import argparse
import hashlib
import logging
import random
import re
from pathlib import Path
from typing import Any, Final, Iterable, Sequence

logger = logging.getLogger(__name__)

import yaml

aYAML = Any  # Loaded-YAML value (dict, list, str, …)


class YAMLPromptTemplateParser:
    # -----------------------------------------------------------------------
    # Constants & type aliases
    # -----------------------------------------------------------------------
    LIST_KEYS: Final[Sequence[str]] = ("values", "options", "choices")
    CHOICE_KEYS: Final[Sequence[str]] = ("choice", "oneOf")

    DEFAULT_WILDCARD_DIR: Final[Path] = Path(__file__).with_name("wildcards")
    MAX_EXPANSION_DEPTH: Final[int] = 64

    VARIABLE_PATTERN = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)")
    BRACE_PATTERN = re.compile(r"\{([^{}]+)\}")  # {a|0.5::b|c}
    WILDCARD_PATTERN = re.compile(r"__([A-Za-z0-9_]+)__")
    FUNCTION_PATTERN = re.compile(
        r"^rand\(\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\)$"
    )

    # -----------------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------------
    def __init__(self, seed: int | None = None, wildcard_dir: Path | str | None = None):
        """
        Parameters
        ----------
        seed : int | None
            If provided, RNG will be seeded to this value for reproducible outputs.
        wildcard_dir : Path | str | None
            Directory containing wildcard `*.txt` files. If None, defaults to
            the module’s `wildcards/` folder.
        """
        if seed is not None:
            self.rng = random.Random(seed)
        else:
            self.rng = random.Random()

        self.seed = seed
        self._wildcard_cache: dict[tuple[Path, str], list[str]] = {}

        if wildcard_dir:
            self.wildcard_dir = Path(wildcard_dir).expanduser().resolve()
        else:
            self.wildcard_dir = self.DEFAULT_WILDCARD_DIR

    # -----------------------------------------------------------------------
    # Wildcards
    # -----------------------------------------------------------------------
    def _load_wildcard(self, name: str) -> list[str]:
        """Return non-blank lines from `directory/name.txt`, cached."""
        key = (self.wildcard_dir, name)
        if key in self._wildcard_cache:
            return self._wildcard_cache[key]

        file_path = self.wildcard_dir / f"{name}.txt"
        try:
            lines = [
                ln.strip()
                for ln in file_path.read_text(encoding="utf-8").splitlines()
                if ln.strip()
            ]
        except FileNotFoundError:
            lines = []

        self._wildcard_cache[key] = lines
        return lines

    # -----------------------------------------------------------------------
    # Expression helpers
    # -----------------------------------------------------------------------
    def _eval_builtin(self, text: str) -> str:
        """Evaluate built-in calls (currently only `rand(min, max)`)."""
        m = self.FUNCTION_PATTERN.match(text)
        if m:
            lo, hi = map(float, m.groups())
            return str(round(self.rng.uniform(lo, hi), 2))
        return text

    # -----------------------------------------------------------------------
    # Expansion helpers (variables → braces → wildcards)
    # -----------------------------------------------------------------------
    def _subst_vars(self, text: str, variables: dict[str, str]) -> str:
        def repl(m: re.Match[str]) -> str:
            return variables.get(m.group(1), m.group(0))

        return self.VARIABLE_PATTERN.sub(repl, text)

    def _choose_brace(self, m: re.Match[str], variables: dict[str, str]) -> str:
        parts = [p.strip() for p in m.group(1).split("|") if p.strip()]
        opts: list[str] = []
        wgts: list[float] = []

        for part in parts:
            if "::" in part:
                w_txt, opt_txt = part.split("::", 1)
                try:
                    weight = float(w_txt)
                except ValueError:
                    weight, opt_txt = 1.0, part
            else:
                weight, opt_txt = 1.0, part

            opts.append(self.expand_string(opt_txt, variables))
            wgts.append(weight)

        if not opts:
            return ""
        return self.rng.choices(opts, wgts)[0]

    def _subst_wildcards(self, text: str) -> str:
        def repl(m: re.Match[str]) -> str:
            name = m.group(1)
            options = self._load_wildcard(name)
            if not options:
                logger.warning(
                    "Wildcard '%s' not found or empty: %s/%s.txt",
                    name, self.wildcard_dir, name,
                )
                return ""
            idx = self._stable_index_for_wildcard(name, len(options))
            return options[idx]

        return self.WILDCARD_PATTERN.sub(repl, text)

    def expand_string(self, text: str, variables: dict[str, str]) -> str:
        """Expand `$vars`, brace lists, and wildcards until stable."""
        # First, substitute all variable references
        expanded = self._subst_vars(text, variables)

        # Iteratively resolve braces and wildcards until stable
        for _ in range(self.MAX_EXPANSION_DEPTH):
            new_text = self.BRACE_PATTERN.sub(
                lambda m: self._choose_brace(m, variables), expanded
            )
            new_text = self._subst_wildcards(new_text)

            if new_text == expanded:
                return new_text.strip()
            expanded = new_text

        raise ValueError(
            f"Expansion depth exceeded ({self.MAX_EXPANSION_DEPTH} iterations). "
            f"Possible self-referencing pattern in: {text[:80]!r}"
        )

    # -----------------------------------------------------------------------
    # choice/oneOf handling
    # -----------------------------------------------------------------------
    def _resolve_choice(
        self, block: dict[str, Any], variables: dict[str, str]
    ) -> str | None:
        chance = float(block.get("chance", 1))
        if chance < 1.0 and self.rng.random() > chance:
            return None

        template = block.get("template", "$value")
        options = next((block[k] for k in self.LIST_KEYS if k in block), None)
        if options is None:
            raise ValueError("choice/oneOf requires 'values', 'options', or 'choices'.")

        texts: list[str] = []
        weights: list[float] = []

        for opt in options:
            if isinstance(opt, dict):
                opt_chance = float(opt.get("chance", 1))
                if opt_chance < 1.0 and self.rng.random() > opt_chance:
                    continue
                name, weight = opt.get("name", ""), float(opt.get("weight", 1))
            else:
                name, weight = opt, 1.0

            texts.append(self.expand_string(str(name), variables))
            weights.append(weight)

        if not texts:
            return None

        chosen = self.rng.choices(texts, weights)[0]
        return self.expand_string(template.replace("$value", chosen), variables)

    # -----------------------------------------------------------------------
    # Item evaluation
    # -----------------------------------------------------------------------
    def _eval_item(self, item: aYAML, variables: dict[str, str]) -> str | None:
        """Evaluate a single item into a prompt string.

        Match priority (first match wins):
        1. Single-key choice wrapper: {"choice": ...} or {"oneOf": ...}
        2. Dict with choice key anywhere: {"choice": ..., "template": ...}
           Note: overlaps with 1, but 1 normalizes shorthand before delegating.
        3. Named entry: {"name": "...", "chance": 0.5, "weight": 2}
        4. Plain string
        5. Fallback: stringify
        """
        # 1) Wrapper shorthand for single-key choice/oneOf blocks
        if (
            isinstance(item, dict)
            and len(item) == 1
            and next(iter(item)) in self.CHOICE_KEYS
        ):
            key = next(iter(item))
            block = item[key]
            if not isinstance(block, dict):
                block = {"values": block}
            return self._resolve_choice(block, variables)

        # 2) Direct mapping with choice keys present anywhere
        if isinstance(item, dict) and any(k in item for k in self.CHOICE_KEYS):
            return self._resolve_choice(item, variables)

        # 3) Named entry with chance
        if isinstance(item, dict) and "name" in item:
            item_chance = float(item.get("chance", 1))
            if item_chance < 1.0 and self.rng.random() > item_chance:
                return None
            return self.expand_string(str(item["name"]), variables)

        # 4) Plain string
        if isinstance(item, str):
            return self.expand_string(item, variables)

        # 5) Fallback: stringify anything else
        return self.expand_string(str(item), variables)

    # -----------------------------------------------------------------------
    # Variable collection
    # -----------------------------------------------------------------------
    def _collect_vars(
        self,
        raw: dict[str, aYAML],
        base: dict[str, str],
    ) -> dict[str, str]:
        vars_: dict[str, str] = dict(base)
        for name, value in raw.items():
            val = self._eval_item(value, vars_)
            if isinstance(val, str):
                val = self._eval_builtin(val)
            vars_[name] = val or ""
        return vars_

    # -----------------------------------------------------------------------
    # Section parsing
    # -----------------------------------------------------------------------
    def _parse_section(self, section: aYAML, variables: dict[str, str]) -> list[str]:
        if section is None:
            return []

        if isinstance(section, dict) and "chance" in section:
            try:
                chance = float(section.get("chance", 1))
            except (TypeError, ValueError):
                raise ValueError(
                    f"Invalid chance on section: {section.get('chance')!r}"
                )
            chance = max(0.0, min(1.0, chance))
            if chance < 1.0 and self.rng.random() > chance:
                return []
            # strip 'chance' so it doesn't leak into template/vars processing
            section = {k: v for k, v in section.items() if k != "chance"}

        # Handle section-local vars
        if isinstance(section, dict) and "vars" in section:
            variables = self._collect_vars(section["vars"], variables)

        # Determine templates for items and blocks
        if isinstance(section, dict):
            raw_item_tpl = section.get("template", "$value")
            raw_block_tpl = section.get("block_template")
        else:
            raw_item_tpl, raw_block_tpl = "$value", None

        item_tpl = self.expand_string(raw_item_tpl, variables)
        block_tpl = (
            self.expand_string(raw_block_tpl, variables) if raw_block_tpl else None
        )

        # Extract list of items
        if isinstance(section, dict):
            list_items: Iterable[aYAML] | None = next(
                (section[k] for k in self.LIST_KEYS if k in section), None
            )
            if list_items is None:
                list_items = []
        elif isinstance(section, list):
            list_items = section
        else:
            list_items = [section]

        merged: list[str] = []
        buffer: list[str] = []

        def flush_buffer() -> None:
            if buffer:
                merged_txt = ", ".join(buffer)
                merged.append(
                    self.expand_string(
                        item_tpl.replace("$value", merged_txt), variables
                    )
                )
                buffer.clear()

        for itm in list_items:
            # Plain strings get buffered
            if isinstance(itm, str):
                buffer.append(self.expand_string(itm, variables))
                continue

            # If next item is a choice, flush buffer first
            is_choice = isinstance(itm, dict) and (
                (len(itm) == 1 and next(iter(itm)) in self.CHOICE_KEYS)
                or any(k in itm for k in self.CHOICE_KEYS)
            )
            if buffer and is_choice:
                ch = self._eval_item(itm, variables)
                if ch is not None:
                    buffer.append(ch)
                flush_buffer()
                continue

            # Otherwise, flush whatever is in buffer, then handle this item
            flush_buffer()
            ev = self._eval_item(itm, variables)
            if ev is not None:
                merged.append(
                    self.expand_string(item_tpl.replace("$value", ev), variables)
                )

        # Flush any remaining buffered strings
        flush_buffer()

        # If everything is plain text without templates, merge into one line
        simple_plain = (
            all(isinstance(e, str) for e in list_items)
            and item_tpl == "$value"
            and block_tpl is None
        )
        if simple_plain:
            return [", ".join(merged)]

        # If there’s a block_template, apply it to the entire merged list
        if block_tpl is not None:
            return [
                self.expand_string(
                    block_tpl.replace("$value", ", ".join(merged)), variables
                )
            ]

        return merged

    def _stable_index_for_wildcard(self, name: str, n: int) -> int:
        if self.seed is None:
            return self.rng.randrange(n)
        key = f"{self.seed}:{str(self.wildcard_dir)}:{name}".encode("utf-8")
        digest = hashlib.sha256(key).digest()
        return int.from_bytes(digest[:8], "big") % n

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------
    def parse_document(self, doc: dict[str, aYAML]) -> list[list[str]]:
        """Flatten `doc` into blocks of prompt lines.

        Parameters
        ----------
        doc : dict[str, aYAML]
            YAML mapping as returned by `yaml.safe_load`.

        Returns
        -------
        list[list[str]]
            Each inner list is one section's flattened prompt lines.
        """
        if not isinstance(doc, dict):
            raise TypeError(
                f"Expected a YAML mapping (dict) at top level, got {type(doc).__name__}. "
                f"Check that your YAML file starts with key: value pairs, not a list."
            )

        # First, collect global variables
        variables = self._collect_vars(doc.get("vars", {}), {})

        blocks: list[list[str]] = []
        for name, section in doc.items():
            if name == "vars":
                continue
            lines = self._parse_section(section, variables)
            if lines:
                blocks.append(lines)

        return blocks


def _parse_var_value(value: str) -> Any:
    import json

    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    try:
        parsed = json.loads(value)
        if isinstance(parsed, (list, dict)):
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass
    return value


def main() -> None:
    from jinja_env import render_template

    ap = argparse.ArgumentParser(
        description="Flatten YAML prompt files into prompt lines (seeded RNG)."
    )
    ap.add_argument(
        "file", type=Path, help="YAML prompt definition file (e.g. prompt.yaml)"
    )
    ap.add_argument(
        "--wildcards-dir",
        type=Path,
        default=None,
        help="Directory containing wildcard `*.txt` files",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for deterministic randomness (optional)",
    )
    ap.add_argument(
        "--var",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Jinja2 context variable (repeatable, e.g. --var enemy=true)",
    )
    args = ap.parse_args()

    jinja_vars: dict[str, Any] = {}
    for var_str in args.var:
        if "=" not in var_str:
            ap.error(f"--var must be KEY=VALUE, got: {var_str!r}")
        key, val = var_str.split("=", 1)
        jinja_vars[key.strip()] = _parse_var_value(val.strip())

    try:
        raw_yaml = args.file.read_text(encoding="utf-8")
    except OSError as err:
        ap.error(f"Cannot read '{args.file}': {err}")

    # Phase 1: Jinja2 preprocessing
    try:
        rendered = render_template(
            raw_yaml,
            jinja_vars=jinja_vars or None,
            search_paths=[args.file.parent.resolve()],
            seed=args.seed,
            wildcard_dir=args.wildcards_dir,
        )
    except Exception as err:
        ap.error(f"Jinja2 error: {err}")

    # Phase 2: YAML parsing
    try:
        data: dict[str, aYAML] = yaml.safe_load(rendered) or {}
    except yaml.YAMLError as err:
        ap.error(f"YAML error: {err}")

    flattener = YAMLPromptTemplateParser(
        seed=args.seed,
        wildcard_dir=args.wildcards_dir,
    )
    blocks = flattener.parse_document(data)

    for i, blk in enumerate(blocks):
        print(*blk, sep="\n")
        if i != len(blocks) - 1:
            print()


if __name__ == "__main__":
    main()
