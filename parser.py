import argparse
import hashlib
import random
import re
from pathlib import Path
from typing import Any, Final, Iterable, Sequence

import yaml

aYAML = Any  # Loaded-YAML value (dict, list, str, …)


class YAMLPromptTemplateParser:
    # -----------------------------------------------------------------------
    # Constants & type aliases
    # -----------------------------------------------------------------------
    LIST_KEYS: Final[Sequence[str]] = ("values", "options", "choices")
    CHOICE_KEYS: Final[Sequence[str]] = ("choice", "oneOf")

    DEFAULT_WILDCARD_DIR: Final[Path] = Path(__file__).with_name("wildcards")
    _WILDCARD_CACHE: dict[tuple[Path, str], list[str]] = {}

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
            # Create a private RNG instance
            self.random = random.Random(seed)
        else:
            # Fall back to Python’s global RNG (not recommended for multi-run reproducibility)
            self.random = random

        self.seed = seed  # keep the original seed

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
        if key in self._WILDCARD_CACHE:
            return self._WILDCARD_CACHE[key]

        file_path = self.wildcard_dir / f"{name}.txt"
        try:
            lines = [
                ln.strip()
                for ln in file_path.read_text(encoding="utf-8").splitlines()
                if ln.strip()
            ]
        except FileNotFoundError:
            lines = []

        self._WILDCARD_CACHE[key] = lines
        return lines

    # -----------------------------------------------------------------------
    # Expression helpers
    # -----------------------------------------------------------------------
    def _eval_builtin(self, text: str) -> str:
        """Evaluate built-in calls (currently only `rand(min, max)`)."""
        m = self.FUNCTION_PATTERN.match(text)
        if m:
            lo, hi = map(float, m.groups())
            return str(round(self.random.uniform(lo, hi), 2))
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

        return self.random.choices(opts, wgts)[0]

    def _subst_wildcards(self, text: str) -> str:
        def repl(m: re.Match[str]) -> str:
            name = m.group(1)
            options = self._load_wildcard(name)
            if not options:
                return m.group(0)
            idx = self._stable_index_for_wildcard(name, len(options))
            return options[idx]

        return self.WILDCARD_PATTERN.sub(repl, text)

    def expand_string(self, text: str, variables: dict[str, str]) -> str:
        """Expand `$vars`, brace lists, and wildcards until stable."""
        # First, substitute all variable references
        expanded = self._subst_vars(text, variables)

        # Then, iteratively resolve braces and wildcards until nothing changes
        while True:
            new_text = self.BRACE_PATTERN.sub(
                lambda m: self._choose_brace(m, variables), expanded
            )
            new_text = self._subst_wildcards(new_text)

            if new_text == expanded:
                return new_text.strip()
            expanded = new_text

    # -----------------------------------------------------------------------
    # choice/oneOf handling
    # -----------------------------------------------------------------------
    def _resolve_choice(
        self, block: dict[str, Any], variables: dict[str, str]
    ) -> str | None:
        if self.random.random() > float(block.get("chance", 1)):
            return None

        template = block.get("template", "$value")
        options = next((block[k] for k in self.LIST_KEYS if k in block), None)
        if options is None:
            raise ValueError("choice/oneOf requires 'values', 'options', or 'choices'.")

        texts: list[str] = []
        weights: list[float] = []

        for opt in options:
            if isinstance(opt, dict):
                if self.random.random() > float(opt.get("chance", 1)):
                    continue
                name, weight = opt.get("name", ""), float(opt.get("weight", 1))
            else:
                name, weight = opt, 1.0

            texts.append(self.expand_string(str(name), variables))
            weights.append(weight)

        if not texts:
            return None

        chosen = self.random.choices(texts, weights)[0]
        return self.expand_string(template.replace("$value", chosen), variables)

    # -----------------------------------------------------------------------
    # Item evaluation
    # -----------------------------------------------------------------------
    def _eval_item(self, item: aYAML, variables: dict[str, str]) -> str | None:
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
            if self.random.random() > float(item.get("chance", 1)):
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
            return self.random.randrange(n)
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
            Each inner list is one section’s flattened prompt lines.
        """

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Flatten YAML prompt files into prompt lines (seeded RNG)."
    )
    parser.add_argument(
        "file", type=Path, help="YAML prompt definition file (e.g. prompt.yaml)"
    )
    parser.add_argument(
        "--wildcards-dir",
        type=Path,
        default=None,
        help="Directory containing wildcard `*.txt` files",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for deterministic randomness (optional)",
    )
    args = parser.parse_args()

    try:
        raw_yaml = args.file.read_text(encoding="utf-8")
    except OSError as err:
        parser.error(f"Cannot read '{args.file}': {err}")

    try:
        data: dict[str, aYAML] = yaml.safe_load(raw_yaml) or {}
    except yaml.YAMLError as err:
        parser.error(f"YAML error: {err}")

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
