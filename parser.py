import hashlib
import logging
import random
import re
from pathlib import Path
from typing import Any, Final, Sequence

import yaml
from wildcards import load_lines

logger = logging.getLogger(__name__)

__all__ = ["YAMLPromptTemplateParser"]


class YAMLPromptTemplateParser:
    VALUES_KEYS: Final[Sequence[str]] = ("values", "options", "choices")
    CHOICE_KEYS: Final[Sequence[str]] = ("choice", "oneOf")

    DEFAULT_WILDCARD_DIR: Final[Path] = Path(__file__).with_name("wildcards")
    MAX_EXPANSION_DEPTH: Final[int] = 64

    VARIABLE_PATTERN = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)")
    BRACE_PATTERN = re.compile(r"\{([^{}]+)\}")  # {a|0.5::b|c}
    WILDCARD_PATTERN = re.compile(r"__([A-Za-z0-9_]+)__")
    FUNCTION_PATTERN = re.compile(
        r"^rand\(\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\)$"
    )

    def __init__(self, seed: int | None = None, wildcard_dir: Path | str | None = None):
        """Create a parser with optional seed and wildcard directory."""
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

    def _load_wildcard(self, name: str) -> list[str]:
        """Return non-blank lines from ``<wildcard_dir>/<name>.txt``, cached."""
        key = (self.wildcard_dir, name)
        if key in self._wildcard_cache:
            return self._wildcard_cache[key]

        lines = load_lines(self.wildcard_dir, name)

        self._wildcard_cache[key] = lines
        return lines

    def _seed_derived_index(self, name: str, n: int) -> int:
        """Return an index derived from the seed and *name*, bypassing RNG state."""
        if self.seed is None:
            return self.rng.randrange(n)
        key = f"{self.seed}:{str(self.wildcard_dir)}:{name}".encode("utf-8")
        digest = hashlib.sha256(key).digest()
        return int.from_bytes(digest[:8], "big") % n

    def _resolve_builtin_call(self, text: str) -> str:
        """Evaluate built-in function calls; currently only ``rand(lo, hi)``."""
        match = self.FUNCTION_PATTERN.match(text)
        if match:
            low, high = map(float, match.groups())
            return str(round(self.rng.uniform(low, high), 2))
        return text

    def _get_list_values(self, block: dict) -> list | None:
        """Find the values/options/choices list in a block dict."""
        for key in self.VALUES_KEYS:
            if key in block:
                return block[key]
        return None

    def _substitute_variables(self, text: str, variables: dict[str, str]) -> str:
        """Replace ``$name`` references with their values from *variables*."""
        def repl(match: re.Match[str]) -> str:
            return variables.get(match.group(1), match.group(0))

        return self.VARIABLE_PATTERN.sub(repl, text)

    def _choose_brace(self, match: re.Match[str], variables: dict[str, str]) -> str:
        """Resolve a ``{a|0.5::b|c}`` brace expression into one chosen option."""
        parts = [p.strip() for p in match.group(1).split("|") if p.strip()]
        options: list[str] = []
        weights: list[float] = []

        for part in parts:
            if "::" in part:
                w_txt, opt_txt = part.split("::", 1)
                try:
                    weight = float(w_txt)
                except ValueError:
                    weight, opt_txt = 1.0, part
            else:
                weight, opt_txt = 1.0, part

            options.append(self.expand_string(opt_txt, variables))
            weights.append(weight)

        if not options:
            return ""
        return self.rng.choices(options, weights)[0]

    def _substitute_wildcards(self, text: str) -> str:
        """Replace ``__name__`` wildcard tokens with lines from text files."""
        def repl(match: re.Match[str]) -> str:
            name = match.group(1)
            candidates = self._load_wildcard(name)
            if not candidates:
                logger.warning(
                    "Wildcard '%s' not found or empty: %s/%s.txt",
                    name, self.wildcard_dir, name,
                )
                return ""
            idx = self._seed_derived_index(name, len(candidates))
            return candidates[idx]

        return self.WILDCARD_PATTERN.sub(repl, text)

    def expand_string(self, text: str, variables: dict[str, str]) -> str:
        """Expand ``$vars``, brace lists, and wildcards until stable."""
        expanded = self._substitute_variables(text, variables)

        def resolve_brace(match: re.Match[str]) -> str:
            return self._choose_brace(match, variables)

        for _ in range(self.MAX_EXPANSION_DEPTH):
            new_text = self.BRACE_PATTERN.sub(resolve_brace, expanded)
            new_text = self._substitute_wildcards(new_text)

            if new_text == expanded:
                return new_text.strip()
            expanded = new_text

        raise ValueError(
            f"Expansion depth exceeded ({self.MAX_EXPANSION_DEPTH} iterations). "
            f"Possible self-referencing pattern in: {text[:80]!r}"
        )

    def _is_choice_item(self, item: Any) -> bool:
        """Check if *item* is a choice/oneOf block (any variant)."""
        if not isinstance(item, dict):
            return False
        if len(item) == 1:
            first_key = next(iter(item))
            if first_key in self.CHOICE_KEYS:
                return True
        return any(k in item for k in self.CHOICE_KEYS)

    def _filter_and_weigh_options(
        self, options: list, variables: dict[str, str]
    ) -> tuple[list[str], list[float]]:
        """Filter *options* by per-item chance, return ``(texts, weights)``."""
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
        return texts, weights

    def _resolve_choice(
        self, block: dict[str, Any], variables: dict[str, str]
    ) -> str | None:
        """Pick one option from a choice/oneOf block, applying weights and chance."""
        chance = float(block.get("chance", 1))
        if chance < 1.0 and self.rng.random() > chance:
            return None

        template = block.get("template", "$value")
        options = self._get_list_values(block)
        if options is None:
            raise ValueError("choice/oneOf requires 'values', 'options', or 'choices'.")

        texts, weights = self._filter_and_weigh_options(options, variables)
        if not texts:
            return None

        chosen = self.rng.choices(texts, weights)[0]
        return self.expand_string(template.replace("$value", chosen), variables)

    def _normalize_choice_block(self, item: dict) -> dict:
        """Normalize a choice wrapper into a standard choice dict."""
        if len(item) != 1:
            return item
        block = item[next(iter(item))]
        if not isinstance(block, dict):
            return {"values": block}
        return block

    def _resolve_item(self, item: Any, variables: dict[str, str]) -> str | None:
        """Resolve a single item into a prompt string (or ``None`` if skipped)."""
        if self._is_choice_item(item):
            return self._resolve_choice(self._normalize_choice_block(item), variables)

        if isinstance(item, dict) and "name" in item:
            item_chance = float(item.get("chance", 1))
            if item_chance < 1.0 and self.rng.random() > item_chance:
                return None
            return self.expand_string(str(item["name"]), variables)

        return self.expand_string(str(item), variables)

    def _collect_vars(
        self,
        raw: dict[str, Any],
        base: dict[str, str],
    ) -> dict[str, str]:
        """Evaluate raw variable definitions and merge onto *base*."""
        variables: dict[str, str] = {**base}
        for name, value in raw.items():
            val = self._resolve_item(value, variables)
            if isinstance(val, str):
                val = self._resolve_builtin_call(val)
            variables[name] = val if val is not None else ""
        return variables

    def _apply_chance(self, section: Any) -> Any | None:
        """Evaluate section-level chance; return cleaned section or ``None`` if skipped."""
        if not isinstance(section, dict) or "chance" not in section:
            return section

        try:
            chance = float(section["chance"])
        except (TypeError, ValueError):
            raise ValueError(f"Invalid chance on section: {section['chance']!r}")
        chance = max(0.0, min(1.0, chance))

        if chance < 1.0 and self.rng.random() > chance:
            return None

        return {k: v for k, v in section.items() if k != "chance"}

    def _extract_section_config(
        self, section: Any, variables: dict[str, str]
    ) -> tuple[dict[str, str], str, str | None]:
        """Extract local vars and templates from a section.

        Returns ``(updated_variables, item_template, block_template)``.
        """
        if isinstance(section, dict) and "vars" in section:
            variables = self._collect_vars(section["vars"], variables)

        if isinstance(section, dict):
            raw_item_tpl = section.get("template", "$value")
            raw_block_tpl = section.get("block_template")
        else:
            raw_item_tpl, raw_block_tpl = "$value", None

        item_tpl = self.expand_string(raw_item_tpl, variables)
        block_tpl = self.expand_string(raw_block_tpl, variables) if raw_block_tpl else None

        return variables, item_tpl, block_tpl

    def _extract_items(self, section: Any) -> list:
        """Extract the list of items from a section."""
        if isinstance(section, dict):
            items = self._get_list_values(section)
            return items if items is not None else []
        if isinstance(section, list):
            return section
        return [section]

    def _apply_item_template(
        self, text: str, item_tpl: str, variables: dict[str, str]
    ) -> str:
        """Apply *item_tpl* to *text* by replacing ``$value``."""
        return self.expand_string(item_tpl.replace("$value", text), variables)

    def _flush_pending(
        self, pending: list[str], item_tpl: str, variables: dict[str, str],
    ) -> str | None:
        if not pending:
            return None
        return self._apply_item_template(", ".join(pending), item_tpl, variables)

    def _render_items(self, items: list, variables: dict[str, str], item_tpl: str) -> list[str]:
        """Render items into a list of template-applied strings."""
        rendered: list[str] = []
        pending: list[str] = []

        for item in items:
            if isinstance(item, str):
                pending.append(self.expand_string(item, variables))
                continue

            if pending and self._is_choice_item(item):
                result = self._resolve_item(item, variables)
                if result is not None:
                    pending.append(result)
                rendered.append(self._apply_item_template(", ".join(pending), item_tpl, variables))
                pending = []
                continue

            flushed = self._flush_pending(pending, item_tpl, variables)
            if flushed is not None:
                rendered.append(flushed)
            pending = []

            result = self._resolve_item(item, variables)
            if result is not None:
                rendered.append(self._apply_item_template(result, item_tpl, variables))

        flushed = self._flush_pending(pending, item_tpl, variables)
        if flushed is not None:
            rendered.append(flushed)

        return rendered

    def _apply_templates(
        self,
        is_simple_plain: bool,
        rendered_lines: list[str],
        block_tpl: str | None,
        variables: dict[str, str],
    ) -> list[str]:
        """Apply block_template or merge plain items into final output."""
        if is_simple_plain:
            return [", ".join(rendered_lines)]

        if block_tpl is not None:
            return [
                self.expand_string(
                    block_tpl.replace("$value", ", ".join(rendered_lines)),
                    variables,
                )
            ]

        return rendered_lines

    def _parse_section(self, section: Any, variables: dict[str, str]) -> list[str]:
        """Parse a single document section into a list of prompt lines."""
        if section is None:
            return []

        section = self._apply_chance(section)
        if section is None:
            return []

        variables, item_tpl, block_tpl = self._extract_section_config(section, variables)
        items = self._extract_items(section)
        is_simple_plain = (
            all(isinstance(e, str) for e in items)
            and item_tpl == "$value"
            and block_tpl is None
        )
        rendered_lines = self._render_items(items, variables, item_tpl)
        return self._apply_templates(is_simple_plain, rendered_lines, block_tpl, variables)

    def parse_document(self, doc: dict[str, Any]) -> list[list[str]]:
        """Flatten *doc* into blocks of prompt lines.

        Parameters
        ----------
        doc : dict[str, Any]
            YAML mapping as returned by ``yaml.safe_load``.

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

        variables = self._collect_vars(doc.get("vars", {}), {})

        blocks: list[list[str]] = []
        for name, section in doc.items():
            if name == "vars":
                continue
            lines = self._parse_section(section, variables)
            if lines:
                blocks.append(lines)

        return blocks
