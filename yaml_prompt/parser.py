import hashlib
import logging
import random
import re
from pathlib import Path
from typing import Any, Final, Sequence

from .wildcards import load_lines

logger = logging.getLogger(__name__)

__all__ = ["YAMLPromptTemplateParser"]


class YAMLPromptTemplateParser:
    VALUES_KEYS: Final[Sequence[str]] = ("values", "options", "choices")
    CHOICE_KEYS: Final[Sequence[str]] = ("choice", "oneOf")

    DEFAULT_WILDCARD_DIR: Final[Path] = Path(__file__).with_name("wildcards")
    MAX_EXPANSION_DEPTH: Final[int] = 64
    MAX_CHOICE_DEPTH: Final[int] = 16

    @staticmethod
    def _stable_select(
        seed: int, items: list[str], weights: list[float] | None = None
    ) -> str:
        """Pick from *items* using SHA-256(seed + joined items) — stable across
        independent parser instances sharing the same seed, regardless of RNG state.
        """
        key = f"{seed}:choice:{'|'.join(items)}".encode("utf-8")
        digest = hashlib.sha256(key).digest()

        if weights is None or all(w == weights[0] for w in weights):
            idx = int.from_bytes(digest[:8], "big") % len(items)
            return items[idx]

        hash_float = int.from_bytes(digest[:8], "big") / (1 << 64)
        total = sum(weights)
        cumulative = 0.0
        for i, w in enumerate(weights):
            cumulative += w / total
            if hash_float < cumulative:
                return items[i]
        return items[-1]

    def _stable_chance(
        self,
        chance: float,
        content: Any,
        *,
        key: str | None = None,
    ) -> bool:
        """Hash-based chance — stable across templates with the same seed.

        When *key* is provided it replaces *content* in the hash, letting
        different blocks share the same coin-flip via an explicit identifier.
        """
        if key is not None:
            raw_key = f"{self.seed}:chance:{key}".encode("utf-8")
        else:
            raw_key = f"{self.seed}:chance:{content}".encode("utf-8")
        digest = hashlib.sha256(raw_key).digest()
        roll = int.from_bytes(digest[:8], "big") / (1 << 64)
        return roll <= chance

    @staticmethod
    def _random_chance(chance: float) -> bool:
        """Truly random chance — ignores seed, varies every render."""
        return random.random() <= chance

    def _parse_chance(
        self, raw: float | dict[str, Any]
    ) -> tuple[float, bool, str | None]:
        """Parse a chance value (float or dict) into components.

        Supports two forms:

        * ``chance: 0.5`` — plain float, stable, auto-keyed from content.
        * ``chance: {value: 0.5, stable: false, key: "tag"}`` — dict with
          optional ``stable`` (default ``True``) and ``key`` fields.

        Parameters
        ----------
        raw:
            Either a numeric value or a dict with ``value``, optional
            ``stable`` (default ``True``), and optional ``key``.

        Returns
        -------
        tuple[float, bool, str | None]
            ``(probability, is_stable, custom_key)``
        """
        if isinstance(raw, dict):
            value = self._safe_chance(raw.get("value", 1))
            stable = bool(raw.get("stable", True))
            key: str | None = None
            if stable and "key" in raw:
                key = str(raw["key"])
            return value, stable, key
        return self._safe_chance(raw), True, None

    def _evaluate_chance(
        self, raw_chance: float | dict[str, Any], content: Any
    ) -> bool:
        """Evaluate a chance gate (float or dict) against *content*.

        Parameters
        ----------
        raw_chance:
            Raw ``chance`` value from YAML — a number or a dict with
            ``value``, ``stable``, and ``key`` fields.
        content:
            Fallback material for hash-based key derivation when no
            custom ``key`` is provided.  The ``chance`` key is stripped
            automatically so that ``chance: 0.5`` and
            ``chance: {value: 0.5}`` hash identically.

        Returns
        -------
        bool
            ``True`` if the content passes the chance gate.
        """
        chance, stable, custom_key = self._parse_chance(raw_chance)
        if chance >= 1.0:
            return True
        if chance <= 0.0:
            return False
        if stable:
            if isinstance(content, dict) and "chance" in content:
                content = {k: v for k, v in content.items() if k != "chance"}
            return self._stable_chance(chance, content, key=custom_key)
        return self._random_chance(chance)

    VARIABLE_PATTERN = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)")
    BRACE_PATTERN = re.compile(r"\{([^{}]+)\}")  # {a|0.5::b|c}
    WILDCARD_PATTERN = re.compile(r"__([A-Za-z0-9_]+)__")
    FUNCTION_PATTERN = re.compile(
        r"^rand\(\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\)$"
    )

    def __init__(self, seed: int | None = None, wildcard_dir: Path | str | None = None):
        """Create a parser with optional seed and wildcard directory."""
        if seed is None:
            seed = random.randint(0, 2**63 - 1)

        self.rng = random.Random(seed)
        self.seed = seed
        self._wildcard_cache: dict[tuple[Path, str], list[str]] = {}

        if wildcard_dir:
            self.wildcard_dir = Path(wildcard_dir).expanduser().resolve()
        else:
            self.wildcard_dir = self.DEFAULT_WILDCARD_DIR

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

        global_vars = self._collect_vars(doc.get("vars", {}), {})

        blocks: list[list[str]] = []
        for name, section in doc.items():
            if name == "vars":
                continue
            lines = self._parse_section(section, global_vars)
            if lines:
                blocks.append(lines)

        return blocks

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

    def _parse_section(self, section: Any, variables: dict[str, str]) -> list[str]:
        """Parse a single document section into a list of prompt lines."""
        if section is None:
            return []

        section = self._apply_chance(section)
        if section is None:
            return []

        variables, item_tpl, block_tpl = self._extract_section_config(
            section, variables
        )
        items = self._extract_items(section)
        is_simple_plain = (
            all(isinstance(e, str) for e in items)
            and item_tpl == "$value"
            and block_tpl is None
        )
        rendered_lines = self._render_items(items, variables, item_tpl)
        return self._apply_templates(
            is_simple_plain, rendered_lines, block_tpl, variables
        )

    def _apply_chance(self, section: Any) -> Any | None:
        """Evaluate section-level chance; return cleaned section or ``None`` if skipped."""
        if not isinstance(section, dict) or "chance" not in section:
            return section

        raw_chance = section["chance"]
        content = {k: v for k, v in section.items() if k != "chance"}

        if not self._evaluate_chance(raw_chance, content):
            return None

        return content

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
        block_tpl = (
            self.expand_string(raw_block_tpl, variables) if raw_block_tpl else None
        )

        return variables, item_tpl, block_tpl

    def _extract_items(self, section: Any) -> list:
        """Extract the list of items from a section."""
        if isinstance(section, dict):
            items = self._get_list_values(section)
            return items if items is not None else []
        if isinstance(section, list):
            return section
        return [section]

    def _render_items(
        self, items: list, variables: dict[str, str], item_tpl: str
    ) -> list[str]:
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
                rendered.append(
                    self._apply_item_template(", ".join(pending), item_tpl, variables)
                )
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

    def _resolve_item(self, item: Any, variables: dict[str, str]) -> str | None:
        """Resolve a single item into a prompt string (or ``None`` if skipped)."""
        if self._is_choice_item(item):
            return self._resolve_choice(self._normalize_choice_block(item), variables)

        if isinstance(item, dict) and "name" in item:
            if "chance" in item and not self._evaluate_chance(item["chance"], item):
                return None
            return self.expand_string(str(item["name"]), variables)

        return self.expand_string(str(item), variables)

    def _is_choice_item(self, item: Any) -> bool:
        """Check if *item* is a choice/oneOf block (any variant)."""
        if not isinstance(item, dict):
            return False
        if len(item) == 1:
            first_key = next(iter(item))
            if first_key in self.CHOICE_KEYS:
                return True
        return any(k in item for k in self.CHOICE_KEYS)

    def _resolve_choice(
        self,
        block: dict[str, Any],
        variables: dict[str, str],
        _depth: int = 0,
    ) -> str | None:
        """Pick one option from a choice/oneOf block, applying weights and chance."""
        if "chance" in block and not self._evaluate_chance(block["chance"], block):
            return None

        template = block.get("template", "$value")
        options = self._get_list_values(block)
        if options is None:
            raise ValueError("choice/oneOf requires 'values', 'options', or 'choices'.")

        texts, weights = self._filter_and_weigh_options(
            options, variables, _depth=_depth
        )
        if not texts:
            return None

        chosen = self._stable_select(self.seed, texts, weights)
        return self.expand_string(template.replace("$value", chosen), variables)

    def _normalize_choice_block(self, item: dict) -> dict:
        """Normalize a choice wrapper into a standard choice dict."""
        if len(item) != 1:
            return item
        block = item[next(iter(item))]
        if not isinstance(block, dict):
            return {"values": block}
        return block

    def _filter_and_weigh_options(
        self,
        options: list,
        variables: dict[str, str],
        _depth: int = 0,
    ) -> tuple[list[str], list[float]]:
        """Filter *options* by per-item chance, return ``(texts, weights)``.

        Nested choice/oneOf blocks are resolved recursively up to
        ``MAX_CHOICE_DEPTH`` levels.
        """
        if _depth > self.MAX_CHOICE_DEPTH:
            raise ValueError(
                f"Nested choice depth exceeded ({self.MAX_CHOICE_DEPTH} levels). "
                f"Check for accidentally recursive choice definitions."
            )

        texts: list[str] = []
        weights: list[float] = []
        for opt in options:
            if isinstance(opt, dict) and self._is_choice_item(opt):
                # Nested choice block — extract sibling weight, resolve recursively.
                weight = self._safe_weight(opt.get("weight", 1))
                clean = {k: v for k, v in opt.items() if k != "weight"}
                resolved = self._resolve_choice(
                    self._normalize_choice_block(clean),
                    variables,
                    _depth=_depth + 1,
                )
                if resolved is None:
                    continue
                texts.append(resolved)
                weights.append(weight)
            elif isinstance(opt, dict):
                if "chance" in opt and not self._evaluate_chance(opt["chance"], opt):
                    continue
                name, weight = (
                    opt.get("name", ""),
                    self._safe_weight(opt.get("weight", 1)),
                )
                texts.append(self.expand_string(str(name), variables))
                weights.append(weight)
            else:
                name, weight = opt, 1.0
                texts.append(self.expand_string(str(name), variables))
                weights.append(weight)
        return texts, weights

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
        return self._stable_select(self.seed, options, weights)

    def _substitute_wildcards(self, text: str) -> str:
        """Replace ``__name__`` wildcard tokens with lines from text files."""

        def repl(match: re.Match[str]) -> str:
            name = match.group(1)
            candidates = self._load_wildcard(name)
            if not candidates:
                logger.warning(
                    "Wildcard '%s' not found or empty: %s/%s.txt",
                    name,
                    self.wildcard_dir,
                    name,
                )
                return ""
            idx = self._seed_derived_index(name, len(candidates))
            return candidates[idx]

        return self.WILDCARD_PATTERN.sub(repl, text)

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

    def _get_list_values(self, block: dict) -> list | None:
        """Find the values/options/choices list in a block dict."""
        for key in self.VALUES_KEYS:
            if key in block:
                return block[key]
        return None

    def _apply_item_template(
        self, text: str, item_tpl: str, variables: dict[str, str]
    ) -> str:
        """Apply *item_tpl* to *text* by replacing ``$value``."""
        return self.expand_string(item_tpl.replace("$value", text), variables)

    def _flush_pending(
        self,
        pending: list[str],
        item_tpl: str,
        variables: dict[str, str],
    ) -> str | None:
        if not pending:
            return None
        return self._apply_item_template(", ".join(pending), item_tpl, variables)

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
        key = f"{self.seed}:{name}".encode("utf-8")
        digest = hashlib.sha256(key).digest()
        return int.from_bytes(digest[:8], "big") % n

    def _resolve_builtin_call(self, text: str) -> str:
        """Evaluate built-in function calls; currently only ``rand(lo, hi)``."""
        match = self.FUNCTION_PATTERN.match(text)
        if match:
            low, high = map(float, match.groups())
            return str(round(self.rng.uniform(low, high), 2))
        return text

    def _safe_chance(self, value: Any) -> float:
        try:
            chance = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"Invalid chance value: {value!r}")
        return max(0.0, min(1.0, chance))

    def _safe_weight(self, value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            raise ValueError(f"Invalid weight value: {value!r}")
