import logging
import random
import re
from pathlib import Path
from typing import Any, Final

from .chance import ChanceEvaluator
from .choice import ChoiceResolver
from .expand import StringExpander

logger = logging.getLogger(__name__)

__all__ = ["YAMLPromptTemplateParser"]


class YAMLPromptTemplateParser:
    DEFAULT_WILDCARD_DIR: Final[Path] = Path(__file__).with_name("wildcards")

    FUNCTION_PATTERN = re.compile(
        r"^rand\(\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\)$"
    )

    def __init__(self, seed: int | None = None, wildcard_dir: Path | str | None = None):
        if seed is None:
            seed = random.randint(0, 2**63 - 1)

        self.seed = seed
        self.rng = random.Random(seed)

        if wildcard_dir:
            wdir = Path(wildcard_dir).expanduser().resolve()
        else:
            wdir = self.DEFAULT_WILDCARD_DIR
        self.wildcard_dir = wdir

        unstable_rng = random.Random()
        self._chance = ChanceEvaluator(seed, rng=unstable_rng)
        self._expander = StringExpander(seed, wdir)
        self._choices = ChoiceResolver(
            seed, self._chance, self._expander.expand, rng=unstable_rng
        )

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
        return [lines for _, lines in self.parse_named_sections(doc)]

    def parse_named_sections(self, doc: dict[str, Any]) -> list[tuple[str, list[str]]]:
        """Flatten *doc* into ``(section_name, prompt_lines)`` pairs.

        Same flattening as :meth:`parse_document`, but the top-level section
        name is preserved alongside each block so callers can route individual
        sections to named outputs.

        Parameters
        ----------
        doc : dict[str, Any]
            YAML mapping as returned by ``yaml.safe_load``.

        Returns
        -------
        list[tuple[str, list[str]]]
            One pair per non-empty top-level section, in document order. The
            reserved ``vars`` and ``separator`` keys are excluded.
        """
        if not isinstance(doc, dict):
            raise TypeError(
                f"Expected a YAML mapping (dict) at top level, got {type(doc).__name__}. "
                f"Check that your YAML file starts with key: value pairs, not a list."
            )

        raw_separator = doc.get("separator")
        inherited_sep = str(raw_separator) if raw_separator is not None else None

        global_vars = self._collect_vars(doc.get("vars", {}), {})

        sections: list[tuple[str, list[str]]] = []
        for name, section in doc.items():
            if name in ("vars", "separator"):
                continue
            lines = self._parse_section(section, global_vars, inherited_sep)
            if lines:
                sections.append((str(name), lines))

        return sections

    def expand_string(self, text: str, variables: dict[str, str]) -> str:
        """Expand ``$vars``, brace lists, and wildcards until stable."""
        return self._expander.expand(text, variables)

    def _parse_section(
        self,
        section: Any,
        variables: dict[str, str],
        inherited_sep: str | None = None,
    ) -> list[str]:
        if section is None:
            return []

        section = self._chance.apply_to_section(section)
        if section is None:
            return []

        variables, item_tpl, block_tpl, separator = self._extract_section_config(
            section, variables, inherited_sep
        )
        if isinstance(section, dict) and "separator" in section:
            child_sep = str(section["separator"])
        else:
            child_sep = inherited_sep
        items = self._extract_items(section)
        is_simple_plain = (
            all(isinstance(e, str) for e in items)
            and item_tpl == "$value"
            and block_tpl is None
        )
        rendered_lines = self._render_items(
            items, variables, item_tpl, separator, child_sep
        )
        return self._apply_templates(
            is_simple_plain, rendered_lines, block_tpl, variables, separator
        )

    def _extract_section_config(
        self,
        section: Any,
        variables: dict[str, str],
        inherited_sep: str | None = None,
    ) -> tuple[dict[str, str], str, str | None, str]:
        if isinstance(section, dict) and "vars" in section:
            variables = self._collect_vars(section["vars"], variables)

        if isinstance(section, dict):
            raw_item_tpl = section.get("template", "$value")
            raw_block_tpl = section.get("block_template")
            explicit_sep = section.get("separator")
        else:
            raw_item_tpl, raw_block_tpl, explicit_sep = "$value", None, None

        item_tpl = self._expander.expand(raw_item_tpl, variables)
        block_tpl = (
            self._expander.expand(raw_block_tpl, variables) if raw_block_tpl else None
        )
        if explicit_sep is not None:
            separator = str(explicit_sep)
        elif inherited_sep is not None:
            separator = inherited_sep
        else:
            separator = ", "

        return variables, item_tpl, block_tpl, separator

    def _extract_items(self, section: Any) -> list:
        if isinstance(section, dict):
            items = self._choices.get_list_values(section)
            return items if items is not None else []
        if isinstance(section, list):
            return section
        return [section]

    def _render_items(
        self,
        items: list,
        variables: dict[str, str],
        item_tpl: str,
        separator: str = ", ",
        inherited_sep: str | None = None,
    ) -> list[str]:
        rendered: list[str] = []
        pending: list[str] = []

        for item in items:
            if isinstance(item, str):
                pending.append(self._expander.expand(item, variables))
                continue

            if pending and (
                self._choices.is_choice_item(item)
                or self._choices.is_chain_item(item)
                or self._choices.is_group_item(item)
            ):
                result = self._resolve_item(item, variables, inherited_sep)
                if result is not None:
                    pending.append(result)
                rendered.append(
                    self._apply_item_template(
                        separator.join(pending), item_tpl, variables
                    )
                )
                pending = []
                continue

            flushed = self._flush_pending(pending, item_tpl, variables, separator)
            if flushed is not None:
                rendered.append(flushed)
            pending = []

            result = self._resolve_item(item, variables, inherited_sep)
            if result is not None:
                rendered.append(self._apply_item_template(result, item_tpl, variables))

        flushed = self._flush_pending(pending, item_tpl, variables, separator)
        if flushed is not None:
            rendered.append(flushed)

        return rendered

    def _apply_templates(
        self,
        is_simple_plain: bool,
        rendered_lines: list[str],
        block_tpl: str | None,
        variables: dict[str, str],
        separator: str = ", ",
    ) -> list[str]:
        if not rendered_lines:
            return []

        if is_simple_plain:
            return [separator.join(rendered_lines)]

        if block_tpl is not None:
            return [
                self._expander.expand(
                    block_tpl.replace("$value", separator.join(rendered_lines)),
                    variables,
                )
            ]

        return [separator.join(rendered_lines)]

    def _resolve_item(
        self,
        item: Any,
        variables: dict[str, str],
        inherited_sep: str | None = None,
    ) -> str | None:
        if self._choices.is_group_item(item):
            return self._choices.resolve_chain(
                self._choices.normalize_block(item),
                variables,
                skip_failed=True,
                inherited_sep=inherited_sep,
            )

        if self._choices.is_chain_item(item):
            return self._choices.resolve_chain(
                self._choices.normalize_block(item),
                variables,
                inherited_sep=inherited_sep,
            )

        if self._choices.is_choice_item(item):
            return self._choices.resolve(
                self._choices.normalize_block(item),
                variables,
                inherited_sep=inherited_sep,
            )

        if isinstance(item, dict) and "name" in item:
            if not self._chance.check(item):
                return None
            return self._expander.expand(str(item["name"]), variables)

        return self._expander.expand(str(item), variables)

    def _collect_vars(
        self,
        raw: dict[str, Any],
        base: dict[str, str],
    ) -> dict[str, str]:
        variables: dict[str, str] = {**base}
        for name, value in raw.items():
            val = self._resolve_item(value, variables)
            if isinstance(val, str):
                val = self._resolve_builtin_call(val)
            variables[name] = val if val is not None else ""
        return variables

    def _apply_item_template(
        self, text: str, item_tpl: str, variables: dict[str, str]
    ) -> str:
        return self._expander.expand(item_tpl.replace("$value", text), variables)

    def _flush_pending(
        self,
        pending: list[str],
        item_tpl: str,
        variables: dict[str, str],
        separator: str = ", ",
    ) -> str | None:
        if not pending:
            return None
        return self._apply_item_template(separator.join(pending), item_tpl, variables)

    def _resolve_builtin_call(self, text: str) -> str:
        match = self.FUNCTION_PATTERN.match(text)
        if match:
            low, high = map(float, match.groups())
            return str(round(self.rng.uniform(low, high), 2))
        return text
