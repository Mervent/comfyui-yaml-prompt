"""Prompt‑template YAML flattener with section‑local variables and built‑ins.

Features
--------
*   **Global and section‑level `vars:`** — string values can reference
    previously defined variables with ``$name``.
*   **Built‑in functions** — currently ``rand(min, max)`` for a one‑shot random
    float; easy to extend with more helpers.
*   **`choice` / `oneOf` blocks** — support weights and per‑option chance.
*   **Brace lists** — inline random picks such as ``{a|0.5::b|c}``.
*   **`template` / `block_template`** — with variable expansion.
*   **Comma‑merging** — consecutive plain strings are merged into a single line
    for compact prompts.

Usage
-----
```
python prompt_parser.py prompt.yaml
```
Prints each top‑level section (``meta:``, ``plot:``, …) as a separate paragraph
ready for Stable‑Diffusion style prompts.
"""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from random import choices, random, uniform
from re import Match
from re import compile as re_compile
from typing import Any, Final, Iterable, Sequence

import yaml

# ---------------------------------------------------------------------------
# Constants & type aliases
# ---------------------------------------------------------------------------

LIST_KEYS: Final[Sequence[str]] = ("values", "options", "choices")
CHOICE_KEYS: Final[Sequence[str]] = ("choice", "oneOf")

aYAML = Any  # Alias for readability: arbitrary loaded‑YAML structure

VARIABLE_PATTERN = re_compile(r"\$([A-Za-z_][A-Za-z0-9_]*)")
BRACE_PATTERN = re_compile(r"\{([^{}]+)\}")  # e.g. {a|0.5::b|c}


# Built-in call pattern (currently only rand)
FUNCTION_PATTERN = re_compile(
    r"^rand\(\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\)$"
)

# ---------------------------------------------------------------------------
# Simple expression helpers
# ---------------------------------------------------------------------------


def evaluate_expression(text: str) -> str:
    """Evaluate supported built-ins such as ``rand(min, max)``."""
    m = FUNCTION_PATTERN.match(text)
    if m:
        lo, hi = map(float, m.groups())
        return str(round(uniform(lo, hi), 2))
    return text


# ---------------------------------------------------------------------------
# Low‑level expansion helpers
# ---------------------------------------------------------------------------


def substitute_variables(text: str, variables: dict[str, str]) -> str:
    """Replace "$name" occurrences with their corresponding values."""

    def _replace(match: Match[str]) -> str:  # noqa: D401
        return variables.get(match.group(1), match.group(0))

    return VARIABLE_PATTERN.sub(_replace, text)


def choose_from_brace(match: Match[str], variables: dict[str, str]) -> str:
    """Resolve a single brace expression like "{a|0.3::b|c}"."""
    parts = [part.strip() for part in match.group(1).split("|") if part.strip()]
    options: list[str] = []
    weights: list[float] = []

    for part in parts:
        if "::" in part:
            weight_text, option_text = part.split("::", 1)
            try:
                weight = float(weight_text)
            except ValueError:  # malformed weight – treat entire string
                weight, option_text = 1.0, part
        else:
            weight, option_text = 1.0, part

        expanded_option = expand_string(option_text, variables)
        options.append(expanded_option)
        weights.append(weight)

    return choices(options, weights)[0]


def expand_string(text: str, variables: dict[str, str]) -> str:
    """Recursively expand variables and brace choices within *text*."""
    # First pass: variables
    expanded = substitute_variables(text, variables)

    # Repeatedly replace braces until the text stabilises
    while True:
        new_expanded = BRACE_PATTERN.sub(
            lambda m: choose_from_brace(m, variables), expanded
        )
        if new_expanded == expanded:
            return new_expanded.strip()
        expanded = new_expanded


# ---------------------------------------------------------------------------
# Choice / oneOf blocks
# ---------------------------------------------------------------------------


def resolve_choice_block(
    block: dict[str, Any], variables: dict[str, str]
) -> str | None:
    """Evaluate a *choice* or *oneOf* mapping, returning the selected string."""
    if random() > float(block.get("chance", 1)):
        return None  # Block skipped due to chance

    template = block.get("template", "$value")
    options_list: list[Any] | None = next(
        (block[key] for key in LIST_KEYS if key in block), None
    )
    if options_list is None:
        message = (
            "choice/oneOf block must contain 'values', 'options', or 'choices' list."
        )
        raise ValueError(message)

    option_texts: list[str] = []
    option_weights: list[float] = []

    for option in options_list:
        # Option can be a plain string or a mapping with extra fields
        if isinstance(option, dict):
            if random() > float(option.get("chance", 1)):
                continue  # Option skipped by chance
            name = option.get("name", "")
            weight = float(option.get("weight", 1))
        else:
            name, weight = option, 1.0

        expanded_name = expand_string(str(name), variables)
        option_texts.append(expanded_name)
        option_weights.append(weight)

    if not option_texts:
        return None  # All options were skipped

    chosen_text = choices(option_texts, option_weights)[0]
    return expand_string(template.replace("$value", chosen_text), variables)


# ---------------------------------------------------------------------------
# List‑item evaluation
# ---------------------------------------------------------------------------


def evaluate_item(item: aYAML, variables: dict[str, str]) -> str | None:
    """Return a fully expanded string for *item*, or *None* if skipped."""
    # Wrapper mapping: {choice: {...}} or {oneOf: {...}}
    if isinstance(item, dict) and len(item) == 1 and next(iter(item)) in CHOICE_KEYS:
        wrapper_key = next(iter(item))
        block = item[wrapper_key]
        if not isinstance(block, dict):  # shorthand list form
            block = {"values": block}
        return resolve_choice_block(block, variables)

    # Direct mapping that already contains choice/oneOf fields
    if isinstance(item, dict) and any(key in item for key in CHOICE_KEYS):
        return resolve_choice_block(item, variables)

    # Named value with optional chance/weight (weight ignored here)
    if isinstance(item, dict) and "name" in item:
        if random() > float(item.get("chance", 1)):
            return None
        return expand_string(str(item["name"]), variables)

    # Plain string
    if isinstance(item, str):
        return expand_string(item, variables)

    # Fallback: any other YAML type → stringify
    return expand_string(str(item), variables)


# ---------------------------------------------------------------------------
# Section parsing
# ---------------------------------------------------------------------------


def parse_section(section_data: aYAML, variables: dict[str, str]) -> list[str]:
    """Convert one top‑level section into a list of output lines."""
    if section_data is None:
        return []

    # Section-local variables (shadow globals)
    if isinstance(section_data, dict) and "vars" in section_data:
        variables = collect_variables(section_data["vars"], variables)
    # Section-level templates – expand variables immediately
    if isinstance(section_data, dict):
        raw_item_tpl = section_data.get("template", "$value")
        raw_blk_tpl = section_data.get("block_template")
    else:
        raw_item_tpl, raw_blk_tpl = "$value", None

    item_template = expand_string(raw_item_tpl, variables)
    block_template = (
        expand_string(raw_blk_tpl, variables) if raw_blk_tpl is not None else None
    )

    # Extract list of items
    if isinstance(section_data, dict):
        list_items: Iterable[aYAML] | None = next(
            (section_data[key] for key in LIST_KEYS if key in section_data),
            None,
        )
        if list_items is None:
            list_items = [section_data]  # The dict itself is the single item
    elif isinstance(section_data, list):
        list_items = section_data
    else:
        # Dict with only template/vars/etc. → no list content
        list_items = [section_data] if not isinstance(section_data, dict) else []

    # Merge consecutive plain strings (plus the first trailing choice)
    merged_lines: list[str] = []
    plain_buffer: list[str] = []

    def flush_buffer() -> None:
        if plain_buffer:
            merged = ", ".join(plain_buffer)
            merged_lines.append(
                expand_string(item_template.replace("$value", merged), variables)
            )
            plain_buffer.clear()

    for list_item in list_items:
        if isinstance(list_item, str):
            plain_buffer.append(expand_string(list_item, variables))
            continue

        # Merge the *first* choice/oneOf following a run of plain strings
        is_choice_dict = isinstance(list_item, dict) and (
            (len(list_item) == 1 and next(iter(list_item)) in CHOICE_KEYS)
            or any(key in list_item for key in CHOICE_KEYS)
        )
        if plain_buffer and is_choice_dict:
            chosen_text = evaluate_item(list_item, variables)
            if chosen_text is not None:
                plain_buffer.append(chosen_text)
            flush_buffer()
            continue

        flush_buffer()
        expanded_item = evaluate_item(list_item, variables)
        if expanded_item is not None:
            merged_lines.append(
                expand_string(item_template.replace("$value", expanded_item), variables)
            )

    flush_buffer()

    # Simple comma‑section: every original entry was a plain string, no templates
    simple_comma_section = (
        all(isinstance(entry, str) for entry in list_items)
        and item_template == "$value"
        and block_template is None
    )
    if simple_comma_section:
        return [", ".join(merged_lines)]

    if block_template is not None:
        # The whole section is wrapped once, joined with commas
        wrapped = expand_string(
            block_template.replace("$value", ", ".join(merged_lines)), variables
        )
        return [wrapped]

    return merged_lines


# ---------------------------------------------------------------------------
# Variable collection & document parsing
# ---------------------------------------------------------------------------


def collect_variables(
    raw_variables: dict[str, aYAML],
    base: dict[str, str] | None = None,
) -> dict[str, str]:
    """Resolve a ``vars:`` mapping, extending *base* if given."""
    resolved: dict[str, str] = dict(base or {})
    for name, value in raw_variables.items():
        expanded = evaluate_item(value, resolved)  # supports nested choices
        if isinstance(expanded, str):  # apply built-ins
            expanded = evaluate_expression(expanded)
        resolved[name] = expanded or ""
    return resolved


def parse_document(document: dict[str, aYAML]) -> list[list[str]]:
    """Convert the YAML document into blocks of output lines."""
    variables = collect_variables(document.get("vars", {}))

    blocks: list[list[str]] = []
    for section_name, section_value in document.items():
        if section_name == "vars":
            continue
        section_lines = parse_section(section_value, variables)
        if section_lines:
            blocks.append(section_lines)
    return blocks


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:  # noqa: D401: keep name concise for __main__ guard
    parser = ArgumentParser(description="Flatten YAML prompt files into prompt lines.")
    parser.add_argument("file", type=Path, help="Path to the YAML prompt definition")
    args = parser.parse_args()

    try:
        yaml_text = args.file.read_text(encoding="utf-8")
    except OSError as error:
        parser.error(f"Unable to read '{args.file}': {error}")

    try:
        yaml_data: dict[str, aYAML] = yaml.safe_load(yaml_text) or {}
    except yaml.YAMLError as error:
        parser.error(f"YAML parsing error: {error}")

    prompt_blocks = parse_document(yaml_data)

    for index, block in enumerate(prompt_blocks):
        print(*block, sep="\n")
        if index != len(prompt_blocks) - 1:
            print()  # blank line between top-level sections


if __name__ == "__main__":
    main()
