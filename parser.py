from __future__ import annotations

"""Prompt‑template YAML flattener
=================================

Features
--------
* **Global & section‑local `vars:`** – reference with ``$name``.
* **Wildcards** – any ``__name__`` token is replaced by a random line from
  ``wildcards/name.txt`` (blank lines ignored). Wildcards work inside
  variables, templates, braces, anywhere.
* **Built‑ins** – currently ``rand(min, max)`` for one‑shot random floats.
* **`choice` / `oneOf` blocks** – with per‑option weight & chance.
* **Brace lists** – inline picks like ``{foo|0.5::bar|baz}``.
* **Templates** – `template:` wraps each item; `block_template:` wraps the
  whole section. Both honour variables & wildcards.
* **Comma merging** – consecutive plain strings are merged to keep prompts
  compact.

CLI Usage
---------
```
python prompt_parser.py prompt.yaml
```
Prints each top‑level section (``meta:``, ``plot:``, …) as a paragraph ready for
Stable‑Diffusion style prompts.
"""

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

_WILDCARD_DIR = Path(__file__).with_name("wildcards")
_WILDCARD_CACHE: dict[str, list[str]] = {}

aYAML = Any  # Loaded‑YAML value (dict, list, str, …)

VARIABLE_PATTERN = re_compile(r"\$([A-Za-z_][A-Za-z0-9_]*)")
BRACE_PATTERN = re_compile(r"\{([^{}]+)\}")  # {a|0.5::b|c}
WILDCARD_PATTERN = re_compile(r"__([A-Za-z0-9_]+)__")
FUNCTION_PATTERN = re_compile(
    r"^rand\(\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\)$"
)

# ---------------------------------------------------------------------------
# Wildcard helpers
# ---------------------------------------------------------------------------


def _load_wildcard(name: str) -> list[str]:
    """Return non‑blank lines from *wildcards/name.txt* (cached)."""
    if name in _WILDCARD_CACHE:
        return _WILDCARD_CACHE[name]

    path = _WILDCARD_DIR / f"{name}.txt"
    try:
        lines = [
            ln.strip()
            for ln in path.read_text(encoding="utf‑8").splitlines()
            if ln.strip()
        ]
    except FileNotFoundError:
        lines = []
    _WILDCARD_CACHE[name] = lines
    return lines


def substitute_wildcards(text: str) -> str:
    """Replace every ``__name__`` with a random line from its wildcard file."""

    def _replace(match: Match[str]) -> str:  # noqa: D401
        options = _load_wildcard(match.group(1))
        return choices(options)[0] if options else match.group(0)

    return WILDCARD_PATTERN.sub(_replace, text)


# ---------------------------------------------------------------------------
# Expression & variable helpers
# ---------------------------------------------------------------------------


def evaluate_expression(text: str) -> str:
    """Evaluate supported built‑ins such as ``rand(0, 1)``."""
    m = FUNCTION_PATTERN.match(text)
    if m:
        lo, hi = map(float, m.groups())
        return str(round(uniform(lo, hi), 2))
    return text


def substitute_variables(text: str, variables: dict[str, str]) -> str:
    """Replace each ``$name`` with the corresponding value."""

    def _replace(match: Match[str]) -> str:  # noqa: D401
        return variables.get(match.group(1), match.group(0))

    return VARIABLE_PATTERN.sub(_replace, text)


# ---------------------------------------------------------------------------
# Recursive string expansion (variables → braces → wildcards)
# ---------------------------------------------------------------------------


def choose_from_brace(match: Match[str], variables: dict[str, str]) -> str:
    parts = [p.strip() for p in match.group(1).split("|") if p.strip()]
    opts, wgts = [], []
    for part in parts:
        if "::" in part:
            w_txt, opt_txt = part.split("::", 1)
            try:
                weight = float(w_txt)
            except ValueError:
                weight, opt_txt = 1.0, part
        else:
            weight, opt_txt = 1.0, part
        opts.append(expand_string(opt_txt, variables))
        wgts.append(weight)
    return choices(opts, wgts)[0]


def expand_string(text: str, variables: dict[str, str]) -> str:
    """Expand `$vars`, brace lists, and wildcards until the text stabilises."""
    expanded = substitute_variables(text, variables)

    while True:
        new_text = BRACE_PATTERN.sub(
            lambda m: choose_from_brace(m, variables), expanded
        )
        new_text = substitute_wildcards(new_text)
        if new_text == expanded:
            return new_text.strip()
        expanded = new_text


# ---------------------------------------------------------------------------
# choice/oneOf handling
# ---------------------------------------------------------------------------


def resolve_choice_block(
    block: dict[str, Any], variables: dict[str, str]
) -> str | None:
    if random() > float(block.get("chance", 1)):
        return None

    template = block.get("template", "$value")
    options = next((block[k] for k in LIST_KEYS if k in block), None)
    if options is None:
        raise ValueError(
            "choice/oneOf block requires 'values', 'options', or 'choices'."
        )

    texts, weights = [], []
    for opt in options:
        if isinstance(opt, dict):
            if random() > float(opt.get("chance", 1)):
                continue
            name, weight = opt.get("name", ""), float(opt.get("weight", 1))
        else:
            name, weight = opt, 1.0
        texts.append(expand_string(str(name), variables))
        weights.append(weight)

    if not texts:
        return None

    chosen = choices(texts, weights)[0]
    return expand_string(template.replace("$value", chosen), variables)


# ---------------------------------------------------------------------------
# Item evaluation (single YAML node → str)
# ---------------------------------------------------------------------------


def evaluate_item(item: aYAML, variables: dict[str, str]) -> str | None:
    # Wrapper shorthand: {choice: ...}
    if isinstance(item, dict) and len(item) == 1 and next(iter(item)) in CHOICE_KEYS:
        key = next(iter(item))
        block = item[key]
        if not isinstance(block, dict):
            block = {"values": block}
        return resolve_choice_block(block, variables)

    # Direct mapping containing choice/oneOf keys
    if isinstance(item, dict) and any(k in item for k in CHOICE_KEYS):
        return resolve_choice_block(item, variables)

    # Named entry with chance (weight ignored here)
    if isinstance(item, dict) and "name" in item:
        if random() > float(item.get("chance", 1)):
            return None
        return expand_string(str(item["name"]), variables)

    # Plain string
    if isinstance(item, str):
        return expand_string(item, variables)

    # Fallback: stringify other YAML types
    return expand_string(str(item), variables)


# ---------------------------------------------------------------------------
# Variable collection
# ---------------------------------------------------------------------------


def collect_variables(
    raw: dict[str, aYAML], base: dict[str, str] | None = None
) -> dict[str, str]:
    vars_: dict[str, str] = dict(base or {})
    for name, value in raw.items():
        expanded = evaluate_item(value, vars_)
        if isinstance(expanded, str):
            expanded = evaluate_expression(expanded)
        vars_[name] = expanded or ""
    return vars_


# ---------------------------------------------------------------------------
# Section parsing
# ---------------------------------------------------------------------------


def parse_section(section: aYAML, variables: dict[str, str]) -> list[str]:
    if section is None:
        return []

    # Section‑local vars
    if isinstance(section, dict) and "vars" in section:
        variables = collect_variables(section["vars"], variables)

    # Templates
    if isinstance(section, dict):
        raw_item_tpl = section.get("template", "$value")
        raw_block_tpl = section.get("block_template")
    else:
        raw_item_tpl, raw_block_tpl = "$value", None

    item_tpl = expand_string(raw_item_tpl, variables)
    block_tpl = expand_string(raw_block_tpl, variables) if raw_block_tpl else None

    # List items
    if isinstance(section, dict):
        list_items: Iterable[aYAML] | None = next(
            (section[k] for k in LIST_KEYS if k in section), None
        )
        if list_items is None:
            list_items = []  # dict only holds templates/vars
    elif isinstance(section, list):
        list_items = section
    else:
        list_items = [section]

    merged, buffer = [], []

    def flush() -> None:
        if buffer:
            merged_txt = ", ".join(buffer)
            merged.append(
                expand_string(item_tpl.replace("$value", merged_txt), variables)
            )
            buffer.clear()

    for itm in list_items:
        if isinstance(itm, str):
            buffer.append(expand_string(itm, variables))
            continue

        is_choice = isinstance(itm, dict) and (
            (len(itm) == 1 and next(iter(itm)) in CHOICE_KEYS)
            or any(k in itm for k in CHOICE_KEYS)
        )
        if buffer and is_choice:
            ch = evaluate_item(itm, variables)
            if ch is not None:
                buffer.append(ch)
            flush()
            continue

        flush()
        ev = evaluate_item(itm, variables)
        if ev is not None:
            merged.append(expand_string(item_tpl.replace("$value", ev), variables))

    flush()

    simple_plain = (
        all(isinstance(entry, str) for entry in list_items)
        and item_tpl == "$value"
        and block_tpl is None
    )
    if simple_plain:
        return [", ".join(merged)]

    if block_tpl is not None:
        return [
            expand_string(block_tpl.replace("$value", ", ".join(merged)), variables)
        ]

    return merged


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Document parsing
# ---------------------------------------------------------------------------


def parse_document(doc: dict[str, aYAML]) -> list[list[str]]:
    variables = collect_variables(doc.get("vars", {}))
    blocks: list[list[str]] = []
    for sec_name, sec_val in doc.items():
        if sec_name == "vars":
            continue
        lines = parse_section(sec_val, variables)
        if lines:
            blocks.append(lines)
    return blocks


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:  # noqa: D401 – concise name for __main__
    parser = ArgumentParser(description="Flatten YAML prompt files into prompt lines.")
    parser.add_argument("file", type=Path, help="Path to the YAML prompt definition")
    args = parser.parse_args()

    try:
        yaml_text = args.file.read_text(encoding="utf‑8")
    except OSError as err:
        parser.error(f"Unable to read '{args.file}': {err}")

    try:
        yaml_data: dict[str, aYAML] = yaml.safe_load(yaml_text) or {}
    except yaml.YAMLError as err:
        parser.error(f"YAML parsing error: {err}")

    blocks = parse_document(yaml_data)

    for i, block in enumerate(blocks):
        print(*block, sep="")
        if i != len(blocks) - 1:
            print()  # blank line between sections


if __name__ == "__main__":
    main()
