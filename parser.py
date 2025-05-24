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

DEFAULT_WILDCARD_DIR = Path(__file__).with_name("wildcards")
_WILDCARD_CACHE: dict[tuple[Path, str], list[str]] = {}

aYAML = Any  # Loaded-YAML value (dict, list, str, …)

VARIABLE_PATTERN = re_compile(r"\$([A-Za-z_][A-Za-z0-9_]*)")
BRACE_PATTERN = re_compile(r"\{([^{}]+)\}")  # {a|0.5::b|c}
WILDCARD_PATTERN = re_compile(r"__([A-Za-z0-9_]+)__")
FUNCTION_PATTERN = re_compile(
    r"^rand\(\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\)$"
)

# ---------------------------------------------------------------------------
# Wildcards
# ---------------------------------------------------------------------------


def _load_wildcard(name: str, directory: Path) -> list[str]:
    """Return non-blank lines from *directory/name.txt* (cached)."""
    key = (directory, name)
    if key in _WILDCARD_CACHE:
        return _WILDCARD_CACHE[key]

    file_path = directory / f"{name}.txt"
    try:
        lines = [
            ln.strip()
            for ln in file_path.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
    except FileNotFoundError:
        lines = []
    _WILDCARD_CACHE[key] = lines
    return lines


# ---------------------------------------------------------------------------
# Expression helpers
# ---------------------------------------------------------------------------


def _eval_builtin(text: str) -> str:
    """Evaluate built-in calls (currently only ``rand``)."""
    m = FUNCTION_PATTERN.match(text)
    if m:
        lo, hi = map(float, m.groups())
        return str(round(uniform(lo, hi), 2))
    return text


# ---------------------------------------------------------------------------
# Expansion helpers (variables → braces → wildcards)
# ---------------------------------------------------------------------------


def _subst_vars(text: str, variables: dict[str, str]) -> str:
    def repl(m: Match[str]) -> str:  # noqa: D401
        return variables.get(m.group(1), m.group(0))

    return VARIABLE_PATTERN.sub(repl, text)


def _choose_brace(m: Match[str], variables: dict[str, str], wildcard_dir: Path) -> str:
    parts = [p.strip() for p in m.group(1).split("|") if p.strip()]
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
        opts.append(expand_string(opt_txt, variables, wildcard_dir))
        wgts.append(weight)
    return choices(opts, wgts)[0]


def _subst_wildcards(text: str, wildcard_dir: Path) -> str:
    def repl(m: Match[str]) -> str:  # noqa: D401
        options = _load_wildcard(m.group(1), wildcard_dir)
        return choices(options)[0] if options else m.group(0)

    return WILDCARD_PATTERN.sub(repl, text)


def expand_string(text: str, variables: dict[str, str], wildcard_dir: Path) -> str:
    """Expand `$vars`, brace lists, and wildcards until stable."""
    expanded = _subst_vars(text, variables)

    while True:
        new_text = BRACE_PATTERN.sub(
            lambda m: _choose_brace(m, variables, wildcard_dir), expanded
        )
        new_text = _subst_wildcards(new_text, wildcard_dir)
        if new_text == expanded:
            return new_text.strip()
        expanded = new_text


# ---------------------------------------------------------------------------
# choice/oneOf handling
# ---------------------------------------------------------------------------


def _resolve_choice(
    block: dict[str, Any], variables: dict[str, str], wildcard_dir: Path
) -> str | None:
    if random() > float(block.get("chance", 1)):
        return None

    template = block.get("template", "$value")
    options = next((block[k] for k in LIST_KEYS if k in block), None)
    if options is None:
        raise ValueError("choice/oneOf requires 'values', 'options', or 'choices'.")

    texts, weights = [], []
    for opt in options:
        if isinstance(opt, dict):
            if random() > float(opt.get("chance", 1)):
                continue
            name, weight = opt.get("name", ""), float(opt.get("weight", 1))
        else:
            name, weight = opt, 1.0
        texts.append(expand_string(str(name), variables, wildcard_dir))
        weights.append(weight)

    if not texts:
        return None

    chosen = choices(texts, weights)[0]
    return expand_string(template.replace("$value", chosen), variables, wildcard_dir)


# ---------------------------------------------------------------------------
# Item evaluation
# ---------------------------------------------------------------------------


def _eval_item(
    item: aYAML, variables: dict[str, str], wildcard_dir: Path
) -> str | None:
    # Wrapper shorthand
    if isinstance(item, dict) and len(item) == 1 and next(iter(item)) in CHOICE_KEYS:
        key = next(iter(item))
        block = item[key]
        if not isinstance(block, dict):
            block = {"values": block}
        return _resolve_choice(block, variables, wildcard_dir)

    # Direct mapping with choice keys
    if isinstance(item, dict) and any(k in item for k in CHOICE_KEYS):
        return _resolve_choice(item, variables, wildcard_dir)

    # Named entry with chance
    if isinstance(item, dict) and "name" in item:
        if random() > float(item.get("chance", 1)):
            return None
        return expand_string(str(item["name"]), variables, wildcard_dir)

    # Plain string
    if isinstance(item, str):
        return expand_string(item, variables, wildcard_dir)

    return expand_string(str(item), variables, wildcard_dir)


# ---------------------------------------------------------------------------
# Variable collection
# ---------------------------------------------------------------------------


def _collect_vars(
    raw: dict[str, aYAML], base: dict[str, str], wildcard_dir: Path
) -> dict[str, str]:
    vars_: dict[str, str] = dict(base)
    for name, value in raw.items():
        val = _eval_item(value, vars_, wildcard_dir)
        if isinstance(val, str):
            val = _eval_builtin(val)
        vars_[name] = val or ""
    return vars_


# ---------------------------------------------------------------------------
# Section parsing
# ---------------------------------------------------------------------------


def _parse_section(
    section: aYAML, variables: dict[str, str], wildcard_dir: Path
) -> list[str]:
    if section is None:
        return []

    # Section-local vars
    if isinstance(section, dict) and "vars" in section:
        variables = _collect_vars(section["vars"], variables, wildcard_dir)

    # Templates
    if isinstance(section, dict):
        raw_item_tpl = section.get("template", "$value")
        raw_block_tpl = section.get("block_template")
    else:
        raw_item_tpl, raw_block_tpl = "$value", None

    item_tpl = expand_string(raw_item_tpl, variables, wildcard_dir)
    block_tpl = (
        expand_string(raw_block_tpl, variables, wildcard_dir) if raw_block_tpl else None
    )

    # List items
    if isinstance(section, dict):
        list_items: Iterable[aYAML] | None = next(
            (section[k] for k in LIST_KEYS if k in section), None
        )
        if list_items is None:
            list_items = []
    elif isinstance(section, list):
        list_items = section
    else:
        list_items = [section]

    merged, buffer = [], []

    def flush() -> None:
        if buffer:
            merged_txt = ", ".join(buffer)
            merged.append(
                expand_string(
                    item_tpl.replace("$value", merged_txt), variables, wildcard_dir
                )
            )
            buffer.clear()

    for itm in list_items:
        if isinstance(itm, str):
            buffer.append(expand_string(itm, variables, wildcard_dir))
            continue

        is_choice = isinstance(itm, dict) and (
            (len(itm) == 1 and next(iter(itm)) in CHOICE_KEYS)
            or any(k in itm for k in CHOICE_KEYS)
        )
        if buffer and is_choice:
            ch = _eval_item(itm, variables, wildcard_dir)
            if ch is not None:
                buffer.append(ch)
            flush()
            continue

        flush()
        ev = _eval_item(itm, variables, wildcard_dir)
        if ev is not None:
            merged.append(
                expand_string(item_tpl.replace("$value", ev), variables, wildcard_dir)
            )

    flush()

    simple_plain = (
        all(isinstance(e, str) for e in list_items)
        and item_tpl == "$value"
        and block_tpl is None
    )
    if simple_plain:
        return [", ".join(merged)]

    if block_tpl is not None:
        return [
            expand_string(
                block_tpl.replace("$value", ", ".join(merged)), variables, wildcard_dir
            )
        ]

    return merged


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_document(
    doc: dict[str, aYAML], *, wildcard_dir: Path | str | None = None
) -> list[list[str]]:
    """Flatten *doc* into blocks of prompt lines.

    Parameters
    ----------
    doc
        YAML mapping as returned by :pyfunc:`yaml.safe_load`.
    wildcard_dir
        Path (or str) to a directory containing wildcard ``*.txt`` files. If
        *None* the default ``<module>/wildcards`` directory is used. Relative
        paths are resolved to absolute.
    """
    wd_path = (
        Path(wildcard_dir).expanduser().resolve()
        if wildcard_dir
        else DEFAULT_WILDCARD_DIR
    )

    variables = _collect_vars(doc.get("vars", {}), {}, wd_path)
    blocks: list[list[str]] = []

    for name, section in doc.items():
        if name == "vars":
            continue
        lines = _parse_section(section, variables, wd_path)
        if lines:
            blocks.append(lines)

    return blocks


# ---------------------------------------------------------------------------
# CLI helper (for manual testing)
# ---------------------------------------------------------------------------


def main() -> None:  # noqa: D401
    ap = ArgumentParser(description="Flatten YAML prompt files into prompt lines.")
    ap.add_argument("file", type=Path, help="YAML prompt definition")
    ap.add_argument(
        "--wildcards-dir",
        type=Path,
        default=DEFAULT_WILDCARD_DIR,
        help="Folder with *.txt wildcards",
    )
    args = ap.parse_args()

    try:
        content = args.file.read_text(encoding="utf-8")
    except OSError as err:
        ap.error(f"Cannot read '{args.file}': {err}")

    try:
        data: dict[str, aYAML] = yaml.safe_load(content) or {}
    except yaml.YAMLError as err:
        ap.error(f"YAML error: {err}")

    blocks = parse_document(data, wildcard_dir=args.wildcards_dir)

    for i, blk in enumerate(blocks):
        print(*blk, sep="\n")
        if i != len(blocks) - 1:
            print()


if __name__ == "__main__":
    main()
