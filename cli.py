"""CLI entry point for the YAML prompt template parser."""

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from jinja_env import render_template
from parser import YAMLPromptTemplateParser


def _parse_var_value(value: str) -> Any:
    """Auto-convert a CLI ``--var`` value to bool / int / float / JSON / str."""
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
    """Parse a YAML prompt file and print flattened prompt lines."""
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
        data: dict[str, Any] = yaml.safe_load(rendered) or {}
    except yaml.YAMLError as err:
        ap.error(f"YAML error: {err}")

    flattener = YAMLPromptTemplateParser(
        seed=args.seed,
        wildcard_dir=args.wildcards_dir,
    )
    blocks = flattener.parse_document(data)

    for i, block in enumerate(blocks):
        print(*block, sep="\n")
        if i != len(blocks) - 1:
            print()


if __name__ == "__main__":
    main()
