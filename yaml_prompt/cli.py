"""CLI entry point for the YAML prompt template parser."""

import argparse
import json
from pathlib import Path
from typing import Any

__all__ = ["main"]

from .pipeline import PipelineError, process_file


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
        result = process_file(
            args.file,
            seed=args.seed,
            wildcard_dir=args.wildcards_dir,
            jinja_vars=jinja_vars or None,
        )
    except PipelineError as error:
        ap.error(str(error))

    for i, block in enumerate(result.blocks):
        print(*block, sep="\n")
        if i != len(result.blocks) - 1:
            print()


def _parse_var_value(value: str) -> Any:
    for convert in _CONVERTERS:
        result = convert(value)
        if result is not None:
            return result
    return value


def _to_bool(value: str) -> bool | None:
    lower = value.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    return None


def _to_int(value: str) -> int | None:
    try:
        return int(value)
    except ValueError:
        return None


def _to_float(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def _to_json_collection(value: str) -> Any:
    try:
        parsed = json.loads(value)
        if isinstance(parsed, (list, dict)):
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass
    return None


_CONVERTERS = (_to_bool, _to_int, _to_float, _to_json_collection)


if __name__ == "__main__":
    main()
