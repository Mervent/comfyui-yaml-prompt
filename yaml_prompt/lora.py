"""Extract and strip `<lora:name:weight>` tags from prompt text."""

from __future__ import annotations

import math
import re

__all__ = ["extract_lora_tags", "strip_lora_tags"]

LORA_TAG_PATTERN = re.compile(r"<lora:([^>]+)>")


def extract_lora_tags(text: str) -> list[tuple[str, float, float]]:
    """Return a deduplicated list of ``(name, model_weight, clip_weight)`` tuples.

    Duplicate lora names keep the first occurrence only.
    """
    results: list[tuple[str, float, float]] = []
    seen: set[str] = set()

    for match in LORA_TAG_PATTERN.finditer(text):
        parsed = _parse_lora_match(match.group(1))
        if parsed is None:
            continue
        name = parsed[0]
        if name in seen:
            continue
        seen.add(name)
        results.append(parsed)

    return results


def strip_lora_tags(text: str) -> str:
    """Remove all ``<lora:...>`` tags and collapse resulting whitespace."""
    stripped = LORA_TAG_PATTERN.sub("", text)
    return re.sub(r"  +", " ", stripped).strip()


def _parse_lora_match(inner: str) -> tuple[str, float, float] | None:
    """Parse the content inside a `<lora:...>` tag into (name, model_w, clip_w).

    Returns ``None`` for empty names.
    """
    parts = inner.split(":")
    name = parts[0].strip()
    if not name:
        return None

    if len(parts) >= 3:
        return (name, _safe_float(parts[1]), _safe_float(parts[2]))
    if len(parts) == 2:
        model_weight = _safe_float(parts[1])
        return (name, model_weight, model_weight)
    return (name, 1.0, 1.0)


def _safe_float(text: str, default: float = 1.0) -> float:
    try:
        value = float(text)
    except ValueError:
        return default
    if not math.isfinite(value):
        return default
    return value
