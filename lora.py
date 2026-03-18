"""Extract and strip `<lora:name:weight>` tags from prompt text."""

from __future__ import annotations

import re

LORA_TAG_PATTERN = re.compile(r"<lora:([^>]+)>")


def _parse_lora_match(inner: str) -> tuple[str, float, float] | None:
    """Parse the content inside a `<lora:...>` tag into (name, model_w, clip_w).

    Returns ``None`` for empty names.
    """
    parts = inner.split(":")
    name = parts[0].strip()
    if not name:
        return None

    model_weight = 1.0
    clip_weight = 1.0

    if len(parts) >= 2:
        try:
            model_weight = float(parts[1])
        except ValueError:
            model_weight = 1.0

    if len(parts) >= 3:
        try:
            clip_weight = float(parts[2])
        except ValueError:
            clip_weight = 1.0
    elif len(parts) == 2:
        clip_weight = model_weight

    return (name, model_weight, clip_weight)


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
