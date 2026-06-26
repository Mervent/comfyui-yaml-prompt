"""Extract and strip `<lora:name:weight>` tags from prompt text."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

__all__ = [
    "extract_lora_tags",
    "extract_lora_tags_lbw",
    "strip_lora_tags",
    "LoraEntry",
]

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


@dataclass(frozen=True)
class LoraEntry:
    """Extended lora descriptor with optional LBW (LoRA Block Weight) fields."""

    name: str
    model_weight: float
    clip_weight: float
    lbw: str | None = None
    lbw_a: float | None = None
    lbw_b: float | None = None


def extract_lora_tags_lbw(text: str) -> list[LoraEntry]:
    """Return a deduplicated list of :class:`LoraEntry` with LBW fields.

    Parses Impact Pack-style syntax::

        <lora:name:0.8:LBW=SD-ALL:A=0.5:B=0.3>
    """
    results: list[LoraEntry] = []
    seen: set[str] = set()

    for match in LORA_TAG_PATTERN.finditer(text):
        entry = _parse_lora_match_lbw(match.group(1))
        if entry is None:
            continue
        if entry.name in seen:
            continue
        seen.add(entry.name)
        results.append(entry)

    return results


def _parse_lora_match_lbw(inner: str) -> LoraEntry | None:
    """Parse ``<lora:...>`` content into a :class:`LoraEntry` with LBW fields."""
    parts = inner.split(":")
    name = parts[0].strip()
    if not name:
        return None

    if not _has_model_extension(name):
        name = name + ".safetensors"

    model_weight = 1.0
    clip_weight: float | None = None
    lbw: str | None = None
    lbw_a: float | None = None
    lbw_b: float | None = None

    numeric_weights: list[float] = []
    for part in parts[1:]:
        stripped = part.strip()
        if stripped.startswith("LBW="):
            lbw = stripped[4:]
        elif stripped.startswith("A="):
            lbw_a = _safe_float(stripped[2:])
        elif stripped.startswith("B="):
            lbw_b = _safe_float(stripped[2:])
        elif len(numeric_weights) < 2:
            numeric_weights.append(_safe_float(stripped))

    if len(numeric_weights) >= 2:
        model_weight = numeric_weights[0]
        clip_weight = numeric_weights[1]
    elif len(numeric_weights) == 1:
        model_weight = numeric_weights[0]

    if clip_weight is None:
        clip_weight = model_weight

    return LoraEntry(
        name=name,
        model_weight=model_weight,
        clip_weight=clip_weight,
        lbw=lbw,
        lbw_a=lbw_a,
        lbw_b=lbw_b,
    )


def _parse_lora_match(inner: str) -> tuple[str, float, float] | None:
    """Parse the content inside a `<lora:...>` tag into (name, model_w, clip_w).

    Returns ``None`` for empty names.
    """
    parts = inner.split(":")
    name = parts[0].strip()
    if not name:
        return None

    if not _has_model_extension(name):
        name = name + ".safetensors"

    if len(parts) >= 3:
        return (name, _safe_float(parts[1]), _safe_float(parts[2]))
    if len(parts) == 2:
        model_weight = _safe_float(parts[1])
        return (name, model_weight, model_weight)
    return (name, 1.0, 1.0)


_MODEL_EXTENSIONS: frozenset[str] = frozenset(
    {".safetensors", ".ckpt", ".pt", ".pth", ".bin"}
)


def _has_model_extension(name: str) -> bool:
    """Return ``True`` if *name* already ends with a known model file extension."""
    lower = name.lower()
    return any(lower.endswith(ext) for ext in _MODEL_EXTENSIONS)


def _safe_float(text: str, default: float = 1.0) -> float:
    try:
        value = float(text)
    except ValueError:
        return default
    if not math.isfinite(value):
        return default
    return value
