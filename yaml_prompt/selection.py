"""Shared deterministic selection primitives.

Both the Jinja2 preprocessing layer and the YAML parser need identical
hash-based selection to guarantee cross-layer agreement (same seed +
same items = same result).  Centralising the logic here eliminates
duplication and makes the contract structural rather than coincidental.
"""

from __future__ import annotations

import hashlib

__all__ = ["stable_select", "seed_derived_index"]


def stable_select(
    seed: int,
    items: list[str],
    weights: list[float] | None = None,
    *,
    key: str | None = None,
) -> str:
    """Pick from *items* using SHA-256(seed + joined items).

    The result is stable across independent callers sharing the same
    seed, regardless of any ``random.Random`` state.

    Parameters
    ----------
    seed:
        Master seed for deterministic hashing.
    items:
        Non-empty list of candidate strings.
    weights:
        Optional per-item weights.  When all weights are equal (or
        ``None``), a uniform selection is used.
    key:
        Optional custom hash key.  When provided, replaces the default
        ``|``-joined items in the hash, enabling correlated selections
        across blocks with different item lists.

    Returns
    -------
    str
        The selected item.
    """
    if key is not None:
        hash_input = f"{seed}:choice:{key}".encode("utf-8")
    else:
        hash_input = f"{seed}:choice:{'|'.join(items)}".encode("utf-8")
    digest = hashlib.sha256(hash_input).digest()

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


def seed_derived_index(seed: int, name: str, n: int) -> int:
    """Return an index in ``[0, n)`` derived from *seed* and *name* via SHA-256.

    Used by both wildcard resolution layers so the Jinja2 ``wildcard()``
    global and the YAML parser ``__name__`` syntax always agree.

    Parameters
    ----------
    seed:
        Master seed.
    name:
        Wildcard identifier (e.g. ``"colors"``).
    n:
        Size of the candidate list.

    Returns
    -------
    int
        Deterministic index in ``[0, n)``.
    """
    key = f"{seed}:{name}".encode("utf-8")
    digest = hashlib.sha256(key).digest()
    return int.from_bytes(digest[:8], "big") % n
