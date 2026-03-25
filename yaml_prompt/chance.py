from __future__ import annotations

import hashlib
import random
from typing import Any

__all__ = ["ChanceEvaluator"]


class ChanceEvaluator:
    """Evaluate probability gates for prompt sections and items.

    Uses SHA-256 hashing for stable (seed-deterministic) chance evaluation,
    or stdlib ``random`` for truly random evaluation.

    Parameters
    ----------
    seed:
        Master seed for deterministic hashing.
    """

    def __init__(self, seed: int) -> None:
        self.seed = seed

    def evaluate(self, raw_chance: float | dict[str, Any], content: Any) -> bool:
        """Evaluate a chance gate (float or dict) against *content*.

        Parameters
        ----------
        raw_chance:
            Raw ``chance`` value from YAML — a number or a dict with
            ``value``, ``stable``, and ``key`` fields.
        content:
            Fallback material for hash-based key derivation when no
            custom ``key`` is provided.  The ``chance`` key is stripped
            automatically so that ``chance: 0.5`` and
            ``chance: {value: 0.5}`` hash identically.

        Returns
        -------
        bool
            ``True`` if the content passes the chance gate.
        """
        chance, stable, custom_key = self._parse_chance(raw_chance)
        if chance >= 1.0:
            return True
        if chance <= 0.0:
            return False
        if stable:
            if isinstance(content, dict) and "chance" in content:
                content = {k: v for k, v in content.items() if k != "chance"}
            return self._stable_chance(chance, content, key=custom_key)
        return self._random_chance(chance)

    def check(self, obj: dict[str, Any]) -> bool:
        """Return ``True`` if *obj* passes its chance gate (or has none)."""
        if "chance" not in obj:
            return True
        return self.evaluate(obj["chance"], obj)

    def apply_to_section(self, section: Any) -> Any | None:
        """Evaluate section-level chance; return cleaned section or ``None``."""
        if not isinstance(section, dict) or "chance" not in section:
            return section
        if not self.evaluate(section["chance"], section):
            return None
        return {k: v for k, v in section.items() if k != "chance"}

    def _stable_chance(
        self,
        chance: float,
        content: Any,
        *,
        key: str | None = None,
    ) -> bool:
        """Hash-based chance — stable across templates with the same seed."""
        if key is not None:
            raw_key = f"{self.seed}:chance:{key}".encode("utf-8")
        else:
            raw_key = f"{self.seed}:chance:{content}".encode("utf-8")
        digest = hashlib.sha256(raw_key).digest()
        roll = int.from_bytes(digest[:8], "big") / (1 << 64)
        return roll <= chance

    @staticmethod
    def _random_chance(chance: float) -> bool:
        """Truly random chance — ignores seed, varies every render."""
        return random.random() <= chance

    def _parse_chance(
        self, raw: float | dict[str, Any]
    ) -> tuple[float, bool, str | None]:
        """Parse a chance value into ``(probability, is_stable, custom_key)``."""
        if isinstance(raw, dict):
            value = self._safe_chance(raw.get("value", 1))
            stable = bool(raw.get("stable", True))
            key: str | None = None
            if stable and "key" in raw:
                key = str(raw["key"])
            return value, stable, key
        return self._safe_chance(raw), True, None

    def _safe_chance(self, value: Any) -> float:
        try:
            chance = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"Invalid chance value: {value!r}")
        return max(0.0, min(1.0, chance))
