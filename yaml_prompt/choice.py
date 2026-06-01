from __future__ import annotations

import random
from collections.abc import Callable
from typing import Any, Final, Sequence

from .chance import ChanceEvaluator
from .selection import stable_select

__all__ = ["ChoiceResolver"]


class ChoiceResolver:
    """Resolve ``choice``/``oneOf`` blocks into selected prompt text.

    Parameters
    ----------
    seed:
        Master seed for deterministic selection.
    chance:
        Evaluator for per-item and per-block chance gates.
    expand_fn:
        Callback for string expansion (``$vars``, braces, wildcards).
    """

    VALUES_KEYS: Final[Sequence[str]] = ("values", "options", "choices")
    CHOICE_KEYS: Final[Sequence[str]] = ("choice", "oneOf")
    MAX_CHOICE_DEPTH: Final[int] = 16

    def __init__(
        self,
        seed: int,
        chance: ChanceEvaluator,
        expand_fn: Callable[[str, dict[str, str]], str],
        *,
        rng: random.Random | None = None,
    ) -> None:
        self.seed = seed
        self._chance = chance
        self._expand = expand_fn
        self._rng = rng or random.Random()

    def resolve(
        self,
        block: dict[str, Any],
        variables: dict[str, str],
        _depth: int = 0,
    ) -> str | None:
        """Pick one option from a choice/oneOf block, applying weights and chance."""
        if not self._chance.check(block):
            return None

        template = block.get("template", "$value")
        options = self.get_list_values(block)
        if options is None:
            raise ValueError("choice/oneOf requires 'values', 'options', or 'choices'.")

        texts, weights = self._filter_and_weigh_options(
            options, variables, _depth=_depth
        )
        if not texts:
            return None

        stable = block.get("stable", True)
        custom_key = block.get("key")

        if stable:
            chosen = stable_select(self.seed, texts, weights, key=custom_key)
        else:
            chosen = self._random_select(texts, weights)

        return self._expand(template.replace("$value", chosen), variables)

    def is_choice_item(self, item: Any) -> bool:
        if not isinstance(item, dict):
            return False
        if len(item) == 1:
            first_key = next(iter(item))
            if first_key in self.CHOICE_KEYS:
                return True
        return any(k in item for k in self.CHOICE_KEYS)

    def normalize_block(self, item: dict) -> dict:
        if len(item) != 1:
            return item
        block = item[next(iter(item))]
        if not isinstance(block, dict):
            return {"values": block}
        return block

    def get_list_values(self, block: dict) -> list | None:
        for key in self.VALUES_KEYS:
            if key in block:
                return block[key]
        return None

    def _filter_and_weigh_options(
        self,
        options: list,
        variables: dict[str, str],
        _depth: int = 0,
    ) -> tuple[list[str], list[float]]:
        """Filter *options* by per-item chance, return ``(texts, weights)``."""
        if _depth > self.MAX_CHOICE_DEPTH:
            raise ValueError(
                f"Nested choice depth exceeded ({self.MAX_CHOICE_DEPTH} levels). "
                f"Check for accidentally recursive choice definitions."
            )

        texts: list[str] = []
        weights: list[float] = []
        for opt in options:
            result = self._resolve_option(opt, variables, _depth)
            if result is not None:
                texts.append(result[0])
                weights.append(result[1])
        return texts, weights

    def _resolve_option(
        self,
        opt: Any,
        variables: dict[str, str],
        _depth: int,
    ) -> tuple[str, float] | None:
        parsed = self._parse_option(opt)
        if parsed is None:
            return None
        content, weight, is_choice = parsed

        if is_choice:
            resolved = self.resolve(
                self.normalize_block(content),
                variables,
                _depth=_depth + 1,
            )
            if resolved is None:
                return None
            return resolved, weight

        return self._expand(str(content), variables), weight

    def _parse_option(self, opt: Any) -> tuple[Any, float, bool] | None:
        if isinstance(opt, dict) and self.is_choice_item(opt):
            weight = self._safe_weight(opt.get("weight", 1))
            clean = {k: v for k, v in opt.items() if k != "weight"}
            return clean, weight, True

        if isinstance(opt, dict):
            if not self._chance.check(opt):
                return None
            return opt.get("name", ""), self._safe_weight(opt.get("weight", 1)), False

        return str(opt), 1.0, False

    def _random_select(self, items: list[str], weights: list[float]) -> str:
        if all(w == weights[0] for w in weights):
            return self._rng.choice(items)
        return self._rng.choices(items, weights=weights, k=1)[0]

    def _safe_weight(self, value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            raise ValueError(f"Invalid weight value: {value!r}")
