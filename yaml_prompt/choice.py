from __future__ import annotations

import random
from collections.abc import Callable
from typing import Any, Final, Sequence

from .chance import ChanceEvaluator
from .selection import stable_select

__all__ = ["ChoiceResolver"]


class ChoiceResolver:
    """Resolve ``choice``/``oneOf``/``chain`` blocks into selected prompt text.

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
    CHAIN_KEY: Final[str] = "chain"
    GROUP_KEY: Final[str] = "group"
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
        inherited_sep: str | None = None,
    ) -> str | None:
        """Pick one option from a choice/oneOf block, applying weights and chance."""
        if not self._chance.check(block):
            return None

        template = block.get("template", "$value")
        options = self.get_list_values(block)
        if options is None:
            raise ValueError("choice/oneOf requires 'values', 'options', or 'choices'.")

        explicit_sep = block.get("separator")
        child_sep = str(explicit_sep) if explicit_sep is not None else inherited_sep
        texts, weights = self._filter_and_weigh_options(
            options, variables, _depth=_depth, inherited_sep=child_sep
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

    def resolve_chain(
        self,
        block: dict[str, Any],
        variables: dict[str, str],
        _depth: int = 0,
        *,
        skip_failed: bool = False,
        inherited_sep: str | None = None,
    ) -> str | None:
        """Process chain/group items sequentially, joining their results.

        Parameters
        ----------
        block : dict[str, Any]
            Normalized chain/group block (must contain a ``values``/
            ``options``/``choices`` list).
        variables : dict[str, str]
            Current variable bindings for ``$var`` expansion.
        _depth : int
            Recursion guard shared with ``resolve()``.
        skip_failed : bool
            When ``False`` (``chain``), stop at the first item that fails its
            chance gate. When ``True`` (``group``), skip failed items and keep
            going so each item is gated independently.
        inherited_sep : str | None
            Separator inherited from an enclosing scope (document or parent
            block). Used when this block declares no ``separator`` of its own,
            taking precedence over the type default (``", "`` for group,
            ``" "`` for chain). Propagated unchanged to nested blocks, or
            replaced by this block's explicit ``separator`` when present.

        Returns
        -------
        str | None
            Joined result of the surviving items, or ``None`` if the
            block-level chance gate fails or no item survives.
        """
        if _depth > self.MAX_CHOICE_DEPTH:
            raise ValueError(
                f"Nested block depth exceeded ({self.MAX_CHOICE_DEPTH} levels). "
                f"Check for accidentally recursive definitions."
            )

        if not self._chance.check(block):
            return None

        explicit_sep = block.get("separator")
        if explicit_sep is not None:
            effective_sep: str | None = str(explicit_sep)
        else:
            effective_sep = inherited_sep
        separator = (
            effective_sep
            if effective_sep is not None
            else (", " if skip_failed else " ")
        )
        template = block.get("template", "$value")
        options = self.get_list_values(block)
        if options is None:
            raise ValueError("chain requires 'values', 'options', or 'choices'.")

        accumulated: list[str] = []
        for item in options:
            resolved = self._resolve_chain_item(item, variables, _depth, effective_sep)
            if resolved is None:
                if skip_failed:
                    continue
                break
            accumulated.append(resolved)

        if not accumulated:
            return None

        result = separator.join(accumulated)
        if template != "$value":
            return self._expand(template.replace("$value", result), variables)
        return result

    def _resolve_chain_item(
        self,
        item: Any,
        variables: dict[str, str],
        _depth: int,
        inherited_sep: str | None = None,
    ) -> str | None:
        if isinstance(item, dict) and self.GROUP_KEY in item:
            return self.resolve_chain(
                self.normalize_block(item),
                variables,
                _depth + 1,
                skip_failed=True,
                inherited_sep=inherited_sep,
            )
        if isinstance(item, dict) and self.CHAIN_KEY in item:
            return self.resolve_chain(
                self.normalize_block(item),
                variables,
                _depth + 1,
                inherited_sep=inherited_sep,
            )
        if self.is_choice_item(item):
            return self.resolve(
                self.normalize_block(item),
                variables,
                _depth + 1,
                inherited_sep=inherited_sep,
            )
        if isinstance(item, dict) and "name" in item:
            if not self._chance.check(item):
                return None
            return self._expand(str(item["name"]), variables)
        if isinstance(item, str):
            return self._expand(item, variables)
        return self._expand(str(item), variables)

    def is_chain_item(self, item: Any) -> bool:
        if not isinstance(item, dict):
            return False
        return self.CHAIN_KEY in item

    def is_group_item(self, item: Any) -> bool:
        if not isinstance(item, dict):
            return False
        return self.GROUP_KEY in item

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
        inherited_sep: str | None = None,
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
            result = self._resolve_option(opt, variables, _depth, inherited_sep)
            if result is not None:
                texts.append(result[0])
                weights.append(result[1])
        return texts, weights

    def _resolve_option(
        self,
        opt: Any,
        variables: dict[str, str],
        _depth: int,
        inherited_sep: str | None = None,
    ) -> tuple[str, float] | None:
        parsed = self._parse_option(opt)
        if parsed is None:
            return None
        content, weight, is_block = parsed

        if is_block:
            if isinstance(content, dict) and self.GROUP_KEY in content:
                resolved = self.resolve_chain(
                    self.normalize_block(content),
                    variables,
                    _depth=_depth + 1,
                    skip_failed=True,
                    inherited_sep=inherited_sep,
                )
            elif isinstance(content, dict) and self.CHAIN_KEY in content:
                resolved = self.resolve_chain(
                    self.normalize_block(content),
                    variables,
                    _depth=_depth + 1,
                    inherited_sep=inherited_sep,
                )
            else:
                resolved = self.resolve(
                    self.normalize_block(content),
                    variables,
                    _depth=_depth + 1,
                    inherited_sep=inherited_sep,
                )
            if resolved is None:
                return None
            return resolved, weight

        return self._expand(str(content), variables), weight

    def _parse_option(self, opt: Any) -> tuple[Any, float, bool] | None:
        if isinstance(opt, dict) and (
            self.is_choice_item(opt)
            or self.is_chain_item(opt)
            or self.is_group_item(opt)
        ):
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
