from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Final

from .selection import seed_derived_index, stable_select
from .wildcards import load_lines

__all__ = ["StringExpander"]

logger = logging.getLogger(__name__)


class StringExpander:
    """Expand ``$vars``, ``{brace|choices}``, and ``__wildcards__`` in strings.

    Parameters
    ----------
    seed:
        Master seed for deterministic brace/wildcard selection.
    wildcard_dir:
        Directory containing ``*.txt`` wildcard files.
    """

    MAX_EXPANSION_DEPTH: Final[int] = 64

    VARIABLE_PATTERN: Final[re.Pattern[str]] = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)")
    BRACE_PATTERN: Final[re.Pattern[str]] = re.compile(r"\{([^{}]+)\}")
    WILDCARD_PATTERN: Final[re.Pattern[str]] = re.compile(r"__([A-Za-z0-9_]+)__")

    def __init__(self, seed: int, wildcard_dir: Path) -> None:
        self.seed = seed
        self.wildcard_dir = wildcard_dir
        self._wildcard_cache: dict[tuple[Path, str], list[str]] = {}

    def expand(self, text: str, variables: dict[str, str]) -> str:
        """Expand ``$vars``, brace lists, and wildcards until stable."""
        expanded = self._substitute_variables(text, variables)

        def resolve_brace(match: re.Match[str]) -> str:
            return self._choose_brace(match, variables)

        for _ in range(self.MAX_EXPANSION_DEPTH):
            new_text = self.BRACE_PATTERN.sub(resolve_brace, expanded)
            new_text = self._substitute_wildcards(new_text)

            if new_text == expanded:
                return new_text.strip()
            expanded = new_text

        raise ValueError(
            f"Expansion depth exceeded ({self.MAX_EXPANSION_DEPTH} iterations). "
            f"Possible self-referencing pattern in: {text[:80]!r}"
        )

    def _substitute_variables(self, text: str, variables: dict[str, str]) -> str:
        def repl(match: re.Match[str]) -> str:
            return variables.get(match.group(1), match.group(0))

        return self.VARIABLE_PATTERN.sub(repl, text)

    def _choose_brace(self, match: re.Match[str], variables: dict[str, str]) -> str:
        parts = [p.strip() for p in match.group(1).split("|") if p.strip()]
        options: list[str] = []
        weights: list[float] = []

        for part in parts:
            if "::" in part:
                w_txt, opt_txt = part.split("::", 1)
                try:
                    weight = float(w_txt)
                except ValueError:
                    weight, opt_txt = 1.0, part
            else:
                weight, opt_txt = 1.0, part

            options.append(self.expand(opt_txt, variables))
            weights.append(weight)

        if not options:
            return ""
        return stable_select(self.seed, options, weights)

    def _substitute_wildcards(self, text: str) -> str:
        def repl(match: re.Match[str]) -> str:
            name = match.group(1)
            candidates = self._load_wildcard(name)
            if not candidates:
                logger.warning(
                    "Wildcard '%s' not found or empty: %s/%s.txt",
                    name,
                    self.wildcard_dir,
                    name,
                )
                return ""
            idx = seed_derived_index(self.seed, name, len(candidates))
            return candidates[idx]

        return self.WILDCARD_PATTERN.sub(repl, text)

    def _load_wildcard(self, name: str) -> list[str]:
        key = (self.wildcard_dir, name)
        if key in self._wildcard_cache:
            return self._wildcard_cache[key]

        lines = load_lines(self.wildcard_dir, name)

        self._wildcard_cache[key] = lines
        return lines
