"""Jinja2 environment configuration for YAML prompt templates.

Two-phase pipeline: Jinja2 renders template -> yaml.safe_load -> parser.
This module owns Phase 1 (Jinja2 preprocessing).
"""

from __future__ import annotations

import hashlib
import logging
import random
from pathlib import Path
from typing import Any

import jinja2

from .wildcards import load_lines

__all__ = ["create_environment", "render_template"]

logger = logging.getLogger(__name__)

# Derived-seed salt so Jinja2 RNG is deterministic but independent
# from the YAML parser RNG (Q3 decision: option C).
_JINJA_SEED_SALT: int = 0x6A696E6A  # "jinj" as 4 ASCII bytes


def render_template(
    raw_text: str,
    *,
    jinja_vars: dict[str, Any] | None = None,
    search_paths: list[Path] | None = None,
    seed: int | None = None,
    wildcard_dir: Path | None = None,
) -> str:
    """Render a Jinja2 template string into plain text (ready for ``yaml.safe_load``).

    Parameters
    ----------
    raw_text:
        Raw template content (may contain Jinja2 syntax).
    jinja_vars:
        Context variables for ``{{ ... }}`` expressions.
    search_paths:
        Directories for ``{% include %}`` resolution.
    seed:
        Master seed for deterministic Jinja2 RNG.
    wildcard_dir:
        Directory for the ``wildcard()`` global.
    """
    env = create_environment(
        search_paths=search_paths,
        seed=seed,
        wildcard_dir=wildcard_dir,
    )
    template = env.from_string(raw_text)
    return template.render(**(jinja_vars or {}))


def create_environment(
    *,
    search_paths: list[Path] | None = None,
    seed: int | None = None,
    wildcard_dir: Path | None = None,
) -> jinja2.Environment:
    """Create a configured Jinja2 ``Environment`` for prompt templates.

    Parameters
    ----------
    search_paths:
        Directories for ``{% include %}`` resolution (Q5: same directory only).
    seed:
        Master seed. Jinja2 RNG is derived via ``seed ^ _JINJA_SEED_SALT``.
    wildcard_dir:
        Directory for the ``wildcard()`` global function.
    """
    fs_paths = [str(p) for p in search_paths] if search_paths else []

    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(fs_paths) if fs_paths else jinja2.BaseLoader(),
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
        undefined=jinja2.StrictUndefined,
    )

    if seed is not None:
        rng = random.Random(seed ^ _JINJA_SEED_SALT)
    else:
        rng = random.Random()

    env.globals.update(_make_globals(rng, wildcard_dir, seed))
    return env


def _make_globals(
    rng: random.Random,
    wildcard_dir: Path | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Build Jinja2 template globals: choice, weighted_choice, rand, wildcard."""

    def choice(*items: str) -> str:
        """Pick one item uniformly at random."""
        if not items:
            return ""
        return rng.choice(items)

    def weighted_choice(items_with_weights: list[list[Any]]) -> str:
        """Pick from weighted items.  Each element is ``[value, weight]``."""
        if not items_with_weights:
            return ""
        values = [i[0] for i in items_with_weights]
        weights = [float(i[1]) for i in items_with_weights]
        return str(rng.choices(values, weights)[0])

    def rand(lo: float = 0.0, hi: float = 1.0) -> float:
        """Random float in *[lo, hi]*, rounded to 2 decimal places."""
        return round(rng.uniform(float(lo), float(hi)), 2)

    def wildcard(name: str) -> str:
        """Pick a random line from ``<wildcard_dir>/<name>.txt``."""
        if wildcard_dir is None:
            logger.warning("wildcard('%s') called but no wildcard_dir configured", name)
            return ""
        lines = load_lines(wildcard_dir, name)
        if not lines:
            logger.warning("Wildcard file not found: %s", wildcard_dir / f"{name}.txt")
            return ""
        if seed is not None:
            key = f"{seed}:{name}".encode("utf-8")
            digest = hashlib.sha256(key).digest()
            idx = int.from_bytes(digest[:8], "big") % len(lines)
            return lines[idx]
        return rng.choice(lines)

    def break_() -> str:
        """Render a YAML section that produces a CLIP BREAK token in the prompt."""
        tag = format(rng.getrandbits(32), '08x')
        return f"_break_{tag}: BREAK"

    @jinja2.pass_context
    def yaml_include(context: Any, filename: str, namespace: str | None = None) -> str:
        """Include a YAML file as a namespaced YAML document."""
        if namespace is None:
            namespace = format(rng.getrandbits(24), '06x')
        env = context.environment
        template = env.get_template(filename)
        rendered = template.render(context.get_all())
        return f"---\n_namespace: {namespace}\n{rendered}\n---"

    return {
        "choice": choice,
        "weighted_choice": weighted_choice,
        "rand": rand,
        "wildcard": wildcard,
        "break": break_,
        "yaml_include": yaml_include,
    }
