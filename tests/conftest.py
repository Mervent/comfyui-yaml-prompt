"""Shared fixtures for yaml-prompt parser tests."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from yaml_prompt.parser import YAMLPromptTemplateParser  # noqa: E402
from yaml_prompt.node import YAMLPromptLoader  # noqa: E402

FIXTURES_DIR = ROOT / "tests" / "fixtures"
WILDCARDS_DIR = FIXTURES_DIR / "wildcards"
INCLUDES_DIR = FIXTURES_DIR / "includes"


@pytest.fixture
def make_parser():
    """Factory: call with optional seed and wildcard_dir."""

    def _factory(seed=None, wildcard_dir=None):
        return YAMLPromptTemplateParser(
            seed=seed,
            wildcard_dir=wildcard_dir or WILDCARDS_DIR,
        )

    return _factory


@pytest.fixture
def parser(make_parser):
    """Parser seeded with 42, using test fixture wildcards."""
    return make_parser(seed=42)


@pytest.fixture
def node_class():
    """The YAMLPromptLoader class itself (for class methods)."""
    return YAMLPromptLoader


@pytest.fixture
def node():
    """A fresh YAMLPromptLoader instance."""
    return YAMLPromptLoader()
