"""Shared fixtures for yaml-prompt parser tests."""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
WILDCARDS_DIR = FIXTURES_DIR / "wildcards"
INCLUDES_DIR = FIXTURES_DIR / "includes"

sys.path.insert(0, str(ROOT))

from parser import YAMLPromptTemplateParser  # noqa: E402
import jinja_env  # noqa: E402
from jinja_env import render_template  # noqa: E402


def _import_node_module():
    """Import node.py despite its relative `from .parser import ...`.

    Creates a synthetic package so that the relative import resolves correctly
    in the test environment where the project isn't pip-installed.
    """
    _PKG = "_yaml_prompt_test_pkg"
    if _PKG in sys.modules:
        return sys.modules[f"{_PKG}.node"]

    pkg = types.ModuleType(_PKG)
    pkg.__path__ = [str(ROOT)]
    pkg.__package__ = _PKG
    sys.modules[_PKG] = pkg
    sys.modules[f"{_PKG}.parser"] = sys.modules["parser"]
    sys.modules[f"{_PKG}.jinja_env"] = sys.modules["jinja_env"]

    spec = importlib.util.spec_from_file_location(
        f"{_PKG}.node", ROOT / "node.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = _PKG
    spec.loader.exec_module(mod)
    sys.modules[f"{_PKG}.node"] = mod
    return mod


_node_mod = _import_node_module()


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
def parser42(make_parser):
    """Parser seeded with 42, using test fixture wildcards."""
    return make_parser(seed=42)


@pytest.fixture
def parser_unseeded(make_parser):
    """Unseeded parser using test fixture wildcards."""
    return make_parser()


@pytest.fixture
def node_class():
    """The YAMLPromptLoader class itself (for class methods)."""
    return _node_mod.YAMLPromptLoader


@pytest.fixture
def node():
    """A fresh YAMLPromptLoader instance."""
    return _node_mod.YAMLPromptLoader()
