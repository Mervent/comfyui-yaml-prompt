import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from yaml_prompt.chance import ChanceEvaluator  # noqa: E402
from yaml_prompt.choice import ChoiceResolver  # noqa: E402
from yaml_prompt.expand import StringExpander  # noqa: E402
from yaml_prompt.node import YAMLPromptLoader  # noqa: E402
from yaml_prompt.parser import YAMLPromptTemplateParser  # noqa: E402

FIXTURES_DIR = ROOT / "tests" / "fixtures"
WILDCARDS_DIR = FIXTURES_DIR / "wildcards"
INCLUDES_DIR = FIXTURES_DIR / "includes"

SEED = 42


@pytest.fixture
def make_parser():
    def _factory(seed=None, wildcard_dir=None):
        return YAMLPromptTemplateParser(
            seed=seed,
            wildcard_dir=wildcard_dir or WILDCARDS_DIR,
        )

    return _factory


@pytest.fixture
def parser(make_parser):
    return make_parser(seed=SEED)


@pytest.fixture
def chance_evaluator():
    return ChanceEvaluator(seed=SEED)


@pytest.fixture
def choice_resolver(chance_evaluator):
    expander = StringExpander(seed=SEED, wildcard_dir=WILDCARDS_DIR)
    return ChoiceResolver(seed=SEED, chance=chance_evaluator, expand_fn=expander.expand)


@pytest.fixture
def node_class():
    return YAMLPromptLoader


@pytest.fixture
def node():
    return YAMLPromptLoader()
