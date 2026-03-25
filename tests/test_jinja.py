"""Tests for Jinja2 preprocessing layer (jinja_env.py)."""

import jinja2
import pytest
import yaml
from conftest import INCLUDES_DIR, WILDCARDS_DIR

from yaml_prompt.jinja_env import render_template
from yaml_prompt.parser import YAMLPromptTemplateParser
from yaml_prompt.pipeline import process_file


def test_passthrough_no_jinja():
    raw = "meta:\n  - ultra-detailed\n  - masterpiece\n"

    result = render_template(raw)

    assert result == raw


def test_brace_syntax_passthrough():
    raw = "style: {a|b|c}\n"

    result = render_template(raw)

    assert result == raw


def test_weighted_brace_passthrough():
    raw = "style: {0.5::a|b}\n"

    result = render_template(raw)

    assert result == raw


def test_dollar_var_passthrough():
    raw = "style: $myvar\n"

    result = render_template(raw)

    assert result == raw


def test_wildcard_passthrough():
    raw = "style: __name__\n"

    result = render_template(raw)

    assert result == raw


def test_variable_expression():
    raw = "style: {{ name }}"

    result = render_template(raw, jinja_vars={"name": "photorealistic"})

    assert result == "style: photorealistic"


def test_strict_undefined():
    raw = "{{ undefined_var }}"

    with pytest.raises(jinja2.UndefinedError):
        render_template(raw)


def test_if_true():
    raw = "{% if enemy %}combat{% endif %}"

    result = render_template(raw, jinja_vars={"enemy": True})

    assert result == "combat"


def test_if_false():
    raw = "{% if enemy %}combat{% endif %}"

    result = render_template(raw, jinja_vars={"enemy": False})

    assert result == ""


def test_if_else():
    raw = "{% if x %}a{% else %}b{% endif %}"

    result_true = render_template(raw, jinja_vars={"x": True})
    result_false = render_template(raw, jinja_vars={"x": False})

    assert result_true == "a"
    assert result_false == "b"


def test_for_loop():
    raw = "{% for item in items %}- {{ item }}\n{% endfor %}"

    result = render_template(raw, jinja_vars={"items": ["a", "b", "c"]})

    assert "- a" in result
    assert "- b" in result
    assert "- c" in result


def test_include_basic():
    raw = "{% include 'part.yaml' %}"

    result = render_template(raw, search_paths=[INCLUDES_DIR])

    assert "included content" in result
    assert "from part" in result


def test_include_conditional():
    raw = "{% if x %}{% include 'part.yaml' %}{% endif %}"

    result_true = render_template(
        raw,
        jinja_vars={"x": True},
        search_paths=[INCLUDES_DIR],
    )
    result_false = render_template(
        raw,
        jinja_vars={"x": False},
        search_paths=[INCLUDES_DIR],
    )

    assert "included content" in result_true
    assert result_false == ""


def test_include_nested():
    raw = "{% include 'nested.yaml' %}"

    result = render_template(raw, search_paths=[INCLUDES_DIR])

    assert "included content" in result
    assert "nested extra" in result


def test_include_missing_file():
    raw = "{% include 'nope.yaml' %}"

    with pytest.raises(jinja2.TemplateNotFound):
        render_template(raw, search_paths=[INCLUDES_DIR])


def test_include_no_search_paths():
    raw = "{% include 'anything.yaml' %}"

    with pytest.raises(jinja2.TemplateNotFound):
        render_template(raw)


def test_whitespace_clean():
    raw = "meta:\n{% if x %}\n  - detailed\n{% endif %}\n  - masterpiece\n"

    result = render_template(raw, jinja_vars={"x": True})

    assert "  - detailed" in result
    assert "  - masterpiece" in result
    parsed = yaml.safe_load(result)
    assert "meta" in parsed


def test_choice_returns_one_of_items():
    raw = "{{ choice('a', 'b', 'c') }}"

    result = render_template(raw, seed=42)

    assert result in ("a", "b", "c")


def test_choice_seeded_deterministic():
    raw = "{{ choice('a', 'b', 'c') }}"

    r1 = render_template(raw, seed=42)
    r2 = render_template(raw, seed=42)

    assert r1 == r2


def test_choice_empty():
    raw = "{{ choice() }}"

    result = render_template(raw, seed=42)

    assert result == ""


def test_weighted_choice():
    raw = "{{ weighted_choice([['a', 100], ['b', 0.001]]) }}"

    results = {render_template(raw, seed=i) for i in range(30)}

    assert "a" in results


def test_weighted_choice_empty():
    raw = "{{ weighted_choice([]) }}"

    result = render_template(raw, seed=42)

    assert result == ""


def test_rand_in_range():
    raw = "{{ rand(0, 1) }}"

    result = render_template(raw, seed=42)

    val = float(result)
    assert 0.0 <= val <= 1.0


def test_rand_seeded_deterministic():
    raw = "{{ rand(0, 10) }}"

    r1 = render_template(raw, seed=42)
    r2 = render_template(raw, seed=42)

    assert r1 == r2


def test_wildcard_picks_from_file():
    raw = "{{ wildcard('colors') }}"

    result = render_template(raw, seed=42, wildcard_dir=WILDCARDS_DIR)

    assert result in ("red", "blue", "green")


def test_wildcard_missing_file():
    raw = "{{ wildcard('nonexistent') }}"

    result = render_template(raw, seed=42, wildcard_dir=WILDCARDS_DIR)

    assert result == ""


def test_wildcard_empty_file():
    raw = "{{ wildcard('empty') }}"

    result = render_template(raw, seed=42, wildcard_dir=WILDCARDS_DIR)

    assert result == ""


def test_wildcard_no_dir():
    raw = "{{ wildcard('colors') }}"

    result = render_template(raw, seed=42)

    assert result == ""


def test_derived_seed_independence():
    template = "meta:\n  values:\n    - {a|b|c}\n"
    with_jinja = "{{ choice('x', 'y') }}\n" + template

    rendered_with = render_template(with_jinja, seed=42)
    yaml_part = "\n".join(rendered_with.split("\n")[1:])
    data_with = yaml.safe_load(yaml_part)
    p1 = YAMLPromptTemplateParser(seed=42)
    result_with = p1.parse_document(data_with)

    rendered_without = render_template(template, seed=42)
    data_without = yaml.safe_load(rendered_without)
    p2 = YAMLPromptTemplateParser(seed=42)
    result_without = p2.parse_document(data_without)

    assert result_with == result_without


def test_full_pipeline(tmp_path):
    (tmp_path / "template.yaml").write_text(
        "vars:\n"
        "  color: {{ default_color }}\n"
        "{% if add_mood %}\n"
        "mood:\n"
        "  - serene\n"
        "  - peaceful\n"
        "{% endif %}\n"
        "meta:\n"
        "  - ultra-detailed\n"
    )

    result = process_file(
        tmp_path / "template.yaml",
        seed=42,
        jinja_vars={
            "default_color": "blue",
            "add_mood": True,
        },
    )

    assert result.prompt == "serene, peaceful\n\nultra-detailed"


def test_break_produces_yaml_section():
    raw = "{{ break() }}"

    result = render_template(raw, seed=42)

    assert result.startswith("_break_")
    assert result.endswith(": BREAK")


def test_break_in_full_pipeline(tmp_path):
    (tmp_path / "template.yaml").write_text(
        "positive:\n"
        "  values:\n"
        "    - beautiful landscape\n"
        "{{ break() }}\n"
        "details:\n"
        "  values:\n"
        "    - detailed, 8k\n"
    )

    result = process_file(tmp_path / "template.yaml", seed=42)

    assert result.prompt == "beautiful landscape\n\nBREAK\n\ndetailed, 8k"


def test_break_deterministic():
    raw = "{{ break() }}"

    r1 = render_template(raw, seed=42)
    r2 = render_template(raw, seed=42)

    assert r1 == r2


def test_full_pipeline_conditional_exclude(tmp_path):
    (tmp_path / "template.yaml").write_text(
        "{% if add_mood %}\n"
        "mood:\n"
        "  - serene\n"
        "{% endif %}\n"
        "meta:\n"
        "  - ultra-detailed\n"
    )

    result = process_file(
        tmp_path / "template.yaml",
        seed=42,
        jinja_vars={"add_mood": False},
    )

    assert result.prompt == "ultra-detailed"


def test_include_override_preserves_position(tmp_path):
    """Child include overrides a parent section's content but keeps its
    original position in the prompt — earlier sections stay earlier."""
    (tmp_path / "child.yaml").write_text(
        "override_me:\n" "  values:\n" "    - from child\n"
    )
    (tmp_path / "parent.yaml").write_text(
        "override_me:\n"
        "  values:\n"
        "    - from parent\n"
        "middle:\n"
        "  values:\n"
        "    - middle content\n"
        "end:\n"
        "  values:\n"
        "    - end content\n"
        "{% include 'child.yaml' %}\n"
    )

    result = process_file(tmp_path / "parent.yaml", seed=42)

    assert result.prompt == "from child\n\nmiddle content\n\nend content"
