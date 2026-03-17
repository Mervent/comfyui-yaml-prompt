"""Tests for expand_string: variable substitution, braces, wildcards."""

import pytest

from parser import YAMLPromptTemplateParser


def test_plain_string(parser42):
    result = parser42.expand_string("hello world", {})

    assert result == "hello world"


def test_variable_substitution(parser42):
    result = parser42.expand_string("$color cat", {"color": "black"})

    assert result == "black cat"


def test_unknown_variable_kept(parser42):
    result = parser42.expand_string("$unknown", {})

    assert result == "$unknown"


def test_multiple_variables(parser42):
    result = parser42.expand_string("$a and $b", {"a": "x", "b": "y"})

    assert result == "x and y"


def test_brace_choice(make_parser):
    p1 = make_parser(seed=42)
    result1 = p1.expand_string("{a|b|c}", {})

    p2 = make_parser(seed=42)
    result2 = p2.expand_string("{a|b|c}", {})

    assert result1 in ("a", "b", "c")
    assert result1 == result2


def test_brace_weighted(make_parser):
    heavy_count = sum(
        1
        for seed in range(100)
        if make_parser(seed=seed).expand_string("{0.9::heavy|0.1::light}", {})
        == "heavy"
    )

    assert heavy_count > 70


def test_brace_empty_option(make_parser):
    p = make_parser(seed=42)

    result = p.expand_string("{a||c}", {})

    assert result in ("a", "c")


def test_brace_all_empty_bug1_regression(make_parser):
    """BUG-1 regression: {|} must not crash (IndexError)."""
    p = make_parser(seed=42)

    result = p.expand_string("{|}", {})

    assert result == ""


def test_brace_nested(make_parser):
    p = make_parser(seed=42)

    result = p.expand_string("{{a|b}|{c|d}}", {})

    assert result in ("a", "b", "c", "d")


def test_wildcard_resolved(make_parser):
    p = make_parser(seed=42)

    result = p.expand_string("__colors__", {})

    assert result in ("red", "blue", "green")


def test_wildcard_missing_file(make_parser):
    """HARDEN-1 regression: missing wildcard -> empty string, no crash."""
    p = make_parser(seed=42)

    result = p.expand_string("__nonexistent__", {})

    assert result == ""


def test_wildcard_stable_with_seed(make_parser):
    results = {make_parser(seed=42).expand_string("__colors__", {}) for _ in range(10)}

    assert len(results) == 1


def test_expansion_depth_limit(tmp_path):
    """HARDEN-2 regression: cyclic wildcards raise ValueError."""
    wc_dir = tmp_path / "wildcards"
    wc_dir.mkdir()
    (wc_dir / "cycle_a.txt").write_text("__cycle_b__\n")
    (wc_dir / "cycle_b.txt").write_text("__cycle_a__\n")
    p = YAMLPromptTemplateParser(seed=42, wildcard_dir=wc_dir)

    with pytest.raises(ValueError, match="Expansion depth exceeded"):
        p.expand_string("__cycle_a__", {})


def test_strip_whitespace(parser42):
    result = parser42.expand_string("  hello  ", {})

    assert result == "hello"


def test_brace_malformed_weight(make_parser):
    p = make_parser(seed=42)

    result = p.expand_string("{abc::text|other}", {})

    assert result in ("abc::text", "other")


def test_wildcard_unseeded(make_parser):
    p = make_parser(seed=None)

    result = p.expand_string("__colors__", {})

    assert result in ("red", "blue", "green")


def test_default_wildcard_dir():
    p = YAMLPromptTemplateParser(seed=42)

    assert p.wildcard_dir == YAMLPromptTemplateParser.DEFAULT_WILDCARD_DIR
