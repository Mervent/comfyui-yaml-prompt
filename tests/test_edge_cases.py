"""Tests for error handling and boundary conditions."""

import pytest

from parser import YAMLPromptTemplateParser


def test_non_dict_document(make_parser):
    """HARDEN-3 regression: non-dict top-level raises TypeError."""
    p = make_parser(seed=42)

    with pytest.raises(TypeError, match="Expected a YAML mapping"):
        p.parse_document(["a", "b"])


def test_invalid_chance_value(make_parser):
    p = make_parser(seed=42)

    with pytest.raises(ValueError, match="Invalid chance"):
        p._parse_section({"chance": "abc", "values": ["a"]}, {})


def test_negative_chance(make_parser):
    p = make_parser(seed=42)

    result = p._parse_section({"chance": -0.5, "values": ["a"]}, {})

    assert result == []


def test_chance_above_1(make_parser):
    p = make_parser(seed=42)

    result = p._parse_section({"chance": 1.5, "values": ["a"]}, {})

    assert result == ["a"]


def test_empty_values_list(make_parser):
    p = make_parser(seed=42)

    result = p._parse_section({"values": []}, {})

    # Empty values + simple_plain path produces one empty string (vacuous join).
    # Filtered out by parse_document's `if lines:` check in practice.
    assert result == [""]


def test_numeric_section_value(make_parser):
    p = make_parser(seed=42)

    result = p._parse_section(42, {})

    assert result == ["42"]


def test_boolean_section_value(make_parser):
    p = make_parser(seed=42)

    result = p._parse_section(True, {})

    assert result == ["True"]


def test_deeply_nested_brace_choice(make_parser):
    p = make_parser(seed=42)

    result = p.expand_string("{{{{{a|b}|{c|d}}|{e|f}}|{g|h}}|i}", {})

    assert result in ("a", "b", "c", "d", "e", "f", "g", "h", "i")


def test_wildcard_dir_nonexistent(tmp_path):
    p = YAMLPromptTemplateParser(seed=42, wildcard_dir=tmp_path / "nope")

    result = p.expand_string("__colors__", {})

    assert result == ""


def test_empty_wildcard_file(make_parser):
    p = make_parser(seed=42)

    result_empty = p.expand_string("__empty__", {})
    result_blanks = p.expand_string("__blanks__", {})

    assert result_empty == ""
    assert result_blanks == ""


def test_collect_vars_with_none_value(make_parser):
    p = make_parser(seed=42)
    doc = {
        "vars": {"x": {"choice": {"chance": 0, "values": ["val"]}}},
        "s": ["got $x"],
    }

    blocks = p.parse_document(doc)

    assert blocks[0] == ["got"]
