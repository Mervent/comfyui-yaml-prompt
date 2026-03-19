"""Tests for error handling and boundary conditions."""

import pytest

from parser import YAMLPromptTemplateParser


def test_non_dict_document(parser):
    """HARDEN-3 regression: non-dict top-level raises TypeError."""
    with pytest.raises(TypeError, match="Expected a YAML mapping"):
        parser.parse_document(["a", "b"])


def test_invalid_chance_value(parser):
    with pytest.raises(ValueError, match="Invalid chance"):
        parser._parse_section({"chance": "abc", "values": ["a"]}, {})


def test_negative_chance(parser):
    result = parser._parse_section({"chance": -0.5, "values": ["a"]}, {})

    assert result == []


def test_chance_above_1(parser):
    result = parser._parse_section({"chance": 1.5, "values": ["a"]}, {})

    assert result == ["a"]


def test_empty_values_list(parser):
    result = parser._parse_section({"values": []}, {})

    # Empty values + simple_plain path produces one empty string (vacuous join).
    # Filtered out by parse_document's `if lines:` check in practice.
    assert result == [""]


def test_numeric_section_value(parser):
    result = parser._parse_section(42, {})

    assert result == ["42"]


def test_boolean_section_value(parser):
    result = parser._parse_section(True, {})

    assert result == ["True"]


def test_deeply_nested_brace_choice(parser):
    result = parser.expand_string("{{{{{a|b}|{c|d}}|{e|f}}|{g|h}}|i}", {})

    assert result in ("a", "b", "c", "d", "e", "f", "g", "h", "i")


def test_wildcard_dir_nonexistent(tmp_path):
    p = YAMLPromptTemplateParser(seed=42, wildcard_dir=tmp_path / "nope")

    result = p.expand_string("__colors__", {})

    assert result == ""


def test_empty_wildcard_file(parser):
    result = parser.expand_string("__empty__", {})

    assert result == ""


def test_blank_lines_wildcard_file(parser):
    result = parser.expand_string("__blanks__", {})

    assert result == ""


def test_collect_vars_with_none_value(parser):
    doc = {
        "vars": {"x": {"choice": {"chance": 0, "values": ["val"]}}},
        "s": ["got $x"],
    }

    blocks = parser.parse_document(doc)

    assert blocks[0] == ["got"]
