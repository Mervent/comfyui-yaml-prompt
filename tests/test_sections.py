"""Tests for _parse_section: templates, block_templates, chance, vars."""

from parser import YAMLPromptTemplateParser


def test_plain_list_section(make_parser):
    p = make_parser(seed=42)

    result = p._parse_section(["a", "b", "c"], {})

    assert result == ["a, b, c"]


def test_section_with_values_key(make_parser):
    p = make_parser(seed=42)

    result = p._parse_section({"values": ["a", "b"]}, {})

    assert result == ["a, b"]


def test_section_template(make_parser):
    p = make_parser(seed=42)

    result = p._parse_section({"template": "($value)", "values": ["x"]}, {})

    assert result == ["(x)"]


def test_section_block_template(make_parser):
    p = make_parser(seed=42)
    section = {
        "block_template": "($value:$w)",
        "vars": {"w": "1.0"},
        "values": ["a", "b"],
    }

    result = p._parse_section(section, {})

    assert result == ["(a, b:1.0)"]


def test_section_chance_skip(make_parser):
    p = make_parser(seed=42)

    result = p._parse_section({"chance": 0, "values": ["a"]}, {})

    assert result == []


def test_section_chance_always(make_parser):
    p = make_parser(seed=42)

    result = p._parse_section({"chance": 1, "values": ["a"]}, {})

    assert result == ["a"]


def test_section_local_vars(make_parser):
    p = make_parser(seed=42)

    result = p._parse_section({"vars": {"x": "hello"}, "values": ["$x"]}, {})

    assert result == ["hello"]


def test_section_local_vars_no_leak(make_parser):
    p = make_parser(seed=42)
    doc = {
        "s1": {"vars": {"x": "hello"}, "values": ["$x"]},
        "s2": {"values": ["$x"]},
    }

    blocks = p.parse_document(doc)

    assert blocks[0] == ["hello"]
    assert blocks[1] == ["$x"]


def test_section_none(make_parser):
    p = make_parser(seed=42)

    result = p._parse_section(None, {})

    assert result == []


def test_section_string(make_parser):
    p = make_parser(seed=42)

    result = p._parse_section("just text", {})

    assert result == ["just text"]


def test_mixed_strings_and_choices(make_parser):
    p = make_parser(seed=42)
    section = ["a", "b", {"choice": {"values": ["x", "y"]}}, "c"]

    result = p._parse_section(section, {})

    assert len(result) == 2
    first_part = result[0]
    assert first_part.startswith("a, b, ")
    tail = first_part.removeprefix("a, b, ")
    assert tail in ("x", "y")
    assert result[1] == "c"


def test_section_dict_without_values_key(make_parser):
    p = make_parser(seed=42)

    result = p._parse_section({"template": "($value)"}, {})

    assert result == []


def test_section_with_choice_returning_none(make_parser):
    p = make_parser(seed=42)
    section = ["a", {"choice": {"values": [{"name": "x", "chance": 0}]}}, "b"]

    result = p._parse_section(section, {})

    assert len(result) >= 1
    assert "a" in result[0]


def test_section_with_named_dict_item(make_parser):
    p = make_parser(seed=42)
    section = [{"name": "special", "weight": 2}]

    result = p._parse_section(section, {})

    assert result == ["special"]


def test_section_empty_after_chance_filter(make_parser):
    p = make_parser(seed=42)
    doc = {"s": {"chance": 0, "values": ["a"]}}

    blocks = p.parse_document(doc)

    assert blocks == []


def test_section_named_item_filtered_by_chance(make_parser):
    p = make_parser(seed=42)
    section = [{"name": "skip_me", "chance": 0}]

    result = p._parse_section(section, {})

    assert result == []
