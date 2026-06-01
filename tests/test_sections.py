def test_plain_list_section(parser):
    result = parser._parse_section(["a", "b", "c"], {})

    assert result == ["a, b, c"]


def test_section_with_values_key(parser):
    result = parser._parse_section({"values": ["a", "b"]}, {})

    assert result == ["a, b"]


def test_section_template(parser):
    result = parser._parse_section({"template": "($value)", "values": ["x"]}, {})

    assert result == ["(x)"]


def test_section_block_template(parser):
    section = {
        "block_template": "($value:$w)",
        "vars": {"w": "1.0"},
        "values": ["a", "b"],
    }

    result = parser._parse_section(section, {})

    assert result == ["(a, b:1.0)"]


def test_section_chance_skip(parser):
    result = parser._parse_section({"chance": 0, "values": ["a"]}, {})

    assert result == []


def test_section_chance_always(parser):
    result = parser._parse_section({"chance": 1, "values": ["a"]}, {})

    assert result == ["a"]


def test_section_local_vars(parser):
    result = parser._parse_section({"vars": {"x": "hello"}, "values": ["$x"]}, {})

    assert result == ["hello"]


def test_section_local_vars_no_leak(parser):
    doc = {
        "s1": {"vars": {"x": "hello"}, "values": ["$x"]},
        "s2": {"values": ["$x"]},
    }

    blocks = parser.parse_document(doc)

    assert blocks[0] == ["hello"]
    assert blocks[1] == ["$x"]


def test_section_none(parser):
    result = parser._parse_section(None, {})

    assert result == []


def test_section_string(parser):
    result = parser._parse_section("just text", {})

    assert result == ["just text"]


def test_mixed_strings_and_choices(parser):
    section = ["a", "b", {"choice": {"values": ["x", "y"]}}, "c"]

    result = parser._parse_section(section, {})

    assert len(result) == 1
    parts = result[0].split(", ")
    assert parts[0] == "a"
    assert parts[1] == "b"
    assert parts[2] in ("x", "y")
    assert parts[3] == "c"


def test_section_dict_without_values_key(parser):
    result = parser._parse_section({"template": "($value)"}, {})

    assert result == []


def test_section_with_choice_returning_none(parser):
    section = ["a", {"choice": {"values": [{"name": "x", "chance": 0}]}}, "b"]

    result = parser._parse_section(section, {})

    assert len(result) >= 1
    assert "a" in result[0]


def test_section_with_named_dict_item(parser):
    section = [{"name": "special", "weight": 2}]

    result = parser._parse_section(section, {})

    assert result == ["special"]


def test_section_empty_after_chance_filter(parser):
    doc = {"s": {"chance": 0, "values": ["a"]}}

    blocks = parser.parse_document(doc)

    assert blocks == []


def test_section_named_item_filtered_by_chance(parser):
    section = [{"name": "skip_me", "chance": 0}]

    result = parser._parse_section(section, {})

    assert result == []


def test_strings_flushed_before_named_item(parser):
    section = ["a", "b", {"name": "special"}]

    result = parser._parse_section(section, {})

    assert result == ["a, b, special"]


def test_section_custom_separator(parser):
    section = {"separator": " | ", "values": ["a", "b", "c"]}

    result = parser._parse_section(section, {})

    assert result == ["a | b | c"]


def test_section_separator_space(parser):
    section = {"separator": " ", "values": ["4k", "hdr", "detailed"]}

    result = parser._parse_section(section, {})

    assert result == ["4k hdr detailed"]


def test_section_separator_with_template(parser):
    section = {"separator": " | ", "template": "($value)", "values": ["a", "b"]}

    result = parser._parse_section(section, {})

    assert result == ["(a | b)"]


def test_section_separator_with_block_template(parser):
    section = {
        "separator": " | ",
        "block_template": "[$value]",
        "values": ["a", "b"],
    }

    result = parser._parse_section(section, {})

    assert result == ["[a | b]"]


def test_section_separator_default_unchanged(parser):
    section = {"values": ["a", "b", "c"]}

    result = parser._parse_section(section, {})

    assert result == ["a, b, c"]


def test_section_separator_empty_string(parser):
    section = {"separator": "", "values": ["a", "b", "c"]}

    result = parser._parse_section(section, {})

    assert result == ["abc"]


def test_section_separator_newline(parser):
    section = {"separator": "\n", "values": ["line1", "line2"]}

    result = parser._parse_section(section, {})

    assert result == ["line1\nline2"]


def test_section_separator_plain_list_unaffected(parser):
    result = parser._parse_section(["a", "b", "c"], {})

    assert result == ["a, b, c"]


def test_section_separator_in_full_document(parser):
    doc = {
        "tags": {"separator": " ", "values": ["4k", "hdr"]},
        "style": {"values": ["painterly", "soft"]},
    }

    blocks = parser.parse_document(doc)

    assert blocks[0] == ["4k hdr"]
    assert blocks[1] == ["painterly, soft"]


def test_choice_items_joined_by_separator(parser):
    doc = {
        "s": {
            "values": [
                "a",
                {"choice": {"values": ["x", "y"]}},
                "b",
            ]
        }
    }

    blocks = parser.parse_document(doc)

    assert len(blocks) == 1
    assert ", " in blocks[0][0]


def test_sections_with_choices_still_separate_blocks(parser):
    doc = {
        "s1": {
            "values": [{"choice": {"values": ["a", "b"]}}]
        },
        "s2": {
            "values": [{"choice": {"values": ["c", "d"]}}]
        },
    }

    blocks = parser.parse_document(doc)

    assert len(blocks) == 2
