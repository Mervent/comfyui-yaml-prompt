import pytest


def test_group_basic_strings(parser):
    result = parser._resolve_item({"group": ["a", "b"]}, {})

    assert result == "a, b"


def test_group_single_item(parser):
    result = parser._resolve_item({"group": ["solo"]}, {})

    assert result == "solo"


def test_group_with_values_key(parser):
    result = parser._resolve_item({"group": {"values": ["a", "b", "c"]}}, {})

    assert result == "a, b, c"


def test_group_all_items_no_chance(parser):
    result = parser._resolve_item({"group": ["a", "b", "c", "d"]}, {})

    assert result == "a, b, c, d"


def test_group_default_separator_is_comma(parser):
    result = parser._resolve_item({"group": ["red", "green"]}, {})

    assert result == "red, green"


def test_group_space_separator_override(parser):
    result = parser._resolve_item(
        {"group": {"separator": " ", "values": ["walking", "slowly"]}}, {}
    )

    assert result == "walking slowly"


def test_group_skips_failed_item(parser):
    result = parser._resolve_item(
        {
            "group": {
                "values": [
                    "a",
                    {"name": "b", "chance": 0},
                    "c",
                ]
            }
        },
        {},
    )

    assert result == "a, c"


def test_group_first_item_chance_zero_continues(parser):
    result = parser._resolve_item(
        {
            "group": {
                "values": [
                    {"name": "a", "chance": 0},
                    "b",
                ]
            }
        },
        {},
    )

    assert result == "b"


def test_group_all_items_fail(parser):
    result = parser._resolve_item(
        {
            "group": {
                "values": [
                    {"name": "a", "chance": 0},
                    {"name": "b", "chance": 0},
                ]
            }
        },
        {},
    )

    assert result is None


def test_group_block_level_chance_zero(parser):
    result = parser._resolve_item({"group": {"chance": 0, "values": ["a", "b"]}}, {})

    assert result is None


def test_group_block_level_chance_one(parser):
    result = parser._resolve_item({"group": {"chance": 1, "values": ["a", "b"]}}, {})

    assert result == "a, b"


def test_group_custom_separator(parser):
    result = parser._resolve_item(
        {"group": {"separator": " | ", "values": ["red", "green", "blue"]}}, {}
    )

    assert result == "red | green | blue"


def test_group_with_template(parser):
    result = parser._resolve_item(
        {"group": {"template": "($value:1.2)", "values": ["fire", "ice"]}}, {}
    )

    assert result == "(fire, ice:1.2)"


def test_group_with_variables(parser):
    result = parser._resolve_item({"group": ["$color", "leather"]}, {"color": "red"})

    assert result == "red, leather"


def test_group_with_brace_expansion(parser):
    result = parser._resolve_item({"group": ["{red|red}", "leather"]}, {})

    assert result == "red, leather"


def test_group_named_items_all_pass(parser):
    result = parser._resolve_item(
        {
            "group": {
                "values": [
                    {"name": "walking"},
                    {"name": "through"},
                    {"name": "forest"},
                ]
            }
        },
        {},
    )

    assert result == "walking, through, forest"


def test_group_options_alias(parser):
    result = parser._resolve_item({"group": {"options": ["x", "y"]}}, {})

    assert result == "x, y"


def test_group_choices_alias(parser):
    result = parser._resolve_item({"group": {"choices": ["p", "q"]}}, {})

    assert result == "p, q"


def test_group_missing_values_key(parser):
    with pytest.raises(ValueError, match="requires 'values', 'options', or 'choices'"):
        parser._resolve_item({"group": {}}, {})


def test_group_empty_values(parser):
    result = parser._resolve_item({"group": {"values": []}}, {})

    assert result is None


def test_group_with_choice_item(parser):
    result = parser._resolve_item(
        {
            "group": {
                "values": [
                    "is wearing",
                    {"choice": {"values": ["red", "blue"]}},
                ]
            }
        },
        {},
    )

    assert result in ("is wearing, red", "is wearing, blue")


def test_group_choice_returns_none_skips(parser):
    result = parser._resolve_item(
        {
            "group": {
                "values": [
                    "before",
                    {"choice": {"chance": 0, "values": ["never"]}},
                    "after",
                ]
            }
        },
        {},
    )

    assert result == "before, after"


def test_group_nested_group(parser):
    result = parser._resolve_item(
        {
            "group": {
                "values": [
                    "outer",
                    {"group": ["inner_a", "inner_b"]},
                ]
            }
        },
        {},
    )

    assert result == "outer, inner_a, inner_b"


def test_group_nested_group_inner_skips(parser):
    result = parser._resolve_item(
        {
            "group": {
                "values": [
                    "start",
                    {
                        "group": {
                            "values": [
                                "mid",
                                {"name": "dropped", "chance": 0},
                                "tail",
                            ]
                        }
                    },
                    "end",
                ]
            }
        },
        {},
    )

    assert result == "start, mid, tail, end"


def test_chain_inside_group(parser):
    result = parser._resolve_item(
        {
            "group": {
                "values": [
                    "start",
                    {"chain": ["c_a", "c_b"]},
                    "end",
                ]
            }
        },
        {},
    )

    assert result == "start, c_a c_b, end"


def test_group_inside_chain(parser):
    result = parser._resolve_item(
        {
            "chain": {
                "values": [
                    "start",
                    {"group": ["g_a", "g_b"]},
                    "end",
                ]
            }
        },
        {},
    )

    assert result == "start g_a, g_b end"


def test_group_inside_choice(parser):
    result = parser._resolve_item(
        {"choice": {"values": [{"group": ["grouped", "result"]}]}}, {}
    )

    assert result == "grouped, result"


def test_group_inside_choice_with_weight(make_parser):
    heavy_count = sum(
        1
        for seed in range(200)
        if make_parser(seed=seed)._resolve_item(
            {
                "choice": {
                    "values": [
                        {"group": ["g_a"], "weight": 10},
                        {"group": ["g_b"], "weight": 1},
                    ]
                }
            },
            {},
        )
        == "g_a"
    )

    assert heavy_count > 150


def test_oneof_inside_group(parser):
    result = parser._resolve_item(
        {
            "group": {
                "values": [
                    {"oneOf": {"values": ["a", "b"]}},
                    {"oneOf": {"values": ["c"]}},
                ]
            }
        },
        {},
    )

    assert result in ("a, c", "b, c")


def test_group_deterministic(make_parser):
    item = {
        "group": {
            "values": [
                "fixed",
                {"choice": {"values": ["a", "b", "c"]}},
            ]
        }
    }

    results = [make_parser(seed=42)._resolve_item(item, {}) for _ in range(10)]

    assert all(r == results[0] for r in results)


def test_group_depth_exceeded(parser):
    block = {"values": ["leaf"]}
    for _ in range(20):
        block = {"values": [{"group": block}]}

    with pytest.raises(ValueError, match="Nested block depth exceeded"):
        parser._resolve_item({"group": block}, {})


def test_group_partial_accumulation(make_parser):
    passed_count = 0
    for seed in range(200):
        result = make_parser(seed=seed)._resolve_item(
            {
                "group": {
                    "values": [
                        "always",
                        {"name": "sometimes", "chance": 0.5},
                        "tail",
                    ]
                }
            },
            {},
        )
        if result and "sometimes" in result:
            passed_count += 1

    assert 50 < passed_count < 150


def test_group_skips_middle_keeps_tail(make_parser):
    for seed in range(50):
        result = make_parser(seed=seed)._resolve_item(
            {
                "group": {
                    "values": [
                        "head",
                        {"name": "maybe", "chance": 0.5},
                        "tail",
                    ]
                }
            },
            {},
        )

        assert result is not None
        assert result.startswith("head")
        assert result.endswith("tail")


def test_group_in_section(parser):
    result = parser._parse_section(
        {"values": ["intro", {"group": ["running", "fast"]}]}, {}
    )

    assert len(result) >= 1
    joined = " ".join(result)
    assert "intro" in joined
    assert "running, fast" in joined


def test_group_in_document(parser):
    doc = {
        "action": {
            "values": [
                {"group": ["walking", "slowly"]},
            ]
        }
    }

    blocks = parser.parse_document(doc)

    assert blocks == [["walking, slowly"]]
