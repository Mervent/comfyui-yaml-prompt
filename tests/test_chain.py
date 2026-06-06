import pytest


def test_chain_basic_strings(parser):
    result = parser._resolve_item(
        {"chain": ["is running", "from car"]}, {}
    )

    assert result == "is running from car"


def test_chain_single_item(parser):
    result = parser._resolve_item({"chain": ["solo"]}, {})

    assert result == "solo"


def test_chain_shorthand_list(parser):
    result = parser._resolve_item(
        {"chain": ["alpha", "beta", "gamma"]}, {}
    )

    assert result == "alpha beta gamma"


def test_chain_with_values_key(parser):
    result = parser._resolve_item(
        {"chain": {"values": ["a", "b", "c"]}}, {}
    )

    assert result == "a b c"


def test_chain_break_on_chance_failure(parser):
    result = parser._resolve_item(
        {
            "chain": {
                "values": [
                    "always here",
                    {"name": "never here", "chance": 0},
                    "also never",
                ]
            }
        },
        {},
    )

    assert result == "always here"


def test_chain_first_item_chance_zero(parser):
    result = parser._resolve_item(
        {
            "chain": {
                "values": [
                    {"name": "nope", "chance": 0},
                    "unreachable",
                ]
            }
        },
        {},
    )

    assert result is None


def test_chain_block_level_chance_zero(parser):
    result = parser._resolve_item(
        {"chain": {"chance": 0, "values": ["a", "b"]}}, {}
    )

    assert result is None


def test_chain_block_level_chance_one(parser):
    result = parser._resolve_item(
        {"chain": {"chance": 1, "values": ["a", "b"]}}, {}
    )

    assert result == "a b"


def test_chain_custom_separator(parser):
    result = parser._resolve_item(
        {"chain": {"separator": ", ", "values": ["red", "green", "blue"]}}, {}
    )

    assert result == "red, green, blue"


def test_chain_with_template(parser):
    result = parser._resolve_item(
        {"chain": {"template": "($value:1.2)", "values": ["fire", "ice"]}}, {}
    )

    assert result == "(fire ice:1.2)"


def test_chain_with_variables(parser):
    result = parser._resolve_item(
        {"chain": ["$color", "dress"]}, {"color": "red"}
    )

    assert result == "red dress"


def test_chain_named_items_all_pass(parser):
    result = parser._resolve_item(
        {
            "chain": {
                "values": [
                    {"name": "walking"},
                    {"name": "through"},
                    {"name": "forest"},
                ]
            }
        },
        {},
    )

    assert result == "walking through forest"


def test_chain_with_choice_item(parser):
    result = parser._resolve_item(
        {
            "chain": {
                "values": [
                    "is wearing",
                    {"choice": {"values": ["red", "blue"]}},
                ]
            }
        },
        {},
    )

    parts = result.split(" ")
    assert parts[:2] == ["is", "wearing"]
    assert parts[2] in ("red", "blue")


def test_chain_choice_returns_none_breaks(parser):
    result = parser._resolve_item(
        {
            "chain": {
                "values": [
                    "before",
                    {"choice": {"chance": 0, "values": ["never"]}},
                    "after",
                ]
            }
        },
        {},
    )

    assert result == "before"


def test_chain_nested_chain(parser):
    result = parser._resolve_item(
        {
            "chain": {
                "values": [
                    "outer",
                    {"chain": ["inner_a", "inner_b"]},
                ]
            }
        },
        {},
    )

    assert result == "outer inner_a inner_b"


def test_chain_nested_chain_inner_partial(parser):
    """Inner chain breaks mid-way but returns its partial result; outer continues."""
    result = parser._resolve_item(
        {
            "chain": {
                "values": [
                    "start",
                    {
                        "chain": {
                            "values": [
                                "mid",
                                {"name": "dropped", "chance": 0},
                                "also dropped",
                            ]
                        }
                    },
                    "end",
                ]
            }
        },
        {},
    )

    assert result == "start mid end"


def test_chain_nested_chain_inner_fully_fails(parser):
    result = parser._resolve_item(
        {
            "chain": {
                "values": [
                    "start",
                    {
                        "chain": {
                            "values": [
                                {"name": "nope", "chance": 0},
                                "unreachable",
                            ]
                        }
                    },
                    "also unreachable",
                ]
            }
        },
        {},
    )

    assert result == "start"


def test_chain_inside_choice(parser):
    result = parser._resolve_item(
        {
            "choice": {
                "values": [
                    {"chain": ["chained", "result"]},
                ]
            }
        },
        {},
    )

    assert result == "chained result"


def test_chain_inside_choice_with_weight(make_parser):
    heavy_count = sum(
        1
        for seed in range(200)
        if make_parser(seed=seed)._resolve_item(
            {
                "choice": {
                    "values": [
                        {"chain": ["chain_a"], "weight": 10},
                        {"chain": ["chain_b"], "weight": 1},
                    ]
                }
            },
            {},
        )
        == "chain_a"
    )

    assert heavy_count > 150


def test_chain_deterministic(make_parser):
    item = {
        "chain": {
            "values": [
                "fixed",
                {"choice": {"values": ["a", "b", "c"]}},
            ]
        }
    }

    results = [make_parser(seed=42)._resolve_item(item, {}) for _ in range(10)]

    assert all(r == results[0] for r in results)


def test_chain_depth_exceeded(parser):
    block = {"values": ["leaf"]}
    for _ in range(20):
        block = {"values": [{"chain": block}]}

    with pytest.raises(ValueError, match="Nested block depth exceeded"):
        parser._resolve_item({"chain": block}, {})


def test_chain_missing_values_key(parser):
    with pytest.raises(ValueError, match="chain requires"):
        parser._resolve_item({"chain": {}}, {})


def test_chain_empty_values(parser):
    result = parser._resolve_item({"chain": {"values": []}}, {})

    assert result is None


def test_chain_in_section(parser):
    result = parser._parse_section(
        {"values": ["intro", {"chain": ["running", "fast"]}]}, {}
    )

    assert len(result) >= 1
    joined = " ".join(result)
    assert "intro" in joined
    assert "running fast" in joined


def test_chain_in_document(parser):
    doc = {
        "action": {
            "values": [
                {"chain": ["walking", "slowly"]},
            ]
        }
    }

    blocks = parser.parse_document(doc)

    assert blocks == [["walking slowly"]]


def test_chain_with_brace_expansion(parser):
    result = parser._resolve_item(
        {"chain": ["{red|red}", "dress"]}, {}
    )

    assert result == "red dress"


def test_chain_options_alias(parser):
    result = parser._resolve_item(
        {"chain": {"options": ["x", "y"]}}, {}
    )

    assert result == "x y"


def test_chain_choices_alias(parser):
    result = parser._resolve_item(
        {"chain": {"choices": ["p", "q"]}}, {}
    )

    assert result == "p q"


def test_chain_partial_accumulation(make_parser):
    passed_count = 0
    for seed in range(200):
        result = make_parser(seed=seed)._resolve_item(
            {
                "chain": {
                    "values": [
                        "always",
                        {"name": "sometimes", "chance": 0.5},
                        "also sometimes",
                    ]
                }
            },
            {},
        )
        if result and "sometimes" in result:
            passed_count += 1

    assert 50 < passed_count < 150


def test_chain_all_items_no_chance(parser):
    result = parser._resolve_item(
        {"chain": ["a", "b", "c", "d"]}, {}
    )

    assert result == "a b c d"
