"""Tests for choice/oneOf resolution via _resolve_item and _resolve_choice."""

import pytest


def test_choice_basic(parser):
    result = parser._resolve_item({"choice": {"values": ["a", "b"]}}, {})

    assert result in ("a", "b")


def test_one_of_alias(parser):
    result = parser._resolve_item({"oneOf": {"values": ["x", "y"]}}, {})

    assert result in ("x", "y")


def test_choice_weighted(make_parser):
    heavy_count = sum(
        1
        for seed in range(200)
        if make_parser(seed=seed)._resolve_item(
            {"choice": {"values": [
                {"name": "a", "weight": 10},
                {"name": "b", "weight": 1},
            ]}},
            {},
        )
        == "a"
    )

    assert heavy_count > 150


def test_choice_with_chance_zero(parser):
    result = parser._resolve_item({"choice": {"chance": 0, "values": ["a"]}}, {})

    assert result is None


def test_choice_chance_1(parser):
    result = parser._resolve_item({"choice": {"chance": 1, "values": ["a"]}}, {})

    assert result == "a"


def test_choice_with_template(parser):
    result = parser._resolve_item(
        {"choice": {"template": "($value:1.2)", "values": ["fire"]}}, {}
    )

    assert result == "(fire:1.2)"


def test_choice_option_with_chance_zero(parser):
    result = parser._resolve_item(
        {"choice": {"values": [{"name": "a", "chance": 0}]}}, {}
    )

    assert result is None


def test_choice_missing_values_key(parser):
    with pytest.raises(ValueError, match="requires 'values', 'options', or 'choices'"):
        parser._resolve_item({"choice": {}}, {})


def test_choice_wrapper_shorthand(parser):
    result = parser._resolve_item({"choice": ["a", "b", "c"]}, {})

    assert result in ("a", "b", "c")


def test_choice_variables_in_options(parser):
    result = parser._resolve_item(
        {"choice": {"values": ["$color ball"]}}, {"color": "red"}
    )

    assert result == "red ball"


def test_choice_multi_key_dict(parser):
    result = parser._resolve_item(
        {"choice": True, "values": ["sword"], "template": "($value:1.2)"}, {}
    )

    assert result == "(sword:1.2)"


def test_named_item_with_chance_skip(parser):
    result = parser._resolve_item({"name": "hello", "chance": 0}, {})

    assert result is None


def test_named_item_without_chance(parser):
    result = parser._resolve_item({"name": "hello"}, {})

    assert result == "hello"
