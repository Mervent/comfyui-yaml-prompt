import pytest


def test_get_list_values_with_values_key(choice_resolver):
    result = choice_resolver.get_list_values({"values": ["a", "b"]})

    assert result == ["a", "b"]


def test_get_list_values_with_options_key(choice_resolver):
    result = choice_resolver.get_list_values({"options": ["x"]})

    assert result == ["x"]


def test_get_list_values_with_choices_key(choice_resolver):
    result = choice_resolver.get_list_values({"choices": [1, 2, 3]})

    assert result == [1, 2, 3]


def test_get_list_values_no_match(choice_resolver):
    result = choice_resolver.get_list_values({"template": "($value)"})

    assert result is None


def test_get_list_values_empty_dict(choice_resolver):
    result = choice_resolver.get_list_values({})

    assert result is None


def test_get_list_values_priority_order(choice_resolver):
    """'values' takes precedence over 'options' when both present."""
    result = choice_resolver.get_list_values({"values": ["a"], "options": ["b"]})

    assert result == ["a"]


def test_is_choice_item_with_choice_key(choice_resolver):
    assert choice_resolver.is_choice_item({"choice": {"values": ["a"]}}) is True


def test_is_choice_item_with_oneof_key(choice_resolver):
    assert choice_resolver.is_choice_item({"oneOf": {"values": ["a"]}}) is True


def test_is_choice_item_non_dict(choice_resolver):
    assert choice_resolver.is_choice_item("plain") is False
    assert choice_resolver.is_choice_item(42) is False


def test_is_choice_item_named_dict(choice_resolver):
    assert choice_resolver.is_choice_item({"name": "hello"}) is False


def test_is_choice_item_multi_key_with_choice(choice_resolver):
    assert choice_resolver.is_choice_item({"choice": True, "values": ["a"]}) is True


def test_normalize_block_single_key(choice_resolver):
    result = choice_resolver.normalize_block({"choice": {"values": ["a", "b"]}})

    assert result == {"values": ["a", "b"]}


def test_normalize_block_multi_key_passthrough(choice_resolver):
    """Multi-key dicts are returned as-is (already in block form)."""
    item = {"choice": True, "values": ["a"], "template": "($value)"}

    result = choice_resolver.normalize_block(item)

    assert result is item


def test_normalize_block_non_dict_wraps_as_values(choice_resolver):
    """Shorthand {"choice": [list]} is normalized to {"values": [list]}."""
    result = choice_resolver.normalize_block({"choice": ["a", "b"]})

    assert result == {"values": ["a", "b"]}


def test_resolve_basic(choice_resolver):
    result = choice_resolver.resolve({"values": ["a", "b"]}, {})

    assert result in ("a", "b")


def test_resolve_chance_zero_skips(choice_resolver):
    result = choice_resolver.resolve({"chance": 0, "values": ["a"]}, {})

    assert result is None


def test_resolve_missing_values_raises(choice_resolver):
    with pytest.raises(ValueError, match="requires 'values'"):
        choice_resolver.resolve({}, {})


def test_safe_weight_valid(choice_resolver):
    assert choice_resolver._safe_weight(2.5) == 2.5
    assert choice_resolver._safe_weight("3") == 3.0


def test_safe_weight_invalid_raises(choice_resolver):
    with pytest.raises(ValueError, match="Invalid weight"):
        choice_resolver._safe_weight("not_a_number")
