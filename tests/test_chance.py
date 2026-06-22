import pytest


def test_random_chance_always_passes_at_one(chance_evaluator):
    assert all(chance_evaluator._random_chance(1.0) for _ in range(100))


def test_random_chance_distribution(chance_evaluator):
    passes = sum(1 for _ in range(1000) if chance_evaluator._random_chance(0.5))

    assert 350 < passes < 650


def test_check_passes_without_chance_key(chance_evaluator):
    assert chance_evaluator.check({"name": "hello"}) is True


def test_check_fails_at_zero(chance_evaluator):
    assert chance_evaluator.check({"name": "hello", "chance": 0}) is False


def test_check_passes_at_one(chance_evaluator):
    assert chance_evaluator.check({"name": "hello", "chance": 1}) is True


def test_evaluate_true_at_one(chance_evaluator):
    assert chance_evaluator.evaluate(1.0, "anything") is True


def test_evaluate_false_at_zero(chance_evaluator):
    assert chance_evaluator.evaluate(0.0, "anything") is False


def test_evaluate_dict_chance(chance_evaluator):
    assert chance_evaluator.evaluate({"value": 1.0}, "anything") is True
    assert chance_evaluator.evaluate({"value": 0.0}, "anything") is False


def test_safe_chance_clamps_negative(chance_evaluator):
    assert chance_evaluator._safe_chance(-5.0) == 0.0


def test_safe_chance_clamps_above_one(chance_evaluator):
    assert chance_evaluator._safe_chance(3.0) == 1.0


def test_safe_chance_invalid_raises(chance_evaluator):
    with pytest.raises(ValueError, match="Invalid chance"):
        chance_evaluator._safe_chance("not_a_number")


def test_apply_to_section_no_chance(chance_evaluator):
    section = {"values": ["a"]}

    result = chance_evaluator.apply_to_section(section)

    assert result == {"values": ["a"]}


def test_apply_to_section_strips_chance_key(chance_evaluator):
    """Passing section is returned without the 'chance' key."""
    section = {"chance": 1.0, "values": ["a"]}

    result = chance_evaluator.apply_to_section(section)

    assert result == {"values": ["a"]}


def test_apply_to_section_returns_none_on_skip(chance_evaluator):
    result = chance_evaluator.apply_to_section({"chance": 0, "values": ["a"]})

    assert result is None


def test_apply_to_section_non_dict_passthrough(chance_evaluator):
    """Strings, None, and other non-dicts pass through unchanged."""
    assert chance_evaluator.apply_to_section("just text") == "just text"
    assert chance_evaluator.apply_to_section(None) is None


def test_promote_chance_plain_float_without_stable(chance_evaluator):
    parent = {"chance": 0.5, "values": ["a"]}

    result = chance_evaluator._promote_chance(parent)

    assert result == 0.5


def test_promote_chance_plain_float_with_sibling_stable(chance_evaluator):
    parent = {"chance": 0.5, "stable": False, "values": ["a"]}

    result = chance_evaluator._promote_chance(parent)

    assert result == {"value": 0.5, "stable": False}


def test_promote_chance_dict_unchanged_despite_sibling_stable(chance_evaluator):
    parent = {"chance": {"value": 0.7}, "stable": False, "values": ["a"]}

    result = chance_evaluator._promote_chance(parent)

    assert result == {"value": 0.7}


def test_check_sibling_stable_false_uses_random(chance_evaluator):
    obj = {"name": "x", "chance": 0.5, "stable": False}

    results = set()
    for _ in range(50):
        results.add(chance_evaluator.check(obj))

    assert len(results) == 2


def test_apply_to_section_sibling_stable_false_uses_random(chance_evaluator):
    section = {"chance": 0.5, "stable": False, "values": ["a"]}

    results = set()
    for _ in range(50):
        results.add(chance_evaluator.apply_to_section(section) is not None)

    assert len(results) == 2
