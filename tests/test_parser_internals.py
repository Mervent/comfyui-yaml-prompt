"""Tests for parser internal helpers — safety net before refactoring."""

from yaml_prompt.parser import YAMLPromptTemplateParser


# --- _get_list_values ---


def test_get_list_values_with_values_key(parser):
    result = parser._get_list_values({"values": ["a", "b"]})

    assert result == ["a", "b"]


def test_get_list_values_with_options_key(parser):
    result = parser._get_list_values({"options": ["x"]})

    assert result == ["x"]


def test_get_list_values_with_choices_key(parser):
    result = parser._get_list_values({"choices": [1, 2, 3]})

    assert result == [1, 2, 3]


def test_get_list_values_no_match(parser):
    result = parser._get_list_values({"template": "($value)"})

    assert result is None


def test_get_list_values_empty_dict(parser):
    result = parser._get_list_values({})

    assert result is None


def test_get_list_values_priority_order(parser):
    """First matching key wins (values checked before options)."""
    result = parser._get_list_values({"values": ["a"], "options": ["b"]})

    assert result == ["a"]


# --- _extract_items ---


def test_extract_items_dict_with_values(parser):
    result = parser._extract_items({"values": ["a", "b"]})

    assert result == ["a", "b"]


def test_extract_items_dict_without_values(parser):
    result = parser._extract_items({"template": "($value)"})

    assert result == []


def test_extract_items_list(parser):
    result = parser._extract_items(["x", "y"])

    assert result == ["x", "y"]


def test_extract_items_scalar_string(parser):
    result = parser._extract_items("hello")

    assert result == ["hello"]


def test_extract_items_scalar_numeric(parser):
    result = parser._extract_items(42)

    assert result == [42]


# --- _extract_section_config ---


def test_extract_section_config_defaults(parser):
    variables, item_tpl, block_tpl = parser._extract_section_config(
        {"values": ["a"]}, {}
    )

    assert item_tpl == "$value"
    assert block_tpl is None
    assert variables == {}


def test_extract_section_config_custom_template(parser):
    _, item_tpl, block_tpl = parser._extract_section_config(
        {"template": "($value)", "values": ["a"]}, {}
    )

    assert item_tpl == "($value)"
    assert block_tpl is None


def test_extract_section_config_block_template(parser):
    _, item_tpl, block_tpl = parser._extract_section_config(
        {"block_template": "[$value]", "values": ["a"]}, {}
    )

    assert item_tpl == "$value"
    assert block_tpl == "[$value]"


def test_extract_section_config_local_vars(parser):
    variables, _, _ = parser._extract_section_config(
        {"vars": {"x": "hello"}, "values": ["$x"]}, {}
    )

    assert variables["x"] == "hello"


def test_extract_section_config_non_dict(parser):
    _, item_tpl, block_tpl = parser._extract_section_config("text", {})

    assert item_tpl == "$value"
    assert block_tpl is None


def test_extract_section_config_inherits_base_vars(parser):
    variables, _, _ = parser._extract_section_config(
        {"values": ["a"]}, {"existing": "var"}
    )

    assert variables["existing"] == "var"


def test_extract_section_config_template_expands_vars(parser):
    _, item_tpl, _ = parser._extract_section_config(
        {"template": "($value:$w)", "vars": {"w": "1.2"}, "values": ["a"]}, {}
    )

    assert item_tpl == "($value:1.2)"


# --- _apply_item_template ---


def test_apply_item_template_identity(parser):
    result = parser._apply_item_template("hello", "$value", {})

    assert result == "hello"


def test_apply_item_template_wrap(parser):
    result = parser._apply_item_template("fire", "($value:1.2)", {})

    assert result == "(fire:1.2)"


def test_apply_item_template_with_variable(parser):
    result = parser._apply_item_template("text", "$prefix $value", {"prefix": "hey"})

    assert result == "hey text"


# --- _flush_pending ---


def test_flush_pending_empty(parser):
    result = parser._flush_pending([], "$value", {})

    assert result is None


def test_flush_pending_single(parser):
    result = parser._flush_pending(["hello"], "$value", {})

    assert result == "hello"


def test_flush_pending_multiple(parser):
    result = parser._flush_pending(["a", "b", "c"], "$value", {})

    assert result == "a, b, c"


def test_flush_pending_with_template(parser):
    result = parser._flush_pending(["a", "b"], "($value)", {})

    assert result == "(a, b)"


# --- _render_items ---


def test_render_items_all_strings(parser):
    result = parser._render_items(["a", "b", "c"], {}, "$value")

    assert result == ["a, b, c"]


def test_render_items_empty(parser):
    result = parser._render_items([], {}, "$value")

    assert result == []


def test_render_items_single_string(parser):
    result = parser._render_items(["hello"], {}, "$value")

    assert result == ["hello"]


def test_render_items_choice_after_strings(parser):
    """Choice item after pending strings: result joins into same line."""
    items = ["a", "b", {"choice": {"values": ["x"]}}]

    result = parser._render_items(items, {}, "$value")

    assert result == ["a, b, x"]


def test_render_items_named_dict_flushes_pending(parser):
    """Named dict item causes pending strings to flush as separate line."""
    items = ["a", "b", {"name": "special"}]

    result = parser._render_items(items, {}, "$value")

    assert result == ["a, b", "special"]


def test_render_items_with_template(parser):
    result = parser._render_items(["fire"], {}, "($value:1.2)")

    assert result == ["(fire:1.2)"]


def test_render_items_choice_returning_none(parser):
    """Choice with chance=0 returns None; pending strings still flushed."""
    items = ["a", {"choice": {"values": [{"name": "x", "chance": 0}]}}]

    result = parser._render_items(items, {}, "$value")

    assert result == ["a"]


def test_render_items_named_dict_with_chance_zero(parser):
    """Named dict with chance=0 is skipped entirely."""
    items = [{"name": "skip", "chance": 0}]

    result = parser._render_items(items, {}, "$value")

    assert result == []


def test_render_items_variable_expansion(parser):
    result = parser._render_items(["$color cat"], {"color": "black"}, "$value")

    assert result == ["black cat"]


# --- _random_chance ---


def test_random_chance_always_passes_at_one():
    assert all(YAMLPromptTemplateParser._random_chance(1.0) for _ in range(100))


def test_random_chance_distribution():
    passes = sum(1 for _ in range(1000) if YAMLPromptTemplateParser._random_chance(0.5))

    assert 350 < passes < 650
