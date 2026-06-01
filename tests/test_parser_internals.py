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


def test_extract_section_config_defaults(parser):
    variables, item_tpl, block_tpl, separator = parser._extract_section_config(
        {"values": ["a"]}, {}
    )

    assert item_tpl == "$value"
    assert block_tpl is None
    assert separator == ", "
    assert variables == {}


def test_extract_section_config_custom_template(parser):
    _, item_tpl, block_tpl, _ = parser._extract_section_config(
        {"template": "($value)", "values": ["a"]}, {}
    )

    assert item_tpl == "($value)"
    assert block_tpl is None


def test_extract_section_config_block_template(parser):
    _, item_tpl, block_tpl, _ = parser._extract_section_config(
        {"block_template": "[$value]", "values": ["a"]}, {}
    )

    assert item_tpl == "$value"
    assert block_tpl == "[$value]"


def test_extract_section_config_local_vars(parser):
    variables, _, _, _ = parser._extract_section_config(
        {"vars": {"x": "hello"}, "values": ["$x"]}, {}
    )

    assert variables["x"] == "hello"


def test_extract_section_config_non_dict(parser):
    _, item_tpl, block_tpl, separator = parser._extract_section_config("text", {})

    assert item_tpl == "$value"
    assert block_tpl is None
    assert separator == ", "


def test_extract_section_config_inherits_base_vars(parser):
    variables, _, _, _ = parser._extract_section_config(
        {"values": ["a"]}, {"existing": "var"}
    )

    assert variables["existing"] == "var"


def test_extract_section_config_template_expands_vars(parser):
    """The template string itself undergoes $var expansion."""
    _, item_tpl, _, _ = parser._extract_section_config(
        {"template": "($value:$w)", "vars": {"w": "1.2"}, "values": ["a"]}, {}
    )

    assert item_tpl == "($value:1.2)"


def test_apply_item_template_identity(parser):
    result = parser._apply_item_template("hello", "$value", {})

    assert result == "hello"


def test_apply_item_template_wrap(parser):
    result = parser._apply_item_template("fire", "($value:1.2)", {})

    assert result == "(fire:1.2)"


def test_apply_item_template_with_variable(parser):
    result = parser._apply_item_template("text", "$prefix $value", {"prefix": "hey"})

    assert result == "hey text"


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
    """Choice result joins into the pending string buffer before flush."""
    items = ["a", "b", {"choice": {"values": ["x"]}}]

    result = parser._render_items(items, {}, "$value")

    assert result == ["a, b, x"]


def test_render_items_named_dict_flushes_pending(parser):
    """Named dict triggers pending string flush as a separate line."""
    items = ["a", "b", {"name": "special"}]

    result = parser._render_items(items, {}, "$value")

    assert result == ["a, b", "special"]


def test_render_items_with_template(parser):
    result = parser._render_items(["fire"], {}, "($value:1.2)")

    assert result == ["(fire:1.2)"]


def test_render_items_choice_returning_none(parser):
    """Pending strings still flush even when the choice is skipped by chance."""
    items = ["a", {"choice": {"values": [{"name": "x", "chance": 0}]}}]

    result = parser._render_items(items, {}, "$value")

    assert result == ["a"]


def test_render_items_named_dict_with_chance_zero(parser):
    items = [{"name": "skip", "chance": 0}]

    result = parser._render_items(items, {}, "$value")

    assert result == []


def test_render_items_variable_expansion(parser):
    result = parser._render_items(["$color cat"], {"color": "black"}, "$value")

    assert result == ["black cat"]


def test_extract_section_config_custom_separator(parser):
    _, _, _, separator = parser._extract_section_config(
        {"separator": " | ", "values": ["a"]}, {}
    )

    assert separator == " | "


def test_extract_section_config_separator_literal(parser):
    _, _, _, separator = parser._extract_section_config(
        {"separator": " . ", "values": ["a"]}, {}
    )

    assert separator == " . "


def test_flush_pending_custom_separator(parser):
    result = parser._flush_pending(["a", "b", "c"], "$value", {}, " | ")

    assert result == "a | b | c"


def test_render_items_custom_separator(parser):
    result = parser._render_items(["a", "b", "c"], {}, "$value", " | ")

    assert result == ["a | b | c"]


def test_render_items_choice_with_custom_separator(parser):
    items = ["a", "b", {"choice": {"values": ["x"]}}]

    result = parser._render_items(items, {}, "$value", " | ")

    assert result == ["a | b | x"]
