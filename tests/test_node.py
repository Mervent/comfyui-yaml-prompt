"""Tests for ComfyUI node integration (YAMLPromptLoader)."""

import time


def test_node_input_types(node_class):
    input_types = node_class.INPUT_TYPES()

    required = input_types["required"]
    assert "file_path" in required
    assert "wildcards_path" in required
    assert "seed" in required
    assert "jinja_vars" in required


def test_node_return_types(node_class):
    assert node_class.RETURN_TYPES == ["STRING", "LORA_STACK"]
    assert node_class.RETURN_NAMES == ["prompt", "lora_stack"]


def test_node_run_valid_file(node, tmp_path):
    yaml_file = tmp_path / "test.yaml"
    yaml_file.write_text("section:\n  - hello\n  - world\n")

    result = node.run(str(yaml_file), "", seed=42, jinja_vars="{}")

    assert isinstance(result, tuple)
    assert len(result) == 2
    assert "hello, world" in result[0]
    assert result[1] == []


def test_node_run_missing_file(node):
    result = node.run("/nonexistent/file.yaml", "", seed=42, jinja_vars="{}")

    assert "File not found" in result[0]
    assert result[1] == []


def test_node_run_invalid_yaml(node, tmp_path):
    yaml_file = tmp_path / "bad.yaml"
    yaml_file.write_text("key: [unclosed\n")

    result = node.run(str(yaml_file), "", seed=42, jinja_vars="{}")

    assert "YAML error" in result[0]
    assert result[1] == []


def test_node_seed_minus_1(node, tmp_path):
    yaml_file = tmp_path / "test.yaml"
    yaml_file.write_text(
        "s:\n  - choice:\n      values: [a, b, c, d, e, f, g, h, i, j]\n"
    )

    results = {node.run(str(yaml_file), "", seed=-1, jinja_vars="{}")[0] for _ in range(20)}

    assert len(results) > 1


def test_node_custom_wildcard_dir(node, tmp_path):
    wc_dir = tmp_path / "my_wildcards"
    wc_dir.mkdir()
    (wc_dir / "custom.txt").write_text("custom_value\n")
    yaml_file = tmp_path / "test.yaml"
    yaml_file.write_text("s:\n  - __custom__\n")

    result = node.run(str(yaml_file), str(wc_dir), seed=42, jinja_vars="{}")

    assert result[0] == "custom_value"
    assert result[1] == []


def test_node_is_changed(node_class):
    v1 = node_class.IS_CHANGED()

    time.sleep(0.01)
    v2 = node_class.IS_CHANGED()

    assert v1 != v2


def test_node_lora_extraction(node, tmp_path):
    yaml_file = tmp_path / "test.yaml"
    yaml_file.write_text("s:\n  - beautiful scenery <lora:detail_v2:0.8>\n")

    result = node.run(str(yaml_file), "", seed=42, jinja_vars="{}")

    assert "detail_v2" not in result[0]
    assert "beautiful scenery" in result[0]
    assert result[1] == [("detail_v2", 0.8, 0.8)]


def test_node_lora_multiple(node, tmp_path):
    yaml_file = tmp_path / "test.yaml"
    yaml_file.write_text(
        "s1:\n  - photo <lora:real:0.7>\n"
        "s2:\n  - style <lora:anime:0.5:0.3>\n"
    )

    result = node.run(str(yaml_file), "", seed=42, jinja_vars="{}")

    assert "<lora:" not in result[0]
    assert result[1] == [("real", 0.7, 0.7), ("anime", 0.5, 0.3)]


def test_node_lora_in_jinja(node, tmp_path):
    yaml_file = tmp_path / "test.yaml"
    yaml_file.write_text(
        "s:\n  - base {% if mode == 'anime' %}<lora:anime:0.8>{% endif %}\n"
    )

    result_no = node.run(str(yaml_file), "", seed=42, jinja_vars='{}')

    assert result_no[1] == []

    result_yes = node.run(str(yaml_file), "", seed=42, jinja_vars='{"mode": "anime"}')

    assert result_yes[1] == [("anime", 0.8, 0.8)]
    assert "<lora:" not in result_yes[0]


def test_node_jinja_vars_json(node, tmp_path):
    yaml_file = tmp_path / "test.yaml"
    yaml_file.write_text(
        "{% if enemy %}\n"
        "combat:\n"
        "  - fighting\n"
        "{% endif %}\n"
        "meta:\n"
        "  - detailed\n"
    )

    result = node.run(str(yaml_file), "", seed=42, jinja_vars='{"enemy": true}')

    assert "fighting" in result[0]
    assert "detailed" in result[0]


def test_node_jinja_vars_empty(node, tmp_path):
    yaml_file = tmp_path / "test.yaml"
    yaml_file.write_text("meta:\n  - detailed\n")

    result = node.run(str(yaml_file), "", seed=42, jinja_vars="")

    assert "detailed" in result[0]


def test_node_jinja_vars_invalid_json(node, tmp_path):
    yaml_file = tmp_path / "test.yaml"
    yaml_file.write_text("meta:\n  - detailed\n")

    result = node.run(str(yaml_file), "", seed=42, jinja_vars="{bad json}")

    assert "Invalid JSON" in result[0]


def test_node_jinja_error(node, tmp_path):
    yaml_file = tmp_path / "test.yaml"
    yaml_file.write_text("{{ undefined_var }}\nmeta:\n  - detailed\n")

    result = node.run(str(yaml_file), "", seed=42, jinja_vars="{}")

    assert "Jinja2 error" in result[0]


def test_node_include_from_same_dir(node, tmp_path):
    part_file = tmp_path / "part.yaml"
    part_file.write_text("extra:\n  - included\n")
    yaml_file = tmp_path / "main.yaml"
    yaml_file.write_text(
        "meta:\n  - detailed\n{% include 'part.yaml' %}\n"
    )

    result = node.run(str(yaml_file), "", seed=42, jinja_vars="{}")

    assert "detailed" in result[0]
    assert "included" in result[0]
