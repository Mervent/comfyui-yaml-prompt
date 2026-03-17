"""Tests for ComfyUI node integration (YAMLPromptLoader)."""

import time


def test_node_input_types(node_class):
    input_types = node_class.INPUT_TYPES()

    required = input_types["required"]
    assert "file_path" in required
    assert "wildcards_path" in required
    assert "seed" in required
    assert "jinja_vars" in required


def test_node_run_valid_file(node, tmp_path):
    yaml_file = tmp_path / "test.yaml"
    yaml_file.write_text("section:\n  - hello\n  - world\n")

    result = node.run(str(yaml_file), "", seed=42, jinja_vars="{}")

    assert isinstance(result, tuple)
    assert len(result) == 1
    assert "hello, world" in result[0]


def test_node_run_missing_file(node):
    result = node.run("/nonexistent/file.yaml", "", seed=42, jinja_vars="{}")

    assert "File not found" in result[0]


def test_node_run_invalid_yaml(node, tmp_path):
    yaml_file = tmp_path / "bad.yaml"
    yaml_file.write_text("key: [unclosed\n")

    result = node.run(str(yaml_file), "", seed=42, jinja_vars="{}")

    assert "YAML error" in result[0]


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


def test_node_is_changed(node_class):
    v1 = node_class.IS_CHANGED()

    time.sleep(0.01)
    v2 = node_class.IS_CHANGED()

    assert v1 != v2
