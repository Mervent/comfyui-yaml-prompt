import pytest

from yaml_prompt.node import YAMLPromptLoader
from yaml_prompt.node_minimax_h3 import REF2VA_FIELDS, YAMLPromptLoaderMiniMaxH3


@pytest.fixture
def h3_node():
    return YAMLPromptLoaderMiniMaxH3()


@pytest.fixture
def write_yaml(tmp_path):
    def _write(content, name="test.yaml"):
        path = tmp_path / name
        path.write_text(content)
        return str(path)

    return _write


def _field(result, name):
    return result[3 + REF2VA_FIELDS.index(name)]


def test_return_names_expose_ref2va_fields():
    assert YAMLPromptLoaderMiniMaxH3.RETURN_NAMES == [
        "prompt",
        "lora_stack",
        "lora_stack_lbw",
        "subject_definitions",
        "summary",
        "retention_analysis",
        "detailed_description",
        "overall_soundscape",
        "non_diegetic_music",
    ]


def test_return_types_are_string_plus_lora_stacks():
    assert YAMLPromptLoaderMiniMaxH3.RETURN_TYPES == [
        "STRING",
        "LORA_STACK",
        "LORA_STACK_LBW",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
    ]


def test_input_types_match_base_loader():
    assert YAMLPromptLoaderMiniMaxH3.INPUT_TYPES() == YAMLPromptLoader.INPUT_TYPES()


def test_ref2va_sections_routed_to_named_outputs(h3_node, write_yaml):
    path = write_yaml(
        "subject_definitions:\n  - tall knight\n  - red armor\n"
        "overall_soundscape:\n  - distant thunder\n"
    )

    result = h3_node.run(path, "", seed=42, jinja_vars="{}")

    assert _field(result, "subject_definitions") == "tall knight, red armor"
    assert _field(result, "overall_soundscape") == "distant thunder"


def test_non_reserved_sections_go_to_prompt(h3_node, write_yaml):
    path = write_yaml(
        "subject_definitions:\n  - a hero\nscene:\n  - castle courtyard\n"
    )

    result = h3_node.run(path, "", seed=42, jinja_vars="{}")

    assert result[0] == "castle courtyard"
    assert "a hero" not in result[0]


def test_missing_music_defaults_to_na(h3_node, write_yaml):
    path = write_yaml("scene:\n  - hi\n")

    result = h3_node.run(path, "", seed=42, jinja_vars="{}")

    assert _field(result, "non_diegetic_music") == "N/A"


def test_missing_ref2va_fields_default_to_empty(h3_node, write_yaml):
    path = write_yaml("scene:\n  - hi\n")

    result = h3_node.run(path, "", seed=42, jinja_vars="{}")

    assert _field(result, "summary") == ""
    assert _field(result, "subject_definitions") == ""


def test_lora_stripped_from_ref2va_output(h3_node, write_yaml):
    path = write_yaml("subject_definitions:\n  - hero <lora:detail_v2:0.8>\n")

    result = h3_node.run(path, "", seed=42, jinja_vars="{}")

    assert "detail_v2" not in _field(result, "subject_definitions")
    assert "hero" in _field(result, "subject_definitions")
    assert result[1] == [("detail_v2.safetensors", 0.8, 0.8)]


def test_keep_lora_tags_preserves_tag_in_ref2va_output(h3_node, write_yaml):
    path = write_yaml("subject_definitions:\n  - hero <lora:detail_v2:0.8>\n")

    result = h3_node.run(path, "", seed=42, jinja_vars="{}", keep_lora_tags=True)

    assert "<lora:detail_v2:0.8>" in _field(result, "subject_definitions")
