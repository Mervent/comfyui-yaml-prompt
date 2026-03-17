"""Tests for parse_document: full document integration and CLI."""

import subprocess
from pathlib import Path

from parser import YAMLPromptTemplateParser

PARSER_PY = str(Path(__file__).resolve().parent.parent / "parser.py")


def test_empty_document(make_parser):
    p = make_parser(seed=42)

    blocks = p.parse_document({})

    assert blocks == []


def test_vars_section_skipped(make_parser):
    p = make_parser(seed=42)

    blocks = p.parse_document({"vars": {"a": "b"}, "section": ["x"]})

    assert len(blocks) == 1
    assert blocks[0] == ["x"]


def test_multiple_sections(make_parser):
    p = make_parser(seed=42)

    blocks = p.parse_document({"s1": ["a"], "s2": ["b"], "s3": ["c"]})

    assert len(blocks) == 3


def test_global_vars_shared(make_parser):
    p = make_parser(seed=42)
    doc = {"vars": {"x": "hi"}, "s1": ["$x"], "s2": ["$x"]}

    blocks = p.parse_document(doc)

    assert blocks[0] == ["hi"]
    assert blocks[1] == ["hi"]


def test_full_document_integration(make_parser):
    p = make_parser(seed=42)
    doc = {
        "vars": {"mood": "fierce", "animal": "__animals__"},
        "meta": ["detailed", "masterpiece"],
        "subject": {
            "template": "($value:1.2)",
            "values": [
                "$mood warrior",
                {"choice": {"values": ["sword", "axe", "spear"]}},
            ],
        },
        "companion": ["a $animal companion"],
    }

    blocks = p.parse_document(doc)

    assert len(blocks) == 3
    assert blocks[0] == ["detailed, masterpiece"]

    subject_line = blocks[1][0]
    assert subject_line.startswith("(fierce warrior, ")
    assert subject_line.endswith(":1.2)")

    companion_line = blocks[2][0]
    assert "companion" in companion_line
    assert any(a in companion_line for a in ("cat", "dog", "bird"))


def test_section_ordering_preserved(make_parser):
    p = make_parser(seed=42)

    blocks = p.parse_document({"first": ["a"], "second": ["b"], "third": ["c"]})

    assert blocks == [["a"], ["b"], ["c"]]


def test_rand_function_in_vars(make_parser):
    p = make_parser(seed=42)
    doc = {"vars": {"w": "rand(0.0, 1.0)"}, "s": ["weight is $w"]}

    blocks = p.parse_document(doc)

    text = blocks[0][0]
    assert text.startswith("weight is ")
    w_val = float(text.removeprefix("weight is "))
    assert 0.0 <= w_val <= 1.0


def test_cli_basic(tmp_path):
    yaml_file = tmp_path / "test.yaml"
    yaml_file.write_text("section:\n  - hello\n  - world\n")

    result = subprocess.run(
        ["python", PARSER_PY, str(yaml_file), "--seed", "42"],
        capture_output=True, text=True,
    )

    assert result.returncode == 0
    assert "hello, world" in result.stdout


def test_cli_with_wildcards_dir(tmp_path):
    wc_dir = tmp_path / "wc"
    wc_dir.mkdir()
    (wc_dir / "items.txt").write_text("sword\n")
    yaml_file = tmp_path / "test.yaml"
    yaml_file.write_text("s:\n  - __items__\n")

    result = subprocess.run(
        ["python", PARSER_PY, str(yaml_file), "--seed", "42",
         "--wildcards-dir", str(wc_dir)],
        capture_output=True, text=True,
    )

    assert result.returncode == 0
    assert "sword" in result.stdout


def test_cli_missing_file():
    result = subprocess.run(
        ["python", PARSER_PY, "/nonexistent.yaml"],
        capture_output=True, text=True,
    )

    assert result.returncode != 0


def test_cli_invalid_yaml(tmp_path):
    yaml_file = tmp_path / "bad.yaml"
    yaml_file.write_text("{{{")

    result = subprocess.run(
        ["python", PARSER_PY, str(yaml_file)],
        capture_output=True, text=True,
    )

    assert result.returncode != 0


def test_cli_multiple_sections_output(tmp_path):
    yaml_file = tmp_path / "multi.yaml"
    yaml_file.write_text("s1:\n  - alpha\ns2:\n  - beta\n")

    result = subprocess.run(
        ["python", PARSER_PY, str(yaml_file), "--seed", "42"],
        capture_output=True, text=True,
    )

    assert result.returncode == 0
    assert "alpha" in result.stdout
    assert "beta" in result.stdout
