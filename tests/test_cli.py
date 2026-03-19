"""Tests for CLI interface (cli.py)."""

import subprocess
from pathlib import Path

CLI_PY = str(Path(__file__).resolve().parent.parent / "cli.py")


def test_cli_basic(tmp_path):
    yaml_file = tmp_path / "test.yaml"
    yaml_file.write_text("section:\n  - hello\n  - world\n")

    result = subprocess.run(
        ["python", CLI_PY, str(yaml_file), "--seed", "42"],
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
        ["python", CLI_PY, str(yaml_file), "--seed", "42",
         "--wildcards-dir", str(wc_dir)],
        capture_output=True, text=True,
    )

    assert result.returncode == 0
    assert "sword" in result.stdout


def test_cli_missing_file():
    result = subprocess.run(
        ["python", CLI_PY, "/nonexistent.yaml"],
        capture_output=True, text=True,
    )

    assert result.returncode != 0


def test_cli_invalid_yaml(tmp_path):
    yaml_file = tmp_path / "bad.yaml"
    yaml_file.write_text("{{{")

    result = subprocess.run(
        ["python", CLI_PY, str(yaml_file)],
        capture_output=True, text=True,
    )

    assert result.returncode != 0


def test_cli_multiple_sections_output(tmp_path):
    yaml_file = tmp_path / "multi.yaml"
    yaml_file.write_text("s1:\n  - alpha\ns2:\n  - beta\n")

    result = subprocess.run(
        ["python", CLI_PY, str(yaml_file), "--seed", "42"],
        capture_output=True, text=True,
    )

    assert result.returncode == 0
    assert "alpha" in result.stdout
    assert "beta" in result.stdout
