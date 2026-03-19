"""Tests for CLI interface (cli.py)."""

import subprocess
from pathlib import Path

CLI_PY = str(Path(__file__).resolve().parent.parent / "cli.py")


def _run_cli(*args):
    return subprocess.run(
        ["python", CLI_PY, *args],
        capture_output=True, text=True,
    )


def test_cli_basic(tmp_path):
    yaml_file = tmp_path / "test.yaml"
    yaml_file.write_text("section:\n  - hello\n  - world\n")

    result = _run_cli(str(yaml_file), "--seed", "42")

    assert result.returncode == 0
    assert "hello, world" in result.stdout


def test_cli_with_wildcards_dir(tmp_path):
    wc_dir = tmp_path / "wc"
    wc_dir.mkdir()
    (wc_dir / "items.txt").write_text("sword\n")
    yaml_file = tmp_path / "test.yaml"
    yaml_file.write_text("s:\n  - __items__\n")

    result = _run_cli(str(yaml_file), "--seed", "42", "--wildcards-dir", str(wc_dir))

    assert result.returncode == 0
    assert "sword" in result.stdout


def test_cli_missing_file():
    result = _run_cli("/nonexistent.yaml")

    assert result.returncode != 0


def test_cli_invalid_yaml(tmp_path):
    yaml_file = tmp_path / "bad.yaml"
    yaml_file.write_text("{{{")

    result = _run_cli(str(yaml_file))

    assert result.returncode != 0


def test_cli_multiple_sections_output(tmp_path):
    yaml_file = tmp_path / "multi.yaml"
    yaml_file.write_text("s1:\n  - alpha\ns2:\n  - beta\n")

    result = _run_cli(str(yaml_file), "--seed", "42")

    assert result.returncode == 0
    assert "alpha" in result.stdout
    assert "beta" in result.stdout
