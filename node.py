from __future__ import annotations

from pathlib import Path
from typing import Final, List

import yaml

from .parser import parse_document


class YAMLPromptLoader:
    """ComfyUI node that loads and parses a YAML prompt file."""

    CATEGORY: Final[str] = "Prompt"
    RETURN_TYPES: Final[List[str]] = ["STRING"]
    RETURN_NAMES: Final[List[str]] = ["prompt"]
    FUNCTION: Final[str] = "run"

    @classmethod
    def INPUT_TYPES(cls):
        """Return node input schema understood by ComfyUI."""
        return {
            "required": {
                "file_path": (
                    "STRING",
                    {
                        "multiline": False,
                        "default": "/absolute/or/relative/path/to/prompt.yaml",
                        "placeholder": "Path to your YAML file …",
                    },
                ),
            },
        }

    def run(self, file_path: str):
        """
        Load *file_path*, parse its YAML content and return a flattened prompt.

        The returned string can be fed directly into downstream prompt nodes.
        """
        path = Path(file_path).expanduser().resolve()
        try:
            yaml_text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return (f"File not found: {path}",)
        except OSError as error:
            return (f"Cannot read file: {error}",)

        try:
            yaml_data = yaml.safe_load(yaml_text) or {}
        except yaml.YAMLError as error:
            return (f"YAML error: {error}",)

        try:
            blocks = parse_document(yaml_data)
        except Exception as error:  # noqa: BLE001 – show any parser error
            return (f"Parser error: {error}",)

        prompt_lines = [line for block in blocks for line in block]
        prompt_text = "\n".join(prompt_lines)
        return (prompt_text,)
