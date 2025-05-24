"""ComfyUI node: Load a YAML prompt definition and flatten it.

Extras added in this revision
-----------------------------
* **Wildcards folder input** – new *wildcards_path* field lets you specify a
  directory containing ``*.txt`` wildcard files. If left empty it defaults to a
  sibling ``wildcards`` directory next to the YAML file.
* The loader sets ``parser._WILDCARD_DIR`` at runtime and clears the cache so
  different nodes can use different wildcard folders in the same workflow.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Final, List

import yaml

from .parser import parse_document


class YAMLPromptLoader:
    """ComfyUI node that loads and parses a YAML prompt file."""

    CATEGORY: Final[str] = "Prompt"
    RETURN_TYPES: Final[List[str]] = ["STRING"]
    RETURN_NAMES: Final[List[str]] = ["prompt"]
    FUNCTION: Final[str] = "run"

    # ---------------------------------------------------------------------
    # ComfyUI input schema
    # ---------------------------------------------------------------------

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
                "wildcards_path": (
                    "STRING",
                    {
                        "multiline": False,
                        "default": "",  # resolved at runtime
                        "placeholder": "Folder with *.txt wildcards (blank → ./wildcards)",
                    },
                ),
            },
        }

    # ------------------------------------------------------------------
    # Main execution
    # ------------------------------------------------------------------

    def run(self, file_path: str, wildcards_path: str):  # noqa: D401 – API fixed by ComfyUI
        """Load *file_path*, parse YAML, and return the flattened prompt."""
        path = Path(file_path).expanduser().resolve()

        # -------------------------------------------------------------
        # Determine and inject wildcard directory
        # -------------------------------------------------------------
        if wildcards_path.strip():
            wildcard_dir = Path(wildcards_path).expanduser().resolve()
        else:
            wildcard_dir = path.parent / "wildcards"

        # -------------------------------------------------------------
        # Read YAML file
        # -------------------------------------------------------------
        try:
            yaml_text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return (f"File not found: {path}",)
        except OSError as error:
            return (f"Cannot read file: {error}",)

        # -------------------------------------------------------------
        # Parse YAML
        # -------------------------------------------------------------
        try:
            yaml_data = yaml.safe_load(yaml_text) or {}
        except yaml.YAMLError as error:
            return (f"YAML error: {error}",)

        try:
            blocks = parse_document(yaml_data, wildcard_dir=wildcard_dir)
        except Exception as error:  # noqa: BLE001 – surface any parser error
            return (f"Parser error: {error}",)

        # -------------------------------------------------------------
        # Flatten to a single prompt string
        # -------------------------------------------------------------
        prompt_lines = [line for block in blocks for line in block]
        prompt_text = "\n\n".join(prompt_lines)
        return (prompt_text,)

    @classmethod
    def IS_CHANGED(cls, *_: Any, **__: Any) -> float:
        return time.time()
