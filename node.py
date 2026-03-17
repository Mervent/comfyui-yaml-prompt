"""ComfyUI node: Load a YAML prompt definition and flatten it."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any, Final, List

import yaml

from .jinja_env import render_template
from .parser import YAMLPromptTemplateParser


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
                "seed": (
                    "INT",
                    {
                        "default": -1,
                        "placeholder": "Seed for deterministic randomness",
                    },
                ),
                "jinja_vars": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "{}",
                        "placeholder": '{"enemy": true, "theme": "dark"}',
                    },
                ),
            },
        }

    # ------------------------------------------------------------------
    # Main execution
    # ------------------------------------------------------------------

    def run(
        self,
        file_path: str,
        wildcards_path: str,
        seed: int,
        jinja_vars: str,
    ):  # noqa: D401 – API fixed by ComfyUI
        """Load *file_path*, preprocess with Jinja2, parse YAML, return prompt."""
        path = Path(file_path).expanduser().resolve()

        if wildcards_path.strip():
            wildcard_dir = Path(wildcards_path).expanduser().resolve()
        else:
            wildcard_dir = path.parent / "wildcards"

        try:
            raw_text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return (f"File not found: {path}",)
        except OSError as error:
            return (f"Cannot read file: {error}",)

        try:
            vars_dict = json.loads(jinja_vars) if jinja_vars.strip() else {}
        except json.JSONDecodeError as error:
            return (f"Invalid JSON in jinja_vars: {error}",)

        if seed == -1:
            seed = random.randint(0, 9999999999999)

        # Phase 1: Jinja2 preprocessing
        try:
            rendered = render_template(
                raw_text,
                jinja_vars=vars_dict or None,
                search_paths=[path.parent],
                seed=seed,
                wildcard_dir=wildcard_dir,
            )
        except Exception as error:  # noqa: BLE001 – surface any Jinja2 error
            return (f"Jinja2 error: {error}",)

        # Phase 2: YAML parsing
        try:
            yaml_data = yaml.safe_load(rendered) or {}
        except yaml.YAMLError as error:
            return (f"YAML error: {error}",)

        try:
            parser = YAMLPromptTemplateParser(seed=seed, wildcard_dir=wildcard_dir)
            blocks = parser.parse_document(yaml_data)
        except Exception as error:  # noqa: BLE001 – surface any parser error
            return (f"Parser error: {error}",)

        prompt_lines = [line for block in blocks for line in block]
        prompt_text = "\n\n".join(prompt_lines)
        return (prompt_text,)

    @classmethod
    def IS_CHANGED(cls, *_: Any, **__: Any) -> float:
        return time.time()
