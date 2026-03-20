"""ComfyUI node: Load a YAML prompt definition and flatten it."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any, Final

from .pipeline import PipelineError, process_file


class YAMLPromptLoader:
    """ComfyUI node that loads and parses a YAML prompt file."""

    CATEGORY: Final[str] = "Prompt"
    RETURN_TYPES: Final[list[str]] = ["STRING", "LORA_STACK"]
    RETURN_NAMES: Final[list[str]] = ["prompt", "lora_stack"]
    FUNCTION: Final[str] = "run"

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
            wildcard_dir = None

        try:
            vars_dict = json.loads(jinja_vars) if jinja_vars.strip() else {}
        except json.JSONDecodeError as error:
            return (f"Invalid JSON in jinja_vars: {error}", [])

        if seed == -1:
            seed = random.randint(0, 9999999999999)

        try:
            result = process_file(
                path,
                seed=seed,
                wildcard_dir=wildcard_dir,
                jinja_vars=vars_dict or None,
            )
        except PipelineError as error:
            return (str(error), [])

        return (result.prompt, result.lora_stack)

    @classmethod
    def IS_CHANGED(cls, *_: Any, **__: Any) -> float:
        return time.time()

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
