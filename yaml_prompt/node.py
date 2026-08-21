"""ComfyUI node: Load a YAML prompt definition and flatten it."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any, Final

from .pipeline import PipelineError, PipelineResult, process_file


def run_yaml_pipeline(
    file_path: str,
    wildcards_path: str,
    seed: int,
    jinja_vars: str,
    keep_lora_tags: bool = False,
) -> PipelineResult:
    path = Path(file_path).expanduser().resolve()

    if wildcards_path.strip():
        wildcard_dir = Path(wildcards_path).expanduser().resolve()
    else:
        wildcard_dir = None

    try:
        vars_dict = json.loads(jinja_vars) if jinja_vars.strip() else {}
    except json.JSONDecodeError as error:
        raise PipelineError(f"Invalid JSON in jinja_vars: {error}") from error

    if seed == -1:
        seed = random.randint(0, 9999999999999)

    return process_file(
        path,
        seed=seed,
        wildcard_dir=wildcard_dir,
        jinja_vars=vars_dict or None,
        keep_lora_tags=keep_lora_tags,
    )


class YAMLPromptLoader:
    """ComfyUI node that loads and parses a YAML prompt file."""

    CATEGORY: Final[str] = "Prompt"
    RETURN_TYPES: Final[list[str]] = ["STRING", "LORA_STACK", "LORA_STACK_LBW"]
    RETURN_NAMES: Final[list[str]] = ["prompt", "lora_stack", "lora_stack_lbw"]
    FUNCTION: Final[str] = "run"

    def run(
        self,
        file_path: str,
        wildcards_path: str,
        seed: int,
        jinja_vars: str,
        keep_lora_tags: bool = False,
    ):  # noqa: D401 – API fixed by ComfyUI
        """Load *file_path*, preprocess with Jinja2, parse YAML, return prompt."""
        result = run_yaml_pipeline(
            file_path,
            wildcards_path=wildcards_path,
            seed=seed,
            jinja_vars=jinja_vars,
            keep_lora_tags=keep_lora_tags,
        )

        return (result.prompt, result.lora_stack, result.lora_stack_lbw)

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
                "keep_lora_tags": (
                    "BOOLEAN",
                    {
                        "default": False,
                    },
                ),
                "jinja_vars": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "placeholder": '{"enemy": true, "theme": "dark"}',
                    },
                ),
            },
        }
