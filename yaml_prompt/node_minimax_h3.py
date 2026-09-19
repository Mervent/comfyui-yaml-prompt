"""ComfyUI node: YAML prompt loader with MiniMax H3 REF2VA field outputs."""

from __future__ import annotations

import time
from typing import Any, Final

from .node import YAMLPromptLoader, run_yaml_pipeline

REF2VA_FIELDS: Final[tuple[str, ...]] = (
    "subject_definitions",
    "summary",
    "retention_analysis",
    "detailed_description",
    "overall_soundscape",
    "non_diegetic_music",
)

SCENE_FIELDS: Final[tuple[str, ...]] = (
    "scene1",
    "scene2",
    "scene3",
    "scene4",
)

_FIELD_DEFAULTS: Final[dict[str, str]] = {"non_diegetic_music": "N/A"}


class YAMLPromptLoaderMiniMaxH3:
    """Load a YAML prompt file and route REF2VA and scene sections to outputs.

    Top-level YAML keys whose names match :data:`REF2VA_FIELDS` or
    :data:`SCENE_FIELDS` are emitted as dedicated ``STRING`` outputs (for the
    MiniMax H3 Director REF2VA text inputs and per-scene fields). The combined
    ``prompt`` output assembles the six REF2VA sections into a MiniMax H3
    ref2va text block, in canonical order, rendering each as a ``name:`` header
    followed by its content on the next line and separated by a blank line; any
    empty section is rendered with ``N/A`` as its content. Lora-stack behaviour
    matches :class:`YAMLPromptLoader`.
    """

    CATEGORY: Final[str] = "Prompt"
    RETURN_TYPES: Final[list[str]] = [
        "STRING",
        "LORA_STACK",
        "LORA_STACK_LBW",
        *["STRING"] * len(REF2VA_FIELDS),
        *["STRING"] * len(SCENE_FIELDS),
    ]
    RETURN_NAMES: Final[list[str]] = [
        "prompt",
        "lora_stack",
        "lora_stack_lbw",
        *REF2VA_FIELDS,
        *SCENE_FIELDS,
    ]
    FUNCTION: Final[str] = "run"

    def run(
        self,
        file_path: str,
        wildcards_path: str,
        seed: int,
        jinja_vars: str,
        keep_lora_tags: bool = False,
    ):
        result = run_yaml_pipeline(
            file_path,
            wildcards_path=wildcards_path,
            seed=seed,
            jinja_vars=jinja_vars,
            keep_lora_tags=keep_lora_tags,
        )

        ref_values = tuple(
            result.sections.get(name, _FIELD_DEFAULTS.get(name, ""))
            for name in REF2VA_FIELDS
        )
        scene_values = tuple(result.sections.get(name, "") for name in SCENE_FIELDS)

        prompt = "\n\n".join(
            f"{name}:\n{value or 'N/A'}"
            for name, value in zip(REF2VA_FIELDS, ref_values)
        )

        return (
            prompt,
            result.lora_stack,
            result.lora_stack_lbw,
            *ref_values,
            *scene_values,
        )

    @classmethod
    def IS_CHANGED(cls, *_: Any, **__: Any) -> float:
        return time.time()

    @classmethod
    def INPUT_TYPES(cls):
        return YAMLPromptLoader.INPUT_TYPES()
