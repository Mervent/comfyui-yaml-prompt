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

RESERVED_FIELDS: Final[tuple[str, ...]] = (*REF2VA_FIELDS, *SCENE_FIELDS)

_FIELD_DEFAULTS: Final[dict[str, str]] = {"non_diegetic_music": "N/A"}


class YAMLPromptLoaderMiniMaxH3:
    """Load a YAML prompt file and route REF2VA and scene sections to outputs.

    Top-level YAML keys whose names match :data:`REF2VA_FIELDS` or
    :data:`SCENE_FIELDS` are emitted as dedicated ``STRING`` outputs (for the
    MiniMax H3 Director REF2VA text inputs and per-scene fields). Every
    remaining section is joined into the combined ``prompt`` output, keeping the
    same lora-stack behaviour as :class:`YAMLPromptLoader`.
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

        prompt = "\n\n".join(
            text
            for name, text in result.sections.items()
            if name not in RESERVED_FIELDS
        )
        ref_values = tuple(
            result.sections.get(name, _FIELD_DEFAULTS.get(name, ""))
            for name in REF2VA_FIELDS
        )
        scene_values = tuple(result.sections.get(name, "") for name in SCENE_FIELDS)

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
