from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Sequence
from typing import Any

import yaml

from .jinja_env import render_template
from .lora import LoraEntry, extract_lora_tags, extract_lora_tags_lbw, strip_lora_tags
from .parser import YAMLPromptTemplateParser

__all__ = ["process_file", "merge_documents", "PipelineError", "PipelineResult"]


class PipelineError(Exception):
    pass


@dataclass(frozen=True)
class PipelineResult:
    prompt: str
    lora_stack: list[tuple[str, float, float]]
    lora_stack_lbw: list[LoraEntry]
    blocks: list[list[str]]
    sections: dict[str, str]


def merge_documents(docs: Sequence[dict[str, Any] | None]) -> dict[str, Any]:
    merged: dict[str, Any] = {}

    for doc in docs:
        if not doc:
            continue
        for key, value in doc.items():
            if (
                key == "vars"
                and key in merged
                and isinstance(merged[key], dict)
                and isinstance(value, dict)
            ):
                merged[key] = {**merged[key], **value}
            else:
                merged[key] = value

    return merged


def process_file(
    file_path: Path,
    *,
    seed: int | None = None,
    wildcard_dir: Path | None = None,
    jinja_vars: dict[str, Any] | None = None,
    keep_lora_tags: bool = False,
) -> PipelineResult:
    if seed is None:
        seed = random.randint(0, 2**63 - 1)

    file_path = file_path.expanduser().resolve()
    if wildcard_dir is None:
        wildcard_dir = file_path.parent / "wildcards"

    try:
        raw_text = file_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise PipelineError(f"File not found: {file_path}")
    except OSError as error:
        raise PipelineError(f"Cannot read file: {error}")

    try:
        rendered = render_template(
            raw_text,
            jinja_vars=jinja_vars,
            search_paths=[file_path.parent],
            seed=seed,
            wildcard_dir=wildcard_dir,
        )
    except Exception as error:
        raise PipelineError(f"Jinja2 error: {error}") from error

    try:
        docs = list(yaml.safe_load_all(rendered))
        yaml_data = merge_documents(docs)
    except PipelineError:
        raise
    except yaml.YAMLError as error:
        raise PipelineError(f"YAML error: {error}")

    try:
        parser = YAMLPromptTemplateParser(seed=seed, wildcard_dir=wildcard_dir)
        named_sections = parser.parse_named_sections(yaml_data)
    except Exception as error:
        raise PipelineError(f"Parser error: {error}") from error

    blocks = [lines for _, lines in named_sections]
    prompt_lines = [line for block in blocks for line in block]
    prompt_text = "\n\n".join(prompt_lines)
    lora_stack = extract_lora_tags(prompt_text)
    lora_stack_lbw = extract_lora_tags_lbw(prompt_text)

    def _finalize(text: str) -> str:
        return text if keep_lora_tags else strip_lora_tags(text)

    sections = {name: _finalize("\n\n".join(lines)) for name, lines in named_sections}
    return PipelineResult(
        prompt=_finalize(prompt_text),
        lora_stack=lora_stack,
        lora_stack_lbw=lora_stack_lbw,
        blocks=blocks,
        sections=sections,
    )
