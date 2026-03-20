from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

try:
    from .jinja_env import render_template
    from .lora import extract_lora_tags, strip_lora_tags
    from .parser import YAMLPromptTemplateParser
except ImportError:
    from jinja_env import render_template
    from lora import extract_lora_tags, strip_lora_tags
    from parser import YAMLPromptTemplateParser

__all__ = ["process_file", "PipelineError", "PipelineResult"]


class PipelineError(Exception):
    pass


@dataclass(frozen=True)
class PipelineResult:
    prompt: str
    lora_stack: list[tuple[str, float, float]]
    blocks: list[list[str]]


def process_file(
    file_path: Path,
    *,
    seed: int | None = None,
    wildcard_dir: Path | None = None,
    jinja_vars: dict[str, Any] | None = None,
) -> PipelineResult:
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
        yaml_data = yaml.safe_load(rendered) or {}
    except yaml.YAMLError as error:
        raise PipelineError(f"YAML error: {error}")

    try:
        parser = YAMLPromptTemplateParser(seed=seed, wildcard_dir=wildcard_dir)
        blocks = parser.parse_document(yaml_data)
    except Exception as error:
        raise PipelineError(f"Parser error: {error}") from error

    prompt_lines = [line for block in blocks for line in block]
    prompt_text = "\n\n".join(prompt_lines)
    lora_stack = extract_lora_tags(prompt_text)
    clean_prompt = strip_lora_tags(prompt_text)
    return PipelineResult(prompt=clean_prompt, lora_stack=lora_stack, blocks=blocks)
