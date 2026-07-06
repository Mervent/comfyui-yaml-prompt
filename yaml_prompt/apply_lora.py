"""ComfyUI node: Apply a LORA_STACK_LBW to MODEL and CLIP with caching."""

from __future__ import annotations

import logging
import time
from typing import Any, Final

from .lora import LoraEntry

logger = logging.getLogger(__name__)

__all__ = ["ApplyLoraStack"]

_LBW_DEFAULT_SEED = 0
_LBW_DEFAULT_A = 4.0
_LBW_DEFAULT_B = 1.0
_LBW_INVERSE = False

_file_cache: dict[str, dict[str, Any]] = {}
_lbw_cache: dict[tuple, tuple[dict, list]] = {}


def _try_import_inspire_lbw():
    try:
        import nodes  # noqa: F811 — ComfyUI runtime

        cls = nodes.NODE_CLASS_MAPPINGS.get("LoraLoaderBlockWeight //Inspire")
        if cls is not None:
            return cls.load_lbw
    except Exception:
        pass
    return None


def _load_lora_file(lora_path: str) -> dict[str, Any]:
    import comfy.utils

    return comfy.utils.load_torch_file(lora_path, safe_load=True)


def _resolve_lora_path(name: str) -> str | None:
    import folder_paths

    return folder_paths.get_full_path("loras", name)


class ApplyLoraStack:
    """Apply LORA_STACK_LBW entries to MODEL/CLIP with per-lora caching."""

    CATEGORY: Final[str] = "Prompt"
    RETURN_TYPES: Final[list[str]] = ["MODEL", "CLIP"]
    RETURN_NAMES: Final[list[str]] = ["model", "clip"]
    FUNCTION: Final[str] = "run"

    def __init__(self) -> None:
        pass

    @classmethod
    def IS_CHANGED(cls, *_: Any, **__: Any) -> float:
        return time.time()

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "lora_stack_lbw": ("LORA_STACK_LBW",),
            },
            "optional": {
                "use_cache": ("BOOLEAN", {"default": True}),
            },
        }

    def run(
        self,
        model: Any,
        clip: Any,
        lora_stack_lbw: list[LoraEntry],
        use_cache: bool = True,
    ) -> tuple[Any, Any]:
        if not use_cache:
            _file_cache.clear()
            _lbw_cache.clear()

        if not lora_stack_lbw:
            return (model, clip)

        t_total = time.perf_counter()
        if use_cache:
            _evict_stale(lora_stack_lbw)

        names = ", ".join(e.name for e in lora_stack_lbw)
        logger.info(
            "── LORA STACK START (%d)%s: %s ──",
            len(lora_stack_lbw),
            "" if use_cache else " [cache disabled]",
            names,
        )

        for i, entry in enumerate(lora_stack_lbw, 1):
            logger.info(
                "  [%d/%d] %s (P=%d)",
                i,
                len(lora_stack_lbw),
                entry.name,
                entry.priority,
            )
            model, clip = self._apply_entry(model, clip, entry, use_cache)

        logger.info(
            "── LORA STACK DONE: %d entries in %.3fs ──",
            len(lora_stack_lbw),
            time.perf_counter() - t_total,
        )
        return (model, clip)

    def _apply_entry(
        self,
        model: Any,
        clip: Any,
        entry: LoraEntry,
        use_cache: bool,
    ) -> tuple[Any, Any]:
        lora_path = _resolve_lora_path(entry.name)
        if lora_path is None:
            logger.warning("LoRA not found, skipping: %s", entry.name)
            return (model, clip)

        if entry.model_weight == 0 and entry.clip_weight == 0:
            logger.info("SKIP LORA (zero weight): %s", entry.name)
            return (model, clip)

        lora_data = _get_lora_file(lora_path, entry.name, use_cache)

        if entry.lbw is not None:
            return self._apply_lbw(model, clip, entry, lora_path, lora_data, use_cache)

        return self._apply_standard(model, clip, entry, lora_path, lora_data)

    def _apply_standard(
        self,
        model: Any,
        clip: Any,
        entry: LoraEntry,
        lora_path: str,
        lora_data: dict[str, Any],
    ) -> tuple[Any, Any]:
        import comfy.sd

        t = time.perf_counter()
        model_out, clip_out = comfy.sd.load_lora_for_models(
            model,
            clip,
            lora_data,
            entry.model_weight,
            entry.clip_weight,
        )
        logger.info(
            "LOAD LORA: %s: %s, %s in %.3fs",
            entry.name,
            entry.model_weight,
            entry.clip_weight,
            time.perf_counter() - t,
        )
        return (model_out, clip_out)

    def _apply_lbw(
        self,
        model: Any,
        clip: Any,
        entry: LoraEntry,
        lora_path: str,
        lora_data: dict[str, Any],
        use_cache: bool,
    ) -> tuple[Any, Any]:
        load_lbw = _try_import_inspire_lbw()
        if load_lbw is None:
            logger.warning(
                "Inspire Pack not installed — ignoring LBW for %s, applying standard",
                entry.name,
            )
            return self._apply_standard(model, clip, entry, lora_path, lora_data)

        lbw_a = entry.lbw_a if entry.lbw_a is not None else _LBW_DEFAULT_A
        lbw_b = entry.lbw_b if entry.lbw_b is not None else _LBW_DEFAULT_B
        block_vector = entry.lbw

        model_id = id(model.model)
        cache_key = (
            lora_path,
            block_vector,
            lbw_a,
            lbw_b,
            _LBW_INVERSE,
            _LBW_DEFAULT_SEED,
            model_id,
        )

        cached = use_cache and cache_key in _lbw_cache
        if cached:
            block_weights, muted_weights = _lbw_cache[cache_key]
            t_lbw = 0.0
        else:
            t = time.perf_counter()
            block_weights, muted_weights, _ = load_lbw(
                model,
                clip,
                lora_data,
                _LBW_INVERSE,
                _LBW_DEFAULT_SEED,
                lbw_a,
                lbw_b,
                block_vector,
            )
            if use_cache:
                _lbw_cache[cache_key] = (block_weights, muted_weights)
            t_lbw = time.perf_counter() - t

        t_apply = time.perf_counter()
        new_model = model.clone()
        new_clip = clip.clone()
        muted_set = set(muted_weights)

        _bulk_add_lbw_patches(
            new_model,
            new_clip,
            block_weights,
            muted_set,
            entry.model_weight,
            entry.clip_weight,
        )
        t_apply = time.perf_counter() - t_apply

        logger.info(
            "LOAD LORA: %s: %s, %s, LBW=%s, A=%s, B=%s (lbw=%.3fs, apply=%.3fs%s)",
            entry.name,
            entry.model_weight,
            entry.clip_weight,
            block_vector,
            lbw_a,
            lbw_b,
            t_lbw,
            t_apply,
            ", cached" if cached else "",
        )

        return (new_model, new_clip)


def _bulk_add_lbw_patches(
    model: Any,
    clip: Any,
    block_weights: dict[str, Any],
    muted_set: set[str],
    strength_model: float,
    strength_clip: float,
) -> None:
    import uuid

    model_sd = model.model.state_dict()
    clip_sd = clip.cond_stage_model.state_dict()

    for k, v in block_weights.items():
        weights, ratio = v
        if k in muted_set:
            continue

        if isinstance(k, tuple):
            key = k[0]
            offset = k[1]
            function = k[2] if len(k) > 2 else None
        else:
            key = k
            offset = None
            function = None

        if "text" in key or "encoder" in key:
            if key in clip_sd:
                current = clip.patcher.patches.get(key, [])
                current.append((strength_clip * ratio, weights, 1.0, offset, function))
                clip.patcher.patches[key] = current
        else:
            if key in model_sd:
                current = model.patches.get(key, [])
                current.append((strength_model * ratio, weights, 1.0, offset, function))
                model.patches[key] = current

    model.patches_uuid = uuid.uuid4()
    clip.patcher.patches_uuid = uuid.uuid4()


def _get_lora_file(lora_path: str, name: str, use_cache: bool) -> dict[str, Any]:
    if use_cache and lora_path in _file_cache:
        logger.info("LORA FILE CACHED: %s", name)
        return _file_cache[lora_path]

    t = time.perf_counter()
    lora_data = _load_lora_file(lora_path)
    if use_cache:
        _file_cache[lora_path] = lora_data
    logger.info("LORA FILE LOADED: %s in %.3fs", name, time.perf_counter() - t)
    return lora_data


def _evict_stale(lora_stack_lbw: list[LoraEntry]) -> None:
    needed_paths: set[str] = set()
    for entry in lora_stack_lbw:
        path = _resolve_lora_path(entry.name)
        if path is not None:
            needed_paths.add(path)

    for path in list(_file_cache):
        if path not in needed_paths:
            del _file_cache[path]

    for key in list(_lbw_cache):
        if key[0] not in needed_paths:
            del _lbw_cache[key]
