"""
[AI_DIRECTIVE]
ROL: Benchmark in-place de memoria Metal para requests de inferencia.
OBJETIVO: Loguear snapshots de memoria en puntos clave del pipeline para
    construir un modelo de decisión basado en datos reales, no suposiciones.
ENTRADAS: request_id, stage, contexto variable (rest_tokens, kv_off, etc.)
SALIDAS: logs/memory_profile.jsonl (JSONL estructurado) + console log
REGLAS INVIOLABLES:
- No tomar decisiones — solo observar y registrar.
- No crashear nunca — todo best-effort.
- No importar mlx a nivel de módulo (puede no estar disponible).
SSoT: Este módulo es la única fuente de profiling de memoria Metal.
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

_LOG_DIR: Optional[Path] = None
_ENABLED: bool = True


def init(log_root: Path) -> None:
    """Initialize profiler with log directory. Call once at startup."""
    global _LOG_DIR
    _LOG_DIR = log_root / "memory_profile"
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("[MEM_PROFILER] Initialized | log_dir=%s", _LOG_DIR)


def _get_metal_memory() -> Dict[str, Optional[int]]:
    """Read Metal GPU memory stats. Returns bytes or None if unavailable."""
    try:
        import mlx.core as mx

        active = None
        peak = None
        cache = None

        # Active memory (current allocation)
        get_active = getattr(mx, "get_active_memory", None) or getattr(
            mx.metal, "get_active_memory", None
        )
        if get_active:
            active = get_active()

        # Peak memory (high watermark since last reset)
        get_peak = getattr(mx, "get_peak_memory", None) or getattr(
            mx.metal, "get_peak_memory", None
        )
        if get_peak:
            peak = get_peak()

        # Cache memory (MLX internal cache)
        get_cache = getattr(mx, "get_cache_memory", None) or getattr(
            mx.metal, "get_cache_memory", None
        )
        if get_cache:
            cache = get_cache()

        return {"active_bytes": active, "peak_bytes": peak, "cache_bytes": cache}
    except Exception:
        return {"active_bytes": None, "peak_bytes": None, "cache_bytes": None}


def _get_system_ram() -> Optional[int]:
    """Total system RAM in bytes."""
    try:
        import os
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except Exception:
        return None


def _bytes_to_gib(b: Optional[int]) -> Optional[float]:
    """Convert bytes to GiB, rounded to 2 decimals."""
    if b is None:
        return None
    return round(b / (1024 ** 3), 2)


def snapshot(
    request_id: str,
    stage: str,
    *,
    is_anthropic: bool = False,
    rest_tokens: Optional[int] = None,
    kv_cache_offset: Optional[int] = None,
    prompt_tokens: Optional[int] = None,
    model_tokens: Optional[int] = None,
    cache_hit_type: Optional[str] = None,
    matched_prefix: Optional[int] = None,
    output_tokens: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Capture a memory snapshot at a pipeline stage.

    Stages:
        PRE_CACHE       Before cache lookup
        POST_CACHE      After cache lookup (rest_tokens known)
        PRE_PREFILL     Right before generation starts
        POST_GENERATION After generation completes + before cache cleanup

    All parameters besides request_id and stage are optional context.
    """
    if not _ENABLED:
        return

    try:
        metal = _get_metal_memory()

        record = {
            "ts": datetime.now().isoformat(timespec="milliseconds"),
            "epoch": time.time(),
            "request_id": request_id[:12],
            "stage": stage,
            "is_anthropic": is_anthropic,
            "metal_active_gib": _bytes_to_gib(metal["active_bytes"]),
            "metal_peak_gib": _bytes_to_gib(metal["peak_bytes"]),
            "metal_cache_gib": _bytes_to_gib(metal["cache_bytes"]),
            "rest_tokens": rest_tokens,
            "kv_cache_offset": kv_cache_offset,
            "prompt_tokens": prompt_tokens,
            "model_tokens": model_tokens,
            "cache_hit_type": cache_hit_type,
            "matched_prefix": matched_prefix,
            "output_tokens": output_tokens,
        }

        if extra:
            record.update(extra)

        # Remove None values for compact logs
        record = {k: v for k, v in record.items() if v is not None}

        # ── Console log (compact one-liner) ──
        active_gib = record.get("metal_active_gib", "?")
        peak_gib = record.get("metal_peak_gib", "?")
        tokens_info = ""
        if rest_tokens is not None:
            tokens_info += f" rest={rest_tokens}"
        if kv_cache_offset is not None:
            tokens_info += f" kv_off={kv_cache_offset}"
        if output_tokens is not None:
            tokens_info += f" out={output_tokens}"

        logger.info(
            "[MEM] %s | req=%s | Metal: active=%.2fGiB peak=%sGiB%s",
            stage,
            request_id[:8],
            active_gib if isinstance(active_gib, (int, float)) else 0,
            peak_gib,
            tokens_info,
        )

        # ── JSONL file ──
        if _LOG_DIR is not None:
            log_file = _LOG_DIR / "snapshots.jsonl"
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    except Exception as e:
        # Never crash the pipeline for profiling
        logger.debug("[MEM_PROFILER] snapshot error: %s", e)


def reset_peak() -> None:
    """Reset MLX peak memory counter for fresh measurement."""
    try:
        import mlx.core as mx
        reset_fn = getattr(mx, "reset_peak_memory", None) or getattr(
            mx.metal, "reset_peak_memory", None
        )
        if reset_fn:
            reset_fn()
    except Exception:
        pass
