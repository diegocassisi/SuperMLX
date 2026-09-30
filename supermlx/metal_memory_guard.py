"""
[AI_DIRECTIVE]
ROL: Guardián centralizado de memoria Metal para SuperMLX.
OBJETIVO: Proteger la estabilidad de inferencia mediante control de:
    (1) wired memory (anti-swap del OS),
    (2) cache limit (techo de buffers scratch),
    (3) shader warmup (pre-compilación Metal),
    (4) cleanup post-request inteligente.
ENTRADAS: modelo MLX cargado, configuración dinámica del sistema.
SALIDAS: llamadas a mx.set_wired_limit / set_cache_limit / clear_cache.
REGLAS INVIOLABLES:
- Prohibido hardcodear valores de memoria — TODO se calcula desde device_info.
- Este módulo es SSoT para mx.set_wired_limit, mx.set_cache_limit, mx.clear_cache.
  Ningún otro módulo debe llamarlos directamente.
- Nunca crashear — toda operación es best-effort con logging.
- Siempre activo — no depende de ENABLE_MOE_CACHE ni feature flags.
SSoT: Este módulo es la única fuente de protección de memoria Metal.
"""

import ctypes
import gc
import logging
import time
from typing import Any, Dict, Optional

from .config import SETTINGS

logger = logging.getLogger(__name__)

# ── Lazy MLX import ───────────────────────────────────────────────────────────
# MLX may not be available in test environments.
_mx = None


def _get_mx():
    """Lazy-import mlx.core to avoid import-time side effects."""
    global _mx
    if _mx is None:
        import mlx.core as mx
        _mx = mx
    return _mx


# ── State ─────────────────────────────────────────────────────────────────────

_initialized: bool = False
_device_total_bytes: int = 0
_max_working_set_bytes: int = 0
_os_overhead_bytes: int = 0
_wired_limit_bytes: int = 0
_cache_limit_bytes: int = 0
_shader_warmup_done: bool = False

# Fractions — derived from device characteristics, not hardcoded.
# These are defaults that can be overridden via env vars.
_WIRED_RESERVE_FRACTION = SETTINGS.metal_wired_reserve_fraction  # Reserve this fraction of max_working_set for KV growth + scratch (NOT wired)
_CACHE_LIMIT_FRACTION = SETTINGS.metal_cache_limit_fraction  # Max buffer cache as fraction of device_total
_PREFILL_RELIEF_THRESHOLD = SETTINGS.metal_prefill_relief_threshold  # Trigger relief when projected usage > this fraction of budget
_CLEANUP_CACHE_THRESHOLD_FRACTION = SETTINGS.metal_cleanup_cache_threshold  # Only clear_cache if cache_memory > this fraction of device_total
_PREFILL_RELIEF_MIN_TOKENS = SETTINGS.metal_prefill_relief_min_tokens  # Skip relief for prefills smaller than this
_KV_BYTES_PER_TOKEN_FP16 = SETTINGS.prefill_kv_bytes_per_token_fp16  # ~64 KB/tok for Qwen3.6 class models at fp16
_SCRATCH_ESTIMATE_GB = SETTINGS.metal_scratch_estimate_gb  # Estimated scratch memory during prefill


# ── Public API ────────────────────────────────────────────────────────────────


def init_protections(model: Any = None, kv_bits: Optional[int] = None) -> Dict[str, Any]:
    """Initialize memory protections after model loading.

    Call once after model weights are materialized in Metal memory.
    Calculates and applies wired_limit and cache_limit dynamically
    based on actual device memory and current active allocation.

    Args:
        model: The loaded MLX model (used for shader warmup later).
        kv_bits: KV cache quantization bits (None = fp16 = 16).

    Returns:
        Dict with applied protection parameters for logging.
    """
    global _initialized, _device_total_bytes, _max_working_set_bytes
    global _os_overhead_bytes, _wired_limit_bytes, _cache_limit_bytes

    mx = _get_mx()
    t0 = time.time()

    # ── Read device characteristics ────────────────────────────────────
    try:
        info = mx.device_info()
        _device_total_bytes = info["memory_size"]
        _max_working_set_bytes = info.get(
            "max_recommended_working_set_size", _device_total_bytes
        )
    except Exception as e:
        logger.error("[GUARD] [ERROR] Failed to read device_info: %s", e)
        _device_total_bytes = 24 * (1024 ** 3)  # fallback 24 GB
        _max_working_set_bytes = _device_total_bytes

    # OS overhead = device_total - max_recommended_working_set
    # On M4 Pro 24GB: 25.77 GB - 19.05 GB ≈ 6.7 GB
    _os_overhead_bytes = _device_total_bytes - _max_working_set_bytes

    active_bytes = mx.get_active_memory()

    # ── Wired limit ────────────────────────────────────────────────────
    # Wire all currently active memory (model weights + KV init).
    # Reserve fraction of max_working_set for KV growth + prefill scratch.
    # On M4 Pro 24GB: max_ws=20.45, reserve(15%)=3.07 → headroom=17.38 → wires ~17.3 GB
    reserve_bytes = int(_max_working_set_bytes * _WIRED_RESERVE_FRACTION)
    headroom = _max_working_set_bytes - reserve_bytes
    _wired_limit_bytes = min(active_bytes, max(headroom, 0))

    _apply_wired_limit(_wired_limit_bytes)

    # ── Cache limit ────────────────────────────────────────────────────
    # Limit the MLX buffer pool so scratch from large prefills
    # doesn't eat all available memory between requests.
    _cache_limit_bytes = int(_device_total_bytes * _CACHE_LIMIT_FRACTION)
    _apply_cache_limit(_cache_limit_bytes)

    _initialized = True
    elapsed_ms = (time.time() - t0) * 1000

    result = {
        "device_total_gb": round(_device_total_bytes / 1e9, 2),
        "max_working_set_gb": round(_max_working_set_bytes / 1e9, 2),
        "os_overhead_gb": round(_os_overhead_bytes / 1e9, 2),
        "active_gb": round(active_bytes / 1e9, 2),
        "wired_limit_gb": round(_wired_limit_bytes / 1e9, 2),
        "cache_limit_gb": round(_cache_limit_bytes / 1e9, 2),
        "elapsed_ms": round(elapsed_ms, 1),
    }

    logger.info(
        "[GUARD] [CONFIG] Parameters | wired_reserve_frac=%.2f | cache_limit_frac=%.2f | "
        "prefill_relief_threshold=%.2f | cleanup_cache_threshold=%.2f | "
        "kv_bytes_per_tok=%d | scratch_estimate=%.1fGB",
        _WIRED_RESERVE_FRACTION, _CACHE_LIMIT_FRACTION,
        _PREFILL_RELIEF_THRESHOLD, _CLEANUP_CACHE_THRESHOLD_FRACTION,
        _KV_BYTES_PER_TOKEN_FP16, _SCRATCH_ESTIMATE_GB,
    )
    logger.info(
        "[GUARD] [INIT] Protections applied | device=%.1fGB | working_set=%.1fGB | "
        "active=%.1fGB | wired=%.1fGB | cache_limit=%.1fGB | %.0fms",
        result["device_total_gb"], result["max_working_set_gb"],
        result["active_gb"], result["wired_limit_gb"],
        result["cache_limit_gb"], elapsed_ms,
    )

    return result


def warmup_shaders(
    model: Any,
    tokenizer: Any,
    cache_factory: Any,
    request_id: str = "warmup",
) -> float:
    """Compile Metal shaders by running a real forward pass.

    Call after init_protections, before first user request.
    This ensures the first request doesn't pay the shader compilation penalty.

    Args:
        model: The loaded MLX model.
        tokenizer: The model tokenizer.
        cache_factory: Callable that creates a fresh KV cache for the model.
        request_id: For logging.

    Returns:
        Elapsed time in seconds.
    """
    global _shader_warmup_done
    if _shader_warmup_done:
        return 0.0

    mx = _get_mx()
    t0 = time.time()

    try:
        # Create a minimal prompt that triggers all shader paths
        # (attention, MLP, embedding, output projection)
        dummy_tokens = mx.array([[tokenizer.eos_token_id or 0]])
        cache = cache_factory()

        # Single forward pass — compiles all Metal shaders
        model(dummy_tokens, cache=cache)
        mx.eval(cache[0].state if hasattr(cache[0], "state") else cache[0])

        # Clean up the dummy cache — don't let it pollute real inference
        del cache
        mx.clear_cache()

        _shader_warmup_done = True
        elapsed = time.time() - t0
        logger.info(
            "[GUARD] [INIT] Shader warmup completed | elapsed=%.2fs | %s",
            elapsed, _metal_mem_str(),
        )
        return elapsed

    except Exception as e:
        _shader_warmup_done = True  # don't retry — let first request handle it
        elapsed = time.time() - t0
        logger.warning(
            "[GUARD] [INIT] Shader warmup failed (first request will be slow): %s | %.2fs",
            e, elapsed,
        )
        return elapsed


def pre_prefill_gate(
    rest_tokens: int,
    kv_bits: Optional[int] = None,
    request_id: str = "",
) -> bool:
    """Evaluate memory state before prefill and take preventive action if needed.

    Estimates projected memory usage after prefill. If it exceeds the
    threshold, runs gc + clear_cache + malloc_zone_pressure_relief.

    Args:
        rest_tokens: Number of tokens to prefill.
        kv_bits: KV quantization bits (None = fp16).
        request_id: For logging.

    Returns:
        True if relief was applied, False if skipped (sufficient headroom).
    """
    if rest_tokens < _PREFILL_RELIEF_MIN_TOKENS:
        return False

    mx = _get_mx()

    try:
        active_bytes = mx.get_active_memory()
        effective_kv_bits = kv_bits or 16
        kv_bytes_per_tok = _KV_BYTES_PER_TOKEN_FP16 * effective_kv_bits / 16
        kv_growth = rest_tokens * kv_bytes_per_tok
        scratch = _SCRATCH_ESTIMATE_GB * 1e9
        projected = active_bytes + kv_growth + scratch
        budget = _get_budget_bytes()
        threshold = budget * _PREFILL_RELIEF_THRESHOLD

        if projected < threshold:
            logger.info(
                "[GUARD] [DECISION] pre_prefill_gate: SKIP (headroom) | "
                "rest=%d | projected=%.2fGB | threshold=%.2fGB | %s",
                rest_tokens, projected / 1e9, threshold / 1e9,
                _metal_mem_str(),
            )
            return False

    except Exception as e:
        logger.warning(
            "[GUARD] [ERROR] pre_prefill_gate estimation failed: %s | "
            "falling through to relief (safe default)", e,
        )
        projected = -1.0
        threshold = -1.0

    # ── Apply relief ───────────────────────────────────────────────────
    mem_before = _metal_mem_str()
    gc.collect()
    mx.clear_cache()
    _malloc_pressure_relief()
    mem_after = _metal_mem_str()

    logger.info(
        "[GUARD] [DECISION] pre_prefill_gate: RELIEF applied | "
        "rest=%d | projected=%.2fGB/%.2fGB | before=%s | after=%s",
        rest_tokens, projected / 1e9, threshold / 1e9,
        mem_before, mem_after,
    )
    return True


def post_request_cleanup(request_id: str = "") -> None:
    """Intelligent post-request memory cleanup.

    Only calls mx.clear_cache() if the buffer cache exceeds a dynamic
    threshold. Avoids destroying the pool unnecessarily (which causes
    re-allocation overhead on the next request).
    """
    mx = _get_mx()

    try:
        cache_bytes = mx.get_cache_memory()
        threshold = int(_device_total_bytes * _CLEANUP_CACHE_THRESHOLD_FRACTION)

        if cache_bytes > threshold:
            gc.collect()
            mx.clear_cache()
            logger.info(
                "[GUARD] [RESULT] post_request_cleanup: cleared | "
                "cache_was=%.2fGB > threshold=%.2fGB | %s",
                cache_bytes / 1e9, threshold / 1e9, _metal_mem_str(),
            )
        else:
            # Light cleanup: just GC Python objects, keep Metal buffer pool
            gc.collect()
            logger.info(
                "[GUARD] [RESULT] post_request_cleanup: kept cache (gc only) | "
                "cache=%.2fGB <= threshold=%.2fGB | %s",
                cache_bytes / 1e9, threshold / 1e9, _metal_mem_str(),
            )

    except Exception as e:
        # Fallback: always clear on error
        gc.collect()
        mx.clear_cache()
        logger.warning("[GUARD] [ERROR] post_request_cleanup fallback: %s", e)


def update_wired_after_change(reason: str = "") -> int:
    """Re-calculate and apply wired limit after a significant memory change.

    Call after expert cache expansion, model reload, or similar events
    that change the active memory footprint.

    Args:
        reason: Description of what changed (for logging).

    Returns:
        New wired limit in bytes.
    """
    global _wired_limit_bytes

    mx = _get_mx()
    active_bytes = mx.get_active_memory()
    reserve_bytes = int(_max_working_set_bytes * _WIRED_RESERVE_FRACTION)
    headroom = _max_working_set_bytes - reserve_bytes
    _wired_limit_bytes = min(active_bytes, max(headroom, 0))

    _apply_wired_limit(_wired_limit_bytes)

    logger.info(
        "[GUARD] [CONFIG] wired_limit updated | reason=%s | active=%.2fGB | "
        "new_wired=%.2fGB | reserve=%.2fGB",
        reason or "unspecified",
        active_bytes / 1e9, _wired_limit_bytes / 1e9, reserve_bytes / 1e9,
    )
    return _wired_limit_bytes


def get_memory_state() -> Dict[str, float]:
    """Return current memory state as a normalized dict (values in GB).

    Safe to call at any time — never raises.
    """
    mx = _get_mx()
    try:
        return {
            "active_gb": round(mx.get_active_memory() / 1e9, 3),
            "cache_gb": round(mx.get_cache_memory() / 1e9, 3),
            "peak_gb": round(mx.get_peak_memory() / 1e9, 3),
            "device_total_gb": round(_device_total_bytes / 1e9, 2),
            "wired_limit_gb": round(_wired_limit_bytes / 1e9, 2),
            "cache_limit_gb": round(_cache_limit_bytes / 1e9, 2),
        }
    except Exception:
        return {"error": "unavailable"}


def get_metal_budget_gb() -> float:
    """Return effective Metal budget in GB for external callers.

    Budget = max_recommended_working_set (or device_total - OS_overhead).
    Can be overridden with METAL_BUDGET_GB env var.
    """
    if SETTINGS.metal_budget_gb:
        return SETTINGS.metal_budget_gb
    return _max_working_set_bytes / 1e9 if _max_working_set_bytes > 0 else 20.0


def force_clear_cache(reason: str = "") -> None:
    """Unconditional clear_cache — for use in emergency/diagnostic paths only.

    Prefer post_request_cleanup() for normal operation.
    """
    mx = _get_mx()
    mem_before = _metal_mem_str()
    gc.collect()
    mx.clear_cache()
    logger.info(
        "[GUARD] [DECISION] force_clear_cache | reason=%s | before=%s | after=%s",
        reason or "unspecified", mem_before, _metal_mem_str(),
    )


# ── Private helpers ───────────────────────────────────────────────────────────


def _apply_wired_limit(limit_bytes: int) -> None:
    """Apply mx.set_wired_limit if available."""
    mx = _get_mx()
    try:
        if hasattr(mx, "set_wired_limit"):
            mx.set_wired_limit(limit_bytes)
            logger.info("[GUARD] [DATA] set_wired_limit applied | %.2fGB", limit_bytes / 1e9)
        elif hasattr(mx.metal, "set_wired_limit"):
            mx.metal.set_wired_limit(limit_bytes)
            logger.info("[GUARD] [DATA] set_wired_limit applied (metal) | %.2fGB", limit_bytes / 1e9)
        else:
            logger.warning("[GUARD] [ERROR] set_wired_limit API not available")
    except Exception as e:
        logger.warning("[GUARD] [ERROR] set_wired_limit failed: %s", e)


def _apply_cache_limit(limit_bytes: int) -> None:
    """Apply mx.set_cache_limit if available."""
    mx = _get_mx()
    try:
        if hasattr(mx, "set_cache_limit"):
            mx.set_cache_limit(limit_bytes)
            logger.info("[GUARD] [DATA] set_cache_limit applied | %.2fGB", limit_bytes / 1e9)
        elif hasattr(mx.metal, "set_cache_limit"):
            mx.metal.set_cache_limit(limit_bytes)
            logger.info("[GUARD] [DATA] set_cache_limit applied (metal) | %.2fGB", limit_bytes / 1e9)
        else:
            logger.warning("[GUARD] [ERROR] set_cache_limit API not available")
    except Exception as e:
        logger.warning("[GUARD] [ERROR] set_cache_limit failed: %s", e)


def _get_budget_bytes() -> float:
    """Return effective budget in bytes for internal calculations."""
    if SETTINGS.metal_budget_gb:
        return SETTINGS.metal_budget_gb * 1e9
    return float(_max_working_set_bytes) if _max_working_set_bytes > 0 else 20e9


def _malloc_pressure_relief() -> None:
    """macOS-specific: tell the C allocator to return freed pages to the OS."""
    try:
        libc = ctypes.CDLL("libSystem.dylib")
        libc.malloc_zone_pressure_relief(0, 0)
    except Exception:
        pass  # Non-macOS or ctypes unavailable


def _metal_mem_str() -> str:
    """Format current Metal memory as a compact string for logging."""
    mx = _get_mx()
    try:
        active = mx.get_active_memory() / 1e9
        peak = mx.get_peak_memory() / 1e9
        cache = mx.get_cache_memory() / 1e9
        return f"metal={active:.2f}GB peak={peak:.2f}GB cache={cache:.2f}GB"
    except Exception:
        return "metal=N/A"
