"""
[AI_DIRECTIVE]
ROL: Adaptive chunked prefill engine
OBJETIVO: Pre-process prompt tokens in memory-aware chunks before generation,
          adapting chunk size based on available Metal memory
ENTRADAS: rest_tokens (suffix to prefill), prompt_cache, model
SALIDAS: Remaining tokens after prefill (usually last token for generation)
REGLAS INVIOLABLES:
- Must hold model_lock during prefill
- Chunk size adapts DOWN on Metal pressure, never UP beyond base
- Progress logging every 2000 tokens
SSoT: This module is the only implementation of adaptive prefill
"""
from __future__ import annotations

import logging
import subprocess
import threading
import time
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import mlx.core as mx
except ImportError:
    mx = None


def _metal_mem_str() -> str:
    """Return a compact Metal memory status string. Best-effort, never raises."""
    try:
        active_gb = mx.get_active_memory() / 1e9
        peak_gb = mx.get_peak_memory() / 1e9
        cache_gb = mx.get_cache_memory() / 1e9
        return f"metal={active_gb:.2f}GB peak={peak_gb:.2f}GB cache={cache_gb:.2f}GB"
    except Exception:
        return ""

# ── Module-level state (set via init()) ──────────────────────────────────────
_settings: Any = None
_terminal_status: Optional[Callable] = None
_pipeline_log: Optional[Callable] = None
_guard: Any = None
_cache_diag: Any = None
_model: Any = None
_prefill_step_size: int = 256
_feature_cache_diag: bool = True
_feature_full_logging: bool = True
_get_metal_budget_gb: Optional[Callable] = None
_ADAPTIVE_PREFILL_SAFETY_MARGIN: float = 0.85
_GPU_YIELD_SECONDS: float = 0.0
_SCRATCH_COEFFICIENT: float = 0.065
_N_LAYERS_SDPA: int = 10
_N_HEADS: int = 16
_SCRATCH_BYTES_PER_ELEMENT: int = 4
_ADAPTIVE_PREFILL_MIN_CHUNK: int = 32


def init(
    *,
    settings: Any,
    terminal_status: Callable,
    pipeline_log: Callable,
    guard: Any,
    cache_diag: Any,
    model: Any,
    prefill_step_size: int = 256,
    feature_cache_diag: bool = True,
    feature_full_logging: bool = True,
    get_metal_budget_gb: Optional[Callable] = None,
    adaptive_prefill_safety_margin: float = 0.85,
    gpu_yield_seconds: float = 0.0,
    scratch_coefficient: float = 0.065,
    n_layers_sdpa: int = 10,
    n_heads: int = 16,
    scratch_bytes_per_element: int = 4,
    adaptive_prefill_min_chunk: int = 32,
) -> None:
    """Initialize with shared state from server2."""
    global _settings, _terminal_status, _pipeline_log, _guard
    global _cache_diag, _model, _prefill_step_size
    global _feature_cache_diag, _feature_full_logging
    global _get_metal_budget_gb, _ADAPTIVE_PREFILL_SAFETY_MARGIN, _GPU_YIELD_SECONDS
    global _SCRATCH_COEFFICIENT, _N_LAYERS_SDPA, _N_HEADS
    global _SCRATCH_BYTES_PER_ELEMENT, _ADAPTIVE_PREFILL_MIN_CHUNK

    _settings = settings
    _terminal_status = terminal_status
    _pipeline_log = pipeline_log
    _guard = guard
    _cache_diag = cache_diag
    _model = model
    _prefill_step_size = prefill_step_size
    _feature_cache_diag = feature_cache_diag
    _feature_full_logging = feature_full_logging
    _get_metal_budget_gb = get_metal_budget_gb
    _ADAPTIVE_PREFILL_SAFETY_MARGIN = adaptive_prefill_safety_margin
    _GPU_YIELD_SECONDS = gpu_yield_seconds
    _SCRATCH_COEFFICIENT = scratch_coefficient
    _N_LAYERS_SDPA = n_layers_sdpa
    _N_HEADS = n_heads
    _SCRATCH_BYTES_PER_ELEMENT = scratch_bytes_per_element
    _ADAPTIVE_PREFILL_MIN_CHUNK = adaptive_prefill_min_chunk


def _start_prefill_progress(
    request_id: str, rest_count: int, log_fn=None, rate: int = 300
) -> Tuple[threading.Event, Optional[threading.Thread]]:
    """Launch a background prefill progress logger. Returns (done_event, thread|None)."""
    done = threading.Event()
    if rest_count <= 1000:
        return done, None
    _log = log_fn or _terminal_status

    def _progress():
        start = time.time()
        est_total = rest_count / rate if rate > 0 else 60
        _log(
            "⏳",
            f"Request {request_id} PREFILL starting | 0% | ETA ~{est_total:.0f}s | "
            f"tokens={rest_count} | {_metal_mem_str()}",
            indent=1,
        )
        
        try:
            last_cleared_mem = mx.get_active_memory()
        except Exception:
            last_cleared_mem = 0
            
        last_log_time = start
        
        while not done.is_set():
            done.wait(0.5)
            if done.is_set():
                break
                
            # --- MEMORY-DRIVEN INTRA-PREFILL RELIEF ---
            try:
                current_mem = mx.get_active_memory()
                if last_cleared_mem > 0 and current_mem - last_cleared_mem >= 600 * 1024 * 1024:  # 600 MB threshold
                    _guard.force_clear_cache("intra_prefill")
                    _log("🧹", f"Memory Guard: Purged intra-prefill transient memory. Was {current_mem/1e9:.2f}GB", indent=2)
                    last_cleared_mem = mx.get_active_memory()
            except Exception:
                pass  # Avoid silent thread death
            # ------------------------------------------

            elapsed = time.time() - start
            if time.time() - last_log_time >= 5.0:
                pct = min(99, (elapsed / est_total) * 100) if est_total > 0 else 0
                eta = max(0, est_total - elapsed)
                _est_tps = rest_count / elapsed if elapsed > 0 else 0
                _log(
                    "🔄",
                    f"Request {request_id} PREFILL | {pct:.0f}% | "
                    f"{elapsed:.0f}s/{est_total:.0f}s | ~{eta:.0f}s remaining | "
                    f"{_est_tps:.0f} tok/s | tokens={rest_count} | {_metal_mem_str()}",
                    indent=1,
                )
                last_log_time = time.time()

    t = threading.Thread(target=_progress, daemon=True)
    t.start()
    return done, t




def _adaptive_prefill_chunk(base_chunk: int, kv_length: int, available_bytes: float) -> int:
    """Compute the largest safe chunk size given available Metal memory.

    Uses the empirically calibrated scratch formula:
        scratch = COEFF × N_LAYERS_SDPA × chunk × kv_length × N_HEADS × 4

    Solves for max chunk:
        chunk_max = available_bytes / (COEFF × N_LAYERS_SDPA × kv_length × N_HEADS × 4)

    Returns min(base_chunk, chunk_max), clamped to [_ADAPTIVE_PREFILL_MIN_CHUNK, base_chunk].
    """
    if kv_length <= 0:
        return base_chunk

    denominator = (
        _SCRATCH_COEFFICIENT
        * _N_LAYERS_SDPA
        * kv_length
        * _N_HEADS
        * _SCRATCH_BYTES_PER_ELEMENT
    )
    if denominator <= 0:
        return base_chunk

    chunk_max = int(available_bytes / denominator)
    chunk = max(_ADAPTIVE_PREFILL_MIN_CHUNK, min(base_chunk, chunk_max))
    return chunk


# ── VM telemetry for adaptive prefill ─────────────────────────────────────



def _get_vm_counters() -> Dict[str, int]:
    """Capture pageouts, pageins, swapins, swapouts from vm_stat (stable BSD API).
    Returns counters dictionary for prefill telemetry. Fail-open returns zeros.
    """
    counters = {"pageouts": 0, "pageins": 0, "swapins": 0, "swapouts": 0}
    try:
        import subprocess
        out = subprocess.check_output(["vm_stat"], text=True, timeout=2)
        for line in out.splitlines():
            line_str = line.strip()
            if line_str.startswith("Pageouts:"):
                counters["pageouts"] = int(line_str.split(":")[1].strip().rstrip("."))
            elif line_str.startswith("Pageins:"):
                counters["pageins"] = int(line_str.split(":")[1].strip().rstrip("."))
            elif line_str.startswith("Swapouts:"):
                counters["swapouts"] = int(line_str.split(":")[1].strip().rstrip("."))
            elif line_str.startswith("Swapins:"):
                counters["swapins"] = int(line_str.split(":")[1].strip().rstrip("."))
    except Exception as _e:
        logger.debug("[VM_STAT] Failed to read vm_stat: %s", _e)
    return counters




def _adaptive_prefill(
    rest_tokens: list,
    prompt_cache: list,
    request_id: str = "",
    leave_last_token: bool = True,
) -> list:
    """Pre-prefill rest_tokens with adaptive chunk sizing.

    If leave_last_token is True: prefills rest_tokens[:-1] and returns rest_tokens[-1:].
    If leave_last_token is False: prefills ALL rest_tokens and returns [].
    The cache is updated in-place.
    """
    if not rest_tokens:
        return []
    if prompt_cache is None:
        return rest_tokens
    if leave_last_token and len(rest_tokens) <= 1:
        return rest_tokens

    tokens_to_prefill = rest_tokens[:-1] if leave_last_token else rest_tokens
    last_token = rest_tokens[-1:] if leave_last_token else []

    total = len(tokens_to_prefill)
    if total == 0:
        return last_token
    processed = 0
    prompt_array = mx.array(tokens_to_prefill)
    base_chunk = _prefill_step_size
    chunk_reductions = 0
    _prefill_t0 = time.time()
    _last_progress_tokens = 0
    _PROGRESS_INTERVAL = 2000  # Log every N tokens
    _diag_pre_prefill = _cache_diag.snapshot(prompt_cache, rest_tokens, "before_adaptive_prefill") if _feature_cache_diag else None

    _vm_before = _get_vm_counters() if _feature_cache_diag else None
    _fwd_eval_total_s = 0.0

    # ANE telemetry snapshot
    _ane_start_cnt = 0
    _gpu_start_cnt = 0
    _wrapped_ane_layers = []
    if getattr(_model, "_ane_injected", False):
        try:
            from lab.ane_code.ane_shim import _get_inner_layers, _ANEWrappedSelfAttention
            for _lyr in _get_inner_layers(_model):
                _at = getattr(_lyr, "self_attn", None)
                if isinstance(_at, _ANEWrappedSelfAttention):
                    _wrapped_ane_layers.append(_at)
            _ane_start_cnt = sum(_w.ane_dispatches for _w in _wrapped_ane_layers)
            _gpu_start_cnt = sum(_w.gpu_fallbacks for _w in _wrapped_ane_layers)
        except Exception:
            pass

    while processed < total:
        # Compute available memory
        budget_bytes = _get_metal_budget_gb() * 1e9
        active_bytes = mx.get_active_memory()
        available = (budget_bytes - active_bytes) * _ADAPTIVE_PREFILL_SAFETY_MARGIN

        # Compute adaptive chunk size based on current kv_length
        # kv_length = tokens already in cache + tokens we've processed
        # Use the first KVCache's offset if available, else track manually
        kv_length = processed  # conservative: our contribution so far
        for c in prompt_cache:
            if hasattr(c, "offset"):
                kv_length = c.offset
                break

        chunk_size = _adaptive_prefill_chunk(base_chunk, kv_length, available)
        n = min(chunk_size, total - processed)

        if chunk_size < base_chunk and chunk_reductions == 0:
            _terminal_status(
                "📐",
                f"Adaptive prefill: chunk reduced {base_chunk}→{chunk_size} "
                f"at kv={kv_length} | avail={available / 1e9:.2f}GB | "
                f"req={request_id[:12] if request_id else '?'}",
            )
        if chunk_size < base_chunk:
            chunk_reductions += 1

        # Forward pass & cache materialization (timed with high-precision perf_counter)
        _fwd_t0 = time.perf_counter()
        chunk = prompt_array[processed : processed + n][None]  # (1, n)
        _model(chunk, cache=prompt_cache)

        # Materialize cache + free scratch memory
        for c in prompt_cache:
            if hasattr(c, "state"):
                mx.eval(c.state)
            else:
                mx.eval(c)
        _fwd_eval_total_s += (time.perf_counter() - _fwd_t0)
        if chunk_reductions > 0:
            _guard.force_clear_cache("chunk_pressure")

        # GPU yield: let other Metal clients (Chrome VideoToolbox) process between chunks
        if _GPU_YIELD_SECONDS > 0:
            time.sleep(_GPU_YIELD_SECONDS)

        processed += n

        # Dark-zone illumination: log progress during large prefills
        if _feature_full_logging and request_id and processed - _last_progress_tokens >= _PROGRESS_INTERVAL:
            _elapsed = time.time() - _prefill_t0
            _tps = processed / _elapsed if _elapsed > 0 else 0
            _pipeline_log("PREFILL", request_id,
                f"progress | {processed}/{total} tok ({processed*100//total}%) | "
                f"{_tps:.0f} tok/s | chunk={n} | {_metal_mem_str()}")
            _last_progress_tokens = processed

    if chunk_reductions > 0:
        _terminal_status(
            "📐",
            f"Adaptive prefill complete: {total} tokens | "
            f"chunk reductions={chunk_reductions} | req={request_id[:12] if request_id else '?'}",
        )

    if _feature_cache_diag and _vm_before is not None:
        _vm_after = _get_vm_counters()
        _d_pageouts = _vm_after["pageouts"] - _vm_before["pageouts"]
        _d_swapouts = _vm_after["swapouts"] - _vm_before["swapouts"]
        _d_swapins = _vm_after["swapins"] - _vm_before["swapins"]
        _fwd_eval_ms = _fwd_eval_total_s * 1000.0
        _total_prefill_ms = (time.time() - _prefill_t0) * 1000.0
        _eval_tps = total / (_fwd_eval_total_s if _fwd_eval_total_s > 0 else 0.001)

        # ANE telemetry delta
        _ane_delta = 0
        _gpu_delta = 0
        if _wrapped_ane_layers:
            _ane_delta = sum(_w.ane_dispatches for _w in _wrapped_ane_layers) - _ane_start_cnt
            _gpu_delta = sum(_w.gpu_fallbacks for _w in _wrapped_ane_layers) - _gpu_start_cnt
            _terminal_status("🍏", f"ANE dispatch: {_ane_delta} ANE | {_gpu_delta} GPU fallback | req={request_id[:12] if request_id else '?'}")

        _pipeline_log(
            "PREFILL_TELEMETRY",
            request_id,
            f"tokens={total} | wall={_total_prefill_ms:.1f}ms | fwd_eval={_fwd_eval_ms:.1f}ms | "
            f"eval_speed={_eval_tps:.1f} tok/s | ane={_ane_delta} | gpu_fallback={_gpu_delta} | "
            f"vm_delta: pageouts={_d_pageouts}, swapouts={_d_swapouts}, swapins={_d_swapins}",
        )

        _cache_diag.compare(_diag_pre_prefill, prompt_cache, rest_tokens, "adaptive_prefill", request_id,
            extra={"tokens": total, "chunks": chunk_reductions})
    return last_token



