"""
[AI_DIRECTIVE]
ROL: Diagnosticar por qué el warmup de Metal shader compilation varía 4x (8s-33s)
OBJETIVO: Capturar estado del sistema antes de cada warmup para correlacionar
ENTRADAS: Modelo Qwen3 MoE
SALIDAS: Tiempos + diagnóstico del sistema a stdout
"""

import time
import sys
import os
import subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
from mlx_lm import load

MODEL_PATH = os.environ.get("MODEL_PATH", "unsloth/Qwen3.6-35B-A3B-UD-MLX-3bit")
CHUNK_SIZE = 256
WARMUP_ATTEMPTS = 5  # run 5 cold-ish warmups


def system_diag():
    """Capture system state before warmup."""
    diag = {}

    # Metal memory
    try:
        diag["metal_active_gb"] = f"{mx.get_active_memory() / 1e9:.2f}"
        diag["metal_peak_gb"] = f"{mx.get_peak_memory() / 1e9:.2f}"
        diag["metal_cache_gb"] = f"{mx.get_cache_memory() / 1e9:.2f}"
    except Exception:
        diag["metal"] = "unavailable"

    # Thermal (macOS)
    try:
        result = subprocess.run(
            ["pmset", "-g", "therm"], capture_output=True, text=True, timeout=2
        )
        for line in result.stdout.splitlines():
            if "CPU_Speed_Limit" in line:
                diag["cpu_speed_limit"] = line.strip().split("=")[-1].strip()
            if "GPU_Speed_Limit" in line:
                diag["gpu_speed_limit"] = line.strip().split("=")[-1].strip()
    except Exception:
        diag["thermal"] = "unavailable"

    # System memory pressure
    try:
        result = subprocess.run(
            ["memory_pressure"], capture_output=True, text=True, timeout=2
        )
        for line in result.stdout.splitlines():
            if "System-wide memory free percentage" in line:
                diag["mem_free_pct"] = line.strip().split(":")[-1].strip()
    except Exception:
        diag["mem_pressure"] = "unavailable"

    return diag


def make_prompt_cache(model):
    from mlx_lm.models.cache import make_prompt_cache
    return make_prompt_cache(model)


def run_warmup(model, prompt_cache, tokens, label="individual"):
    """Run one forward pass + eval and return the time."""
    chunk = mx.array(tokens)[None]
    model(chunk, cache=prompt_cache)

    t0 = time.perf_counter()
    if label == "individual":
        for c in prompt_cache:
            if hasattr(c, "state"):
                mx.eval(c.state)
            else:
                mx.eval(c)
    else:
        to_eval = []
        for c in prompt_cache:
            if hasattr(c, "state"):
                if isinstance(c.state, (list, tuple)):
                    to_eval.extend(a for a in c.state if a is not None and isinstance(a, mx.array))
                elif isinstance(c.state, mx.array):
                    to_eval.append(c.state)
            elif hasattr(c, "keys") and hasattr(c, "values"):
                if c.keys is not None:
                    to_eval.append(c.keys)
                if c.values is not None:
                    to_eval.append(c.values)
        if to_eval:
            mx.eval(*to_eval)
    t1 = time.perf_counter()
    return t1 - t0


def main():
    print(f"Loading model: {MODEL_PATH}")
    model, tokenizer = load(MODEL_PATH)
    print(f"Model loaded. Layers: {len(model.layers)}")

    test_text = "This is a benchmark test. " * (CHUNK_SIZE // 6)
    tokens = tokenizer.encode(test_text)[:CHUNK_SIZE]
    print(f"Test tokens: {len(tokens)}")
    print()

    # ── Phase 1: COLD warmups with full cleanup between each ──
    print("=" * 70)
    print("WARMUP DIAGNOSTIC — Full cleanup between each attempt")
    print("=" * 70)

    for attempt in range(WARMUP_ATTEMPTS):
        # Force full cleanup
        mx.clear_cache()
        import gc
        gc.collect()
        mx.clear_cache()

        # System diagnostics BEFORE warmup
        diag = system_diag()
        diag_str = " | ".join(f"{k}={v}" for k, v in diag.items())

        # Fresh cache each time
        cache = make_prompt_cache(model)

        print(f"\n  attempt {attempt + 1}:")
        print(f"    pre-state: {diag_str}")

        t_total_start = time.perf_counter()

        # Full forward + eval (individual, like current code)
        chunk = mx.array(tokens)[None]

        t_fwd_start = time.perf_counter()
        model(chunk, cache=cache)
        t_fwd_end = time.perf_counter()

        t_eval_start = time.perf_counter()
        for c in cache:
            if hasattr(c, "state"):
                mx.eval(c.state)
            else:
                mx.eval(c)
        t_eval_end = time.perf_counter()

        t_total_end = time.perf_counter()

        fwd_ms = (t_fwd_end - t_fwd_start) * 1000
        eval_ms = (t_eval_end - t_eval_start) * 1000
        total_ms = (t_total_end - t_total_start) * 1000

        print(f"    model()  : {fwd_ms:.1f} ms  (graph build, lazy)")
        print(f"    mx.eval(): {eval_ms:.1f} ms  (GPU execution + shader compile)")
        print(f"    TOTAL    : {total_ms:.1f} ms")

        # Post-state
        diag_post = system_diag()
        print(f"    post-state: metal_active={diag_post.get('metal_active_gb', '?')}GB"
              f" peak={diag_post.get('metal_peak_gb', '?')}GB"
              f" cache={diag_post.get('metal_cache_gb', '?')}GB")

        del cache
        mx.clear_cache()

        # Wait 2s between attempts to let system settle
        if attempt < WARMUP_ATTEMPTS - 1:
            print("    (waiting 2s...)")
            time.sleep(2)

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
