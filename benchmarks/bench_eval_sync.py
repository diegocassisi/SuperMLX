"""
[AI_DIRECTIVE]
ROL: Benchmark de overhead mx.eval individual vs batched en prefill
OBJETIVO: Medir si 40 mx.eval individuales (1 por capa) son más lentos que 1 mx.eval batched
ENTRADAS: Modelo Qwen3 MoE cargado desde HuggingFace cache
SALIDAS: Tiempos comparativos a stdout
REGLAS INVIOLABLES:
- No modificar el modelo ni el cache
- Medir solo el paso de mx.eval, no el forward pass
"""

import time
import sys
import os

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
from mlx_lm import load

MODEL_PATH = os.environ.get("MODEL_PATH", "unsloth/Qwen3.6-35B-A3B-UD-MLX-3bit")
WARMUP_ROUNDS = 3
MEASURE_ROUNDS = 10
CHUNK_SIZE = 256  # tokens per prefill chunk (realistic)


def make_prompt_cache(model):
    """Create a fresh prompt cache matching the model architecture."""
    from mlx_lm.models.cache import make_prompt_cache
    return make_prompt_cache(model)


def run_forward_and_eval_individual(model, prompt_cache, tokens):
    """Run forward pass then eval each cache layer individually (current code)."""
    chunk = mx.array(tokens)[None]  # (1, N)
    model(chunk, cache=prompt_cache)

    t0 = time.perf_counter()
    for c in prompt_cache:
        if hasattr(c, "state"):
            mx.eval(c.state)
        else:
            mx.eval(c)
    t1 = time.perf_counter()
    return t1 - t0


def run_forward_and_eval_batched(model, prompt_cache, tokens):
    """Run forward pass then eval all cache layers in one call (proposed)."""
    chunk = mx.array(tokens)[None]  # (1, N)
    model(chunk, cache=prompt_cache)

    t0 = time.perf_counter()
    to_eval = []
    for c in prompt_cache:
        if hasattr(c, "state"):
            if isinstance(c.state, (list, tuple)):
                to_eval.extend(a for a in c.state if a is not None and isinstance(a, mx.array))
            elif isinstance(c.state, mx.array):
                to_eval.append(c.state)
        elif hasattr(c, "keys") and hasattr(c, "values"):
            # KVCache — eval keys and values
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

    # Generate a realistic token sequence
    test_text = "This is a benchmark test. " * (CHUNK_SIZE // 6)
    tokens = tokenizer.encode(test_text)[:CHUNK_SIZE]
    print(f"Test tokens: {len(tokens)}")
    print(f"Warmup rounds: {WARMUP_ROUNDS}, Measure rounds: {MEASURE_ROUNDS}")
    print()

    # ── BENCHMARK: Batched eval (FIRST) ──
    print("=" * 60)
    print("METHOD B: 1 batched mx.eval call (proposed) — RUNS FIRST")
    print("=" * 60)

    times_batched = []
    for i in range(WARMUP_ROUNDS + MEASURE_ROUNDS):
        cache = make_prompt_cache(model)
        mx.clear_cache()
        dt = run_forward_and_eval_batched(model, cache, tokens)
        if i >= WARMUP_ROUNDS:
            times_batched.append(dt)
            print(f"  round {i - WARMUP_ROUNDS + 1:2d}: {dt * 1000:.3f} ms")
        else:
            print(f"  warmup {i + 1}: {dt * 1000:.3f} ms")
        del cache
        mx.clear_cache()

    # ── BENCHMARK: Individual eval (SECOND) ──
    print()
    print("=" * 60)
    print("METHOD A: 40 individual mx.eval calls (current code) — RUNS SECOND")
    print("=" * 60)

    times_individual = []
    for i in range(WARMUP_ROUNDS + MEASURE_ROUNDS):
        cache = make_prompt_cache(model)
        mx.clear_cache()
        dt = run_forward_and_eval_individual(model, cache, tokens)
        if i >= WARMUP_ROUNDS:
            times_individual.append(dt)
            print(f"  round {i - WARMUP_ROUNDS + 1:2d}: {dt * 1000:.3f} ms")
        else:
            print(f"  warmup {i + 1}: {dt * 1000:.3f} ms")
        del cache
        mx.clear_cache()

    # ── RESULTS ──
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    avg_ind = sum(times_individual) / len(times_individual) * 1000
    avg_bat = sum(times_batched) / len(times_batched) * 1000
    min_ind = min(times_individual) * 1000
    min_bat = min(times_batched) * 1000

    print(f"Individual (40 evals):  avg={avg_ind:.3f} ms  min={min_ind:.3f} ms")
    print(f"Batched    (1 eval):    avg={avg_bat:.3f} ms  min={min_bat:.3f} ms")
    print(f"Difference:             avg={avg_ind - avg_bat:+.3f} ms  min={min_ind - min_bat:+.3f} ms")
    if avg_bat > 0:
        print(f"Speedup:                {avg_ind / avg_bat:.2f}x")
    print()


if __name__ == "__main__":
    main()
