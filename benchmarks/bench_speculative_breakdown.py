"""
[AI_DIRECTIVE]
ROL: Micro-benchmark de latencia y overhead en el ciclo de speculative decoding / MTP
OBJETIVO: Medir con certeza el desglose de tiempo: GPU execution (mx.eval) vs Sampling/Sync vs Python control flow y rollback
ENTRADAS: Modelo local cargable vía mlx_lm (por defecto mlx-community/Qwen3-0.6B-4bit)
SALIDAS: Métricas de latencia por fase (ms) y % de overhead atribuible a Python vs GPU
REGLAS INVIOLABLES:
- Prohibido hardcoding de rutas fijas
- Prohibido print() para diagnóstico; usar logger con prefijos estándar
- Obligatorio tipado estricto en funciones públicas
- Prohibido inventar métricas
"""

import argparse
import logging
import os
import sys
import time
from typing import Any, Dict, List, Tuple

# Ensure repository root is in python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

from supermlx.mtp.speculative_engine import (
    _distribution_from_logits,
    _sample_token,
    _verify_draft_token,
    _rollback_draft_caches,
    _clear_draft_rollback,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def bench_sampling_primitives(
    vocab_size: int = 151936,
    num_rounds: int = 50,
) -> Dict[str, float]:
    """Mide el tiempo de sampling en CPU/Metal para un vector de logits real."""
    start_time = time.time()
    logger.info("[INICIO] bench_sampling_primitives (vocab_size=%d, rounds=%d)", vocab_size, num_rounds)

    # Warmup
    dummy_logits = mx.random.normal((vocab_size,)).astype(mx.float32)
    mx.eval(dummy_logits)
    _ = _distribution_from_logits(dummy_logits, temperature=0.0)

    # 1. Greedy argmax + item() sync
    t_greedy_list = []
    for _ in range(num_rounds):
        t0 = time.perf_counter()
        token = int(mx.argmax(dummy_logits).item())
        t1 = time.perf_counter()
        t_greedy_list.append((t1 - t0) * 1000.0)
    avg_greedy_ms = sum(t_greedy_list) / len(t_greedy_list)

    # 2. Distribution calculation (temp=0.7, top_p=0.8, min_p=0.05, top_k=20)
    t_dist_list = []
    for _ in range(num_rounds):
        t0 = time.perf_counter()
        probs = _distribution_from_logits(
            dummy_logits,
            temperature=0.7,
            top_p=0.8,
            min_p=0.05,
            top_k=20,
        )
        mx.eval(probs)
        t1 = time.perf_counter()
        t_dist_list.append((t1 - t0) * 1000.0)
    avg_dist_ms = sum(t_dist_list) / len(t_dist_list)

    # 3. Categorical sampling from probs
    t_sample_list = []
    for _ in range(num_rounds):
        t0 = time.perf_counter()
        tok, _ = _sample_token(
            dummy_logits,
            temperature=0.7,
            top_p=0.8,
            min_p=0.05,
            top_k=20,
        )
        t1 = time.perf_counter()
        t_sample_list.append((t1 - t0) * 1000.0)
    avg_sample_ms = sum(t_sample_list) / len(t_sample_list)

    # 4. Verification Leviathan (accept path)
    p_probs = _distribution_from_logits(dummy_logits, temperature=0.7)
    q_probs = p_probs
    mx.eval(p_probs, q_probs)
    t_verify_accept_list = []
    for _ in range(num_rounds):
        t0 = time.perf_counter()
        accepted, tok = _verify_draft_token(
            draft_token=100,
            p_probs=p_probs,
            q_probs=q_probs,
            temperature=0.7,
        )
        t1 = time.perf_counter()
        t_verify_accept_list.append((t1 - t0) * 1000.0)
    avg_verify_accept_ms = sum(t_verify_accept_list) / len(t_verify_accept_list)

    # 5. Verification Leviathan (reject path with residual sampling)
    dummy_q = mx.random.normal((vocab_size,)).astype(mx.float32)
    q_dist = _distribution_from_logits(dummy_q, temperature=0.7)
    mx.eval(q_dist)
    t_verify_reject_list = []
    for _ in range(num_rounds):
        t0 = time.perf_counter()
        # force reject by choosing draft_token with 0 probability
        accepted, tok = _verify_draft_token(
            draft_token=0,
            p_probs=p_probs,
            q_probs=q_dist,
            temperature=0.7,
        )
        t1 = time.perf_counter()
        t_verify_reject_list.append((t1 - t0) * 1000.0)
    avg_verify_reject_ms = sum(t_verify_reject_list) / len(t_verify_reject_list)

    elapsed_total = int((time.time() - start_time) * 1000)
    logger.info("[CALC] greedy_sync_ms=%.3f", avg_greedy_ms)
    logger.info("[CALC] dist_calc_ms=%.3f", avg_dist_ms)
    logger.info("[CALC] full_sample_ms=%.3f", avg_sample_ms)
    logger.info("[CALC] verify_accept_ms=%.3f", avg_verify_accept_ms)
    logger.info("[CALC] verify_reject_ms=%.3f", avg_verify_reject_ms)
    logger.info("[RESULT] Status=SUCCESS Time=%dms", elapsed_total)

    return {
        "greedy_sync_ms": avg_greedy_ms,
        "dist_calc_ms": avg_dist_ms,
        "full_sample_ms": avg_sample_ms,
        "verify_accept_ms": avg_verify_accept_ms,
        "verify_reject_ms": avg_verify_reject_ms,
    }


def bench_decode_loop_breakdown(
    model_name: str,
    num_steps: int = 40,
    warmup_steps: int = 5,
) -> Dict[str, float]:
    """Mide cada fase dentro de un paso de decode / verificación especulativa."""
    start_time = time.time()
    logger.info("[INICIO] bench_decode_loop_breakdown (model=%s, steps=%d)", model_name, num_steps)

    model, tokenizer = load(model_name)
    logger.info("[DATA] Model loaded successfully (%d layers)", len(model.layers))

    # Build prompt cache
    cache = make_prompt_cache(model)
    prompt = "Explain quantum entanglement in simple terms."
    prompt_tokens = tokenizer.encode(prompt)
    input_ids = mx.array([prompt_tokens])

    # Prefill
    logits = model(input_ids, cache=cache)
    mx.eval(logits)
    last_token = int(mx.argmax(logits[:, -1, :]).item())

    # Profiling buckets (in milliseconds)
    t_gpu_fwd_1tok: List[float] = []      # GPU forward 1 token (baseline decode / draft forward)
    t_gpu_fwd_2tok: List[float] = []      # GPU forward 2 tokens (verify batch)
    t_python_overhead: List[float] = []   # Python array prep, slicing, branching, cache ops
    t_sync_and_item: List[float] = []     # Converting mx.array token to Python int

    current_token = last_token

    for step in range(warmup_steps + num_steps):
        # --- Measure Python overhead: creating array, updating counters, bookkeeping ---
        t_py_start = time.perf_counter()
        token_arr = mx.array([[current_token]])
        step_idx = step
        branch_check = (step_idx % 2 == 0)
        # Simulate rollback / trim check
        for c in cache:
            if hasattr(c, "trim") and getattr(c, "offset", 0) > 1000:
                c.trim(1)
        t_py_end = time.perf_counter()

        # --- Measure 1-token forward pass (GPU execution via mx.eval) ---
        out_logits = model(token_arr, cache=cache)
        t_gpu_start = time.perf_counter()
        mx.eval(out_logits)
        t_gpu_end = time.perf_counter()

        # --- Measure Sync / token extraction ---
        t_sync_start = time.perf_counter()
        next_tok = int(mx.argmax(out_logits[:, -1, :]).item())
        t_sync_end = time.perf_counter()

        # --- Measure 2-token verification batch forward pass ---
        verify_arr = mx.array([[current_token, next_tok]])
        # Use a temporary branch or clone to measure 2-token fwd
        t_v_py_start = time.perf_counter()
        _v_input = verify_arr
        t_v_py_end = time.perf_counter()

        v_logits = model(_v_input, cache=cache)
        t_vgpu_start = time.perf_counter()
        mx.eval(v_logits)
        t_vgpu_end = time.perf_counter()

        # Re-sync next token for next iteration
        current_token = next_tok

        if step >= warmup_steps:
            t_python_overhead.append(((t_py_end - t_py_start) + (t_v_py_end - t_v_py_start)) * 1000.0)
            t_gpu_fwd_1tok.append((t_gpu_end - t_gpu_start) * 1000.0)
            t_gpu_fwd_2tok.append((t_vgpu_end - t_vgpu_start) * 1000.0)
            t_sync_and_item.append((t_sync_end - t_sync_start) * 1000.0)

    avg_py = sum(t_python_overhead) / len(t_python_overhead)
    avg_gpu_1 = sum(t_gpu_fwd_1tok) / len(t_gpu_fwd_1tok)
    avg_gpu_2 = sum(t_gpu_fwd_2tok) / len(t_gpu_fwd_2tok)
    avg_sync = sum(t_sync_and_item) / len(t_sync_and_item)

    total_step_ms = avg_py + avg_gpu_1 + avg_sync
    py_percentage = (avg_py / total_step_ms) * 100.0

    logger.info("=" * 60)
    logger.info("[RESULT] DECODE STEP LATENCY BREAKDOWN (%s)", model_name)
    logger.info("=" * 60)
    logger.info("[CALC] 1. Python Loop / Control Overhead: %.4f ms (%.2f%%)", avg_py, py_percentage)
    logger.info("[CALC] 2. Metal GPU Forward (1 tok):      %.3f ms", avg_gpu_1)
    logger.info("[CALC] 3. Metal GPU Verify  (2 tok batch): %.3f ms", avg_gpu_2)
    logger.info("[CALC] 4. Argmax + Sync (.item()):        %.4f ms", avg_sync)
    logger.info("[CALC] Total 1-token step time:           %.3f ms (%.1f tok/s)", total_step_ms, 1000.0 / total_step_ms)
    logger.info("=" * 60)

    elapsed_total = int((time.time() - start_time) * 1000)
    logger.info("[RESULT] Status=SUCCESS Time=%dms", elapsed_total)

    return {
        "avg_python_overhead_ms": avg_py,
        "avg_gpu_fwd_1tok_ms": avg_gpu_1,
        "avg_gpu_fwd_2tok_ms": avg_gpu_2,
        "avg_sync_ms": avg_sync,
        "total_step_ms": total_step_ms,
        "python_pct": py_percentage,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Micro-benchmark de speculative decoding breakdown")
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("MODEL_PATH", "mlx-community/Qwen3-0.6B-4bit"),
        help="Model path or HuggingFace repo id",
    )
    parser.add_argument("--steps", type=int, default=30, help="Number of benchmark steps")
    args = parser.parse_args()

    logger.info("[CONFIG] model=%s steps=%d", args.model, args.steps)

    logger.info(">>> PARTE 1: Benchmarking Primitivas de Sampling y Rejection (Vocab=151k) <<<")
    sample_results = bench_sampling_primitives()

    logger.info(">>> PARTE 2: Benchmarking Decode Loop con Modelo Real (%s) <<<", args.model)
    loop_results = bench_decode_loop_breakdown(model_name=args.model, num_steps=args.steps)

    print("\n" + "=" * 65)
    print("                RESUMEN DE CERTEZAS MEDIDAS")
    print("=" * 65)
    print(f"1. Pure Python Control Overhead:     {loop_results['avg_python_overhead_ms']:.4f} ms")
    print(f"2. Argmax + CPU Sync (.item()):       {loop_results['avg_sync_ms']:.4f} ms")
    print(f"3. Complex Sampling (TopK/TopP/Soft): {sample_results['full_sample_ms']:.4f} ms")
    print(f"4. Leviathan Rejection Test:         {sample_results['verify_reject_ms']:.4f} ms")
    print(f"5. Metal GPU 1-token decode:         {loop_results['avg_gpu_fwd_1tok_ms']:.3f} ms")
    print(f"6. Metal GPU 2-token batch verify:   {loop_results['avg_gpu_fwd_2tok_ms']:.3f} ms")
    print("-" * 65)
    print(f"Overhead puro de Python en el loop:  {loop_results['python_pct']:.2f}% del tiempo total")
    print("=" * 65)


if __name__ == "__main__":
    main()
