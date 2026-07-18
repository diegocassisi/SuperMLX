#!/usr/bin/env python3
"""Stress test for prefill chunk sizes — measures peak Metal memory empirically.

Usage (each invocation is a separate process):
    python stress_prefill.py --tokens 25000 --chunk 512
    python stress_prefill.py --tokens 31000 --chunk 512
    python stress_prefill.py --tokens 31000 --chunk 256 --diverse

Output: JSON line to stdout + results.jsonl file for comparison.
Exit code 0 = success, non-zero = OOM/crash (process killed by Metal).
"""

import argparse
import json
import os
import random
import sys
import time

# Force offline mode — model should already be cached
os.environ["HF_HUB_OFFLINE"] = "1"

import mlx.core as mx
from mlx_lm import load


# Model params (verified from config.json text_config)
N_HEADS = 16
N_LAYERS_SDPA = 10  # full_attention_interval=4, 40 layers → 10 SDPA layers
BYTES_PER_ELEMENT = 4  # fp32 assumption for QK^T scratch


def predicted_scratch_gb(chunk_size: int, kv_length: int) -> float:
    """Theoretical upper bound: assumes full QK^T materialization across all SDPA layers."""
    return (N_LAYERS_SDPA * chunk_size * kv_length * N_HEADS * BYTES_PER_ELEMENT) / 1e9


def run_prefill(model, tokenizer, prompt_tokens: mx.array, chunk_size: int,
                post_load_active: float):
    """Run prefill only, return peak memory, timing, and per-chunk log."""

    # Create cache
    cache = model.make_cache() if hasattr(model, "make_cache") else None

    t0 = time.perf_counter()

    total = prompt_tokens.shape[0]
    processed = 0
    chunk_log = []
    global_peak_gb = mx.get_peak_memory() / 1e9  # track absolute peak manually

    # Prefill loop (same pattern as mlx_lm generate_step lines 430-451)
    while total - processed > 1:
        remaining = (total - processed) - 1
        n = min(chunk_size, remaining)
        chunk = prompt_tokens[processed : processed + n][None]  # (1, n)

        # Reset peak BEFORE this chunk so post-chunk peak measures THIS chunk only
        mx.reset_peak_memory()

        # Forward pass
        model(chunk, cache=cache)

        # Materialize cache state + free transient scratch
        if cache is not None:
            for c in cache:
                if hasattr(c, "state"):
                    mx.eval(c.state)
                else:
                    mx.eval(c)
        mx.clear_cache()

        processed += n
        kv_length = processed

        # Memory snapshot — peak is per-chunk (reset above), active is absolute
        chunk_peak_gb = mx.get_peak_memory() / 1e9
        active_gb = mx.get_active_memory() / 1e9
        chunk_scratch_gb = round(chunk_peak_gb - active_gb, 4)
        predicted_gb = predicted_scratch_gb(n, kv_length)

        # Track absolute peak
        if chunk_peak_gb > global_peak_gb:
            global_peak_gb = chunk_peak_gb

        entry = {
            "chunk_idx": len(chunk_log),
            "kv_length": kv_length,
            "chunk_tokens": n,
            "chunk_peak_gb": round(chunk_peak_gb, 3),
            "active_gb": round(active_gb, 3),
            "chunk_scratch_gb": round(chunk_scratch_gb, 4),
            "predicted_scratch_gb": round(predicted_gb, 4),
            "ratio_measured_vs_predicted": round(chunk_scratch_gb / predicted_gb, 3) if predicted_gb > 0.001 else None,
        }
        chunk_log.append(entry)

        # Print progress: every 10 chunks + last 5 before end
        pct = processed / total * 100
        remaining_chunks = max(0, (total - processed - 1)) // chunk_size + 1
        if len(chunk_log) % 10 == 0 or remaining_chunks <= 5 or processed + 1 >= total:
            ratio_str = f"ratio={entry['ratio_measured_vs_predicted']:.3f}" if entry['ratio_measured_vs_predicted'] is not None else "ratio=N/A"
            print(
                f"  [{pct:5.1f}%] kv={kv_length:>6d} chunk={n:>4d} "
                f"peak={chunk_peak_gb:.2f}GB active={active_gb:.2f}GB "
                f"scratch={chunk_scratch_gb:.4f}GB pred={predicted_gb:.4f}GB {ratio_str}",
                file=sys.stderr,
                flush=True,
            )

    # Process last token
    mx.reset_peak_memory()
    last_token = prompt_tokens[-1:][None]
    logits = model(last_token, cache=cache)
    mx.eval(logits)
    mx.clear_cache()

    last_peak = mx.get_peak_memory() / 1e9
    if last_peak > global_peak_gb:
        global_peak_gb = last_peak

    elapsed = time.perf_counter() - t0
    final_active_gb = mx.get_active_memory() / 1e9

    # Find the chunk with max measured scratch
    max_scratch_entry = max(chunk_log, key=lambda e: e["chunk_scratch_gb"])

    return {
        "elapsed_s": round(elapsed, 2),
        "global_peak_gb": round(global_peak_gb, 3),
        "final_active_gb": round(final_active_gb, 3),
        "kv_cache_growth_gb": round(final_active_gb - post_load_active, 3),
        "tokens_per_sec": round(total / elapsed, 1),
        "total_chunks": len(chunk_log),
        "max_chunk_scratch_gb": round(max_scratch_entry["chunk_scratch_gb"], 4),
        "max_scratch_at_kv": max_scratch_entry["kv_length"],
        "max_scratch_ratio": max_scratch_entry["ratio_measured_vs_predicted"],
        "chunk_log_last10": chunk_log[-10:],
    }


def main():
    parser = argparse.ArgumentParser(description="Stress test prefill chunk sizes")
    parser.add_argument("--tokens", type=int, required=True, help="Number of prompt tokens")
    parser.add_argument("--chunk", type=int, required=True, help="Prefill chunk size")
    parser.add_argument("--diverse", action="store_true",
                        help="Use random diverse tokens (not repeated) to stress MoE routing")
    parser.add_argument(
        "--model",
        type=str,
        default="unsloth/Qwen3.6-35B-A3B-UD-MLX-3bit",
        help="Model path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="stress_results.jsonl",
        help="Output JSONL file",
    )
    args = parser.parse_args()

    print(f"=== Stress Prefill Test ===", file=sys.stderr)
    print(f"  Model: {args.model}", file=sys.stderr)
    print(f"  Tokens: {args.tokens}", file=sys.stderr)
    print(f"  Chunk: {args.chunk}", file=sys.stderr)
    print(f"  Diverse tokens: {args.diverse}", file=sys.stderr)
    print(f"  MLX version: {mx.__version__}", file=sys.stderr)
    device_mem_gb = mx.device_info()["memory_size"] / 1e9
    print(f"  Device memory: {device_mem_gb:.1f} GB", file=sys.stderr)
    print(file=sys.stderr, flush=True)

    # Load model
    print("Loading model...", file=sys.stderr, flush=True)
    t0 = time.perf_counter()
    model, tokenizer = load(args.model, tokenizer_config={"trust_remote_code": True})
    load_time = time.perf_counter() - t0

    post_load_peak = mx.get_peak_memory() / 1e9
    post_load_active = mx.get_active_memory() / 1e9

    print(f"Model loaded in {load_time:.1f}s", file=sys.stderr)
    print(
        f"Post-load memory: peak={post_load_peak:.2f}GB "
        f"active={post_load_active:.2f}GB",
        file=sys.stderr,
        flush=True,
    )

    # Create prompt tokens
    if args.diverse:
        # Random tokens from vocab (excluding special tokens at the extremes)
        vocab_size = getattr(tokenizer, "vocab_size", 248320)
        safe_range = (100, vocab_size - 100)
        token_ids = [random.randint(*safe_range) for _ in range(args.tokens)]
        prompt_tokens = mx.array(token_ids)
        print(f"Prompt: {args.tokens} tokens (diverse random, range {safe_range})",
              file=sys.stderr, flush=True)
    else:
        prompt_tokens = mx.array([198] * args.tokens)
        print(f"Prompt: {args.tokens} tokens (synthetic, token=198 repeated)",
              file=sys.stderr, flush=True)

    # Run prefill
    print(f"\nRunning prefill with chunk_size={args.chunk}...", file=sys.stderr, flush=True)
    result = run_prefill(model, tokenizer, prompt_tokens, args.chunk, post_load_active)

    # Build output record
    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": args.model,
        "tokens": args.tokens,
        "chunk_size": args.chunk,
        "diverse_tokens": args.diverse,
        "mlx_version": mx.__version__,
        "device_memory_gb": round(device_mem_gb, 1),
        "post_load_peak_gb": round(post_load_peak, 3),
        "post_load_active_gb": round(post_load_active, 3),
        **result,
        "status": "OK",
    }

    # Output
    json_line = json.dumps(record)
    print(json_line)  # stdout — machine-readable

    print(f"\n=== RESULT ===", file=sys.stderr)
    print(f"  Status: OK", file=sys.stderr)
    print(f"  Global peak: {result['global_peak_gb']:.3f} GB", file=sys.stderr)
    print(f"  Final active: {result['final_active_gb']:.3f} GB", file=sys.stderr)
    print(f"  KV cache growth: {result['kv_cache_growth_gb']:.3f} GB", file=sys.stderr)
    print(f"  Max chunk scratch: {result['max_chunk_scratch_gb']:.4f} GB (at kv={result['max_scratch_at_kv']})", file=sys.stderr)
    print(f"  Scratch ratio (measured/predicted): {result['max_scratch_ratio']}", file=sys.stderr)
    print(f"  Time: {result['elapsed_s']:.1f}s", file=sys.stderr)
    print(f"  Speed: {result['tokens_per_sec']:.0f} tok/s", file=sys.stderr)
    print(f"  Headroom to device: {device_mem_gb - result['global_peak_gb']:.1f} GB", file=sys.stderr, flush=True)

    # Append to JSONL file
    output_path = os.path.join(os.path.dirname(__file__), args.output)
    with open(output_path, "a") as f:
        f.write(json_line + "\n")
    print(f"  Results appended to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
