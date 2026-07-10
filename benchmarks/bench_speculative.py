"""Benchmark: baseline vs speculative decoding with draft model.

Usage (server must be stopped first):
  venv/bin/python benchmarks/bench_speculative.py
"""

import time
import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

TARGET_MODEL = "mlx-community/Agents-A1-3bit"
DRAFT_MODEL = "mlx-community/Qwen3.5-0.8B-MLX-4bit"
PROMPT = "Write a Python function that implements binary search on a sorted list. Include docstring and edge case handling."
MAX_TOKENS = 256
TEMPERATURE = 0.2
REPEATS = 3


def bench_generate(model, tokenizer, prompt_tokens, draft_model=None, label=""):
    """Run generation and return tok/s."""
    times = []
    sampler = make_sampler(temp=TEMPERATURE, top_p=0.95)
    for r in range(REPEATS):
        tokens = []
        t0 = time.perf_counter()
        for resp in stream_generate(
            model,
            tokenizer,
            prompt_tokens,
            max_tokens=MAX_TOKENS,
            draft_model=draft_model,
            sampler=sampler,
        ):
            tokens.append(resp.token)
        elapsed = time.perf_counter() - t0
        tok_s = len(tokens) / elapsed
        times.append(tok_s)
        print(f"  [{label}] run {r+1}/{REPEATS}: {len(tokens)} tokens in {elapsed:.2f}s = {tok_s:.1f} tok/s")
    avg = sum(times) / len(times)
    print(f"  [{label}] average: {avg:.1f} tok/s\n")
    return avg


def main():
    print(f"Loading target model: {TARGET_MODEL}")
    model, tokenizer = load(TARGET_MODEL)
    mx.eval(model.parameters())
    print("Target loaded.\n")

    # Build prompt
    messages = [{"role": "user", "content": PROMPT}]
    prompt_tokens = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, enable_thinking=True
    )

    # Baseline
    print("=" * 60)
    print("BASELINE (no draft)")
    print("=" * 60)
    baseline = bench_generate(model, tokenizer, prompt_tokens, label="baseline")

    # Load draft
    print(f"Loading draft model: {DRAFT_MODEL}")
    draft_model, _ = load(DRAFT_MODEL)
    mx.eval(draft_model.parameters())
    print("Draft loaded.\n")

    # Speculative
    print("=" * 60)
    print("SPECULATIVE (with draft)")
    print("=" * 60)
    speculative = None
    try:
        speculative = bench_generate(model, tokenizer, prompt_tokens, draft_model=draft_model, label="speculative")
    except ValueError as e:
        print(f"  ❌ INCOMPATIBLE: {e}")
        print(f"  Agents-A1 uses ArraysCache (GatedDeltaNet layers).")
        print(f"  mlx-lm speculative decoding requires trimmable KVCache.\n")

    # Results
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Baseline:    {baseline:.1f} tok/s")
    if speculative is not None:
        speedup = speculative / baseline if baseline > 0 else 0
        print(f"  Speculative: {speculative:.1f} tok/s")
        print(f"  Speedup:     {speedup:.2f}x")
    else:
        print(f"  Speculative: N/A (incompatible cache type)")
    print(f"  Memory:      {mx.metal.get_active_memory() / 1e9:.2f} GB active")
    print(f"               {mx.metal.get_peak_memory() / 1e9:.2f} GB peak")


if __name__ == "__main__":
    main()
