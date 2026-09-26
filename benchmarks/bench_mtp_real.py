"""
[AI_DIRECTIVE]
ROL: Benchmark de MTP speculative decoding con el modelo real.
OBJETIVO: Medir acceptance rate y tokens/s con el modelo Qwen3.6-35B-A3B + cabezal MTP real.
ENTRADAS: Modelo cargado desde disco (MODEL_PATH + MTP_WEIGHTS_PATH de .env)
SALIDAS: Métricas de generación: alpha%, tokens/s, tokens generados.
REGLAS INVIOLABLES:
- Prohibido print() para debug. Usar logger.
- Medir con el modelo real, no mocks.
SSoT: Este script mide el estado actual del motor MTP.
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

# Load .env manually (minimal, no dotenv dependency)
env_path = PROJECT_ROOT / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, value = line.partition("=")
            # Strip inline comments: "256  # comment" → "256"
            if "  #" in value:
                value = value[:value.index("  #")]
            elif "\t#" in value:
                value = value[:value.index("\t#")]
            os.environ.setdefault(key.strip(), value.strip())

import mlx.core as mx

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("bench_mtp")

MODEL_PATH = os.environ.get("MODEL_PATH", "unsloth/Qwen3.6-35B-A3B-UD-MLX-3bit")
MTP_WEIGHTS = PROJECT_ROOT / os.environ.get("MTP_WEIGHTS_PATH", "models/Qwen3.6-35B-A3B-MTP-FP16")

PROMPT_SHORT = "Explain the key differences between speculative decoding and standard autoregressive generation in large language models. Be detailed and technical."

PROMPT_LONG = """You are an expert AI systems architect specializing in high-performance inference on Apple Silicon. You have deep knowledge of MLX, Metal GPU programming, KV cache management, and speculative decoding techniques. Your responses should be technically precise and well-structured.

The user is building a local LLM server called SuperMLX that runs Qwen3.6-35B-A3B (a Mixture of Experts model with GatedDeltaNet recurrent layers) on a Mac Mini M4 Pro with 64GB unified memory. The server implements Multi-Token Prediction (MTP) speculative decoding with a single MTP head that predicts one token ahead.

Current architecture:
- Trunk: 40 layers (mix of full attention and GatedDeltaNet linear attention)
- MTP Head: 1 MoE layer with its own KV cache
- Quantization: 3-bit trunk weights, 4-bit MTP head
- Framework: MLX 0.32 with custom model shims
- Features: Dual-slot KV cache, Tool Prefix Cache (TPC), session-aware routing

The user has been optimizing MTP acceptance rate and throughput. Current metrics show 77.7% acceptance rate with MTP throughput roughly matching autoregressive baseline (~64 t/s). The following fixes have been applied:
1. Post-normalization: MTP head now receives post-norm hidden states (matching VeOmni documentation)
2. Both pairs feeding: On acceptance, both (hidden_confirmed, draft_tok) and (hidden_draft, bonus_tok) are fed to the MTP head
3. MTP prompt prefill: The MTP head receives full prompt context during prefill
4. Selective projection: logits_to_keep parameter avoids unnecessary vocabulary projections

Given this context, explain what additional optimizations could push MTP throughput meaningfully above the autoregressive baseline. Consider the specific constraints of depth-1 MTP on Apple Silicon with a quantized MoE trunk."""

PROMPT = PROMPT_SHORT
MAX_TOKENS = 200
RUNS = 3


def load_model():
    """Load the real model + MTP head, same flow as server.py."""
    from supermlx.mtp import (
        install_qwen3_5_mtp_trunk_shim,
        inject_qwen3_5_mtp_support,
        validate_qwen3_5_mtp_support,
    )

    install_qwen3_5_mtp_trunk_shim()

    from mlx_lm import load

    logger.info("[INICIO] Loading model: %s", MODEL_PATH)
    model, tokenizer = load(MODEL_PATH, tokenizer_config={"trust_remote_code": True})
    logger.info("[DATA] ✓ Model loaded")

    mtp_cfg_path = MTP_WEIGHTS / "config.json"
    if not mtp_cfg_path.exists():
        logger.error("[ERROR] MTP config not found: %s", mtp_cfg_path)
        sys.exit(1)

    with open(mtp_cfg_path) as f:
        mtp_cfg = json.load(f)

    ok = inject_qwen3_5_mtp_support(model, MTP_WEIGHTS, mtp_cfg)
    if not (ok and validate_qwen3_5_mtp_support(model)):
        logger.error("[ERROR] MTP injection failed")
        sys.exit(1)

    logger.info("[RESULT] ✓ MTP head injected and validated")
    return model, tokenizer


def bench_mtp(model, tokenizer, label: str = "current", prompt: str = PROMPT_SHORT):
    """Run MTP speculative generation and measure metrics."""
    from supermlx.mtp.speculative_engine import stream_generate_mtp

    prompt_tokens = tokenizer.encode(prompt)
    logger.info("[CONFIG] Prompt tokens: %d, max_tokens: %d, runs: %d", len(prompt_tokens), MAX_TOKENS, RUNS)

    results = []
    for run in range(RUNS):
        tokens = []
        drafts_accepted = 0
        drafts_attempted = 0

        tic = time.perf_counter()
        tic_gen = None
        for n, resp in enumerate(stream_generate_mtp(
            model, tokenizer, mx.array(prompt_tokens),
            max_tokens=MAX_TOKENS,
        )):
            if n == 0:
                ttft = time.perf_counter() - tic
                tic_gen = time.perf_counter()
            else:
                tokens.append(resp.token)
            drafts_accepted = getattr(resp, "drafts_accepted", 0)
            drafts_attempted = getattr(resp, "drafts_attempted", 0)
        elapsed_gen = time.perf_counter() - tic_gen if tic_gen else 0.001
        elapsed_total = time.perf_counter() - tic

        alpha = 100.0 * drafts_accepted / max(drafts_attempted, 1)
        gen_tps = len(tokens) / max(elapsed_gen, 0.001)
        e2e_tps = (len(tokens) + 1) / max(elapsed_total, 0.001)
        results.append(dict(
            run=run + 1,
            tokens=len(tokens) + 1,
            ttft_s=round(ttft, 3),
            gen_tps=round(gen_tps, 1),
            e2e_tps=round(e2e_tps, 1),
            alpha=round(alpha, 1),
            accepted=drafts_accepted,
            attempted=drafts_attempted,
        ))
        logger.info(
            "[RESULT] Run %d/%d: %d tokens in %.2fs (gen=%.1f t/s, e2e=%.1f t/s, ttft=%.2fs) alpha=%.1f%% (%d/%d)",
            run + 1, RUNS, len(tokens) + 1, elapsed_total, gen_tps, e2e_tps, ttft, alpha, drafts_accepted, drafts_attempted,
        )

    avg_gen_tps = sum(r["gen_tps"] for r in results) / len(results)
    avg_e2e_tps = sum(r["e2e_tps"] for r in results) / len(results)
    avg_alpha = sum(r["alpha"] for r in results) / len(results)
    logger.info(
        "[RESULT] === %s AVERAGE: gen=%.1f t/s, e2e=%.1f t/s, alpha=%.1f%% ===",
        label.upper(), avg_gen_tps, avg_e2e_tps, avg_alpha,
    )

    output_path = PROJECT_ROOT / "benchmarks" / f"mtp_bench_{label}.json"
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(dict(label=label, model=MODEL_PATH, prompt_tokens=len(prompt_tokens),
                       max_tokens=MAX_TOKENS, runs=results,
                       avg_gen_tps=round(avg_gen_tps, 1), avg_e2e_tps=round(avg_e2e_tps, 1),
                       avg_alpha=round(avg_alpha, 1)), f, indent=2)
    logger.info("[DATA] Results saved to %s", output_path)
    return avg_gen_tps, avg_alpha


def bench_autoregressive(model, tokenizer, label: str = "autoregressive", prompt: str = PROMPT_SHORT):
    """Run standard autoregressive generation for baseline comparison."""
    from mlx_lm import stream_generate

    prompt_tokens = tokenizer.encode(prompt)
    logger.info("[CONFIG] Autoregressive baseline — prompt tokens: %d, max_tokens: %d", len(prompt_tokens), MAX_TOKENS)

    results = []
    for run in range(RUNS):
        tokens = []
        tic = time.perf_counter()
        tic_gen = None
        for n, resp in enumerate(stream_generate(
            model, tokenizer, prompt, max_tokens=MAX_TOKENS,
        )):
            if n == 0:
                ttft = time.perf_counter() - tic
                tic_gen = time.perf_counter()
            else:
                tokens.append(resp.token)
        elapsed_gen = time.perf_counter() - tic_gen if tic_gen else 0.001
        elapsed_total = time.perf_counter() - tic

        gen_tps = len(tokens) / max(elapsed_gen, 0.001)
        e2e_tps = (len(tokens) + 1) / max(elapsed_total, 0.001)
        results.append(dict(run=run + 1, tokens=len(tokens) + 1, ttft_s=round(ttft, 3), gen_tps=round(gen_tps, 1), e2e_tps=round(e2e_tps, 1)))
        logger.info("[RESULT] AR Run %d/%d: %d tokens in %.2fs (gen=%.1f t/s, e2e=%.1f t/s, ttft=%.2fs)",
                    run + 1, RUNS, len(tokens) + 1, elapsed_total, gen_tps, e2e_tps, ttft)

    avg_gen_tps = sum(r["gen_tps"] for r in results) / len(results)
    avg_e2e_tps = sum(r["e2e_tps"] for r in results) / len(results)
    logger.info("[RESULT] === AUTOREGRESSIVE AVERAGE: gen=%.1f t/s, e2e=%.1f t/s ===", avg_gen_tps, avg_e2e_tps)

    output_path = PROJECT_ROOT / "benchmarks" / f"mtp_bench_{label}.json"
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(dict(label=label, model=MODEL_PATH, prompt_tokens=len(prompt_tokens),
                       max_tokens=MAX_TOKENS, runs=results,
                       avg_gen_tps=round(avg_gen_tps, 1), avg_e2e_tps=round(avg_e2e_tps, 1)), f, indent=2)
    logger.info("[DATA] Results saved to %s", output_path)
    return avg_gen_tps


if __name__ == "__main__":
    model, tokenizer = load_model()

    for prompt_name, prompt_text in [("short", PROMPT_SHORT), ("long", PROMPT_LONG)]:
        prompt_tokens = tokenizer.encode(prompt_text)
        logger.info("")
        logger.info("#" * 60)
        logger.info("[INICIO] PROMPT: %s (%d tokens)", prompt_name.upper(), len(prompt_tokens))
        logger.info("#" * 60)

        logger.info("=" * 60)
        logger.info("[INICIO] BASELINE: Autoregressive (no MTP)")
        logger.info("=" * 60)
        ar_tps = bench_autoregressive(model, tokenizer, label=f"ar_{prompt_name}", prompt=prompt_text)

        logger.info("")
        logger.info("=" * 60)
        logger.info("[INICIO] MTP Speculative Decoding")
        logger.info("=" * 60)
        mtp_tps, mtp_alpha = bench_mtp(model, tokenizer, label=f"mtp_{prompt_name}", prompt=prompt_text)

        logger.info("")
        logger.info("=" * 60)
        speedup = (mtp_tps / ar_tps - 1) * 100 if ar_tps > 0 else 0
        logger.info("[RESULT] %s PROMPT: AR=%.1f t/s vs MTP=%.1f t/s (%.1f%% %s) alpha=%.1f%%",
                    prompt_name.upper(), ar_tps, mtp_tps, abs(speedup),
                    "faster" if speedup > 0 else "SLOWER", mtp_alpha)
        logger.info("=" * 60)
