#!/usr/bin/env python3
"""
[AI_DIRECTIVE]
ROL: REAP Expert Profiling — mide saliencia por expert para decisiones de poda.
OBJETIVO: Correr modelo completo (128 experts), capturar gate routing, generar JSON de saliencia.
ENTRADAS: Model path (HF o local), output JSON path, target experts count.
SALIDAS: JSON con saliencia per-layer per-expert + recomendación de poda.
REGLAS INVIOLABLES:
- Offline only. Carga ALL experts sin predictive cache.
- mx.eval en hooks es aceptable (profiling, no producción).
- NUNCA podar expert con selection_rate > 5% (high-impact protection).
SSoT: Output JSON es la única fuente para reap_prune.py.
"""

import argparse
import gc
import json
import logging
import time
from collections import defaultdict
from pathlib import Path

import mlx.core as mx
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

VERSION = "1.0.0"

# ── Profiling Dataset ─────────────────────────────────────────────────────────
# 15 conversations covering: tool calling, JSON, code, reasoning, multilingual.

PROFILE_CONVERSATIONS = [
    # 1. Tool calling — Bash
    [
        {"role": "system", "content": "You are a helpful coding assistant with access to tools: Bash(command), Write(file_path, content), Read(file_path)."},
        {"role": "user", "content": "Create a Python script that reads a CSV file, processes the data, and generates a matplotlib bar chart showing sales by category. Save it to /tmp/chart.py"},
    ],
    # 2. Tool calling — Write
    [
        {"role": "system", "content": "You have tools: Write(file_path, content), Bash(command). Complete the task step by step."},
        {"role": "user", "content": "Set up a basic Express.js server with three endpoints: GET /health, POST /api/data, and GET /api/data/:id. Include error handling and input validation."},
    ],
    # 3. Tool calling — multi-step
    [
        {"role": "system", "content": "You have tools: Bash(command), Write(file_path, content), Read(file_path). Execute step by step."},
        {"role": "user", "content": "Write a bash script that monitors disk usage, sends alerts when any partition exceeds 90%, and rotates log files older than 7 days. Save to /tmp/monitor.sh and make it executable."},
    ],
    # 4. JSON generation
    [
        {"role": "system", "content": "You are a helpful assistant that always responds in valid JSON."},
        {"role": "user", "content": "Generate a JSON array of 10 software products with fields: id (int), name (string), version (string), price (float), categories (array of strings), metadata (object with author and license fields)."},
    ],
    # 5. JSON schema
    [
        {"role": "system", "content": "You are an API that returns structured data."},
        {"role": "user", "content": "Return a JSON object representing a REST API specification with endpoints, methods, request/response schemas, and authentication requirements for a user management service."},
    ],
    # 6. Code — Python
    [
        {"role": "system", "content": "You are an expert Python programmer."},
        {"role": "user", "content": "Implement a complete binary search tree with insert, delete, search, and in-order traversal. Include type hints and docstrings."},
    ],
    # 7. Code — ETL
    [
        {"role": "system", "content": "You are a data engineer."},
        {"role": "user", "content": "Write a Python ETL pipeline using pandas that reads multiple CSV files from a directory, cleans the data (handle nulls, normalize dates, deduplicate), joins on a common key, and outputs a single Parquet file."},
    ],
    # 8. Reasoning — technical analysis
    [
        {"role": "system", "content": "You are an analytical assistant."},
        {"role": "user", "content": "Analyze the trade-offs between using PostgreSQL vs MongoDB for a real-time analytics dashboard that ingests 10,000 events per second. Consider query patterns, scaling, and operational complexity."},
    ],
    # 9. Reasoning — system design
    [
        {"role": "system", "content": "You are a senior software architect."},
        {"role": "user", "content": "Design the architecture for a rate limiter service that handles 100k requests per second with multiple rate limiting strategies (fixed window, sliding window, token bucket). Include data structures and distributed coordination."},
    ],
    # 10. Mathematical/technical
    [
        {"role": "system", "content": "You are a mathematics tutor."},
        {"role": "user", "content": "Explain the Fast Fourier Transform algorithm step by step, including the butterfly diagram, bit-reversal permutation, and how it achieves O(n log n) complexity."},
    ],
    # 11. Creative coding — HTML
    [
        {"role": "system", "content": "You are an expert web developer."},
        {"role": "user", "content": "Create a single-file HTML page with an interactive Conway's Game of Life simulation using Canvas. Include play/pause, speed control, and the ability to draw cells with mouse clicks."},
    ],
    # 12. Debugging
    [
        {"role": "system", "content": "You are a debugging assistant."},
        {"role": "user", "content": "This Python code raises a TypeError. Fix it and explain why:\n\ndef merge_dicts(*dicts):\n    result = {}\n    for d in dicts:\n        for k, v in d:\n            if k in result:\n                result[k] += v\n            else:\n                result[k] = v\n    return result\n\nprint(merge_dicts({'a': 1}, {'a': 2, 'b': 3}))"},
    ],
    # 13. Spanish / multilingual
    [
        {"role": "system", "content": "Eres un asistente que responde en español."},
        {"role": "user", "content": "Explica cómo funciona el protocolo TCP/IP, incluyendo el three-way handshake, control de flujo, y manejo de congestión. Usa ejemplos prácticos."},
    ],
    # 14. Short response (edge case)
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ],
    # 15. Technical documentation
    [
        {"role": "system", "content": "You are a technical writer."},
        {"role": "user", "content": "Write comprehensive documentation for a WebSocket-based real-time notification system, including architecture overview, message protocol, reconnection strategy, and code examples in Python."},
    ],
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def numpy_softmax(x, axis=-1):
    """Numerically stable softmax in numpy."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def find_moe_blocks(model):
    """Find all MoE blocks in model. Returns list of (layer_idx, moe_block)."""
    blocks = []
    for i, layer in enumerate(getattr(model, "layers", [])):
        # Qwen: layer.mlp has switch_mlp + gate
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "switch_mlp"):
            blocks.append((i, layer.mlp))
            continue
        # DeepSeek/other: layer.block_sparse_moe
        if hasattr(layer, "block_sparse_moe") and hasattr(layer.block_sparse_moe, "switch_mlp"):
            blocks.append((i, layer.block_sparse_moe))
    return blocks


def install_gate_hooks(moe_blocks):
    """Hook gate.__call__ on each MoE block to capture raw logits.

    Returns:
        gate_store: dict[layer_idx] -> list of numpy logit arrays
        restore_fns: list of callables to restore original gates
    """
    gate_store = {i: [] for i, _ in moe_blocks}
    restore_fns = []

    for layer_idx, block in moe_blocks:
        gate = block.gate
        original_call = gate.__call__

        def make_hook(orig, layer_i):
            def hooked_gate(x):
                logits = orig(x)
                # Eval + store (offline profiling, eval is OK)
                mx.eval(logits)
                gate_store[layer_i].append(
                    np.array(logits).reshape(-1, logits.shape[-1])
                )
                return logits
            return hooked_gate

        gate.__call__ = make_hook(original_call, layer_idx)
        restore_fns.append(lambda g=gate, o=original_call: setattr(g, "__call__", o))

    return gate_store, restore_fns


def compute_saliency(gate_store, num_experts, top_k, protection_rate=0.05):
    """Compute per-expert saliency from captured gate logits.

    Args:
        gate_store: dict[layer_idx] -> list of numpy arrays [tokens, num_experts]
        num_experts: total expert count
        top_k: number of experts selected per token
        protection_rate: experts with selection_rate > this are protected from pruning

    Returns:
        dict with per-layer per-expert saliency data
    """
    layers_result = {}

    for layer_idx in sorted(gate_store.keys()):
        data = gate_store[layer_idx]
        if not data:
            continue

        all_logits = np.concatenate(data, axis=0)  # [total_tokens, num_experts]
        total_tokens = all_logits.shape[0]

        # Softmax scores
        scores = numpy_softmax(all_logits, axis=-1)

        # Top-K selection per token
        top_inds = np.argpartition(-scores, kth=top_k - 1, axis=-1)[:, :top_k]
        top_scores = np.take_along_axis(scores, top_inds, axis=-1)

        experts = {}
        for e in range(num_experts):
            mask = (top_inds == e)
            freq = int(mask.sum())
            if freq > 0:
                mean_score = float(np.mean(top_scores[mask]))
                saliency = freq * mean_score
            else:
                mean_score = 0.0
                saliency = 0.0

            sel_rate = freq / total_tokens if total_tokens > 0 else 0.0
            experts[str(e)] = {
                "frequency": freq,
                "selection_rate": round(sel_rate, 6),
                "mean_score": round(mean_score, 6),
                "saliency": round(saliency, 4),
                "protected": sel_rate > protection_rate,
            }

        layers_result[str(layer_idx)] = {
            "total_tokens": int(total_tokens),
            "experts": experts,
        }

    return layers_result


def compute_prune_recommendation(layers_result, num_experts, target_experts):
    """Determine which experts to prune per layer.

    Never prunes protected experts (selection_rate > 5%).
    """
    prune_map = {}
    protected_saved = 0

    for layer_key, layer_data in layers_result.items():
        experts = layer_data["experts"]
        # Sort by saliency ascending (lowest = prune first)
        sorted_experts = sorted(experts.items(), key=lambda x: x[1]["saliency"])

        n_to_prune = num_experts - target_experts
        prune_ids = []
        for eid_str, stats in sorted_experts:
            if len(prune_ids) >= n_to_prune:
                break
            if stats["protected"]:
                protected_saved += 1
                continue
            prune_ids.append(int(eid_str))

        prune_map[layer_key] = sorted(prune_ids)

    return prune_map, protected_saved


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="REAP Expert Profiling — profile MoE expert saliency for pruning."
    )
    parser.add_argument(
        "--model", default="mlx-community/Qwen3.6-35B-A3B-4bit",
        help="HuggingFace model path or local directory",
    )
    parser.add_argument(
        "--output", default="scripts/expert_saliency.json",
        help="Output JSON path for saliency data",
    )
    parser.add_argument(
        "--target-experts", type=int, default=100,
        help="Target number of experts per layer after pruning",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=50,
        help="Max tokens to generate per prompt (more = more decode routing data)",
    )
    parser.add_argument(
        "--protection-rate", type=float, default=0.05,
        help="Experts with selection_rate above this are protected from pruning",
    )
    args = parser.parse_args()

    t0 = time.time()
    logger.info("[INICIO] REAP Profiling v%s", VERSION)
    logger.info("[CONFIG] model=%s", args.model)
    logger.info("[CONFIG] target_experts=%d, max_tokens=%d, protection_rate=%.2f",
                args.target_experts, args.max_tokens, args.protection_rate)

    # ── Load model ────────────────────────────────────────────────────────
    from mlx_lm import load, generate

    logger.info("[DATA] Loading model (all experts)...")
    model, tokenizer = load(args.model)
    mem_gb = mx.get_active_memory() / 1e9
    logger.info("[DATA] ✓ Model loaded. Active memory: %.1f GB", mem_gb)

    # ── Find MoE blocks ──────────────────────────────────────────────────
    moe_blocks = find_moe_blocks(model)
    if not moe_blocks:
        logger.error("[ERROR] No MoE blocks found. Is this an MoE model?")
        return

    num_experts = moe_blocks[0][1].gate.weight.shape[0]
    top_k = getattr(moe_blocks[0][1], "num_experts_per_tok", 2)
    logger.info("[CONFIG] MoE layers=%d, num_experts=%d, top_k=%d",
                len(moe_blocks), num_experts, top_k)

    if args.target_experts >= num_experts:
        logger.error("[ERROR] target_experts (%d) >= num_experts (%d). Nothing to prune.",
                     args.target_experts, num_experts)
        return

    # ── Install gate hooks ────────────────────────────────────────────────
    gate_store, restore_fns = install_gate_hooks(moe_blocks)
    logger.info("[CONFIG] Gate hooks installed on %d layers", len(moe_blocks))

    # ── Run profiling ─────────────────────────────────────────────────────
    total_tokens_profiled = 0
    for i, conv in enumerate(PROFILE_CONVERSATIONS):
        # Format prompt using chat template
        if hasattr(tokenizer, "apply_chat_template"):
            prompt = tokenizer.apply_chat_template(
                conv, tokenize=False, add_generation_prompt=True,
            )
        else:
            prompt = "\n".join(m["content"] for m in conv)

        prompt_tokens = len(tokenizer.encode(prompt))
        logger.info("[DATA] Prompt %d/%d | %d input tokens | category: %s",
                    i + 1, len(PROFILE_CONVERSATIONS), prompt_tokens,
                    conv[0]["content"][:50])

        try:
            output = generate(
                model, tokenizer, prompt=prompt,
                max_tokens=args.max_tokens, verbose=False,
            )
            # Count tokens profiled (input + generated)
            output_tokens = len(tokenizer.encode(output))
            total_tokens_profiled += prompt_tokens + output_tokens
            logger.info("[DATA] ✓ Generated %d tokens", output_tokens)
        except Exception as e:
            logger.warning("[ERROR] Prompt %d failed: %s", i + 1, str(e))
            continue

        # Clear graph between prompts to manage memory
        mx.clear_cache()
        gc.collect()

    # ── Restore original gates ────────────────────────────────────────────
    for fn in restore_fns:
        fn()

    # ── Compute saliency ─────────────────────────────────────────────────
    logger.info("[CALC] Computing saliency across %d layers...", len(moe_blocks))
    layers_result = compute_saliency(
        gate_store, num_experts, top_k, args.protection_rate,
    )

    # ── Compute prune recommendation ─────────────────────────────────────
    prune_map, protected_saved = compute_prune_recommendation(
        layers_result, num_experts, args.target_experts,
    )
    n_to_prune = num_experts - args.target_experts

    logger.info("[CALC] Prune recommendation: %d experts per layer", n_to_prune)
    if protected_saved > 0:
        logger.info("[CALC] %d expert-layer pairs protected (selection_rate > %.1f%%)",
                    protected_saved, args.protection_rate * 100)

    # ── Build output ─────────────────────────────────────────────────────
    result = {
        "model": args.model,
        "num_experts": num_experts,
        "top_k": top_k,
        "total_prompts": len(PROFILE_CONVERSATIONS),
        "total_tokens_profiled": total_tokens_profiled,
        "protection_rate": args.protection_rate,
        "profiled_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "layers": layers_result,
        "prune_recommendation": {
            "target_experts": args.target_experts,
            "experts_to_prune": n_to_prune,
            "protected_experts_saved": protected_saved,
            "per_layer": prune_map,
        },
    }

    # ── Save ─────────────────────────────────────────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    elapsed = time.time() - t0
    logger.info("[RESULT] Status=SUCCESS Time=%ds", int(elapsed))
    logger.info("[RESULT] Saved to %s", output_path)
    logger.info("[RESULT] Next: python scripts/reap_prune.py --saliency %s", output_path)


if __name__ == "__main__":
    main()
