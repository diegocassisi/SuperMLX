#!/usr/bin/env python3
"""
[AI_DIRECTIVE]
ROL: REAP Expert Pruning — elimina experts de baja saliencia de safetensors.
OBJETIVO: Leer saliencia JSON, remover experts podados, escribir modelo nuevo.
ENTRADAS: Saliency JSON (de reap_profile.py), modelo original, output dir.
SALIDAS: Nuevo modelo con N_pruned experts en safetensors + config actualizado.
REGLAS INVIOLABLES:
- NUNCA modifica el modelo original. Escribe en output_dir separado.
- Verifica shapes antes de operar. Abort si no coinciden.
- Copia tokenizer files sin modificar.
SSoT: Saliency JSON es la única fuente de decisiones de poda.
"""

import argparse
import gc
import json
import logging
import shutil
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

VERSION = "1.0.0"

# Tokenizer/config files to copy unchanged
_COPY_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "merges.txt",
    "vocab.json",
    "generation_config.json",
]

# MoE projection names (must match expert_cache.py)
_PROJ_NAMES = ("gate_proj", "up_proj", "down_proj")

# Tensor name patterns for MoE components
# Format: layers.{i}.mlp.{component}
_GATE_PATTERN = "layers.{layer}.mlp.gate.weight"
_EXPERT_PATTERN = "layers.{layer}.mlp.switch_mlp.{proj}.{param}"


def resolve_model_path(model_path: str) -> Path:
    """Resolve HuggingFace model path to local directory."""
    p = Path(model_path)
    if p.exists():
        return p
    # Try HF cache
    from mlx_lm.utils import hf_repo_to_path
    return Path(hf_repo_to_path(model_path))


def load_safetensors_index(model_dir: Path) -> dict:
    """Load the safetensors index (maps tensor names to shard files)."""
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            return json.load(f)
    # Single-file model
    sf_path = model_dir / "model.safetensors"
    if sf_path.exists():
        return {"weight_map": {}}  # will load from single file
    raise FileNotFoundError(f"No safetensors found in {model_dir}")


def load_tensor(model_dir: Path, index: dict, tensor_name: str) -> np.ndarray:
    """Load a single tensor from safetensors by name."""
    from safetensors import safe_open

    weight_map = index.get("weight_map", {})
    shard_file = weight_map.get(tensor_name)
    if shard_file:
        path = model_dir / shard_file
    else:
        # Single file fallback
        path = model_dir / "model.safetensors"

    with safe_open(str(path), framework="numpy") as f:
        if tensor_name in f.keys():
            return f.get_tensor(tensor_name)

    raise KeyError(f"Tensor {tensor_name} not found in {path}")


def prune_tensor(tensor: np.ndarray, keep_mask: np.ndarray) -> np.ndarray:
    """Remove experts from tensor along axis 0.

    Args:
        tensor: shape [num_experts, ...] — any MoE tensor with expert dim first
        keep_mask: boolean array of shape [num_experts]

    Returns:
        Pruned tensor with shape [target_experts, ...]
    """
    return tensor[keep_mask]


def main():
    parser = argparse.ArgumentParser(
        description="REAP Expert Pruning — create pruned model from saliency profile."
    )
    parser.add_argument(
        "--model", default="mlx-community/Qwen3.6-35B-A3B-4bit",
        help="Original model path (HF or local)",
    )
    parser.add_argument(
        "--saliency", required=True,
        help="Path to expert_saliency.json from reap_profile.py",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Output directory for pruned model",
    )
    parser.add_argument(
        "--target-experts", type=int, default=None,
        help="Override target experts (default: from saliency JSON)",
    )
    args = parser.parse_args()

    t0 = time.time()
    logger.info("[INICIO] REAP Pruning v%s", VERSION)

    # ── Load saliency data ────────────────────────────────────────────────
    with open(args.saliency) as f:
        saliency = json.load(f)

    num_experts = saliency["num_experts"]
    target = args.target_experts or saliency["prune_recommendation"]["target_experts"]
    prune_map = saliency["prune_recommendation"]["per_layer"]

    logger.info("[CONFIG] model=%s", args.model)
    logger.info("[CONFIG] num_experts=%d → target=%d (pruning %d per layer)",
                num_experts, target, num_experts - target)
    logger.info("[CONFIG] output_dir=%s", args.output_dir)

    # ── Resolve paths ─────────────────────────────────────────────────────
    model_dir = resolve_model_path(args.model)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("[DATA] Source: %s", model_dir)

    # ── Load safetensors index ────────────────────────────────────────────
    index = load_safetensors_index(model_dir)
    weight_map = index.get("weight_map", {})

    # Identify all tensor names
    all_tensor_names = set(weight_map.keys())
    if not all_tensor_names:
        # Single file — enumerate tensors
        from safetensors import safe_open
        sf_path = model_dir / "model.safetensors"
        with safe_open(str(sf_path), framework="numpy") as f:
            all_tensor_names = set(f.keys())

    logger.info("[DATA] Total tensors in model: %d", len(all_tensor_names))

    # ── Identify MoE layers from prune map ────────────────────────────────
    moe_layer_indices = sorted(int(k) for k in prune_map.keys())
    logger.info("[CONFIG] MoE layers to prune: %d layers %s",
                len(moe_layer_indices), moe_layer_indices[:5])

    # ── Build keep masks per layer ────────────────────────────────────────
    keep_masks = {}
    for layer_key, prune_ids in prune_map.items():
        mask = np.ones(num_experts, dtype=bool)
        mask[prune_ids] = False
        keep_masks[layer_key] = mask
        logger.info("[CALC] Layer %s: pruning %d experts %s",
                    layer_key, len(prune_ids),
                    prune_ids[:5] if len(prune_ids) > 5 else prune_ids)

    # ── Process tensors ───────────────────────────────────────────────────
    # Strategy: load all tensors, prune MoE ones, write new shards.
    # We write a single output shard for simplicity.

    from safetensors import safe_open
    from safetensors.numpy import save_file

    pruned_tensors = {}
    pruned_count = 0
    copied_count = 0

    # Collect all shard files
    shard_files = set(weight_map.values()) if weight_map else {"model.safetensors"}

    for shard_name in sorted(shard_files):
        shard_path = model_dir / shard_name
        logger.info("[DATA] Processing shard: %s", shard_name)

        with safe_open(str(shard_path), framework="numpy") as f:
            for tensor_name in f.keys():
                tensor = f.get_tensor(tensor_name)

                # Check if this tensor belongs to a MoE layer
                is_moe = False
                for layer_idx in moe_layer_indices:
                    layer_key = str(layer_idx)

                    # Gate weight: layers.{i}.mlp.gate.weight
                    gate_name = _GATE_PATTERN.format(layer=layer_idx)
                    if tensor_name == gate_name:
                        # Gate: shape [num_experts, hidden_dim]
                        if tensor.shape[0] != num_experts:
                            logger.error(
                                "[ERROR] Gate %s shape[0]=%d != num_experts=%d. Aborting.",
                                tensor_name, tensor.shape[0], num_experts,
                            )
                            return
                        pruned_tensors[tensor_name] = prune_tensor(tensor, keep_masks[layer_key])
                        pruned_count += 1
                        is_moe = True
                        break

                    # Expert projections: layers.{i}.mlp.switch_mlp.{proj}.{param}
                    for proj in _PROJ_NAMES:
                        for param in ("weight", "scales", "biases"):
                            expert_name = _EXPERT_PATTERN.format(
                                layer=layer_idx, proj=proj, param=param,
                            )
                            if tensor_name == expert_name:
                                if tensor.shape[0] != num_experts:
                                    logger.error(
                                        "[ERROR] %s shape[0]=%d != num_experts=%d. Aborting.",
                                        tensor_name, tensor.shape[0], num_experts,
                                    )
                                    return
                                pruned_tensors[tensor_name] = prune_tensor(
                                    tensor, keep_masks[layer_key],
                                )
                                pruned_count += 1
                                is_moe = True
                                break
                        if is_moe:
                            break

                    if is_moe:
                        break

                if not is_moe:
                    # Non-MoE tensor: copy as-is
                    pruned_tensors[tensor_name] = tensor
                    copied_count += 1

        gc.collect()

    logger.info("[CALC] Pruned %d MoE tensors, copied %d non-MoE tensors",
                pruned_count, copied_count)

    # ── Write pruned safetensors ──────────────────────────────────────────
    # Write as a single shard for simplicity
    output_sf = output_dir / "model.safetensors"
    logger.info("[DATA] Writing pruned model to %s...", output_sf)

    save_file(pruned_tensors, str(output_sf))
    del pruned_tensors
    gc.collect()

    output_size_gb = output_sf.stat().st_size / 1e9
    logger.info("[DATA] ✓ Written %.2f GB", output_size_gb)

    # ── Write index (single shard) ────────────────────────────────────────
    # Re-read tensor names from the written file for the index
    with safe_open(str(output_sf), framework="numpy") as f:
        tensor_names = list(f.keys())

    new_index = {
        "metadata": {"total_size": int(output_sf.stat().st_size)},
        "weight_map": {name: "model.safetensors" for name in tensor_names},
    }
    index_path = output_dir / "model.safetensors.index.json"
    with open(index_path, "w") as f:
        json.dump(new_index, f, indent=2)

    # ── Update config.json ────────────────────────────────────────────────
    config_src = model_dir / "config.json"
    with open(config_src) as f:
        config = json.load(f)

    # Update expert count
    original_experts = config.get("num_local_experts", config.get("num_experts", num_experts))
    config["num_local_experts"] = target
    if "num_experts" in config:
        config["num_experts"] = target

    # Add pruning metadata
    config["_reap_pruning"] = {
        "original_experts": original_experts,
        "pruned_to": target,
        "saliency_source": str(Path(args.saliency).name),
        "pruned_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    config_out = output_dir / "config.json"
    with open(config_out, "w") as f:
        json.dump(config, f, indent=2)
    logger.info("[DATA] ✓ config.json updated: num_local_experts %d → %d",
                original_experts, target)

    # ── Copy tokenizer and other files ────────────────────────────────────
    for filename in _COPY_FILES:
        src = model_dir / filename
        if src.exists():
            shutil.copy2(src, output_dir / filename)

    # Also copy any .py model files (model architecture code)
    for py_file in model_dir.glob("*.py"):
        shutil.copy2(py_file, output_dir / py_file.name)

    # ── Summary ───────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    logger.info("[RESULT] Status=SUCCESS Time=%ds", int(elapsed))
    logger.info("[RESULT] Pruned model at: %s", output_dir)
    logger.info("[RESULT] Experts: %d → %d", original_experts, target)
    logger.info("[RESULT] Size: %.2f GB", output_size_gb)
    logger.info("")
    logger.info("[RESULT] To use with SuperMLX:")
    logger.info("  MODEL_PATH=%s MOE_EXPERT_CAPACITY=%d python -m supermlx.server",
                output_dir, target)


if __name__ == "__main__":
    main()
