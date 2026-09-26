"""
[AI_DIRECTIVE]
ROL: Cuantiza el MTP head FP16 a 8-bit para obtener mejor calidad que 4-bit sin exceder RAM.
OBJETIVO: Crear head 8-bit como compromiso entre calidad y tamaño.
ENTRADAS: models/Qwen3.6-35B-A3B-MTP-FP16/model.safetensors (bf16)
SALIDAS: models/Qwen3.6-35B-A3B-MTP-8bit/model.safetensors + config.json
"""

import json
import logging
import shutil
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("quantize_mtp")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FP16_DIR = PROJECT_ROOT / "models" / "Qwen3.6-35B-A3B-MTP-FP16"
OUTPUT_DIR = PROJECT_ROOT / "models" / "Qwen3.6-35B-A3B-MTP-8bit"

BITS = 8
GROUP_SIZE = 64


def main() -> None:
    from mlx_lm.models.qwen3_5 import TextModelArgs

    # 1. Load config
    with open(FP16_DIR / "config.json") as f:
        config = json.load(f)
    text_config = config["text_config"]
    args = TextModelArgs.from_dict(text_config)

    # 2. Build MTP module structure
    from supermlx.mtp.qwen_mtp_shim import _make_qwen3_5_mtp_module
    mtp_module = _make_qwen3_5_mtp_module(args)

    # 3. Load FP16 weights and sanitize MoE keys (PyTorch -> MLX switch_mlp)
    raw_weights = dict(mx.load(str(FP16_DIR / "model.safetensors")))
    sanitized: dict[str, mx.array] = {}
    for k, v in raw_weights.items():
        if k == "layers.0.mlp.experts.gate_up_proj":
            mid = v.shape[-2] // 2
            sanitized["layers.0.mlp.switch_mlp.gate_proj.weight"] = v[..., :mid, :]
            sanitized["layers.0.mlp.switch_mlp.up_proj.weight"] = v[..., mid:, :]
        elif k == "layers.0.mlp.experts.down_proj":
            sanitized["layers.0.mlp.switch_mlp.down_proj.weight"] = v
        elif "norm" in k and v.ndim == 1:
            sanitized[k] = v + 1.0
        else:
            sanitized[k] = v

    mtp_module.load_weights(list(sanitized.items()), strict=True)
    mx.eval(mtp_module.parameters())
    logger.info("[DATA] ✓ FP16 weights loaded and sanitized (strict=True): %d tensors", len(sanitized))

    # 4. Quantize to 8-bit
    nn.quantize(mtp_module, group_size=GROUP_SIZE, bits=BITS)
    mx.eval(mtp_module.parameters())
    logger.info("[CALC] ✓ Quantized to %d-bit, group_size=%d", BITS, GROUP_SIZE)

    # 5. Save quantized weights
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    from mlx.utils import tree_flatten
    flat = dict(tree_flatten(mtp_module.parameters()))
    mx.save_safetensors(str(OUTPUT_DIR / "model.safetensors"), flat)
    total_mb = (OUTPUT_DIR / "model.safetensors").stat().st_size / (1024 * 1024)
    logger.info("[RESULT] ✓ Saved: %s (%.1f MB)", OUTPUT_DIR / "model.safetensors", total_mb)

    # 6. Save config with quantization info
    config["quantization_config"] = {
        "group_size": GROUP_SIZE,
        "bits": BITS,
        "mode": "affine",
    }
    with open(OUTPUT_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    logger.info("[RESULT] ✓ Config saved: %s", OUTPUT_DIR / "config.json")


if __name__ == "__main__":
    main()
