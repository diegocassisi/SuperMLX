"""
[AI_DIRECTIVE]
ROL: Extrae los tensores MTP del modelo original Qwen3.6-35B-A3B en FP16.
OBJETIVO: Crear un MTP head sin cuantizar para medir el techo de alpha.
ENTRADAS: Shards 25 y 26 de Qwen/Qwen3.6-35B-A3B (HuggingFace).
SALIDAS: models/Qwen3.6-35B-A3B-MTP-FP16/model.safetensors + config.json
REGLAS INVIOLABLES:
- Descargar un shard a la vez para minimizar uso de disco.
- Borrar cada shard inmediatamente después de extraer las claves MTP.
- No cargar el modelo completo en memoria.
"""

import json
import logging
import os
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("extract_mtp")

REPO_ID = "Qwen/Qwen3.6-35B-A3B"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "models" / "Qwen3.6-35B-A3B-MTP-FP16"

# MTP tensors and their shards
MTP_SHARDS = {
    "model-00025-of-00026.safetensors": ["mtp.layers.0.mlp.experts.gate_up_proj"],
    "model-00026-of-00026.safetensors": [
        "mtp.fc.weight",
        "mtp.layers.0.input_layernorm.weight",
        "mtp.layers.0.mlp.experts.down_proj",
        "mtp.layers.0.mlp.gate.weight",
        "mtp.layers.0.mlp.shared_expert.down_proj.weight",
        "mtp.layers.0.mlp.shared_expert.gate_proj.weight",
        "mtp.layers.0.mlp.shared_expert.up_proj.weight",
        "mtp.layers.0.mlp.shared_expert_gate.weight",
        "mtp.layers.0.post_attention_layernorm.weight",
        "mtp.layers.0.self_attn.k_norm.weight",
        "mtp.layers.0.self_attn.k_proj.weight",
        "mtp.layers.0.self_attn.o_proj.weight",
        "mtp.layers.0.self_attn.q_norm.weight",
        "mtp.layers.0.self_attn.q_proj.weight",
        "mtp.layers.0.self_attn.v_proj.weight",
        "mtp.norm.weight",
        "mtp.pre_fc_norm_embedding.weight",
        "mtp.pre_fc_norm_hidden.weight",
    ],
}


def main() -> None:
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from safetensors.torch import save_file
    import torch

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_tensors: dict[str, torch.Tensor] = {}

    for shard_name, mtp_keys in MTP_SHARDS.items():
        logger.info("[INICIO] Descargando shard: %s", shard_name)
        local_path = hf_hub_download(repo_id=REPO_ID, filename=shard_name)
        logger.info("[DATA] ✓ Shard descargado: %s", local_path)

        # Extract only MTP keys using safe_open (memory-efficient)
        with safe_open(local_path, framework="pt") as f:
            available_keys = f.keys()
            for key in mtp_keys:
                if key in available_keys:
                    # Strip "mtp." prefix to match our shim's expected format
                    clean_key = key[4:] if key.startswith("mtp.") else key
                    tensor = f.get_tensor(key)
                    all_tensors[clean_key] = tensor
                    logger.info("[DATA] ✓ %s: shape=%s, dtype=%s", clean_key, list(tensor.shape), tensor.dtype)
                else:
                    logger.warning("[ERROR] Clave %s no encontrada en shard", key)

        logger.info("[RESULT] Shard %s procesado. %d tensores extraídos hasta ahora.", shard_name, len(all_tensors))

    # Save extracted MTP tensors
    output_path = OUTPUT_DIR / "model.safetensors"
    save_file(all_tensors, str(output_path))
    total_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info("[RESULT] ✓ MTP head FP16 guardado: %s (%.1f MB)", output_path, total_mb)

    # Create config without quantization
    config = {
        "block_size": 3,
        "model_type": "qwen3_5_mtp",
        "text_config": {
            "attention_bias": False,
            "attention_dropout": 0.0,
            "attn_output_gate": True,
            "bos_token_id": 248044,
            "dtype": "bfloat16",
            "eos_token_id": 248044,
            "full_attention_interval": 4,
            "head_dim": 256,
            "hidden_act": "silu",
            "hidden_size": 2048,
            "initializer_range": 0.02,
            "layer_types": ["linear_attention"] * 3 + ["full_attention"] + (["linear_attention"] * 3 + ["full_attention"]) * 9,
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 128,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 32,
            "linear_value_head_dim": 128,
            "mamba_ssm_dtype": "float32",
            "max_position_embeddings": 262144,
            "model_type": "qwen3_5_moe_text",
            "moe_intermediate_size": 512,
            "mtp_num_hidden_layers": 1,
            "mtp_use_dedicated_embeddings": False,
            "num_attention_heads": 16,
            "num_experts": 256,
            "num_experts_per_tok": 8,
            "num_hidden_layers": 40,
            "num_key_value_heads": 2,
            "output_router_logits": False,
            "pad_token_id": None,
            "partial_rotary_factor": 0.25,
            "rms_norm_eps": 1e-06,
            "rope_parameters": {
                "mrope_interleaved": True,
                "mrope_section": [11, 11, 10],
                "partial_rotary_factor": 0.25,
                "rope_theta": 10000000,
                "rope_type": "default",
            },
            "router_aux_loss_coef": 0.001,
            "shared_expert_intermediate_size": 512,
            "tie_word_embeddings": False,
            "use_cache": True,
            "vocab_size": 248320,
        },
        "tie_word_embeddings": False,
    }

    config_path = OUTPUT_DIR / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    logger.info("[RESULT] ✓ Config guardado: %s", config_path)
    logger.info("[RESULT] ✓ Extracción completa. Total: %d tensores, %.1f MB", len(all_tensors), total_mb)


if __name__ == "__main__":
    main()
