"""Quantize MTP head from bf16 to 3-bit and save alongside Agents-A1-3bit.

Downloads mtp.safetensors from wang-yang/Agents-A1-MTPLX-Q4,
quantizes large weight matrices to 3-bit (matching main model),
and saves as mtp.safetensors in the local model cache.
"""
import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import hf_hub_download, scan_cache_dir
from pathlib import Path
import json
import shutil

SRC_REPO = "wang-yang/Agents-A1-MTPLX-Q4"
DST_REPO = "mlx-community/Agents-A1-3bit"
BITS = 3
GROUP_SIZE = 64  # match main model quantization


def find_model_path(repo_id: str) -> Path:
    cache = scan_cache_dir()
    for repo in cache.repos:
        if repo.repo_id == repo_id:
            for rev in repo.revisions:
                return Path(rev.snapshot_path)
    raise FileNotFoundError(f"Model {repo_id} not in cache")


def quantize_weight(w: mx.array, bits: int, group_size: int):
    """Quantize a weight matrix using mlx's affine quantization."""
    return mx.quantize(w, bits=bits, group_size=group_size)


def main():
    # Load source MTP tensors (bf16)
    src_path = hf_hub_download(SRC_REPO, "mtp.safetensors")
    print(f"Loading MTP tensors from {src_path}")
    tensors = mx.load(src_path)
    print(f"  {len(tensors)} tensors, bf16")

    # Quantize large weight matrices, keep small ones as-is
    quantized = {}
    kept_fp = 0
    quantized_count = 0

    for name, tensor in sorted(tensors.items()):
        # Only quantize 2D weight matrices (not biases, norms, etc.)
        if tensor.ndim == 2 and tensor.shape[0] >= 256 and tensor.shape[1] >= 256:
            w_q, scales, biases = quantize_weight(tensor, BITS, GROUP_SIZE)
            quantized[name] = w_q
            quantized[name.replace(".weight", ".scales").replace(".weight", ".scales") if ".weight" in name else name + "_scales"] = scales
            quantized[name.replace(".weight", ".biases").replace(".weight", ".biases") if ".weight" in name else name + "_biases"] = biases
            quantized_count += 1
        else:
            quantized[name] = tensor
            kept_fp += 1

    print(f"  Quantized: {quantized_count} tensors to {BITS}-bit")
    print(f"  Kept fp: {kept_fp} tensors")

    # Save alongside the model
    dst_dir = find_model_path(DST_REPO)
    dst_path = dst_dir / "mtp.safetensors"

    # Backup existing if any
    if dst_path.exists():
        backup = dst_path.with_suffix(".safetensors.bak")
        shutil.copy2(dst_path, backup)
        print(f"  Backed up existing to {backup}")

    mx.save_safetensors(str(dst_path), quantized)
    size_mb = dst_path.stat().st_size / 1024 / 1024
    print(f"  Saved: {dst_path} ({size_mb:.1f} MB)")

    # Update config.json to indicate MTP support
    config_path = dst_dir / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    # Backup config
    config_bak = config_path.with_suffix(".json.bak")
    if not config_bak.exists():
        shutil.copy2(config_path, config_bak)
        print(f"  Backed up config to {config_bak}")

    # The text_config should already have mtp_num_hidden_layers from the base
    tc = config.get("text_config", config)
    if tc.get("mtp_num_hidden_layers") is None:
        tc["mtp_num_hidden_layers"] = 1
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print("  Updated config.json: mtp_num_hidden_layers=1")
    else:
        print(f"  Config already has mtp_num_hidden_layers={tc['mtp_num_hidden_layers']}")

    print("\nDone! MTP head ready.")
    return str(dst_path)


if __name__ == "__main__":
    main()
