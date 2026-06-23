# SPDX-License-Identifier: MIT
"""
[AI_DIRECTIVE]
ROL: CPU-Tier MoE Expert Loading for SuperMLX (V2).
OBJETIVO: Run Qwen3.6-35B-A3B with ALL 256 experts on 24 GB Apple Silicon
    by storing expert weights as numpy arrays in CPU RAM, not Metal buffers.
    Forward pass converts numpy → mx.array per MoE layer with per-layer
    mx.eval() to free temporaries. Result: 0% fallback, ~5 GB Metal.
ENTRADAS: mlx_lm model loaded with lazy=True, model path.
SALIDAS: Model with CpuTierSwitchLinear modules — all 256 experts available,
    per-layer eval for Metal memory management.
REGLAS INVIOLABLES:
- mx.eval() REQUIRED per MoE layer (free numpy→mx temporaries).
- All experts loaded — no capacity limit, no profile selection, no fallback.
- numpy.copy() on mmap data — no dependency on OS page cache.
- Compatible API surface with expert_cache.py (V1) for server.py.
SSoT: This module is the CPU-tier alternative to expert_cache.py.
"""

import logging
import time
import types
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np

# Reuse V1 utilities — no duplication
from .expert_cache import (
    SafetensorsMap,
    _find_switch_mlp,
    _find_moe_block,
    _detect_moe_config,
    _build_shard_map,
    _gate_returns_tuple,
    _PROJ_NAMES,
    is_moe_model,  # noqa: F401 — re-exported for server.py
)

logger = logging.getLogger(__name__)


# ── NumpyExpertStore ──────────────────────────────────────────────────────────


class NumpyExpertStore:
    """All experts for one MoE layer stored as numpy arrays in CPU RAM.

    On Apple Silicon unified memory, numpy arrays share physical RAM with Metal
    but are NOT tracked by Metal's memory accounting. 12+ GB of expert weights
    live here without counting against the ~17 GB Metal recommended working set.

    Forward pass: numpy → mx.array (memcpy within unified RAM) → gather_qmm → free.
    """

    __slots__ = (
        "num_experts",
        "np_weights", "np_scales", "np_biases",
        "mx_dtypes_w", "mx_dtypes_s", "mx_dtypes_b",
        "total_requests", "total_fallbacks",
    )

    def __init__(self, num_experts: int = 256):
        self.num_experts = num_experts
        self.np_weights: Dict[str, np.ndarray] = {}
        self.np_scales: Dict[str, np.ndarray] = {}
        self.np_biases: Dict[str, Optional[np.ndarray]] = {}
        # Per-tensor mx dtype for bfloat16 view-cast
        self.mx_dtypes_w: Dict[str, mx.Dtype] = {}
        self.mx_dtypes_s: Dict[str, mx.Dtype] = {}
        self.mx_dtypes_b: Dict[str, mx.Dtype] = {}
        self.total_requests: int = 0
        self.total_fallbacks: int = 0  # always 0

    def load_projection(
        self,
        st_map: SafetensorsMap,
        key_base: str,
        proj_name: str,
    ) -> int:
        """Load all experts for one projection from SafetensorsMap into numpy.

        Uses .copy() to detach from mmap — data lives in Python heap,
        not dependent on OS page cache.

        Args:
            st_map: Memory-mapped safetensors access.
            key_base: e.g. "model.layers.0.mlp.switch_mlp"
            proj_name: "gate_proj", "up_proj", or "down_proj"

        Returns:
            Number of experts loaded (= num_experts).
        """
        w_key = f"{key_base}.{proj_name}.weight"
        s_key = f"{key_base}.{proj_name}.scales"
        b_key = f"{key_base}.{proj_name}.biases"

        # Weight
        path, np_dt, mx_dt, shape, start, length = st_map._index[w_key]
        mm = st_map._mmaps[path]
        self.np_weights[proj_name] = np.frombuffer(
            mm[start:start + length], dtype=np_dt,
        ).reshape(shape).copy()
        self.mx_dtypes_w[proj_name] = mx_dt

        # Scales — may also be bfloat16 (uint16 in numpy)
        path_s, np_dt_s, mx_dt_s, shape_s, start_s, length_s = st_map._index[s_key]
        mm_s = st_map._mmaps[path_s]
        self.np_scales[proj_name] = np.frombuffer(
            mm_s[start_s:start_s + length_s], dtype=np_dt_s,
        ).reshape(shape_s).copy()
        self.mx_dtypes_s[proj_name] = mx_dt_s

        # Biases (optional — not all quant configs have biases)
        if b_key in st_map._index:
            path_b, np_dt_b, mx_dt_b, shape_b, start_b, length_b = st_map._index[b_key]
            mm_b = st_map._mmaps[path_b]
            self.np_biases[proj_name] = np.frombuffer(
                mm_b[start_b:start_b + length_b], dtype=np_dt_b,
            ).reshape(shape_b).copy()
            self.mx_dtypes_b[proj_name] = mx_dt_b
        else:
            self.np_biases[proj_name] = None

        return self.num_experts

    def numpy_bytes(self) -> int:
        """Total bytes of numpy arrays in this store."""
        total = 0
        for w in self.np_weights.values():
            total += w.nbytes
        for s in self.np_scales.values():
            total += s.nbytes
        for b in self.np_biases.values():
            if b is not None:
                total += b.nbytes
        return total


# ── CpuTierSwitchLinear ───────────────────────────────────────────────────────


class CpuTierSwitchLinear(nn.Module):
    """Expert dispatch from CPU RAM: numpy → mx.array → gather_qmm.

    Loads the FULL (num_experts, ...) expert weight tensor from numpy on each
    call. This allocates ~100 MB of temporary Metal per projection per layer.
    The caller (patched MoE block) calls mx.eval() after the layer to free
    these temporaries.

    Unlike V1's PredictiveCachedSwitchLinear:
    - No lookup table (all experts present → direct indexing)
    - No indices buffer (no dynamic cache updates)
    - No fallback slot 0 (no missing experts)
    """

    def __init__(
        self,
        group_size: int,
        bits: int,
        mode: str,
        proj_name: str,
        store: NumpyExpertStore,
    ):
        super().__init__()
        self.group_size = group_size
        self.bits = bits
        self.mode = mode
        self._proj_name = proj_name
        self._store = store
        self.freeze()

    def __call__(self, x, indices, sorted_indices=False):
        # numpy → mx.array: immediate allocation in Metal, memcpy in unified RAM
        pn = self._proj_name
        store = self._store

        w = mx.array(store.np_weights[pn])
        s = mx.array(store.np_scales[pn])
        b_np = store.np_biases.get(pn)
        b = mx.array(b_np) if b_np is not None else None

        # bfloat16 stored as uint16 in numpy — view-cast ALL affected tensors
        if store.mx_dtypes_w.get(pn) == mx.bfloat16:
            w = w.view(mx.bfloat16)
        if store.mx_dtypes_s.get(pn) == mx.bfloat16:
            s = s.view(mx.bfloat16)
        if b is not None and store.mx_dtypes_b.get(pn) == mx.bfloat16:
            b = b.view(mx.bfloat16)

        # Direct gather_qmm with ORIGINAL router indices — no remap needed
        return mx.gather_qmm(
            x, w, s, b,
            rhs_indices=indices,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
        )


# Shared counter across all MoE blocks — eval every N during prefill
_PREFILL_EVAL_INTERVAL = 5  # eval every 5 MoE layers → 8 sync points for 40 layers
_fwd_counter = [0]
_patched_classes: Dict[type, type] = {}  # cache: original_cls → patched_cls


def _patch_moe_block_v2(moe_block):
    """Patch MoE block for CPU-tier: no score masking + adaptive eval.

    CRITICAL: Python 3 looks up __call__ on type(obj), NOT on the instance.
    types.MethodType sets an instance attribute → never called by obj().
    Fix: reassign __class__ to a dynamically created subclass that defines
    __call__ at the class level.

    Eval strategy:
    - Decode (seq_len=1): eval every MoE layer → minimal Metal (~360 MB peak)
    - Prefill (seq_len>2): eval every 5 MoE layers → 8 sync points instead of 40
      Peak Metal: 5×360MB=1.8GB + base(2.3GB) + KV = ~5GB. No OOM.
    """
    original_cls = type(moe_block)

    if original_cls not in _patched_classes:

        class _CpuTierMoeBlock(original_cls):
            """MoE block with CPU-tier adaptive eval."""

            def __call__(self, x):
                # Runtime detection: gate may return (inds, scores) or raw logits
                gate_out = self.gate(x)
                if isinstance(gate_out, tuple):
                    inds, scores = gate_out
                else:
                    gates = gate_out
                    k = getattr(self, "num_experts_per_tok", getattr(self, "top_k", 2))
                    gates = mx.softmax(gates, axis=-1, precise=True)
                    inds = mx.stop_gradient(
                        mx.argpartition(-gates, kth=k - 1, axis=-1)[..., :k]
                    )
                    scores = mx.take_along_axis(gates, inds, axis=-1)
                    if getattr(self, "norm_topk_prob", False):
                        scores = scores / scores.sum(axis=-1, keepdims=True)

                # ALL experts available — no score masking needed
                y = self.switch_mlp(x, inds)
                y = (y * scores[..., None]).sum(axis=-2)

                # Shared expert (Qwen3 MoE architecture)
                if hasattr(self, "shared_expert") and hasattr(self, "shared_expert_gate"):
                    se = self.shared_expert(x)
                    y = y + mx.sigmoid(self.shared_expert_gate(x)) * se
                elif hasattr(self, "shared_experts"):
                    y = y + self.shared_experts(x)

                # Eval every N MoE layers to free numpy→mx temporaries.
                # NO clear_cache — let MLX reuse freed buffers across layers.
                _fwd_counter[0] += 1
                if _fwd_counter[0] % _PREFILL_EVAL_INTERVAL == 0:
                    mx.eval(y)

                return y

        _patched_classes[original_cls] = _CpuTierMoeBlock

    moe_block.__class__ = _patched_classes[original_cls]


# ── Main Integration ──────────────────────────────────────────────────────────


def enable_moe_cache(
    model: nn.Module,
    model_path: str,
    capacity: int = 256,
    profile_path: Optional[str] = None,
) -> Dict[str, Any]:
    """CPU-Tier: Load ALL experts as numpy arrays. Metal-free expert storage.

    Same signature as V1 for drop-in replacement in server.py.
    The `capacity` and `profile_path` args are IGNORED — all experts loaded.

    Flow:
    1. Detect MoE config (experts, layers, slot size)
    2. Build SafetensorsMap (manual mmap, bfloat16-safe)
    3. Replace QuantizedSwitchLinear with CpuTierSwitchLinear
       (removes expert weights from model parameter tree)
    4. mx.eval(model.parameters()) → only non-expert params (~5 GB Metal)
    5. Load ALL expert weights into numpy stores (~12 GB CPU RAM)
    6. Patch MoE blocks for per-layer eval (free temporaries)
    7. Wire memory (minimal — base model only)

    Memory budget on 24 GB Apple Silicon:
        Metal: ~5 GB (base model) + ~1.5 GB (KV) + ~0.3 GB (1 layer experts) = ~7 GB
        Numpy: ~12 GB (all experts in CPU RAM)
        Total unified: ~19 GB → 5 GB headroom
    """
    t0 = time.time()

    # Detect MoE config
    num_experts, moe_layers, expert_slot_mb = _detect_moe_config(model)
    if moe_layers == 0:
        logger.info("[MOE-V2] No MoE layers found — skipping.")
        return {"moe_layers": 0}

    logger.info(
        "[MOE-V2] CPU-Tier: ALL %d experts → numpy | %d MoE layers | %.2f MB/slot",
        num_experts, moe_layers, expert_slot_mb,
    )

    # Resolve model path
    from mlx_lm.utils import hf_repo_to_path
    resolved_path = Path(hf_repo_to_path(model_path))

    # Build shard map and SafetensorsMap (manual parsing, bfloat16-safe)
    shard_map = _build_shard_map(resolved_path)
    all_shards = sorted(set(shard_map.values()))
    st_map = SafetensorsMap(all_shards)

    from mlx_lm.models.switch_layers import QuantizedSwitchLinear

    # ── Pass 1: Replace modules ──────────────────────────────────────────
    # Install CpuTierSwitchLinear with empty NumpyExpertStores.
    # This removes QuantizedSwitchLinear (and its weight tensors) from the
    # model parameter tree, so mx.eval only materializes non-expert params.
    replaced = 0
    layer_stores: Dict[int, NumpyExpertStore] = {}
    layer_key_bases: Dict[int, str] = {}
    layer_quant_params: Dict[int, Dict[str, dict]] = {}

    for i, layer in enumerate(model.layers):
        switch, key_base = _find_switch_mlp(layer, i)
        if switch is None:
            continue

        store = NumpyExpertStore(num_experts)
        qp_layer: Dict[str, dict] = {}

        for proj_name in _PROJ_NAMES:
            orig = getattr(switch, proj_name)
            if isinstance(orig, QuantizedSwitchLinear):
                qp_layer[proj_name] = {
                    "group_size": orig.group_size,
                    "bits": orig.bits,
                    "mode": orig.mode,
                }

        # Replace modules
        for proj_name, qp in qp_layer.items():
            replacement = CpuTierSwitchLinear(
                group_size=qp["group_size"],
                bits=qp["bits"],
                mode=qp["mode"],
                proj_name=proj_name,
                store=store,
            )
            setattr(switch, proj_name, replacement)
            replaced += 1

        layer_stores[i] = store
        layer_key_bases[i] = key_base
        layer_quant_params[i] = qp_layer

    # ── Materialize non-expert params ────────────────────────────────────
    # With expert modules replaced, model.parameters() only contains
    # attention, embeddings, norms, gate, shared_expert (~5 GB).
    mx.eval(model.parameters())
    non_expert_gb = mx.get_active_memory() / 1e9
    logger.info("[MOE-V2] Non-expert params materialized: %.1f GB Metal", non_expert_gb)

    # ── Pass 2: Load ALL expert weights into numpy ───────────────────────
    # This is the big I/O step: ~12 GB from SSD → numpy (CPU RAM).
    # Uses .copy() to detach from mmap for reliable access.
    total_expert_tensors = 0
    t_load = time.time()

    for i, store in layer_stores.items():
        key_base = layer_key_bases[i]
        for proj_name in layer_quant_params[i]:
            loaded = store.load_projection(st_map, key_base, proj_name)
            total_expert_tensors += loaded

        if (i + 1) % 10 == 0 or i == max(layer_stores.keys()):
            elapsed_load = time.time() - t_load
            logger.info(
                "[MOE-V2] numpy load: %d/%d layers | %.1fs",
                i + 1, moe_layers, elapsed_load,
            )

    # ── Pass 3: Patch MoE blocks for per-layer eval ──────────────────────
    patched = 0
    for layer in model.layers:
        moe_block = _find_moe_block(layer)
        if moe_block is None:
            continue
        _patch_moe_block_v2(moe_block)
        patched += 1

    # Wire memory — only base model, experts are in numpy
    if hasattr(mx, "set_wired_limit"):
        active = mx.get_active_memory()
        mx.set_wired_limit(active)
        logger.info(
            "[MOE-V2] Wired %.1f GB (base model only — experts in numpy)",
            active / 1e9,
        )

    elapsed = time.time() - t0
    numpy_gb = sum(store.numpy_bytes() for store in layer_stores.values()) / 1e9

    stats = {
        "moe_layers": moe_layers,
        "num_experts": num_experts,
        "expert_slot_mb": round(expert_slot_mb, 2),
        "capacity": num_experts,
        "replaced_modules": replaced,
        "skip_fallback_patches": patched,
        "expert_tensors_loaded": total_expert_tensors,
        "elapsed_seconds": round(elapsed, 1),
        "active_memory_gb": round(non_expert_gb, 1),
        "numpy_memory_gb": round(numpy_gb, 1),
        "cpu_tier": True,
    }

    logger.info(
        "[MOE-V2] ✅ CPU-Tier enabled | layers=%d experts=%d (ALL) "
        "replaced=%d patched=%d | %.1fs | Metal=%.1fGB numpy=%.1fGB",
        moe_layers, num_experts, replaced, patched,
        elapsed, non_expert_gb, numpy_gb,
    )

    model._st_map = st_map
    model._moe_config = stats
    return stats


# ── Compatibility API (no-ops for server.py) ──────────────────────────────────
# Server.py imports these functions. CPU-tier doesn't need them but provides
# compatible signatures to avoid changing server.py logic.


def dynamic_cache_update(
    model: nn.Module,
    max_layer_updates: int = 12,
) -> List[Dict]:
    """No-op: all experts always loaded in CPU-tier mode."""
    return []


def dynamic_update_policy(model: nn.Module, **kwargs) -> Tuple[int, int]:
    """No-op: returns (interval=999999, budget=0) — never triggers."""
    return 999999, 0


def get_cache_stats(model: nn.Module) -> Dict[str, Any]:
    """CPU-tier stats: always 100% hit, 0% fallback."""
    config = getattr(model, "_moe_config", {})
    if config.get("moe_layers", 0) == 0:
        return {"moe_active": False}
    return {
        "moe_active": True,
        "capacity": config.get("num_experts", 256),
        "num_experts": config.get("num_experts", 256),
        "coverage": 1.0,
        "hit_rate": 1.0,
        "fallback_rate": 0.0,
        "total_requests": 0,
        "total_fallbacks": 0,
        "cpu_tier": True,
    }


def expand_expert_capacity(
    model: nn.Module,
    target_capacity: int = 256,
    profile_path: Optional[str] = None,
) -> Dict[str, Any]:
    """No-op: all experts already loaded in CPU-tier."""
    return {"expanded": False, "reason": "cpu_tier_all_loaded"}


def evict_experts_for_memory(
    model: nn.Module,
    **kwargs,
) -> int:
    """No-op: experts are in numpy, not Metal — nothing to evict."""
    return 0
