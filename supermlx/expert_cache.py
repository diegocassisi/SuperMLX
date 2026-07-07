# SPDX-License-Identifier: MIT
"""
[AI_DIRECTIVE]
ROL: MoE Expert Caching engine for SuperMLX (Predictive Phase 3 architecture).
OBJETIVO: Run large MoE models (35B+ Qwen3.6-A3B) on 24 GB Apple Silicon by loading
    only profiled experts into GPU RAM via zero-eval predictive dispatch.
ENTRADAS: mlx_lm model loaded with lazy=True, model path, expert profile JSON.
SALIDAS: Model with PredictiveCachedSwitchLinear modules — zero mx.eval() in forward,
    compact tensors (capacity, out, in), GPU lookup table for expert→slot remap.
REGLAS INVIOLABLES:
- NEVER call mx.eval() inside forward pass (breaks async_eval pipeline).
- Manual safetensors parsing for bfloat16 safety (no safe_open).
- Fallback to slot 0 with score zeroing + renormalize (no NaN).
- No-op for dense models.
- All memory estimates in MB, not bytes.
SSoT: This module is the single source of MoE expert management in SuperMLX.
"""

import json
import logging
import mmap
import struct
import time
import types
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from . import ssd_prefetch

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

_PROJ_NAMES = ("gate_proj", "up_proj", "down_proj")
_MAX_SWAPS_PER_LAYER = 10

# safetensors dtype → (numpy dtype, mlx dtype)
_ST_DTYPE_MAP = {
    "F16": (np.float16, mx.float16),
    "F32": (np.float32, mx.float32),
    "I8": (np.int8, mx.int8),
    "I16": (np.int16, mx.int16),
    "I32": (np.int32, mx.int32),
    "I64": (np.int64, mx.int64),
    "U8": (np.uint8, mx.uint8),
    "U16": (np.uint16, mx.uint16),
    "U32": (np.uint32, mx.uint32),
    "U64": (np.uint64, mx.uint64),
    # bfloat16: numpy has no bfloat16, load as uint16, view-cast in MLX
    "BF16": (np.uint16, mx.bfloat16),
}


# ── SafetensorsMap ────────────────────────────────────────────────────────────
# Manual header parsing + mmap. Does NOT use safetensors.safe_open() because
# that library mishandles bfloat16 with the MLX framework backend.


class SafetensorsMap:
    """Memory-mapped safetensors access with manual header parsing.

    Parses headers at init, mmaps each shard file. Supports:
    - get_tensor(key): full tensor load via numpy view of mmap region
    - get_expert_slices(key, expert_ids): byte-level per-expert row slicing

    bfloat16 tensors are loaded as uint16 numpy arrays then view-cast to
    mx.bfloat16 — the only safe path on MLX.
    """

    def __init__(self, shard_paths: List[str]):
        self._mmaps: Dict[str, mmap.mmap] = {}
        self._fds: Dict[str, object] = {}
        # {tensor_key: (shard_path, np_dtype, mx_dtype, shape, byte_offset, byte_length)}
        self._index: Dict[str, tuple] = {}

        for path in shard_paths:
            path = str(path)
            if path in self._mmaps:
                continue
            fd = open(path, "rb")
            header_size = struct.unpack("<Q", fd.read(8))[0]
            header_json = fd.read(header_size)
            header = json.loads(header_json)
            data_offset = 8 + header_size

            mm = mmap.mmap(fd.fileno(), 0, access=mmap.ACCESS_READ)
            self._mmaps[path] = mm
            self._fds[path] = fd

            for key, meta in header.items():
                if key == "__metadata__":
                    continue
                dtype_str = meta["dtype"]
                shape = meta["shape"]
                offsets = meta["data_offsets"]
                if dtype_str not in _ST_DTYPE_MAP:
                    logger.warning("[SAFETENSORS] Unknown dtype %s for key %s", dtype_str, key)
                    continue
                np_dt, mx_dt = _ST_DTYPE_MAP[dtype_str]
                start = data_offset + offsets[0]
                length = offsets[1] - offsets[0]
                self._index[key] = (path, np_dt, mx_dt, shape, start, length)

        # Alias language_model. prefixed keys (multimodal models)
        _LM_PREFIX = "language_model."
        aliased = {}
        for key, val in self._index.items():
            if key.startswith(_LM_PREFIX):
                aliased[key[len(_LM_PREFIX):]] = val
        self._index.update(aliased)

        logger.info(
            "[SAFETENSORS] Mapped %d shards, %d tensor keys",
            len(self._mmaps), len(self._index),
        )

    def get_tensor(self, key: str) -> mx.array:
        """Load a full tensor by key."""
        path, np_dt, mx_dt, shape, start, length = self._index[key]
        mm = self._mmaps[path]
        buf = mm[start:start + length]
        arr_np = np.frombuffer(buf, dtype=np_dt).reshape(shape)
        result = mx.array(arr_np)
        if mx_dt == mx.bfloat16:
            result = result.view(mx.bfloat16)
        return result

    def get_expert_slices(self, key: str, expert_ids) -> mx.array:
        """Load specific expert rows from a stacked (E, ...) tensor.

        Reads only the bytes for requested experts via mmap — no full tensor
        materialization.
        """
        path, np_dt, mx_dt, shape, start, length = self._index[key]
        mm = self._mmaps[path]
        ids = np.asarray(expert_ids).reshape(-1)
        row_bytes = length // shape[0]
        row_shape = shape[1:]

        parts = []
        for eid in ids:
            row_start = start + int(eid) * row_bytes
            buf = mm[row_start:row_start + row_bytes]
            parts.append(np.frombuffer(buf, dtype=np_dt).reshape(row_shape))

        stacked = np.stack(parts)
        result = mx.array(stacked)
        if mx_dt == mx.bfloat16:
            result = result.view(mx.bfloat16)
        return result

    def __contains__(self, key: str) -> bool:
        return key in self._index

    def __getitem__(self, key: str) -> mx.array:
        return self.get_tensor(key)

    def prefetch_experts(
        self,
        key_base: str,
        expert_ids: List[int],
        proj_names: tuple = _PROJ_NAMES,
    ) -> None:
        """Async prefetch expert byte ranges via F_RDADVISE.

        Issues read-ahead hints to macOS kernel so pages are warm
        in the page cache before the subsequent get_expert_slices call.
        Non-blocking, fire-and-forget. Safe to call anytime.
        """
        ssd_prefetch.prefetch_experts(
            self._index, self._fds, key_base, expert_ids, proj_names,
        )

    def close(self):
        ssd_prefetch.shutdown()
        for mm in self._mmaps.values():
            mm.close()
        for fd in self._fds.values():
            fd.close()
        self._mmaps.clear()
        self._fds.clear()
        self._index.clear()


# ── Model Inspection ──────────────────────────────────────────────────────────


def _find_switch_mlp(layer, layer_idx: Optional[int] = None):
    """Find the SwitchGLU module in a model layer.

    Returns (switch_mlp, key_prefix_base) or (None, None).
    Supports Qwen (layer.mlp.switch_mlp) and Mixtral (layer.block_sparse_moe.switch_mlp).
    """
    prefix = f"model.layers.{layer_idx}" if layer_idx is not None else None

    if hasattr(layer, "mlp") and hasattr(layer.mlp, "switch_mlp"):
        switch = layer.mlp.switch_mlp
        key_base = f"{prefix}.mlp.switch_mlp" if prefix else "mlp.switch_mlp"
        return switch, key_base

    if hasattr(layer, "block_sparse_moe") and hasattr(layer.block_sparse_moe, "switch_mlp"):
        switch = layer.block_sparse_moe.switch_mlp
        key_base = f"{prefix}.block_sparse_moe.switch_mlp" if prefix else "block_sparse_moe.switch_mlp"
        return switch, key_base

    return None, None


def _find_moe_block(layer):
    """Find the MoE block (parent of switch_mlp) in a layer."""
    if hasattr(layer, "mlp") and hasattr(layer.mlp, "switch_mlp"):
        return layer.mlp
    if hasattr(layer, "block_sparse_moe") and hasattr(layer.block_sparse_moe, "switch_mlp"):
        return layer.block_sparse_moe
    return None


def _detect_num_experts(switch_mlp) -> int:
    """Detect number of experts from a SwitchGLU module."""
    for name in _PROJ_NAMES:
        proj = getattr(switch_mlp, name, None)
        if proj is not None and hasattr(proj, "weight"):
            return proj.weight.shape[0]
    return 256


def is_moe_model(model: nn.Module) -> bool:
    """Check if a model has MoE layers (SwitchLinear modules)."""
    for layer in getattr(model, "layers", []):
        switch, _ = _find_switch_mlp(layer)
        if switch is not None:
            return True
    return False


def _detect_moe_config(model: nn.Module) -> Tuple[int, int, float]:
    """Detect MoE configuration.

    Returns (num_experts, num_moe_layers, expert_slot_mb).
    expert_slot_mb = size of one expert across all 3 projections.
    """
    num_experts = 0
    moe_layers = 0
    expert_slot_mb = 0.0

    for layer in getattr(model, "layers", []):
        switch, _ = _find_switch_mlp(layer)
        if switch is None:
            continue
        moe_layers += 1
        if num_experts == 0:
            num_experts = _detect_num_experts(switch)
        if expert_slot_mb == 0.0:
            proj = getattr(switch, "gate_proj", None) or getattr(switch, "up_proj", None)
            if proj is not None and hasattr(proj, "weight"):
                per_expert = proj.weight.nbytes // num_experts
                for attr in ("scales", "biases"):
                    t = getattr(proj, attr, None)
                    if t is not None:
                        per_expert += t.nbytes // num_experts
                expert_slot_mb = per_expert * 3 / 1e6

    return num_experts, moe_layers, expert_slot_mb


# ── PredictiveExpertCache ─────────────────────────────────────────────────────
# Per-layer cache with GPU-resident weight tensors and lookup table.


class PredictiveExpertCache:
    """Per-layer expert cache with GPU lookup table for zero-eval dispatch.

    Pre-loads a subset of experts into Metal memory at startup. Forward pass
    uses a lookup table to remap global expert IDs to cache slots entirely
    on GPU — no mx.eval needed. Uncached experts map to slot 0 (fallback).
    """
    __slots__ = (
        'capacity', 'full_capacity', 'num_experts', 'lookup', 'hit_mask',
        'weights', 'scales', 'biases',
        'cached_ids', 'cached_set',
        'frequency', 'session_frequency', 'historical_frequency',
        'last_active', 'step',
        '_indices_buffer',
        '_shard_paths', '_key_prefixes', '_shard_map',
        '_st_map',
        'total_requests', 'total_fallbacks',
        'pinned_set',
    )

    def __init__(self, capacity: int, num_experts: int = 256):
        self.capacity = capacity
        self.full_capacity = capacity  # Original capacity for breathe_up
        self.num_experts = num_experts
        self.weights: Dict[str, mx.array] = {}
        self.scales: Dict[str, mx.array] = {}
        self.biases: Dict[str, Optional[mx.array]] = {}
        self.lookup: Optional[mx.array] = None
        self.hit_mask: Optional[mx.array] = None
        self.cached_ids: List[int] = []
        self.cached_set: set = set()
        self.frequency: Dict[int, int] = {}           # Combined view (session + historical)
        self.session_frequency: Dict[int, int] = {}    # This session only
        self.historical_frequency: Dict[int, int] = {} # Loaded from disk
        self.last_active: Dict[int, int] = {}
        self.step: int = 0
        self._indices_buffer: List[mx.array] = []
        self._shard_paths: Dict[str, str] = {}
        self._key_prefixes: Dict[str, str] = {}
        self._shard_map: Optional[Dict[str, str]] = None
        self._st_map: Optional[SafetensorsMap] = None
        self.total_requests: int = 0
        self.total_fallbacks: int = 0
        self.pinned_set: set = set()

    def export_frequency(self) -> Dict[int, int]:
        """Return copy of session-only frequency counts (for persistence)."""
        return dict(self.session_frequency)

    def breathing_priority(self, eid: int) -> float:
        """Weighted priority score for breathing eviction decisions.

        Session usage dominates (3x) so current-task experts are protected.
        Historical baseline provides a floor (1x) for experts not yet used.
        Never-used-anywhere experts get 0 → evicted first.
        """
        return self.session_frequency.get(eid, 0) * 3 + self.historical_frequency.get(eid, 0) * 1

    def get_coverage_ratio(self) -> float:
        """Fraction of total routing traffic covered by currently cached experts.

        Uses combined frequency: sum(freq for cached) / sum(all freq).
        Returns 1.0 if no frequency data yet.
        """
        if not self.frequency:
            return 1.0
        total = sum(self.frequency.values())
        if total == 0:
            return 1.0
        covered = sum(self.frequency.get(eid, 0) for eid in self.cached_set)
        return covered / total



    def build_lookup(self, cached_ids: List[int]) -> None:
        """Build GPU-resident lookup table and hit mask from cached expert IDs.

        Uncached IDs map to slot 0 (fallback).
        """
        self.cached_ids = list(cached_ids)
        self.cached_set = set(cached_ids)
        for eid in cached_ids:
            self.frequency.setdefault(eid, 1)
            self.last_active.setdefault(eid, 0)
        self.rebuild_lookup()

    def rebuild_lookup(self) -> None:
        """Rebuild lookup table and hit mask from current cached_ids."""
        lookup_np = np.zeros(self.num_experts, dtype=np.int32)
        hit_np = np.zeros(self.num_experts, dtype=np.float32)
        for slot, eid in enumerate(self.cached_ids):
            lookup_np[eid] = slot
            hit_np[eid] = 1.0
        self.lookup = mx.array(lookup_np)
        self.hit_mask = mx.array(hit_np)

    def remap(self, indices: mx.array) -> mx.array:
        """Map global expert IDs to cache slots. Pure mx.array op, no eval."""
        return self.lookup[indices]

    def _lcp_priority(self, eid: int) -> float:
        """LCP (Locality-Count Priority) score for eviction decisions."""
        mu = self.frequency.get(eid, 0)
        nu = self.step - self.last_active.get(eid, 0)
        return mu * (0.25 ** (nu / 128))

    def update(self) -> Dict[str, int]:
        """Process buffered indices and swap cold experts for missed ones.

        Call between tokens. Skips the last buffered entry (in-flight due
        to async_eval double-buffering).

        Returns dict with swaps, fallbacks, requests counts.
        """
        if len(self._indices_buffer) < 2:
            return {"swaps": 0, "fallbacks": 0, "requests": 0}

        to_process = self._indices_buffer[:-1]
        self._indices_buffer = self._indices_buffer[-1:]

        all_requested: set = set()
        for indices in to_process:
            flat = np.asarray(indices.reshape(-1))
            unique = set(int(x) for x in np.unique(flat))
            all_requested |= unique

        self.step += 1
        for eid in all_requested:
            self.frequency[eid] = self.frequency.get(eid, 0) + 1
            self.session_frequency[eid] = self.session_frequency.get(eid, 0) + 1
            self.last_active[eid] = self.step

        misses = all_requested - self.cached_set
        n_requests = len(all_requested)
        n_fallbacks = len(misses)
        self.total_requests += n_requests
        self.total_fallbacks += n_fallbacks

        if not misses or not self._shard_paths:
            return {"swaps": 0, "fallbacks": n_fallbacks, "requests": n_requests}

        # Find eviction candidates (not active, not pinned), sorted by LCP priority
        evict_candidates = [
            (self._lcp_priority(eid), slot, eid)
            for slot, eid in enumerate(self.cached_ids)
            if eid not in all_requested and eid not in self.pinned_set
        ]
        evict_candidates.sort()

        swaps: List[Tuple[int, int, int]] = []
        for new_eid in sorted(misses):
            if not evict_candidates:
                break
            _, slot, old_eid = evict_candidates.pop(0)
            swaps.append((slot, old_eid, new_eid))

        swaps = swaps[:_MAX_SWAPS_PER_LAYER]

        if not swaps:
            return {"swaps": 0, "fallbacks": n_fallbacks, "requests": n_requests}

        # Prefetch byte ranges before loading — warms page cache via F_RDADVISE
        swap_eids = [new_eid for _, _, new_eid in swaps]
        if self._st_map is not None:
            for proj_name in _PROJ_NAMES:
                key_prefix = self._key_prefixes.get(proj_name)
                if key_prefix:
                    self._st_map.prefetch_experts(key_prefix, swap_eids, (proj_name,))

        # Load new experts and swap into stacked tensors
        new_eids = mx.array(swap_eids)
        slot_indices = mx.array([slot for slot, _, _ in swaps])
        for proj_name in _PROJ_NAMES:
            key_prefix = self._key_prefixes[proj_name]
            w_key = f"{key_prefix}.weight"
            s_key = f"{key_prefix}.scales"
            b_key = f"{key_prefix}.biases"

            # Load via mmap — pages pre-warmed by F_RDADVISE above
            new_w = self._st_map.get_expert_slices(w_key, new_eids)
            new_s = self._st_map.get_expert_slices(s_key, new_eids)
            new_b = self._st_map.get_expert_slices(b_key, new_eids) if b_key in self._st_map else None

            if new_b is None:
                mx.eval(new_w, new_s)
            else:
                mx.eval(new_w, new_s, new_b)

            # Swap in-place via pop/reassign (MLX array mutation pattern)
            w = self.weights.pop(proj_name)
            w[slot_indices] = new_w
            self.weights[proj_name] = w

            s = self.scales.pop(proj_name)
            s[slot_indices] = new_s
            self.scales[proj_name] = s

            if self.biases[proj_name] is not None and new_b is not None:
                b = self.biases.pop(proj_name)
                b[slot_indices] = new_b
                self.biases[proj_name] = b

        mx.clear_cache()

        # Update bookkeeping
        for slot, old_eid, new_eid in swaps:
            self.cached_set.discard(old_eid)
            self.cached_set.add(new_eid)
            self.cached_ids[slot] = new_eid
            self.frequency.pop(old_eid, None)
            self.last_active.pop(old_eid, None)

        self.rebuild_lookup()
        mx.eval(self.lookup, self.hit_mask)

        return {"swaps": len(swaps), "fallbacks": n_fallbacks, "requests": n_requests}


# ── PredictiveCachedSwitchLinear ──────────────────────────────────────────────
# Zero-eval forward pass using pre-stacked tensors and GPU lookup remap.


class PredictiveCachedSwitchLinear(nn.Module):
    """Zero-eval expert dispatch using pre-loaded weights and GPU lookup table.

    The forward pass stays entirely lazy — indices are remapped via a
    pre-built lookup table on GPU, and gather_qmm uses pre-loaded weight
    tensors already in Metal memory. No mx.eval until the output token.

    Only up_proj captures router indices for dynamic cache updates
    (SwitchGLU calls up_proj first).
    """

    def __init__(
        self,
        group_size: int,
        bits: int,
        mode: str,
        proj_name: str,
        cache: PredictiveExpertCache,
    ):
        super().__init__()
        self.group_size = group_size
        self.bits = bits
        self.mode = mode
        self._proj_name = proj_name
        self._cache = cache
        self.freeze()

    def __call__(self, x, indices, sorted_indices=False):
        # Only up_proj captures indices (avoid triple-buffering)
        if self._proj_name == "up_proj":
            self._cache._indices_buffer.append(indices)

        local_indices = self._cache.remap(indices)
        # Remap breaks sorted order — always pass sorted_indices=False
        return mx.gather_qmm(
            x,
            self._cache.weights[self._proj_name],
            self._cache.scales[self._proj_name],
            self._cache.biases[self._proj_name],
            rhs_indices=local_indices,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
        )


# ── Expert Profile ────────────────────────────────────────────────────────────


def load_expert_profile(path: str) -> Dict:
    """Load universal expert profile from JSON."""
    with open(path) as f:
        return json.load(f)


# ── Shard Map ─────────────────────────────────────────────────────────────────


def _build_shard_map(model_path: Path) -> Dict[str, str]:
    """Read model.safetensors.index.json and return {key: absolute_shard_path}."""
    index_path = model_path / "model.safetensors.index.json"
    if not index_path.exists():
        single = model_path / "model.safetensors"
        if single.exists():
            return {"__single__": str(single)}
        return {}

    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]

    shard_map = {key: str(model_path / shard) for key, shard in weight_map.items()}

    # Alias language_model. prefixed keys (multimodal models)
    _LM_PREFIX = "language_model."
    has_lm_prefix = any(k.startswith(_LM_PREFIX) for k in shard_map)
    if has_lm_prefix:
        aliased = {}
        for key, path in shard_map.items():
            if key.startswith(_LM_PREFIX):
                aliased[key[len(_LM_PREFIX):]] = path
        shard_map.update(aliased)

    return shard_map


# ── Skip-Fallback ─────────────────────────────────────────────────────────────
# Monkey-patches MoE blocks to zero out scores for uncached experts.


def enable_skip_fallback(model: nn.Module) -> int:
    """Patch MoE blocks to zero scores for missing experts and renormalize.

    Without this, missing experts fall back to cache slot 0 (wrong expert),
    injecting wrong-expert outputs weighted by the real router score.

    With skip-fallback, missing experts get score=0 and remaining hits are
    renormalized. The residual connection passes through unchanged input
    for the missing expert's contribution.

    Returns number of MoE blocks patched.
    """
    patched = 0
    for i, layer in enumerate(model.layers):
        moe_block = _find_moe_block(layer)
        if moe_block is None:
            continue
        switch = getattr(moe_block, "switch_mlp", None)
        if switch is None:
            continue
        proj = getattr(switch, "up_proj", None)
        if not isinstance(proj, PredictiveCachedSwitchLinear):
            continue

        cache = proj._cache
        _patch_moe_block(moe_block, cache)
        patched += 1

    if patched > 0:
        logger.info("[MOE] Skip-fallback enabled on %d MoE blocks", patched)
    return patched


def _gate_returns_tuple(moe_block) -> bool:
    """Check if the gate returns (inds, scores) tuple vs raw logits."""
    gate = moe_block.gate
    return not isinstance(gate, nn.Linear)


def _patch_moe_block(moe_block, cache: PredictiveExpertCache):
    """Replace MoE block __call__ with score-masking version."""
    gate_returns_tuple = _gate_returns_tuple(moe_block)

    def patched_call(self, x):
        if gate_returns_tuple:
            inds, scores = self.gate(x)
        else:
            gates = self.gate(x)
            k = getattr(self, "num_experts_per_tok", getattr(self, "top_k", 2))
            gates = mx.softmax(gates, axis=-1, precise=True)
            inds = mx.stop_gradient(mx.argpartition(-gates, kth=k - 1, axis=-1)[..., :k])
            scores = mx.take_along_axis(gates, inds, axis=-1)
            if getattr(self, "norm_topk_prob", False):
                scores = scores / scores.sum(axis=-1, keepdims=True)

        # Zero out scores for uncached experts
        mask = cache.hit_mask[inds]
        scores = scores * mask
        score_sum = scores.sum(axis=-1, keepdims=True)
        # Guard against div-by-zero when ALL top-K experts are uncached
        scores = mx.where(score_sum > 0, scores / score_sum, scores)

        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2)

        # Shared expert (Qwen3 MoE architecture)
        if hasattr(self, "shared_expert") and hasattr(self, "shared_expert_gate"):
            se = self.shared_expert(x)
            y = y + mx.sigmoid(self.shared_expert_gate(x)) * se
        elif hasattr(self, "shared_experts"):
            y = y + self.shared_experts(x)

        return y

    moe_block.__call__ = types.MethodType(patched_call, moe_block)


# ── Dynamic Cache Update ─────────────────────────────────────────────────────


def dynamic_cache_update(
    model: nn.Module,
    max_layer_updates: int = 12,
) -> List[Dict]:
    """Process buffered router indices and swap cold experts for missed ones.

    Call between tokens during generation. Handles async_eval double-buffering
    by skipping in-flight indices.

    Args:
        model: Model with PredictiveCachedSwitchLinear modules.
        max_layer_updates: Max layers to perform swaps on per call.

    Returns:
        Per-layer stats: [{"layer": i, "swaps": n, "fallbacks": n, "requests": n}]
    """
    stats = []
    swap_budget = max_layer_updates
    for i, layer in enumerate(model.layers):
        switch, _ = _find_switch_mlp(layer, i)
        if switch is None:
            continue
        proj = getattr(switch, "up_proj", None)
        if not isinstance(proj, PredictiveCachedSwitchLinear):
            continue
        cache = proj._cache

        if swap_budget > 0:
            layer_stats = cache.update()
            if layer_stats["swaps"] > 0:
                swap_budget -= 1
        else:
            # Track stats but defer swaps
            if len(cache._indices_buffer) < 2:
                layer_stats = {"swaps": 0, "fallbacks": 0, "requests": 0}
            else:
                to_process = cache._indices_buffer[:-1]
                cache._indices_buffer = cache._indices_buffer[-1:]
                all_requested: set = set()
                for indices in to_process:
                    flat = np.asarray(indices.reshape(-1))
                    all_requested |= set(int(x) for x in np.unique(flat))
                cache.step += 1
                for eid in all_requested:
                    cache.frequency[eid] = cache.frequency.get(eid, 0) + 1
                    cache.last_active[eid] = cache.step
                misses = all_requested - cache.cached_set
                cache.total_requests += len(all_requested)
                cache.total_fallbacks += len(misses)
                layer_stats = {"swaps": 0, "fallbacks": len(misses), "requests": len(all_requested)}

        layer_stats["layer"] = i
        stats.append(layer_stats)
    return stats


def dynamic_update_policy(
    gen_tokens: int,
    last_fallback_rate: float,
    no_swap_streak: int,
    low_fallback_streak: int,
) -> Tuple[int, int]:
    """Adaptive dynamic update interval and budget.

    Returns (interval, budget) where:
    - interval: call dynamic_cache_update every N tokens
    - budget: max_layer_updates per call
    """
    if gen_tokens <= 5:
        interval, budget = 1, 48
    elif gen_tokens <= 20:
        interval, budget = 2, 24
    else:
        interval, budget = 4, 12

    if no_swap_streak >= 3:
        interval = max(interval, 6)

    if low_fallback_streak >= 6 and last_fallback_rate < 0.08:
        interval = max(interval, 12)
        budget = 8

    if low_fallback_streak >= 12 and last_fallback_rate < 0.05:
        interval = max(interval, 20)
        budget = 8

    return interval, budget


# ── Main Integration ──────────────────────────────────────────────────────────


def enable_moe_cache(
    model: nn.Module,
    model_path: str,
    capacity: int = 100,
    profile_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Enable MoE expert caching on a lazily-loaded model.

    This is the main entry point. Call after mlx_lm.load(lazy=True).
    Do NOT call mx.eval(model.parameters()) before this — it would
    materialize all 19.5 GB of expert weights.

    Flow:
    1. Detect MoE config
    2. Build SafetensorsMap (manual mmap, bfloat16-safe)
    3. Pass 1: Replace QuantizedSwitchLinear with PredictiveCachedSwitchLinear
       (removes expert weights from model parameter tree)
    4. mx.eval(model.parameters()) → only non-expert params (~1.4 GB)
    5. Pass 2: Load expert weights via byte-level mmap into caches
    6. Enable skip-fallback + wire memory

    Args:
        model: Model loaded with mlx_lm.load(lazy=True). NOT yet eval'd.
        model_path: HuggingFace model path or local directory.
        capacity: Max experts per layer (default 100 for 24GB).
        profile_path: Path to expert profile JSON.

    Returns:
        Dict with stats.
    """
    t0 = time.time()

    # Detect MoE config
    num_experts, moe_layers, expert_slot_mb = _detect_moe_config(model)
    if moe_layers == 0:
        logger.info("[MOE] No MoE layers found — skipping.")
        return {"moe_layers": 0}

    effective_cap = min(capacity, num_experts)

    # Resolve model path
    from mlx_lm.utils import hf_repo_to_path
    resolved_path = Path(hf_repo_to_path(model_path))

    # Build shard map and SafetensorsMap (manual parsing, bfloat16-safe)
    shard_map = _build_shard_map(resolved_path)
    all_shards = sorted(set(shard_map.values()))
    st_map = SafetensorsMap(all_shards)

    # Load profile
    profile = None
    if profile_path and Path(profile_path).exists():
        profile = load_expert_profile(profile_path)
        logger.info(
            "[MOE] Profile loaded: %d prompts",
            profile.get("num_prompts", 0),
        )

    logger.info(
        "[MOE] Config: %d layers, %d experts, cap=%d, slot=%.2f MB",
        moe_layers, num_experts, effective_cap, expert_slot_mb,
    )

    from mlx_lm.models.switch_layers import QuantizedSwitchLinear

    # ── Pass 1: Replace modules ──────────────────────────────────────────
    # Install PredictiveCachedSwitchLinear with EMPTY caches.
    # This removes QuantizedSwitchLinear (and its 256-expert weight tensors)
    # from the model parameter tree, so the subsequent mx.eval only
    # materializes non-expert params (~1.4 GB instead of 19.5 GB).
    replaced = 0
    layer_caches: Dict[int, Tuple[PredictiveExpertCache, List[int]]] = {}

    for i, layer in enumerate(model.layers):
        switch, key_base = _find_switch_mlp(layer, i)
        if switch is None:
            continue

        cached_ids = _select_experts_for_layer(
            i, num_experts, effective_cap, profile,
        )
        pred_cache = PredictiveExpertCache(effective_cap, num_experts)
        pred_cache._shard_map = shard_map
        pred_cache._st_map = st_map

        # Extract quant params from originals BEFORE replacing them
        quant_params = {}
        for proj_name in _PROJ_NAMES:
            orig = getattr(switch, proj_name)
            if isinstance(orig, QuantizedSwitchLinear):
                quant_params[proj_name] = {
                    "group_size": orig.group_size,
                    "bits": orig.bits,
                    "mode": orig.mode,
                }
                # Store key prefixes for dynamic updates
                key_prefix = f"{key_base}.{proj_name}"
                pred_cache._shard_paths[proj_name] = shard_map.get(f"{key_prefix}.weight", "")
                pred_cache._key_prefixes[proj_name] = key_prefix

        # Replace modules (drops original weight tensors from model params)
        for proj_name, qp in quant_params.items():
            replacement = PredictiveCachedSwitchLinear(
                group_size=qp["group_size"],
                bits=qp["bits"],
                mode=qp["mode"],
                proj_name=proj_name,
                cache=pred_cache,
            )
            setattr(switch, proj_name, replacement)
            replaced += 1

        layer_caches[i] = (pred_cache, cached_ids)

    # ── Materialize non-expert params ────────────────────────────────────
    # Now that QuantizedSwitchLinear modules are gone, model.parameters()
    # only contains attention, embeddings, norms, gate, shared_expert.
    mx.eval(model.parameters())
    non_expert_gb = mx.get_active_memory() / 1e9
    logger.info("[MOE] Non-expert params materialized: %.1f GB", non_expert_gb)

    # ── Pass 2: Load expert weights ──────────────────────────────────────
    # Load from SafetensorsMap via byte-level mmap slicing into caches.
    # F_RDADVISE prefetch: while loading layer N, prefetch layer N+1's
    # byte ranges so pages are warm when we get there.
    total_expert_tensors = 0
    layer_indices = list(layer_caches.keys())
    for idx, i in enumerate(layer_indices):
        pred_cache, cached_ids = layer_caches[i]
        expert_ids_arr = np.array(cached_ids, dtype=np.int32)

        # Prefetch NEXT layer's byte ranges while this layer loads
        if idx + 1 < len(layer_indices):
            next_i = layer_indices[idx + 1]
            next_cache, next_ids = layer_caches[next_i]
            for proj_name in _PROJ_NAMES:
                kp = next_cache._key_prefixes.get(proj_name)
                if kp:
                    st_map.prefetch_experts(kp, next_ids, (proj_name,))

        for proj_name in _PROJ_NAMES:
            key_prefix = pred_cache._key_prefixes.get(proj_name)
            if key_prefix is None:
                continue
            w_key = f"{key_prefix}.weight"
            s_key = f"{key_prefix}.scales"
            b_key = f"{key_prefix}.biases"

            w = st_map.get_expert_slices(w_key, expert_ids_arr)
            s = st_map.get_expert_slices(s_key, expert_ids_arr)
            b = st_map.get_expert_slices(b_key, expert_ids_arr) if b_key in st_map else None

            pred_cache.weights[proj_name] = w
            pred_cache.scales[proj_name] = s
            pred_cache.biases[proj_name] = b
            total_expert_tensors += len(cached_ids)

        # Eval per-layer to control peak memory
        to_eval = []
        for proj_name in _PROJ_NAMES:
            if proj_name in pred_cache.weights:
                to_eval.extend([pred_cache.weights[proj_name], pred_cache.scales[proj_name]])
                if pred_cache.biases.get(proj_name) is not None:
                    to_eval.append(pred_cache.biases[proj_name])
        if to_eval:
            mx.eval(*to_eval)

        # Build lookup + hit mask
        pred_cache.build_lookup(cached_ids)
        mx.eval(pred_cache.lookup, pred_cache.hit_mask)

        # Pin universal experts from profile
        if profile:
            _pin_from_profile(pred_cache, i, profile)

    # Enable skip-fallback on MoE blocks
    patched = enable_skip_fallback(model)

    # Wire memory to prevent OS paging.
    # Reserve PREFILL_SCRATCH_GB for large-context prefill scratch;
    # wiring too aggressively causes Metal OOM during mx.eval() on long prompts.
    if hasattr(mx, "set_wired_limit"):
        PREFILL_SCRATCH_RESERVE_GB = 8.0
        active = mx.get_active_memory()
        metal_total = mx.device_info()["memory_size"]
        headroom = int(metal_total - PREFILL_SCRATCH_RESERVE_GB * 1e9)
        wired = min(active, headroom)
        mx.set_wired_limit(wired)
        logger.info(
            "[MOE] Wired %.1f GB in Metal residency set (%.1f GB reserved for prefill)",
            wired / 1e9, PREFILL_SCRATCH_RESERVE_GB,
        )

    elapsed = time.time() - t0
    active_gb = mx.get_active_memory() / 1e9

    prefetch_stats = ssd_prefetch.get_stats()
    stats = {
        "moe_layers": moe_layers,
        "num_experts": num_experts,
        "expert_slot_mb": round(expert_slot_mb, 2),
        "capacity": effective_cap,
        "replaced_modules": replaced,
        "skip_fallback_patches": patched,
        "expert_tensors_loaded": total_expert_tensors,
        "elapsed_seconds": round(elapsed, 1),
        "active_memory_gb": round(active_gb, 1),
        "prefetch_issued": prefetch_stats["issued"],
        "prefetch_failed": prefetch_stats["failed"],
    }

    logger.info(
        "[MOE] ✅ Enabled | layers=%d experts=%d cap=%d "
        "replaced=%d loaded=%d patched=%d time=%.1fs mem=%.1fGB",
        moe_layers, num_experts, effective_cap,
        replaced, total_expert_tensors, patched, elapsed, active_gb,
    )

    # Store on model for later access
    model._st_map = st_map
    model._moe_config = stats

    return stats


def expand_expert_capacity(
    model: nn.Module,
    target_capacity: int,
    profile_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Expand expert cache from initial_capacity to target_capacity.

    Two-stage loading: startup uses low capacity (safe for cold prefill),
    then this function expands after the KV cache is warm (only tiny suffix
    prefills needed, so more experts fit in the scratch headroom).

    Operates in-place on existing PredictiveExpertCache instances:
    1. Creates new larger stacked tensors
    2. Copies existing loaded experts
    3. Loads additional experts from SSD
    4. Rebuilds lookup tables

    Must be called under model_lock.
    """
    t0 = time.time()

    config = getattr(model, "_moe_config", None)
    st_map = getattr(model, "_st_map", None)
    if config is None or st_map is None:
        logger.warning("[MOE] expand_expert_capacity: no MoE config found")
        return {"expanded": False}

    num_experts = config["num_experts"]
    old_capacity = config["capacity"]
    effective_target = min(target_capacity, num_experts)

    if effective_target <= old_capacity:
        logger.info("[MOE] expand: target %d <= current %d, skip", effective_target, old_capacity)
        return {"expanded": False, "reason": "target <= current"}

    profile = None
    if profile_path and Path(profile_path).exists():
        profile = load_expert_profile(profile_path)

    expanded_layers = 0
    total_new_experts = 0

    for i, layer in enumerate(model.layers):
        switch, _ = _find_switch_mlp(layer, i)
        if switch is None:
            continue
        proj = getattr(switch, "up_proj", None)
        if not isinstance(proj, PredictiveCachedSwitchLinear):
            continue

        cache = proj._cache
        old_ids = list(cache.cached_ids)
        old_set = set(old_ids)

        # Select experts for expanded capacity
        target_ids = _select_experts_for_layer(i, num_experts, effective_target, profile)
        new_ids = [eid for eid in target_ids if eid not in old_set]
        new_ids = new_ids[:effective_target - old_capacity]

        if not new_ids:
            continue

        new_eids_arr = np.array(new_ids, dtype=np.int32)

        # Prefetch byte ranges before loading from SSD
        for proj_name in _PROJ_NAMES:
            kp = cache._key_prefixes.get(proj_name)
            if kp:
                st_map.prefetch_experts(kp, new_ids, (proj_name,))

        # Expand stacked tensors for each projection
        for proj_name in _PROJ_NAMES:
            old_w = cache.weights.get(proj_name)
            old_s = cache.scales.get(proj_name)
            old_b = cache.biases.get(proj_name)
            if old_w is None:
                continue

            key_prefix = cache._key_prefixes.get(proj_name)
            if key_prefix is None:
                continue

            # Load new expert weights from SSD
            w_key = f"{key_prefix}.weight"
            s_key = f"{key_prefix}.scales"
            b_key = f"{key_prefix}.biases"

            new_w = st_map.get_expert_slices(w_key, new_eids_arr)
            new_s = st_map.get_expert_slices(s_key, new_eids_arr)
            new_b = st_map.get_expert_slices(b_key, new_eids_arr) if b_key in st_map else None

            if new_b is None:
                mx.eval(new_w, new_s)
            else:
                mx.eval(new_w, new_s, new_b)

            # Concatenate old + new into expanded stacked tensor
            cache.weights[proj_name] = mx.concatenate([old_w, new_w], axis=0)
            cache.scales[proj_name] = mx.concatenate([old_s, new_s], axis=0)
            if old_b is not None and new_b is not None:
                cache.biases[proj_name] = mx.concatenate([old_b, new_b], axis=0)

        # Eval the expanded tensors
        to_eval = []
        for proj_name in _PROJ_NAMES:
            if proj_name in cache.weights:
                to_eval.extend([cache.weights[proj_name], cache.scales[proj_name]])
                if cache.biases.get(proj_name) is not None:
                    to_eval.append(cache.biases[proj_name])
        if to_eval:
            mx.eval(*to_eval)

        # Update bookkeeping
        all_ids = old_ids + new_ids
        cache.capacity = len(all_ids)
        cache.cached_ids = all_ids
        cache.cached_set = set(all_ids)
        for eid in new_ids:
            cache.frequency.setdefault(eid, 1)
            cache.last_active.setdefault(eid, 0)

        # Rebuild lookup + hit mask
        cache.rebuild_lookup()
        mx.eval(cache.lookup, cache.hit_mask)

        # Re-pin from profile
        if profile:
            _pin_from_profile(cache, i, profile)

        expanded_layers += 1
        total_new_experts += len(new_ids)

    mx.clear_cache()

    # Re-wire memory with new footprint
    if hasattr(mx, "set_wired_limit"):
        PREFILL_SCRATCH_RESERVE_GB = 4.0  # Less reserve needed with warm cache
        active = mx.get_active_memory()
        metal_total = mx.device_info()["memory_size"]
        headroom = int(metal_total - PREFILL_SCRATCH_RESERVE_GB * 1e9)
        wired = min(active, headroom)
        mx.set_wired_limit(wired)

    elapsed = time.time() - t0
    active_gb = mx.get_active_memory() / 1e9

    # Update model config
    model._moe_config["capacity"] = effective_target
    model._moe_config["active_memory_gb"] = round(active_gb, 1)

    logger.info(
        "[MOE] ✅ Expanded: %d→%d cap | +%d experts across %d layers | "
        "%.1fs | mem=%.1fGB",
        old_capacity, effective_target, total_new_experts,
        expanded_layers, elapsed, active_gb,
    )

    return {
        "expanded": True,
        "old_capacity": old_capacity,
        "new_capacity": effective_target,
        "new_experts_loaded": total_new_experts,
        "layers_expanded": expanded_layers,
        "elapsed_seconds": round(elapsed, 1),
        "active_memory_gb": round(active_gb, 1),
    }


def _select_experts_for_layer(
    layer_idx: int,
    num_experts: int,
    capacity: int,
    profile: Optional[Dict],
) -> List[int]:
    """Select which experts to cache for a layer.

    If profile exists, uses activation counts (top-C by frequency).
    Otherwise fills with experts 0..C-1.
    """
    if profile is None:
        return list(range(min(capacity, num_experts)))

    layer_data = profile.get("layers", {}).get(str(layer_idx))
    if layer_data is None:
        return list(range(min(capacity, num_experts)))

    counts = layer_data.get("activation_counts", {})
    sorted_experts = sorted(
        ((int(eid), int(cnt)) for eid, cnt in counts.items()),
        key=lambda x: (-x[1], x[0]),
    )

    # Top-C experts from profile
    selected = [eid for eid, _ in sorted_experts[:capacity]]

    # Fill remaining slots with sequential experts not already selected
    if len(selected) < capacity:
        selected_set = set(selected)
        for eid in range(num_experts):
            if len(selected) >= capacity:
                break
            if eid not in selected_set:
                selected.append(eid)

    return selected


def _pin_from_profile(
    cache: PredictiveExpertCache,
    layer_idx: int,
    profile: Dict,
    threshold: float = 0.5,
) -> None:
    """Pin universal experts (activated in >threshold fraction of prompts)."""
    num_prompts = profile.get("num_prompts", 1)
    min_count = int(threshold * num_prompts)

    layer_data = profile.get("layers", {}).get(str(layer_idx))
    if layer_data is None:
        return

    counts = layer_data.get("activation_counts", {})
    pinned = set(
        int(eid) for eid, cnt in counts.items()
        if int(cnt) >= min_count
    )
    cache.pinned_set = pinned & cache.cached_set


# ── Stats / Diagnostics ──────────────────────────────────────────────────────


def get_cache_stats(model: nn.Module) -> Dict[str, Any]:
    """Get aggregate fallback stats for all MoE layers."""
    total_requests = 0
    total_fallbacks = 0

    for i, layer in enumerate(model.layers):
        switch, _ = _find_switch_mlp(layer, i)
        if switch is None:
            continue
        proj = getattr(switch, "up_proj", None)
        if not isinstance(proj, PredictiveCachedSwitchLinear):
            continue
        cache = proj._cache
        total_requests += cache.total_requests
        total_fallbacks += cache.total_fallbacks

    if total_requests == 0:
        return {"moe_active": False}

    fallback_rate = total_fallbacks / total_requests
    return {
        "moe_active": True,
        "total_requests": total_requests,
        "total_fallbacks": total_fallbacks,
        "hit_rate": round(1 - fallback_rate, 4),
        "fallback_rate": round(fallback_rate, 4),
    }


def breathe_down(
    model: nn.Module,
    target_count: int,
    log_fn: Optional[Any] = None,
) -> Dict[str, Any]:
    """Contract expert cache to target_count per layer, freeing GPU memory.

    Evicts least-used experts based on frequency stats. Pinned experts are
    protected. Rebuilds stacked tensors in-place — the old larger tensors
    are released by mx.clear_cache().

    Must be called under model_lock.

    Args:
        model: Model with PredictiveCachedSwitchLinear modules.
        target_count: Target experts per layer (e.g. 100).
        log_fn: Optional terminal status logger.

    Returns:
        Dict with before/after counts, freed memory, coverage ratio.
    """
    t0 = time.time()
    before_mem = mx.get_active_memory() / 1e9

    config = getattr(model, "_moe_config", None)
    if config is None:
        return {"breathed": False, "reason": "no_moe_config"}

    num_experts = config["num_experts"]
    old_capacity = config["capacity"]
    target_count = max(target_count, 1)

    if target_count >= old_capacity:
        return {"breathed": False, "reason": "target >= current"}

    layers_contracted = 0
    total_evicted = 0
    avg_coverage = 0.0

    for i, layer in enumerate(model.layers):
        switch, _ = _find_switch_mlp(layer, i)
        if switch is None:
            continue
        proj = getattr(switch, "up_proj", None)
        if not isinstance(proj, PredictiveCachedSwitchLinear):
            continue

        cache = proj._cache

        if len(cache.cached_ids) <= target_count:
            avg_coverage += cache.get_coverage_ratio()
            continue

        # Sort by breathing_priority descending; pinned experts always retained
        scored = []
        for slot, eid in enumerate(cache.cached_ids):
            if eid in cache.pinned_set:
                scored.append((float('inf'), slot, eid))
            else:
                scored.append((cache.breathing_priority(eid), slot, eid))
        scored.sort(key=lambda x: -x[0])

        # Keep top target_count
        keep = scored[:target_count]
        keep_slots = sorted([slot for _, slot, _ in keep])
        keep_ids = [cache.cached_ids[s] for s in keep_slots]
        evicted = len(cache.cached_ids) - len(keep_ids)

        # Rebuild stacked tensors with only kept slots
        keep_indices = mx.array(keep_slots)
        for proj_name in _PROJ_NAMES:
            old_w = cache.weights.get(proj_name)
            old_s = cache.scales.get(proj_name)
            old_b = cache.biases.get(proj_name)
            if old_w is None:
                continue

            cache.weights[proj_name] = old_w[keep_indices]
            cache.scales[proj_name] = old_s[keep_indices]
            if old_b is not None:
                cache.biases[proj_name] = old_b[keep_indices]

        # Eval the sliced tensors to materialize before clearing old ones
        to_eval = []
        for proj_name in _PROJ_NAMES:
            if proj_name in cache.weights:
                to_eval.extend([cache.weights[proj_name], cache.scales[proj_name]])
                if cache.biases.get(proj_name) is not None:
                    to_eval.append(cache.biases[proj_name])
        if to_eval:
            mx.eval(*to_eval)

        # Update bookkeeping
        cache.cached_ids = keep_ids
        cache.cached_set = set(keep_ids)
        cache.capacity = len(keep_ids)
        cache.rebuild_lookup()
        mx.eval(cache.lookup, cache.hit_mask)

        layers_contracted += 1
        total_evicted += evicted
        avg_coverage += cache.get_coverage_ratio()

    mx.clear_cache()
    import gc; gc.collect()

    after_mem = mx.get_active_memory() / 1e9
    freed = before_mem - after_mem
    moe_layers = config["moe_layers"]
    avg_coverage = avg_coverage / max(moe_layers, 1)

    # Update model config
    config["capacity"] = target_count

    elapsed = time.time() - t0

    if log_fn:
        log_fn("🫁",
            f"BREATHE ↓ experts={old_capacity}→{target_count} "
            f"(evicted {total_evicted}) | freed={freed:.1f}GB | "
            f"headroom={_get_headroom_gb():.1f}GB | "
            f"coverage={avg_coverage:.1%} | {elapsed:.1f}s")

    return {
        "breathed": True,
        "direction": "down",
        "before": old_capacity,
        "after": target_count,
        "evicted": total_evicted,
        "freed_gb": round(freed, 2),
        "coverage": round(avg_coverage, 4),
        "elapsed_s": round(elapsed, 1),
    }


def breathe_up(
    model: nn.Module,
    log_fn: Optional[Any] = None,
) -> Dict[str, Any]:
    """Expand expert cache back to full capacity from SSD.

    Reloads evicted experts using SafetensorsMap byte-level mmap.
    Must be called under model_lock after generation completes.

    Args:
        model: Model with PredictiveCachedSwitchLinear modules.
        log_fn: Optional terminal status logger.

    Returns:
        Dict with reload counts, time, memory.
    """
    t0 = time.time()

    config = getattr(model, "_moe_config", None)
    st_map = getattr(model, "_st_map", None)
    if config is None or st_map is None:
        return {"breathed": False, "reason": "no_moe_config"}

    current_capacity = config["capacity"]
    # Use the full_capacity stored on each cache, fallback to num_experts
    target_capacity = None  # Will detect from first cache

    total_loaded = 0
    layers_expanded = 0

    for i, layer in enumerate(model.layers):
        switch, _ = _find_switch_mlp(layer, i)
        if switch is None:
            continue
        proj = getattr(switch, "up_proj", None)
        if not isinstance(proj, PredictiveCachedSwitchLinear):
            continue

        cache = proj._cache
        if target_capacity is None:
            target_capacity = cache.full_capacity

        if len(cache.cached_ids) >= target_capacity:
            continue

        old_ids = list(cache.cached_ids)
        old_set = set(old_ids)

        # Select experts to reload — use frequency-ordered selection
        # to prioritize historically useful experts
        all_candidates = []
        for eid in range(cache.num_experts):
            if eid not in old_set:
                freq = cache.frequency.get(eid, 0)
                all_candidates.append((freq, eid))
        all_candidates.sort(key=lambda x: -x[0])

        needed = target_capacity - len(old_ids)
        new_ids = [eid for _, eid in all_candidates[:needed]]

        if not new_ids:
            continue

        new_eids_arr = np.array(new_ids, dtype=np.int32)

        # Prefetch byte ranges
        for proj_name in _PROJ_NAMES:
            kp = cache._key_prefixes.get(proj_name)
            if kp:
                st_map.prefetch_experts(kp, new_ids, (proj_name,))

        # Load and concatenate
        for proj_name in _PROJ_NAMES:
            key_prefix = cache._key_prefixes.get(proj_name)
            if key_prefix is None:
                continue

            w_key = f"{key_prefix}.weight"
            s_key = f"{key_prefix}.scales"
            b_key = f"{key_prefix}.biases"

            new_w = st_map.get_expert_slices(w_key, new_eids_arr)
            new_s = st_map.get_expert_slices(s_key, new_eids_arr)
            new_b = st_map.get_expert_slices(b_key, new_eids_arr) if b_key in st_map else None

            if new_b is None:
                mx.eval(new_w, new_s)
            else:
                mx.eval(new_w, new_s, new_b)

            old_w = cache.weights.pop(proj_name)
            cache.weights[proj_name] = mx.concatenate([old_w, new_w], axis=0)

            old_s = cache.scales.pop(proj_name)
            cache.scales[proj_name] = mx.concatenate([old_s, new_s], axis=0)

            if cache.biases.get(proj_name) is not None and new_b is not None:
                old_b = cache.biases.pop(proj_name)
                cache.biases[proj_name] = mx.concatenate([old_b, new_b], axis=0)

        # Eval concatenated tensors
        to_eval = []
        for proj_name in _PROJ_NAMES:
            if proj_name in cache.weights:
                to_eval.extend([cache.weights[proj_name], cache.scales[proj_name]])
                if cache.biases.get(proj_name) is not None:
                    to_eval.append(cache.biases[proj_name])
        if to_eval:
            mx.eval(*to_eval)

        # Update bookkeeping
        all_ids = old_ids + new_ids
        cache.capacity = len(all_ids)
        cache.cached_ids = all_ids
        cache.cached_set = set(all_ids)
        for eid in new_ids:
            cache.frequency.setdefault(eid, 1)
            cache.last_active.setdefault(eid, 0)

        cache.rebuild_lookup()
        mx.eval(cache.lookup, cache.hit_mask)

        layers_expanded += 1
        total_loaded += len(new_ids)

    mx.clear_cache()

    # Re-wire memory
    if hasattr(mx, "set_wired_limit"):
        PREFILL_SCRATCH_RESERVE_GB = 4.0
        active = mx.get_active_memory()
        metal_total = mx.device_info()["memory_size"]
        headroom = int(metal_total - PREFILL_SCRATCH_RESERVE_GB * 1e9)
        wired = min(active, headroom)
        mx.set_wired_limit(wired)

    elapsed = time.time() - t0
    active_gb = mx.get_active_memory() / 1e9

    if target_capacity is not None:
        config["capacity"] = target_capacity

    if log_fn:
        log_fn("🫁",
            f"BREATHE ↑ experts={current_capacity}→{target_capacity or current_capacity} "
            f"(+{total_loaded} reloaded) | {elapsed:.1f}s from SSD | "
            f"mem={active_gb:.1f}GB | ready=full")

    return {
        "breathed": True,
        "direction": "up",
        "before": current_capacity,
        "after": target_capacity or current_capacity,
        "loaded": total_loaded,
        "elapsed_s": round(elapsed, 1),
        "active_memory_gb": round(active_gb, 1),
    }


def _get_headroom_gb() -> float:
    """Current GPU headroom in GB."""
    try:
        active = mx.get_active_memory() / 1e9
        total = mx.device_info()["memory_size"] / 1e9
        return total - active - 2.0  # 2GB OS reserve
    except Exception:
        return 0.0


# ── Frequency Stats Persistence ──────────────────────────────────────────────


def save_frequency_stats(
    model: nn.Module,
    path: str,
    merge_existing: bool = True,
) -> None:
    """Persist per-layer expert session frequency stats to JSON.

    Saves session-only counts merged additively with existing historical file.
    This way the file accumulates across sessions without double-counting.
    """
    from pathlib import Path

    session_stats: Dict[str, Dict[str, int]] = {}

    for i, layer in enumerate(model.layers):
        switch, _ = _find_switch_mlp(layer, i)
        if switch is None:
            continue
        proj = getattr(switch, "up_proj", None)
        if not isinstance(proj, PredictiveCachedSwitchLinear):
            continue
        cache = proj._cache
        if cache.session_frequency:
            session_stats[str(i)] = {str(eid): cnt for eid, cnt in cache.session_frequency.items()}

    if not session_stats:
        return

    # Merge session counts into existing historical file
    if merge_existing:
        existing = load_frequency_stats(path)
        if existing:
            for layer_key, freqs in session_stats.items():
                if layer_key in existing:
                    for eid, cnt in freqs.items():
                        existing[layer_key][eid] = existing[layer_key].get(eid, 0) + cnt
                else:
                    existing[layer_key] = freqs
            session_stats = existing

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump({"version": 1, "layers": session_stats}, f, indent=1)

    logger.info("[MOE] Frequency stats saved: %s (%d layers)", path, len(session_stats))


def load_frequency_stats(path: str) -> Optional[Dict[str, Dict[str, int]]]:
    """Load frequency stats from JSON. Returns None if file doesn't exist."""
    from pathlib import Path
    p = Path(path)
    if not p.exists():
        return None
    try:
        with open(p) as f:
            data = json.load(f)
        return data.get("layers", {})
    except Exception as e:
        logger.warning("[MOE] Failed to load frequency stats: %s", e)
        return None


def apply_historical_frequency(
    model: nn.Module,
    path: str,
) -> int:
    """Load historical frequency stats into live caches.

    Populates historical_frequency (baseline) and adds to frequency
    (combined view). Does NOT touch session_frequency.
    Call at startup before any generation.
    Returns number of layers seeded.
    """
    historical = load_frequency_stats(path)
    if not historical:
        return 0

    seeded = 0
    for i, layer in enumerate(model.layers):
        switch, _ = _find_switch_mlp(layer, i)
        if switch is None:
            continue
        proj = getattr(switch, "up_proj", None)
        if not isinstance(proj, PredictiveCachedSwitchLinear):
            continue

        cache = proj._cache
        layer_data = historical.get(str(i))
        if not layer_data:
            continue

        for eid_str, cnt in layer_data.items():
            eid = int(eid_str)
            cache.historical_frequency[eid] = cnt
            cache.frequency[eid] = cache.frequency.get(eid, 0) + cnt
        seeded += 1

    if seeded > 0:
        logger.info("[MOE] Historical frequency applied: %d layers from %s", seeded, path)
    return seeded

