"""
cache_engine.py — Hybrid-aware prompt cache for SuperMLX.

Extends mlx-lm's LRUPromptCache with:

1. fetch_nearest_cache: handles "longer" matches on hybrid models
   (ArraysCache + KVCache) by doing per-layer KVCache trim when
   can_trim_prompt_cache returns False. Returns FetchResult diagnostics.

2. strip_response_tokens: trims generated response tokens from a
   cache before re-insertion, preventing contamination.

3. snapshot_arrays_cache / rollback_arrays_cache: snapshot/rollback
   of ArraysCache layers for ephemeral large content (PDF, tool-result).
   KVCache layers are trimmed; ArraysCache layers are restored from snapshot.

   TRIGGER CONTRACT: caller (server.py) is responsible for calling
   snapshot_arrays_cache BEFORE prefilling tokens > SNAPSHOT_THRESHOLD,
   and rollback_arrays_cache if next request's matched_prefix <
   snapshot_position. Decision logic lives in server.py (same layer
   as Memory Guard), NOT in this module.
"""

import copy
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
from mlx_lm.models.cache import (
    LRUPromptCache,
    can_trim_prompt_cache,
    trim_prompt_cache,
)


# ── Diagnostics ──────────────────────────────────────────────────────────────

class HitType(Enum):
    EXACT = "exact"
    SHORTER = "shorter"
    LONGER_TRIMMED = "longer_trimmed"
    MISS = "miss"


@dataclass
class FetchResult:
    """Diagnostics from fetch_nearest_cache."""
    hit_type: HitType
    matched_prefix_len: int = 0
    trimmed_tokens: int = 0


# ── Hybrid trim helpers ──────────────────────────────────────────────────────

def _is_arrays_cache(layer: Any) -> bool:
    """Check if a cache layer is an ArraysCache (recurrent state, not KVCache)."""
    # NOTE: Cannot use `not layer.is_trimmable()` — the Marconi monkey-patch
    # makes is_trimmable() return True for ArraysCache.
    try:
        from mlx_lm.models.cache import ArraysCache
        return isinstance(layer, ArraysCache)
    except ImportError:
        # Fallback: duck-typing (has .cache list but no .offset)
        return (
            hasattr(layer, "cache")
            and isinstance(layer.cache, list)
            and not hasattr(layer, "offset")
        )


def _trim_hybrid(cache: list, n: int) -> bool:
    """Trim n tokens from KVCache layers only (skip ArraysCache).

    Returns True if at least one layer was trimmed.
    """
    trimmed = 0
    for layer in cache:
        if (
            hasattr(layer, "is_trimmable")
            and layer.is_trimmable()
            and hasattr(layer, "trim")
        ):
            layer.trim(n)
            trimmed += 1
    return trimmed > 0


def strip_response_tokens(
    cache: list, n: int
) -> bool:
    """Strip n response tokens from cache (in-place).

    Handles both pure KVCache and hybrid (ArraysCache + KVCache) models.
    Returns True if trim succeeded.
    """
    if n <= 0:
        return True
    if can_trim_prompt_cache(cache):
        trim_prompt_cache(cache, n)
        return True
    return _trim_hybrid(cache, n)


# ── ArraysCache Snapshot/Rollback ────────────────────────────────────────────

def snapshot_arrays_cache(cache: list) -> Dict[str, Any]:
    """Snapshot the state of all ArraysCache layers for later rollback.

    Creates a REAL copy of each mx.array in ArraysCache layers using
    mx.array() + mx.eval() to force materialization. This prevents
    the lazy-reference problem where in-place forward pass updates
    would modify the snapshot.

    KVCache layers are NOT snapshotted (they are trimmable and don't
    need rollback — use _trim_hybrid instead).

    Returns an opaque dict with:
        - "layers": {layer_idx: [mx.array copies]}
        - "kv_offsets": {layer_idx: offset} for KVCache layers (for trim)
        - "nbytes": total bytes of the snapshot

    TRIGGER CONTRACT: caller (server.py) MUST call this BEFORE
    prefilling tokens > SNAPSHOT_THRESHOLD.
    """
    snapshot = {"layers": {}, "kv_offsets": {}, "nbytes": 0}

    for i, layer in enumerate(cache):
        if _is_arrays_cache(layer):
            copied = []
            for c in layer.cache:
                if c is not None:
                    arr = mx.array(c)
                    copied.append(arr)
                    snapshot["nbytes"] += arr.nbytes
                else:
                    copied.append(None)
            snapshot["layers"][i] = copied
        elif hasattr(layer, "offset"):
            # Record KVCache offset for trim on rollback
            snapshot["kv_offsets"][i] = int(layer.offset)

    # Force materialization — without this, mx.array() may hold lazy refs
    if snapshot["layers"]:
        all_arrays = [
            a for arrays in snapshot["layers"].values()
            for a in arrays if a is not None
        ]
        if all_arrays:
            mx.eval(*all_arrays)

    return snapshot


def rollback_arrays_cache(cache: list, snapshot: Dict[str, Any]) -> bool:
    """Restore ArraysCache layers from a snapshot and trim KVCache layers.

    - ArraysCache layers: in-place replace of .cache arrays from snapshot
    - KVCache layers: trimmed back to the offset recorded in snapshot

    Returns True if rollback succeeded (at least one layer restored).
    """
    if not snapshot.get("layers") and not snapshot.get("kv_offsets"):
        return False

    restored = 0

    # Restore ArraysCache layers
    for idx_str, arrays in snapshot.get("layers", {}).items():
        idx = int(idx_str) if isinstance(idx_str, str) else idx_str
        if idx < len(cache) and _is_arrays_cache(cache[idx]):
            # Deep copy again to keep the snapshot reusable
            cache[idx].cache = [
                mx.array(a) if a is not None else None
                for a in arrays
            ]
            restored += 1

    # Trim KVCache layers back to snapshot position
    for idx_str, old_offset in snapshot.get("kv_offsets", {}).items():
        idx = int(idx_str) if isinstance(idx_str, str) else idx_str
        if idx < len(cache) and hasattr(cache[idx], "offset"):
            current_offset = int(cache[idx].offset)
            if current_offset > old_offset:
                trim_n = current_offset - old_offset
                if hasattr(cache[idx], "trim"):
                    cache[idx].trim(trim_n)
                    restored += 1

    # Materialize restored arrays
    if restored > 0:
        to_eval = []
        for idx_str in snapshot.get("layers", {}):
            idx = int(idx_str) if isinstance(idx_str, str) else idx_str
            if idx < len(cache) and _is_arrays_cache(cache[idx]):
                to_eval.extend(
                    a for a in cache[idx].cache if a is not None
                )
        if to_eval:
            mx.eval(*to_eval)

    return restored > 0


@dataclass
class HybridGenerationCheckpoint:
    """Prompt-only checkpoint captured immediately before hybrid generation.

    ``cache_key_len`` is the canonical-key prefix represented by the physical
    cache.  SuperMLX intentionally checkpoints with the final prompt token
    still pending, so exact retries can safely reprocess that one token.

    ``model_offset`` and ``model_prefix_hash`` enable defense-in-depth
    verification: on cache lookup the caller can confirm that
    model_tokens[:model_offset] in the new request hashes identically to what
    was checkpointed, preventing silent divergence from healing or
    canonicalization changes.
    """

    snapshot: Dict[str, Any]
    cache_key_len: int
    model_offset: int = 0
    model_prefix_hash: int = 0


def capture_hybrid_generation_checkpoint(
    cache: list,
    cache_key_len: int,
    model_offset: int = 0,
    model_prefix_hash: int = 0,
) -> Optional[HybridGenerationCheckpoint]:
    """Capture recurrent state and KV offsets for a hybrid prompt cache.

    Returns ``None`` for pure KV caches, invalid key lengths, or cache types
    without ArraysCache state.  Callers must treat ``None`` as non-reusable
    after generation; partially trimming only the KV layers is unsafe.
    """
    if not cache or cache_key_len <= 0:
        return None
    # With the Marconi monkey-patch, can_trim_prompt_cache() returns True even
    # for hybrid caches. Check for ArraysCache layers directly to decide if a
    # checkpoint is needed (pure KV caches don't need one).
    _has_arrays = any(_is_arrays_cache(c) for c in cache)
    if not _has_arrays:
        return None
    snapshot = snapshot_arrays_cache(cache)
    if not snapshot.get("layers"):
        return None
    return HybridGenerationCheckpoint(
        snapshot=snapshot,
        cache_key_len=cache_key_len,
        model_offset=model_offset,
        model_prefix_hash=model_prefix_hash,
    )


def restore_hybrid_generation_checkpoint(
    cache: list,
    checkpoint: HybridGenerationCheckpoint,
) -> bool:
    """Restore every recurrent layer and verify every recorded KV offset."""
    if checkpoint is None or not cache:
        return False

    snapshot = checkpoint.snapshot
    for idx in snapshot.get("layers", {}):
        idx = int(idx) if isinstance(idx, str) else idx
        if idx >= len(cache) or not _is_arrays_cache(cache[idx]):
            return False

    if not rollback_arrays_cache(cache, snapshot):
        return False

    for idx, expected_offset in snapshot.get("kv_offsets", {}).items():
        idx = int(idx) if isinstance(idx, str) else idx
        if idx >= len(cache) or not hasattr(cache[idx], "offset"):
            return False
        if int(cache[idx].offset) != int(expected_offset):
            return False
    return True


def prepare_cache_for_insertion(
    cache: list,
    cache_key: List[int],
    generated_tokens: List[int],
    checkpoint: Optional[HybridGenerationCheckpoint],
) -> Optional[Tuple[List[int], List[int]]]:
    """Return safe insertion inputs, or ``None`` when cache reuse is unsafe.

    Pure KV caches retain the existing trim-on-insert behavior.  Hybrid caches
    with generated tokens are reusable only after a complete checkpoint
    restore; failure is deliberately fail-closed.
    """
    if not generated_tokens or can_trim_prompt_cache(cache):
        return list(cache_key), list(generated_tokens)
    if checkpoint is None:
        return None
    if not restore_hybrid_generation_checkpoint(cache, checkpoint):
        return None
    if checkpoint.cache_key_len > len(cache_key):
        return None
    return list(cache_key[: checkpoint.cache_key_len]), []


# ── HybridPromptCache ────────────────────────────────────────────────────────

class HybridPromptCache(LRUPromptCache):
    """LRUPromptCache with hybrid model support for longer matches.

    mlx-lm's fetch_nearest_cache skips "longer" matches when
    can_trim_prompt_cache is False (hybrid models like Qwen3 MoE).
    This subclass handles them via per-layer KVCache trim.

    Returns (cache, rest_tokens, FetchResult) with diagnostics.
    """

    def fetch_nearest_cache(self, model: Any, tokens: List[int]):
        result = self._trie.search(model, tokens)

        if result.exact is not None:
            cache_entry = self._trie.get(result.model, result.exact)
            return (
                copy.deepcopy(cache_entry.prompt_cache),
                [],
                FetchResult(
                    hit_type=HitType.EXACT,
                    matched_prefix_len=len(result.exact),
                ),
            )

        short_length = len(result.shorter) if result.shorter is not None else 0

        if result.longer is not None and result.common_prefix > short_length:
            cache_entry = self._trie.get(result.model, result.longer)
            cache = copy.deepcopy(cache_entry.prompt_cache)
            prefix = min(len(tokens) - 1, result.common_prefix)
            num_to_trim = len(result.longer) - prefix

            # Pure KVCache: use mlx-lm's trim
            if can_trim_prompt_cache(cache):
                trim_prompt_cache(cache, num_to_trim)
                return (
                    cache,
                    tokens[prefix:],
                    FetchResult(
                        hit_type=HitType.LONGER_TRIMMED,
                        matched_prefix_len=prefix,
                        trimmed_tokens=num_to_trim,
                    ),
                )

            # Hybrid: per-layer KVCache trim
            if _trim_hybrid(cache, num_to_trim):
                return (
                    cache,
                    tokens[prefix:],
                    FetchResult(
                        hit_type=HitType.LONGER_TRIMMED,
                        matched_prefix_len=prefix,
                        trimmed_tokens=num_to_trim,
                    ),
                )

            # Trim failed entirely — fall through to shorter

        if short_length > 0:
            cache_entry = self._trie.get(result.model, result.shorter)
            return (
                copy.deepcopy(cache_entry.prompt_cache),
                tokens[short_length:],
                FetchResult(
                    hit_type=HitType.SHORTER,
                    matched_prefix_len=short_length,
                ),
            )

        return (
            None,
            tokens,
            FetchResult(hit_type=HitType.MISS),
        )
