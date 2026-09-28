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


ASTRA Modification
Hybrid-aware prompt cache with conservative reuse.

Recurrent caches are never shortened without a complete checkpoint.
Rollback is staged before committing. Callers remain responsible for
matching physical model tokens to cache keys and validating prefix hashes.
Snapshot schema now includes complete recurrent layer states; legacy
array-only snapshots are rejected. Requires independent deepcopy semantics
for mutable cache objects and compatible MLX persistent array semantics.
"""

import copy
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from supermlx.cache_types import is_recurrent_layer

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

# _is_arrays_cache removed — use is_recurrent_layer from cache_types (SSoT)
_is_arrays_cache = is_recurrent_layer  # backward compat alias for internal refs


def _trim_hybrid(cache: list, n: int) -> bool:
    """Compatibility helper: never partially trim a recurrent cache."""
    return strip_response_tokens(cache, n)


def strip_response_tokens(cache: list, n: int) -> bool:
    """Trim only non-recurrent caches; failed caches must be discarded."""
    if n <= 0:
        return True
    if not cache or any(_is_arrays_cache(c) for c in cache):
        return False
    if not can_trim_prompt_cache(cache):
        return False
    offsets = [getattr(c, "offset", None) for c in cache]
    if any(o is None or int(o) < n for o in offsets):
        return False
    trim_prompt_cache(cache, n)
    return all(int(c.offset) == int(o) - n for c, o in zip(cache, offsets))


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
    snapshot = {"layers": {}, "states": {}, "kv_offsets": {}, "nbytes": 0}

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
            state = copy.deepcopy(layer)
            state.cache = copied
            snapshot["states"][i] = state
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
    """Stage restoration on copies, committing only after full validation.

    Preserves snapshot-time recurrent metadata as well as arrays. Legacy
    snapshots without complete recurrent states are intentionally rejected.
    """
    try:
        layers = {int(i): state for i, state in snapshot.get("states", {}).items()}
        offsets = {int(i): int(o) for i, o in snapshot.get("kv_offsets", {}).items()}
        recurrent = {i for i, c in enumerate(cache) if _is_arrays_cache(c)}
        others = set(range(len(cache))) - recurrent
        if not cache or set(layers) != recurrent or set(offsets) != others:
            return False
        for i, state in layers.items():
            if type(state) is not type(cache[i]):
                return False
        for i, old in offsets.items():
            c = cache[i]
            if old < 0 or not hasattr(c, "offset") or int(c.offset) < old:
                return False
            if int(c.offset) > old and not (
                callable(getattr(c, "trim", None))
                and callable(getattr(c, "is_trimmable", None))
                and c.is_trimmable()
            ):
                return False
        # Stage restoration: superficial copy of cache list, deepcopy only recurrent states
        staged = list(cache)
        for i, state in layers.items():
            staged[i] = copy.deepcopy(state)
        for i, old in offsets.items():
            c = staged[i]
            delta = int(c.offset) - old
            if delta:
                c.trim(delta)
            if int(c.offset) != old:
                return False
        arrays = [a for i in layers for a in staged[i].cache if a is not None]
        if arrays:
            mx.eval(*arrays)
    except Exception:
        return False
    cache[:] = staged
    return True


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
    if not generated_tokens:
        return list(cache_key), []
    if not any(_is_arrays_cache(c) for c in cache):
        if not can_trim_prompt_cache(cache):
            return None
        return list(cache_key), list(generated_tokens)
    if checkpoint is None:
        return None
    if not 0 < checkpoint.cache_key_len <= len(cache_key):
        return None
    if not restore_hybrid_generation_checkpoint(cache, checkpoint):
        return None
    return list(cache_key[: checkpoint.cache_key_len]), []


# ── HybridPromptCache ────────────────────────────────────────────────────────

class HybridPromptCache(LRUPromptCache):
    """LRUPromptCache with hybrid model support for longer matches.

    mlx-lm's fetch_nearest_cache skips "longer" matches when
    can_trim_prompt_cache is False (hybrid models like Qwen3 MoE).
    This subclass rejects them unless a valid shorter entry exists.

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
            source = cache_entry.prompt_cache
            prefix = min(len(tokens) - 1, result.common_prefix)
            num_to_trim = len(result.longer) - prefix
            # Reject recurrent candidates before copying. A shorter valid
            # entry remains eligible below, otherwise return MISS.
            if (
                prefix >= 0
                and num_to_trim >= 0
                and not any(_is_arrays_cache(c) for c in source)
                and can_trim_prompt_cache(source)
            ):
                cache = copy.deepcopy(source)
                if strip_response_tokens(cache, num_to_trim):
                    return (
                        cache,
                        tokens[prefix:],
                        FetchResult(
                            hit_type=HitType.LONGER_TRIMMED,
                            matched_prefix_len=prefix,
                            trimmed_tokens=num_to_trim,
                        ),
                    )

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
