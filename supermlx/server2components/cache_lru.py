"""
[AI_DIRECTIVE]
ROL: LRU prompt cache for KV state reuse across requests
OBJETIVO: Almacenar y reusar KV cache entries para evitar prefill completo,
          logrando 97%+ cache hit rate en conversaciones multi-turno
ENTRADAS: model name, token sequences, prompt_cache KV states
SALIDAS: Best-match cache entry (prompt_cache, common_prefix_length, total_tokens)
REGLAS INVIOLABLES:
- Thread-safe: caller must hold prompt_cache_lock for all mutations
- Pinned entries never evicted (Kripper Base Slot pattern)
- TTL + LRU eviction for memory management
- Block-aligned token matching for optimal prefix reuse
SSoT: Este módulo es la única implementación de prompt cache LRU
"""
from __future__ import annotations

import hashlib
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Module-level callable for terminal status output
_terminal_status_fn: Optional[Callable] = None
_settings: Any = None


def init(*, settings: Any, terminal_status_fn: Callable) -> None:
    """Initialize with shared state from server2."""
    global _terminal_status_fn, _settings
    _terminal_status_fn = terminal_status_fn
    _settings = settings


def _block_chain_hashes(
    tokens: Tuple[int, ...],
    block_size: int,
) -> List[Tuple[bytes, int]]:
    """Return [(chain_hash, prefix_len), ...] for each block prefix. chain_hash[i] = H(prev || block_i)."""
    if block_size <= 0 or not tokens:
        return []
    out: List[Tuple[bytes, int]] = []
    prev = b""
    for i in range(0, len(tokens), block_size):
        block = tokens[i : i + block_size]
        block_bytes = b"".join(t.to_bytes(4, "big") for t in block)
        h = hashlib.sha256(prev + block_bytes).digest()
        prefix_len = min(i + block_size, len(tokens))
        out.append((h, prefix_len))
        prev = h
    return out


class LRUPromptCache:
    @dataclass
    class CacheEntry:
        prompt_cache: List[Any]
        tokens: Tuple[int, ...]
        count: int
        touched_at: float
        pinned: bool = False  # Kripper Base Slot: nunca evictado si True

    def __init__(self, max_size=10, ttl_seconds=1800):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.block_size = max(16, getattr(_settings, "prompt_cache_block_size", 16))

        # Core Flat Cache: (model, exact_tokens) -> CacheEntry
        self._entries: Dict[Tuple[str, Tuple[int, ...]], self.CacheEntry] = {}
        # Block Hash Index: (model, chain_hash) -> Set of token sequences
        self._block_index: Dict[Tuple[str, bytes], set] = {}

    def _is_expired(self, entry):
        if entry.pinned:  # Kripper Base Slot: nunca expira por TTL
            return False
        if self.ttl_seconds <= 0:
            return False
        return (time.time() - entry.touched_at) > self.ttl_seconds

    def prune_expired(self):
        now = time.time()
        stale_keys = [k for k, v in self._entries.items() if self._is_expired(v)]
        for k in stale_keys:
            self._delete(k[0], k[1])
        if "HOUSEKEEPING_STAGING_MANAGER" in globals() and "PROMPT_CACHE" in globals() and self is PROMPT_CACHE:
            HOUSEKEEPING_STAGING_MANAGER.prune(
                self,
                now=now,
                ttl_seconds=getattr(_settings, "housekeeping_staging_ttl_seconds", 300.0),
                max_entries=getattr(_settings, "housekeeping_staging_max_entries", 2),
            )

    def _delete(self, model, tokens, _reaper_telemetry=False):
        key = (model, tuple(tokens))
        if key not in self._entries:
            return

        # Clean up the block index to prevent memory leaks
        chain_pairs = _block_chain_hashes(tokens, self.block_size)
        for chain_hash, _ in chain_pairs:
            idx_key = (model, chain_hash)
            if idx_key in self._block_index:
                self._block_index[idx_key].discard(key[1])
                if not self._block_index[idx_key]:
                    del self._block_index[idx_key]

        del self._entries[key]



        # Return evicted Metal GPU buffers to the OS pool immediately.
        # Without this, MLX holds the backing Metal buffers in its pool even
        # after the Python reference is dropped, causing monotonic GPU memory
        # growth during long sessions with repeated cache divergence.
        try:
            _guard.force_clear_cache("cache_entry_delete")
        except Exception:
            pass

    def _extract(self, model, tokens):
        """Pop entry from LRU and return the LIVE reference (no deepcopy).

        Design: model_lock guarantees only one request runs at a time, so there
        is no concurrent modification risk. The caller gets the live KV object,
        stream_generate extends it in-place, and _insert_cache_entries
        re-inserts the updated state after generation completes.

        Memory impact: halves KV footprint during generation (1 object instead
        of 2).  Previous design kept the original AND a deepcopy alive
        simultaneously, which was the primary OOM trigger on 24 GB hardware.
        """
        key = (model, tuple(tokens))
        entry = self._entries[key]
        entry.touched_at = time.time()
        entry.count += 1

        # Pop from the LRU dict so the slot is FREE during generation.
        # _insert_cache_entries will re-insert the extended state when done.
        del self._entries[key]
        # Clean up block index to keep index consistent.
        chain_pairs = _block_chain_hashes(tokens, self.block_size)
        for chain_hash, _ in chain_pairs:
            idx_key = (model, chain_hash)
            if idx_key in self._block_index:
                self._block_index[idx_key].discard(key[1])
                if not self._block_index[idx_key]:
                    del self._block_index[idx_key]

        # Return a CacheEntry wrapping the original (un-copied) prompt_cache.
        return self.CacheEntry(
            entry.prompt_cache,  # Live reference — NO deepcopy
            entry.tokens,
            entry.count,
            entry.touched_at,
        )

    def _evict_optimal(self):
        """
        Cost-Aware Eviction: Finds the entry with the highest eviction score.
        Protects long 'trunks' and frequently used templates; penalizes old, short branches.
        Kripper Base Slot: entries with pinned=True are immune to eviction.
        Only if ALL slots are pinned, the oldest is evicted (safety fallback).
        """
        now = time.time()
        best_key = None
        max_score = -1.0

        for key, entry in self._entries.items():
            if entry.pinned:  # Kripper slot: nunca evictar el base
                continue
            age_seconds = max(1.0, now - entry.touched_at)

            # Use square root to create a balanced gravity for long chains.
            # A 10,000 token chain has 10x more protection than a 100 token chain.
            length_weight = math.sqrt(len(entry.tokens))
            freq_weight = math.log1p(entry.count) + 1.0

            # Higher score = more likely to be evicted. (Old and Short -> High Score)
            score = age_seconds / (length_weight * freq_weight)

            if score > max_score:
                max_score = score
                best_key = key

        # Safety fallback: if ALL slots are pinned, evict the oldest pinned
        # (evita deadlock de memoria por pinning excesivo)
        if best_key is None:
            oldest_key = None
            oldest_time = float("inf")
            for key, entry in self._entries.items():
                if entry.touched_at < oldest_time:
                    oldest_time = entry.touched_at
                    oldest_key = key
            if oldest_key:
                _terminal_status_fn(
                    "⚠️",
                    "Kripper slot: all slots are pinned, evicting the oldest",
                )
                self._delete(oldest_key[0], oldest_key[1])
            return

        if best_key:
            self._delete(best_key[0], best_key[1])

    def _cull_redundant_prefixes(self, model, new_tokens):
        """
        Frees up slots by removing strict prefixes. Since the cache can trim
        longer chains to serve shorter ones, shorter strict prefixes are wasted slots.
        Kripper Base Slot: pinned entries are NEVER culled — they are structural.
        """
        new_len = len(new_tokens)
        to_delete = []
        for m, t_tup in self._entries.keys():
            if m == model and len(t_tup) < new_len:
                # Never cull pinned entries (Kripper slots)
                entry = self._entries.get((m, t_tup))
                if entry and entry.pinned:
                    continue
                # If the existing shorter cache is a strict prefix of the new one
                if new_tokens[: len(t_tup)] == t_tup:
                    to_delete.append((m, t_tup))

        to_delete.sort(key=lambda x: len(x[1]), reverse=True)
        if to_delete:
            # Spare the longest prefix from deletion
            to_delete.pop(0)

        for k in to_delete:
            self._delete(k[0], k[1])

    def evict_unpinned(self) -> int:
        """
        Evict all non-pinned entries immediately to free GPU memory.

        Called before large EMBEDDED prefills to prevent Metal OOM:
          With 36-layer Qwen3.5-9B, each KV slot takes ~4.5GB.
          EMBEDDED deepcopy needs: Kripper(4.5) + SessionCache(4.5) + EMBEDDED_KV(4.5) = 13.5GB
          Evicting session cache first: Kripper(4.5) + EMBEDDED_KV(4.5) = 9GB  → safe.

        Returns number of entries evicted.
        """
        to_delete = [
            (m, t) for (m, t), e in self._entries.items() if not e.pinned
        ]
        for m, t in to_delete:
            self._delete(m, t)
        if to_delete:
            try:
                _guard.force_clear_cache("batch_evict")  # Return Metal buffers immediately
            except Exception:
                pass
        return len(to_delete)

    def fetch_nearest_cache(self, model, tokens):
        self.prune_expired()
        tokens_tup = tuple(tokens)

        # Fast path: exact-token key lookup is O(1) and avoids scanning large
        # candidate sets in block index under multi-session workloads.
        exact_key = (model, tokens_tup)
        if exact_key in self._entries:
            entry = self._extract(model, tokens_tup)
            if len(tokens_tup) > 1:
                return (
                    entry.prompt_cache,
                    tokens_tup[-1:],
                    tokens_tup,
                    "exact",
                    len(tokens_tup) - 1,
                )
            return entry.prompt_cache, [], tokens_tup, "exact", len(tokens_tup)

        # 1. Hash the incoming request into blocks
        chain_pairs = _block_chain_hashes(tokens_tup, self.block_size)
        if not chain_pairs:
            return None, tokens, tokens, "miss", 0

        best_prefix_len = 0
        best_cached_tokens = None

        # 2. Walk blocks backwards. For each block level, collect ALL candidates
        # that match up to that block boundary and pick the LONGEST one.
        # Previously "first matching candidate wins" — but sets are unordered, so
        # a short heartbeat/branch entry could win over a long conversation entry
        # that shares the same early block hashes, causing a near-total cache flush.
        for chain_hash, req_prefix_len in reversed(chain_pairs):
            idx_key = (model, chain_hash)
            if idx_key in self._block_index:
                candidate_tokens_set = self._block_index[idx_key]
                best_candidate_at_level = None
                best_candidate_len = -1
                for candidate_tokens in candidate_tokens_set:
                    if tokens_tup[:req_prefix_len] == candidate_tokens[:req_prefix_len]:
                        if len(candidate_tokens) > best_candidate_len:
                            best_candidate_len = len(candidate_tokens)
                            best_candidate_at_level = candidate_tokens
                if best_candidate_at_level is not None:
                    best_prefix_len = req_prefix_len
                    best_cached_tokens = best_candidate_at_level
            if best_cached_tokens is not None:
                break

        # 3. If no block matched, return miss
        if best_cached_tokens is None:
            return None, tokens, tokens, "miss", 0

        # MEMORY GUARD: If match ratio is trivially low (< 5%), return miss.
        # Avoids using a KV entry that shares almost nothing with the current
        # prompt (e.g. EMBEDDED vs MAIN), which would waste the slot for zero benefit.
        match_ratio = best_prefix_len / max(len(tokens_tup), 1)
        if match_ratio < 0.05:
            return None, tokens, tokens, "miss", 0
        entry = self._extract(model, best_cached_tokens)

        # 5. Extend match inside the current block
        while best_prefix_len < min(len(tokens_tup), len(best_cached_tokens)):
            if tokens_tup[best_prefix_len] == best_cached_tokens[best_prefix_len]:
                best_prefix_len += 1
            else:
                break

        # 6. Return sliced/trimmed cache
        if best_prefix_len == len(tokens_tup):
            # Request is fully covered by selected cache.
            # Trimming is delegated to Fix-31 v3 (Model Space Math)
            if len(tokens_tup) > 1:
                return (
                    entry.prompt_cache,
                    tokens_tup[-1:],
                    best_cached_tokens,
                    "exact",
                    len(tokens_tup) - 1,
                )
            return entry.prompt_cache, [], best_cached_tokens, "exact", len(tokens_tup)

        if best_prefix_len < len(tokens_tup):
            # Shorter Match (We have the prefix, compute the rest)
            # Trimming is delegated to Fix-31 v3 (Model Space Math)
            return (
                entry.prompt_cache,
                list(tokens_tup)[best_prefix_len:],
                best_cached_tokens,
                "shorter",
                best_prefix_len,
            )

        # Longer cache: request is a strict prefix of cached
        # Trimming is delegated to Fix-31 v3 (Model Space Math)
        return entry.prompt_cache, [], best_cached_tokens, "longer", len(tokens_tup)

    def insert_cache(self, model, tokens, prompt_cache, pinned: bool = False):
        self.prune_expired()
        tokens_tup = tuple(tokens)
        key = (model, tokens_tup)
        now = time.time()

        if key in self._entries:
            self._entries[key].count += 1
            self._entries[key].touched_at = now
            if pinned:  # Kripper slot: upgrade a pinned si se re-inserta como base
                self._entries[key].pinned = True
            return

        # NOTE: Eviction guard removed — with all sessions ~13K tokens,
        # the guard was locking the first stale entry forever.
        # Tool normalization alone handles cross-session compatibility.

        # 1. Subsumption: Cull redundant prefixes to organically free up space
        self._cull_redundant_prefixes(model, tokens_tup)

        # 2. Insert into flat dictionary
        self._entries[key] = self.CacheEntry(
            prompt_cache, tokens_tup, 1, now, pinned=pinned
        )

        # 3. Map block hashes
        chain_pairs = _block_chain_hashes(tokens_tup, self.block_size)
        for chain_hash, _ in chain_pairs:
            idx_key = (model, chain_hash)
            if idx_key not in self._block_index:
                self._block_index[idx_key] = set()
            self._block_index[idx_key].add(tokens_tup)

        # 4. Enforce max size using the Cost-Aware Eviction
        while len(self._entries) > self.max_size:
            self._evict_optimal()

    def contains_tokens(self, model, tokens):
        key = (model, tuple(tokens))
        return key in self._entries


