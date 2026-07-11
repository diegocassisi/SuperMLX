"""
cache_engine.py — Hybrid-aware prompt cache for SuperMLX.

Extends mlx-lm's LRUPromptCache with two additions:

1. fetch_nearest_cache: handles "longer" matches on hybrid models
   (ArraysCache + KVCache) by doing per-layer KVCache trim when
   can_trim_prompt_cache returns False.

2. strip_response_tokens: trims generated response tokens from a
   cache before re-insertion, preventing contamination.
"""

import copy
from typing import Any, List, Optional, Tuple

from mlx_lm.models.cache import (
    LRUPromptCache,
    can_trim_prompt_cache,
    trim_prompt_cache,
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


class HybridPromptCache(LRUPromptCache):
    """LRUPromptCache with hybrid model support for longer matches.

    mlx-lm's fetch_nearest_cache skips "longer" matches when
    can_trim_prompt_cache is False (hybrid models like Qwen3 MoE).
    This subclass handles them via per-layer KVCache trim.
    """

    def fetch_nearest_cache(self, model: Any, tokens: List[int]):
        result = self._trie.search(model, tokens)

        if result.exact is not None:
            cache_entry = self._trie.get(result.model, result.exact)
            return copy.deepcopy(cache_entry.prompt_cache), []

        short_length = len(result.shorter) if result.shorter is not None else 0

        if result.longer is not None and result.common_prefix > short_length:
            cache_entry = self._trie.get(result.model, result.longer)
            cache = copy.deepcopy(cache_entry.prompt_cache)
            prefix = min(len(tokens) - 1, result.common_prefix)
            num_to_trim = len(result.longer) - prefix

            # Pure KVCache: use mlx-lm's trim
            if can_trim_prompt_cache(cache):
                trim_prompt_cache(cache, num_to_trim)
                return cache, tokens[prefix:]

            # Hybrid: per-layer KVCache trim
            if _trim_hybrid(cache, num_to_trim):
                return cache, tokens[prefix:]

            # Trim failed entirely — fall through to shorter

        if short_length > 0:
            cache_entry = self._trie.get(result.model, result.shorter)
            return copy.deepcopy(cache_entry.prompt_cache), tokens[short_length:]

        return None, tokens
