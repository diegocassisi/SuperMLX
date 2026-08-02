"""
[AI_DIRECTIVE]
ROL: Centralized cache layer type detection for SuperMLX
OBJETIVO: Single source of truth for identifying ArraysCache vs KVCache layers
ENTRADAS: prompt_cache list from mlx-lm
SALIDAS: CacheLayerInfo classifications
REGLAS INVIOLABLES:
- No direct imports from mlx_lm.models.cache.ArraysCache
- Duck-typing only — works across mlx-lm versions
- Classify once, use everywhere
SSoT: This module is the ONLY place that determines cache layer types
"""

import logging
from dataclasses import dataclass
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class CacheLayerInfo:
    """Classification of a single cache layer."""
    layer_idx: int
    is_recurrent: bool       # True = ArraysCache / GatedDeltaNet state
    is_kv: bool              # True = KVCache (has keys/values/offset)
    cache_ref: Any           # Direct reference to the layer object

    @property
    def is_trimmable(self) -> bool:
        """KV layers are trimmable; recurrent layers are not (without monkey-patch)."""
        return self.is_kv and not self.is_recurrent


def _detect_recurrent(layer: Any) -> bool:
    """Duck-type detection: recurrent layer has .state or .cache list but no .offset.

    CRITICAL: KV cache variants (KVCache, QuantizedKVCache, RotatingKVCache)
    all have .offset. Recurrent layers (ArraysCache / GatedDeltaNet) do NOT.
    Check .offset FIRST to exclude all KV variants.
    """
    # All KV cache types have .offset — exclude them immediately
    if hasattr(layer, "offset"):
        return False
    # Primary: has 'state' attribute with list/tuple (GatedDeltaNet pattern)
    if hasattr(layer, "state") and isinstance(getattr(layer, "state", None), (list, tuple)):
        return True
    # Fallback: has .cache list (ArraysCache pattern)
    if (
        hasattr(layer, "cache")
        and isinstance(getattr(layer, "cache", None), list)
    ):
        return True
    return False


def _detect_kv(layer: Any) -> bool:
    """Duck-type detection: KVCache has .offset and .keys/.values."""
    return hasattr(layer, "offset") and (hasattr(layer, "keys") or hasattr(layer, "values"))


def classify_cache_layers(prompt_cache: list) -> List[CacheLayerInfo]:
    """Classify all layers in a prompt cache. Call once at model load or cache creation.

    Returns a list of CacheLayerInfo, one per layer, in order.
    """
    if not prompt_cache:
        return []

    result = []
    for idx, layer in enumerate(prompt_cache):
        is_rec = _detect_recurrent(layer)
        is_kv = _detect_kv(layer)
        result.append(CacheLayerInfo(
            layer_idx=idx,
            is_recurrent=is_rec,
            is_kv=is_kv,
            cache_ref=layer,
        ))

    n_rec = sum(1 for c in result if c.is_recurrent)
    n_kv = sum(1 for c in result if c.is_kv)
    logger.info("[INIT] cache_types: %d layers classified | %d recurrent | %d KV",
                len(result), n_rec, n_kv)
    return result


def is_hybrid_cache(classifications: List[CacheLayerInfo]) -> bool:
    """True if the cache has both recurrent and KV layers (hybrid model)."""
    has_rec = any(c.is_recurrent for c in classifications)
    has_kv = any(c.is_kv for c in classifications)
    return has_rec and has_kv


def has_recurrent_layers(prompt_cache: list) -> bool:
    """Quick check without full classification — for guards that just need yes/no."""
    return any(_detect_recurrent(layer) for layer in prompt_cache)


def is_recurrent_layer(layer: Any) -> bool:
    """Check a single layer. Drop-in replacement for _is_arrays_cache()."""
    return _detect_recurrent(layer)
