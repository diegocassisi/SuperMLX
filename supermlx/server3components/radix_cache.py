"""
RadixPromptCache — prefix cache por radix tree, inspirado en RadixAttention
(SGLang), adaptado a single-user / single-request-at-a-time.

Diferencia con cache_lru.py actual:
  - cache_lru.py: hashea bloques de tokens de tamaño fijo, compara
    secuencialmente contra entradas de una LRU plana.
  - RadixPromptCache: cada nodo del árbol es un segmento de tokens
    compartido por todos sus descendientes. match_prefix() camina el árbol
    una vez y devuelve el prefijo común más largo en O(profundidad), no
    O(entradas).

No incluye (a propósito, por ser single-user local):
  - cache-aware scheduling / reordenamiento de cola de requests
  - continuous batching
  - eviction por prioridad entre tenants

Sí incluye: eviction LRU simple a nivel de nodo, igual que hoy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class RadixNode:
    tokens: list[int]                        # segmento de tokens de ESTE nodo
    children: dict[int, "RadixNode"] = field(default_factory=dict)  # keyed por primer token del hijo
    kv_cache: Any = None                      # mlx prompt cache asociado a este segmento
    last_used: float = 0.0
    ref_count: int = 0                        # >0 mientras un request activo lo usa
    # TODO Commit 1: Kripper Dual-Slot — agregar slot_type: Literal["main", "compact"]
    # para modelar MAIN vs COMPACT. Decidir si son dos árboles separados o un
    # flag por nodo. pinned=True hoy no se invoca en código activo (verificado
    # por grep) pero el diseño debe soportarlo.


class RadixPromptCache:
    """
    Un solo árbol, un solo hilo activo (single-user). No hay lock de
    contienda entre requests porque no hay concurrencia real — el lock
    existente (model_lock) sigue siendo suficiente, este objeto no agrega
    uno propio.
    """

    def __init__(self, max_tokens: int):
        self.root = RadixNode(tokens=[])
        self.max_tokens = max_tokens
        self._total_tokens = 0

    def match_prefix(self, tokens: list[int]) -> tuple[list[int], Any]:
        """
        Devuelve (matched_prefix_tokens, kv_cache_del_nodo_mas_profundo_con_kv).
        El caller (cache_lookup phase) resta len(matched_prefix_tokens) de
        len(tokens) para saber el rest_count a prefillear.

        kv_cache puede ser None si el match estructural existe pero ningún
        nodo en el camino tiene KV reutilizable (ej: primer request, o split
        parcial en la raíz). El caller debe tratar kv_cache=None como MISS
        funcional (rest_count = len(tokens)).

        TODO: portar la lógica de _block_chain_hashes existente como
        fallback de verificación — comparar contenido real de tokens, no
        solo el árbol, antes de confiar en un match (la contaminación KV
        que ya tuviste vino de un falso positivo de este tipo).
        """
        node = self.root
        matched: list[int] = []
        best_kv = None  # último nodo con kv_cache válido
        remaining = tokens
        while remaining and remaining[0] in node.children:
            child = node.children[remaining[0]]
            shared = _common_prefix_len(remaining, child.tokens)
            matched.extend(remaining[:shared])
            if shared < len(child.tokens):
                # Match parcial → requiere split del nodo. NO es solo
                # partir listas de tokens del árbol: el kv_cache del nodo
                # es un tensor mlx indexado por posición (RoPE incluido).
                # Split implica slicing alineado a esa posición exacta y
                # generar dos handles de cache válidos, uno por mitad.
                # Implementar en Commit 1 (no diferir) — sin esto, todo
                # turn que EXTIENDE un prefijo ya cacheado (el caso común
                # en conversación multi-turn) cae a rest_count completo.
                break
            remaining = remaining[shared:]
            node = child
            if child.kv_cache is not None:
                best_kv = child.kv_cache
        return matched, best_kv

    def insert(self, tokens: list[int], kv_cache: Any) -> None:
        """TODO: insertar/actualizar el árbol tras un generate() exitoso."""
        raise NotImplementedError

    def evict_lru(self, need_tokens: int) -> None:
        """TODO: portar la política de eviction actual de cache_lru.py."""
        raise NotImplementedError


def _common_prefix_len(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i
