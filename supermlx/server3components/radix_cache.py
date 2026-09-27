"""
[AI_DIRECTIVE]
ROL: Prefix cache por Radix Tree (RadixAttention) para reutilización de KV cache.
OBJETIVO: Almacenar secuencias de tokens y sus estados KV asociados con matching
          O(profundidad), bifurcación de prefijos compartidos (node split),
          eviction LRU con soporte Kripper Dual-Slot (MAIN vs COMPACT) y slots pinned.
ENTRADAS: Secuencias de tokens (list[int]), tensores KV de MLX / objetos de cache, slot_type.
SALIDAS: Tupla (matched_prefix_tokens, kv_cache_del_nodo_mas_profundo).
REGLAS INVIOLABLES:
- Single-user local: thread-safety coordinado externamente por model_lock.
- Node split obligatorio en match parcial tanto en lectura como en inserción.
- Kripper Dual-Slot: árboles aislados para 'main' y 'compact' bajo un mismo presupuesto max_tokens.
- Nodos pinned inmunes a eviction LRU salvo deadlock total.
- Sin imports circulares; dependencias inyectadas o resueltas con duck-typing limpio.
DECISIÓN ARQUITECTURAL (Commit 3.5 — Coexistencia TPC vs Radix Tree):
- Opción (b) adoptada: Subsistemas separados con orden de consulta determinístico estricto.
  1. RadixPromptCache es el SSoT primario para secuencias conversacionales dinámicas multi-turno.
  2. TPC (ToolPrefixCache) es el baseline inmutable de arranque en frío (cold-start) para requests con herramientas.
  3. Orden de consulta: RadixPromptCache primero. Solo ante MISS funcional total se consulta TPC como fallback.
  4. Garantía de no-contaminación: TPC retorna clones aislados; el estado extendido se indexa en el Radix Tree
     en postprocess(), permitiendo que los turnos subsiguientes resuelvan 100% dentro del árbol radix.
SSoT: RadixPromptCache es la única implementación de cache radix para server3.
"""
from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

logger = logging.getLogger(__name__)

@dataclass
class RadixNode:
    tokens: list[int]                                                # Segmento de tokens de este nodo
    children: dict[int, "RadixNode"] = field(default_factory=dict)  # Keyed por primer token del hijo
    kv_cache: Any = None                                            # MLX prompt cache asociado al segmento
    last_used: float = field(default_factory=time.time)
    ref_count: int = 0                                              # >0 mientras un request activo lo usa
    slot_type: Literal["main", "compact"] = "main"                  # Kripper Dual-Slot
    pinned: bool = False                                            # Inmune a eviction LRU si True
    parent: Optional["RadixNode"] = None                            # Referencia al nodo padre para eviction/split


class RadixPromptCache:
    """
    Cache de prefijos basado en Radix Tree (RadixAttention).
    Maneja dos árboles lógicos ('main' y 'compact') para aislar turnos de conversación
    principal de resúmenes de contexto o background tasks, compartiendo el límite max_tokens.
    """

    def __init__(
        self,
        max_tokens: int = 131072,
        guard: Any = None,
        trim_fn: Optional[Callable[[Any, int], None]] = None,
    ):
        self.max_tokens = max_tokens
        self.guard = guard
        self.trim_fn = trim_fn
        self._total_tokens = 0
        self._roots: dict[str, RadixNode] = {
            "main": RadixNode(tokens=[], slot_type="main"),
            "compact": RadixNode(tokens=[], slot_type="compact"),
        }

    @property
    def root(self) -> RadixNode:
        """Compatibilidad con el skeleton original (slot main por defecto)."""
        return self._roots["main"]

    def get_root(self, slot: str = "main") -> RadixNode:
        if slot not in self._roots:
            self._roots[slot] = RadixNode(tokens=[], slot_type=slot)  # type: ignore[arg-type]
        return self._roots[slot]

    @property
    def total_tokens(self) -> int:
        return self._total_tokens

    def match_prefix(
        self,
        tokens: list[int],
        slot: str = "main",
        min_match_ratio: float = 0.05,
        min_match_tokens: int = 16,
    ) -> tuple[list[int], Any]:
        """
        Recorre el árbol correspondiente a 'slot' y devuelve:
        (matched_prefix_tokens, kv_cache_del_nodo_mas_profundo_con_kv).

        Si ocurre un match parcial dentro de un nodo (shared < len(child.tokens)),
        realiza el split del nodo en el acto, permitiendo bifurcaciones futuras
        sin degradar a rest_count completo.

        Aplica dos guards críticos para prevenir contaminación KV:
        1. Content verification: comprueba token por token que matched == tokens[:len(matched)].
        2. Prefix-match-ratio guard: descarta matches con ratio < 5% en prompts de tamaño >= min_match_tokens.
        """
        if not tokens:
            return [], None

        node = self.get_root(slot)
        matched: list[int] = []
        best_kv = None
        remaining = list(tokens)

        while remaining and remaining[0] in node.children:
            child = node.children[remaining[0]]
            shared = _common_prefix_len(remaining, child.tokens)
            if shared == 0:
                break

            if shared < len(child.tokens):
                # Match parcial: split del nodo
                prefix_node = self._split_node(child, shared)
                matched.extend(prefix_node.tokens)
                prefix_node.last_used = time.time()
                if prefix_node.kv_cache is not None:
                    best_kv = prefix_node.kv_cache
                break

            # Match total del segmento del child
            matched.extend(child.tokens)
            child.last_used = time.time()
            if child.kv_cache is not None:
                best_kv = child.kv_cache

            remaining = remaining[shared:]
            node = child

        # ── GUARD 1: Verificación estricta de contenido real de tokens ──
        if matched and tokens[: len(matched)] != matched:
            logger.error(
                "[RADIX CONTAMINATION GUARD] Divergencia detectada: tokens de cache no coinciden con prompt real!"
            )
            return [], None

        # ── GUARD 2: Prefix-match-ratio guard (portado de cache_lru.py) ──
        # Evita atar el KV a prefijos triviales (< 5%) en prompts largos.
        if matched and len(tokens) >= min_match_tokens:
            match_ratio = len(matched) / len(tokens)
            if match_ratio < min_match_ratio:
                logger.debug(
                    "[RADIX GUARD] Match ratio trivial (%.3f < %.3f) — descartado como miss",
                    match_ratio,
                    min_match_ratio,
                )
                return [], None

        isolated_kv = copy.deepcopy(best_kv) if best_kv is not None else None
        return matched, isolated_kv

    def insert(
        self,
        tokens: list[int],
        kv_cache: Any,
        slot: str = "main",
        pinned: bool = False,
        is_compact: bool = False,
    ) -> None:
        """
        Inserta una secuencia de tokens y su KV cache en el árbol del slot especificado.
        Si is_compact es True, fuerza slot='compact'.
        Si la inserción supera max_tokens, dispara evict_lru para mantener el presupuesto.
        """
        if is_compact:
            slot = "compact"
        if not tokens:
            return

        kv_to_store = copy.deepcopy(kv_cache) if kv_cache is not None else None
        node = self.get_root(slot)
        remaining = list(tokens)

        while remaining:
            first_token = remaining[0]
            if first_token not in node.children:
                # Caso: no hay hijo con este token inicial -> crear hoja con el sufijo restante
                new_leaf = RadixNode(
                    tokens=remaining,
                    kv_cache=kv_to_store,
                    slot_type=slot,  # type: ignore[arg-type]
                    pinned=pinned,
                    last_used=time.time(),
                    parent=node,
                )
                node.children[first_token] = new_leaf
                self._total_tokens += len(remaining)
                remaining = []
                break

            child = node.children[first_token]
            shared = _common_prefix_len(remaining, child.tokens)

            if shared < len(child.tokens):
                # Split necesario para insertar la nueva rama
                prefix_node = self._split_node(child, shared)
                remaining = remaining[shared:]
                if remaining:
                    new_leaf = RadixNode(
                        tokens=remaining,
                        kv_cache=kv_to_store,
                        slot_type=slot,  # type: ignore[arg-type]
                        pinned=pinned,
                        last_used=time.time(),
                        parent=prefix_node,
                    )
                    prefix_node.children[remaining[0]] = new_leaf
                    self._total_tokens += len(remaining)
                else:
                    # El nuevo camino coincide exactamente con el prefix recién creado
                    prefix_node.kv_cache = kv_to_store
                    prefix_node.pinned = pinned or prefix_node.pinned
                    prefix_node.last_used = time.time()
                remaining = []
                break
            else:
                # El child completo coincide con el prefijo de remaining
                remaining = remaining[shared:]
                node = child
                if not remaining:
                    # Coincidencia exacta con un nodo preexistente -> actualizar KV
                    node.kv_cache = kv_to_store
                    node.pinned = pinned or node.pinned
                    node.last_used = time.time()
                    break

        # Mantener presupuesto de memoria
        if self._total_tokens > self.max_tokens:
            self.evict_lru(self._total_tokens - self.max_tokens)

    def _split_node(self, child: RadixNode, split_idx: int) -> RadixNode:
        """
        Divide child en dos nodos:
        - prefix_node: tokens[:split_idx], que pasa a ocupar el lugar de child bajo su parent.
        - suffix_node (child modificado): tokens[split_idx:], que pasa a ser hijo de prefix_node.
        Maneja el slicing/trim del kv_cache de forma coherente.
        """
        parent = child.parent
        if parent is None:
            raise ValueError("No se puede hacer split de un nodo raíz sin parent.")

        prefix_tokens = child.tokens[:split_idx]
        suffix_tokens = child.tokens[split_idx:]

        # Crear prefix_node ocupando el lugar de child bajo parent
        prefix_node = RadixNode(
            tokens=prefix_tokens,
            slot_type=child.slot_type,
            pinned=child.pinned,
            last_used=child.last_used,
            parent=parent,
        )

        # Modificar child para que sea el suffix_node
        child.tokens = suffix_tokens
        child.parent = prefix_node

        # Actualizar relaciones en el árbol
        parent.children[prefix_tokens[0]] = prefix_node
        prefix_node.children[suffix_tokens[0]] = child

        # Manejo del KV cache en el split:
        # child mantiene su kv_cache original que representa la secuencia completa.
        # prefix_node obtiene un slice/trim del kv_cache si es factible.
        if child.kv_cache is not None:
            prefix_node.kv_cache = _slice_kv_for_prefix(child.kv_cache, len(suffix_tokens), trim_fn=self.trim_fn)

        return prefix_node

    def evict_lru(self, need_tokens: int) -> int:
        """
        Desaloja nodos hoja según política LRU hasta liberar al menos need_tokens.
        Respeta la protección de nodos pinned (Kripper Base Slot pattern).
        Retorna la cantidad total de tokens liberados.
        """
        freed = 0
        while (freed < need_tokens or self._total_tokens > self.max_tokens) and self._total_tokens > 0:
            candidates: list[RadixNode] = []
            pinned_candidates: list[RadixNode] = []

            for root in self._roots.values():
                self._collect_leaf_candidates(root, candidates, pinned_candidates)

            target: Optional[RadixNode] = None
            if candidates:
                # Seleccionar la hoja menos recientemente usada (LRU)
                candidates.sort(key=lambda n: n.last_used)
                target = candidates[0]
            elif pinned_candidates:
                # Fallback de seguridad: si todas las hojas están pinned y se requiere memoria,
                # desalojar la más vieja para evitar deadlock de memoria.
                logger.warning("[RADIX] Todos los nodos hoja están pinned. Aplicando fallback de desalojo.")
                pinned_candidates.sort(key=lambda n: n.last_used)
                target = pinned_candidates[0]
            else:
                # Ninguna hoja disponible (ej: todos en ref_count > 0 activo)
                break

            if target is None or target.parent is None:
                break

            # Desalojar el nodo hoja
            num_tokens = len(target.tokens)
            del target.parent.children[target.tokens[0]]
            self._total_tokens -= num_tokens
            freed += num_tokens
            target.kv_cache = None

            # Si el padre quedó sin hijos y sin KV propio, podarlo si no es raíz
            self._prune_empty_ancestors(target.parent)

        if freed > 0 and self.guard is not None:
            try:
                self.guard.force_clear_cache("radix_evict")
            except Exception:
                pass

        return freed

    def _collect_leaf_candidates(
        self,
        node: RadixNode,
        candidates: list[RadixNode],
        pinned_candidates: list[RadixNode],
    ) -> None:
        """Recorre recursivamente buscando nodos hoja con ref_count == 0."""
        for child in list(node.children.values()):
            if not child.children:
                # Es nodo hoja
                if child.ref_count == 0:
                    if child.pinned:
                        pinned_candidates.append(child)
                    else:
                        candidates.append(child)
            else:
                self._collect_leaf_candidates(child, candidates, pinned_candidates)

    def _prune_empty_ancestors(self, node: Optional[RadixNode]) -> None:
        """Elimina nodos internos intermedios que se quedaron sin hijos y sin kv_cache."""
        current = node
        while current is not None and current.parent is not None:
            if not current.children and current.kv_cache is None and current.ref_count == 0:
                parent = current.parent
                if current.tokens and current.tokens[0] in parent.children:
                    del parent.children[current.tokens[0]]
                self._total_tokens -= len(current.tokens)
                current = parent
            else:
                break

    def evict_unpinned(self) -> int:
        """Desaloja todas las hojas no pineadas inmediatamente (útil para liberar memoria GPU)."""
        freed = 0
        while True:
            candidates: list[RadixNode] = []
            dummy_pinned: list[RadixNode] = []
            for root in self._roots.values():
                self._collect_leaf_candidates(root, candidates, dummy_pinned)
            if not candidates:
                break
            for leaf in candidates:
                if leaf.parent is not None and leaf.tokens and leaf.tokens[0] in leaf.parent.children:
                    n_tok = len(leaf.tokens)
                    del leaf.parent.children[leaf.tokens[0]]
                    self._total_tokens -= n_tok
                    freed += n_tok
                    leaf.kv_cache = None
                    self._prune_empty_ancestors(leaf.parent)
        if freed > 0 and self.guard is not None:
            try:
                self.guard.force_clear_cache("radix_evict_unpinned")
            except Exception:
                pass
        return freed

    def contains_tokens(self, tokens: list[int], slot: str = "main") -> bool:
        """Verifica si la secuencia exacta de tokens está registrada en el árbol."""
        matched, _ = self.match_prefix(tokens, slot=slot)
        return len(matched) == len(tokens)


def _common_prefix_len(a: list[int], b: list[int]) -> int:
    """Calcula la longitud del prefijo común entre dos listas de enteros."""
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def _slice_kv_for_prefix(kv_cache: Any, tokens_to_trim: int, trim_fn: Optional[Callable[[Any, int], None]] = None) -> Any:
    """
    Intenta crear una versión recortada del kv_cache para un nodo prefijo tras un split.
    Soporta:
    1. Objetos mock de test con método .trim(n) o .slice(n).
    2. Función de trim externa inyectada (trim_fn) sobre deepcopy.
    3. mlx_lm trim_prompt_cache estándar si está disponible (exclusivamente para pure-KV).
    
    REGLA INVIOLABLE (Fail-Closed):
    Si el cache contiene alguna capa recurrente (ArraysCache / Marconi pattern),
    retorna None inmediatamente. El monkey-patch Marconi causa que can_trim_prompt_cache()
    devuelva True, pero su trim() ejecuta rollback() en lugar de recorte posicional,
    lo que corrompería silenciosamente el estado del nodo. Ante capas recurrentes,
    la única operación válida es checkpoint/rollback, por lo que el prefijo debe forzar re-prefill.
    """
    if kv_cache is None or tokens_to_trim <= 0:
        return kv_cache

    # ── GUARD FAIL-CLOSED: Capas recurrentes (ArraysCache / hybrid models) ──
    try:
        from ..cache_types import is_recurrent_layer
    except (ImportError, ValueError):
        try:
            from supermlx.cache_types import is_recurrent_layer
        except Exception:
            is_recurrent_layer = None

    if isinstance(kv_cache, list):
        for layer in kv_cache:
            if (is_recurrent_layer is not None and is_recurrent_layer(layer)) or type(layer).__name__ == "ArraysCache":
                logger.debug(
                    "[RADIX_SPLIT] Fail-closed: capa recurrente detectada (%s), rechazando slice para prefix node",
                    type(layer).__name__,
                )
                return None

    def _clean_metadata(obj: Any) -> Any:
        targets = [obj]
        if isinstance(obj, (list, tuple)) and len(obj) > 0:
            targets.append(obj[0])
        for target in targets:
            for attr in ("__model_offset__", "__model_prefix_hash__"):
                if hasattr(target, attr):
                    try:
                        delattr(target, attr)
                    except Exception:
                        setattr(target, attr, None)
        return obj

    # Caso 1: Soporte directo de objetos mock o custom de pruebas unitarias
    if hasattr(kv_cache, "trim") and callable(kv_cache.trim):
        try:
            copied = copy.deepcopy(kv_cache)
            copied.trim(tokens_to_trim)
            return _clean_metadata(copied)
        except Exception as err:
            logger.warning("[RADIX_SPLIT] Error recortando kv_cache custom: %s", err)

    # Caso 2: Función trim externa inyectada
    if trim_fn is not None:
        try:
            copied = copy.deepcopy(kv_cache)
            trim_fn(copied, tokens_to_trim)
            return _clean_metadata(copied)
        except Exception as err:
            logger.warning("[RADIX_SPLIT] Error en trim_fn inyectada: %s", err)

    # Caso 3: mlx_lm nativo (solo para caches pure-KV verificados)
    try:
        from mlx_lm.models.cache import can_trim_prompt_cache, trim_prompt_cache
        if isinstance(kv_cache, list):
            copied = copy.deepcopy(kv_cache)
            if can_trim_prompt_cache(copied):
                trim_prompt_cache(copied, tokens_to_trim)
                return _clean_metadata(copied)
    except ImportError:
        pass
    except Exception as err:
        logger.warning("[RADIX_SPLIT] Error en trim_prompt_cache nativo: %s", err)

    return None

