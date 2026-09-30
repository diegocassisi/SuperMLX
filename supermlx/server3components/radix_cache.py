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
    slot_type: Literal["main", "compact"] = "main"                  # Kripper Dual-Slot
    pinned: bool = False                                            # Inmune a eviction LRU si True
    parent: Optional["RadixNode"] = None                            # Referencia al nodo padre para eviction/split
    kv_nbytes: int = 0                                              # Bytes de arrays que posee kv_cache (medido al asignar)


def _kv_nbytes(kv_cache: Any) -> int:
    """
    Suma los bytes de todos los arrays alcanzables desde kv_cache (keys/values, estado
    recurrente, snapshots de checkpoint). Cuenta buffers reservados, no sólo el offset
    ocupado: es la memoria que el nodo retiene. Deduplica por id() dentro del objeto.
    """
    if kv_cache is None:
        return 0
    total = 0
    seen: set[int] = set()
    stack: list[Any] = [kv_cache]
    while stack:
        obj = stack.pop()
        if obj is None or isinstance(obj, (int, float, str, bool, bytes)):
            continue
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        nbytes = getattr(obj, "nbytes", None) if hasattr(obj, "shape") else None
        if isinstance(nbytes, int):
            total += nbytes
        elif isinstance(obj, dict):
            stack.extend(obj.values())
        elif isinstance(obj, (list, tuple)):
            stack.extend(obj)
        elif hasattr(obj, "__dict__"):
            stack.extend(vars(obj).values())
    return total


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
        # Presupuesto de memoria (desactivado hasta configure_memory_budget)
        self._kv_budget_bytes: Optional[int] = None   # budget Metal - pesos del modelo
        self._scratch_reserve_bytes: int = 0          # scratch de prefill/decode fuera del árbol
        self._newest: Optional[RadixNode] = None      # nodo del último insert: nunca se desaloja
        self._log_fn: Optional[Callable[[str], None]] = None

    def configure_memory_budget(
        self,
        metal_budget_bytes: int,
        weights_bytes: int,
        scratch_reserve_bytes: int = 0,
        log_fn: Optional[Callable[[str], None]] = None,
    ) -> None:
        """
        Activa el desalojo por memoria. Lo que el árbol puede retener es:
            metal_budget - weights - (copia de trabajo + scratch)
        donde la copia de trabajo es el tamaño del último KV insertado (el próximo turno
        lo deep-copia vía match_prefix y lo extiende).
        """
        self._kv_budget_bytes = max(0, int(metal_budget_bytes) - int(weights_bytes))
        self._scratch_reserve_bytes = max(0, int(scratch_reserve_bytes))
        self._log_fn = log_fn

    @property
    def owned_bytes(self) -> int:
        """Bytes de KV que retienen los nodos del árbol (todos los slots)."""
        total = 0
        stack = list(self._roots.values())
        while stack:
            node = stack.pop()
            total += node.kv_nbytes
            stack.extend(node.children.values())
        return total

    def memory_limit_bytes(self) -> Optional[int]:
        """Límite actual para owned_bytes, o None si el presupuesto no está configurado."""
        if self._kv_budget_bytes is None:
            return None
        working_copy = self._newest.kv_nbytes if self._newest is not None else 0
        return max(0, self._kv_budget_bytes - self._scratch_reserve_bytes - working_copy)

    @staticmethod
    def _set_kv(node: RadixNode, kv_cache: Any) -> None:
        node.kv_cache = kv_cache
        node.kv_nbytes = _kv_nbytes(kv_cache)

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

        # Hacer lugar ANTES del deepcopy: la copia nueva no debe coexistir con nodos
        # que igual se van a desalojar (evita un pico transitorio de un KV extra).
        if self._kv_budget_bytes is not None and kv_cache is not None:
            incoming = _kv_nbytes(kv_cache)
            limit_after = max(0, self._kv_budget_bytes - self._scratch_reserve_bytes - incoming)
            self.evict_to_bytes(max(0, limit_after - incoming), reason="pre_insert")

        kv_to_store = copy.deepcopy(kv_cache) if kv_cache is not None else None
        stored_node: Optional[RadixNode] = None
        node = self.get_root(slot)
        remaining = list(tokens)

        while remaining:
            first_token = remaining[0]
            if first_token not in node.children:
                # Caso: no hay hijo con este token inicial -> crear hoja con el sufijo restante
                new_leaf = RadixNode(
                    tokens=remaining,
                    slot_type=slot,  # type: ignore[arg-type]
                    pinned=pinned,
                    last_used=time.time(),
                    parent=node,
                )
                self._set_kv(new_leaf, kv_to_store)
                stored_node = new_leaf
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
                        slot_type=slot,  # type: ignore[arg-type]
                        pinned=pinned,
                        last_used=time.time(),
                        parent=prefix_node,
                    )
                    self._set_kv(new_leaf, kv_to_store)
                    stored_node = new_leaf
                    prefix_node.children[remaining[0]] = new_leaf
                    self._total_tokens += len(remaining)
                else:
                    # El nuevo camino coincide exactamente con el prefix recién creado
                    self._set_kv(prefix_node, kv_to_store)
                    stored_node = prefix_node
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
                    self._set_kv(node, kv_to_store)
                    stored_node = node
                    node.pinned = pinned or node.pinned
                    node.last_used = time.time()
                    break

        if stored_node is not None and stored_node.kv_cache is not None:
            self._newest = stored_node

        # Mantener presupuesto de tokens
        if self._total_tokens > self.max_tokens:
            self.evict_lru(self._total_tokens - self.max_tokens)

        # Mantener presupuesto de memoria
        limit = self.memory_limit_bytes()
        if limit is not None:
            self.evict_to_bytes(limit, reason="post_insert")
            if self._log_fn is not None:
                try:
                    self._log_fn(
                        f"[RADIX_MEM] insert={len(tokens)} tok "
                        f"node={(self._newest.kv_nbytes if self._newest else 0) / 1e9:.3f}GB | "
                        f"owned={self.owned_bytes / 1e9:.3f}GB limit={limit / 1e9:.3f}GB | "
                        f"kv_budget={self._kv_budget_bytes / 1e9:.2f}GB "
                        f"scratch={self._scratch_reserve_bytes / 1e9:.2f}GB | "
                        f"kv_nodes={self._count_kv_nodes()}"
                    )
                except Exception:
                    pass

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
            self._set_kv(prefix_node, _slice_kv_for_prefix(child.kv_cache, len(suffix_tokens), trim_fn=self.trim_fn))

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
                # Ninguna hoja desalojable disponible
                break

            if target is None or target.parent is None:
                break

            # Desalojar el nodo hoja
            num_tokens = len(target.tokens)
            del target.parent.children[target.tokens[0]]
            self._total_tokens -= num_tokens
            freed += num_tokens
            self._set_kv(target, None)

            # Si el padre quedó sin hijos y sin KV propio, podarlo si no es raíz
            self._prune_empty_ancestors(target.parent)

        if freed > 0 and self.guard is not None:
            try:
                self.guard.force_clear_cache("radix_evict")
            except Exception:
                pass

        return freed

    def evict_to_bytes(self, limit_bytes: int, reason: str = "") -> int:
        """
        Libera KV (LRU) hasta que owned_bytes <= limit_bytes. Retorna bytes liberados.

        A diferencia de evict_lru, considera también nodos internos: en una conversación
        lineal cada turno deja un nodo intermedio con el KV completo de su prefijo, y esos
        nodos nunca llegan a ser hoja. En un nodo interno sólo se suelta el KV (la
        estructura queda para los hijos); una hoja se elimina del árbol.
        Nunca toca el nodo recién insertado. No hay ref_count: match_prefix entrega un
        deepcopy y el desalojo sólo corre dentro de insert() bajo model_lock, por lo que
        ninguna generación en curso referencia un nodo del árbol.
        Pinned sólo se desaloja si no queda otra opción (igual que evict_lru).
        """
        freed = 0
        evicted_nodes = 0
        owned = self.owned_bytes
        while owned > limit_bytes:
            candidates: list[RadixNode] = []
            pinned_candidates: list[RadixNode] = []
            stack = list(self._roots.values())
            while stack:
                node = stack.pop()
                stack.extend(node.children.values())
                if (
                    node.parent is None
                    or node.kv_cache is None
                    or node is self._newest
                ):
                    continue
                (pinned_candidates if node.pinned else candidates).append(node)

            pool = candidates or pinned_candidates
            if not pool:
                logger.warning(
                    "[RADIX_MEM] Sin candidatos desalojables: owned=%.3fGB > limit=%.3fGB (reason=%s)",
                    owned / 1e9, limit_bytes / 1e9, reason,
                )
                break
            if not candidates:
                logger.warning("[RADIX_MEM] Sólo quedan nodos pinned. Aplicando fallback de desalojo.")

            # LRU; a igualdad de timestamp, el más superficial primero (menos contexto reutilizable)
            target = min(pool, key=lambda n: (n.last_used, _node_depth(n)))
            node_bytes = target.kv_nbytes
            if target.children:
                self._set_kv(target, None)
            else:
                del target.parent.children[target.tokens[0]]
                self._total_tokens -= len(target.tokens)
                self._set_kv(target, None)
                self._prune_empty_ancestors(target.parent)
            freed += node_bytes
            evicted_nodes += 1
            owned -= node_bytes

        if freed > 0:
            logger.info(
                "[RADIX_MEM] evict_to_bytes(%s): %d nodos, %.3fGB liberados, owned=%.3fGB limit=%.3fGB",
                reason, evicted_nodes, freed / 1e9, owned / 1e9, limit_bytes / 1e9,
            )
            if self._log_fn is not None:
                try:
                    self._log_fn(
                        f"[RADIX_MEM] evicted {evicted_nodes} node(s) ({reason}) | freed={freed / 1e9:.3f}GB | "
                        f"owned={owned / 1e9:.3f}GB limit={limit_bytes / 1e9:.3f}GB"
                    )
                except Exception:
                    pass
            if self.guard is not None:
                try:
                    self.guard.force_clear_cache("radix_evict_bytes")
                except Exception:
                    pass
        return freed

    def _count_kv_nodes(self) -> int:
        count = 0
        stack = list(self._roots.values())
        while stack:
            node = stack.pop()
            if node.kv_cache is not None:
                count += 1
            stack.extend(node.children.values())
        return count

    def _collect_leaf_candidates(
        self,
        node: RadixNode,
        candidates: list[RadixNode],
        pinned_candidates: list[RadixNode],
    ) -> None:
        """Recorre recursivamente buscando nodos hoja desalojables."""
        for child in list(node.children.values()):
            if not child.children:
                # Es nodo hoja (el recién insertado nunca es candidato)
                if child is not self._newest:
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
            if not current.children and current.kv_cache is None:
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
                    self._set_kv(leaf, None)
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


def _node_depth(node: RadixNode) -> int:
    """Tokens desde la raíz hasta el final de node."""
    depth = 0
    current: Optional[RadixNode] = node
    while current is not None:
        depth += len(current.tokens)
        current = current.parent
    return depth


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

