# SPDX-License-Identifier: MIT
"""
[AI_DIRECTIVE]
ROL: Gestor de Staging Seguro Multiturno para Housekeeping (Hermes / Agentes)
OBJETIVO: Preservar el KV Cache entre llamadas consecutivas de herramientas/housekeeping,
          aislando el estado del prompt cache principal y aplicando dual-space math.
ENTRADAS: Session ID, model name, canonical prompt_tokens, model_tokens, prompt_cache,
          checkpoints de generación y snapshots de cache.
SALIDAS: Instancias de HousekeepingStagingEntry, rest_tokens ajustados por model_offset,
          y restauración fail-closed hacia PROMPT_CACHE.
REGLAS INVIOLABLES:
- Prohibido usar matched_prefix_len para indexar model_tokens (dual-space invariant).
- Prohibido try/except silencioso sin logger.
- Obligatorio fail-closed: ante falla de restore_hybrid_generation_checkpoint o rollback,
  descartar staging y nunca reinsertar datos corruptos.
- Prohibido hardcoding de herramientas, URLs o identificadores de clientes.
SSoT: Este módulo es la única fuente de verdad para el ciclo de vida de staging de housekeeping.
"""

from dataclasses import dataclass
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from .cache_engine import (
    HybridGenerationCheckpoint,
    restore_hybrid_generation_checkpoint,
    rollback_arrays_cache,
)

logger = logging.getLogger(__name__)


@dataclass
class HousekeepingStagingEntry:
    """Represents a session-isolated staged cache during multi-turn housekeeping."""

    session_id: str
    model_name: str
    base_snapshot: Optional[Dict[str, Any]]
    base_tokens: Optional[Tuple[int, ...]]
    staged_cache: Any
    staged_canonical_tokens: Tuple[int, ...]
    staged_model_tokens: Tuple[int, ...]
    staged_model_offset: int
    created_at: float
    last_touched_at: float
    turn_count: int = 1


class HousekeepingStagingManager:
    """Manages multi-turn housekeeping staging caches per session and model.

    Enforces dual-space token alignment, fail-closed checkpoint validation,
    and automatic TTL/LRU capacity pruning.
    """

    def __init__(self) -> None:
        self._staging: Dict[Tuple[str, str], HousekeepingStagingEntry] = {}

    def get_entry(
        self, session_id: str, model_name: str
    ) -> Optional[HousekeepingStagingEntry]:
        """Retrieve active staging entry for a given session and model."""
        if not session_id or not model_name:
            return None
        return self._staging.get((session_id, model_name))

    def has_staging(self, session_id: str, model_name: str) -> bool:
        """Check if an active staging entry exists."""
        if not session_id or not model_name:
            return False
        return (session_id, model_name) in self._staging

    def check_continuation(
        self,
        session_id: str,
        model_name: str,
        prompt_tokens: List[int],
        model_tokens: List[int],
    ) -> Optional[Tuple[HousekeepingStagingEntry, int, int]]:
        """Validate continuation for multi-turn housekeeping using Dual-Space Math.

        Verifies:
        1. Canonical space: prompt_tokens must start with entry.staged_canonical_tokens.
        2. Model space: model_tokens[:entry.staged_model_offset] must match
           entry.staged_model_tokens[:entry.staged_model_offset].

        Returns:
            Tuple of (entry, matched_canonical_len, staged_model_offset) on valid hit,
            or None if mismatch / no entry.
        """
        if not session_id or not model_name:
            return None

        entry = self._staging.get((session_id, model_name))
        if entry is None:
            return None

        canon_len = len(entry.staged_canonical_tokens)
        canon_match = (
            len(prompt_tokens) >= canon_len
            and tuple(prompt_tokens[:canon_len]) == entry.staged_canonical_tokens
        )

        model_off = entry.staged_model_offset
        model_match = (
            0 < model_off <= len(model_tokens)
            and tuple(model_tokens[:model_off])
            == entry.staged_model_tokens[:model_off]
        )

        if canon_match and model_match:
            entry.last_touched_at = time.time()
            entry.turn_count += 1
            logger.info(
                "[DATA] Housekeeping staging continuation HIT | session=%s turn=%d | "
                "canon_match=%d model_off=%d",
                session_id,
                entry.turn_count,
                canon_len,
                model_off,
            )
            return entry, canon_len, model_off

        logger.info(
            "[DECISION] Housekeeping staging MISMATCH | session=%s | "
            "canon_match=%s model_match=%s",
            session_id,
            canon_match,
            model_match,
        )
        return None

    def store_or_update(
        self,
        session_id: str,
        model_name: str,
        staged_cache: Any,
        prompt_tokens: List[int],
        model_tokens: List[int],
        hybrid_checkpoint: Optional[HybridGenerationCheckpoint],
        base_snapshot: Optional[Dict[str, Any]] = None,
        base_tokens: Optional[Tuple[int, ...]] = None,
    ) -> bool:
        """Restore hybrid checkpoint (fail-closed) and update staging state.

        Returns True on successful staging, False on checkpoint restoration failure.
        """
        if not session_id or not model_name or staged_cache is None:
            return False

        key = (session_id, model_name)
        existing = self._staging.get(key)

        # 1. Restore hybrid checkpoint to pre-generation state
        if hybrid_checkpoint is not None:
            restored = restore_hybrid_generation_checkpoint(
                staged_cache, hybrid_checkpoint
            )
            if not restored:
                logger.error(
                    "[ERROR] restore_hybrid_generation_checkpoint FAILED (fail-closed) | "
                    "session=%s model=%s. Discarding staging.",
                    session_id,
                    model_name,
                )
                self._staging.pop(key, None)
                return False

        # 2. Determine model-space offset
        if hybrid_checkpoint is not None and hybrid_checkpoint.model_offset > 0:
            model_off = hybrid_checkpoint.model_offset
        elif model_tokens:
            model_off = len(model_tokens) - 1
        else:
            model_off = len(prompt_tokens) - 1

        # 3. Preserve original base snapshot and tokens from turn 1
        resolved_base_snapshot = (
            existing.base_snapshot if existing is not None else base_snapshot
        )
        resolved_base_tokens = (
            existing.base_tokens if existing is not None else base_tokens
        )
        resolved_created_at = (
            existing.created_at if existing is not None else time.time()
        )
        turn_cnt = (existing.turn_count + 1) if existing is not None else 1

        self._staging[key] = HousekeepingStagingEntry(
            session_id=session_id,
            model_name=model_name,
            base_snapshot=resolved_base_snapshot,
            base_tokens=resolved_base_tokens,
            staged_cache=staged_cache,
            staged_canonical_tokens=tuple(prompt_tokens),
            staged_model_tokens=tuple(model_tokens)
            if model_tokens
            else tuple(prompt_tokens),
            staged_model_offset=model_off,
            created_at=resolved_created_at,
            last_touched_at=time.time(),
            turn_count=turn_cnt,
        )

        logger.info(
            "[RESULT] Housekeeping staging stored | session=%s turn=%d model_off=%d",
            session_id,
            turn_cnt,
            model_off,
        )
        return True

    def close_and_restore(
        self, session_id: str, model_name: str, prompt_cache_store: Any
    ) -> bool:
        """Close staging entry, rollback to base snapshot, and re-insert into prompt_cache_store.

        Returns True if base was successfully restored to the store, False otherwise.
        """
        if not session_id or not model_name:
            return False

        key = (session_id, model_name)
        entry = self._staging.pop(key, None)
        if entry is None:
            return False

        if entry.base_snapshot is None or entry.base_tokens is None:
            logger.info(
                "[DECISION] Housekeeping staging closed (no base snapshot to restore) | session=%s",
                session_id,
            )
            return False

        try:
            restored = rollback_arrays_cache(entry.staged_cache, entry.base_snapshot)
            if restored:
                if prompt_cache_store is not None and hasattr(
                    prompt_cache_store, "insert_cache"
                ):
                    prompt_cache_store.insert_cache(
                        entry.model_name,
                        list(entry.base_tokens),
                        entry.staged_cache,
                    )
                logger.info(
                    "[RESULT] Housekeeping staging rolled back & restored to base | "
                    "session=%s tokens=%d",
                    session_id,
                    len(entry.base_tokens),
                )
                return True
            else:
                logger.error(
                    "[ERROR] Housekeeping staging rollback_arrays_cache FAILED | "
                    "session=%s. Base cache lost.",
                    session_id,
                )
                return False
        except Exception as exc:
            logger.error(
                "[ERROR] Exception in close_and_restore | session=%s: %s",
                session_id,
                exc,
            )
            return False

    def discard(self, session_id: str, model_name: str) -> None:
        """Discard staging without restoring to prompt_cache_store."""
        if session_id and model_name:
            self._staging.pop((session_id, model_name), None)

    def prune(
        self,
        prompt_cache_store: Optional[Any] = None,
        now: Optional[float] = None,
        ttl_seconds: float = 300.0,
        max_entries: int = 2,
    ) -> int:
        """Prune expired and excess staging entries under prompt_cache_lock.

        Returns the number of pruned entries.
        """
        if not self._staging:
            return 0

        if now is None:
            now = time.time()

        pruned_count = 0

        # 1. TTL Pruning
        expired_keys = [
            k for k, v in self._staging.items() if (now - v.last_touched_at) > ttl_seconds
        ]
        for k in expired_keys:
            entry = self._staging.pop(k, None)
            if entry is not None:
                pruned_count += 1
                self._safely_restore_entry(entry, prompt_cache_store, reason="TTL")

        # 2. Capacity Pruning (LRU by last_touched_at)
        while len(self._staging) > max_entries:
            oldest_key = min(
                self._staging.keys(), key=lambda k: self._staging[k].last_touched_at
            )
            entry = self._staging.pop(oldest_key, None)
            if entry is not None:
                pruned_count += 1
                self._safely_restore_entry(
                    entry, prompt_cache_store, reason="CAPACITY"
                )

        return pruned_count

    def _safely_restore_entry(
        self,
        entry: HousekeepingStagingEntry,
        prompt_cache_store: Optional[Any],
        reason: str,
    ) -> None:
        """Attempt rollback and re-insertion of a pruned staging entry."""
        if entry.base_snapshot is None or entry.base_tokens is None:
            logger.info(
                "[DECISION] Pruned entry without base snapshot (%s) | session=%s",
                reason,
                entry.session_id,
            )
            return

        try:
            restored = rollback_arrays_cache(entry.staged_cache, entry.base_snapshot)
            if (
                restored
                and prompt_cache_store is not None
                and hasattr(prompt_cache_store, "insert_cache")
            ):
                prompt_cache_store.insert_cache(
                    entry.model_name,
                    list(entry.base_tokens),
                    entry.staged_cache,
                )
                logger.info(
                    "[RESULT] Pruned entry restored to base (%s) | session=%s tokens=%d",
                    reason,
                    entry.session_id,
                    len(entry.base_tokens),
                )
            elif not restored:
                logger.warning(
                    "[ERROR] Pruned entry rollback FAILED (%s) | session=%s",
                    reason,
                    entry.session_id,
                )
        except Exception as exc:
            logger.error(
                "[ERROR] Exception during pruned entry restoration (%s) | session=%s: %s",
                reason,
                entry.session_id,
                exc,
            )
