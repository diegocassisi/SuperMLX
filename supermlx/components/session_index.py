"""
[AI_DIRECTIVE]
ROL: Per-session KV cache index for prefix-based cache lookups
OBJETIVO: Trackear cache keys por sesión para lograr 97%+ hit rate entre turnos
ENTRADAS: SessionContext (session_id, parent_session_id), cache_key (token list)
SALIDAS: Best-match cache entry from LRUPromptCache
REGLAS INVIOLABLES:
- Thread-safe: todas las operaciones under prompt_cache_lock del caller
- Pruning automático de sesiones idle (configurable via max_idle_seconds)
- Anchor prefixes mantienen stable points para branch returns
SSoT: Este módulo es la única fuente de lógica de session-aware cache routing
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ..message_pipeline import SessionContext


class SessionIndex:
    """Per-session index for prompt cache lookups with anchor prefix tracking."""

    @dataclass
    class SessionState:
        parent_session_id: Optional[str]
        touched_at: float
        keys: deque
        anchors: deque

    def __init__(self, max_entries_per_session: int, max_idle_seconds: int):
        self._max_entries_per_session = max(1, max_entries_per_session)
        self._max_idle_seconds = max_idle_seconds
        self._max_anchor_entries = max(4, self._max_entries_per_session * 4)
        self._anchor_stride_tokens = 2048
        self._sessions: Dict[str, SessionIndex.SessionState] = {}

    def _prune_idle(self) -> None:
        if self._max_idle_seconds <= 0:
            return
        now = time.time()
        stale = [
            session_id
            for session_id, state in self._sessions.items()
            if (now - state.touched_at) > self._max_idle_seconds
        ]
        for session_id in stale:
            self._sessions.pop(session_id, None)

    @staticmethod
    def _lcp_len(a: List[int], b: Tuple[int, ...]) -> int:
        limit = min(len(a), len(b))
        idx = 0
        while idx < limit and a[idx] == b[idx]:
            idx += 1
        return idx

    @staticmethod
    def _append_unique_bounded(
        queue: deque, key_tuple: Tuple[int, ...], limit: int
    ) -> None:
        try:
            queue.remove(key_tuple)
        except ValueError:
            pass
        queue.append(key_tuple)
        while len(queue) > limit:
            queue.popleft()

    def register_cache_key(
        self, session_ctx: "SessionContext", cache_key: List[int]
    ) -> None:
        self._prune_idle()
        session_id = (session_ctx.session_id or "").strip()
        if not session_id:
            return

        state = self._sessions.get(session_id)
        if state is None:
            state = self.SessionState(
                parent_session_id=session_ctx.parent_session_id,
                touched_at=time.time(),
                keys=deque(),
                anchors=deque(),
            )
            self._sessions[session_id] = state

        if (
            session_ctx.parent_session_id
            and session_ctx.parent_session_id != session_id
        ):
            state.parent_session_id = session_ctx.parent_session_id
        state.touched_at = time.time()

        key_tuple = tuple(cache_key)
        self._append_unique_bounded(
            state.keys, key_tuple, self._max_entries_per_session
        )

        # Keep additional anchor prefixes so branch returns can reuse older stable
        # points even when recent keys are from another branch/tool-heavy turn.
        should_add_anchor = False
        if not state.anchors:
            should_add_anchor = True
        else:
            last_anchor_len = len(state.anchors[-1])
            if (len(key_tuple) - last_anchor_len) >= self._anchor_stride_tokens:
                should_add_anchor = True
        if should_add_anchor:
            self._append_unique_bounded(
                state.anchors, key_tuple, self._max_anchor_entries
            )

    def select_best_cache(
        self,
        model_name: str,
        prompt_tokens: List[int],
        session_ctx: "SessionContext",
        prompt_cache_store: Any,  # LRUPromptCache — avoids circular import
    ):
        """Always use global cache; session/conversation/thread ID is ignored for lookup."""
        prompt_cache_store.prune_expired()
        self._prune_idle()
        selected = prompt_cache_store.fetch_nearest_cache(model_name, prompt_tokens)
        return (*selected, "global")
