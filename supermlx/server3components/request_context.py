"""
RequestContext — contrato de estado para el pipeline de Fase D.

No reemplaza ServerState (state.py): ServerState es estado *global* del
server (model, tokenizer, locks). RequestContext es estado *por request*,
vive y muere con _handle_chat_completion.

Single-user, sin batching: un solo RequestContext activo a la vez, no hay
queue ni scheduling entre requests concurrentes.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class RequestContext:
    # ── identidad ──────────────────────────────────────────────
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    t_start: float = field(default_factory=time.time)

    # ── input crudo (seteado en preprocess) ───────────────────
    raw_messages: list[dict] = field(default_factory=list)
    body: dict = field(default_factory=dict)
    is_streaming: bool = False
    is_anthropic: bool = False

    # ── post-canonicalización ─────────────────────────────────
    canonical_messages: Optional[list[dict]] = None
    session_id: Optional[str] = None
    tool_calls: list[dict] = field(default_factory=list)
    healed: bool = False

    # ── cache_lookup ───────────────────────────────────────────
    prompt_tokens: Optional[list[int]] = None
    rest_count: int = 0                 # tokens no cubiertos por cache hit
    cache_hit_ratio: Optional[float] = None
    prompt_cache: Any = None            # handle al objeto de mlx cache

    # ── generate ───────────────────────────────────────────────
    cache_has_recurrent_layers: bool = False  # True → postprocess restaura checkpoint y descarta generated_tokens
    checkpoint: Any = None              # hybrid checkpoint pre-generación
    generated_tokens: list[int] = field(default_factory=list)
    finish_reason: Optional[str] = None

    # ── postprocess ────────────────────────────────────────────
    response_text: str = ""
    usage: dict = field(default_factory=dict)

    def elapsed_ms(self) -> float:
        return (time.time() - self.t_start) * 1000
