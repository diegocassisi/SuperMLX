"""
[AI_DIRECTIVE]
ROL: RequestContext — contrato de estado por request para el pipeline de 4 fases de Fase D.
OBJETIVO: Encapsular todo el estado mutado y compartido a lo largo del ciclo de vida de un chat completion.
ENTRADAS: Parámetros del request HTTP (body, headers, streaming flags).
SALIDAS: Objeto de contexto que transita por preprocess -> cache_lookup -> generate -> postprocess.
REGLAS INVIOLABLES:
- Un solo RequestContext activo a la vez (single-user local).
- No sustituye a ServerState (que gestiona recursos globales persistentes).
- Tipado estricto en todos los atributos.
SSoT: Este módulo es la única definición de RequestContext para server3.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


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

    # ── tokens y prompts renderizados (preprocess) ────────────
    model_tokens: list[int] = field(default_factory=list)
    prompt_tokens: Optional[list[int]] = None
    prompt: str = ""
    cache_prompt: str = ""
    enable_thinking: bool = False
    is_compact: bool = False
    is_housekeeping: bool = False
    is_ephemeral: bool = False
    session_ctx: Any = None
    housekeeping_conv_model_boundary: Optional[int] = None
    housekeeping_conv_prompt_tokens: Optional[list[int]] = None
    pipeline_timings: dict = field(default_factory=dict)

    # ── vlm opcionales (preprocess) ───────────────────────────
    vlm_pixel_values: Any = None
    vlm_mask: Any = None
    vlm_kwargs: Any = None

    # ── cache_lookup ───────────────────────────────────────────
    cache_key: Optional[list[int]] = None
    skip_cache_store: bool = False
    rest_count: int = 0                 # tokens no cubiertos por cache hit
    rest_tokens: list[int] = field(default_factory=list)
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
