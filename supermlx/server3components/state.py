"""
[AI_DIRECTIVE]
ROL: Contrato de dependencias para módulos extraídos de server2.py
OBJETIVO: Centralizar las ~15 variables globales compartidas en un dataclass tipado,
          eliminando acoplamiento implícito entre componentes
ENTRADAS: Globals de server2.py (model, tokenizer, caches, guards, etc.)
SALIDAS: ServerState dataclass que los módulos reciben en lugar de acceder a globals
REGLAS INVIOLABLES:
- from __future__ import annotations OBLIGATORIO (evita NameError en forward refs)
- Este dataclass es SOLO para construcción directa (ServerState(model=..., ...))
- NO usar typing.get_type_hints() en runtime — falla porque LRUPromptCache/SessionIndex
  no existen en el namespace de state.py
- Agregar campos nuevos AL FINAL para minimizar impacto en módulos existentes
SSoT: plan_server2_refactor.md v5, FASE 0.5
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import threading


@dataclass
class ServerState:
    """Shared state contract for extracted server2 modules.

    IMPORTANT: This dataclass uses forward-referenced types (strings in annotations)
    thanks to `from __future__ import annotations`. This means:
    - Direct construction works: ServerState(model=model_obj, ...)
    - typing.get_type_hints(ServerState) WILL FAIL at runtime (the types like
      LRUPromptCache don't exist in this module's namespace)
    - This is intentional — we only need construction, not runtime introspection
    """

    # ── Core model ────────────────────────────────────────────────────────
    model: Any                          # MLX model (loaded via mlx_lm or mlx_vlm)
    tokenizer: Any                      # HuggingFace tokenizer
    model_lock: threading.Lock          # Serializes all model access (no batching)
    is_vlm: bool                        # True if vision-language model detected
    processor: Any                      # VLM processor (None for text-only models)

    # ── Cache subsystem ───────────────────────────────────────────────────
    prompt_cache: "LRUPromptCache"      # MAIN KV cache store (LRU max=2)
    session_index: "SessionIndex"       # Per-session prefix tracking with block-hash
    prompt_cache_lock: threading.Lock   # Protects prompt_cache mutations

    # ── Subsystems ────────────────────────────────────────────────────────
    guard: Any                          # MetalMemoryGuard (pre/post request memory mgmt)
    tpc: Any                            # ToolPrefixCache (pre-computed system+tools KV)
    dpc: Any                            # DPCState (disk persistence controller)
    healing_store: Any                  # HealingStore (conversation healing snapshots)
    thinking_tracker: Any               # ThinkingTracker (<think> tag tracking per request)
    tool_call_tracker: Any              # ToolCallTracker (infinite loop detection)

    # ── Config ────────────────────────────────────────────────────────────
    settings: Any                       # SETTINGS singleton (env-driven config)

    # ── Concurrency ───────────────────────────────────────────────────────
    console_lock: threading.Lock        # Protects terminal output
