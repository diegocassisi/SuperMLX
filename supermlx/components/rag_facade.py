"""
[AI_DIRECTIVE]
ROL: Facade para RAG enrichment y prompt compression (server)
OBJETIVO: Centralizar la inicialización, config, y estado de rag_enricher
          para que server.py solo importe este módulo
ENTRADAS: env vars (FEATURE_RAG_ENRICHMENT, RAG_WORKSPACE_ROOT, etc.)
SALIDAS: is_rag_available(), is_compressor_available(), enrich_messages(), compress_messages()
REGLAS INVIOLABLES:
- No duplicar lógica de rag_enricher — solo wrappear
- Exponer estado via funciones, no globals
- init() es idempotente
SSoT: rag_enricher.py es la implementación, este módulo es el wiring para server
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from ..config import SETTINGS

logger = logging.getLogger(__name__)

# ── Internal state ───────────────────────────────────────────────────────────
_rag_module: Any = None
_rag_available: bool = False
_compressor_module: Any = None
_compressor_available: bool = False
_terminal_status_fn: Optional[Callable] = None

# ── Config (from SETTINGS) ───────────────────────────────────────────────────
FEATURE_RAG_ENRICHMENT = SETTINGS.feature_rag_enrichment
FEATURE_RAG_WORKSPACE_ROOT = SETTINGS.rag_workspace_root
FEATURE_RAG_RELEVANCE_THRESHOLD = 1.6  # Qwen3-Embed asymmetric

FEATURE_COMPRESSOR = SETTINGS.feature_compressor
FEATURE_COMPRESSION_THRESHOLD = SETTINGS.compression_threshold
FEATURE_COMPRESSION_GUARD = SETTINGS.compression_guard


def init(*, terminal_status_fn: Optional[Callable] = None) -> Dict[str, Any]:
    """Initialize RAG enricher and compressor. Returns status dict for logging.

    Idempotent — safe to call multiple times.
    """
    global _rag_module, _rag_available, _compressor_module, _compressor_available
    global _terminal_status_fn

    if terminal_status_fn is not None:
        _terminal_status_fn = terminal_status_fn

    def _log(icon: str, msg: str) -> None:
        if _terminal_status_fn:
            _terminal_status_fn(icon, msg)
        else:
            logger.info("%s %s", icon, msg)

    status: Dict[str, Any] = {
        "rag_available": False,
        "compressor_available": False,
    }

    # ── RAG Enricher ──────────────────────────────────────────────────────
    if FEATURE_RAG_ENRICHMENT:
        try:
            from supermlx import rag_enricher as _rag_mod
            _rag_mod.RELEVANCE_THRESHOLD = FEATURE_RAG_RELEVANCE_THRESHOLD
            _rag_module = _rag_mod
            _rag_available = True

            _log("🔍", (
                f"RAG Enricher: ACTIVATED"
                f" | top_k={_rag_mod.TOP_K}"
                f" | relevance_threshold={FEATURE_RAG_RELEVANCE_THRESHOLD}"
                f" | embedding={_rag_mod.EMBED_MODEL_NAME} (MPS)"
            ))

            if FEATURE_RAG_WORKSPACE_ROOT:
                try:
                    _log("🔍", f"RAG: indexando workspace | root={FEATURE_RAG_WORKSPACE_ROOT}")
                    _rag_mod.init(codebase_root=FEATURE_RAG_WORKSPACE_ROOT)
                except Exception as e:
                    _log("⚠️", f"RAG: workspace init failed ({e}) — enricher active but index empty")
            else:
                _log("ℹ️", "RAG: no RAG_WORKSPACE_ROOT set — enricher active but index empty")

            status["rag_available"] = True
        except ImportError as e:
            _log("⚠️", f"RAG Enricher: import failed ({e}) — DISABLED")
        except Exception as e:
            _log("⚠️", f"RAG Enricher: init failed ({e}) — DISABLED")
    else:
        _log("ℹ️", "RAG Enricher: DISABLED (FEATURE_RAG_ENRICHMENT=False)")

    # ── Compressor ────────────────────────────────────────────────────────
    if FEATURE_COMPRESSOR:
        try:
            if _rag_module is not None:
                _compressor_module = _rag_module
            else:
                from supermlx import rag_enricher as _comp_mod
                _compressor_module = _comp_mod
            _compressor_available = True

            _log("📦", (
                f"Compressor: ACTIVATED | threshold={FEATURE_COMPRESSION_THRESHOLD} tok | "
                f"guard={FEATURE_COMPRESSION_GUARD} msgs"
            ))

            # Preload reranker if available
            try:
                _compressor_module._load_reranker()
            except Exception:
                pass

            status["compressor_available"] = True
        except ImportError as e:
            _log("⚠️", f"Compressor: import failed ({e}) — DISABLED")
        except Exception as e:
            _log("⚠️", f"Compressor: init failed ({e}) — DISABLED")
    else:
        _log("ℹ️", "Compressor: DISABLED (FEATURE_COMPRESSOR=False)")

    return status


# ── Public API ───────────────────────────────────────────────────────────────

def is_rag_available() -> bool:
    return _rag_available


def is_compressor_available() -> bool:
    return _compressor_available


def get_rag_module() -> Any:
    """Return the rag_enricher module (for enrich_messages calls)."""
    return _rag_module


def get_compressor_module() -> Any:
    """Return the compressor module (for compress_messages calls)."""
    return _compressor_module


def get_compression_threshold() -> int:
    return FEATURE_COMPRESSION_THRESHOLD


def get_compression_guard() -> int:
    return FEATURE_COMPRESSION_GUARD


def get_relevance_threshold() -> float:
    return FEATURE_RAG_RELEVANCE_THRESHOLD
