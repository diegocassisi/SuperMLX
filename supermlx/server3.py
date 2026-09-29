# SPDX-License-Identifier: MIT
"""
[AI_DIRECTIVE]
ROL: SuperMLX — Production-Grade MLX Inference Server for Agentic AI
OBJETIVO: Servidor de inferencia OpenAI/Anthropic compatible con soporte MoE, KV cache persistente, Dual-Phase Sampling y protección OOM
ENTRADAS: Solicitudes HTTP (/v1/chat/completions, /v1/messages) con prompts, herramientas y tokens
SALIDAS: Streaming de eventos SSE o payloads JSON de inferencia con tokens, razonamiento y tool calls
REGLAS INVIOLABLES:
- Prohibido modificar KV cache o pesos fuera del model_lock
- Prohibido try/except silencioso sin logger
- Obligatorio Dual-Phase Sampling con conmutación dinámica de temperatura según estado del tracker
SSoT: server.py es la única fuente de verdad (SSoT) para la ejecución de inferencia en SuperMLX.
═══════════════════════════════════════════════════════════════════════════════

─── MULTI-MODEL SUPPORT ──────────────────────────────────────────────────────

  Family     Models                   Tool Call Format          Thinking       Status
  ────────   ──────────────────────   ─────────────────────    ────────────   ──────
  qwen3      Qwen 3 / 3.5            <tool_call><function=>   <think>        ✅ Production
  hermes     Hermes 3 (NousResearch)  <tool_call> + JSON       <think>        ✅ Supported
  glm4       GLM-4 / 4.5 / 4.7       <tool_call> + JSON       <think>        ✅ Supported
  gemma4     Gemma 4 (Google)         <|tool_call|> + JSON     <|think|>      ✅ Supported
  deepseek   DeepSeek V3 / R1         JSON (via template)      <think>        ✅ Supported
  generic    Phi-4, other ChatML      <tool_call> + JSON       <think>        ⚠️  Fallback

  Tokenization: tokenizer.apply_chat_template() — model-agnostic.
  Extraction:   _extract_openai_tool_calls() with multi-family fallback chain.

─── QUICK START ──────────────────────────────────────────────────────────────

  Minimal:
    python SuperMLX.py

  Production (TPC + cache persistence):
    FORCE_TEXT_MODE=true \\
      CACHE_PERSIST_PATH=logs/warmup_cache.safetensors \\
      python SuperMLX.py

  Endpoints:
    MLX Direct:     http://0.0.0.0:8080/v1/chat/completions
    Sidecar:        http://0.0.0.0:8081/v1/chat/completions  (scripts, sensors)

─── ACTIVE FEATURES ──────────────────────────────────────────────────────────

  ✅ On by default:
    • KV Cache (LRU max=2)       MAIN prompt cache with session-aware lookups
    • Tool Prefix Cache (TPC)   Pre-computed system+tools KV cache, disk persistence (tool_prefix_cache.py)
    • Post-Reaper Cache Reload  Automatic disk reload after idle eviction (v1.4.0)
    • Cache Canonicalization     Volatile fields masked → 97%+ cache hit rate
    • Memory Guard              Pre-prefill Metal RAM check with auto-eviction
    • Tool Loop Breaker         Breaks infinite tool-call retry cycles (3 detection modes)
    • Session-Aware Routing     Per-session prefix tracking with block-hash index
    • Min-Suffix Pollution Wash Re-prefill ≥256 tokens on cache hit with response pollution
    • Per-Layer Trim            Selective KVCache trim for hybrid architecture disk saves
    • Frozen Cache Snapshot     Prompt-only snapshot for post-generation cache recovery
    • Compact Guard             Overflow + memory pressure detection, triggers client compaction

  🔧 Optional (env vars):
    • RAG Enrichment            FEATURE_RAG_ENRICHMENT=true  (LanceDB + embeddings)
    • Prompt Compressor         FEATURE_COMPRESSOR=true      (LLMLingua-2 + reranker)
    • Vision Model Support      Auto-detected from config    (mlx-vlm)
    • Cascade Routing           FEATURE_CASCADE=true         (frontier API fallback)
    • Native KV Quantization    KV_BITS=4                    (4-bit / 8-bit per session)

  ❌ Not implemented:
    • Continuous Batching       model_lock serializes requests (single-request pipeline)

─── CONFIGURATION ────────────────────────────────────────────────────────────

  Core:
    MODEL_PATH                      (mlx-community/Qwen3.5-9B-4bit)  HuggingFace ID or local path
    MODEL_FAMILY                    (auto)      qwen3 | hermes | glm4 | gemma4 | deepseek | generic
    FORCE_TEXT_MODE                 (false)     Skip VLM detection for text-only models
    MLX_PORT                        (8080)      MLX engine port
    SIDECAR_PORT                    (8081)      Sidecar port (0 = disabled)

  KV Cache:
    MAX_KV_SIZE                     (196608)    Max tokens per session
    PROMPT_CACHE_MAX_ENTRIES_GLOBAL (2)         Max LRU entries (2 = safe for 24GB)

    KV_BITS                         (None)      Native quantization (4 / 8 / None)

  Persistence (DPC):
    CACHE_PERSIST_PATH              ("")        Disk path for MAIN cache
    (EMBEDDED_CACHE_PERSIST_PATH removed — COMPACT cache eliminated in FASE-C1)

  Memory:
    MEMORY_GUARD_THRESHOLD_GB       (auto)      total_ram - 8GB (0 = disabled)

  See .env.example for the full reference with all supported variables.

─── MEMORY BUDGET (24GB M4 Pro) ──────────────────────────────────────────────

    Model Qwen3.5-9B-4bit:    ~5.0 GB
    2 MAIN KV entries:        ~9.0 GB  (2 x 4.5GB)
    Scratch prefill:          ~5.0 GB
    ──────────────────────────────────
    Peak total:               ~19.0 GB → safe with Memory Guard at 19.2GB
    Concurrent agents:        1-2 with warm cache

─── ARCHITECTURE ─────────────────────────────────────────────────────────────

  Claude Code ──→ MLX Engine :8080
                       │
                  PROMPT_CACHE
                   (LRU max=2)
                       │
                 ┌─────┴─────┐
                 │Mem Guard   │
                 │(Metal RAM) │
                 └─────┬─────┘
                       ↓
                    stream_generate(model, prompt_cache=...)
                                      ▲
  External tools ──→ Sidecar :8081 ───┘
                     (ephemeral cache, optional RAG, same model_lock)
"""
import os
# HuggingFace download progress bars: shown by default (useful for first-time model downloads).
# Set HF_SHOW_DOWNLOAD_PROGRESS=false in .env to suppress on cached setups.
if os.getenv("HF_SHOW_DOWNLOAD_PROGRESS", "true").lower() in ("0", "false", "no", "off"):
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import json
import time
import subprocess
import atexit
import copy
import threading
import tempfile
import re
import uuid
import hashlib
import sys
import shutil
import signal
import math
import logging
import traceback

logger = logging.getLogger("supermlx.server3")
from datetime import datetime
# Overflow guard: reject requests that would exceed safe prefill limits (used by Compact Guard)
_max_safe_prefill_tokens = int(os.environ.get("MAX_SAFE_PREFILL_TOKENS", "82192"))
def _should_signal_overflow(rest_count: int) -> bool:
    return rest_count > _max_safe_prefill_tokens
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from dotenv import load_dotenv

# --- THE METAL CRASH FIX (CPU QUARANTINE) ---
import torch

# Blindfold PyTorch so it never touches the Apple Silicon GPU (MPS).
# This forces torchvision to use the CPU, preventing Metal Command Buffer
# collisions with MLX during massive OpenClaw prefills.
torch.backends.mps.is_built = lambda: False
# --------------------------------------------

import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler
from supermlx.mtp import stream_generate_mtp
from mlx_lm.models.cache import (
    make_prompt_cache,
    can_trim_prompt_cache,
    trim_prompt_cache,
    ArraysCache,
)
import supermlx.tool_prefix_cache as _tpc

# ── Marconi Pattern: ArraysCache monkey-patch ────────────────────────────────
# Ref: "Gestión Avanzada de Caché Híbrida y Estados Recurrentes en LLMs"
#
# mlx-lm's ArraysCache (recurrent state for Gated DeltaNet / Mamba layers)
# lacks checkpoint/rollback/trim, causing can_trim_prompt_cache() to return
# False for hybrid models (Qwen3 MoE). This forces cold starts (60-120s)
# on every prompt change.
#
# Solution: patch ArraysCache with checkpoint/rollback/trim so the normal
# KVCache trim path works. trim() is a no-op that calls rollback(), matching
# the design from mlx-lm PRs #1462/#1497 (speculative decoding).
#
# Cost: ~61MB per snapshot (30 ArraysCache layers × ~2MB each). Negligible
# on Apple Silicon unified memory.
# ─────────────────────────────────────────────────────────────────────────────
import copy as _copy

def _ac_checkpoint(self):
    """Snapshot current recurrent state tensors (deep copy, O(1) per layer)."""
    self._supermlx_snapshot = _copy.deepcopy(self.cache)

def _ac_rollback(self):
    """Restore recurrent state from last checkpoint."""
    if hasattr(self, '_supermlx_snapshot') and self._supermlx_snapshot is not None:
        self.cache = _copy.deepcopy(self._supermlx_snapshot)
        mx.eval(*[a for a in self.cache if a is not None])

def _ac_trim(self, n):
    """No-op positional trim — rollback recurrent state instead.
    
    KVCache.trim(n) reduces offset by n (positional).
    ArraysCache has no positional offset; trim() restores the last
    checkpoint, which is the correct semantic equivalent for recurrent state.
    Returns n to satisfy trim_prompt_cache()'s return contract.
    """
    self._ac_rollback()
    return n

def _ac_is_trimmable(self):
    """Allow can_trim_prompt_cache() to return True for hybrid caches."""
    return True

ArraysCache.checkpoint = _ac_checkpoint
ArraysCache.rollback = _ac_rollback
ArraysCache._ac_rollback = _ac_rollback
ArraysCache.trim = _ac_trim
ArraysCache.is_trimmable = _ac_is_trimmable

print("[INIT] ArraysCache monkey-patched: checkpoint/rollback/trim (Marconi pattern)")

# Optional VLM support (Blaizzy/mlx-vlm). If unavailable, is_vlm is always False.
try:
    import mlx_vlm
    from mlx_vlm import load as load_vlm
    from mlx_vlm import stream_generate as stream_generate_vlm
    from mlx_vlm.utils import load_config as load_vlm_config
    from mlx_vlm.utils import prepare_inputs as vlm_prepare_inputs
    from mlx_vlm.utils import load_image as vlm_load_image
    from mlx_vlm.prompt_utils import get_chat_template

    mlx_vlm_available = True
except ImportError:
    mlx_vlm_available = False
    load_vlm = None
    stream_generate_vlm = None
    load_vlm_config = None
    vlm_prepare_inputs = None
    vlm_load_image = None
    get_chat_template = None

SCRIPT_DIR = Path(__file__).resolve().parent.parent  # repo root (one level above supermlx/)
DOTENV_PATH = SCRIPT_DIR / ".env"


if DOTENV_PATH.exists():
    load_dotenv(dotenv_path=DOTENV_PATH, override=True)

__version__ = "2.1.0-dev"


# ── Configuration (extracted to config.py) ────────────────────────────────────
from .config import (
    build_settings,
    _env_str, _env_int, _env_float, _env_bool,
)

SETTINGS = build_settings(script_dir=SCRIPT_DIR)

# ══════════════════════════════════════════════════════════════════════════════
# FEATURE FLAGS — Hardcoded toggles para activar/desactivar componentes.
# Cambiar True/False y reiniciar el servidor. Sin env vars por ahora — rapidez.
# ══════════════════════════════════════════════════════════════════════════════

# Prompt compression: historial largo → LanceDB → solo contexto relevante.
# Requiere rag_enricher.py + lancedb + sentence-transformers.
# Env: FEATURE_COMPRESSOR=true | COMPRESSION_THRESHOLD=6000 | COMPRESSION_GUARD=6
# FEATURE_COMPRESSOR: now in rag_facade


# Hermes compact prompt swap: when Hermes sends a compaction request,
# replace its generic summarization prompt with Claude Code's structured
# COMPACT_PROMPT (9 sections, analysis+summary tags). Default off.
FEATURE_HERMES_COMPACT_SWAP   = _env_str("HERMES_COMPACT_SWAP", "false").lower() in ("1", "true", "yes")

# Last-seen Hermes system prompt + tools — injected into compact requests
# so TPC prefix matches and the pre-computed KV cache is reused.
# Persisted to disk so it survives server restarts.
_last_hermes_context: dict = {}  # {"system": str, "tools": list}
_HERMES_CONTEXT_FILE = "hermes_context.json"

def _save_hermes_context() -> None:
    """Persist _last_hermes_context to disk alongside TPC files."""
    try:
        from pathlib import Path
        _dir = Path(os.environ.get("CACHE_PERSIST_PATH", "logs")).parent if "CACHE_PERSIST_PATH" in os.environ else Path("logs")
        # Use TPC's cache dir if available
        import supermlx.tool_prefix_cache as _tpc_mod
        if _tpc_mod._cache_dir:
            _dir = _tpc_mod._cache_dir
        _path = _dir / _HERMES_CONTEXT_FILE
        _path.write_text(json.dumps(_last_hermes_context, ensure_ascii=False))
    except Exception as _e:
        logger.warning("[CONTEXT] Falló guardado de hermes_context: %s", _e)

def _load_hermes_context() -> None:
    """Load persisted _last_hermes_context from disk."""
    global _last_hermes_context
    try:
        from pathlib import Path
        import supermlx.tool_prefix_cache as _tpc_mod
        if _tpc_mod._cache_dir:
            _path = _tpc_mod._cache_dir / _HERMES_CONTEXT_FILE
            if _path.exists():
                _last_hermes_context = json.loads(_path.read_text())
    except Exception as _e:
        logger.warning("[CONTEXT] Falló carga de hermes_context: %s", _e)

# Prefill step size: tokens processed per chunk during prompt prefill.
# Smaller = less Metal scratch memory (flash attention scratch ≈ 0.065 × chunk × kv_len × n_heads × 4).
# Benchmark (mlx 0.31.1, Qwen3.5-35B-A3B-3bit, 31K tokens):
#   chunk=512 → 0.66GB scratch, 516 tok/s
#   chunk=256 → 0.32GB scratch, 520 tok/s  ← same speed, half scratch
#   chunk=128 → 0.16GB scratch, 429 tok/s  ← 17% slower
PREFILL_STEP_SIZE             = int(_env_str("PREFILL_STEP_SIZE", "256"))

# RAG codebase enrichment: inyecta chunks relevantes del codebase en el context.
# Requiere rag_enricher.py + lancedb + sentence-transformers.
# Env: FEATURE_RAG_ENRICHMENT=true | RAG_WORKSPACE_ROOT=/path/to/workspace
# FEATURE_RAG_ENRICHMENT: now in rag_facade
# FEATURE_RAG_WORKSPACE_ROOT: now in rag_facade

# Diagnostic HTTP headers (X-Pipeline-Compression-Ms, X-Pipeline-RAG-Ms, etc.)
FEATURE_DIAGNOSTIC_HEADERS = True

# Logging exhaustivo de todas las etapas del pipeline.
# Master switch: si False, todos los FEATURE_LOG_* se desactivan.
FEATURE_FULL_LOGGING       = True

# Sub-flags for logging (only active when FEATURE_FULL_LOGGING = True)
FEATURE_LOG_PROMPTS        = _env_bool("LOG_PROMPTS", True)    # Dump de prompts/messages a disco (logs/requests/)
FEATURE_LOG_TOOLS          = _env_bool("LOG_TOOLS", True)      # Detalle de tool schemas + tool_results en historial
FEATURE_LOG_COMPRESSION    = True   # Before/after compression + chunks + timing
FEATURE_LOG_RAG            = True   # Chunks RAG inyectados, scores de relevancia
FEATURE_LOG_CACHE          = True   # Hit/miss/shorter, stable prefix, evictions
FEATURE_LOG_HEALING        = True   # Healing store operations (hits, store size)
FEATURE_LOG_GENERATION     = True   # tps, timing, token breakdown por etapa
FEATURE_LOG_RESP           = True   # Response normalization, tool extraction, think stripping

# FIX-31 Cache Diagnostics: opt-in before/after snapshots on every cache mutation.
# Captures KV offset, key-tail hash, recurrent state fingerprint, and divergence detection.
# Overhead: ~1ms per cache operation. Set FEATURE_CACHE_DIAG=false to disable.
FEATURE_CACHE_DIAG         = _env_bool("FEATURE_CACHE_DIAG", True)

# Tool call audit log: raw model XML → extracted args → delivered JSON.
# Writes logs/requests/<req_id>/tool_audit.jsonl for debugging pipeline transforms.
# Off by default — enable only when investigating tool call corruption.
FEATURE_TOOL_AUDIT_LOG     = _env_bool("TOOL_AUDIT_LOG", False)

# Expert Routing Logger: standalone diagnostic that records which experts are activated
# per layer, per request. Purely observational — zero impact on model output or cache.
# When disabled, the class is defined but attach() is never called: zero runtime overhead.
# Output: logs/expert_routing.json. Env: EXPERT_ROUTING_LOG=true
FEATURE_EXPERT_ROUTING_LOG = _env_bool("EXPERT_ROUTING_LOG", False)


def _tool_audit_write(request_id: str, stage: str, data: dict) -> None:
    """Append one audit entry to <log_root>/requests/<req_id>/tool_audit.jsonl.

    Uses _PIPELINE_LOG_DIR (absolute Path) — same directory as inbound.json.
    Best-effort: never raises.
    """
    if not FEATURE_TOOL_AUDIT_LOG:
        return
    entry = {"ts": datetime.now().isoformat(), "req": request_id[:8], "stage": stage}
    entry.update(data)
    try:
        _req_dir = _PIPELINE_LOG_DIR / request_id
        _req_dir.mkdir(parents=True, exist_ok=True)
        _audit_path = _req_dir / "tool_audit.jsonl"
        with open(_audit_path, "a", encoding="utf-8") as _af:
            _af.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as _ae:
        _pipeline_log("TOOL_AUDIT", request_id, f"write failed: {_ae}")
# Expert Routing Logger: enabled via EXPERT_ROUTING_LOG=true in .env.
# Implementation lives in expert_cache.py (start_expert_routing/end_expert_routing/save_routing_stats).
# No attach/detach needed — routing state is module-level inside expert_cache.

# Compressor: config (env-var driven above, these are runtime defaults for rag_enricher)
# FEATURE_COMPRESSION_THRESHOLD: now in rag_facade
# FEATURE_COMPRESSION_GUARD: now in rag_facade

# ── Compression Cache: extracted to server3components/compress_cache.py ────────
from .server3components import compress_cache as _cc
_compress_cache_get = _cc.cache_get
_compress_cache_put = _cc.cache_put
_compress_with_cache = _cc.compress_with_cache

# FEATURE_RAG_RELEVANCE_THRESHOLD: now in rag_facade

# Tool Call Loop Breaker: detect and break infinite tool-call retry loops.
# When the model retries the same failed tool N consecutive times, inject a
# stop instruction into the last tool_result so the model gives up and responds
# with text instead.  Env: TOOL_LOOP_BREAKER=true (default), TOOL_LOOP_MAX_RETRIES=3.
FEATURE_TOOL_LOOP_BREAKER     = _env_bool("TOOL_LOOP_BREAKER", True)
TOOL_LOOP_MAX_RETRIES         = _env_int("TOOL_LOOP_MAX_RETRIES", 3)

# Healing Store: restore <think> blocks stripped by clients back into assistant
# messages so the KV cache matches the original generation.
# With PRESERVE_THINKING=true, healing becomes essential: it restores thinking
# that clients strip, and the template preserves it for the model to see.
FEATURE_HEALING               = _env_bool("HEALING", True)

# Preserve Thinking: pass preserve_thinking=True to apply_chat_template so
# the Qwen3.6 template keeps <think> blocks from previous assistant turns.
# Qwen3.6 was designed to see its own reasoning chain (preserve_thinking);
# without it, the template strips thinking and the model loses coherence.
FEATURE_PRESERVE_THINKING     = _env_bool("PRESERVE_THINKING", True)

# Housekeeping Cache Borrow: when True, housekeeping requests USE the
# conversation cache (good model quality) but RESTORE it afterwards
# so the next normal request still gets a cache hit.
# When False, housekeeping skips cache lookup entirely (TPC cold start).
FEATURE_HOUSEKEEPING_CACHE_BORROW = _env_bool("HOUSEKEEPING_CACHE_BORROW", True)

# warmup_manager: disk I/O primitives for KV cache persistence (used by TPC + FIX-31 recovery).
# TPC (tool_prefix_cache.py) is the SSoT for prefix caching.
from . import warmup_manager as _wm
from . import metal_memory_guard as _guard
from .cache_engine import (
    HybridGenerationCheckpoint,
    capture_hybrid_generation_checkpoint,
    prepare_cache_for_insertion,
    restore_hybrid_generation_checkpoint,
    snapshot_arrays_cache,
    rollback_arrays_cache,
)
from .housekeeping_staging import (
    HousekeepingStagingManager,
    HousekeepingStagingEntry,
    find_housekeeping_split_index,
)
HOUSEKEEPING_STAGING_MANAGER = HousekeepingStagingManager()


# ── CASCADE ROUTING: DESCONECTADO ─────────────────────────────────────────────
# Extracted to server3components/cascade_routing.py and DISABLED.
# Was: forward to frontier API (Gemini) when RAG confidence is low.
# Re-enable by importing cascade_routing and calling init().
FEATURE_CASCADE = False  # DESCONECTADO
CASCADE_API_URL = ""
CASCADE_API_KEY = ""
CASCADE_MODEL = ""
CASCADE_RAG_THRESHOLD = 2.0
CASCADE_TIMEOUT_S = 60


# ══════════════════════════════════════════════════════════════════════════════

# VLM model_type whitelist (from mlx-vlm 0.4.3 models/ directory).
VLM_MODEL_TYPES = frozenset(
    {
        "aya_vision",
        "deepseek_vl_v2",
        "deepseekocr",
        "deepseekocr_2",
        "dots_ocr",
        "ernie4_5_moe_vl",
        "falcon_ocr",
        "falcon_perception",
        "fastvlm",
        "florence2",
        "gemma3",
        "gemma3n",
        "gemma4",
        "glm4v",
        "glm4v_moe",
        "glm_ocr",
        "granite4_vision",
        "granite_vision",
        "hunyuan_vl",
        "idefics2",
        "idefics3",
        "internvl_chat",
        "jina_vlm",
        "kimi_vl",
        "lfm2_vl",
        "llama4",
        "llava",
        "llava_bunny",
        "llava_next",
        "minicpmo",
        "mistral3",
        "mistral4",
        "mllama",
        "molmo",
        "molmo2",
        "molmo_point",
        "moondream3",
        "multi_modality",
        "paddleocr_vl",
        "paligemma",
        "phi3_v",
        "phi4_siglip",
        "phi4mm",
        "pixtral",
        "qwen2_5_vl",
        "qwen2_vl",
        "qwen3_5",
        "qwen3_5_moe",
        "qwen3_omni_moe",
        "qwen3_vl",
        "qwen3_vl_moe",
        "rfdetr",
        "sam3",
        "sam3_1",
        "smolvlm",
    }
)


def _resolve_model_path_and_config():
    """Resolve MODEL_PATH to a local directory and load config.json. Returns (path, config dict or None)."""
    path_str = SETTINGS.model_path
    path = Path(path_str)
    if path.exists() and path.is_dir():
        config_path = path / "config.json"
        if config_path.exists():
            try:
                with open(config_path, encoding="utf-8") as f:
                    return path, json.load(f)
            except Exception as _cfg_err:
                logger.warning("[CONFIG] Falló parseo de config.json en %s: %s", config_path, _cfg_err)
                return path, None
        return path, None
    if mlx_vlm_available:
        try:
            from huggingface_hub import snapshot_download

            # Offline-first: use cached snapshot, fallback to network
            _was_offline = os.environ.get("HF_HUB_OFFLINE")
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            try:
                path = Path(
                    snapshot_download(
                        repo_id=path_str,
                        allow_patterns=[
                            "*.json",
                            "*.safetensors",
                            "*.model",
                            "*.tiktoken",
                            "*.py",
                            "*.jinja",
                        ],
                    )
                )
            except Exception:
                # Not cached yet — retry with network
                if _was_offline is None:
                    os.environ.pop("HF_HUB_OFFLINE", None)
                path = Path(
                    snapshot_download(
                        repo_id=path_str,
                        allow_patterns=[
                            "*.json",
                            "*.safetensors",
                            "*.model",
                            "*.tiktoken",
                            "*.py",
                            "*.jinja",
                        ],
                    )
                )
            config_path = path / "config.json"
            if config_path.exists():
                with open(config_path, encoding="utf-8") as f:
                    return path, json.load(f)
            return path, None
        except Exception as _e:
            logger.warning("[CONFIG] Error leyendo config de modelo en %s: %s", path, _e)
    return None, None


def _is_vlm_config(config: Optional[Dict[str, Any]]) -> bool:
    """True if config indicates a VLM (vision) model. Prefer exact model_type to avoid text-only (e.g. Qwen3-Coder) being misclassified."""
    if not config or not mlx_vlm_available:
        return False
    model_type = (config.get("model_type") or "").strip().lower()
    if model_type in VLM_MODEL_TYPES:
        return True
    vc = config.get("vision_config")
    if isinstance(vc, dict) and len(vc) > 0:
        return True
    archs = config.get("architectures") or []
    if any("VL" in str(a) or "Vision" in str(a) for a in archs):
        return True
    return False


# GLOBAL LOCK: Prevents concurrent GPU access (Fixes the crash)
model_lock = threading.Lock()
prompt_cache_lock = threading.Lock()
console_lock = threading.Lock()

# ── Session Log: persist all console output to lastlog.md ─────────────────────
import re as _re
_ANSI_STRIP_RE = _re.compile(r'\033\[[0-9;]*m')
_LASTLOG_PATH = SETTINGS.log_root.parent / "lastlog.md"

# Truncate on server start (new session)
try:
    with open(_LASTLOG_PATH, "w", encoding="utf-8") as _f:
        _f.write(f"# SuperMLX Session Log — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n```\n")
except Exception as _e:
    logger.warning("[LOG] No se pudo inicializar lastlog.md: %s", _e)


# ── Request Logger: extracted to server3components/request_logger.py ───────────
from .server3components import request_logger as _rlog
_rlog.init(
    console_lock=console_lock,
    feature_full_logging=FEATURE_FULL_LOGGING,
    feature_log_prompts=FEATURE_LOG_PROMPTS,
    lastlog_path=_LASTLOG_PATH,
    pipeline_log_dir=SETTINGS.log_root / "requests",
)
_console_emit = _rlog.console_emit


# FIX-31 Cache Diagnostics singleton (shares console_lock for output)
from .cache_diag import cache_diag as _cache_diag
_cache_diag.enabled = FEATURE_CACHE_DIAG
_cache_diag._lock = console_lock
_cache_diag._emit_fn = _console_emit


def _memory_guard_pre_prefill(request_id: str = "") -> int:
    """
    KRIPPER MEMORY GUARD: Check Metal GPU memory before prefill.
    If active memory exceeds the threshold (default: total RAM - 8GB OS reserve),
    evict unpinned cache entries from BOTH stores to prevent OOM.
    Returns the number of entries evicted (0 if no action needed).

    This protects against OOM when external processes (browser, apps, other
    servers) consume memory alongside the MLX inference server.
    """
    threshold_gb = SETTINGS.memory_guard_threshold_gb
    if threshold_gb <= 0:
        return 0  # Guard disabled
    threshold_bytes = int(threshold_gb * (1024 ** 3))
    try:
        # Use non-deprecated API first, fallback to legacy
        get_mem = getattr(mx, 'get_active_memory', None) or getattr(mx.metal, 'get_active_memory', None)
        if get_mem is None:
            return 0
        active_bytes = get_mem()
        if active_bytes < threshold_bytes:
            return 0
        # Over threshold — evict KV cache stores
        # (MoE expert memory is managed by Expert Breathing — breathe_down/up)
        evicted = 0
        # Step 2: Evict KV cache entries
        with prompt_cache_lock:
            evicted += PROMPT_CACHE.evict_unpinned()
        _guard.force_clear_cache("memory_guard_eviction")
        post_bytes = get_mem()
        if FEATURE_FULL_LOGGING:
            _terminal_status(
                "⚠️",
                f"MEMORY GUARD: evicted {evicted} entries | "
                f"Metal was {active_bytes / (1024**3):.1f}GiB > {threshold_gb:.1f}GiB threshold | "
                f"now {post_bytes / (1024**3):.1f}GiB",
            )
        else:
            _terminal_status(
                "⚠️",
                f"MEMORY GUARD: evicted {evicted} cache entries ({active_bytes / (1024**3):.1f}GiB → {post_bytes / (1024**3):.1f}GiB)",
            )
        return evicted
    except Exception:
        return 0  # Never crash on guard failure




# ── Tool parsing (extracted to tool_parsing.py) ──────────────────────────────
from .tool_parsing import (
    _strip_thinking_from_content, _extract_thinking_text,
    _extract_enable_thinking,
    _normalize_assistant_text, _extract_openai_tool_calls,
    get_think_token_ids,
    _sanitize_tool_calls,
)
from .thinking_tracker import ThinkingTracker, ThinkingTrackerV2, ThinkingEvent
from .tool_call_tracker import ToolCallTracker, ToolCallEvent
# ── Message pipeline (extracted to message_pipeline.py) ──────────────────────
from .message_pipeline import (
    SessionContext,
    _get_healing_hash, _heal_messages,
    _break_tool_call_loop,
    _count_roles, _summarize_tool_results, _estimate_token_count,
    _is_slug_gen_request, _is_title_gen_request, _is_rag_bypass_request,
    _is_hermes_housekeeping_request,
    _prepare_messages_for_template,
    _scrub_cache_key,
    _canonicalize_messages, _extract_session_context,
    _assert_cache_key_safety, _hoist_system_messages,
)
from .cache_diag import _debug_token_divergence
from .anthropic_compat import (
    CLAUDE_MODEL_ALIASES, anthropic_to_openai_body,
    openai_to_anthropic_response, build_anthropic_sse_events,
    _sse as _sse_event,
)
from . import memory_profiler as _mem_profiler



# --- STATELESS MESSAGE HEALING STORE ---
HEALING_STORE: OrderedDict = OrderedDict()
HEALING_STORE_LOCK = threading.Lock()
MAX_HEALING_STORE = 2000  # Generous size to survive deep multi-agent sessions




_PIPELINE_LOG_DIR = SETTINGS.log_root / "requests"


# ANSI constants — re-exported from request_logger for direct use in server2.py
_ANSI_YELLOW = _rlog.ANSI_YELLOW
_ANSI_RED = _rlog.ANSI_RED
_ANSI_DIM = _rlog.ANSI_DIM
_ANSI_RESET = _rlog.ANSI_RESET
_HIGHLIGHT_STAGES = _rlog.HIGHLIGHT_STAGES
_delta_tracker = _rlog.delta_tracker
_pipeline_log = _rlog.pipeline_log
_fmt_tc_for_log = _rlog.fmt_tc_for_log
_write_request_log = _rlog.write_request_log



# _cascade_forward_request: REMOVED — was in cascade_routing.py (DESCONECTADO)
def _cascade_forward_request(*args, **kwargs):
    """Cascade routing is DISCONNECTED. This stub prevents NameError."""
    raise RuntimeError("Cascade routing is DISCONNECTED")




# ══════════════════════════════════════════════════════════════════════════════
# COMPONENT INITIALIZATION — RAG, Compressor
# Deferred to after model load (see post-model-load section below).
# These globals track runtime availability after dependency checks.
# ══════════════════════════════════════════════════════════════════════════════

# _compressor_available: set after rag_facade.init()
# _compressor_module: set after rag_facade.init()

# _rag_available: set after rag_facade.init()
# _rag_module: set after rag_facade.init()

_terminal_status = _rlog.terminal_status




# _block_chain_hashes moved to cache_lru.py
from .server3components.cache_lru import _block_chain_hashes


# ── LRUPromptCache: extracted to server3components/cache_lru.py ────────────────
from .server3components.cache_lru import LRUPromptCache, init as _cache_lru_init
_cache_lru_init(settings=SETTINGS, terminal_status_fn=_terminal_status)


PROMPT_CACHE = LRUPromptCache(
    max_size=SETTINGS.prompt_cache_max_entries_global,
    ttl_seconds=SETTINGS.prompt_cache_ttl_seconds,
)

# ── RadixPromptCache: SSoT para Fase D (RadixAttention) ────────────────────────
from .server3components.radix_cache import RadixPromptCache
RADIX_PROMPT_CACHE = RadixPromptCache(
    max_tokens=SETTINGS.max_kv_size,
    guard=_guard,
)


# Last tools list and system body from a MAIN (non-compact) request.
# Used by _prewarm_post_compact to build canonical cache keys that match
# the dual pipeline (model_tokens vs prompt_tokens/canonical).
_LAST_MAIN_TOOLS: Optional[List[Dict[str, Any]]] = None
_LAST_MAIN_SYSTEM_BODY: Optional[str] = None


def _housekeeping_staging_reaper_loop() -> None:
    while True:
        time.sleep(30.0)
        try:
            with prompt_cache_lock:
                if "HOUSEKEEPING_STAGING_MANAGER" in globals() and "PROMPT_CACHE" in globals():
                    HOUSEKEEPING_STAGING_MANAGER.prune(
                        PROMPT_CACHE,
                        ttl_seconds=getattr(SETTINGS, "housekeeping_staging_ttl_seconds", 300.0),
                        max_entries=getattr(SETTINGS, "housekeeping_staging_max_entries", 2),
                    )
        except Exception as _reaper_err:
            logger.error("[ERROR] Housekeeping staging reaper loop error: %s", _reaper_err)


_staging_reaper_thread = threading.Thread(
    target=_housekeeping_staging_reaper_loop,
    name="housekeeping_staging_reaper",
    daemon=True,
)
_staging_reaper_thread.start()



# ── SessionIndex: extracted to server3components/session_index.py ──────────────
from .server3components.session_index import SessionIndex

SESSION_INDEX = SessionIndex(
    max_entries_per_session=SETTINGS.prompt_cache_max_entries_per_session,
    max_idle_seconds=SETTINGS.prompt_cache_session_max_idle_seconds,
)


# ── Stable Prefix: extracted to server3components/stable_prefix.py ─────────────
from .server3components.stable_prefix import (
    _SessionTurnRecord,
    SESSION_TURN_STORE,
    _normalize_message_content_for_diff,
    _message_diff,
    _stable_prefix_token_len,
    _compute_msg_token_boundaries,
    _update_session_turn_store,
    init as _stable_prefix_init,
)

# ── Pipeline Fase D: RequestContext, ServerState y preprocess ─────────────────
from .server3components.request_context import RequestContext
from .server3components.state import ServerState
from .server3components.pipeline import (
    preprocess as _pipeline_preprocess,
    cache_lookup as _pipeline_cache_lookup,
    generate as _pipeline_generate,
    postprocess as _pipeline_postprocess,
)


def _get_current_server_state() -> ServerState:
    """Construye un snapshot de ServerState para el pipeline de 4 fases."""
    return ServerState(
        model=model,
        tokenizer=tokenizer,
        model_lock=model_lock,
        is_vlm=is_vlm,
        processor=processor,
        prompt_cache=PROMPT_CACHE,
        session_index=SESSION_INDEX,
        prompt_cache_lock=prompt_cache_lock,
        guard=_guard,
        tpc=_tpc,
        dpc=_DPC,
        healing_store=HEALING_STORE,
        thinking_tracker=_thinking_tracker,
        tool_call_tracker=_tool_call_tracker,
        settings=SETTINGS,
        console_lock=console_lock,
        radix_cache=RADIX_PROMPT_CACHE,
    )


class CacheSessionTranscriptLogger:
    def __init__(self, cache_session_id: str):
        now = datetime.now()
        day_dir = SETTINGS.log_root / now.strftime("%Y-%m-%d")
        day_dir.mkdir(parents=True, exist_ok=True)
        self.path = day_dir / f"cache-session-{cache_session_id}.log"
        if not self.path.exists():
            self._write_line(f"# cache_session_id={cache_session_id}")
            self._write_line(f"# created_at={self._ts()}")

    def _ts(self):
        return datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")

    def _write_line(self, line):
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line)
            f.write("\n")

    def log(self, direction, payload, request_id: str):
        self._write_line(
            f"[{self._ts()}] request_id={request_id} direction={direction}"
        )
        if isinstance(payload, (dict, list)):
            self._write_line(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            self._write_line(str(payload))
        self._write_line("")


def _cache_session_id(tokens: List[int]) -> str:
    raw = ",".join(str(tok) for tok in tokens[:1024])
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _cache_log_session_id(
    session_ctx: SessionContext, cache_session_tokens: List[int]
) -> str:
    # Prefer stable conversation/session identifier for grouped diagnostics.
    session_raw = (session_ctx.session_id or "").strip()
    if session_raw:
        return hashlib.sha1(session_raw.encode("utf-8")).hexdigest()[:16]
    return _cache_session_id(cache_session_tokens)




# _extract_images_from_messages + _prepare_messages_for_vlm moved to vlm_pipeline.py
from .server3components.vlm_pipeline import vlm_extract_images as _extract_images_from_messages
from .server3components.vlm_pipeline import vlm_prepare_messages as _prepare_messages_for_vlm



# ── VLM Pipeline: extracted to server3components/vlm_pipeline.py ───────────────
from .server3components import vlm_pipeline as _vlm
_vlm.init(feature_preserve_thinking=FEATURE_PRESERVE_THINKING)
_vlm_prompt_and_inputs = _vlm.vlm_prompt_and_inputs
_vlm_sync_before_generation = _vlm.vlm_sync_before_generation




# ── Shared helpers: deduplicate streaming / non-streaming do_POST paths ───────

def _metal_mem_str() -> str:
    """Return a compact Metal memory status string. Best-effort, never raises."""
    try:
        active_gb = mx.get_active_memory() / 1e9
        peak_gb = mx.get_peak_memory() / 1e9
        cache_gb = mx.get_cache_memory() / 1e9
        return f"metal={active_gb:.2f}GB peak={peak_gb:.2f}GB cache={cache_gb:.2f}GB"
    except Exception as _m_err:
        logger.warning("[METAL] Error leyendo memoria Metal: %s", _m_err)
        return ""


# ── Prefill memory constants ──────────────────────────────────────────────
# These are now managed by metal_memory_guard (SSoT).
# Kept as a thin redirect for _adaptive_prefill which still needs the budget.

def _get_metal_budget_gb() -> float:
    """Redirect to metal_memory_guard SSoT."""
    return _guard.get_metal_budget_gb()

# Threshold for Expert Breathing: rest tokens above this trigger breathe_down
_BREATHE_DOWN_REST_THRESHOLD = int(os.environ.get("BREATHE_DOWN_REST_THRESHOLD", "15000"))

# GPU yield: insert a tiny sleep between generated tokens so other Metal clients
# (Chrome/Safari VideoToolbox) can squeeze command buffers into the GPU queue.
# 0 = disabled (default), 1-2 ms is enough for smooth video playback alongside inference.
_GPU_YIELD_SECONDS = float(os.environ.get("GPU_YIELD_MS", "0")) / 1000.0



def _pre_prefill_memory_relief(request_id: str, rest_count: int) -> None:
    """Free OS and Metal memory before large prefills to reduce peak pressure.

    Delegates memory relief to metal_memory_guard (SSoT).
    Also triggers Expert Breathing (breathe_down) for large MAIN requests.
    """
    # Memory relief: delegated to metal_memory_guard (SSoT for clear_cache/gc/malloc)
    _guard.pre_prefill_gate(rest_count, kv_bits=SETTINGS.kv_bits, request_id=request_id)

    # Expert Breathing: contract experts for large MAIN prefills (delegated to expert_cache)
    if rest_count >= _BREATHE_DOWN_REST_THRESHOLD:
        try:
            from .expert_cache import moe_pre_prefill_hook
            moe_pre_prefill_hook(
                model=model,
                rest_count=rest_count,
                threshold=_BREATHE_DOWN_REST_THRESHOLD,
                request_id=request_id,
                log_fn=_terminal_status,
            )
        except Exception as _be:
            _terminal_status("⚠️", f"MoE pre-prefill hook error: {_be} | req={request_id[:8]}")


# ── Adaptive prefill: extracted to server3components/adaptive_prefill.py ───────
from .server3components import adaptive_prefill as _ap_mod
from .server3components.adaptive_prefill import (
    _start_prefill_progress,
    _adaptive_prefill,
    _adaptive_prefill_chunk,
    _get_vm_counters,
)
# _ap_mod.init() called in run()

def _update_healing_store(raw_text: str, message_text: str, tool_calls: Optional[List],
                          user_context: str = "") -> None:
    """Store raw response keyed by the stripped version's hash.

    With PRESERVE_THINKING=true, thinking blocks are kept in the store so
    healing can restore them for the next turn. The template's
    preserve_thinking flag ensures they're included in the prompt.
    Without PRESERVE_THINKING, thinking is stripped to avoid prompt inflation.
    """
    if not FEATURE_PRESERVE_THINKING:
        # Legacy behavior: strip thinking to avoid prompt inflation
        raw_text = _strip_thinking_from_content(raw_text)
    if raw_text == message_text:
        return
    # FIX: _strip_thinking_from_content has a safety guard that refuses to return
    # empty string, and its line 190 may strip the </think> TAG while leaving
    # the thinking TEXT. In both cases message_text has thinking residue that the
    # client never sees (thinking is sent as a separate SSE block).
    # Derive the client-visible portion directly from raw_text, which always
    # preserves the </think> boundary marker.
    if message_text and re.search(r'</think>', raw_text, re.IGNORECASE):
        _after_think = re.sub(
            r'^.*?</think>\s*', '', raw_text,
            flags=re.DOTALL | re.IGNORECASE
        )
        # Remove tool call blocks (already extracted into tool_calls param)
        _after_think = re.sub(
            r'<tool_call>.*?</tool_call>', '', _after_think,
            flags=re.DOTALL | re.IGNORECASE
        ).strip()
        # Strip residual orphan tags
        _after_think = re.sub(
            r'</(?:think|tool_call)>\s*', '', _after_think,
            flags=re.IGNORECASE
        ).strip()
        message_text = _after_think
    h = _get_healing_hash(message_text, tool_calls, user_context)
    if not h:
        return
    # DIAGNOSTIC: log exactly what goes into the hash at storage time
    _mt = (message_text or "").strip()
    _tc_n = len(tool_calls) if tool_calls else 0
    _uc_n = len(user_context.strip()) if user_context else 0
    _raw_n = len(raw_text) if raw_text else 0
    _mt_first = repr(_mt[:60]) if _mt else "''"
    _mt_last = repr(_mt[-40:]) if len(_mt) > 60 else ""
    # Build tool_call detail for diagnostic
    _tc_detail = ""
    if tool_calls:
        for _ti, _t in enumerate(tool_calls[:2]):  # max 2
            _fn = _t.get("function", {})
            _args_str = _fn.get("arguments", "")
            try:
                _akeys = list(json.loads(_args_str).keys())
            except Exception as _json_err:
                logger.warning("[HEAL_STORE] Error parseando JSON de arguments: %s", _json_err)
                _akeys = []
            _tc_detail += (
                f" | tc[{_ti}]={{id={_t.get('id','')[:16]}, fn={_fn.get('name','')}, "
                f"args_len={len(_args_str)}, args_keys={_akeys}}}"
            )
    with console_lock:
        _console_emit(
            f"  [HEAL_STORE] SAVE | hash={h[:16]} | "
            f"msg_text_len={len(_mt)} | tc={_tc_n} | user_ctx_len={_uc_n} | "
            f"raw_len={_raw_n} | first={_mt_first}"
            + (f" | last={_mt_last}" if _mt_last else "")
            + _tc_detail
        )
    with HEALING_STORE_LOCK:
        HEALING_STORE[h] = raw_text
        HEALING_STORE.move_to_end(h, last=True)
        while len(HEALING_STORE) > MAX_HEALING_STORE:
            HEALING_STORE.popitem(last=False)


# ── Post-generation cache logic: extracted to server3components/post_generation.py ──
from .server3components import post_generation as _post_gen
from .server3components.post_generation import (
    _post_generation_cache_update,
    _build_timing_dict,
    _log_generation_telemetry,
    _insert_cache_entries,
)
# _post_gen.init() is called in run() after all deps are available

def _tokenize_prompt(prompt):
    if isinstance(prompt, str):
        add_special_tokens = tokenizer.bos_token is None or not prompt.startswith(
            tokenizer.bos_token
        )
        return tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
    if isinstance(prompt, list):
        return prompt
    return list(prompt)


# _insert_cache_entries: moved to post_generation.py

def _kv_cache_offset(cache: Any) -> Optional[int]:
    """
    Return the current KV offset (number of original tokens actually stored in the
    KV cache) from the first cache layer after any trim_prompt_cache() call.

    KVCache and QuantizedKVCache both expose a public `.offset` int attribute
    (mlx_lm/models/cache.py).  `prompt_cache` is a list of per-layer caches.

    Returns None if the cache is None, empty, or does not expose .offset.
    Callers must fall back to the canonical `matched_prefix_len` in that case.
    """
    try:
        layers = cache if isinstance(cache, (list, tuple)) else [cache]
        # Pass 1: prioritize layers with explicit .offset (KVCache, RotatingKVCache).
        # This avoids reading from ArraysCache whose internal shape[2] is
        # a static dimension (e.g., 8192 conv_dim), NOT a token offset.
        for layer in layers:
            if hasattr(layer, "offset"):
                return int(layer.offset)
        
        # Pass 2: fallback for models that only use ArraysCache-style layers
        # (no KVCache at all). Only reached if NO layer has .offset.
        for layer in layers:
            if hasattr(layer, "cache") and isinstance(layer.cache, list):
                for c in layer.cache:
                    if c is not None and len(c.shape) >= 3:
                        return int(c.shape[2])
    except (IndexError, TypeError, AttributeError) as _e:
        logger.warning("[CACHE] Error leyendo offset en capas de cache: %s", _e)
    return None


def _capture_hybrid_checkpoint_before_generation(
    *,
    request_id: str,
    prompt_cache: Any,
    prompt_tokens: List[int],
    model_tokens: List[int],
    rest_tokens: List[int],
) -> Optional[HybridGenerationCheckpoint]:
    """Capture a clean hybrid checkpoint with the last prompt token pending.
    
    With the Marconi monkey-patch, can_trim_prompt_cache() returns True even
    for hybrid caches. We detect ArraysCache layers directly and invoke their
    native checkpoint() before capturing the SuperMLX snapshot.
    """
    if not prompt_cache:
        if FEATURE_FULL_LOGGING:
            _pipeline_log(
                "CACHE",
                request_id,
                "HYBRID_CHECKPOINT skipped: prompt_cache is None/empty",
            )
        return None

    # Check if any layer is ArraysCache (recurrent state that needs checkpoint)
    _has_recurrent_layers = any(
        hasattr(c, '_supermlx_snapshot') or (hasattr(c, 'is_trimmable') and isinstance(c, ArraysCache))
        for c in prompt_cache
    )
    if not _has_recurrent_layers:
        if FEATURE_FULL_LOGGING:
            _pipeline_log(
                "CACHE",
                request_id,
                "HYBRID_CHECKPOINT skipped: no ArraysCache layers (pure KV)",
            )
        return None

    # Invoke native checkpoint() on each ArraysCache layer (Marconi pattern)
    for c in prompt_cache:
        if isinstance(c, ArraysCache) and hasattr(c, 'checkpoint'):
            c.checkpoint()

    remaining = len(rest_tokens)
    expected_model_offset = len(model_tokens) - remaining
    actual_model_offset = _kv_cache_offset(prompt_cache)
    _v15_tolerated = False
    if actual_model_offset != expected_model_offset:
        # FIX-31 v15 compat: when v15 accepted a minor extension, the cache
        # holds extra tokens from the previous turn's response. The physical
        # offset exceeds model_tokens by the extension size. Allow checkpoint
        # capture using the actual physical offset — the recurrent state
        # contamination was already deemed acceptable by v15.
        _MINOR_EXT_CKPT_TOLERANCE = 128
        _ext_delta = (actual_model_offset - expected_model_offset) if actual_model_offset > expected_model_offset else 0
        if 0 < _ext_delta <= _MINOR_EXT_CKPT_TOLERANCE:
            _v15_tolerated = True
            _pipeline_log(
                "CACHE",
                request_id,
                f"HYBRID_CHECKPOINT: v15 minor extension tolerated | "
                f"physical={actual_model_offset} expected={expected_model_offset} delta={_ext_delta}",
            )
        else:
            _pipeline_log(
                "CACHE",
                request_id,
                "HYBRID_CHECKPOINT skipped: physical/model offset mismatch | "
                f"physical={actual_model_offset} expected={expected_model_offset}",
            )
            return None

    # When v15 tolerated, skip hash metadata: the offset exceeds model_tokens
    # length, making model_prefix_hash unreliable for next-lookup verification.
    # model_offset=0 ensures __model_offset__ won't match _kv_off on next
    # lookup, so the hash check at FIX-31 L5528 is naturally bypassed.
    if _v15_tolerated:
        _ckpt_model_offset = 0
        _ckpt_model_hash = 0
    else:
        _ckpt_model_offset = actual_model_offset
        _ckpt_model_hash = hash(tuple(model_tokens[:actual_model_offset]))

    canonical_key_len = len(prompt_tokens) - remaining
    try:
        checkpoint = capture_hybrid_generation_checkpoint(
            prompt_cache,
            canonical_key_len,
            model_offset=_ckpt_model_offset,
            model_prefix_hash=_ckpt_model_hash,
        )
        if checkpoint is None:
            _pipeline_log(
                "CACHE",
                request_id,
                "HYBRID_CHECKPOINT unavailable: no recurrent layers captured",
            )
            return None
        _pipeline_log(
            "CACHE",
            request_id,
            "HYBRID_CHECKPOINT captured | "
            f"key={canonical_key_len} | kv_offset={actual_model_offset} | "
            f"arrays={len(checkpoint.snapshot.get('layers', {}))} | "
            f"size={checkpoint.snapshot.get('nbytes', 0) / (1024**2):.1f}MB"
            f"{' | v15_no_hash=true' if _v15_tolerated else ''}",
        )
        _cache_diag.snapshot(prompt_cache, list(model_tokens[:min(actual_model_offset, len(model_tokens))]), "hybrid_capture")
        return checkpoint
    except Exception as checkpoint_err:
        _terminal_status(
            "⚠️",
            f"Hybrid checkpoint failed ({checkpoint_err}) — cache will not be reused",
        )
        return None








from .sampling import DualPhaseSampler, safe_make_sampler, build_dual_phase_sampler


def _build_sampler(
    body: dict,
    enable_thinking: Optional[bool] = None,
    tracker: Any = None,
    is_compact: bool = False,
):
    """
    Build a sampler and logits processors with anti-loop defaults.
    Sampler handles: Dual-Phase (think_temp -> resp_temp), top_p, top_k, min_p.
    Logits processors handle: repetition_penalty, presence_penalty (penalty application).
    Returns (sampler, logits_processors, applied_kwargs).
    """
    # ── Sampler params ────────────────────────────────────────────────────
    top_p = body.get("top_p", SETTINGS.default_top_p)
    top_k = body.get("top_k", SETTINGS.default_top_k)
    min_p = body.get("min_p", SETTINGS.default_min_p)

    has_tools = bool(body.get("tools"))
    body_temp = body.get("temperature")

    # Determine temperatures for thinking and response phases:
    body_thinking_temp = body.get("thinking_temperature")
    if body_thinking_temp is not None and isinstance(body_thinking_temp, (int, float)):
        think_temp = float(body_thinking_temp)
    elif is_compact:
        think_temp = SETTINGS.compaction_temperature
    else:
        think_temp = SETTINGS.thinking_temperature

    if is_compact:
        resp_temp = SETTINGS.compaction_temperature
    elif has_tools:
        resp_temp = SETTINGS.tool_calling_temperature
    elif body_temp is not None and isinstance(body_temp, (int, float)):
        resp_temp = float(body_temp)
    else:
        resp_temp = (
            SETTINGS.tool_calling_temperature
            if has_tools
            else SETTINGS.response_temperature
        )

    # Determine active tracker
    active_tracker = tracker if tracker is not None else globals().get("_thinking_tracker")

    # Resolve whether thinking is active for this generation
    if is_compact:
        thinking_active = False
    else:
        thinking_active = enable_thinking if enable_thinking is not None else SETTINGS.default_thinking

    sampler, applied_kwargs = build_dual_phase_sampler(
        think_temp=think_temp,
        resp_temp=resp_temp,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        enable_thinking=thinking_active,
        tracker=active_tracker,
    )

    # ── Logits processors (penalty application) ───────────────────────────
    _rep_penalty = body.get("repetition_penalty", SETTINGS.default_repetition_penalty)
    _rep_ctx = body.get("repetition_context_size", SETTINGS.default_repetition_context_size)
    _pres_penalty = body.get("presence_penalty", SETTINGS.default_presence_penalty)
    _pres_ctx = body.get("presence_context_size", SETTINGS.default_presence_context_size)

    logits_processors = None
    try:
        from mlx_lm.sample_utils import make_logits_processors
        logits_processors = make_logits_processors(
            repetition_penalty=_rep_penalty,
            repetition_context_size=_rep_ctx,
            presence_penalty=_pres_penalty,
            presence_context_size=_pres_ctx,
        )
        if not logits_processors:
            logits_processors = None  # Empty list → None (no-op)
    except (ImportError, TypeError):
        logits_processors = None  # mlx_lm version without make_logits_processors

    applied_kwargs["repetition_penalty"] = _rep_penalty
    applied_kwargs["repetition_context_size"] = _rep_ctx
    applied_kwargs["presence_penalty"] = _pres_penalty
    applied_kwargs["presence_context_size"] = _pres_ctx
    applied_kwargs["logits_processors_active"] = logits_processors is not None

    return sampler, logits_processors, applied_kwargs



# Model loading: LM or VLM based on config.
model = None
tokenizer = None
processor = None
is_vlm = False
vlm_config = None
_thinking_tracker = None

_terminal_status(
    "🚀",
    f"Loading model: {SETTINGS.model_path} (exposed as: {SETTINGS.proxy_model_id})",
)
_resolved_path, _config = _resolve_model_path_and_config()
if _config and _is_vlm_config(_config) and not SETTINGS.force_text_mode:
    if not mlx_vlm_available:
        raise RuntimeError(
            "Model config indicates a VLM but mlx-vlm is not installed. "
            "Install with: pip install mlx-vlm (and optionally mlx-vlm[torch] for some models)."
        )
    # Workaround: when torchvision is missing, transformers sets VIDEO_PROCESSOR_MAPPING_NAMES
    # values to None, then video_processor_class_from_name does "class_name in extractors"
    # and raises TypeError. Patch to skip None so processor can load (image-only path).
    try:
        import importlib
        from transformers.models.auto import video_processing_auto as _vpa
        from transformers.models.auto.configuration_auto import (
            model_type_to_module_name,
        )

        def _patched_video_processor_class_from_name(class_name: str):
            for module_name, extractors in _vpa.VIDEO_PROCESSOR_MAPPING_NAMES.items():
                if extractors is not None and class_name in extractors:
                    mod_name = model_type_to_module_name(module_name)
                    module = importlib.import_module(
                        f".{mod_name}", "transformers.models"
                    )
                    try:
                        return getattr(module, class_name)
                    except AttributeError:
                        continue
            for extractor in _vpa.VIDEO_PROCESSOR_MAPPING._extra_content.values():
                if getattr(extractor, "__name__", None) == class_name:
                    return extractor
            main_module = importlib.import_module("transformers")
            if hasattr(main_module, class_name):
                return getattr(main_module, class_name)
            return None

        _vpa.video_processor_class_from_name = _patched_video_processor_class_from_name
    except Exception as _vlm_patch_err:
        logger.warning("[VLM] Falló parche de video_processor_class_from_name: %s", _vlm_patch_err)
    model, processor = load_vlm(
        SETTINGS.model_path, tokenizer_config={"trust_remote_code": True}
    )
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    is_vlm = True
    vlm_config = (
        _config if isinstance(_config, dict) else getattr(model, "config", None)
    )
    if vlm_config is None and hasattr(model, "config"):
        vlm_config = getattr(model.config, "__dict__", None) or {}
    _terminal_status("✅", "VLM model loaded (mlx-vlm).")
    try:
        from transformers.utils import is_torchvision_available

        if is_torchvision_available():
            _terminal_status(
                "⚡",
                "Torch acceleration: enabled (fast image/video processor). "
                "If you see Metal 'uncommitted encoder' crash, try: pip install mlx-vlm (without [torch] extra).",
            )
        else:
            _terminal_status(
                "⚠️",
                "Torch acceleration: disabled. Install mlx-vlm[torch] for much faster image processing.",
            )
    except Exception:
        _terminal_status(
            "⚠️", "Torch acceleration: unknown (torch/torchvision not detected)."
        )
else:
    if SETTINGS.force_text_mode and _config and _is_vlm_config(_config):
        _terminal_status("⚠️", "FORCE_TEXT_MODE=true → VLM config detected but loading as TEXT-ONLY (faster)")
    # Force offline mode: model already cached in ~/.cache/huggingface/
    # Prevents "Fetching N files" network check on every boot.
    # If model isn't cached yet, we catch the error and retry online.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    # Detect MoE models: load lazily to avoid materializing all experts (19.5 GB).
    # After module replacement, only non-expert params (~1.4 GB) are materialized.
    _is_moe_path = any(tag in SETTINGS.model_path for tag in ("A3B", "A14B", "MoE", "moe", "Mixtral", "mixtral", "Agents-A1"))
    _load_kwargs = {"tokenizer_config": {"trust_remote_code": True}}
    if _is_moe_path and getattr(SETTINGS, "enable_moe_cache", False):
        _load_kwargs["lazy"] = True
        _terminal_status("🔧", "MoE model detected — loading with lazy=True (experts as placeholders)")
    if SETTINGS.enable_mtp:
        from supermlx.mtp import install_qwen3_5_mtp_trunk_shim, inject_qwen3_5_mtp_support, validate_qwen3_5_mtp_support
        install_qwen3_5_mtp_trunk_shim()
    try:
        model, tokenizer = load(SETTINGS.model_path, **_load_kwargs)
    except Exception as _offline_err:
        _terminal_status("⚠️", f"Offline load failed ({_offline_err}) — retrying with network...")
        os.environ.pop("HF_HUB_OFFLINE", None)
        model, tokenizer = load(SETTINGS.model_path, **_load_kwargs)
    _terminal_status("✅", "Model loaded (mlx-lm).")
    _terminal_status("⚡", "Torch acceleration: N/A (text-only model).")

    if SETTINGS.enable_mtp:
        _mtp_dir = Path(SETTINGS.mtp_weights_path)
        _mtp_cfg_path = _mtp_dir / "config.json"
        if _mtp_cfg_path.exists():
            import json as _json
            with open(_mtp_cfg_path) as _f:
                _mtp_cfg = _json.load(_f)
            _injected = inject_qwen3_5_mtp_support(model, _mtp_dir, _mtp_cfg)
            if _injected and validate_qwen3_5_mtp_support(model):
                _terminal_status("🚀", f"MTP Speculative Decoding enabled ({_mtp_dir.name})")
            else:
                _terminal_status("⚠️", "MTP injection failed — falling back to standard autoregressive")
        else:
            _terminal_status("⚠️", f"MTP config not found at {_mtp_cfg_path} — running standard autoregressive")

    # ── Apple Neural Engine (ANE) Prefill Injection ────────────────────────
    if getattr(SETTINGS, "enable_ane", False) or os.environ.get("ENABLE_ANE", "").strip().lower() in {"1", "true", "yes"}:
        try:
            from lab.ane_code.ane_shim import inject_ane_support, validate_ane_support
            _ane_buckets = getattr(SETTINGS, "ane_prefill_buckets", [64, 128, 256])
            _ane_ok = inject_ane_support(model, {}, buckets=_ane_buckets)
            if _ane_ok and validate_ane_support(model):
                _terminal_status("🍏", f"Apple Neural Engine (ANE) acceleration enabled (buckets={_ane_buckets})")
            else:
                _terminal_status("⚠️", "ANE injection failed — falling back to standard GPU path")
        except Exception as _ane_err:
            _terminal_status("⚠️", f"ANE setup error: {_ane_err} — falling back to standard GPU path")

    # ── Resolve think token IDs (once, at load) ───────────────────────────
    _think_start_id, _think_end_id = get_think_token_ids(tokenizer)
    if _think_start_id is not None:
        _terminal_status("🧠", f"Think tokens resolved: start={_think_start_id} end={_think_end_id}")
    else:
        _terminal_status("ℹ️", "Think tokens: not found in vocab — using text-based detection")
    # ThinkingTracker: SSoT for thinking state — created once, reset() per request
    if SETTINGS.thinking_budget_mode == "v2":
        _forced_end_tokens = [_think_end_id] if _think_end_id is not None else []
        if not _forced_end_tokens:
            # Fallback: encode </think>
            try:
                _forced_end_tokens = list(tokenizer.encode("</think>", add_special_tokens=False))
            except Exception:
                _forced_end_tokens = []
        _thinking_tracker = ThinkingTrackerV2(
            think_start_id=_think_start_id,
            think_end_id=_think_end_id,
            enable_thinking=SETTINGS.default_thinking,
            budget=SETTINGS.max_thinking_tokens,
            forced_end_tokens=_forced_end_tokens,
        )
        _terminal_status("🧠", f"ThinkingTrackerV2 initialized (logits-forcing mode, budget={SETTINGS.max_thinking_tokens})")
    else:
        _thinking_tracker = ThinkingTracker(
            think_start_id=_think_start_id,
            think_end_id=_think_end_id,
            enable_thinking=SETTINGS.default_thinking,
        )
        _terminal_status("🧠", "ThinkingTracker v1 initialized (break mode)")
    _tool_call_tracker = ToolCallTracker()
    _terminal_status("🔧", "ToolCallTracker initialized (SSoT for tool call streaming state)")

    # Some models (e.g. Agents-A1) ship chat_template.jinja separately instead of
    # embedding it in tokenizer_config.json. Load it so tools get injected into the prompt.
    if not getattr(tokenizer, "chat_template", None):
        try:
            from huggingface_hub import hf_hub_download
            _jinja_path = hf_hub_download(SETTINGS.model_path, "chat_template.jinja")
            with open(_jinja_path, "r") as f:
                tokenizer.chat_template = f.read()
            _terminal_status("🔧", "Loaded chat_template.jinja into tokenizer (was missing from tokenizer_config)")
        except Exception as _jinja_err:
            logger.warning("[TOKENIZER] No se pudo cargar chat_template.jinja: %s", _jinja_err)

# ── MoE Expert Cache ──────────────────────────────────────────────────────
# ── MoE Expert Subsystem ──────────────────────────────────────────────────
_moe_stats = {}
try:
    from .expert_cache import init_moe_subsystem
    _moe_stats = init_moe_subsystem(model, SETTINGS.model_path, SETTINGS, log_fn=_terminal_status)
except ImportError:
    pass  # expert_cache not available — dense model, no action needed
except Exception as _moe_err:
    _terminal_status("⚠️", f"MoE Subsystem init failed: {_moe_err}")

# ── Stable Prefix init (needs tokenizer + is_vlm from model load) ────────
_stable_prefix_init(
    tokenizer=tokenizer,
    is_vlm=is_vlm,
    feature_preserve_thinking=FEATURE_PRESERVE_THINKING,
    max_idle_seconds=SETTINGS.prompt_cache_session_max_idle_seconds,
)

# ── Metal Memory Guard (SSoT) ────────────────────────────────────────────
# Must run AFTER model load + MoE init (needs active memory to be final).
# Applies wired_limit (anti-swap) + cache_limit (scratch ceiling).
from . import metal_memory_guard as _guard
try:
    _terminal_status("🛡️", "Metal Guard: initializing (wired limit + cache ceiling + shader warmup)...")
    _guard_t0 = time.time()
    _guard_result = _guard.init_protections(model=model, kv_bits=SETTINGS.kv_bits)
    _guard_init_ms = int((time.time() - _guard_t0) * 1000)
    _terminal_status(
        "🛡️",
        f"Metal Guard: wired={_guard_result['wired_limit_gb']:.1f}GB | "
        f"cache_limit={_guard_result['cache_limit_gb']:.1f}GB | "
        f"active={_guard_result['active_gb']:.1f}GB | "
        f"budget={_guard_result['max_working_set_gb']:.1f}GB ({_guard_init_ms}ms)",
    )
    # Mechanism 3: compile Metal shaders via a real forward pass
    _shader_elapsed = _guard.warmup_shaders(
        model=model,
        tokenizer=tokenizer,
        cache_factory=lambda: make_prompt_cache(model),
        request_id="startup",
    )
    if _shader_elapsed > 0:
        _terminal_status("🛡️", f"Metal Guard: shaders compiled in {_shader_elapsed:.2f}s")
except Exception as _guard_err:
    _terminal_status("⚠️", f"Metal Guard init failed (non-fatal): {_guard_err}")

# ── Late init: wire _guard + HSM into cache_lru (defined after model load) ──
_cache_lru_init(
    settings=SETTINGS, terminal_status_fn=_terminal_status,
    guard=_guard, housekeeping_staging_manager=HOUSEKEEPING_STAGING_MANAGER,
    prompt_cache_ref=PROMPT_CACHE,
)

# ── Expert Routing Logger ─────────────────────────────────────────────────
if FEATURE_EXPERT_ROUTING_LOG and _moe_stats.get("moe_layers", 0) > 0:
    _terminal_status("🔬", "Expert Routing Logger: active (output: logs/expert_routing.json)")
elif FEATURE_EXPERT_ROUTING_LOG:
    _terminal_status("⚠️", "Expert Routing Logger: EXPERT_ROUTING_LOG=true but no MoE layers found")
_terminal_status(
    "🧠",
    (
        "Global Cache Config loaded | "
        f"max_entries={SETTINGS.prompt_cache_max_entries_global} | "
        f"ttl_seconds={SETTINGS.prompt_cache_ttl_seconds} | "
        f"canonicalize_tool_context={SETTINGS.cache_canonicalize_tool_context} | "
        f"healing_store_capacity={MAX_HEALING_STORE}"
    ),
)
_kv_desc = f"bits={SETTINGS.kv_bits}" if SETTINGS.kv_bits is not None else "OFF"
if SETTINGS.kv_bits is not None:
    _kv_desc += f" scheme={SETTINGS.kv_quant_scheme} group_size={SETTINGS.kv_group_size}"
_terminal_status("🗜️", f"KV Cache Quantization (native mlx-lm): {_kv_desc}")

# ══════════════════════════════════════════════════════════════════════════════
# Each component checks its FEATURE flag, then tries to import dependencies.
# Failures are logged but never crash startup — graceful degradation.
# ══════════════════════════════════════════════════════════════════════════════
# --- RAG Enricher ---
# ── RAG + Compressor: extracted to server3components/rag_facade.py ─────────────
from .server3components import rag_facade as _rag_facade
_rag_facade.init(terminal_status_fn=_terminal_status)
_rag_available = _rag_facade.is_rag_available()
_rag_module = _rag_facade.get_rag_module()
_compressor_available = _rag_facade.is_compressor_available()
_compressor_module = _rag_facade.get_compressor_module()
FEATURE_COMPRESSOR = _rag_facade.FEATURE_COMPRESSOR
FEATURE_RAG_ENRICHMENT = _rag_facade.FEATURE_RAG_ENRICHMENT
FEATURE_COMPRESSION_THRESHOLD = _rag_facade.FEATURE_COMPRESSION_THRESHOLD
FEATURE_COMPRESSION_GUARD = _rag_facade.FEATURE_COMPRESSION_GUARD
FEATURE_RAG_RELEVANCE_THRESHOLD = _rag_facade.FEATURE_RAG_RELEVANCE_THRESHOLD



# --- Feature flags summary ---
_active_features = []
if _compressor_available:
    _active_features.append("Compressor")
if _rag_available:
    _active_features.append("RAG")
if FEATURE_DIAGNOSTIC_HEADERS:
    _active_features.append("DiagHeaders")
if FEATURE_FULL_LOGGING:
    _active_features.append("FullLog")
if FEATURE_PRESERVE_THINKING:
    _active_features.append("PreserveThinking")
if FEATURE_HEALING:
    _active_features.append("Healing")
if FEATURE_TOOL_LOOP_BREAKER:
    _active_features.append("LoopBreaker")
_terminal_status(
    "🏁",
    f"Active features: [{', '.join(_active_features) if _active_features else 'BASE ONLY'}]"
)

# ══════════════════════════════════════════════════════════════════════════════
# TPC STARTUP + KV CACHE PERSISTENCE
# Reduces first-request cold-start from ~40s to <2s.
#
# TPC (tool_prefix_cache.py) pre-computes and persists the system+tools
# KV cache to disk. On restart, loads from disk → instant warm start.
#
# CACHE_PERSIST_PATH  — directory for TPC disk state.
# ══════════════════════════════════════════════════════════════════════════════

# Shared state: startup event + frozen cache for FIX-31 pollution recovery.
_DPC = _wm.DPCState()
_WARMUP_DONE = _DPC.warmup_done  # Signals TPC startup complete (gates first request)




# Pattern for startup detection (robust: case-insensitive, survives OpenClaw text changes)
_STARTUP_MSG_PATTERN = re.compile(
    r"(new session|/new|/reset|session startup|run your session startup)",
    re.IGNORECASE,
)


def _warmup_save_cache(
    tokens: List[int], prompt_cache: Any, path: Path, prefix_hash: Optional[str] = None
) -> bool:
    """Thin wrapper → warmup_manager.save_cache(). Backward compat for auto-save callers."""
    return _wm.save_cache(tokens, prompt_cache, path, prefix_hash=prefix_hash, log_fn=_terminal_status)




def _run_tpc_startup() -> None:
    """Background startup — load Tool Prefix Cache (TPC) from disk.

    TPC stores the pre-computed KV cache for the system+tools prefix,
    eliminating cold-start prefill on server restart.
    """
    _t0 = time.perf_counter()
    _mem_before = mx.get_active_memory() / 1e9

    _DPC.warmup_done.set()
    _DPC.disk_cache_saved = True

    if SETTINGS.cache_persist_path:
        _tpc.init(SETTINGS.cache_persist_path)
        _tpc.load_from_disk(model, SETTINGS.max_kv_size)
        _load_hermes_context()

    _elapsed = time.perf_counter() - _t0
    _mem_after = mx.get_active_memory() / 1e9
    _mem_delta = _mem_after - _mem_before
    _terminal_status(
        "⏱️",
        f"TPC startup: completo | elapsed={_elapsed:.1f}s | "
        f"active_mem={_mem_after:.3f}GB | delta={_mem_delta:+.3f}GB"
    )


# ── Launch TPC startup thread ────────────────────────────────────────────────
if SETTINGS.cache_persist_path:
    _warmup_thread = threading.Thread(
        target=_run_tpc_startup, daemon=True, name="tpc-startup"
    )
    _warmup_thread.start()
    _terminal_status(
        "🔥",
        f"TPC: loading tool prefix cache from disk...",
    )
else:
    _terminal_status(
        "ℹ️",
        "TPC: DISABLED — set CACHE_PERSIST_PATH to enable",
    )
    _WARMUP_DONE.set()

# ── Metal cache limit ────────────────────────────────────────────────────────
# Managed by metal_memory_guard.init_protections() (see above).
# Dynamic cache_limit calculated as 25% of device_total (~6.4 GB on 24GB M4 Pro).


# ══════════════════════════════════════════════════════════════════════════════
# ADAPTIVE PREFILL — dynamic chunk sizing based on available Metal memory
# ══════════════════════════════════════════════════════════════════════════════
#
# Calibrated against empirical benchmark (mlx 0.31.1, Qwen3.5-35B-A3B-3bit):
#   theoretical_scratch = N_LAYERS_SDPA × chunk × kv_len × N_HEADS × 4
#   measured_scratch    ≈ 0.065 × theoretical_scratch  (flash attention tiling)
#
# The function pre-prefills rest_tokens[:-1] with adaptive chunks, then returns
# only the last token. stream_generate sees 1 token → skips its own prefill loop
# (generate_step line 430: `while total - processed > 1` is False) → goes
# straight to decoding.
# ══════════════════════════════════════════════════════════════════════════════

# ── Scratch model constants (from config.json + benchmark calibration) ──────
# These are extracted once at import time from _config (module-level dict).
_text_cfg = (_config or {}).get("text_config", _config or {})
_N_HEADS = _text_cfg.get("num_attention_heads", 16)
_FULL_ATTN_INTERVAL = _text_cfg.get("full_attention_interval", 4)
_N_LAYERS = _text_cfg.get("num_hidden_layers", 40)
_N_LAYERS_SDPA = _N_LAYERS // _FULL_ATTN_INTERVAL  # layers with full SDPA attention
_SCRATCH_COEFFICIENT = 0.065  # calibrated: measured/predicted ratio from benchmark
_SCRATCH_BYTES_PER_ELEMENT = 4  # fp32 for QK^T score matrix
_ADAPTIVE_PREFILL_MIN_CHUNK = 32  # never go below this
_ADAPTIVE_PREFILL_SAFETY_MARGIN = 0.85  # use only 85% of available memory for scratch

_terminal_status(
    "📐",
    f"Adaptive prefill params: n_heads={_N_HEADS} sdpa_layers={_N_LAYERS_SDPA} "
    f"coeff={_SCRATCH_COEFFICIENT} min_chunk={_ADAPTIVE_PREFILL_MIN_CHUNK}",
)


# _adaptive_prefill_chunk, _get_vm_counters, _adaptive_prefill: moved to adaptive_prefill.py



def _stream_generate_kwargs(prompt_tokens, max_tokens, sampler, prompt_cache, logits_processors=None):
    kwargs = {
        "model": model,
        "tokenizer": tokenizer,
        "prompt": prompt_tokens,
        "max_tokens": max_tokens,
        "sampler": sampler,
        "prompt_cache": prompt_cache,
        "max_kv_size": SETTINGS.max_kv_size,
        "kv_group_size": SETTINGS.kv_group_size,
        "prefill_step_size": PREFILL_STEP_SIZE,
    }
    if SETTINGS.kv_bits is not None:
        kwargs["kv_bits"] = SETTINGS.kv_bits
    if logits_processors is not None:
        kwargs["logits_processors"] = logits_processors
    return kwargs


def _stream_generate_unified(
    rest_tokens,
    max_tokens,
    sampler,
    prompt_cache,
    vlm_pixel_values=None,
    vlm_mask=None,
    vlm_kwargs=None,
    cache_match_type="miss",
    logits_processors=None,
):
    """
    Yields response objects with .text and .token (LM: GenerationResponse, VLM: GenerationResult).
    Dynamically checks for image tokens in rest_tokens to prevent Metal Segmentation Faults.
    """
    if is_vlm:
        # Prevent Segfault: We MUST pass vision tensors if rest_tokens contains image placeholders.
        # If we have a massive chunk of rest_tokens (e.g., > 100) on a 'shorter' hit,
        # it almost certainly contains the system/user image tokens.
        has_image_tokens = False
        if rest_tokens:
            # Check for common Qwen3/GLM vision token IDs, or fallback to length heuristic
            vision_tokens = {151652, 151653, 151654, 151655}  # Common Qwen vision IDs
            has_image_tokens = any(tok in vision_tokens for tok in rest_tokens)
            if not has_image_tokens and len(rest_tokens) > 100:
                has_image_tokens = True

        use_vision = (
            cache_match_type == "miss" or has_image_tokens
        ) and vlm_pixel_values is not None

        rest_ids = (
            mx.array([rest_tokens], dtype=mx.int32)
            if rest_tokens
            else mx.array([[0]], dtype=mx.int32)
        )

        sliced_mask = None
        if vlm_mask is not None:
            if rest_tokens:
                sliced_mask = vlm_mask[..., -len(rest_tokens) :]
            else:
                sliced_mask = vlm_mask[..., -1:]

        kwargs = {
            "input_ids": rest_ids,
            "pixel_values": vlm_pixel_values if use_vision else None,
            "mask": sliced_mask if use_vision else None,
            **(vlm_kwargs if vlm_kwargs else {}),
            "prompt_cache": prompt_cache,
            "max_tokens": max_tokens,
            "sampler": sampler,
            "max_kv_size": SETTINGS.max_kv_size,
            "kv_group_size": SETTINGS.kv_group_size,
            "kv_quant_scheme": SETTINGS.kv_quant_scheme,
            "quantized_kv_start": SETTINGS.quantized_kv_start,
        }
        if SETTINGS.kv_bits is not None:
            kwargs["kv_bits"] = SETTINGS.kv_bits
        for resp in stream_generate_vlm(model, processor, "", image=None, **kwargs):
            yield resp
    else:
        # Dynamic cache update state for MoE models
        _dcu_gen_count = 0
        _dcu_no_swap_streak = 0
        _dcu_low_fb_streak = 0
        _dcu_last_fb_rate = 1.0
        try:
            from .expert_cache import (
                dynamic_cache_needed,
                dynamic_cache_update,
                dynamic_update_policy,
            )
            _dcu_enabled = dynamic_cache_needed(model)
            _dcu_func = dynamic_cache_update if _dcu_enabled else None
            _dcu_policy = dynamic_update_policy if _dcu_enabled else None
            if not _dcu_enabled and hasattr(model, "_moe_config") and model._moe_config.get("moe_layers", 0) > 0:
                _cfg = model._moe_config
                _terminal_status(
                    "⚡",
                    f"MOE dynamic updates disabled: full expert residency "
                    f"{_cfg.get('capacity', 0)}/{_cfg.get('num_experts', 0)}",
                )
        except ImportError:
            _dcu_enabled = False
            _dcu_func = None
            _dcu_policy = None

        _stream_gen_fn = (
            stream_generate_mtp
            if (SETTINGS.enable_mtp and getattr(model, "mtp", None) is not None)
            else stream_generate
        )
        for resp in _stream_gen_fn(
            **_stream_generate_kwargs(rest_tokens, max_tokens, sampler, prompt_cache, logits_processors=logits_processors)
        ):
            yield resp

            # MoE dynamic cache update — swap cold experts for hot ones between tokens
            if _dcu_enabled:
                _dcu_gen_count += 1
                interval, budget = _dcu_policy(
                    _dcu_gen_count, _dcu_last_fb_rate,
                    _dcu_no_swap_streak, _dcu_low_fb_streak,
                )
                if _dcu_gen_count % interval == 0:
                    stats = _dcu_func(model, max_layer_updates=budget)
                    swaps = sum(s.get("swaps", 0) for s in stats)
                    fallbacks = sum(s.get("fallbacks", 0) for s in stats)
                    requests = sum(s.get("requests", 0) for s in stats)
                    if requests > 0:
                        _dcu_last_fb_rate = fallbacks / requests
                    if swaps == 0:
                        _dcu_no_swap_streak += 1
                    else:
                        _dcu_no_swap_streak = 0
                    if requests > 0 and (fallbacks / requests) <= 0.005:
                        _dcu_low_fb_streak += 1
                    else:
                        _dcu_low_fb_streak = 0


# ══════════════════════════════════════════════════════════════════════════════
# SIDECAR HANDLER — Lightweight OpenAI-compatible endpoint for non-OpenClaw use
# ══════════════════════════════════════════════════════════════════════════════
#
# Design principles:
#   • Shares model in RAM with APIHandler (0 extra model memory)
#   • Ephemeral KV cache per request (make_prompt_cache → generate → discard)
#   • Optional RAG enrichment (financial papers, codebase — same embedder)
#   • Same model_lock — queues behind OpenClaw (no starvation, just waiting)
#   • No healing, loop breaker, session index, or Kripper pipeline
#   • Minimal code path = minimal latency overhead
#
# Memory at rest: 0 bytes (no persistent cache)
# Memory during generation: ~144KB × context_tokens (ephemeral, freed on completion)
# ══════════════════════════════════════════════════════════════════════════════

# ── SidecarHandler: extracted to server3components/sidecar_handler.py ──────────
from .server3components.sidecar_handler import SidecarHandler, init as _sidecar_init
# _sidecar_init() is called in run() after all deps are available


class APIHandler(BaseHTTPRequestHandler):
    # Per-request state set during routing.  Checked at response time to
    # decide between OpenAI and Anthropic wire formats.
    _is_anthropic: bool = False
    _anthropic_model: str = ""

    def log_message(self, format, *args):
        # Keep terminal output focused on custom request lifecycle lines.
        return

    def _route_path(self) -> str:
        """Strip query string from self.path for route matching.

        Claude Code appends ?beta=true to /v1/messages which broke plain
        string matching against the path."""
        return self.path.split("?")[0].rstrip("/")

    def do_HEAD(self):
        """Claude Code sends HEAD / as a connectivity health-check."""
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()

    def do_GET(self):
        route = self._route_path()
        if route in ("/v1/models", "/models"):
            # Include Claude model aliases so Claude Code discovers models
            # with the 'claude-' prefix it requires.
            models_data = [
                {
                    "id": SETTINGS.proxy_model_id,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "mlx",
                }
            ]
            for alias in CLAUDE_MODEL_ALIASES:
                models_data.append({
                    "id": alias,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "anthropic",
                })
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(
                json.dumps({"object": "list", "data": models_data}).encode("utf-8")
            )
            return
        if route == "":
            # Root health-check (Claude Code sends HEAD /, browsers GET /)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok"}).encode("utf-8"))
            return
        self.send_error(404, "Not Found")

    def do_POST(self):
        route = self._route_path()
        if route in ("/v1/models", "/models"):
            return self.do_GET()

        # Anthropic token-count stub — Claude Code calls this for validation.
        if route in ("/v1/messages/count_tokens", "/messages/count_tokens"):
            try:
                cl = int(self.headers.get("Content-Length", 0))
                self.rfile.read(cl)
            except Exception as _count_err:
                logger.warning("[HTTP] Error leyendo body en count_tokens: %s", _count_err)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"input_tokens": 0}).encode("utf-8"))
            return

        # ── Ephemeral endpoint: TPC reuse WITHOUT session cache write ──
        # Auxiliary tasks (background_review, compact, etc.) hit this route
        # to benefit from the TPC prefix but never evict PROMPT_CACHE slots.
        _is_ephemeral_route = route in (
            "/v1/ephemeral/messages", "/ephemeral/messages",
            "/v1/ephemeral/chat/completions", "/ephemeral/chat/completions",
        )

        # ── Anthropic Messages API: inline translation ───────────────
        # Translate the Anthropic body to OpenAI format and fall through
        # to the same pipeline that handles /v1/chat/completions.
        if route in ("/v1/messages", "/messages") or route in ("/v1/ephemeral/messages", "/ephemeral/messages"):
            try:
                cl = int(self.headers["Content-Length"])
                raw = json.loads(self.rfile.read(cl).decode("utf-8"))
            except Exception:
                self.send_error(400, "Bad Request")
                return
            self._is_anthropic = True
            self._anthropic_model = raw.get("model", "claude-sonnet-4-20250514")

            # ── COMPACT TOOL STRIP ────────────────────────────────────
            # Compact requests say "Do NOT call any tools" but Claude Code
            # still sends all 28 tool definitions (~130K chars, ~30K tokens).
            # Stripping them cuts prefill from ~37K to ~5K model_tokens,
            # eliminating OOM and reducing compact time from ~130s to ~17s.
            _compact_stripped_tools = 0
            _hermes_compact_swapped = False
            if raw.get("tools") or raw.get("messages"):
                _last_user = ""
                _system_text = ""
                for _m in raw.get("messages", []):
                    if _m.get("role") == "system":
                        _sc = _m.get("content", "")
                        if isinstance(_sc, str):
                            _system_text = _sc
                for _m in reversed(raw.get("messages", [])):
                    if _m.get("role") == "user":
                        _c = _m.get("content", "")
                        if isinstance(_c, str):
                            _last_user = _c
                        elif isinstance(_c, list):
                            _last_user = " ".join(
                                b.get("text", "") for b in _c
                                if isinstance(b, dict) and b.get("type") == "text"
                            )
                        break

                # Capture system+tools from normal requests for compact TPC reuse
                if raw.get("tools") and _system_text:
                    _new_hash = hashlib.md5((_system_text + str(len(raw["tools"]))).encode()).hexdigest()[:8]
                    if _last_hermes_context.get("_hash") != _new_hash:
                        _last_hermes_context["system"] = _system_text
                        _last_hermes_context["tools"] = list(raw["tools"])
                        _last_hermes_context["_hash"] = _new_hash
                        _save_hermes_context()

                _is_compact_request = False
                # Claude Code compact: detect by user message phrase
                if (raw.get("tools") or raw.get("messages")) and "CRITICAL: Respond with TEXT ONLY" in _last_user[:200]:
                    if raw.get("tools"):
                        _compact_stripped_tools = len(raw["tools"])
                        raw["tools"] = []
                        _terminal_status("🪶",
                            f"COMPACT TOOL STRIP: removed {_compact_stripped_tools} tool definitions from compact request")
                    raw.pop("thinking", None)
                    _is_compact_request = True

                # Hermes compact: detect by user message OR system prompt phrase
                # Hermes sends compact as a single user message (no system msg).
                elif (FEATURE_HERMES_COMPACT_SWAP
                      and (("summarization agent" in _last_user[:200]
                            and "context checkpoint" in _last_user[:200])
                           or ("summarization agent" in _system_text[:200]
                               and "context checkpoint" in _system_text[:200]))):
                    # Strip tools if present
                    if raw.get("tools"):
                        _compact_stripped_tools = len(raw["tools"])
                        raw["tools"] = []
                    raw.pop("thinking", None)
                    # Prepend Claude Code's structured COMPACT_PROMPT to the user message
                    # Hermes already serialized the conversation as user content — keep it,
                    # just replace the generic instruction with the structured one.
                    from .server_compact import (
                        COMPACT_PROMPT as _CC_COMPACT_PROMPT,
                        extract_hermes_compact_parts,
                        _max_frozen_summary_chars,
                    )
                    _frozen_hermes_summary = ""
                    for _m in reversed(raw.get("messages", [])):
                        if _m.get("role") == "user":
                            _original = _m.get("content", "")
                            if isinstance(_original, str):
                                _prev_sum, _turns_data = extract_hermes_compact_parts(_original, log_fn=_terminal_status)
                                if _prev_sum and len(_prev_sum) <= _max_frozen_summary_chars():
                                    _frozen_hermes_summary = _prev_sum
                                    _m["content"] = _CC_COMPACT_PROMPT + "\n\n" + _turns_data
                                    _terminal_status("🪶", f"HERMES COMPACT: frozen summary preserved ({len(_prev_sum)} chars) — summarizing new turns only")
                                else:
                                    if _prev_sum:
                                        _terminal_status("⚠️", f"HERMES COMPACT: frozen summary ({len(_prev_sum)} chars) exceeds cap -> consolidating")
                                    _m["content"] = _CC_COMPACT_PROMPT + (f"\n\n{_prev_sum}\n\n{_turns_data}" if _prev_sum else f"\n\n{_turns_data}")
                            break

                    # ── TPC REUSE: inject saved system+tools into compact ──
                    # The compact arrives without system prompt or tools, so the
                    # TPC prefix (18K tokens) can't match. By injecting the last
                    # Hermes system+tools, the compact's tokenized prefix matches
                    # the TPC and gets the pre-computed KV cache for free.
                    _tpc_injected_into_compact = False
                    if _last_hermes_context.get("system") and _last_hermes_context.get("tools"):
                        # Inject system message
                        _has_system = any(_m.get("role") == "system" for _m in raw.get("messages", []))
                        if not _has_system:
                            raw.setdefault("messages", []).insert(0, {
                                "role": "system",
                                "content": _last_hermes_context["system"]
                            })
                        # Inject tools (don't count these as stripped)
                        raw["tools"] = _last_hermes_context["tools"]
                        _tpc_injected_into_compact = True

                    _hermes_compact_swapped = True
                    _is_compact_request = True
                    _terminal_status("🪶",
                        f"HERMES COMPACT SWAP: replaced prompt + stripped {_compact_stripped_tools} tools"
                        f" | TPC context injected={_tpc_injected_into_compact}")

            body = anthropic_to_openai_body(raw, SETTINGS.proxy_model_id)

            if _hermes_compact_swapped or _is_compact_request:
                # Signal to _handle_chat_completion to treat this as housekeeping.
                # The HOUSEKEEPING_CACHE_BORROW mechanism will:
                # 1. Snapshot the conversation cache before compact generation
                # 2. Generate the summary (compact prompt)
                # 3. Restore the conversation cache after generation
                # This avoids the cold start that was caused by evicting here.
                body["_supermlx_compact"] = True
                body["_frozen_summary"] = _frozen_hermes_summary if _hermes_compact_swapped else ""
                body["enable_thinking"] = False
                body["temperature"] = SETTINGS.compaction_temperature
                body.pop("thinking", None)

            # Ephemeral route flag → skip session cache save
            if _is_ephemeral_route:
                body["_supermlx_ephemeral"] = True

            # Jump past the body-parse block that follows.
            self._handle_chat_completion(body)
            return

        if route not in (
            "/v1/chat/completions", "/chat/completions",
            "/v1/ephemeral/chat/completions", "/ephemeral/chat/completions",
        ):
            try:
                self.send_error(404, "Not Found")
            except BrokenPipeError:
                pass
            return

        self._is_anthropic = False
        self._anthropic_model = ""
        try:
            content_length = int(self.headers["Content-Length"])
            body = json.loads(self.rfile.read(content_length).decode("utf-8"))
        except Exception as _parse_err:
            logger.warning("[HTTP] Error parseando request body: %s", _parse_err)
            self.send_error(400, "Bad Request")
            return
        # Ephemeral route flag → skip session cache save
        if _is_ephemeral_route:
            body["_supermlx_ephemeral"] = True
        self._handle_chat_completion(body)

    def _handle_chat_completion(self, body):

        # ── TPC STARTUP GATE ─────────────────────────────────────────────────
        # If TPC is loading from disk and boot hasn't finished yet, the first
        # request waits. For all subsequent requests _WARMUP_DONE.is_set() == True
        # → no overhead (O(1) check).
        # Timeout 120s: if boot fails, request proceeds with cold start.
        if not _WARMUP_DONE.is_set() and SETTINGS.cache_persist_path:
            _wg_t0 = time.time()
            _WARMUP_DONE.wait(timeout=120)
            _wg_elapsed = time.time() - _wg_t0
            if _wg_elapsed > 0.5:  # Only log if actually waited
                logger.info(
                    "[DATA] TPC startup gate: request queued %.1fs | cache_ready=%s",
                    _wg_elapsed, _WARMUP_DONE.is_set(),
                )

        # ── END TPC STARTUP GATE ────────────────────────────────────────────

        _wg_elapsed = 0.0 # Placeholder if gate didn't run

        request_id = uuid.uuid4().hex[:12]

        # ── SLUG-GEN FAST PATH (FIX-14) ────────────────────────────────────────
        # Avoid 40s prefill hangs for non-essential background tasks.
        raw_messages_inbound = body.get("messages", [])
        if _is_slug_gen_request(raw_messages_inbound):
            _pipeline_log("BYPASS", request_id, "Slug-gen detected | Intercepting with hardcoded response")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            resp = {
                "id": f"chatcmpl-{request_id}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": SETTINGS.proxy_model_id,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "chat-session"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 100, "completion_tokens": 2, "total_tokens": 102}
            }
            self.wfile.write(json.dumps(resp).encode("utf-8"))
            return

        # ── NL TITLE FAST PATH ──────────────────────────────────────────────────
        # Hermes title_generation auxiliary requests are intercepted here and
        # answered with Apple NaturalLanguage.framework keyword extraction
        # instead of burning a full LLM prefill cycle (~10-40s) for a trivial
        # side-task. Typical response time: <1ms.
        if _is_title_gen_request(raw_messages_inbound):
            from .nl_title import generate_title_nl, build_title_response
            # Extract the user message (contains "User: <text>\n\nAssistant: <text>")
            _title_user_msg = ""
            for _tm in raw_messages_inbound:
                if (_tm.get("role") or "").lower() == "user":
                    _title_user_msg = str(_tm.get("content") or "")
                    break
            _t0_title = time.time()
            try:
                _nl_title = generate_title_nl(_title_user_msg)
            except Exception as _nl_err:
                _pipeline_log("NL_TITLE", request_id,
                    f"NL.framework crashed: {_nl_err!r} — falling through to LLM")
                _nl_title = None
            _title_ms = (time.time() - _t0_title) * 1000
            if _nl_title:
                _pipeline_log("NL_TITLE", request_id,
                    f"Title generated via NaturalLanguage.framework | "
                    f'"{_nl_title}" | {_title_ms:.1f}ms')
                _terminal_status("🏷️", f"NL Title: \"{_nl_title}\" ({_title_ms:.1f}ms)", indent=1)
                resp = build_title_response(_nl_title, request_id, SETTINGS.proxy_model_id)
                _is_streaming_title = body.get("stream", False)
                if _is_streaming_title and self._is_anthropic:
                    # Anthropic SSE format (for Hermes via /v1/messages)
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    _msg_id = f"msg_{uuid.uuid4().hex[:24]}"
                    self.wfile.write(_sse_event("message_start", {
                        "type": "message_start",
                        "message": {
                            "id": _msg_id, "type": "message", "role": "assistant",
                            "model": self._anthropic_model, "content": [],
                            "stop_reason": None, "stop_sequence": None,
                            "usage": {"input_tokens": 50, "output_tokens": 0},
                        },
                    }).encode("utf-8"))
                    self.wfile.write(_sse_event("content_block_start", {
                        "type": "content_block_start", "index": 0,
                        "content_block": {"type": "text", "text": ""},
                    }).encode("utf-8"))
                    self.wfile.write(_sse_event("content_block_delta", {
                        "type": "content_block_delta", "index": 0,
                        "delta": {"type": "text_delta", "text": _nl_title},
                    }).encode("utf-8"))
                    self.wfile.write(_sse_event("content_block_stop", {
                        "type": "content_block_stop", "index": 0,
                    }).encode("utf-8"))
                    self.wfile.write(_sse_event("message_delta", {
                        "type": "message_delta",
                        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                        "usage": {"output_tokens": len(_nl_title.split())},
                    }).encode("utf-8"))
                    self.wfile.write(_sse_event("message_stop", {"type": "message_stop"}).encode("utf-8"))
                    self.wfile.flush()
                elif _is_streaming_title:
                    # OpenAI SSE format (for Claude Code / OpenClaw)
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()
                    _chunk_base = {
                        "id": resp["id"],
                        "object": "chat.completion.chunk",
                        "created": resp["created"],
                        "model": resp["model"],
                    }
                    _role_chunk = {**_chunk_base, "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]}
                    self.wfile.write(f"data: {json.dumps(_role_chunk)}\n\n".encode("utf-8"))
                    _content_chunk = {**_chunk_base, "choices": [{"index": 0, "delta": {"content": _nl_title}, "finish_reason": None}]}
                    self.wfile.write(f"data: {json.dumps(_content_chunk)}\n\n".encode("utf-8"))
                    _stop_chunk = {**_chunk_base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
                    self.wfile.write(f"data: {json.dumps(_stop_chunk)}\n\n".encode("utf-8"))
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                else:
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps(resp).encode("utf-8"))
                return
            else:
                _pipeline_log("NL_TITLE", request_id,
                    "NL.framework unavailable — falling through to LLM")

        # ── FASE D.1: PREPROCESS (coexistencia) ───────────────────────────
        _server_state = _get_current_server_state()
        ctx = RequestContext(
            request_id=request_id,
            body=body,
            is_streaming=body.get("stream", False),
            is_anthropic=self._is_anthropic,
        )
        _pipeline_preprocess(ctx, _server_state)

        prompt_tokens = ctx.prompt_tokens or []
        model_tokens = ctx.model_tokens or []
        prompt = ctx.prompt
        cache_key = ctx.cache_key or list(prompt_tokens)
        raw_messages = ctx.raw_messages or []
        messages = ctx.canonical_messages or ctx.raw_messages
        tools = ctx.body.get("tools")
        enable_thinking = ctx.enable_thinking
        _is_compact = ctx.is_compact
        _is_ephemeral = ctx.is_ephemeral
        _is_housekeeping = ctx.is_housekeeping
        _housekeeping_conv_model_boundary = ctx.housekeeping_conv_model_boundary
        _housekeeping_conv_prompt_tokens = ctx.housekeeping_conv_prompt_tokens
        _housekeeping_staging_hit = getattr(ctx, "housekeeping_staging_hit", False)
        _pipeline_timings = ctx.pipeline_timings or {}
        _frozen_summary = str(ctx.body.get("_frozen_summary", "") or "")
        session_ctx = ctx.session_ctx
        _session_id_for_turn = ctx.session_id or ""
        vlm_pixel_values = ctx.vlm_pixel_values
        vlm_mask = ctx.vlm_mask
        vlm_kwargs = ctx.vlm_kwargs
        cache_key_delta_chars = len(model_tokens) - len(prompt_tokens)
        prompt_was_normalized = len(prompt_tokens) != len(model_tokens)
        reasoning_control = _extract_enable_thinking(body, default_thinking=SETTINGS.default_thinking)
        # ── FASE D.2: CACHE_LOOKUP (RadixPromptCache + TPC) ───────────────
        _pipeline_cache_lookup(ctx, _server_state, RADIX_PROMPT_CACHE)
        _pipeline_log("CACHE_LOOKUP_RADIX", request_id,
            f"hit_ratio={ctx.cache_hit_ratio or 0.0:.3f} | rest_count={ctx.rest_count} | "
            f"has_kv={ctx.prompt_cache is not None}")

        prompt_cache = ctx.prompt_cache
        rest_count = ctx.rest_count
        matched_prefix_len = (
            max(0, len(model_tokens) - rest_count)
            if rest_count <= len(model_tokens)
            else 0
        )
        cache_match_type = getattr(ctx, "cache_match_type", "hit" if matched_prefix_len > 0 else "miss")
        cache_selection_source = getattr(ctx, "cache_selection_source", "radix")
        if 0 < rest_count <= len(model_tokens):
            rest_tokens = model_tokens[-rest_count:]
        else:
            rest_tokens = list(model_tokens)
        cache_session_tokens = prompt_tokens
        sampler, logits_processors, sampler_kwargs = _build_sampler(
            body, enable_thinking=enable_thinking, is_compact=_is_compact
        )
        # V2: inject reasoning budget logits processor
        if isinstance(_thinking_tracker, ThinkingTrackerV2):
            _budget_proc = _thinking_tracker.make_logits_processor()
            if logits_processors is None:
                logits_processors = [_budget_proc]
            else:
                logits_processors.append(_budget_proc)
        # Always use server DEFAULT_MAX_TOKENS — Claude Code sends max_tokens=8192
        # which truncates long code generation. We override it entirely.
        max_tokens = SETTINGS.default_max_tokens
        # Compact runner (context compaction) needs short output.
        # Cap total budget to avoid wasting 35s on thinking for a summary.
        # Note: title generation is intercepted earlier by the NL fast path.
        is_streaming = body.get("stream", False)

        # ── CONTEXT STATUS — unconditional, every turn ────────────────────
        _ctx_total = len(model_tokens)
        _ctx_max = SETTINGS.max_kv_size
        _ctx_pct = (_ctx_total / max(1, _ctx_max)) * 100
        _ctx_headroom = max(0, _ctx_max - _ctx_total)
        _ctx_ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        _ctx_line = (
            f"  📊 [CONTEXT STATUS] {_ctx_ts} req={request_id[:8]} | "
            f"context={_ctx_total}/{_ctx_max} ({_ctx_pct:.1f}%) | "
            f"output_budget={max_tokens} | headroom={_ctx_headroom}"
        )
        if _ctx_pct >= 50:
            _ctx_line = f"{_ANSI_RED}{_ctx_line}{_ANSI_RESET}"
        else:
            _ctx_line = f"{_ANSI_YELLOW}{_ctx_line}{_ANSI_RESET}"
        with console_lock:
            _console_emit(_ctx_line)

        acquired = False
        generated_tokens = []
        message_text = ""
        generation_started_at = None
        first_token_at = None
        queue_started_at = time.time()
        hybrid_generation_checkpoint: Optional[HybridGenerationCheckpoint] = None
        _task_detector: Any = None
        _loop_broken: int = int(getattr(ctx, "loop_broken", 0) or 0)

        # --- STABLE-PREFIX TELEMETRY DEFAULTS ---
        stable_prefix_token_len_computed = 0
        stable_prefix_msg_count_computed = 0
        stable_prefix_diff_descriptors: List[Dict[str, Any]] = []

        try:
            model_lock.acquire(blocking=True)
            acquired = True
            _lock_wait_ms = (time.time() - queue_started_at) * 1000
            if FEATURE_FULL_LOGGING and _lock_wait_ms > 100:
                _pipeline_log("LOCK_WAIT", request_id,
                    f"model_lock acquired | waited={_lock_wait_ms:.0f}ms")

            generation_started_at = time.time()
            wait_seconds = generation_started_at - queue_started_at
            with prompt_cache_lock:
                prompt_cache = ctx.prompt_cache
                if prompt_cache is None and model is not None:
                    cache_model = model.language_model if is_vlm else model
                    prompt_cache = make_prompt_cache(
                        cache_model, max_kv_size=SETTINGS.max_kv_size
                    )
                    ctx.prompt_cache = prompt_cache

                _kv_off = _kv_cache_offset(prompt_cache)
                _session_id_for_turn = (session_ctx.session_id or "").strip() if session_ctx else ""
                _mem_profiler.snapshot(
                    request_id,
                    "PRE_CACHE",
                    is_anthropic=self._is_anthropic,
                    prompt_tokens=len(prompt_tokens) if prompt_tokens else None,
                    model_tokens=len(model_tokens) if model_tokens else None,
                )
                if FEATURE_FULL_LOGGING:
                    _pipeline_log(
                        "CACHE_LOOKUP",
                        request_id,
                        f"radix hit={cache_match_type} | kv_off={_kv_off} | "
                        f"rest={len(rest_tokens)} | matched={matched_prefix_len}/{len(prompt_tokens)}",
                    )

            rest_count = ctx.rest_count if ctx.rest_count is not None else len(rest_tokens)

            # ── COMPACT GUARD: OVERFLOW + MEMORY PRESSURE ─────────────────
            # Applies to ALL requests unconditionally (including Anthropic).
            # Prevents OOM crashes by rejecting oversized prefills before they start.
            # Uses "prompt is too long" wording to trigger Claude Code's reactive compact.
            _memory_pressure = False
            _pressure_reason = ""
            if _should_signal_overflow(rest_count):
                _memory_pressure = True
                _pressure_reason = f"rest={rest_count} tokens exceed safe prefill limit ({_max_safe_prefill_tokens})"
            else:
                try:
                    _get_mem = getattr(mx, 'get_active_memory', None) or getattr(mx.metal, 'get_active_memory', None)
                    if _get_mem:
                        _active_gb = _get_mem() / 1e9
                        _threshold_gb = float(os.environ.get("MEMORY_COMPACT_THRESHOLD_GB", "21.0"))
                        # Use model_tokens (actual token count) not prompt_tokens
                        # (canonical cache key, can be inflated by scrub masking).
                        _real_token_count = len(model_tokens) if model_tokens else len(prompt_tokens)
                        if _active_gb > _threshold_gb and _real_token_count > 80000:
                            _memory_pressure = True
                            _pressure_reason = f"metal={_active_gb:.1f}GB > {_threshold_gb}GB threshold (prompt={_real_token_count} tokens)"
                except Exception as _mem_err:
                    logger.warning("[COMPACT_GUARD] Error consultando memoria Metal: %s", _mem_err)

            if _memory_pressure:
                _terminal_status("⚠️",
                    f"COMPACT GUARD: {_pressure_reason} "
                    f"— sending 'prompt is too long' to trigger client compaction")
                _overflow_error = json.dumps({
                    "type": "error",
                    "error": {
                        "type": "invalid_request_error",
                        "message": f"prompt is too long: {len(prompt_tokens)} tokens > {len(prompt_tokens) - 1000} maximum",
                    }
                }).encode("utf-8")
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(_overflow_error)))
                self.end_headers()
                self.wfile.write(_overflow_error)
                if acquired:
                    _guard.force_clear_cache("overflow_error")
                    model_lock.release()
                    acquired = False
                return

            # ── PIPELINE LOG: CACHE & TQ ───────────────────────────────────
            if FEATURE_FULL_LOGGING and FEATURE_LOG_CACHE:
                _pipeline_log("CACHE", request_id,
                    f"session_id={session_ctx.session_id[:16]} | source={session_ctx.source}")
                _pipeline_log("CACHE", request_id,
                    f"global_lookup: match_type={cache_match_type} | matched_prefix={matched_prefix_len}/{len(prompt_tokens)}")
                _pipeline_log("CACHE", request_id,
                    f"RESULTADO: cache_hit={cache_match_type} | rest_tokens={rest_count} | source={cache_selection_source}")
            

            cache_session_id = _cache_log_session_id(session_ctx, cache_session_tokens)
            if SETTINGS.enable_request_logging:
                try:
                    request_logger = CacheSessionTranscriptLogger(
                        cache_session_id=cache_session_id
                    )
                except Exception as _rlog_err:
                    logger.warning("[LOG] Error instanciando CacheSessionTranscriptLogger: %s", _rlog_err)
                    request_logger = None

            if request_logger:
                request_logger.log(
                    "prompt",
                    {
                        "request_meta": {
                            "path": self.path,
                            "stream": bool(body.get("stream", False)),
                            "model": body.get("model", SETTINGS.proxy_model_id),
                            "model_family": SETTINGS.model_family,
                            "temperature": body.get(
                                "temperature", SETTINGS.default_temperature
                            ),
                            "max_tokens": body.get(
                                "max_tokens", SETTINGS.default_max_tokens
                            ),
                            "enable_thinking": enable_thinking,
                            "thinking_source": reasoning_control["source"],
                            "thinking_raw": reasoning_control["raw"],
                            "cache_session_id": cache_session_id,
                            "cache_match_type": cache_match_type,
                            "cache_selection_source": cache_selection_source,
                            "matched_prefix_len": matched_prefix_len,
                            "cache_prompt_normalized": prompt_was_normalized,
                            "cache_key_normalized": prompt_was_normalized,
                            "cache_key_delta_chars": cache_key_delta_chars,
                            "prompt_tokens": len(prompt_tokens),
                            "session_id": session_ctx.session_id,
                            "parent_session_id": session_ctx.parent_session_id,
                            "branch_id": session_ctx.branch_id,
                            "session_source": session_ctx.source,
                            # --- M6 telemetry: stable-prefix metrics ---
                            "stable_prefix_msg_count": stable_prefix_msg_count_computed,
                            "stable_prefix_token_len": stable_prefix_token_len_computed,
                            "stable_prefix_diff": stable_prefix_diff_descriptors[
                                :8
                            ],  # cap to avoid log bloat
                            **(
                                {
                                    "vlm_format_prefix_stable": False,
                                    "prompt_token_prefix": list(prompt_tokens[:64]),
                                }
                                if is_vlm
                                else {}
                            ),
                        },
                        "messages": messages,
                        "tools": tools,
                        "rendered_prompt": prompt,
                    },
                    request_id=request_id,
                )
                sampler_payload = {
                    "applied_kwargs": sampler_kwargs,
                    "rest_tokens": rest_count,
                    "matched_prefix_len": matched_prefix_len,
                }
                if SETTINGS.vlm_cache_debug and is_vlm and prompt_tokens:
                    sampler_payload["prompt_token_prefix"] = list(prompt_tokens[:64])
                request_logger.log("sampler", sampler_payload, request_id=request_id)

            prompt_len = len(prompt_tokens)
            hit_ratio = (
                (matched_prefix_len / prompt_len * 100) if prompt_len > 0 else 0.0
            )

            if hit_ratio >= 85:
                cache_light = "🟢"
            elif hit_ratio >= 50:
                cache_light = "🟡"
            else:
                cache_light = "🔴"

            _terminal_status(
                "📨",
                f"{cache_light} Cache: {hit_ratio:.1f}% ({cache_match_type}) | "
                f"tokens={matched_prefix_len}/{prompt_len} | rest={rest_count} | "
                f"stream={body.get('stream', False)} | thinking={enable_thinking}",
                request_id=request_id, stage="REQ",
            )

            # ── DPC: Auto-capture is handled by auto-save + prefix hash in
            # _insert_cache_entries → no need for seed file refresh. ────────

            _terminal_status(
                "⏳",
                f"Prefill started | wait={wait_seconds:.2f}s | rest_tokens={rest_count} | "
                f"session={session_ctx.session_id[:16]} ({cache_selection_source}) | family={SETTINGS.model_family}",
                request_id=request_id, stage="PREFILL",
            )
            # Expert Routing: start tracking this request (non-streaming path)
            if FEATURE_EXPERT_ROUTING_LOG:
                try:
                    from .expert_cache import start_expert_routing
                    start_expert_routing(request_id)
                except (ImportError, Exception) as _er_err:
                    logger.warning("[EXPERT_ROUTING] No se pudo iniciar expert routing: %s", _er_err)
            if FEATURE_FULL_LOGGING:
                _pipeline_log("PRE_GEN", request_id,
                    f"entering generation | rest={rest_count} | stream={is_streaming} | "
                    f"{_metal_mem_str()}")
            _mem_profiler.snapshot(request_id, "PRE_PREFILL", is_anthropic=self._is_anthropic, rest_tokens=rest_count, kv_cache_offset=_kv_off)
            _mem_profiler.reset_peak()

            _mtp_stats: Optional[Dict[str, Any]] = None
            _task_detector: Any = None

            if not is_streaming:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                if FEATURE_DIAGNOSTIC_HEADERS:
                    self.send_header("X-Pipeline-Compression-Ms", f"{_pipeline_timings.get('compress', 0):.1f}")
                    self.send_header("X-Pipeline-RAG-Ms", f"{_pipeline_timings.get('rag', 0):.1f}")
                    self.send_header("X-Pipeline-Heal-Ms", f"{_pipeline_timings.get('heal', 0):.1f}")
                self.end_headers()

                _pre_prefill_memory_relief(request_id, rest_count)

                if not is_vlm:
                    # ── FASE D.3: GENERATE (adaptive prefill + checkpoint) ────
                    _pipeline_generate(ctx, _server_state)
                    rest_tokens = ctx.rest_tokens if ctx.rest_tokens else rest_tokens
                    hybrid_generation_checkpoint = ctx.checkpoint

                _prefill_done, _prefill_thread = _start_prefill_progress(request_id, rest_count)

                generated_parts = []
                # ThinkingTracker: reset for this request (SSoT)
                _thinking_tracker.reset(enable_thinking=enable_thinking)
                _max_thinking_ns = SETTINGS.max_thinking_tokens
                progress_last_at = time.time()

                from .sampling import DynamicTaskDetector
                _client_has_custom_temp = body.get("temperature") is not None and isinstance(body.get("temperature"), (int, float))
                _task_detector = DynamicTaskDetector(
                    sampler=sampler,
                    request_id=request_id,
                    client_has_custom_temp=_client_has_custom_temp,
                    client_temp=float(body.get("temperature")) if _client_has_custom_temp else None,
                    log_fn=_terminal_status,
                    pipeline_log_fn=_pipeline_log if FEATURE_FULL_LOGGING else None,
                )

                _terminal_status(
                    "⚙️",
                    f"Generation started (decoding on GPU) | max_tokens={max_tokens} | thinking={enable_thinking}",
                    request_id=request_id, stage="GEN",
                )

                for response in _stream_generate_unified(
                    rest_tokens,
                    max_tokens,
                    sampler,
                    prompt_cache,
                    vlm_pixel_values=vlm_pixel_values,
                    vlm_mask=vlm_mask,
                    vlm_kwargs=vlm_kwargs,
                    cache_match_type=cache_match_type,
                    logits_processors=logits_processors,
                ):
                    generated_parts.append(response.text)
                    generated_tokens.append(int(response.token))
                    if hasattr(response, "drafts_attempted") and response.drafts_attempted > 0:
                        _mtp_stats = {
                            "alpha": response.alpha,
                            "accepted": response.drafts_accepted,
                            "attempted": response.drafts_attempted,
                            "draft_accepted": getattr(response, "draft_accepted", True),
                        }
                    if _GPU_YIELD_SECONDS > 0:
                        time.sleep(_GPU_YIELD_SECONDS)
                    if first_token_at is None:
                        first_token_at = time.time()
                        _prefill_done.set()  # Stop prefill progress
                    # Feed token to tracker (SSoT)
                    _ns_event = _thinking_tracker.feed(int(response.token), response.text)
                    _task_detector.feed(response.text, _thinking_tracker.is_responding)
                    if _ns_event != ThinkingEvent.NONE:
                        _pipeline_log("THINK", request_id,
                            f"{_ns_event.name} at token {len(generated_tokens)} (non-stream)")
                    # THINK_LOOP_BREAK: DISABLED — was a workaround for when
                    # repetition_penalty wasn't read. Now rep penalty works and
                    # n-gram detector handles real loops. Re-enable if needed.
                    # if _thinking_tracker.is_looping:
                    #     ...
                    #     break
                    # THINKING_LIMIT (v1 only — v2 handles via logits processor)
                    if (not isinstance(_thinking_tracker, ThinkingTrackerV2)
                            and _thinking_tracker.is_thinking and _max_thinking_ns > 0):
                        if _thinking_tracker.thinking_count >= _max_thinking_ns:
                            _terminal_status(
                                "🛑",
                                f"THINKING LIMIT: {_thinking_tracker.thinking_count} tokens in <think> "
                                f"(limit={_max_thinking_ns}). Breaking generation.",
                                indent=1,
                            )
                            break
                    if (
                        len(generated_tokens) % 64 == 0
                        and (time.time() - progress_last_at) >= 1.0
                    ):
                        progress_last_at = time.time()
                        _decode_tps = len(generated_tokens) / (time.time() - first_token_at) if first_token_at else 0
                        _mtp_str = (
                            f" | α={_mtp_stats['alpha']:.1f}% ({_mtp_stats['accepted']}/{_mtp_stats['attempted']})"
                            if _mtp_stats and _mtp_stats.get("attempted", 0) > 0
                            else ""
                        )
                        if _is_compact:
                            _stage_str = f"T={getattr(sampler, 'resp_temp', 0.0):.2f} [COMPACT]"
                        elif _thinking_tracker and getattr(_thinking_tracker, "is_thinking", False):
                            _stage_str = f"T={getattr(sampler, 'think_temp', 0.50):.2f} (THINK)"
                        elif locals().get("_tool_call_tracker") and getattr(_tool_call_tracker, "is_buffering", False):
                            _tag = getattr(_task_detector, "tag", "TOOLS") if _task_detector and getattr(_task_detector, "tag", "DEFAULT") != "DEFAULT" else "TOOLS"
                            _stage_str = f"T={getattr(sampler, 'resp_temp', 0.10):.2f} [{_tag}]"
                        else:
                            _tag = getattr(_task_detector, "tag", "DEFAULT") if _task_detector else "DEFAULT"
                            _stage_str = f"T={getattr(sampler, 'resp_temp', getattr(sampler, 'temp', 0.30)):.2f} [{_tag}]"
                        _terminal_status(
                            "⏳",
                            f"generated_tokens={len(generated_tokens)} | {_decode_tps:.1f} tok/s | {_stage_str}{_mtp_str} | {_metal_mem_str()}",
                            request_id=request_id, stage="DECODE",
                        )
                # ── THINKING CLEANUP (non-stream) ─────────────────────────
                # If generation ended while still in THINKING state, inject </think>.
                # V2 doesn't need cleanup — </think> was generated via logits processor.
                if (not isinstance(_thinking_tracker, ThinkingTrackerV2)
                        and enable_thinking and _thinking_tracker.is_thinking and generated_parts):
                    _joined_tc = "".join(generated_parts)
                    _func_m = re.search(r'<?function=\w', _joined_tc, re.IGNORECASE)
                    if _func_m:
                        _fpos = _func_m.start()
                        _inject = "</think>\n"
                        if "</tool_call>" in _joined_tc.lower() and "<tool_call>" not in _joined_tc.lower():
                            _inject += "<tool_call>\n"
                        generated_parts.clear()
                        generated_parts.append(_joined_tc[:_fpos] + _inject + _joined_tc[_fpos:])
                        _pipeline_log("GEN", request_id,
                            f"THINK_CLEANUP: injected </think> + <tool_call> before bare function call (non-stream)")
                    else:
                        generated_parts.append("</think>\n")
                        _pipeline_log("GEN", request_id,
                            "THINK_CLEANUP: injected synthetic </think> after forced break (non-stream)")
                    _thinking_tracker.force_exit()
                    _pipeline_log("THINK", request_id,
                        f"FORCE_EXIT (non-stream) | thinking={_thinking_tracker.thinking_count} responding={_thinking_tracker.responding_count}")

                response_text = "".join(generated_parts)
                raw_response_text = response_text
                response_text = _normalize_assistant_text(
                    response_text, enable_thinking, SETTINGS.model_family
                )
                # DEBUG: raw model output before tool extraction
                if FEATURE_FULL_LOGGING:
                    _pipeline_log("RAW_OUT", request_id,
                        f"raw_response ({len(response_text)} chars): {repr(response_text[:1000])}")
                message_text, tool_calls = _extract_openai_tool_calls(
                    response_text, SETTINGS.model_family, allowed_tools=tools or []
                )
                # TOOL_RAW: exactly what the model generated (pre-processing)
                if tool_calls:
                    _pipeline_log("TOOL_RAW", request_id,
                        f"model_output: {_fmt_tc_for_log(tool_calls)}")
                # TOOL_COMPAT: normalize aliases
                if tool_calls:
                    tool_calls, _alias_count = _sanitize_tool_calls(tool_calls, request_id)
                    if _alias_count:
                        _pipeline_log("TOOL_COMPAT", request_id,
                            f"normalized {_alias_count} tool call(s) (alias remapping)")
                    # TOOL_OUT: exactly what goes to Hermes (post-processing)
                    _pipeline_log("TOOL_OUT", request_id,
                        f"to_client: {_fmt_tc_for_log(tool_calls)}")
                # Hide <think> blocks from the client whenever reasoning was requested.
                if enable_thinking:
                    message_text = _strip_thinking_from_content(message_text)
                    # THINKING_LEAK_GUARD: if tracker says ALL tokens were thinking
                    # (responding_count==0), the model never produced visible content.
                    # _strip_thinking_from_content may fail to strip when <think> tag
                    # is missing (qwen3/deepseek/hermes/glm4 skip _normalize injection).
                    if _thinking_tracker.responding_count == 0 and _thinking_tracker.thinking_count > 0:
                        message_text = ""

                    # Extract last user message for healing hash context
                    _heal_user_ctx = ""
                    for _hm in reversed(raw_messages):
                        if (_hm.get("role") or "").strip().lower() == "user":
                            _hc = _hm.get("content", "")
                            _heal_user_ctx = _hc if isinstance(_hc, str) else str(_hc)[:500]
                            break
                    _update_healing_store(raw_response_text, message_text, tool_calls, _heal_user_ctx)

                finish_reason = "tool_calls" if tool_calls else "stop"
                # LOOP BREAK ESCALATION: if LOOP_BREAK fired but model still
                # generated tool_calls, strip them and force stop. The model
                # ignored the injected instruction — intercept at output level.
                if _loop_broken >= 3 and tool_calls:
                    _terminal_status("🛑",
                        f"LOOP ESCALATION: model ignored LOOP_BREAK, stripping tool_calls",
                        indent=1)
                    message_text = (
                        "Your last tool call was identical to a previous one and was blocked "
                        "to prevent an infinite loop. Continue working on your current task "
                        "using a different approach or different arguments."
                    )
                    tool_calls = []
                    finish_reason = "stop"
                # ── FASE D.4: POSTPROCESS (RadixAttention insertion + telemetry) ──
                # NOTE: NO extender cache_key con generated_tokens aquí.
                # postprocess() maneja internamente la bifurcación:
                #   pure-KV: inserta base_key + generated_tokens
                #   híbrido: inserta base_key[:checkpoint_len] (descarta generated)
                ctx.cache_key = list(cache_key)
                ctx.generated_tokens = list(generated_tokens)
                ctx.finish_reason = finish_reason
                ctx.skip_cache_store = _is_housekeeping
                _pipeline_postprocess(ctx, _server_state, RADIX_PROMPT_CACHE)

                if _frozen_summary:
                    message_text = f"{_frozen_summary}\n\n---\n\n{message_text}"

                if self._is_anthropic:
                    full_response = openai_to_anthropic_response(
                        message_text, tool_calls, finish_reason,
                        self._anthropic_model,
                        prompt_input_tokens=len(model_tokens),
                    )
                    self.wfile.write(json.dumps(full_response).encode("utf-8"))
                else:
                    response_id = f"chatcmpl-{int(time.time())}"
                    full_response = {
                        "id": response_id,
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": SETTINGS.proxy_model_id,
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": message_text},
                                "finish_reason": finish_reason,
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "total_tokens": 0,
                        },
                    }
                    if tool_calls:
                        full_response["choices"][0]["message"]["tool_calls"] = tool_calls
                    self.wfile.write(json.dumps(full_response).encode("utf-8"))
                timing = _build_timing_dict(first_token_at, generation_started_at, rest_count, generated_tokens, mtp_stats=_mtp_stats)
                if request_logger:
                    _think_log = _extract_thinking_text(raw_response_text)
                    request_logger.log(
                        "generation",
                        {
                            "mode": "non-stream",
                            "timing": timing,
                            "thinking_text": _think_log,
                            "thinking_tokens": _thinking_tracker.thinking_count,
                            "response_text": message_text,
                            "raw_response_text": raw_response_text,
                            "normalized_response_text": response_text,
                            "assistant_message_text": message_text,
                            "tool_calls": tool_calls,
                            "finish_reason": finish_reason,
                        },
                        request_id=request_id,
                    )

                if FEATURE_FULL_LOGGING:
                    _log_generation_telemetry(
                        request_id, generation_started_at, generated_tokens,
                        message_text, tool_calls, enable_thinking, finish_reason,
                        timing,
                        thinking_token_count=_thinking_tracker.thinking_count if enable_thinking else None,
                    )
            else:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                if self._is_anthropic:
                    self.send_header("Connection", "close")
                else:
                    self.send_header("Connection", "keep-alive")
                self.end_headers()

                response_id = f"chatcmpl-{int(time.time())}"
                # ── ANTHROPIC: send message_start + content_block_start immediately ──
                _anthropic_streaming = self._is_anthropic
                _anthropic_streamed_text = []  # chunks already sent via SSE
                _anthropic_block_idx = 0
                # ThinkingTracker: reset for this request (SSoT)
                _thinking_tracker.reset(enable_thinking=enable_thinking)
                _tool_call_tracker = ToolCallTracker()  # Fresh tracker per request
                _max_thinking = SETTINGS.max_thinking_tokens  # 0=unlimited
                if _anthropic_streaming:
                    _msg_id = f"msg_{uuid.uuid4().hex[:24]}"
                    _msg_start = _sse_event("message_start", {
                        "type": "message_start",
                        "message": {
                            "id": _msg_id, "type": "message", "role": "assistant",
                            "model": self._anthropic_model, "content": [],
                            "stop_reason": None, "stop_sequence": None,
                            "usage": {"input_tokens": len(model_tokens), "output_tokens": 0},
                        },
                    })
                    # Start with thinking block when thinking is enabled,
                    # otherwise start with text block directly.
                    if enable_thinking:
                        _block_start = _sse_event("content_block_start", {
                            "type": "content_block_start", "index": 0,
                            "content_block": {"type": "thinking", "thinking": ""},
                        })
                    else:
                        _block_start = _sse_event("content_block_start", {
                            "type": "content_block_start", "index": 0,
                            "content_block": {"type": "text", "text": ""},
                        })
                    self.wfile.write(_msg_start.encode("utf-8"))
                    self.wfile.write(_block_start.encode("utf-8"))
                    self.wfile.flush()
                    _pipeline_log("WIRE", request_id, f"ANTHROPIC SSE: sent message_start + content_block_start (thinking={enable_thinking})")
                    if _frozen_summary:
                        _frozen_prefix = f"{_frozen_summary}\n\n---\n\n"
                        _delta_evt = _sse_event("content_block_delta", {
                            "type": "content_block_delta",
                            "index": _anthropic_block_idx,
                            "delta": {"type": "text_delta", "text": _frozen_prefix},
                        })
                        self.wfile.write(_delta_evt.encode("utf-8"))
                        self.wfile.flush()
                        _anthropic_streamed_text.append(_frozen_prefix)
                else:
                    role_chunk = {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": SETTINGS.proxy_model_id,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"role": "assistant"},
                                "finish_reason": None,
                            }
                        ],
                    }
                    _wire_payload = f"data: {json.dumps(role_chunk)}\n\n"
                    _pipeline_log("WIRE", request_id, f"SEND role_chunk | len={len(_wire_payload)}")
                    self.wfile.write(_wire_payload.encode("utf-8"))
                    self.wfile.flush()
                    _pipeline_log("WIRE", request_id, "FLUSH role_chunk OK")
                    if _frozen_summary:
                        _frozen_prefix = f"{_frozen_summary}\n\n---\n\n"
                        _chunk = {
                            "id": response_id,
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": SETTINGS.proxy_model_id,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": _frozen_prefix},
                                    "finish_reason": None,
                                }
                            ],
                        }
                        self.wfile.write(f"data: {json.dumps(_chunk)}\n\n".encode("utf-8"))
                        self.wfile.flush()

                # --- SSE KEEPALIVE THREAD ---
                # During prefill (~58s) and decode, no SSE events are sent because
                # we must collect all tokens to post-process (strip <think>, extract
                # tool_calls).  Without keepalives the idle TCP connection dies and
                # the client sees \"Connection error\".
                # SSE spec: lines starting with ':' are comments, ignored by clients.
                _keepalive_stop = threading.Event()
                _keepalive_wfile = self.wfile  # capture for thread
                _keepalive_interval = 5  # seconds
                _wfile_lock = threading.Lock()  # protect concurrent writes

                def _sse_keepalive_sender():
                    while not _keepalive_stop.wait(_keepalive_interval):
                        try:
                            with _wfile_lock:
                                _keepalive_wfile.write(b": keepalive\n\n")
                                _keepalive_wfile.flush()
                        except Exception:
                            break  # client disconnected

                _keepalive_thread = threading.Thread(
                    target=_sse_keepalive_sender, daemon=True
                )
                _keepalive_thread.start()

                _pre_prefill_memory_relief(request_id, rest_count)

                if not is_vlm:
                    # ── FASE D.3: GENERATE (adaptive prefill + checkpoint) ────
                    _pipeline_generate(ctx, _server_state)
                    rest_tokens = ctx.rest_tokens if ctx.rest_tokens else rest_tokens
                    hybrid_generation_checkpoint = ctx.checkpoint

                _prefill_done_s, _ = _start_prefill_progress(request_id, rest_count)

                raw_parts = []

                # Expert Routing: start tracking this request (streaming path)
                if FEATURE_EXPERT_ROUTING_LOG:
                    try:
                        from .expert_cache import start_expert_routing
                        start_expert_routing(request_id)
                    except (ImportError, Exception) as _er_err:
                        logger.warning("[EXPERT_ROUTING] No se pudo iniciar expert routing: %s", _er_err)

                progress_last_at = time.time()
                from .sampling import DynamicTaskDetector
                _client_has_custom_temp = body.get("temperature") is not None and isinstance(body.get("temperature"), (int, float))
                _task_detector = DynamicTaskDetector(
                    sampler=sampler,
                    request_id=request_id,
                    client_has_custom_temp=_client_has_custom_temp,
                    client_temp=float(body.get("temperature")) if _client_has_custom_temp else None,
                    log_fn=_terminal_status,
                    pipeline_log_fn=_pipeline_log if FEATURE_FULL_LOGGING else None,
                )

                _server_loop_warned = False
                _server_loop_warned_token = 0
                _terminal_status(
                    "⚙️",
                    f"Generation started (decoding on GPU) | max_tokens={max_tokens} | thinking={enable_thinking}",
                    request_id=request_id, stage="GEN",
                )

                try:
                    for response in _stream_generate_unified(
                        rest_tokens,
                        max_tokens,
                        sampler,
                        prompt_cache,
                        vlm_pixel_values=vlm_pixel_values,
                        vlm_mask=vlm_mask,
                        vlm_kwargs=vlm_kwargs,
                        cache_match_type=cache_match_type,
                        logits_processors=logits_processors,
                    ):
                        generated_tokens.append(int(response.token))
                        if hasattr(response, "drafts_attempted") and response.drafts_attempted > 0:
                            _mtp_stats = {
                                "alpha": response.alpha,
                                "accepted": response.drafts_accepted,
                                "attempted": response.drafts_attempted,
                                "draft_accepted": getattr(response, "draft_accepted", True),
                            }
                        if _GPU_YIELD_SECONDS > 0:
                            time.sleep(_GPU_YIELD_SECONDS)
                        if first_token_at is None:
                            first_token_at = time.time()
                            _prefill_done_s.set()  # Stop prefill progress
                        response_text = response.text

                        # ── THINKING TRACKER (SSoT) ──────────────────────
                        # Feed every token to the centralized tracker.
                        # It detects <think>/<\/think> transitions, counts tokens,
                        # and detects thinking loops (repeated </think>).
                        _think_event = _thinking_tracker.feed(int(response.token), response_text)
                        _task_detector.feed(response_text, _thinking_tracker.is_responding)
                        if _think_event != ThinkingEvent.NONE:
                            _pipeline_log("THINK", request_id,
                                f"{_think_event.name} at token {len(generated_tokens)} (stream)")

                        # THINKING_LIMIT: break if too many tokens in thinking
                        # V1 only — v2 handles via logits processor (no break needed)
                        if (not isinstance(_thinking_tracker, ThinkingTrackerV2)
                                and _thinking_tracker.is_thinking and _max_thinking > 0):
                            if _thinking_tracker.thinking_count >= _max_thinking:
                                _terminal_status(
                                    "🛑",
                                    f"THINKING LIMIT: {_thinking_tracker.thinking_count} tokens in <think> block "
                                    f"(limit={_max_thinking}). Forcing generation stop.",
                                    indent=1,
                                )
                                _pipeline_log("GEN", request_id,
                                    f"THINKING_LIMIT_HIT: {_thinking_tracker.thinking_count} thinking tokens, "
                                    f"limit={_max_thinking}. Breaking generation loop.")
                                break

                        # THINK_LOOP_BREAK: DISABLED — was a workaround for when
                        # repetition_penalty wasn't read. Now rep penalty works and
                        # n-gram detector handles real loops. Re-enable if needed.
                        # if _thinking_tracker.is_looping:
                        #     ...
                        #     break

                        # ── N-GRAM LOOP DETECTION ────────────────────────
                        # Controlled via SETTINGS (SSoT): disabled by default (false)
                        # to prevent false positives on repetitive code/matrices/JSON.
                        if SETTINGS.ngram_loop_detection:
                            _NGRAM_SIZE = SETTINGS.ngram_size
                            _NGRAM_MAX_REPEATS = SETTINGS.ngram_max_repeats
                            _NGRAM_CHECK_INTERVAL = SETTINGS.ngram_check_interval
                            _NGRAM_WINDOW = SETTINGS.ngram_window
                            _n_gen = len(generated_tokens)
                            if (
                                _n_gen >= _NGRAM_SIZE * 2
                                and _n_gen % _NGRAM_CHECK_INTERVAL == 0
                            ):
                                _tail = tuple(generated_tokens[-_NGRAM_SIZE:])
                                _search_region = generated_tokens[:-_NGRAM_SIZE]
                                # Limit search to recent window — real loops are nearby
                                _window_start = max(0, len(_search_region) - _NGRAM_WINDOW)
                                _search_region = _search_region[_window_start:]
                                _repeat_count = 0
                                for _si in range(len(_search_region) - _NGRAM_SIZE + 1):
                                    if tuple(_search_region[_si:_si + _NGRAM_SIZE]) == _tail:
                                        _repeat_count += 1
                                        if _repeat_count >= _NGRAM_MAX_REPEATS:
                                            break
                                if _repeat_count >= _NGRAM_MAX_REPEATS:
                                    if SETTINGS.ngram_nudge_enabled and not _server_loop_warned:
                                        _server_loop_warned = True
                                        _server_loop_warned_token = _n_gen
                                        _terminal_status(
                                            "⚠️",
                                            f"NGRAM LOOP DETECTADO: {_NGRAM_SIZE}-token sequence repeated "
                                            f"{_repeat_count + 1}x after {_n_gen} tokens. Steering Nudge activado.",
                                            indent=1,
                                        )
                                        _pipeline_log("GEN", request_id,
                                            f"NGRAM_LOOP_WARN: {_NGRAM_SIZE}-gram repeated "
                                            f"{_repeat_count + 1}x at token {_n_gen}. Steering Nudge activado.")
                                    else:
                                        if not SETTINGS.ngram_nudge_enabled or (_n_gen - _server_loop_warned_token) >= SETTINGS.ngram_grace_tokens:
                                            _terminal_status(
                                                "🛑",
                                                f"NGRAM LOOP: {_NGRAM_SIZE}-token sequence repeated "
                                                f"{_repeat_count + 1}x after {_n_gen} tokens. "
                                                f"Forcing generation stop.",
                                                indent=1,
                                            )
                                            _pipeline_log("GEN", request_id,
                                                f"NGRAM_LOOP_BREAK: {_NGRAM_SIZE}-gram repeated "
                                                f"{_repeat_count + 1}x at token {_n_gen}. "
                                                f"Breaking generation loop.")
                                            break
                        if response_text:
                            raw_parts.append(response_text)
                            if hasattr(sampler, "feed_thinking_text"):
                                sampler.feed_thinking_text(response_text)
                            # ── ANTHROPIC LIVE STREAMING (tracker-based) ────
                            if _anthropic_streaming:
                                if _think_event == ThinkingEvent.EXIT_THINKING:
                                    # Transition: close thinking block, open text block
                                    with _wfile_lock:
                                        self.wfile.write(_sse_event("content_block_stop", {
                                            "type": "content_block_stop", "index": 0,
                                        }).encode("utf-8"))
                                        _anthropic_block_idx = 1
                                        self.wfile.write(_sse_event("content_block_start", {
                                            "type": "content_block_start", "index": 1,
                                            "content_block": {"type": "text", "text": ""},
                                        }).encode("utf-8"))
                                        self.wfile.flush()
                                elif _thinking_tracker.is_thinking:
                                    # Still in thinking — stream as thinking_delta
                                    with _wfile_lock:
                                        self.wfile.write(_sse_event("content_block_delta", {
                                            "type": "content_block_delta", "index": 0,
                                            "delta": {"type": "thinking_delta", "thinking": response_text},
                                        }).encode("utf-8"))
                                        self.wfile.flush()
                                elif _thinking_tracker.is_responding:
                                    # In response mode — use ToolCallTracker to decide
                                    # what to stream vs buffer
                                    _tc_event = _tool_call_tracker.feed(response_text)
                                    if _tc_event == ToolCallEvent.ENTER_TOOL_CALL:
                                        _pipeline_log("TOOL_TRACK", request_id,
                                            f"ENTER_TOOL_CALL at token {len(generated_tokens)}")
                                    elif _tc_event == ToolCallEvent.EXIT_TOOL_CALL:
                                        _tc_last = _tool_call_tracker.last_completed
                                        _pipeline_log("TOOL_TRACK", request_id,
                                            f"EXIT_TOOL_CALL at token {len(generated_tokens)} | "
                                            f"has_params={_tc_last['has_parameters'] if _tc_last else '?'} | "
                                            f"incomplete={_tc_last.get('incomplete', False) if _tc_last else '?'}")

                                    # Stream flushable text (text before/between tool calls)
                                    _flush = _tool_call_tracker.flushable_text
                                    if _flush and not _tool_call_tracker.is_buffering:
                                        try:
                                            with _wfile_lock:
                                                self.wfile.write(_sse_event("content_block_delta", {
                                                    "type": "content_block_delta", "index": _anthropic_block_idx,
                                                    "delta": {"type": "text_delta", "text": _flush},
                                                }).encode("utf-8"))
                                                self.wfile.flush()
                                            _anthropic_streamed_text.append(_flush)
                                        except BrokenPipeError:
                                            _pipeline_log("WIRE", request_id,
                                                "❌ BROKEN PIPE during Anthropic SSE streaming")
                                            raise
                        if (
                            len(generated_tokens) % 64 == 0
                            and (time.time() - progress_last_at) >= 1.0
                        ):
                            progress_last_at = time.time()
                            _decode_tps = len(generated_tokens) / (time.time() - first_token_at) if first_token_at else 0
                            _mtp_str = (
                                f" | α={_mtp_stats['alpha']:.1f}% ({_mtp_stats['accepted']}/{_mtp_stats['attempted']})"
                                if _mtp_stats and _mtp_stats.get("attempted", 0) > 0
                                else ""
                            )
                            if _is_compact:
                                _stage_str = f"T={getattr(sampler, 'resp_temp', 0.0):.2f} [COMPACT]"
                            elif _thinking_tracker and getattr(_thinking_tracker, "is_thinking", False):
                                _stage_str = f"T={getattr(sampler, 'think_temp', 0.50):.2f} (THINK)"
                            elif locals().get("_tool_call_tracker") and getattr(_tool_call_tracker, "is_buffering", False):
                                _tag = getattr(_task_detector, "tag", "TOOLS") if _task_detector and getattr(_task_detector, "tag", "DEFAULT") != "DEFAULT" else "TOOLS"
                                _stage_str = f"T={getattr(sampler, 'resp_temp', 0.10):.2f} [{_tag}]"
                            else:
                                _tag = getattr(_task_detector, "tag", "DEFAULT") if _task_detector else "DEFAULT"
                                _stage_str = f"T={getattr(sampler, 'resp_temp', getattr(sampler, 'temp', 0.30)):.2f} [{_tag}]"
                            _terminal_status(
                                "⏳",
                                f"generated_tokens={len(generated_tokens)} | {_decode_tps:.1f} tok/s | {_stage_str}{_mtp_str} | {_metal_mem_str()}",
                                request_id=request_id, stage="DECODE",
                            )
                finally:
                    # Stop keepalive thread BEFORE writing actual content chunks
                    _keepalive_stop.set()
                    _keepalive_thread.join(timeout=2)

                # ── THINKING CLEANUP (stream) ──────────────────────────────
                # If generation ended while still in THINKING state (e.g. THINKING_LIMIT
                # or NGRAM_LOOP broke mid-thinking), inject synthetic </think> and
                # properly transition Anthropic SSE blocks.
                if (not isinstance(_thinking_tracker, ThinkingTrackerV2)
                        and enable_thinking and _thinking_tracker.is_thinking and raw_parts):
                    _joined_tc = "".join(raw_parts)
                    _func_m = re.search(r'<?function=\w', _joined_tc, re.IGNORECASE)
                    if _func_m:
                        _fpos = _func_m.start()
                        _inject = "</think>\n"
                        if "</tool_call>" in _joined_tc.lower() and "<tool_call>" not in _joined_tc.lower():
                            _inject += "<tool_call>\n"
                        raw_parts.clear()
                        raw_parts.append(_joined_tc[:_fpos] + _inject + _joined_tc[_fpos:])
                    else:
                        raw_parts.append("</think>\n")
                    # Mark tracker as exited so SSE output uses streaming path
                    _thinking_tracker.force_exit()
                    _pipeline_log("THINK", request_id,
                        f"FORCE_EXIT (stream) | thinking={_thinking_tracker.thinking_count} responding={_thinking_tracker.responding_count}")
                    if _anthropic_streaming:
                        # Close the open thinking block (index 0)
                        self.wfile.write(_sse_event("content_block_stop", {
                            "type": "content_block_stop", "index": 0,
                        }).encode("utf-8"))
                        # Open text block (index 1) for any post-think content
                        _anthropic_block_idx = 1
                        self.wfile.write(_sse_event("content_block_start", {
                            "type": "content_block_start", "index": 1,
                            "content_block": {"type": "text", "text": ""},
                        }).encode("utf-8"))
                        self.wfile.flush()
                    _pipeline_log("GEN", request_id,
                        f"THINK_CLEANUP: injected </think>{' + <tool_call>' if _func_m else ''} "
                        f"({'before bare function call' if _func_m else 'after forced break'}) (stream)")

                # ── TOOL CALL TRACKER FINALIZE ─────────────────────────────
                _tc_final = _tool_call_tracker.finalize()
                if _tc_final == ToolCallEvent.EXIT_TOOL_CALL:
                    _tc_last = _tool_call_tracker.last_completed
                    _pipeline_log("TOOL_TRACK", request_id,
                        f"INCOMPLETE at EOS | has_params={_tc_last['has_parameters'] if _tc_last else '?'}")
                if _tool_call_tracker.call_count > 0:
                    _pipeline_log("TOOL_TRACK", request_id,
                        f"summary: {_tool_call_tracker.summary()}")
                # Flush any remaining holdback text
                _tc_remaining = _tool_call_tracker.flushable_text
                if _tc_remaining and _anthropic_streaming:
                    try:
                        with _wfile_lock:
                            self.wfile.write(_sse_event("content_block_delta", {
                                "type": "content_block_delta", "index": _anthropic_block_idx,
                                "delta": {"type": "text_delta", "text": _tc_remaining},
                            }).encode("utf-8"))
                            self.wfile.flush()
                        _anthropic_streamed_text.append(_tc_remaining)
                    except BrokenPipeError:
                        pass

                full_text = "".join(raw_parts)
                raw_full_text = full_text
                full_text = _normalize_assistant_text(
                    full_text, enable_thinking, SETTINGS.model_family
                )
                # DEBUG: raw model output before tool extraction (streaming path)
                if FEATURE_FULL_LOGGING:
                    _pipeline_log("RAW_OUT", request_id,
                        f"raw_response ({len(full_text)} chars): {repr(full_text[:2000])}")
                message_text, tool_calls = _extract_openai_tool_calls(
                    full_text, SETTINGS.model_family, allowed_tools=tools or []
                )
                # TOOL_RAW: exactly what the model generated (pre-processing)
                if tool_calls:
                    _pipeline_log("TOOL_RAW", request_id,
                        f"model_output: {_fmt_tc_for_log(tool_calls)}")
                # Audit stage 2: extracted args
                if FEATURE_TOOL_AUDIT_LOG and tool_calls:
                    import hashlib as _hl
                    for _atc in tool_calls:
                        _afn = _atc.get("function", {})
                        try:
                            _aargs = json.loads(_afn.get("arguments", "{}"))
                        except Exception as _tc_json_err:
                            logger.warning("[TOOL_AUDIT] Error parseando JSON de arguments: %s", _tc_json_err)
                            _aargs = {}
                        _aarg_lens = {k: len(str(v)) for k, v in _aargs.items()}
                        _aargs_json = _afn.get("arguments", "")
                        _tool_audit_write(request_id, "extracted", {
                            "tool": _afn.get("name", "?"),
                            "args_keys": list(_aargs.keys()),
                            "arg_lens": _aarg_lens,
                            "args_json_len": len(_aargs_json),
                            "args_json_hash": hashlib.md5(_aargs_json.encode()).hexdigest()[:8],
                        })
                # TOOL_COMPAT: normalize aliases
                if tool_calls:
                    tool_calls, _alias_count = _sanitize_tool_calls(tool_calls, request_id)
                    if _alias_count:
                        _pipeline_log("TOOL_COMPAT", request_id,
                            f"normalized {_alias_count} tool call(s) (alias remapping)")
                    # TOOL_OUT: exactly what goes to Hermes (post-processing)
                    _pipeline_log("TOOL_OUT", request_id,
                        f"to_client: {_fmt_tc_for_log(tool_calls)}")
                # Hide <think> blocks from the client whenever reasoning was requested.
                if enable_thinking:
                    message_text = _strip_thinking_from_content(message_text)
                    # THINKING_LEAK_GUARD: if tracker says ALL tokens were thinking
                    # (responding_count==0), the model never produced visible content.
                    # _strip_thinking_from_content may fail to strip when <think> tag
                    # is missing (qwen3/deepseek/hermes/glm4 skip _normalize injection).
                    if _thinking_tracker.responding_count == 0 and _thinking_tracker.thinking_count > 0:
                        message_text = ""

                finish_reason = "tool_calls" if tool_calls else "stop"
                # LOOP BREAK ESCALATION: if LOOP_BREAK fired but model still
                # generated tool_calls, strip them and force stop.
                if _loop_broken >= 3 and tool_calls:
                    _terminal_status("🛑",
                        f"LOOP ESCALATION: model ignored LOOP_BREAK, stripping tool_calls",
                        indent=1)
                    message_text = (
                        "Your last tool call was identical to a previous one and was blocked "
                        "to prevent an infinite loop. Continue working on your current task "
                        "using a different approach or different arguments."
                    )
                    tool_calls = []
                    finish_reason = "stop"


                # ── ANTHROPIC SSE OUTPUT ──────────────────────────────────
                if self._is_anthropic:
                    if _anthropic_streaming and (_thinking_tracker.has_exited_thinking or not enable_thinking):
                        # SSE_TEXT_RECOVERY: safety net — if nothing was streamed
                        # as text_delta (e.g. model never emitted </think> token
                        # and THINK_CLEANUP injected it), send the full text now.
                        if message_text and not _anthropic_streamed_text:
                            with _wfile_lock:
                                self.wfile.write(_sse_event("content_block_delta", {
                                    "type": "content_block_delta",
                                    "index": _anthropic_block_idx,
                                    "delta": {"type": "text_delta", "text": message_text},
                                }).encode("utf-8"))
                                self.wfile.flush()
                            _anthropic_streamed_text.append(message_text)
                            _pipeline_log("WIRE", request_id,
                                f"SSE_TEXT_RECOVERY: sent {len(message_text)} chars "
                                f"(not streamed during generation)")
                        # We already streamed content_block_delta events during
                        # generation. Now close the current block and message.
                        _stop_reason = "end_turn"
                        if tool_calls:
                            _stop_reason = "tool_use"
                        elif finish_reason == "length":
                            _stop_reason = "max_tokens"
                        _closing_events = [
                            _sse_event("content_block_stop", {
                                "type": "content_block_stop", "index": _anthropic_block_idx,
                            }),
                        ]
                        # If there are tool_calls, add tool_use blocks
                        if tool_calls:
                            _tc_idx = _anthropic_block_idx + 1
                            for tc in tool_calls:
                                func = tc.get("function", {})
                                try:
                                    tc_input = json.loads(func.get("arguments", "{}"))
                                except (json.JSONDecodeError, TypeError):
                                    tc_input = {"raw": func.get("arguments", "")}
                                tc_id = tc.get("id", f"toolu_{uuid.uuid4().hex[:12]}")
                                _closing_events.append(_sse_event("content_block_start", {
                                    "type": "content_block_start", "index": _tc_idx,
                                    "content_block": {
                                        "type": "tool_use", "id": tc_id,
                                        "name": func.get("name", ""), "input": {},
                                    },
                                }))
                                # Emit input_json_delta in 512-char chunks.
                                # The real Anthropic API streams tool inputs incrementally.
                                # Sending one huge delta for large write_file content (>5KB JS)
                                # gets fragmented by TCP and can be cut by the client SDK buffer.
                                _tc_json = json.dumps(tc_input, ensure_ascii=False)
                                # Audit stage 3: delivered JSON to Hermes
                                if FEATURE_TOOL_AUDIT_LOG:
                                    import hashlib as _hl
                                    _tool_audit_write(request_id, "delivered", {
                                        "tool": func.get("name", "?"),
                                        "delivered_json_len": len(_tc_json),
                                        "delivered_json_hash": _hl.md5(_tc_json.encode()).hexdigest()[:8],
                                        "delivered_keys": list(tc_input.keys()),
                                        "delivered_arg_lens": {k: len(str(v)) for k, v in tc_input.items()},
                                    })
                                _TC_CHUNK = 512
                                for _jc_start in range(0, max(1, len(_tc_json)), _TC_CHUNK):
                                    _closing_events.append(_sse_event("content_block_delta", {
                                        "type": "content_block_delta", "index": _tc_idx,
                                        "delta": {
                                            "type": "input_json_delta",
                                            "partial_json": _tc_json[_jc_start:_jc_start + _TC_CHUNK],
                                        },
                                    }))
                                _closing_events.append(_sse_event("content_block_stop", {
                                    "type": "content_block_stop", "index": _tc_idx,
                                }))
                                _tc_idx += 1
                        _closing_events.append(_sse_event("message_delta", {
                            "type": "message_delta",
                            "delta": {"stop_reason": _stop_reason, "stop_sequence": None},
                            "usage": {"output_tokens": len(generated_tokens)},
                        }))
                        _closing_events.append(_sse_event("message_stop", {"type": "message_stop"}))
                        try:
                            for ev in _closing_events:
                                self.wfile.write(ev.encode("utf-8"))
                            self.wfile.flush()
                            self.close_connection = True
                            _streamed_chars = sum(len(s) for s in _anthropic_streamed_text)
                            _pipeline_log("WIRE", request_id,
                                f"ANTHROPIC SSE streamed | chars={_streamed_chars} | finish={_stop_reason}")
                        except BrokenPipeError:
                            _pipeline_log("WIRE", request_id,
                                "❌ BROKEN PIPE on Anthropic SSE closing — client disconnected")
                            raise
                    else:
                        # Fallback: model didn't think or streaming wasn't active.
                        # Use the original buffered approach.
                        anthropic_events = build_anthropic_sse_events(
                            message_text, tool_calls, finish_reason,
                            self._anthropic_model,
                            prompt_input_tokens=len(model_tokens),
                        )
                        try:
                            for ev in anthropic_events:
                                self.wfile.write(ev.encode("utf-8"))
                            self.wfile.flush()
                            self.close_connection = True
                            _pipeline_log("WIRE", request_id,
                                f"ANTHROPIC SSE delivered | blocks={len(anthropic_events)} | finish={finish_reason}")
                        except BrokenPipeError:
                            _pipeline_log("WIRE", request_id,
                                "❌ BROKEN PIPE on Anthropic SSE — client disconnected")
                            raise
                # ── OPENAI SSE OUTPUT (original) ─────────────────────────
                else:
                    if message_text:
                        chunk = {
                            "id": response_id,
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": SETTINGS.proxy_model_id,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": message_text},
                                    "finish_reason": None,
                                }
                            ],
                        }
                        _wire_payload = f"data: {json.dumps(chunk)}\n\n"
                        _pipeline_log("WIRE", request_id, f"SEND content_chunk | content_len={len(message_text)} | wire_len={len(_wire_payload)}")
                        try:
                            self.wfile.write(_wire_payload.encode("utf-8"))
                            self.wfile.flush()
                            _pipeline_log("WIRE", request_id, "FLUSH content_chunk OK")
                        except BrokenPipeError:
                            _pipeline_log("WIRE", request_id, "❌ BROKEN PIPE on content_chunk — client already disconnected")
                            raise

                    if tool_calls:
                        for idx, tc in enumerate(tool_calls):
                            tc_chunk = {
                                "id": response_id,
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": SETTINGS.proxy_model_id,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"tool_calls": [{**tc, "index": idx}]},
                                        "finish_reason": None,
                                    }
                                ],
                            }
                            _wire_payload = f"data: {json.dumps(tc_chunk)}\n\n"
                            _tc_name = tc.get('function', {}).get('name', '?')
                            _pipeline_log("WIRE", request_id, f"SEND tool_call_chunk[{idx}] | name={_tc_name} | wire_len={len(_wire_payload)}")
                            try:
                                self.wfile.write(_wire_payload.encode("utf-8"))
                                self.wfile.flush()
                                _pipeline_log("WIRE", request_id, f"FLUSH tool_call_chunk[{idx}] OK")
                            except BrokenPipeError:
                                _pipeline_log("WIRE", request_id, f"❌ BROKEN PIPE on tool_call_chunk[{idx}] — client already disconnected")
                                raise

                    final_chunk = {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": SETTINGS.proxy_model_id,
                        "choices": [
                            {"index": 0, "delta": {}, "finish_reason": finish_reason}
                        ],
                    }
                    _wire_payload = f"data: {json.dumps(final_chunk)}\n\n"
                    _pipeline_log("WIRE", request_id, f"SEND final_chunk | finish_reason={finish_reason} | wire_len={len(_wire_payload)}")
                    try:
                        self.wfile.write(_wire_payload.encode("utf-8"))
                        self.wfile.write(b"data: [DONE]\n\n")
                        self.wfile.flush()
                        _pipeline_log("WIRE", request_id, "FLUSH final+DONE OK — response fully delivered")
                    except BrokenPipeError:
                        _pipeline_log("WIRE", request_id, "❌ BROKEN PIPE on final_chunk — client disconnected before receiving response")
                        raise
                timing = _build_timing_dict(first_token_at, generation_started_at, rest_count, generated_tokens, mtp_stats=_mtp_stats)
                if request_logger:
                    _think_log = _extract_thinking_text(raw_full_text)
                    request_logger.log(
                        "generation",
                        {
                            "mode": "stream",
                            "timing": timing,
                            "thinking_text": _think_log,
                            "thinking_tokens": _thinking_tracker.thinking_count,
                            "response_text": message_text,
                            "raw_response_text": raw_full_text,
                            "normalized_response_text": full_text,
                            "assistant_message_text": message_text,
                            "tool_calls": tool_calls,
                            "finish_reason": finish_reason,
                        },
                        request_id=request_id,
                    )

                if FEATURE_FULL_LOGGING:
                    _log_generation_telemetry(
                        request_id, generation_started_at, generated_tokens,
                        message_text, tool_calls, enable_thinking, finish_reason,
                        timing,
                        thinking_token_count=_thinking_tracker.thinking_count if enable_thinking else None,
                    )

                # ── DEFERRED POST-PROCESSING ──────────────────────────────
                # These operations are heavy (cache trim with mx.eval, healing
                # store, pre-warmup) and were previously run BEFORE sending
                # closing events, causing a 2+ second SSE gap that could make
                # the client time out. Now they run AFTER the response is
                # fully delivered to the client.
                if enable_thinking:
                    # Extract last user message for healing hash context
                    _heal_user_ctx = ""
                    for _hm in reversed(raw_messages):
                        if (_hm.get("role") or "").strip().lower() == "user":
                            _hc = _hm.get("content", "")
                            _heal_user_ctx = _hc if isinstance(_hc, str) else str(_hc)[:500]
                            break
                    _update_healing_store(raw_full_text, message_text, tool_calls, _heal_user_ctx)

                # ── FASE D.4: POSTPROCESS (RadixAttention insertion + telemetry) ──
                ctx.cache_key = list(cache_key)
                ctx.generated_tokens = list(generated_tokens)
                ctx.finish_reason = finish_reason
                ctx.skip_cache_store = _is_housekeeping
                _pipeline_postprocess(ctx, _server_state, RADIX_PROMPT_CACHE)
        except BrokenPipeError:
            _terminal_status(
                "⚠️",
                f"Request {request_id} client disconnected (BrokenPipeError)",
                indent=1,
            )
            if request_logger:
                request_logger.log(
                    "generation",
                    "client disconnected (BrokenPipeError)",
                    request_id=request_id,
                )
            # ── CACHE SALVAGE ──────────────────────────────────────────────
            # _extract() removed the cache entry from the LRU store during
            # fetch (line ~1168). Normally _post_generation_cache_update
            # re-inserts it after generation. But BrokenPipeError skips that
            # path, leaving the LRU empty → next request cold-starts from 0%.
            # Salvage: re-insert the cache using the same post-generation
            # logic (restores hybrid checkpoint, trims generated tokens).
            # Wrapped in try/except: failure = current behavior (cache lost).
            try:
                if prompt_cache is not None and len(cache_key) > 0:
                    ctx.cache_key = list(cache_key)
                    ctx.generated_tokens = list(generated_tokens)
                    ctx.finish_reason = "disconnect"
                    _pipeline_postprocess(ctx, _server_state, RADIX_PROMPT_CACHE)
                    _terminal_status(
                        "♻️",
                        f"Request {request_id} cache salvaged after disconnect",
                        indent=1,
                    )
            except Exception as _salvage_err:
                _terminal_status(
                    "⚠️",
                    f"Request {request_id} cache salvage failed: {_salvage_err}",
                    indent=1,
                )
        except Exception as e:
            _err_str = str(e).lower()
            _is_oom = any(k in _err_str for k in ("out of memory", "memory", "allocation", "metal"))
            if _is_oom and not generated_tokens:
                # OOM during prefill, before any content was streamed.
                # Send SSE error event with "prompt is too long" to trigger
                # Claude Code's reactive compact (auto-compacts and retries).
                _terminal_status("🧠", f"Request {request_id} OOM during prefill — sending 'prompt is too long' to trigger compact", indent=1)
                try:
                    _oom_error = json.dumps({
                        "type": "error",
                        "error": {
                            "type": "invalid_request_error",
                            "message": f"prompt is too long: OOM during prefill ({e})",
                        }
                    })
                    self.wfile.write(f"event: error\ndata: {_oom_error}\n\n".encode("utf-8"))
                    self.wfile.flush()
                except Exception as _werr:
                    logger.warning("[ERROR] Falló envío SSE de oom_error: %s", _werr)
                try:
                    _guard.force_clear_cache("oom_error")
                except Exception as _gerr:
                    logger.warning("[ERROR] Falló force_clear_cache tras OOM: %s", _gerr)
            else:
                _terminal_status("❌", f"Request {request_id} failed: {e}", indent=1)
                import traceback
                _pipeline_log("ERROR", request_id, f"traceback: {traceback.format_exc()}")
            if request_logger:
                request_logger.log("generation", f"error: {e}", request_id=request_id)
        finally:
            # Cancel prefill progress threads (may still be running if generation
            # failed before producing the first token).
            for _evt_name in ("_prefill_done", "_prefill_done_s"):
                _evt = locals().get(_evt_name)
                if _evt is not None:
                    _evt.set()

            if generation_started_at is not None:
                end_at = time.time()
                elapsed = max(end_at - generation_started_at, 1e-9)
                output_tokens = len(generated_tokens)
                speed = output_tokens / elapsed if output_tokens else 0.0

                if enable_thinking:
                    _t_thinking = _thinking_tracker.thinking_count
                    _t_visible = (
                        len(_tokenize_prompt(message_text)) if message_text else 0
                    )
                    _t_markup = max(0, output_tokens - _t_thinking - _t_visible)
                    token_breakdown = (
                        f"{output_tokens} (thinking: {_t_thinking}, "
                        f"visible: {_t_visible}, markup: {_t_markup})"
                    )
                else:
                    token_breakdown = f"{output_tokens}"

                if first_token_at is not None:
                    prefill_seconds = first_token_at - generation_started_at
                    decode_seconds = max(end_at - first_token_at, 1e-9)
                    decode_tps = (
                        output_tokens / decode_seconds if output_tokens else 0.0
                    )
                    prefill_tps = (
                        rest_count / prefill_seconds if prefill_seconds > 0 else 0.0
                    )
                    _mtp_suffix = ""
                    if _mtp_stats and _mtp_stats.get("attempted", 0) > 0:
                        _mtp_suffix = (
                            f" | mtp_alpha={_mtp_stats['alpha']:.1f}% "
                            f"({_mtp_stats['accepted']}/{_mtp_stats['attempted']})"
                        )
                    _final_tag = getattr(_task_detector, "tag", "DEFAULT") if _task_detector else "DEFAULT"
                    _final_temp = getattr(sampler, "resp_temp", getattr(sampler, "temp", 0.30)) if sampler else 0.30
                    _req_log = (
                            f"Request {request_id} finished | output_tokens={token_breakdown} | "
                            f"elapsed={elapsed:.2f}s | tok/s={speed:.2f} | "
                            f"prefill={prefill_seconds:.2f}s ({prefill_tps:.0f} tok/s) | "
                            f"decode={decode_seconds:.2f}s ({decode_tps:.1f} tok/s) | "
                            f"task={_final_tag} | temp={_final_temp:.2f}{_mtp_suffix}"
                    )
                    # Append MoE stats if active
                    _moe_suffix = ""
                    try:
                        from .expert_cache import get_cache_stats
                        _mcs = get_cache_stats(model)
                        if _mcs.get("moe_active"):
                            _moe_suffix = (
                                f" | moe_hit={_mcs['hit_rate']:.0%}"
                                f" fallback={_mcs['fallback_rate']:.0%}"
                            )
                    except (ImportError, Exception) as _moe_err:
                        logger.warning("[STATS] Error obteniendo moe stats: %s", _moe_err)
                    # Metal memory snapshot — monitor fragmentation and pressure
                    _mem_suffix = ""
                    try:
                        _m_active = mx.get_active_memory() / 1e9
                        _m_cache = mx.get_cache_memory() / 1e9
                        _m_peak = mx.get_peak_memory() / 1e9
                        _mem_suffix = f" | mem={_m_active:.1f}GB active/{_m_cache:.1f}GB cache/{_m_peak:.1f}GB peak"
                    except Exception as _mem_err:
                        logger.warning("[STATS] Error leyendo snapshot de memoria Metal: %s", _mem_err)
                    _terminal_status("✅", _req_log + _moe_suffix + _mem_suffix, indent=1)
                else:
                    _final_tag = getattr(_task_detector, "tag", "DEFAULT") if _task_detector else "DEFAULT"
                    _final_temp = getattr(sampler, "resp_temp", getattr(sampler, "temp", 0.30)) if sampler else 0.30
                    _terminal_status(
                        "✅",
                        (
                            f"Request {request_id} finished | output_tokens={token_breakdown} | "
                            f"elapsed={elapsed:.2f}s | tok/s={speed:.2f} | "
                            f"task={_final_tag} | temp={_final_temp:.2f}"
                        ),
                        indent=1,
                    )
            try:
                from .router_tracer import flush_request_trace
                _prompt_preview = str(locals().get("_last_user", locals().get("last_user_msg", locals().get("prompt", ""))))
                flush_request_trace(request_id, prompt_preview=_prompt_preview)
            except Exception as _trc_err:
                logger.warning("[TRACER] Error ejecutando flush_request_trace: %s", _trc_err)
            # On normal exit or Python exception, release the lock. On process abort (e.g. Metal
            # "uncommitted encoder" crash), finally may not run, so the "leaked semaphore" warning
            # at shutdown is expected; fixing the Metal crash resolves it.
            if acquired:
                # (not released). No need to null-out prompt_cache here — the LRU manages it.
                # MAIN cache is always untouched.

                _mem_profiler.snapshot(request_id, "POST_GENERATION", is_anthropic=self._is_anthropic, rest_tokens=rest_count if 'rest_count' in dir() else None, output_tokens=output_tokens if 'output_tokens' in dir() else None, kv_cache_offset=_kv_off if '_kv_off' in dir() else None)
                _guard.post_request_cleanup(request_id)
                if FEATURE_FULL_LOGGING:
                    _pipeline_log("METAL", request_id, f"post_request_cleanup via guard")
                    if generation_started_at is not None:
                        _held_ms = (time.time() - generation_started_at) * 1000
                        _pipeline_log("METAL", request_id, f"model_lock released | held_for={_held_ms/1000:.2f}s")

                # ── MoE Post-Generation Hook ──────────────────────────────
                try:
                    from .expert_cache import moe_post_generation_hook
                    moe_post_generation_hook(
                        model=model,
                        output_tokens=output_tokens,
                        settings=SETTINGS,
                        request_id=request_id,
                        log_fn=_terminal_status,
                    )
                except Exception as _moe_post_err:
                    _terminal_status("⚠️", f"MoE post-generation hook error: {_moe_post_err}")

                model_lock.release()


def run():
    server_address = (SETTINGS.mlx_host, SETTINGS.mlx_port)
    httpd = ThreadingHTTPServer(server_address, APIHandler)

    # ── Module init: adaptive_prefill ─────────────────────────────────────
    _ap_mod.init(
        settings=SETTINGS,
        terminal_status=_terminal_status,
        pipeline_log=_pipeline_log,
        guard=_guard,
        cache_diag=_cache_diag,
        model=model,
        prefill_step_size=PREFILL_STEP_SIZE,
        feature_cache_diag=FEATURE_CACHE_DIAG,
        feature_full_logging=FEATURE_FULL_LOGGING,
        get_metal_budget_gb=_get_metal_budget_gb,
        adaptive_prefill_safety_margin=_ADAPTIVE_PREFILL_SAFETY_MARGIN,
        gpu_yield_seconds=_GPU_YIELD_SECONDS,
        scratch_coefficient=_SCRATCH_COEFFICIENT,
        n_layers_sdpa=_N_LAYERS_SDPA,
        n_heads=_N_HEADS,
        scratch_bytes_per_element=_SCRATCH_BYTES_PER_ELEMENT,
        adaptive_prefill_min_chunk=_ADAPTIVE_PREFILL_MIN_CHUNK,
    )

    # ── Module init: post_generation ──────────────────────────────────────
    _post_gen.init(
        settings=SETTINGS,
        terminal_status=_terminal_status,
        pipeline_log=_pipeline_log,
        tokenize_prompt=_tokenize_prompt,
        thinking_tracker=_thinking_tracker,
        prompt_cache=PROMPT_CACHE,
        session_index=SESSION_INDEX,
        cache_diag=_cache_diag,
        kv_cache_offset=_kv_cache_offset,
        update_session_turn_store=_update_session_turn_store,
        dpc=_DPC,
        wm=_wm,
        warmup_save_cache=_warmup_save_cache if '_warmup_save_cache' in dir() else None,
        housekeeping_staging_manager=HOUSEKEEPING_STAGING_MANAGER if 'HOUSEKEEPING_STAGING_MANAGER' in dir() else None,
        feature_housekeeping_cache_borrow=FEATURE_HOUSEKEEPING_CACHE_BORROW,
        feature_log_generation=FEATURE_LOG_GENERATION,
        feature_log_resp=FEATURE_LOG_RESP,
        is_vlm=is_vlm,
        prepare_cache_for_insertion=prepare_cache_for_insertion,
        restore_hybrid_checkpoint=restore_hybrid_generation_checkpoint,
        rollback_arrays_cache=rollback_arrays_cache,
        can_trim_prompt_cache=can_trim_prompt_cache,
        trim_prompt_cache=trim_prompt_cache,
        HybridGenerationCheckpoint=HybridGenerationCheckpoint,
    )

    # ── SIDECAR: Start lightweight endpoint in daemon thread ───────────
    sidecar_httpd = None
    if SETTINGS.sidecar_port > 0:
        try:
            _sidecar_init(
                settings=SETTINGS,
                terminal_status=_terminal_status,
                pipeline_log=_pipeline_log,
                guard=_guard,
                model_lock=model_lock,
                model=model,
                tokenizer=tokenizer,
                thinking_tracker=_thinking_tracker,
                make_prompt_cache=make_prompt_cache,
                stream_generate=stream_generate,
                stream_generate_mtp=stream_generate_mtp if SETTINGS.enable_mtp else None,
                build_sampler=_build_sampler,
                stream_generate_kwargs_fn=_stream_generate_kwargs,
                tokenize_prompt=_tokenize_prompt,
                prepare_messages_for_template=_prepare_messages_for_template,
                strip_thinking_from_content=_strip_thinking_from_content,
                rag_available=_rag_available,
                rag_module=_rag_module if _rag_available else None,
                feature_preserve_thinking=FEATURE_PRESERVE_THINKING,
                ThinkingEvent=ThinkingEvent,
            )
            sidecar_address = (SETTINGS.mlx_host, SETTINGS.sidecar_port)
            sidecar_httpd = ThreadingHTTPServer(sidecar_address, SidecarHandler)
            sidecar_thread = threading.Thread(
                target=sidecar_httpd.serve_forever,
                name="sidecar",
                daemon=True,  # Dies automatically on main thread exit
            )
            sidecar_thread.start()
        except OSError as e:
            _terminal_status("⚠️", f"Sidecar: failed to bind port {SETTINGS.sidecar_port} ({e})")
            sidecar_httpd = None

    # ── Wait for TPC startup to complete before accepting requests ─────
    # Without this, the first request arrives before TPC loads from disk
    # → cache miss → full prefill. Better to delay SYSTEM READY and
    # guarantee cache hits from the first request.
    if not _WARMUP_DONE.is_set():
        logger.info("[DATA] Waiting for TPC startup to complete...")
        _WARMUP_DONE.wait(timeout=300)  # 5 min max, TPC load typically takes <1s

    # Two-stage expert expansion: handled in post-response hook (_stream method).
    # Expanding at startup causes OOM: cap=170 (13.6 GB) + 33K cold prefill (3.3 GB)
    # = 16.9 GB > 17.18 GB Metal limit. Instead, first request runs at cap=100
    # (safe for cold prefill), then expand_expert_capacity() runs after response.

    _mem_profiler.init(SETTINGS.log_root)
    import mlx.core as _mx_banner
    _mlx_ver = getattr(_mx_banner, '__version__', 'unknown')
    _host = SETTINGS.mlx_host
    _port = SETTINGS.mlx_port
    _base = f"http://{_host}:{_port}"
    print("\n" + "=" * 82)
    print(f"🟢 SYSTEM READY — SuperMLX v{__version__}")
    print(f"   MLX v{_mlx_ver}  •  {'VLM (vision)' if is_vlm else 'LM (text-only)'}")
    print("-" * 82)
    _col = 49  # column width for endpoint text before TPC
    print(f"   {'ENDPOINT':<{_col}}TPC   SESSION   THINKING")
    print("   OpenAI")
    print(f"     {_base + '/v1/chat/completions':<{_col - 2}}✅      ✅        ✅")
    print(f"     {_base + '/v1/ephemeral/chat/completions':<{_col - 2}}✅      ❌        ❌")
    print("   Anthropic")
    print(f"     {_base + '/v1/messages':<{_col - 2}}✅      ✅        ✅")
    print(f"     {_base + '/v1/ephemeral/messages':<{_col - 2}}✅      ❌        ❌")
    if sidecar_httpd:
        rag_tag = " + RAG" if (SETTINGS.sidecar_enable_rag and _rag_available) else ""
        print(f"   Sidecar{rag_tag}")
        _sc_url = f"http://{_host}:{SETTINGS.sidecar_port}"
        print(f"     {_sc_url:<{_col - 2}}❌      ❌        per-req")
    elif SETTINGS.sidecar_port > 0:
        print(f"   Sidecar: FAILED (port {SETTINGS.sidecar_port})")
    else:
        print(f"   Sidecar: DISABLED")
    print("=" * 82 + "\n")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        # Persist expert frequency stats for cross-session learning
        try:
            from .expert_cache import save_frequency_stats
            _stats_path = os.path.join("logs", "expert_stats.json")
            save_frequency_stats(model, _stats_path)
            if os.path.exists(_stats_path):
                print(f"[SHUTDOWN] Expert frequency stats saved to {_stats_path}")
            else:
                print("[SHUTDOWN] Expert frequency stats: nothing to save (session_frequency empty)")
        except Exception as _save_err:
            print(f"[SHUTDOWN] Expert frequency stats FAILED: {_save_err}")
        # Expert Routing: save per-request routing data on shutdown
        if FEATURE_EXPERT_ROUTING_LOG:
            try:
                from .expert_cache import save_routing_stats, end_expert_routing
                end_expert_routing()  # Safety: clear any open label
                _erl_path = os.path.join("logs", "expert_routing.json")
                if save_routing_stats(_erl_path):
                    print(f"[SHUTDOWN] Expert routing log saved to {_erl_path}")
                else:
                    print("[SHUTDOWN] Expert routing log: nothing to save")
            except Exception as _erl_shutdown_err:
                print(f"[SHUTDOWN] Expert routing log FAILED: {_erl_shutdown_err}")
        if sidecar_httpd:
            sidecar_httpd.shutdown()


if __name__ == "__main__":
    run()
