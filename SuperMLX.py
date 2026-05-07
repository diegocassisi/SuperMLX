# SPDX-License-Identifier: MIT
"""
SuperMLX — Production-Grade MLX Inference Server for Agentic AI

OpenAI-compatible inference server on Apple Silicon (MLX). Serves agentic AI
requests with persistent KV cache, multi-agent isolation, and OOM protection,
maximizing cache hits to minimize TTFT.

This file is the single source of inference logic.
warmup_manager.py handles DPC. rag_enricher.py handles RAG/Compressor.
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

  Production (DPC + cache persistence):
    FORCE_TEXT_MODE=true \\
      CACHE_PERSIST_PATH=logs/warmup_cache.safetensors \\
      EMBEDDED_CACHE_PERSIST_PATH=logs/embedded_cache.safetensors \\
      python SuperMLX.py

  Endpoints:
    LiteLLM Proxy:  http://0.0.0.0:4000/v1/chat/completions  (point your framework here)
    MLX Direct:     http://0.0.0.0:8080/v1/chat/completions
    Sidecar:        http://0.0.0.0:8081/v1/chat/completions  (scripts, sensors)

─── ACTIVE FEATURES ──────────────────────────────────────────────────────────

  ✅ On by default:
    • Dual-Slot KV Cache        Isolated MAIN + COMPACT LRU stores per agent type
    • Dynamic Prefix Capture    Auto-capture, hash validation, disk persistence (warmup_manager.py)
    • Post-Reaper Cache Reload  Automatic disk reload after idle eviction (v1.4.0)
    • Cache Canonicalization     Volatile fields masked → 97%+ cache hit rate
    • Compact Runner Detector   Multi-signal routing: tools + keywords + anti-MAIN guard
    • Memory Guard              Pre-prefill Metal RAM check with auto-eviction
    • Tool Loop Breaker         Breaks infinite tool-call retry cycles (3 detection modes)
    • Emergency Compression     LLMLingua-2 last-resort OOM defense
    • Session-Aware Routing     Per-session prefix tracking with block-hash index
    • Min-Suffix Pollution Wash Re-prefill ≥256 tokens on cache hit with response pollution
    • Per-Layer Trim            Selective KVCache trim for hybrid architecture disk saves
    • Frozen Cache Snapshot     Prompt-only snapshot for post-generation cache recovery
    • LiteLLM Reverse Proxy     OpenAI-compatible routing layer

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
    PROXY_PORT                      (4000)      LiteLLM proxy port
    SIDECAR_PORT                    (8081)      Sidecar port (0 = disabled)

  KV Cache:
    MAX_KV_SIZE                     (196608)    Max tokens per session
    PROMPT_CACHE_MAX_ENTRIES_GLOBAL (2)         Max LRU entries (2 = safe for 24GB)
    PROMPT_CACHE_TTL_SECONDS        (1800)      Entry TTL before reaper prunes
    CACHE_REAPER_INTERVAL_SECONDS   (60)        Reaper check frequency
    KV_BITS                         (None)      Native quantization (4 / 8 / None)

  Persistence (DPC):
    CACHE_PERSIST_PATH              ("")        Disk path for MAIN cache
    EMBEDDED_CACHE_PERSIST_PATH     ("")        Disk path for COMPACT cache

  Memory:
    MEMORY_GUARD_THRESHOLD_GB       (auto)      total_ram - 8GB (0 = disabled)

  See .env.example for the full reference with all supported variables.

─── MEMORY BUDGET (24GB M4 Pro) ──────────────────────────────────────────────

    Model Qwen3.5-9B-4bit:    ~5.0 GB
    2 MAIN KV entries:        ~9.0 GB  (2 × 4.5GB)
    1 COMPACT KV entry:       ~4.5 GB
    Scratch prefill:          ~5.0 GB
    ──────────────────────────────────
    Peak total:               ~23.5 GB → safe with Memory Guard at 19.2GB
    Concurrent agents:        1–2 with warm cache

─── ARCHITECTURE ─────────────────────────────────────────────────────────────

  OpenClaw / Claude Code ──→ LiteLLM Proxy :4000 ──→ MLX Engine :8080
                                                        │
                                _detect_compact_runner()
                                 │              │
                           MAIN path       COMPACT path
                                 │              │
                          PROMPT_CACHE    PROMPT_CACHE_COMPACT
                           (LRU max=2)      (LRU max=1)
                                 │              │
                          ┌──────┴──────┐       │
                          │Memory Guard │       │
                          │(Metal RAM)  │       │
                          └──────┬──────┘       │
                                 ↓              ↓
                              stream_generate(model, prompt_cache=...)
                                                 ▲
  External tools ──→ Sidecar :8081 ──────────────┘
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
from datetime import datetime
try:
    from emergency_compressor import emergency_compress_if_needed, should_signal_overflow
    _emergency_compressor_available = True
except ImportError:
    _emergency_compressor_available = False
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
from mlx_lm.models.cache import (
    make_prompt_cache,
    can_trim_prompt_cache,
    trim_prompt_cache,
)

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

SCRIPT_DIR = Path(__file__).resolve().parent
DOTENV_PATH = SCRIPT_DIR / ".env"

# Thread-local VLM diagnostics (used_prefix_stable, etc.) for cache-debug logging.
_vlm_diagnostics = threading.local()
if DOTENV_PATH.exists():
    load_dotenv(dotenv_path=DOTENV_PATH, override=True)

__version__ = "1.4.0"


# ── Configuration (extracted to config.py) ────────────────────────────────────
from config import (
    Settings, build_settings,
    _env_str, _env_int, _env_float, _env_bool,
    _env_str_any, _env_int_any, _env_bool_any, _env_kv_bits,
    _normalize_model_family, _infer_model_family,
)

SETTINGS = build_settings(script_dir=SCRIPT_DIR)

# ══════════════════════════════════════════════════════════════════════════════
# FEATURE FLAGS — Hardcoded toggles para activar/desactivar componentes.
# Cambiar True/False y reiniciar el servidor. Sin env vars por ahora — rapidez.
# ══════════════════════════════════════════════════════════════════════════════

# Prompt compression: historial largo → LanceDB → solo contexto relevante.
# Requiere rag_enricher.py + lancedb + sentence-transformers.
# Env: FEATURE_COMPRESSOR=true | COMPRESSION_THRESHOLD=6000 | COMPRESSION_GUARD=6
FEATURE_COMPRESSOR            = _env_str("FEATURE_COMPRESSOR", "false").lower() in ("1", "true", "yes")

# Emergency content compression (LLMLingua-2): independent of FEATURE_COMPRESSOR.
# Activates on rest_tokens > MAX_SAFE_PREFILL_TOKENS as last-resort OOM defense.
# Only needs LLMLingua (CPU BERT), not the full reranker+compression pipeline.
FEATURE_EMERGENCY_COMPRESS    = _env_str("EMERGENCY_CONTENT_COMPRESS", "true").lower() in ("1", "true", "yes")

# RAG codebase enrichment: inyecta chunks relevantes del codebase en el context.
# Requiere rag_enricher.py + lancedb + sentence-transformers.
# Env: FEATURE_RAG_ENRICHMENT=true | RAG_WORKSPACE_ROOT=/path/to/workspace
FEATURE_RAG_ENRICHMENT        = _env_str("FEATURE_RAG_ENRICHMENT", "false").lower() in ("1", "true", "yes")
FEATURE_RAG_WORKSPACE_ROOT    = _env_str("RAG_WORKSPACE_ROOT", "")

# Headers HTTP de diagnóstico (X-Pipeline-Compression-Ms, X-Pipeline-RAG-Ms, etc.)
FEATURE_DIAGNOSTIC_HEADERS = True

# Logging exhaustivo de todas las etapas del pipeline.
# Master switch: si False, todos los FEATURE_LOG_* se desactivan.
FEATURE_FULL_LOGGING       = True

# Sub-flags de logging (sólo activos cuando FEATURE_FULL_LOGGING = True)
FEATURE_LOG_PROMPTS        = _env_bool("LOG_PROMPTS", True)    # Dump de prompts/messages a disco (logs/requests/)
FEATURE_LOG_TOOLS          = _env_bool("LOG_TOOLS", True)      # Detalle de tool schemas + tool_results en historial
FEATURE_LOG_COMPRESSION    = True   # Antes/después de compresión + chunks + timing
FEATURE_LOG_RAG            = True   # Chunks RAG inyectados, scores de relevancia
FEATURE_LOG_CACHE          = True   # Hit/miss/shorter, stable prefix, evictions
FEATURE_LOG_HEALING        = True   # Healing store operations (hits, store size)
FEATURE_LOG_GENERATION     = True   # tps, timing, token breakdown por etapa
FEATURE_LOG_RESP           = True   # Response normalization, tool extraction, think stripping

# Compressor: config (env-var driven above, these are runtime defaults for rag_enricher)
FEATURE_COMPRESSION_THRESHOLD = _env_int("COMPRESSION_THRESHOLD", 6000)
FEATURE_COMPRESSION_GUARD     = _env_int("COMPRESSION_GUARD", 6)

# ── COMPRESSION CACHE: per-session reuse of compressed output ────────────────
# Problem: the compressor re-processes ALL history every request, producing
# slightly different output each time → canonical tokens diverge → KV cache
# drops from 97% to 78% on every new user message (destroying 5K+ prefill tokens).
# Solution: cache the compressed output per session. On subsequent requests,
# reuse the cached compressed messages and only append truly new messages.
# This makes canonical tokens stable across requests → 97%+ cache hit always.
# Keyed by _comp_session (X-Session-ID header or body session_id).
_COMPRESS_CACHE: Dict[str, Dict] = {}  # session_id → {"msgs": [...], "input_count": N, "ts": float}
_COMPRESS_CACHE_LOCK = threading.Lock()
_COMPRESS_CACHE_MAX_SESSIONS = 8  # Max sessions to cache (LRU eviction)
_COMPRESS_CACHE_TTL = 3600  # Seconds before a cached entry expires


def _compress_cache_get(session_id: str, current_msg_count: int) -> Optional[List[Dict]]:
    """Return cached compressed messages if valid for this session + message count.
    
    Cache is valid when:
    - Entry exists for this session_id
    - The current message count >= cached input_count (conversation grew, not shrunk)
    - Entry hasn't expired (TTL)
    
    Returns None if cache miss (caller should do full compression).
    Returns the cached compressed messages if hit.
    """
    with _COMPRESS_CACHE_LOCK:
        entry = _COMPRESS_CACHE.get(session_id)
        if entry is None:
            return None
        if time.time() - entry["ts"] > _COMPRESS_CACHE_TTL:
            del _COMPRESS_CACHE[session_id]
            return None
        # If conversation shrunk (e.g., /new or /reset), invalidate
        if current_msg_count < entry["input_count"]:
            del _COMPRESS_CACHE[session_id]
            return None
        return entry["msgs"]


def _compress_cache_put(session_id: str, compressed_msgs: List[Dict], input_count: int) -> None:
    """Store compressed output for a session. LRU eviction if over capacity."""
    with _COMPRESS_CACHE_LOCK:
        _COMPRESS_CACHE[session_id] = {
            "msgs": compressed_msgs,
            "input_count": input_count,
            "ts": time.time(),
        }
        # LRU eviction: drop oldest if over capacity
        if len(_COMPRESS_CACHE) > _COMPRESS_CACHE_MAX_SESSIONS:
            oldest_key = min(_COMPRESS_CACHE, key=lambda k: _COMPRESS_CACHE[k]["ts"])
            del _COMPRESS_CACHE[oldest_key]


def _compress_with_cache(
    raw_messages: List[Dict],
    comp_session: str,
    compressor_module,
    threshold: int,
    request_id: str,
) -> Tuple[List[Dict], float, bool]:
    """Compress messages with session-level caching.
    
    Returns: (compressed_messages, elapsed_ms, cache_hit)
    
    Strategy:
    1. Check if we have cached compressed output for this session
    2. If YES and conversation only grew (append-only):
       - Take the cached compressed messages (system + compressed history)
       - Identify truly new messages (delta between cached input_count and current)
       - Append the new messages to the cached output
       → Canonical tokens stay identical up to the new messages → high KV cache hit
    3. If NO: full compression, then store result for future requests
    """
    t0 = time.time()
    input_count = len(raw_messages)
    est_tokens = sum(len(json.dumps(m)) // 4 for m in raw_messages)
    
    # Below threshold → no compression needed, but still cache the pass-through
    if est_tokens <= threshold:
        return raw_messages, 0.0, False
    
    # Check cache
    cached = _compress_cache_get(comp_session, input_count)
    if cached is not None:
        # Cache hit: reuse compressed base + append new messages
        cached_input_count = _COMPRESS_CACHE.get(comp_session, {}).get("input_count", 0)
        if cached_input_count > 0 and input_count > cached_input_count:
            # New messages arrived since last compression
            new_messages = raw_messages[cached_input_count:]
            result = cached + new_messages
        elif input_count == cached_input_count:
            # Same message count → exact reuse (e.g., retry)
            result = list(cached)
        else:
            # Shouldn't happen (guard in _compress_cache_get), but fallback
            result = list(cached)
        
        elapsed_ms = (time.time() - t0) * 1000
        # Update cache with new input_count (conversation grew)
        _compress_cache_put(comp_session, result, input_count)
        return result, elapsed_ms, True
    
    # Cache miss: full compression
    compressed = compressor_module.compress_messages(
        raw_messages,
        session_key=comp_session,
    )
    elapsed_ms = (time.time() - t0) * 1000
    
    # Store in cache for future requests
    _compress_cache_put(comp_session, compressed, input_count)
    
    return compressed, elapsed_ms, False


# RAG: config
FEATURE_RAG_TOP_K             = 5      # chunks a recuperar de LanceDB
FEATURE_RAG_RELEVANCE_THRESHOLD = 1.6  # Qwen3-Embed asimétrico (docs sin prefijo). Probado 0.8: filtra demasiado

# Tool Call Loop Breaker: detect and break infinite tool-call retry loops.
# When the model retries the same failed tool N consecutive times, inject a
# stop instruction into the last tool_result so the model gives up and responds
# with text instead.  Env: TOOL_LOOP_BREAKER=true (default), TOOL_LOOP_MAX_RETRIES=3.
FEATURE_TOOL_LOOP_BREAKER     = _env_bool("TOOL_LOOP_BREAKER", True)
TOOL_LOOP_MAX_RETRIES         = _env_int("TOOL_LOOP_MAX_RETRIES", 3)

# Dynamic Prefix Capture (DPC): replaces warmup_seed.txt with auto-capture + hash validation.
# Managed by warmup_manager.py. No manual seed files needed.
import warmup_manager as _wm

# ── CASCADE ROUTING ──────────────────────────────────────────────────────────
# Forward a frontier API cuando RAG confidence es baja (no hay knowledge local).
# Env: FEATURE_CASCADE=true | CASCADE_API_KEY=<key> | CASCADE_API_URL=<url>
FEATURE_CASCADE                = _env_bool("FEATURE_CASCADE", False)
CASCADE_API_URL                = _env_str("CASCADE_API_URL", "")
CASCADE_API_KEY                = _env_str("CASCADE_API_KEY", "")
CASCADE_MODEL                  = _env_str("CASCADE_MODEL", "")
CASCADE_RAG_THRESHOLD          = _env_float("CASCADE_RAG_THRESHOLD", 2.0)
CASCADE_TIMEOUT_S              = _env_int("CASCADE_TIMEOUT_S", 60)

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
            except Exception:
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
        except Exception:
            pass
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
        # Over threshold — evict from both stores
        evicted = 0
        with prompt_cache_lock:
            evicted += PROMPT_CACHE.evict_unpinned()
            evicted += PROMPT_CACHE_COMPACT.evict_unpinned()
        mx.clear_cache()
        import gc; gc.collect()
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


proxy_process = None
proxy_config_path = None

TOOL_CALL_PATTERN = re.compile(
    r"<tool_call>(.*?)</tool_call>", re.DOTALL | re.IGNORECASE
)
# Gemma 4 uses pipe-delimited tokens: <|tool_call|>...<|/tool_call|> or <|tool_call|>...<|tool_response|>
GEMMA4_TOOL_CALL_PATTERN = re.compile(
    r"<\|tool_call\|>(.*?)(?:<\|/tool_call\|>|<\|tool_response\|>)", re.DOTALL | re.IGNORECASE
)
ARG_PAIR_PATTERN = re.compile(
    r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>",
    re.DOTALL | re.IGNORECASE,
)
QWEN_FUNCTION_PATTERN = re.compile(
    r"<function=([^>\s]+)>\s*(.*?)\s*</function>", re.DOTALL | re.IGNORECASE
)
QWEN_PARAMETER_PATTERN = re.compile(
    r"<parameter=([^>\s]+)>\s*(.*?)\s*</parameter>", re.DOTALL | re.IGNORECASE
)
INBOUND_META_MESSAGE_ID_PATTERN = re.compile(r'("message_id"\s*:\s*")[^"]+(")')
# NOTE: INBOUND_TRUSTED_CONTEXT_BLOCK_PATTERN, CACHE_STABLE_INBOUND_CONTEXT_BLOCK, and
# INBOUND_CONTEXT_TO_PROJECT_BOUNDARY_PATTERN were removed (Phase 7 Fix 2, 2026-03-14).
# The Inbound Context block is now handled structurally by _canonicalize_inbound_context_block()
# using plain string anchors with no DOTALL regex.  Do NOT re-add a DOTALL regex here —
# see Phase 5 in PLAN.md for why that approach caused a 17k-char lobotomy bug.
# OpenClaw sub-agent completion event: "Stats: runtime 1m52s • tokens 0 (in 0 / out 0)"
# The runtime duration is volatile (changes with each execution).  The UUIDs in
# session_key and session_id are frozen per injection (stable across turns), so only
# the runtime field needs to be normalised.  Single-line, structurally bounded — safe.
SUBAGENT_STATS_PATTERN = re.compile(r"(Stats: runtime\s+)[^\n•]+", re.IGNORECASE)
# Cache normalization: scrub volatile request-specific text so retries hit cache.
CACHE_TIME_PATTERN = re.compile(r"Current time is[^\n]+\.", re.IGNORECASE)
# OpenClaw user message injects "Current time: Tuesday, April..." (colon variant, not covered above)
CACHE_TIME_COLON_PATTERN = re.compile(
    r"Current time:\s*[^\n]+", re.IGNORECASE
)
CACHE_CCH_PATTERN = re.compile(r"cch=[a-zA-Z0-9]+;?", re.IGNORECASE)
CACHE_BILLING_HEADER_PATTERN = re.compile(
    r"-anthropic-billing-header:\s*[a-zA-Z0-9\-]+", re.IGNORECASE
)
# Claude Code injects variable <system-reminder> text (e.g. "gentle reminder" vs "improve or augment");
# normalize so stream=true vs stream=false retries hit cache.
CACHE_SYSTEM_REMINDER_PATTERN = re.compile(
    r"<system-reminder>.*?</system-reminder>", re.DOTALL | re.IGNORECASE
)
# OpenClaw injects <available_skills>...</available_skills> dynamically based on installed skills.
# ANY skill install/update changes this block and breaks cache at token ~691.
# Replace the entire block with a stable sentinel so the cache key is invariant to skill changes.
CACHE_SKILLS_BLOCK_PATTERN = re.compile(
    r"<available_skills>.*?</available_skills>", re.DOTALL | re.IGNORECASE
)
# OpenClaw injects a ## Runtime line with volatile fields (model, channel, thinking level, node version).
# This line changes per request and breaks the stable prefix after the skills block.
CACHE_RUNTIME_LINE_PATTERN = re.compile(
    r"^Runtime:.*$", re.MULTILINE | re.IGNORECASE
)
# Strip reasoning from response content when returning to client (so reasoning is hidden).
# Full think blocks (<think>...</think>) and "orphan" </think> (reasoning with no opening tag, e.g. GLM-style).
THINK_TAG_STRIP_PATTERN = re.compile(
    r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE
)
# Gemma 4 uses pipe-delimited think tokens: <|think|>...</|think|> or <|/think|>
GEMMA4_THINK_STRIP_PATTERN = re.compile(
    r"<\|think\|>.*?<\|/think\|>\s*", re.DOTALL | re.IGNORECASE
)
GEMMA4_THINK_ORPHAN_PATTERN = re.compile(
    r"^.*?<\|/think\|>\s*", re.DOTALL | re.IGNORECASE
)
# Orphan </think>: content from start up to and including </think> so we hide reasoning when model
# outputs "reasoning text</think>\n\nanswer" without a leading <think> tag.
THINK_ORPHAN_CLOSE_PATTERN = re.compile(r"^.*?</think>\s*", re.DOTALL | re.IGNORECASE)


def _strip_thinking_from_content(text: str) -> str:
    """Remove <think>...</think> and <|think|>...<|/think|> blocks so reasoning is hidden."""
    if not isinstance(text, str):
        return text
    # First remove full think blocks (Qwen/GLM/Hermes/DeepSeek style).
    out = THINK_TAG_STRIP_PATTERN.sub("", text)
    # Then remove Gemma 4 pipe-style think blocks.
    out = GEMMA4_THINK_STRIP_PATTERN.sub("", out)
    # Remove orphan close tags (both styles).
    out = THINK_ORPHAN_CLOSE_PATTERN.sub("", out, count=1)
    out = GEMMA4_THINK_ORPHAN_PATTERN.sub("", out, count=1)
    return out.strip()


# --- STATELESS MESSAGE HEALING STORE ---
HEALING_STORE: OrderedDict = OrderedDict()
HEALING_STORE_LOCK = threading.Lock()
MAX_HEALING_STORE = 2000  # Generous size to survive deep multi-agent sessions


def _get_healing_hash(
    text: str, tool_calls: Optional[List[Dict[str, Any]]] = None
) -> Optional[str]:
    """
    Creates a robust SHA-256 hash of the assistant's output.
    If the text is empty but has tool calls, we hash the canonicalized tool calls.
    """
    base = (text or "").strip()
    if tool_calls:
        try:
            # Sort keys to ensure deterministic hashing of tool calls
            base += json.dumps(tool_calls, sort_keys=True)
        except Exception:
            pass

    if not base:
        return None

    return hashlib.sha256(base.encode("utf-8")).hexdigest()


def _heal_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Intercepts incoming messages. If a stripped assistant message matches
    a hash in our store, we swap it back to the full version (with <think>).
    """
    healed = []
    for msg in messages:
        m = dict(msg)
        if (m.get("role") or "").strip().lower() == "assistant":
            content = m.get("content", "")
            tool_calls = m.get("tool_calls")

            # Handle standard text content
            if isinstance(content, str):
                h = _get_healing_hash(content, tool_calls)
                if h:
                    with HEALING_STORE_LOCK:
                        if h in HEALING_STORE:
                            m["content"] = HEALING_STORE[h]
                            # CRITICAL: Prevent the Jinja template from double-rendering
                            # the tool calls, since the raw text already contains them.
                            m.pop("tool_calls", None)
                            HEALING_STORE.move_to_end(h, last=True)

            # Handle VLM list content (e.g., [{"type": "text", "text": "..."}])
            elif isinstance(content, list):
                new_content = []
                healed_any = False
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_part = part.get("text") or part.get("content") or ""
                        h = _get_healing_hash(text_part, tool_calls)
                        if h:
                            with HEALING_STORE_LOCK:
                                if h in HEALING_STORE:
                                    new_content.append(
                                        {**part, "text": HEALING_STORE[h]}
                                    )
                                    healed_any = True
                                    HEALING_STORE.move_to_end(h, last=True)
                                    continue
                    new_content.append(part)
                m["content"] = new_content
                if healed_any:
                    # CRITICAL: Prevent double-rendering for VLM multi-part messages too
                    m.pop("tool_calls", None)

        healed.append(m)
    return healed


# ══════════════════════════════════════════════════════════════════════════════
# TOOL CALL LOOP BREAKER
# ══════════════════════════════════════════════════════════════════════════════

def _extract_tool_call_signature(asst_msg: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    """Extract (tool_name, args_json) from an assistant message with tool_calls."""
    tc_list = asst_msg.get("tool_calls") or []
    if not tc_list or not isinstance(tc_list[0], dict):
        return None
    func = tc_list[0].get("function", {})
    name = func.get("name", "")
    args = func.get("arguments", "")
    if isinstance(args, dict):
        args = json.dumps(args, sort_keys=True)
    return (name, args) if name else None


def _break_tool_call_loop(
    messages: List[Dict[str, Any]], request_id: str = ""
) -> Tuple[List[Dict[str, Any]], bool]:
    """
    Detect and break infinite tool-call retry loops.

    Three detection modes:
    1. ERROR LOOP: Same tool fails N consecutive times (original behavior).
    2. DUPLICATE LOOP: Same tool + same args called 2+ times (even if "successful").
    3. SPAM LOOP: Same tool called 5+ times consecutively regardless of args/results.

    Returns (messages, loop_was_broken).
    """
    if not FEATURE_TOOL_LOOP_BREAKER or len(messages) < 4:
        return messages, False

    # ── MODE 2 & 3: Scan for duplicate calls and same-tool spam ──────────
    # Walk backward collecting consecutive (assistant→tool) pairs regardless of error status.
    SPAM_THRESHOLD = 5  # same tool N times in a row = spam
    DUPLICATE_THRESHOLD = 2  # exact same call (name+args) N times = duplicate

    consecutive_calls: List[Dict[str, Any]] = []  # [{tool_name, tool_args, tool_idx}]
    i = len(messages) - 1

    while i >= 1:
        tool_msg = messages[i]
        asst_msg = messages[i - 1]

        if (tool_msg.get("role") or "").lower() != "tool":
            break
        if (asst_msg.get("role") or "").lower() != "assistant":
            break

        sig = _extract_tool_call_signature(asst_msg)
        if not sig:
            break

        consecutive_calls.append({
            "tool_name": sig[0], "tool_args": sig[1], "tool_idx": i
        })
        i -= 2

    if consecutive_calls:
        # Check DUPLICATE: exact same (name, args) repeated
        seen_sigs: Dict[str, int] = {}
        for c in consecutive_calls:
            key = f"{c['tool_name']}::{c['tool_args']}"
            seen_sigs[key] = seen_sigs.get(key, 0) + 1

        for sig_key, count in seen_sigs.items():
            if count >= DUPLICATE_THRESHOLD:
                dup_name = sig_key.split("::")[0]
                return _inject_loop_stop(
                    messages, consecutive_calls[0]["tool_idx"],
                    dup_name, count, "DUPLICATE", request_id,
                    f"You already called '{dup_name}' with the EXACT same arguments "
                    f"{count} times and got the same results each time. "
                    f"The information you need is NOT available via this tool. "
                    f"DO NOT call '{dup_name}' again. "
                    f"Respond with a text message using your own knowledge instead."
                )

        # Check SPAM: same tool name called too many times (different args OK)
        if len(consecutive_calls) >= SPAM_THRESHOLD:
            first_name = consecutive_calls[0]["tool_name"]
            same_name_count = sum(1 for c in consecutive_calls if c["tool_name"] == first_name)
            if same_name_count >= SPAM_THRESHOLD:
                return _inject_loop_stop(
                    messages, consecutive_calls[0]["tool_idx"],
                    first_name, same_name_count, "SPAM", request_id,
                    f"You have called '{first_name}' {same_name_count} consecutive times "
                    f"without making progress. STOP searching. "
                    f"DO NOT call '{first_name}' again. "
                    f"Respond with a text message using the information you already have, "
                    f"or explain that you couldn't find what you needed."
                )

    # ── MODE 1: Original error-based loop detection ──────────────────────
    failures: List[Dict[str, Any]] = []
    i = len(messages) - 1

    while i >= 1:
        tool_msg = messages[i]
        asst_msg = messages[i - 1]

        if (tool_msg.get("role") or "").lower() != "tool":
            break
        if (asst_msg.get("role") or "").lower() != "assistant":
            break

        content = str(tool_msg.get("content", ""))
        is_error = (
            '"status": "error"' in content
            or '"status":"error"' in content
            or '"error":' in content[:200]
        )
        if not is_error:
            break

        tc_list = asst_msg.get("tool_calls") or []
        if not tc_list:
            break

        tool_name = (
            tc_list[0].get("function", {}).get("name", "unknown")
            if isinstance(tc_list[0], dict)
            else "unknown"
        )
        failures.append({"tool_name": tool_name, "tool_idx": i})
        i -= 2

    if len(failures) < TOOL_LOOP_MAX_RETRIES:
        return messages, False

    name_counts: Dict[str, int] = {}
    for f in failures:
        name_counts[f["tool_name"]] = name_counts.get(f["tool_name"], 0) + 1
    most_common_name = max(name_counts, key=name_counts.get)  # type: ignore[arg-type]
    most_common_count = name_counts[most_common_name]

    if most_common_count < TOOL_LOOP_MAX_RETRIES:
        return messages, False

    return _inject_loop_stop(
        messages, failures[0]["tool_idx"],
        most_common_name, most_common_count, "ERROR", request_id,
        f"The tool '{most_common_name}' has failed "
        f"{most_common_count} consecutive times with errors. "
        f"DO NOT retry this tool. DO NOT call '{most_common_name}' again. "
        f"Instead, respond with a text message to the user explaining "
        f"that '{most_common_name}' is currently unavailable and what the error was."
    )


def _inject_loop_stop(
    messages: List[Dict[str, Any]], tool_idx: int,
    tool_name: str, count: int, mode: str, request_id: str,
    instruction: str,
) -> Tuple[List[Dict[str, Any]], bool]:
    """Inject a stop instruction into the last tool result to break the loop."""
    messages = list(messages)
    messages[tool_idx] = dict(messages[tool_idx])
    original_content = str(messages[tool_idx].get("content", ""))
    messages[tool_idx]["content"] = original_content + f"\n\n⚠️ SYSTEM LOOP BREAKER: {instruction}"

    if FEATURE_FULL_LOGGING:
        _pipeline_log(
            "LOOP_BREAK", request_id,
            f"ACTIVATED ({mode}) | tool='{tool_name}' | "
            f"consecutive_calls={count} | "
            f"injected stop instruction at msg[{tool_idx}]",
        )

    return messages, True


# ---------------------------------------


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE LOGGING — Full observability for every pipeline stage.
# Every component logs: when activated, what received, what did, what returned,
# how long it took.  "Si no se loguea, no existe."
# ══════════════════════════════════════════════════════════════════════════════

_PIPELINE_LOG_DIR = SETTINGS.log_root / "requests"


# ANSI color codes for terminal log highlighting
_ANSI_YELLOW = "\033[33m"
_ANSI_RED = "\033[31m"
_ANSI_RESET = "\033[0m"
# Stages that get yellow highlighting (compression/compaction events)
_HIGHLIGHT_STAGES = {"COMPRESS", "COMPACT_RUNNER"}

def _pipeline_log(
    stage: str,
    request_id: str,
    message: str,
    data: dict = None,
    indent: int = 1,
) -> None:
    """Unified pipeline logger.  Each stage has its own tag.

    Only fires when FEATURE_FULL_LOGGING = True.
    Per-stage sub-flags (FEATURE_LOG_TOOLS, etc.) are checked by the caller,
    not here — keeps this function fast and simple.
    """
    if not FEATURE_FULL_LOGGING:
        return

    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]  # HH:MM:SS.mmm
    tag = f"[{stage}]"
    pad = "  " * max(indent, 0)
    line = f"{pad}{tag} {ts} req={request_id[:8]} | {message}"

    # Highlight compression/compaction stages in yellow
    if stage in _HIGHLIGHT_STAGES:
        line = f"{_ANSI_YELLOW}{line}{_ANSI_RESET}"

    with console_lock:
        print(line, flush=True)

    # Optionally dump structured data to disk
    if data is not None and FEATURE_LOG_PROMPTS:
        _write_request_log(request_id, stage, data)


def _write_request_log(request_id: str, stage: str, data: Any) -> None:
    """Write pipeline stage data to logs/requests/{request_id}/{stage}.json.

    Best-effort: never raises — a log failure must not crash the pipeline.
    """
    try:
        req_dir = _PIPELINE_LOG_DIR / request_id
        req_dir.mkdir(parents=True, exist_ok=True)
        out_path = req_dir / f"{stage.lower().replace(' ', '_')}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    except Exception:
        pass  # Best-effort — never crash the pipeline for a log


def _count_roles(messages: List[Dict[str, Any]]) -> Dict[str, int]:
    """Count messages by role for inbound logging."""
    counts: Dict[str, int] = {}
    for m in messages:
        role = (m.get("role") or "unknown").strip().lower()
        counts[role] = counts.get(role, 0) + 1
    return counts


def _summarize_tool_results(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract tool result messages for logging (role=tool only)."""
    results = []
    for m in messages:
        if (m.get("role") or "").strip().lower() == "tool":
            content = m.get("content", "")
            content_len = len(content) if isinstance(content, str) else len(str(content))
            results.append({
                "tool_call_id": m.get("tool_call_id", "?"),
                "name": m.get("name", "?"),
                "content_len": content_len,
            })
    return results


def _estimate_token_count(messages: List[Dict[str, Any]]) -> int:
    """Rough token estimate: ~4 chars per token. Used for logging only, never for logic."""
    total_chars = 0
    for m in messages:
        content = m.get("content", "")
        if isinstance(content, str):
            total_chars += len(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    total_chars += len(part.get("text", "") or part.get("content", "") or "")
    return total_chars // 4


# ══════════════════════════════════════════════════════════════════════════════
# KRIPPER DUAL-SLOT — Compact runner detector
# ══════════════════════════════════════════════════════════════════════════════

# Keywords injected by OpenClaw's buildEmbeddedSystemPrompt() during compaction.
# Source: dist/pi-embedded-BYdcxQ5A.js — compactEmbeddedPiSessionDirect()
_COMPACT_RUNNER_SIGNALS: List[str] = [
    "compact",
    "compaction",
    "summarize",
    "summary of",
    "context window",
    "conversation history",
    "prior conversation",
    "previous conversation",
    "Produce a compact, factual summary",  # Real OpenClaw source signal (pi-embedded.js)
]

def _is_slug_gen_request(messages: List[Dict[str, Any]]) -> bool:
    """Detects the specific OpenClaw slug-generation request."""
    if not messages: return False
    # Check last user message for slug generation instructions
    last_msg = str(messages[-1].get("content") or "")
    return "filename slug" in last_msg.lower() and "short 1-2 word" in last_msg.lower()


# ── RAG BYPASS SIGNALS ────────────────────────────────────────────────────
# Requests matching these patterns are lightweight/internal OpenClaw operations
# that should NOT be enriched with RAG context. RAG injection on these inflates
# a ~500 token request to 50K+, causing 170s prefill on a 9B model.
_RAG_BYPASS_SIGNALS_USER = [
    "write any lasting notes to memory",
    "store durable memories now",
    "session nearing compaction",
    "reply with no_reply if nothing to store",
    "<text_to_summarize>",
]
_RAG_BYPASS_SIGNALS_SYSTEM = [
    "session nearing compaction",
    "store durable memories now",
]

def _is_rag_bypass_request(messages: List[Dict[str, Any]]) -> bool:
    """
    Detects requests that should bypass RAG enrichment:
    - OpenClaw memory flush turns (pre-compaction save)
    - Pure summarization requests (no domain knowledge needed)
    - Internal maintenance operations

    These are lightweight requests where injecting RAG context is
    counterproductive: it inflates tokens and wastes prefill time.
    """
    if not messages:
        return False

    # Check last user message for flush/summarization signals
    for msg in reversed(messages):
        role = (msg.get("role") or "").lower()
        content = msg.get("content", "")
        if not isinstance(content, str):
            continue
        content_lower = content.lower()

        if role == "user":
            if any(sig in content_lower for sig in _RAG_BYPASS_SIGNALS_USER):
                return True
            break  # only check the last user message

    # Check system prompt for flush signals
    for msg in messages:
        if (msg.get("role") or "").lower() == "system":
            content = msg.get("content", "")
            if isinstance(content, str):
                content_lower = content.lower()
                if any(sig in content_lower for sig in _RAG_BYPASS_SIGNALS_SYSTEM):
                    return True

    return False


def _detect_compact_runner(
    messages: List[Dict[str, Any]],
    tools: Any,
) -> bool:
    """
    Detects if the incoming request is from OpenClaw's compact runner
    (compactEmbeddedPiSessionDirect), not from the MAIN agent.

    The compact runner:
      - Sends 0 tools (never uses function calling)
      - Injects a system prompt with compaction-specific language
      - Sends the full session history as messages (can be long)

    The old heuristic (len(tools) <= 2) was wrong: compact sends 0 tools,
    and MAIN in conversational mode may also send 0 tools.

    Decision: Guard 1 (no tools) is necessary but not sufficient.
    Guard 2 (system prompt keywords) is the decisive signal.
    Guard 3 (long history, no tools) catches edge cases without system prompt.
    """
    # Guard 1: compact never uses tools. If tools present → definitely MAIN.
    has_tools = tools is not None and len(tools) > 0
    if has_tools:
        return False

    # Guard 2: system prompt contains compaction keywords (primary signal).
    for msg in messages:
        if (msg.get("role") or "").lower() == "system":
            content = msg.get("content", "")
            if isinstance(content, str):
                content_lower = content.lower()
                if any(sig in content_lower for sig in _COMPACT_RUNNER_SIGNALS):
                    return True
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        text = (part.get("text") or part.get("content") or "").lower()
                        if any(sig in text for sig in _COMPACT_RUNNER_SIGNALS):
                            return True

    # Guard 3: very long history with no tools — likely compact.
    # Conservative threshold: MAIN conversational rarely exceeds 8K tokens without tools.
    # Guard 3b: If system prompt contains MAIN-agent indicators, do NOT flag as compact.
    # This prevents false positives when MAIN runs in conversational mode (no tools, long history).
    _MAIN_AGENT_INDICATORS = ["<tools>", "function", "you are", "tool_choice", "<environment"]
    est_tokens = _estimate_token_count(messages)
    if est_tokens > 8000:
        for msg in messages:
            if (msg.get("role") or "").lower() == "system":
                content = msg.get("content", "")
                if isinstance(content, str):
                    content_lower = content.lower()
                    if any(ind in content_lower for ind in _MAIN_AGENT_INDICATORS):
                        return False  # Looks like MAIN, not compact
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict):
                            text = (part.get("text") or part.get("content") or "").lower()
                            if any(ind in text for ind in _MAIN_AGENT_INDICATORS):
                                return False
        return True

    return False


# ── CASCADE: Forward to Frontier API ─────────────────────────────────────────

def _cascade_forward_request(body: Dict[str, Any], request_id: str,
                             handler=None, is_streaming: bool = False) -> Optional[Dict[str, Any]]:
    """Forward request to a frontier API (Gemini, etc.) when RAG confidence is low.

    Uses urllib (stdlib) — zero extra dependencies. The frontier API must be
    OpenAI-compatible (/v1/chat/completions or equivalent).

    Two modes:
        - Non-streaming (is_streaming=False): returns parsed JSON response dict.
        - Streaming (is_streaming=True): proxies SSE lines directly to handler.wfile,
          returns None. Requires handler to be passed.

    Raises: Exception on timeout, HTTP error, or parse failure.
    """
    import urllib.request
    import urllib.error

    # Build the request payload — pass through most fields from the original body
    cascade_body = {
        "model": CASCADE_MODEL,
        "messages": body.get("messages", []),
        "stream": is_streaming,
    }
    # Pass through optional fields if present
    for _key in ("temperature", "top_p", "max_tokens", "tools", "tool_choice"):
        if _key in body:
            cascade_body[_key] = body[_key]

    _payload = json.dumps(cascade_body, ensure_ascii=False).encode("utf-8")

    _req = urllib.request.Request(
        CASCADE_API_URL,
        data=_payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {CASCADE_API_KEY}",
        },
        method="POST",
    )

    _t0 = time.time()
    try:
        if is_streaming and handler is not None:
            # ── STREAMING: proxy SSE lines directly to client ──
            handler.send_response(200)
            handler.send_header("Content-Type", "text/event-stream")
            handler.send_header("Cache-Control", "no-cache")
            handler.send_header("X-Cascade-Active", "true")
            handler.send_header("X-Cascade-Model", CASCADE_MODEL)
            handler.end_headers()

            with urllib.request.urlopen(_req, timeout=CASCADE_TIMEOUT_S) as resp:
                for line in resp:
                    handler.wfile.write(line)
            handler.wfile.flush()

            _elapsed_ms = int((time.time() - _t0) * 1000)
            _pipeline_log("CASCADE", request_id,
                f"frontier streamed | status=200 | {_elapsed_ms}ms")
            return None  # Response already sent via SSE
        else:
            # ── NON-STREAMING: parse and return JSON ──
            with urllib.request.urlopen(_req, timeout=CASCADE_TIMEOUT_S) as resp:
                _resp_data = resp.read().decode("utf-8")
                _elapsed_ms = int((time.time() - _t0) * 1000)
                _pipeline_log("CASCADE", request_id,
                    f"frontier responded | status={resp.status} | {_elapsed_ms}ms | "
                    f"resp_len={len(_resp_data)}")
                return json.loads(_resp_data)
    except urllib.error.HTTPError as e:
        _err_body = ""
        try:
            _err_body = e.read().decode("utf-8")[:500]
        except Exception:
            pass
        raise RuntimeError(
            f"Cascade HTTP {e.code}: {e.reason} | body={_err_body}"
        ) from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Cascade connection error: {e.reason}") from e


# ══════════════════════════════════════════════════════════════════════════════
# COMPONENT INITIALIZATION — RAG, Compressor
# Deferred to after model load (see post-model-load section below).
# These globals track runtime availability after dependency checks.
# ══════════════════════════════════════════════════════════════════════════════

_compressor_available = False
_compressor_module = None     # Will be set to rag_enricher module

_rag_available = False
_rag_module = None            # Will be set to rag_enricher module (same module)

def _terminal_status(icon: str, message: str, indent: int = 0) -> None:
    with console_lock:
        ts = datetime.now().strftime("%H:%M:%S")
        pad = "  " * max(indent, 0)
        line = f"{pad}{icon} [{ts}] {message}"
        # Highlight emergency compressor lines in yellow
        if "EMERGENCY COMPRESSOR" in message:
            line = f"{_ANSI_YELLOW}{line}{_ANSI_RESET}"
        print(line, flush=True)


def _debug_token_divergence(
    tokenizer,
    current_tokens: List[int],
    stored_tokens: Tuple[int, ...],
    context_window: int = 5,
):
    """Finds and prints exactly where two token sequences diverge for cache debugging."""
    min_len = min(len(current_tokens), len(stored_tokens))
    diverge_idx = -1

    for i in range(min_len):
        if current_tokens[i] != stored_tokens[i]:
            diverge_idx = i
            break

    # Limit how much we print: last few token IDs and short decoded snippets only
    max_tokens_show = 10
    max_text_len = 200

    start_idx = max(0, diverge_idx - context_window)
    tokens_before = current_tokens[start_idx:diverge_idx]
    if len(tokens_before) > max_tokens_show:
        tokens_before = tokens_before[-max_tokens_show:]
    end_idx_current = min(len(current_tokens), diverge_idx + context_window + 1)
    end_idx_stored = min(len(stored_tokens), diverge_idx + context_window + 1)

    print(f"\n" + "=" * 50)
    print(f"🚨 CACHE DIVERGENCE DETECTED AT INDEX {diverge_idx} 🚨")
    print(f"Token IDs before divergence (last {len(tokens_before)}): {tokens_before}")

    try:
        matching_text = tokenizer.decode(current_tokens[start_idx:diverge_idx])
        if len(matching_text) > max_text_len:
            matching_text = "..." + matching_text[-max_text_len:].strip()
        print(f"Matching text leading up: {repr(matching_text)}")

        curr_divergent_token = current_tokens[diverge_idx]
        stor_divergent_token = stored_tokens[diverge_idx]
        print(
            f"\n❌ Current Request Token [{diverge_idx}]: ID {curr_divergent_token} -> {repr(tokenizer.decode([curr_divergent_token]))}"
        )
        print(
            f"❌ Stored Cache Token  [{diverge_idx}]: ID {stor_divergent_token} -> {repr(tokenizer.decode([stor_divergent_token]))}"
        )

        curr_context_after = tokenizer.decode(
            current_tokens[diverge_idx + 1 : end_idx_current]
        )
        stor_context_after = tokenizer.decode(
            stored_tokens[diverge_idx + 1 : end_idx_stored]
        )
        if len(curr_context_after) > max_text_len:
            curr_context_after = curr_context_after[:max_text_len] + "..."
        if len(stor_context_after) > max_text_len:
            stor_context_after = stor_context_after[:max_text_len] + "..."
        print(f"\nCurrent context after: {repr(curr_context_after)}")
        print(f"Stored context after:  {repr(stor_context_after)}")
    except Exception as e:
        print(f"Could not decode tokens: {e}")
    print("=" * 50 + "\n")


def _block_chain_hashes(
    tokens: Tuple[int, ...],
    block_size: int,
) -> List[Tuple[bytes, int]]:
    """Return [(chain_hash, prefix_len), ...] for each block prefix. chain_hash[i] = H(prev || block_i)."""
    if block_size <= 0 or not tokens:
        return []
    out: List[Tuple[bytes, int]] = []
    prev = b""
    for i in range(0, len(tokens), block_size):
        block = tokens[i : i + block_size]
        block_bytes = b"".join(t.to_bytes(4, "big") for t in block)
        h = hashlib.sha256(prev + block_bytes).digest()
        prefix_len = min(i + block_size, len(tokens))
        out.append((h, prefix_len))
        prev = h
    return out


class LRUPromptCache:
    @dataclass
    class CacheEntry:
        prompt_cache: List[Any]
        tokens: Tuple[int, ...]
        count: int
        touched_at: float
        pinned: bool = False  # Kripper Base Slot: nunca evictado si True

    def __init__(self, max_size=10, ttl_seconds=1800):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.block_size = max(16, getattr(SETTINGS, "prompt_cache_block_size", 16))

        # Core Flat Cache: (model, exact_tokens) -> CacheEntry
        self._entries: Dict[Tuple[str, Tuple[int, ...]], self.CacheEntry] = {}
        # Block Hash Index: (model, chain_hash) -> Set of token sequences
        self._block_index: Dict[Tuple[str, bytes], set] = {}

    def _is_expired(self, entry):
        if entry.pinned:  # Kripper Base Slot: nunca expira por TTL
            return False
        if self.ttl_seconds <= 0:
            return False
        return (time.time() - entry.touched_at) > self.ttl_seconds

    def prune_expired(self):
        now = time.time()
        stale_keys = [k for k, v in self._entries.items() if self._is_expired(v)]
        for k in stale_keys:
            self._delete(k[0], k[1])

    def _delete(self, model, tokens, _reaper_telemetry=False):
        key = (model, tuple(tokens))
        if key not in self._entries:
            return

        # Clean up the block index to prevent memory leaks
        chain_pairs = _block_chain_hashes(tokens, self.block_size)
        for chain_hash, _ in chain_pairs:
            idx_key = (model, chain_hash)
            if idx_key in self._block_index:
                self._block_index[idx_key].discard(key[1])
                if not self._block_index[idx_key]:
                    del self._block_index[idx_key]

        del self._entries[key]



        # Return evicted Metal GPU buffers to the OS pool immediately.
        # Without this, MLX holds the backing Metal buffers in its pool even
        # after the Python reference is dropped, causing monotonic GPU memory
        # growth during long sessions with repeated cache divergence.
        try:
            mx.clear_cache()
        except Exception:
            pass

    def _extract(self, model, tokens):
        """Pop entry from LRU and return the LIVE reference (no deepcopy).

        Design: model_lock guarantees only one request runs at a time, so there
        is no concurrent modification risk. The caller gets the live KV object,
        stream_generate extends it in-place, and _insert_cache_entries
        re-inserts the updated state after generation completes.

        Memory impact: halves KV footprint during generation (1 object instead
        of 2).  Previous design kept the original AND a deepcopy alive
        simultaneously, which was the primary OOM trigger on 24 GB hardware.
        """
        key = (model, tuple(tokens))
        entry = self._entries[key]
        entry.touched_at = time.time()
        entry.count += 1

        # Pop from the LRU dict so the slot is FREE during generation.
        # _insert_cache_entries will re-insert the extended state when done.
        del self._entries[key]
        # Clean up block index to keep index consistent.
        chain_pairs = _block_chain_hashes(tokens, self.block_size)
        for chain_hash, _ in chain_pairs:
            idx_key = (model, chain_hash)
            if idx_key in self._block_index:
                self._block_index[idx_key].discard(key[1])
                if not self._block_index[idx_key]:
                    del self._block_index[idx_key]

        # Return a CacheEntry wrapping the original (un-copied) prompt_cache.
        return self.CacheEntry(
            entry.prompt_cache,  # Live reference — NO deepcopy
            entry.tokens,
            entry.count,
            entry.touched_at,
        )

    def _evict_optimal(self):
        """
        Cost-Aware Eviction: Finds the entry with the highest eviction score.
        Protects long 'trunks' and frequently used templates; penalizes old, short branches.
        Kripper Base Slot: entradas con pinned=True son inmunes a evicción.
        Solo si TODOS los slots están pinned, se evicta el más viejo (safety fallback).
        """
        now = time.time()
        best_key = None
        max_score = -1.0

        for key, entry in self._entries.items():
            if entry.pinned:  # Kripper slot: nunca evictar el base
                continue
            age_seconds = max(1.0, now - entry.touched_at)

            # Use square root to create a balanced gravity for long chains.
            # A 10,000 token chain has 10x more protection than a 100 token chain.
            length_weight = math.sqrt(len(entry.tokens))
            freq_weight = math.log1p(entry.count) + 1.0

            # Higher score = more likely to be evicted. (Old and Short -> High Score)
            score = age_seconds / (length_weight * freq_weight)

            if score > max_score:
                max_score = score
                best_key = key

        # Safety fallback: si TODOS los slots están pinned, evictar el pinned más viejo
        # (evita deadlock de memoria por pinning excesivo)
        if best_key is None:
            oldest_key = None
            oldest_time = float("inf")
            for key, entry in self._entries.items():
                if entry.touched_at < oldest_time:
                    oldest_time = entry.touched_at
                    oldest_key = key
            if oldest_key:
                _terminal_status(
                    "⚠️",
                    "Kripper slot: todos los slots están pinned, evictando el más viejo",
                )
                self._delete(oldest_key[0], oldest_key[1])
            return

        if best_key:
            self._delete(best_key[0], best_key[1])

    def _cull_redundant_prefixes(self, model, new_tokens):
        """
        Frees up slots by removing strict prefixes. Since the cache can trim
        longer chains to serve shorter ones, shorter strict prefixes are wasted slots.
        Kripper Base Slot: pinned entries are NEVER culled — they are structural.
        """
        new_len = len(new_tokens)
        to_delete = []
        for m, t_tup in self._entries.keys():
            if m == model and len(t_tup) < new_len:
                # Never cull pinned entries (Kripper slots)
                entry = self._entries.get((m, t_tup))
                if entry and entry.pinned:
                    continue
                # If the existing shorter cache is a strict prefix of the new one
                if new_tokens[: len(t_tup)] == t_tup:
                    to_delete.append((m, t_tup))

        to_delete.sort(key=lambda x: len(x[1]), reverse=True)
        if to_delete:
            # Spare the longest prefix from deletion
            to_delete.pop(0)

        for k in to_delete:
            self._delete(k[0], k[1])

    def evict_unpinned(self) -> int:
        """
        Evict all non-pinned entries immediately to free GPU memory.

        Called before large EMBEDDED prefills to prevent Metal OOM:
          With 36-layer Qwen3.5-9B, each KV slot takes ~4.5GB.
          EMBEDDED deepcopy needs: Kripper(4.5) + SessionCache(4.5) + EMBEDDED_KV(4.5) = 13.5GB
          Evicting session cache first: Kripper(4.5) + EMBEDDED_KV(4.5) = 9GB  → safe.

        Returns number of entries evicted.
        """
        to_delete = [
            (m, t) for (m, t), e in self._entries.items() if not e.pinned
        ]
        for m, t in to_delete:
            self._delete(m, t)
        if to_delete:
            try:
                mx.clear_cache()  # Return Metal buffers immediately
            except Exception:
                pass
        return len(to_delete)

    def fetch_nearest_cache(self, model, tokens):
        self.prune_expired()
        tokens_tup = tuple(tokens)

        # Fast path: exact-token key lookup is O(1) and avoids scanning large
        # candidate sets in block index under multi-session workloads.
        exact_key = (model, tokens_tup)
        if exact_key in self._entries:
            entry = self._extract(model, tokens_tup)
            if len(tokens_tup) > 1:
                return (
                    entry.prompt_cache,
                    tokens_tup[-1:],
                    tokens_tup,
                    "exact",
                    len(tokens_tup) - 1,
                )
            return entry.prompt_cache, [], tokens_tup, "exact", len(tokens_tup)

        # 1. Hash the incoming request into blocks
        chain_pairs = _block_chain_hashes(tokens_tup, self.block_size)
        if not chain_pairs:
            return None, tokens, tokens, "miss", 0

        best_prefix_len = 0
        best_cached_tokens = None

        # 2. Walk blocks backwards. For each block level, collect ALL candidates
        # that match up to that block boundary and pick the LONGEST one.
        # Previously "first matching candidate wins" — but sets are unordered, so
        # a short heartbeat/branch entry could win over a long conversation entry
        # that shares the same early block hashes, causing a near-total cache flush.
        for chain_hash, req_prefix_len in reversed(chain_pairs):
            idx_key = (model, chain_hash)
            if idx_key in self._block_index:
                candidate_tokens_set = self._block_index[idx_key]
                best_candidate_at_level = None
                best_candidate_len = -1
                for candidate_tokens in candidate_tokens_set:
                    if tokens_tup[:req_prefix_len] == candidate_tokens[:req_prefix_len]:
                        if len(candidate_tokens) > best_candidate_len:
                            best_candidate_len = len(candidate_tokens)
                            best_candidate_at_level = candidate_tokens
                if best_candidate_at_level is not None:
                    best_prefix_len = req_prefix_len
                    best_cached_tokens = best_candidate_at_level
            if best_cached_tokens is not None:
                break

        # 3. If no block matched, return miss
        if best_cached_tokens is None:
            return None, tokens, tokens, "miss", 0

        # MEMORY GUARD: If match ratio is trivially low (< 5%), return miss.
        # Avoids using a KV entry that shares almost nothing with the current
        # prompt (e.g. EMBEDDED vs MAIN), which would waste the slot for zero benefit.
        match_ratio = best_prefix_len / max(len(tokens_tup), 1)
        if match_ratio < 0.05:
            return None, tokens, tokens, "miss", 0
        entry = self._extract(model, best_cached_tokens)

        # 5. Extend match inside the current block
        while best_prefix_len < min(len(tokens_tup), len(best_cached_tokens)):
            if tokens_tup[best_prefix_len] == best_cached_tokens[best_prefix_len]:
                best_prefix_len += 1
            else:
                break

        # 6. Return sliced/trimmed cache
        if best_prefix_len == len(tokens_tup):
            # Request is fully covered by selected cache.
            # Trimming is delegated to Fix-31 v3 (Model Space Math)
            if len(tokens_tup) > 1:
                return (
                    entry.prompt_cache,
                    tokens_tup[-1:],
                    best_cached_tokens,
                    "exact",
                    len(tokens_tup) - 1,
                )
            return entry.prompt_cache, [], best_cached_tokens, "exact", len(tokens_tup)

        if best_prefix_len < len(tokens_tup):
            # Shorter Match (We have the prefix, compute the rest)
            # Trimming is delegated to Fix-31 v3 (Model Space Math)
            return (
                entry.prompt_cache,
                list(tokens_tup)[best_prefix_len:],
                best_cached_tokens,
                "shorter",
                best_prefix_len,
            )

        # Longer cache: request is a strict prefix of cached
        # Trimming is delegated to Fix-31 v3 (Model Space Math)
        return entry.prompt_cache, [], best_cached_tokens, "longer", len(tokens_tup)

    def insert_cache(self, model, tokens, prompt_cache, pinned: bool = False):
        self.prune_expired()
        tokens_tup = tuple(tokens)
        key = (model, tokens_tup)
        now = time.time()

        if key in self._entries:
            self._entries[key].count += 1
            self._entries[key].touched_at = now
            if pinned:  # Kripper slot: upgrade a pinned si se re-inserta como base
                self._entries[key].pinned = True
            return

        # NOTE: Eviction guard removed — with all sessions ~13K tokens,
        # the guard was locking the first stale entry forever.
        # Tool normalization alone handles cross-session compatibility.

        # 1. Subsumption: Cull redundant prefixes to organically free up space
        self._cull_redundant_prefixes(model, tokens_tup)

        # 2. Insert into flat dictionary
        self._entries[key] = self.CacheEntry(
            prompt_cache, tokens_tup, 1, now, pinned=pinned
        )

        # 3. Map block hashes
        chain_pairs = _block_chain_hashes(tokens_tup, self.block_size)
        for chain_hash, _ in chain_pairs:
            idx_key = (model, chain_hash)
            if idx_key not in self._block_index:
                self._block_index[idx_key] = set()
            self._block_index[idx_key].add(tokens_tup)

        # 4. Enforce max size using the Cost-Aware Eviction
        while len(self._entries) > self.max_size:
            self._evict_optimal()

    def contains_tokens(self, model, tokens):
        key = (model, tuple(tokens))
        return key in self._entries

    def extract_exact_cache(self, model, tokens):
        if not self.contains_tokens(model, tokens):
            return None
        return self._extract(model, tuple(tokens))


PROMPT_CACHE = LRUPromptCache(
    max_size=SETTINGS.prompt_cache_max_entries_global,
    ttl_seconds=SETTINGS.prompt_cache_ttl_seconds,
)

# KRIPPER DUAL-SLOT: LRU dedicado para el compact runner de OpenClaw.
# Isolado de PROMPT_CACHE para que nunca evicte el slot MAIN.
# max_size=1: el compact genera un resumen puntual, no necesita histórico.
PROMPT_CACHE_COMPACT = LRUPromptCache(
    max_size=1,
    ttl_seconds=SETTINGS.prompt_cache_ttl_seconds,
)


@dataclass(frozen=True)
class SessionContext:
    session_id: str
    parent_session_id: Optional[str]
    branch_id: Optional[str]
    source: str


class SessionIndex:
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

    def _lineage_chain(self, session_id: str, max_depth: int = 8) -> List[str]:
        chain: List[str] = []
        seen = set()
        current = session_id
        depth = 0
        while current and current not in seen and depth < max_depth:
            chain.append(current)
            seen.add(current)
            state = self._sessions.get(current)
            if state is None:
                break
            current = state.parent_session_id
            depth += 1
        return chain

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
        self, session_ctx: SessionContext, cache_key: List[int]
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

    def _selection_from_exact_entry(
        self,
        prompt_tokens: List[int],
        cache_tokens: Tuple[int, ...],
        cache_entry: Any,
    ):
        prefix_len = self._lcp_len(prompt_tokens, cache_tokens)
        if prefix_len <= 0:
            return None

        if len(cache_tokens) > prefix_len:
            if not can_trim_prompt_cache(cache_entry.prompt_cache):
                return None
            trim_prompt_cache(cache_entry.prompt_cache, len(cache_tokens) - prefix_len)

        if prefix_len == len(prompt_tokens):
            if len(prompt_tokens) > 1 and can_trim_prompt_cache(
                cache_entry.prompt_cache
            ):
                trim_prompt_cache(cache_entry.prompt_cache, 1)
                return (
                    cache_entry.prompt_cache,
                    prompt_tokens[-1:],
                    list(cache_tokens),
                    "exact",
                    len(prompt_tokens) - 1,
                )
            return (
                cache_entry.prompt_cache,
                prompt_tokens,
                list(cache_tokens),
                "exact",
                len(prompt_tokens),
            )

        return (
            cache_entry.prompt_cache,
            prompt_tokens[prefix_len:],
            list(cache_tokens),
            "shorter",
            prefix_len,
        )

    def select_best_cache(
        self,
        model_name: str,
        prompt_tokens: List[int],
        session_ctx: SessionContext,
        prompt_cache_store: LRUPromptCache,
    ):
        """Always use global cache; session/conversation/thread ID is ignored for lookup."""
        prompt_cache_store.prune_expired()
        self._prune_idle()
        selected = prompt_cache_store.fetch_nearest_cache(model_name, prompt_tokens)
        return (*selected, "global")


SESSION_INDEX = SessionIndex(
    max_entries_per_session=SETTINGS.prompt_cache_max_entries_per_session,
    max_idle_seconds=SETTINGS.prompt_cache_session_max_idle_seconds,
)


# --- MESSAGE-AWARE STABLE-PREFIX CACHE (Phase 2) ---


@dataclass
class _SessionTurnRecord:
    """Lightweight per-session record of the last completed turn's message structure."""

    messages: List[Dict[str, Any]]  # normalised message list used for diff key
    msg_token_lens: List[int]  # token count for each message (in order)
    total_prompt_tokens: (
        int  # sum(msg_token_lens); equals len(prompt_tokens) for that turn
    )
    touched_at: float


# Global store: session_id -> _SessionTurnRecord
# Protected by prompt_cache_lock (same lock used for PROMPT_CACHE).
SESSION_TURN_STORE: Dict[str, _SessionTurnRecord] = {}
_SESSION_TURN_MAX_IDLE_SECONDS: int = SETTINGS.prompt_cache_session_max_idle_seconds


def _normalize_message_content_for_diff(msg: Dict[str, Any]) -> str:
    """
    Return a normalised string representation of a message's content for use
    ONLY as a diff key. The original message is never modified.

    Currently strips:
    - Leading/trailing whitespace differences (trailing space is a real FP-1 trigger)
    - No other normalisation until confirmed from real OpenCode/OpenClaw logs.

    M4 IMPLEMENTATION NOTE: When real session logs from OpenCode/OpenClaw are
    audited, add confirmed volatile-field stripping here (timestamps, system-reminder
    injections, etc.). The post-render _scrub_cache_key() patterns are a reference
    but are applied at the serialised-string level — they need to be adapted here at
    the per-message level after real-traffic confirmation.
    """
    content = msg.get("content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        # Multi-part content (e.g. VLM image+text). Use JSON with stripped text parts.
        parts = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text":
                    parts.append({"type": "text", "text": part.get("text", "").strip()})
                else:
                    parts.append(part)
            else:
                parts.append(part)
        return json.dumps(parts, sort_keys=True, ensure_ascii=False)
    return json.dumps(content, sort_keys=True, ensure_ascii=False)


def _message_diff(
    prev_msgs: List[Dict[str, Any]],
    curr_msgs: List[Dict[str, Any]],
) -> Tuple[int, List[Dict[str, Any]]]:
    """
    Diff two message lists at message-boundary level.

    Returns:
        stable_prefix_count (int): number of messages from the START of both lists
            that are normalisation-identical in role and content. These messages'
            KV state can be safely reused.
        descriptors (list): list of change descriptors for telemetry/debugging.

    Rules:
    - Two messages are "equal" if their role is identical AND their normalised
      content (_normalize_message_content_for_diff) is identical.
    - Only the LEADING equal block contributes to stable_prefix_count.
      If messages [0..K-1] are equal and message[K] differs, stable_prefix_count = K.
    - Later equal blocks (after a change) are NOT counted — RoPE position
      correctness requires contiguous prefix reuse only.
    """
    stable_prefix_count = 0
    for i, (prev, curr) in enumerate(zip(prev_msgs, curr_msgs)):
        if prev.get("role") == curr.get("role") and _normalize_message_content_for_diff(
            prev
        ) == _normalize_message_content_for_diff(curr):
            stable_prefix_count = i + 1
        else:
            break

    descriptors: List[Dict[str, Any]] = []
    n = max(len(prev_msgs), len(curr_msgs))
    for i in range(n):
        p = prev_msgs[i] if i < len(prev_msgs) else None
        c = curr_msgs[i] if i < len(curr_msgs) else None
        if p is None:
            descriptors.append({"idx": i, "op": "insert", "role": c.get("role")})
        elif c is None:
            descriptors.append({"idx": i, "op": "delete", "role": p.get("role")})
        elif p.get("role") != c.get("role") or _normalize_message_content_for_diff(
            p
        ) != _normalize_message_content_for_diff(c):
            descriptors.append(
                {
                    "idx": i,
                    "op": "replace",
                    "role_prev": p.get("role"),
                    "role_curr": c.get("role"),
                }
            )

    return stable_prefix_count, descriptors


def _stable_prefix_token_len(
    session_id: str,
    curr_msgs: List[Dict[str, Any]],
) -> Tuple[int, int, List[Dict[str, Any]]]:
    """
    For a given session and incoming message list, determine the stable prefix
    in tokens by consulting the SESSION_TURN_STORE.

    Returns:
        stable_token_len (int): number of tokens at the start of the prompt that
            are known-stable from the previous turn. 0 if no prior turn or no match.
        stable_msg_count (int): number of leading messages that are stable.
        descriptors (list): diff descriptors for telemetry.

    Must be called while holding prompt_cache_lock (reads SESSION_TURN_STORE).
    """
    record = SESSION_TURN_STORE.get(session_id)
    if record is None:
        return 0, 0, []

    stable_msg_count, descriptors = _message_diff(record.messages, curr_msgs)
    if stable_msg_count == 0:
        return 0, 0, descriptors

    # Sum the token lengths of the stable prefix messages
    stable_token_len = sum(record.msg_token_lens[:stable_msg_count])
    return stable_token_len, stable_msg_count, descriptors


def _compute_msg_token_boundaries(
    messages: List[Dict[str, Any]],
    prompt_tokens: List[int],
) -> List[int]:
    """
    Compute accurate per-message token boundary lengths using cumulative chat-template
    rendering. Each boundary is the cumulative token count up to and including that
    message in the rendered prompt.

    Strategy: render messages[0..i] (without generation prompt) through the actual
    chat template and measure the token count. The per-message length is the diff
    between consecutive cumulative counts.

    Returns a list of per-message token counts (same length as messages).
    The last message gets len(prompt_tokens) - sum(prev) as ground truth.

    For VLM prompts: tokenizer may not have apply_chat_template; falls back to
    equal distribution with last-message remainder correction.
    """
    n = len(messages)
    if n == 0:
        return []

    msg_token_lens: List[int] = []

    # Prefer apply_chat_template if available (gives exact boundaries)
    if (
        tokenizer is not None
        and hasattr(tokenizer, "apply_chat_template")
        and getattr(tokenizer, "chat_template", None) is not None
        and not is_vlm
    ):
        try:
            cumulative_lengths: List[int] = []
            prev_len = 0
            for i in range(n):
                # Render prefix messages[0..i] without generation prompt so the
                # boundary lands exactly at the end of message i.
                # We suppress add_generation_prompt to avoid including the assistant
                # start token in the boundary count.
                prefix_toks = tokenizer.apply_chat_template(
                    messages[: i + 1],
                    tokenize=True,
                    add_generation_prompt=False,
                )
                if isinstance(prefix_toks, list):
                    cur_len = len(prefix_toks)
                else:
                    cur_len = prev_len
                msg_token_lens.append(max(0, cur_len - prev_len))
                prev_len = cur_len
            # Correct the last bucket using prev_len, which after the loop holds the
            # token count for the full message list rendered WITHOUT a generation
            # prompt (last loop iteration called apply_chat_template(messages[:n], ...,
            # add_generation_prompt=False)). We intentionally do NOT use
            # len(prompt_tokens) here because prompt_tokens was rendered with
            # add_generation_prompt=True (and enable_thinking=True for Qwen3), which
            # appends tokens such as "<|im_start|>assistant\n<think>\n\n". If those
            # tokens were absorbed into the last-message bucket,
            # _stable_prefix_token_len would sum past the real message content into
            # the generation-prompt region. On the next request the same position
            # holds a different token depending on how the new response starts
            # (e.g. <think>\n ID 198 vs <think>\n\n ID 271), causing the
            # "cache divergence at <think>" bug with Qwen3 + tool calls.
            # prev_len gives a boundary that stops exactly at the last real message.
            if msg_token_lens:
                prefix_sum = sum(msg_token_lens[:-1])
                msg_token_lens[-1] = max(0, prev_len - prefix_sum)
            return msg_token_lens
        except Exception:
            pass  # Fall through to approximation

    # Fallback: divide evenly, correct last bucket with remainder
    per = len(prompt_tokens) // max(1, n)
    msg_token_lens = [per] * (n - 1) + [max(0, len(prompt_tokens) - per * (n - 1))]
    return msg_token_lens


def _update_session_turn_store(
    session_id: str,
    messages: List[Dict[str, Any]],
    prompt_tokens: List[int],
) -> None:
    """
    Record the per-message token boundary information for this session's completed turn.
    Must be called while holding prompt_cache_lock.

    Per-message token lengths are computed via cumulative chat-template rendering
    (_compute_msg_token_boundaries) to give exact boundaries. The stable-prefix
    secondary lookup depends on these boundaries being accurate: if the boundary
    is over-estimated, the lookup prefix extends into the next (changed) message
    and the block-hash mismatches.
    """
    if not session_id or not messages:
        return

    # Prune stale entries. A value of 0 means "never expire" (infinite TTL),
    # consistent with LRUPromptCache._is_expired which returns False when
    # ttl_seconds <= 0. Without this guard, _SESSION_TURN_MAX_IDLE_SECONDS=0
    # makes (now - touched_at) > 0 always True, pruning every record on every
    # write and silently destroying the stable-prefix mechanism for all sessions.
    now = time.time()
    if _SESSION_TURN_MAX_IDLE_SECONDS > 0:
        stale = [
            sid
            for sid, rec in SESSION_TURN_STORE.items()
            if (now - rec.touched_at) > _SESSION_TURN_MAX_IDLE_SECONDS
        ]
        for sid in stale:
            del SESSION_TURN_STORE[sid]

    # Guard: only write if the new message list is a strict append of the existing record.
    # If the existing record's messages are NOT a prefix of the incoming list, this request
    # is from a sub-agent or parallel branch that has a structurally different conversation
    # history on the same session_id. Writing would clobber the orchestrator's record and
    # corrupt the stable-prefix diff for the next orchestrator turn. Skip silently.
    existing = SESSION_TURN_STORE.get(session_id)
    if existing is not None:
        prev = existing.messages
        n_prev = len(prev)
        if len(messages) < n_prev:
            # Shorter than what we already have — definitely not an append. Skip.
            return
        for i in range(n_prev):
            if prev[i].get("role") != messages[i].get(
                "role"
            ) or _normalize_message_content_for_diff(
                prev[i]
            ) != _normalize_message_content_for_diff(messages[i]):
                # A prior message changed — this is not a linear continuation.
                # The diff logic would still produce a valid (possibly lower) stable_prefix_count,
                # but the stored record would now reflect a diverged branch. Skip to preserve
                # the best-known linear record for this session.
                return

    msg_token_lens = _compute_msg_token_boundaries(messages, prompt_tokens)

    SESSION_TURN_STORE[session_id] = _SessionTurnRecord(
        messages=messages,
        msg_token_lens=msg_token_lens,
        total_prompt_tokens=len(prompt_tokens),
        touched_at=now,
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


def _flatten_content(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_content = ""
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text_content += part.get("text", "")
            elif isinstance(part, str):
                text_content += part
        return text_content
    return content


def _prepare_messages_for_template(messages):
    normalized = []
    for msg in messages:
        m = dict(msg)
        m["content"] = _flatten_content(m.get("content", ""))

        # OpenAI tool_calls use function.arguments as a JSON string.
        # GLM's template expects arguments to be a mapping so it can iterate items().
        if m.get("role") == "assistant" and isinstance(m.get("tool_calls"), list):
            fixed_tool_calls = []
            for tc in m["tool_calls"]:
                tc_copy = dict(tc)
                fn = tc_copy.get("function")
                if isinstance(fn, dict):
                    fn_copy = dict(fn)
                    fn_name = str(fn_copy.get("name", "")).strip().lower()
                    args = fn_copy.get("arguments")
                    if isinstance(args, str):
                        try:
                            fn_copy["arguments"] = json.loads(args)
                        except Exception:
                            fn_copy["arguments"] = {"raw": args}
                    if (
                        SETTINGS.normalize_write_tool_content_for_prompt
                        and fn_name == "write"
                        and isinstance(fn_copy.get("arguments"), dict)
                        and "content" in fn_copy["arguments"]
                    ):
                        # Keep write semantics (path + write intent) while stabilizing
                        # prompt tokens by replacing huge inline content with a digest tag.
                        raw_content = fn_copy["arguments"]["content"]
                        if isinstance(raw_content, str):
                            raw_bytes = raw_content.encode("utf-8")
                            digest = hashlib.sha1(raw_bytes).hexdigest()[:16]
                            placeholder = (
                                f"__WRITE_CONTENT_OMITTED__sha1={digest};"
                                f"bytes={len(raw_bytes)};chars={len(raw_content)};"
                                f"lines={raw_content.count(chr(10)) + 1}__"
                            )
                        else:
                            serialized = json.dumps(
                                raw_content, ensure_ascii=False, sort_keys=True
                            )
                            raw_bytes = serialized.encode("utf-8")
                            digest = hashlib.sha1(raw_bytes).hexdigest()[:16]
                            placeholder = (
                                f"__WRITE_CONTENT_OMITTED_NONSTRING__sha1={digest};"
                                f"bytes={len(raw_bytes)}__"
                            )
                        args_copy = dict(fn_copy["arguments"])
                        args_copy["content"] = placeholder
                        fn_copy["arguments"] = args_copy
                    tc_copy["function"] = fn_copy
                fixed_tool_calls.append(tc_copy)
            m["tool_calls"] = fixed_tool_calls

        normalized.append(m)
    return normalized


def _messages_have_images(messages: List[Dict[str, Any]]) -> bool:
    """True if any message has content list containing image_url or input_image."""
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict):
                t = (part.get("type") or "").strip().lower()
                if t in ("image_url", "input_image"):
                    return True
                if "image_url" in part or "input_image" in part:
                    return True
    return False


def _extract_images_from_messages(messages: List[Dict[str, Any]]) -> List[Any]:
    """
    Extract image sources from OpenAI-style content (type image_url / input_image).
    Returns list in message order (data URL strings or PIL Images for prepare_inputs).
    """
    import base64
    from io import BytesIO

    images = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            url = None
            if part.get("type") == "image_url":
                u = part.get("image_url") or {}
                url = u.get("url") if isinstance(u, dict) else None
            elif part.get("type") == "input_image":
                u = part.get("input_image") or part.get("image_url") or {}
                url = (
                    u.get("url")
                    if isinstance(u, dict)
                    else u
                    if isinstance(u, str)
                    else None
                )
            if not url or not isinstance(url, str):
                continue
            url = url.strip()
            if url.startswith("data:image/") and "," in url:
                try:
                    _, b64 = url.split(",", 1)
                    from PIL import Image

                    img = Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")
                    images.append(img)
                except Exception:
                    images.append(url)
            else:
                images.append(url)
    return images


def _prepare_messages_for_vlm(
    messages: List[Dict[str, Any]], tools: Optional[Any] = None
) -> List[Dict[str, Any]]:
    """
    Normalize messages for VLM: fix tool_calls like _prepare_messages_for_template,
    but preserve content as list (text + image_url) so get_chat_template can insert image tokens.
    """
    normalized = []
    for msg in messages:
        m = dict(msg)
        content = m.get("content", "")
        if isinstance(content, list):
            new_content = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text = part.get("text") or part.get("content") or ""
                    # NOTE: do NOT normalize here — model must see original content.
                    # Canonicalization for cache key happens via _canonicalize_messages()
                    # before rendering, and only in the cache key pipeline.
                    new_content.append({**part, "text": text})
                else:
                    new_content.append(part)
            m["content"] = new_content
        elif isinstance(content, str):
            m["content"] = content
        if m.get("role") == "assistant" and isinstance(m.get("tool_calls"), list):
            fixed_tool_calls = []
            for tc in m["tool_calls"]:
                tc_copy = dict(tc)
                fn = tc_copy.get("function")
                if isinstance(fn, dict):
                    fn_copy = dict(fn)
                    args = fn_copy.get("arguments")
                    if isinstance(args, str):
                        try:
                            fn_copy["arguments"] = json.loads(args)
                        except Exception:
                            fn_copy["arguments"] = {"raw": args}
                    tc_copy["function"] = fn_copy
                fixed_tool_calls.append(tc_copy)
            m["tool_calls"] = fixed_tool_calls
        normalized.append(m)
    return normalized


def _vlm_prompt_and_inputs(
    processor_any,
    config: Dict[str, Any],
    messages: List[Dict[str, Any]],
    images: List[Any],
    tools: Optional[Any] = None,
    enable_thinking: Optional[bool] = None,
    **kwargs: Any,
) -> Tuple[Any, Any, Any, Any]:
    """
    Build formatted prompt and run prepare_inputs for VLM using native chat templates.
    Returns (input_ids, pixel_values, mask, vlm_kwargs) where input_ids is mx.array; mask may be None.
    """
    template_kwargs = dict(kwargs)
    if tools is not None:
        template_kwargs["tools"] = tools
    if enable_thinking is not None:
        template_kwargs["enable_thinking"] = enable_thinking

    template_processor = None
    if processor_any is not None and hasattr(processor_any, "apply_chat_template"):
        if getattr(processor_any, "chat_template", None) is not None:
            template_processor = processor_any
    if (
        template_processor is None
        and getattr(processor_any, "tokenizer", None) is not None
    ):
        tok = processor_any.tokenizer
        if (
            hasattr(tok, "apply_chat_template")
            and getattr(tok, "chat_template", None) is not None
        ):
            template_processor = tok

    formatted = ""
    if template_processor is not None:
        try:
            formatted = template_processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **template_kwargs,
            )
            if isinstance(formatted, list):
                formatted = formatted[0] if formatted else ""
            formatted = str(formatted)
        except Exception:
            formatted = ""

    # Fallback to mlx_vlm's get_chat_template if HF template fails/is missing
    if not formatted:
        out = get_chat_template(
            processor_any,
            messages,
            add_generation_prompt=True,
            tokenize=False,
            **template_kwargs,
        )
        if isinstance(out, list):
            out = out[0].get("content", "") if out else ""
        formatted = str(out) if out else ""

    # NOTE: do NOT scrub the formatted string here — this is the model input.
    # Post-render scrubbing for the cache key is applied in do_POST on the
    # cache_prompt_raw string (canonical pipeline), never on the model input.

    inputs = vlm_prepare_inputs(
        processor_any,
        images=images if images else None,
        prompts=formatted,
        add_special_tokens=True,
        return_tensors="mlx",
    )

    input_ids = inputs.get("input_ids")
    pixel_values = inputs.get("pixel_values")
    mask = inputs.get("attention_mask")
    vlm_kwargs = {
        k: v
        for k, v in inputs.items()
        if k not in ["input_ids", "pixel_values", "attention_mask"]
    }

    return input_ids, pixel_values, mask, vlm_kwargs


def _vlm_sync_before_generation(pixel_values: Any, mask: Any) -> None:
    """
    Flush Metal work before VLM generation to prevent PyTorch and MLX from colliding.
    """
    try:
        import torch

        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.synchronize()
            torch.mps.empty_cache()  # CRITICAL: Force PyTorch to release all Metal encoders
    except Exception:
        pass

    to_eval = []
    if pixel_values is not None and hasattr(pixel_values, "shape"):
        to_eval.append(pixel_values)
    if mask is not None and hasattr(mask, "shape"):
        to_eval.append(mask)
    if to_eval:
        try:
            import mlx.core as mx

            mx.eval(*to_eval)
        except Exception:
            pass


def _should_enable_thinking(body):
    if isinstance(body.get("enable_thinking"), bool):
        return body["enable_thinking"]
    # Default to enabled
    return SETTINGS.default_thinking


def _reasoning_level_to_enable_thinking(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    # Anthropic-style dict: {"type": "disabled"} / {"type": "enabled"}
    if isinstance(value, dict):
        t = value.get("type")
        if isinstance(t, str):
            return _reasoning_level_to_enable_thinking(t)  # recurse con el string
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"off", "none", "disable", "disabled", "false", "0", "no"}:
            return False
        if normalized in {
            "minimal",
            "low",
            "medium",
            "high",
            "xhigh",
            "default",
            "on",
            "enabled",
            "true",
            "1",
            "yes",
            "auto",
        }:
            return True
    return None


def _extract_enable_thinking(body: Dict[str, Any]) -> Dict[str, Any]:
    """
    Determine enable_thinking from explicit flag first, then common reasoning fields.

    Supported sources:
    - enable_thinking: bool
    - thinking: str|bool|dict (OpenClaw / Anthropic style)
    - reasoning_effort: str (OpenAI / LiteLLM style)
    - reasoning: str|dict (Responses-style)
    - metadata.{thinking, reasoning, reasoning_effort}
    - extra_body.{thinking, reasoning, reasoning_effort}
    """
    if isinstance(body.get("enable_thinking"), bool):
        return {
            "enable_thinking": body["enable_thinking"],
            "source": "enable_thinking",
            "raw": body["enable_thinking"],
        }

    for key in ("thinking", "reasoning_effort"):
        mapped = _reasoning_level_to_enable_thinking(body.get(key))
        if mapped is not None:
            return {
                "enable_thinking": mapped,
                "source": key,
                "raw": body.get(key),
            }

    reasoning_obj = body.get("reasoning")
    if isinstance(reasoning_obj, dict):
        for nested_key in ("enabled", "effort", "level", "type"):
            mapped = _reasoning_level_to_enable_thinking(reasoning_obj.get(nested_key))
            if mapped is not None:
                return {
                    "enable_thinking": mapped,
                    "source": f"reasoning.{nested_key}",
                    "raw": reasoning_obj.get(nested_key),
                }
    else:
        mapped = _reasoning_level_to_enable_thinking(reasoning_obj)
        if mapped is not None:
            return {
                "enable_thinking": mapped,
                "source": "reasoning",
                "raw": reasoning_obj,
            }

    for container_key in ("metadata", "extra_body"):
        container = body.get(container_key)
        if not isinstance(container, dict):
            continue
        for key in ("thinking", "reasoning_effort"):
            mapped = _reasoning_level_to_enable_thinking(container.get(key))
            if mapped is not None:
                return {
                    "enable_thinking": mapped,
                    "source": f"{container_key}.{key}",
                    "raw": container.get(key),
                }
        nested_reasoning = container.get("reasoning")
        if isinstance(nested_reasoning, dict):
            for nested_key in ("enabled", "effort", "level", "type"):
                mapped = _reasoning_level_to_enable_thinking(
                    nested_reasoning.get(nested_key)
                )
                if mapped is not None:
                    return {
                        "enable_thinking": mapped,
                        "source": f"{container_key}.reasoning.{nested_key}",
                        "raw": nested_reasoning.get(nested_key),
                    }
        else:
            mapped = _reasoning_level_to_enable_thinking(nested_reasoning)
            if mapped is not None:
                return {
                    "enable_thinking": mapped,
                    "source": f"{container_key}.reasoning",
                    "raw": nested_reasoning,
                }

    return {
        "enable_thinking": _should_enable_thinking(body),
        "source": "default",
        "raw": None,
    }


def _normalize_assistant_text(text, enable_thinking, model_family):
    if not isinstance(text, str):
        return text
    # Qwen3/DeepSeek/Hermes/GLM output can be sensitive to synthetic prefix injection.
    # Keep raw text stable so next-turn prompt tokens stay cache-friendly.
    if model_family in ("qwen3", "deepseek", "hermes", "glm4"):
        return text
    # Gemma 4 uses <|think|> instead of <think>
    if model_family == "gemma4":
        if enable_thinking and text and not text.lstrip().startswith("<|think|>"):
            return "<|think|>" + text
        return text
    if enable_thinking and text and not text.lstrip().startswith("<think>"):
        return "<think>" + text
    return text


def _coerce_arg_value(raw_value):
    value = raw_value.strip()
    try:
        return json.loads(value)
    except Exception:
        return value


def _extract_openai_tool_calls(text, model_family):
    if not isinstance(text, str):
        return text, []
    # Quick-exit: no tool call markers at all
    _has_standard = "<tool_call>" in text
    _has_gemma4 = "<|tool_call|>" in text
    if not _has_standard and not _has_gemma4:
        return text, []

    def _parse_legacy_block(body):
        name_match = re.match(r"^([^\s<]+)", body)
        if not name_match:
            return None
        tool_name = name_match.group(1).strip()
        if not tool_name:
            return None
        args = {}
        for arg_key, arg_value in ARG_PAIR_PATTERN.findall(body):
            key = arg_key.strip()
            if not key:
                continue
            args[key] = _coerce_arg_value(arg_value)
        return {
            "id": f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {
                "name": tool_name,
                "arguments": json.dumps(args, ensure_ascii=False),
            },
        }

    def _parse_qwen_block(body):
        function_match = QWEN_FUNCTION_PATTERN.search(body)
        if not function_match:
            return None
        tool_name = function_match.group(1).strip()
        fn_body = function_match.group(2)
        if not tool_name:
            return None
        args = {}
        for key, value in QWEN_PARAMETER_PATTERN.findall(fn_body):
            param_key = key.strip()
            if not param_key:
                continue
            args[param_key] = _coerce_arg_value(value)
        return {
            "id": f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {
                "name": tool_name,
                "arguments": json.dumps(args, ensure_ascii=False),
            },
        }

    def _parse_json_direct_block(body):
        """Parse JSON-direct tool call format used by GLM-4, Hermes 3, Gemma 4, DeepSeek.
        Expects body to be raw JSON: {"name": "func", "arguments": {...}}
        or {"function": {"name": ..., "arguments": ...}} (OpenAI-compat variant).
        """
        try:
            data = json.loads(body)
        except (json.JSONDecodeError, ValueError):
            return None
        if not isinstance(data, dict):
            return None
        # Variant 1: {"name": "...", "arguments": {...}}
        tool_name = data.get("name")
        arguments = data.get("arguments")
        # Variant 2: {"function": {"name": ..., "arguments": ...}}
        if not tool_name and "function" in data:
            fn = data["function"]
            if isinstance(fn, dict):
                tool_name = fn.get("name")
                arguments = fn.get("arguments")
        if not tool_name:
            return None
        # arguments can be dict or already-serialized string
        if isinstance(arguments, dict):
            args_str = json.dumps(arguments, ensure_ascii=False)
        elif isinstance(arguments, str):
            args_str = arguments
        else:
            args_str = json.dumps(arguments or {}, ensure_ascii=False)
        return {
            "id": f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {
                "name": tool_name,
                "arguments": args_str,
            },
        }

    tool_calls = []
    remove_spans = []

    # Determine which pattern(s) to use based on model family
    patterns = []
    if _has_gemma4:
        patterns.append(GEMMA4_TOOL_CALL_PATTERN)
    if _has_standard:
        patterns.append(TOOL_CALL_PATTERN)

    for pattern in patterns:
        for match in pattern.finditer(text):
            body = match.group(1).strip()
            if not body:
                continue
            parsed = None
            if model_family == "qwen3":
                # Qwen priority: <function=NAME> first, then legacy, then JSON
                parsed = _parse_qwen_block(body)
                if parsed is None:
                    parsed = _parse_legacy_block(body)
                if parsed is None:
                    parsed = _parse_json_direct_block(body)
            elif model_family in ("gemma4", "deepseek", "hermes", "glm4"):
                # JSON-first families: try JSON direct, then fall back
                parsed = _parse_json_direct_block(body)
                if parsed is None:
                    parsed = _parse_legacy_block(body)
                if parsed is None:
                    parsed = _parse_qwen_block(body)
            else:
                # Generic: legacy first, then JSON, then Qwen
                parsed = _parse_legacy_block(body)
                if parsed is None:
                    parsed = _parse_json_direct_block(body)
                if parsed is None:
                    parsed = _parse_qwen_block(body)
            if parsed is None:
                continue
            tool_calls.append(parsed)
            remove_spans.append(match.span())

    if not remove_spans:
        return text, tool_calls

    # Sort spans to handle overlapping patterns from multiple passes
    remove_spans.sort(key=lambda s: s[0])
    cleaned_parts = []
    cursor = 0
    for start, end in remove_spans:
        if start < cursor:
            continue  # Skip overlapping span
        if start > cursor:
            cleaned_parts.append(text[cursor:start])
        cursor = end
    if cursor < len(text):
        cleaned_parts.append(text[cursor:])
    cleaned_text = "".join(cleaned_parts).strip()
    return cleaned_text, tool_calls


def _tokenize_prompt(prompt):
    if isinstance(prompt, str):
        add_special_tokens = tokenizer.bos_token is None or not prompt.startswith(
            tokenizer.bos_token
        )
        return tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
    if isinstance(prompt, list):
        return prompt
    return list(prompt)


def _insert_cache_entries(
    model_name: str,
    session_ctx: "SessionContext",
    cache_key: List[int],
    prompt_cache: Any,
    generated_tokens: List[int],
    tool_calls: Optional[List[Dict[str, Any]]] = None,
    cache_hit_ratio: float = 1.0,
    prompt_cache_store_override: Optional["LRUPromptCache"] = None,
    is_warmup_candidate: bool = False,  # True solo para el startup /new (messages==2, MAIN runner)
) -> None:
    """
    Insert the standard full-turn cache entry and, for tool-call or VLM turns,
    also insert a prompt-only checkpoint. The checkpoint helps next-turn prefix
    reuse when provider-side tool-call serialisation differs between turns, or
    when VLM vision tokens change the effective prompt boundary.

    For plain LM turns without tool calls the full-key entry alone is sufficient:
    the next request will get a 'shorter' hit via _cull_redundant_prefixes or
    trim, without requiring a second deepcopy here. Gating the deepcopy prevents
    two full KV tensors (~2–3 GB each at 20k tokens) from living simultaneously
    in GPU memory on every non-tool turn.

    Cache eviction protection: when running with a single cache entry
    (PROMPT_CACHE_MAX_ENTRIES_GLOBAL=1), skip insertion if the current request
    had a very low cache hit ratio (<30%), which indicates a cross-session
    request that would destroy the primary session's warm cache.

    KRIPPER DUAL-SLOT: prompt_cache_store_override allows routing to a
    dedicated LRU (e.g. PROMPT_CACHE_COMPACT) without touching PROMPT_CACHE.
    """
    # NOTE: Eviction guard removed — it was blocking ALL insertions because
    # a tiny initial entry (285 tokens) caused all subsequent entries to have
    # hit_ratio < 30%, preventing the cache from ever being populated.
    # TODO: If cross-session thrashing is a problem, compare entry token counts
    # instead of hit ratios (only block if existing entry is LARGER than new one).

    # Select which LRU store to use: override (compact slot) or global MAIN store.
    _store = prompt_cache_store_override if prompt_cache_store_override is not None else PROMPT_CACHE

    # Only insert a prompt-only checkpoint for turns where it actually helps:
    # - tool_calls: next turn's prompt differs from cache key due to serialisation
    # - is_vlm: VLM requests need an explicit prefix entry for vision-token reuse
    # Plain LM turns get 'shorter' cache hits from the full-key entry alone.
    # MEMORY GUARD: Skip the expensive deepcopy when running with very limited cache
    # entries (<2). The deepcopy temporarily doubles GPU memory (~3 GB at 25K tokens)
    # and is the primary source of OOM crashes. The eviction guard already protects
    # the primary session's cache, so the checkpoint is not strictly necessary.
    # NOTE: >= 2 (was > 2) — with max_entries=4, the deepcopy is safe. (PEND-04 fix 2026-04-07)
    if (
        SETTINGS.prompt_cache_max_entries_global >= 2
        and (tool_calls or is_vlm)
        and generated_tokens
        and can_trim_prompt_cache(prompt_cache)
        and len(cache_key) > len(generated_tokens)
    ):
        try:
            prompt_only_cache = copy.deepcopy(prompt_cache)
            trim_prompt_cache(prompt_only_cache, len(generated_tokens))
            prompt_only_key = cache_key[: -len(generated_tokens)]
            _store.insert_cache(model_name, prompt_only_key, prompt_only_cache)
            SESSION_INDEX.register_cache_key(session_ctx, prompt_only_key)
        except Exception:
            pass

    _store.insert_cache(model_name, cache_key, prompt_cache)
    SESSION_INDEX.register_cache_key(session_ctx, cache_key)

    # ── AUTO-SAVE MAIN CACHE TO DISK ──────────────────────────────────────────
    _active_store = prompt_cache_store_override if prompt_cache_store_override is not None else PROMPT_CACHE
    _is_compact_save = (prompt_cache_store_override is PROMPT_CACHE_COMPACT)

    if _is_compact_save:
        # ── AUTO-SAVE EMBEDDED/COMPACT CACHE ──────────────────────────────
        # Guardar el estado del compact runner en EMBEDDED_CACHE_PERSIST_PATH.
        # Solo si: hay path configurado, no fue guardado aún (o hit rate bajo), y tokens suficientes.
        embedded_persist_path = Path(SETTINGS.embedded_cache_persist_path) if SETTINGS.embedded_cache_persist_path else None
        _emb_should_save = False
        _emb_save_reason = ""
        if embedded_persist_path and len(cache_key) > 1000:
            if not _DPC.embedded_cache_saved:
                _emb_should_save = True
                _emb_save_reason = f"first_save (hit={cache_hit_ratio:.0%})"
            elif cache_hit_ratio < 0.85:
                _emb_should_save = True
                _emb_save_reason = f"stale_cache (hit={cache_hit_ratio:.0%} < 85%)"
        if _emb_should_save:
            _DPC.embedded_cache_saved = True
            _terminal_status(
                "💾",
                f"Auto-save EMBEDDED: → {embedded_persist_path.name} | reason={_emb_save_reason} | tokens={len(cache_key)}",
            )
            threading.Thread(
                target=_warmup_save_cache,
                args=(list(cache_key), prompt_cache, embedded_persist_path),
                kwargs={"prefix_hash": None},
                daemon=True,
                name="disk-embedded-cache-save",
            ).start()
    else:
        # ── AUTO-SAVE MAIN CACHE TO DISK ──────────────────────────────────────────
        persist_path = Path(SETTINGS.cache_persist_path) if SETTINGS.cache_persist_path else None

        _should_save = False
        _save_reason = ""
        if persist_path and is_warmup_candidate and len(cache_key) > 5000:
            # Diagnóstico explícito — siempre visible cuando es un real startup
            _terminal_status(
                "🔍",
                f"Auto-save check: saved={_DPC.disk_cache_saved} | hit={cache_hit_ratio:.0%} | "
                f"tokens={len(cache_key)} | path={persist_path.name}",
            )
            if not _DPC.disk_cache_saved:
                # Primera vez: siempre guardar (sea buen o mal hit rate)
                _should_save = True
                _save_reason = f"first_save (hit={cache_hit_ratio:.0%})"
            elif cache_hit_ratio < 0.85:
                # Cache en disco desactualizado — sobreescribir con el real
                _should_save = True
                _save_reason = f"stale_cache (hit={cache_hit_ratio:.0%} < 85%)"

        if _should_save:
            _DPC.disk_cache_saved = True
            # ── FIX: Save PROMPT-ONLY tokens, not prompt+response ────────────
            # cache_key at this point = prompt_tokens + generated_tokens.
            # Saving generated_tokens to disk contaminates the warmup cache:
            # on restart, the KV states encode the previous response, causing
            # hallucinations on the first request. Strip response tokens.
            _prompt_only_key = cache_key[: -len(generated_tokens)] if generated_tokens else list(cache_key)
            _terminal_status("💾", f"Auto-save: capturing prompt-only cache → {persist_path.name} | reason={_save_reason} | tokens={len(_prompt_only_key)} (stripped {len(generated_tokens)} response tok)")
            # Trim prompt_cache to remove generated token KV states.
            # CRITICAL: if deepcopy or trim fails, ABORT the save entirely.
            # A contaminated cache (with response tokens baked in) causes tool
            # hallucinations on restart — far worse than a cold start.
            _save_cache = None
            if generated_tokens and can_trim_prompt_cache(prompt_cache):
                try:
                    import copy as _copy_mod
                    _save_cache = _copy_mod.deepcopy(prompt_cache)
                    _pre_trim_off = _kv_cache_offset(_save_cache)
                    trim_prompt_cache(_save_cache, len(generated_tokens))
                    _post_trim_off = _kv_cache_offset(_save_cache)
                    # Verify trim actually reduced the offset
                    if (_post_trim_off is not None and _pre_trim_off is not None
                            and _post_trim_off >= _pre_trim_off):
                        _terminal_status("⚠️", f"Auto-save ABORTED: trim did not reduce offset ({_pre_trim_off} → {_post_trim_off})")
                        _save_cache = None
                except Exception as _trim_err:
                    _terminal_status("⚠️", f"Auto-save ABORTED: deepcopy/trim failed ({_trim_err})")
                    _save_cache = None  # NEVER fallback to saving contaminated cache
            elif not generated_tokens:
                # No response tokens to strip — safe to save as-is
                _save_cache = prompt_cache
            else:
                # Hybrid cache (e.g., Qwen3.5: ArraysCache + KVCache):
                # can_trim_prompt_cache() requires ALL layers trimmable, but
                # ArraysCache (linear attention) is never trimmable.
                # Fix: deepcopy + trim only the trimmable layers (KVCache).
                # Non-trimmable layers (ArraysCache) have no .keys →
                # _warmup_save_cache skips them anyway.
                _any_trimmable = any(
                    hasattr(layer, 'is_trimmable') and layer.is_trimmable()
                    for layer in prompt_cache
                )
                if _any_trimmable:
                    try:
                        import copy as _copy_mod
                        _save_cache = _copy_mod.deepcopy(prompt_cache)
                        # Read offset from TRIMMABLE layers only (KVCache),
                        # not ArraysCache whose internal shape[2] is static.
                        _pre_trim_off = None
                        for layer in _save_cache:
                            if hasattr(layer, 'is_trimmable') and layer.is_trimmable() and hasattr(layer, 'offset'):
                                _pre_trim_off = int(layer.offset)
                                break
                        _n_trimmed = 0
                        for layer in _save_cache:
                            if hasattr(layer, 'is_trimmable') and layer.is_trimmable() and hasattr(layer, 'trim'):
                                layer.trim(len(generated_tokens))
                                _n_trimmed += 1
                        _post_trim_off = None
                        for layer in _save_cache:
                            if hasattr(layer, 'is_trimmable') and layer.is_trimmable() and hasattr(layer, 'offset'):
                                _post_trim_off = int(layer.offset)
                                break
                        if (_post_trim_off is not None and _pre_trim_off is not None
                                and _post_trim_off >= _pre_trim_off):
                            _terminal_status("⚠️", f"Auto-save ABORTED: per-layer trim did not reduce offset ({_pre_trim_off} → {_post_trim_off})")
                            _save_cache = None
                        else:
                            _terminal_status("💾", f"Auto-save: per-layer trim OK | trimmed={_n_trimmed}/{len(_save_cache)} layers | offset {_pre_trim_off} → {_post_trim_off}")
                            # FIX-31 v7 + DPC: Update frozen cache with this CLEAN
                            # trimmed copy so pollution restore has full prompt
                            # coverage (not just seed tokens).
                            _DPC.frozen_cache = _save_cache
                            _DPC.frozen_tokens = list(_prompt_only_key)
                            _terminal_status("🧊", f"DPC: frozen cache updated | {len(_save_cache)} layers | {len(_prompt_only_key)} tokens")
                    except Exception as _trim_err:
                        _terminal_status("⚠️", f"Auto-save ABORTED: per-layer trim failed ({_trim_err})")
                        _save_cache = None
                else:
                    _terminal_status("⚠️", f"Auto-save SKIPPED: no trimmable layers ({type(prompt_cache[0]).__name__ if prompt_cache else 'empty'})")
                    _save_cache = None
            if _save_cache is not None:
                # DPC: compute prefix hash from the prompt-only tokens for auto-healing
                _prefix_hash = _wm.compute_prefix_hash(_prompt_only_key)
                threading.Thread(
                    target=_warmup_save_cache,
                    args=(_prompt_only_key, _save_cache, persist_path),
                    kwargs={"prefix_hash": _prefix_hash},
                    daemon=True,
                    name="dpc-cache-save",
                ).start()
            else:
                _DPC.disk_cache_saved = False  # Allow retry on next request




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
    except (IndexError, TypeError, AttributeError):
        pass
    return None


def _assert_cache_key_safety(
    original_prompt: str,
    cache_key_prompt: str,
    context: str = "",
) -> bool:
    """
    Safety invariant check for the dual-pipeline architecture.
    Asserts that the cache key is not dramatically shorter than the original prompt,
    which would indicate over-matching normalization silently deleting content.

    Only called when CACHE_NORM_SAFETY_CHECK=true (off by default — no latency impact).

    Returns True if invariant holds.  On violation: logs an error and returns False.
    The caller must fall back to using original_prompt as the cache key.
    """
    if not isinstance(original_prompt, str) or not isinstance(cache_key_prompt, str):
        return True
    orig_len = len(original_prompt)
    if orig_len == 0:
        return True
    key_len = len(cache_key_prompt)
    ratio = key_len / orig_len
    if ratio < 0.90:
        _terminal_status(
            "❌",
            f"[CACHE-SAFETY] cache_key is {ratio:.1%} of original prompt "
            f"({key_len} vs {orig_len} chars) — normalization over-matched. "
            f"Falling back to original prompt as cache key. context={context}",
        )
        return False
    return True


def _scrub_cache_key(prompt: str) -> str:
    """
    Post-render scrub of the CACHE KEY ONLY — never called on the model input.

    KRIPPER FIX-13: Length-Preserving Scrubbing.
    Instead of short sentinels, we mask volatile text with '0's of identical length
    to keep character/token offsets stable. This prevents 'rest_tokens inflation'.
    """
    if not isinstance(prompt, str):
        return prompt
    if not SETTINGS.cache_canonicalize_tool_context:
        return prompt

    def _mask(match):
        """Helper to replace match with a string of same length (0-padding)."""
        return "0" * len(match.group(0))

    # Scrub timestamp and Claude Code telemetry.
    normalized = CACHE_TIME_PATTERN.sub(_mask, prompt)
    normalized = CACHE_TIME_COLON_PATTERN.sub(_mask, normalized)
    normalized = CACHE_CCH_PATTERN.sub(_mask, normalized)
    normalized = CACHE_BILLING_HEADER_PATTERN.sub(_mask, normalized)
    normalized = CACHE_SYSTEM_REMINDER_PATTERN.sub(_mask, normalized)
    normalized = CACHE_SKILLS_BLOCK_PATTERN.sub(_mask, normalized)
    normalized = CACHE_RUNTIME_LINE_PATTERN.sub(_mask, normalized)
    return normalized


# _normalize_prompt_for_cache alias removed (Phase 7 Fix 4, 2026-03-14).
# No call sites remain after Phase 6 cleanup.  Use _scrub_cache_key() directly.


# --- DUAL-PIPELINE: message-struct level canonicalization ---
# Structural anchors for the OpenClaw "Inbound Context (trusted metadata)" block.
# These constants are intentionally plain strings — no regex — so there is no
# risk of accidental over-matching on user content.
_INBOUND_CONTEXT_HEADER = (
    "## Group Chat Context\n## Inbound Context (trusted metadata)\n"
)
_INBOUND_CONTEXT_FENCE_OPEN = "```json"
_INBOUND_CONTEXT_FENCE_CLOSE = "```"
# Stable sentinel that replaces the ENTIRE Group Chat Context / Inbound Context
# block (header + description lines + JSON fence) in the canonical cache key.
# Using a single sentinel for the whole block — instead of replacing only the
# JSON payload — ensures canonical form is IDENTICAL whether or not OpenClaw
# included the block in a given turn.  OpenClaw sometimes omits the block
# (confirmed in Phase 8 session logs: 44059 chars in T2, 43478 chars in T7),
# which caused the entire KV cache to diverge at the omission point and produced
# a 50.4% cache hit instead of the expected ~98%.
_INBOUND_CONTEXT_STABLE_SECTION = "__STABLE_INBOUND_CONTEXT_SECTION__"
# Anchor used for the "block absent" injection path.
_PROJECT_CONTEXT_ANCHOR = "\n# Project Context"


def _canonicalize_inbound_context_block(content: str) -> str:
    """
    Canonicalize the OpenClaw "Group Chat Context / Inbound Context" section.

    Two cases are handled so the canonical form is IDENTICAL regardless of whether
    OpenClaw included the block in a given turn (Phase 8 finding: the block is
    sometimes present, sometimes absent, causing a 50.4% VLM cache hit instead of
    the expected ~98% when the block is omitted in one turn but was present in the
    turn whose KV state is cached).

    Case A — block IS present:
        Replace the ENTIRE block (from the '## Group Chat Context' line through
        the closing ``` fence) with _INBOUND_CONTEXT_STABLE_SECTION.
        Canonical: '...<before>\\n__STABLE_INBOUND_CONTEXT_SECTION__\\n# Project Context...'

    Case B — block is ABSENT but '\\n# Project Context' anchor exists:
        Insert _INBOUND_CONTEXT_STABLE_SECTION immediately before the anchor.
        Canonical: same as Case A.

    If neither condition is met, the content is returned unchanged (safe no-op).

    Safety guarantees:
    - Pure string operations, no regex, no DOTALL.
    - Only touches the bounded region between the header and closing ```.
    - Stops scanning after 10 description lines (cannot cross section boundaries).
    - Idempotent: already-stable content is returned unchanged in one fast check.
    """
    # Fast idempotency check.
    if _INBOUND_CONTEXT_STABLE_SECTION in content:
        return content

    header_pos = content.find(_INBOUND_CONTEXT_HEADER)

    if header_pos != -1:
        # --- Case A: block present ---
        # Find the start of the '## Group Chat Context' line.
        # header_pos already points there since _INBOUND_CONTEXT_HEADER begins with it.
        # Walk forward (up to 10 lines) to find the opening ```json fence.
        search_start = header_pos + len(_INBOUND_CONTEXT_HEADER)
        fence_open_pos = -1
        cursor = search_start
        for _ in range(10):
            line_end = content.find("\n", cursor)
            if line_end == -1:
                break
            line = content[cursor:line_end]
            if line.startswith(_INBOUND_CONTEXT_FENCE_OPEN):
                fence_open_pos = cursor
                break
            # Stop if a new section header is encountered — prevents over-scanning.
            if line.startswith("## ") or line.startswith("# "):
                break
            cursor = line_end + 1

        if fence_open_pos == -1:
            # JSON fence not found within 10 lines — leave unchanged (safe).
            return content

        # Find the closing ``` fence on its own line.
        fence_line_end = content.find("\n", fence_open_pos)
        if fence_line_end == -1:
            return content
        json_body_start = fence_line_end + 1
        fence_close_pos = content.find(
            "\n" + _INBOUND_CONTEXT_FENCE_CLOSE, json_body_start
        )
        if fence_close_pos == -1:
            return content
        # fence_close_pos + 1 = first ` of the closing fence.
        # We advance past the full closing fence line (```\n) to get the remainder.
        fence_end = fence_close_pos + 1 + len(_INBOUND_CONTEXT_FENCE_CLOSE)
        # Skip the trailing newline if present.
        if fence_end < len(content) and content[fence_end] == "\n":
            fence_end += 1

        # Determine prefix boundary: keep the content immediately before the block.
        # header_pos may be preceded by a newline we want to preserve.
        block_start = header_pos

        return (
            content[:block_start]
            + _INBOUND_CONTEXT_STABLE_SECTION
            + "\n"
            + content[fence_end:]
        )

    else:
        # --- Case B: block absent but # Project Context anchor present ---
        # OpenClaw sometimes omits the Inbound Context section entirely.
        # Insert the stable sentinel at the same relative position so the cache key
        # matches turns where the block was present (Case A canonical form).
        anchor_pos = content.find(_PROJECT_CONTEXT_ANCHOR)
        if anchor_pos == -1:
            return content  # No recognisable anchor — safe no-op.
        # Insert stable sentinel on its own line immediately before the anchor.
        # _PROJECT_CONTEXT_ANCHOR starts with '\n', so after anchor_pos we have
        # '\n# Project Context'. We insert before that newline.
        return (
            content[:anchor_pos]
            + "\n"
            + _INBOUND_CONTEXT_STABLE_SECTION
            + content[anchor_pos:]
        )


def _canonicalize_messages(
    messages: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Pre-render canonicalization: returns (original_messages, canonical_messages).

    - original_messages  : deep copy of the input, never modified.
      This is what the model always receives (fed to apply_chat_template for generation).
    - canonical_messages : deep copy with volatile-but-semantically-neutral fields
      replaced by stable tokens.  Used ONLY for cache key computation.

    Volatile fields normalised in the canonical copy (message-struct level only):
    1. "message_id" values inside JSON content strings → __STABLE_MSG_ID__
       (handles OpenClaw / Claude Code per-request IDs that change every turn)
    2. OpenClaw "Inbound Context (trusted metadata)" JSON block → __STABLE_INBOUND_META__
       (handled structurally via string anchors, no DOTALL regex)

    No other mutations.  In particular, tool results and <think> blocks are
    preserved verbatim — the model sees them intact.
    """
    if not SETTINGS.cache_canonicalize_tool_context:
        # Canonicalization disabled: both pipelines see the same content.
        original = copy.deepcopy(messages)
        return original, copy.deepcopy(original)

    original: List[Dict[str, Any]] = copy.deepcopy(messages)
    canonical: List[Dict[str, Any]] = copy.deepcopy(messages)

    for msg in canonical:
        content = msg.get("content", "")
        if isinstance(content, str):
            # 1. Stable message IDs.
            content = INBOUND_META_MESSAGE_ID_PATTERN.sub(
                r"\1__STABLE_MSG_ID__\2", content
            )
            # 2. Inbound Context block (system message — present or absent).
            content = _canonicalize_inbound_context_block(content)
            # 3. Sub-agent completion stats (volatile runtime duration in user messages).
            content = SUBAGENT_STATS_PATTERN.sub(r"\1__STABLE_RUNTIME__", content)
            msg["content"] = content
        elif isinstance(content, list):
            new_parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text = part.get("text", "")
                    text = INBOUND_META_MESSAGE_ID_PATTERN.sub(
                        r"\1__STABLE_MSG_ID__\2", text
                    )
                    text = _canonicalize_inbound_context_block(text)
                    text = SUBAGENT_STATS_PATTERN.sub(r"\1__STABLE_RUNTIME__", text)
                    new_parts.append({**part, "text": text})
                else:
                    new_parts.append(part)
            msg["content"] = new_parts

    return original, canonical


def _extract_session_context(
    body: Dict[str, Any], prompt_tokens: List[int]
) -> SessionContext:
    def _read_any_id(
        container: Optional[Dict[str, Any]], keys: List[str]
    ) -> Optional[str]:
        if not isinstance(container, dict):
            return None
        for key in keys:
            value = container.get(key)
            if value is None:
                continue
            if isinstance(value, str) and value.strip():
                return value.strip()
            if isinstance(value, (int, float)):
                return str(value)
        return None

    metadata = body.get("metadata")
    extra_body = body.get("extra_body")

    id_keys = [
        "session_id",
        "conversation_id",
        "thread_id",
        "chat_id",
        "conversation",
        "session",
    ]
    parent_keys = [
        "parent_session_id",
        "parent_id",
        "source_session_id",
        "origin_session_id",
    ]
    branch_keys = ["branch_id", "branch", "subsession_id", "subagent_id"]

    session_id = (
        _read_any_id(body, id_keys)
        or _read_any_id(metadata, id_keys)
        or _read_any_id(extra_body, id_keys)
    )
    parent_session_id = (
        _read_any_id(body, parent_keys)
        or _read_any_id(metadata, parent_keys)
        or _read_any_id(extra_body, parent_keys)
    )
    branch_id = (
        _read_any_id(body, branch_keys)
        or _read_any_id(metadata, branch_keys)
        or _read_any_id(extra_body, branch_keys)
    )

    source = "request"
    if not session_id:
        raw = ",".join(str(tok) for tok in prompt_tokens[:128])
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
        session_id = f"implicit-{digest}"
        source = "derived_from_prompt_prefix"

    if parent_session_id == session_id:
        parent_session_id = None

    return SessionContext(
        session_id=session_id,
        parent_session_id=parent_session_id,
        branch_id=branch_id,
        source=source,
    )


def _build_sampler(body):
    """
    Build a sampler with conservative anti-loop defaults.
    If mlx_lm in this environment does not support some kwargs, fall back safely.
    """
    temperature = body.get("temperature", SETTINGS.default_temperature)
    if not isinstance(temperature, (int, float)):
        temperature = 0.1
    if temperature < 0.01:
        temperature = 0

    # Anti-loop defaults can be overridden by request body.
    candidate_kwargs = {
        "temp": temperature,
        "top_p": body.get("top_p", SETTINGS.default_top_p),
        "top_k": body.get("top_k", SETTINGS.default_top_k),
        "min_p": body.get("min_p", SETTINGS.default_min_p),
        "repetition_penalty": body.get(
            "repetition_penalty", SETTINGS.default_repetition_penalty
        ),
        "repetition_context_size": body.get(
            "repetition_context_size", SETTINGS.default_repetition_context_size
        ),
    }

    # Some mlx_lm versions don't support all params. Drop unsupported keys progressively.
    kwargs = dict(candidate_kwargs)
    while True:
        try:
            return make_sampler(**kwargs), kwargs
        except TypeError as e:
            msg = str(e)
            removed = False
            for key in list(kwargs.keys()):
                # Typical error: unexpected keyword argument 'xyz'
                if f"'{key}'" in msg:
                    kwargs.pop(key, None)
                    removed = True
                    break
            if not removed:
                # Unknown failure mode, use minimal safe sampler.
                return make_sampler(temp=temperature), {"temp": temperature}


def _find_pids_listening_on_port(port: int) -> List[int]:
    try:
        result = subprocess.run(
            ["lsof", "-nP", "-t", f"-iTCP:{port}", "-sTCP:LISTEN"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return []
    if result.returncode not in (0, 1):
        return []
    pids: List[int] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            pid = int(line)
        except ValueError:
            continue
        if pid != os.getpid():
            pids.append(pid)
    return pids


def _stop_stale_litellm_on_proxy_port(port: int) -> None:
    pids = _find_pids_listening_on_port(port)
    if not pids:
        return
    stopped_any = False
    for pid in pids:
        cmdline = ""
        try:
            ps_result = subprocess.run(
                ["ps", "-p", str(pid), "-o", "command="],
                check=False,
                capture_output=True,
                text=True,
            )
            cmdline = ps_result.stdout.strip().lower()
        except Exception:
            pass
        if "litellm" not in cmdline:
            continue
        try:
            os.kill(pid, signal.SIGTERM)
            stopped_any = True
            time.sleep(0.2)
            try:
                os.kill(pid, 0)
            except OSError:
                continue
            os.kill(pid, signal.SIGKILL)
        except OSError:
            continue
    if stopped_any:
        _terminal_status(
            "♻️", f"Stopped stale LiteLLM process(es) on port {port} before restart."
        )


def start_litellm_proxy():
    global proxy_process, proxy_config_path
    _stop_stale_litellm_on_proxy_port(SETTINGS.proxy_port)
    _terminal_status("🌉", f"Launching LiteLLM Proxy on port {SETTINGS.proxy_port}...")

    # Use proxy config so unsupported OpenAI params (e.g. "store") are dropped.
    # request_timeout (seconds): allow long prefills (e.g. 75k tokens) so client retries
    # don't trigger a timeout death spiral; 20 min is generous for local MLX.
    config_yaml = f"""model_list:
  - model_name: {SETTINGS.proxy_model_id}
    litellm_params:
      model: {SETTINGS.proxy_model_id}
      api_base: http://127.0.0.1:{SETTINGS.mlx_port}/v1
      api_key: local
      timeout: 1200
      stream_timeout: 1200
      # Keep these request fields when drop_params=true so this server can map
      # OpenClaw/Claude reasoning intent into tokenizer enable_thinking.
      allowed_openai_params:
        - reasoning_effort
litellm_settings:
  drop_params: true
  request_timeout: 1200
  stream_timeout: 1200
"""
    fd, temp_path = tempfile.mkstemp(prefix="litellm-qwen-", suffix=".yaml")
    with os.fdopen(fd, "w") as config_file:
        config_file.write(config_yaml)
    proxy_config_path = temp_path

    litellm_cli = Path(sys.executable).with_name("litellm")
    if litellm_cli.exists():
        cmd = [
            str(litellm_cli),
            "--config",
            proxy_config_path,
            "--port",
            str(SETTINGS.proxy_port),
            "--host",
            "0.0.0.0",
        ]
    else:
        # Fallback path for environments where the CLI script is not next to Python.
        resolved = shutil.which("litellm")
        if resolved:
            cmd = [
                resolved,
                "--config",
                proxy_config_path,
                "--port",
                str(SETTINGS.proxy_port),
                "--host",
                "0.0.0.0",
            ]
        else:
            cmd = [
                sys.executable,
                "-m",
                "litellm",
                "--config",
                proxy_config_path,
                "--port",
                str(SETTINGS.proxy_port),
                "--host",
                "0.0.0.0",
            ]

    my_env = os.environ.copy()
    my_env["OPENAI_API_KEY"] = "local"
    proxy_process = subprocess.Popen(
        cmd, env=my_env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
    )
    time.sleep(SETTINGS.proxy_startup_wait_seconds)
    if proxy_process.poll() is not None:
        stderr_output = ""
        if proxy_process.stderr is not None:
            try:
                stderr_output = proxy_process.stderr.read().decode(
                    "utf-8", errors="replace"
                )
            except Exception:
                stderr_output = ""
        raise RuntimeError(
            "LiteLLM proxy failed to start. "
            + (
                f"stderr: {stderr_output.strip()}"
                if stderr_output
                else "No stderr captured."
            )
        )
    _terminal_status(
        "✅",
        f"LiteLLM Proxy ready at http://127.0.0.1:{SETTINGS.proxy_port} (model: {SETTINGS.proxy_model_id})",
    )


def cleanup():
    global proxy_config_path
    if proxy_process:
        _terminal_status("🧹", "Shutting down LiteLLM Proxy...")
        proxy_process.terminate()
        proxy_process.wait()
    if proxy_config_path:
        try:
            Path(proxy_config_path).unlink(missing_ok=True)
        except Exception:
            pass
    _terminal_status("👋", "MLX Server stopped.")


atexit.register(cleanup)

# Model loading: LM or VLM based on config.
model = None
tokenizer = None
processor = None
is_vlm = False
vlm_config = None

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
    except Exception:
        pass
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
    try:
        model, tokenizer = load(
            SETTINGS.model_path, tokenizer_config={"trust_remote_code": True}
        )
    except Exception as _offline_err:
        _terminal_status("⚠️", f"Offline load failed ({_offline_err}) — retrying with network...")
        os.environ.pop("HF_HUB_OFFLINE", None)
        model, tokenizer = load(
            SETTINGS.model_path, tokenizer_config={"trust_remote_code": True}
        )
    _terminal_status("✅", "Model loaded (mlx-lm).")
    _terminal_status("⚡", "Torch acceleration: N/A (text-only model).")
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
if FEATURE_RAG_ENRICHMENT:
    try:
        import rag_enricher as _rag_mod
        _rag_mod.RELEVANCE_THRESHOLD = FEATURE_RAG_RELEVANCE_THRESHOLD
        _rag_module = _rag_mod
        _rag_available = True
        _terminal_status(
            "🔍",
            f"RAG Enricher: ACTIVATED"
            f" | top_k={_rag_mod.TOP_K}"
            f" | relevance_threshold={FEATURE_RAG_RELEVANCE_THRESHOLD}"
            f" | embedding={_rag_mod.EMBED_MODEL_NAME} (MPS)"
        )
        # RAG index: bloqueante — SYSTEM READY aparece solo cuando el índice esté listo.
        if FEATURE_RAG_WORKSPACE_ROOT:
            try:
                _terminal_status("🔍", f"RAG: indexando workspace | root={FEATURE_RAG_WORKSPACE_ROOT}")
                _rag_mod.init(codebase_root=FEATURE_RAG_WORKSPACE_ROOT)
                _terminal_status("✅", "RAG: workspace index ready")
            except Exception as _idx_err:
                _terminal_status("⚠️", f"RAG: indexing failed ({_idx_err})")
        else:
            _terminal_status("ℹ️", "RAG: no RAG_WORKSPACE_ROOT set — enricher activo pero índice vacío")
    except ImportError as e:
        _terminal_status("⚠️", f"RAG Enricher: UNAVAILABLE (import failed: {e})")
    except Exception as e:
        _terminal_status("❌", f"RAG Enricher: FAILED to initialize ({e})")
else:
    _terminal_status("ℹ️", "RAG Enricher: DISABLED (FEATURE_RAG_ENRICHMENT=False)")

# --- Prompt Compressor ---
if FEATURE_COMPRESSOR:
    try:
        # Compressor uses the same rag_enricher module
        if _rag_module is not None:
            _compressor_module = _rag_module
        else:
            import rag_enricher as _comp_mod
            _compressor_module = _comp_mod
        _compressor_available = True
        _terminal_status(
            "🗜️",
            f"Compressor: ACTIVATED | threshold={FEATURE_COMPRESSION_THRESHOLD} tok | "
            f"guard={FEATURE_COMPRESSION_GUARD} msgs"
        )
        # Precarga de modelos auxiliares al startup — evita 20s de penalty en el primer request.
        # Los modelos se leen de ~/.cache/huggingface/ (local, sin internet).
        try:
            _terminal_status("🗜️", "Compressor: precargando reranker + LLMLingua (CPU)...")
            _compressor_module._load_reranker()
            _compressor_module._load_llmlingua()
            _terminal_status("✅", "Compressor: modelos auxiliares listos en CPU")
        except Exception as _preload_err:
            _terminal_status("⚠️", f"Compressor: precarga parcial ({_preload_err}) — cargará lazy en primer request")
    except ImportError as e:
        _terminal_status("⚠️", f"Compressor: UNAVAILABLE (import failed: {e})")
    except Exception as e:
        _terminal_status("❌", f"Compressor: FAILED to initialize ({e})")
else:
    _terminal_status("ℹ️", "Compressor: DISABLED (FEATURE_COMPRESSOR=False)")

# --- Emergency Compressor standalone init ---
# When FEATURE_COMPRESSOR is disabled but EMERGENCY_CONTENT_COMPRESS is enabled,
# we still need _compressor_module loaded with LLMLingua for emergency compression.
# This only loads the LLMLingua model (CPU BERT, ~200MB) — not the full reranker.
if not FEATURE_COMPRESSOR and FEATURE_EMERGENCY_COMPRESS and _compressor_module is None:
    try:
        if _rag_module is not None:
            _compressor_module = _rag_module
        else:
            import rag_enricher as _comp_mod
            _compressor_module = _comp_mod
        try:
            _terminal_status("🗜️", "Emergency Compressor: precargando LLMLingua (CPU)...")
            _compressor_module._load_llmlingua()
            _terminal_status("✅", "Emergency Compressor: LLMLingua listo en CPU (standalone)")
        except Exception as _preload_err:
            _terminal_status("⚠️", f"Emergency Compressor: precarga parcial ({_preload_err}) — cargará lazy")
    except ImportError as e:
        _terminal_status("⚠️", f"Emergency Compressor: UNAVAILABLE (import failed: {e})")
        _compressor_module = None
    except Exception as e:
        _terminal_status("❌", f"Emergency Compressor: FAILED to initialize ({e})")
        _compressor_module = None

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
_terminal_status(
    "🏁",
    f"Active features: [{', '.join(_active_features) if _active_features else 'BASE ONLY'}]"
)

# ══════════════════════════════════════════════════════════════════════════════
# STARTUP WARMUP + KV CACHE PERSISTENCE
# Reduces first-request cold-start from ~40s to <2s.
#
# Dynamic Prefix Capture (DPC) — replaces manual warmup_seed.txt:
#   - Boot: loads prefix hash from disk (instant)
#   - First request: boundary detection → hash → validate → load or cold-start
#   - Auto-capture: saves prefix KV + hash after cold start
#   - Self-healing: invalidates cache if SOUL/tools change
#
# CACHE_PERSIST_PATH  — path to save/load KV state between restarts.
#                       E.g.: logs/warmup_cache.safetensors
#                       Eliminates 40s prefill cost across server restarts.
# ══════════════════════════════════════════════════════════════════════════════

# DPC shared state — replaces _WARMUP_SEED_HASH, _WARMUP_CACHE_FROZEN, etc.
_DPC = _wm.DPCState()
_WARMUP_DONE = _DPC.warmup_done  # Alias for backward compat (warmup gate, sidecar wait)

# Post-reaper reload signal: set by reaper when it prunes entries,
# cleared by request handler after launching reload thread.
# Separate from _WARMUP_DONE to avoid impacting sidecar or DPCState.
_CACHE_REAPED = threading.Event()
_CACHE_RELOAD_LOCK = threading.Lock()  # prevents double-reload on concurrent requests


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


def _warmup_load_cache(path: Path) -> Tuple[Optional[List[int]], Optional[Any]]:
    """Thin wrapper → warmup_manager.load_cache(). Backward compat for embedded cache load."""
    return _wm.load_cache(path, model, SETTINGS.max_kv_size, is_vlm=is_vlm, log_fn=_terminal_status)


def _run_startup_warmup() -> None:
    """
    Background startup — Dynamic Prefix Capture (DPC).

    No seed file needed. Boot sequence:
      1. Load prefix hash from disk (instant)
      2. Load EMBEDDED cache → PROMPT_CACHE_COMPACT
      3. Load MAIN cache → PROMPT_CACHE (tentative, validated on first request)
      4. Signal warmup_done

    First real request handles validation + cold start if needed.
    """
    _wm.run_startup(
        state=_DPC,
        cache_persist_path=SETTINGS.cache_persist_path,
        embedded_cache_persist_path=SETTINGS.embedded_cache_persist_path,
        model=model,
        max_kv_size=SETTINGS.max_kv_size,
        is_vlm=is_vlm,
        prompt_cache_main=PROMPT_CACHE,
        prompt_cache_compact=PROMPT_CACHE_COMPACT,
        prompt_cache_lock=prompt_cache_lock,
        model_path=SETTINGS.model_path,
        log_fn=_terminal_status,
    )


# ── Launch DPC daemon thread ────────────────────────────────────────────────
if SETTINGS.cache_persist_path:
    _warmup_thread = threading.Thread(
        target=_run_startup_warmup, daemon=True, name="dpc-startup"
    )
    _warmup_thread.start()
    _terminal_status(
        "🔥",
        f"DPC: thread launched | "
        f"persist={'✓ ' + Path(SETTINGS.cache_persist_path).name if SETTINGS.cache_persist_path else 'none'}",
    )
else:
    _terminal_status(
        "ℹ️",
        "DPC: DISABLED — set CACHE_PERSIST_PATH to enable",
    )
    _WARMUP_DONE.set()

# ── Background cache reaper: prune expired KV entries even when idle ─────────
# Without this, expired cache entries hold Metal GPU buffers until the next
# request triggers prune_expired(). This causes memory pressure to stay high
# (yellow in Activity Monitor) even when no sessions are active.
#
# Key insight: mx.clear_cache() releases MLX's internal buffer pool back to
# Metal, but Metal's allocator keeps GPU memory mapped until macOS forces
# eviction at ~98% pressure. Setting mx.metal.set_cache_limit(0) tells Metal
# to NOT hoard freed buffers, releasing them to the OS immediately.
_CACHE_REAPER_INTERVAL_SECONDS = _env_int("CACHE_REAPER_INTERVAL_SECONDS", 60)

# Tell Metal to release freed GPU buffers above the model footprint.
# set_cache_limit(0) is too aggressive (causes alloc/dealloc thrashing during generation).
# ~6GB keeps model weights + warmup cached, but releases KV session buffers after use.
_METAL_CACHE_LIMIT_BYTES = int(6.5 * (1024 ** 3))  # 6.5 GB ≈ model + warmup
try:
    # Prefer non-deprecated API (mx.set_cache_limit), fallback to mx.metal.set_cache_limit
    _set_cache_limit = getattr(mx, 'set_cache_limit', None) or getattr(mx.metal, 'set_cache_limit', None)
    if _set_cache_limit:
        _set_cache_limit(_METAL_CACHE_LIMIT_BYTES)
        _terminal_status("🧹", f"Metal cache limit: {_METAL_CACHE_LIMIT_BYTES / 1e9:.1f}GB (model + warmup)")
except Exception:
    pass


def _cache_reaper_loop():
    """Periodically prune expired cache entries and release Metal memory."""
    _get_active = getattr(mx, 'get_active_memory', None) or getattr(mx.metal, 'get_active_memory', None)
    _get_cache = getattr(mx, 'get_cache_memory', None) or getattr(mx.metal, 'get_cache_memory', None)
    _get_peak = getattr(mx, 'get_peak_memory', None) or getattr(mx.metal, 'get_peak_memory', None)

    def _mem_snapshot(label):
        pass  # Telemetry disabled

    while True:
        time.sleep(_CACHE_REAPER_INTERVAL_SECONDS)
        try:
            pruned = 0
            with prompt_cache_lock:
                before_main = len(PROMPT_CACHE._entries)
                before_compact = len(PROMPT_CACHE_COMPACT._entries)

                # [1/4] Before any pruning
                if before_main > 0 or before_compact > 0:
                    _mem_snapshot(f"1/4 before prune | entries_main={before_main} entries_compact={before_compact}")

                # Prune with telemetry enabled
                for k in [k for k, v in PROMPT_CACHE._entries.items() if PROMPT_CACHE._is_expired(v)]:
                    PROMPT_CACHE._delete(k[0], k[1], _reaper_telemetry=True)
                    pruned += 1
                for k in [k for k, v in PROMPT_CACHE_COMPACT._entries.items() if PROMPT_CACHE_COMPACT._is_expired(v)]:
                    PROMPT_CACHE_COMPACT._delete(k[0], k[1], _reaper_telemetry=True)
                    pruned += 1

            if pruned > 0:
                # [3/4] After gc.collect
                import gc; gc.collect()
                _mem_snapshot("3/4 after gc.collect")

                # [4/4] After mx.clear_cache
                mx.clear_cache()
                _mem_snapshot("4/4 after mx.clear_cache")

                _terminal_status(
                    "🧹",
                    f"CACHE REAPER: pruned {pruned} expired entries | "
                    f"active={_get_active()/1e9:.2f}GB | "
                    f"cache={_get_cache()/1e9:.2f}GB",
                )
                # Signal next request to reload cache from disk (post-reaper warm-up).
                if SETTINGS.cache_persist_path:
                    _CACHE_REAPED.set()
                    _terminal_status("🧹", "CACHE REAPER: signaled _CACHE_REAPED → next request will reload from disk")
        except Exception:
            pass  # Never crash the reaper


_reaper_thread = threading.Thread(
    target=_cache_reaper_loop, daemon=True, name="cache-reaper"
)
_reaper_thread.start()
_terminal_status("🧹", f"Cache reaper: active (every {_CACHE_REAPER_INTERVAL_SECONDS}s)")


# ══════════════════════════════════════════════════════════════════════════════


def _stream_generate_kwargs(prompt_tokens, max_tokens, sampler, prompt_cache):
    kwargs = {
        "model": model,
        "tokenizer": tokenizer,
        "prompt": prompt_tokens,
        "max_tokens": max_tokens,
        "sampler": sampler,
        "prompt_cache": prompt_cache,
        "max_kv_size": SETTINGS.max_kv_size,
        "kv_group_size": SETTINGS.kv_group_size,
    }
    if SETTINGS.kv_bits is not None:
        kwargs["kv_bits"] = SETTINGS.kv_bits
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
        for resp in stream_generate(
            **_stream_generate_kwargs(rest_tokens, max_tokens, sampler, prompt_cache)
        ):
            yield resp


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

class SidecarHandler(BaseHTTPRequestHandler):
    """Lightweight handler for non-OpenClaw queries sharing the loaded model."""

    def log_message(self, format, *args):
        return  # Suppress default HTTP logging
#Cambio hecho por Opus para evitar error (inofensivo) en sidecar en la pantalla de log
    def handle_one_request(self):
        """Suppress ConnectionResetError from health probes disconnecting early."""
        try:
            super().handle_one_request()
        except ConnectionResetError:
            pass  # Cosmetic: client closed before we read the request

    def do_GET(self):
        if self.path.rstrip("/") in ("/v1/models", "/models"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "object": "list",
                "data": [{
                    "id": SETTINGS.proxy_model_id,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "mlx-sidecar",
                }]
            }).encode("utf-8"))
            return
        # Health check
        if self.path.rstrip("/") in ("/health", "/"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "status": "ok",
                "model": SETTINGS.model_path,
                "sidecar": True,
                "rag_enabled": SETTINGS.sidecar_enable_rag and _rag_available,
            }).encode("utf-8"))
            return
        self.send_error(404, "Not Found")

    def do_POST(self):
        if self.path.rstrip("/") not in ("/v1/chat/completions", "/chat/completions"):
            self.send_error(404, "Not Found")
            return

        try:
            content_length = int(self.headers["Content-Length"])
            body = json.loads(self.rfile.read(content_length).decode("utf-8"))
        except Exception:
            self.send_error(400, "Bad Request")
            return

        request_id = uuid.uuid4().hex[:12]
        messages = body.get("messages", [])
        is_streaming = body.get("stream", False)
        max_tokens = min(
            body.get("max_tokens", SETTINGS.sidecar_max_tokens),
            SETTINGS.sidecar_max_tokens,
        )

        if not messages:
            self.send_error(400, "No messages provided")
            return

        _terminal_status(
            "🛸", f"Sidecar request {request_id} | "
            f"msgs={len(messages)} | stream={is_streaming} | max_tokens={max_tokens}",
            indent=1,
        )

        # ── RAG ENRICHMENT (optional) ──────────────────────────────────────
        if SETTINGS.sidecar_enable_rag and _rag_available and _rag_module is not None:
            try:
                rag_t0 = time.time()
                messages = _rag_module.enrich_messages(
                    messages, relevance_threshold=SETTINGS.sidecar_rag_threshold
                )
                rag_ms = (time.time() - rag_t0) * 1000
                _terminal_status(
                    "🔍", f"Sidecar RAG: enriched in {rag_ms:.0f}ms",
                    indent=2,
                )
            except Exception as e:
                _terminal_status(
                    "⚠️", f"Sidecar RAG: enrichment failed ({e}) — proceeding without",
                    indent=2,
                )

        # ── TOKENIZE ───────────────────────────────────────────────────────
        try:
            # Hoist system messages to position 0 (Qwen3.5 requirement)
            system_parts = []
            non_system = []
            for m in messages:
                if m.get("role") == "system":
                    c = m.get("content", "")
                    if isinstance(c, str) and c.strip():
                        system_parts.append(c.strip())
                else:
                    non_system.append(m)
            if system_parts:
                messages = [{"role": "system", "content": "\n\n".join(system_parts)}] + non_system

            prepared = _prepare_messages_for_template(messages)

            # Resolve enable_thinking: support both top-level (legacy) and
            # chat_template_kwargs (Qwen3.5 official API)
            _ct_kwargs = body.get("chat_template_kwargs", {})
            _enable_thinking = _ct_kwargs.get(
                "enable_thinking",
                body.get("enable_thinking", SETTINGS.default_thinking),
            )

            if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
                prompt = tokenizer.apply_chat_template(
                    prepared,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=_enable_thinking,
                )
            else:
                prompt = prepared[-1]["content"] if prepared else ""

            prompt_tokens = _tokenize_prompt(prompt)
        except Exception as e:
            _terminal_status("❌", f"Sidecar tokenization failed: {e}")
            self.send_error(500, f"Tokenization error: {e}")
            return

        _terminal_status(
            "📊", f"Sidecar {request_id}: {len(prompt_tokens)} prompt tokens",
            indent=2,
        )

        # ── BUILD SAMPLER ──────────────────────────────────────────────────
        sampler, _ = _build_sampler(body)

        # ── GENERATE (under model_lock) ────────────────────────────────────
        acquired = False
        generation_started_at = None
        queue_started_at = time.time()

        try:
            model_lock.acquire(blocking=True)
            acquired = True
            generation_started_at = time.time()
            wait_seconds = generation_started_at - queue_started_at

            if wait_seconds > 1.0:
                _terminal_status(
                    "⏳", f"Sidecar {request_id}: waited {wait_seconds:.1f}s for model_lock",
                    indent=2,
                )

            # Ephemeral KV cache — created fresh, discarded after generation
            ephemeral_cache = make_prompt_cache(model)

            if is_streaming:
                self._handle_streaming(
                    request_id, prompt_tokens, max_tokens,
                    sampler, ephemeral_cache, generation_started_at, prepared,
                )
            else:
                self._handle_non_streaming(
                    request_id, prompt_tokens, max_tokens,
                    sampler, ephemeral_cache, generation_started_at, prepared,
                    enable_thinking=_enable_thinking,
                )

        except Exception as e:
            _terminal_status("❌", f"Sidecar {request_id}: generation error: {e}")
            try:
                self.send_error(500, str(e))
            except Exception:
                pass
        finally:
            if acquired:
                # Free ephemeral KV cache immediately
                mx.clear_cache()
                import gc; gc.collect()
                if generation_started_at:
                    held_ms = (time.time() - generation_started_at) * 1000
                    _terminal_status(
                        "🔓", f"Sidecar {request_id}: model_lock released ({held_ms:.0f}ms)",
                        indent=2,
                    )
                model_lock.release()

    def _handle_non_streaming(self, request_id, prompt_tokens, max_tokens,
                               sampler, ephemeral_cache, gen_start, prepared_messages_ref,
                               enable_thinking=True):
        """Generate complete response and send as single JSON."""
        full_text = ""
        token_count = 0

        for resp in stream_generate(
            **_stream_generate_kwargs(prompt_tokens, max_tokens, sampler, ephemeral_cache)
        ):
            full_text += resp.text
            token_count += 1

        # ── STRIP THINKING from response ──────────────────────────────────
        # Only apply think-stripping when thinking was enabled.
        # With enable_thinking=False, the model generates content directly
        # (no <think> tags), so stripping would cause a false-positive retry.
        if not enable_thinking:
            # No-think mode: content is the direct output, no stripping needed
            pass
        elif "</think>" in full_text:
            # Normal case: take everything after </think>
            full_text = full_text.split("</think>", 1)[1].strip()
        elif "<think>" in full_text:
            # Explicit <think> tag without closing — truncated
            full_text = full_text[:full_text.index("<think>")].strip()
        else:
            # No </think> found — model spent all max_tokens thinking.
            # Retry WITHOUT thinking to get a direct answer.
            _terminal_status(
                "🔄", f"Sidecar {request_id}: thinking truncated at {token_count} tokens — "
                f"retrying with enable_thinking=False",
                indent=2,
            )
            # Re-tokenize with thinking disabled
            retry_prompt = tokenizer.apply_chat_template(
                prepared_messages_ref,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            retry_tokens = _tokenize_prompt(retry_prompt)
            retry_cache = make_prompt_cache(model)
            full_text = ""
            token_count = 0
            for resp in stream_generate(
                **_stream_generate_kwargs(retry_tokens, max_tokens, sampler, retry_cache)
            ):
                full_text += resp.text
                token_count += 1
            # Final strip just in case
            full_text = _strip_thinking_from_content(full_text)

        elapsed = time.time() - gen_start
        tps = token_count / elapsed if elapsed > 0 else 0

        response = {
            "id": f"chatcmpl-sc-{request_id}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": SETTINGS.proxy_model_id,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": full_text},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": len(prompt_tokens),
                "completion_tokens": token_count,
                "total_tokens": len(prompt_tokens) + token_count,
            },
        }

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(response).encode("utf-8"))

        _terminal_status(
            "✅", f"Sidecar {request_id}: {token_count} tokens | "
            f"{elapsed:.1f}s | {tps:.1f} tok/s",
            indent=1,
        )

    def _handle_streaming(self, request_id, prompt_tokens, max_tokens,
                           sampler, ephemeral_cache, gen_start, prepared_messages_ref):
        """Stream SSE chunks as they're generated."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        token_count = 0
        full_text = ""
        in_thinking = False

        for resp in stream_generate(
            **_stream_generate_kwargs(prompt_tokens, max_tokens, sampler, ephemeral_cache)
        ):
            token_count += 1
            text = resp.text

            # Simple think-tag stripping for streaming
            if "<think>" in text:
                in_thinking = True
                continue
            if "</think>" in text:
                in_thinking = False
                continue
            if in_thinking:
                continue

            full_text += text

            chunk = {
                "id": f"chatcmpl-sc-{request_id}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": SETTINGS.proxy_model_id,
                "choices": [{
                    "index": 0,
                    "delta": {"content": text},
                    "finish_reason": None,
                }],
            }
            try:
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode("utf-8"))
                self.wfile.flush()
            except BrokenPipeError:
                _terminal_status("⚠️", f"Sidecar {request_id}: client disconnected")
                return

        # Send final chunk
        final_chunk = {
            "id": f"chatcmpl-sc-{request_id}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": SETTINGS.proxy_model_id,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }],
        }
        try:
            self.wfile.write(f"data: {json.dumps(final_chunk)}\n\n".encode("utf-8"))
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except BrokenPipeError:
            pass

        elapsed = time.time() - gen_start
        tps = token_count / elapsed if elapsed > 0 else 0
        _terminal_status(
            "✅", f"Sidecar {request_id}: {token_count} tokens streamed | "
            f"{elapsed:.1f}s | {tps:.1f} tok/s",
            indent=1,
        )


class APIHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Keep terminal output focused on custom request lifecycle lines.
        return

    def do_GET(self):
        if self.path.rstrip("/") in ("/v1/models", "/models"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(
                json.dumps(
                    {
                        "object": "list",
                        "data": [
                            {
                                "id": SETTINGS.proxy_model_id,
                                "object": "model",
                                "created": int(time.time()),
                                "owned_by": "mlx",
                            }
                        ],
                    }
                ).encode("utf-8")
            )
            return
        self.send_error(404, "Not Found")

    def do_POST(self):
        if self.path.rstrip("/") in ("/v1/models", "/models"):
            return self.do_GET()

        if self.path.rstrip("/") not in ("/v1/chat/completions", "/chat/completions"):
            try:
                self.send_error(404, "Not Found")
            except BrokenPipeError:
                pass
            return

        try:
            content_length = int(self.headers["Content-Length"])
            body = json.loads(self.rfile.read(content_length).decode("utf-8"))
        except Exception:
            self.send_error(400, "Bad Request")
            return

        # ── DPC GATE ────────────────────────────────────────────────────────
        # Si DPC está configurado y aún no terminó el boot, el primer request espera
        # hasta que el cache hash esté cargado. Para todos los requests siguientes
        # _WARMUP_DONE.is_set() == True → no hay overhead (operación O(1)).
        # Timeout de 120s por si el boot falla: el request procede igual (cold start).
        if not _WARMUP_DONE.is_set() and SETTINGS.cache_persist_path:
            _wg_t0 = time.time()
            _WARMUP_DONE.wait(timeout=120)
            _wg_elapsed = time.time() - _wg_t0
            if _wg_elapsed > 0.5:  # Solo loguear si realmente esperó
                with console_lock:
                    print(
                        f"  [WARMUP_GATE] {datetime.now().strftime('%H:%M:%S')} "
                        f"request queued {_wg_elapsed:.1f}s for warmup | "
                        f"cache_ready={_WARMUP_DONE.is_set()}",
                        flush=True,
                    )

        # ── POST-REAPER RELOAD GATE ─────────────────────────────────────────
        # If the cache reaper pruned entries while idle, reload from disk on the
        # first incoming request. Uses double-check locking to prevent concurrent
        # requests from launching multiple reload threads simultaneously.
        if _CACHE_REAPED.is_set() and SETTINGS.cache_persist_path:
            with _CACHE_RELOAD_LOCK:
                if _CACHE_REAPED.is_set():  # double-check inside lock
                    _CACHE_REAPED.clear()
                    _terminal_status("🔥", "DPC: post-reaper reload triggered | reloading from disk...")
                    _reload_done = threading.Event()
                    def _do_reload(_done=_reload_done):
                        _run_startup_warmup()
                        _done.set()
                    threading.Thread(target=_do_reload, daemon=True, name="dpc-post-reaper").start()
                    _reload_done.wait(timeout=120)
        # ── END POST-REAPER RELOAD GATE ─────────────────────────────────────

        # ── END DPC GATE ────────────────────────────────────────────────────

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
        tools = body.get("tools")

        # KRIPPER DUAL-SLOT: Detect OpenClaw's compact runner.
        # The compact runner sends ZERO tools and a system prompt with compaction keywords.
        # The old detector (len(tools) <= 2) was incorrect — compact sends 0 tools,
        # not 2 — and MAIN requests in conversational mode also may have 0 tools.
        # This detector is multi-signal and much more reliable.
        _raw_messages_for_detect = body.get("messages", [])
        _is_embedded_agent = _detect_compact_runner(
            messages=_raw_messages_for_detect,
            tools=tools,
        )
        if _is_embedded_agent and FEATURE_FULL_LOGGING:
            _pipeline_log("TOOLS", request_id,
                f"COMPACT_RUNNER: detected (system-prompt keywords + 0 tools) — routed to PROMPT_CACHE_COMPACT")

        reasoning_control = _extract_enable_thinking(body)
        enable_thinking = reasoning_control["enable_thinking"]

        # ── PIPELINE LOG: INBOUND ──────────────────────────────────────────
        _pipeline_t0 = time.time()
        _pipeline_timings = {}  # stage -> ms
        raw_messages_inbound = body.get("messages", [])

        # ── TOOL RESULT TRUNCATION (medida temporal hasta implementar plan §11) ──
        # Impide que web_fetch/pdf de documentos grandes vuelen la memoria del GPU.
        # Cuando §11 esté implementado (workspace efímero + embeddings), este bloque
        # puede eliminarse porque el modelo nunca recibirá el documento completo.
        _MAX_TOOL_RESULT_CHARS = int(os.environ.get("MAX_TOOL_RESULT_CHARS", "20000"))
        _tool_truncations = 0
        for _msg in raw_messages_inbound:
            if (_msg.get("role") or "").lower() == "tool":
                _tc = _msg.get("content", "")
                if isinstance(_tc, str) and len(_tc) > _MAX_TOOL_RESULT_CHARS:
                    _msg["content"] = (
                        _tc[:_MAX_TOOL_RESULT_CHARS]
                        + f"\n\n[TRUNCADO: el resultado de la herramienta excedió {_MAX_TOOL_RESULT_CHARS} caracteres. "
                        f"Implementar plan §11 (workspace efímero + embeddings) para eliminar este límite.]"
                    )
                    _tool_truncations += 1
        if _tool_truncations > 0:
            _pipeline_log("INBOUND", request_id,
                f"tool_result_truncation={_tool_truncations} | limit={_MAX_TOOL_RESULT_CHARS} chars "
                f"| TEMP measure until plan §11 (ephemeral workspace)")


        if FEATURE_FULL_LOGGING:
            _roles = _count_roles(raw_messages_inbound)
            _roles_str = ", ".join(f"{r}:{c}" for r, c in sorted(_roles.items()))
            _est_tok = _estimate_token_count(raw_messages_inbound)
            _last_user = ""
            for _m in reversed(raw_messages_inbound):
                if (_m.get("role") or "").lower() == "user":
                    _c = _m.get("content", "")
                    if isinstance(_c, str):
                        _last_user = _c[:100].replace("\n", " ")
                    elif isinstance(_c, list):
                        for _part in _c:
                            if isinstance(_part, dict) and _part.get("type") == "text":
                                _last_user = (_part.get("text", "") or "")[:100].replace("\n", " ")
                                break
                    break
            _pipeline_log("INBOUND", request_id,
                f"messages={len(raw_messages_inbound)} | roles=[{_roles_str}] | "
                f"stream={body.get('stream', False)} | model={body.get('model', '?')} | "
                f"thinking={enable_thinking}")
            _pipeline_log("INBOUND", request_id,
                f"estimated_tokens={_est_tok} | tools={len(tools) if tools else 0} | "
                f"last_user_msg=\"{_last_user}\"")
            if FEATURE_LOG_PROMPTS:
                _pipeline_log("INBOUND", request_id, "messages dumped to disk",
                    data={"messages": raw_messages_inbound, "tools": tools,
                          "stream": body.get("stream", False), "model": body.get("model", "?"),
                          "enable_thinking": enable_thinking})
            # ── PIPELINE LOG: MSG_IN ──────────────────────────────────────────
            _agent_tag = "EMBEDDED" if _is_embedded_agent else "MAIN"
            _msg_in_content = None
            _msg_in_source = "user"
            for _m in reversed(raw_messages_inbound):
                _role = (_m.get("role") or "").lower()
                if _role == "tool":
                    _c = _m.get("content", "")
                    if isinstance(_c, str):
                        _msg_in_content = _c[:400].replace("\n", " ")
                    _msg_in_source = "tool_result"
                    break
                elif _role == "user":
                    _c = _m.get("content", "")
                    if isinstance(_c, str):
                        _msg_in_content = _c[:400].replace("\n", " ")
                    elif isinstance(_c, list):
                        for _part in _c:
                            if isinstance(_part, dict) and _part.get("type") == "text":
                                _msg_in_content = _part.get("text", "")[:400].replace("\n", " ")
                                break
                    _msg_in_source = "user"
                    break
            _pipeline_log("MSG_IN", request_id,
                f"[{_agent_tag}] src={_msg_in_source} | {_msg_in_content!r}")

        # ── PIPELINE LOG: TOOLS ────────────────────────────────────────────
        if FEATURE_FULL_LOGGING and FEATURE_LOG_TOOLS and tools:
            for _ti, _t in enumerate(tools):
                _fn = _t.get("function", {}) if isinstance(_t, dict) else {}
                _pipeline_log("TOOLS", request_id,
                    f"tool[{_ti}]: name={_fn.get('name', '?')} | "
                    f"params={list((_fn.get('parameters', {}).get('properties', {}) or {}).keys())}")
            _tool_results = _summarize_tool_results(raw_messages_inbound)
            if _tool_results:
                _pipeline_log("TOOLS", request_id,
                    f"tool_results_in_history={len(_tool_results)}")
                if len(_tool_results) > 0:
                    _last_tr = _tool_results[-1]
                    _pipeline_log("TOOLS", request_id,
                        f"tool_result[last]: name={_last_tr['name']} | content_len={_last_tr['content_len']}")

        vlm_pixel_values = None
        vlm_mask = None
        vlm_input_ids_raw = None
        vlm_kwargs = None

        if is_vlm:
            raw_messages = body.get("messages", [])
            # --- APPLY STATELESS HEALING ---
            healed_messages = _heal_messages(raw_messages)

            messages = _prepare_messages_for_vlm(healed_messages, tools=tools)
            images = _extract_images_from_messages(body.get("messages", []))

            with model_lock:
                vlm_input_ids_raw, vlm_pixel_values, vlm_mask, vlm_kwargs = (
                    _vlm_prompt_and_inputs(
                        processor,
                        vlm_config or {},
                        messages,
                        images,
                        tools=tools,
                        enable_thinking=enable_thinking,
                    )
                )
                _vlm_sync_before_generation(vlm_pixel_values, vlm_mask)

            if vlm_input_ids_raw is not None:
                if hasattr(vlm_input_ids_raw, "flatten"):
                    model_tokens = vlm_input_ids_raw.flatten().tolist()
                else:
                    model_tokens = list(vlm_input_ids_raw)
            else:
                model_tokens = []

            # --- VLM DUAL PIPELINE: canonical cache key (Phase 8) ---
            # The model always prefills from model_tokens (original messages, above).
            # For the cache lookup we apply the same canonicalization as the LM path:
            # _canonicalize_messages + _scrub_cache_key.  This ensures volatile fields
            # in the system message (Inbound Context block, message IDs, etc.) do not
            # cause cache divergence when OpenClaw changes them between turns.
            # We use the VLM tokenizer's text-only apply_chat_template (CPU, no GPU
            # required) so the canonical key is derived without a second model_lock
            # acquisition.  Falls back to model_tokens if tokenizer is unavailable.
            prompt_tokens = model_tokens  # safe fallback
            if SETTINGS.cache_canonicalize_tool_context:
                try:
                    _, canonical_msgs_vlm = _canonicalize_messages(healed_messages)
                    canon_prepared = _prepare_messages_for_vlm(
                        canonical_msgs_vlm, tools=tools
                    )
                    # Resolve the text-only tokenizer from the VLM processor.
                    _vlm_tok = None
                    if processor is not None:
                        if (
                            hasattr(processor, "tokenizer")
                            and processor.tokenizer is not None
                            and hasattr(processor.tokenizer, "apply_chat_template")
                            and getattr(processor.tokenizer, "chat_template", None)
                        ):
                            _vlm_tok = processor.tokenizer
                        elif hasattr(processor, "apply_chat_template") and getattr(
                            processor, "chat_template", None
                        ):
                            _vlm_tok = processor
                    if _vlm_tok is not None:
                        _canon_fmt = _vlm_tok.apply_chat_template(
                            canon_prepared,
                            tokenize=False,
                            add_generation_prompt=True,
                            tools=tools,
                            enable_thinking=enable_thinking,
                        )
                        _canon_fmt = _scrub_cache_key(str(_canon_fmt))
                        _canon_ids = _vlm_tok.encode(_canon_fmt)
                        if isinstance(_canon_ids, list) and _canon_ids:
                            prompt_tokens = _canon_ids
                except Exception:
                    pass  # Fall back to model_tokens — correct but no canonicalization

            prompt = ""

        else:
            raw_messages = body.get("messages", [])  # Reorderer disabled — personality regression

            # ── PIPELINE: PROMPT COMPRESSION ───────────────────────────────
            # KRIPPER FASE 3: Guard — never compress compact runner requests.
            # The compact runner sends the full session history intentionally;
            # compressing it would corrupt the summary output.
            if _compressor_available and FEATURE_COMPRESSOR and not _is_embedded_agent:
                _pre_compress_count = len(raw_messages)
                _pre_compress_est = _estimate_token_count(raw_messages)
                try:
                    # KRIPPER FASE 3: Use X-Session-ID header for semantic isolation.
                    # Falls back to body fields, then a stable hash derived from the
                    # system message content.  request_id is NEVER a valid fallback
                    # because it's unique per request → compression cache never hits.
                    _sys_msg_hash = None
                    for _m in raw_messages:
                        if _m.get("role") == "system":
                            _sc = _m.get("content", "")
                            if isinstance(_sc, str) and len(_sc) > 100:
                                import hashlib as _hl
                                _sys_msg_hash = "sys-" + _hl.md5(_sc[:2000].encode()).hexdigest()[:12]
                                break
                    _comp_session = (
                        self.headers.get("X-Session-ID")
                        or body.get("session_id")
                        or body.get("user")
                        or _sys_msg_hash
                        or request_id
                    )
                    raw_messages, _compress_ms, _compress_cache_hit = _compress_with_cache(
                        raw_messages=raw_messages,
                        comp_session=_comp_session,
                        compressor_module=_compressor_module,
                        threshold=FEATURE_COMPRESSION_THRESHOLD,
                        request_id=request_id,
                    )
                    _pipeline_timings["compress"] = _compress_ms
                    _post_compress_est = _estimate_token_count(raw_messages)
                    if FEATURE_FULL_LOGGING and FEATURE_LOG_COMPRESSION:
                        if _pre_compress_est <= FEATURE_COMPRESSION_THRESHOLD:
                            _pipeline_log("COMPRESS", request_id,
                                f"SKIPPED | {_pre_compress_est} tok < threshold {FEATURE_COMPRESSION_THRESHOLD}")
                        elif _compress_cache_hit:
                            _pipeline_log("COMPRESS", request_id,
                                f"CACHE_HIT | {_pre_compress_count} msgs → {len(raw_messages)} msgs | "
                                f"~{_pre_compress_est} tok → ~{_post_compress_est} tok | "
                                f"session={str(_comp_session)[:16]} | {_compress_ms:.0f}ms (saved ~5s)")
                        else:
                            _pipeline_log("COMPRESS", request_id,
                                f"ACTIVATED | {_pre_compress_count} msgs → {len(raw_messages)} msgs | "
                                f"~{_pre_compress_est} tok → ~{_post_compress_est} tok | "
                                f"session={str(_comp_session)[:16]} | {_compress_ms:.0f}ms")
                except Exception as _ce:
                    if FEATURE_FULL_LOGGING:
                        _pipeline_log("COMPRESS", request_id, f"ERROR: {_ce}")

            # ── PIPELINE: RAG ENRICHMENT ───────────────────────────────────
            # KRIPPER FASE 3: Guard — never enrich compact runner requests.
            # The compact runner reads the session as-is; RAG injection would
            # pollute the summary with irrelevant codebase chunks.
            # FASE 4: Also skip RAG for memory flush, summarization, and other
            # lightweight internal operations that don't need domain context.
            _skip_rag = _is_embedded_agent or _is_rag_bypass_request(raw_messages)
            _rag_meta = {"best_score": 999.0, "chunks_found": 0, "query_used": "", "search_ms": 0}
            if _rag_available and FEATURE_RAG_ENRICHMENT and not _skip_rag:
                _rag_t0 = time.time()
                try:
                    # Use metadata variant for cascade routing signals
                    if hasattr(_rag_module, 'enrich_messages_with_metadata'):
                        raw_messages, _rag_meta = _rag_module.enrich_messages_with_metadata(
                            raw_messages,
                        )
                    else:
                        raw_messages = _rag_module.enrich_messages(
                            raw_messages,
                        )
                    _rag_ms = (time.time() - _rag_t0) * 1000
                    _pipeline_timings["rag"] = _rag_ms
                    if FEATURE_FULL_LOGGING and FEATURE_LOG_RAG:
                        _post_rag_est = _estimate_token_count(raw_messages)
                        _pipeline_log("RAG", request_id,
                            f"ACTIVATED | enriched msgs={len(raw_messages)} | "
                            f"~{_post_rag_est} tok | {_rag_ms:.0f}ms | "
                            f"best_score={_rag_meta.get('best_score', '?'):.3f} | "
                            f"chunks={_rag_meta.get('chunks_found', 0)}")
                except Exception as _re:
                    if FEATURE_FULL_LOGGING:
                        _pipeline_log("RAG", request_id, f"ERROR: {_re}")
            elif _skip_rag and FEATURE_FULL_LOGGING and FEATURE_LOG_RAG:
                _reason = "compact_runner" if _is_embedded_agent else "rag_bypass(flush/summarize)"
                _pipeline_log("RAG", request_id, f"SKIPPED | reason={_reason}")

            # ── PIPELINE: CASCADE ROUTING ──────────────────────────────────
            # If RAG confidence is low AND cascade is enabled, forward to frontier.
            # Signals: RAG best_score > threshold, or explicit X-Force-Cascade header.
            _force_cascade = self.headers.get("X-Force-Cascade", "").lower() in ("true", "1", "yes")
            _block_cascade = self.headers.get("X-No-Cascade", "").lower() in ("true", "1", "yes")
            _cascade_triggered = False

            if (FEATURE_CASCADE and CASCADE_API_URL and CASCADE_API_KEY
                    and not _is_embedded_agent and not _skip_rag and not _block_cascade):
                _rag_best = _rag_meta.get("best_score", 999.0)
                _rag_chunks = _rag_meta.get("chunks_found", 0)
                _should_cascade = (
                    _force_cascade
                    or (_rag_chunks == 0 and _rag_best >= CASCADE_RAG_THRESHOLD)
                    or (_rag_best >= CASCADE_RAG_THRESHOLD)
                )
                if _should_cascade:
                    _cascade_triggered = True
                    _cascade_reason = "forced" if _force_cascade else f"rag_score={_rag_best:.3f}>={CASCADE_RAG_THRESHOLD}"
                    _pipeline_log("CASCADE", request_id,
                        f"ACTIVATED | reason={_cascade_reason} | "
                        f"rag_chunks={_rag_chunks} | best_score={_rag_best:.3f} | "
                        f"target={CASCADE_MODEL}@{CASCADE_API_URL[:60]}")
                    try:
                        _is_stream_request = body.get("stream", False)
                        _cascade_response = _cascade_forward_request(
                            body=body,
                            request_id=request_id,
                            handler=self,
                            is_streaming=_is_stream_request,
                        )
                        if _is_stream_request:
                            # Streaming: response already proxied via SSE
                            _pipeline_log("CASCADE", request_id,
                                f"COMPLETED (stream) | model={CASCADE_MODEL}")
                            return  # Done — SSE already sent to client
                        else:
                            # Non-streaming: serialize and send JSON
                            _resp_bytes = json.dumps(_cascade_response, ensure_ascii=False).encode("utf-8")
                            self.send_response(200)
                            self.send_header("Content-Type", "application/json")
                            self.send_header("X-Cascade-Active", "true")
                            self.send_header("X-Cascade-Reason", _cascade_reason)
                            self.send_header("X-Cascade-Model", CASCADE_MODEL)
                            self.send_header("Content-Length", str(len(_resp_bytes)))
                            self.end_headers()
                            self.wfile.write(_resp_bytes)
                            _pipeline_log("CASCADE", request_id,
                                f"COMPLETED | model={CASCADE_MODEL} | "
                                f"response_bytes={len(_resp_bytes)}")
                            return  # Done — skip local model entirely
                    except Exception as _ce:
                        _pipeline_log("CASCADE", request_id,
                            f"FAILED — falling back to local | error={_ce}")
                        _cascade_triggered = False  # Fallback to local generation

            # --- APPLY STATELESS HEALING ---
            _heal_t0 = time.time()
            healed_messages = _heal_messages(raw_messages)
            _heal_ms = (time.time() - _heal_t0) * 1000
            _pipeline_timings["heal"] = _heal_ms

            # ── PIPELINE LOG: HEALING ──────────────────────────────────────
            if FEATURE_FULL_LOGGING and FEATURE_LOG_HEALING:
                _healed_count = 0
                for _hi, (_orig, _healed) in enumerate(zip(raw_messages, healed_messages)):
                    if _orig.get("content") != _healed.get("content"):
                        _healed_count += 1
                        _pipeline_log("HEAL", request_id,
                            f"heal[{_hi}]: role={_healed.get('role')} | restored content")
                _pipeline_log("HEAL", request_id,
                    f"input={len(raw_messages)} msgs | healed={_healed_count} | "
                    f"store_size={len(HEALING_STORE)}/{MAX_HEALING_STORE} | {_heal_ms:.1f}ms")

            # ── PIPELINE: TOOL CALL LOOP BREAKER ─────────────────────────────
            # Detect and break infinite tool-call retry loops BEFORE canonicalization.
            # The stop instruction must reach the model (via original_messages) and
            # must also be reflected in the cache key (via canonical_messages).
            healed_messages, _loop_broken = _break_tool_call_loop(
                healed_messages, request_id
            )

            # --- DUAL PIPELINE: split model input from cache key at message-struct level ---
            # original_messages → rendered → model sees this (never normalized)
            # canonical_messages → rendered → scrubbed → cache lookup key only
            original_messages, canonical_messages = _canonicalize_messages(
                healed_messages
            )

            # --- Qwen3.5 STRICT RULE: system messages MUST be at the beginning ---
            # After compress/RAG/heal, system messages can be scattered.
            # Merge all system messages into one at position 0.
            # NOTE: This merges RAG system messages too, which buries them in msg[0].
            # See parking lot: RAG hoisting exception (audit 2026-04-26, 45/46 misplaced).
            # Deferred until personality regression testing is available.
            def _hoist_system_messages(msgs):
                system_parts = []
                non_system = []
                for m in msgs:
                    if m.get("role") == "system":
                        c = m.get("content", "")
                        if isinstance(c, str) and c.strip():
                            system_parts.append(c.strip())
                    else:
                        non_system.append(m)
                if system_parts:
                    merged_system = {"role": "system", "content": "\n\n".join(system_parts)}
                    return [merged_system] + non_system
                return non_system

            original_messages = _hoist_system_messages(original_messages)
            canonical_messages = _hoist_system_messages(canonical_messages)

            messages = _prepare_messages_for_template(original_messages)
            cache_messages = _prepare_messages_for_template(canonical_messages)

            if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    tools=tools,
                    enable_thinking=enable_thinking,
                )
                cache_prompt_raw = tokenizer.apply_chat_template(
                    cache_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    tools=tools,
                    enable_thinking=enable_thinking,
                )
            else:
                prompt = messages[-1]["content"] if messages else ""
                cache_prompt_raw = (
                    cache_messages[-1]["content"] if cache_messages else ""
                )

            # Post-render scrub: atomic, line-scoped patterns only (cch=, billing header,
            # timestamps, system-reminder). Never applied to the model input.
            cache_prompt = _scrub_cache_key(cache_prompt_raw)
            # Safety invariant: cache key must not be dramatically shorter than the original.
            if SETTINGS.cache_norm_safety_check:
                if not _assert_cache_key_safety(
                    prompt, cache_prompt, context="lm_path"
                ):
                    # Normalization over-matched — fall back to using the original prompt
                    # as the cache key to prevent invisible content loss.
                    cache_prompt = prompt
            # Cache lookup uses canonical token sequence; model input uses original tokens.
            prompt_tokens = _tokenize_prompt(cache_prompt)
            model_tokens = _tokenize_prompt(prompt)
            prompt_was_normalized = cache_prompt != prompt
            cache_key_delta_chars = len(prompt) - len(cache_prompt)

            # ── PIPELINE LOG: CANONICALIZATION ─────────────────────────────
            if FEATURE_FULL_LOGGING:
                _pipeline_log("CANON", request_id,
                    f"original_prompt_len={len(prompt)} | canonical_prompt_len={len(cache_prompt)} | "
                    f"delta={cache_key_delta_chars} chars | normalized={prompt_was_normalized}")
                _pipeline_log("CANON", request_id,
                    f"prompt_tokens={len(prompt_tokens)} | model_tokens={len(model_tokens)}")
                if FEATURE_LOG_PROMPTS:
                    _pipeline_log("CANON", request_id, "prompts dumped to disk",
                        data={"model_prompt_len": len(prompt),
                              "canonical_prompt_len": len(cache_prompt),
                              "prompt_tokens_count": len(prompt_tokens),
                              "model_tokens_count": len(model_tokens)})

        session_ctx = _extract_session_context(body, prompt_tokens)
        if is_vlm:
            # prompt_was_normalized: True when canonical pipeline fired and produced
            # different token count from model_tokens (i.e. canonicalization changed
            # something). When the canonical pipeline falls back to model_tokens (no
            # tokenizer available or canonicalization disabled), prompt_tokens ==
            # model_tokens and we report no normalization.
            prompt_was_normalized = bool(
                SETTINGS.cache_canonicalize_tool_context
                and prompt_tokens is not model_tokens
                and len(prompt_tokens) != len(model_tokens)
            )
            cache_key_delta_chars = len(model_tokens) - len(prompt_tokens)
        cache_key = prompt_tokens[:]
        prompt_cache = None
        # model_tokens: tokens derived from the original (unmodified) prompt.
        # The model always prefills from this sequence, never from the canonical cache key.
        rest_tokens = model_tokens
        cache_session_tokens = prompt_tokens
        cache_match_type = "miss"
        matched_prefix_len = 0
        cache_selection_source = "none"
        rest_count = len(model_tokens)
        request_logger = None
        sampler, sampler_kwargs = _build_sampler(body)
        max_tokens = body.get("max_tokens", SETTINGS.default_max_tokens)
        is_streaming = body.get("stream", False)

        acquired = False
        generated_tokens = []
        message_text = ""
        generation_started_at = None
        first_token_at = None
        queue_started_at = time.time()

        # --- STABLE-PREFIX TELEMETRY DEFAULTS ---
        stable_prefix_token_len_computed = 0
        stable_prefix_msg_count_computed = 0
        stable_prefix_diff_descriptors: List[Dict[str, Any]] = []

        try:
            model_lock.acquire(blocking=True)
            acquired = True

            generation_started_at = time.time()
            wait_seconds = generation_started_at - queue_started_at
            with prompt_cache_lock:
                # KRIPPER DUAL-SLOT: Route to the correct LRU based on agent type.
                # COMPACT runner → PROMPT_CACHE_COMPACT (isolated, max_size=1)
                # MAIN agent    → PROMPT_CACHE         (HOT slot, never touched by compact)
                # This replaces the old evict_unpinned() approach which destroyed MAIN
                # cache before compact, causing 70-110s cold-starts on the next MAIN request.
                _active_cache_store = PROMPT_CACHE_COMPACT if _is_embedded_agent else PROMPT_CACHE
                if _is_embedded_agent and FEATURE_FULL_LOGGING:
                    _pipeline_log("CACHE", request_id,
                        "COMPACT_RUNNER: using PROMPT_CACHE_COMPACT — MAIN cache untouched")

                # --- M2: Compute stable prefix from message-level diff ---
                _session_id_for_turn = (session_ctx.session_id or "").strip()
                if _session_id_for_turn and not _is_embedded_agent:
                    (
                        stable_prefix_token_len_computed,
                        stable_prefix_msg_count_computed,
                        stable_prefix_diff_descriptors,
                    ) = _stable_prefix_token_len(_session_id_for_turn, messages)

                # --- Cache lookup: MAIN uses SESSION_INDEX, COMPACT uses direct LRU ---
                # Cache lookup operates on prompt_tokens (canonical key).
                # rest_tokens will be overridden below to use model_tokens (original).
                (
                    prompt_cache,
                    _rest_tokens_canonical,
                    cache_session_tokens,
                    cache_match_type,
                    matched_prefix_len,
                    cache_selection_source,
                ) = SESSION_INDEX.select_best_cache(
                    model_name=SETTINGS.model_path,
                    prompt_tokens=prompt_tokens,
                    session_ctx=session_ctx,
                    prompt_cache_store=_active_cache_store,
                )
                # Model always prefills from original tokens (dual-pipeline invariant).
                # Use the actual KV cache offset (number of original tokens stored)
                # rather than matched_prefix_len (which is a canonical token count and
                # may differ when _canonicalize_messages() changes token lengths).
                _kv_off = _kv_cache_offset(prompt_cache)
                rest_tokens = model_tokens[
                    _kv_off if _kv_off is not None else matched_prefix_len :
                ]

                # --- M3: Stable-prefix fallback ---
                # If the global cache lookup found fewer cached tokens than the
                # message-level diff says are stable, try a secondary lookup
                # using the stable prefix as the minimum acceptable match.
                # Only triggers when there is a real improvement available.
                if (
                    stable_prefix_token_len_computed > matched_prefix_len
                    and stable_prefix_token_len_computed > 0
                    and stable_prefix_token_len_computed < len(prompt_tokens)
                ):
                    # Try to find a cache entry that covers at least stable_prefix_token_len_computed tokens.
                    stable_prefix_toks = prompt_tokens[:stable_prefix_token_len_computed]
                    (
                        sp_cache,
                        _sp_rest_tokens_canonical,
                        sp_cache_session_tokens,
                        sp_match_type,
                        sp_matched_prefix_len,
                    ) = PROMPT_CACHE.fetch_nearest_cache(
                        SETTINGS.model_path, stable_prefix_toks
                    )
                    if (
                        sp_cache is not None
                        and sp_matched_prefix_len > matched_prefix_len
                    ):
                        prompt_cache = sp_cache
                        cache_session_tokens = sp_cache_session_tokens
                        matched_prefix_len = sp_matched_prefix_len
                        cache_match_type = "stable_prefix_" + sp_match_type
                        cache_selection_source = "stable_prefix"
            
            # --- REAL-MATCH SYNC (FIX-31 v3) ---
            # Two coordinate systems:
            #   - CANONICAL space: prompt_tokens / cache_session_tokens (scrubbed,
            #     used for cache LOOKUP — same volatile fields masked identically).
            #   - MODEL space: model_tokens / _kv_cache_offset (original text,
            #     used for actual prefill and KV cache management).
            #
            # matched_prefix_len lives in CANONICAL space (telemetry only).
            # rest_tokens and KV trim operate in MODEL space exclusively.
            #
            # Bug history:
            #   v1 — compared model_tokens vs canonical → diverged at Runtime: line.
            #   v2 — fixed comparison but FIX-16/17 mixed spaces: canonical
            #         matched_prefix_len (19849) >= model _m_len (15848) always
            #         fired, forcing rest_tokens=1 → garbage output.
            if prompt_cache is not None and cache_session_tokens is not None:
                # 1. Canonical match (determines IF the cache entry is valid).
                _raw_match = 0
                _limit = min(len(prompt_tokens), len(cache_session_tokens))
                for i in range(_limit):
                    if prompt_tokens[i] == cache_session_tokens[i]:
                        _raw_match += 1
                    else:
                        break
                
                # 2. Telemetry: canonical match length (NOT used for offset math).
                matched_prefix_len = _raw_match
                
                # 3. Model-space offset: how many original tokens the KV cache holds.
                _kv_off = _kv_cache_offset(prompt_cache)
                _m_len = len(model_tokens)
                
                _terminal_status("DEBUG", f"FIX-31 v4: _kv_off={_kv_off} type(layer)={type(prompt_cache[0] if prompt_cache else None)}")

                if _kv_off is not None:
                    # FIX-31 v4: Sliding window cap bypass
                    # If physical tensor size (_kv_off) is capped (e.g., 8192 sliding window limit) 
                    # but the logical match (matched_prefix_len) proves we matched far beyond it.
                    _cache_extended = (
                        (cache_session_tokens is not None
                         and len(cache_session_tokens) > len(prompt_tokens))
                        or _kv_off > _m_len  # physical KV has response tokens
                    )
                    
                    # 1. Sliding window cap bypass & frozen restore optimization
                    # This happens when the logical match is larger than the physical cache size
                    if _kv_off < _m_len and matched_prefix_len > _kv_off:
                        if _cache_extended:
                            if _DPC.frozen_cache is not None:
                                _restored = _wm.restore_frozen_snapshot(
                                    _DPC.frozen_cache, log_fn=_terminal_status
                                )
                                if _restored is not None:
                                    prompt_cache = _restored
                                    _warmup_len = len(_DPC.frozen_tokens) if _DPC.frozen_tokens else 0
                                    _suffix_len = max(1, len(prompt_tokens) - _warmup_len)
                                    _terminal_status("DEBUG",
                                        f"FIX-31 v9: Pollution → restored frozen cache. "
                                        f"warmup={_warmup_len} | suffix={_suffix_len}")
                                else:
                                    _suffix_len = _m_len
                                    prompt_cache = None
                            else:
                                # No frozen cache available — full re-prefill is the only safe option.
                                _suffix_len = _m_len
                                prompt_cache = None
                        else:
                            _suffix_len = max(1, len(prompt_tokens) - matched_prefix_len)
                            
                        rest_tokens = model_tokens[-_suffix_len:]
                        _terminal_status("DEBUG", f"FIX-31 v9: Sliding window trap bypassed. rest_tokens={_suffix_len}")
                        
                    else:
                        # 2. Unified cache alignment
                        # Handles exact match, shorter match, pollution, and normal prefix
                        _canonical_suffix = max(1, len(prompt_tokens) - matched_prefix_len)
                        
                        if _cache_extended:
                            # For hybrid models, we need a min-suffix to wash out the recurrent state.
                            _MIN_POLLUTION_SUFFIX = 256
                            _suffix_len = max(_MIN_POLLUTION_SUFFIX, _canonical_suffix)
                            _terminal_status("DEBUG",
                                f"FIX-31 v9: Cache polluted. canonical_suffix={_canonical_suffix} | effective={_suffix_len}")
                        else:
                            _suffix_len = _canonical_suffix
                            
                        _trim_to = max(0, _m_len - _suffix_len)
                        
                        if _kv_off > _trim_to:
                            if can_trim_prompt_cache(prompt_cache):
                                trim_prompt_cache(prompt_cache, _kv_off - _trim_to)
                                rest_tokens = model_tokens[_trim_to:]
                                _terminal_status("DEBUG", f"FIX-31 v9: Trimmed KVCache {_kv_off} -> {_trim_to}")
                            else:
                                rest_tokens = model_tokens[-_suffix_len:]
                                _terminal_status("DEBUG", f"FIX-31 v9: Hybrid wash with {_suffix_len} tokens")
                        else:
                            rest_tokens = model_tokens[_kv_off:]
                            _terminal_status("DEBUG", f"FIX-31 v9: Normal continuation from {_kv_off}")
                else:
                    # No KV offset available — full prefill.
                    rest_tokens = model_tokens
                    _terminal_status("DEBUG", "FIX-31 v4: Forced rest_tokens = model_tokens because _kv_off is None")

            # --- PERFECT HIT GUARD (FIX-16/17 v2) ---
            # Ensure rest_tokens is never empty (generator crash prevention).
            # All logic uses model-space measurements exclusively.
            _m_len = len(model_tokens)
            if rest_tokens is None:
                rest_tokens = model_tokens
            if len(rest_tokens) == 0 and _m_len > 0:
                rest_tokens = model_tokens[max(0, _m_len - 1):]
                _kv_off = _kv_cache_offset(prompt_cache)
                if _kv_off and _kv_off > _m_len - 1:
                    if can_trim_prompt_cache(prompt_cache):
                        trim_prompt_cache(prompt_cache, _kv_off - (_m_len - 1))
            # ---------------------------------

            # --- DEBUG BLOCK ---
            # Only fire when the cache divergence is genuinely unexpected:
            # - On a full miss (no cache entry found at all), or
            # - On a 'shorter' hit where the matched prefix is significantly
            #   less than what the stable-prefix layer expected to be cached.
            #   A 'shorter' hit at the normal end-of-last-turn boundary is
            #   expected behaviour and should not generate noise.
            _debug_unexpected_miss = cache_match_type == "miss" or (
                cache_match_type in ("shorter",)
                and stable_prefix_token_len_computed > 0
                and matched_prefix_len < stable_prefix_token_len_computed - 64
            )
            if (
                SETTINGS.vlm_cache_debug
                and cache_session_tokens
                and _debug_unexpected_miss
            ):
                # Compare the prompt against the actual candidate the global cache evaluated
                _debug_token_divergence(
                    tokenizer, prompt_tokens, cache_session_tokens, context_window=8
                )
            # -----------------------

            # ── MEMORY GUARD: check before prefill ──────────────────────────
            # If external processes have pushed Metal memory above threshold,
            # evict cache entries NOW before allocating new KV buffers.
            _guard_evicted = _memory_guard_pre_prefill(request_id)
            if _guard_evicted > 0:
                # Cache was evicted — force a full miss to re-create from scratch
                prompt_cache = None
                cache_match_type = "miss"
                matched_prefix_len = 0
                rest_tokens = model_tokens
                rest_count = len(rest_tokens)

            if prompt_cache is None:
                cache_model = (
                    model.language_model
                    if is_vlm and hasattr(model, "language_model")
                    else model
                )
                prompt_cache = make_prompt_cache(
                    cache_model, max_kv_size=SETTINGS.max_kv_size
                )
                # Full miss: model prefills all original tokens (never canonical).
                rest_tokens = model_tokens


            rest_count = len(rest_tokens) if rest_tokens is not None else _m_len

            # ── PIPELINE: EMERGENCY COMPRESSION (A5) ───────────────────────
            # If rest_tokens exceeds MAX_SAFE_PREFILL after normal compression + cache,
            # re-compress with aggressive settings (guard=2) and re-do the pipeline.
            # This is self-healing: the response ARRIVES without error or retry.
            # Origin: OOM crash 2026-04-26 (wiki/MLXServer/challenges/out-of-memory-v83-fix)
            if (
                _emergency_compressor_available
                and _compressor_module is not None
                and not _is_embedded_agent
                and rest_count is not None
            ):
                _emergency_result = emergency_compress_if_needed(
                    raw_messages=raw_messages,
                    rest_tokens_count=rest_count,
                    compressor_module=_compressor_module,
                    session_key=_comp_session if '_comp_session' in dir() else request_id,
                    log_fn=_terminal_status,
                    request_id=request_id,
                )
                if _emergency_result is not None:
                    # Emergency compression succeeded — re-run pipeline from healing
                    raw_messages = _emergency_result
                    _pipeline_timings["emergency_compress"] = time.time()

                    # Re-heal, re-canonicalize, re-tokenize
                    healed_messages = _heal_messages(raw_messages)
                    original_messages, canonical_messages = _canonicalize_messages(healed_messages)
                    original_messages = _hoist_system_messages(original_messages)
                    canonical_messages = _hoist_system_messages(canonical_messages)
                    messages = _prepare_messages_for_template(original_messages)
                    cache_messages = _prepare_messages_for_template(canonical_messages)

                    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
                        prompt = tokenizer.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True,
                            tools=tools, enable_thinking=enable_thinking,
                        )
                        cache_prompt_raw = tokenizer.apply_chat_template(
                            cache_messages, tokenize=False, add_generation_prompt=True,
                            tools=tools, enable_thinking=enable_thinking,
                        )
                    else:
                        prompt = messages[-1]["content"] if messages else ""
                        cache_prompt_raw = cache_messages[-1]["content"] if cache_messages else ""

                    cache_prompt = _scrub_cache_key(cache_prompt_raw)
                    prompt_tokens = _tokenize_prompt(cache_prompt)
                    model_tokens = _tokenize_prompt(prompt)

                    # Re-do cache lookup with compressed tokens
                    with prompt_cache_lock:
                        (
                            prompt_cache, _rest_tokens_canonical,
                            cache_session_tokens, cache_match_type,
                            matched_prefix_len, cache_selection_source,
                        ) = SESSION_INDEX.select_best_cache(
                            model_name=SETTINGS.model_path,
                            prompt_tokens=prompt_tokens,
                            session_ctx=session_ctx,
                            prompt_cache_store=_active_cache_store,
                        )
                        _kv_off = _kv_cache_offset(prompt_cache)
                        rest_tokens = model_tokens[
                            _kv_off if _kv_off is not None else matched_prefix_len :
                        ]

                    rest_count = len(rest_tokens) if rest_tokens is not None else len(model_tokens)
                    _terminal_status("🚨",
                        f"EMERGENCY COMPRESSOR: pipeline re-done | new rest_tokens={rest_count} | "
                        f"cache_hit={cache_match_type} | matched={matched_prefix_len}/{len(prompt_tokens)}")

                    # If STILL over limit after emergency compression, signal overflow (Capa 3)
                    # Return "context length exceeded" → OpenClaw auto-compacts + retries transparently.
                    # The user never sees this error; OpenClaw handles it automatically.
                    if should_signal_overflow(rest_count):
                        _terminal_status("⚠️",
                            f"EMERGENCY COMPRESSOR: still over limit after compression | "
                            f"rest={rest_count} — signaling overflow to client")
                        _overflow_error = json.dumps({
                            "error": {
                                "message": f"context length exceeded: {rest_count} tokens exceed safe prefill limit",
                                "type": "invalid_request_error",
                                "code": "context_length_exceeded",
                            }
                        }).encode("utf-8")
                        self.send_response(400)
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Content-Length", str(len(_overflow_error)))
                        self.end_headers()
                        self.wfile.write(_overflow_error)
                        if acquired:
                            mx.clear_cache()
                            import gc; gc.collect()
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
                    f"stable_prefix: computed={stable_prefix_token_len_computed} tok")
                _pipeline_log("CACHE", request_id,
                    f"RESULTADO: cache_hit={cache_match_type} | rest_tokens={rest_count} | source={cache_selection_source}")
            

            cache_session_id = _cache_log_session_id(session_ctx, cache_session_tokens)
            if SETTINGS.enable_request_logging:
                try:
                    request_logger = CacheSessionTranscriptLogger(
                        cache_session_id=cache_session_id
                    )
                except Exception:
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
                                    "vlm_format_prefix_stable": getattr(
                                        _vlm_diagnostics, "used_prefix_stable", False
                                    ),
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
                f"Request {request_id} | {cache_light} Cache: {hit_ratio:.1f}% ({cache_match_type}) | "
                f"tokens={matched_prefix_len}/{prompt_len} | rest={rest_count} | "
                f"stream={body.get('stream', False)} | thinking={enable_thinking}",
            )

            # ── DPC: Auto-capture is handled by auto-save + prefix hash in
            # _insert_cache_entries → no need for seed file refresh. ────────

            _terminal_status(
                "⚙️",
                f"Generation started | wait={wait_seconds:.2f}s | prefill={rest_count} | "
                f"session={session_ctx.session_id[:16]} ({cache_selection_source}) | family={SETTINGS.model_family}",
                indent=1,
            )

            if not is_streaming:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                if FEATURE_DIAGNOSTIC_HEADERS:
                    self.send_header("X-Pipeline-Compression-Ms", f"{_pipeline_timings.get('compress', 0):.1f}")
                    self.send_header("X-Pipeline-RAG-Ms", f"{_pipeline_timings.get('rag', 0):.1f}")
                    self.send_header("X-Pipeline-Heal-Ms", f"{_pipeline_timings.get('heal', 0):.1f}")
                self.end_headers()

                # --- PREFILL PROGRESS INDICATOR ---
                _prefill_done = threading.Event()
                _prefill_rate = 300  # estimated tok/s from logs
                def _prefill_progress():
                    _start = time.time()
                    _est_total = rest_count / _prefill_rate if _prefill_rate > 0 else 60
                    # Log inmediato al arrancar — elimina el gap silencioso post-FLUSH
                    _terminal_status(
                        "⏳",
                        f"Request {request_id} PREFILL starting | 0% | ETA ~{_est_total:.0f}s | "
                        f"tokens={rest_count}",
                        indent=1,
                    )

                    while not _prefill_done.is_set():
                        _prefill_done.wait(5.0)
                        if _prefill_done.is_set():
                            break
                        _elapsed = time.time() - _start
                        _pct = min(99, (_elapsed / _est_total) * 100) if _est_total > 0 else 0
                        _eta = max(0, _est_total - _elapsed)
                        _terminal_status(
                            "🔄",
                            f"Request {request_id} PREFILL | {_pct:.0f}% | "
                            f"{_elapsed:.0f}s/{_est_total:.0f}s | ~{_eta:.0f}s remaining | "
                            f"tokens={rest_count}",
                            indent=1,
                        )
                if rest_count > 1000:
                    _prefill_thread = threading.Thread(target=_prefill_progress, daemon=True)
                    _prefill_thread.start()
                else:
                    _prefill_thread = None

                generated_parts = []
                progress_last_at = time.time()
                for response in _stream_generate_unified(
                    rest_tokens,
                    max_tokens,
                    sampler,
                    prompt_cache,
                    vlm_pixel_values=vlm_pixel_values,
                    vlm_mask=vlm_mask,
                    vlm_kwargs=vlm_kwargs,
                    cache_match_type=cache_match_type,
                ):
                    generated_parts.append(response.text)
                    generated_tokens.append(int(response.token))
                    if first_token_at is None:
                        first_token_at = time.time()
                        _prefill_done.set()  # Stop prefill progress
                    if (
                        len(generated_tokens) % 64 == 0
                        and (time.time() - progress_last_at) >= 1.0
                    ):
                        progress_last_at = time.time()
                        _terminal_status(
                            "⏳",
                            f"Request {request_id} in progress | generated_tokens={len(generated_tokens)}",
                            indent=1,
                        )
                response_text = "".join(generated_parts)
                raw_response_text = response_text
                response_text = _normalize_assistant_text(
                    response_text, enable_thinking, SETTINGS.model_family
                )
                message_text, tool_calls = _extract_openai_tool_calls(
                    response_text, SETTINGS.model_family
                )
                # Hide <think> blocks from the client whenever reasoning was requested.
                if enable_thinking:
                    message_text = _strip_thinking_from_content(message_text)

                    # --- NEW HEALING STORE LOGIC ---
                    if raw_response_text != message_text:
                        h = _get_healing_hash(message_text, tool_calls)
                        if h:
                            with HEALING_STORE_LOCK:
                                HEALING_STORE[h] = raw_response_text
                                HEALING_STORE.move_to_end(h, last=True)
                                while len(HEALING_STORE) > MAX_HEALING_STORE:
                                    HEALING_STORE.popitem(last=False)

                finish_reason = "tool_calls" if tool_calls else "stop"
                cache_key.extend(generated_tokens)
                with prompt_cache_lock:
                    _cache_hit_ratio = matched_prefix_len / max(len(prompt_tokens), 1) if prompt_tokens else 1.0
                    if not _is_embedded_agent:
                        # MAIN: insert into PROMPT_CACHE (HOT slot)
                        # MAIN: insert into PROMPT_CACHE (HOT slot)
                        # FIX PEND-03 (2026-04-07): Detección estricta de startup vs slug-gen.
                        # Usa regex case-insensitive para sobrevivir cambios de texto en OpenClaw.
                        _last_msg_content = str(messages[-1].get("content", ""))
                        _is_real_startup = (
                            len(messages) == 2
                            and bool(_STARTUP_MSG_PATTERN.search(_last_msg_content))
                            and "filename slug" not in _last_msg_content.lower()
                        )
                        _insert_cache_entries(
                            model_name=SETTINGS.model_path,
                            session_ctx=session_ctx,
                            cache_key=cache_key,
                            prompt_cache=prompt_cache,
                            generated_tokens=generated_tokens,
                            tool_calls=tool_calls,
                            cache_hit_ratio=_cache_hit_ratio,
                            is_warmup_candidate=_is_real_startup,
                        )
                    else:
                        # COMPACT RUNNER: insert into PROMPT_CACHE_COMPACT (isolated slot).
                        # This lets subsequent compact calls hit cache, and MAIN is NEVER touched.
                        _insert_cache_entries(
                            model_name=SETTINGS.model_path,
                            session_ctx=session_ctx,
                            cache_key=cache_key,
                            prompt_cache=prompt_cache,
                            generated_tokens=generated_tokens,
                            tool_calls=tool_calls,
                            cache_hit_ratio=_cache_hit_ratio,
                            prompt_cache_store_override=PROMPT_CACHE_COMPACT,
                        )
                        if FEATURE_FULL_LOGGING:
                            _pipeline_log("CACHE", request_id,
                                f"COMPACT_RUNNER: inserted into PROMPT_CACHE_COMPACT | "
                                f"MAIN cache slots preserved: {len(PROMPT_CACHE._entries)}")
                    # --- M5: Update session turn record for next-turn stable-prefix lookup ---
                    if _session_id_for_turn:
                        _update_session_turn_store(
                            _session_id_for_turn,
                            messages,
                            prompt_tokens,
                        )

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
                if request_logger:
                    timing = {}
                    if first_token_at is not None and generation_started_at is not None:
                        timing["prefill_seconds"] = (
                            first_token_at - generation_started_at
                        )
                        timing["decode_seconds"] = time.time() - first_token_at
                        timing["prefill_tps"] = (
                            rest_count / timing["prefill_seconds"]
                            if timing["prefill_seconds"] > 0
                            else None
                        )
                        timing["decode_tps"] = (
                            len(generated_tokens) / timing["decode_seconds"]
                            if timing["decode_seconds"] > 0
                            else None
                        )
                    request_logger.log(
                        "generation",
                        {
                            "mode": "non-stream",
                            "timing": timing,
                            "raw_response_text": raw_response_text,
                            "normalized_response_text": response_text,
                            "assistant_message_text": message_text,
                            "tool_calls": tool_calls,
                            "finish_reason": finish_reason,
                        },
                        request_id=request_id,
                    )

                if FEATURE_FULL_LOGGING:
                    if FEATURE_LOG_GENERATION:
                        _gen_ms = (time.time() - generation_started_at) * 1000
                        _pipeline_log("GEN", request_id,
                            f"finished: {len(generated_tokens)} tokens in {_gen_ms/1000:.2f}s "
                            f"({timing.get('decode_tps', 0):.1f} tok/s decode)")
                        _non_reas = len(_tokenize_prompt(message_text)) if message_text else 0
                        _reas = max(0, len(generated_tokens) - _non_reas)
                        _pipeline_log("GEN", request_id,
                            f"thinking_tokens={_reas} | output_tokens={_non_reas}")
                    if FEATURE_LOG_RESP:
                        _pipeline_log("RESP", request_id,
                            f"raw_text={len(generated_tokens)} tokens | normalize_applied=true")
                        if tool_calls:
                            _pipeline_log("RESP", request_id,
                                f"tool_calls_extracted={len(tool_calls)} {[tc.get('function', {}).get('name') for tc in tool_calls]}")
                        if enable_thinking:
                            _pipeline_log("RESP", request_id,
                                f"think_block_stripped=true ({_reas} tokens hidden from client)")
                        _pipeline_log("RESP", request_id,
                            f"message_to_client={_non_reas} tokens | finish_reason={finish_reason}")
                        # ── PIPELINE LOG: MSG_OUT (non-streaming) ─────────────────────────
                        _agent_tag_out = "EMBEDDED" if _is_embedded_agent else "MAIN"
                        if tool_calls:
                            for _tc in tool_calls:
                                _tc_name = _tc.get("function", {}).get("name", "?")
                                _tc_args = str(_tc.get("function", {}).get("arguments", ""))[:300].replace("\n", " ")
                                _pipeline_log("MSG_OUT", request_id,
                                    f"[{_agent_tag_out}] tool_call: {_tc_name}({_tc_args})")
                        elif message_text:
                            _out_preview = message_text[:300].replace("\n", " ")
                            _pipeline_log("MSG_OUT", request_id,
                                f"[{_agent_tag_out}] text: {_out_preview!r}")
            else:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.end_headers()

                response_id = f"chatcmpl-{int(time.time())}"
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

                # --- SSE KEEPALIVE THREAD ---
                # During prefill (~58s) and decode, no SSE events are sent because
                # we must collect all tokens to post-process (strip <think>, extract
                # tool_calls).  Without keepalives the idle TCP connection dies and
                # LiteLLM / OpenClaw sees "Connection error".
                # SSE spec: lines starting with ':' are comments, ignored by clients.
                _keepalive_stop = threading.Event()
                _keepalive_wfile = self.wfile  # capture for thread
                _keepalive_interval = 5  # seconds

                def _sse_keepalive_sender():
                    while not _keepalive_stop.wait(_keepalive_interval):
                        try:
                            _keepalive_wfile.write(b": keepalive\n\n")
                            _keepalive_wfile.flush()
                        except Exception:
                            break  # client disconnected

                _keepalive_thread = threading.Thread(
                    target=_sse_keepalive_sender, daemon=True
                )
                _keepalive_thread.start()

                # --- PREFILL PROGRESS INDICATOR (streaming) ---
                _prefill_done_s = threading.Event()
                _prefill_rate_s = 300
                def _prefill_progress_s():
                    _start = time.time()
                    _est_total = rest_count / _prefill_rate_s if _prefill_rate_s > 0 else 60
                    # Log inmediato al arrancar — elimina el gap silencioso post-FLUSH (streaming)
                    _terminal_status(
                        "⏳",
                        f"Request {request_id} PREFILL starting | 0% | ETA ~{_est_total:.0f}s | "
                        f"tokens={rest_count}",
                        indent=1,
                    )
                    while not _prefill_done_s.is_set():
                        _prefill_done_s.wait(5.0)
                        if _prefill_done_s.is_set():
                            break
                        _elapsed = time.time() - _start
                        _pct = min(99, (_elapsed / _est_total) * 100) if _est_total > 0 else 0
                        _eta = max(0, _est_total - _elapsed)
                        _terminal_status(
                            "🔄",
                            f"Request {request_id} PREFILL | {_pct:.0f}% | "
                            f"{_elapsed:.0f}s/{_est_total:.0f}s | ~{_eta:.0f}s remaining | "
                            f"tokens={rest_count}",
                            indent=1,
                        )
                if rest_count > 1000:
                    _prefill_thread_s = threading.Thread(target=_prefill_progress_s, daemon=True)
                    _prefill_thread_s.start()

                raw_parts = []

                progress_last_at = time.time()
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
                    ):
                        generated_tokens.append(int(response.token))
                        if first_token_at is None:
                            first_token_at = time.time()
                            _prefill_done_s.set()  # Stop prefill progress
                        response_text = response.text
                        if response_text:
                            raw_parts.append(response_text)
                        if (
                            len(generated_tokens) % 64 == 0
                            and (time.time() - progress_last_at) >= 1.0
                        ):
                            progress_last_at = time.time()
                            _terminal_status(
                                "⏳",
                                f"Request {request_id} in progress | generated_tokens={len(generated_tokens)}",
                                indent=1,
                            )
                finally:
                    # Stop keepalive thread BEFORE writing actual content chunks
                    _keepalive_stop.set()
                    _keepalive_thread.join(timeout=2)

                full_text = "".join(raw_parts)
                raw_full_text = full_text
                full_text = _normalize_assistant_text(
                    full_text, enable_thinking, SETTINGS.model_family
                )
                message_text, tool_calls = _extract_openai_tool_calls(
                    full_text, SETTINGS.model_family
                )
                # Hide <think> blocks from the client whenever reasoning was requested.
                if enable_thinking:
                    message_text = _strip_thinking_from_content(message_text)

                    # --- NEW HEALING STORE LOGIC ---
                    if raw_full_text != message_text:
                        h = _get_healing_hash(message_text, tool_calls)
                        if h:
                            with HEALING_STORE_LOCK:
                                HEALING_STORE[h] = raw_full_text
                                HEALING_STORE.move_to_end(h, last=True)
                                while len(HEALING_STORE) > MAX_HEALING_STORE:
                                    HEALING_STORE.popitem(last=False)

                cache_key.extend(generated_tokens)
                with prompt_cache_lock:
                    _cache_hit_ratio = matched_prefix_len / max(len(prompt_tokens), 1) if prompt_tokens else 1.0
                    if not _is_embedded_agent:
                        # FIX B2: is_warmup_candidate must be passed in streaming path too.
                        _last_msg_content_s = str(messages[-1].get("content", ""))
                        _is_real_startup_s = (
                            len(messages) == 2
                            and bool(_STARTUP_MSG_PATTERN.search(_last_msg_content_s))
                            and "filename slug" not in _last_msg_content_s.lower()
                        )
                        _insert_cache_entries(
                            model_name=SETTINGS.model_path,
                            session_ctx=session_ctx,
                            cache_key=cache_key,
                            prompt_cache=prompt_cache,
                            generated_tokens=generated_tokens,
                            tool_calls=tool_calls,
                            cache_hit_ratio=_cache_hit_ratio,
                            is_warmup_candidate=_is_real_startup_s,
                        )
                    else:
                        # COMPACT RUNNER: insert into PROMPT_CACHE_COMPACT (isolated slot).
                        # This lets subsequent compact calls hit cache, and MAIN is NEVER touched.
                        # (Aligned with non-streaming path — audit fix B2)
                        _cache_hit_ratio = matched_prefix_len / max(len(prompt_tokens), 1) if prompt_tokens else 1.0
                        _insert_cache_entries(
                            model_name=SETTINGS.model_path,
                            session_ctx=session_ctx,
                            cache_key=cache_key,
                            prompt_cache=prompt_cache,
                            generated_tokens=generated_tokens,
                            tool_calls=tool_calls,
                            cache_hit_ratio=_cache_hit_ratio,
                            prompt_cache_store_override=PROMPT_CACHE_COMPACT,
                        )
                        if FEATURE_FULL_LOGGING:
                            _pipeline_log("CACHE", request_id,
                                f"COMPACT_RUNNER: inserted into PROMPT_CACHE_COMPACT (streaming) | "
                                f"MAIN cache slots preserved: {len(PROMPT_CACHE._entries)}")
                    # --- M5: Update session turn record for next-turn stable-prefix lookup ---
                    if _session_id_for_turn:
                        _update_session_turn_store(
                            _session_id_for_turn,
                            messages,
                            prompt_tokens,
                        )

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

                finish_reason = "tool_calls" if tool_calls else "stop"
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
                if request_logger:
                    timing = {}
                    if first_token_at is not None and generation_started_at is not None:
                        timing["prefill_seconds"] = (
                            first_token_at - generation_started_at
                        )
                        timing["decode_seconds"] = time.time() - first_token_at
                        timing["prefill_tps"] = (
                            rest_count / timing["prefill_seconds"]
                            if timing["prefill_seconds"] > 0
                            else None
                        )
                        timing["decode_tps"] = (
                            len(generated_tokens) / timing["decode_seconds"]
                            if timing["decode_seconds"] > 0
                            else None
                        )
                    request_logger.log(
                        "generation",
                        {
                            "mode": "stream",
                            "timing": timing,
                            "raw_response_text": raw_full_text,
                            "normalized_response_text": full_text,
                            "assistant_message_text": message_text,
                            "tool_calls": tool_calls,
                            "finish_reason": finish_reason,
                        },
                        request_id=request_id,
                    )

                if FEATURE_FULL_LOGGING:
                    if FEATURE_LOG_GENERATION:
                        _gen_ms = (time.time() - generation_started_at) * 1000
                        _pipeline_log("GEN", request_id,
                            f"finished: {len(generated_tokens)} tokens in {_gen_ms/1000:.2f}s "
                            f"({timing.get('decode_tps', 0):.1f} tok/s decode)")
                        _non_reas = len(_tokenize_prompt(message_text)) if message_text else 0
                        _reas = max(0, len(generated_tokens) - _non_reas)
                        _pipeline_log("GEN", request_id,
                            f"thinking_tokens={_reas} | output_tokens={_non_reas}")
                    if FEATURE_LOG_RESP:
                        _pipeline_log("RESP", request_id,
                            f"raw_text={len(generated_tokens)} tokens | normalize_applied=true")
                        if tool_calls:
                            _pipeline_log("RESP", request_id,
                                f"tool_calls_extracted={len(tool_calls)} {[tc.get('function', {}).get('name') for tc in tool_calls]}")
                        if enable_thinking:
                            _pipeline_log("RESP", request_id,
                                f"think_block_stripped=true ({_reas} tokens hidden from client)")
                        _pipeline_log("RESP", request_id,
                            f"message_to_client={_non_reas} tokens | finish_reason={finish_reason}")
                        # ── PIPELINE LOG: MSG_OUT (streaming) ────────────────────────────
                        _agent_tag_out = "EMBEDDED" if _is_embedded_agent else "MAIN"
                        if tool_calls:
                            for _tc in tool_calls:
                                _tc_name = _tc.get("function", {}).get("name", "?")
                                _tc_args = str(_tc.get("function", {}).get("arguments", ""))[:300].replace("\n", " ")
                                _pipeline_log("MSG_OUT", request_id,
                                    f"[{_agent_tag_out}] tool_call: {_tc_name}({_tc_args})")
                        elif message_text:
                            _out_preview = message_text[:300].replace("\n", " ")
                            _pipeline_log("MSG_OUT", request_id,
                                f"[{_agent_tag_out}] text: {_out_preview!r}")

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
        except Exception as e:
            _terminal_status("❌", f"Request {request_id} failed: {e}", indent=1)
            if request_logger:
                request_logger.log("generation", f"error: {e}", request_id=request_id)
        finally:
            if generation_started_at is not None:
                end_at = time.time()
                elapsed = max(end_at - generation_started_at, 1e-9)
                output_tokens = len(generated_tokens)
                speed = output_tokens / elapsed if output_tokens else 0.0

                non_reasoning_tokens = (
                    len(_tokenize_prompt(message_text)) if message_text else 0
                )
                reasoning_tokens = max(0, output_tokens - non_reasoning_tokens)
                token_breakdown = (
                    f"{output_tokens} (reasoning: {reasoning_tokens}, output: {non_reasoning_tokens})"
                    if enable_thinking
                    else f"{output_tokens}"
                )

                if first_token_at is not None:
                    prefill_seconds = first_token_at - generation_started_at
                    decode_seconds = max(end_at - first_token_at, 1e-9)
                    decode_tps = (
                        output_tokens / decode_seconds if output_tokens else 0.0
                    )
                    prefill_tps = (
                        rest_count / prefill_seconds if prefill_seconds > 0 else 0.0
                    )
                    _terminal_status(
                        "✅",
                        (
                            f"Request {request_id} finished | output_tokens={token_breakdown} | "
                            f"elapsed={elapsed:.2f}s | tok/s={speed:.2f} | "
                            f"prefill={prefill_seconds:.2f}s ({prefill_tps:.0f} tok/s) | "
                            f"decode={decode_seconds:.2f}s ({decode_tps:.1f} tok/s)"
                        ),
                        indent=1,
                    )
                else:
                    _terminal_status(
                        "✅",
                        (
                            f"Request {request_id} finished | output_tokens={token_breakdown} | "
                            f"elapsed={elapsed:.2f}s | tok/s={speed:.2f}"
                        ),
                        indent=1,
                    )
            # On normal exit or Python exception, release the lock. On process abort (e.g. Metal
            # "uncommitted encoder" crash), finally may not run, so the "leaked semaphore" warning
            # at shutdown is expected; fixing the Metal crash resolves it.
            if acquired:
                # KRIPPER DUAL-SLOT: compact runner's KV cache is stored in PROMPT_CACHE_COMPACT
                # (not released). No need to null-out prompt_cache here — the LRU manages it.
                # MAIN cache is always untouched.
                mx.clear_cache()
                import gc; gc.collect()
                if FEATURE_FULL_LOGGING:
                    _pipeline_log("METAL", request_id, f"mx.clear_cache() + gc.collect() called"
                        f"{' | COMPACT slot retained in PROMPT_CACHE_COMPACT' if _is_embedded_agent else ''}")
                    if generation_started_at is not None:
                        _held_ms = (time.time() - generation_started_at) * 1000
                        _pipeline_log("METAL", request_id, f"model_lock released | held_for={_held_ms/1000:.2f}s")
                model_lock.release()


def run():
    start_litellm_proxy()
    server_address = (SETTINGS.mlx_host, SETTINGS.mlx_port)
    httpd = ThreadingHTTPServer(server_address, APIHandler)

    # ── SIDECAR: Start lightweight endpoint in daemon thread ───────────
    sidecar_httpd = None
    if SETTINGS.sidecar_port > 0:
        try:
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

    # ── Wait for warmup to complete before accepting requests ──────────
    # Without this, the first request arrives while warmup is still prefilling
    # in background → 0% cache miss → 75s prefill. Better to delay SYSTEM READY
    # by ~82s and guarantee cache hits from the first request.
    if not _WARMUP_DONE.is_set():
        print("\n⏳ Waiting for warmup to complete before accepting requests...")
        _WARMUP_DONE.wait(timeout=300)  # 5 min max, warmup typically takes ~82s

    print("\n" + "=" * 50)
    print("🟢 SYSTEM READY")
    print(f"   • Mode:         {'VLM (vision)' if is_vlm else 'LM (text-only)'}")
    print(f"   • MLX Engine:   http://{SETTINGS.mlx_host}:{SETTINGS.mlx_port}")
    print(f"   • LiteLLM:      http://0.0.0.0:{SETTINGS.proxy_port}")
    if sidecar_httpd:
        rag_tag = " + RAG" if (SETTINGS.sidecar_enable_rag and _rag_available) else ""
        print(f"   • Sidecar:      http://{SETTINGS.mlx_host}:{SETTINGS.sidecar_port}{rag_tag}")
    elif SETTINGS.sidecar_port > 0:
        print(f"   • Sidecar:      FAILED (port {SETTINGS.sidecar_port})")
    else:
        print(f"   • Sidecar:      DISABLED")
    print("=" * 50 + "\n")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if sidecar_httpd:
            sidecar_httpd.shutdown()


if __name__ == "__main__":
    run()
