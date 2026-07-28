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
    from .emergency_compressor import emergency_compress_if_needed, should_signal_overflow, _max_safe_prefill
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

__version__ = "1.4.2"


# ── Configuration (extracted to config.py) ────────────────────────────────────
from .config import (
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
    except Exception:
        pass  # Best-effort, non-critical

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
    except Exception:
        pass

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
FEATURE_RAG_ENRICHMENT        = _env_str("FEATURE_RAG_ENRICHMENT", "false").lower() in ("1", "true", "yes")
FEATURE_RAG_WORKSPACE_ROOT    = _env_str("RAG_WORKSPACE_ROOT", "")

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
FEATURE_RAG_RELEVANCE_THRESHOLD = 1.6  # Qwen3-Embed asymmetric (docs without prefix). Tested 0.8: filters too much

# Tool Call Loop Breaker: detect and break infinite tool-call retry loops.
# When the model retries the same failed tool N consecutive times, inject a
# stop instruction into the last tool_result so the model gives up and responds
# with text instead.  Env: TOOL_LOOP_BREAKER=true (default), TOOL_LOOP_MAX_RETRIES=3.
FEATURE_TOOL_LOOP_BREAKER     = _env_bool("TOOL_LOOP_BREAKER", True)
TOOL_LOOP_MAX_RETRIES         = _env_int("TOOL_LOOP_MAX_RETRIES", 3)

# Healing Store: restore <think> blocks stripped by clients back into assistant
# messages so the KV cache matches the original generation.
# DISABLE (HEALING=false) to test without healing — the Qwen3 template already
# strips thinking from previous turns, so healing is likely unnecessary and
# introduces tool_call format variance that destabilizes the cache.
FEATURE_HEALING               = _env_bool("HEALING", True)

# Housekeeping Cache Borrow: when True, housekeeping requests USE the
# conversation cache (good model quality) but RESTORE it afterwards
# so the next normal request still gets a cache hit.
# When False, housekeeping skips cache lookup entirely (TPC cold start).
FEATURE_HOUSEKEEPING_CACHE_BORROW = _env_bool("HOUSEKEEPING_CACHE_BORROW", True)

# Dynamic Prefix Capture (DPC): replaces warmup_seed.txt with auto-capture + hash validation.
# Managed by warmup_manager.py. No manual seed files needed.
from . import warmup_manager as _wm
from .cache_engine import (
    HybridGenerationCheckpoint,
    capture_hybrid_generation_checkpoint,
    prepare_cache_for_insertion,
    restore_hybrid_generation_checkpoint,
    snapshot_arrays_cache,
    rollback_arrays_cache,
)

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

# ── Session Log: persist all console output to lastlog.md ─────────────────────
import re as _re
_ANSI_STRIP_RE = _re.compile(r'\033\[[0-9;]*m')
_LASTLOG_PATH = SETTINGS.log_root.parent / "lastlog.md"

# Truncate on server start (new session)
try:
    with open(_LASTLOG_PATH, "w", encoding="utf-8") as _f:
        _f.write(f"# SuperMLX Session Log — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n```\n")
except Exception:
    pass


def _console_emit(line: str) -> None:
    """Print to stdout AND append to lastlog.md (ANSI-stripped). Must be called under console_lock."""
    print(line, flush=True)
    try:
        clean = _ANSI_STRIP_RE.sub("", line)
        with open(_LASTLOG_PATH, "a", encoding="utf-8") as f:
            f.write(clean + "\n")
    except Exception:
        pass  # Never crash for a log write


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

# ── Tool parsing (extracted to tool_parsing.py) ──────────────────────────────
from .tool_parsing import (
    TOOL_CALL_PATTERN, GEMMA4_TOOL_CALL_PATTERN, ARG_PAIR_PATTERN,
    QWEN_FUNCTION_PATTERN, QWEN_PARAMETER_PATTERN,
    THINK_TAG_STRIP_PATTERN, GEMMA4_THINK_STRIP_PATTERN,
    GEMMA4_THINK_ORPHAN_PATTERN, THINK_ORPHAN_CLOSE_PATTERN,
    _strip_thinking_from_content, _extract_thinking_text, _should_enable_thinking,
    _reasoning_level_to_enable_thinking, _extract_enable_thinking,
    _normalize_assistant_text, _coerce_arg_value, _extract_openai_tool_calls,
)
# ── Message pipeline (extracted to message_pipeline.py) ──────────────────────
from .message_pipeline import (
    INBOUND_META_MESSAGE_ID_PATTERN, SUBAGENT_STATS_PATTERN,
    CACHE_TIME_PATTERN, CACHE_TIME_COLON_PATTERN, CACHE_CCH_PATTERN,
    CACHE_BILLING_HEADER_PATTERN, CACHE_SYSTEM_REMINDER_PATTERN,
    CACHE_SKILLS_BLOCK_PATTERN, CACHE_RUNTIME_LINE_PATTERN,
    SessionContext,
    _get_healing_hash, _heal_messages, _extract_tool_call_signature,
    _break_tool_call_loop, _inject_loop_stop,
    _count_roles, _summarize_tool_results, _estimate_token_count,
    _is_slug_gen_request, _is_title_gen_request, _is_rag_bypass_request, _detect_compact_runner,
    _is_hermes_housekeeping_request,
    _flatten_content, _prepare_messages_for_template,
    _scrub_cache_key, _canonicalize_inbound_context_block,
    _canonicalize_messages, _extract_session_context,
    _assert_cache_key_safety, _hoist_system_messages,
    _COMPACT_RUNNER_SIGNALS, _RAG_BYPASS_SIGNALS_USER, _RAG_BYPASS_SIGNALS_SYSTEM,
)
from .debug_tools import _debug_token_divergence
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


# ANSI color codes for terminal log highlighting
_ANSI_YELLOW = "\033[33m"
_ANSI_RED = "\033[31m"
_ANSI_DIM = "\033[2m"
_ANSI_RESET = "\033[0m"
# Stages that get yellow highlighting (compression/compaction events)
_HIGHLIGHT_STAGES = {"COMPRESS", "COMPACT_RUNNER"}


# ── Auto-Delta Tracker ───────────────────────────────────────────────────────
# Tracks timestamps per request to compute inter-stage deltas automatically.
# Maintains rolling stats per transition type for adaptive slow-detection.
# Thread-safe: always called under console_lock.

class _DeltaTracker:
    """Tracks last timestamp per request and rolling stats per transition type."""

    __slots__ = ("_req_last", "_transition_history", "_history_size")

    def __init__(self, history_size: int = 50):
        self._req_last: Dict[str, tuple] = {}          # req[:8] -> (time.time(), stage)
        self._transition_history: Dict[str, deque] = {} # "FROM→TO" -> deque of delta_secs
        self._history_size = history_size

    def record(self, request_id: str, stage: str):
        """Record a timestamp. Returns (delta_secs, from_stage, is_slow) or None."""
        now = time.time()
        key = request_id[:8]
        prev = self._req_last.get(key)
        self._req_last[key] = (now, stage)

        if prev is None:
            return None

        prev_ts, prev_stage = prev
        delta = now - prev_ts

        # Track per-transition rolling history
        trans_key = f"{prev_stage}→{stage}"
        hist = self._transition_history.get(trans_key)
        if hist is None:
            hist = deque(maxlen=self._history_size)
            self._transition_history[trans_key] = hist
        hist.append(delta)

        # Adaptive threshold: P90 × 2, minimum 1.0s
        is_slow = False
        if len(hist) >= 5:
            sorted_hist = sorted(hist)
            p90 = sorted_hist[int(len(sorted_hist) * 0.9)]
            threshold = max(1.0, p90 * 2)
            is_slow = delta > threshold
        elif delta > 5.0:
            # Cold start fallback: flag anything > 5s
            is_slow = True

        return (delta, prev_stage, is_slow)

    def cleanup(self, request_id: str):
        """Remove tracking for a completed request."""
        self._req_last.pop(request_id[:8], None)


_delta_tracker = _DeltaTracker()


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

    Auto-delta: appends Δ=Xs to the line when the gap from the previous
    stage (same request) exceeds 500ms.  Paints the line RED when the gap
    exceeds the adaptive threshold (P90×2 of rolling history).
    """
    if not FEATURE_FULL_LOGGING:
        return

    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]  # HH:MM:SS.mmm
    tag = f"[{stage}]"
    pad = "  " * max(indent, 0)

    # Auto-delta tracking
    delta_info = _delta_tracker.record(request_id, stage)
    delta_suffix = ""
    if delta_info:
        delta_secs, _from_stage, is_slow = delta_info
        if delta_secs >= 0.5:  # Only annotate deltas >= 500ms
            delta_suffix = f" Δ={delta_secs:.3f}s"

    line = f"{pad}{tag} {ts} req={request_id[:8]} | {message}{delta_suffix}"

    # Color priority: red (slow) > yellow (highlight stage) > default
    if delta_info and delta_info[2]:  # is_slow
        line = f"{_ANSI_RED}{line}{_ANSI_RESET}"
    elif stage in _HIGHLIGHT_STAGES:
        line = f"{_ANSI_YELLOW}{line}{_ANSI_RESET}"

    with console_lock:
        _console_emit(line)

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

def _terminal_status(icon: str, message: str, indent: int = 0, *,
                     request_id: str = None, stage: str = None) -> None:
    """Terminal status logger.

    When request_id and stage are provided, uses unified pipeline format
    with auto-delta tracking:  icon [STAGE] HH:MM:SS.mmm req=xxx | message [Δ=Xs]
    Otherwise uses legacy emoji format:  icon [HH:MM:SS] message

    Backward compatible — all existing calls work unchanged.
    """
    now = datetime.now()

    if request_id and stage:
        # ── Unified format with delta tracking ──
        ts = now.strftime("%H:%M:%S.%f")[:-3]
        tag = f"[{stage}]"
        pad = "  " * max(indent, 0)

        delta_info = _delta_tracker.record(request_id, stage)
        delta_suffix = ""
        if delta_info:
            delta_secs, _from_stage, is_slow = delta_info
            if delta_secs >= 0.5:
                delta_suffix = f" Δ={delta_secs:.3f}s"

        line = f"{pad}{icon} {tag} {ts} req={request_id[:8]} | {message}{delta_suffix}"

        if delta_info and delta_info[2]:  # is_slow
            line = f"{_ANSI_RED}{line}{_ANSI_RESET}"
        elif "EMERGENCY COMPRESSOR" in message:
            line = f"{_ANSI_YELLOW}{line}{_ANSI_RESET}"
    else:
        # ── Legacy format (startup, system messages) ──
        ts = now.strftime("%H:%M:%S.%f")[:-3]
        pad = "  " * max(indent, 0)
        line = f"{pad}  {icon} {ts} | {message}"
        if "EMERGENCY COMPRESSOR" in message:
            line = f"{_ANSI_YELLOW}{line}{_ANSI_RESET}"

    with console_lock:
        _console_emit(line)




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
        Kripper Base Slot: entries with pinned=True are immune to eviction.
        Only if ALL slots are pinned, the oldest is evicted (safety fallback).
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

        # Safety fallback: if ALL slots are pinned, evict the oldest pinned
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
                    "Kripper slot: all slots are pinned, evicting the oldest",
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
# max_size=1: compact generates a one-shot summary, no history needed.
PROMPT_CACHE_COMPACT = LRUPromptCache(
    max_size=1,
    ttl_seconds=SETTINGS.prompt_cache_ttl_seconds,
)

# Last tools list and system body from a MAIN (non-compact) request.
# Used by _prewarm_post_compact to build canonical cache keys that match
# the dual pipeline (model_tokens vs prompt_tokens/canonical).
_LAST_MAIN_TOOLS: Optional[List[Dict[str, Any]]] = None
_LAST_MAIN_SYSTEM_BODY: Optional[str] = None


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



# ── Shared helpers: deduplicate streaming / non-streaming do_POST paths ───────

def _metal_mem_str() -> str:
    """Return a compact Metal memory status string. Best-effort, never raises."""
    try:
        active_gb = mx.get_active_memory() / 1e9
        peak_gb = mx.get_peak_memory() / 1e9
        cache_gb = mx.get_cache_memory() / 1e9
        return f"metal={active_gb:.2f}GB peak={peak_gb:.2f}GB cache={cache_gb:.2f}GB"
    except Exception:
        return ""


# Minimum tokens to even consider running memory relief (avoid overhead for tiny prefills)
_PREFILL_MEMORY_RELIEF_MIN_TOKENS = 1000

# Empirical KV cache bytes per token at FP16 (Qwen3.6-27B-class: ~65 KB/tok).
# Scaled down by KV_BITS at runtime.  Override via PREFILL_KV_BYTES_PER_TOKEN env var.
_PREFILL_KV_BYTES_PER_TOKEN_FP16 = int(os.environ.get("PREFILL_KV_BYTES_PER_TOKEN_FP16", "65536"))

# Safety margin: run relief when projected usage exceeds this fraction of Metal budget
_PREFILL_RELIEF_BUDGET_FRACTION = float(os.environ.get("PREFILL_RELIEF_BUDGET_FRACTION", "0.90"))

# ── Metal budget (effective memory ceiling for prefill estimation) ─────────
_metal_budget_gb_cached: Optional[float] = None

def _get_metal_budget_gb() -> float:
    """Return effective Metal budget in GB for dynamic guard calculations.

    Priority: METAL_BUDGET_GB env var > device_info - 3GB (OS overhead).
    Cached after first call. Note: iogpu.wired_limit_mb is NOT used here —
    that controls MoE expert pinning, not the Metal allocation ceiling.
    """
    global _metal_budget_gb_cached
    if _metal_budget_gb_cached is not None:
        return _metal_budget_gb_cached

    # 1. Explicit env var (most reliable)
    _env_val = os.environ.get("METAL_BUDGET_GB")
    if _env_val:
        _metal_budget_gb_cached = float(_env_val)
        return _metal_budget_gb_cached

    # 2. Device total minus OS/system overhead (~3GB)
    # On a 24GiB Mac: device_info reports 25.77GB, OOM occurs ~22.4GB → 3.3GB overhead.
    try:
        _device_gb = mx.device_info()["memory_size"] / 1e9
        _metal_budget_gb_cached = _device_gb - 3.0
    except Exception:
        _metal_budget_gb_cached = 20.0  # conservative default
    return _metal_budget_gb_cached

# Threshold for Expert Breathing: rest tokens above this trigger breathe_down
_BREATHE_DOWN_REST_THRESHOLD = int(os.environ.get("BREATHE_DOWN_REST_THRESHOLD", "15000"))

# GPU yield: insert a tiny sleep between generated tokens so other Metal clients
# (Chrome/Safari VideoToolbox) can squeeze command buffers into the GPU queue.
# 0 = disabled (default), 1-2 ms is enough for smooth video playback alongside inference.
_GPU_YIELD_SECONDS = float(os.environ.get("GPU_YIELD_MS", "0")) / 1000.0

# Track whether breathing is active for the current request
_breathing_active = False


def _find_switch_mlp_for_breathing(layer):
    """Wrapper for expert_cache._find_switch_mlp without layer_idx (not needed here)."""
    from .expert_cache import _find_switch_mlp
    return _find_switch_mlp(layer)


def _pre_prefill_memory_relief(request_id: str, rest_count: int, is_embedded_agent: bool = False) -> None:
    """Free OS and Metal memory before large prefills to reduce peak pressure.

    Runs gc.collect + mx.clear_cache + malloc_zone_pressure_relief
    (macOS-specific: tells the C allocator to return freed pages to the OS).
    Also triggers Expert Breathing (breathe_down) for large MAIN requests.

    Decision is memory-based: estimates projected Metal usage after prefill
    (current + KV cache growth + scratch) and only fires when it would exceed
    _PREFILL_RELIEF_BUDGET_FRACTION of the Metal budget.  Tiny prefills
    (< _PREFILL_MEMORY_RELIEF_MIN_TOKENS) are always skipped.
    """
    global _breathing_active

    # Skip trivially small prefills (no benefit from relief)
    if rest_count < _PREFILL_MEMORY_RELIEF_MIN_TOKENS:
        return

    # ── Memory-based gate: project post-prefill usage ──────────────
    _projected_gb = -1.0  # fallback if estimation fails
    _threshold_gb = -1.0
    try:
        _active_gb = mx.get_active_memory() / 1e9
        _kv_bits = SETTINGS.kv_bits or 16  # None means FP16
        _kv_bytes_per_tok = _PREFILL_KV_BYTES_PER_TOKEN_FP16 * _kv_bits / 16
        _kv_growth_gb = rest_count * _kv_bytes_per_tok / 1e9
        _scratch_gb = 1.5  # activation/attention scratch during prefill
        _projected_gb = _active_gb + _kv_growth_gb + _scratch_gb
        _budget_gb = _get_metal_budget_gb()
        _threshold_gb = _budget_gb * _PREFILL_RELIEF_BUDGET_FRACTION

        if _projected_gb < _threshold_gb:
            _pipeline_log("METAL", request_id,
                f"PRE_PREFILL_RELIEF: SKIPPED (headroom) | "
                f"rest={rest_count} | projected={_projected_gb:.2f}GB | "
                f"threshold={_threshold_gb:.2f}GB ({_PREFILL_RELIEF_BUDGET_FRACTION:.0%} of {_budget_gb:.1f}GB) | "
                f"{_metal_mem_str()}")
            return
    except Exception:
        pass  # If estimation fails, fall through to relief (safe default)
    _mem_before = _metal_mem_str()
    import gc as _gc
    _gc.collect()
    mx.clear_cache()
    # macOS: return freed malloc pages to the OS
    try:
        import ctypes
        _libc = ctypes.CDLL("libSystem.dylib")
        _libc.malloc_zone_pressure_relief(0, 0)
    except Exception:
        pass  # Non-macOS or ctypes unavailable

    # Expert Breathing: contract experts for large MAIN prefills
    if (
        not is_embedded_agent
        and rest_count >= _BREATHE_DOWN_REST_THRESHOLD
        and hasattr(model, "_moe_config")
        and model._moe_config.get("capacity", 0) > 100
    ):
        try:
            from .expert_cache import breathe_down, PredictiveCachedSwitchLinear
            config = model._moe_config
            current_cap = config.get("capacity", 256)
            num_experts = config.get("num_experts", 256)
            # Count experts with non-zero breathing priority (session*3 + historical*1)
            # Target = keep only the ones that have SOME usage signal, evict the rest
            min_used = current_cap  # worst case: keep all
            for layer in model.layers:
                switch, _ = _find_switch_mlp_for_breathing(layer)
                if switch is None:
                    continue
                proj = getattr(switch, "up_proj", None)
                if not isinstance(proj, PredictiveCachedSwitchLinear):
                    continue
                cache = proj._cache
                used_count = sum(1 for eid in cache.cached_ids if cache.breathing_priority(eid) > 0)
                min_used = min(min_used, used_count)
            target = max(min_used, 64)  # Never below 64
            if target < current_cap:
                result = breathe_down(model, target, log_fn=_terminal_status)
                if result.get("breathed"):
                    _breathing_active = True
        except Exception as _be:
            _terminal_status("⚠️", f"BREATHE DOWN failed: {_be} | req={request_id[:8]}")

    _mem_after = _metal_mem_str()
    _pipeline_log("METAL", request_id,
        f"PRE_PREFILL_RELIEF: gc+clear_cache+malloc_pressure | "
        f"rest={rest_count} | projected={_projected_gb:.2f}GB/{_threshold_gb:.2f}GB | "
        f"before={_mem_before} | after={_mem_after}")


def _start_prefill_progress(
    request_id: str, rest_count: int, log_fn=None, rate: int = 300
) -> Tuple[threading.Event, Optional[threading.Thread]]:
    """Launch a background prefill progress logger. Returns (done_event, thread|None)."""
    done = threading.Event()
    if rest_count <= 1000:
        return done, None
    _log = log_fn or _terminal_status

    def _progress():
        start = time.time()
        est_total = rest_count / rate if rate > 0 else 60
        _log(
            "⏳",
            f"Request {request_id} PREFILL starting | 0% | ETA ~{est_total:.0f}s | "
            f"tokens={rest_count} | {_metal_mem_str()}",
            indent=1,
        )
        
        try:
            last_cleared_mem = mx.get_active_memory()
        except Exception:
            last_cleared_mem = 0
            
        last_log_time = start
        
        while not done.is_set():
            done.wait(0.5)
            if done.is_set():
                break
                
            # --- MEMORY-DRIVEN INTRA-PREFILL RELIEF ---
            try:
                current_mem = mx.get_active_memory()
                if last_cleared_mem > 0 and current_mem - last_cleared_mem >= 600 * 1024 * 1024:  # 600 MB threshold
                    mx.clear_cache()
                    _log("🧹", f"Memory Guard: Purged intra-prefill transient memory. Was {current_mem/1e9:.2f}GB", indent=2)
                    last_cleared_mem = mx.get_active_memory()
            except Exception:
                pass  # Avoid silent thread death
            # ------------------------------------------

            elapsed = time.time() - start
            if time.time() - last_log_time >= 5.0:
                pct = min(99, (elapsed / est_total) * 100) if est_total > 0 else 0
                eta = max(0, est_total - elapsed)
                _est_tps = rest_count / elapsed if elapsed > 0 else 0
                _log(
                    "🔄",
                    f"Request {request_id} PREFILL | {pct:.0f}% | "
                    f"{elapsed:.0f}s/{est_total:.0f}s | ~{eta:.0f}s remaining | "
                    f"{_est_tps:.0f} tok/s | tokens={rest_count} | {_metal_mem_str()}",
                    indent=1,
                )
                last_log_time = time.time()

    t = threading.Thread(target=_progress, daemon=True)
    t.start()
    return done, t


def _update_healing_store(raw_text: str, message_text: str, tool_calls: Optional[List]) -> None:
    """Store the raw (with <think>) response keyed by the stripped version's hash."""
    if raw_text == message_text:
        return
    h = _get_healing_hash(message_text, tool_calls)
    if not h:
        return
    with HEALING_STORE_LOCK:
        HEALING_STORE[h] = raw_text
        HEALING_STORE.move_to_end(h, last=True)
        while len(HEALING_STORE) > MAX_HEALING_STORE:
            HEALING_STORE.popitem(last=False)


def _prewarm_post_compact(
    summary_text: str,
    messages: List[Dict[str, Any]],
    request_id: str,
) -> None:
    """Pre-warm MAIN cache after compact using TPC + dual pipeline.

    The compact runner just generated a summary. The old MAIN cache is dead.
    This function builds a pre-warmed cache using the SAME dual pipeline as
    the main request path:
    - CANONICAL key (prompt_tokens space) for cache lookup matching
    - MODEL tokens for KV state computation (via TPC clone + delta prefill)

    Steps:
    1. Build messages [system, summary_user_msg] with stored tools
    2. Run dual pipeline: canonical → scrub → tokenize = canonical_key
    3. Run model pipeline: original → tokenize = model_tokens
    4. Clone TPC, prefill delta = model_tokens[tpc_len:]
    5. Evict dead MAIN + COMPACT, insert (canonical_key, new_cache)

    Must be called while model_lock is held (MLX is not thread-safe).
    """
    import copy

    if not _tpc.is_initialized():
        _terminal_status("⚠️",
            f"POST-COMPACT: TPC not available, skipping | req={request_id[:8]}")
        return

    if _LAST_MAIN_TOOLS is None or _LAST_MAIN_SYSTEM_BODY is None:
        _terminal_status("⚠️",
            f"POST-COMPACT: no MAIN tools/system captured yet, skipping | req={request_id[:8]}")
        return

    t0 = time.time()

    # 1. Build the expected post-compact messages
    from .server_compact import COMPACT_USER_WRAPPER

    summary_user_msg = {
        "role": "user",
        "content": COMPACT_USER_WRAPPER.format(summary=summary_text),
    }
    prewarm_messages = [
        {"role": "system", "content": _LAST_MAIN_SYSTEM_BODY},
        summary_user_msg,
    ]

    # 2. Dual pipeline — same as _handle_chat_completion (L4265-L4312)
    enable_thinking = SETTINGS.default_thinking
    original_msgs, canonical_msgs = _canonicalize_messages(prewarm_messages)
    original_msgs = _hoist_system_messages(original_msgs)
    canonical_msgs = _hoist_system_messages(canonical_msgs)
    original_msgs = _prepare_messages_for_template(original_msgs, SETTINGS.normalize_write_tool_content_for_prompt)
    canonical_msgs = _prepare_messages_for_template(canonical_msgs, SETTINGS.normalize_write_tool_content_for_prompt)

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        model_prompt = tokenizer.apply_chat_template(
            original_msgs, tokenize=False, add_generation_prompt=True,
            tools=_LAST_MAIN_TOOLS, enable_thinking=enable_thinking,
        )
        cache_prompt_raw = tokenizer.apply_chat_template(
            canonical_msgs, tokenize=False, add_generation_prompt=True,
            tools=_LAST_MAIN_TOOLS, enable_thinking=enable_thinking,
        )
    else:
        _terminal_status("⚠️",
            f"POST-COMPACT: no chat template, skipping | req={request_id[:8]}")
        return

    cache_prompt = _scrub_cache_key(cache_prompt_raw, SETTINGS.cache_canonicalize_tool_context)
    canonical_key = _tokenize_prompt(cache_prompt)
    model_tokens = _tokenize_prompt(model_prompt)

    # 3. Verify TPC prefix alignment with model_tokens
    tpc_tokens = list(_tpc._prefix_tokens)
    tpc_len = len(tpc_tokens)

    if len(model_tokens) <= tpc_len:
        _terminal_status("⚠️",
            f"POST-COMPACT: model_tokens({len(model_tokens)}) <= TPC({tpc_len}), skipping | req={request_id[:8]}")
        return

    if model_tokens[:tpc_len] != tpc_tokens:
        _terminal_status("⚠️",
            f"POST-COMPACT: TPC prefix mismatch, skipping | req={request_id[:8]}")
        return

    # 4. Clone TPC and prefill only the delta (summary tokens)
    new_cache = copy.deepcopy(_tpc._prefix_cache)
    delta_model_tokens = model_tokens[tpc_len:]

    try:
        ok = _tpc.prefill_cache_only(
            delta_model_tokens,
            model,
            new_cache,
            prefill_step_size=PREFILL_STEP_SIZE,
            kv_bits=SETTINGS.kv_bits,
            kv_group_size=64,
            quantized_kv_start=0,
        )
        if not ok:
            _terminal_status("⚠️",
                f"POST-COMPACT: prefill postcondition failed, skipping | req={request_id[:8]}")
            return
    except Exception as e:
        _terminal_status("⚠️",
            f"POST-COMPACT: delta prefill failed ({e}), skipping | req={request_id[:8]}")
        return

    # 5. Evict dead MAIN + COMPACT caches, insert with canonical key
    with prompt_cache_lock:
        PROMPT_CACHE.evict_unpinned()
        PROMPT_CACHE_COMPACT.evict_unpinned()
        PROMPT_CACHE.insert_cache(SETTINGS.model_path, canonical_key, new_cache)

    mx.clear_cache()

    elapsed_ms = (time.time() - t0) * 1000
    _terminal_status("🔥",
        f"POST-COMPACT PRE-WARMUP: TPC({tpc_len}) + delta({len(delta_model_tokens)}) → MAIN | "
        f"canonical_key={len(canonical_key)} | model_kv={len(model_tokens)} | "
        f"{elapsed_ms:.0f}ms | {_metal_mem_str()} | req={request_id[:8]}")





def _post_generation_cache_update(
    *,
    request_id: str,
    messages: List[Dict[str, Any]],
    prompt_tokens: List[int],
    cache_key: List[int],
    prompt_cache: Any,
    generated_tokens: List[int],
    tool_calls: Optional[List],
    matched_prefix_len: int,
    session_ctx: Any,
    session_id_for_turn: str,
    is_embedded_agent: bool,
    hybrid_checkpoint: Optional[HybridGenerationCheckpoint] = None,
    skip_cache_store: bool = False,
    housekeeping_pre_snapshot: Optional[Dict] = None,
    housekeeping_original_tokens: Optional[tuple] = None,
) -> None:
    """
    Shared post-generation logic: insert cache entries (MAIN or COMPACT),
    detect startup warmup candidate, update session turn store.
    Must be called while holding prompt_cache_lock.
    """
    _cache_hit_ratio = matched_prefix_len / max(len(prompt_tokens), 1) if prompt_tokens else 1.0

    if skip_cache_store:
        # BORROW & RETURN: if we have a pre-housekeeping snapshot, restore
        # the ArraysCache to its pre-housekeeping state and re-insert the
        # trie entry with the original key.  This preserves the cache for
        # the next normal request while still giving housekeeping good quality.
        if (
            FEATURE_HOUSEKEEPING_CACHE_BORROW
            and housekeeping_pre_snapshot is not None
            and housekeeping_original_tokens is not None
            and prompt_cache is not None
        ):
            _restored = rollback_arrays_cache(prompt_cache, housekeeping_pre_snapshot)
            if _restored:
                PROMPT_CACHE.insert_cache(
                    SETTINGS.model_path,
                    list(housekeeping_original_tokens),
                    prompt_cache,
                )
                _pipeline_log(
                    "CACHE",
                    request_id,
                    f"HOUSEKEEPING_BORROW: cache restored & re-inserted | "
                    f"original_key_len={len(housekeeping_original_tokens)} | "
                    f"kv_off={_kv_cache_offset(prompt_cache)}",
                )
            else:
                _pipeline_log(
                    "CACHE",
                    request_id,
                    "HOUSEKEEPING_BORROW: rollback FAILED — cache entry lost",
                )
        else:
            _pipeline_log(
                "CACHE",
                request_id,
                "HERMES_HOUSEKEEPING: cache store skipped — session turn store updated only",
            )
        if session_id_for_turn:
            _update_session_turn_store(session_id_for_turn, messages, prompt_tokens)
        return

    _diag_pre_restore = _cache_diag.snapshot(prompt_cache, cache_key, "before_hybrid_restore")
    prepared = (
        prepare_cache_for_insertion(
            prompt_cache,
            cache_key,
            generated_tokens,
            hybrid_checkpoint,
        )
        if prompt_cache is not None
        else None
    )
    if prepared is None:
        # Detailed diagnostics for cache discard
        if prompt_cache is None:
            _discard_reason = "prompt_cache is None"
        elif hybrid_checkpoint is None:
            _discard_reason = (
                f"hybrid_checkpoint is None | "
                f"can_trim={can_trim_prompt_cache(prompt_cache)} | "
                f"gen_tokens={len(generated_tokens)}"
            )
        else:
            # checkpoint existed but restore/validation failed
            _restore_ok = restore_hybrid_generation_checkpoint(prompt_cache, hybrid_checkpoint)
            _discard_reason = (
                f"restore_failed={not _restore_ok} | "
                f"ckpt_key_len={hybrid_checkpoint.cache_key_len} | "
                f"cache_key_len={len(cache_key)} | "
                f"kv_off={_kv_cache_offset(prompt_cache)}"
            )
        _pipeline_log(
            "CACHE",
            request_id,
            f"post-generation cache discarded: {_discard_reason}",
        )
        if session_id_for_turn:
            _update_session_turn_store(session_id_for_turn, messages, prompt_tokens)
        return

    cache_key, generated_tokens = prepared
    if hybrid_checkpoint is not None:
        _pipeline_log(
            "CACHE",
            request_id,
            "HYBRID_CHECKPOINT restored | "
            f"key={len(cache_key)} | kv_offset={_kv_cache_offset(prompt_cache)} | "
            "all recurrent layers rolled back",
        )
        _cache_diag.compare(_diag_pre_restore, prompt_cache, cache_key, "hybrid_restore", request_id,
            extra={"key_len": len(cache_key), "gen_tokens": len(generated_tokens)})
        # Attach verification metadata to the cache list so it survives
        # the trie store.  On the next lookup, Normal continuation can
        # verify that model_tokens[:_kv_off] hashes identically.
        try:
            prompt_cache.__model_prefix_hash__ = hybrid_checkpoint.model_prefix_hash
            prompt_cache.__model_offset__ = hybrid_checkpoint.model_offset
        except (TypeError, AttributeError):
            pass  # list subclasses or frozen objects — skip silently


    if not is_embedded_agent:
        # MAIN: any first-turn request with enough tokens qualifies for warmup save.
        # The old _STARTUP_MSG_PATTERN required phrases like "/new" or "session startup"
        # that Claude Code never sends — so warmup_cache.safetensors was never created.
        # Now: messages==2 (first turn) AND no cache on disk yet (or hash invalidated).
        # The 5000-token guard in _insert_cache_entries still filters noise.
        _is_real_startup = (
            len(messages) == 2
            and not _DPC.disk_cache_saved
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
        # COMPACT RUNNER: insert into isolated PROMPT_CACHE_COMPACT
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

    # M5: Update session turn record for next-turn stable-prefix lookup
    if session_id_for_turn:
        _update_session_turn_store(session_id_for_turn, messages, prompt_tokens)


def _build_timing_dict(
    first_token_at: Optional[float],
    generation_started_at: Optional[float],
    rest_count: int,
    generated_tokens: List[int],
) -> Dict[str, Any]:
    """Build the prefill/decode timing dict used by request_logger and telemetry."""
    timing: Dict[str, Any] = {}
    if first_token_at is not None and generation_started_at is not None:
        timing["prefill_seconds"] = first_token_at - generation_started_at
        timing["decode_seconds"] = time.time() - first_token_at
        timing["prefill_tps"] = (
            rest_count / timing["prefill_seconds"]
            if timing["prefill_seconds"] > 0 else None
        )
        timing["decode_tps"] = (
            len(generated_tokens) / timing["decode_seconds"]
            if timing["decode_seconds"] > 0 else None
        )
    return timing


def _log_generation_telemetry(
    request_id: str,
    generation_started_at: float,
    generated_tokens: List[int],
    message_text: str,
    tool_calls: Optional[List],
    enable_thinking: bool,
    finish_reason: str,
    is_embedded_agent: bool,
    timing: Dict[str, Any],
) -> None:
    """Shared post-generation pipeline logging (GEN + RESP + MSG_OUT)."""
    if FEATURE_LOG_GENERATION:
        _gen_ms = (time.time() - generation_started_at) * 1000
        _pipeline_log("GEN", request_id,
            f"finished: {len(generated_tokens)} tokens in {_gen_ms/1000:.2f}s "
            f"({timing.get('decode_tps', 0):.1f} tok/s decode)")
        _non_reas = len(_tokenize_prompt(message_text)) if message_text else 0
        _reas = max(0, len(generated_tokens) - _non_reas)
        _pipeline_log("GEN", request_id,
            f"thinking_tokens={_reas} | output_tokens={_non_reas}")
    else:
        _non_reas = len(_tokenize_prompt(message_text)) if message_text else 0
        _reas = max(0, len(generated_tokens) - _non_reas)

    if FEATURE_LOG_RESP:
        _pipeline_log("RESP", request_id,
            f"raw_text={len(generated_tokens)} tokens | normalize_applied=true")
        if tool_calls:
            _pipeline_log("RESP", request_id,
                f"tool_calls_extracted={len(tool_calls)} "
                f"{[tc.get('function', {}).get('name') for tc in tool_calls]}")
        if enable_thinking:
            _pipeline_log("RESP", request_id,
                f"think_block_stripped=true ({_reas} tokens hidden from client)")
        _pipeline_log("RESP", request_id,
            f"message_to_client={_non_reas} tokens | finish_reason={finish_reason}")
        _agent_tag_out = "EMBEDDED" if is_embedded_agent else "MAIN"
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
    is_warmup_candidate: bool = False,
) -> None:
    _store = prompt_cache_store_override if prompt_cache_store_override is not None else PROMPT_CACHE
    if (
        prompt_cache_store_override is None
        and SETTINGS.prompt_cache_max_entries_global <= 1
        and _store._entries
    ):
        _existing_len = max(
            (len(e.tokens) for e in _store._entries.values()), default=0
        )
        _new_len = len(cache_key)
        if _existing_len > _new_len * 1.5:
            # Existing entry is >50% larger — don't evict it.
            return

    # ── CLEAN CACHE BEFORE INSERTION ────────────────────────────────────────
    # After generation, prompt_cache contains KV states for prompt + response.
    # The next request will include the response as TEXT in its prompt, so the
    # response KV states in the cache are stale/wrong. Saving them causes
    # FIX-31 v10 to detect "contamination" and force a full cold start.
    #
    # Strategy depends on available cache slots:
    # - max_entries >= 2: deepcopy + trim → save BOTH clean and full versions
    # - max_entries == 1: trim IN-PLACE → save ONLY the clean version
    #
    # For hybrid caches (Qwen3 MoE: ArraysCache + KVCache), can_trim_prompt_cache
    # returns False because it requires ALL layers trimmable. In that case, we
    # trim only the trimmable KVCache layers per-layer (matching disk-save logic).
    if generated_tokens and len(cache_key) > len(generated_tokens):
        _diag_pre_insert = _cache_diag.snapshot(prompt_cache, cache_key, "before_insert_trim")
        if can_trim_prompt_cache(prompt_cache):
            # All layers trimmable (pure KVCache model)
            if SETTINGS.prompt_cache_max_entries_global >= 2 and (tool_calls or is_vlm):
                # Multi-slot: deepcopy for a prompt-only checkpoint
                try:
                    prompt_only_cache = copy.deepcopy(prompt_cache)
                    trim_prompt_cache(prompt_only_cache, len(generated_tokens))
                    prompt_only_key = cache_key[: -len(generated_tokens)]
                    _store.insert_cache(model_name, prompt_only_key, prompt_only_cache)
                    SESSION_INDEX.register_cache_key(session_ctx, prompt_only_key)
                except Exception:
                    pass
            else:
                # Single-slot: trim in-place
                try:
                    _pre_off = _kv_cache_offset(prompt_cache)
                    trim_prompt_cache(prompt_cache, len(generated_tokens))
                    cache_key = cache_key[: -len(generated_tokens)]
                    _post_off = _kv_cache_offset(prompt_cache)
                    _terminal_status("🧹",
                        f"Cache trim in-place (full): {_pre_off} → {_post_off} | "
                        f"stripped {len(generated_tokens)} response tokens")
                except Exception as _trim_err:
                    _terminal_status("⚠️",
                        f"Cache trim in-place FAILED ({_trim_err}) — saving as-is")
        else:
            # A hybrid cache can only be reused after restoring its pre-generation
            # ArraysCache checkpoint.  Partial KV-only trim publishes a false clean
            # key while recurrent layers still contain the response.
            _terminal_status(
                "⚠️",
                "Hybrid cache discarded: generated response reached insertion "
                "without a complete recurrent-state rollback",
            )
            return

    _store.insert_cache(model_name, cache_key, prompt_cache)
    SESSION_INDEX.register_cache_key(session_ctx, cache_key)
    # FIX-31 DIAG: after insertion — verify cache is clean
    if 'generated_tokens' in dir() and generated_tokens:
        _cache_diag.compare(
            _diag_pre_insert if '_diag_pre_insert' in dir() else None,
            prompt_cache, cache_key, "cache_insert", "",
            extra={"gen_tokens": len(generated_tokens), "key_len": len(cache_key)})

    # ── AUTO-SAVE MAIN CACHE TO DISK ──────────────────────────────────────────
    _active_store = prompt_cache_store_override if prompt_cache_store_override is not None else PROMPT_CACHE
    _is_compact_save = (prompt_cache_store_override is PROMPT_CACHE_COMPACT)

    if _is_compact_save:
        # ── AUTO-SAVE EMBEDDED/COMPACT CACHE ──────────────────────────────
        # Guardar el estado del compact runner en EMBEDDED_CACHE_PERSIST_PATH.
        # Only if: persist path configured, not yet saved (or low hit rate), and enough tokens.
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
            # Explicit diagnostic — always visible on real startup
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
                            # FIX-31 DIAG: verify frozen cache is clean after update
                            _cache_diag.compare(None, _save_cache, list(_prompt_only_key), "frozen_update", "",
                                extra={"layers": len(_save_cache), "tokens": len(_prompt_only_key)})
                    except Exception as _trim_err:
                        _terminal_status("⚠️", f"Auto-save ABORTED: per-layer trim failed ({_trim_err})")
                        _save_cache = None
                else:
                    _terminal_status("⚠️", f"Auto-save SKIPPED: no trimmable layers ({type(prompt_cache[0]).__name__ if prompt_cache else 'empty'})")
                    _save_cache = None
            if _save_cache is not None:
                # DPC: compute prefix hash from the prompt-only tokens for auto-healing
                _prefix_hash = _wm.compute_prefix_hash(
                    _prompt_only_key,
                    model_path=SETTINGS.model_path,
                    kv_bits=SETTINGS.kv_bits,
                )
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
        "presence_penalty": body.get(
            "presence_penalty", SETTINGS.default_presence_penalty
        ),
        "presence_context_size": body.get(
            "presence_context_size", SETTINGS.default_presence_context_size
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
    # Detect MoE models: load lazily to avoid materializing all experts (19.5 GB).
    # After module replacement, only non-expert params (~1.4 GB) are materialized.
    _is_moe_path = any(tag in SETTINGS.model_path for tag in ("A3B", "A14B", "MoE", "moe", "Mixtral", "mixtral", "Agents-A1"))
    _load_kwargs = {"tokenizer_config": {"trust_remote_code": True}}
    if _is_moe_path:
        _load_kwargs["lazy"] = True
        _terminal_status("🔧", "MoE model detected — loading with lazy=True (experts as placeholders)")
    try:
        model, tokenizer = load(SETTINGS.model_path, **_load_kwargs)
    except Exception as _offline_err:
        _terminal_status("⚠️", f"Offline load failed ({_offline_err}) — retrying with network...")
        os.environ.pop("HF_HUB_OFFLINE", None)
        model, tokenizer = load(SETTINGS.model_path, **_load_kwargs)
    _terminal_status("✅", "Model loaded (mlx-lm).")
    _terminal_status("⚡", "Torch acceleration: N/A (text-only model).")

    # Some models (e.g. Agents-A1) ship chat_template.jinja separately instead of
    # embedding it in tokenizer_config.json. Load it so tools get injected into the prompt.
    if not getattr(tokenizer, "chat_template", None):
        try:
            from huggingface_hub import hf_hub_download
            _jinja_path = hf_hub_download(SETTINGS.model_path, "chat_template.jinja")
            with open(_jinja_path, "r") as f:
                tokenizer.chat_template = f.read()
            _terminal_status("🔧", "Loaded chat_template.jinja into tokenizer (was missing from tokenizer_config)")
        except Exception:
            pass  # No jinja file available, model uses a different template mechanism

# ── MoE Expert Cache ──────────────────────────────────────────────────────
# Phase 3 predictive cache: lazy load → module replacement → selective expert
# materialization → zero-eval forward pass. See docs/MOE_EXPERT_CACHE.md.
_moe_stats = {}
try:
    from .expert_cache import is_moe_model, enable_moe_cache
    if is_moe_model(model):
        _terminal_status("🔧", "MoE model detected — enabling predictive expert cache...")
        # mx.eval of non-expert params happens INSIDE enable_moe_cache,
        # AFTER module replacement, to avoid materializing all 19.5 GB.
        _moe_stats = enable_moe_cache(
            model,
            SETTINGS.model_path,
            capacity=SETTINGS.moe_expert_capacity,
            profile_path=SETTINGS.moe_expert_profile or None,
        )
        _terminal_status(
            "✅",
            f"MoE Expert Cache: {_moe_stats.get('moe_layers', 0)} layers, "
            f"{_moe_stats.get('num_experts', 0)} experts, "
            f"cap={_moe_stats.get('capacity', 0)}, "
            f"loaded={_moe_stats.get('expert_tensors_loaded', 0)}, "
            f"mem={_moe_stats.get('active_memory_gb', 0):.1f}GB",
        )
        # ── Qwen3.6 full-residency guard ─────────────────────────────────
        # Qwen3.6-35B-A3B MUST run with all 256 experts resident.
        # Partial capacity causes severe quality degradation and hallucinations.
        _moe_cap = _moe_stats.get("capacity", 0)
        _moe_ne = _moe_stats.get("num_experts", 0)
        if _moe_cap < _moe_ne and "A3B" in SETTINGS.model_path:
            raise RuntimeError(
                f"Qwen3.6-35B-A3B requires ALL {_moe_ne} experts but only "
                f"{_moe_cap} are loaded. Set MOE_EXPERT_CAPACITY={_moe_ne} "
                f"and MOE_TARGET_CAPACITY={_moe_ne}. Refusing to start with "
                f"partial experts — quality degradation and hallucinations."
            )
        # Load historical expert frequency stats for breathing decisions
        try:
            from .expert_cache import apply_historical_frequency
            _stats_path = os.path.join("logs", "expert_stats.json")
            _seeded = apply_historical_frequency(model, _stats_path)
            if _seeded > 0:
                _terminal_status("📊", f"Expert frequency stats loaded: {_seeded} layers from previous sessions")
        except Exception as _freq_err:
            _terminal_status("⚠️", f"Historical frequency load failed: {_freq_err}")
except ImportError:
    pass  # expert_cache not available — dense model, no action needed
except Exception as _moe_err:
    _terminal_status("⚠️", f"MoE Expert Cache failed: {_moe_err}")

# Two-stage expert loading: expand after first successful response
_moe_expand_pending = (
    SETTINGS.moe_target_capacity > SETTINGS.moe_expert_capacity
    and _moe_stats.get("moe_layers", 0) > 0
)
if _moe_expand_pending:
    _terminal_status("📋",
        f"Staged loading: {SETTINGS.moe_expert_capacity}→{SETTINGS.moe_target_capacity} "
        f"experts after first warm response")
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
        from . import rag_enricher as _rag_mod
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
        # RAG index: blocking — SYSTEM READY appears only when the index is ready.
        if FEATURE_RAG_WORKSPACE_ROOT:
            try:
                _terminal_status("🔍", f"RAG: indexando workspace | root={FEATURE_RAG_WORKSPACE_ROOT}")
                _rag_mod.init(codebase_root=FEATURE_RAG_WORKSPACE_ROOT)
                _terminal_status("✅", "RAG: workspace index ready")
            except Exception as _idx_err:
                _terminal_status("⚠️", f"RAG: indexing failed ({_idx_err})")
        else:
            _terminal_status("ℹ️", "RAG: no RAG_WORKSPACE_ROOT set — enricher active but index empty")
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
            from . import rag_enricher as _comp_mod
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
            _terminal_status("⚠️", f"Compressor: partial preload ({_preload_err}) — will load lazily on first request")
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
            from . import rag_enricher as _comp_mod
            _compressor_module = _comp_mod
        try:
            _terminal_status("🗜️", "Emergency Compressor: precargando LLMLingua (CPU)...")
            _compressor_module._load_llmlingua()
            _terminal_status("✅", "Emergency Compressor: LLMLingua listo en CPU (standalone)")
        except Exception as _preload_err:
            _terminal_status("⚠️", f"Emergency Compressor: partial preload ({_preload_err}) — will load lazily")
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
    _t0 = time.perf_counter()
    _mem_before = mx.get_active_memory() / 1e9
    _terminal_status("⏱️", f"DPC warmup: iniciando | active_mem={_mem_before:.3f}GB")

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
    # Load tool prefix KV cache from disk (non-blocking — first request triggers
    # compute if no saved state exists).
    if SETTINGS.cache_persist_path:
        _tpc.init(SETTINGS.cache_persist_path)
        _tpc.load_from_disk(model, SETTINGS.max_kv_size)
        _load_hermes_context()

    _elapsed = time.perf_counter() - _t0
    _mem_after = mx.get_active_memory() / 1e9
    _mem_delta = _mem_after - _mem_before
    _terminal_status(
        "⏱️",
        f"DPC warmup: completo | elapsed={_elapsed:.1f}s | "
        f"active_mem={_mem_after:.3f}GB | delta={_mem_delta:+.3f}GB"
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

# ── Metal cache limit ────────────────────────────────────────────────────────
# Tell Metal to release freed GPU buffers above the model footprint.
# set_cache_limit(0) is too aggressive (causes alloc/dealloc thrashing during generation).
# ~6GB keeps model weights + warmup cached, but releases KV session buffers after use.
_METAL_CACHE_LIMIT_BYTES = int(6.5 * (1024 ** 3))  # 6.5 GB ≈ model + warmup
try:
    _set_cache_limit = getattr(mx, 'set_cache_limit', None) or getattr(mx.metal, 'set_cache_limit', None)
    if _set_cache_limit:
        _set_cache_limit(_METAL_CACHE_LIMIT_BYTES)
        _terminal_status("🧹", f"Metal cache limit: {_METAL_CACHE_LIMIT_BYTES / 1e9:.1f}GB (model + warmup)")
except Exception:
    pass


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


def _adaptive_prefill_chunk(base_chunk: int, kv_length: int, available_bytes: float) -> int:
    """Compute the largest safe chunk size given available Metal memory.

    Uses the empirically calibrated scratch formula:
        scratch = COEFF × N_LAYERS_SDPA × chunk × kv_length × N_HEADS × 4

    Solves for max chunk:
        chunk_max = available_bytes / (COEFF × N_LAYERS_SDPA × kv_length × N_HEADS × 4)

    Returns min(base_chunk, chunk_max), clamped to [_ADAPTIVE_PREFILL_MIN_CHUNK, base_chunk].
    """
    if kv_length <= 0:
        return base_chunk

    denominator = (
        _SCRATCH_COEFFICIENT
        * _N_LAYERS_SDPA
        * kv_length
        * _N_HEADS
        * _SCRATCH_BYTES_PER_ELEMENT
    )
    if denominator <= 0:
        return base_chunk

    chunk_max = int(available_bytes / denominator)
    chunk = max(_ADAPTIVE_PREFILL_MIN_CHUNK, min(base_chunk, chunk_max))
    return chunk


def _adaptive_prefill(
    rest_tokens: list,
    prompt_cache: list,
    request_id: str = "",
) -> list:
    """Pre-prefill rest_tokens[:-1] with adaptive chunk sizing.

    Returns the remaining tokens (just the last one) to pass to stream_generate.
    The cache is updated in-place.

    If rest_tokens has <= 1 token, returns rest_tokens unchanged (no prefill needed).
    """
    if len(rest_tokens) <= 1:
        return rest_tokens

    tokens_to_prefill = rest_tokens[:-1]
    last_token = rest_tokens[-1:]  # keep as list for stream_generate

    total = len(tokens_to_prefill)
    processed = 0
    prompt_array = mx.array(tokens_to_prefill)
    base_chunk = PREFILL_STEP_SIZE
    chunk_reductions = 0
    _prefill_t0 = time.time()
    _last_progress_tokens = 0
    _PROGRESS_INTERVAL = 2000  # Log every N tokens
    _diag_pre_prefill = _cache_diag.snapshot(prompt_cache, rest_tokens, "before_adaptive_prefill")

    while processed < total:
        # Compute available memory
        budget_bytes = _get_metal_budget_gb() * 1e9
        active_bytes = mx.get_active_memory()
        available = (budget_bytes - active_bytes) * _ADAPTIVE_PREFILL_SAFETY_MARGIN

        # Compute adaptive chunk size based on current kv_length
        # kv_length = tokens already in cache + tokens we've processed
        # Use the first KVCache's offset if available, else track manually
        kv_length = processed  # conservative: our contribution so far
        for c in prompt_cache:
            if hasattr(c, "offset"):
                kv_length = c.offset
                break

        chunk_size = _adaptive_prefill_chunk(base_chunk, kv_length, available)
        n = min(chunk_size, total - processed)

        if chunk_size < base_chunk and chunk_reductions == 0:
            _terminal_status(
                "📐",
                f"Adaptive prefill: chunk reduced {base_chunk}→{chunk_size} "
                f"at kv={kv_length} | avail={available / 1e9:.2f}GB | "
                f"req={request_id[:12] if request_id else '?'}",
            )
        if chunk_size < base_chunk:
            chunk_reductions += 1

        # Forward pass
        chunk = prompt_array[processed : processed + n][None]  # (1, n)
        model(chunk, cache=prompt_cache)

        # Materialize cache + free scratch
        for c in prompt_cache:
            if hasattr(c, "state"):
                mx.eval(c.state)
            else:
                mx.eval(c)
        mx.clear_cache()

        # GPU yield: let other Metal clients (Chrome VideoToolbox) process between chunks
        if _GPU_YIELD_SECONDS > 0:
            time.sleep(_GPU_YIELD_SECONDS)

        processed += n

        # Dark-zone illumination: log progress during large prefills
        if FEATURE_FULL_LOGGING and request_id and processed - _last_progress_tokens >= _PROGRESS_INTERVAL:
            _elapsed = time.time() - _prefill_t0
            _tps = processed / _elapsed if _elapsed > 0 else 0
            _pipeline_log("PREFILL", request_id,
                f"progress | {processed}/{total} tok ({processed*100//total}%) | "
                f"{_tps:.0f} tok/s | chunk={n} | {_metal_mem_str()}")
            _last_progress_tokens = processed

    if chunk_reductions > 0:
        _terminal_status(
            "📐",
            f"Adaptive prefill complete: {total} tokens | "
            f"chunk reductions={chunk_reductions} | req={request_id[:12] if request_id else '?'}",
        )

    _cache_diag.compare(_diag_pre_prefill, prompt_cache, rest_tokens, "adaptive_prefill", request_id,
        extra={"tokens": total, "chunks": chunk_reductions})
    return last_token


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
        "prefill_step_size": PREFILL_STEP_SIZE,
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

        for resp in stream_generate(
            **_stream_generate_kwargs(rest_tokens, max_tokens, sampler, prompt_cache)
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
            f"from={self.client_address[0]} | msgs={len(messages)} | stream={is_streaming} | max_tokens={max_tokens}",
            indent=1,
        )
        _last_content = str(messages[-1].get("content", ""))[:200] if messages else ""
        _terminal_status("🛸", f"Sidecar {request_id} | preview: '{_last_content}'", indent=2)

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

            prepared = _prepare_messages_for_template(messages, SETTINGS.normalize_write_tool_content_for_prompt)

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
            except Exception:
                pass
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"input_tokens": 0}).encode("utf-8"))
            return

        # ── Anthropic Messages API: inline translation ───────────────
        # Translate the Anthropic body to OpenAI format and fall through
        # to the same pipeline that handles /v1/chat/completions.
        if route in ("/v1/messages", "/messages"):
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

                # Claude Code compact: detect by user message phrase
                if raw.get("tools") and "CRITICAL: Respond with TEXT ONLY" in _last_user[:200]:
                    _compact_stripped_tools = len(raw["tools"])
                    raw["tools"] = []
                    _terminal_status("🪶",
                        f"COMPACT TOOL STRIP: removed {_compact_stripped_tools} tool definitions from compact request")

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
                    # Prepend Claude Code's structured COMPACT_PROMPT to the user message
                    # Hermes already serialized the conversation as user content — keep it,
                    # just replace the generic instruction with the structured one.
                    from .server_compact import COMPACT_PROMPT as _CC_COMPACT_PROMPT
                    for _m in reversed(raw.get("messages", [])):
                        if _m.get("role") == "user":
                            _original = _m.get("content", "")
                            if isinstance(_original, str):
                                # Strip Hermes's generic instruction, keep conversation data
                                # The conversation data typically starts after the first newline block
                                _conv_start = _original.find("\n\n")
                                _conv_data = _original[_conv_start:] if _conv_start > 0 else "\n\n" + _original
                                _m["content"] = _CC_COMPACT_PROMPT + _conv_data
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
                    _terminal_status("🪶",
                        f"HERMES COMPACT SWAP: replaced prompt + stripped {_compact_stripped_tools} tools"
                        f" | TPC context injected={_tpc_injected_into_compact}")

            body = anthropic_to_openai_body(raw, SETTINGS.proxy_model_id)

            if _hermes_compact_swapped:
                # Compact invalidates the entire conversation cache.
                # Evict BEFORE generation to free memory for the large compact
                # prefill and ensure no stale entries survive.
                with prompt_cache_lock:
                    PROMPT_CACHE.evict_unpinned()
                import gc; gc.collect()
                _terminal_status("🧹",
                    "COMPACT EVICTION: cleared PROMPT_CACHE — next request will rebuild via TPC warmup")
                # Signal to _handle_chat_completion to skip cache store.
                # The compact's cache (serialized conversation + instructions)
                # is a one-off and must NOT be stored — only the RESULT matters,
                # which Hermes will send as the next normal request.
                body["_supermlx_compact"] = True

            # Jump past the body-parse block that follows.
            self._handle_chat_completion(body)
            return

        if route not in ("/v1/chat/completions", "/chat/completions"):
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
        except Exception:
            self.send_error(400, "Bad Request")
            return
        self._handle_chat_completion(body)

    def _handle_chat_completion(self, body):

        # ── DPC GATE ────────────────────────────────────────────────────────
        # If DPC is configured and boot hasn't finished yet, the first request waits
        # until the cache hash is loaded. For all subsequent requests
        # _WARMUP_DONE.is_set() == True → no overhead (O(1) operation).
        # Timeout de 120s por si el boot falla: el request procede igual (cold start).
        if not _WARMUP_DONE.is_set() and SETTINGS.cache_persist_path:
            _wg_t0 = time.time()
            _WARMUP_DONE.wait(timeout=120)
            _wg_elapsed = time.time() - _wg_t0
            if _wg_elapsed > 0.5:  # Only log if actually waited
                with console_lock:
                    print(
                        f"  [WARMUP_GATE] {datetime.now().strftime('%H:%M:%S')} "
                        f"request queued {_wg_elapsed:.1f}s for warmup | "
                        f"cache_ready={_WARMUP_DONE.is_set()}",
                        flush=True,
                    )



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

        _is_housekeeping = _is_hermes_housekeeping_request(_raw_messages_for_detect)
        _is_compact = body.pop("_supermlx_compact", False)
        if _is_compact:
            _is_housekeeping = True  # reuse same skip_cache_store path
            _pipeline_log("CACHE", request_id,
                "HERMES_COMPACT: cache store will be skipped — cache was evicted pre-generation")
        elif _is_housekeeping:
            _pipeline_log("CACHE", request_id,
                "HERMES_HOUSEKEEPING: detected — cache store will be skipped to avoid contaminating conversation cache")

        reasoning_control = _extract_enable_thinking(body, default_thinking=SETTINGS.default_thinking)
        enable_thinking = reasoning_control["enable_thinking"]

        # ── PIPELINE LOG: INBOUND ──────────────────────────────────────────
        _pipeline_t0 = time.time()
        _pipeline_timings = {}  # stage -> ms
        raw_messages_inbound = body.get("messages", [])

        # ── TOOL RESULT SIZE LOGGING (pass-through, no truncation) ─────────
        # Previously truncated tool results > MAX_TOOL_RESULT_CHARS (20K chars),
        # silently mutilating skills (up to 80% loss) and data. Removed because:
        # 1. A model cannot truncate its own skills — it degrades silently
        # 2. The COMPACT GUARD (line ~4978) already handles oversized prompts
        #    by returning a standard 400 error that Hermes handles via compaction
        # 3. Per Gabriel Pinto: "the model isn't lying, it's answering from
        #    incomplete data because nothing in the input flagged the cut"
        # Ref: wiki/MLXServer/challenges/tool_result_truncation.md
        _large_tool_results = 0
        _largest_tool_chars = 0
        for _msg in raw_messages_inbound:
            if (_msg.get("role") or "").lower() == "tool":
                _tc = _msg.get("content", "")
                if isinstance(_tc, str) and len(_tc) > 20000:
                    _large_tool_results += 1
                    _largest_tool_chars = max(_largest_tool_chars, len(_tc))
        if _large_tool_results > 0:
            _pipeline_log("INBOUND", request_id,
                f"large_tool_results={_large_tool_results} | largest={_largest_tool_chars} chars "
                f"| pass-through (no truncation)")


        # Always estimate prompt tokens — needed for Anthropic usage reporting
        # even when FEATURE_FULL_LOGGING is off.
        _est_tok = _estimate_token_count(raw_messages_inbound)

        if FEATURE_FULL_LOGGING:
            _roles = _count_roles(raw_messages_inbound)
            _roles_str = ", ".join(f"{r}:{c}" for r, c in sorted(_roles.items()))
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
            healed_messages = _heal_messages(raw_messages, HEALING_STORE, HEALING_STORE_LOCK)

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
                    _, canonical_msgs_vlm = _canonicalize_messages(healed_messages, SETTINGS.cache_canonicalize_tool_context)
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
                        _canon_fmt = _scrub_cache_key(str(_canon_fmt), SETTINGS.cache_canonicalize_tool_context)
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
            if _compressor_available and FEATURE_COMPRESSOR and not _is_embedded_agent and not self._is_anthropic:
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
            _skip_rag = _is_embedded_agent or _is_rag_bypass_request(raw_messages) or self._is_anthropic
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
                _reason = "compact_runner" if _is_embedded_agent else ("anthropic" if self._is_anthropic else "rag_bypass(flush/summarize)")
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
            if FEATURE_HEALING:
                healed_messages = _heal_messages(raw_messages, HEALING_STORE, HEALING_STORE_LOCK)
            else:
                healed_messages = list(raw_messages)
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
                healed_messages, request_id,
                enabled=FEATURE_TOOL_LOOP_BREAKER,
                max_retries=TOOL_LOOP_MAX_RETRIES,
                log_fn=_pipeline_log if FEATURE_FULL_LOGGING else None,
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
            original_messages = _hoist_system_messages(original_messages)
            canonical_messages = _hoist_system_messages(canonical_messages)

            messages = _prepare_messages_for_template(original_messages, SETTINGS.normalize_write_tool_content_for_prompt)
            cache_messages = _prepare_messages_for_template(canonical_messages, SETTINGS.normalize_write_tool_content_for_prompt)

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
            cache_prompt = _scrub_cache_key(cache_prompt_raw, SETTINGS.cache_canonicalize_tool_context)
            # Safety invariant: cache key must not be dramatically shorter than the original.
            if SETTINGS.cache_norm_safety_check:
                if not _assert_cache_key_safety(
                    prompt, cache_prompt, context="lm_path",
                    log_fn=_terminal_status,
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
        # Always use server DEFAULT_MAX_TOKENS — Claude Code sends max_tokens=8192
        # which truncates long code generation. We override it entirely.
        max_tokens = SETTINGS.default_max_tokens
        # Compact runner (context compaction) needs short output.
        # Cap total budget to avoid wasting 35s on thinking for a summary.
        # Note: title generation is intercepted earlier by the NL fast path.
        if _is_embedded_agent:
            max_tokens = 256
        is_streaming = body.get("stream", False)

        acquired = False
        generated_tokens = []
        message_text = ""
        generation_started_at = None
        first_token_at = None
        queue_started_at = time.time()
        hybrid_generation_checkpoint: Optional[HybridGenerationCheckpoint] = None

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
                # KRIPPER DUAL-SLOT: Route to the correct LRU based on agent type.
                # COMPACT runner → PROMPT_CACHE_COMPACT (isolated, max_size=1)
                # MAIN agent    → PROMPT_CACHE         (HOT slot, never touched by compact)
                # This replaces the old evict_unpinned() approach which destroyed MAIN
                # cache before compact, causing 70-110s cold-starts on the next MAIN request.
                _active_cache_store = PROMPT_CACHE_COMPACT if _is_embedded_agent else PROMPT_CACHE
                _mem_profiler.snapshot(request_id, "PRE_CACHE", is_anthropic=self._is_anthropic, prompt_tokens=len(prompt_tokens) if prompt_tokens else None, model_tokens=len(model_tokens) if model_tokens else None)
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
                #
                # HOUSEKEEPING CACHE STRATEGY:
                # When HOUSEKEEPING_CACHE_BORROW=True: allow normal cache lookup
                # (good quality), snapshot beforehand, restore+re-insert after.
                # When False: bypass trie lookup (TPC cold start, lower quality).
                _diag_pre_lookup = _cache_diag.snapshot(prompt_cache, list(prompt_tokens), "before_cache_lookup")
                _housekeeping_pre_snapshot = None
                _housekeeping_original_tokens = None
                if _is_housekeeping and not FEATURE_HOUSEKEEPING_CACHE_BORROW:
                    prompt_cache = None
                    _rest_tokens_canonical = prompt_tokens
                    cache_session_tokens = prompt_tokens
                    cache_match_type = "miss"
                    matched_prefix_len = 0
                    cache_selection_source = "housekeeping_bypass"
                    if FEATURE_FULL_LOGGING:
                        _pipeline_log("CACHE", request_id,
                            "HOUSEKEEPING_BYPASS: skipping trie lookup to protect MAIN cache entry")
                else:
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
                _mem_profiler.snapshot(request_id, "POST_CACHE", is_anthropic=self._is_anthropic, rest_tokens=len(rest_tokens), kv_cache_offset=_kv_off, cache_hit_type=cache_match_type, matched_prefix=matched_prefix_len, prompt_tokens=len(prompt_tokens) if prompt_tokens else None)
                _cache_diag.compare(_diag_pre_lookup, prompt_cache, list(prompt_tokens), "cache_lookup", request_id,
                    extra={"hit": cache_match_type, "matched": matched_prefix_len, "rest": len(rest_tokens)})
                if FEATURE_FULL_LOGGING:
                    _pipeline_log("CACHE_LOOKUP", request_id,
                        f"completed | hit={cache_match_type} | kv_off={_kv_off} | "
                        f"rest={len(rest_tokens)} | matched={matched_prefix_len}/{len(prompt_tokens)}")

                # --- M3: Stable-prefix fallback ---
                # If the global cache lookup found fewer cached tokens than the
                # message-level diff says are stable, try a secondary lookup
                # using the stable prefix as the minimum acceptable match.
                # Only triggers when there is a real improvement available.
                if (
                    (not _is_housekeeping or FEATURE_HOUSEKEEPING_CACHE_BORROW)
                    and stable_prefix_token_len_computed > matched_prefix_len
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

                # --- HOUSEKEEPING BORROW: snapshot before prefill ---
                # If housekeeping found a cache entry (via main lookup or M3),
                # snapshot the ArraysCache NOW (before prefill modifies it).
                # After generation, we'll restore this snapshot and re-insert
                # the entry with cache_session_tokens as the key.
                if (
                    _is_housekeeping
                    and FEATURE_HOUSEKEEPING_CACHE_BORROW
                    and prompt_cache is not None
                    and cache_session_tokens is not None
                ):
                    _housekeeping_pre_snapshot = snapshot_arrays_cache(prompt_cache)
                    _housekeeping_original_tokens = tuple(cache_session_tokens)
                    _pipeline_log("CACHE", request_id,
                        f"HOUSEKEEPING_BORROW: snapshot captured | "
                        f"original_key_len={len(_housekeeping_original_tokens)} | "
                        f"snapshot_bytes={_housekeeping_pre_snapshot.get('nbytes', 0)}")
            
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
                # FIX-31 DIAG: snapshot before real-match sync
                _diag_pre_fix31 = _cache_diag.snapshot(prompt_cache, cache_key, "before_fix31_sync")
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
                
                _terminal_status("🐞", f"FIX-31 v4: _kv_off={_kv_off} type(layer)={type(prompt_cache[0] if prompt_cache else None)}",
                    request_id=request_id, stage="DEBUG")

                # ── Marconi early trim ────────────────────────────────────
                # With ArraysCache monkey-patched (is_trimmable=True), we can
                # trim the stale overflow BEFORE FIX-31. KVCache layers reduce
                # offset; ArraysCache layers rollback to checkpoint.
                #
                # CRITICAL: _kv_off is model-space, matched_prefix_len is
                # canonical-space. Compute model-space match point to avoid
                # the v14 space-mismatch bug.
                _canonical_rest = max(1, len(prompt_tokens) - matched_prefix_len)
                _model_match_point = _m_len - _canonical_rest  # model-space
                if (
                    _kv_off is not None
                    and _kv_off > _model_match_point
                    and can_trim_prompt_cache(prompt_cache)
                ):
                    _marconi_trim_n = _kv_off - _model_match_point
                    trim_prompt_cache(prompt_cache, _marconi_trim_n)
                    _kv_off = _kv_cache_offset(prompt_cache)
                    rest_tokens = model_tokens[-_canonical_rest:]
                    _terminal_status("✅",
                        f"Marconi trim: {_marconi_trim_n} stale tokens trimmed | "
                        f"kv_off={_kv_off} | rest={len(rest_tokens)}",
                        request_id=request_id, stage="DEBUG")
                    _cache_diag.compare(_diag_pre_fix31, prompt_cache, cache_key,
                        "marconi_trim", request_id,
                        extra={"trimmed": _marconi_trim_n, "kv_off": _kv_off})
                elif (
                    _kv_off is not None
                    and _kv_off <= _model_match_point
                    and can_trim_prompt_cache(prompt_cache)
                ):
                    # Marconi normal continuation: cache is aligned, just prefill delta
                    rest_tokens = model_tokens[-_canonical_rest:]
                    _terminal_status("✅",
                        f"Marconi: normal continuation | kv_off={_kv_off} | rest={len(rest_tokens)}",
                        request_id=request_id, stage="DEBUG")
                # ──────────────────────────────────────────────────────────
                elif _kv_off is not None:
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
                                    _terminal_status("🐞",
                                        f"FIX-31 v9: Pollution → restored frozen cache. "
                                        f"warmup={_warmup_len} | suffix={_suffix_len}",
                                        request_id=request_id, stage="DEBUG")
                                    _cache_diag.compare(_diag_pre_fix31, prompt_cache, cache_key, "restore_frozen", request_id,
                                        extra={"warmup": _warmup_len, "suffix": _suffix_len})
                                else:
                                    _suffix_len = _m_len
                                    prompt_cache = None
                                    PROMPT_CACHE.evict_unpinned()
                                    gc.collect()
                            else:
                                # No frozen cache available — full re-prefill is the only safe option.
                                _suffix_len = _m_len
                                prompt_cache = None
                                PROMPT_CACHE.evict_unpinned()
                                gc.collect()
                        else:
                            _suffix_len = max(1, len(prompt_tokens) - matched_prefix_len)
                            
                        rest_tokens = model_tokens[-_suffix_len:]
                        _terminal_status("🐞", f"FIX-31 v9: Sliding window trap bypassed. rest_tokens={_suffix_len}",
                            request_id=request_id, stage="DEBUG")
                        _cache_diag.compare(_diag_pre_fix31, prompt_cache, cache_key, "fix31_sliding_window", request_id,
                            extra={"suffix": _suffix_len, "kv_off": _kv_off, "m_len": _m_len})
                        
                    else:
                        # 2. Unified cache alignment
                        # Handles exact match, shorter match, pollution, and normal prefix
                        _canonical_suffix = max(1, len(prompt_tokens) - matched_prefix_len)
                        # FIX-31 v14: suffix must be in MODEL space (same as _kv_off and _m_len).
                        # Before: _suffix_len was in canonical space, _trim_to = _m_len - _suffix_len
                        # mixed spaces, causing off-by-N (typically ~27 tokens for Qwen3) that
                        # triggered v13 cold starts on every single request.
                        # Fix: compute model-space suffix = _m_len - _kv_off (how many model tokens
                        # the cache does NOT cover), then take the max with canonical_suffix.
                        _model_suffix = max(1, _m_len - _kv_off) if _kv_off is not None else _canonical_suffix
                        
                        if _cache_extended:
                            # FIX-31 v15: If the cache extension is tiny (< 128 response
                            # tokens residual), the recurrent state contamination is
                            # negligible against a 48K+ context. Skip the 512-token wash
                            # to avoid catastrophic cold starts (30K+ re-prefill).
                            _extension_size = _kv_off - _m_len if _kv_off > _m_len else 0
                            _MINOR_EXTENSION_THRESHOLD = 128
                            if _extension_size < _MINOR_EXTENSION_THRESHOLD:
                                # Minor extension: accept the slightly stale recurrent state.
                                # The model's attention over N correct KV tokens will dominate.
                                # Set _trim_to = _kv_off so the downstream `_kv_off > _trim_to`
                                # check is False, preventing entry into the wash/cold-start path.
                                _suffix_len = max(_model_suffix, 1)
                                _trim_to = _kv_off  # ensures _kv_off > _trim_to is False
                                rest_tokens = model_tokens[-_suffix_len:]
                                _terminal_status("🐞",
                                    f"FIX-31 v15: Minor extension ({_extension_size} tok) — "
                                    f"accepted | rest={len(rest_tokens)}",
                                    request_id=request_id, stage="DEBUG")
                                _cache_diag.compare(_diag_pre_fix31, prompt_cache, cache_key, "v15_minor_ext", request_id,
                                    extra={"extension": _extension_size, "rest": len(rest_tokens)})
                            else:
                                # For hybrid models with significant pollution, we need a
                                # min-suffix to wash out the recurrent state.
                                _MIN_POLLUTION_SUFFIX = 512
                                _suffix_len = max(_MIN_POLLUTION_SUFFIX, _model_suffix)
                                _terminal_status("🐞",
                                    f"FIX-31 v9: Cache polluted. canonical_suffix={_canonical_suffix} | "
                                    f"model_suffix={_model_suffix} | extension={_extension_size} | effective={_suffix_len}",
                                    request_id=request_id, stage="DEBUG")
                                # Fall through to _trim_to logic below
                                _trim_to = max(0, _m_len - _suffix_len)
                        elif not _cache_extended:
                            _suffix_len = _model_suffix
                            _trim_to = max(0, _m_len - _suffix_len)
                        else:
                            # v15 minor extension handled — _trim_to stays at _kv_off (no trim needed)
                            _trim_to = _kv_off
                        
                        if _kv_off > _trim_to:
                            if can_trim_prompt_cache(prompt_cache):
                                trim_prompt_cache(prompt_cache, _kv_off - _trim_to)
                                rest_tokens = model_tokens[_trim_to:]
                                _terminal_status("🐞", f"FIX-31 v9: Trimmed KVCache {_kv_off} -> {_trim_to}",
                                    request_id=request_id, stage="DEBUG")
                                _cache_diag.compare(_diag_pre_fix31, prompt_cache, cache_key, "trim", request_id,
                                    extra={"from": _kv_off, "to": _trim_to})
                            else:
                                # Hybrid wash: reprocess suffix while keeping the polluted KV as
                                # context. Risk: if KV is large, the attention computation for
                                # the suffix over N existing tokens requires O(suffix × KV) scratch.
                                # The scratch per attention layer ≈ prefill_chunk × KV × n_heads × 4 bytes.
                                # With PREFILL_STEP_SIZE=512 and 16 GQA heads:
                                #   512 × 100K × 16 × 4 = 3.3 GB/layer (safe on 24 GB)
                                # Previously hardcoded at 40K which forced catastrophic cold starts
                                # (128s + OOM risk) even when only 397 tokens needed washing.
                                #
                                # CRITICAL: Hybrid wash does NOT fix ArraysCache (GatedDeltaNet)
                                # contamination. Recurrent state accumulates ALL prior tokens —
                                # washing a suffix just adds on top of the wrong state.
                                # When DPC loaded a cache from a DIFFERENT session, the recurrent
                                # layers contain context from the old conversation, causing
                                # cross-session hallucinations. Force cold start in this case.
                                _has_recurrent = not can_trim_prompt_cache(prompt_cache)
                                _HYBRID_WASH_KV_LIMIT = 100_000
                                # _kv_off > _m_len means the physical KV holds tokens beyond
                                # the current request. But normal same-session continuation
                                # leaves a small overshoot (generated tokens from last turn).
                                # True DPC contamination diverges by thousands of tokens.
                                _DPC_CONTAMINATION_THRESHOLD = 2000
                                if _has_recurrent and _kv_off is not None and _kv_off > _m_len + _DPC_CONTAMINATION_THRESHOLD:
                                    # FIX-31 v11: Before cold-starting, check SESSION_TURN_STORE to
                                    # distinguish same-session shrink (Hermes compacted, Hermes varies
                                    # msg count) from genuine cross-session DPC contamination.
                                    #
                                    # SESSION_TURN_STORE is keyed by real session_id — NOT a content
                                    # hash. stable_prefix_token_len_computed (computed at line ~4539)
                                    # tells us how many tokens from the start of this prompt were
                                    # stable across the prior turn of THIS session.
                                    #
                                    # If ≥90% of the current prompt is stable from the same session,
                                    # the recurrent state accumulated so far is coherent — hybrid wash
                                    # of the delta is safe and avoids the 57s cold-start penalty.
                                    # Threshold 90% (not 98%) absorbs canonicalization variance.
                                    #
                                    # If no session_id, no prior turn, or low stability → genuine
                                    # contamination → cold start as before (v10 path).
                                    _is_same_session_shrink = (
                                        bool(_session_id_for_turn)
                                        and stable_prefix_token_len_computed > 0
                                        and stable_prefix_token_len_computed >= int(_m_len * 0.90)
                                    )
                                    if _is_same_session_shrink:
                                        _v11_delta = _m_len - stable_prefix_token_len_computed
                                        _v11_wash = max(512, _v11_delta)
                                        rest_tokens = model_tokens[-_v11_wash:]
                                        _terminal_status("🔧",
                                            f"FIX-31 v11: Same-session shrink "
                                            f"(stable={stable_prefix_token_len_computed}/{_m_len}"
                                            f"={stable_prefix_token_len_computed/_m_len:.1%}) "
                                            f"kv_off={_kv_off} — hybrid wash {_v11_wash} tok "
                                            f"(avoided cold start)")
                                    else:
                                        # FIX-31 v10 revised: Before cold-starting, try per-layer
                                        # KVCache trim to bring offset within range. ArraysCache
                                        # retains its recurrent state (same as hybrid wash for
                                        # smaller overshoots). Only cold-start if trim fails.
                                        _trim_excess = _kv_off - _m_len
                                        _n_trimmed_v10 = 0
                                        try:
                                            for layer in prompt_cache:
                                                if hasattr(layer, 'is_trimmable') and layer.is_trimmable() and hasattr(layer, 'trim'):
                                                    layer.trim(_trim_excess)
                                                    _n_trimmed_v10 += 1
                                        except Exception as _trim_err:
                                            _terminal_status("⚠️",
                                                f"FIX-31 v10: per-layer trim failed ({_trim_err})")
                                            _n_trimmed_v10 = 0

                                        if _n_trimmed_v10 > 0:
                                            # Trim succeeded — hybrid wash the suffix
                                            _post_trim_off = _kv_cache_offset(prompt_cache)
                                            _v10_wash = max(512, _canonical_suffix)
                                            rest_tokens = model_tokens[-_v10_wash:]
                                            _terminal_status("🧹",
                                                f"FIX-31 v10: KVCache trimmed {_kv_off} → {_post_trim_off} | "
                                                f"excess={_trim_excess} | trimmed {_n_trimmed_v10} layers | "
                                                f"hybrid wash {_v10_wash} tok (avoided cold start)")
                                        else:
                                            # No trimmable layers — genuine cold start
                                            prompt_cache = None
                                            if _is_embedded_agent:
                                                PROMPT_CACHE_COMPACT.evict_unpinned()
                                            else:
                                                PROMPT_CACHE.evict_unpinned()
                                            import gc; gc.collect()
                                            rest_tokens = model_tokens
                                            _terminal_status("⚠️",
                                                f"FIX-31 v10: ArraysCache contaminated by DPC "
                                                f"(kv_off={_kv_off} > request_len={_m_len}+{_DPC_CONTAMINATION_THRESHOLD}) — cold start",
                                                request_id=request_id, stage="WARN")
                                elif _kv_off is not None and _kv_off > _HYBRID_WASH_KV_LIMIT:
                                    prompt_cache = None
                                    if _is_embedded_agent:
                                        PROMPT_CACHE_COMPACT.evict_unpinned()
                                    else:
                                        PROMPT_CACHE.evict_unpinned()
                                    import gc; gc.collect()
                                    rest_tokens = model_tokens
                                    _terminal_status("⚠️",
                                        f"FIX-31 v9: KV too large for hybrid wash "
                                        f"({_kv_off} > {_HYBRID_WASH_KV_LIMIT}) — cold start",
                                        request_id=request_id, stage="WARN")
                                else:
                                    # FIX-31 v13: ArraysCache divergent — fail-closed cold start.
                                    # Hybrid wash cannot fix recurrent state contamination:
                                    # ArraysCache accumulates ALL prior tokens; washing a suffix
                                    # just adds on top of wrong state.  Additionally, the wash
                                    # produces offset = _kv_off + (_suffix_len - 1) which does
                                    # NOT equal len(model_tokens) - 1 when _kv_off ≠
                                    # len(model_tokens) - _suffix_len (Bug 4: healing shifts
                                    # matched_prefix by 1 relative to the previous checkpoint's
                                    # canonical_key, breaking the checkpoint invariant).
                                    # Cold start produces a clean cache with correct checkpoints.
                                    prompt_cache = None
                                    if _is_embedded_agent:
                                        PROMPT_CACHE_COMPACT.evict_unpinned()
                                    else:
                                        PROMPT_CACHE.evict_unpinned()
                                    import gc; gc.collect()
                                    rest_tokens = model_tokens
                                    _terminal_status("⚠️",
                                        f"FIX-31 v13: ArraysCache divergent "
                                        f"(kv_off={_kv_off} > trim_to={_trim_to}, "
                                        f"suffix={_suffix_len}) — cold start (fail-closed)",
                                        request_id=request_id, stage="WARN")

                        else:
                            # Normal continuation — _kv_off ≤ _trim_to.
                            # Defense-in-depth: verify model prefix hash if metadata
                            # was attached by a previous checkpoint.
                            _stored_hash = getattr(prompt_cache, '__model_prefix_hash__', None)
                            _stored_off = getattr(prompt_cache, '__model_offset__', None)
                            if (_stored_hash is not None
                                    and _stored_off is not None
                                    and _stored_off == _kv_off):
                                _current_hash = hash(tuple(model_tokens[:_kv_off]))
                                if _current_hash != _stored_hash:
                                    # Model prefix diverged despite canonical match — cold start.
                                    prompt_cache = None
                                    if _is_embedded_agent:
                                        PROMPT_CACHE_COMPACT.evict_unpinned()
                                    else:
                                        PROMPT_CACHE.evict_unpinned()
                                    import gc; gc.collect()
                                    rest_tokens = model_tokens
                                    _terminal_status("⚠️",
                                        f"FIX-31 v13: model prefix hash mismatch at offset {_kv_off} "
                                        f"— cold start (fail-closed)",
                                        request_id=request_id, stage="WARN")
                                else:
                                    rest_tokens = model_tokens[_kv_off:]
                                    _terminal_status("🐞", f"FIX-31 v9: Normal continuation from {_kv_off} (hash verified)",
                                        request_id=request_id, stage="DEBUG")
                            else:
                                rest_tokens = model_tokens[_kv_off:]
                                _terminal_status("🐞", f"FIX-31 v9: Normal continuation from {_kv_off}",
                                    request_id=request_id, stage="DEBUG")
                else:
                    # No KV offset available — full prefill.
                    rest_tokens = model_tokens
                    _terminal_status("🐞", "FIX-31 v4: Forced rest_tokens = model_tokens because _kv_off is None",
                        request_id=request_id, stage="DEBUG")

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

            # ── CAPTURE TOOLS + SYSTEM for post-compact pre-warmup ────────
            if not _is_embedded_agent and tools:
                global _LAST_MAIN_TOOLS, _LAST_MAIN_SYSTEM_BODY
                _LAST_MAIN_TOOLS = tools
                for _m in messages:
                    if (_m.get("role") or "").lower() == "system":
                        _LAST_MAIN_SYSTEM_BODY = _m.get("content", "")
                        break

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
                # ── TOOL PREFIX KV INJECTION ─────────────────────────────────
                # On LRU miss, inject pre-computed KV for system+tools (~33K tok).
                # Only for MAIN agent requests with tools (not compact runner).
                _tpc_injected = False
                if (
                    not _is_embedded_agent
                    and tools
                    and not is_vlm
                    and _tpc.is_configured()
                ):
                    try:
                        # Use the full system message (including billing header) for the prefix.
                        # Stripping the header caused tokenization to diverge from model_tokens.
                        # The hash invalidates on Claude Code version changes (rare, acceptable).
                        _sys_body = ""
                        for _m in messages:
                            if (_m.get("role") or "").lower() == "system":
                                _sys_body = _m.get("content", "")
                                break
                        if _sys_body:
                            _pc_clone, _ptoks, _recomputed = _tpc.get_prefix_cache_clone(
                                system_body=_sys_body,
                                tools=tools,
                                tokenizer=tokenizer,
                                model=cache_model,
                                max_kv_size=SETTINGS.max_kv_size,
                                kv_bits=getattr(SETTINGS, "kv_bits", None),
                                enable_thinking=enable_thinking,
                            )
                            if _pc_clone is not None and _ptoks:
                                _ptok_len = len(_ptoks)
                                # Verify prefix alignment: model_tokens must start with prefix
                                if (len(model_tokens) > _ptok_len
                                        and model_tokens[:_ptok_len] == _ptoks):
                                    prompt_cache = _pc_clone
                                    rest_tokens = model_tokens[_ptok_len:]
                                    _tpc_injected = True
                                    _terminal_status(
                                        "🔧",
                                        f"TPC: injected | prefix={_ptok_len} tok | "
                                        f"rest={len(rest_tokens)} tok | "
                                        f"recomputed={_recomputed}",
                                    )
                                    _cache_diag.compare(_diag_pre_fix31 if '_diag_pre_fix31' in dir() else None,
                                        prompt_cache, cache_key, "tpc_inject", request_id,
                                        extra={"prefix_len": _ptok_len, "recomputed": _recomputed})
                                else:
                                    _terminal_status(
                                        "⚠️",
                                        f"TPC: prefix mismatch — falling back to full prefill "
                                        f"(prefix_len={_ptok_len} model_len={len(model_tokens)})",
                                    )
                    except Exception as _tpc_err:
                        _terminal_status("⚠️", f"TPC: error — falling back to full prefill ({_tpc_err})")

                if not _tpc_injected:
                    prompt_cache = make_prompt_cache(
                        cache_model, max_kv_size=SETTINGS.max_kv_size
                    )
                    # Full miss: model prefills all original tokens (never canonical).
                    rest_tokens = model_tokens
                    _cache_diag.compare(_diag_pre_fix31 if '_diag_pre_fix31' in dir() else None,
                        prompt_cache, cache_key, "cold_start", request_id,
                        extra={"rest": len(rest_tokens)})


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
                # For Anthropic: skip LLMLingua compression (BERT 512-token limit)
                # but KEEP the overflow signal to prevent OOM crashes.
                if self._is_anthropic:
                    _emergency_result = None
                else:
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
                    healed_messages = _heal_messages(raw_messages, HEALING_STORE, HEALING_STORE_LOCK)
                    original_messages, canonical_messages = _canonicalize_messages(healed_messages, SETTINGS.cache_canonicalize_tool_context)
                    original_messages = _hoist_system_messages(original_messages)
                    canonical_messages = _hoist_system_messages(canonical_messages)
                    messages = _prepare_messages_for_template(original_messages, SETTINGS.normalize_write_tool_content_for_prompt)
                    cache_messages = _prepare_messages_for_template(canonical_messages, SETTINGS.normalize_write_tool_content_for_prompt)

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

                    cache_prompt = _scrub_cache_key(cache_prompt_raw, SETTINGS.cache_canonicalize_tool_context)
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

            # ── DIAGNOSTIC: prompt_tokens vs model_tokens divergence ──────
            # FIXME: temporary — remove after diagnosing 146K vs 34K bug on dense models
            if prompt_tokens and model_tokens and abs(len(prompt_tokens) - len(model_tokens)) > len(model_tokens) * 0.5:
                _terminal_status("🔍",
                    f"TOKEN DIVERGENCE: prompt_tokens={len(prompt_tokens)} model_tokens={len(model_tokens)} "
                    f"ratio={len(prompt_tokens)/max(1,len(model_tokens)):.2f}x | "
                    f"pt_type={type(prompt_tokens).__name__} mt_type={type(model_tokens).__name__} | "
                    f"pt[0]={prompt_tokens[0] if prompt_tokens else '?'} (type={type(prompt_tokens[0]).__name__ if prompt_tokens else '?'}) | "
                    f"mt[0]={model_tokens[0] if model_tokens else '?'} (type={type(model_tokens[0]).__name__ if model_tokens else '?'}) | "
                    f"cache_prompt_type={type(cache_prompt).__name__ if 'cache_prompt' in dir() else 'N/A'} "
                    f"cache_prompt_len={len(cache_prompt) if 'cache_prompt' in dir() and cache_prompt else 'N/A'}")

            # ── COMPACT GUARD: OVERFLOW + MEMORY PRESSURE ─────────────────
            # Applies to ALL requests unconditionally (including Anthropic).
            # Prevents OOM crashes by rejecting oversized prefills before they start.
            # Uses "prompt is too long" wording to trigger Claude Code's reactive compact.
            _memory_pressure = False
            _pressure_reason = ""
            if should_signal_overflow(rest_count):
                _memory_pressure = True
                _pressure_reason = f"rest={rest_count} tokens exceed safe prefill limit ({_max_safe_prefill()})"
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
                except Exception:
                    pass

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
                "⚙️",
                f"Generation started | wait={wait_seconds:.2f}s | prefill={rest_count} | "
                f"session={session_ctx.session_id[:16]} ({cache_selection_source}) | family={SETTINGS.model_family}",
                request_id=request_id, stage="GEN",
            )
            if FEATURE_FULL_LOGGING:
                _pipeline_log("PRE_GEN", request_id,
                    f"entering generation | rest={rest_count} | stream={is_streaming} | "
                    f"{_metal_mem_str()}")
            _mem_profiler.snapshot(request_id, "PRE_PREFILL", is_anthropic=self._is_anthropic, rest_tokens=rest_count, kv_cache_offset=_kv_off)
            _mem_profiler.reset_peak()

            if not is_streaming:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                if FEATURE_DIAGNOSTIC_HEADERS:
                    self.send_header("X-Pipeline-Compression-Ms", f"{_pipeline_timings.get('compress', 0):.1f}")
                    self.send_header("X-Pipeline-RAG-Ms", f"{_pipeline_timings.get('rag', 0):.1f}")
                    self.send_header("X-Pipeline-Heal-Ms", f"{_pipeline_timings.get('heal', 0):.1f}")
                self.end_headers()

                _pre_prefill_memory_relief(request_id, rest_count, is_embedded_agent=_is_embedded_agent)

                # Adaptive pre-prefill: process rest_tokens[:-1] with dynamic chunks,
                # pass only the last token to stream_generate (skips its fixed-step loop).
                if not is_vlm and len(rest_tokens) > 1:
                    rest_tokens = _adaptive_prefill(rest_tokens, prompt_cache, request_id)

                if not is_vlm:
                    hybrid_generation_checkpoint = _capture_hybrid_checkpoint_before_generation(
                        request_id=request_id,
                        prompt_cache=prompt_cache,
                        prompt_tokens=prompt_tokens,
                        model_tokens=model_tokens,
                        rest_tokens=rest_tokens,
                    )

                _prefill_done, _prefill_thread = _start_prefill_progress(request_id, rest_count)

                generated_parts = []
                _thinking_token_count_ns = 0
                _think_close_count_ns = 0
                _in_think_ns = False
                _max_thinking_ns = 128 if _is_embedded_agent else SETTINGS.max_thinking_tokens
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
                    if _GPU_YIELD_SECONDS > 0:
                        time.sleep(_GPU_YIELD_SECONDS)
                    if first_token_at is None:
                        first_token_at = time.time()
                        _prefill_done.set()  # Stop prefill progress
                    # Track think state for token limit
                    if "<think>" in response.text:
                        _in_think_ns = True
                    if "</think>" in response.text:
                        _in_think_ns = False
                        _think_close_count_ns += 1
                    # ── REPEATED </think> BREAK (non-stream) ─────────
                    # After the first </think>, any subsequent </think>
                    # means the model is looping. Stop immediately.
                    if _think_close_count_ns >= 2:
                        # Truncate to content before second </think>
                        _full = "".join(generated_parts)
                        _first_c = _full.index("</think>")
                        _second_c = _full.index("</think>", _first_c + len("</think>"))
                        generated_parts.clear()
                        generated_parts.append(_full[:_second_c].rstrip())
                        _terminal_status(
                            "🛑",
                            f"THINK_LOOP_BREAK: second </think> detected after "
                            f"{len(generated_tokens)} tokens. Model is looping — "
                            f"stopping generation.",
                            indent=1,
                        )
                        _pipeline_log("GEN", request_id,
                            f"THINK_LOOP_BREAK: repeated </think> at token "
                            f"{len(generated_tokens)}. Truncated and stopped.")
                        break
                    if _in_think_ns and _max_thinking_ns > 0:
                        _thinking_token_count_ns += 1
                        if _thinking_token_count_ns >= _max_thinking_ns:
                            _terminal_status(
                                "🛑",
                                f"THINKING LIMIT: {_thinking_token_count_ns} tokens in <think> "
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
                        _terminal_status(
                            "⏳",
                            f"generated_tokens={len(generated_tokens)} | {_decode_tps:.1f} tok/s | {_metal_mem_str()}",
                            request_id=request_id, stage="DECODE",
                        )
                # ── THINKING CLEANUP (non-stream) ─────────────────────────
                # If generation ended while still inside <think> (e.g. THINKING_LIMIT
                # or NGRAM_LOOP broke mid-thinking), inject synthetic </think> so
                # downstream _strip_thinking_from_content can properly strip it.
                # If the model jumped straight to a tool call (bare function=)
                # without closing </think>, insert </think> BEFORE the function
                # call and wrap with <tool_call> if the opening tag is missing.
                if enable_thinking and _in_think_ns and generated_parts:
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
                # Hide <think> blocks from the client whenever reasoning was requested.
                if enable_thinking:
                    message_text = _strip_thinking_from_content(message_text)

                    _update_healing_store(raw_response_text, message_text, tool_calls)

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
                cache_key.extend(generated_tokens)
                with prompt_cache_lock:
                    _post_generation_cache_update(
                        request_id=request_id,
                        messages=messages,
                        prompt_tokens=prompt_tokens,
                        cache_key=cache_key,
                        prompt_cache=prompt_cache,
                        generated_tokens=generated_tokens,
                        tool_calls=tool_calls,
                        matched_prefix_len=matched_prefix_len,
                        session_ctx=session_ctx,
                        session_id_for_turn=_session_id_for_turn,
                        is_embedded_agent=_is_embedded_agent,
                        hybrid_checkpoint=hybrid_generation_checkpoint,
                        skip_cache_store=_is_housekeeping,
                        housekeeping_pre_snapshot=_housekeeping_pre_snapshot,
                        housekeeping_original_tokens=_housekeeping_original_tokens,
                    )

                # POST-COMPACT PRE-WARMUP: after compact finishes, pre-warm MAIN
                # with the summary so the next request doesn't cold-start 50K tokens.
                # Guard: skip if MAIN already has a warm cache larger than TPC —
                # title/slug generators are classified as compact runners but don't
                # invalidate MAIN (they run in PROMPT_CACHE_COMPACT).
                _main_max_len = max(
                    (len(e.tokens) for e in PROMPT_CACHE._entries.values()), default=0
                ) if PROMPT_CACHE._entries else 0
                _tpc_len = len(_tpc._prefix_tokens) if _tpc.is_initialized() else 0
                if _is_embedded_agent and message_text and len(message_text) > 100 and _main_max_len <= _tpc_len:
                    try:
                        _prewarm_post_compact(
                            summary_text=message_text,
                            messages=messages,
                            request_id=request_id,
                        )
                    except Exception as _pw_err:
                        _terminal_status("⚠️",
                            f"POST-COMPACT PRE-WARMUP failed: {_pw_err} | req={request_id[:8]}")



                # Expert Breathing: restore full capacity after generation
                global _breathing_active
                if _breathing_active:
                    try:
                        from .expert_cache import breathe_up
                        breathe_up(model, log_fn=_terminal_status)
                    except Exception as _bu_err:
                        _terminal_status("⚠️", f"BREATHE UP failed: {_bu_err} | req={request_id[:8]}")
                    finally:
                        _breathing_active = False

                if self._is_anthropic:
                    full_response = openai_to_anthropic_response(
                        message_text, tool_calls, finish_reason,
                        self._anthropic_model,
                        prompt_input_tokens=_est_tok,
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
                timing = _build_timing_dict(first_token_at, generation_started_at, rest_count, generated_tokens)
                if request_logger:
                    _think_log = _extract_thinking_text(raw_response_text)
                    request_logger.log(
                        "generation",
                        {
                            "mode": "non-stream",
                            "timing": timing,
                            "thinking_text": _think_log,
                            "thinking_tokens": len(_think_log) // 4 if _think_log else 0,
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
                        _is_embedded_agent, timing,
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
                _seen_think_close = False  # True only after </think> is detected in stream
                _anthropic_streamed_text = []  # chunks already sent via SSE
                _anthropic_block_idx = 0
                _anthropic_thinking_streamed = []  # thinking chunks for the thinking block
                if _anthropic_streaming:
                    _msg_id = f"msg_{uuid.uuid4().hex[:24]}"
                    _msg_start = _sse_event("message_start", {
                        "type": "message_start",
                        "message": {
                            "id": _msg_id, "type": "message", "role": "assistant",
                            "model": self._anthropic_model, "content": [],
                            "stop_reason": None, "stop_sequence": None,
                            "usage": {"input_tokens": _est_tok, "output_tokens": 0},
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
                        _seen_think_close = True  # No thinking expected
                    self.wfile.write(_msg_start.encode("utf-8"))
                    self.wfile.write(_block_start.encode("utf-8"))
                    self.wfile.flush()
                    _pipeline_log("WIRE", request_id, f"ANTHROPIC SSE: sent message_start + content_block_start (thinking={enable_thinking})")
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

                # --- SSE KEEPALIVE THREAD ---
                # During prefill (~58s) and decode, no SSE events are sent because
                # we must collect all tokens to post-process (strip <think>, extract
                # tool_calls).  Without keepalives the idle TCP connection dies and
                # LiteLLM / OpenClaw sees "Connection error".
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

                _pre_prefill_memory_relief(request_id, rest_count, is_embedded_agent=_is_embedded_agent)

                # Adaptive pre-prefill: process rest_tokens[:-1] with dynamic chunks,
                # pass only the last token to stream_generate (skips its fixed-step loop).
                if not is_vlm and len(rest_tokens) > 1:
                    rest_tokens = _adaptive_prefill(rest_tokens, prompt_cache, request_id)

                if not is_vlm:
                    hybrid_generation_checkpoint = _capture_hybrid_checkpoint_before_generation(
                        request_id=request_id,
                        prompt_cache=prompt_cache,
                        prompt_tokens=prompt_tokens,
                        model_tokens=model_tokens,
                        rest_tokens=rest_tokens,
                    )

                _prefill_done_s, _ = _start_prefill_progress(request_id, rest_count)

                raw_parts = []
                _thinking_token_count = 0
                _max_thinking = 128 if _is_embedded_agent else SETTINGS.max_thinking_tokens  # 0=unlimited

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
                        if _GPU_YIELD_SECONDS > 0:
                            time.sleep(_GPU_YIELD_SECONDS)
                        if first_token_at is None:
                            first_token_at = time.time()
                            _prefill_done_s.set()  # Stop prefill progress
                        response_text = response.text

                        # ── THINKING TOKEN LIMIT ──────────────────────────
                        # Count tokens while still inside <think> block.
                        # When limit is exceeded, break generation to prevent
                        # circular reasoning loops that produce 0 output.
                        if not _seen_think_close and _max_thinking > 0:
                            _thinking_token_count += 1
                            if _thinking_token_count >= _max_thinking:
                                _terminal_status(
                                    "🛑",
                                    f"THINKING LIMIT: {_thinking_token_count} tokens in <think> block "
                                    f"(limit={_max_thinking}). Forcing generation stop.",
                                    indent=1,
                                )
                                _pipeline_log("GEN", request_id,
                                    f"THINKING_LIMIT_HIT: {_thinking_token_count} thinking tokens, "
                                    f"limit={_max_thinking}. Breaking generation loop.")
                                break

                        # ── N-GRAM LOOP DETECTION ────────────────────────
                        # Detect verbatim phrase repetition in generated output.
                        # Every _NGRAM_CHECK_INTERVAL tokens, check if the last
                        # _NGRAM_SIZE tokens appeared earlier in RECENT output.
                        # Uses a proximity window to avoid false positives on
                        # code generation where identical patterns (HTML tags,
                        # repeated function signatures) appear far apart.
                        # Real infinite loops repeat every ~20-40 tokens;
                        # a 300-token window catches those easily.
                        _NGRAM_SIZE = 20         # tokens per n-gram
                        _NGRAM_MAX_REPEATS = 3   # max allowed repeats before break
                        _NGRAM_CHECK_INTERVAL = 32  # check every N tokens
                        _NGRAM_WINDOW = 300      # only search this far back for repeats
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
                            # ── ANTHROPIC LIVE STREAMING ─────────────────────
                            # The chat template puts <think>\n in the PROMPT when
                            # enable_thinking=True. Model output is the continuation
                            # AFTER that tag — so output never starts with <think>.
                            # All output is thinking until </think> appears.
                            if _anthropic_streaming and not _seen_think_close:
                                _acc = "".join(raw_parts)
                                if "</think>" in _acc:
                                    _seen_think_close = True
                                    # Close thinking block (index 0)
                                    with _wfile_lock:
                                        self.wfile.write(_sse_event("content_block_stop", {
                                            "type": "content_block_stop", "index": 0,
                                        }).encode("utf-8"))
                                        # Open text block (index 1)
                                        _anthropic_block_idx = 1
                                        self.wfile.write(_sse_event("content_block_start", {
                                            "type": "content_block_start", "index": 1,
                                            "content_block": {"type": "text", "text": ""},
                                        }).encode("utf-8"))
                                        # NOTE: Do NOT stream _post_think as text_delta here.
                                        # The text after </think> may contain tool call XML
                                        # fragments (e.g. malformed `function=browser_navigate>`).
                                        # Streaming it raw would send corrupted content as visible
                                        # text to Hermes, which stores it in history and feeds it
                                        # back to the model as context — causing progressive
                                        # degradation.  Instead, SSE_TEXT_RECOVERY (below) sends
                                        # the post-processed message_text after tool extraction.
                                        self.wfile.flush()
                                else:
                                    # Still in thinking — stream as thinking_delta
                                    _new_think = response_text
                                    if _new_think:
                                        with _wfile_lock:
                                            self.wfile.write(_sse_event("content_block_delta", {
                                                "type": "content_block_delta", "index": 0,
                                                "delta": {"type": "thinking_delta", "thinking": _new_think},
                                            }).encode("utf-8"))
                                            self.wfile.flush()
                                        _anthropic_thinking_streamed.append(_new_think)
                            elif _anthropic_streaming and _seen_think_close:
                                # ── REPEATED </think> BREAK ──────────────────
                                # After the first </think>, any subsequent </think>
                                # means the model is looping (generating multiple
                                # response blocks). Stop generation immediately
                                # to free the GPU and avoid sending duplicates.
                                _post_acc = "".join(raw_parts)
                                # Count </think> in full accumulated text — first one
                                # is the real close, any more = loop
                                if _post_acc.count("</think>") >= 2:
                                    # Truncate raw_parts to content before second </think>
                                    _first_close = _post_acc.index("</think>")
                                    _second_close = _post_acc.index("</think>", _first_close + len("</think>"))
                                    _clean = _post_acc[:_second_close].rstrip()
                                    raw_parts.clear()
                                    raw_parts.append(_clean)
                                    _terminal_status(
                                        "🛑",
                                        f"THINK_LOOP_BREAK: second </think> detected after "
                                        f"{len(generated_tokens)} tokens. Model is looping — "
                                        f"stopping generation.",
                                        indent=1,
                                    )
                                    _pipeline_log("GEN", request_id,
                                        f"THINK_LOOP_BREAK: repeated </think> at token "
                                        f"{len(generated_tokens)}. Truncated and stopped.")
                                    break

                                # Suppress raw <tool_call> XML from being streamed as text.
                                # The post-generation code extracts tool_calls and sends
                                # proper Anthropic tool_use blocks. Track open/close tags
                                # so text AFTER a closed </tool_call> resumes streaming.
                                _acc_text = _post_acc
                                _acc_lower = _acc_text.lower()
                                _in_tool_block = (
                                    bool(tools)
                                    and _acc_lower.count("<tool_call") > _acc_lower.count("</tool_call>")
                                )
                                if _in_tool_block:
                                    # Inside an unclosed tool_call — suppress text streaming.
                                    pass
                                else:
                                    _delta_ev = _sse_event("content_block_delta", {
                                        "type": "content_block_delta", "index": _anthropic_block_idx,
                                        "delta": {"type": "text_delta", "text": response_text},
                                    })
                                    try:
                                        with _wfile_lock:
                                            self.wfile.write(_delta_ev.encode("utf-8"))
                                            self.wfile.flush()
                                        _anthropic_streamed_text.append(response_text)
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
                            _terminal_status(
                                "⏳",
                                f"generated_tokens={len(generated_tokens)} | {_decode_tps:.1f} tok/s | {_metal_mem_str()}",
                                request_id=request_id, stage="DECODE",
                            )
                finally:
                    # Stop keepalive thread BEFORE writing actual content chunks
                    _keepalive_stop.set()
                    _keepalive_thread.join(timeout=2)

                # ── THINKING CLEANUP (stream) ──────────────────────────────
                # If generation ended while still inside <think> (e.g. THINKING_LIMIT
                # or NGRAM_LOOP broke mid-thinking), inject synthetic </think> and
                # properly transition Anthropic SSE blocks.
                # If the model jumped straight to a tool call (bare function=)
                # without closing </think>, insert </think> BEFORE the function
                # call and wrap with <tool_call> if the opening tag is missing.
                if enable_thinking and not _seen_think_close and raw_parts:
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
                    _seen_think_close = True
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
                # Hide <think> blocks from the client whenever reasoning was requested.
                if enable_thinking:
                    message_text = _strip_thinking_from_content(message_text)

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
                    if _anthropic_streaming and _seen_think_close:
                        # FIX: Qwen3.6 sometimes generates the response text
                        # BEFORE </think> (no reasoning, just answer + closing
                        # tag).  During streaming, all tokens were sent as
                        # thinking_delta because _seen_think_close was False.
                        # The post-processed message_text has the correct answer
                        # but it was never sent as text_delta.  Send it now.
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
                                f"that were streamed as thinking_delta")
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
                                _closing_events.append(_sse_event("content_block_delta", {
                                    "type": "content_block_delta", "index": _tc_idx,
                                    "delta": {
                                        "type": "input_json_delta",
                                        "partial_json": json.dumps(tc_input, ensure_ascii=False),
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
                            prompt_input_tokens=_est_tok,
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
                timing = _build_timing_dict(first_token_at, generation_started_at, rest_count, generated_tokens)
                if request_logger:
                    _think_log = _extract_thinking_text(raw_full_text)
                    request_logger.log(
                        "generation",
                        {
                            "mode": "stream",
                            "timing": timing,
                            "thinking_text": _think_log,
                            "thinking_tokens": len(_think_log) // 4 if _think_log else 0,
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
                        _is_embedded_agent, timing,
                    )

                # ── DEFERRED POST-PROCESSING ──────────────────────────────
                # These operations are heavy (cache trim with mx.eval, healing
                # store, pre-warmup) and were previously run BEFORE sending
                # closing events, causing a 2+ second SSE gap that could make
                # the client time out. Now they run AFTER the response is
                # fully delivered to the client.
                if enable_thinking:
                    _update_healing_store(raw_full_text, message_text, tool_calls)

                cache_key.extend(generated_tokens)
                with prompt_cache_lock:
                    _post_generation_cache_update(
                        request_id=request_id,
                        messages=messages,
                        prompt_tokens=prompt_tokens,
                        cache_key=cache_key,
                        prompt_cache=prompt_cache,
                        generated_tokens=generated_tokens,
                        tool_calls=tool_calls,
                        matched_prefix_len=matched_prefix_len,
                        session_ctx=session_ctx,
                        session_id_for_turn=_session_id_for_turn,
                        is_embedded_agent=_is_embedded_agent,
                        hybrid_checkpoint=hybrid_generation_checkpoint,
                        skip_cache_store=_is_housekeeping,
                        housekeeping_pre_snapshot=_housekeeping_pre_snapshot,
                        housekeeping_original_tokens=_housekeeping_original_tokens,
                    )

                # POST-COMPACT PRE-WARMUP: after compact finishes, pre-warm MAIN
                # with the summary so the next request doesn't cold-start 50K tokens.
                # Guard: skip if MAIN already has a warm cache larger than TPC.
                _main_max_len = max(
                    (len(e.tokens) for e in PROMPT_CACHE._entries.values()), default=0
                ) if PROMPT_CACHE._entries else 0
                _tpc_len = len(_tpc._prefix_tokens) if _tpc.is_initialized() else 0
                if _is_embedded_agent and message_text and len(message_text) > 100 and _main_max_len <= _tpc_len:
                    try:
                        _prewarm_post_compact(
                            summary_text=message_text,
                            messages=messages,
                            request_id=request_id,
                        )
                    except Exception as _pw_err:
                        _terminal_status("⚠️",
                            f"POST-COMPACT PRE-WARMUP failed: {_pw_err} | req={request_id[:8]}")

                # Expert Breathing: restore full capacity after generation
                if _breathing_active:
                    try:
                        from .expert_cache import breathe_up
                        breathe_up(model, log_fn=_terminal_status)
                    except Exception as _bu_err:
                        _terminal_status("⚠️", f"BREATHE UP failed: {_bu_err} | req={request_id[:8]}")
                    finally:
                        _breathing_active = False

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
                except Exception:
                    pass
                try:
                    mx.clear_cache()
                    import gc; gc.collect()
                except Exception:
                    pass
            else:
                _terminal_status("❌", f"Request {request_id} failed: {e}", indent=1)
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
                    _req_log = (
                            f"Request {request_id} finished | output_tokens={token_breakdown} | "
                            f"elapsed={elapsed:.2f}s | tok/s={speed:.2f} | "
                            f"prefill={prefill_seconds:.2f}s ({prefill_tps:.0f} tok/s) | "
                            f"decode={decode_seconds:.2f}s ({decode_tps:.1f} tok/s)"
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
                    except (ImportError, Exception):
                        pass
                    # Metal memory snapshot — monitor fragmentation and pressure
                    _mem_suffix = ""
                    try:
                        _m_active = mx.get_active_memory() / 1e9
                        _m_cache = mx.get_cache_memory() / 1e9
                        _m_peak = mx.get_peak_memory() / 1e9
                        _mem_suffix = f" | mem={_m_active:.1f}GB active/{_m_cache:.1f}GB cache/{_m_peak:.1f}GB peak"
                    except Exception:
                        pass
                    _terminal_status("✅", _req_log + _moe_suffix + _mem_suffix, indent=1)
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

                _mem_profiler.snapshot(request_id, "POST_GENERATION", is_anthropic=self._is_anthropic, rest_tokens=rest_count if 'rest_count' in dir() else None, output_tokens=output_tokens if 'output_tokens' in dir() else None, kv_cache_offset=_kv_off if '_kv_off' in dir() else None)
                mx.clear_cache()
                import gc; gc.collect()
                if FEATURE_FULL_LOGGING:
                    _pipeline_log("METAL", request_id, f"mx.clear_cache() + gc.collect() called"
                        f"{' | COMPACT slot retained in PROMPT_CACHE_COMPACT' if _is_embedded_agent else ''}")
                    if generation_started_at is not None:
                        _held_ms = (time.time() - generation_started_at) * 1000
                        _pipeline_log("METAL", request_id, f"model_lock released | held_for={_held_ms/1000:.2f}s")

                # ── TWO-STAGE EXPERT EXPANSION ────────────────────────────
                # After first successful response, expand expert capacity.
                # First request runs at cap=100 (safe for 33K cold prefill),
                # subsequent requests use expanded cap with warm cache.
                global _moe_expand_pending
                if _moe_expand_pending and output_tokens > 0:
                    try:
                        from .expert_cache import expand_expert_capacity
                        _terminal_status("🔄",
                            f"Expanding experts: {SETTINGS.moe_expert_capacity}"
                            f"→{SETTINGS.moe_target_capacity}...")
                        _expand_stats = expand_expert_capacity(
                            model,
                            target_capacity=SETTINGS.moe_target_capacity,
                            profile_path=SETTINGS.moe_expert_profile or None,
                        )
                        if _expand_stats.get("expanded"):
                            _terminal_status("✅",
                                f"Expert expansion complete: "
                                f"{_expand_stats['old_capacity']}→{_expand_stats['new_capacity']} | "
                                f"+{_expand_stats['new_experts_loaded']} experts | "
                                f"{_expand_stats['elapsed_seconds']}s | "
                                f"mem={_expand_stats['active_memory_gb']}GB")
                    except Exception as _exp_err:
                        _terminal_status("⚠️", f"Expert expansion failed: {_exp_err}")
                    _moe_expand_pending = False

                model_lock.release()


def run():
    #start_litellm_proxy()
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

    # Two-stage expert expansion: handled in post-response hook (_stream method).
    # Expanding at startup causes OOM: cap=170 (13.6 GB) + 33K cold prefill (3.3 GB)
    # = 16.9 GB > 17.18 GB Metal limit. Instead, first request runs at cap=100
    # (safe for cold prefill), then expand_expert_capacity() runs after response.

    _mem_profiler.init(SETTINGS.log_root)
    print("\n" + "=" * 50)
    print("🟢 SYSTEM READY")
    print(f"   • Mode:         {'VLM (vision)' if is_vlm else 'LM (text-only)'}")
    print(f"   • MLX Engine:   http://{SETTINGS.mlx_host}:{SETTINGS.mlx_port}")
    #print(f"   • LiteLLM:      http://0.0.0.0:{SETTINGS.proxy_port}")
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
        # Persist expert frequency stats for cross-session learning
        try:
            from .expert_cache import save_frequency_stats
            _stats_path = os.path.join("logs", "expert_stats.json")
            save_frequency_stats(model, _stats_path)
        except Exception:
            pass  # Best-effort on shutdown
        if sidecar_httpd:
            sidecar_httpd.shutdown()


if __name__ == "__main__":
    run()
