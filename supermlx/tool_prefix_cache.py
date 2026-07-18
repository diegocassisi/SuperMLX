"""
[AI_DIRECTIVE]
ROL: Gestión del KV cache pre-computado para tool definitions estáticas de Claude Code.
OBJETIVO: Eliminar el overhead de prefill de ~33K tokens de tools en cada request.
          Pre-computa el KV cache una vez, lo serializa, y lo inyecta en cada request.
ENTRADAS: tools[] del request inbound, model, tokenizer, SETTINGS
SALIDAS: prompt_cache clonado listo para prefill de solo la conversación
REGLAS INVIOLABLES:
- Nunca mutar el prefix cache base — siempre clonar antes de retornar
- Nunca silenciar excepciones en compute_prefix_kv (fallar explícitamente)
- El hash debe cubrir el system prompt cacheable + tools para detectar cambios
SSoT: El prefix_hash en disco es la fuente de verdad para validar el cache
"""

import copy
import hashlib
import json
import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
from mlx_lm.generate import maybe_quantize_kv_cache
from mlx_lm.models.cache import make_prompt_cache

from supermlx.warmup_manager import load_cache, save_cache

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
_HASH_FILENAME = "tool_prefix.hash"
_KV_FILENAME = "tool_prefix_kv.safetensors"
_TOKENS_FILENAME = "tool_prefix_tokens.json"
# Bump to invalidate disk caches produced with the ghost-token generate_step.
_TPC_CACHE_VERSION = 2

# ── Module state (singleton) ───────────────────────────────────────────────────
_prefix_cache: Optional[List[Any]] = None   # The base KV cache — NEVER mutate
_prefix_tokens: Optional[List[int]] = None  # Token IDs of the cached prefix
_prefix_hash: Optional[str] = None          # Hash of system_body + tools that built this cache
_cache_dir: Optional[Path] = None           # Resolved cache directory


def init(cache_persist_path: str) -> None:
    """
    Set the directory for tool prefix cache files.
    Call once at server startup before any request.

    Args:
        cache_persist_path: Path to the existing warmup cache file (e.g. logs/warmup_cache.safetensors).
                            Tool prefix files are stored in the same directory.
    """
    global _cache_dir
    _cache_dir = Path(cache_persist_path).parent
    logger.info("[CONFIG] tool_prefix_cache dir=%s", _cache_dir)


def compute_tools_hash(system_body: str, tools: List[Dict], kv_bits: Optional[int] = None) -> str:
    """
    Compute a stable MD5 hash over the system body + tool definitions + kv_bits.

    Args:
        system_body: System prompt text (billing header stripped).
        tools: Tool definition list from the inbound request.
        kv_bits: KV cache quantization bits (None, 4, or 8). Included in hash
                 so changing quantization invalidates the cached prefix.

    Returns:
        Hex MD5 string.
    """
    payload = (
        system_body + "\x00"
        + json.dumps(tools, sort_keys=True, ensure_ascii=False) + "\x00"
        + str(kv_bits) + "\x00"
        + str(_TPC_CACHE_VERSION)
    )
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def _hash_path() -> Path:
    assert _cache_dir is not None, "tool_prefix_cache.init() not called"
    return _cache_dir / _HASH_FILENAME


def _kv_path() -> Path:
    assert _cache_dir is not None, "tool_prefix_cache.init() not called"
    return _cache_dir / _KV_FILENAME


def _tokens_path() -> Path:
    assert _cache_dir is not None, "tool_prefix_cache.init() not called"
    return _cache_dir / _TOKENS_FILENAME


def _load_stored_hash() -> Optional[str]:
    p = _hash_path()
    if p.exists():
        return p.read_text().strip()
    return None


def _save_stored_hash(h: str) -> None:
    _hash_path().write_text(h)


def load_from_disk(model: Any, max_kv_size: Optional[int]) -> bool:
    """
    Try to load the tool prefix KV cache from disk into module state.

    Returns True if loaded successfully, False otherwise.
    Must be called at startup after init().
    """
    global _prefix_cache, _prefix_tokens, _prefix_hash

    stored_hash = _load_stored_hash()
    if stored_hash is None:
        logger.info("[DATA] tool_prefix_cache: no hash on disk — cold start")
        return False

    tokens, cache = load_cache(_kv_path(), model, max_kv_size, log_fn=_log_fn)
    if tokens is None or cache is None:
        logger.info("[DATA] tool_prefix_cache: KV file missing or corrupt — cold start")
        return False

    _prefix_tokens = tokens
    _prefix_cache = cache
    _prefix_hash = stored_hash
    logger.info("[DATA] ✓ tool_prefix_cache loaded | tokens=%d | hash=%s", len(tokens), stored_hash[:8])
    return True


def prefill_cache_only(
    tokens: List[int],
    model: Any,
    cache: List[Any],
    *,
    prefill_step_size: int = 512,
    kv_bits: Optional[int] = None,
    kv_group_size: int = 64,
    quantized_kv_start: int = 0,
) -> bool:
    """
    Populate KV cache with tokens via chunked forward passes, WITHOUT
    generating any output tokens.

    Unlike generate_step(max_tokens=1), this does NOT produce a ghost token,
    so the cache offset after completion equals exactly initial_offset + len(tokens).
    Critical for hybrid caches (ArraysCache + KVCache) where
    can_trim_prompt_cache() returns False and ghost tokens cannot be removed.

    Includes KV quantization per chunk (same as generate_step) and a
    postcondition check on the final offset.

    Returns True if offset postcondition holds, False otherwise.
    Callers MUST NOT save/insert the cache when False is returned.
    """
    # Record initial offset for postcondition
    initial_offset = 0
    for c in cache:
        if hasattr(c, "offset"):
            initial_offset = int(c.offset)
            break

    prompt_array = mx.array(tokens)
    total = len(tokens)
    processed = 0

    while processed < total:
        n = min(prefill_step_size, total - processed)
        chunk = prompt_array[processed : processed + n][None]  # (1, n)
        model(chunk, cache=cache)
        if kv_bits is not None:
            maybe_quantize_kv_cache(
                cache,
                quantized_kv_start=quantized_kv_start,
                kv_group_size=kv_group_size,
                kv_bits=kv_bits,
            )
        mx.eval([c.state if hasattr(c, "state") else c for c in cache])
        mx.clear_cache()
        processed += n

    # ── Postcondition: offset integrity ──────────────────────────────────
    final_offset = None
    for c in cache:
        if hasattr(c, "offset"):
            final_offset = int(c.offset)
            break

    expected_offset = initial_offset + total
    if final_offset is not None and final_offset != expected_offset:
        logger.error(
            "[POSTCONDITION] prefill_cache_only: offset mismatch | "
            f"expected={expected_offset} actual={final_offset} "
            f"(initial={initial_offset} + tokens={total})"
        )
        return False

    return True

def compute_and_save(
    system_body: str,
    tools: List[Dict],
    tools_hash: str,
    tokenizer: Any,
    model: Any,
    max_kv_size: Optional[int],
    kv_bits: Optional[int],
    enable_thinking: bool = False,
    prefill_step_size: int = 512,
) -> bool:
    """
    Tokenize system_body + tools, run prefill-only forward pass, save KV to disk.

    This is called when the hash mismatches (tools changed) or on first cold start.
    Blocks until the KV cache is fully computed and saved.

    Args:
        system_body: System prompt text without the billing header line.
        tools: Tool definitions list.
        tools_hash: Pre-computed hash (from compute_tools_hash).
        tokenizer: The model tokenizer.
        model: The loaded MLX model.
        max_kv_size: Max KV cache size setting.
        kv_bits: Bits for KV quantization (None = no quantization).
        enable_thinking: Whether to enable thinking mode in chat template.
        prefill_step_size: Chunk size for prefill progress.

    Returns:
        True on success, raises on failure.
    """
    global _prefix_cache, _prefix_tokens, _prefix_hash

    t0 = time.time()
    logger.info("[INICIO] tool_prefix_cache.compute_and_save | kv_bits=%s", kv_bits)

    # Build a minimal messages list: one system message with system_body only,
    # no user/assistant turns. Tools are injected via the tools= kwarg.
    # Some chat templates (e.g. Agents-A1) require at least one user message
    # to render. Try without first; on failure, add a dummy user and strip it
    # from the rendered text before tokenizing.
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        _tpl_kwargs = dict(
            tokenize=False,
            add_generation_prompt=False,
            tools=tools,
            enable_thinking=enable_thinking,
        )
        try:
            prefix_text = tokenizer.apply_chat_template(
                [{"role": "system", "content": system_body}],
                **_tpl_kwargs,
            )
        except Exception:
            # Template requires a user message — render with dummy, strip it
            full_text = tokenizer.apply_chat_template(
                [{"role": "system", "content": system_body},
                 {"role": "user", "content": "."}],
                **_tpl_kwargs,
            )
            _marker = "<|im_start|>user"
            _pos = full_text.rfind(_marker)
            if _pos > 0:
                prefix_text = full_text[:_pos]
                logger.info("[DATA] dummy user stripped at pos=%d", _pos)
            else:
                prefix_text = full_text
                logger.warning("[DATA] dummy user marker not found — using full text")
        prefix_tokens = tokenizer.encode(prefix_text, add_special_tokens=False)
    else:
        prefix_tokens = tokenizer.encode(system_body, add_special_tokens=True)

    logger.info("[DATA] prefix tokens=%d", len(prefix_tokens))

    # Create fresh KV cache
    prefix_prompt = mx.array(prefix_tokens)
    cache = make_prompt_cache(model, max_kv_size=max_kv_size)

    # Chunked prefill: populate KV cache without generating output tokens.
    # generate_step(max_tokens=1) leaves a ghost token (+1 offset) that
    # cannot be trimmed in hybrid caches (ArraysCache.is_trimmable() = False),
    # breaking the checkpoint offset check and killing all cache reuse.
    ok = prefill_cache_only(
        prefix_tokens,
        model,
        cache,
        prefill_step_size=prefill_step_size,
        kv_bits=kv_bits,
        kv_group_size=64,
        quantized_kv_start=0,
    )
    if not ok:
        raise RuntimeError(
            "tool_prefix_cache: prefill postcondition failed — offset mismatch"
        )

    elapsed_prefill = int((time.time() - t0) * 1000)
    logger.info("[CALC] prefill done | elapsed=%dms", elapsed_prefill)

    # Save to disk
    ok = save_cache(prefix_tokens, cache, _kv_path(), prefix_hash=tools_hash, log_fn=_log_fn)
    if not ok:
        raise RuntimeError("tool_prefix_cache: save_cache failed")

    # Save hash separately for quick lookup at startup
    _save_stored_hash(tools_hash)

    # Save tokens for debug inspection
    _tokens_path().write_text(json.dumps(prefix_tokens))

    # Update module state
    _prefix_cache = cache
    _prefix_tokens = prefix_tokens
    _prefix_hash = tools_hash

    elapsed_total = int((time.time() - t0) * 1000)
    logger.info("[RESULT] Status=SUCCESS tool_prefix_cache saved | tokens=%d | elapsed=%dms", len(prefix_tokens), elapsed_total)
    return True


def get_prefix_cache_clone(
    system_body: str,
    tools: List[Dict],
    tokenizer: Any,
    model: Any,
    max_kv_size: Optional[int],
    kv_bits: Optional[int],
    enable_thinking: bool = False,
) -> Tuple[Optional[List[Any]], Optional[List[int]], bool]:
    """
    Main entry point for server.py.

    Given the current request's system_body and tools, returns a CLONED
    prompt_cache pre-populated with the tool prefix KV, ready for prefill
    of just the conversation tokens.

    If tools changed (hash mismatch), recomputes the KV cache synchronously
    before returning.

    Args:
        system_body: System prompt without billing header.
        tools: Tool definitions from the request.
        tokenizer, model, max_kv_size, kv_bits: Model config.
        enable_thinking: Chat template thinking flag.

    Returns:
        (cloned_cache, prefix_tokens, was_recomputed)
        cloned_cache is None if the module is not initialized (init() not called).
    """
    global _prefix_cache, _prefix_tokens, _prefix_hash

    if _cache_dir is None:
        logger.warning("[DECISION] tool_prefix_cache not initialized — skipping")
        return None, None, False

    current_hash = compute_tools_hash(system_body, tools, kv_bits=kv_bits)

    if _prefix_hash != current_hash or _prefix_cache is None:
        reason = "hash_mismatch" if _prefix_cache is not None else "cold_start"
        logger.info("[DECISION] tool_prefix_cache %s | old=%s new=%s",
                    reason,
                    (_prefix_hash or "none")[:8],
                    current_hash[:8])
        compute_and_save(
            system_body=system_body,
            tools=tools,
            tools_hash=current_hash,
            tokenizer=tokenizer,
            model=model,
            max_kv_size=max_kv_size,
            kv_bits=kv_bits,
            enable_thinking=enable_thinking,
        )
        was_recomputed = True
    else:
        was_recomputed = False

    # Clone the base cache so mutations during generation don't corrupt it
    cloned = copy.deepcopy(_prefix_cache)
    return cloned, list(_prefix_tokens), was_recomputed


def is_initialized() -> bool:
    """Return True if the prefix cache is loaded and ready."""
    return _prefix_cache is not None and _prefix_hash is not None


def is_configured() -> bool:
    """Return True if init() was called (cache dir set), regardless of whether
    the KV cache has been loaded or computed yet."""
    return _cache_dir is not None


def stats() -> Dict:
    """Return current state for logging/sidecar."""
    return {
        "initialized": is_initialized(),
        "prefix_tokens": len(_prefix_tokens) if _prefix_tokens else 0,
        "hash": _prefix_hash[:8] if _prefix_hash else None,
        "cache_dir": str(_cache_dir) if _cache_dir else None,
    }


def _log_fn(emoji: str, msg: str) -> None:
    logger.info("%s %s", emoji, msg)
