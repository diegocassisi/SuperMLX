"""
[AI_DIRECTIVE]
ROL: Per-session cache for compressed message output
OBJETIVO: Evitar re-comprimir todo el historial en cada request, manteniendo
          tokens canónicos estables para lograr 97%+ KV cache hit rate
ENTRADAS: session_id, raw_messages, compressor_module, threshold
SALIDAS: (compressed_messages, elapsed_ms, cache_hit)
REGLAS INVIOLABLES:
- Thread-safe: todas las operaciones bajo _COMPRESS_CACHE_LOCK
- LRU eviction cuando se excede MAX_SESSIONS
- TTL-based expiry para evitar stale data
- Si la conversación shrunk (count < cached), invalidar
SSoT: Este módulo es la única fuente de caching de compresión de mensajes
"""
import json
import threading
import time
from typing import Any, Dict, List, Optional, Tuple


# ── Compression cache state ──────────────────────────────────────────────────
_COMPRESS_CACHE: Dict[str, Dict] = {}  # session_id → {"msgs": [...], "input_count": N, "ts": float}
_COMPRESS_CACHE_LOCK = threading.Lock()
_COMPRESS_CACHE_MAX_SESSIONS = 8  # Max sessions to cache (LRU eviction)
_COMPRESS_CACHE_TTL = 3600  # Seconds before a cached entry expires


def cache_get(session_id: str, current_msg_count: int) -> Optional[List[Dict]]:
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


def cache_put(session_id: str, compressed_msgs: List[Dict], input_count: int) -> None:
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


def compress_with_cache(
    raw_messages: List[Dict],
    comp_session: str,
    compressor_module: Any,
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
    cached = cache_get(comp_session, input_count)
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
            # Shouldn't happen (guard in cache_get), but fallback
            result = list(cached)

        elapsed_ms = (time.time() - t0) * 1000
        # Update cache with new input_count (conversation grew)
        cache_put(comp_session, result, input_count)
        return result, elapsed_ms, True

    # Cache miss: full compression
    compressed = compressor_module.compress_messages(
        raw_messages,
        session_key=comp_session,
    )
    elapsed_ms = (time.time() - t0) * 1000

    # Store in cache for future requests
    cache_put(comp_session, compressed, input_count)

    return compressed, elapsed_ms, False
