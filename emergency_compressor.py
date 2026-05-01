"""
emergency_compressor.py — LLMLingua-2 Content Compression Guard for SuperMLX

When rest_tokens exceeds the safe prefill limit, this module compresses the
TEXT CONTENT of messages in the "middle zone" using LLMLingua-2 (CPU, BERT).

Unlike the previous version which tried to DROP messages (ineffective when the
normal compressor already cleaned up), this version compresses the actual content
of remaining messages — making them shorter while preserving semantic meaning.

Zone architecture:
    [system]       ← PROTECTED (personality, identity)
    [first N msgs] ← PROTECTED (essence of the discussion)
    [middle zone]  ← COMPRESSED by LLMLingua-2 (verbose back-and-forth)
    [last M msgs]  ← PROTECTED (current context + question)

LLMLingua-2 is NOT a summarizer — it removes redundant words/tokens
while preserving key information. Runs on CPU (BERT model), does not
compete with the MLX model on GPU.

Usage in SuperMLX pipeline (after cache lookup, before prefill):

    from emergency_compressor import emergency_compress_if_needed

    result = emergency_compress_if_needed(
        raw_messages=raw_messages,
        rest_tokens_count=len(rest_tokens),
        compressor_module=_compressor_module,
        log_fn=_terminal_status,
        request_id=request_id,
    )
    if result is not None:
        raw_messages = result
        # Re-tokenize and re-lookup needed after emergency compression

Environment variables:
    MAX_SAFE_PREFILL_TOKENS        (default: 20000)  — threshold to trigger
    EMERGENCY_CONTENT_COMPRESS     (default: true)   — feature flag
    EMERGENCY_CONTENT_COMPRESS_RATIO (default: 0.5)  — compression ratio
    EMERGENCY_PROTECT_HEAD         (default: 2)      — msgs to protect at start
    EMERGENCY_PROTECT_TAIL         (default: 2)      — msgs to protect at end

Origin: OOM crash analysis 2026-04-26 (Diego + Opus 4)
Wiki:   knowledge/wiki/MLXServer/challenges/out-of-memory-v83-fix
"""

import os
import time
from typing import Callable, Dict, List, Optional


# ── Configuration ─────────────────────────────────────────────────────────────

def _env_int(key: str, default: int) -> int:
    v = os.environ.get(key, "")
    try:
        return int(v)
    except (ValueError, TypeError):
        return default


def _env_float(key: str, default: float) -> float:
    v = os.environ.get(key, "")
    try:
        return float(v)
    except (ValueError, TypeError):
        return default


def _env_bool(key: str, default: bool) -> bool:
    v = os.environ.get(key, "").strip().lower()
    if v in ("true", "1", "yes"):
        return True
    if v in ("false", "0", "no"):
        return False
    return default


# NOTE: Read lazily via function because SuperMLX imports this module BEFORE
# load_dotenv() runs. Module-level reads would always get the default.
def _max_safe_prefill() -> int:
    return _env_int("MAX_SAFE_PREFILL_TOKENS", 20000)


def _content_compress_enabled() -> bool:
    return _env_bool("EMERGENCY_CONTENT_COMPRESS", True)


def _content_compress_ratio() -> float:
    return _env_float("EMERGENCY_CONTENT_COMPRESS_RATIO", 0.5)


# How many non-system messages to protect at head and tail.
# Head = essence of the discussion. Tail = current context.
PROTECT_HEAD = _env_int("EMERGENCY_PROTECT_HEAD", 2)
PROTECT_TAIL = _env_int("EMERGENCY_PROTECT_TAIL", 2)

# Roles whose content can be compressed
_COMPRESSIBLE_ROLES = {"user", "assistant"}

# Minimum content length (chars) worth compressing — skip short messages
_MIN_CONTENT_LEN = 200


# ── Core ──────────────────────────────────────────────────────────────────────

def emergency_compress_if_needed(
    raw_messages: List[Dict],
    rest_tokens_count: int,
    compressor_module,
    session_key: str = "default",
    log_fn: Optional[Callable] = None,
    request_id: str = "",
) -> Optional[List[Dict]]:
    """Check if rest_tokens exceeds safe prefill limit and compress content if needed.

    Uses LLMLingua-2 to compress the TEXT CONTENT of messages in the middle zone,
    preserving system messages, head messages (discussion essence), and tail
    messages (current context).

    Args:
        raw_messages:       The current message list (may already be normally compressed).
        rest_tokens_count:  Number of tokens that need to be prefilled.
        compressor_module:  The rag_enricher module (needs _compress_with_llmlingua).
        session_key:        Session key (unused but kept for API compat).
        log_fn:             Optional logging function (signature: log_fn(icon, message)).
        request_id:         Request ID for logging.

    Returns:
        Messages with compressed content if triggered and effective.
        None if no compression was needed or feature is disabled.
    """
    _threshold = _max_safe_prefill()
    if rest_tokens_count <= _threshold:
        return None  # No emergency — within safe limits

    if not _content_compress_enabled():
        if log_fn:
            log_fn(
                "⚠️",
                f"EMERGENCY COMPRESSOR: rest_tokens={rest_tokens_count} > "
                f"{_threshold} but EMERGENCY_CONTENT_COMPRESS=false. "
                f"Skipping — signaling overflow. | req={request_id[:8]}",
            )
        return None

    # Check if the compressor module has LLMLingua
    compress_fn = getattr(compressor_module, "_compress_with_llmlingua", None)
    if compress_fn is None:
        if log_fn:
            log_fn(
                "⚠️",
                f"EMERGENCY COMPRESSOR: rest_tokens={rest_tokens_count} > "
                f"{_threshold} but _compress_with_llmlingua not available. "
                f"Skipping. | req={request_id[:8]}",
            )
        return None

    # ── Identify zones ────────────────────────────────────────────────────
    # Split messages into: system | head (protected) | middle (compress) | tail (protected)
    system_msgs = []
    non_system_msgs = []
    for m in raw_messages:
        if (m.get("role") or "").lower() == "system":
            system_msgs.append(m)
        else:
            non_system_msgs.append(m)

    n = len(non_system_msgs)
    head_count = min(PROTECT_HEAD, n)
    tail_count = min(PROTECT_TAIL, max(0, n - head_count))

    head_msgs = non_system_msgs[:head_count]
    tail_msgs = non_system_msgs[n - tail_count:] if tail_count > 0 else []
    middle_msgs = non_system_msgs[head_count:n - tail_count] if tail_count > 0 else non_system_msgs[head_count:]

    if not middle_msgs:
        if log_fn:
            log_fn(
                "⚠️",
                f"EMERGENCY COMPRESSOR: rest_tokens={rest_tokens_count} > "
                f"{_threshold} but no middle zone messages to compress "
                f"(total={n}, head={head_count}, tail={tail_count}). "
                f"Signaling overflow. | req={request_id[:8]}",
            )
        return None

    # ── Compress middle zone ──────────────────────────────────────────────
    ratio = _content_compress_ratio()
    t0 = time.time()
    compressed_count = 0
    total_chars_before = 0
    total_chars_after = 0

    if log_fn:
        log_fn(
            "🚨",
            f"EMERGENCY COMPRESSOR: rest_tokens={rest_tokens_count} > "
            f"{_threshold} | LLMLingua-2 content compress | "
            f"ratio={ratio} | middle={len(middle_msgs)} msgs | "
            f"protected: head={head_count} tail={tail_count} | "
            f"req={request_id[:8]}",
        )

    for msg in middle_msgs:
        role = (msg.get("role") or "").lower()
        content = msg.get("content", "")

        # Only compress text content from user/assistant roles
        if role not in _COMPRESSIBLE_ROLES:
            continue
        if not isinstance(content, str):
            continue
        if len(content) < _MIN_CONTENT_LEN:
            continue

        total_chars_before += len(content)

        try:
            compressed_text = compress_fn(content, ratio=ratio)
            if compressed_text and len(compressed_text) < len(content):
                msg["content"] = compressed_text
                total_chars_after += len(compressed_text)
                compressed_count += 1
            else:
                total_chars_after += len(content)
        except Exception as e:
            total_chars_after += len(content)
            if log_fn:
                log_fn(
                    "⚠️",
                    f"EMERGENCY COMPRESSOR: LLMLingua-2 failed on msg: {e} | "
                    f"req={request_id[:8]}",
                )

    elapsed_ms = (time.time() - t0) * 1000

    if compressed_count == 0:
        if log_fn:
            log_fn(
                "⚠️",
                f"EMERGENCY COMPRESSOR: no messages compressed "
                f"(0 eligible in middle zone). Signaling overflow. | "
                f"req={request_id[:8]}",
            )
        return None

    chars_saved = total_chars_before - total_chars_after
    tokens_saved_est = chars_saved // 4  # rough estimate

    if log_fn:
        reduction_pct = (1 - total_chars_after / max(total_chars_before, 1)) * 100
        log_fn(
            "🚨",
            f"EMERGENCY COMPRESSOR: compressed {compressed_count} msgs | "
            f"~{total_chars_before // 4} tok → ~{total_chars_after // 4} tok | "
            f"saved ~{tokens_saved_est} tok ({reduction_pct:.0f}%) | "
            f"{elapsed_ms:.0f}ms | req={request_id[:8]}",
        )

    # Check if compression was meaningful (>10% reduction)
    if total_chars_after >= total_chars_before * 0.90:
        if log_fn:
            log_fn(
                "⚠️",
                f"EMERGENCY COMPRESSOR: compression was ineffective "
                f"(<10% reduction). Signaling overflow. | req={request_id[:8]}",
            )
        return None

    # Reassemble: system + head + compressed middle + tail
    result = system_msgs + head_msgs + middle_msgs + tail_msgs
    return result


def should_signal_overflow(rest_tokens_count: int) -> bool:
    """Check if rest_tokens still exceeds safe limit after emergency compression.

    If True, the caller should return a 'context length exceeded' error
    to trigger OpenClaw auto-compaction + retry (Capa 3 in the defense plan).
    """
    return rest_tokens_count > _max_safe_prefill()
