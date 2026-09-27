"""
[AI_DIRECTIVE]
ROL: Message-aware stable-prefix cache for cross-turn KV reuse
OBJETIVO: Detectar qué mensajes son idénticos entre turnos para reusar el
          prefix KV exacto, evitando re-prefill de contexto estable
ENTRADAS: session_id, messages (normalized), prompt_tokens
SALIDAS: stable_token_len (# tokens reutilizables), msg_token_boundaries
REGLAS INVIOLABLES:
- Caller debe tener prompt_cache_lock antes de llamar
- Solo el prefix contiguo de mensajes idénticos es reutilizable (por RoPE)
- No modificar los mensajes originales — normalización es solo para diff
SSoT: Este módulo es la única fuente de lógica de stable-prefix
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

# ── Module-level state (set via init()) ──────────────────────────────────────
_tokenizer: Any = None
_is_vlm: bool = False
_feature_preserve_thinking: bool = True
_max_idle_seconds: int = 1800


def init(
    *,
    tokenizer: Any,
    is_vlm: bool,
    feature_preserve_thinking: bool,
    max_idle_seconds: int,
) -> None:
    """Initialize with shared state from server2."""
    global _tokenizer, _is_vlm, _feature_preserve_thinking, _max_idle_seconds
    _tokenizer = tokenizer
    _is_vlm = is_vlm
    _feature_preserve_thinking = feature_preserve_thinking
    _max_idle_seconds = max_idle_seconds


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
_max_idle_seconds: int = SETTINGS.prompt_cache_session_max_idle_seconds


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
        _tokenizer is not None
        and hasattr(_tokenizer, "apply_chat_template")
        and getattr(_tokenizer, "chat_template", None) is not None
        and not _is_vlm
    ):
        try:
            prev_len = 0
            for i in range(n):
                # Render prefix messages[0..i] without generation prompt so the
                # boundary lands exactly at the end of message i.
                # We suppress add_generation_prompt to avoid including the assistant
                # start token in the boundary count.
                prefix_toks = _tokenizer.apply_chat_template(
                    messages[: i + 1],
                    tokenize=True,
                    add_generation_prompt=False,
                    preserve_thinking=_feature_preserve_thinking,
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
    # ttl_seconds <= 0. Without this guard, _max_idle_seconds=0
    # makes (now - touched_at) > 0 always True, pruning every record on every
    # write and silently destroying the stable-prefix mechanism for all sessions.
    now = time.time()
    if _max_idle_seconds > 0:
        stale = [
            sid
            for sid, rec in SESSION_TURN_STORE.items()
            if (now - rec.touched_at) > _max_idle_seconds
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

