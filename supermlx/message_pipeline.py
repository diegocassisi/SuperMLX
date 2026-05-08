# SPDX-License-Identifier: MIT
"""
SuperMLX message pipeline: canonicalization, healing, loop breaking, detection.

All functions are pure (input → output). Globals are passed as parameters.
"""
import copy
import hashlib
import json
import re
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

# ── Canonicalization regex patterns ───────────────────────────────────────────
INBOUND_META_MESSAGE_ID_PATTERN = re.compile(r'("message_id"\s*:\s*")[^"]+(")')
SUBAGENT_STATS_PATTERN = re.compile(r"(Stats: runtime\s+)[^\n•]+", re.IGNORECASE)
CACHE_TIME_PATTERN = re.compile(r"Current time is[^\n]+\.", re.IGNORECASE)
CACHE_TIME_COLON_PATTERN = re.compile(r"Current time:\s*[^\n]+", re.IGNORECASE)
CACHE_CCH_PATTERN = re.compile(r"cch=[a-zA-Z0-9]+;?", re.IGNORECASE)
CACHE_BILLING_HEADER_PATTERN = re.compile(
    r"-anthropic-billing-header:\s*[a-zA-Z0-9\-]+", re.IGNORECASE
)
CACHE_SYSTEM_REMINDER_PATTERN = re.compile(
    r"<system-reminder>.*?</system-reminder>", re.DOTALL | re.IGNORECASE
)
CACHE_SKILLS_BLOCK_PATTERN = re.compile(
    r"<available_skills>.*?</available_skills>", re.DOTALL | re.IGNORECASE
)
CACHE_RUNTIME_LINE_PATTERN = re.compile(
    r"^Runtime:.*$", re.MULTILINE | re.IGNORECASE
)

# ── Inbound Context structural constants ─────────────────────────────────────
_INBOUND_CONTEXT_HEADER = (
    "## Group Chat Context\n## Inbound Context (trusted metadata)\n"
)
_INBOUND_CONTEXT_FENCE_OPEN = "```json"
_INBOUND_CONTEXT_FENCE_CLOSE = "```"
_INBOUND_CONTEXT_STABLE_SECTION = "__STABLE_INBOUND_CONTEXT_SECTION__"
_PROJECT_CONTEXT_ANCHOR = "\n# Project Context"

# ── Compact runner / RAG bypass signals ──────────────────────────────────────
_COMPACT_RUNNER_SIGNALS: List[str] = [
    "compact", "compaction", "summarize", "summary of",
    "context window", "conversation history", "prior conversation",
    "previous conversation", "Produce a compact, factual summary",
]
_RAG_BYPASS_SIGNALS_USER = [
    "write any lasting notes to memory", "store durable memories now",
    "session nearing compaction", "reply with no_reply if nothing to store",
    "<text_to_summarize>",
]
_RAG_BYPASS_SIGNALS_SYSTEM = [
    "session nearing compaction", "store durable memories now",
]

# ── SessionContext dataclass ─────────────────────────────────────────────────
@dataclass
class SessionContext:
    session_id: str
    parent_session_id: Optional[str]
    branch_id: Optional[str]
    source: str


# ── Pure functions ───────────────────────────────────────────────────────────

def _get_healing_hash(
    text: str, tool_calls: Optional[List[Dict[str, Any]]] = None
) -> Optional[str]:
    """Creates a robust SHA-256 hash of the assistant's output."""
    base = (text or "").strip()
    if tool_calls:
        try:
            base += json.dumps(tool_calls, sort_keys=True)
        except Exception:
            pass
    if not base:
        return None
    return hashlib.sha256(base.encode("utf-8")).hexdigest()


def _heal_messages(
    messages: List[Dict[str, Any]],
    healing_store: OrderedDict,
    healing_store_lock,
) -> List[Dict[str, Any]]:
    """Swap stripped assistant messages back to full version (with <think>)."""
    healed = []
    for msg in messages:
        m = dict(msg)
        if (m.get("role") or "").strip().lower() == "assistant":
            content = m.get("content", "")
            tool_calls = m.get("tool_calls")
            if isinstance(content, str):
                h = _get_healing_hash(content, tool_calls)
                if h:
                    with healing_store_lock:
                        if h in healing_store:
                            m["content"] = healing_store[h]
                            m.pop("tool_calls", None)
                            healing_store.move_to_end(h, last=True)
            elif isinstance(content, list):
                new_content = []
                healed_any = False
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_part = part.get("text") or part.get("content") or ""
                        h = _get_healing_hash(text_part, tool_calls)
                        if h:
                            with healing_store_lock:
                                if h in healing_store:
                                    new_content.append({**part, "text": healing_store[h]})
                                    healed_any = True
                                    healing_store.move_to_end(h, last=True)
                                    continue
                    new_content.append(part)
                m["content"] = new_content
                if healed_any:
                    m.pop("tool_calls", None)
        healed.append(m)
    return healed


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


def _inject_loop_stop(
    messages: List[Dict[str, Any]], tool_idx: int,
    tool_name: str, count: int, mode: str, request_id: str,
    instruction: str, log_fn: Optional[Callable] = None,
) -> Tuple[List[Dict[str, Any]], bool]:
    """Inject a stop instruction into the last tool result to break the loop."""
    messages = list(messages)
    messages[tool_idx] = dict(messages[tool_idx])
    original_content = str(messages[tool_idx].get("content", ""))
    messages[tool_idx]["content"] = original_content + f"\n\n⚠️ SYSTEM LOOP BREAKER: {instruction}"
    if log_fn:
        log_fn(
            "LOOP_BREAK", request_id,
            f"ACTIVATED ({mode}) | tool='{tool_name}' | "
            f"consecutive_calls={count} | "
            f"injected stop instruction at msg[{tool_idx}]",
        )
    return messages, True


def _break_tool_call_loop(
    messages: List[Dict[str, Any]], request_id: str = "",
    enabled: bool = True, max_retries: int = 3,
    log_fn: Optional[Callable] = None,
) -> Tuple[List[Dict[str, Any]], bool]:
    """Detect and break infinite tool-call retry loops (ERROR/DUPLICATE/SPAM)."""
    if not enabled or len(messages) < 4:
        return messages, False

    SPAM_THRESHOLD = 5
    DUPLICATE_THRESHOLD = 2
    consecutive_calls: List[Dict[str, Any]] = []
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
        consecutive_calls.append({"tool_name": sig[0], "tool_args": sig[1], "tool_idx": i})
        i -= 2

    if consecutive_calls:
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
                    f"Respond with a text message using your own knowledge instead.",
                    log_fn,
                )
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
                    f"or explain that you couldn't find what you needed.",
                    log_fn,
                )

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
            if isinstance(tc_list[0], dict) else "unknown"
        )
        failures.append({"tool_name": tool_name, "tool_idx": i})
        i -= 2

    if len(failures) < max_retries:
        return messages, False

    name_counts: Dict[str, int] = {}
    for f in failures:
        name_counts[f["tool_name"]] = name_counts.get(f["tool_name"], 0) + 1
    most_common_name = max(name_counts, key=name_counts.get)  # type: ignore[arg-type]
    most_common_count = name_counts[most_common_name]

    if most_common_count < max_retries:
        return messages, False

    return _inject_loop_stop(
        messages, failures[0]["tool_idx"],
        most_common_name, most_common_count, "ERROR", request_id,
        f"The tool '{most_common_name}' has failed "
        f"{most_common_count} consecutive times with errors. "
        f"DO NOT retry this tool. DO NOT call '{most_common_name}' again. "
        f"Instead, respond with a text message to the user explaining "
        f"that '{most_common_name}' is currently unavailable and what the error was.",
        log_fn,
    )


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
    """Rough token estimate: ~4 chars per token. Used for logging only."""
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


def _is_slug_gen_request(messages: List[Dict[str, Any]]) -> bool:
    """Detects the specific OpenClaw slug-generation request."""
    if not messages:
        return False
    last_msg = str(messages[-1].get("content") or "")
    return "filename slug" in last_msg.lower() and "short 1-2 word" in last_msg.lower()


def _is_rag_bypass_request(messages: List[Dict[str, Any]]) -> bool:
    """Detects requests that should bypass RAG enrichment."""
    if not messages:
        return False
    for msg in reversed(messages):
        role = (msg.get("role") or "").lower()
        content = msg.get("content", "")
        if not isinstance(content, str):
            continue
        content_lower = content.lower()
        if role == "user":
            if any(sig in content_lower for sig in _RAG_BYPASS_SIGNALS_USER):
                return True
            break
    for msg in messages:
        if (msg.get("role") or "").lower() == "system":
            content = msg.get("content", "")
            if isinstance(content, str):
                content_lower = content.lower()
                if any(sig in content_lower for sig in _RAG_BYPASS_SIGNALS_SYSTEM):
                    return True
    return False


def _detect_compact_runner(messages: List[Dict[str, Any]], tools: Any) -> bool:
    """Detects if the incoming request is from OpenClaw's compact runner."""
    has_tools = tools is not None and len(tools) > 0
    if has_tools:
        return False
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
    _MAIN_AGENT_INDICATORS = ["<tools>", "function", "you are", "tool_choice", "<environment"]
    est_tokens = _estimate_token_count(messages)
    if est_tokens > 8000:
        for msg in messages:
            if (msg.get("role") or "").lower() == "system":
                content = msg.get("content", "")
                if isinstance(content, str):
                    content_lower = content.lower()
                    if any(ind in content_lower for ind in _MAIN_AGENT_INDICATORS):
                        return False
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict):
                            text = (part.get("text") or part.get("content") or "").lower()
                            if any(ind in text for ind in _MAIN_AGENT_INDICATORS):
                                return False
        return True
    return False


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


def _prepare_messages_for_template(messages, normalize_write_tool_content: bool = False):
    """Normalize messages for Jinja template rendering."""
    normalized = []
    for msg in messages:
        m = dict(msg)
        m["content"] = _flatten_content(m.get("content", ""))
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
                        normalize_write_tool_content
                        and fn_name == "write"
                        and isinstance(fn_copy.get("arguments"), dict)
                        and "content" in fn_copy["arguments"]
                    ):
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
                            serialized = json.dumps(raw_content, ensure_ascii=False, sort_keys=True)
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


def _scrub_cache_key(prompt: str, canonicalize: bool = True) -> str:
    """Post-render scrub of the CACHE KEY ONLY — length-preserving masking."""
    if not isinstance(prompt, str):
        return prompt
    if not canonicalize:
        return prompt

    def _mask(match):
        return "0" * len(match.group(0))

    normalized = CACHE_TIME_PATTERN.sub(_mask, prompt)
    normalized = CACHE_TIME_COLON_PATTERN.sub(_mask, normalized)
    normalized = CACHE_CCH_PATTERN.sub(_mask, normalized)
    normalized = CACHE_BILLING_HEADER_PATTERN.sub(_mask, normalized)
    normalized = CACHE_SYSTEM_REMINDER_PATTERN.sub(_mask, normalized)
    normalized = CACHE_SKILLS_BLOCK_PATTERN.sub(_mask, normalized)
    normalized = CACHE_RUNTIME_LINE_PATTERN.sub(_mask, normalized)
    return normalized


def _canonicalize_inbound_context_block(content: str) -> str:
    """Canonicalize the OpenClaw Inbound Context section (pure string ops, no regex)."""
    if _INBOUND_CONTEXT_STABLE_SECTION in content:
        return content

    header_pos = content.find(_INBOUND_CONTEXT_HEADER)

    if header_pos != -1:
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
            if line.startswith("## ") or line.startswith("# "):
                break
            cursor = line_end + 1
        if fence_open_pos == -1:
            return content
        fence_line_end = content.find("\n", fence_open_pos)
        if fence_line_end == -1:
            return content
        json_body_start = fence_line_end + 1
        fence_close_pos = content.find("\n" + _INBOUND_CONTEXT_FENCE_CLOSE, json_body_start)
        if fence_close_pos == -1:
            return content
        fence_end = fence_close_pos + 1 + len(_INBOUND_CONTEXT_FENCE_CLOSE)
        if fence_end < len(content) and content[fence_end] == "\n":
            fence_end += 1
        block_start = header_pos
        return content[:block_start] + _INBOUND_CONTEXT_STABLE_SECTION + "\n" + content[fence_end:]
    else:
        anchor_pos = content.find(_PROJECT_CONTEXT_ANCHOR)
        if anchor_pos == -1:
            return content
        return content[:anchor_pos] + "\n" + _INBOUND_CONTEXT_STABLE_SECTION + content[anchor_pos:]


def _canonicalize_messages(
    messages: List[Dict[str, Any]], canonicalize: bool = True,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Pre-render canonicalization: returns (original_messages, canonical_messages)."""
    if not canonicalize:
        original = copy.deepcopy(messages)
        return original, copy.deepcopy(original)

    original: List[Dict[str, Any]] = copy.deepcopy(messages)
    canonical: List[Dict[str, Any]] = copy.deepcopy(messages)

    for msg in canonical:
        content = msg.get("content", "")
        if isinstance(content, str):
            content = INBOUND_META_MESSAGE_ID_PATTERN.sub(r"\1__STABLE_MSG_ID__\2", content)
            content = _canonicalize_inbound_context_block(content)
            content = SUBAGENT_STATS_PATTERN.sub(r"\1__STABLE_RUNTIME__", content)
            msg["content"] = content
        elif isinstance(content, list):
            new_parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text = part.get("text", "")
                    text = INBOUND_META_MESSAGE_ID_PATTERN.sub(r"\1__STABLE_MSG_ID__\2", text)
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
    def _read_any_id(container: Optional[Dict[str, Any]], keys: List[str]) -> Optional[str]:
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
    id_keys = ["session_id", "conversation_id", "thread_id", "chat_id", "conversation", "session"]
    parent_keys = ["parent_session_id", "parent_id", "source_session_id", "origin_session_id"]
    branch_keys = ["branch_id", "branch", "subsession_id", "subagent_id"]

    session_id = (
        _read_any_id(body, id_keys) or _read_any_id(metadata, id_keys)
        or _read_any_id(extra_body, id_keys)
    )
    parent_session_id = (
        _read_any_id(body, parent_keys) or _read_any_id(metadata, parent_keys)
        or _read_any_id(extra_body, parent_keys)
    )
    branch_id = (
        _read_any_id(body, branch_keys) or _read_any_id(metadata, branch_keys)
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
        session_id=session_id, parent_session_id=parent_session_id,
        branch_id=branch_id, source=source,
    )


def _hoist_system_messages(msgs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge all system messages into one at position 0.

    Required by Qwen3.5 strict rule: system messages MUST be at the beginning.
    After compress/RAG/heal, system messages can be scattered — this fixes ordering.
    """
    system_parts: List[str] = []
    non_system: List[Dict[str, Any]] = []
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


def _assert_cache_key_safety(
    original_prompt: str,
    cache_key_prompt: str,
    context: str = "",
    log_fn: Optional[Callable] = None,
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
        if log_fn:
            log_fn(
                "❌",
                f"[CACHE-SAFETY] cache_key is {ratio:.1%} of original prompt "
                f"({key_len} vs {orig_len} chars) — normalization over-matched. "
                f"Falling back to original prompt as cache key. context={context}",
            )
        return False
    return True
