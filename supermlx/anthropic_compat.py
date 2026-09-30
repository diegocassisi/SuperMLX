# SPDX-License-Identifier: MIT
"""
[AI_DIRECTIVE]
ROL: Anthropic Messages API translation functions for SuperMLX.
OBJETIVO: Translate between Anthropic /v1/messages format and OpenAI
          /v1/chat/completions format so Claude Code can use the MLX engine.
ENTRADAS: Anthropic-format dicts / OpenAI-format dicts
SALIDAS: Translated dicts in the opposite format
REGLAS INVIOLABLES:
- Prohibido lógica HTTP (eso vive en server.py)
- Prohibido hardcoding de URLs o model names
- Solo funciones puras de traducción de formato
SSoT: server.py maneja toda la lógica de inferencia y HTTP
"""
import json
import uuid
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# ── Claude model aliases ─────────────────────────────────────────────────────
# Claude Code sends these model names.  We accept any and route to the local
# model.  This set is also used by server.py to populate /v1/models.
CLAUDE_MODEL_ALIASES = frozenset({
    "claude-sonnet-4-20250514",
    "claude-opus-4-20250514",
    "claude-opus-4-8",
    "claude-3-5-sonnet-20241022",
    "claude-3-5-sonnet-latest",
    "claude-3-5-haiku-20241022",
    "claude-3-opus-20240229",
    "claude-3-haiku-20240307",
})


# ── Anthropic → OpenAI ───────────────────────────────────────────────────────

def anthropic_to_openai_body(anthropic_body: dict, local_model_id: str) -> dict:
    """Full Anthropic Messages request → OpenAI chat/completions request."""
    messages: List[Dict[str, Any]] = []

    # system
    system = anthropic_body.get("system")
    if system:
        if isinstance(system, list):
            sys_text = "\n".join(
                b.get("text", "") for b in system if b.get("type") == "text"
            ) or json.dumps(system, ensure_ascii=False)
        else:
            sys_text = str(system)
        messages.append({"role": "system", "content": sys_text})

    # messages
    for msg in anthropic_body.get("messages", []):
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if isinstance(content, str):
            messages.append({"role": role, "content": content})
            continue
        if not isinstance(content, list):
            messages.append({"role": role, "content": str(content)})
            continue

        text_parts: List[str] = []
        tool_calls: List[dict] = []
        tool_results: List[dict] = []

        for block in content:
            bt = block.get("type", "")
            if bt == "text":
                text_parts.append(block.get("text", ""))
            elif bt == "tool_use":
                tool_calls.append({
                    "id": block.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": json.dumps(
                            block.get("input", {}), ensure_ascii=False),
                    },
                })
            elif bt == "tool_result":
                rc = block.get("content", "")
                if isinstance(rc, list):
                    rc = "\n".join(
                        b.get("text", "") for b in rc if b.get("type") == "text"
                    ) or json.dumps(rc, ensure_ascii=False)
                tool_results.append({
                    "tool_call_id": block.get("tool_use_id", ""),
                    "content": str(rc),
                })

        if role == "assistant":
            am: Dict[str, Any] = {"role": "assistant"}
            if text_parts:
                am["content"] = "\n".join(text_parts)
            if tool_calls:
                am["tool_calls"] = tool_calls
                am.setdefault("content", "")
            messages.append(am)
        elif role == "user":
            for tr in tool_results:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tr["tool_call_id"],
                    "content": tr["content"],
                })
            if text_parts:
                messages.append({"role": "user",
                                 "content": "\n".join(text_parts)})
        else:
            messages.append({
                "role": role,
                "content": "\n".join(text_parts) if text_parts else "",
            })

    # body
    openai_body: Dict[str, Any] = {
        "model": local_model_id,
        "messages": messages,
        "max_tokens": anthropic_body.get("max_tokens", 4096),
        "stream": anthropic_body.get("stream", False),
    }
    if "temperature" in anthropic_body:
        openai_body["temperature"] = anthropic_body["temperature"]
    if "top_p" in anthropic_body:
        openai_body["top_p"] = anthropic_body["top_p"]
    if "stop_sequences" in anthropic_body:
        openai_body["stop"] = anthropic_body["stop_sequences"]

    # tools
    tools = anthropic_body.get("tools")
    if tools:
        oai_tools = []
        for t in tools:
            oai_tools.append({
                "type": "function",
                "function": {
                    "name": t.get("name", ""),
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {}),
                },
            })
        openai_body["tools"] = oai_tools

    return openai_body


# ── OpenAI → Anthropic (non-streaming) ───────────────────────────────────────

def openai_to_anthropic_response(
    message_text: str,
    tool_calls: list,
    finish_reason: str,
    requested_model: str,
    prompt_input_tokens: int = 0,
    output_tokens: int = 0,
) -> dict:
    """Build an Anthropic Messages response from already-processed generation
    output (text + tool_calls) that came out of the existing SuperMLX pipeline.

    prompt_input_tokens: estimated Anthropic prompt token count. Claude Code
    uses usage.input_tokens from the response to drive auto-compact decisions.
    Reporting 0 makes Claude Code think the context is empty → never compacts.
    """
    content_blocks: List[Dict[str, Any]] = []

    if message_text:
        content_blocks.append({"type": "text", "text": message_text})

    stop_reason = "end_turn"
    if tool_calls:
        stop_reason = "tool_use"
        for tc in tool_calls:
            func = tc.get("function", {})
            try:
                tc_input = json.loads(func.get("arguments", "{}"))
            except (json.JSONDecodeError, TypeError):
                tc_input = {"raw": func.get("arguments", "")}
            content_blocks.append({
                "type": "tool_use",
                "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:12]}"),
                "name": func.get("name", ""),
                "input": tc_input,
            })
    elif finish_reason == "length":
        stop_reason = "max_tokens"

    if not content_blocks:
        content_blocks.append({"type": "text", "text": ""})

    return {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "model": requested_model,
        "content": content_blocks,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": prompt_input_tokens,
            "output_tokens": output_tokens,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    }


# ── OpenAI → Anthropic (streaming) ──────────────────────────────────────────

def build_anthropic_sse_events(
    message_text: str,
    tool_calls: list,
    finish_reason: str,
    requested_model: str,
    prompt_input_tokens: int = 0,
) -> List[str]:
    """Build a complete list of Anthropic SSE event strings from the
    already-processed generation output.  Ready to write to the wire.

    prompt_input_tokens: estimated Anthropic prompt token count for
    Claude Code auto-compact. See openai_to_anthropic_response docstring.
    """
    message_id = f"msg_{uuid.uuid4().hex[:24]}"
    events: List[str] = []

    # 1. message_start — input_tokens drives Claude Code's auto-compact math
    events.append(_sse("message_start", {
        "type": "message_start",
        "message": {
            "id": message_id, "type": "message", "role": "assistant",
            "model": requested_model, "content": [],
            "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": prompt_input_tokens, "output_tokens": 0},
        },
    }))

    # 2. content blocks
    block_idx = 0
    if message_text:
        events.append(_sse("content_block_start", {
            "type": "content_block_start", "index": block_idx,
            "content_block": {"type": "text", "text": ""},
        }))
        events.append(_sse("content_block_delta", {
            "type": "content_block_delta", "index": block_idx,
            "delta": {"type": "text_delta", "text": message_text},
        }))
        events.append(_sse("content_block_stop", {
            "type": "content_block_stop", "index": block_idx,
        }))
        block_idx += 1

    stop_reason = "end_turn"
    if tool_calls:
        stop_reason = "tool_use"
        for tc in tool_calls:
            func = tc.get("function", {})
            try:
                tc_input = json.loads(func.get("arguments", "{}"))
            except (json.JSONDecodeError, TypeError):
                tc_input = {"raw": func.get("arguments", "")}
            tc_id = tc.get("id", f"toolu_{uuid.uuid4().hex[:12]}")
            events.append(_sse("content_block_start", {
                "type": "content_block_start", "index": block_idx,
                "content_block": {
                    "type": "tool_use", "id": tc_id,
                    "name": func.get("name", ""), "input": {},
                },
            }))
            events.append(_sse("content_block_delta", {
                "type": "content_block_delta", "index": block_idx,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": json.dumps(tc_input, ensure_ascii=False),
                },
            }))
            events.append(_sse("content_block_stop", {
                "type": "content_block_stop", "index": block_idx,
            }))
            block_idx += 1
    elif finish_reason == "length":
        stop_reason = "max_tokens"

    # 3. message_delta + message_stop
    events.append(_sse("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": {"output_tokens": 0},
    }))
    events.append(_sse("message_stop", {"type": "message_stop"}))

    return events


def _sse(event_type: str, data: dict) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
