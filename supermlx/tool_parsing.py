# SPDX-License-Identifier: MIT
"""
SuperMLX tool call parsing: regex patterns, thinking extraction, tool call extraction.

All functions are pure (input → output). No state, no globals, no model access.
"""
import json
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple

# ── Tool call patterns ────────────────────────────────────────────────────────

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

# ── Thinking/reasoning patterns ───────────────────────────────────────────────

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
# Orphan <think> without </think>: generation was cut mid-thinking (e.g. THINKING_LIMIT,
# NGRAM_LOOP). Strip everything from <think> to end-of-string.
THINK_ORPHAN_OPEN_PATTERN = re.compile(r"<think>.*$", re.DOTALL | re.IGNORECASE)
GEMMA4_THINK_ORPHAN_OPEN_PATTERN = re.compile(r"<\|think\|>.*$", re.DOTALL | re.IGNORECASE)


# ── Functions ─────────────────────────────────────────────────────────────────

def _strip_thinking_from_content(text: str) -> str:
    """Remove <think>...</think> and <|think|>...<|/think|> blocks so reasoning is hidden."""
    if not isinstance(text, str):
        return text
    # First remove full think blocks (Qwen/GLM/Hermes/DeepSeek style).
    out = THINK_TAG_STRIP_PATTERN.sub("", text)
    # Then remove Gemma 4 pipe-style think blocks.
    out = GEMMA4_THINK_STRIP_PATTERN.sub("", out)
    # Remove orphan close tags (both styles), but only if content remains after.
    candidate = THINK_ORPHAN_CLOSE_PATTERN.sub("", out, count=1)
    if candidate.strip():
        out = candidate
    candidate = GEMMA4_THINK_ORPHAN_PATTERN.sub("", out, count=1)
    if candidate.strip():
        out = candidate
    # Remove orphan open tags (generation cut mid-thinking without </think>).
    candidate = THINK_ORPHAN_OPEN_PATTERN.sub("", out)
    if candidate.strip():
        out = candidate
    candidate = GEMMA4_THINK_ORPHAN_OPEN_PATTERN.sub("", out)
    if candidate.strip():
        out = candidate
    return out.strip()


def _should_enable_thinking(body, default_thinking: bool = True):
    if isinstance(body.get("enable_thinking"), bool):
        return body["enable_thinking"]
    # Default to enabled
    return default_thinking


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


def _extract_enable_thinking(body: Dict[str, Any], default_thinking: bool = True) -> Dict[str, Any]:
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
        "enable_thinking": _should_enable_thinking(body, default_thinking),
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
    # Handle Python-style booleans that json.loads rejects (case-sensitive)
    if value in ("True", "False"):
        return value == "True"
    try:
        return json.loads(value)
    except Exception:
        return value



# Pattern matching markdown code blocks: ```lang\ncontent\n```
_MARKDOWN_CODE_BLOCK = re.compile(
    r"```(\w*)\n(.*?)```", re.DOTALL
)


def _convert_markdown_tool_calls(text: str) -> str:
    """Convert markdown code blocks into <tool_call> XML format.

    Models conditioned by Claude Code's system prompt sometimes output tool calls
    as markdown code blocks (```bash\\nls -la\\n```) instead of the structured
    <tool_call><function=terminal> format. This rewrites those blocks so the main
    extraction chain picks them up.
    """
    # Map markdown language hints to interpreter commands
    _INTERPRETERS = {
        "python": "python3", "python3": "python3", "py": "python3",
        "node": "node", "javascript": "node", "js": "node",
        "typescript": "npx ts-node", "ts": "npx ts-node",
        "ruby": "ruby", "rb": "ruby",
        "perl": "perl",
    }
    _SHELL_LANGS = {"bash", "sh", "shell", "zsh"}

    def _replace_block(match):
        lang = match.group(1).strip().lower()
        content = match.group(2).strip()
        if not content:
            return match.group(0)
        if lang in _SHELL_LANGS:
            cmd = content
        elif lang in _INTERPRETERS:
            interp = _INTERPRETERS[lang]
            cmd = f"{interp} << 'HEREDOC_EOF'\n{content}\nHEREDOC_EOF"
        else:
            # Unknown language — leave as-is (display code, not a command)
            return match.group(0)
        return (
            f"\n<tool_call>\n<function=terminal>\n"
            f"<parameter=command>\n{cmd}\n</parameter>\n"
            f"</function>\n</tool_call>\n"
        )

    return _MARKDOWN_CODE_BLOCK.sub(_replace_block, text)


def _extract_openai_tool_calls(text, model_family):
    if not isinstance(text, str):
        return text, []

    # Agents-A1 and similar models conditioned by Claude's system prompt may output
    # tool calls as markdown code blocks instead of <tool_call> XML. Convert them
    # so the existing parser chain picks them up.
    if model_family == "qwen3" and "<tool_call>" not in text and "```" in text:
        text = _convert_markdown_tool_calls(text)

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
        body_clean = body.strip()
        if body_clean.startswith("```"):
            lines = body_clean.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            body_clean = "\n".join(lines).strip()

        try:
            data = json.loads(body_clean)
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
            cursor = max(cursor, end)
            continue  # Skip overlapping span
        if start > cursor:
            cleaned_parts.append(text[cursor:start])
        cursor = end
    if cursor < len(text):
        cleaned_parts.append(text[cursor:])
    cleaned_text = "".join(cleaned_parts).strip()
    return cleaned_text, tool_calls
