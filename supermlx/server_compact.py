"""
[AI_DIRECTIVE]
ROL: Server-side conversation compaction using the local model.
OBJETIVO: Compress long conversations into summaries to prevent OOM and improve quality.
ENTRADAS: raw_messages (OpenAI format), model, tokenizer
SALIDAS: Compacted message list with summary replacing old messages.
REGLAS INVIOLABLES:
- Compact prompt from Claude Code (compact/prompt.ts) — do NOT modify semantics.
- Must fit within safe prefill limit (MAX_SAFE_COMPACT_TOKENS).
- Strip <analysis> scratchpad, keep <summary> only.
- Preserve system messages and recent tail messages.
SSoT: Prompt text is the single source for compact behavior.
"""

import logging
import re
import time
import json
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

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


def _compact_temperature() -> float:
    return _env_float("COMPACTION_TEMPERATURE", 0.20)


def _compact_min_p() -> float:
    return _env_float("DEFAULT_MIN_P", 0.04)


def _compact_top_p() -> float:
    return _env_float("DEFAULT_TOP_P", 1.0)


# Max tokens for the compact call itself (conversation + prompt).
# Must be below OOM threshold. Default 35K leaves room for the ~2K prompt.
def _max_safe_compact_tokens() -> int:
    return _env_int("MAX_SAFE_COMPACT_TOKENS", 35000)


# How many recent messages to preserve verbatim (not summarized).
def _compact_tail_count() -> int:
    return _env_int("COMPACT_PRESERVE_TAIL", 4)


# Maximum characters for accumulated frozen summary before forcing consolidation (~4K tokens)
def _max_frozen_summary_chars() -> int:
    return _env_int("MAX_FROZEN_SUMMARY_CHARS", 16000)


# Max output tokens for the summary generation.
COMPACT_MAX_OUTPUT_TOKENS = 8192


# ── Compact Prompt (from Claude Code compact/prompt.ts) ───────────────────────

COMPACT_PROMPT2 = """Your task is to create a detailed summary of the conversation so far, paying close attention to the user's explicit requests and your previous actions.
This summary should be thorough in capturing technical details, code patterns, and architectural decisions that would be essential for continuing development work without losing context.

Before providing your final summary, wrap your analysis in <analysis> tags to organize your thoughts and ensure you've covered all necessary points. In your analysis process:

1. Chronologically analyze each message and section of the conversation. For each section thoroughly identify:
   - The user's explicit requests and intents
   - Your approach to addressing the user's requests
   - Key decisions, technical concepts and code patterns
   - Specific details like:
     - file names
     - full code snippets
     - function signatures
     - file edits
   - Errors that you ran into and how you fixed them
   - Pay special attention to specific user feedback that you received, especially if the user told you to do something differently.
2. Double-check for technical accuracy and completeness, addressing each required element thoroughly.

Your summary should include the following sections:

1. Primary Request and Intent: Capture all of the user's explicit requests and intents in detail
2. Key Technical Concepts: List all important technical concepts, technologies, and frameworks discussed.
3. Files and Code Sections: Enumerate specific files and code sections examined, modified, or created. Pay special attention to the most recent messages and include full code snippets where applicable and include a summary of why this file read or edit is important.
4. Errors and fixes: List all errors that you ran into, and how you fixed them. Pay special attention to specific user feedback that you received, especially if the user told you to do something differently.
5. Problem Solving: Document problems solved and any ongoing troubleshooting efforts.
6. All user messages: List ALL user messages that are not tool results. These are critical for understanding the users' feedback and changing intent.
7. Pending Tasks: Outline any pending tasks that you have explicitly been asked to work on.
8. Current Work: Describe in detail precisely what was being worked on immediately before this summary request, paying special attention to the most recent messages from both user and assistant. Include file names and code snippets where applicable.
9. Optional Next Step: List the next step that you will take that is related to the most recent work you were doing. IMPORTANT: ensure that this step is DIRECTLY in line with the user's most recent explicit requests, and the task you were working on immediately before this summary request. If your last task was concluded, then only list next steps if they are explicitly in line with the users request. Do not start on tangential requests or really old requests that were already completed without confirming with the user first.
                       If there is a next step, include direct quotes from the most recent conversation showing exactly what task you were working on and where you left off. This should be verbatim to ensure there's no drift in task interpretation.

CRITICAL: Respond with TEXT ONLY. Do NOT call any tools.
Your entire response must be an <analysis> block followed by a <summary> block.

Please provide your summary based on the conversation so far, following this structure and ensuring precision and thoroughness in your response."""

COMPACT_PROMPT = """Your task is to create a detailed summary of the conversation so far, paying close attention to the user's explicit requests and your previous actions.
This summary should be thorough in capturing technical decisions, architectural patterns, and exact state needed to resume development without losing continuity.

CRITICAL INSTRUCTIONS ON CODE AND FILES:
- Files on disk are the Single Source of Truth (SSoT). Do NOT output full file contents or full code snippets for files that already exist on disk. Reference them by exact path, current status, and key function signatures/exports.
- Ephemeral / unsaved code: If there is an in-progress draft, uncommitted snippet, or code that failed before being written to disk, DO preserve that specific snippet verbatim so work is not lost.
- Active working frontier: For the file actively being modified right before this summary, specify: exact path, the specific function/section in progress, and the immediate next change needed.
- Log clipping vs errors: Markers like "[payload omitted for context compaction — see file on disk]" or "...[tool output omitted for context compaction]" in tool call history are purely prompt length limits for this summary. They are NOT runtime errors, tool failures, or file corruptions. Never report a tool call as corrupted or failed simply because its serialized arguments were clipped for length.

Before providing your final summary, ensure you cover all necessary points directly:

1. Analyze each message and section of the conversation:
   - The user's explicit requests, intents, and feedback
   - Your approach to addressing the requests
   - Key architectural decisions, concepts, and interfaces
   - Files created or modified (paths and status)
   - Real errors encountered (actual command failures or exceptions) and how they were resolved
2. Double-check for technical accuracy and avoid assuming that clipped tool displays represent broken files.

Your summary should include the following sections:

1. Primary Request and Intent: Capture all of the user's explicit requests and intents in detail.
2. Key Technical Concepts: List all important technical concepts, architectures, and conventions agreed upon.
3. Files and Code Status:
   - Existing files on disk: List exact paths, roles, and key exports/APIs. (NO full file bodies).
   - In-flight / unsaved drafts: Include any pending code snippet that was NOT yet saved to disk.
4. Errors and Fixes: List genuine errors encountered (compiler/runtime errors, test failures, user corrections) and their fixes. Do NOT invent errors from clipped tool arguments.
5. All User Messages: List key user messages and corrections (critical for understanding changing intent).
6. Pending Tasks: Outline tasks explicitly requested by the user that remain to be done.
7. Current Work & Next Step:
   - Active file and exact location of current work.
   - Specific, concrete next action to resume immediately without asking clarifying questions.

CRITICAL: Respond with TEXT ONLY directly inside <summary>...</summary> tags. Do NOT call any tools. Do NOT output <think>, reasoning, or analysis tags.

Please provide your summary based on the conversation so far, following this structure."""


COMPACT_USER_WRAPPER = """This session is being continued from a previous conversation that ran out of context. The summary below covers the earlier portion of the conversation.

{summary}

Continue the conversation from where it left off without asking the user any further questions. Resume directly — do not acknowledge the summary, do not recap what was happening, do not preface with "I'll continue" or similar. Pick up the last task as if the break never happened."""


# ── Core Logic ────────────────────────────────────────────────────────────────

def format_compact_summary(raw_text: str) -> str:
    """Strip <analysis> scratchpad, extract <summary> content."""
    # Strip analysis
    text = re.sub(r"<analysis>[\s\S]*?</analysis>", "", raw_text)
    # Extract summary content
    match = re.search(r"<summary>([\s\S]*?)</summary>", text)
    if match:
        text = f"Summary:\n{match.group(1).strip()}"
    # Clean whitespace
    text = re.sub(r"\n\n+", "\n\n", text)
    return text.strip()


def extract_hermes_compact_parts(original: str, log_fn: Optional[Callable] = None) -> Tuple[str, str]:
    """Extract (previous_summary, turns_to_summarize) from Hermes compaction request,
    stripping all Hermes template instructions (## Historical Task, ## Goal, etc.)
    so they do not conflict with COMPACT_PROMPT."""
    if not isinstance(original, str) or not original:
        return ("", "")

    focus_part = ""
    focus_idx = original.find("\nFOCUS TOPIC:")
    if focus_idx == -1:
        focus_idx = original.find("FOCUS TOPIC:")
    if focus_idx != -1:
        focus_part = "\n\n" + original[focus_idx:].strip()
        body_text = original[:focus_idx]
    else:
        body_text = original

    if "PREVIOUS SUMMARY:" in body_text and "NEW TURNS TO INCORPORATE:" in body_text:
        ps_idx = body_text.find("PREVIOUS SUMMARY:")
        nt_idx = body_text.find("NEW TURNS TO INCORPORATE:")
        prev_summary_part = body_text[ps_idx:nt_idx].strip()
        after_nt = body_text[nt_idx:]
        end_markers = [
            "\n\nUpdate the summary using",
            "\n\n## Historical Task",
            "\n\n## Goal",
            "\n\nUse this exact structure:",
        ]
        end_pos = len(after_nt)
        for m in end_markers:
            p = after_nt.find(m)
            if p != -1 and p < end_pos:
                end_pos = p
        turns_part = after_nt[:end_pos].strip()
        return (prev_summary_part, f"{turns_part}{focus_part}".strip())

    if "TURNS TO SUMMARIZE:" in body_text:
        ts_idx = body_text.find("TURNS TO SUMMARIZE:")
        after_ts = body_text[ts_idx:]
        end_markers = [
            "\n\nUse this exact structure:",
            "\n\n## Historical Task",
            "\n\n## Goal",
            "\n\nUpdate the summary using",
        ]
        end_pos = len(after_ts)
        for m in end_markers:
            p = after_ts.find(m)
            if p != -1 and p < end_pos:
                end_pos = p
        turns_part = after_ts[:end_pos].strip()
        return ("", f"{turns_part}{focus_part}".strip())

    msg = "Hermes format fallback triggered — markers not found, formato pudo haber cambiado"
    if log_fn:
        log_fn("⚠️", msg)
    else:
        logger.warning("[FALLBACK] %s", msg)
    conv_start = original.find("\n\n")
    raw_fallback = original[conv_start:] if conv_start > 0 else original
    return ("", raw_fallback.strip())


def extract_hermes_compact_content(original: str, log_fn: Optional[Callable] = None) -> str:
    """Retrocompatible wrapper returning combined string for callers expecting a single string."""
    prev_part, turns_part = extract_hermes_compact_parts(original, log_fn=log_fn)
    if prev_part and turns_part:
        return f"\n\n{prev_part}\n\n{turns_part}"
    return f"\n\n{prev_part or turns_part}"


def serialize_messages_to_text(messages: List[Dict[str, Any]], max_chars: int = 140000) -> str:
    """Serialize OpenAI-format messages to plain text for the compact prompt.
    Truncates from the HEAD (oldest) to fit within max_chars."""
    lines = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")

        # Handle tool_calls in assistant messages
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            for tc in tool_calls:
                func = tc.get("function", {})
                name = func.get("name", "?")
                args = func.get("arguments", "")
                if isinstance(args, dict):
                    args = json.dumps(args, ensure_ascii=False)
                # Truncate very long args (file writes, etc.)
                if len(args) > 500:
                    args = args[:500] + "...[payload omitted for context compaction — see file on disk]"
                lines.append(f"[{role}] Tool call: {name}({args})")

        if isinstance(content, list):
            # Anthropic-style content blocks
            texts = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        texts.append(block.get("text", ""))
                    elif block.get("type") == "tool_result":
                        sub = block.get("content", "")
                        if isinstance(sub, list):
                            for s in sub:
                                if isinstance(s, dict) and s.get("type") == "text":
                                    texts.append(s.get("text", ""))
                        elif isinstance(sub, str):
                            texts.append(sub)
                elif isinstance(block, str):
                    texts.append(block)
            content = "\n".join(texts)

        if content:
            # Truncate very long individual messages (tool results, file reads)
            if len(content) > 2000 and role == "tool":
                content = content[:2000] + "\n...[tool output omitted for context compaction]"
            lines.append(f"[{role}] {content}")

    full_text = "\n\n".join(lines)

    # Truncate from HEAD if too long
    if len(full_text) > max_chars:
        full_text = "[earlier turns omitted for context compaction]\n\n" + full_text[-max_chars:]

    return full_text


def compact_conversation(
    raw_messages: List[Dict[str, Any]],
    model: Any,
    tokenizer: Any,
    log_fn: Optional[Callable] = None,
    request_id: str = "",
    enable_thinking: bool = False,
) -> Optional[List[Dict[str, Any]]]:
    """Compact a conversation using the loaded model.

    Takes the raw OpenAI-format messages, generates a summary using the model,
    and returns a new message list with old messages replaced by the summary.

    Returns None if compaction fails or isn't worth it.
    """
    import mlx.core as mx
    from mlx_lm.generate import generate_step

    t0 = time.time()
    tail_count = _compact_tail_count()

    # Split messages: system | body | tail (preserved)
    system_msgs = []
    body_msgs = []
    for msg in raw_messages:
        if (msg.get("role") or "").lower() == "system":
            system_msgs.append(msg)
        else:
            body_msgs.append(msg)

    n = len(body_msgs)
    if n <= tail_count + 2:
        if log_fn:
            log_fn("⚠️", f"COMPACT: too few messages to compact ({n}). Skipping. | req={request_id[:8]}")
        return None

    # ── DETECCIÓN DE RESUMEN PREVIO CONGELADO (Anti-degradación recursiva) ──
    _WRAPPER_TRIGGER = "This session is being continued from a previous conversation"
    _WRAPPER_MID = "The summary below covers the earlier portion of the conversation.\n\n"
    _WRAPPER_TAIL = "\n\nContinue the conversation from where it left off"

    frozen_summary = ""
    max_frozen_limit = _max_frozen_summary_chars()
    if body_msgs and isinstance(body_msgs[0].get("content"), str):
        first_content = body_msgs[0]["content"]
        if _WRAPPER_TRIGGER in first_content:
            start_idx = first_content.find(_WRAPPER_MID)
            end_idx = first_content.find(_WRAPPER_TAIL)
            extracted = ""
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                extracted = first_content[start_idx + len(_WRAPPER_MID):end_idx].strip()
            elif start_idx != -1:
                extracted = first_content[start_idx + len(_WRAPPER_MID):].strip()

            if len(extracted) > max_frozen_limit:
                if log_fn:
                    log_fn("⚠️", f"COMPACT: frozen summary exceeds cap ({len(extracted)} > {max_frozen_limit} chars). Consolidating via model.")
                frozen_summary = ""
            else:
                frozen_summary = extracted
                body_msgs = body_msgs[1:]
                n = len(body_msgs)

    tail_msgs = body_msgs[n - tail_count:]
    summarize_msgs = body_msgs[:n - tail_count]

    if log_fn:
        log_fn("🔄", f"COMPACT: starting | msgs_to_summarize={len(summarize_msgs)} | "
               f"tail_preserved={len(tail_msgs)} | req={request_id[:8]}")

    # Serialize messages to text, truncate to fit safe limits
    # ~4 chars per token, leave room for prompt (~2K tokens = ~8K chars)
    max_chars = (_max_safe_compact_tokens() - 2000) * 4
    conversation_text = serialize_messages_to_text(summarize_msgs, max_chars=max_chars)

    # Build the compact call messages
    compact_messages = [
        {"role": "user", "content": f"Here is the conversation to summarize:\n\n{conversation_text}"},
        {"role": "assistant", "content": "I'll analyze and summarize this conversation."},
        {"role": "user", "content": COMPACT_PROMPT},
    ]

    # Tokenize
    try:
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            prompt_text = tokenizer.apply_chat_template(
                compact_messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        else:
            prompt_text = compact_messages[-1]["content"]

        prompt_tokens = mx.array(tokenizer.encode(prompt_text))
        token_count = len(prompt_tokens)

        if log_fn:
            log_fn("🔄", f"COMPACT: tokenized | prompt_tokens={token_count} | "
                   f"safe_limit={_max_safe_compact_tokens()} | req={request_id[:8]}")

        if token_count > _max_safe_compact_tokens():
            if log_fn:
                log_fn("⚠️", f"COMPACT: prompt too large ({token_count} > {_max_safe_compact_tokens()}). "
                       f"Cannot compact safely. | req={request_id[:8]}")
            return None

    except Exception as e:
        logger.error("[ERROR] COMPACT tokenization failed: %s", str(e))
        if log_fn:
            log_fn("⚠️", f"COMPACT: tokenization failed: {e} | req={request_id[:8]}")
        return None

    # Generate summary
    try:
        if log_fn:
            log_fn("🔄", f"COMPACT: generating summary... | req={request_id[:8]}")

        # Use mlx_lm generate with calibrated compaction sampler
        from mlx_lm.sample_utils import make_sampler
        from mlx_lm.utils import GenerationResponse

        compact_sampler = make_sampler(
            temp=_compact_temperature(),
            top_p=_compact_top_p(),
            min_p=_compact_min_p(),
        )
        tokens_generated = []
        for response in generate_step(
            prompt=prompt_tokens,
            model=model,
            max_tokens=COMPACT_MAX_OUTPUT_TOKENS,
            sampler=compact_sampler,
        ):
            if isinstance(response, GenerationResponse):
                token = response.token
            else:
                # Older mlx_lm returns (token, logprobs)
                token = response[0] if isinstance(response, tuple) else response
            
            token_val = token.item() if hasattr(token, 'item') else int(token)
            tokens_generated.append(token_val)

            # Stop on EOS
            if token_val == tokenizer.eos_token_id:
                break
            # Also stop on </summary> detection
            if len(tokens_generated) > 20:
                partial = tokenizer.decode(tokens_generated[-20:])
                if "</summary>" in partial:
                    break

        mx.eval(mx.zeros(1))  # sync

        summary_text = tokenizer.decode(tokens_generated)

        if log_fn:
            log_fn("🔄", f"COMPACT: generated {len(tokens_generated)} tokens | req={request_id[:8]}")

    except Exception as e:
        logger.error("[ERROR] COMPACT generation failed: %s", str(e))
        if log_fn:
            log_fn("⚠️", f"COMPACT: generation failed: {e} | req={request_id[:8]}")
        return None

    # Parse and format summary
    formatted_summary = format_compact_summary(summary_text)
    if not formatted_summary or len(formatted_summary) < 100:
        if log_fn:
            log_fn("⚠️", f"COMPACT: summary too short or empty. Skipping. | req={request_id[:8]}")
        return None

    # Concatenate frozen summary with new summary without re-processing
    if frozen_summary:
        combined_summary = f"{frozen_summary}\n\n---\n\n{formatted_summary}"
    else:
        combined_summary = formatted_summary

    # Build compacted message list
    summary_user_msg = {
        "role": "user",
        "content": COMPACT_USER_WRAPPER.format(summary=combined_summary),
    }

    result = system_msgs + [summary_user_msg] + tail_msgs

    elapsed_ms = (time.time() - t0) * 1000
    if log_fn:
        log_fn("✅", f"COMPACT: done | {len(raw_messages)} msgs → {len(result)} msgs | "
               f"summary={len(formatted_summary)} chars | {elapsed_ms:.0f}ms | req={request_id[:8]}")

    return result
