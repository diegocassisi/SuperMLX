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


# Max tokens for the compact call itself (conversation + prompt).
# Must be below OOM threshold. Default 35K leaves room for the ~2K prompt.
def _max_safe_compact_tokens() -> int:
    return _env_int("MAX_SAFE_COMPACT_TOKENS", 35000)


# How many recent messages to preserve verbatim (not summarized).
def _compact_tail_count() -> int:
    return _env_int("COMPACT_PRESERVE_TAIL", 4)


# Max output tokens for the summary generation.
COMPACT_MAX_OUTPUT_TOKENS = 4096


# ── Compact Prompt (from Claude Code compact/prompt.ts) ───────────────────────

COMPACT_PROMPT = """Your task is to create a detailed summary of the conversation so far, paying close attention to the user's explicit requests and your previous actions.
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
                    args = args[:500] + "...[truncated]"
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
                content = content[:2000] + "\n...[truncated tool output]"
            lines.append(f"[{role}] {content}")

    full_text = "\n\n".join(lines)

    # Truncate from HEAD if too long
    if len(full_text) > max_chars:
        full_text = "[earlier conversation truncated]\n\n" + full_text[-max_chars:]

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

        # Use mlx_lm generate
        from mlx_lm.utils import GenerationResponse
        tokens_generated = []
        for response in generate_step(
            prompt=prompt_tokens,
            model=model,
            max_tokens=COMPACT_MAX_OUTPUT_TOKENS,
            temp=0.3,
            top_p=0.95,
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

    # Build compacted message list
    summary_user_msg = {
        "role": "user",
        "content": COMPACT_USER_WRAPPER.format(summary=formatted_summary),
    }

    result = system_msgs + [summary_user_msg] + tail_msgs

    elapsed_ms = (time.time() - t0) * 1000
    if log_fn:
        log_fn("✅", f"COMPACT: done | {len(raw_messages)} msgs → {len(result)} msgs | "
               f"summary={len(formatted_summary)} chars | {elapsed_ms:.0f}ms | req={request_id[:8]}")

    return result
