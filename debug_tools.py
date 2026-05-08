# SPDX-License-Identifier: MIT
"""
SuperMLX debug utilities — cache divergence analysis, token inspection.

These are diagnostic tools, not hot-path code. Separated to reduce
the main orchestrator's line count.
"""
from typing import List, Tuple


def _debug_token_divergence(
    tokenizer,
    current_tokens: List[int],
    stored_tokens: Tuple[int, ...],
    context_window: int = 5,
    log_fn=None,
):
    """Finds and logs exactly where two token sequences diverge for cache debugging."""
    min_len = min(len(current_tokens), len(stored_tokens))
    diverge_idx = -1

    for i in range(min_len):
        if current_tokens[i] != stored_tokens[i]:
            diverge_idx = i
            break

    # Limit how much we print: last few token IDs and short decoded snippets only
    max_tokens_show = 10
    max_text_len = 200

    start_idx = max(0, diverge_idx - context_window)
    tokens_before = current_tokens[start_idx:diverge_idx]
    if len(tokens_before) > max_tokens_show:
        tokens_before = tokens_before[-max_tokens_show:]
    end_idx_current = min(len(current_tokens), diverge_idx + context_window + 1)
    end_idx_stored = min(len(stored_tokens), diverge_idx + context_window + 1)

    _log = log_fn or print
    _log(f"\n" + "=" * 50)
    _log(f"🚨 CACHE DIVERGENCE DETECTED AT INDEX {diverge_idx} 🚨")
    _log(f"Token IDs before divergence (last {len(tokens_before)}): {tokens_before}")

    try:
        matching_text = tokenizer.decode(current_tokens[start_idx:diverge_idx])
        if len(matching_text) > max_text_len:
            matching_text = "..." + matching_text[-max_text_len:].strip()
        _log(f"Matching text leading up: {repr(matching_text)}")

        curr_divergent_token = current_tokens[diverge_idx]
        stor_divergent_token = stored_tokens[diverge_idx]
        _log(
            f"\n❌ Current Request Token [{diverge_idx}]: ID {curr_divergent_token} -> {repr(tokenizer.decode([curr_divergent_token]))}"
        )
        _log(
            f"❌ Stored Cache Token  [{diverge_idx}]: ID {stor_divergent_token} -> {repr(tokenizer.decode([stor_divergent_token]))}"
        )

        curr_context_after = tokenizer.decode(
            current_tokens[diverge_idx + 1 : end_idx_current]
        )
        stor_context_after = tokenizer.decode(
            stored_tokens[diverge_idx + 1 : end_idx_stored]
        )
        if len(curr_context_after) > max_text_len:
            curr_context_after = curr_context_after[:max_text_len] + "..."
        if len(stor_context_after) > max_text_len:
            stor_context_after = stor_context_after[:max_text_len] + "..."
        _log(f"\nCurrent context after: {repr(curr_context_after)}")
        _log(f"Stored context after:  {repr(stor_context_after)}")
    except Exception as e:
        _log(f"Could not decode tokens: {e}")
    _log("=" * 50 + "\n")
