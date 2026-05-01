"""
prompt_deduplicator.py — Tool Result & Definition Deduplication
═══════════════════════════════════════════════════════════════════════════════

[AI_DIRECTIVE]
ROL: Deduplica contenido repetido en mensajes OpenClaw para ahorrar tokens.
OBJETIVO: Reducir tokens de prefill sin romper KV cache (no reordena nada).
ENTRADAS: List[Dict] messages (OpenClaw format)
SALIDAS: List[Dict] messages deduplicados + metadata
REGLAS INVIOLABLES:
- Prohibido reordenar mensajes (rompe KV cache)
- Prohibido tocar los últimos N mensajes (guard)
- Solo operar sobre tool results y definitions
SSoT: Este módulo es la única fuente de dedup. SuperMLX no reimplementa.

Extraído de prompt_reorderer.py (2026-05-01):
- deduplicate_tool_results() — reemplaza tool_results repetidos con referencia
- deduplicate_definitions() — quita <definitions> duplicados del system prompt

Funciones ELIMINADAS (incompatibles con KV cache):
- reorder_system_prompt() — cambiaba orden de tokens → cache miss siempre
"""

import hashlib
import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# DEDUPLICATE TOOL RESULTS — Remove repeated tool results from conversation
# ═══════════════════════════════════════════════════════════════════════════════

# Minimum content length to consider for deduplication (skip short results)
_DEDUP_MIN_CHARS = 200

_DEDUP_REFERENCE_MSG = (
    "[Contenido ya presente en el contexto del sistema — "
    "no se repite para optimizar uso de memoria. "
    "Consultá el system prompt para el contenido completo.]"
)


def _content_hash(text: str) -> str:
    """Hash normalized content for dedup comparison."""
    normalized = text.strip().lower()
    return hashlib.md5(normalized.encode()).hexdigest()


def _extract_content_text(content: Any) -> str:
    """Extract plain text from message content (string or list format)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
        return "\n".join(parts)
    return ""


def deduplicate_tool_results(
    messages: List[Dict[str, Any]],
    system_content: str = "",
    guard_last_n: int = 4,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Remove duplicated tool result content from conversation history.

    Detects when the model reads the same file multiple times (e.g.,
    IDENTITY.md appears 6 times in tool results) and replaces all but
    the LAST occurrence with a compact reference message.

    Also detects tool results that duplicate content already present in
    the system prompt (Project Context) and replaces those too.

    Args:
        messages: Full messages array.
        system_content: System prompt content for cross-reference dedup.
        guard_last_n: Never touch the last N messages (preserve recent context).

    Returns:
        (deduped_messages, metadata)
    """
    if not messages or len(messages) < 3:
        return messages, {"duplicates_found": 0, "tokens_saved": 0}

    # Phase 1: Build hash set from system prompt content
    system_hashes: Set[str] = set()
    if system_content:
        file_blocks = re.split(r"^## /home/.+?\.md$", system_content, flags=re.MULTILINE)
        for block in file_blocks:
            block_text = block.strip()
            if len(block_text) >= _DEDUP_MIN_CHARS:
                system_hashes.add(_content_hash(block_text))

    # Phase 2: Scan tool results for duplicates
    seen_hashes: Dict[str, List[int]] = {}
    safe_end = max(0, len(messages) - guard_last_n)

    for i, msg in enumerate(messages):
        if msg.get("role") != "tool":
            continue
        content_text = _extract_content_text(msg.get("content", ""))
        if len(content_text) < _DEDUP_MIN_CHARS:
            continue
        h = _content_hash(content_text)
        if h not in seen_hashes:
            seen_hashes[h] = []
        seen_hashes[h].append(i)

    # Phase 3: Identify which messages to deduplicate
    dedup_targets: List[Tuple[int, str, int]] = []

    for h, indices in seen_hashes.items():
        if len(indices) > 1:
            for idx in indices[:-1]:
                if idx < safe_end:
                    content = _extract_content_text(messages[idx].get("content", ""))
                    dedup_targets.append((idx, "repeated_tool_result", len(content)))
        elif len(indices) == 1 and h in system_hashes:
            idx = indices[0]
            if idx < safe_end:
                content = _extract_content_text(messages[idx].get("content", ""))
                dedup_targets.append((idx, "duplicates_system_prompt", len(content)))

    if not dedup_targets:
        return messages, {"duplicates_found": 0, "tokens_saved": 0}

    # Phase 4: Apply deduplication
    deduped = list(messages)
    total_chars_saved = 0
    details = []

    for idx, reason, chars_saved in dedup_targets:
        original_content = _extract_content_text(deduped[idx].get("content", ""))
        preview = original_content[:80].replace("\n", " ")
        deduped[idx] = dict(deduped[idx])
        deduped[idx]["content"] = _DEDUP_REFERENCE_MSG
        total_chars_saved += chars_saved
        details.append({
            "index": idx,
            "reason": reason,
            "preview": preview,
            "chars_saved": chars_saved,
        })

    tokens_saved = total_chars_saved // 4
    logger.info(
        "[DEDUP] Removed %d duplicate tool results | ~%d tokens freed | details: %s",
        len(dedup_targets), tokens_saved,
        ", ".join(f"msg[{d['index']}]:{d['reason']}" for d in details[:5]),
    )

    return deduped, {
        "duplicates_found": len(dedup_targets),
        "tokens_saved": tokens_saved,
        "details": details,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# DEDUPLICATE DEFINITIONS — Remove repeated <definitions> blocks
# ═══════════════════════════════════════════════════════════════════════════════

_DEFINITIONS_BLOCK_PATTERN = re.compile(
    r"<definitions>.*?</definitions>",
    re.DOTALL,
)


def deduplicate_definitions(system_content: str) -> Tuple[str, int]:
    """Remove duplicate <definitions> blocks from system prompt.

    Keeps the FIRST occurrence, removes subsequent duplicates.

    Returns:
        (cleaned_content, chars_saved)
    """
    matches = list(_DEFINITIONS_BLOCK_PATTERN.finditer(system_content))

    if len(matches) <= 1:
        return system_content, 0

    chars_saved = 0
    result = system_content
    for match in reversed(matches[1:]):
        removed_text = match.group(0)
        chars_saved += len(removed_text)
        result = result[:match.start()] + result[match.end():]

    if chars_saved > 0:
        logger.info(
            "[DEDUP] Removed %d duplicate definition blocks | ~%d tokens freed",
            len(matches) - 1, chars_saved // 4,
        )

    return result, chars_saved


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def process_messages(
    messages: List[Dict[str, Any]],
    deduplicate: bool = True,
    dedup_definitions_flag: bool = True,
    guard_last_n: int = 4,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Deduplication pipeline for OpenClaw messages.

    1. Deduplicate definitions within system prompt
    2. Deduplicate repeated tool results in history

    NOTE: NO reordering. Reordering breaks KV cache.

    Returns:
        (optimized_messages, metadata)
    """
    if not messages:
        return messages, {"optimized": False}

    result = list(messages)
    meta: Dict[str, Any] = {"optimized": False}

    # Step 1: Process system prompt definitions
    system_content = ""
    system_idx = None
    for i, msg in enumerate(result):
        if msg.get("role") == "system":
            system_idx = i
            content = msg.get("content", "")
            if isinstance(content, str):
                system_content = content
            elif isinstance(content, list):
                system_content = "\n".join(
                    p.get("text", "") for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                )
            break

    if system_idx is not None and system_content and dedup_definitions_flag:
        original_len = len(system_content)
        system_content, defs_saved = deduplicate_definitions(system_content)
        meta["definitions_deduped"] = defs_saved > 0
        meta["definitions_chars_saved"] = defs_saved
        if len(system_content) != original_len:
            result[system_idx] = dict(result[system_idx])
            result[system_idx]["content"] = system_content
            meta["optimized"] = True

    # Step 2: Deduplicate tool results
    if deduplicate:
        result, dedup_meta = deduplicate_tool_results(
            result,
            system_content=system_content,
            guard_last_n=guard_last_n,
        )
        meta["dedup"] = dedup_meta
        if dedup_meta.get("duplicates_found", 0) > 0:
            meta["optimized"] = True

    # Summary
    total_savings = (
        meta.get("definitions_chars_saved", 0)
        + meta.get("dedup", {}).get("tokens_saved", 0) * 4
    )
    meta["total_tokens_saved"] = total_savings // 4

    if meta["optimized"]:
        logger.info(
            "[DEDUP] Pipeline complete | deduped=%d | defs_deduped=%s | ~%d tokens saved",
            meta.get("dedup", {}).get("duplicates_found", 0),
            meta.get("definitions_deduped", False),
            meta["total_tokens_saved"],
        )

    return result, meta
