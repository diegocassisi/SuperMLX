# SPDX-License-Identifier: MIT
"""
NL Title Generator — Zero-model title generation using Apple NaturalLanguage.framework.

Intercepts Hermes title_generation requests at the server level and responds
with keyword-extracted titles in <1ms, avoiding the full LLM prefill cycle
(typically 10-40s) for a trivial side-task.

Uses NLTagger (NLTagSchemeLexicalClass) to extract nouns, verbs, and adjectives,
then assembles them into a readable title. The framework runs on ANE/CPU
automatically — Apple manages the dispatch.

Requires: pyobjc-framework-NaturalLanguage (pip install pyobjc-framework-NaturalLanguage)
Falls back gracefully if not installed — returns None and the server proceeds
with normal LLM inference.
"""

import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger("supermlx.nl_title")

# ── Lazy NL.framework import ─────────────────────────────────────────────────
_NL = None
_NL_AVAILABLE: Optional[bool] = None


def _ensure_nl():
    """Lazy-load NaturalLanguage.framework. Returns True if available."""
    global _NL, _NL_AVAILABLE
    if _NL_AVAILABLE is not None:
        return _NL_AVAILABLE
    try:
        import NaturalLanguage as NL
        _NL = NL
        _NL_AVAILABLE = True
        logger.info("NaturalLanguage.framework loaded — NL title intercept active")
    except ImportError:
        _NL_AVAILABLE = False
        logger.warning(
            "pyobjc-framework-NaturalLanguage not installed — NL title intercept disabled. "
            "Install with: pip install pyobjc-framework-NaturalLanguage"
        )
    return _NL_AVAILABLE


# ── Stopwords (EN + ES) ──────────────────────────────────────────────────────
_STOP = {
    # English
    "the", "a", "an", "in", "on", "at", "to", "for", "of", "with", "by",
    "is", "are", "was", "were", "be", "been", "being", "do", "does", "did",
    "can", "could", "will", "would", "shall", "should", "may", "might",
    "have", "has", "had", "this", "that", "these", "those", "it", "its",
    "my", "your", "our", "their", "his", "her", "me", "you", "we", "they",
    "i", "he", "she", "what", "how", "why", "when", "where", "which", "who",
    "not", "no", "and", "or", "but", "if", "then", "so", "just", "also",
    "each", "other", "between", "from", "up", "out", "about", "into",
    "some", "any", "all", "more", "most", "very", "too", "much", "many",
    "here", "there", "now", "only", "still", "already", "please", "thanks",
    "s", "t", "re", "ve", "ll", "d", "m",  # contractions
    # Spanish
    "el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del",
    "en", "con", "por", "para", "como", "que", "qué", "cómo", "mi", "mis",
    "tu", "tus", "su", "sus", "nos", "es", "son", "está", "están", "al",
    "se", "le", "lo", "me", "te", "ya", "más", "muy", "si", "sí", "o",
    "pero", "sin", "sobre", "entre", "hay", "ese", "esa", "esto", "eso",
}

# Filler verbs too generic for titles
_FILLER_VERBS = {
    # English
    "help", "make", "get", "let", "want", "need", "try", "give",
    "tell", "show", "know", "see", "go", "come", "take", "find", "put",
    "keep", "turn", "start", "begin", "look", "think", "ask",
    # Spanish
    "quiero", "puedo", "puede", "puedes", "necesito", "dame", "dime", "haz",
    "hacer", "tener", "poder", "deber", "saber", "decir", "poner",
    "ayuda", "ayudar", "explicar", "explica", "explicame",
}


def generate_title_nl(user_message: str, max_words: int = 5) -> Optional[str]:
    """Generate a session title using NaturalLanguage.framework keyword extraction.

    Returns a title string or None if NL.framework is unavailable.
    Typical latency: <1ms (after first-call warmup of ~50ms per language).
    """
    if not _ensure_nl():
        return None

    if not user_message or not user_message.strip():
        return "New Session"

    # Extract the actual user text from either format:
    #   Hermes:     "User: <snippet>\n\nAssistant: <snippet>"
    #   Claude Code: "<session>\n<user text>\n</session>\n\nWrite the title..."
    text = user_message
    if text.startswith("User: "):
        # Hermes format — split at "Assistant: " and take user part
        parts = text.split("\n\nAssistant: ", 1)
        text = parts[0][6:]  # Remove "User: " prefix
    elif "<session>" in text:
        # Claude Code format — extract content between <session> tags
        import re
        match = re.search(r"<session>\s*(.*?)\s*</session>", text, re.DOTALL)
        if match:
            text = match.group(1)

    # Cap input length
    text = text[:300]

    NL = _NL
    tagger = NL.NLTagger.alloc().initWithTagSchemes_([NL.NLTagSchemeLexicalClass])
    tagger.setString_(text)
    text_len = len(text)

    keywords = []

    results = tagger.tagsInRange_unit_scheme_options_tokenRanges_(
        (0, text_len),
        NL.NLTokenUnitWord,
        NL.NLTagSchemeLexicalClass,
        NL.NLTaggerOmitWhitespace | NL.NLTaggerOmitPunctuation,
        None,
    )

    if results and len(results) == 2:
        tags, ranges = results
        for tag, rng in zip(tags, ranges):
            loc = rng.rangeValue().location
            length = rng.rangeValue().length
            word = text[loc:loc + length]
            word_lower = word.lower()

            # Skip stopwords and filler verbs
            if word_lower in _STOP or word_lower in _FILLER_VERBS:
                continue

            # Keep nouns, verbs, adjectives, and untagged words (often proper nouns)
            if tag in (NL.NLTagNoun, NL.NLTagVerb, NL.NLTagAdjective, NL.NLTagOtherWord):
                keywords.append(word.capitalize())
                if len(keywords) >= max_words:
                    break

    if not keywords:
        # Fallback: first meaningful words from message
        words = [w for w in text.split()[:8] if w.lower() not in _STOP]
        keywords = [w.capitalize() for w in words[:max_words]]

    title = " ".join(keywords) if keywords else "New Session"

    # Enforce max length
    if len(title) > 80:
        title = title[:77] + "..."

    return title


def build_title_response(
    title: str,
    request_id: str,
    model_id: str,
) -> Dict[str, Any]:
    """Build an OpenAI-compatible chat completion response with the generated title."""
    import time as _time
    return {
        "id": f"chatcmpl-{request_id}",
        "object": "chat.completion",
        "created": int(_time.time()),
        "model": model_id,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": title},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": 50,
            "completion_tokens": len(title.split()),
            "total_tokens": 50 + len(title.split()),
        },
        "x_supermlx_nl_title": True,  # Signal for debugging
    }
