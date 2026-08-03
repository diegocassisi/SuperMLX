"""
[AI_DIRECTIVE]
ROL: State machine centralizada para detección de bloques <think>...</think>
OBJETIVO: SSoT para thinking detection en todo SuperMLX (sidecar, non-stream, stream)
ENTRADAS: Token IDs y texto de cada token generado
SALIDAS: ThinkingEvent (ENTER_THINKING, EXIT_THINKING, NONE) + estado actual
REGLAS INVIOLABLES:
- Prohibido duplicar lógica de thinking detection fuera de este módulo
- Prohibido usar regex para detección runtime (solo post-processing strip)
- Obligatorio: toda transición thinking↔responding pasa por feed()
SSoT: Este módulo es la ÚNICA fuente de verdad para el estado thinking/responding
"""

import logging
import re
from enum import Enum, auto
from typing import Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Think Token Candidates ────────────────────────────────────────────────────
# Ordered by priority: first match wins during vocab lookup.
_THINK_TOKEN_CANDIDATES: List[Tuple[str, str]] = [
    ("<think>", "</think>"),         # Qwen3, GLM4, DeepSeek, Hermes
    ("<|think|>", "<|/think|>"),     # Gemma 4
]

# Text-based fallback patterns (when model has no dedicated think tokens)
_THINK_END_TEXT = "</think>"
_THINK_END_GEMMA = "<|/think|>"


class ThinkingEvent(Enum):
    """Events emitted by ThinkingTracker.feed()."""
    NONE = auto()             # Normal token, no state change
    ENTER_THINKING = auto()   # Transitioned into thinking
    EXIT_THINKING = auto()    # Transitioned out of thinking (→ responding)


class ThinkingState(Enum):
    """Internal state of the tracker."""
    THINKING = auto()         # Inside <think> block
    RESPONDING = auto()       # Outside <think>, generating response


class ThinkingTracker:
    """Centralized state machine for think block detection.

    Usage:
        tracker = ThinkingTracker.from_tokenizer(tokenizer, enable_thinking=True)
        for token in generated_tokens:
            event = tracker.feed(token.id, token.text)
            if event == ThinkingEvent.EXIT_THINKING:
                # transition SSE blocks, etc.
    """

    def __init__(
        self,
        think_start_id: Optional[int] = None,
        think_end_id: Optional[int] = None,
        enable_thinking: bool = True,
    ) -> None:
        self._think_start_id = think_start_id
        self._think_end_id = think_end_id

        # When enable_thinking=True, the chat template injects <think>\n
        # at the end of the prompt. The model starts generating INSIDE
        # the thinking block — so initial state is THINKING.
        if enable_thinking:
            self._state = ThinkingState.THINKING
        else:
            self._state = ThinkingState.RESPONDING

        self._thinking_count: int = 0    # Total tokens generated while THINKING
        self._responding_count: int = 0  # Total tokens generated while RESPONDING
        self._exit_count: int = 0        # How many times </think> was detected
        self._total_count: int = 0       # Total tokens fed

        # Text accumulator for fallback detection (only used when no token IDs)
        self._text_acc: str = ""
        self._has_token_ids = think_end_id is not None

    @classmethod
    def from_tokenizer(
        cls,
        tokenizer: Any,
        enable_thinking: bool = True,
    ) -> "ThinkingTracker":
        """Create a tracker with token IDs resolved from the tokenizer vocab."""
        start_id, end_id = _resolve_think_token_ids(tokenizer)
        return cls(
            think_start_id=start_id,
            think_end_id=end_id,
            enable_thinking=enable_thinking,
        )

    def feed(self, token_id: int, token_text: str) -> ThinkingEvent:
        """Feed a generated token and get the resulting event.

        Args:
            token_id: The integer token ID from the model.
            token_text: The decoded text of this token.

        Returns:
            ThinkingEvent indicating state transition (or NONE).
        """
        self._total_count += 1
        event = ThinkingEvent.NONE

        if self._has_token_ids:
            event = self._detect_by_token_id(token_id)
        else:
            event = self._detect_by_text(token_text)

        # Update counters AFTER detection (so the boundary token itself
        # counts toward the state it ENTERED, not the one it LEFT)
        if self._state == ThinkingState.THINKING:
            self._thinking_count += 1
        else:
            self._responding_count += 1

        return event

    def _detect_by_token_id(self, token_id: int) -> ThinkingEvent:
        """Primary detection: exact token ID match."""
        if self._state == ThinkingState.THINKING:
            if token_id == self._think_end_id:
                self._state = ThinkingState.RESPONDING
                self._exit_count += 1
                return ThinkingEvent.EXIT_THINKING
            # Nested <think> while already thinking: ignore
        elif self._state == ThinkingState.RESPONDING:
            if token_id == self._think_start_id:
                self._state = ThinkingState.THINKING
                return ThinkingEvent.ENTER_THINKING
            if token_id == self._think_end_id:
                # Repeated </think> while responding — count it
                self._exit_count += 1
        return ThinkingEvent.NONE

    def _detect_by_text(self, token_text: str) -> ThinkingEvent:
        """Fallback detection: accumulate text and search for tags."""
        self._text_acc += token_text

        if self._state == ThinkingState.THINKING:
            # Check for </think> or <|/think|> in accumulated text
            for end_tag in (_THINK_END_TEXT, _THINK_END_GEMMA):
                if end_tag in self._text_acc:
                    self._state = ThinkingState.RESPONDING
                    self._exit_count += 1
                    # Keep text after the tag for response
                    self._text_acc = self._text_acc.split(end_tag, 1)[1]
                    return ThinkingEvent.EXIT_THINKING
        elif self._state == ThinkingState.RESPONDING:
            for start_tag in ("<think>", "<|think|>"):
                if start_tag in self._text_acc:
                    self._state = ThinkingState.THINKING
                    self._text_acc = self._text_acc.split(start_tag, 1)[1]
                    return ThinkingEvent.ENTER_THINKING
            # Check repeated </think>
            for end_tag in (_THINK_END_TEXT, _THINK_END_GEMMA):
                if end_tag in self._text_acc:
                    self._exit_count += 1
                    self._text_acc = self._text_acc.split(end_tag, 1)[1]

        return ThinkingEvent.NONE

    # ── State queries ─────────────────────────────────────────────────────

    @property
    def is_thinking(self) -> bool:
        """True if currently inside a <think> block."""
        return self._state == ThinkingState.THINKING

    @property
    def is_responding(self) -> bool:
        """True if currently generating visible response."""
        return self._state == ThinkingState.RESPONDING

    @property
    def thinking_count(self) -> int:
        """Number of tokens generated while in THINKING state."""
        return self._thinking_count

    @property
    def responding_count(self) -> int:
        """Number of tokens generated while in RESPONDING state."""
        return self._responding_count

    @property
    def total_count(self) -> int:
        """Total tokens fed to the tracker."""
        return self._total_count

    @property
    def exit_count(self) -> int:
        """How many </think> transitions were detected.
        
        1 = normal (single think block).
        2+ = model is looping (repeated </think> = THINK_LOOP_BREAK signal).
        """
        return self._exit_count

    @property
    def has_exited_thinking(self) -> bool:
        """True if at least one </think> was detected."""
        return self._exit_count > 0

    @property
    def is_looping(self) -> bool:
        """True if model emitted 2+ </think> tags (thinking loop detected)."""
        return self._exit_count >= 3

    def reset(self, enable_thinking: bool = True) -> None:
        """Reset tracker for a new generation (reuse same token IDs)."""
        self._state = ThinkingState.THINKING if enable_thinking else ThinkingState.RESPONDING
        self._thinking_count = 0
        self._responding_count = 0
        self._exit_count = 0
        self._total_count = 0
        self._text_acc = ""

    def force_exit(self) -> None:
        """Force transition to RESPONDING state.

        Used by THINK_CLEANUP when it injects a synthetic </think>
        after generation ended mid-thinking. This ensures downstream
        SSE output logic knows thinking has "exited".
        """
        self._state = ThinkingState.RESPONDING
        self._exit_count += 1


# ── Module-level helpers ──────────────────────────────────────────────────────

def _resolve_think_token_ids(tokenizer: Any) -> Tuple[Optional[int], Optional[int]]:
    """Resolve think start/end token IDs from the tokenizer vocabulary.

    Returns (start_id, end_id) or (None, None) if no think tokens found.
    """
    vocab = {}
    if hasattr(tokenizer, "get_vocab"):
        vocab = tokenizer.get_vocab()
    elif hasattr(tokenizer, "vocab"):
        vocab = tokenizer.vocab

    if not vocab:
        return (None, None)

    for start_str, end_str in _THINK_TOKEN_CANDIDATES:
        start_id = vocab.get(start_str)
        end_id = vocab.get(end_str)
        if start_id is not None and end_id is not None:
            return (start_id, end_id)

    return (None, None)
