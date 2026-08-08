"""
[AI_DIRECTIVE]
ROL: State machine para detección real-time de tool calls durante streaming
OBJETIVO: Bufferar XML de tool calls durante generación, prevenir leak de tags
          al cliente, detectar tool calls vacíos/inválidos en tiempo real
ENTRADAS: Texto de cada token generado (desde el loop de generación)
SALIDAS: ToolCallEvent + texto flusheable para el cliente + tool call bufferizado
REGLAS INVIOLABLES:
- Corre token-por-token durante streaming, nunca post-hoc
- NO reemplaza tool_parsing.py (que hace extracción post-hoc con regex)
- Debe manejar tags que abarcan múltiples tokens
- Obligatorio: toda transición responding↔buffering pasa por feed()
SSoT: Este módulo es la ÚNICA fuente de verdad para "estamos dentro de un tool call"
      durante streaming
"""

import logging
from enum import Enum, auto
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Tag Definitions ──────────────────────────────────────────────────────────
# Qwen3 / standard format
_OPEN_TAG = "<tool_call>"       # 11 chars
_CLOSE_TAG = "</tool_call>"     # 12 chars
# Gemma4 format
_OPEN_TAG_GEMMA = "<|tool_call|>"    # 13 chars
_CLOSE_TAG_GEMMA = "<|/tool_call|>"  # 14 chars

_MAX_TAG_LEN = 14  # Longest tag for holdback buffer sizing


class ToolCallEvent(Enum):
    """Events emitted by ToolCallTracker.feed()."""
    NONE = auto()                # Normal token, no state change
    ENTER_TOOL_CALL = auto()     # Transitioned into tool call buffering
    EXIT_TOOL_CALL = auto()      # Tool call completed (check .last_completed)


class ToolCallState(Enum):
    """Internal state of the tracker."""
    RESPONDING = auto()          # Normal text, safe to stream to client
    BUFFERING = auto()           # Inside <tool_call>, accumulating


class ToolCallTracker:
    """Streaming state machine for tool call detection.

    Usage:
        tracker = ToolCallTracker()
        for token in generated_tokens:
            event = tracker.feed(token.text)

            if tracker.is_buffering:
                pass  # Don't stream to client
            else:
                text = tracker.flushable_text
                if text:
                    send_to_client(text)

            if event == ToolCallEvent.EXIT_TOOL_CALL:
                call = tracker.last_completed
                if not call["has_parameters"]:
                    logger.warning("Empty tool call detected")

        # On EOS:
        event = tracker.finalize()
    """

    def __init__(self) -> None:
        self._state = ToolCallState.RESPONDING
        self._pending: str = ""           # Holdback for partial open-tag match
        self._tool_buffer: str = ""       # Accumulated tool call content
        self._flushable: str = ""         # Text safe to send to client NOW
        self._completed: List[Dict] = []  # Completed tool call records
        self._has_parameters: bool = False
        self._call_count: int = 0
        self._total_tokens: int = 0

    def feed(self, token_text: str) -> ToolCallEvent:
        """Feed a decoded token and get the resulting event.

        After calling feed(), check:
        - .flushable_text: text safe to stream to client
        - .is_buffering: True if inside a tool call (don't stream)
        """
        self._flushable = ""
        self._total_tokens += 1

        if self._state == ToolCallState.RESPONDING:
            return self._handle_responding(token_text)
        else:
            return self._handle_buffering(token_text)

    def _handle_responding(self, text: str) -> ToolCallEvent:
        """RESPONDING state: look for <tool_call> or <|tool_call|>."""
        self._pending += text

        # Check for full open tag (both formats)
        for open_tag in (_OPEN_TAG, _OPEN_TAG_GEMMA):
            idx = self._pending.find(open_tag)
            if idx >= 0:
                # Text before tag is safe to flush
                self._flushable = self._pending[:idx]
                # Content after tag starts the tool buffer
                self._tool_buffer = self._pending[idx + len(open_tag):]
                self._pending = ""
                self._state = ToolCallState.BUFFERING
                self._has_parameters = False
                return ToolCallEvent.ENTER_TOOL_CALL

        # No full tag found — flush text that can't be part of a tag
        overlap = _suffix_prefix_overlap(self._pending)
        safe = len(self._pending) - overlap
        if safe > 0:
            self._flushable = self._pending[:safe]
            self._pending = self._pending[safe:]

        return ToolCallEvent.NONE

    def _handle_buffering(self, text: str) -> ToolCallEvent:
        """BUFFERING state: accumulate until </tool_call> or <|/tool_call|>."""
        self._tool_buffer += text

        # Track parameter presence incrementally
        if not self._has_parameters and "<parameter" in self._tool_buffer:
            self._has_parameters = True

        # Check for close tag (both formats)
        for close_tag in (_CLOSE_TAG, _CLOSE_TAG_GEMMA):
            idx = self._tool_buffer.find(close_tag)
            if idx >= 0:
                tool_content = self._tool_buffer[:idx]
                remainder = self._tool_buffer[idx + len(close_tag):]

                # Final parameter check on complete content
                has_params = "<parameter" in tool_content

                self._completed.append({
                    "text": tool_content,
                    "has_parameters": has_params,
                    "incomplete": False,
                })
                self._call_count += 1

                # Reset for next potential tool call
                self._tool_buffer = ""
                self._state = ToolCallState.RESPONDING

                # Remainder goes to pending for next cycle
                self._pending = remainder
                if self._pending:
                    # Flush safe portion of remainder
                    overlap = _suffix_prefix_overlap(self._pending)
                    safe = len(self._pending) - overlap
                    if safe > 0:
                        self._flushable = self._pending[:safe]
                        self._pending = self._pending[safe:]

                return ToolCallEvent.EXIT_TOOL_CALL

        return ToolCallEvent.NONE

    def finalize(self) -> Optional[ToolCallEvent]:
        """Call when generation ends (EOS). Handles incomplete tool calls
        and flushes remaining pending text."""
        if self._state == ToolCallState.BUFFERING:
            # Tool call never closed — mark as incomplete
            has_params = "<parameter" in self._tool_buffer
            self._completed.append({
                "text": self._tool_buffer,
                "has_parameters": has_params,
                "incomplete": True,
            })
            self._call_count += 1
            self._tool_buffer = ""
            self._state = ToolCallState.RESPONDING
            logger.warning(
                "[TOOL_TRACK] incomplete tool call at EOS | "
                "has_params=%s | buffered_chars=%d",
                has_params, len(self._tool_buffer),
            )
            return ToolCallEvent.EXIT_TOOL_CALL

        # Flush any remaining pending text
        if self._pending:
            self._flushable = self._pending
            self._pending = ""

        return None

    # ── State queries ─────────────────────────────────────────────────────

    @property
    def flushable_text(self) -> str:
        """Text that is safe to stream to the client right now."""
        return self._flushable

    @property
    def is_buffering(self) -> bool:
        """True if currently inside a <tool_call> block."""
        return self._state == ToolCallState.BUFFERING

    @property
    def is_responding(self) -> bool:
        """True if generating normal visible text."""
        return self._state == ToolCallState.RESPONDING

    @property
    def last_completed(self) -> Optional[Dict]:
        """Most recently completed tool call record, or None."""
        return self._completed[-1] if self._completed else None

    @property
    def completed_calls(self) -> List[Dict]:
        """All completed tool call records."""
        return list(self._completed)

    @property
    def call_count(self) -> int:
        """Number of completed tool calls in this generation."""
        return self._call_count

    @property
    def total_tokens(self) -> int:
        """Total tokens fed to the tracker."""
        return self._total_tokens

    def summary(self) -> str:
        """One-line summary for logging."""
        empty = sum(1 for c in self._completed if not c["has_parameters"])
        incomplete = sum(1 for c in self._completed if c.get("incomplete"))
        parts = [f"calls={self._call_count}"]
        if empty:
            parts.append(f"empty={empty}")
        if incomplete:
            parts.append(f"incomplete={incomplete}")
        parts.append(f"tokens={self._total_tokens}")
        return " | ".join(parts)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _suffix_prefix_overlap(text: str) -> int:
    """How many chars at the end of `text` could be the start of an open tag?

    Checks both standard (<tool_call>) and Gemma4 (<|tool_call|>) tags.
    Returns the length of the longest suffix of `text` that is a prefix
    of any open tag.
    """
    best = 0
    for tag in (_OPEN_TAG, _OPEN_TAG_GEMMA):
        max_check = min(len(text), len(tag) - 1)
        for i in range(max_check, 0, -1):
            if text[-i:] == tag[:i]:
                best = max(best, i)
                break  # Found longest for this tag
    return best
