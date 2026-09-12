"""
[AI_DIRECTIVE]
ROL: ThinkingTracker V2 con reasoning budget via logits processor (Opción A)
OBJETIVO: Limitar tokens en bloques <think> forzando </think> via manipulación
          de logits — el modelo "genera" </think> naturalmente, KV cache consistente
ENTRADAS: Token IDs generados (feed()), logits del modelo (processor)
SALIDAS: ThinkingEvent (igual que v1) + logits modificados cuando forcing
REGLAS INVIOLABLES:
- Prohibido duplicar lógica de thinking detection (hereda de ThinkingTracker)
- Prohibido break del loop de generación — el forzado es via logits
- Obligatorio: misma interfaz pública que ThinkingTracker (drop-in replacement)
- Obligatorio: logits processor es stateless respecto a mx — lee estado del tracker
SSoT: ThinkingTracker (v1) es la fuente de verdad para detección de thinking.
      Este módulo EXTIENDE con budget enforcement via logits.
"""

import logging
import time
from typing import Any, Callable, List, Optional, Tuple

import mlx.core as mx

from .thinking_tracker import ThinkingEvent, ThinkingState, ThinkingTracker

logger = logging.getLogger(__name__)


class ThinkingTrackerV2(ThinkingTracker):
    """ThinkingTracker with reasoning budget enforcement via logits processor.

    Drop-in replacement for ThinkingTracker. Same interface: feed(), is_thinking,
    is_responding, thinking_count, etc.

    Adds: make_logits_processor() that returns a callable to inject into
    mlx-lm's logits_processors list. When the thinking budget is exhausted,
    the processor forces the model to emit </think> token-by-token by setting
    all other logits to -inf (like llama.cpp's reasoning-budget sampler).

    Usage:
        tracker = ThinkingTrackerV2.from_tokenizer(
            tokenizer, enable_thinking=True, budget=4096,
        )
        processor = tracker.make_logits_processor()
        # Add processor to logits_processors list passed to stream_generate
        for token in generated_tokens:
            event = tracker.feed(token.id, token.text)
            # EXIT_THINKING fires naturally when forced </think> is emitted
    """

    def __init__(
        self,
        think_start_id: Optional[int] = None,
        think_end_id: Optional[int] = None,
        enable_thinking: bool = True,
        budget: int = 0,
        forced_end_tokens: Optional[List[int]] = None,
    ) -> None:
        super().__init__(
            think_start_id=think_start_id,
            think_end_id=think_end_id,
            enable_thinking=enable_thinking,
        )
        self._budget = budget                          # 0 = unlimited
        self._forced_end_tokens = forced_end_tokens or []  # token IDs for </think>
        self._forcing = False                          # True when forcing </think>
        self._force_pos = 0                            # next position in forced_end_tokens
        self._budget_exhausted_at: Optional[float] = None  # timestamp for telemetry

    @classmethod
    def from_tokenizer(
        cls,
        tokenizer: Any,
        enable_thinking: bool = True,
        budget: int = 0,
    ) -> "ThinkingTrackerV2":
        """Create a v2 tracker with token IDs resolved from the tokenizer vocab."""
        start_id, end_id = _resolve_think_token_ids_v2(tokenizer)
        forced_end = _resolve_forced_end_tokens(tokenizer, end_id)
        return cls(
            think_start_id=start_id,
            think_end_id=end_id,
            enable_thinking=enable_thinking,
            budget=budget,
            forced_end_tokens=forced_end,
        )

    def feed(self, token_id: int, token_text: str) -> ThinkingEvent:
        """Feed a generated token — extends v1 with budget enforcement.

        After the parent detects state transitions, checks if the thinking
        budget is exhausted. If so, activates forcing mode. The logits
        processor (from make_logits_processor()) will then force the next
        token(s) to be the </think> sequence.

        When the forced </think> token is emitted, the parent's detection
        naturally fires EXIT_THINKING — no special handling needed.
        """
        event = super().feed(token_id, token_text)

        if event == ThinkingEvent.EXIT_THINKING:
            # Natural exit (model decided, or our forced token triggered it)
            if self._forcing:
                elapsed_ms = int((time.time() - (self._budget_exhausted_at or time.time())) * 1000)
                logger.info(
                    "[RESULT] Reasoning budget forcing complete: "
                    "forced %d tokens in %dms",
                    self._force_pos, elapsed_ms,
                )
            self._forcing = False
            self._force_pos = 0
            return event

        if self._forcing:
            # Advance position in forced sequence
            self._force_pos += 1
            if self._force_pos >= len(self._forced_end_tokens):
                # All forced tokens emitted but parent didn't detect EXIT
                # This can happen if forced_end_tokens doesn't match the
                # token ID that ThinkingTracker uses for detection.
                # Force exit as safety fallback.
                logger.warning(
                    "[ERROR] Forced sequence complete but EXIT_THINKING not "
                    "detected — forcing exit as fallback"
                )
                self._forcing = False
                self._force_pos = 0
                self.force_exit()
                return ThinkingEvent.EXIT_THINKING
            return event

        # Check if budget is exhausted
        if (
            self.is_thinking
            and self._budget > 0
            and self.thinking_count >= self._budget
            and self._forced_end_tokens
        ):
            self._forcing = True
            self._force_pos = 0
            self._budget_exhausted_at = time.time()
            logger.info(
                "[DECISION] Reasoning budget exhausted (%d/%d tokens), "
                "forcing </think> via logits (%d forced tokens)",
                self.thinking_count, self._budget, len(self._forced_end_tokens),
            )

        return event

    def make_logits_processor(self) -> Callable[[mx.array, mx.array], mx.array]:
        """Return a logits processor for mlx-lm's generate_step.

        The returned callable has signature (tokens, logits) -> logits,
        matching mlx-lm's logits_processors protocol.

        When not forcing, it's a no-op passthrough.
        When forcing, it sets all logits to -inf except the forced token,
        guaranteeing the sampler picks it regardless of temperature/top_p.
        """
        tracker = self  # closure captures reference to live tracker state

        def _reasoning_budget_processor(
            tokens: mx.array, logits: mx.array
        ) -> mx.array:
            if not tracker._forcing:
                return logits

            if tracker._force_pos >= len(tracker._forced_end_tokens):
                return logits

            forced_token_id = tracker._forced_end_tokens[tracker._force_pos]

            # Set all logits to -inf except the forced token (like llama.cpp).
            # FIX: Use mx.where to avoid NaN from (-inf + inf) per IEEE 754.
            # The original .at[].add() did: restore - (-inf) = +inf, then
            # -inf + inf = NaN → broken sampler. mx.where is clean and safe.
            mask = mx.arange(logits.shape[-1]) == forced_token_id  # [vocab_size] bool
            result = mx.where(mask, logits, float("-inf"))          # broadcast over batch
            return result

        return _reasoning_budget_processor

    # ── State queries (v2 additions) ──────────────────────────────────────

    @property
    def is_forcing(self) -> bool:
        """True if currently forcing the </think> sequence via logits."""
        return self._forcing

    @property
    def budget(self) -> int:
        """Configured budget (0 = unlimited)."""
        return self._budget

    @property
    def budget_remaining(self) -> int:
        """Tokens remaining in budget (0 if unlimited or exhausted)."""
        if self._budget <= 0:
            return 0
        return max(0, self._budget - self.thinking_count)

    def reset(self, enable_thinking: bool = True) -> None:
        """Reset tracker for a new generation (reuse same token IDs + budget)."""
        super().reset(enable_thinking)
        self._forcing = False
        self._force_pos = 0
        self._budget_exhausted_at = None


# ── Module-level helpers ──────────────────────────────────────────────────────

def _resolve_think_token_ids_v2(
    tokenizer: Any,
) -> Tuple[Optional[int], Optional[int]]:
    """Resolve think start/end token IDs from the tokenizer vocabulary.

    Delegates to ThinkingTracker's resolution logic via from_tokenizer,
    but returns the raw IDs for v2's constructor.
    """
    # Reuse v1's resolution logic
    from .thinking_tracker import _resolve_think_token_ids
    return _resolve_think_token_ids(tokenizer)


def _resolve_forced_end_tokens(
    tokenizer: Any,
    think_end_id: Optional[int],
) -> List[int]:
    """Resolve the token ID sequence to force when closing a think block.

    Strategy:
    1. If think_end_id is a dedicated vocab token, use it directly (single token).
    2. Otherwise, encode "</think>" to get the multi-token sequence.

    Returns:
        List of token IDs representing the </think> sequence to force.
    """
    # Prefer the dedicated vocab token (most models have one)
    if think_end_id is not None:
        logger.info(
            "[CONFIG] Forced end tokens: single vocab token ID=%d",
            think_end_id,
        )
        return [think_end_id]

    # Fallback: encode the text
    end_tags = ["</think>", "<|/think|>"]
    for tag in end_tags:
        try:
            tokens = tokenizer.encode(tag, add_special_tokens=False)
            if tokens:
                logger.info(
                    "[CONFIG] Forced end tokens: encoded '%s' → %s",
                    tag, tokens,
                )
                return list(tokens)
        except Exception:
            continue

    logger.warning(
        "[ERROR] Could not resolve forced end tokens — "
        "v2 budget enforcement will not work"
    )
    return []
