"""
[AI_DIRECTIVE]
ROL: Controlador de vuelo para generación en tiempo real (in-flight interventions).
OBJETIVO: Analizar el stream de tokens durante la generación y decidir intervenciones
          (nudge injection, loop break, spark coupling) sin tocar el KV cache directamente.
ENTRADAS:
- Token IDs generados por el speculative engine (via record_token + evaluate).
- Sampler con tracker para determinar fase thinking/responding.
- Tokenizer para codificar textos de nudge.
SALIDAS:
- FlightAction con decisión de inyección, break, o None (continuar).
REGLAS INVIOLABLES:
- Prohibido acceder al KV cache. Solo retorna decisiones; el engine ejecuta la inyección.
- Prohibido print(). Usar logger con prefijos estándar.
- Obligatorio tipado en funciones públicas.
- Toda la lógica de política (cuándo inyectar, qué texto, loop detection) vive aquí.
SSoT: Única fuente de decisión para intervenciones in-flight durante generación MTP.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

from supermlx.config import SETTINGS

logger = logging.getLogger(__name__)


@dataclass
class FlightAction:
    """Resultado de evaluación del flight controller."""
    inject_tokens: Optional[List[int]] = None   # Token IDs a inyectar en KV cache
    hard_break: bool = False                      # Cortar generación inmediatamente
    label: str = ""                               # Label para logging
    spark_cause: Optional[str] = None             # Causa para Thermal Spark coupling


class FlightController:
    """Analiza el stream de tokens y decide intervenciones in-flight.

    Responsabilidades (policy):
    - N-gram loop detection (todas las fases)
    - Epistemic nudge scheduling (solo thinking)
    - Closure nudge scheduling (solo thinking)
    - Loop steering nudge (solo thinking)
    - Thermal Spark coupling en nudges

    NO responsable de (mechanism — lo hace el engine):
    - Forward passes / KV cache manipulation
    - Token sampling
    - MTP draft/verify
    """

    def __init__(
        self,
        tokenizer: Any,
        sampler: Optional[Any] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.sampler = sampler

        # ── Nudge scheduling state ──
        self._last_nudge_token: int = 0
        self._last_closure_nudge_token: int = 0

        # ── Loop detection state ──
        self._loop_nudge_applied: bool = False
        self._loop_nudge_token: int = 0
        self._generated_tokens_history: List[int] = []

        logger.info("[CONFIG] FlightController inicializado")

    @property
    def generated_tokens_history(self) -> List[int]:
        """Acceso de lectura a la historia de tokens para telemetría externa."""
        return self._generated_tokens_history

    def record_token(self, token_id: int) -> None:
        """Registra un token generado en la historia (para loop detection)."""
        self._generated_tokens_history.append(token_id)

    def record_tokens(self, token_ids: List[int]) -> None:
        """Registra múltiples tokens generados (e.g. nudge tokens)."""
        self._generated_tokens_history.extend(token_ids)

    def _is_thinking(self) -> bool:
        """Determina si el modelo está en fase thinking via el tracker del sampler."""
        if self.sampler is None:
            return False
        return (
            hasattr(self.sampler, "tracker")
            and getattr(self.sampler.tracker, "is_thinking", False)
        )

    def _get_thinking_count(self, tokens_generated: int) -> int:
        """Obtiene el conteo de thinking tokens del tracker, con fallback."""
        if self.sampler is not None and hasattr(self.sampler, "tracker"):
            return getattr(self.sampler.tracker, "thinking_count", tokens_generated)
        return tokens_generated

    def _check_ngram_loop(self) -> Tuple[bool, bool]:
        """Detecta loops por repetición de n-gramas.

        Returns:
            (trigger_nudge, trigger_hard_break)
        """
        if not SETTINGS.ngram_loop_detection:
            return False, False

        _n_hist = len(self._generated_tokens_history)
        if _n_hist < SETTINGS.ngram_size * 2:
            return False, False
        if _n_hist % SETTINGS.ngram_check_interval != 0:
            return False, False

        _tail = tuple(self._generated_tokens_history[-SETTINGS.ngram_size:])
        _search_region = self._generated_tokens_history[:-SETTINGS.ngram_size]
        _window_start = max(0, len(_search_region) - SETTINGS.ngram_window)
        _search_region = _search_region[_window_start:]

        _repeat_count = 0
        for _si in range(len(_search_region) - SETTINGS.ngram_size + 1):
            if tuple(_search_region[_si:_si + SETTINGS.ngram_size]) == _tail:
                _repeat_count += 1
                if _repeat_count >= SETTINGS.ngram_max_repeats:
                    break

        if _repeat_count < SETTINGS.ngram_max_repeats:
            return False, False

        # Loop detected — decide nudge vs hard break
        if SETTINGS.ngram_nudge_enabled and not self._loop_nudge_applied and self._is_thinking():
            return True, False  # nudge
        else:
            if self._loop_nudge_applied:
                if (_n_hist - self._loop_nudge_token) >= SETTINGS.ngram_grace_tokens:
                    logger.warning(
                        "[DECISION] 🛑 N-Gram loop persistió tras gracia (%d tok). Corte duro.",
                        _n_hist - self._loop_nudge_token,
                    )
                    return False, True  # hard break
                return False, False  # still in grace period
            return False, True  # hard break (no nudge available)

    def _check_nudges(self, tokens_generated: int, trigger_loop_nudge: bool) -> Optional[FlightAction]:
        """Evalúa si se debe inyectar un nudge (loop/closure/epistemic).

        Solo aplica durante thinking phase.
        """
        if not self._is_thinking():
            return None
        if self.tokenizer is None:
            return None

        _thinking_count = self._get_thinking_count(tokens_generated)
        _n_hist = len(self._generated_tokens_history)

        # ── Prioridad 0: Loop Steering Nudge ──
        if trigger_loop_nudge:
            raw_text = SETTINGS.ngram_nudge_text
            label = "Loop Steering Nudge"
            spark_cause = "ngram_loop_nudge"
            is_loop = True
            is_closure = False

        # ── Prioridad 1: Closure Nudge ──
        elif (
            SETTINGS.closure_nudge_interval > 0
            and _thinking_count >= SETTINGS.closure_nudge_interval
            and (_thinking_count - self._last_closure_nudge_token) >= SETTINGS.closure_nudge_interval
        ):
            raw_text = SETTINGS.closure_nudge_text
            label = "Closure Nudge"
            spark_cause = "closure_nudge"
            is_loop = False
            is_closure = True

        # ── Prioridad 2: Epistemic Nudge ──
        elif (
            SETTINGS.feature_epistemic_nudge
            and _thinking_count >= SETTINGS.epistemic_nudge_min_tokens
            and (_thinking_count - self._last_nudge_token) >= SETTINGS.epistemic_nudge_interval
        ):
            raw_text = SETTINGS.epistemic_nudge_text
            label = "Epistemic Nudge"
            spark_cause = "epistemic_nudge"
            is_loop = False
            is_closure = False

        else:
            return None  # No nudge needed

        # Tokenize nudge text
        clean_text = raw_text.strip()
        nudge_text = f"\n\n{clean_text}\n"
        nudge_token_ids = self.tokenizer.encode(nudge_text)
        if not nudge_token_ids:
            return None

        # Update scheduling state
        self._last_nudge_token = _thinking_count
        if is_closure:
            self._last_closure_nudge_token = _thinking_count
        if is_loop:
            self._loop_nudge_applied = True
            self._loop_nudge_token = _n_hist

        # Activate Thermal Spark if configured
        if (
            self.sampler is not None
            and hasattr(self.sampler, "spark_controller")
            and self.sampler.spark_controller is not None
            and self.sampler.spark_controller.enabled
        ):
            self.sampler.spark_controller.spark_remaining_tokens = SETTINGS.spark_pulse_duration
            self.sampler.spark_controller.last_spark_token = _thinking_count
            self.sampler.spark_controller.last_cause = spark_cause

        logger.info(
            "[SPARK] ⚡ %s inyectado en token %d (%d tokens)",
            label, _thinking_count, len(nudge_token_ids),
        )

        return FlightAction(
            inject_tokens=nudge_token_ids,
            hard_break=False,
            label=label,
            spark_cause=spark_cause,
        )

    def evaluate(self, tokens_generated: int, confirmed_token: int) -> Optional[FlightAction]:
        """Punto de entrada principal. Evalúa si se necesita intervención.

        Called once per engine iteration, BEFORE the verify step.

        Args:
            tokens_generated: Total tokens generados hasta ahora.
            confirmed_token: Último token confirmado (para contexto).

        Returns:
            FlightAction si se necesita intervención, None si se debe continuar.
        """
        # 1. N-gram loop detection (aplica a todas las fases)
        trigger_loop_nudge, trigger_hard_break = self._check_ngram_loop()

        if trigger_hard_break:
            _n_hist = len(self._generated_tokens_history)
            logger.warning(
                "[DECISION] 🛑 NGRAM LOOP HARD BREAK en token %d (%d-gram repetido)",
                _n_hist, SETTINGS.ngram_size,
            )
            return FlightAction(hard_break=True, label="N-Gram Hard Break")

        # 2. Nudge evaluation (loop nudge / closure / epistemic — solo thinking)
        nudge_action = self._check_nudges(tokens_generated, trigger_loop_nudge)
        if nudge_action is not None:
            return nudge_action

        return None
