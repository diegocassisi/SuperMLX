# SPDX-License-Identifier: MIT
"""
[AI_DIRECTIVE]
ROL: SuperMLX Sampling Engine & Dual-Phase Strategy
OBJETIVO: Proveer samplers deterministas y Dual-Phase para modelos de razonamiento (thinking -> response)
ENTRADAS: Logits de MLX, parámetros de muestreo (temp, min_p, top_p, top_k) y estado de ThinkingTracker
SALIDAS: Tokens seleccionados o callables de muestreo compatibles con mlx-lm
REGLAS INVIOLABLES:
- Prohibido cargar modelos pesados en este módulo (módulo puramente matemático/estocástico)
- Prohibido hardcodear números mágicos fuera de los fallbacks tipados
- Obligatoria conmutación dinámica de fase condicionada a tracker.is_thinking
SSoT: Este archivo es la única fuente de verdad (SSoT) para la lógica de DualPhaseSampler en SuperMLX.
"""

from typing import Any, Callable, Dict, Optional, Tuple
import mlx.core as mx
from mlx_lm.sample_utils import make_sampler


class DualPhaseSampler:
    """Dynamic Dual-Phase Sampler for reasoning models.

    Routes token selection to think_sampler while tracker.is_thinking is True,
    and switches to resp_sampler as soon as tracker transitions to responding.
    Supports dynamic adaptation of response temperature during generation.
    """

    def __init__(
        self,
        think_sampler: Callable[[mx.array], mx.array],
        resp_sampler: Callable[[mx.array], mx.array],
        tracker: Any = None,
        think_temp: float = 0.5,
        resp_temp: float = 0.1,
        top_p: float = 1.0,
        top_k: int = -1,
        min_p: float = 0.0,
    ) -> None:
        self.think_sampler = think_sampler
        self.resp_sampler = resp_sampler
        self.tracker = tracker
        self.think_temp = think_temp
        self.resp_temp = resp_temp
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p

    def update_response_temperature(self, new_temp: float) -> None:
        """Dynamically adapts the response temperature without interrupting generation."""
        if new_temp < 0.01:
            new_temp = 0.0
        self.resp_sampler, _ = safe_make_sampler(new_temp, self.top_p, self.top_k, self.min_p)
        self.resp_temp = new_temp

    def __call__(self, logits: mx.array) -> mx.array:
        if self.tracker is not None and getattr(self.tracker, "is_thinking", False):
            return self.think_sampler(logits)
        return self.resp_sampler(logits)


def safe_make_sampler(
    temp: float, top_p: float, top_k: int, min_p: float
) -> Tuple[Callable[[mx.array], mx.array], Dict[str, Any]]:
    """Helper to build a sampler with progressive fallback if any kwarg is unsupported."""
    kwargs: Dict[str, Any] = {
        "temp": temp,
        "top_p": top_p,
        "top_k": top_k,
        "min_p": min_p,
    }
    while True:
        try:
            return make_sampler(**kwargs), kwargs
        except TypeError as e:
            msg = str(e)
            removed = False
            for key in list(kwargs.keys()):
                if f"'{key}'" in msg:
                    kwargs.pop(key, None)
                    removed = True
                    break
            if not removed:
                return make_sampler(temp=temp), {"temp": temp}


def build_dual_phase_sampler(
    think_temp: float,
    resp_temp: float,
    top_p: float,
    top_k: int,
    min_p: float,
    enable_thinking: bool = True,
    tracker: Any = None,
) -> Tuple[Callable[[mx.array], mx.array], Dict[str, Any]]:
    """Builds a dual-phase sampler or single-phase sampler based on configuration."""
    if think_temp < 0.01:
        think_temp = 0.0
    if resp_temp < 0.01:
        resp_temp = 0.0

    think_sampler, applied_think_kwargs = safe_make_sampler(think_temp, top_p, top_k, min_p)
    resp_sampler, applied_resp_kwargs = safe_make_sampler(resp_temp, top_p, top_k, min_p)
    if enable_thinking and tracker is not None:
        sampler = DualPhaseSampler(
            think_sampler,
            resp_sampler,
            tracker=tracker,
            think_temp=think_temp,
            resp_temp=resp_temp,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
        )
        applied_kwargs = {
            "mode": "dual_phase",
            "think_temp": think_temp,
            "resp_temp": resp_temp,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
        }
    else:
        sampler = resp_sampler
        applied_kwargs = {
            "mode": "single_phase",
            "temp": resp_temp,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
        }

    return sampler, applied_kwargs
