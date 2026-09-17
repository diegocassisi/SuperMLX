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

CHANGELOG (parches sobre la versión previa, API y contrato sin cambios):
- TASK_TAG_PATTERN anclado al inicio del buffer (antes: .search() en cualquier posición,
  podía disparar un cambio de temperatura a mitad de respuesta si el modelo mencionaba
  "[TASK: ...]" como texto normal, no como directiva).
- MAX_HEADER_CHARS: cota dura de crecimiento de buffer también en el path de tool-call
  (antes solo el path de task-tag tenía freno vía el fallback de 40 caracteres).
- TaskHeaderFilter/strip_task_header: elimina el tag del texto que efectivamente ve el
  usuario (antes el "[TASK: CODE]" quedaba filtrando en el output final).
- Validación explícita de temperaturas no-negativas/finitas en update_response_temperature
  y en la construcción del sampler (antes un valor negativo pasaba silencioso a make_sampler).
- safe_make_sampler valida top_p/min_p en [0,1] y top_k entero >= -1 ANTES de intentar
  construir el sampler, pero conserva el fallback progresivo de kwargs no soportados
  (protección ante drift de firma de mlx-lm entre versiones) — no se reemplaza ese
  mecanismo, solo se le agrega validación de valores.
"""

import math
import re
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
import mlx.core as mx
from mlx_lm.sample_utils import make_sampler
from .config import SETTINGS

logger = logging.getLogger("supermlx.sampling")

# Anclado al inicio del buffer: un tag solo cuenta como directiva de protocolo si es
# lo primero que aparece, no si el modelo lo menciona en medio del texto.
TASK_TAG_PATTERN = re.compile(
    r'^(?:\[\s*(?:TASK:\s*)?([A-Za-z0-9_]+)\s*\])|^(?:TASK:\s*([A-Za-z0-9_]+))',
    re.IGNORECASE,
)

TOOL_NAME_PATTERN = re.compile(
    r'(?:<function=([a-zA-Z0-9_\-\.]+)>)|(?:\"name\"\s*:\s*\"([a-zA-Z0-9_\-\.]+)\")',
    re.IGNORECASE,
)

# Cota dura de crecimiento de buffer. Evita costo O(n^2) del re.sub de limpieza si
# nunca se llega a resolver un match (p.ej. un tool call cuyo nombre nunca matchea).
MAX_HEADER_CHARS = 256

DYNAMIC_TEMP_PROTOCOL_DIRECTIVE = ("[MANDATORY TASK TAG DIRECTIVE] CRITICO: La primerisima palabra de tu respuesta final (fuera de <think>) DEBE ser obligatoriamente alguna de estas: [TASK: CODE] [TASK: TECH] [TASK: PROSE] [TASK: CREATIVE]")


def _validate_temperature(value: float) -> float:
    """Rechaza temperaturas negativas/no-finitas antes de que lleguen a make_sampler."""
    if isinstance(value, bool) or not math.isfinite(value) or value < 0:
        raise ValueError(f"temperature must be finite and nonnegative, got {value!r}")
    return float(value)


def resolve_task_tag_temperature(tag: str) -> Optional[float]:
    """Mapea una etiqueta semántica declarada por el modelo a su temperatura configurada."""
    tag_clean = tag.upper().strip()
    if tag_clean in ("CODE", "CODIGO", "MATH", "MATE"):
        return SETTINGS.temp_resp_code
    elif tag_clean in ("TECH", "TECNICO", "TECHNICAL"):
        return SETTINGS.temp_resp_tech
    elif tag_clean in ("PROSE", "PROSA", "CHAT"):
        return SETTINGS.temp_resp_prose
    elif tag_clean in ("CREATIVE", "CREATIVO", "BRAINSTORM"):
        return SETTINGS.temp_resp_creative
    return None


@dataclass(frozen=True)
class ThinkingSchedule:
    """Schedule continuo o bucketizado de enfriamiento durante la fase de thinking."""
    initial: float
    minimum: float
    budget: int
    exponent: float = 3.0
    fraction: float = 0.25

    def __post_init__(self) -> None:
        _validate_temperature(self.initial)
        _validate_temperature(self.minimum)
        if self.minimum > self.initial:
            raise ValueError("thinking minimum must not exceed initial temperature")
        if self.budget <= 0:
            raise ValueError("thinking schedule requires a positive token budget")
        if not math.isfinite(self.exponent) or self.exponent <= 0:
            raise ValueError("thinking exponent must be finite and positive")
        if not 0 < self.fraction <= 1:
            raise ValueError("thinking cooling fraction must be in (0, 1]")

    def temperature(self, tokens: int) -> float:
        progress = min(1.0, max(0.0, tokens / (self.budget * self.fraction)))
        return self.minimum + (self.initial - self.minimum) * (1.0 - progress) ** self.exponent


def parse_thinking_schedule(schedule_str: str) -> List[Tuple[int, float]]:
    """Parsea una cadena de schedule tipo '0:0.80,256:0.60,1024:0.35,3000:0.10'.

    Retorna una lista ordenada de tuplas (min_tokens, temperature) para los buckets.
    """
    if not schedule_str or not schedule_str.strip():
        return []
    buckets: List[Tuple[int, float]] = []
    pairs = [p.strip() for p in schedule_str.split(",") if p.strip()]
    for pair in pairs:
        if ":" not in pair:
            continue
        tok_str, temp_str = pair.split(":", 1)
        try:
            tokens = int(tok_str.strip())
            temp = _validate_temperature(float(temp_str.strip()))
            if tokens < 0:
                continue
            buckets.append((tokens, temp))
        except (ValueError, TypeError):
            continue
    buckets.sort(key=lambda x: x[0])
    if buckets and buckets[0][0] > 0:
        buckets.insert(0, (0, buckets[0][1]))
    return buckets


class TaskHeaderFilter:
    """Oculta del output visible al usuario el header [TASK: ...] cuando es reconocido,
    preservando el resto del texto verbatim. Sin esto, el tag queda filtrando en la
    respuesta final que ve el usuario.
    """

    def __init__(self) -> None:
        self.buffer: str = ""
        self.done: bool = False

    def feed(self, text: str) -> str:
        if self.done:
            return text
        self.buffer += text
        clean = self.buffer.lstrip()
        match = TASK_TAG_PATTERN.match(clean)
        if match and resolve_task_tag_temperature(match.group(1) or match.group(2)) is not None:
            self.done = True
            result = clean[match.end():]
            self.buffer = ""
            return result
        possible = (
            not clean
            or clean.startswith("[")
            or "task:".startswith(clean.lower())
            or clean.lower().startswith("task:")
        )
        if match or not possible or len(self.buffer) >= MAX_HEADER_CHARS:
            return self.finish()
        return ""

    def finish(self) -> str:
        clean = self.buffer.lstrip()
        match = TASK_TAG_PATTERN.match(clean + "\n")
        result = self.buffer
        if match and resolve_task_tag_temperature(match.group(1) or match.group(2)) is not None:
            result = clean[min(match.end(), len(clean)):]
        self.buffer = ""
        self.done = True
        return result


def strip_task_header(text: str) -> str:
    filter_ = TaskHeaderFilter()
    return filter_.feed(text) + filter_.finish()


class DynamicTaskDetector:
    """Detecta la etiqueta de protocolo semántica en los primeros tokens de respuesta
    y actualiza la temperatura de respuesta del sampler de forma dinámica.
    """

    def __init__(
        self,
        sampler: Any,
        request_id: str,
        client_has_custom_temp: bool = False,
        client_temp: Optional[float] = None,
        log_fn: Optional[Callable[..., Any]] = None,
        pipeline_log_fn: Optional[Callable[..., Any]] = None,
    ) -> None:
        self.sampler = sampler
        self.request_id = request_id
        self.client_has_custom_temp = client_has_custom_temp
        self.client_temp = client_temp
        self.log_fn = log_fn
        self.pipeline_log_fn = pipeline_log_fn
        self.detected: bool = False
        self.buffer: str = ""
        self.tag: str = "DEFAULT"

    def feed(self, text: str, is_responding: bool) -> None:
        if self.detected or not is_responding or not text:
            return

        # Cap duro: evita crecimiento sin cota del buffer (y del costo del re.sub sobre
        # él) si nunca se resuelve un match, p.ej. un tool call cuyo nombre no matchea.
        remaining = MAX_HEADER_CHARS - len(self.buffer)
        if remaining <= 0:
            self.buffer = self.buffer[:MAX_HEADER_CHARS]
        else:
            self.buffer += text[:remaining]

        clean_buffer = re.sub(r'^(?:\s*</?think>\s*)*\s*', '', self.buffer, flags=re.IGNORECASE)

        if any(marker in clean_buffer.lower() for marker in ("<tool_call", "<function=", "<|tool_call|")):
            tool_name_match = TOOL_NAME_PATTERN.search(clean_buffer)
            # Si aún no tenemos el nombre de la tool y el buffer es corto, esperar más tokens
            if (
                not tool_name_match
                and len(clean_buffer) < 80
                and not ("\n" in clean_buffer and len(clean_buffer) > 30)
                and len(self.buffer) < MAX_HEADER_CHARS
            ):
                return

            tool_name = (tool_name_match.group(1) or tool_name_match.group(2)).lower() if tool_name_match else ""
            is_coding_tool = any(k in tool_name for k in ("write", "edit", "patch", "replace", "create", "modify", "code", "file"))

            self.detected = True
            if is_coding_tool:
                self.tag = "CODE"
                target_temp = SETTINGS.temp_resp_code
            else:
                self.tag = "TOOLS"
                target_temp = SETTINGS.tool_calling_temperature

            if hasattr(self.sampler, "update_response_temperature"):
                self.sampler.update_response_temperature(target_temp)

            if self.log_fn:
                self.log_fn(
                    "💻" if is_coding_tool else "🔧",
                    f"Tool Call [{tool_name or 'generic'}] detectado ➔ Resp Temp={target_temp:.2f} [{self.tag}]",
                    request_id=self.request_id,
                )
            if self.pipeline_log_fn:
                self.pipeline_log_fn(
                    "SAMPLING",
                    self.request_id,
                    f"Tool call detected ({tool_name or 'generic'}) -> Resp Temp={target_temp:.2f} [{self.tag}]",
                )
            return

        # Anclado: solo cuenta como directiva si aparece al inicio del buffer limpio,
        # no en cualquier posición (evita falsos positivos a mitad de respuesta).
        match = TASK_TAG_PATTERN.match(clean_buffer)
        if match:
            raw_tag = match.group(1) or match.group(2)
            target_temp = resolve_task_tag_temperature(raw_tag)
            if target_temp is not None:
                self.detected = True
                self.tag = raw_tag.upper().strip()
                if hasattr(self.sampler, "update_response_temperature"):
                    self.sampler.update_response_temperature(target_temp)
                    if self.log_fn:
                        self.log_fn(
                            "🏷️",
                            f"Tag [{self.tag}] ➔ Resp Temp={target_temp:.2f}",
                            request_id=self.request_id,
                        )
                    if self.pipeline_log_fn:
                        self.pipeline_log_fn(
                            "SAMPLING",
                            self.request_id,
                            f"Dynamic Temp adapted to {target_temp:.2f} via [{self.tag}]",
                        )
                return
        if len(clean_buffer) > 40 or len(self.buffer) >= MAX_HEADER_CHARS:
            self.detected = True
            self.tag = "DEFAULT"
            fallback_temp = (
                min(self.client_temp, SETTINGS.response_temperature)
                if self.client_has_custom_temp and self.client_temp is not None
                else SETTINGS.response_temperature
            )
            if hasattr(self.sampler, "update_response_temperature"):
                self.sampler.update_response_temperature(fallback_temp)
            current_temp = getattr(self.sampler, "resp_temp", fallback_temp)
            preview = repr(clean_buffer[:30])
            if self.log_fn:
                self.log_fn(
                    "🏷️",
                    f"Sin Tag de tarea en prefijo ({preview}) ➔ Fallback temp={current_temp:.2f}",
                    request_id=self.request_id,
                )
            if self.pipeline_log_fn:
                self.pipeline_log_fn(
                    "SAMPLING",
                    self.request_id,
                    f"No task tag detected in prefix {preview} -> Fallback temp={current_temp:.2f}",
                )


class DualPhaseSampler:
    """Dynamic Dual-Phase Sampler for reasoning models.

    Routes token selection to think_sampler while tracker.is_thinking is True,
    and switches to resp_sampler as soon as tracker transitions to responding.
    Supports dynamic adaptation of response temperature during generation and
    pre-instantiated bucketized thinking temperature schedules.
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
        think_buckets: Optional[List[Tuple[int, Callable[[mx.array], mx.array], float]]] = None,
    ) -> None:
        self.think_sampler = think_sampler
        self.resp_sampler = resp_sampler
        self.tracker = tracker
        self._think_temp = _validate_temperature(think_temp)
        self.resp_temp = _validate_temperature(resp_temp)
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.think_buckets = think_buckets or []

    @property
    def think_temp(self) -> float:
        """Returns the active thinking temperature (reads bucket if scheduled)."""
        if self.think_buckets and self.tracker is not None:
            count = getattr(self.tracker, "thinking_count", 0)
            selected_temp = self.think_buckets[0][2]
            for min_tok, _, temp in self.think_buckets:
                if count >= min_tok:
                    selected_temp = temp
                else:
                    break
            return selected_temp
        return self._think_temp

    @think_temp.setter
    def think_temp(self, value: float) -> None:
        self._think_temp = _validate_temperature(value)

    def update_response_temperature(self, new_temp: float) -> None:
        """Dynamically adapts the response temperature without interrupting generation."""
        new_temp = _validate_temperature(new_temp)
        if new_temp < 0.01:
            new_temp = 0.0
        self.resp_sampler, _ = safe_make_sampler(new_temp, self.top_p, self.top_k, self.min_p)
        self.resp_temp = new_temp

    def __call__(self, logits: mx.array) -> mx.array:
        if self.tracker is not None and getattr(self.tracker, "is_thinking", False):
            if self.think_buckets:
                count = getattr(self.tracker, "thinking_count", 0)
                active_sampler = self.think_buckets[0][1]
                for min_tok, sampler_fn, _ in self.think_buckets:
                    if count >= min_tok:
                        active_sampler = sampler_fn
                    else:
                        break
                return active_sampler(logits)
            return self.think_sampler(logits)
        return self.resp_sampler(logits)


def safe_make_sampler(
    temp: float, top_p: float, top_k: int, min_p: float
) -> Tuple[Callable[[mx.array], mx.array], Dict[str, Any]]:
    """Helper to build a sampler with progressive fallback if any kwarg is unsupported.

    Valida los VALORES antes de intentar construir (temp no-negativa/finita,
    top_p/min_p en [0,1], top_k entero >= -1). Esto es independiente del fallback
    progresivo de abajo, que sigue existiendo para absorber drift de FIRMA de
    make_sampler entre versiones de mlx-lm sin romper en producción.
    """
    temp = _validate_temperature(temp)
    if not (0 <= top_p <= 1):
        raise ValueError(f"top_p must be in [0, 1], got {top_p!r}")
    if not (0 <= min_p <= 1):
        raise ValueError(f"min_p must be in [0, 1], got {min_p!r}")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < -1:
        raise ValueError(f"top_k must be an integer >= -1, got {top_k!r}")

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
    schedule: Optional[str] = None,
) -> Tuple[Callable[[mx.array], mx.array], Dict[str, Any]]:
    """Builds a dual-phase sampler or single-phase sampler based on configuration."""
    think_temp = _validate_temperature(think_temp)
    resp_temp = _validate_temperature(resp_temp)
    if think_temp < 0.01:
        think_temp = 0.0
    if resp_temp < 0.01:
        resp_temp = 0.0

    resp_sampler, applied_resp_kwargs = safe_make_sampler(resp_temp, top_p, top_k, min_p)
    if enable_thinking and tracker is not None:
        effective_schedule = schedule
        if effective_schedule is None and getattr(SETTINGS, "thinking_schedule_enabled", False):
            effective_schedule = getattr(SETTINGS, "thinking_temp_schedule", "")

        think_buckets: List[Tuple[int, Callable[[mx.array], mx.array], float]] = []
        if effective_schedule:
            parsed = parse_thinking_schedule(effective_schedule)
            for min_tok, b_temp in parsed:
                clean_b_temp = 0.0 if b_temp < 0.01 else b_temp
                b_sampler, _ = safe_make_sampler(clean_b_temp, top_p, top_k, min_p)
                think_buckets.append((min_tok, b_sampler, clean_b_temp))

        if think_buckets:
            initial_think_temp = think_buckets[0][2]
            think_sampler = think_buckets[0][1]
        else:
            initial_think_temp = think_temp
            think_sampler, _ = safe_make_sampler(initial_think_temp, top_p, top_k, min_p)

        sampler = DualPhaseSampler(
            think_sampler,
            resp_sampler,
            tracker=tracker,
            think_temp=initial_think_temp,
            resp_temp=resp_temp,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            think_buckets=think_buckets,
        )
        applied_kwargs = {
            "mode": "dual_phase",
            "think_temp": initial_think_temp,
            "resp_temp": resp_temp,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
        }
        if think_buckets and effective_schedule:
            applied_kwargs["thinking_schedule"] = effective_schedule
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
