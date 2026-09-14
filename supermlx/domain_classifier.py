"""
[AI_DIRECTIVE]
ROL: Clasificador semántico O(1) de intención de dominio por expertos MoE de prefill.
OBJETIVO: Identificar en tiempo real el dominio del prompt (Prosa vs Código/Matemática vs Default)
          para suministrar la temperatura de respuesta óptima sin latencia ni modelos secundarios.
ENTRADAS: Conjunto de expertos Top-K activados durante el prefill de Layer 20 (Set[int]).
SALIDAS: Tupla tipada (nombre_dominio: str, temperatura_asignada: float).
REGLAS INVIOLABLES:
- Prohibido hardcoding de IDs de expertos o temperaturas (SSoT en config / .env)
- Evaluación puramente de conjuntos en O(1) (cero sincronización pesada ni bloqueo)
- Prohibido modificar logits ni alterar el flujo determinístico de inferencia
- Respetar siempre la temperatura explícita si el cliente la proveyó en su payload
SSoT: Este módulo es la única fuente de verdad para la clasificación de dominio y asignación de temperatura adaptativa.
"""

import logging
import threading
from typing import Set, Tuple, Dict, Any, Optional
import mlx.core as mx
from .config import SETTINGS

logger = logging.getLogger("supermlx.domain_classifier")

# Contexto local por hilo para vincular de forma segura el prefill con el sampler del request
_thread_context = threading.local()


def _parse_markers(csv_str: str) -> Set[int]:
    """Parsea una lista de enteros separados por comas desde la configuración."""
    try:
        return {int(x.strip()) for x in csv_str.split(",") if x.strip()}
    except Exception as e:
        logger.error("[ERROR] Error parseando marcadores '%s': %s", csv_str, e)
        return set()


class DomainClassifier:
    """Clasificador de dominio basado en marcadores de expertos de prefill MoE."""

    def __init__(self) -> None:
        self.enabled: bool = SETTINGS.adaptive_temperature_enabled
        self.layer: int = SETTINGS.adaptive_temperature_layer
        self.markers_prosa: Set[int] = _parse_markers(SETTINGS.adaptive_markers_prosa)
        self.markers_codigo_mate: Set[int] = _parse_markers(SETTINGS.adaptive_markers_codigo_mate)

        self.temp_prosa: float = SETTINGS.adaptive_temperature_prosa
        self.temp_codigo_mate: float = SETTINGS.adaptive_temperature_codigo_mate
        self.temp_default: float = SETTINGS.adaptive_temperature_default

        logger.info(
            "[CONFIG] DomainClassifier inicializado | layer=%d | Prosa(E%s)=%.2f | Codigo/Mate(N=%d)=%.2f | Default=%.2f | Enabled=%s",
            self.layer,
            sorted(list(self.markers_prosa)),
            self.temp_prosa,
            len(self.markers_codigo_mate),
            self.temp_codigo_mate,
            self.temp_default,
            self.enabled,
        )

    def classify(self, top_experts: Set[int]) -> Tuple[str, float]:
        """
        Clasifica el dominio a partir del conjunto de expertos activados en prefill.
        Retorna (dominio, temperatura_recomendada).
        """
        if not self.enabled or not top_experts:
            return "DEFAULT", self.temp_default

        # 1. Regla Prosa (E19 validado en holdout con Precision 100%, Recall 90%)
        if bool(top_experts & self.markers_prosa):
            matched = top_experts & self.markers_prosa
            logger.info("[DECISION] Dominio PROSA detectado (Marcadores: %s) ➔ Temp=%.2f", sorted(list(matched)), self.temp_prosa)
            return "PROSA", self.temp_prosa

        # 2. Regla Código / Matemática (validado en holdout con Precision 90-100%)
        if bool(top_experts & self.markers_codigo_mate):
            matched = top_experts & self.markers_codigo_mate
            logger.info("[DECISION] Dominio CODIGO_MATE detectado (Marcadores: %s) ➔ Temp=%.2f", sorted(list(matched)), self.temp_codigo_mate)
            return "CODIGO_MATE", self.temp_codigo_mate

        # 3. Default fallback
        logger.info("[DECISION] Dominio DEFAULT (sin marcadores específicos) ➔ Temp=%.2f", self.temp_default)
        return "DEFAULT", self.temp_default


# Instancia global (Singleton SSoT)
_domain_classifier = DomainClassifier()


def get_domain_classifier() -> DomainClassifier:
    """Retorna la instancia singleton del clasificador de dominio."""
    return _domain_classifier


def classify_prefill_domain(top_experts: Set[int]) -> Tuple[str, float]:
    """Acceso funcional directo a la clasificación de dominio en O(1)."""
    return _domain_classifier.classify(top_experts)


def register_request_sampler(
    sampler: Any,
    request_id: str,
    client_specified_temp: bool = False,
    log_fn: Optional[Any] = None
) -> None:
    """Registra el sampler activo para el hilo actual del request."""
    _thread_context.sampler = sampler
    _thread_context.request_id = request_id
    _thread_context.client_specified_temp = client_specified_temp
    _thread_context.domain_classified = False
    _thread_context.log_fn = log_fn


def unregister_request_sampler() -> None:
    """Limpia el contexto del hilo al terminar el request."""
    _thread_context.sampler = None
    _thread_context.request_id = None
    _thread_context.client_specified_temp = False
    _thread_context.domain_classified = False
    _thread_context.log_fn = None


def on_prefill_routed(inds: mx.array) -> Tuple[str, float]:
    """
    Invocado desde el hook de Layer 20 durante el prefill (cuando num_tokens > 1).
    Extrae los expertos únicos, clasifica el dominio y actualiza el sampler de respuesta.
    """
    if getattr(_thread_context, "domain_classified", False):
        return "ALREADY_CLASSIFIED", 0.0

    try:
        # inds tiene forma (..., K) en prefill
        mx.eval(inds)
        unique_experts = set(inds.reshape(-1).tolist())

        domain, temp = _domain_classifier.classify(unique_experts)
        _thread_context.domain_classified = True

        sampler = getattr(_thread_context, "sampler", None)
        client_override = getattr(_thread_context, "client_specified_temp", False)
        req_id = getattr(_thread_context, "request_id", "?")
        log_fn = getattr(_thread_context, "log_fn", None)

        if sampler is not None and hasattr(sampler, "update_response_temperature"):
            if client_override:
                logger.info("[DECISION] Req %s: Dominio %s detectado, pero se respeta temperatura explícita del cliente.", req_id, domain)
                if log_fn:
                    log_fn("🏷️", f"Dominio detectado: {domain} (respetando temp cliente)")
            else:
                sampler.update_response_temperature(temp)
                logger.info("[DECISION] Req %s: Temperatura adaptada a %.2f (Dominio: %s)", req_id, temp, domain)
                if log_fn:
                    log_fn("🏷️", f"Dominio detectado: {domain} ➔ Temp={temp:.2f}")

        return domain, temp
    except Exception as e:
        logger.error("[ERROR] Error en on_prefill_routed: %s", e)
        return "ERROR", _domain_classifier.temp_default
