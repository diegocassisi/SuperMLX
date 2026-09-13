"""
[AI_DIRECTIVE]
ROL: Tap no-bloqueante de prefill MoE en Layer medio (Layer 20) para Clasificador de Temperatura Adaptativa y Telemetría.
OBJETIVO: Capturar los índices de routing durante el prefill de prompt para clasificar el dominio (Prosa vs Código/Matemática)
          y modular la temperatura en DualPhaseSampler en O(1), con soporte opcional de volcado de trazas de diagnóstico.
ENTRADAS: Activaciones del Layer target (Layer 20 por defecto) SparseMoeBlock de Qwen MoE.
SALIDAS: Notificación en prefill a domain_classifier y registro estructurado si MEASURE_ROUTER_ENTROPY=true.
REGLAS INVIOLABLES:
- Prohibido modificar o alterar logits de inferencia
- Cero sincronización pesada en el loop de decodificación (solo se evalúa en prefill)
- SSoT en config.SETTINGS
"""

import os
import time
import json
import logging
from typing import Optional, List, Dict, Any
from pathlib import Path
from collections import Counter
import mlx.core as mx
from .config import SETTINGS

logger = logging.getLogger("supermlx.router_tracer")

_TRACER_ENABLED: bool = os.getenv("MEASURE_ROUTER_ENTROPY", "false").lower() in ("1", "true", "yes")
_TARGET_LAYER: int = SETTINGS.adaptive_temperature_layer
_DOMAIN_TRACE_FILE: Path = Path(os.getenv("DOMAIN_TRACE_PATH", str(Path(__file__).resolve().parent.parent / "domain_smoke_test_trace.jsonl")))

# Buffer en memoria para recolectar tensores sin sincronizar con la CPU (modo diagnóstico)
_prefill_inds_tensors: List[mx.array] = []
_is_decode_phase: bool = False


def _get_layers(model: Any) -> list:
    if hasattr(model, "layers"):
        return model.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "language_model") and hasattr(model.language_model, "model") and hasattr(model.language_model.model, "layers"):
        return model.language_model.model.layers
    raise AttributeError("No layers found on model")


def attach_router_tracer(model: Any) -> bool:
    """
    Instala un interceptor pasivo en el bloque MoE del layer especificado (por defecto Layer 20).
    Utiliza re-asignación de __class__ para garantizar intercepción de dunder __call__ en Python.
    Retorna True si fue instalado exitosamente.
    """
    global _TRACER_ENABLED, _TARGET_LAYER
    _adaptive_enabled = SETTINGS.adaptive_temperature_enabled

    if not _TRACER_ENABLED and not _adaptive_enabled:
        logger.debug("[CONFIG] Ni RouterTracer ni Adaptive Temperature activos — tap inactivo.")
        return False

    try:
        layers = _get_layers(model)
        target_layer = layers[_TARGET_LAYER]
        mlp_block = target_layer.mlp

        OrigClass = type(mlp_block)

        class PassiveTappedMoeBlock(OrigClass):
            def __call__(self, x: mx.array) -> mx.array:
                global _is_decode_phase

                # 1. Routing idéntico al estándar
                if self.sharding_group is not None:
                    from mlx.nn.layers.distributed import sum_gradients
                    x = sum_gradients(self.sharding_group)(x)

                gates = self.gate(x)
                gates = mx.softmax(gates, axis=-1, precise=True)

                k = self.top_k
                inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
                scores = mx.take_along_axis(gates, inds, axis=-1)

                # 2. Renormalización al 7-simplex
                scores_norm = scores / mx.sum(scores, axis=-1, keepdims=True)

                # 3. Captura exclusiva durante la fase de prefill (x.shape[-2] > 1)
                num_tokens = x.shape[-2] if len(x.shape) >= 2 else 1
                if num_tokens > 1 and not _is_decode_phase:
                    # A. Notificar al clasificador adaptativo de temperatura
                    if SETTINGS.adaptive_temperature_enabled:
                        try:
                            from .domain_classifier import on_prefill_routed
                            on_prefill_routed(inds)
                        except Exception as e:
                            logger.error("[ERROR] Fallo al despachar on_prefill_routed: %s", e)

                    # B. Guardar tensor para diagnóstico si el tracer está activo
                    if _TRACER_ENABLED:
                        _prefill_inds_tensors.append(inds.reshape(-1))

                elif num_tokens == 1:
                    _is_decode_phase = True

                # 4. Forward estándar de expertos (única pasada)
                y = self.switch_mlp(x, inds)
                y_routed = (y * scores_norm[..., None]).sum(axis=-2)

                shared_y = self.shared_expert(x)
                shared_y = mx.sigmoid(self.shared_expert_gate(x)) * shared_y
                y_out = y_routed + shared_y

                if self.sharding_group is not None:
                    y_out = mx.distributed.all_sum(y_out, group=self.sharding_group)

                return y_out

        mlp_block.__class__ = PassiveTappedMoeBlock

        if SETTINGS.adaptive_temperature_enabled:
            print(f"  🧠 Adaptive Temperature activo en Layer {_TARGET_LAYER} (Clasificador O(1))")
            logger.info("[INICIO] Adaptive Temperature activo en Layer %d (Clasificador O(1))", _TARGET_LAYER)
        if _TRACER_ENABLED:
            print(f"  📐 RouterTracer diagnóstico activo en Layer {_TARGET_LAYER}")
            logger.info("[INICIO] RouterTracer diagnóstico activo en Layer %d", _TARGET_LAYER)

        return True

    except Exception as e:
        logger.error("[ERROR] No se pudo instalar el tap en el Layer %d: %s", _TARGET_LAYER, e)
        return False


def flush_request_trace(request_id: str, prompt_preview: str = "") -> Optional[Dict[str, Any]]:
    """
    Sincroniza en un único lote los índices de expertos activados durante el prefill al terminar el request.
    Solo escribe en disco si MEASURE_ROUTER_ENTROPY=true.
    Limpia el estado de prefill para el siguiente request.
    """
    global _prefill_inds_tensors, _is_decode_phase
    _is_decode_phase = False

    try:
        from .domain_classifier import unregister_request_sampler
        unregister_request_sampler()
    except Exception:
        pass

    if not _TRACER_ENABLED or not _prefill_inds_tensors:
        _prefill_inds_tensors.clear()
        return None

    try:
        t0 = time.time()
        stacked = mx.concatenate(_prefill_inds_tensors, axis=0)
        mx.eval(stacked)
        inds = stacked.tolist()
        _prefill_inds_tensors.clear()
        sync_time_ms = round((time.time() - t0) * 1000, 2)

        if not inds:
            return None

        counts = Counter(inds)
        total_activations = len(inds)
        top_15 = counts.most_common(15)

        record = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "request_id": request_id,
            "prompt_preview": prompt_preview[:120].replace("\n", " "),
            "layer": _TARGET_LAYER,
            "total_prefill_activations": total_activations,
            "unique_experts": len(counts),
            "top_experts": [
                {"expert": exp, "count": cnt, "pct": round(cnt / total_activations * 100, 2)}
                for exp, cnt in top_15
            ],
            "full_histogram": {str(k): v for k, v in counts.items()},
            "sync_time_ms": sync_time_ms
        }

        with open(_DOMAIN_TRACE_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        return record

    except Exception as e:
        logger.error("[ERROR] Falló el flush de diagnóstico en Layer %d: %s", _TARGET_LAYER, e)
        _prefill_inds_tensors.clear()
        return None
