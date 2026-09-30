"""
[AI_DIRECTIVE]
ROL: Motor de verificación y sampling especulativo para MTP (Multi-Token Prediction) en SuperMLX.
OBJETIVO: Implementar el algoritmo exacto de Rejection Sampling (Leviathan & Chen) con corrección residual para MTP depth=1.
ENTRADAS:
- model: Modelo con soporte MTP inyectado (validate_qwen3_5_mtp_support == True).
- prompt_tokens: mx.array de tokens iniciales.
- max_tokens: int, presupuesto de generación.
- temperature, top_p, min_p: parámetros de sampling.
- on_token_callback: Optional callback del FlightController para intervenciones in-flight.
SALIDAS:
- Generator con tokens emitidos, estadísticas de aceptación (alpha), tokens/s y conteos.
REGLAS INVIOLABLES:
- Prohibido modificar la distribución de probabilidad del modelo base (rejection sampling matemáticamente exacto).
- Prohibido print(). Usar exclusivamente logger con prefijos estándar ([INICIO], [CONFIG], [CALC], [DECISION], [RESULT], [ERROR]).
- Obligatorio tipado estricto en funciones públicas.
- Obligatorio recolectar telemetría de tasa de aceptación (alpha = accepted_drafts / total_drafts).
- Este módulo implementa MECANISMO (cómo inyectar tokens en el cache). La POLÍTICA (cuándo/qué inyectar)
  vive en supermlx.flight_controller.FlightController.
SSoT: Leviathan, Mattson, Liang (2023) / Chen et al. (2023) Speculative Decoding Theorem.

ASTRA Modification
Integration requirements:
- forward_with_hidden(..., n_confirmed=1) snapshots recurrent state after
  the confirmed input and before the speculative input.
- mtp_forward caches its confirmed input, not its predicted output.
- The caller advances any adaptive sampler tracker for each emitted token.
- Sampling implemented here supports temperature/top_k/min_p/top_p only,
  in that order; arbitrary sampler callables/processors are not reproduced.
- Exact rejection sampling is subject to floating-point arithmetic.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

import mlx.core as mx
from mlx_lm.sample_utils import apply_min_p, apply_top_k, apply_top_p
from supermlx.config import SETTINGS

logger = logging.getLogger(__name__)


def _distribution_from_logits(
    logits: mx.array,
    temperature: float = 0.0,
    top_p: float = 1.0,
    min_p: float = 0.0,
    top_k: int = 0,
) -> mx.array:
    """Distribución final idéntica a mlx_lm.sample_utils.make_sampler."""
    if logits.ndim != 1 or logits.size == 0:
        raise ValueError("Expected a nonempty 1D logits vector")
    logits = logits.astype(mx.float32)
    if temperature <= 0.0:
        return (mx.arange(logits.shape[0]) == mx.argmax(logits)).astype(mx.float32)
    logprobs = logits - mx.logsumexp(logits)
    if 0.0 < top_p < 1.0:
        logprobs = apply_top_p(logprobs, top_p)
    if min_p > 0.0:
        logprobs = apply_min_p(logprobs, min_p)
    if top_k > 0:
        logprobs = apply_top_k(logprobs, top_k)
    return mx.softmax(logprobs / temperature)


def _apply_logits_processors(
    processors: Optional[List[Callable[[mx.array, mx.array], mx.array]]],
    tokens: mx.array,
    logits: mx.array,
) -> mx.array:
    """Aplica la lista de logits_processors (repetition penalty, etc.) sobre logits 1D o 2D."""
    if not processors or tokens is None or tokens.size == 0:
        return logits
    is_1d = (logits.ndim == 1)
    x = logits[None] if is_1d else logits
    for proc in processors:
        x = proc(tokens, x)
    return x[0] if is_1d else x


def _is_cache_mtp_safe(prompt_cache: Optional[List[Any]], max_tokens: int) -> Tuple[bool, str]:
    """Evalúa dinámicamente por request si las capas en prompt_cache admiten rollback seguro."""
    if not prompt_cache:
        return True, ""
    try:
        from mlx_lm.models.cache import QuantizedKVCache, RotatingKVCache, BatchRotatingKVCache
    except ImportError:
        QuantizedKVCache = ()
        RotatingKVCache = ()
        BatchRotatingKVCache = ()

    for idx, layer in enumerate(prompt_cache):
        if isinstance(layer, QuantizedKVCache):
            return False, f"layer {idx} is QuantizedKVCache (quantized KV not trimmable for rejection rollback)"

        if isinstance(layer, (RotatingKVCache, BatchRotatingKVCache)):
            is_trimmable_fn = getattr(layer, "is_trimmable", None)
            if is_trimmable_fn is None or not is_trimmable_fn():
                offset = getattr(layer, "offset", 0)
                max_size = getattr(layer, "max_size", 0)
                return False, f"layer {idx} RotatingKVCache is not currently trimmable (offset {offset} >= max_size {max_size})"
            offset = getattr(layer, "offset", 0)
            max_size = getattr(layer, "max_size", 0)
            if max_size > 0 and (offset + max_tokens > max_size):
                return False, f"layer {idx} RotatingKVCache offset ({offset}) + max_tokens ({max_tokens}) > max_size ({max_size}); rollback would fail mid-generation"
    return True, ""


def _extract_sampling_params(
    sampler: Optional[Any],
    kwargs: Dict[str, Any],
) -> Tuple[Dict[str, Any], bool]:
    """
    Extrae parámetros de sampling del sampler o kwargs.
    Retorna (params_dict, is_opaque_sampler).
    """
    if sampler is None:
        return {
            "temp": float(kwargs.get("temperature", kwargs.get("temp", 0.0))),
            "top_p": float(kwargs.get("top_p", 1.0)),
            "min_p": float(kwargs.get("min_p", 0.0)),
            "top_k": int(kwargs.get("top_k", 0)),
        }, False

    # 1. Contrato explícito _mtp_sampling_params
    if hasattr(sampler, "_mtp_sampling_params"):
        p = sampler._mtp_sampling_params
        p_dict = p() if callable(p) else dict(p)
        return {
            "temp": float(p_dict.get("temp", p_dict.get("temperature", 0.0))),
            "top_p": float(p_dict.get("top_p", 1.0)),
            "min_p": float(p_dict.get("min_p", 0.0)),
            "top_k": int(p_dict.get("top_k", 0)),
        }, False

    # 2. DualPhaseSampler u objetos con atributos de temperatura
    if (
        hasattr(sampler, "resp_temp")
        or hasattr(sampler, "think_temp")
        or hasattr(sampler, "temp")
    ):
        return {
            "temp": float(getattr(sampler, "temp", getattr(sampler, "resp_temp", 0.0))),
            "top_p": float(getattr(sampler, "top_p", 1.0)),
            "min_p": float(getattr(sampler, "min_p", 0.0)),
            "top_k": int(getattr(sampler, "top_k", 0)),
        }, False

    # 3. Kwargs explícitos acompañando a sampler opaco
    if "temperature" in kwargs or "temp" in kwargs:
        return {
            "temp": float(kwargs.get("temperature", kwargs.get("temp"))),
            "top_p": float(kwargs.get("top_p", 1.0)),
            "min_p": float(kwargs.get("min_p", 0.0)),
            "top_k": int(kwargs.get("top_k", 0)),
        }, False

    # 4. Sampler opaco sin contrato
    if callable(sampler):
        return {}, True

    return {
        "temp": 0.0,
        "top_p": 1.0,
        "min_p": 0.0,
        "top_k": 0,
    }, False


def _sample_token(
    logits: mx.array,
    temperature: float = 0.0,
    top_p: float = 1.0,
    min_p: float = 0.0,
    top_k: int = 0,
) -> Tuple[int, mx.array]:
    probs = _distribution_from_logits(logits, temperature, top_p, min_p, top_k)
    if temperature <= 0.0:
        return int(mx.argmax(logits).item()), probs
    # log(0) = -inf: filtered-out tokens stay impossible.
    return int(mx.random.categorical(mx.log(probs)).item()), probs


def _verify_draft_token(
    draft_token: int,
    p_probs: mx.array,
    q_probs: mx.array,
    temperature: float,
) -> Tuple[bool, int]:
    if temperature <= 0.0:
        target = int(mx.argmax(p_probs).item())
        return draft_token == target, target
    p_val = p_probs[draft_token]
    q_val = q_probs[draft_token]
    # q_val is positive because the draft was sampled from q.
    accept = mx.random.uniform() * q_val < mx.minimum(p_val, q_val)
    if bool(accept.item()):
        return True, draft_token
    residual = mx.maximum(p_probs - q_probs, 0.0)
    # categorical normalizes the residual weights implicitly. No epsilon or
    # fallback to p: either would alter the intended distribution.
    total = mx.sum(residual)
    if not bool((mx.isfinite(total) & (total > 0)).item()):
        raise FloatingPointError("Invalid rejection residual; check logits and cache state")
    token = int(mx.random.categorical(mx.log(residual)).item())
    return False, token


def _rollback_draft_caches(
    model_cache: List[Any],
    mtp_cache: Optional[List[Any]] = None,
) -> None:
    """
    Restaura los cachés del trunk al estado previo al draft rechazado.
    - Capas SSM (ArraysCache): restaura (conv_snap, ssm_snap) de rollback_state.
    - Capas de atención (KVCache): recorta 1 token con trim(1).
    El cabezal MTP NO se recorta porque nunca procesó el draft rechazado.
    """
    for c in model_cache:
        if hasattr(c, "rollback_state") and c.rollback_state is not None:
            conv_snap, ssm_snap = c.rollback_state
            c[0] = conv_snap
            c[1] = ssm_snap
            c.rollback_state = None
            if hasattr(c, "advance"):
                c.advance(-1)
        elif hasattr(c, "trim") and getattr(c, "offset", 0) > 0:
            c.trim(1)



def _clear_draft_rollback(model_cache: List[Any]) -> None:
    """Limpia el rollback_state cuando el draft fue aceptado."""
    for c in model_cache:
        if hasattr(c, "rollback_state"):
            c.rollback_state = None


class MTPSpeculativeEngine:
    """
    Motor de generación especulativa con Multi-Token Prediction (MTP) integrado para SuperMLX.
    """

    def __init__(
        self,
        model: Any,
        temperature: float = 0.0,
        top_p: float = 1.0,
        min_p: float = 0.0,
        top_k: int = 0,
        sampler: Optional[Any] = None,
        adaptive_temperature: bool = False,
        tokenizer: Optional[Any] = None,
        on_token_callback: Optional[Any] = None,
        prefill_step_size: int = 256,
        logits_processors: Optional[List[Callable]] = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.min_p = min_p
        self.top_k = top_k
        self.sampler = sampler
        self.adaptive_temperature = adaptive_temperature
        self.tokenizer = tokenizer
        self._on_token_callback = on_token_callback
        self.prefill_step_size = prefill_step_size
        self.logits_processors = logits_processors
        logger.info(
            "[CONFIG] MTPSpeculativeEngine inicializado: temp=%.2f, top_p=%.2f, min_p=%.2f, top_k=%d, prefill_step=%d, logits_processors=%s, adaptive_temp=%s",
            self.temperature,
            self.top_p,
            self.min_p,
            self.top_k,
            self.prefill_step_size,
            bool(self.logits_processors),
            self.adaptive_temperature,
        )

    def _get_sampling_params(self) -> Tuple[float, float, float, int]:
        """
        Resuelve (temp, top_p, min_p, top_k) dinámicamente según la fase del sampler
        (DualPhaseSampler / FEATURE_ADAPTIVE_TEMPERATURE) o el modo estático.
        """
        if self.sampler is None:
            return self.temperature, self.top_p, self.min_p, self.top_k

        # 1. Contrato explícito si existe
        if hasattr(self.sampler, "_mtp_sampling_params"):
            p = self.sampler._mtp_sampling_params
            p_dict = p() if callable(p) else dict(p)
            return (
                float(p_dict.get("temp", p_dict.get("temperature", self.temperature))),
                float(p_dict.get("top_p", self.top_p)),
                float(p_dict.get("min_p", self.min_p)),
                int(p_dict.get("top_k", self.top_k)),
            )

        if not self.adaptive_temperature:
            return self.temperature, self.top_p, self.min_p, self.top_k

        # 2. Temperatura activa según la fase (thinking vs response)
        temp = self.temperature
        if hasattr(self.sampler, "tracker") and getattr(self.sampler.tracker, "is_thinking", False):
            temp = getattr(self.sampler, "think_temp", self.temperature)
        elif hasattr(self.sampler, "resp_temp"):
            temp = getattr(self.sampler, "resp_temp", self.temperature)
        elif hasattr(self.sampler, "temp"):
            temp = getattr(self.sampler, "temp", self.temperature)

        # 3. Filtros de sampling del sampler activo
        top_p = getattr(self.sampler, "top_p", self.top_p)
        min_p = getattr(self.sampler, "min_p", self.min_p)
        top_k = getattr(self.sampler, "top_k", self.top_k)

        # Sincronizar min_p durante pulso de Thermal Spark si está activo
        if (
            hasattr(self.sampler, "spark_controller")
            and getattr(self.sampler.spark_controller, "is_spark_active", False)
        ):
            min_p = getattr(self.sampler.spark_controller, "spark_min_p", min_p)

        return temp, top_p, min_p, top_k

    def _inject_tokens(
        self,
        inject_token_ids: List[int],
        cache: List[Any],
        mtp_cache: List[Any],
        confirmed_token: int,
        tokens_generated: int,
        drafts_accepted: int,
        drafts_attempted: int,
        curr_alpha: float,
        max_tokens: int = 512,
        eos_token_id: Optional[int] = None,
    ) -> Tuple[List[Dict[str, Any]], int, int, Any, List[Any], int, Any]:
        """Inject tokens into KV cache and prepare next draft.

        This is the MECHANISM for injection. The POLICY (when/what to inject)
        lives in FlightController.
        """
        # Respetar presupuesto de max_tokens
        rem_budget = max(0, max_tokens - tokens_generated)
        inject_token_ids = list(inject_token_ids)[:rem_budget]

        # Respetar EOS
        if eos_token_id is not None and eos_token_id in inject_token_ids:
            eos_idx = inject_token_ids.index(eos_token_id)
            inject_token_ids = inject_token_ids[:eos_idx + 1]

        if not inject_token_ids:
            return [], confirmed_token, None, mtp_cache, tokens_generated, None, None

        # Forward confirmed_token + all nudge tokens except last through trunk
        tokens_to_forward = [confirmed_token] + list(inject_token_ids[:-1])
        fwd_arr = mx.array([tokens_to_forward], dtype=mx.int32)
        fwd_logits, fwd_hidden = self.model.forward_with_hidden(
            fwd_arr, cache=cache
        )
        mx.eval(fwd_logits, fwd_hidden)

        # Build token dicts to yield (caller does actual yield from generate())
        tokens_to_yield: List[Dict[str, Any]] = []
        for n_tok in inject_token_ids:
            tokens_generated += 1
            # Notify flight controller of injected token
            if self._on_token_callback:
                cb = self._on_token_callback
                if hasattr(cb, '__self__') and hasattr(cb.__self__, 'record_token'):
                    cb.__self__.record_token(int(n_tok))
            tokens_to_yield.append({
                "token": int(n_tok),
                "is_speculative": False,
                "accepted": True,
                "tokens_generated": tokens_generated,
                "drafts_accepted": drafts_accepted,
                "drafts_attempted": drafts_attempted,
                "alpha": curr_alpha,
            })
            if eos_token_id is not None and int(n_tok) == eos_token_id:
                break

        # Last injected token becomes the new confirmed token
        new_confirmed = int(inject_token_ids[-1])
        hidden_at_confirmed = fwd_hidden[:, -1:, :]
        new_mtp_cache = self.model.make_mtp_cache()

        if (eos_token_id is not None and new_confirmed == eos_token_id) or tokens_generated >= max_tokens:
            return tokens_to_yield, new_confirmed, hidden_at_confirmed, new_mtp_cache, tokens_generated, None, None

        # Generate new speculative draft from the last injected token
        mtp_logits = self.model.mtp_forward(
            hidden_at_confirmed,
            mx.array([[new_confirmed]]),
            mtp_cache=new_mtp_cache,
        )
        mx.eval(mtp_logits)
        next_temp, next_top_p, next_min_p, next_top_k = self._get_sampling_params()
        draft_tok, q2_probs = _sample_token(
            mtp_logits[0, -1],
            temperature=next_temp,
            top_p=next_top_p,
            min_p=next_min_p,
            top_k=next_top_k,
        )

        return tokens_to_yield, new_confirmed, hidden_at_confirmed, new_mtp_cache, tokens_generated, draft_tok, q2_probs

    def generate(
        self,
        prompt_tokens: mx.array,
        max_tokens: int = 512,
        eos_token_id: Optional[int] = None,
        prompt_cache: Optional[List[Any]] = None,
        mtp_cache: Optional[List[Any]] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Ejecuta el bucle de generación especulativa MTP rindiendo tokens y telemetría en tiempo real.
        Utiliza verificación en batch de 2 tokens (n_confirmed=1) con emisión de bonus token (1+alpha).
        """
        logger.info("[INICIO] Generación especulativa MTP iniciada (max_tokens=%d)", max_tokens)
        if max_tokens < 0:
            raise ValueError("max_tokens must be nonnegative")
        if max_tokens == 0:
            return
        start_time = time.perf_counter()

        # 1. Inicialización de KV caches (trunk + cabeza MTP)
        from mlx_lm.models.cache import make_prompt_cache

        cache = prompt_cache if prompt_cache is not None else make_prompt_cache(self.model)
        if mtp_cache is None:
            mtp_cache = self.model.make_mtp_cache()

        # 2. Prefill del prompt: trunk + cabezal MTP sincronizados por bloques.
        # El cabezal MTP necesita los pares (hidden[i], token[i+1]) del prompt
        # para tener contexto de atención. Sin esto, su attention está vacía.
        if prompt_tokens.ndim == 1:
            prompt_tokens = prompt_tokens[None]

        tokens_context = [int(t) for t in prompt_tokens[0].tolist()] if prompt_tokens.size > 0 else []
        previous_hidden = None
        for offset in range(0, prompt_tokens.shape[1], self.prefill_step_size):
            chunk = prompt_tokens[:, offset:offset + self.prefill_step_size]
            is_last_chunk = (offset + chunk.shape[1] == prompt_tokens.shape[1])
            logits, hidden = self.model.forward_with_hidden(
                chunk, cache=cache, logits_to_keep=1 if is_last_chunk else 0)

            # Alimentar pares (hidden[i], token[i+1]) al cabezal MTP
            if previous_hidden is not None:
                head_hidden = mx.concatenate([previous_hidden, hidden[:, :-1]], axis=1)
                head_tokens = chunk
            else:
                head_hidden = hidden[:, :-1]
                head_tokens = chunk[:, 1:]

            if head_tokens.size:
                self.model.mtp_forward(
                    head_hidden, head_tokens, mtp_cache=mtp_cache, logits_to_keep=0)

            mx.eval(hidden, [c.state for c in cache],
                    [c.state for c in mtp_cache if getattr(c, "offset", 0) > 0])
            previous_hidden = hidden[:, -1:]

        pre_norm = previous_hidden

        curr_temp, curr_top_p, curr_min_p, curr_top_k = self._get_sampling_params()
        first_token_logits = logits[0, -1]
        if self.logits_processors and tokens_context:
            first_token_logits = _apply_logits_processors(
                self.logits_processors, mx.array(tokens_context), first_token_logits
            )
        token_t1, _ = _sample_token(
            first_token_logits,
            temperature=curr_temp,
            top_p=curr_top_p,
            min_p=curr_min_p,
            top_k=curr_top_k,
        )
        tokens_context.append(int(token_t1))

        tokens_generated = 1
        drafts_attempted = 0
        drafts_accepted = 0
        curr_alpha = 0.0

        # Helper: notify flight controller of generated tokens
        def _fc_record(token_id: int) -> None:
            if self._on_token_callback:
                cb = self._on_token_callback
                if hasattr(cb, '__self__') and hasattr(cb.__self__, 'record_token'):
                    cb.__self__.record_token(token_id)

        _fc_record(int(token_t1))

        yield {
            "token": token_t1,
            "is_speculative": False,
            "accepted": True,
            "tokens_generated": tokens_generated,
            "drafts_accepted": drafts_accepted,
            "drafts_attempted": drafts_attempted,
            "alpha": 0.0,
        }

        if eos_token_id is not None and token_t1 == eos_token_id:
            logger.info("[DECISION] EOS token alcanzado en prefill. Finalizando.")
            return

        if tokens_generated >= max_tokens:
            return

        # 3. Primer draft MTP a partir del último estado oculto del prompt prefill
        hidden_at_confirmed = pre_norm[:, -1:, :]
        confirmed_token = token_t1

        mtp_logits = self.model.mtp_forward(
            hidden_at_confirmed,
            mx.array([[confirmed_token]]),
            mtp_cache=mtp_cache,
        )
        mx.eval(mtp_logits)

        curr_temp, curr_top_p, curr_min_p, curr_top_k = self._get_sampling_params()
        draft_tok, q2_probs = _sample_token(
            mtp_logits[0, -1],
            temperature=curr_temp,
            top_p=curr_top_p,
            min_p=curr_min_p,
            top_k=curr_top_k,
        )

        while tokens_generated < max_tokens:
            # ── FLIGHT CONTROLLER HOOK ────────────────────────────────────
            if self._on_token_callback is not None:
                action = self._on_token_callback(tokens_generated, confirmed_token)
                if action is not None:
                    if getattr(action, "hard_break", False):
                        break
                    inject_list = getattr(action, "inject_tokens", None)
                    if inject_list:
                        # Inject tokens into KV cache and resume speculative decoding
                        inject_result = self._inject_tokens(
                            inject_list, cache, mtp_cache,
                            confirmed_token, tokens_generated,
                            drafts_accepted, drafts_attempted, curr_alpha,
                            max_tokens=max_tokens,
                            eos_token_id=eos_token_id,
                        )
                        tokens_to_yield, confirmed_token, hidden_at_confirmed, mtp_cache, tokens_generated, draft_tok, q2_probs = inject_result
                        # Yield injected tokens to consumer
                        for tok_dict in tokens_to_yield:
                            yield tok_dict
                        if tokens_generated >= max_tokens or (eos_token_id is not None and confirmed_token == eos_token_id):
                            break
                        if draft_tok is None:
                            break
                        continue

            # 4. Fase VERIFY en batch de 2 tokens: [confirmed_token, draft_tok]
            # n_confirmed=1 le indica a GatedDeltaNet que guarde snapshot SSM entre ellos.
            y_with_draft = mx.array([[confirmed_token, draft_tok]])
            target_logits, hidden = self.model.forward_with_hidden(
                y_with_draft,
                cache=cache,
                n_confirmed=1,
            )

            verify_logits = target_logits[0, 0]
            bonus_logits = target_logits[0, 1]

            if self.logits_processors and tokens_context:
                verify_logits = _apply_logits_processors(
                    self.logits_processors, mx.array(tokens_context), verify_logits
                )

            curr_temp, curr_top_p, curr_min_p, curr_top_k = self._get_sampling_params()
            p_probs = _distribution_from_logits(
                verify_logits,
                temperature=curr_temp,
                top_p=curr_top_p,
                min_p=curr_min_p,
                top_k=curr_top_k,
            )

            is_accepted, committed_token = _verify_draft_token(
                draft_token=draft_tok,
                p_probs=p_probs,
                q_probs=q2_probs,
                temperature=curr_temp,
            )
            drafts_attempted += 1

            if is_accepted:
                _clear_draft_rollback(cache)
                drafts_accepted += 1
                curr_alpha = (drafts_accepted / max(drafts_attempted, 1)) * 100.0

                tokens_context.append(int(draft_tok))
                # Emitir draft token confirmado
                tokens_generated += 1
                _fc_record(int(draft_tok))
                yield {
                    "token": draft_tok,
                    "is_speculative": True,
                    "accepted": True,
                    "tokens_generated": tokens_generated,
                    "drafts_accepted": drafts_accepted,
                    "drafts_attempted": drafts_attempted,
                    "alpha": curr_alpha,
                }
                if eos_token_id is not None and draft_tok == eos_token_id:
                    logger.info("[DECISION] EOS token alcanzado por draft. Finalizando.")
                    break
                if tokens_generated >= max_tokens:
                    break

                # Resolve again after the consumer processes the accepted token.
                curr_temp, curr_top_p, curr_min_p, curr_top_k = self._get_sampling_params()
                if self.logits_processors and tokens_context:
                    bonus_logits = _apply_logits_processors(
                        self.logits_processors, mx.array(tokens_context), bonus_logits
                    )
                # Muestrear y emitir bonus token proyectado por el trunk en posición 1
                bonus_tok, _ = _sample_token(
                    bonus_logits,
                    temperature=curr_temp,
                    top_p=curr_top_p,
                    min_p=curr_min_p,
                    top_k=curr_top_k,
                )
                tokens_context.append(int(bonus_tok))
                tokens_generated += 1
                _fc_record(int(bonus_tok))
                yield {
                    "token": bonus_tok,
                    "is_speculative": False,
                    "accepted": True,
                    "tokens_generated": tokens_generated,
                    "drafts_accepted": drafts_accepted,
                    "drafts_attempted": drafts_attempted,
                    "alpha": curr_alpha,
                }
                if eos_token_id is not None and bonus_tok == eos_token_id:
                    logger.info("[DECISION] EOS token alcanzado por bonus token. Finalizando.")
                    break
                if tokens_generated >= max_tokens:
                    break

                # Siguiente ciclo: alimentar ambos pares al MTP head para mantener coherencia posicional.
                # Par 1: (hidden_confirmed, draft_tok) — solo actualiza cache, sin proyectar a vocab.
                hidden_at_confirmed = hidden[:, 0:1, :]
                self.model.mtp_forward(
                    hidden_at_confirmed,
                    mx.array([[draft_tok]]),
                    mtp_cache=mtp_cache,
                    logits_to_keep=0,
                )
                # Par 2: (hidden_draft, bonus_tok) — genera el siguiente draft.
                hidden_at_draft = hidden[:, 1:2, :]
                mtp_logits = self.model.mtp_forward(
                    hidden_at_draft,
                    mx.array([[bonus_tok]]),
                    mtp_cache=mtp_cache,
                )
                next_temp, next_top_p, next_min_p, next_top_k = self._get_sampling_params()
                next_draft_tok, q2_probs = _sample_token(
                    mtp_logits[0, -1],
                    temperature=next_temp,
                    top_p=next_top_p,
                    min_p=next_min_p,
                    top_k=next_top_k,
                )
                confirmed_token = bonus_tok
                draft_tok = next_draft_tok

            else:
                # RECHAZO: Rollback en trunk únicamente (el cabezal MTP nunca evaluó el draft rechazado)
                _rollback_draft_caches(cache, mtp_cache=None)
                curr_alpha = (drafts_accepted / max(drafts_attempted, 1)) * 100.0

                tokens_context.append(int(committed_token))
                # Emitir el token corregido
                tokens_generated += 1
                _fc_record(int(committed_token))
                yield {
                    "token": committed_token,
                    "is_speculative": False,
                    "accepted": False,
                    "tokens_generated": tokens_generated,
                    "drafts_accepted": drafts_accepted,
                    "drafts_attempted": drafts_attempted,
                    "alpha": curr_alpha,
                }
                if eos_token_id is not None and committed_token == eos_token_id:
                    logger.info("[DECISION] EOS token alcanzado por token corregido. Finalizando.")
                    break
                if tokens_generated >= max_tokens:
                    break

                # Siguiente ciclo tras rechazo: el estado oculto es la posición 0 (confirmed_token)
                hidden_at_confirmed = hidden[:, 0:1, :]
                mtp_logits = self.model.mtp_forward(
                    hidden_at_confirmed,
                    mx.array([[committed_token]]),
                    mtp_cache=mtp_cache,
                )
                next_temp, next_top_p, next_min_p, next_top_k = self._get_sampling_params()
                next_draft_tok, q2_probs = _sample_token(
                    mtp_logits[0, -1],
                    temperature=next_temp,
                    top_p=next_top_p,
                    min_p=next_min_p,
                    top_k=next_top_k,
                )
                confirmed_token = committed_token
                draft_tok = next_draft_tok

        elapsed = time.perf_counter() - start_time
        alpha = (drafts_accepted / max(drafts_attempted, 1)) * 100.0
        tps = tokens_generated / max(elapsed, 0.001)

        logger.info(
            "[RESULT] MTP Finalizado: %d tokens en %.2fs (%.1f t/s) | Tasa Aceptación alpha=%.1f%% (%d/%d)",
            tokens_generated,
            elapsed,
            tps,
            alpha,
            drafts_accepted,
            drafts_attempted,
        )


def stream_generate_mtp(
    model: Any,
    tokenizer: Any,
    prompt: Union[str, mx.array, List[int]],
    max_tokens: int = 512,
    sampler: Optional[Any] = None,
    prompt_cache: Optional[List[Any]] = None,
    mtp_cache: Optional[List[Any]] = None,
    adaptive_temperature: Optional[bool] = None,
    logits_processors: Optional[List[Callable]] = None,
    prefill_step_size: Optional[int] = None,
    **kwargs: Any,
) -> Generator[Any, None, None]:
    """
    Generador compatible con stream_generate() de mlx_lm para integración transparente en SuperMLX.
    Produce instancias GenerationResponse con .text, .token, .from_draft y métricas de generación.
    Soporta sincronización dinámica con DualPhaseSampler y FEATURE_ADAPTIVE_TEMPERATURE.
    """
    from mlx_lm.generate import GenerationResponse
    from mlx_lm.tokenizer_utils import TokenizerWrapper

    if adaptive_temperature is None:
        try:
            from supermlx.config import SETTINGS
            adaptive_temperature = getattr(SETTINGS, "mtp_adaptive_temperature", True)
        except Exception:
            adaptive_temperature = True

    if tokenizer is not None and not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)

    if not isinstance(prompt, mx.array):
        if isinstance(prompt, str):
            add_special_tokens = tokenizer is None or tokenizer.bos_token is None or not prompt.startswith(
                tokenizer.bos_token
            )
            prompt = tokenizer.encode(prompt, add_special_tokens=add_special_tokens) if tokenizer else []
        prompt = mx.array(prompt)

    # Protección contra prompt / rest_tokens vacío
    if prompt.size == 0:
        logger.warning("[DECISION] Prompt vacío recibido en stream_generate_mtp. Retornando sin generar.")
        return

    # Guard de Caché Dinámico por Request: derivar a autoregresivo si el cache no es trimmable
    cache_safe, cache_reason = _is_cache_mtp_safe(prompt_cache, max_tokens)
    if not cache_safe:
        logger.warning(
            "[FALLBACK] Cache KV incompatible con rollback de MTP (%s); derivando a autoregresivo estándar",
            cache_reason,
        )
        import mlx_lm
        fallback_gen = getattr(mlx_lm, "stream_generate", None)
        if fallback_gen is None:
            from mlx_lm.generate import stream_generate as fallback_gen
        for resp in fallback_gen(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=sampler,
            prompt_cache=prompt_cache,
            logits_processors=logits_processors,
            **kwargs,
        ):
            if not hasattr(resp, "alpha"):
                resp.alpha = 0.0
            if not hasattr(resp, "drafts_accepted"):
                resp.drafts_accepted = 0
            if not hasattr(resp, "drafts_attempted"):
                resp.drafts_attempted = 0
            if not hasattr(resp, "from_draft"):
                resp.from_draft = False
            yield resp
        return

    # Extraer parámetros de sampling base del sampler; derivar si es opaco sin contrato
    sampling_params, is_opaque = _extract_sampling_params(sampler, kwargs)
    if is_opaque:
        logger.warning(
            "[FALLBACK] Sampler opaco sin contrato de parámetros (_mtp_sampling_params / temp); "
            "derivando a autoregresivo estándar de mlx_lm"
        )
        import mlx_lm
        fallback_gen = getattr(mlx_lm, "stream_generate", None)
        if fallback_gen is None:
            from mlx_lm.generate import stream_generate as fallback_gen
        for resp in fallback_gen(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=sampler,
            prompt_cache=prompt_cache,
            logits_processors=logits_processors,
            **kwargs,
        ):
            if not hasattr(resp, "alpha"):
                resp.alpha = 0.0
            if not hasattr(resp, "drafts_accepted"):
                resp.drafts_accepted = 0
            if not hasattr(resp, "drafts_attempted"):
                resp.drafts_attempted = 0
            if not hasattr(resp, "from_draft"):
                resp.from_draft = False
            yield resp
        return

    # Flight controller: policy for in-flight interventions (nudge, loop, spark)
    from supermlx.flight_controller import FlightController
    controller = FlightController(tokenizer=tokenizer, sampler=sampler) if tokenizer else None

    step_size = prefill_step_size if prefill_step_size is not None else kwargs.get("prefill_step_size", 256)
    effective_logits_processors = logits_processors if logits_processors is not None else kwargs.get("logits_processors", None)

    engine = MTPSpeculativeEngine(
        model=model,
        temperature=sampling_params["temp"],
        top_p=sampling_params["top_p"],
        min_p=sampling_params["min_p"],
        top_k=sampling_params["top_k"],
        sampler=sampler,
        adaptive_temperature=adaptive_temperature,
        tokenizer=tokenizer,
        on_token_callback=controller.evaluate if controller else None,
        prefill_step_size=step_size,
        logits_processors=effective_logits_processors,
    )

    detokenizer = tokenizer.detokenizer
    detokenizer.reset()

    eos_ids = set(getattr(tokenizer, "eos_token_ids", set()) or ())
    primary_eos = getattr(tokenizer, "eos_token_id", None)
    if primary_eos is not None:
        eos_ids.add(primary_eos)

    tic = time.perf_counter()
    tokens_yielded = 0
    token_id = None
    is_eos = False
    last_step = {}

    for step in engine.generate(
        prompt_tokens=prompt,
        max_tokens=max_tokens,
        eos_token_id=primary_eos,
        prompt_cache=prompt_cache,
        mtp_cache=mtp_cache,
    ):
        token_id = int(step["token"])
        last_step = step
        is_eos = token_id in eos_ids
        if is_eos:
            tokens_yielded += 1
            break

        detokenizer.add_token(token_id)
        tokens_yielded += 1

        if tokens_yielded >= max_tokens:
            break

        response = GenerationResponse(
            text=detokenizer.last_segment,
            token=token_id,
            logprobs=None,
            from_draft=step.get("is_speculative", False),
            prompt_tokens=prompt.size,
            prompt_tps=0.0,
            generation_tokens=tokens_yielded,
            generation_tps=tokens_yielded / max(time.perf_counter() - tic, 1e-6),
            peak_memory=mx.get_peak_memory() / 1e9,
            finish_reason=None,
        )
        response.drafts_accepted = step.get("drafts_accepted", 0)
        response.drafts_attempted = step.get("drafts_attempted", 0)
        response.alpha = step.get("alpha", 0.0)
        response.draft_accepted = step.get("accepted", True)

        yield response

    detokenizer.finalize()
    if token_id is not None:
        final_response = GenerationResponse(
            text=detokenizer.last_segment,
            token=token_id,
            logprobs=None,
            from_draft=last_step.get("is_speculative", False),
            prompt_tokens=prompt.size,
            prompt_tps=0.0,
            generation_tokens=tokens_yielded,
            generation_tps=tokens_yielded / max(time.perf_counter() - tic, 1e-6),
            peak_memory=mx.get_peak_memory() / 1e9,
            finish_reason="stop" if is_eos else "length",
        )
        final_response.drafts_accepted = last_step.get("drafts_accepted", 0)
        final_response.drafts_attempted = last_step.get("drafts_attempted", 0)
        final_response.alpha = last_step.get("alpha", 0.0)
        final_response.draft_accepted = last_step.get("accepted", True)
        yield final_response