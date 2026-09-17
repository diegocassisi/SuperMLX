"""
[AI_DIRECTIVE]
ROL: Motor de verificación y sampling especulativo para MTP (Multi-Token Prediction) en SuperMLX.
OBJETIVO: Implementar el algoritmo exacto de Rejection Sampling (Leviathan & Chen) con corrección residual para MTP depth=1.
ENTRADAS:
- model: Modelo con soporte MTP inyectado (validate_qwen3_5_mtp_support == True).
- prompt_tokens: mx.array de tokens iniciales.
- max_tokens: int, presupuesto de generación.
- temperature, top_p, min_p: parámetros de sampling.
SALIDAS:
- Generator con tokens emitidos, estadísticas de aceptación (alpha), tokens/s y conteos.
REGLAS INVIOLABLES:
- Prohibido modificar la distribución de probabilidad del modelo base (rejection sampling matemáticamente exacto).
- Prohibido print(). Usar exclusivamente logger con prefijos estándar ([INICIO], [CONFIG], [CALC], [DECISION], [RESULT], [ERROR]).
- Obligatorio tipado estricto en funciones públicas.
- Obligatorio recolectar telemetría de tasa de aceptación (alpha = accepted_drafts / total_drafts).
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
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import mlx.core as mx
from supermlx.config import SETTINGS

logger = logging.getLogger(__name__)


def _distribution_from_logits(
    logits: mx.array,
    temperature: float = 0.0,
    top_p: float = 1.0,
    min_p: float = 0.0,
    top_k: int = 0,
) -> mx.array:
    """Final distribution for a single vocabulary vector; greedy is one-hot."""
    if logits.ndim != 1 or logits.size == 0:
        raise ValueError("Expected a nonempty 1D logits vector")
    if not 0.0 < top_p <= 1.0 or not 0.0 <= min_p <= 1.0 or top_k < 0:
        raise ValueError("Invalid top_p, min_p or top_k")
    logits = logits.astype(mx.float32)
    if temperature <= 0.0:
        return (mx.arange(logits.shape[0]) == mx.argmax(logits)).astype(mx.float32)
    scaled = logits / temperature
    if top_k > 0:
        k = min(top_k, scaled.shape[0])
        threshold = mx.min(mx.topk(scaled, k=k))
        scaled = mx.where(scaled < threshold, -float("inf"), scaled)
    if min_p > 0.0:
        # Equivalent to softmax(x) >= min_p * max(softmax(x)).
        threshold = mx.max(scaled) + mx.log(mx.array(min_p, dtype=mx.float32))
        scaled = mx.where(scaled < threshold, -float("inf"), scaled)
    probs = mx.softmax(scaled)
    if top_p < 1.0:
        order = mx.argsort(-probs)
        ordered = probs[order]
        cumulative = mx.cumsum(ordered)
        remove = mx.concatenate([mx.array([False]), cumulative[:-1] > top_p])
        ordered = mx.where(remove, 0.0, ordered)
        probs = ordered[mx.argsort(order)]
        probs = probs / mx.sum(probs)
    return probs


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
    Restaura los cachés del trunk y del MTP al estado previo al draft rechazado.
    - Capas SSM (ArraysCache): restaura (conv_snap, ssm_snap) de rollback_state.
    - Capas de atención (KVCache): recorta 1 token con trim(1).
    - Cabezal MTP (KVCache): recorta 1 token con trim(1) para mantener sincronía posicional.
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

    if mtp_cache is not None:
        for c in mtp_cache:
            if hasattr(c, "trim") and getattr(c, "offset", 0) > 0:
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
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.min_p = min_p
        self.top_k = top_k
        self.sampler = sampler
        self.adaptive_temperature = adaptive_temperature
        self.tokenizer = tokenizer
        self._last_nudge_token: int = 0
        logger.info(
            "[CONFIG] MTPSpeculativeEngine inicializado: temp=%.2f, top_p=%.2f, min_p=%.2f, top_k=%d, adaptive_temp=%s",
            self.temperature,
            self.top_p,
            self.min_p,
            self.top_k,
            self.adaptive_temperature,
        )

    def _get_sampling_params(self) -> Tuple[float, float, float, int]:
        """
        Resuelve (temp, top_p, min_p, top_k) dinámicamente según la fase del sampler
        (DualPhaseSampler / FEATURE_ADAPTIVE_TEMPERATURE) o el modo estático.
        """
        if not self.adaptive_temperature or self.sampler is None:
            return self.temperature, self.top_p, self.min_p, self.top_k

        # 1. Temperatura activa según la fase (thinking vs response)
        temp = self.temperature
        if hasattr(self.sampler, "tracker") and getattr(self.sampler.tracker, "is_thinking", False):
            temp = getattr(self.sampler, "think_temp", self.temperature)
        elif hasattr(self.sampler, "resp_temp"):
            temp = getattr(self.sampler, "resp_temp", self.temperature)
        elif hasattr(self.sampler, "temp"):
            temp = getattr(self.sampler, "temp", self.temperature)

        # 2. Filtros de sampling del sampler activo
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

        # 2. Prefill inicial del prompt
        if prompt_tokens.ndim == 1:
            prompt_tokens = prompt_tokens[None]

        logits, pre_norm = self.model.forward_with_hidden(prompt_tokens, cache=cache)
        mx.eval(logits, pre_norm)

        curr_temp, curr_top_p, curr_min_p, curr_top_k = self._get_sampling_params()
        token_t1, _ = _sample_token(
            logits[0, -1],
            temperature=curr_temp,
            top_p=curr_top_p,
            min_p=curr_min_p,
            top_k=curr_top_k,
        )

        tokens_generated = 1
        drafts_attempted = 0
        drafts_accepted = 0

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
            # ── IN-SITU EPISTEMIC NUDGE INJECTION ────────────────────────
            if (
                SETTINGS.feature_epistemic_nudge
                and self.tokenizer is not None
                and hasattr(self.sampler, "tracker")
                and getattr(self.sampler.tracker, "is_thinking", False)
            ):
                _thinking_count = getattr(self.sampler.tracker, "thinking_count", tokens_generated)
                if (
                    _thinking_count >= SETTINGS.epistemic_nudge_min_tokens
                    and (_thinking_count - self._last_nudge_token) >= SETTINGS.epistemic_nudge_interval
                ):
                    nudge_text = SETTINGS.epistemic_nudge_text
                    nudge_tokens = self.tokenizer.encode(nudge_text)
                    if nudge_tokens:
                        nudge_arr = mx.array([nudge_tokens], dtype=mx.int32)
                        nudge_logits, nudge_hidden = self.model.forward_with_hidden(
                            nudge_arr, cache=cache
                        )
                        mx.eval(nudge_logits, nudge_hidden)

                        for n_tok in nudge_tokens:
                            tokens_generated += 1
                            yield {
                                "token": int(n_tok),
                                "is_speculative": False,
                                "accepted": True,
                                "tokens_generated": tokens_generated,
                                "drafts_accepted": drafts_accepted,
                                "drafts_attempted": drafts_attempted,
                                "alpha": curr_alpha if 'curr_alpha' in locals() else 0.0,
                            }

                        confirmed_token = int(nudge_tokens[-1])
                        hidden_at_confirmed = nudge_hidden[:, -1:, :]
                        self._last_nudge_token = _thinking_count

                        # Acoplar Thermal Spark durante la pausa reflexiva si está configurado
                        if (
                            hasattr(self.sampler, "spark_controller")
                            and self.sampler.spark_controller is not None
                            and self.sampler.spark_controller.enabled
                        ):
                            self.sampler.spark_controller.spark_remaining_tokens = SETTINGS.spark_pulse_duration
                            self.sampler.spark_controller.last_spark_token = _thinking_count
                            self.sampler.spark_controller.last_cause = "epistemic_nudge"

                        # Generar nuevo draft especulativo a partir del último token del nudge
                        mtp_logits = self.model.mtp_forward(
                            hidden_at_confirmed,
                            mx.array([[confirmed_token]]),
                            mtp_cache=mtp_cache,
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
                        logger.info(
                            "[SPARK] ⚡ Epistemic Nudge inyectado en token %d (%d tokens)",
                            _thinking_count,
                            len(nudge_tokens),
                        )
                        if tokens_generated >= max_tokens:
                            break

            # 4. Fase VERIFY en batch de 2 tokens: [confirmed_token, draft_tok]
            # n_confirmed=1 le indica a GatedDeltaNet que guarde snapshot SSM entre ellos.
            y_with_draft = mx.array([[confirmed_token, draft_tok]])
            target_logits, hidden = self.model.forward_with_hidden(
                y_with_draft,
                cache=cache,
                n_confirmed=1,
            )
            mx.eval(target_logits, hidden)

            verify_logits = target_logits[0, 0]
            bonus_logits = target_logits[0, 1]

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

                # Emitir draft token confirmado
                tokens_generated += 1
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
                # Muestrear y emitir bonus token proyectado por el trunk en posición 1
                bonus_tok, _ = _sample_token(
                    bonus_logits,
                    temperature=curr_temp,
                    top_p=curr_top_p,
                    min_p=curr_min_p,
                    top_k=curr_top_k,
                )
                tokens_generated += 1
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

                # Siguiente ciclo: el estado oculto confirmado para el MTP es la posición 1 (draft_tok)
                hidden_at_draft = hidden[:, 1:2, :]
                mtp_logits = self.model.mtp_forward(
                    hidden_at_draft,
                    mx.array([[bonus_tok]]),
                    mtp_cache=mtp_cache,
                )
                mx.eval(mtp_logits)
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
                # RECHAZO: Rollback simétrico en trunk y en cabezal MTP
                _rollback_draft_caches(cache, mtp_cache=mtp_cache)
                curr_alpha = (drafts_accepted / max(drafts_attempted, 1)) * 100.0

                # Emitir el token corregido
                tokens_generated += 1
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
                mx.eval(mtp_logits)
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

    if not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)

    if not isinstance(prompt, mx.array):
        if isinstance(prompt, str):
            add_special_tokens = tokenizer.bos_token is None or not prompt.startswith(
                tokenizer.bos_token
            )
            prompt = tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
        prompt = mx.array(prompt)

    # Extraer parámetros de sampling base del sampler si están presentes
    temp = getattr(sampler, "temp", getattr(sampler, "resp_temp", 0.0)) if sampler is not None else 0.0
    min_p = getattr(sampler, "min_p", 0.0) if sampler is not None else 0.0
    top_p = getattr(sampler, "top_p", 1.0) if sampler is not None else 1.0
    top_k = getattr(sampler, "top_k", 0) if sampler is not None else 0

    engine = MTPSpeculativeEngine(
        model=model,
        temperature=temp,
        top_p=top_p,
        min_p=min_p,
        top_k=top_k,
        sampler=sampler,
        adaptive_temperature=adaptive_temperature,
        tokenizer=tokenizer,
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