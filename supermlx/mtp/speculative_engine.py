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
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Generator, List, Optional, Tuple

import mlx.core as mx

logger = logging.getLogger(__name__)


def _sample_token(
    logits: mx.array,
    temperature: float = 0.0,
    top_p: float = 1.0,
    min_p: float = 0.0,
    top_k: int = 0,
) -> Tuple[int, mx.array]:
    """
    Muestrea un token a partir de logits y devuelve (token_id, probs).
    Si temperature <= 0.0, ejecuta argmax determinístico (greedy).
    Garantiza que probs devuelto sea la distribución final normalizada post-filtros
    (temperatura, min_p, top_k, top_p) requerida para paridad en Rejection Sampling.
    """
    if temperature <= 0.0:
        token_id = int(mx.argmax(logits, axis=-1).item())
        probs = mx.softmax(logits, axis=-1)
        return token_id, probs

    # 1. Escalamiento por temperatura
    scaled_logits = logits / temperature

    # 2. Top-K filtering si aplica
    if top_k > 0:
        kth_val = mx.topk(scaled_logits, k=top_k)[..., -1:]
        scaled_logits = mx.where(scaled_logits < kth_val, -1e9, scaled_logits)

    # 3. Min-P filtering si aplica
    if min_p > 0.0:
        probs_raw = mx.softmax(scaled_logits, axis=-1)
        max_prob = mx.max(probs_raw, axis=-1, keepdims=True)
        threshold = max_prob * min_p
        scaled_logits = mx.where(probs_raw < threshold, -1e9, scaled_logits)

    probs = mx.softmax(scaled_logits, axis=-1)

    # 4. Top-P / Nucleus sampling si aplica
    if top_p < 1.0:
        sorted_indices = mx.argsort(-probs, axis=-1)
        sorted_probs = probs[sorted_indices]
        cumulative_probs = mx.cumsum(sorted_probs, axis=-1)
        cutoff_mask = cumulative_probs > top_p
        cutoff_mask = mx.concatenate([mx.array([False]), cutoff_mask[:-1]])
        probs = mx.where(cutoff_mask, 0.0, probs)
        probs = probs / mx.sum(probs)

    token_id = int(mx.random.categorical(mx.log(probs + 1e-12)).item())
    return token_id, probs


def _verify_draft_token(
    draft_token: int,
    p_probs: mx.array,
    q_probs: mx.array,
    temperature: float,
) -> Tuple[bool, int]:
    """
    Aplica el teorema de Rejection Sampling de Leviathan & Chen:
    - Probabilidad de aceptación: alpha = min(1, p(x) / q(x))
    - Si se rechaza: muestrea de la distribución residual (p - q)+ / sum(p - q)+
    """
    if temperature <= 0.0:
        # En modo greedy, la verificación es una comparación de igualdad exacta
        target_token = int(mx.argmax(p_probs, axis=-1).item())
        is_accepted = (draft_token == target_token)
        committed_token = draft_token if is_accepted else target_token
        return is_accepted, committed_token

    p_val = float(p_probs[draft_token].item())
    q_val = float(q_probs[draft_token].item())
    ratio = p_val / max(q_val, 1e-12)

    # Moneda de aceptación
    u = float(mx.random.uniform().item())
    if u < ratio:
        return True, draft_token

    # Si se rechaza, muestreo de la distribución residual
    residual = mx.maximum(0.0, p_probs - q_probs)
    residual_sum = float(mx.sum(residual).item())
    if residual_sum > 1e-9:
        residual_probs = residual / residual_sum
        resampled_token = int(mx.random.categorical(mx.log(residual_probs + 1e-12)).item())
    else:
        resampled_token = int(mx.random.categorical(mx.log(p_probs + 1e-12)).item())

    return False, resampled_token


def _rollback_draft_caches(
    model_cache: List[Any],
    mtp_cache: Optional[List[Any]] = None,
) -> None:
    """
    Restaura los cachés del trunk y del MTP al estado previo al draft rechazado.
    - Capas SSM (ArraysCache): restaura (conv_snap, ssm_snap) de rollback_state.
    - Capas de atención (KVCache): recorta 1 token con trim(1).
    - Cabezal MTP (KVCache): recorta 1 token con trim(1) para no desfasar posiciones RoPE.
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
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.min_p = min_p
        self.top_k = top_k
        self.sampler = sampler
        self.adaptive_temperature = adaptive_temperature
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
        start_time = time.time()

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
        token_t1, p1_probs = _sample_token(
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
            verify_pred, p_probs = _sample_token(
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

        elapsed = time.time() - start_time
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
    **kwargs,
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
    )

    detokenizer = tokenizer.detokenizer
    detokenizer.reset()

    eos_ids = getattr(tokenizer, "eos_token_ids", set())
    primary_eos = getattr(tokenizer, "eos_token_id", None)

    tic = time.perf_counter()
    tokens_yielded = 0

    for step in engine.generate(
        prompt_tokens=prompt,
        max_tokens=max_tokens,
        eos_token_id=primary_eos,
        prompt_cache=prompt_cache,
        mtp_cache=mtp_cache,
    ):
        token_id = int(step["token"])
        detokenizer.add_token(token_id)
        tokens_yielded += 1

        is_eos = (token_id in eos_ids) or (token_id == primary_eos)
        finish_reason = "stop" if is_eos else ("length" if tokens_yielded >= max_tokens else None)

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
            finish_reason=finish_reason,
        )
        response.drafts_accepted = step.get("drafts_accepted", 0)
        response.drafts_attempted = step.get("drafts_attempted", 0)
        response.alpha = step.get("alpha", 0.0)
        response.draft_accepted = step.get("accepted", True)

        yield response

        if finish_reason is not None:
            break

    detokenizer.finalize()
