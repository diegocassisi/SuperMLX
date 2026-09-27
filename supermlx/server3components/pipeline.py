"""
[AI_DIRECTIVE]
ROL: Pipeline de 4 fases para _handle_chat_completion en server3 (Fase D).
OBJETIVO: Descomponer el procesamiento de chat completions en 4 fases secuenciales
          puras/mutadoras sobre RequestContext y ServerState:
          1. preprocess: parseo, canonicalización dual, healing, compresión y RAG.
          2. cache_lookup: consulta al RadixPromptCache y resolución de prefijo.
          3. generate: prefill adaptativo y decode con captura de checkpoint pre-decode.
          4. postprocess: actualización del RadixPromptCache y formateo de respuesta.
ENTRADAS: ctx (RequestContext), state (ServerState), radix (RadixPromptCache).
SALIDAS: ctx mutado in-place con estado final de respuesta y telemetría.
REGLAS INVIOLABLES:
- Single-user local: sin scheduler de colas ni continuous batching.
- No imports circulares: dependencias de modelo/tokenizer pasan vía state.
- Coexistencia paso a paso según el plan de refactorización.
SSoT: pipeline.py es la única definición de las 4 fases de ejecución para server3.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from .request_context import RequestContext
from .radix_cache import RadixPromptCache
from .state import ServerState

# SuperMLX pure helpers
from ..message_pipeline import (
    _canonicalize_messages,
    _extract_session_context,
    _heal_messages,
    _break_tool_call_loop,
    _hoist_system_messages,
    _prepare_messages_for_template,
    _scrub_cache_key,
    _assert_cache_key_safety,
    _count_roles,
    _summarize_tool_results,
    _estimate_token_count,
    _is_hermes_housekeeping_request,
    _is_rag_bypass_request,
)
from ..tool_parsing import _extract_enable_thinking
from .compress_cache import compress_with_cache as _compress_with_cache
from ..housekeeping_staging import find_housekeeping_split_index

logger = logging.getLogger(__name__)


def _tokenize_text(tokenizer: Any, text: str) -> list[int]:
    """Tokeniza un string utilizando el tokenizer del modelo de forma consistente."""
    if not isinstance(text, str):
        return list(text) if isinstance(text, list) else []
    if tokenizer is None:
        return []
    bos = getattr(tokenizer, "bos_token", None)
    add_special = bos is None or not text.startswith(bos)
    try:
        return tokenizer.encode(text, add_special_tokens=add_special)
    except TypeError:
        return tokenizer.encode(text)


def preprocess(ctx: RequestContext, state: ServerState) -> None:
    """
    Fase 1: Parseo + canonicalización dual + RAG + healing + compresión.
    Muta ctx in-place seteando canonical_messages, healed, prompt_tokens,
    model_tokens, y flags operativos (is_compact, enable_thinking, etc.).
    """
    body = ctx.body
    request_id = ctx.request_id
    raw_messages_inbound = body.get("messages", [])
    tools = body.get("tools")
    ctx.tool_calls = tools or []

    # Detectar flags de housekeeping, compact y ephemeral
    is_housekeeping = _is_hermes_housekeeping_request(raw_messages_inbound)
    is_compact = bool(body.get("_supermlx_compact", False))
    is_ephemeral = bool(body.get("_supermlx_ephemeral", False))

    if is_ephemeral or is_compact:
        is_housekeeping = True

    ctx.is_housekeeping = is_housekeeping
    ctx.is_compact = is_compact
    ctx.is_ephemeral = is_ephemeral

    # Reasoning control (thinking tokens)
    default_thinking = getattr(state.settings, "default_thinking", True)
    reasoning_control = _extract_enable_thinking(body, default_thinking=default_thinking)
    enable_thinking = False if (is_compact or is_ephemeral) else reasoning_control["enable_thinking"]
    ctx.enable_thinking = enable_thinking

    # ── RUTA VLM (Vision-Language Model) ────────────────────────────────────────
    if state.is_vlm:
        raw_messages = list(raw_messages_inbound)
        # Stateless healing en VLM
        healing_lock = getattr(state, "healing_store_lock", None) or threading.Lock()
        healed_messages = _heal_messages(
            raw_messages,
            state.healing_store,
            healing_lock,
        )
        ctx.healed = (healed_messages != raw_messages)

        from .vlm_pipeline import (
            vlm_prepare_messages as _prep_vlm,
            vlm_extract_images as _extract_vlm_imgs,
            vlm_prompt_and_inputs as _vlm_inputs,
            vlm_sync_before_generation as _vlm_sync,
        )

        messages = _prep_vlm(healed_messages, tools=tools)
        images = _extract_vlm_imgs(body.get("messages", []))

        with state.model_lock:
            vlm_input_ids_raw, vlm_pixel_values, vlm_mask, vlm_kwargs = _vlm_inputs(
                state.processor,
                getattr(state, "vlm_config", {}) or {},
                messages,
                images,
                tools=tools,
                enable_thinking=enable_thinking,
            )
            _vlm_sync(vlm_pixel_values, vlm_mask)

        ctx.vlm_pixel_values = vlm_pixel_values
        ctx.vlm_mask = vlm_mask
        ctx.vlm_kwargs = vlm_kwargs

        if vlm_input_ids_raw is not None:
            if hasattr(vlm_input_ids_raw, "flatten"):
                model_tokens = vlm_input_ids_raw.flatten().tolist()
            else:
                model_tokens = list(vlm_input_ids_raw)
        else:
            model_tokens = []

        ctx.model_tokens = model_tokens
        prompt_tokens = model_tokens

        # Canonical key para VLM
        if getattr(state.settings, "cache_canonicalize_tool_context", True):
            try:
                _, canonical_msgs_vlm = _canonicalize_messages(
                    healed_messages,
                    state.settings.cache_canonicalize_tool_context,
                )
                ctx.canonical_messages = canonical_msgs_vlm
                canon_prepared = _prep_vlm(canonical_msgs_vlm, tools=tools)
                vlm_tok = getattr(state.processor, "tokenizer", state.processor)
                if vlm_tok is not None and hasattr(vlm_tok, "apply_chat_template"):
                    canon_fmt = vlm_tok.apply_chat_template(
                        canon_prepared,
                        tokenize=False,
                        add_generation_prompt=True,
                        tools=tools,
                        enable_thinking=enable_thinking,
                    )
                    canon_fmt = _scrub_cache_key(str(canon_fmt), state.settings.cache_canonicalize_tool_context)
                    canon_ids = vlm_tok.encode(canon_fmt)
                    if isinstance(canon_ids, list) and canon_ids:
                        prompt_tokens = canon_ids
            except Exception as e:
                logger.debug("[PREPROCESS VLM] Error en canonicalización VLM: %s", e)

        ctx.prompt_tokens = prompt_tokens
        ctx.raw_messages = raw_messages
        if ctx.canonical_messages is None:
            ctx.canonical_messages = healed_messages

    # ── RUTA LM ESTÁNDAR ────────────────────────────────────────────────────────
    else:
        raw_messages = list(raw_messages_inbound)

        # 1. Compresión opcional de prompts
        feat_compressor = getattr(state.settings, "feature_compressor", False)
        comp_thresh = getattr(state.settings, "feature_compression_threshold", 4000)
        compressor_mod = getattr(state, "compressor_module", None)
        if feat_compressor and compressor_mod is not None and not ctx.is_anthropic:
            try:
                comp_session = ctx.body.get("session_id") or request_id
                raw_messages, comp_ms, _ = _compress_with_cache(
                    raw_messages=raw_messages,
                    comp_session=comp_session,
                    compressor_module=compressor_mod,
                    threshold=comp_thresh,
                    request_id=request_id,
                )
                ctx.pipeline_timings["compress"] = comp_ms
            except Exception as ce:
                logger.debug("[PREPROCESS] Compresión falló: %s", ce)

        # 2. RAG Enrichment
        rag_mod = getattr(state, "rag_module", None)
        feat_rag = getattr(state.settings, "feature_rag_enrichment", False)
        skip_rag = _is_rag_bypass_request(raw_messages) or ctx.is_anthropic
        if feat_rag and rag_mod is not None and not skip_rag:
            t0_rag = time.time()
            try:
                if hasattr(rag_mod, "enrich_messages_with_metadata"):
                    raw_messages, _ = rag_mod.enrich_messages_with_metadata(raw_messages)
                else:
                    raw_messages = rag_mod.enrich_messages(raw_messages)
                ctx.pipeline_timings["rag"] = (time.time() - t0_rag) * 1000
            except Exception as re:
                logger.debug("[PREPROCESS] RAG enrichment falló: %s", re)

        # 3. Stateless message healing
        feat_healing = getattr(state.settings, "feature_healing", True)
        t0_heal = time.time()
        if feat_healing and state.healing_store is not None:
            healing_lock = getattr(state, "healing_store_lock", None) or threading.Lock()
            healed_messages = _heal_messages(
                raw_messages,
                state.healing_store,
                healing_lock,
            )
        else:
            healed_messages = list(raw_messages)
        ctx.pipeline_timings["heal"] = (time.time() - t0_heal) * 1000
        ctx.healed = (healed_messages != raw_messages)

        # 4. Tool call loop breaker
        feat_loop_breaker = getattr(state.settings, "feature_tool_loop_breaker", True)
        loop_max_retries = getattr(state.settings, "tool_loop_max_retries", 3)
        healed_messages, _ = _break_tool_call_loop(
            healed_messages,
            request_id,
            enabled=feat_loop_breaker,
            max_retries=loop_max_retries,
        )

        # 5. Dual pipeline: canonicalización para cache key vs mensaje crudo para modelo
        original_messages, canonical_messages = _canonicalize_messages(healed_messages)
        original_messages = _hoist_system_messages(original_messages)
        canonical_messages = _hoist_system_messages(canonical_messages)

        norm_write = getattr(state.settings, "normalize_write_tool_content_for_prompt", True)
        messages = _prepare_messages_for_template(original_messages, norm_write)
        cache_messages = _prepare_messages_for_template(canonical_messages, norm_write)

        preserve_thinking = getattr(state.settings, "feature_preserve_thinking", True)
        tok = state.tokenizer
        if hasattr(tok, "apply_chat_template") and getattr(tok, "chat_template", None):
            prompt = tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                tools=tools,
                enable_thinking=enable_thinking,
                preserve_thinking=preserve_thinking,
            )
            cache_prompt_raw = tok.apply_chat_template(
                cache_messages,
                tokenize=False,
                add_generation_prompt=True,
                tools=tools,
                enable_thinking=enable_thinking,
                preserve_thinking=preserve_thinking,
            )
        else:
            prompt = messages[-1]["content"] if messages else ""
            cache_prompt_raw = cache_messages[-1]["content"] if cache_messages else ""

        # Post-render scrub de campos volátiles
        canon_ctx = getattr(state.settings, "cache_canonicalize_tool_context", True)
        cache_prompt = _scrub_cache_key(cache_prompt_raw, canon_ctx)

        # Safety check de la clave normalizada
        if getattr(state.settings, "cache_norm_safety_check", True):
            if not _assert_cache_key_safety(prompt, cache_prompt, context="lm_path"):
                cache_prompt = prompt

        prompt_tokens = _tokenize_text(tok, cache_prompt)
        model_tokens = _tokenize_text(tok, prompt)

        ctx.raw_messages = raw_messages
        ctx.canonical_messages = canonical_messages
        ctx.prompt = prompt
        ctx.cache_prompt = cache_prompt
        ctx.prompt_tokens = prompt_tokens
        ctx.model_tokens = model_tokens

        # 6. Housekeeping split-prefill boundary
        if is_housekeeping and getattr(state.settings, "feature_housekeeping_staging", False) and not is_ephemeral:
            hk_idx = find_housekeeping_split_index(messages)
            if hk_idx > 0 and hasattr(tok, "apply_chat_template") and tok.chat_template:
                try:
                    c_prompt_raw = tok.apply_chat_template(
                        messages[:hk_idx],
                        tokenize=False,
                        add_generation_prompt=False,
                        tools=tools,
                        preserve_thinking=preserve_thinking,
                    )
                    c_cache_raw = tok.apply_chat_template(
                        cache_messages[:hk_idx],
                        tokenize=False,
                        add_generation_prompt=False,
                        tools=tools,
                        preserve_thinking=preserve_thinking,
                    )
                    c_cache_scrubbed = _scrub_cache_key(c_cache_raw, canon_ctx)
                    c_model_toks = _tokenize_text(tok, c_prompt_raw)
                    c_prompt_toks = _tokenize_text(tok, c_cache_scrubbed)
                    if len(c_model_toks) <= len(model_tokens) and model_tokens[: len(c_model_toks)] == c_model_toks:
                        ctx.housekeeping_conv_model_boundary = len(c_model_toks)
                        ctx.housekeeping_conv_prompt_tokens = c_prompt_toks
                except Exception as hk_exc:
                    logger.warning("[PREPROCESS] Falló cálculo de housekeeping boundary: %s", hk_exc)

    # Contexto de sesión
    session_ctx = _extract_session_context(body, ctx.prompt_tokens or [])
    ctx.session_ctx = session_ctx
    if session_ctx is not None and hasattr(session_ctx, "session_id"):
        ctx.session_id = session_ctx.session_id


def cache_lookup(ctx: RequestContext, state, radix: RadixPromptCache) -> None:
    """Fase 2: Reemplaza la resolución actual contra cache_lru."""
    raise NotImplementedError


def generate(ctx: RequestContext, state) -> None:
    """Fase 3: Prefill adaptativo + decode."""
    raise NotImplementedError


def postprocess(ctx: RequestContext, state, radix: RadixPromptCache) -> None:
    """Fase 4: Inserción en radix tree + telemetría + respuesta."""
    raise NotImplementedError
