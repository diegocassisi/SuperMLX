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


def materialize_cache(cache: Any) -> None:
    """Evalúa forzosamente y sincroniza todos los tensores del KV cache en el hilo actual.

    Obligatorio en entornos multi-hilo (ThreadingHTTPServer) para evitar que tensores
    lazy conserven dependencias de Stream(gpu, N) ligadas al hilo de origen.
    """
    if not cache:
        return
    try:
        import mlx.core as mx  # type: ignore[import-untyped]
    except ImportError:
        return

    arrays_to_eval = []
    try:
        for c in cache:
            if hasattr(c, "state") and c.state is not None:
                if isinstance(c.state, mx.array):
                    arrays_to_eval.append(c.state)
                elif isinstance(c.state, (list, tuple)):
                    arrays_to_eval.extend([a for a in c.state if isinstance(a, mx.array)])
            if hasattr(c, "cache") and c.cache is not None:
                if isinstance(c.cache, (list, tuple)):
                    arrays_to_eval.extend([a for a in c.cache if isinstance(a, mx.array)])
            if hasattr(c, "keys") and isinstance(c.keys, mx.array):
                arrays_to_eval.append(c.keys)
            if hasattr(c, "values") and isinstance(c.values, mx.array):
                arrays_to_eval.append(c.values)
        if arrays_to_eval:
            mx.eval(*arrays_to_eval)
        if hasattr(mx, "synchronize"):
            mx.synchronize()
    except Exception as e:
        logger.error("[MATERIALIZE_CACHE] Error evaluando tensores de cache: %s", str(e))
        raise


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
        healed_messages, _loop_broken = _break_tool_call_loop(
            healed_messages,
            request_id,
            enabled=feat_loop_breaker,
            max_retries=loop_max_retries,
        )
        ctx.loop_broken = int(_loop_broken or 0)

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


def cache_lookup(ctx: RequestContext, state: ServerState, radix: RadixPromptCache) -> None:
    """
    Fase 2: Búsqueda en RadixPromptCache y resolución determinística con TPC fallback.
    Muta ctx in-place: ctx.prompt_cache, ctx.rest_count, ctx.cache_hit_ratio.
    """
    tokens = ctx.prompt_tokens or []
    slot = "compact" if ctx.is_compact else "main"

    # 1. Búsqueda primaria en RadixPromptCache
    matched, kv = radix.match_prefix(tokens, slot=slot)

    # Si hay match estructural pero kv_cache es None -> MISS funcional (P4)
    if kv is not None and matched:
        ctx.prompt_cache = kv
        matched_len = len(matched)
        ctx.rest_count = max(0, len(ctx.model_tokens) - matched_len) if ctx.model_tokens else max(0, len(tokens) - matched_len)
        ctx.cache_hit_ratio = matched_len / max(len(tokens), 1)
        return

    # 2. Fallback determinístico a TPC si no hubo hit en Radix y hay tools configuradas (Opción B)
    tpc = getattr(state, "tpc", None)
    if (
        tpc is not None
        and hasattr(tpc, "is_configured")
        and tpc.is_configured()
        and ctx.tool_calls
        and not getattr(state, "is_vlm", False)
    ):
        try:
            sys_body = ""
            for m in ctx.raw_messages:
                if (m.get("role") or "").lower() == "system":
                    sys_body = m.get("content", "")
                    break
            if sys_body:
                pc_clone, ptoks, _ = tpc.get_prefix_cache_clone(
                    system_body=sys_body,
                    tools=ctx.tool_calls,
                    tokenizer=state.tokenizer,
                    model=state.model,
                    max_kv_size=getattr(state.settings, "max_kv_size", 131072),
                    kv_bits=getattr(state.settings, "kv_bits", None),
                    enable_thinking=ctx.enable_thinking,
                    model_path=getattr(state.settings, "model_path", ""),
                    prefill_step_size=getattr(state.settings, "prefill_step_size", 512),
                )
                if pc_clone is not None and ptoks:
                    ptok_len = len(ptoks)
                    target_tokens = ctx.model_tokens or tokens
                    if len(target_tokens) >= ptok_len and target_tokens[:ptok_len] == ptoks:
                        ctx.prompt_cache = pc_clone
                        ctx.rest_count = max(0, len(target_tokens) - ptok_len)
                        ctx.cache_hit_ratio = ptok_len / max(len(tokens), 1)
                        return
        except Exception as tpc_err:
            logger.debug("[CACHE_LOOKUP] TPC fallback failed: %s", tpc_err)

    # 3. Cache Miss total
    ctx.prompt_cache = None
    ctx.rest_count = len(ctx.model_tokens) if ctx.model_tokens else len(tokens)
    ctx.cache_hit_ratio = 0.0


from ..cache_engine import (
    capture_hybrid_generation_checkpoint,
    restore_hybrid_generation_checkpoint,
    _is_arrays_cache,
)
from .adaptive_prefill import _adaptive_prefill


def generate(
    ctx: RequestContext,
    state: ServerState,
    generator_fn: Optional[Callable[[list[int], RequestContext, ServerState], Any]] = None,
) -> None:
    """
    Fase 3: Prefill adaptativo + captura de checkpoint pre-decode + generación de tokens.
    Muta ctx in-place:
    - ctx.cache_has_recurrent_layers: flag crítico para postprocess.
    - ctx.checkpoint: snapshot limpio antes de que cualquier token de decode contamine el estado.
    - ctx.generated_tokens: lista de token IDs producidos.
    - ctx.finish_reason: motivo de terminación ('stop', 'length', etc.).
    """
    model_toks = ctx.model_tokens or ctx.prompt_tokens or []
    # Calcular los tokens restantes a prefillear según rest_count
    if ctx.rest_count > 0 and ctx.rest_count <= len(model_toks):
        rest_tokens = model_toks[-ctx.rest_count :]
    else:
        rest_tokens = list(model_toks)

    # 1. Detección estricta de capas recurrentes (ArraysCache / Marconi pattern)
    has_recurrent = False
    if ctx.prompt_cache and isinstance(ctx.prompt_cache, list):
        has_recurrent = any(_is_arrays_cache(c) for c in ctx.prompt_cache)
    ctx.cache_has_recurrent_layers = has_recurrent

    # 2. Adaptive prefill: procesar rest_tokens[:-1] en chunks dinámicos
    if not getattr(state, "is_vlm", False) and len(rest_tokens) > 1:
        try:
            rest_tokens = _adaptive_prefill(rest_tokens, ctx.prompt_cache, ctx.request_id)
            materialize_cache(ctx.prompt_cache)
        except Exception as ap_err:
            logger.debug("[GENERATE] Adaptive prefill fallback: %s", ap_err)

    ctx.rest_tokens = rest_tokens
    ctx.rest_count = len(rest_tokens)

    # 3. Checkpoint capture ANTES de generar (contrato inviolable contra contaminación KV)
    if has_recurrent and not getattr(state, "is_vlm", False) and ctx.prompt_cache:
        try:
            # Invocar checkpoint() nativo en cada ArraysCache
            for c in ctx.prompt_cache:
                if hasattr(c, "checkpoint") and callable(c.checkpoint):
                    c.checkpoint()

            remaining = len(rest_tokens)
            expected_offset = max(0, len(model_toks) - remaining)
            canonical_key_len = max(0, len(ctx.prompt_tokens or []) - remaining)

            ctx.checkpoint = capture_hybrid_generation_checkpoint(
                ctx.prompt_cache,
                cache_key_len=canonical_key_len,
                model_offset=expected_offset,
                model_prefix_hash=hash(tuple(model_toks[:expected_offset])),
            )
        except Exception as ckpt_err:
            logger.debug("[GENERATE] Falló captura de hybrid checkpoint: %s", ckpt_err)

    # 4. Ejecución del generador si está provisto
    if generator_fn is not None:
        generator_fn(rest_tokens, ctx, state)
        materialize_cache(ctx.prompt_cache)
    else:
        # Fallback para pruebas o cuando el decode es gestionado externamente
        if not ctx.finish_reason:
            ctx.finish_reason = "stop"


def postprocess(
    ctx: RequestContext,
    state: ServerState,
    radix: Optional[RadixPromptCache] = None,
) -> None:
    """
    Fase 4: Post-procesamiento, restauración de checkpoint (si híbrido), inserción en RadixPromptCache y telemetría.

    DOS caminos según ctx.cache_has_recurrent_layers:
    1. Pure-KV (False):
       - Clave a insertar: ctx.cache_key + (ctx.generated_tokens or [])
       - Inserción en RadixPromptCache con todo el contexto generado.
    2. Híbrido (True):
       - Restaurar checkpoint primero vía restore_hybrid_generation_checkpoint(ctx.prompt_cache, ctx.checkpoint)
       - Si la restauración es exitosa:
           checkpoint_key = ctx.cache_key[: ctx.checkpoint.cache_key_len]
           radix.insert(checkpoint_key, ctx.prompt_cache, slot=slot)
           Los generated_tokens se descartan enteros del cache para evitar contaminación.
       - Si falla: descarta la inserción (fail-closed).
    """
    if radix is None and state is not None:
        radix = getattr(state, "radix_cache", None)

    slot = "compact" if getattr(ctx, "is_compact", False) else "main"

    # Preparar uso y métricas
    prompt_len = len(ctx.model_tokens or ctx.prompt_tokens or [])
    gen_len = len(ctx.generated_tokens or [])
    ctx.usage = {
        "prompt_tokens": prompt_len,
        "completion_tokens": gen_len,
        "total_tokens": prompt_len + gen_len,
    }

    # Verificar si se debe omitir la persistencia en cache
    if getattr(ctx, "skip_cache_store", False) or ctx.prompt_cache is None or radix is None:
        logger.debug(
            "[POSTPROCESS] Omitiendo inserción en radix cache (skip=%s, cache_is_none=%s, radix_is_none=%s)",
            getattr(ctx, "skip_cache_store", False),
            ctx.prompt_cache is None,
            radix is None,
        )
        return

    base_key = (
        list(ctx.cache_key)
        if getattr(ctx, "cache_key", None)
        else list(ctx.prompt_tokens or ctx.model_tokens or [])
    )

    lock = getattr(state, "prompt_cache_lock", None)

    def _do_insert():
        if ctx.cache_has_recurrent_layers:
            # ── CAMINO HÍBRIDO ──────────────────────────────────────────────
            if ctx.checkpoint is not None:
                restored = restore_hybrid_generation_checkpoint(ctx.prompt_cache, ctx.checkpoint)
                if restored:
                    ckpt_len = ctx.checkpoint.cache_key_len
                    checkpoint_key = base_key[:ckpt_len]
                    materialize_cache(ctx.prompt_cache)
                    radix.insert(checkpoint_key, ctx.prompt_cache, slot=slot)
                    logger.info(
                        "[POSTPROCESS] [DECISION] Híbrido restaurado e insertado: %d tokens en slot '%s' (generated_tokens descartados)",
                        len(checkpoint_key),
                        slot,
                    )
                else:
                    logger.warning(
                        "[POSTPROCESS] [ERROR] Falló restore_hybrid_generation_checkpoint — cache descartado (fail-closed)"
                    )
            else:
                logger.warning(
                    "[POSTPROCESS] [ERROR] Cache híbrido sin checkpoint disponible — cache descartado (fail-closed)"
                )
        else:
            # ── CAMINO PURE-KV ──────────────────────────────────────────────
            full_key = base_key + (ctx.generated_tokens or [])
            materialize_cache(ctx.prompt_cache)
            radix.insert(full_key, ctx.prompt_cache, slot=slot)
            logger.info(
                "[POSTPROCESS] [DECISION] Pure-KV insertado: %d tokens (base=%d + gen=%d) en slot '%s'",
                len(full_key),
                len(base_key),
                len(ctx.generated_tokens or []),
                slot,
            )

    if lock is not None:
        with lock:
            _do_insert()
    else:
        _do_insert()

