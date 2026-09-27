"""
Pipeline de 4 fases para _handle_chat_completion, single-user.

No hay Scheduler como clase separada: en SGLang el scheduler existe para
decidir QUÉ request de la cola correr y cómo batchearlo con otros. Acá,
con un solo request a la vez, ese rol se reduce a "cache_lookup decide
cuánto hay que prefillear" — no hace falta el objeto.

Cada función recibe (ctx: RequestContext, state: ServerState) y muta ctx
in-place. server2.py llama a las 4 en secuencia; los fast-paths (slug-gen,
NL title) siguen resueltos ANTES de crear el RequestContext, no pasan
por acá.
"""
from __future__ import annotations

from .request_context import RequestContext
from .radix_cache import RadixPromptCache


def preprocess(ctx: RequestContext, state) -> None:
    """
    Parseo + canonicalización + RAG + healing store.
    Fuente: server2.py L~2400-2900 (a confirmar rango exacto al extraer).

    TODO: mover _update_healing_store() acá, seteando ctx.healed.
    TODO: mover canonicalización de mensajes, seteando ctx.canonical_messages.
    """
    raise NotImplementedError


def cache_lookup(ctx: RequestContext, state, radix: RadixPromptCache) -> None:
    """
    Reemplaza la resolución actual contra cache_lru.

    matched, kv = radix.match_prefix(ctx.prompt_tokens)
    ctx.rest_count = len(ctx.prompt_tokens) - len(matched)
    ctx.prompt_cache = kv
    ctx.cache_hit_ratio = len(matched) / max(len(ctx.prompt_tokens), 1)

    DECISIÓN (ya tomada, no TODO): stable_prefix.py sigue SEPARADO.
    Responde una pregunta distinta a la del radix tree — dónde cambió la
    conversación entre turns (session boundary, para healing), no dónde
    matchea el KV cache. Fusionarlos acopla session semantics con cache
    lookup, que es justo lo que complicaba a server2. Esta fase llama a
    ambos, radix.match_prefix() y el boundary check de stable_prefix,
    como dos pasos independientes.

    TODO — DECISIÓN PENDIENTE antes de Commit 4: TPC (tool_prefix_cache,
    ctx no lo modela todavía) cachea tokens de tool calls por un camino
    separado del radix tree. Si ambos cachean la misma región sin
    coordinarse, hay riesgo de doble invalidación (uno cree que el
    segmento es válido, el otro ya lo evictó). Definir: ¿TPC se funde
    dentro del radix tree como otro tipo de nodo, o queda aparte con un
    orden de consulta explícito (radix primero, TPC como fallback)?
    """
    raise NotImplementedError


def generate(ctx: RequestContext, state) -> None:
    """
    Prefill (adaptive_prefill.py) + decode (_stream_generate_unified).
    Setea ctx.checkpoint antes de generar (_capture_hybrid_checkpoint),
    ctx.generated_tokens y ctx.finish_reason al terminar.

    Riesgo alto — esta es la fase que toca la zona de contaminación KV
    histórica. No tocar sin smoke test streaming + sesión concurrente.

    CONTRATO DE CHECKPOINT (verificado contra cache_engine.py, no supuesto):
    ctx.checkpoint se captura ACÁ, antes de generar (equivalente a
    capture_hybrid_generation_checkpoint). Hay DOS caminos distintos en
    postprocess(), no un restore-then-insert único:

    1. Cache SIN capas recurrentes (pure KV): postprocess() inserta
       (cache_key, generated_tokens) completo — el cache SÍ crece con
       la respuesta, vía trim_prompt_cache estándar.
    2. Cache CON capas recurrentes (tu caso — GatedDeltaNet+attention):
       postprocess() debe restaurar el checkpoint (recurrente Y KV en
       lockstep — restore_hybrid_generation_checkpoint hace trim() de
       las capas KV también, no solo rollback de las recurrentes) e
       insertar SOLO (cache_key[:checkpoint_len], []) — los tokens
       generados se descartan enteros del cache, nunca se insertan.
       No hay "insertar contaminado" posible porque la key nunca es
       más larga que el estado físico real.

    ctx necesita un campo para distinguir estos dos caminos (ej.
    ctx.cache_has_recurrent_layers: bool) — no está en el skeleton
    original de RequestContext, agregarlo.
    """
    raise NotImplementedError


def postprocess(ctx: RequestContext, state, radix: RadixPromptCache) -> None:
    """
    Insert en radix tree + telemetría + formateo de respuesta.
    Equivalente a post_generation.py (A8) + cache_engine.prepare_cache_for_insertion.

    DOS caminos según ctx.cache_has_recurrent_layers:

    1. False (pure-KV): radix.insert(cache_key + generated_tokens, kv_cache)
       — el cache crece con la respuesta, vía trim_prompt_cache estándar.

    2. True (híbrido, GatedDeltaNet+attention): primero restaurar checkpoint
       (restore_hybrid_generation_checkpoint hace trim de capas KV + rollback
       de recurrentes en lockstep), después radix.insert(cache_key[:checkpoint_len], kv_cache)
       — los generated_tokens se descartan enteros, nunca se insertan.
    """
    raise NotImplementedError
