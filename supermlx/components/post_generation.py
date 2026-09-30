"""
[AI_DIRECTIVE]
ROL: Post-generation cache orchestration
OBJETIVO: Insert generated KV states into PROMPT_CACHE + SessionIndex + disk,
          handle hybrid checkpoint restoration, and log generation telemetry
ENTRADAS: prompt_cache, generated_tokens, cache_key, hybrid_checkpoint
SALIDAS: Mutates PROMPT_CACHE, SessionIndex, warmup disk
REGLAS INVIOLABLES:
- Caller must hold prompt_cache_lock
- Never save contaminated cache (response tokens in prompt KV)
- Hybrid checkpoint must restore recurrent state before insertion
SSoT: This module is the only place where post-generation cache insertion happens
"""
from __future__ import annotations

import copy
import hashlib
import logging
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Module-level state (set via init()) ──────────────────────────────────────
_settings: Any = None
_terminal_status: Optional[Callable] = None
_pipeline_log: Optional[Callable] = None
_tokenize_prompt: Optional[Callable] = None
_thinking_tracker: Any = None
_prompt_cache: Any = None      # LRUPromptCache instance
_session_index: Any = None     # SessionIndex instance
_cache_diag: Any = None
_kv_cache_offset_fn: Optional[Callable] = None
_update_session_turn_store_fn: Optional[Callable] = None
_dpc: Any = None               # DiskPersistenceContext
_wm: Any = None                # WarmupManager
_warmup_save_cache_fn: Optional[Callable] = None
_housekeeping_staging_manager: Any = None
_feature_housekeeping_cache_borrow: bool = False
_feature_log_generation: bool = True
_feature_log_resp: bool = True
_is_vlm: bool = False

# Imports from cache_engine (done at init time to avoid circular)
_prepare_cache_for_insertion: Optional[Callable] = None
_restore_hybrid_checkpoint: Optional[Callable] = None
_rollback_arrays_cache: Optional[Callable] = None
_can_trim: Optional[Callable] = None
_trim_prompt_cache: Optional[Callable] = None
_HybridGenerationCheckpoint: Any = None


def init(
    *,
    settings: Any,
    terminal_status: Callable,
    pipeline_log: Callable,
    tokenize_prompt: Callable,
    thinking_tracker: Any,
    prompt_cache: Any,
    session_index: Any,
    cache_diag: Any,
    kv_cache_offset: Callable,
    update_session_turn_store: Callable,
    dpc: Any,
    wm: Any,
    warmup_save_cache: Optional[Callable] = None,
    housekeeping_staging_manager: Any = None,
    feature_housekeeping_cache_borrow: bool = False,
    feature_log_generation: bool = True,
    feature_log_resp: bool = True,
    is_vlm: bool = False,
    prepare_cache_for_insertion: Callable,
    restore_hybrid_checkpoint: Callable,
    rollback_arrays_cache: Callable,
    can_trim_prompt_cache: Callable,
    trim_prompt_cache: Callable,
    HybridGenerationCheckpoint: Any,
) -> None:
    """Initialize with shared state from server."""
    global _settings, _terminal_status, _pipeline_log, _tokenize_prompt
    global _thinking_tracker, _prompt_cache, _session_index, _cache_diag
    global _kv_cache_offset_fn, _update_session_turn_store_fn
    global _dpc, _wm, _warmup_save_cache_fn, _housekeeping_staging_manager
    global _feature_housekeeping_cache_borrow, _feature_log_generation, _feature_log_resp
    global _is_vlm
    global _prepare_cache_for_insertion, _restore_hybrid_checkpoint, _rollback_arrays_cache
    global _can_trim, _trim_prompt_cache, _HybridGenerationCheckpoint

    _settings = settings
    _terminal_status = terminal_status
    _pipeline_log = pipeline_log
    _tokenize_prompt = tokenize_prompt
    _thinking_tracker = thinking_tracker
    _prompt_cache = prompt_cache
    _session_index = session_index
    _cache_diag = cache_diag
    _kv_cache_offset_fn = kv_cache_offset
    _update_session_turn_store_fn = update_session_turn_store
    _dpc = dpc
    _wm = wm
    _warmup_save_cache_fn = warmup_save_cache
    _housekeeping_staging_manager = housekeeping_staging_manager
    _feature_housekeeping_cache_borrow = feature_housekeeping_cache_borrow
    _feature_log_generation = feature_log_generation
    _feature_log_resp = feature_log_resp
    _is_vlm = is_vlm
    _prepare_cache_for_insertion = prepare_cache_for_insertion
    _restore_hybrid_checkpoint = restore_hybrid_checkpoint
    _rollback_arrays_cache = rollback_arrays_cache
    _can_trim = can_trim_prompt_cache
    _trim_prompt_cache = trim_prompt_cache
    _HybridGenerationCheckpoint = HybridGenerationCheckpoint

def _post_generation_cache_update(
    *,
    request_id: str,
    messages: List[Dict[str, Any]],
    prompt_tokens: List[int],
    cache_key: List[int],
    prompt_cache: Any,
    generated_tokens: List[int],
    tool_calls: Optional[List],
    matched_prefix_len: int,
    session_ctx: Any,
    session_id_for_turn: str,
    hybrid_checkpoint: Optional[_HybridGenerationCheckpoint] = None,
    skip_cache_store: bool = False,
    housekeeping_pre_snapshot: Optional[Dict] = None,
    housekeeping_original_tokens: Optional[tuple] = None,
    model_tokens: Optional[List[int]] = None,
    is_compact: bool = False,
) -> None:
    """
    Shared post-generation logic: insert cache entries (MAIN or COMPACT),
    detect startup warmup candidate, update session turn store.
    Must be called while holding prompt_cache_lock.
    """
    _cache_hit_ratio = matched_prefix_len / max(len(prompt_tokens), 1) if prompt_tokens else 1.0

    if skip_cache_store:
        if (
            _settings.feature_housekeeping_staging
            and not is_compact
            and session_id_for_turn
            and prompt_cache is not None
        ):
            success = _housekeeping_staging_manager.store_or_update(
                session_id=session_id_for_turn,
                model_name=_settings.model_path,
                staged_cache=prompt_cache,
                prompt_tokens=prompt_tokens,
                model_tokens=model_tokens if model_tokens is not None else prompt_tokens,
                hybrid_checkpoint=hybrid_checkpoint,
                base_snapshot=housekeeping_pre_snapshot,
                base_tokens=housekeeping_original_tokens,
            )
            if success:
                _pipeline_log(
                    "CACHE",
                    request_id,
                    f"HOUSEKEEPING_STAGING: staged successfully for session {session_id_for_turn} | "
                    f"hybrid_restored={hybrid_checkpoint is not None} | "
                    f"kv_off={_kv_cache_offset_fn(prompt_cache)}",
                )
            else:
                _pipeline_log(
                    "CACHE",
                    request_id,
                    f"HOUSEKEEPING_STAGING: checkpoint restoration FAILED (fail-closed) | "
                    f"session={session_id_for_turn}",
                )
                if housekeeping_pre_snapshot is not None and housekeeping_original_tokens is not None:
                    _restored = _rollback_arrays_cache(prompt_cache, housekeeping_pre_snapshot)
                    if _restored:
                        _prompt_cache.insert_cache(
                            _settings.model_path,
                            list(housekeeping_original_tokens),
                            prompt_cache,
                        )
                        _pipeline_log(
                            "CACHE",
                            request_id,
                            "HOUSEKEEPING_STAGING: base conversation cache rescued & re-inserted",
                        )
        elif (
            _feature_housekeeping_cache_borrow
            and housekeeping_pre_snapshot is not None
            and housekeeping_original_tokens is not None
            and prompt_cache is not None
        ):
            _restored = _rollback_arrays_cache(prompt_cache, housekeeping_pre_snapshot)
            if _restored:
                _prompt_cache.insert_cache(
                    _settings.model_path,
                    list(housekeeping_original_tokens),
                    prompt_cache,
                )
                _pipeline_log(
                    "CACHE",
                    request_id,
                    f"HOUSEKEEPING_BORROW: cache restored & re-inserted | "
                    f"original_key_len={len(housekeeping_original_tokens)} | "
                    f"kv_off={_kv_cache_offset_fn(prompt_cache)}",
                )
            else:
                _pipeline_log(
                    "CACHE",
                    request_id,
                    "HOUSEKEEPING_BORROW: rollback FAILED — cache entry lost",
                )
        else:
            _pipeline_log(
                "CACHE",
                request_id,
                "HERMES_HOUSEKEEPING: cache store skipped — session turn store updated only",
            )
        if session_id_for_turn:
            _update_session_turn_store_fn(session_id_for_turn, messages, prompt_tokens)
        return

    _diag_pre_restore = _cache_diag.snapshot(prompt_cache, cache_key, "before_hybrid_restore")
    prepared = (
        _prepare_cache_for_insertion(
            prompt_cache,
            cache_key,
            generated_tokens,
            hybrid_checkpoint,
        )
        if prompt_cache is not None
        else None
    )
    if prepared is None:
        # Detailed diagnostics for cache discard
        if prompt_cache is None:
            _discard_reason = "prompt_cache is None"
        elif hybrid_checkpoint is None:
            _discard_reason = (
                f"hybrid_checkpoint is None | "
                f"can_trim={_can_trim(prompt_cache)} | "
                f"gen_tokens={len(generated_tokens)}"
            )
        else:
            # checkpoint existed but restore/validation failed
            _restore_ok = _restore_hybrid_checkpoint(prompt_cache, hybrid_checkpoint)
            _discard_reason = (
                f"restore_failed={not _restore_ok} | "
                f"ckpt_key_len={hybrid_checkpoint.cache_key_len} | "
                f"cache_key_len={len(cache_key)} | "
                f"kv_off={_kv_cache_offset_fn(prompt_cache)}"
            )
        _pipeline_log(
            "CACHE",
            request_id,
            f"post-generation cache discarded: {_discard_reason}",
        )
        if session_id_for_turn:
            _update_session_turn_store_fn(session_id_for_turn, messages, prompt_tokens)
        return

    cache_key, generated_tokens = prepared
    if hybrid_checkpoint is not None:
        _pipeline_log(
            "CACHE",
            request_id,
            "HYBRID_CHECKPOINT restored | "
            f"key={len(cache_key)} | kv_offset={_kv_cache_offset_fn(prompt_cache)} | "
            "all recurrent layers rolled back",
        )
        _cache_diag.compare(_diag_pre_restore, prompt_cache, cache_key, "hybrid_restore", request_id,
            extra={"key_len": len(cache_key), "gen_tokens": len(generated_tokens)})
        # Attach verification metadata to the cache list so it survives
        # the trie store.  On the next lookup, Normal continuation can
        # verify that model_tokens[:_kv_off] hashes identically.
        try:
            prompt_cache.__model_prefix_hash__ = hybrid_checkpoint.model_prefix_hash
            prompt_cache.__model_offset__ = hybrid_checkpoint.model_offset
        except (TypeError, AttributeError):
            pass  # list subclasses or frozen objects — skip silently
        if prompt_cache and hasattr(prompt_cache[0], "__dict__"):
            try:
                prompt_cache[0].__model_prefix_hash__ = hybrid_checkpoint.model_prefix_hash
                prompt_cache[0].__model_offset__ = hybrid_checkpoint.model_offset
            except (TypeError, AttributeError):
                pass


    # MAIN path (always — compact runner removed in FASE-C1)
    # MAIN: any first-turn request with enough tokens qualifies for warmup save.
    # The old _STARTUP_MSG_PATTERN required phrases like "/new" or "session startup"
    # that Claude Code never sends — so warmup_cache.safetensors was never created.
    # Now: messages==2 (first turn) AND no cache on disk yet (or hash invalidated).
    # The 5000-token guard in _insert_cache_entries still filters noise.
    _is_real_startup = (
        len(messages) == 2
        and not _dpc.disk_cache_saved
    )
    _insert_cache_entries(
        model_name=_settings.model_path,
        session_ctx=session_ctx,
        cache_key=cache_key,
        prompt_cache=prompt_cache,
        generated_tokens=generated_tokens,
        tool_calls=tool_calls,
        cache_hit_ratio=_cache_hit_ratio,
        is_warmup_candidate=_is_real_startup,
    )
    # M5: Update session turn record for next-turn stable-prefix lookup
    if session_id_for_turn:
        _update_session_turn_store_fn(session_id_for_turn, messages, prompt_tokens)




def _build_timing_dict(
    first_token_at: Optional[float],
    generation_started_at: Optional[float],
    rest_count: int,
    generated_tokens: List[int],
    mtp_stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the prefill/decode timing dict used by request_logger and telemetry."""
    timing: Dict[str, Any] = {}
    if first_token_at is not None and generation_started_at is not None:
        timing["prefill_seconds"] = first_token_at - generation_started_at
        timing["decode_seconds"] = time.time() - first_token_at
        timing["prefill_tps"] = (
            rest_count / timing["prefill_seconds"]
            if timing["prefill_seconds"] > 0 else None
        )
        timing["decode_tps"] = (
            len(generated_tokens) / timing["decode_seconds"]
            if timing["decode_seconds"] > 0 else None
        )
    if mtp_stats and mtp_stats.get("attempted", 0) > 0:
        timing["mtp_alpha"] = mtp_stats["alpha"]
        timing["mtp_accepted"] = mtp_stats["accepted"]
        timing["mtp_attempted"] = mtp_stats["attempted"]
    return timing




def _log_generation_telemetry(
    request_id: str,
    generation_started_at: float,
    generated_tokens: List[int],
    message_text: str,
    tool_calls: Optional[List],
    enable_thinking: bool,
    finish_reason: str,
    timing: Dict[str, Any],
    thinking_token_count: Optional[int] = None,
) -> None:
    """Shared post-generation pipeline logging (GEN + RESP + MSG_OUT)."""
    if thinking_token_count is not None and thinking_token_count > 0:
        # Use the real count from the generation loop (token-based, accurate)
        _reas = thinking_token_count
        _non_reas = max(0, len(generated_tokens) - _reas)
    else:
        # Fallback: estimate from retokenized message_text (less accurate)
        _non_reas = len(_tokenize_prompt(message_text)) if message_text else 0
        _reas = max(0, len(generated_tokens) - _non_reas)

    if _feature_log_generation:
        _gen_ms = (time.time() - generation_started_at) * 1000
        _mtp_telemetry = ""
        if timing.get("mtp_alpha") is not None:
            _mtp_telemetry = f" | α={timing['mtp_alpha']:.1f}% ({timing['mtp_accepted']}/{timing['mtp_attempted']})"
        _pipeline_log("GEN", request_id,
            f"finished: {len(generated_tokens)} tokens in {_gen_ms/1000:.2f}s "
            f"({timing.get('decode_tps', 0):.1f} tok/s decode){_mtp_telemetry}")
        _pipeline_log("GEN", request_id,
            f"thinking_tokens={_reas} | output_tokens={_non_reas}")
        # DIAGNOSTIC: model skipped thinking and reasoned in visible text
        if enable_thinking and _thinking_tracker.thinking_count <= 1 and _thinking_tracker.responding_count > 10:
            _pipeline_log("THINK", request_id,
                f"EMPTY_THINKING_BLOCK | thinking={_thinking_tracker.thinking_count} "
                f"responding={_thinking_tracker.responding_count} "
                f"(model reasoning in visible text)")
    else:
        pass  # _reas/_non_reas already computed above

    if _feature_log_resp:
        _pipeline_log("RESP", request_id,
            f"raw_text={len(generated_tokens)} tokens | normalize_applied=true")
        if tool_calls:
            _pipeline_log("RESP", request_id,
                f"tool_calls_extracted={len(tool_calls)} "
                f"{[tc.get('function', {}).get('name') for tc in tool_calls]}")
        if enable_thinking:
            _pipeline_log("RESP", request_id,
                f"think_block_stripped=true ({_reas} tokens hidden from client)")
        _pipeline_log("RESP", request_id,
            f"message_to_client={_non_reas} tokens | finish_reason={finish_reason}")
        _agent_tag_out = "MAIN"
        if tool_calls:
            for _tc in tool_calls:
                _tc_name = _tc.get("function", {}).get("name", "?")
                _tc_args = str(_tc.get("function", {}).get("arguments", ""))[:300].replace("\n", " ")
                _pipeline_log("MSG_OUT", request_id,
                    f"[{_agent_tag_out}] tool_call: {_tc_name}({_tc_args})")
        elif message_text:
            _out_preview = message_text[:300].replace("\n", " ")
            _pipeline_log("MSG_OUT", request_id,
                f"[{_agent_tag_out}] text: {_out_preview!r}")




def _insert_cache_entries(
    model_name: str,
    session_ctx: "SessionContext",
    cache_key: List[int],
    prompt_cache: Any,
    generated_tokens: List[int],
    tool_calls: Optional[List[Dict[str, Any]]] = None,
    cache_hit_ratio: float = 1.0,
    prompt_cache_store_override: Optional["LRUPromptCache"] = None,
    is_warmup_candidate: bool = False,
) -> None:
    _store = prompt_cache_store_override if prompt_cache_store_override is not None else _prompt_cache
    if (
        prompt_cache_store_override is None
        and _settings.prompt_cache_max_entries_global <= 1
        and _store._entries
    ):
        _existing_len = max(
            (len(e.tokens) for e in _store._entries.values()), default=0
        )
        _new_len = len(cache_key)
        if _existing_len > _new_len * 1.5:
            # Existing entry is >50% larger — don't evict it.
            return

    # ── CLEAN CACHE BEFORE INSERTION ────────────────────────────────────────
    # After generation, prompt_cache contains KV states for prompt + response.
    # The next request will include the response as TEXT in its prompt, so the
    # response KV states in the cache are stale/wrong. Saving them causes
    # FIX-31 v10 to detect "contamination" and force a full cold start.
    #
    # Strategy depends on available cache slots:
    # - max_entries >= 2: deepcopy + trim → save BOTH clean and full versions
    # - max_entries == 1: trim IN-PLACE → save ONLY the clean version
    #
    # For hybrid caches (Qwen3 MoE: ArraysCache + KVCache), can_trim_prompt_cache
    # returns False because it requires ALL layers trimmable. In that case, we
    # trim only the trimmable KVCache layers per-layer (matching disk-save logic).
    if generated_tokens and len(cache_key) > len(generated_tokens):
        _diag_pre_insert = _cache_diag.snapshot(prompt_cache, cache_key, "before_insert_trim")
        if _can_trim(prompt_cache):
            # All layers trimmable (pure KVCache or Marconi hybrid)
            if tool_calls:
                # ── TOOL CALL TURN: PRESERVE FULL KV CACHE ───────────────────
                # In agentic workflows (Hermes, Claude Code), a tool call is ALWAYS
                # followed by the client executing the tool and returning the tool_result.
                # Preserving the generated tool call in the KV cache allows the next turn
                # to hit >98% cache instead of discarding 8K-15K tokens and re-prefilling
                # for 35+ seconds.
                _full_kv_off = _kv_cache_offset_fn(prompt_cache)
                if _full_kv_off is not None:
                    _prefix_hash = hash(tuple(cache_key[:_full_kv_off]))
                    try:
                        prompt_cache.__model_offset__ = _full_kv_off
                        prompt_cache.__model_prefix_hash__ = _prefix_hash
                    except (TypeError, AttributeError):
                        pass
                    if prompt_cache and hasattr(prompt_cache[0], "__dict__"):
                        try:
                            prompt_cache[0].__model_offset__ = _full_kv_off
                            prompt_cache[0].__model_prefix_hash__ = _prefix_hash
                        except (TypeError, AttributeError):
                            pass

                if _settings.prompt_cache_max_entries_global >= 2:
                    # Multi-slot: also deepcopy a prompt-only checkpoint into a secondary slot
                    try:
                        prompt_only_cache = copy.deepcopy(prompt_cache)
                        _trim_prompt_cache(prompt_only_cache, len(generated_tokens))
                        prompt_only_key = cache_key[: -len(generated_tokens)]
                        _store.insert_cache(model_name, prompt_only_key, prompt_only_cache)
                        _session_index.register_cache_key(session_ctx, prompt_only_key)
                    except Exception:
                        pass

                _terminal_status(
                    "💾",
                    f"Cache preserved for tool turn: {_full_kv_off} tokens "
                    f"(retained {len(generated_tokens)} response/tool tokens | hash verified)",
                )
            elif _settings.prompt_cache_max_entries_global >= 2 and _is_vlm:
                # Multi-slot VLM: deepcopy for a prompt-only checkpoint
                try:
                    prompt_only_cache = copy.deepcopy(prompt_cache)
                    _trim_prompt_cache(prompt_only_cache, len(generated_tokens))
                    prompt_only_key = cache_key[: -len(generated_tokens)]
                    _store.insert_cache(model_name, prompt_only_key, prompt_only_cache)
                    _session_index.register_cache_key(session_ctx, prompt_only_key)
                except Exception:
                    pass
            else:
                # Conversational non-tool turn (or single-slot without tools):
                # Trim response tokens to keep prompt-only baseline clean.
                try:
                    _pre_off = _kv_cache_offset_fn(prompt_cache)
                    _trim_prompt_cache(prompt_cache, len(generated_tokens))
                    cache_key = cache_key[: -len(generated_tokens)]
                    _post_off = _kv_cache_offset_fn(prompt_cache)
                    _terminal_status("🧹",
                        f"Cache trim in-place (full): {_pre_off} → {_post_off} | "
                        f"stripped {len(generated_tokens)} response tokens")
                except Exception as _trim_err:
                    _terminal_status("⚠️",
                        f"Cache trim in-place FAILED ({_trim_err}) — saving as-is")
        else:
            # A hybrid cache can only be reused after restoring its pre-generation
            # ArraysCache checkpoint.  Partial KV-only trim publishes a false clean
            # key while recurrent layers still contain the response.
            _terminal_status(
                "⚠️",
                "Hybrid cache discarded: generated response reached insertion "
                "without a complete recurrent-state rollback",
            )
            return

    _store.insert_cache(model_name, cache_key, prompt_cache)
    _session_index.register_cache_key(session_ctx, cache_key)
    # FIX-31 DIAG: after insertion — verify cache is clean
    if 'generated_tokens' in dir() and generated_tokens:
        _cache_diag.compare(
            _diag_pre_insert if '_diag_pre_insert' in dir() else None,
            prompt_cache, cache_key, "cache_insert", "",
            extra={"gen_tokens": len(generated_tokens), "key_len": len(cache_key)})

    # ── AUTO-SAVE MAIN CACHE TO DISK ──────────────────────────────────────────
    _active_store = prompt_cache_store_override if prompt_cache_store_override is not None else _prompt_cache

    # ── AUTO-SAVE MAIN CACHE TO DISK ──────────────────────────────────────────
    persist_path = Path(_settings.cache_persist_path) if _settings.cache_persist_path else None

    # ── WARMUP CACHE AUTO-SAVE: DESCONECTADO (REEMPLAZADO POR TPC) ─────────────
    # El sistema utiliza Tool Prefix Cache (TPC) como única fuente de verdad (SSoT)
    # para la persistencia del prefijo (logs/tool_prefix_cache.safetensors).
    # El guardado de warmup_cache.safetensors queda formalmente desconectado.
    _should_save = False

    if _should_save:
        _dpc.disk_cache_saved = True
        # ── FIX: Save PROMPT-ONLY tokens, not prompt+response ────────────
        # cache_key at this point = prompt_tokens + generated_tokens.
        # Saving generated_tokens to disk contaminates the warmup cache:
        # on restart, the KV states encode the previous response, causing
        # hallucinations on the first request. Strip response tokens.
        _prompt_only_key = cache_key[: -len(generated_tokens)] if generated_tokens else list(cache_key)
        _terminal_status("💾", f"Auto-save: capturing prompt-only cache → {persist_path.name} | reason={_save_reason} | tokens={len(_prompt_only_key)} (stripped {len(generated_tokens)} response tok)")
        # Trim prompt_cache to remove generated token KV states.
        # CRITICAL: if deepcopy or trim fails, ABORT the save entirely.
        # A contaminated cache (with response tokens baked in) causes tool
        # hallucinations on restart — far worse than a cold start.
        _save_cache = None
        if generated_tokens and _can_trim(prompt_cache):
            try:
                import copy as _copy_mod
                _save_cache = _copy_mod.deepcopy(prompt_cache)
                _pre_trim_off = _kv_cache_offset_fn(_save_cache)
                _trim_prompt_cache(_save_cache, len(generated_tokens))
                _post_trim_off = _kv_cache_offset_fn(_save_cache)
                # Verify trim actually reduced the offset
                if (_post_trim_off is not None and _pre_trim_off is not None
                        and _post_trim_off >= _pre_trim_off):
                    _terminal_status("⚠️", f"Auto-save ABORTED: trim did not reduce offset ({_pre_trim_off} → {_post_trim_off})")
                    _save_cache = None
            except Exception as _trim_err:
                _terminal_status("⚠️", f"Auto-save ABORTED: deepcopy/trim failed ({_trim_err})")
                _save_cache = None  # NEVER fallback to saving contaminated cache
        elif not generated_tokens:
            # No response tokens to strip — safe to save as-is
            _save_cache = prompt_cache
        else:
            # Hybrid cache (e.g., Qwen3.5: ArraysCache + KVCache):
            # _can_trim() requires ALL layers trimmable, but
            # ArraysCache (linear attention) is never trimmable.
            # Fix: deepcopy + trim only the trimmable layers (KVCache).
            # Non-trimmable layers (ArraysCache) have no .keys →
            # _warmup_save_cache skips them anyway.
            _any_trimmable = any(
                hasattr(layer, 'is_trimmable') and layer.is_trimmable()
                for layer in prompt_cache
            )
            if _any_trimmable:
                try:
                    import copy as _copy_mod
                    _save_cache = _copy_mod.deepcopy(prompt_cache)
                    # Read offset from TRIMMABLE layers only (KVCache),
                    # not ArraysCache whose internal shape[2] is static.
                    _pre_trim_off = None
                    for layer in _save_cache:
                        if hasattr(layer, 'is_trimmable') and layer.is_trimmable() and hasattr(layer, 'offset'):
                            _pre_trim_off = int(layer.offset)
                            break
                    _n_trimmed = 0
                    for layer in _save_cache:
                        if hasattr(layer, 'is_trimmable') and layer.is_trimmable() and hasattr(layer, 'trim'):
                            layer.trim(len(generated_tokens))
                            _n_trimmed += 1
                    _post_trim_off = None
                    for layer in _save_cache:
                        if hasattr(layer, 'is_trimmable') and layer.is_trimmable() and hasattr(layer, 'offset'):
                            _post_trim_off = int(layer.offset)
                            break
                    if (_post_trim_off is not None and _pre_trim_off is not None
                            and _post_trim_off >= _pre_trim_off):
                        _terminal_status("⚠️", f"Auto-save ABORTED: per-layer trim did not reduce offset ({_pre_trim_off} → {_post_trim_off})")
                        _save_cache = None
                    else:
                        _terminal_status("💾", f"Auto-save: per-layer trim OK | trimmed={_n_trimmed}/{len(_save_cache)} layers | offset {_pre_trim_off} → {_post_trim_off}")
                        # FIX-31 v7 + DPC: Update frozen cache with this CLEAN
                        # trimmed copy so pollution restore has full prompt
                        # coverage (not just seed tokens).
                        _dpc.frozen_cache = _save_cache
                        _dpc.frozen_tokens = list(_prompt_only_key)
                        _terminal_status("🧊", f"DPC: frozen cache updated | {len(_save_cache)} layers | {len(_prompt_only_key)} tokens")
                        # FIX-31 DIAG: verify frozen cache is clean after update
                        _cache_diag.compare(None, _save_cache, list(_prompt_only_key), "frozen_update", "",
                            extra={"layers": len(_save_cache), "tokens": len(_prompt_only_key)})
                except Exception as _trim_err:
                    _terminal_status("⚠️", f"Auto-save ABORTED: per-layer trim failed ({_trim_err})")
                    _save_cache = None
            else:
                _terminal_status("⚠️", f"Auto-save SKIPPED: no trimmable layers ({type(prompt_cache[0]).__name__ if prompt_cache else 'empty'})")
                _save_cache = None
        if _save_cache is not None:
            # DPC: compute prefix hash from the prompt-only tokens for auto-healing
            _prefix_hash = _wm.compute_prefix_hash(
                _prompt_only_key,
                model_path=_settings.model_path,
                kv_bits=_settings.kv_bits,
            )
            threading.Thread(
                target=_warmup_save_cache_fn,
                args=(_prompt_only_key, _save_cache, persist_path),
                kwargs={"prefix_hash": _prefix_hash},
                daemon=True,
                name="dpc-cache-save",
            ).start()
        else:
            _dpc.disk_cache_saved = False  # Allow retry on next request





