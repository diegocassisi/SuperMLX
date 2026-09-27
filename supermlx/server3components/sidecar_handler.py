"""
[AI_DIRECTIVE]
ROL: Sidecar HTTP handler for lightweight, cache-less inference
OBJETIVO: Servir requests en puerto 8081 con ephemeral KV cache (sin PROMPT_CACHE/TPC)
ENTRADAS: HTTP POST /v1/chat/completions (OpenAI-compatible)
SALIDAS: JSON response o SSE stream
REGLAS INVIOLABLES:
- NUNCA tocar PROMPT_CACHE — solo ephemeral KV cache (make_prompt_cache → generate → discard)
- Adquirir model_lock antes de generar
- Liberar model_lock y limpiar cache en finally
SSoT: Este módulo es la única implementación del sidecar endpoint
"""
from __future__ import annotations

import json
import time
import uuid
from http.server import BaseHTTPRequestHandler
from typing import Any, Callable, Dict, List, Optional

# ── Module-level state (set via init()) ──────────────────────────────────────
_settings: Any = None
_terminal_status: Optional[Callable] = None
_pipeline_log: Optional[Callable] = None
_guard: Any = None
_model_lock: Any = None
_model: Any = None
_tokenizer: Any = None
_thinking_tracker: Any = None
_make_prompt_cache: Optional[Callable] = None
_stream_generate: Optional[Callable] = None
_stream_generate_mtp: Optional[Callable] = None
_build_sampler: Optional[Callable] = None
_stream_generate_kwargs: Optional[Callable] = None
_tokenize_prompt: Optional[Callable] = None
_prepare_messages_for_template: Optional[Callable] = None
_strip_thinking_from_content: Optional[Callable] = None
_rag_available: bool = False
_rag_module: Any = None
_feature_preserve_thinking: bool = True
_ThinkingEvent: Any = None


def init(
    *,
    settings: Any,
    terminal_status: Callable,
    pipeline_log: Callable,
    guard: Any,
    model_lock: Any,
    model: Any,
    tokenizer: Any,
    thinking_tracker: Any,
    make_prompt_cache: Callable,
    stream_generate: Callable,
    stream_generate_mtp: Optional[Callable],
    build_sampler: Callable,
    stream_generate_kwargs_fn: Callable,
    tokenize_prompt: Callable,
    prepare_messages_for_template: Callable,
    strip_thinking_from_content: Callable,
    rag_available: bool,
    rag_module: Any,
    feature_preserve_thinking: bool,
    ThinkingEvent: Any,
) -> None:
    """Initialize with shared state from server2."""
    global _settings, _terminal_status, _pipeline_log, _guard
    global _model_lock, _model, _tokenizer, _thinking_tracker
    global _make_prompt_cache, _stream_generate, _stream_generate_mtp
    global _build_sampler, _stream_generate_kwargs, _tokenize_prompt
    global _prepare_messages_for_template, _strip_thinking_from_content
    global _rag_available, _rag_module, _feature_preserve_thinking, _ThinkingEvent

    _settings = settings
    _terminal_status = terminal_status
    _pipeline_log = pipeline_log
    _guard = guard
    _model_lock = model_lock
    _model = model
    _tokenizer = tokenizer
    _thinking_tracker = thinking_tracker
    _make_prompt_cache = make_prompt_cache
    _stream_generate = stream_generate
    _stream_generate_mtp = stream_generate_mtp
    _build_sampler = build_sampler
    _stream_generate_kwargs = stream_generate_kwargs_fn
    _tokenize_prompt = tokenize_prompt
    _prepare_messages_for_template = prepare_messages_for_template
    _strip_thinking_from_content = strip_thinking_from_content
    _rag_available = rag_available
    _rag_module = rag_module
    _feature_preserve_thinking = feature_preserve_thinking
    _ThinkingEvent = ThinkingEvent


class SidecarHandler(BaseHTTPRequestHandler):
    """Lightweight handler for non-OpenClaw queries sharing the loaded model."""

    def log_message(self, format, *args):
        return  # Suppress default HTTP logging

    def handle_one_request(self):
        """Suppress ConnectionResetError from health probes disconnecting early."""
        try:
            super().handle_one_request()
        except ConnectionResetError:
            pass  # Cosmetic: client closed before we read the request

    def do_GET(self):
        if self.path.rstrip("/") in ("/v1/models", "/models"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "object": "list",
                "data": [{
                    "id": _settings.proxy_model_id,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "mlx-sidecar",
                }]
            }).encode("utf-8"))
            return
        # Health check
        if self.path.rstrip("/") in ("/health", "/"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "status": "ok",
                "model": _settings.model_path,
                "sidecar": True,
                "rag_enabled": _settings.sidecar_enable_rag and _rag_available,
            }).encode("utf-8"))
            return
        self.send_error(404, "Not Found")

    def do_POST(self):
        if self.path.rstrip("/") not in ("/v1/chat/completions", "/chat/completions"):
            self.send_error(404, "Not Found")
            return

        try:
            content_length = int(self.headers["Content-Length"])
            body = json.loads(self.rfile.read(content_length).decode("utf-8"))
        except Exception:
            self.send_error(400, "Bad Request")
            return

        request_id = uuid.uuid4().hex[:12]
        messages = body.get("messages", [])
        is_streaming = body.get("stream", False)
        max_tokens = min(
            body.get("max_tokens", _settings.sidecar_max_tokens),
            _settings.sidecar_max_tokens,
        )

        if not messages:
            self.send_error(400, "No messages provided")
            return

        _terminal_status(
            "🛸", f"Sidecar request {request_id} | "
            f"from={self.client_address[0]} | msgs={len(messages)} | stream={is_streaming} | max_tokens={max_tokens}",
            indent=1,
        )
        _last_content = str(messages[-1].get("content", ""))[:200] if messages else ""
        _terminal_status("🛸", f"Sidecar {request_id} | preview: '{_last_content}'", indent=2)

        # ── RAG ENRICHMENT (optional) ──────────────────────────────────────
        if _settings.sidecar_enable_rag and _rag_available and _rag_module is not None:
            try:
                rag_t0 = time.time()
                messages = _rag_module.enrich_messages(
                    messages, relevance_threshold=_settings.sidecar_rag_threshold
                )
                rag_ms = (time.time() - rag_t0) * 1000
                _terminal_status(
                    "🔍", f"Sidecar RAG: enriched in {rag_ms:.0f}ms",
                    indent=2,
                )
            except Exception as e:
                _terminal_status(
                    "⚠️", f"Sidecar RAG: enrichment failed ({e}) — proceeding without",
                    indent=2,
                )

        # ── TOKENIZE ───────────────────────────────────────────────────────
        try:
            # Hoist system messages to position 0 (Qwen3.5 requirement)
            system_parts = []
            non_system = []
            for m in messages:
                if m.get("role") == "system":
                    c = m.get("content", "")
                    if isinstance(c, str) and c.strip():
                        system_parts.append(c.strip())
                else:
                    non_system.append(m)
            if system_parts:
                messages = [{"role": "system", "content": "\n\n".join(system_parts)}] + non_system

            prepared = _prepare_messages_for_template(messages, _settings.normalize_write_tool_content_for_prompt)

            # Resolve enable_thinking: support both top-level (legacy) and
            # chat_template_kwargs (Qwen3.5 official API)
            _ct_kwargs = body.get("chat_template_kwargs", {})
            _enable_thinking = _ct_kwargs.get(
                "enable_thinking",
                body.get("enable_thinking", _settings.default_thinking),
            )

            if hasattr(_tokenizer, "apply_chat_template") and _tokenizer.chat_template:
                prompt = _tokenizer.apply_chat_template(
                    prepared,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=_enable_thinking,
                    preserve_thinking=_feature_preserve_thinking,
                )
            else:
                prompt = prepared[-1]["content"] if prepared else ""

            prompt_tokens = _tokenize_prompt(prompt)
        except Exception as e:
            _terminal_status("❌", f"Sidecar tokenization failed: {e}")
            self.send_error(500, f"Tokenization error: {e}")
            return

        _terminal_status(
            "📊", f"Sidecar {request_id}: {len(prompt_tokens)} prompt tokens",
            indent=2,
        )

        sampler, logits_processors, _ = _build_sampler(body, enable_thinking=_enable_thinking)

        # ── GENERATE (under model_lock) ────────────────────────────────────
        acquired = False
        generation_started_at = None
        queue_started_at = time.time()

        try:
            _model_lock.acquire(blocking=True)
            acquired = True
            generation_started_at = time.time()
            wait_seconds = generation_started_at - queue_started_at

            if wait_seconds > 1.0:
                _terminal_status(
                    "⏳", f"Sidecar {request_id}: waited {wait_seconds:.1f}s for model_lock",
                    indent=2,
                )

            # Ephemeral KV cache — created fresh, discarded after generation
            ephemeral_cache = _make_prompt_cache(_model)

            if is_streaming:
                self._handle_streaming(
                    request_id, prompt_tokens, max_tokens,
                    sampler, ephemeral_cache, generation_started_at, prepared,
                    logits_processors=logits_processors,
                )
            else:
                self._handle_non_streaming(
                    request_id, prompt_tokens, max_tokens,
                    sampler, ephemeral_cache, generation_started_at, prepared,
                    enable_thinking=_enable_thinking,
                    logits_processors=logits_processors,
                )

        except Exception as e:
            _terminal_status("❌", f"Sidecar {request_id}: generation error: {e}")
            try:
                self.send_error(500, str(e))
            except Exception:
                pass
        finally:
            if acquired:
                # Free ephemeral KV cache immediately
                _guard.post_request_cleanup(request_id)
                if generation_started_at:
                    held_ms = (time.time() - generation_started_at) * 1000
                    _terminal_status(
                        "🔓", f"Sidecar {request_id}: model_lock released ({held_ms:.0f}ms)",
                        indent=2,
                    )
                _model_lock.release()

    def _handle_non_streaming(self, request_id, prompt_tokens, max_tokens,
                               sampler, ephemeral_cache, gen_start, prepared_messages_ref,
                               enable_thinking=True, logits_processors=None):
        """Generate complete response and send as single JSON."""
        full_text = ""
        token_count = 0

        for resp in _stream_generate(
            **_stream_generate_kwargs(prompt_tokens, max_tokens, sampler, ephemeral_cache, logits_processors=logits_processors)
        ):
            full_text += resp.text
            token_count += 1
            if hasattr(sampler, "feed_thinking_text") and resp.text:
                sampler.feed_thinking_text(resp.text)

        # ── STRIP THINKING from response ──────────────────────────────────
        if not enable_thinking:
            pass
        elif "</think>" in full_text:
            full_text = full_text.split("</think>", 1)[1].strip()
        elif "<think>" in full_text:
            full_text = full_text[:full_text.index("<think>")].strip()
        else:
            # No </think> found — model spent all max_tokens thinking.
            # Retry WITHOUT thinking to get a direct answer.
            _terminal_status(
                "🔄", f"Sidecar {request_id}: thinking truncated at {token_count} tokens — "
                f"retrying with enable_thinking=False",
                indent=2,
            )
            # Re-tokenize with thinking disabled
            retry_prompt = _tokenizer.apply_chat_template(
                prepared_messages_ref,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
                preserve_thinking=_feature_preserve_thinking,
            )
            retry_tokens = _tokenize_prompt(retry_prompt)
            retry_cache = _make_prompt_cache(_model)
            full_text = ""
            token_count = 0
            for resp in _stream_generate(
                **_stream_generate_kwargs(retry_tokens, max_tokens, sampler, retry_cache, logits_processors=logits_processors)
            ):
                full_text += resp.text
                token_count += 1
            # Final strip just in case
            full_text = _strip_thinking_from_content(full_text)

        elapsed = time.time() - gen_start
        tps = token_count / elapsed if elapsed > 0 else 0

        response = {
            "id": f"chatcmpl-sc-{request_id}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": _settings.proxy_model_id,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": full_text},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": len(prompt_tokens),
                "completion_tokens": token_count,
                "total_tokens": len(prompt_tokens) + token_count,
            },
        }

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(response).encode("utf-8"))

        _terminal_status(
            "✅", f"Sidecar {request_id}: {token_count} tokens | "
            f"{elapsed:.1f}s | {tps:.1f} tok/s",
            indent=1,
        )

    def _handle_streaming(self, request_id, prompt_tokens, max_tokens,
                           sampler, ephemeral_cache, gen_start, prepared_messages_ref,
                           logits_processors=None):
        """Stream SSE chunks as they're generated."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        token_count = 0
        full_text = ""
        # ThinkingTracker: SSoT — starts in THINKING when enable_thinking=True
        _thinking_tracker.reset(enable_thinking=True)

        _sidecar_gen_fn = (
            _stream_generate_mtp
            if (_settings.enable_mtp and getattr(_model, "mtp", None) is not None)
            else _stream_generate
        )
        for resp in _sidecar_gen_fn(
            **_stream_generate_kwargs(prompt_tokens, max_tokens, sampler, ephemeral_cache, logits_processors=logits_processors)
        ):
            token_count += 1
            text = resp.text
            _sc_event = _thinking_tracker.feed(int(resp.token), text)
            if _sc_event != _ThinkingEvent.NONE:
                _pipeline_log("THINK", request_id,
                    f"{_sc_event.name} at token {token_count} (sidecar)")

            # Skip thinking tokens — only stream visible response
            if _thinking_tracker.is_thinking:
                continue

            full_text += text

            chunk = {
                "id": f"chatcmpl-sc-{request_id}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": _settings.proxy_model_id,
                "choices": [{
                    "index": 0,
                    "delta": {"content": text},
                    "finish_reason": None,
                }],
            }
            try:
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode("utf-8"))
                self.wfile.flush()
            except BrokenPipeError:
                _terminal_status("⚠️", f"Sidecar {request_id}: client disconnected")
                return

        # Send final chunk
        final_chunk = {
            "id": f"chatcmpl-sc-{request_id}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": _settings.proxy_model_id,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }],
        }
        try:
            self.wfile.write(f"data: {json.dumps(final_chunk)}\n\n".encode("utf-8"))
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except BrokenPipeError:
            pass

        elapsed = time.time() - gen_start
        tps = token_count / elapsed if elapsed > 0 else 0
        _terminal_status(
            "✅", f"Sidecar {request_id}: {token_count} tokens streamed | "
            f"{elapsed:.1f}s | {tps:.1f} tok/s",
            indent=1,
        )
