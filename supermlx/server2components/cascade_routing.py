"""
[AI_DIRECTIVE]
ROL: Cascade routing — forward requests to a frontier API
OBJETIVO: When local model confidence is low (RAG score > threshold),
          forward the request to a frontier API (Gemini, OpenAI, etc.)
ENTRADAS: OpenAI-compatible request body, request_id, optional handler for SSE
SALIDAS: JSON response dict (non-streaming) or None (streaming, proxied via SSE)
REGLAS INVIOLABLES:
- Only activate when FEATURE_CASCADE=true AND API credentials configured
- Always fall back to local model on cascade failure
- Zero extra dependencies (uses urllib from stdlib)
SSoT: This module is the only implementation of cascade routing
"""
from __future__ import annotations

import json
import logging
import os
import time
import urllib.request
import urllib.error
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

# ── Module-level state (set via init()) ──────────────────────────────────────
_pipeline_log: Optional[Callable] = None

# ── Config (from env) ────────────────────────────────────────────────────────
def _env_str(key: str, default: str) -> str:
    return os.environ.get(key, default)

def _env_bool(key: str, default: bool) -> bool:
    val = os.environ.get(key, str(default)).strip().lower()
    return val in ("1", "true", "yes")

def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, str(default)))
    except (ValueError, TypeError):
        return default

def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, str(default)))
    except (ValueError, TypeError):
        return default

FEATURE_CASCADE = _env_bool("FEATURE_CASCADE", False)
CASCADE_API_URL = _env_str("CASCADE_API_URL", "")
CASCADE_API_KEY = _env_str("CASCADE_API_KEY", "")
CASCADE_MODEL = _env_str("CASCADE_MODEL", "")
CASCADE_RAG_THRESHOLD = _env_float("CASCADE_RAG_THRESHOLD", 2.0)
CASCADE_TIMEOUT_S = _env_int("CASCADE_TIMEOUT_S", 60)


def init(*, pipeline_log: Callable) -> None:
    """Initialize with shared state from server2."""
    global _pipeline_log
    _pipeline_log = pipeline_log


def is_available() -> bool:
    """Return True if cascade routing is configured and enabled."""
    return bool(FEATURE_CASCADE and CASCADE_API_URL and CASCADE_API_KEY)


def cascade_forward_request(
    body: Dict[str, Any],
    request_id: str,
    handler: Any = None,
    is_streaming: bool = False,
) -> Optional[Dict[str, Any]]:
    """Forward request to a frontier API (Gemini, etc.) when RAG confidence is low.

    Uses urllib (stdlib) — zero extra dependencies. The frontier API must be
    OpenAI-compatible (/v1/chat/completions or equivalent).

    Two modes:
        - Non-streaming (is_streaming=False): returns parsed JSON response dict.
        - Streaming (is_streaming=True): proxies SSE lines directly to handler.wfile,
          returns None. Requires handler to be passed.

    Raises: Exception on timeout, HTTP error, or parse failure.
    """
    # Build the request payload — pass through most fields from the original body
    cascade_body = {
        "model": CASCADE_MODEL,
        "messages": body.get("messages", []),
        "stream": is_streaming,
    }
    # Pass through optional fields if present
    for _key in ("temperature", "top_p", "max_tokens", "tools", "tool_choice"):
        if _key in body:
            cascade_body[_key] = body[_key]

    _payload = json.dumps(cascade_body, ensure_ascii=False).encode("utf-8")

    _req = urllib.request.Request(
        CASCADE_API_URL,
        data=_payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {CASCADE_API_KEY}",
        },
        method="POST",
    )

    _log = _pipeline_log or (lambda *a, **k: None)

    _t0 = time.time()
    try:
        if is_streaming and handler is not None:
            # ── STREAMING: proxy SSE lines directly to client ──
            handler.send_response(200)
            handler.send_header("Content-Type", "text/event-stream")
            handler.send_header("Cache-Control", "no-cache")
            handler.send_header("X-Cascade-Active", "true")
            handler.send_header("X-Cascade-Model", CASCADE_MODEL)
            handler.end_headers()

            with urllib.request.urlopen(_req, timeout=CASCADE_TIMEOUT_S) as resp:
                for line in resp:
                    handler.wfile.write(line)
            handler.wfile.flush()

            _elapsed_ms = int((time.time() - _t0) * 1000)
            _log("CASCADE", request_id,
                f"frontier streamed | status=200 | {_elapsed_ms}ms")
            return None  # Response already sent via SSE
        else:
            # ── NON-STREAMING: parse and return JSON ──
            with urllib.request.urlopen(_req, timeout=CASCADE_TIMEOUT_S) as resp:
                _resp_data = resp.read().decode("utf-8")
                _elapsed_ms = int((time.time() - _t0) * 1000)
                _log("CASCADE", request_id,
                    f"frontier responded | status={resp.status} | {_elapsed_ms}ms | "
                    f"resp_len={len(_resp_data)}")
                return json.loads(_resp_data)
    except urllib.error.HTTPError as e:
        _err_body = ""
        try:
            _err_body = e.read().decode("utf-8")[:500]
        except Exception:
            pass
        raise RuntimeError(
            f"Cascade HTTP {e.code}: {e.reason} | body={_err_body}"
        ) from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Cascade connection error: {e.reason}") from e
