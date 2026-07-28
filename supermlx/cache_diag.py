"""
FIX-31 Cache Diagnostics — Opt-in cache state tracing.

Controlled by FEATURE_CACHE_DIAG env flag (default: false).
When enabled, instruments every cache mutation with before/after snapshots
and divergence detection. Near-zero overhead (~1ms per operation).

Usage in server.py:
    from .cache_diag import cache_diag

    snap = cache_diag.snapshot(prompt_cache, cache_key, "before_trim")
    trim_prompt_cache(prompt_cache, n)
    cache_diag.compare(snap, prompt_cache, cache_key, "trim", request_id, extra={...})
"""

from __future__ import annotations

import hashlib
import time
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# ANSI colors (same as server.py)
_ANSI_RED = "\033[31m"
_ANSI_CYAN = "\033[36m"
_ANSI_MAGENTA = "\033[35m"
_ANSI_DIM = "\033[2m"
_ANSI_RESET = "\033[0m"


@dataclass
class DiagSnapshot:
    """Immutable snapshot of cache state at a point in time."""
    label: str
    timestamp: float
    kv_offset: Optional[int]
    cache_key_len: int
    last_8_tokens: Tuple[int, ...]
    key_tail_hash: str          # SHA-256 of last 64 tokens
    layer_count: int
    arrays_cache_count: int     # layers with .state (recurrent)
    kv_cache_count: int         # layers with .offset (attention)
    recurrent_fingerprint: Optional[str]  # lightweight fingerprint of recurrent state
    cache_obj_id: int           # id(prompt_cache) for identity tracking


@dataclass
class DiagResult:
    """Result of comparing two snapshots around an operation."""
    operation: str
    request_id: str
    before: DiagSnapshot
    after: DiagSnapshot
    offset_delta: int
    hash_changed: bool
    recurrent_changed: bool
    unexpected_divergence: bool  # True if hash changed but operation shouldn't have changed it
    duration_ms: float
    extra: Dict[str, Any] = field(default_factory=dict)


def _kv_offset(cache: Any) -> Optional[int]:
    """Extract KV offset from cache layers (mirrors server._kv_cache_offset)."""
    try:
        layers = cache if isinstance(cache, (list, tuple)) else [cache]
        for layer in layers:
            if hasattr(layer, "offset"):
                return int(layer.offset)
        for layer in layers:
            if hasattr(layer, "cache") and isinstance(layer.cache, list):
                for c in layer.cache:
                    if c is not None and len(c.shape) >= 3:
                        return int(c.shape[2])
    except (IndexError, TypeError, AttributeError):
        pass
    return None


def _key_tail_hash(cache_key: list, n: int = 64) -> str:
    """SHA-256 of the last N tokens in cache_key."""
    if not cache_key:
        return "empty"
    tail = cache_key[-n:]
    raw = b"".join(t.to_bytes(4, "big", signed=True) for t in tail)
    return hashlib.sha256(raw).hexdigest()[:16]


def _recurrent_fingerprint(cache: Any) -> Optional[str]:
    """Lightweight fingerprint of ArraysCache recurrent state.
    
    Takes the first recurrent layer's state, reads 16 float values,
    and hashes them. Detects unexpected state mutations.
    Cost: ~0.1ms (no GPU sync, reads from cached CPU mirror).
    """
    try:
        layers = cache if isinstance(cache, (list, tuple)) else [cache]
        for layer in layers:
            if hasattr(layer, "state") and layer.state is not None:
                import mlx.core as mx
                # Take first 16 values from state for fingerprinting
                state = layer.state
                if isinstance(state, list):
                    state = state[0] if state else None
                if state is None:
                    continue
                # Flatten and take first 16 values
                flat = state.reshape(-1)[:16]
                values = flat.tolist()
                raw = str(values).encode()
                return hashlib.sha256(raw).hexdigest()[:16]
    except Exception:
        pass
    return None


def _count_layer_types(cache: Any) -> Tuple[int, int, int]:
    """Returns (total_layers, arrays_cache_count, kv_cache_count)."""
    if not cache:
        return (0, 0, 0)
    layers = cache if isinstance(cache, (list, tuple)) else [cache]
    total = len(layers)
    arrays = sum(1 for l in layers if hasattr(l, "state"))
    kv = sum(1 for l in layers if hasattr(l, "offset"))
    return (total, arrays, kv)


# Operations that are EXPECTED to change the key tail hash
_MUTATING_OPS = frozenset({
    "prefill", "adaptive_prefill", "generate", "stream_generate",
    "trim", "trim_prompt_cache", "cold_start", "make_prompt_cache",
    "tpc_inject", "restore_frozen", "hybrid_restore",
    "cache_insert", "marconi_trim",
})

# Operations that should NOT change the hash (read-only or identity)
_READONLY_OPS = frozenset({
    "cache_lookup", "hybrid_capture",
})


class CacheDiagnostics:
    """Thread-safe cache diagnostics with before/after snapshots.
    
    All methods are no-ops when disabled (FEATURE_CACHE_DIAG=false).
    """

    def __init__(self, enabled: bool = False, console_lock: threading.Lock = None):
        self.enabled = enabled
        self._lock = console_lock or threading.Lock()
        self._emit_fn = None  # Set by server.py to _console_emit for lastlog.md capture
        self._request_ops: Dict[str, List[DiagResult]] = {}  # req[:8] -> results

    def snapshot(
        self,
        cache: Any,
        cache_key: list,
        label: str = "",
    ) -> Optional[DiagSnapshot]:
        """Capture a lightweight snapshot of current cache state."""
        if not self.enabled:
            return None

        total, arrays, kv = _count_layer_types(cache)
        return DiagSnapshot(
            label=label,
            timestamp=time.time(),
            kv_offset=_kv_offset(cache),
            cache_key_len=len(cache_key) if cache_key else 0,
            last_8_tokens=tuple(cache_key[-8:]) if cache_key else (),
            key_tail_hash=_key_tail_hash(cache_key),
            layer_count=total,
            arrays_cache_count=arrays,
            kv_cache_count=kv,
            recurrent_fingerprint=_recurrent_fingerprint(cache),
            cache_obj_id=id(cache) if cache else 0,
        )

    def compare(
        self,
        before: Optional[DiagSnapshot],
        cache: Any,
        cache_key: list,
        operation: str,
        request_id: str,
        extra: Dict[str, Any] = None,
    ) -> Optional[DiagResult]:
        """Compare before snapshot with current state, log divergence."""
        if not self.enabled or before is None:
            return None

        after = self.snapshot(cache, cache_key, f"after_{operation}")
        if after is None:
            return None

        offset_delta = (after.kv_offset or 0) - (before.kv_offset or 0)
        hash_changed = before.key_tail_hash != after.key_tail_hash
        recurrent_changed = (
            before.recurrent_fingerprint is not None
            and after.recurrent_fingerprint is not None
            and before.recurrent_fingerprint != after.recurrent_fingerprint
        )

        # Detect unexpected divergence
        unexpected = False
        if operation in _READONLY_OPS and (hash_changed or recurrent_changed):
            unexpected = True
        if operation not in _MUTATING_OPS and operation not in _READONLY_OPS:
            # Unknown op — flag if anything changed
            unexpected = hash_changed or recurrent_changed

        duration_ms = (after.timestamp - before.timestamp) * 1000

        result = DiagResult(
            operation=operation,
            request_id=request_id[:8],
            before=before,
            after=after,
            offset_delta=offset_delta,
            hash_changed=hash_changed,
            recurrent_changed=recurrent_changed,
            unexpected_divergence=unexpected,
            duration_ms=duration_ms,
            extra=extra or {},
        )

        # Log to console
        self._emit(result)

        # Track per-request
        req_key = request_id[:8]
        if req_key not in self._request_ops:
            self._request_ops[req_key] = []
        self._request_ops[req_key].append(result)
        # Cap at 100 per request
        if len(self._request_ops[req_key]) > 100:
            self._request_ops[req_key] = self._request_ops[req_key][-50:]

        return result

    def _emit(self, r: DiagResult) -> None:
        """Print diagnostic line to console."""
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        tag = f"[CACHE_DIAG]"

        # Build compact state diff
        offset_str = f"off={r.before.kv_offset}→{r.after.kv_offset}"
        hash_str = f"hash={'CHANGED' if r.hash_changed else 'ok'}"
        recur_str = f"recur={'CHANGED' if r.recurrent_changed else 'ok'}"
        obj_str = f"obj={'SAME' if r.before.cache_obj_id == r.after.cache_obj_id else 'NEW'}"

        parts = [
            f"{r.operation}",
            offset_str,
            hash_str,
            recur_str,
            obj_str,
            f"{r.duration_ms:.1f}ms",
        ]

        # Add extra info
        if r.extra:
            for k, v in list(r.extra.items())[:4]:
                parts.append(f"{k}={v}")

        line = f"  {tag} {ts} req={r.request_id} | {' | '.join(parts)}"

        # Color: RED for unexpected divergence, MAGENTA for expected mutations, DIM for no-change
        if r.unexpected_divergence:
            line = f"{_ANSI_RED}  ⚠️ DIVERGENCE {line}{_ANSI_RESET}"
        elif r.hash_changed or r.recurrent_changed:
            line = f"{_ANSI_MAGENTA}{line}{_ANSI_RESET}"
        else:
            line = f"{_ANSI_DIM}{line}{_ANSI_RESET}"

        with self._lock:
            if self._emit_fn:
                self._emit_fn(line)
            else:
                print(line, flush=True)

    def cleanup(self, request_id: str) -> None:
        """Remove tracking for completed request."""
        self._request_ops.pop(request_id[:8], None)

    def summary(self, request_id: str) -> Optional[str]:
        """Return a one-line summary of all ops for a request."""
        ops = self._request_ops.get(request_id[:8])
        if not ops:
            return None
        divergences = sum(1 for o in ops if o.unexpected_divergence)
        mutations = sum(1 for o in ops if o.hash_changed or o.recurrent_changed)
        return (
            f"CACHE_DIAG summary req={request_id[:8]}: "
            f"{len(ops)} ops | {mutations} mutations | "
            f"{divergences} UNEXPECTED divergences"
        )


# ── Module-level singleton (initialized as disabled; server.py enables it) ──
cache_diag = CacheDiagnostics(enabled=False)
