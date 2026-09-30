"""
[AI_DIRECTIVE]
ROL: Unified pipeline logging for server — terminal output, auto-delta tracking, disk logs
OBJETIVO: Centralizar todas las funciones de logging de request pipeline en un módulo reutilizable
ENTRADAS: stage tags, request IDs, messages, structured data
SALIDAS: Console output (with ANSI colors), lastlog.md, per-request JSON logs
REGLAS INVIOLABLES:
- Never crash the pipeline for a log failure (best-effort everywhere)
- Thread-safe: always acquire console_lock before writing
- FEATURE_FULL_LOGGING master switch controls all pipeline_log output
SSoT: Este módulo es la única fuente de logging de pipeline para server.py
"""
import json
import re
import time
import threading
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


# ── ANSI color codes for terminal log highlighting ───────────────────────────
ANSI_YELLOW = "\033[33m"
ANSI_RED = "\033[31m"
ANSI_DIM = "\033[2m"
ANSI_RESET = "\033[0m"
ANSI_STRIP_RE = re.compile(r'\033\[[0-9;]*m')

# Stages that get yellow highlighting (compression/compaction events)
HIGHLIGHT_STAGES = {"COMPRESS", "COMPACT_RUNNER"}


# ── Module-level state (set via init()) ──────────────────────────────────────
_console_lock: threading.Lock = threading.Lock()
_feature_full_logging: bool = True
_feature_log_prompts: bool = True
_lastlog_path: Optional[Path] = None
_pipeline_log_dir: Optional[Path] = None


def init(
    *,
    console_lock: threading.Lock,
    feature_full_logging: bool,
    feature_log_prompts: bool,
    lastlog_path: Path,
    pipeline_log_dir: Path,
) -> None:
    """Initialize the request logger with shared state from server.

    Must be called once during server startup, before any logging calls.
    """
    global _console_lock, _feature_full_logging, _feature_log_prompts
    global _lastlog_path, _pipeline_log_dir

    _console_lock = console_lock
    _feature_full_logging = feature_full_logging
    _feature_log_prompts = feature_log_prompts
    _lastlog_path = lastlog_path
    _pipeline_log_dir = pipeline_log_dir


# ── Auto-Delta Tracker ───────────────────────────────────────────────────────
# Tracks timestamps per request to compute inter-stage deltas automatically.
# Maintains rolling stats per transition type for adaptive slow-detection.
# Thread-safe: always called under console_lock.

class DeltaTracker:
    """Tracks last timestamp per request and rolling stats per transition type."""

    __slots__ = ("_req_last", "_transition_history", "_history_size")

    def __init__(self, history_size: int = 50):
        self._req_last: Dict[str, tuple] = {}          # req[:8] -> (time.time(), stage)
        self._transition_history: Dict[str, deque] = {} # "FROM→TO" -> deque of delta_secs
        self._history_size = history_size

    def record(self, request_id: str, stage: str):
        """Record a timestamp. Returns (delta_secs, from_stage, is_slow) or None."""
        now = time.time()
        key = request_id[:8]
        prev = self._req_last.get(key)
        self._req_last[key] = (now, stage)

        if prev is None:
            return None

        prev_ts, prev_stage = prev
        delta = now - prev_ts

        # Track per-transition rolling history
        trans_key = f"{prev_stage}→{stage}"
        hist = self._transition_history.get(trans_key)
        if hist is None:
            hist = deque(maxlen=self._history_size)
            self._transition_history[trans_key] = hist
        hist.append(delta)

        # Adaptive threshold: P90 × 2, minimum 1.0s
        is_slow = False
        if len(hist) >= 5:
            sorted_hist = sorted(hist)
            p90 = sorted_hist[int(len(sorted_hist) * 0.9)]
            threshold = max(1.0, p90 * 2)
            is_slow = delta > threshold
        elif delta > 5.0:
            # Cold start fallback: flag anything > 5s
            is_slow = True

        return (delta, prev_stage, is_slow)

    def cleanup(self, request_id: str):
        """Remove tracking for a completed request."""
        self._req_last.pop(request_id[:8], None)


# Module-level singleton
delta_tracker = DeltaTracker()


# ── Console Emitter ──────────────────────────────────────────────────────────

def console_emit(line: str) -> None:
    """Print to stdout AND append to lastlog.md (ANSI-stripped). Must be called under console_lock."""
    print(line, flush=True)
    if _lastlog_path is None:
        return
    try:
        clean = ANSI_STRIP_RE.sub("", line)
        with open(_lastlog_path, "a", encoding="utf-8") as f:
            f.write(clean + "\n")
    except Exception:
        pass  # Never crash for a log write


# ── Pipeline Logger ──────────────────────────────────────────────────────────

def pipeline_log(
    stage: str,
    request_id: str,
    message: str,
    data: dict = None,
    indent: int = 1,
) -> None:
    """Unified pipeline logger.  Each stage has its own tag.

    Only fires when FEATURE_FULL_LOGGING = True.
    Per-stage sub-flags (FEATURE_LOG_TOOLS, etc.) are checked by the caller,
    not here — keeps this function fast and simple.

    Auto-delta: appends Δ=Xs to the line when the gap from the previous
    stage (same request) exceeds 500ms.  Paints the line RED when the gap
    exceeds the adaptive threshold (P90×2 of rolling history).
    """
    if not _feature_full_logging:
        return

    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]  # HH:MM:SS.mmm
    tag = f"[{stage}]"
    pad = "  " * max(indent, 0)

    # Auto-delta tracking
    delta_info = delta_tracker.record(request_id, stage)
    delta_suffix = ""
    if delta_info:
        delta_secs, _from_stage, is_slow = delta_info
        if delta_secs >= 0.5:  # Only annotate deltas >= 500ms
            delta_suffix = f" Δ={delta_secs:.3f}s"

    line = f"{pad}{tag} {ts} req={request_id[:8]} | {message}{delta_suffix}"

    # Color priority: red (slow) > yellow (highlight stage) > default
    if delta_info and delta_info[2]:  # is_slow
        line = f"{ANSI_RED}{line}{ANSI_RESET}"
    elif stage in HIGHLIGHT_STAGES:
        line = f"{ANSI_YELLOW}{line}{ANSI_RESET}"

    with _console_lock:
        console_emit(line)

    # Optionally dump structured data to disk
    if data is not None and _feature_log_prompts:
        write_request_log(request_id, stage, data)


# ── Tool Call Formatter ──────────────────────────────────────────────────────

def fmt_tc_for_log(tool_calls: list) -> str:
    """Format tool calls for logging with smart truncation.

    Long arg values (>30 chars) are shown as: first10......last10
    Short values are shown as-is.
    """
    parts = []
    for tc in tool_calls:
        fn = tc.get("function", {})
        name = fn.get("name", "?")
        raw_args = fn.get("arguments", "{}")
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except (json.JSONDecodeError, TypeError):
            args = raw_args
        if isinstance(args, dict):
            fmt_args = {}
            for k, v in args.items():
                sv = str(v)
                if len(sv) > 30:
                    fmt_args[k] = f"{sv[:10]}......{sv[-10:]}"
                else:
                    fmt_args[k] = v
            parts.append(f"{name}({json.dumps(fmt_args, ensure_ascii=False)})")
        else:
            s = str(args)
            if len(s) > 80:
                s = s[:40] + "..." + s[-40:]
            parts.append(f"{name}({s})")
    return " | ".join(parts)


# ── Request Log Writer ───────────────────────────────────────────────────────

def write_request_log(request_id: str, stage: str, data: Any) -> None:
    """Write pipeline stage data to logs/requests/{request_id}/{stage}.json.

    Best-effort: never raises — a log failure must not crash the pipeline.
    """
    if _pipeline_log_dir is None:
        return
    try:
        req_dir = _pipeline_log_dir / request_id
        req_dir.mkdir(parents=True, exist_ok=True)
        out_path = req_dir / f"{stage.lower().replace(' ', '_')}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    except Exception:
        pass  # Best-effort — never crash the pipeline for a log


# ── Terminal Status Logger ───────────────────────────────────────────────────

def terminal_status(icon: str, message: str, indent: int = 0, *,
                    request_id: str = None, stage: str = None) -> None:
    """Terminal status logger.

    When request_id and stage are provided, uses unified pipeline format
    with auto-delta tracking:  icon [STAGE] HH:MM:SS.mmm req=xxx | message [Δ=Xs]
    Otherwise uses legacy emoji format:  icon [HH:MM:SS] message

    Backward compatible — all existing calls work unchanged.
    """
    now = datetime.now()

    if request_id and stage:
        # ── Unified format with delta tracking ──
        ts = now.strftime("%H:%M:%S.%f")[:-3]
        tag = f"[{stage}]"
        pad = "  " * max(indent, 0)

        delta_info = delta_tracker.record(request_id, stage)
        delta_suffix = ""
        if delta_info:
            delta_secs, _from_stage, is_slow = delta_info
            if delta_secs >= 0.5:
                delta_suffix = f" Δ={delta_secs:.3f}s"

        line = f"{pad}{icon} {tag} {ts} req={request_id[:8]} | {message}{delta_suffix}"

        if delta_info and delta_info[2]:  # is_slow
            line = f"{ANSI_RED}{line}{ANSI_RESET}"
    else:
        # ── Legacy format (startup, system messages) ──
        ts = now.strftime("%H:%M:%S.%f")[:-3]
        pad = "  " * max(indent, 0)
        line = f"{pad}  {icon} {ts} | {message}"

    with _console_lock:
        console_emit(line)
