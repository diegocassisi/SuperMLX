"""
SuperMLX Pipeline Monitor — Real-time observability dashboard.

Reads structured pipeline logs from SuperMLX and displays them in a
Streamlit dashboard. No modifications to the server needed — reads logs
from the configured logs directory.

Usage:
    pip install streamlit plotly psutil
    streamlit run pipeline_monitor.py -- --logs-dir ./logs
"""

import json
import os
import re
import subprocess
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import streamlit as st

# ── Config ────────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent
DEFAULT_LOGS_DIR = SCRIPT_DIR / "logs"

# Parse --logs-dir from sys.argv (Streamlit passes args after --)
_logs_dir_str = os.environ.get("SUPERMLX_LOGS_DIR", "")
for i, arg in enumerate(sys.argv):
    if arg == "--logs-dir" and i + 1 < len(sys.argv):
        _logs_dir_str = sys.argv[i + 1]
        break

LOGS_DIR = Path(_logs_dir_str) if _logs_dir_str else DEFAULT_LOGS_DIR

# ── Page Config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="SuperMLX Monitor",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styles ────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@400;600;700&display=swap');

    .stApp {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 50%, #0d0d1f 100%);
        color: #e0e0e8;
    }

    h1, h2, h3 { font-family: 'Inter', sans-serif; }
    .stMetric label { color: #8888aa !important; font-family: 'Inter', sans-serif; }
    .stMetric [data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.8rem !important;
    }

    div[data-testid="stExpander"] {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
    }

    .pipeline-stage {
        padding: 8px 14px;
        border-radius: 8px;
        margin: 4px 0;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
    }

    .stage-compress { background: rgba(147,51,234,0.15); border-left: 3px solid #9333ea; }
    .stage-rag { background: rgba(59,130,246,0.15); border-left: 3px solid #3b82f6; }
    .stage-heal { background: rgba(34,197,94,0.15); border-left: 3px solid #22c55e; }
    .stage-cache { background: rgba(234,179,8,0.15); border-left: 3px solid #eab308; }
    .stage-gen { background: rgba(249,115,22,0.15); border-left: 3px solid #f97316; }
    .stage-emergency { background: rgba(239,68,68,0.15); border-left: 3px solid #ef4444; }
    .stage-default { background: rgba(255,255,255,0.05); border-left: 3px solid #666; }

    .metric-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }

    /* Force word-wrap on all code/pre blocks */
    pre, code, .stCode, .stCodeBlock {
        white-space: pre-wrap !important;
        word-wrap: break-word !important;
        word-break: break-word !important;
        overflow-wrap: break-word !important;
        max-width: 100% !important;
        overflow-x: hidden !important;
    }

    /* Constrain expander content */
    div[data-testid="stExpander"] > div {
        max-width: 100% !important;
        overflow-x: hidden !important;
    }

    /* Main content area */
    .block-container {
        max-width: 100% !important;
        overflow-x: hidden !important;
    }

    /* Alternating message backgrounds */
    .msg-even {
        background: rgba(255,255,255,0.02);
        border-left: 4px solid rgba(255,255,255,0.15);
        padding: 10px 14px;
        margin: 8px 0;
        border-radius: 8px;
    }
    .msg-odd {
        background: rgba(100,140,255,0.06);
        border-left: 4px solid rgba(100,140,255,0.25);
        padding: 10px 14px;
        margin: 8px 0;
        border-radius: 8px;
    }
    .msg-system {
        border-left: 4px solid #9333ea;
        background: rgba(147,51,234,0.12);
        padding: 10px 14px;
        margin: 8px 0;
        border-radius: 8px;
    }
    .msg-user {
        border-left: 4px solid #3b82f6;
        background: rgba(59,130,246,0.12);
        padding: 10px 14px;
        margin: 8px 0;
        border-radius: 8px;
    }
    .msg-assistant {
        border-left: 4px solid #22c55e;
        background: rgba(34,197,94,0.12);
        padding: 10px 14px;
        margin: 8px 0;
        border-radius: 8px;
    }
    .msg-tool {
        border-left: 4px solid #f97316;
        background: rgba(249,115,22,0.12);
        padding: 10px 14px;
        margin: 8px 0;
        border-radius: 8px;
    }
    .msg-header {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        font-size: 0.9rem;
        margin-bottom: 6px;
        color: #c0c0d0;
    }
    .msg-body {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        white-space: pre-wrap;
        word-wrap: break-word;
        word-break: break-word;
        overflow-wrap: break-word;
        color: #d0d0e0;
        max-height: 400px;
        overflow-y: auto;
        line-height: 1.4;
    }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

STAGE_COLORS = {
    "COMPRESS": "stage-compress",
    "RAG": "stage-rag",
    "HEAL": "stage-heal",
    "CACHE": "stage-cache",
    "CANON": "stage-cache",
    "TQ": "stage-cache",
    "GEN": "stage-gen",
    "RESP": "stage-gen",
    "MSG_OUT": "stage-gen",
    "WIRE": "stage-gen",
    "EMERGENCY": "stage-emergency",
    "LOOP_BREAK": "stage-emergency",
    "INBOUND": "stage-default",
    "MSG_IN": "stage-default",
    "TOOLS": "stage-default",
    "CASCADE": "stage-compress",
    "BYPASS": "stage-default",
    "METAL": "stage-gen",
}


def _get_metal_memory_gb():
    """Get current Metal GPU memory usage via mlx (if available)."""
    try:
        import mlx.core as mx
        get_mem = getattr(mx, 'get_active_memory', None) or getattr(mx.metal, 'get_active_memory', None)
        if get_mem:
            return get_mem() / (1024**3)
    except Exception:
        pass
    return None


def _get_system_memory():
    """Get system memory info via psutil."""
    try:
        import psutil
        vm = psutil.virtual_memory()
        return {
            "total_gb": vm.total / (1024**3),
            "available_gb": vm.available / (1024**3),
            "used_gb": vm.used / (1024**3),
            "percent": vm.percent,
        }
    except ImportError:
        return None


def _parse_terminal_line(line: str):
    """Parse a _terminal_status or _pipeline_log line into structured data."""
    # _pipeline_log format: "  [STAGE] HH:MM:SS.mmm req=XXXXXXXX | message"
    pipeline_match = re.match(
        r'\s*\[(\w+)\]\s+(\d{2}:\d{2}:\d{2}\.\d{3})\s+req=(\w+)\s+\|\s+(.*)',
        line
    )
    if pipeline_match:
        return {
            "type": "pipeline",
            "stage": pipeline_match.group(1),
            "time": pipeline_match.group(2),
            "request_id": pipeline_match.group(3),
            "message": pipeline_match.group(4),
        }

    # _terminal_status format: "  ICON [HH:MM:SS] message"
    status_match = re.match(
        r'\s*(.+?)\s+\[(\d{2}:\d{2}:\d{2})\]\s+(.*)',
        line
    )
    if status_match:
        return {
            "type": "status",
            "icon": status_match.group(1),
            "time": status_match.group(2),
            "message": status_match.group(3),
        }

    return None


def _parse_session_log(filepath: Path):
    """Parse a cache-session log file into structured entries."""
    entries = []
    try:
        content = filepath.read_text(encoding="utf-8", errors="ignore")
        # Split on timestamp headers
        blocks = re.split(r'\[(\d{4}-\d{2}-\d{2}T[\d:.]+)\]\s+request_id=(\w+)\s+direction=(\w+)', content)
        # blocks[0] is before first match, then groups of (ts, req_id, direction, body)
        for i in range(1, len(blocks) - 3, 4):
            ts = blocks[i]
            req_id = blocks[i+1]
            direction = blocks[i+2]
            body_raw = blocks[i+3].strip()
            try:
                body = json.loads(body_raw)
            except (json.JSONDecodeError, ValueError):
                body = {"raw": body_raw[:500]}

            entries.append({
                "timestamp": ts,
                "request_id": req_id,
                "direction": direction,
                "data": body,
            })
    except Exception as e:
        st.error(f"Error parsing {filepath.name}: {e}")
    return entries


def _get_available_dates():
    """List available log dates."""
    dates = []
    if LOGS_DIR.exists():
        for d in sorted(LOGS_DIR.iterdir(), reverse=True):
            if d.is_dir() and re.match(r'\d{4}-\d{2}-\d{2}', d.name):
                dates.append(d.name)
    return dates


def _get_sessions_for_date(date_str: str):
    """List session log files for a date."""
    date_dir = LOGS_DIR / date_str
    sessions = []
    if date_dir.exists():
        for f in sorted(date_dir.iterdir(), reverse=True):
            if f.name.startswith("cache-session-") and f.suffix == ".log":
                sessions.append(f)
    return sessions


def _extract_request_metrics(entries):
    """Extract key metrics from parsed session entries."""
    metrics = []
    for entry in entries:
        if entry["direction"] == "prompt":
            data = entry["data"]
            meta = data.get("request_meta", {})
            metrics.append({
                "request_id": entry["request_id"],
                "timestamp": entry["timestamp"],
                "prompt_tokens": meta.get("prompt_tokens", 0),
                "matched_prefix_len": meta.get("matched_prefix_len", 0),
                "cache_match_type": meta.get("cache_match_type", "?"),
                "cache_selection_source": meta.get("cache_selection_source", "?"),
                "enable_thinking": meta.get("enable_thinking", False),
                "session_id": meta.get("session_id", "?")[:16],
                "messages_count": len(data.get("messages", [])),
                "tools_count": len(data.get("tools", []) or []),
                "stable_prefix_token_len": meta.get("stable_prefix_token_len", 0),
            })
        elif entry["direction"] == "sampler":
            data = entry["data"]
            # Find matching prompt entry
            for m in metrics:
                if m["request_id"] == entry["request_id"]:
                    m["rest_tokens"] = data.get("rest_tokens", 0)
                    m["temperature"] = data.get("applied_kwargs", {}).get("temp", "?")
                    break
        elif entry["direction"] == "generation":
            data = entry["data"]
            for m in metrics:
                if m["request_id"] == entry["request_id"]:
                    timing = data.get("timing", {})
                    m["prefill_s"] = timing.get("prefill_seconds", 0)
                    m["decode_s"] = timing.get("decode_seconds", 0)
                    m["prefill_tps"] = timing.get("prefill_tps", 0)
                    m["decode_tps"] = timing.get("decode_tps", 0)
                    m["finish_reason"] = data.get("finish_reason", "?")
                    m["has_tool_calls"] = bool(data.get("tool_calls"))
                    # Count thinking vs output tokens
                    raw = data.get("raw_response_text", "")
                    normalized = data.get("normalized_response_text", "")
                    m["raw_len"] = len(raw)
                    m["normalized_len"] = len(normalized)
                    break
    return metrics


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🔬 SuperMLX Monitor")
    st.markdown("---")

    available_dates = _get_available_dates()
    if not available_dates:
        st.warning(f"No logs found in {LOGS_DIR}")
        st.stop()

    # Auto-select today if available
    today_str = datetime.now().strftime("%Y-%m-%d")
    default_date_idx = 0
    if today_str in available_dates:
        default_date_idx = available_dates.index(today_str)

    selected_date = st.selectbox("📅 Date", available_dates, index=default_date_idx)
    sessions = _get_sessions_for_date(selected_date)

    if not sessions:
        st.warning(f"No sessions for {selected_date}")
        st.stop()

    # Auto-select the most recently modified (= active) session
    sessions_with_mtime = [(f, f.stat().st_mtime) for f in sessions]
    sessions_with_mtime.sort(key=lambda x: x[1], reverse=True)
    sessions_sorted = [f for f, _ in sessions_with_mtime]

    session_labels = []
    for f in sessions_sorted:
        mtime = datetime.fromtimestamp(f.stat().st_mtime)
        size_mb = f.stat().st_size / (1024 * 1024)
        sid = f.stem.replace("cache-session-", "")
        label = f"🔗 {sid[:12]} | {mtime.strftime('%H:%M')} | {size_mb:.1f}MB"
        if f == sessions_sorted[0]:
            label = f"⭐ {sid[:12]} | {mtime.strftime('%H:%M')} | {size_mb:.1f}MB (active)"
        session_labels.append(label)

    selected_idx = st.selectbox("🔗 Session", range(len(sessions_sorted)),
                                format_func=lambda i: session_labels[i])
    selected_session = sessions_sorted[selected_idx]

    st.markdown("---")
    mtime = datetime.fromtimestamp(selected_session.stat().st_mtime)
    st.markdown(f"📁 `{LOGS_DIR}`")
    st.markdown(f"🕐 Last write: **{mtime.strftime('%H:%M:%S')}**")
    st.markdown(f"📦 Size: **{selected_session.stat().st_size / (1024*1024):.1f} MB**")

    # Auto-refresh
    auto_refresh = st.checkbox("🔄 Auto-refresh (10s)", value=False)
    if auto_refresh:
        time.sleep(0.1)
        st.rerun()


# ── Main ──────────────────────────────────────────────────────────────────────

st.markdown("# 🔬 SuperMLX Pipeline Monitor")
st.caption(f"Session: `{selected_session.name}` | Date: `{selected_date}`")

# Parse session
entries = _parse_session_log(selected_session)
metrics = _extract_request_metrics(entries)

if not metrics:
    st.info("No requests found in this session log.")
    st.stop()

# ── Compute all dashboard values ─────────────────────────────────────────────

cache_hits_count = sum(1 for m in metrics if m.get("cache_match_type") not in ("miss", "?"))
hit_pct = (cache_hits_count / len(metrics) * 100) if metrics else 0
avg_rest = sum(m.get("rest_tokens", 0) for m in metrics) / max(len(metrics), 1)
avg_prefill = sum(m.get("prefill_s", 0) for m in metrics) / max(len(metrics), 1)
avg_decode_tps = sum(m.get("decode_tps", 0) for m in metrics) / max(len(metrics), 1)

_total_prefill = sum(m.get("prefill_s", 0) for m in metrics)
_cache_hits = [m for m in metrics if m.get("cache_match_type") not in ("miss", "?")]
_cache_misses = [m for m in metrics if m.get("cache_match_type") in ("miss", "?")]
_avg_miss_prefill = (
    sum(m.get("prefill_s", 0) for m in _cache_misses) / max(len(_cache_misses), 1)
)
_avg_hit_prefill = (
    sum(m.get("prefill_s", 0) for m in _cache_hits) / max(len(_cache_hits), 1)
)
_estimated_no_cache = _avg_miss_prefill * len(metrics) if _cache_misses else _total_prefill
_time_saved = max(0, _estimated_no_cache - _total_prefill)
_speedup = _estimated_no_cache / max(_total_prefill, 0.1)

sys_mem = _get_system_memory()
metal_gb = _get_metal_memory_gb()
_ram_used = f"{sys_mem['used_gb']:.1f}" if sys_mem else "—"
_ram_total = f"{sys_mem['total_gb']:.1f}" if sys_mem else "—"
_ram_pct = sys_mem['percent'] if sys_mem else 0
_metal_str = f"{metal_gb:.1f}" if metal_gb is not None else "—"

# Cache hit gauge color
if hit_pct >= 85:
    _gauge_color = "#22c55e"
    _gauge_glow = "rgba(34,197,94,0.3)"
elif hit_pct >= 50:
    _gauge_color = "#eab308"
    _gauge_glow = "rgba(234,179,8,0.3)"
else:
    _gauge_color = "#ef4444"
    _gauge_glow = "rgba(239,68,68,0.3)"

# ── Dashboard HTML ───────────────────────────────────────────────────────────

st.markdown(f"""
<style>
    .dash-grid {{
        display: grid;
        grid-template-columns: 1fr 1fr 1fr;
        gap: 16px;
        margin: 16px 0 24px 0;
    }}
    .dash-hero {{
        grid-column: 1 / -1;
        display: grid;
        grid-template-columns: 1fr 1fr 1fr;
        gap: 16px;
    }}
    .card {{
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 20px 24px;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
    }}
    .card-hero {{
        background: linear-gradient(135deg, rgba(34,197,94,0.12) 0%, rgba(59,130,246,0.08) 100%);
        border: 1px solid rgba(34,197,94,0.2);
        border-radius: 20px;
        padding: 28px 32px;
        text-align: center;
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
    }}
    .card-gauge {{
        background: linear-gradient(135deg, rgba(255,255,255,0.06) 0%, rgba(255,255,255,0.02) 100%);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 20px;
        padding: 28px 32px;
        text-align: center;
        backdrop-filter: blur(16px);
    }}
    .card-label {{
        font-family: 'Inter', -apple-system, sans-serif;
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: rgba(255,255,255,0.45);
        margin-bottom: 6px;
    }}
    .card-value {{
        font-family: 'JetBrains Mono', 'SF Mono', monospace;
        font-size: 1.6rem;
        font-weight: 700;
        color: #f0f0f8;
        line-height: 1.2;
    }}
    .card-value-hero {{
        font-family: 'JetBrains Mono', 'SF Mono', monospace;
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #22c55e, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1.1;
    }}
    .card-sub {{
        font-family: 'Inter', -apple-system, sans-serif;
        font-size: 0.75rem;
        color: rgba(255,255,255,0.35);
        margin-top: 4px;
    }}
    .gauge-ring {{
        width: 100px;
        height: 100px;
        border-radius: 50%;
        background: conic-gradient(
            {_gauge_color} 0deg,
            {_gauge_color} {hit_pct * 3.6}deg,
            rgba(255,255,255,0.06) {hit_pct * 3.6}deg,
            rgba(255,255,255,0.06) 360deg
        );
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 8px auto;
        box-shadow: 0 0 24px {_gauge_glow};
    }}
    .gauge-inner {{
        width: 76px;
        height: 76px;
        border-radius: 50%;
        background: #13132e;
        display: flex;
        align-items: center;
        justify-content: center;
    }}
    .gauge-text {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.4rem;
        font-weight: 700;
        color: {_gauge_color};
    }}
    .dash-row {{
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 12px;
        margin-top: 12px;
    }}
    .card-sm {{
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px;
        padding: 14px 16px;
        backdrop-filter: blur(8px);
    }}
    .card-sm .card-value {{
        font-size: 1.2rem;
    }}
</style>

<div class="dash-hero">
    <div class="card-gauge">
        <div class="card-label">Cache Hit Rate</div>
        <div class="gauge-ring">
            <div class="gauge-inner">
                <span class="gauge-text">{hit_pct:.0f}%</span>
            </div>
        </div>
        <div class="card-sub">{cache_hits_count} hits / {len(_cache_misses)} misses</div>
    </div>
    <div class="card-hero">
        <div class="card-label">Time Saved This Session</div>
        <div class="card-value-hero">{_time_saved:.0f}s</div>
        <div class="card-sub">
            {_speedup:.1f}× speedup &nbsp;·&nbsp;
            {_total_prefill:.0f}s actual vs ~{_estimated_no_cache:.0f}s without cache
        </div>
    </div>
    <div class="card-gauge">
        <div class="card-label">System Memory</div>
        <div style="margin: 12px 0;">
            <div class="card-value" style="font-size: 1.4rem;">{_ram_used} / {_ram_total} GiB</div>
            <div class="card-sub" style="margin-top: 8px;">RAM {_ram_pct:.0f}% &nbsp;·&nbsp; Metal GPU {_metal_str} GiB</div>
        </div>
        <div style="background: rgba(255,255,255,0.06); border-radius: 4px; height: 6px; margin-top: 12px; overflow: hidden;">
            <div style="background: {'#ef4444' if _ram_pct > 85 else '#eab308' if _ram_pct > 70 else '#22c55e'}; height: 100%; width: {_ram_pct}%; border-radius: 4px; transition: width 0.3s;"></div>
        </div>
    </div>
</div>

<div class="dash-row">
    <div class="card-sm">
        <div class="card-label">Requests</div>
        <div class="card-value">{len(metrics)}</div>
    </div>
    <div class="card-sm">
        <div class="card-label">Avg Prefill</div>
        <div class="card-value">{avg_prefill:.1f}s</div>
        <div class="card-sub">cold {_avg_miss_prefill:.1f}s · warm {_avg_hit_prefill:.1f}s</div>
    </div>
    <div class="card-sm">
        <div class="card-label">Decode Speed</div>
        <div class="card-value">{avg_decode_tps:.1f} <span style="font-size: 0.7rem; color: rgba(255,255,255,0.4);">tok/s</span></div>
    </div>
    <div class="card-sm">
        <div class="card-label">Avg Rest Tokens</div>
        <div class="card-value">{avg_rest:,.0f}</div>
        <div class="card-sub">tokens to prefill after cache</div>
    </div>
    <div class="card-sm">
        <div class="card-label">Total Prefill</div>
        <div class="card-value">{_total_prefill:.0f}s</div>
        <div class="card-sub">across {len(metrics)} requests</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Request Timeline ──────────────────────────────────────────────────────────

st.markdown("## 📋 Request Timeline")
st.caption("Most recent request first.")

# Reverse: newest request at the top
metrics_display = list(reversed(metrics))
entries_reversed = True  # Flag for matching entries by full request_id

for i, m in enumerate(metrics_display):
    req_id = m.get("request_id", "?")[:8]
    prompt_tok = m.get("prompt_tokens", 0)
    rest_tok = m.get("rest_tokens", 0)
    cache_type = m.get("cache_match_type", "?")
    matched = m.get("matched_prefix_len", 0)
    prefill_s = m.get("prefill_s", 0)
    decode_tps = m.get("decode_tps", 0)
    finish = m.get("finish_reason", "?")
    thinking = m.get("enable_thinking", False)
    msgs = m.get("messages_count", 0)
    tools = m.get("tools_count", 0)

    # Cache color indicator
    if prompt_tok > 0 and matched > 0:
        hit_ratio = matched / prompt_tok * 100
    else:
        hit_ratio = 0
    if hit_ratio >= 85:
        cache_light = "🟢"
    elif hit_ratio >= 50:
        cache_light = "🟡"
    else:
        cache_light = "🔴"

    # Rest tokens danger indicator
    rest_danger = "🚨" if rest_tok > 20000 else ("⚠️" if rest_tok > 10000 else "")

    # Timestamp
    ts = m.get("timestamp", "")
    ts_display = ""
    if ts:
        try:
            ts_dt = datetime.fromisoformat(ts)
            ts_display = ts_dt.strftime("%H:%M:%S")
        except (ValueError, TypeError):
            ts_display = ts[:8]

    header = (
        f"{cache_light} `{ts_display}` `{req_id}` | "
        f"cache={cache_type} ({hit_ratio:.0f}%) | "
        f"rest={rest_tok:,} {rest_danger} | "
        f"prefill={prefill_s:.1f}s | "
        f"decode={decode_tps:.1f} tok/s | "
        f"finish={finish}"
    )

    with st.expander(header, expanded=False):
        dcol1, dcol2, dcol3, dcol4 = st.columns(4)
        with dcol1:
            st.markdown(f"**Prompt Tokens:** `{prompt_tok:,}`")
            st.markdown(f"**Matched Prefix:** `{matched:,}`")
            st.markdown(f"**Rest Tokens:** `{rest_tok:,}`")
        with dcol2:
            st.markdown(f"**Messages:** `{msgs}`")
            st.markdown(f"**Tools:** `{tools}`")
            st.markdown(f"**Thinking:** `{thinking}`")
        with dcol3:
            st.markdown(f"**Prefill:** `{prefill_s:.2f}s`")
            prefill_tps = m.get("prefill_tps", 0)
            st.markdown(f"**Prefill TPS:** `{prefill_tps:.0f}`")
            st.markdown(f"**Decode TPS:** `{decode_tps:.1f}`")
        with dcol4:
            st.markdown(f"**Cache Source:** `{m.get('cache_selection_source', '?')}`")
            st.markdown(f"**Stable Prefix:** `{m.get('stable_prefix_token_len', 0):,}`")
            st.markdown(f"**Temperature:** `{m.get('temperature', '?')}`")

        # Show raw prompt/generation data if available
        # Use full request_id for matching (not truncated)
        full_req_id = m.get("request_id", "?")
        matching_entries = [e for e in entries if e["request_id"] == full_req_id]

        if matching_entries:
            st.markdown("---")
            for entry in matching_entries:
                direction = entry["direction"]
                if direction == "prompt":
                    st.markdown(f"### 📥 Inbound Messages ({msgs} msgs)")

                    # Show rendered prompt if available
                    rendered = entry["data"].get("rendered_prompt", "")
                    if rendered:
                        st.markdown(f"**📜 Rendered Prompt** ({len(rendered):,} chars, ~{len(rendered)//4:,} tok)")
                        st.code(rendered, language="text")

                    # Show each message with role-based coloring
                    msg_list = entry["data"].get("messages", [])
                    for mi, msg in enumerate(msg_list):
                        role = msg.get("role", "?")
                        content = msg.get("content", "")
                        if isinstance(content, list):
                            content = str(content)
                        elif not isinstance(content, str):
                            content = str(content)

                        role_emoji = {"system": "⚙️", "user": "👤", "assistant": "🤖", "tool": "🔧"}.get(role, "❓")
                        role_class = f"msg-{role}" if role in ("system", "user", "assistant", "tool") else ("msg-even" if mi % 2 == 0 else "msg-odd")
                        content_len = len(content)
                        est_tokens = content_len // 4

                        import html as html_mod
                        escaped = html_mod.escape(content)
                        st.markdown(
                            f'<div class="{role_class}">'
                            f'<div class="msg-header">{role_emoji} {role.upper()} &mdash; {est_tokens:,} tok &mdash; msg[{mi}]</div>'
                            f'<div class="msg-body">{escaped}</div>'
                            f'</div>',
                            unsafe_allow_html=True
                        )

                elif direction == "generation":
                    st.markdown("### 📤 Generation Output")
                    raw = entry["data"].get("raw_response_text", "")
                    assistant_text = entry["data"].get("assistant_message_text", "")
                    tool_calls = entry["data"].get("tool_calls", [])

                    import html as html_mod

                    if raw:
                        thinking_match = re.search(r'<think>(.*?)</think>', raw, re.DOTALL)
                        if thinking_match:
                            think_text = thinking_match.group(1)
                            st.markdown(
                                f'<div class="msg-system">'
                                f'<div class="msg-header">🧠 THINKING &mdash; {len(think_text)//4:,} tok</div>'
                                f'<div class="msg-body">{html_mod.escape(think_text)}</div>'
                                f'</div>',
                                unsafe_allow_html=True
                            )

                    if assistant_text:
                        st.markdown(
                            f'<div class="msg-assistant">'
                            f'<div class="msg-header">💬 RESPONSE &mdash; {len(assistant_text)//4:,} tok</div>'
                            f'<div class="msg-body">{html_mod.escape(assistant_text)}</div>'
                            f'</div>',
                            unsafe_allow_html=True
                        )

                    if tool_calls:
                        for tc in tool_calls:
                            func = tc.get("function", {})
                            tc_str = f"{func.get('name', '?')}({func.get('arguments', '')})"
                            st.markdown(
                                f'<div class="msg-tool">'
                                f'<div class="msg-header">🔧 TOOL CALL</div>'
                                f'<div class="msg-body">{html_mod.escape(tc_str)}</div>'
                                f'</div>',
                                unsafe_allow_html=True
                            )

# ── Footer ────────────────────────────────────────────────────────────────────

st.markdown("---")
st.caption(f"SuperMLX Pipeline Monitor v1.4.2 | Logs: `{LOGS_DIR}` | {datetime.now().strftime('%H:%M:%S')}")

if auto_refresh:
    time.sleep(10)
    st.rerun()
