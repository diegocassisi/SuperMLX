"""
Temporal anchor module for MLXTurboQuant / SuperMLX1.3.83.

Purpose:
- inject a small, explicit time context block into model input
- give Qwen a deterministic weekday/date lookup table
- prevent year/timezone hallucinations from partial timestamps like
  "Apr 8 23:23"

Recommended insertion points in SuperMLX1.3.83.py:

1. Main LM path:
   healed_messages, _loop_broken = _break_tool_call_loop(...)
   healed_messages = temporal_anchor.inject_temporal_anchor(healed_messages, config)
   original_messages, canonical_messages = _canonicalize_messages(healed_messages)

2. Sidecar path:
   messages = temporal_anchor.inject_temporal_anchor(messages, config)

3. Cache scrub:
   cache_prompt = temporal_anchor.scrub_temporal_anchor(cache_prompt)
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo


DAYS_ES = [
    "Lunes",
    "Martes",
    "Miercoles",
    "Jueves",
    "Viernes",
    "Sabado",
    "Domingo",
]

MONTHS_ES = [
    "Enero",
    "Febrero",
    "Marzo",
    "Abril",
    "Mayo",
    "Junio",
    "Julio",
    "Agosto",
    "Septiembre",
    "Octubre",
    "Noviembre",
    "Diciembre",
]

DEFAULT_TIMEZONE = "America/Argentina/Buenos_Aires"
DEFAULT_TAG_NAME = "temporal_anchor"

_STARTUP_PATTERN = re.compile(
    r"(new session|/new|/reset|session startup|run your session startup)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class TemporalAnchorConfig:
    timezone_name: str = DEFAULT_TIMEZONE
    include_week_map: bool = True
    week_days: int = 7
    inject_mode: str = "always"  # "always" | "startup_only"
    tag_name: str = DEFAULT_TAG_NAME


def _safe_tz(timezone_name: str) -> ZoneInfo:
    try:
        return ZoneInfo(timezone_name)
    except Exception:
        return ZoneInfo("UTC")


def _coerce_now(now: Optional[datetime], tz: ZoneInfo) -> datetime:
    if now is None:
        return datetime.now(tz)
    if now.tzinfo is None:
        return now.replace(tzinfo=tz)
    return now.astimezone(tz)


def _format_gmt_offset(dt: datetime) -> str:
    offset = dt.utcoffset() or timedelta(0)
    total_minutes = int(offset.total_seconds() // 60)
    sign = "+" if total_minutes >= 0 else "-"
    total_minutes = abs(total_minutes)
    hours, minutes = divmod(total_minutes, 60)
    if minutes == 0:
        return f"GMT{sign}{hours}"
    return f"GMT{sign}{hours:02d}:{minutes:02d}"


def _build_human_date_es(now_local: datetime) -> str:
    day_name = DAYS_ES[now_local.weekday()]
    month_name = MONTHS_ES[now_local.month - 1]
    return f"{day_name}, {now_local.day} de {month_name} de {now_local.year}"


def _build_upcoming_week(now_local: datetime, days: int) -> str:
    lines = ["<upcoming_week>"]
    for offset in range(max(0, days)):
        day_dt = now_local + timedelta(days=offset)
        day_name = DAYS_ES[day_dt.weekday()]
        lines.append(
            f'  <day offset="{offset}" name="{day_name}" date="{day_dt.date().isoformat()}" />'
        )
    lines.append("</upcoming_week>")
    return "\n".join(lines)


def _anchor_open(tag_name: str) -> str:
    return f"<{tag_name}>"


def _anchor_close(tag_name: str) -> str:
    return f"</{tag_name}>"


def _anchor_pattern(tag_name: str) -> re.Pattern[str]:
    return re.compile(
        re.escape(_anchor_open(tag_name)) + r"[\s\S]*?" + re.escape(_anchor_close(tag_name)),
        re.IGNORECASE,
    )


def _flatten_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text", "")))
            elif isinstance(part, str):
                parts.append(part)
        return "".join(parts)
    return str(content)


def is_startup_turn(messages: List[Dict[str, Any]]) -> bool:
    if not messages:
        return False
    for msg in reversed(messages):
        if (msg.get("role") or "").lower() != "user":
            continue
        text = _flatten_text_content(msg.get("content", ""))
        return bool(_STARTUP_PATTERN.search(text))
    return False


def has_temporal_anchor(
    messages: List[Dict[str, Any]],
    tag_name: str = DEFAULT_TAG_NAME,
) -> bool:
    open_tag = _anchor_open(tag_name)
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str) and open_tag in content:
            return True
        if isinstance(content, list):
            for part in content:
                if (
                    isinstance(part, dict)
                    and part.get("type") == "text"
                    and open_tag in str(part.get("text", ""))
                ):
                    return True
    return False


def build_temporal_anchor(
    *,
    now: Optional[datetime] = None,
    config: Optional[TemporalAnchorConfig] = None,
) -> str:
    config = config or TemporalAnchorConfig()
    tz = _safe_tz(config.timezone_name)
    now_local = _coerce_now(now, tz)
    now_utc = now_local.astimezone(timezone.utc)

    lines = [
        _anchor_open(config.tag_name),
        f"<current_date_iso>{now_local.date().isoformat()}</current_date_iso>",
        (
            "<current_local_datetime>"
            f"{now_local.isoformat(timespec='seconds')}"
            "</current_local_datetime>"
        ),
        (
            "<current_utc_datetime>"
            f"{now_utc.isoformat(timespec='seconds')}"
            "</current_utc_datetime>"
        ),
        f"<current_date_es>{_build_human_date_es(now_local)}</current_date_es>",
        (
            '<current_weekday_num monday_zero="true">'
            f"{now_local.weekday()}"
            "</current_weekday_num>"
        ),
        f"<timezone_name>{config.timezone_name}</timezone_name>",
        f"<timezone_offset>{_format_gmt_offset(now_local)}</timezone_offset>",
        (
            "<temporal_rule>"
            "If a file, tool, or message shows a partial timestamp without year or timezone "
            '(for example "Apr 8 23:23"), treat it as ambiguous and do not infer the missing parts.'
            "</temporal_rule>"
        ),
        (
            "<temporal_rule>"
            "If exact dating matters, prefer the explicit ISO dates in this block or verify with a tool."
            "</temporal_rule>"
        ),
    ]

    if config.include_week_map:
        lines.append(_build_upcoming_week(now_local, config.week_days))

    lines.append(_anchor_close(config.tag_name))
    return "\n".join(lines)


def inject_temporal_anchor(
    messages: List[Dict[str, Any]],
    config: Optional[TemporalAnchorConfig] = None,
) -> List[Dict[str, Any]]:
    """
    Return a deep-copied message list with a standalone system message appended.

    The server already hoists/merges system messages later, so this module stays
    intentionally dumb and self-contained.
    """
    config = config or TemporalAnchorConfig()
    cloned = copy.deepcopy(messages)

    if has_temporal_anchor(cloned, tag_name=config.tag_name):
        return cloned

    if config.inject_mode == "startup_only" and not is_startup_turn(cloned):
        return cloned

    cloned.append(
        {
            "role": "system",
            "content": build_temporal_anchor(config=config),
        }
    )
    return cloned


def scrub_temporal_anchor(
    prompt: str,
    *,
    tag_name: str = DEFAULT_TAG_NAME,
    mask_char: str = "0",
) -> str:
    """
    Length-preserving scrub for cache-key use only.

    This mirrors the server's existing cache scrubber strategy.
    """
    if not isinstance(prompt, str):
        return prompt

    pattern = _anchor_pattern(tag_name)

    def _mask(match: re.Match[str]) -> str:
        return mask_char * len(match.group(0))

    return pattern.sub(_mask, prompt)


__all__ = [
    "DEFAULT_TAG_NAME",
    "DEFAULT_TIMEZONE",
    "TemporalAnchorConfig",
    "build_temporal_anchor",
    "has_temporal_anchor",
    "inject_temporal_anchor",
    "is_startup_turn",
    "scrub_temporal_anchor",
]
