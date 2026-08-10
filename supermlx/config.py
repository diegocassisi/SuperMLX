# SPDX-License-Identifier: MIT
"""
SuperMLX configuration: environment helpers, Settings dataclass, model family detection.

All functions are pure (read env → return value). No side effects.
"""
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return default if value is None else value.strip()


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _env_str_any(names: List[str], default: str) -> str:
    for name in names:
        raw = os.getenv(name)
        if raw is not None and raw.strip() != "":
            return raw.strip()
    return default


def _env_int_any(names: List[str], default: int) -> int:
    for name in names:
        raw = os.getenv(name)
        if raw is None or raw.strip() == "":
            continue
        try:
            return int(raw)
        except ValueError:
            continue
    return default


def _env_bool_any(names: List[str], default: bool) -> bool:
    for name in names:
        raw = os.getenv(name)
        if raw is None or raw.strip() == "":
            continue
        normalized = raw.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return default


def _env_kv_bits(name: str, default: Optional[int]) -> Optional[int]:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    normalized = raw.strip().upper()
    if normalized == "OFF":
        return None
    try:
        return int(float(normalized))  # int() so "4.0" → 4, not 4.0
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    model_path: str
    model_family: str
    force_text_mode: bool
    mlx_host: str
    mlx_port: int
    proxy_port: int
    prompt_cache_max_entries_global: int
    prompt_cache_max_entries_per_session: int
    prompt_cache_ttl_seconds: int
    prompt_cache_session_max_idle_seconds: int
    max_kv_size: int
    kv_group_size: int
    kv_bits: Optional[float]
    kv_quant_scheme: str
    quantized_kv_start: int
    default_temperature: float
    default_top_p: float
    default_top_k: int
    default_min_p: float
    default_repetition_penalty: float
    default_repetition_context_size: int
    default_presence_penalty: float
    default_presence_context_size: int
    default_max_tokens: int
    enable_request_logging: bool
    default_thinking: bool
    max_thinking_tokens: int            # Max tokens in <think> block before forcing transition (0=unlimited)
    vlm_cache_debug: bool
    normalize_write_tool_content_for_prompt: bool
    cache_canonicalize_tool_context: bool
    cache_session_partitioning: bool
    prompt_cache_block_size: int
    cache_use_block_index: bool
    cache_norm_safety_check: bool
    log_root: Path
    proxy_startup_wait_seconds: float
    proxy_model_id: str
    cache_persist_path: str
    embedded_cache_persist_path: str  # Kripper slot for EMBEDDED agents (compaction/memory)
    memory_guard_threshold_gb: float  # Metal RAM threshold (GB) for pre-prefill eviction (0=disabled)
    sidecar_port: int                 # Sidecar port for non-OpenClaw queries (0=disabled)
    sidecar_max_tokens: int           # Max tokens for sidecar responses (keep low to minimize lock time)
    sidecar_enable_rag: bool          # Enable RAG enrichment on sidecar requests
    sidecar_rag_threshold: float      # L2 distance threshold for sidecar RAG (stricter than OpenClaw's 1.6)
    moe_expert_capacity: int          # Max experts per MoE layer (0=auto, based on RAM)
    moe_target_capacity: int          # Target capacity after warmup (0=no expansion)
    moe_expert_profile: str           # Path to expert profile JSON for MoE pinning (""=none)
    moe_shallow_pin_layers: int       # Pin top experts in first N MoE layers (0=disabled)
    moe_shallow_pin_top: int          # How many experts to pin per shallow layer
    compile_model: bool               # Wrap model with mx.compile for JIT op fusion
    async_prefill: bool               # Use mx.async_eval in adaptive prefill loop


def _normalize_model_family(value: Optional[str]) -> str:
    raw = (value or "").strip().lower()
    if raw in {"qwen", "qwen3", "qen3", "qwopus", "agents"}:
        return "qwen3"
    if raw in {"glm", "glm4", "glm-4"}:
        return "glm4"
    if raw in {"gemma", "gemma4", "gemma-4"}:
        return "gemma4"
    if raw in {"deepseek", "deepseek-v3", "deepseek-r1"}:
        return "deepseek"
    if raw in {"hermes", "hermes3", "hermes-3"}:
        return "hermes"
    return "generic"


def _infer_model_family(model_path: str) -> str:
    normalized = (model_path or "").strip().lower()
    if "qwen" in normalized or "qwopus" in normalized or "agents-a1" in normalized:
        return "qwen3"
    if "glm" in normalized:
        return "glm4"
    if "gemma" in normalized:
        return "gemma4"
    if "deepseek" in normalized:
        return "deepseek"
    if "hermes" in normalized:
        return "hermes"
    return "generic"


def build_settings(script_dir: Path = Path(__file__).parent) -> Settings:
    """Build Settings from environment variables. Pure function (reads env only)."""
    _default_model = "mlx-community/Qwen3.5-9B-4bit"
    model_path = _env_str("MODEL_PATH", _default_model)
    model_family = _normalize_model_family(
        _env_str("MODEL_FAMILY", _infer_model_family(model_path))
    )
    proxy_model_id = _env_str(
        "PROXY_MODEL_ID",
        model_path if model_path.startswith("openai/") else f"openai/{model_path}",
    )
    return Settings(
        model_path=model_path,
        model_family=model_family,
        force_text_mode=_env_bool("FORCE_TEXT_MODE", False),
        mlx_host=_env_str("MLX_HOST", "0.0.0.0"),
        mlx_port=_env_int("MLX_PORT", 8080),
        proxy_port=_env_int("PROXY_PORT", 4000),
        prompt_cache_max_entries_global=_env_int_any(
            ["PROMPT_CACHE_MAX_ENTRIES_GLOBAL", "PROMPT_CACHE_MAX_SIZE"],
            2,  # 24GB M4 Pro: modelo(5GB) + 2 MAIN entries(9GB) + 1 COMPACT(4.5GB) + scratch(5GB) = ~23.5GB
        ),
        prompt_cache_max_entries_per_session=_env_int_any(
            ["PROMPT_CACHE_MAX_ENTRIES_PER_SESSION"],
            2,
        ),
        prompt_cache_ttl_seconds=_env_int("PROMPT_CACHE_TTL_SECONDS", 0),  # 0 = disabled (no time-based expiry)
        prompt_cache_session_max_idle_seconds=_env_int_any(
            ["PROMPT_CACHE_SESSION_MAX_IDLE_SECONDS"],
            30 * 60,
        ),
        max_kv_size=_env_int("MAX_KV_SIZE", 196608),
        kv_group_size=_env_int("KV_GROUP_SIZE", 64),
        kv_bits=_env_kv_bits("KV_BITS", None),
        kv_quant_scheme=_env_str("KV_QUANT_SCHEME", "uniform"),
        #cambiado por diego, original 50
        quantized_kv_start=_env_int("QUANTIZED_KV_START", 0),
        default_temperature=_env_float("DEFAULT_TEMPERATURE", 0.6),
        default_top_p=_env_float("DEFAULT_TOP_P", 0.95),
        default_top_k=_env_int("DEFAULT_TOP_K", 20),
        default_min_p=_env_float("DEFAULT_MIN_P", 0.0),
        default_repetition_penalty=_env_float("DEFAULT_REPETITION_PENALTY", 1.1),
        default_repetition_context_size=_env_int(
            "DEFAULT_REPETITION_CONTEXT_SIZE", 64
        ),
        default_presence_penalty=_env_float("DEFAULT_PRESENCE_PENALTY", 0.0),
        default_presence_context_size=_env_int("DEFAULT_PRESENCE_CONTEXT_SIZE", 20),
        default_max_tokens=_env_int("DEFAULT_MAX_TOKENS", 2048),
        enable_request_logging=_env_bool("ENABLE_REQUEST_LOGGING", True),
        default_thinking=_env_bool("DEFAULT_THINKING", True),
        max_thinking_tokens=_env_int("MAX_THINKING_TOKENS", 4096),
        vlm_cache_debug=_env_bool("VLM_CACHE_DEBUG", False),
        normalize_write_tool_content_for_prompt=_env_bool(
            "NORMALIZE_WRITE_TOOL_CONTENT_FOR_PROMPT", False
        ),
        cache_canonicalize_tool_context=_env_bool_any(
            ["CACHE_CANONICALIZE_TOOL_CONTEXT"],
            True,
        ),
        cache_session_partitioning=_env_bool_any(
            ["CACHE_SESSION_PARTITIONING"],
            True,
        ),
        prompt_cache_block_size=_env_int("PROMPT_CACHE_BLOCK_SIZE", 16),
        cache_use_block_index=_env_bool_any(
            ["CACHE_USE_BLOCK_INDEX"],
            True,
        ),
        cache_norm_safety_check=_env_bool("CACHE_NORM_SAFETY_CHECK", False),
        log_root=Path(_env_str("LOG_ROOT", str(script_dir / "logs"))),
        proxy_startup_wait_seconds=_env_float("PROXY_STARTUP_WAIT_SECONDS", 2.0),
        proxy_model_id=proxy_model_id,
        cache_persist_path=_env_str("CACHE_PERSIST_PATH", ""),
        embedded_cache_persist_path=_env_str("EMBEDDED_CACHE_PERSIST_PATH", ""),
        # Memory guard: evict cache entries if Metal RAM exceeds this threshold (GB).
        # Default: total RAM minus 8GB OS reserve (not 80% of total — the server
        # doesn't own all RAM; macOS + apps consume ~5-7GB permanently).
        # Set to 0 to disable.
        memory_guard_threshold_gb=_env_float(
            "MEMORY_GUARD_THRESHOLD_GB",
            round(os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / (1024**3) - 8.0, 1)
            if hasattr(os, 'sysconf') else 16.0,
        ),
        # Sidecar: lightweight OpenAI-compatible endpoint for non-OpenClaw queries.
        # Shares model in RAM, uses ephemeral KV cache (0 memory at rest).
        # Set to 0 to disable.
        sidecar_port=_env_int("SIDECAR_PORT", 8081),
        sidecar_max_tokens=_env_int("SIDECAR_MAX_TOKENS", 16384),
        sidecar_enable_rag=_env_bool("SIDECAR_ENABLE_RAG", True),
        sidecar_rag_threshold=_env_float("SIDECAR_RAG_THRESHOLD", 1.4),
        moe_expert_capacity=_env_int("MOE_EXPERT_CAPACITY", 100),
        moe_target_capacity=_env_int("MOE_TARGET_CAPACITY", 0),
        moe_expert_profile=_env_str("MOE_EXPERT_PROFILE", ""),
        moe_shallow_pin_layers=_env_int("MOE_SHALLOW_PIN_LAYERS", 5),
        moe_shallow_pin_top=_env_int("MOE_SHALLOW_PIN_TOP", 6),
        compile_model=_env_bool("SUPERMLX_COMPILE_MODEL", False),
        async_prefill=_env_bool("SUPERMLX_ASYNC_PREFILL", False),
    )
