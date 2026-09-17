# SPDX-License-Identifier: MIT
"""
[AI_DIRECTIVE]
ROL: SuperMLX configuration: environment helpers, Settings dataclass, model family detection.
OBJETIVO: Cargar, validar y centralizar todas las variables de configuración del servidor desde .env y entorno.
ENTRADAS: Variables de entorno del sistema operativo / archivo .env
SALIDAS: Instancia inmutable Settings con tipado y valores por defecto calibrados
REGLAS INVIOLABLES:
- Prohibido hardcoding de rutas o números mágicos fuera de los defaults declarados
- Prohibida la lógica con efectos secundarios; todas las funciones deben ser puras
- Obligatorio mantener paridad de tipado estricto con Settings dataclass
SSoT: Este módulo es la única fuente de verdad (SSoT) para la configuración del servidor.
"""
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

try:
    from dotenv import load_dotenv
    _dotenv_path = Path(__file__).resolve().parent.parent / ".env"
    if _dotenv_path.exists():
        load_dotenv(dotenv_path=_dotenv_path, override=False)
except ImportError:
    pass


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
    thinking_temperature: float
    thinking_schedule_enabled: bool
    thinking_temp_schedule: str
    thinking_temperature_min: float
    thinking_cooling_exponent: float
    thinking_cooling_fraction: float
    temp_resp_precise: float
    temp_resp_explain: float
    temp_resp_explore: float
    response_temperature: float
    tool_calling_temperature: float
    compaction_temperature: float
    temp_resp_code: float
    temp_resp_tech: float
    temp_resp_prose: float
    temp_resp_creative: float
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
    thinking_budget_mode: str            # "v1" (break+inject) | "v2" (logits forcing, KV-consistent)
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
    ngram_loop_detection: bool        # Enable verbatim n-gram repetition loop detector
    ngram_max_repeats: int            # Max allowed repeats before breaking generation
    ngram_size: int                   # Tokens per n-gram pattern
    ngram_window: int                 # Search window size in tokens
    ngram_check_interval: int         # Interval in tokens between loop checks
    adaptive_temperature_enabled: bool # Enable prefill domain-adaptive temperature
    adaptive_temperature_layer: int    # Layer to tap for domain classification (default: 20)
    adaptive_temperature_prosa: float  # Response temperature for prose domain
    adaptive_temperature_codigo_mate: float # Response temperature for code/math domain
    adaptive_temperature_default: float # Fallback response temperature
    adaptive_markers_prosa: str        # Comma-separated expert IDs for prose
    adaptive_markers_codigo_mate: str  # Comma-separated expert IDs for code/math
    enable_mtp: bool                   # Multi-Token Prediction (MTP) speculative decoding
    mtp_weights_path: str              # Local directory containing MTP weights and config
    mtp_adaptive_temperature: bool     # Sync MTP speculative sampling temperature with model (DualPhase / Domain-Adaptive)
    thinking_thermal_spark: bool       # Enable Thermal Spark (re-heating pulse) in thinking mode
    spark_min_tokens_threshold: int    # Minimum thinking tokens before sparks are allowed
    spark_periodic_interval: int       # Interval in tokens for periodic heartbeat sparks
    spark_pulse_duration: int          # Duration of the spark in tokens
    spark_temperature: float           # Temperature during thermal spark
    spark_min_p: float                 # min_p safety filter during thermal spark
    spark_cooldown_tokens: int         # Minimum tokens between consecutive sparks


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
        thinking_temperature=_env_float("THINKING_TEMPERATURE", 0.50),
        thinking_schedule_enabled=_env_bool("THINKING_SCHEDULE_ENABLED", False),
        thinking_temp_schedule=_env_str(
            "THINKING_TEMP_SCHEDULE", "0:0.80,256:0.60,1024:0.35,3000:0.10"
        ),
        thinking_temperature_min=_env_float("THINKING_TEMPERATURE_MIN", 0.10),
        thinking_cooling_exponent=_env_float("THINKING_COOLING_EXPONENT", 3.0),
        thinking_cooling_fraction=_env_float("THINKING_COOLING_FRACTION", 0.25),
        temp_resp_precise=_env_float("TEMP_RESP_PRECISE", 0.10),
        temp_resp_explain=_env_float("TEMP_RESP_EXPLAIN", 0.50),
        temp_resp_explore=_env_float("TEMP_RESP_EXPLORE", 0.85),
        response_temperature=_env_float("RESPONSE_TEMPERATURE", 0.30),
        tool_calling_temperature=_env_float("TOOL_CALLING_TEMPERATURE", 0.10),
        compaction_temperature=_env_float("COMPACTION_TEMPERATURE", 0.20),
        temp_resp_code=_env_float("TEMP_RESP_CODE", 0.10),
        temp_resp_tech=_env_float("TEMP_RESP_TECH", 0.30),
        temp_resp_prose=_env_float("TEMP_RESP_PROSE", 0.65),
        temp_resp_creative=_env_float("TEMP_RESP_CREATIVE", 0.85),
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
        thinking_budget_mode=_env_str("THINKING_BUDGET_MODE", "v1"),
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
        ngram_loop_detection=_env_bool("NGRAM_LOOP_DETECTION", False),
        ngram_max_repeats=_env_int("NGRAM_MAX_REPEATS", 12),
        ngram_size=_env_int("NGRAM_SIZE", 35),
        ngram_window=_env_int("NGRAM_WINDOW", 400),
        ngram_check_interval=_env_int("NGRAM_CHECK_INTERVAL", 32),
        adaptive_temperature_enabled=_env_bool("FEATURE_ADAPTIVE_TEMPERATURE", False),
        adaptive_temperature_layer=_env_int("ADAPTIVE_TEMP_LAYER", 20),
        adaptive_temperature_prosa=_env_float("ADAPTIVE_TEMP_PROSA", 0.85),
        adaptive_temperature_codigo_mate=_env_float("ADAPTIVE_TEMP_CODIGO_MATE", 0.10),
        adaptive_temperature_default=_env_float("ADAPTIVE_TEMP_DEFAULT", 0.40),
        adaptive_markers_prosa=_env_str("ADAPTIVE_MARKERS_PROSA", "19"),
        adaptive_markers_codigo_mate=_env_str("ADAPTIVE_MARKERS_CODIGO_MATE", "72,133,23,148,42,130,161,198,226"),
        enable_mtp=_env_bool("ENABLE_MTP", False),
        mtp_weights_path=_env_str("MTP_WEIGHTS_PATH", "models/Qwen3.6-35B-A3B-MTP-MLX"),
        mtp_adaptive_temperature=_env_bool("MTP_ADAPTIVE_TEMPERATURE", True),
        thinking_thermal_spark=_env_bool("FEATURE_THINKING_THERMAL_SPARK", True),
        spark_min_tokens_threshold=_env_int("SPARK_MIN_TOKENS_THRESHOLD", 1200),
        spark_periodic_interval=_env_int("SPARK_PERIODIC_INTERVAL", 1500),
        spark_pulse_duration=_env_int("SPARK_PULSE_DURATION", 35),
        spark_temperature=_env_float("SPARK_TEMPERATURE", 0.88),
        spark_min_p=_env_float("SPARK_MIN_P", 0.05),
        spark_cooldown_tokens=_env_int("SPARK_COOLDOWN_TOKENS", 250),
    )


# Default global Settings instance
SETTINGS = build_settings()

