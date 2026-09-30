# Changelog

All notable changes to SuperMLX are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [3.0.0] — 2026-09-30

First release of the rewritten server. Versions 1.5 – 2.x were internal checkpoints
(see the git tags); this entry summarizes everything since 1.4.x.

### Added
- **Radix prompt cache** (prefix tree of KV states) with eviction against a Metal memory
  budget, and a **4-phase request pipeline** (preprocess → cache lookup → generate → postprocess).
- **MTP speculative decoding** for models with a Multi-Token-Prediction head.
- **Thinking control:** token budget enforced by logits forcing (KV-consistent), optional
  temperature schedule, thermal sparks and epistemic/closure nudges.
- **Task-aware temperature** via `[TASK: CODE|TECH|PROSE|CREATIVE]` tags.
- **Metal memory guard** and adaptive prefill sized from the device.
- **Anthropic API** (`/v1/messages`, `count_tokens`), **ephemeral endpoints** that bypass the
  session cache, and the **sidecar** port.
- Optional **MoE expert cache** with frequency-profile pinning.
- `.env.example` documenting every setting, verified against `config.py` by a test.

### Changed
- Single server module `supermlx/server.py`; helpers live in `supermlx/components/`.
- All configuration is read in `supermlx/config.py`; no module reads the environment directly
  (except the Hugging Face `HF_*` variables).
- Compaction requests share the global `DEFAULT_MIN_P`, `DEFAULT_TOP_P` and
  `COMPACTION_TEMPERATURE` instead of keeping their own defaults.
- Dependencies trimmed to what the code imports; `torch` and `pyobjc` moved to extras.

### Removed
- LiteLLM proxy (the server is reached directly on `MLX_PORT`), cascade routing,
  the embedded-agent / COMPACT cache slot, the pipeline monitor dashboard.
- Domain-adaptive temperature and the PRECISE/EXPLAIN/EXPLORE profiles (kept as archived ideas).

### Known limitations
- Vision-language models are not operational yet.
- One request at a time (no continuous batching).

---

## [1.4.1] — 2026-05-07

### Fixed
- **CRITICAL:** `_assert_cache_key_safety` was deleted during modularization but still
  called at runtime — would crash with `NameError` if `CACHE_NORM_SAFETY_CHECK=true`.
  Migrated to `message_pipeline.py` with parameterized `log_fn`.
- Orphaned `@dataclass(frozen=True)` decorator leaked onto `SessionIndex` class after
  `SessionContext` extraction, causing `FrozenInstanceError` on startup.

### Changed
- **Modularization Phase 3:** Extracted ~750 lines of core pipeline logic from the
  monolithic `SuperMLX.py` into three independent modules:
  - `config.py` — Environment helpers + `Settings` dataclass (pure, no side effects).
  - `tool_parsing.py` — Regex patterns, `<think>` extraction, OpenAI tool-call parsing.
  - `message_pipeline.py` — Canonicalization, healing, loop breaker, detection,
    session context extraction, cache key safety check.
- All extracted functions are pure (input → output). Global stores (`HEALING_STORE`,
  `_terminal_status`) are passed as parameters to maintain module independence.
- `SuperMLX.py` reduced from ~5800 to ~4891 lines (~16% reduction).
- **Deduplication:** Extracted 3 shared helpers replacing 6 copy-pasted blocks
  between streaming and non-streaming paths:
  - `_start_prefill_progress()` — replaces 2× ~30-line inline closures.
  - `_update_healing_store()` — replaces 2× ~8-line inline blocks.
  - `_post_generation_cache_update()` — replaces 2× ~40-line cache insertion blocks.
  - `_build_timing_dict()` — replaces 2× ~15-line inline timing dict construction.
  - `_log_generation_telemetry()` — replaces 2× ~33-line inline GEN/RESP/MSG_OUT logging.
- **Module extraction:** `_debug_token_divergence()` moved to `debug_tools.py`;
  `_hoist_system_messages()` moved to `message_pipeline.py` (was nested, recreated
  per request).

### Removed
- `_ANSI_RED` — defined but never referenced (only `_ANSI_YELLOW` was used).
- `_vlm_diagnostics` — thread-local never written to, always returned `False`.
- `_messages_have_images()` — 14-line function with zero call sites.
- `_warmup_load_cache()` — backward-compat wrapper with zero callers post-modularization.
- `_mem_snapshot()` — empty function (`pass`) called 4 times in cache reaper; dead telemetry.
- 17 consecutive blank lines (sed extraction residue).
- Orphaned comment referencing deleted Phase 6 function.

---

## [1.4.0] — 2026-05-07

### Added
- **Post-reaper cache reload:** When the background reaper prunes expired KV cache
  entries after a period of inactivity, the next incoming request now triggers an
  automatic reload from disk (`warmup_cache.safetensors`) via a background thread.
  Eliminates the permanent cold-start state that occurred after overnight idle.
  Uses double-check locking to prevent concurrent requests from triggering
  multiple simultaneous reload threads.
- **`CACHE_REAPER_INTERVAL_SECONDS` env var:** Reaper interval is now configurable
  (default: 60s). Enables short-cycle testing without modifying code
  (`CACHE_REAPER_INTERVAL_SECONDS=30 + PROMPT_CACHE_TTL_SECONDS=120` → reaper
  fires in ~2 minutes for validation).

### Fixed
- Cache reaper no longer leaves the inference pipeline in a permanent cold-start
  state after idle eviction. Previously, expired entries were pruned but never
  reloaded, forcing every subsequent request through a full 30s prefill until
  the next server restart.

---

## [1.3.0] — 2026-04

### Added
- **Kripper Dual-Slot KV Cache:** Two isolated LRU stores — `PROMPT_CACHE` (main
  agent) and `PROMPT_CACHE_COMPACT` (compaction/embedded agent). Eliminates
  cross-agent cache eviction: the compactor running in background never displaces
  the main agent's warm cache.
- **Compact runner detector:** Multi-signal detection (`_detect_compact_runner`)
  using tool schema count, system prompt keyword matching, and token volume
  heuristics. Routes compaction requests to the COMPACT slot transparently.
- **Dynamic Prefix Capture (DPC):** Replaces manual seed files. Auto-captures the
  system prompt KV state from the first real request, validates with SHA-256 hash,
  persists to disk (`safetensors` format), and restores across server restarts.
  Reduces cold-start TTFT from ~32s to <2s. Self-heals if system prompt changes.
- **Frozen cache snapshot:** Separate read-only cache snapshot for post-generation
  pollution recovery. Auto-updates with prompt-only KV state after cold starts.
- **Per-layer trim (auto-save):** Deep-copy + selective trim of KVCache layers only
  (preserving ArraysCache layers) before disk saves. Required for hybrid
  architectures like Qwen3.5 that mix softmax and linear attention layers.
- **Native KV quantization:** Optional per-session KV cache quantization (4-bit / 8-bit)
  via `KV_BITS` env var. Uses MLX built-in quantized cache implementation.
  Reduces memory footprint at the cost of minor quality degradation.
- **Tool Call Loop Breaker:** Three detection modes — error loops (model retries a
  failed tool call), duplicate calls (identical tool+args consecutive), and spam
  calls (same tool repeated N times). Injects a stop instruction into the last
  tool result to break infinite retry cycles.
- **Sidecar endpoint (port 8081):** Lightweight OpenAI-compatible endpoint for
  non-agentic use (trading sensors, scripts). Shares the loaded model (zero extra
  RAM), uses ephemeral KV cache per request, optional RAG enrichment.
  Configurable via `SIDECAR_PORT`, `SIDECAR_MAX_TOKENS`, `SIDECAR_ENABLE_RAG`.
- **Emergency content compression:** LLMLingua-2 last-resort compression for
  prompts exceeding `MAX_SAFE_PREFILL_TOKENS`. Operates independently of the
  standard compressor pipeline.
- **Min-suffix pollution wash:** Detects response token pollution in the KV cache
  on cache hits (canonical suffix mismatch). Re-prefills ≥256 tokens to flush
  polluted state before generation. Prevents model attention degradation.
- **VLM support:** Optional vision model support via `mlx-vlm`. Auto-detected from
  `config.json` model type. Bypass with `FORCE_TEXT_MODE=true`.
- **Cascade routing:** Forward requests to a frontier API (OpenAI, Gemini) when
  local RAG confidence is below threshold. Configurable via `FEATURE_CASCADE`,
  `CASCADE_API_URL`, `CASCADE_API_KEY`, `CASCADE_MODEL`.

### Changed
- Memory Guard threshold default changed from fixed 80% to `total_ram - 8GB`
  (more accurate for Apple Silicon where macOS + apps consume ~5-7GB permanently).

---

## [1.2.0] — 2026-03

### Added
- **Cache Canonicalization Pipeline:** Dual-pipeline architecture where the model
  receives the original prompt (correct output) while cache lookups use a
  canonicalized version where volatile fields (timestamps, message IDs, skill
  blocks, runtime metadata) are masked to stable sentinels.
  Achieves 97%+ KV cache hit rates where naive implementations average 50%.
- **Session-aware routing:** `SessionIndex` tracks per-session turn history for
  stable prefix computation across multi-turn agentic conversations.
- **Block-hash prefix index:** SHA-256 block-chain index over token sequences
  enables fuzzy prefix matching for imprecise cache hits (shorter/longer).
- **Compression cache:** Per-session caching of LLMLingua-2 compressed output.
  Ensures canonical tokens remain stable across requests within a session,
  preventing cache invalidation from non-deterministic compression output.
- **RAG codebase enrichment:** Optional LanceDB + sentence-transformers pipeline
  for injecting relevant codebase chunks into request context. Configurable via
  `FEATURE_RAG_ENRICHMENT`, `RAG_WORKSPACE_ROOT`.
- **Prompt compressor:** LLMLingua-2 + reranker pipeline for long conversation
  histories. Configurable via `FEATURE_COMPRESSOR`, `COMPRESSION_THRESHOLD`.

---

## [1.0.0] — 2026-02

### Added
- **OpenAI-compatible inference server** on Apple Silicon via MLX backend (`mlx-lm`).
- **LiteLLM reverse proxy** on port 4000: exposes a unified OpenAI-format endpoint
  that routes to the local MLX server. Compatible with any OpenAI client.
- **LRU prompt cache** (`LRUPromptCache`): global cache with configurable max
  entries, TTL, and background reaper. Reuses prefilled KV states across requests
  to eliminate redundant prefill computation.
- **Multi-model family support:** Automatic detection and routing for Qwen3/3.5,
  GLM-4/4.5, Gemma 4, DeepSeek V3/R1, Hermes 3, and generic ChatML models.
  Each family has a dedicated tool-call parser and thinking-tag extractor.
- **Thinking tag stripping:** `<think>...</think>` blocks are stripped from client
  responses. Reasoning tokens are tracked in diagnostics but hidden from output.
- **Memory Guard:** Pre-prefill Metal RAM check that evicts unpinned cache entries
  if GPU memory exceeds the configured threshold. Prevents OOM crashes during
  long agentic sessions.
- **Streaming + non-streaming:** Full SSE streaming and single-response JSON modes
  for all endpoints.
- **Background cache reaper:** Periodic expiration of idle cache entries to release
  Metal GPU buffers even when no requests are active.
