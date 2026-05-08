# SuperMLX

**Production-grade MLX inference server for agentic AI on Apple Silicon.**

SuperMLX is an OpenAI-compatible inference server built specifically for multi-agent AI workflows running on Apple Silicon. While other servers handle 1:1 chat well, they break down when agentic frameworks (OpenClaw, Claude Code, Cursor) run sub-agents, compact memory, and retry failed tool calls — all hitting the same model concurrently.

SuperMLX solves the 5 problems that make local agentic AI unusable:

| Problem | mlx-lm | Ollama | llama.cpp | SuperMLX |
|---------|:------:|:------:|:---------:|:--------:|
| KV cache survives sub-agent compaction | ❌ | ❌ | ❌ | ✅ Dual-Slot |
| Auto-persists & restores warm cache | ❌ | ❌ | ❌ | ✅ DPC |
| Cache stable despite volatile metadata | ❌ | ❌ | ❌ | ✅ Canonicalization |
| Breaks infinite tool-call loops | ❌ | ❌ | ❌ | ✅ Loop Breaker |
| Recovers from idle cache eviction | ❌ | ❌ | ❌ | ✅ Post-Reaper Reload |

---

## Quick Start

```bash
# Clone and install
git clone https://github.com/YOUR_USER/SuperMLX.git
cd SuperMLX
pip install -r requirements.txt

# Copy and configure
cp .env.example .env
# Edit .env: set MODEL_PATH, adjust MEMORY_GUARD_THRESHOLD_GB for your RAM

# Run
python SuperMLX.py
```

Three endpoints start automatically:

| Endpoint | Port | Purpose |
|----------|:----:|---------|
| **LiteLLM Proxy** | 4000 | Point your agentic framework here (OpenAI-compatible) |
| **MLX Direct** | 8080 | Raw MLX endpoint (used by the proxy internally) |
| **Sidecar** | 8081 | Lightweight endpoint for scripts/sensors (ephemeral cache) |

---

## Key Innovations

### 1. Kripper Dual-Slot KV Cache

Two isolated LRU stores — `PROMPT_CACHE` (main agent) and `PROMPT_CACHE_COMPACT` (compaction agent). When your framework runs a background compaction agent, it uses the COMPACT slot. The main agent's warm cache is never evicted.

### 2. Dynamic Prefix Capture (DPC)

No manual seed files. SuperMLX auto-captures the system prompt KV state from the first real request, validates it with a SHA-256 hash, and persists to disk. On restart (or after idle eviction), the cache reloads in <2 seconds instead of a 32-second cold prefill.

### 3. Cache Canonicalization

Dual-pipeline architecture: the model sees the original prompt (correct output), but cache lookups use a canonicalized version where volatile fields (timestamps, message IDs, runtime metadata) are masked to stable sentinels. Achieves **97%+ cache hit rates** where naive implementations average 50%.

### 4. Tool Call Loop Breaker

Three detection modes — error loops (model retries a failed tool call), duplicate calls (identical tool+args), and spam calls (same tool repeated N times). Injects a stop instruction to break infinite retry cycles. Solves a universal pain point in agentic AI.

### 5. Post-Reaper Cache Reload *(v1.4.0)*

When the background reaper prunes expired cache entries after idle time, the next request triggers an automatic reload from disk via a background thread. Uses double-check locking to prevent concurrent reload races. Eliminates permanent cold-start degradation after overnight idle.

---

## Architecture

```
OpenClaw / Claude Code ──→ LiteLLM Proxy :4000 ──→ MLX Engine :8080
                                                      │
                              _detect_compact_runner()
                               │              │
                         MAIN path       COMPACT path
                               │              │
                        PROMPT_CACHE    PROMPT_CACHE_COMPACT
                         (LRU max=2)      (LRU max=1)
                               │              │
                        ┌──────┴──────┐       │
                        │Memory Guard │       │
                        │(Metal RAM)  │       │
                        └──────┬──────┘       │
                               ↓              ↓
                            stream_generate(model, prompt_cache=...)
                                               ▲
  External tools ──→ Sidecar :8081 ────────────┘
                     (ephemeral cache, optional RAG, same model_lock)
```

---

## Configuration

All configuration is via environment variables (`.env` file). See [`.env.example`](.env.example) for the full reference.

### Essential Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `mlx-community/Qwen3.5-9B-4bit` | HuggingFace model ID or local path |
| `FORCE_TEXT_MODE` | `false` | Skip VLM detection for text-only models (faster load) |
| `MAX_KV_SIZE` | `196608` | Max tokens in KV cache per session |
| `MEMORY_GUARD_THRESHOLD_GB` | `total_ram - 8` | GPU RAM threshold for cache eviction (0 = disabled) |
| `CACHE_PERSIST_PATH` | `""` | Path for DPC disk persistence (e.g. `logs/warmup_cache.safetensors`) |
| `PROMPT_CACHE_TTL_SECONDS` | `1800` | Cache entry TTL before reaper prunes it |
| `CACHE_REAPER_INTERVAL_SECONDS` | `60` | How often the reaper checks for expired entries |

### Feature Flags

| Variable | Default | Description |
|----------|---------|-------------|
| `FEATURE_RAG_ENRICHMENT` | `false` | Inject relevant codebase chunks via LanceDB |
| `FEATURE_COMPRESSOR` | `false` | LLMLingua-2 prompt compression for long histories |
| `EMERGENCY_CONTENT_COMPRESS` | `true` | Last-resort OOM compression defense |
| `TOOL_LOOP_BREAKER` | `true` | Detect and break infinite tool-call loops |
| `FEATURE_CASCADE` | `false` | Forward to frontier API on low RAG confidence |
| `KV_BITS` | `None` | Native KV quantization (4 or 8 bit) |

---

## Multi-Model Support

SuperMLX auto-detects the model family from `MODEL_PATH` and applies the correct tool-call parser and thinking-tag format:

| Family | Models | Tool Call Format | Thinking | Status |
|--------|--------|:----------------:|:--------:|:------:|
| `qwen3` | Qwen 3/3.5 | `<tool_call><function=>` | `<think>` | ✅ Production-tested |
| `hermes` | Hermes 3 | `<tool_call>` + JSON | `<think>` | ⚠️ Not tested |
| `glm4` | GLM-4/4.5/4.7 | `<tool_call>` + JSON | `<think>` | ⚠️ Not tested |
| `gemma4` | Gemma 4 | `<\|tool_call\|>` + JSON | `<\|think\|>` | ⚠️ Not tested |
| `deepseek` | DeepSeek V3/R1 | JSON (via template) | `<think>` | ⚠️ Not tested |

> Models marked ⚠️ have dedicated parsers but are not yet validated in production.
> Community reports welcome via [Discussions](../../discussions).

---

## Memory Budget (24GB M4 Pro)

```
Model Qwen3.5-9B-4bit:   ~5.0 GB
2 MAIN KV entries:        ~9.0 GB  (2 × 4.5GB)
1 COMPACT KV entry:       ~4.5 GB
Scratch prefill:          ~5.0 GB
──────────────────────────────────
Total peak:               ~23.5 GB → safe with Memory Guard at 19.2GB
Concurrent agents:        1-2 with warm cache
```

For 12GB machines, use `PROMPT_CACHE_MAX_ENTRIES_GLOBAL=1` and `MEMORY_GUARD_THRESHOLD_GB=10.5`.

---

## Modules

| File | Purpose |
|------|---------| 
| `SuperMLX.py` | Main server: HTTP handler, cache management, generation pipeline |
| `config.py` | Environment helpers + `Settings` dataclass (pure, no side effects) |
| `tool_parsing.py` | Regex patterns, `<think>` extraction, OpenAI tool-call parsing |
| `message_pipeline.py` | Canonicalization, healing, loop breaker, detection, session context |
| `debug_tools.py` | Cache divergence analysis, token inspection diagnostics |
| `warmup_manager.py` | Dynamic Prefix Capture: disk persistence, hash validation, startup reload |
| `rag_enricher.py` | RAG enrichment + LLMLingua-2 compression pipeline |
| `emergency_compressor.py` | Last-resort content compression for OOM prevention |
| `pipeline_monitor.py` | Streamlit real-time observability dashboard |

### Why `SuperMLX.py` is large

A typical web server can split cleanly into independent modules. An MLX inference
server cannot — because **GPU memory is a shared physical resource** that doesn't
modularize.

The KV cache, model weights, and tokenizer live in Metal GPU buffers managed by
`mx`. Every component that touches generation — the HTTP handler, cache lookup,
memory guard, prefill, decode, post-generation cache update — needs direct access
to the same `mx` runtime, the same `model` singleton, and the same `prompt_cache_lock`
mutex. This creates a "gravity well" where core logic is pulled toward the center.

**What we extracted** (~1,400 lines across 4 modules): everything that is *pure* —
config parsing, regex patterns, message canonicalization, healing, loop detection,
debug diagnostics. These functions take inputs and return outputs with no GPU state.

**What stays in `SuperMLX.py`** (~4,900 lines): everything coupled to GPU state —
`LRUPromptCache` (calls `mx.clear_cache()`), the generation loop, the HTTP handler,
the cache reaper thread, and the memory guard. Extracting these would require either
circular imports or passing 10+ parameters through every call, trading real complexity
for cosmetic file splitting.

This is not a limitation to fix — it's an architectural reality of GPU-bound servers.
Other MLX/llama.cpp servers with comparable features have similar structure.

### Pipeline Monitor

Real-time dashboard that reads SuperMLX's structured logs — no server modifications needed.

```bash
# Install monitor dependencies
pip install streamlit plotly psutil

# Run (auto-detects ./logs in the same directory)
streamlit run pipeline_monitor.py

# Or point to a specific logs directory
streamlit run pipeline_monitor.py -- --logs-dir /path/to/logs
```

Shows per-request: cache hit rate, prefill/decode timing, rest tokens, tool calls, thinking blocks, and full message history with role-colored rendering. Includes system RAM and Metal GPU memory metrics.

---

## Benchmarks

Measured on Mac Mini M4 Pro (24GB), Qwen3.5-9B-4bit:

| Metric | Cold Start | Warm Cache |
|--------|:----------:|:----------:|
| **TTFT** | ~32s | ~1-3s |
| **Decode speed** | 41-44 tok/s | 41-44 tok/s |
| **Cache hit rate** | 0% | 97%+ |
| **Prefill throughput** | 300-330 tok/s | N/A (cached) |

## Roadmap — v2.0 (Future Rewrite)

v1.x works and is stable. The items below are architectural improvements that would
require rewriting the core server. They are **aspirational, not committed** — contributions
are welcome.

| Feature | Description |
|---------|-------------|
| **Pipeline Architecture** | Replace the monolithic `do_POST` with a staged pipeline: `Adapter IN → Pipeline(canon → RAG → compress → cache → gen) → Adapter OUT`. Each stage receives a `RequestContext` dataclass — zero globals, testable per stage. |
| **Multi-API Adapters** | Pluggable input/output adapters for OpenAI, Anthropic, and future API formats. The pipeline stays the same; only the request parsing and response formatting change. |
| **Continuous Batching** | Process multiple concurrent requests on the GPU by interleaving token generation across users. Enables true parallel sidecar + main generation. Requires `mlx-lm.BatchGenerator` integration. |
| **Paged KV Cache** | Block-based KV cache with Copy-on-Write and prefix sharing (trie or hash-indexed), replacing the current per-session LRU slots. Better memory utilization for multi-session workloads. |
| **Structured Output** | JSON Schema-constrained generation via grammar-based sampling (e.g. `lm-format-enforcer`). Guarantees valid tool-call JSON without retry loops. |
| **Multi-Model Serving** | LRU model eviction + pinning + per-model TTL. Load multiple models in unified memory, swap on demand. |
| **Context Scaling** | Report scaled token counts so agentic frameworks (Claude Code, OpenClaw) trigger auto-compact at the right timing. |

> These ideas draw inspiration from [omlx](https://github.com/jundot/omlx),
> [vllm-mlx](https://github.com/waybarrios/vllm-mlx), and
> [mlx-openai-server](https://github.com/cubist38/mlx-openai-server) —
> excellent projects solving complementary problems on Apple Silicon.

---

## Acknowledgments

Cache architecture inspired by [openclaw-claude-code-mlx-server](https://github.com/nicobrenner/openclaw-claude-code-mlx-server). All cache canonicalization, Dynamic Prefix Capture, tool-call loop breaking, cascade routing, post-reaper reload, and RAG enrichment are original contributions.

## License

[MIT](LICENSE) — Copyright (c) 2026 Diego Cassisi
