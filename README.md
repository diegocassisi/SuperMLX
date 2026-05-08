<p align="center">
  <img src="assets/banner.png" alt="SuperMLX — Agentic inference for Apple Silicon" width="800">
</p>

# ⚡ SuperMLX — Agentic inference for Apple Silicon

**Make small local models work for agentic AI. No cloud required.**

SuperMLX is an OpenAI-compatible inference server that turns a local model on your Mac into a practical backend for multi-agent frameworks. Built and tested with [OpenClaw](https://github.com/openclaw/openclaw), it works with any framework that speaks the OpenAI API (Claude Code, Cursor, LangChain, etc.). It eliminates the infrastructure overhead that makes local inference unusable: 30+ second waiting.., cache misses on every turn, memory exhaustion, and infinite tool-call loops — making it a smart local agent solution.

[Install](#quick-start) · [Features](#features) · [Architecture](#architecture) · [Configuration](#configuration) · [Benchmarks](#benchmarks) · [Pipeline Monitor](#pipeline-monitor) · [Roadmap](#roadmap--v20)

---

## Why This Exists

Frontier APIs prefill 30,000-token prompts in milliseconds. Agentic frameworks were designed for that reality — a typical session injects personality files, tool definitions, memory, and user config before the first message. On Claude, that's invisible. On a local model with 24GB of RAM, it's a **32-second cold start every turn** if you don't manage the KV cache correctly.

SuperMLX's approach: reuse the KV cache aggressively. When a 13,000-token prompt grows by 200 tokens on the next turn, only the 200 new tokens get prefilled — dropping time-to-first-token from 32s to under 3s. The hard part is making cache reuse survive the chaos that agentic frameworks create: sub-agent compaction, volatile metadata, idle eviction, and retry storms.

| Problem | Other servers | SuperMLX |
|---------|:------------:|:---------|
| KV cache survives sub-agent compaction | ❌ | ✅ Dual-Slot |
| Auto-persists & restores warm cache across restarts | ❌ | ✅ DPC |
| Cache stable despite volatile metadata every turn | ❌ | ✅ Canonicalization |
| Breaks infinite tool-call retry loops | ❌ | ✅ Loop Breaker |
| Recovers from idle cache eviction automatically | ❌ | ✅ Post-Reaper Reload |

---

## Quick Start

```bash
# Install
pip install supermlx

# Configure
cp .env.example .env
# Edit .env: set MODEL_PATH, adjust MEMORY_GUARD_THRESHOLD_GB for your RAM

# Run
supermlx
```

Or from source:

```bash
git clone https://github.com/diegocassisi/SuperMLX.git
cd SuperMLX
pip install -e .
supermlx
```

Three endpoints start automatically:

| Endpoint | Port | Purpose |
|----------|:----:|---------|
| **LiteLLM Proxy** | 4000 | Point your agentic framework here (OpenAI-compatible) |
| **MLX Direct** | 8080 | Raw MLX engine (used by the proxy internally) |
| **Sidecar** | 8081 | Lightweight endpoint for scripts,  and other needs (ephemeral cache, no session tracking) |

---

## Features

### Kripper Dual-Slot KV Cache

Two isolated LRU stores — one for the main agent, one for compaction sub-agents. When your framework runs a background compaction, it uses the COMPACT slot. The main agent's warm cache is never evicted.

### Dynamic Prefix Capture (DPC)

No manual seed files. Auto-captures the system prompt KV state from the first real request, validates with SHA-256, and persists to disk. On restart, the cache reloads in <2s instead of a 32s cold prefill.

### Cache Canonicalization

The model sees the original prompt. Cache lookups use a version where volatile fields (timestamps, message IDs, billing headers) are masked to stable sentinels. **97%+ cache hit rates** where naive implementations average 50%.

### Tool Call Loop Breaker

Three detection modes — error loops, duplicate calls, and spam calls. Injects a stop instruction to break infinite retry cycles. Solves a universal pain point in agentic AI.

### RAG Enrichment *(optional)*

Injects relevant codebase context via LanceDB vector search before generation. Increases prompt length but meaningfully improves output on code tasks where the model lacks project-specific knowledge. Configurable, bypassable per-request.

### Cascade Routing *(experimental)*

Allows the local model to delegate to a frontier API (Gemini, Claude, etc.) when RAG confidence is low. The idea: a 9B model handles 80% of tasks at zero API cost; the remaining 20% get forwarded to a larger model. Triggering logic and cost-aware routing are areas for contributors.

### Post-Reaper Cache Reload

Background reaper prunes expired cache entries after idle time. Next request triggers automatic reload from disk via double-check locking. Eliminates permanent cold-start degradation after overnight idle.

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

All via `.env`. See [`.env.example`](.env.example) for the full reference.

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `mlx-community/Qwen3.5-9B-4bit` | HuggingFace model ID or local path |
| `FORCE_TEXT_MODE` | `false` | Skip VLM detection for text-only models |
| `MAX_KV_SIZE` | `196608` | Max tokens in KV cache per session |
| `MEMORY_GUARD_THRESHOLD_GB` | `total_ram - 8` | GPU RAM threshold for cache eviction |
| `CACHE_PERSIST_PATH` | `""` | DPC disk path (e.g. `logs/warmup_cache.safetensors`) |
| `PROMPT_CACHE_TTL_SECONDS` | `1800` | TTL before reaper prunes cache entries |
| `FEATURE_RAG_ENRICHMENT` | `false` | LanceDB codebase context injection |
| `FEATURE_COMPRESSOR` | `false` | LLMLingua-2 prompt compression |
| `FEATURE_CASCADE` | `false` | Frontier API fallback routing |
| `TOOL_LOOP_BREAKER` | `true` | Break infinite tool-call loops |
| `KV_BITS` | `None` | Native KV quantization (4 or 8 bit) |

---

## Multi-Model Support

Auto-detects model family from `MODEL_PATH` and applies the correct tool-call parser:

| Family | Models | Thinking | Status |
|--------|--------|:--------:|:------:|
| `qwen3` | Qwen 3 / 3.5 | `<think>` | ✅ Production-tested |
| `hermes` | Hermes 3 | `<think>` | ⚠️ Untested |
| `glm4` | GLM-4 / 4.5 / 4.7 | `<think>` | ⚠️ Untested |
| `gemma4` | Gemma 4 | `<\|think\|>` | ⚠️ Untested |
| `deepseek` | DeepSeek V3 / R1 | `<think>` | ⚠️ Untested |

> Models marked ⚠️ have dedicated parsers but are not validated in production. Community testing welcome.

---

## Pipeline Monitor

Real-time Streamlit dashboard — reads structured logs, no server modifications needed.

```bash
pip install supermlx[monitor]
streamlit run supermlx/pipeline_monitor.py -- --logs-dir ./logs
```

**Dashboard includes:**
- 🎯 Cache hit gauge with session-level benchmark (time saved, speedup ×)
- ⏱️ Per-request prefill/decode timing and token breakdown
- 🔧 Tool call extraction and loop detection events
- 💬 Full message history with role-colored rendering
- 💾 System RAM and Metal GPU memory metrics

---

## Benchmarks

Mac Mini M4 Pro (24GB), Qwen3.5-9B-4bit:

| Metric | Cold Start | Warm Cache |
|--------|:----------:|:----------:|
| **TTFT** | ~32s | ~1-3s |
| **Decode speed** | 38-44 tok/s | 38-44 tok/s |
| **Cache hit rate** | 0% | 97%+ |
| **Prefill throughput** | 300-330 tok/s | N/A (cached) |

---

## Memory Requirements

| RAM | Model | Cache Slots | Config Needed |
|:---:|-------|:-----------:|---------------|
| **16 GB** | Qwen3.5-9B-4bit | 1 MAIN | `PROMPT_CACHE_MAX_ENTRIES_GLOBAL=1`, `MEMORY_GUARD_THRESHOLD_GB=10.5` |
| **24 GB** | Qwen3.5-9B-4bit | 2 MAIN + 1 COMPACT | Works out of the box |

Memory Guard auto-evicts cache entries before Metal runs out. Set `MEMORY_GUARD_THRESHOLD_GB` for your machine and it handles the rest.

---

## Tested Environment

SuperMLX has been developed and tested on:

- **Hardware:** Mac Mini M4 Pro, 24GB unified memory
- **Model:** `mlx-community/Qwen3.5-9B-4bit` (~5GB)
- **Framework:** [OpenClaw](https://github.com/openclaw/openclaw)

Other model families have dedicated parsers but have **not been validated in production**. Community reports welcome.

> **Tip:** Agentic frameworks inject large system prompts (12,000+ tokens) designed for frontier models. On local models, invest time trimming these files to what the model actually needs. A leaner prompt improves both speed and output quality.

---

## Project Structure

| Module | Purpose |
|--------|---------|
| `supermlx/server.py` | Main server: HTTP handler, cache management, generation pipeline |
| `supermlx/config.py` | Environment helpers + `Settings` dataclass |
| `supermlx/tool_parsing.py` | Tool-call parsing, `<think>` extraction |
| `supermlx/message_pipeline.py` | Canonicalization, healing, loop breaker, session context |
| `supermlx/warmup_manager.py` | DPC: disk persistence, hash validation, startup reload |
| `supermlx/rag_enricher.py` | RAG enrichment + LLMLingua-2 compression |
| `supermlx/emergency_compressor.py` | Last-resort OOM compression |
| `supermlx/debug_tools.py` | Cache divergence diagnostics |
| `supermlx/pipeline_monitor.py` | Streamlit observability dashboard |

<details>
<summary><strong>Why <code>server.py</code> is large (~4,900 lines)</strong></summary>

In Architectural reality of GPU-bound servers GPU memory is a shared physical resource that doesn't modularize. The KV cache, model weights, and tokenizer live in Metal GPU buffers. Every component that touches generation needs direct access to the same `mx` runtime, the same model singleton, and the same `prompt_cache_lock` mutex.

</details>

---

## Roadmap — v2.0

v1.x works and is stable. These are aspirational improvements — contributions welcome.

| Feature | Description |
|---------|-------------|
| **Pipeline Architecture** | Staged pipeline: `Adapter IN → Pipeline → Adapter OUT`. `RequestContext` dataclass — zero globals, testable per stage. |
| **Multi-API Adapters** | Pluggable OpenAI / Anthropic / future format adapters. |
| **Continuous Batching** | Concurrent GPU generation via `mlx-lm.BatchGenerator`. |
| **Paged KV Cache** | Block-based with CoW and prefix sharing. |
| **Structured Output** | JSON Schema-constrained sampling (grammar-based). |
| **Multi-Model Serving** | LRU eviction + pinning + per-model TTL. |
| **Context Scaling** | Scaled token counts for agentic auto-compact timing. |
---


## License

[MIT](LICENSE) — Copyright (c) 2026 Diego Cassisi
