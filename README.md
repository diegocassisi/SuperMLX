<p align="center">
  <img src="assets/banner.png" alt="SuperMLX — Agentic inference for Apple Silicon" width="800">
</p>

# SuperMLX

**A local inference server for agentic workloads on Apple Silicon, built on [MLX](https://github.com/ml-explore/mlx).**

Agent frameworks (Claude Code, Hermes, OpenClaw, Cursor, LangChain…) send large, repetitive prompts: system prompt, tool schemas, memory and a long history on every turn. On a local model that means re-prefilling tens of thousands of tokens each time. SuperMLX keeps the KV cache alive across turns, survives the things agents do to break caches (compaction, volatile metadata, retries), and keeps a big model inside the memory of a laptop.

It speaks both the **OpenAI** and the **Anthropic** HTTP APIs, so most clients work by just changing the base URL.

> **Status:** active development, single-user, one request at a time (no continuous batching). Developed and tested mainly with Qwen3.5 / Qwen3.6 models on a 24 GB Mac.

---

## Features

**Cache**
- **Radix prompt cache** — a prefix tree of KV states with a memory budget derived from the Metal working set. A new turn only prefills the tokens that changed.
- **Tool-prefix cache** — the system prompt + tool definitions are precomputed once and persisted to disk.
- **Cache canonicalization** — volatile fields (timestamps, message ids, billing headers) are masked in the cache key, so equivalent prompts hit the same prefix. The model still sees the original prompt.
- **Session-aware lookups** and a warm-up cache that survives restarts.

**Generation**
- **MTP speculative decoding** for models that ship a Multi-Token-Prediction head.
- **Thinking control** — a token budget for `<think>` blocks enforced by logits forcing (the KV cache stays consistent), optional temperature schedule and re-heating "sparks".
- **Task-aware temperature** — the model opens its answer with `[TASK: CODE|TECH|PROSE|CREATIVE]` and the tag selects the temperature; requests with tools and compaction requests have their own.
- **Loop protection** — n-gram repetition detector with an in-context "nudge" before a hard break, and a tool-call loop breaker.
- **Tool calling** — parsing and healing of malformed tool calls for Qwen3, Hermes, GLM-4, Gemma-4 and DeepSeek style templates.

**Memory**
- **Metal memory guard** — sizes the wired limit and cache ceiling from the device, checks memory before each prefill and evicts cache entries instead of crashing.
- **Adaptive prefill** — chunk size adapts to the available memory.
- **MoE expert cache** *(optional)* — predictive loading and pinning of experts for MoE models.

**Integration**
- OpenAI and Anthropic endpoints, streaming and non-streaming, plus separate **ephemeral** endpoints and a **sidecar** port for traffic that must not use the session cache (see [Endpoints](#endpoints)).
- Optional RAG enrichment and prompt compression (LanceDB, LLMLingua-2).

An experimental Apple Neural Engine prefill path exists but is disabled by default.

---

## Requirements

- Apple Silicon Mac (M-series) with macOS
- Python 3.11 – 3.13
- Enough unified memory for your model plus KV cache (a 35B-A3B model at 3-bit runs on 24 GB)

## Quick start

```bash
git clone https://github.com/diegocassisi/SuperMLX.git
cd SuperMLX
python -m venv .venv && source .venv/bin/activate
pip install -e .

cp .env.example .env          # set MODEL_PATH, everything else is optional
python SuperMLX.py            # or: supermlx
```

### Endpoints

All API endpoints are served on the main port (`8080` by default). The sidecar has its own port.

| Endpoint | Tool-prefix cache | Session KV cache | Thinking |
|---|:---:|:---:|:---:|
| **OpenAI** `POST /v1/chat/completions` | ✅ | ✅ | ✅ |
| **OpenAI** `POST /v1/ephemeral/chat/completions` | ✅ | ❌ | ❌ |
| **Anthropic** `POST /v1/messages` | ✅ | ✅ | ✅ |
| **Anthropic** `POST /v1/ephemeral/messages` | ✅ | ❌ | ❌ |
| **Sidecar** `http://host:8081` (OpenAI-style) | ❌ | ❌ | per request |

Also available: `GET /v1/models` and `POST /v1/messages/count_tokens`. Streaming and non-streaming are supported on all of them.

**Main endpoints** are for the agent itself: they use the session KV cache, so each turn only prefills what changed, and they support thinking.

**Ephemeral endpoints** (`/v1/ephemeral/...`) are for everything around the agent that should *not* touch its cache: title generation, summaries, memory and other housekeeping calls. They never read or write the session cache and run with thinking off, so they cannot evict or pollute the main conversation. Because they are separate URLs, you can point those auxiliary services at them independently of the main agent, in any client that lets you set a base URL per service.

**Sidecar** is a lightweight extra port for scripts, sensors and other tools: no session tracking, an ephemeral per-request cache, optional RAG enrichment and its own limits (`SIDECAR_PORT`, `SIDECAR_MAX_TOKENS`, `SIDECAR_ENABLE_RAG`). Set `SIDECAR_PORT=0` to disable it. It shares the loaded model, so it takes turns with the main endpoints.

On startup the server prints this table for the configured host and ports.

### Example: Hermes agent

Hermes runs auxiliary services (title generation, context compression, web extraction…) as separate model calls. Point the agent's main model at `http://127.0.0.1:8080/v1` (session cache + thinking) and every auxiliary service at the ephemeral endpoint, so those side calls never disturb the main conversation's KV cache:

```yaml
auxiliary:
  title_generation:
    provider: custom
    model: local
    base_url: http://127.0.0.1:8080/v1/ephemeral   # no session cache, thinking off
    api_key: "none"
  compression:
    provider: custom
    model: local
    base_url: http://127.0.0.1:8080/v1/ephemeral
    api_key: "none"
  web_extract:
    provider: custom
    model: local
    base_url: http://127.0.0.1:8080/v1/ephemeral
    api_key: "none"
```

Each auxiliary service has its own `base_url`, so you can also send some of them elsewhere, for example a cloud model for `web_extract` and the local ephemeral endpoint for the rest. The same idea works in any client that lets you set a base URL per task.

Quick check:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"local","messages":[{"role":"user","content":"Hello"}],"max_tokens":64}'
```

Using it with Claude Code:

```bash
ANTHROPIC_BASE_URL=http://127.0.0.1:8080 claude
```

### Optional extras

```bash
pip install -e '.[rag]'       # RAG enrichment + prompt compression
pip install -e '.[titles]'    # Apple NaturalLanguage session titles (macOS)
```

---

## Configuration

Everything is configured through environment variables, normally in a `.env` file. Only `MODEL_PATH` is required.

[`.env.example`](.env.example) documents **every** variable — purpose, units and default — grouped by section (model, network, sampling, thinking, cache, memory, MTP, loop protection, RAG, diagnostics). The defaults in that file are checked against `supermlx/config.py` by a test, so it cannot drift.

A few variables you will probably touch first:

| Variable | What it does |
|---|---|
| `MODEL_PATH` | Hugging Face id or local path of the MLX model |
| `MAX_KV_SIZE` | Maximum context held in the KV cache |
| `ENABLE_MTP`, `MTP_WEIGHTS_PATH` | Speculative decoding with the model's MTP head |
| `MAX_THINKING_TOKENS`, `THINKING_BUDGET_MODE` | Reasoning budget and how it is enforced |
| `MEMORY_GUARD_THRESHOLD_GB` | Metal memory level that triggers cache eviction |
| `KV_BITS` | KV cache quantization (`OFF`, 4 or 8) |

## Supported models

| Family | Models | Status |
|---|---|---|
| `qwen3` | Qwen3 / 3.5 / 3.6 (dense and MoE) | Developed and tested |
| `hermes`, `glm4`, `gemma4`, `deepseek` | Corresponding chat/tool templates | Parser support, lightly tested |

The family is inferred from `MODEL_PATH`; override it with `MODEL_FAMILY`.

---

## Project layout

```
SuperMLX.py            launcher (python SuperMLX.py)
supermlx/
  server.py            HTTP server, request handling, wiring
  config.py            all settings — the single source of truth
  components/          pipeline (4 phases), radix cache, LRU cache, sidecar, post-generation...
  mtp/                 MTP speculative decoding (engine + model shim)
  sampling.py          dual-phase sampler, thinking schedule, task detector
  cache_engine.py, tool_prefix_cache.py, expert_cache.py, metal_memory_guard.py ...
tests/                 pytest suites (most run without a GPU)
benchmarks/            benchmark and stress scripts
profiles/              MoE expert-frequency profiles
docs/                  design notes
```

## Tests

```bash
pip install -e '.[dev]'
pytest tests/test_radix_cache.py tests/test_env_example.py     # fast, no model needed
```

Some suites (`*_real_model*`, `scale_radix_17k.py`) need a model and a running server; they are meant for manual validation.

## Design notes

[`docs/`](docs) contains the longer write-ups: [MoE expert cache](docs/MOE_EXPERT_CACHE.md), [expert breathing](docs/breathing_architecture.md) and [tool-calling fixes](docs/TOOL_CALLING_FIXES.md).

## Limitations

- One request at a time: a model lock serializes generation.
- Text-only: vision-language models are not operational yet. The code path exists but is unfinished; set `FORCE_TEXT_MODE=true` to load a VLM checkpoint as text-only.
- macOS / Apple Silicon only.
- Quality of the tool-call parsers outside the Qwen family is not guaranteed.

## License

MIT — see [LICENSE](LICENSE).
