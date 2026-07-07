# Expert Breathing — Arquitectura Completa

Sistema de gestión dinámica de memoria GPU para MoE (Mixture-of-Experts) en SuperMLX.
Contrata y expande expertos en Metal RAM según presión de memoria, con aprendizaje
cross-sesión mediante estadísticas de frecuencia persistidas en disco.

## Configuración (config.py)

```
MOE_EXPERT_CAPACITY=100          # Expertos iniciales por capa (capacidad de arranque)
MOE_TARGET_CAPACITY=0            # 0 = sin expansión post-warmup; N = expandir a N expertos
MOE_EXPERT_PROFILE=""            # JSON de perfil de expertos (activaciones por prompt)
MOE_SHALLOW_PIN_LAYERS=5         # Primeras N capas MoE con pinning de top-k
MOE_SHALLOW_PIN_TOP=6            # Top-k expertos a pinnear por capa superficial
BREATHE_DOWN_REST_THRESHOLD=15000 # Tokens de rest que activan breathe_down
```

## Ciclo de vida completo

```
  ┌─────────────────────────────────────────────────────────────────────────────────┐
  │                        EXPERT BREATHING LIFECYCLE                               │
  └─────────────────────────────────────────────────────────────────────────────────┘

  ┌──────────┐
  │  BOOT     │ enable_moe_cache(capacity=100)
  │  PHASE    │ → detect MoE config (num_experts, moe_layers, slot_mb)
  │           │ → build SafetensorsMap (mmap, bfloat16-safe)
  │           │ → Pass 1: replace QuantizedSwitchLinear → PredictiveCachedSwitchLinear
  │           │ → Pass 2: load initial experts via byte-level mmap
  │           │ → apply_historical_frequency() from logs/expert_stats.json
  │           │ → set model._moe_config["capacity"] = 100
  └─────┬────┘
        │
        ▼
  ┌─────────────────────────────────────────────────────────────────────────────────┐
  │  EXPERT SELECTION (per layer)                                                   │
  │                                                                                 │
  │  _select_experts_for_layer(layer_idx, num_experts, capacity, profile)           │
  │    ├─ profile exists → top-C by activation_counts (from profile JSON)           │
  │    └─ no profile    → sequential 0..C-1                                        │
  │                                                                                 │
  │  _pin_from_profile(cache, layer_idx, profile, threshold=0.5)                    │
  │    └─ Pin experts activated in >50% of profile prompts                          │
  │       → cache.pinned_set = intersection(profile_top & cached_set)              │
  │       → pinned experts NEVER evicted by breathing                               │
  └─────┬────────────────────────────┬──────────────────────────────────────────────┘
        │                            │
        ▼                            ▼
  ┌──────────────────────┐   ┌─────────────────────────────────────────────────────┐
  │ STAGED LOADING       │   │ PER-REQUEST FLOW                                    │
  │ (server.py:2996)     │   │                                                     │
  │ If target > capacity │   │ 1. PRE-PREFILL (server.py:4990, 5241)              │
  │ → wait first response│   │   _pre_prefill_memory_relief(request_id, rest)     │
  │ → then expand_after_ │   │   ├─ gc.collect() + mx.clear_cache()               │
  │   first_response()   │   │   ├─ malloc_zone_pressure_relief (macOS)           │
  │                      │   │   └─ IF rest >= 15000 AND capacity > 100:          │
  │                      │   │      breathe_down(model, target)  ← BREATHE DOWN  │
  └──────────────────────┘   │                                                     │
                             │ 2. DURING GENERATION                                │
                             │   PredictiveCachedSwitchLinear.__call__(x, indices)  │
                             │   ├─ Capture router indices in _indices_buffer       │
                             │   │   (only up_proj, to avoid triple-buffering)     │
                             │   │   indices: [batch, seq, top_k] → expert IDs     │
                             │   │                                                │
                             │   ├─ Remap via GPU lookup table (zero-eval):        │
                             │   │   local_indices = cache.remap(indices)          │
                             │   │   → lookup[global_eid] = cache_slot            │
                             │   │   → uncached → slot 0 (fallback)              │
                             │   │                                                │
                             │   ├─ mx.gather_qmm(x, cached_weights, ...)          │
                             │   │   → uses pre-loaded stacked tensors             │
                             │   │   → NO mx.eval() in forward pass               │
                             │   │                                                │
                             │   └─ PER-TOKEN: dynamic_cache_update(model)         │
                             │       ├─ Process _indices_buffer (skip last: async) │
                             │       ├─ Update frequency counters                  │
                             │       ├─ Find misses = requested - cached_set       │
                             │       ├─ Evict by LCP priority: μ·0.25^(ν/128)     │
                             │       ├─ Prefetch via F_RDADVISE (macOS)           │
                             │       ├─ Load new experts via mmap slices           │
                             │       ├─ Swap in-place: w[slot] = new_w            │
                             │       └─ Rebuild lookup table + hit_mask            │
                             │                                                      │
                             │   dynamic_update_policy(gen_tokens, fallback_rate)    │
                             │   → adaptive interval: 1/2/4 tokens                  │
                             │   → adaptive budget: 48/24/12 swaps/call             │
                             │   → slows down if no swaps needed                     │
                             └──────────────────────┬───────────────────────────────┘
                                                    │
                                                    ▼
  ┌─────────────────────────────────────────────────────────────────────────────────┐
  │  POST-GENERATION (server.py:5100-5109, 5390-5397)                               │
  │                                                                                 │
  │  if _breathing_active:                                                         │
  │    breathe_up(model)  ← BREATHE UP                                             │
  │    └─ Reload evicted experts from SSD via SafetensorsMap                        │
  │       → frequency-ordered selection (most-used first)                           │
  │       → concatenate new weights: old + new                                      │
  │       → rebuild lookup + hit_mask                                               │
  │       → re-wire Metal wired limit                                               │
  │       → model._moe_config["capacity"] = full_capacity (100)                     │
  │       → _breathing_active = False                                               │
  └─────────────────────────────────────────────────────────────────────────────────┘
        │
        ▼
  ┌─────────────────────────────────────────────────────────────────────────────────┐
  │  SHUTDOWN (server.py:5801)                                                      │
  │                                                                                 │
  │  save_frequency_stats(model, "logs/expert_stats.json")                         │
  │    ├─ Extract session_frequency from each layer cache                           │
  │    ├─ Merge additively into existing historical file                            │
  │    └─ Write JSON: {"version": 1, "layers": {"0": {eid: count, ...}, ...}}      │
  │                                                                                 │
  │  → Next boot: apply_historical_frequency() loads this file                     │
  │  → Session frequency resets each session; historical accumulates               │
  └─────────────────────────────────────────────────────────────────────────────────┘
```

## Datos por capa: PredictiveExpertCache

```
┌──────────────────────────────────────────────────────────────────────┐
│                    PredictiveExpertCache                             │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────┐  ┌──────────────────┐  ┌──────────────────┐       │
│  │ capacity      │  │ cached_ids       │  │ cached_set       │       │
│  │ 100 → 64 → 100│  │ [0,1,3,5,...]   │  │ {0,1,3,5,...}   │       │
│  └─────────────┘  └──────────────────┘  └──────────────────┘       │
│                                                                      │
│  Frequency tracking (3 views):                                      │
│  ┌────────────────────┐  ┌──────────────────┐  ┌────────────────┐  │
│  │ session_frequency   │  │ historical_freq  │  │ frequency      │  │
│  │ {eid: count}       │  │ {eid: count}     │  │ session+hist   │  │
│  │ Reset per session  │  │ Persisted to     │  │ = combined     │  │
│  │                    │  │ disk             │  │ = sum(both)    │  │
│  └────────────────────┘  └──────────────────┘  └────────────────┘  │
│                                                                      │
│  ┌─────────────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │ last_active: {eid:  │  │ pinned_set:   │  │ step: int        │   │
│  │   step_number}      │  │ {eid}        │  │ (incremented     │   │
│  │                     │  │ (protected)   │  │  per update)     │   │
│  └─────────────────────┘  └──────────────┘  └──────────────────┘   │
│                                                                      │
│  GPU tensors:                                                       │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ weights: Dict[str, mx.array]  (proj_name → stacked weights) │   │
│  │ scales:  Dict[str, mx.array]  (proj_name → stacked scales)  │   │
│  │ biases:  Dict[str, mx.array]  (proj_name → stacked biases)  │   │
│  │ lookup:  mx.array (num_experts,) → cache slot indices        │   │
│  │ hit_mask: mx.array (num_experts,) → 1.0 if cached, 0.0 else │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  SSD access:                                                        │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ _st_map: SafetensorsMap (mmap of model shards)              │   │
│  │ _key_prefixes: {proj_name: "model.layers.N.mlp.switch_mlp.X"}│
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

## Decision de eviction: LCP (Locality-Count Priority)

```
  LCP(eid) = μ × 0.25^(ν / 128)

  μ = frequency[eid]          (total usage: session + historical)
  ν = step - last_active[eid] (tokens since last use)

  Decay: 0.25^0 = 1.0 (just used)
         0.25^1 = 0.25 (128 tokens ago)
         0.25^2 = 0.0625 (256 tokens ago)
         0.25^3 = 0.016  (384 tokens ago)

  → Recency matters exponentially; frequently-used experts stay hot.
  → Eviction order: lowest LCP first (least-used, oldest).
  → Pinned experts: LCP = +∞ (never evicted).
```

## Breathing Priority (contraction decision)

```
  breathing_priority(eid) = session_frequency[eid] × 3 + historical_frequency[eid] × 1

  Rationale:
  - Session usage × 3: protege expertos activos en la tarea actual
  - Historical × 1:    baseline de uso cross-sesión
  - Never-used: 0 → evicted first
  - Pinned: +∞ → never evicted

  breathe_down target = min across layers of:
    count(eid where breathing_priority(eid) > 0)
  → Keep only experts with SOME usage signal, evict the rest.
  → Floor: 64 experts per layer (never below).
```

## Estados de capacidad

```
  ┌──────────────────────────────────────────────────────────────────┐
  │  State machine:                                                  │
  │                                                                  │
  │  BOOT: capacity = 100 (moe_expert_capacity)                     │
  │    │                                                             │
  │    ├─ If moe_target_capacity > 100:                             │
  │    │   expand_after_first_response() → capacity = target (e.g. 256)
  │    │                                                            │
  │    ├─ PRE-PREFILL: rest_tokens >= 15000 AND capacity > 100:     │
  │    │   breathe_down(target) → capacity = min_used (e.g. 64)    │
  │    │   _breathing_active = True                                 │
  │    │                                                            │
  │    ├─ DURING GENERATION:                                       │
  │    │   dynamic_cache_update() → swap cold→hot per miss         │
  │    │   (capacity stays same, just swaps experts in-place)       │
  │    │                                                            │
  │    └─ POST-GENERATION:                                         │
  │        if _breathing_active:                                    │
  │          breathe_up() → capacity = full_capacity (100 or 256)   │
  │          _breathing_active = False                               │
  │                                                                  │
  │  SHUTDOWN: save_frequency_stats() → persist session_frequency   │
  │    → merged into logs/expert_stats.json                          │
  │    → loaded at next boot via apply_historical_frequency()       │
  └──────────────────────────────────────────────────────────────────┘
```

## Persistencia de frecuencias

```
  File: logs/expert_stats.json

  Structure:
  {
    "version": 1,
    "layers": {
      "0": { "0": 15, "3": 8, "12": 22, ... },
      "1": { "2": 12, "5": 30, "7": 5, ... },
      "2": { "1": 7, "4": 18, ... },
      ...
    }
  }

  Merge strategy (save_frequency_stats):
  - Load existing historical file
  - For each layer: additive merge (existing[eid] += session[eid])
  - Write merged file
  → Historical counts accumulate across sessions without double-counting
  → session_frequency resets each session (fresh start for current task)

  Load strategy (apply_historical_frequency):
  - Load JSON → populate historical_frequency dict per layer
  - Add to frequency dict (combined view = session + historical)
  - Does NOT touch session_frequency (kept separate for breathing priority)
```

## Archivos clave

| Archivo | Función |
|---------|---------|
| `expert_cache.py` | `PredictiveExpertCache`, `breathe_down()`, `breathe_up()`, `dynamic_cache_update()` |
| `server.py` | `_pre_prefill_memory_relief()`, `_breathing_active` flag, `save_frequency_stats()` |
| `config.py` | `moe_expert_capacity`, `moe_target_capacity`, `moe_expert_profile` |
| `logs/expert_stats.json` | Frecuencias históricas persistidas cross-sesión |

## Flujo de datos de un request

```
  Request arrives (rest_count tokens)
    │
    ├─ [rest_count >= 15000 AND capacity > 100]
    │   └─ breathe_down(model, target)
    │       ├─ Evict low-priority experts per layer
    │       ├─ Rebuild stacked tensors (slice to keep)
    │       ├─ Rebuild lookup + hit_mask
    │       ├─ Update capacity (100 → ~64)
    │       └─ _breathing_active = True
    │
    ├─ Generation loop
    │   ├─ PredictiveCachedSwitchLinear.__call__()
    │   │   ├─ Capture indices in _indices_buffer
    │   │   ├─ Remap via GPU lookup table
    │   │   └─ gather_qmm with cached weights
    │   │
    │   └─ dynamic_cache_update() every N tokens
    │       ├─ Process buffer, update frequency
    │       ├─ Swap cold→hot experts (up to 12/call)
    │       └─ Prefetch via F_RDADVISE
    │
    ├─ Generation complete
    │   ├─ save_frequency_stats() → persist to disk
    │   │   └─ Merge session_frequency → historical
    │   │
    │   └─ if _breathing_active:
    │       └─ breathe_up(model)
    │           ├─ Reload evicted experts from SSD
    │           ├─ Concatenate weights (old + new)
    │           ├─ Rebuild lookup + hit_mask
    │           ├─ Restore capacity (64 → 100)
    │           └─ _breathing_active = False
    │
    └─ Next request → repeat
```

## Métricas de diagnóstico

```
  get_cache_stats(model) →
  {
    "moe_active": True,
    "total_requests": 15234,
    "total_fallbacks": 892,
    "hit_rate": 0.9415,
    "fallback_rate": 0.0585
  }

  breathe_down returns:
  {
    "breathed": True,
    "direction": "down",
    "before": 100,
    "after": 64,
    "evicted": 36,
    "freed_gb": 2.3,
    "coverage": 0.92,
    "elapsed_s": 1.4
  }

  breathe_up returns:
  {
    "breathed": True,
    "direction": "up",
    "before": 64,
    "after": 100,
    "loaded": 36,
    "elapsed_s": 3.2,
    "active_memory_gb": 8.5
  }
```