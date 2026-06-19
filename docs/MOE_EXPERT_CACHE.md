# MoE Expert Cache — Design Document

## Problema

SuperMLX necesita correr modelos MoE grandes (Qwen3.6-35B-A3B: 256 experts × 40 layers, activa 8/token) en Apple Silicon con 24 GB de RAM unificada, sirviendo Claude Code que manda prompts de hasta 116K tokens.

Cargar el modelo completo consume 19.5 GB. Con KV cache + generation overhead → OOM.

mlx-moe resuelve esto cargando solo ~8.5 GB (100 experts de 256 por layer). El resto se carga on-demand vía mmap.

## Referencia: cómo funciona mlx-moe

mlx-moe opera en 3 fases. Revisado archivo por archivo:

### Fase 1 — Lazy Load + Module Replacement
**Fuente:** `generate.py:62`, `core.py:32-59`

```
model, tokenizer = mlx_lm.load(model_name, lazy=True)   →  tensores lazy (0 bytes en GPU)
enable_lazy_experts(model, model_path, capacity=100, predictive=True)
mx.eval(model.parameters())   →  ~1.4 GB (solo attention, embeddings, norms)
```

`enable_lazy_experts()` con `predictive=True` llama internamente a `_enable_cached()`:
- Por cada MoE layer, crea un `ExpertCache(capacity)` compartido entre gate/up/down_proj
- Reemplaza cada `QuantizedSwitchLinear` por `CachedQuantizedSwitchLinear`
- El `CachedQuantizedSwitchLinear` NO tiene tensores propios — carga bajo demanda

**Detalle clave**: `mx.eval(model.parameters())` después de `enable_lazy_experts()` solo materializa
los parámetros que NO fueron reemplazados. Los experts quedan como placeholders.

### Fase 2 — Profile o Warmup → Predictive Upgrade
**Fuente:** `generate.py:172-179`, `persistence.py:343-439`, `core.py:175-374`

Dos caminos para poblar el cache:

**Camino A — Profile (frío, sin generación):**
```python
upgrade_from_profile(model, model_path, capacity, profile, pin_top_k=32)
```
1. Lee activation_counts del profile JSON
2. Inyecta frecuencias en los Phase 2 ExpertCache
3. Llama `upgrade_to_predictive()` internamente

**Camino B — Warmup (caliente, genera tokens):**
```python
mlx_lm.generate(model, tokenizer, prompt=prompt, max_tokens=50)
upgrade_to_predictive(model, model_path, capacity)
```

Ambos convergen en `upgrade_to_predictive()` que hace:

#### `upgrade_to_predictive()` — 3 pasadas
**Fuente:** `core.py:175-374`

**Pasada 1**: Harvest — extrae top-C experts del LCP cache de cada layer.
Si hay menos de C experts descubiertos, rellena con filler (experts 0, 1, 2...).

**Pasada 2**: Disk Loading — agrupa loads por shard file para minimizar I/O.
Cada shard se abre una sola vez con `mx.load()` o `SafetensorsMap` (mmap).
Evalúa per-layer (no todo junto) para controlar pico de memoria.

**Pasada 3**: Assembly — por cada layer:
- Crea `PredictiveExpertCache(capacity, num_experts=256)`
- Stacks harvested + loaded experts en tensores `(capacity, out, in)` per-proj
- Construye lookup table GPU: `np.array(256) → mx.array` (expert_id → slot index)
- Construye hit_mask GPU: `mx.array(256)` (1.0 si cacheado, 0.0 si no)
- Instala `PredictiveCachedSwitchLinear` como reemplazo final

### Fase 3 — Forward Pass (zero-eval)
**Fuente:** `modules.py:404-441`

```python
class PredictiveCachedSwitchLinear:
    def __call__(self, x, indices, sorted_indices=False):
        # Solo up_proj captura indices (evita triple-buffering)
        if self._proj_name == "up_proj":
            self._cache._indices_buffer.append(indices)

        # Remap: expert 187 → slot 42 (o slot 0 si no cacheado)
        local_indices = self._cache.remap(indices)   # lookup[indices], pure GPU

        return mx.gather_qmm(
            x,
            self._cache.weights[self._proj_name],     # (100, out, in) — NO (256, out, in)
            self._cache.scales[self._proj_name],
            self._cache.biases[self._proj_name],
            rhs_indices=local_indices,
            transpose=True, group_size=..., bits=..., mode=...
        )
```

**Características críticas:**
- **Zero mx.eval**: el forward pass entero es lazy — no hay sync points internos
- **Fallback a slot 0**: expert no cacheado se remapea silenciosamente a slot 0
- **Tensor compacto**: `gather_qmm` opera sobre `(100, out, in)` no `(256, out, in)` → 2.5x menos VRAM
- **Solo up_proj captura**: SwitchGLU llama up_proj primero, luego gate_proj, luego down_proj. Capturar solo en uno evita buffer triplicado.

### Skip-Fallback
**Fuente:** `core.py:787-861`

Monkey-patches cada MoE block para que los experts no cacheados no contaminen:
```python
def patched_call(self, x):
    inds, scores = gate(x)          # router scores
    mask = cache.hit_mask[inds]     # 1.0 si cacheado, 0.0 si no
    scores = scores * mask          # zero out missing expert scores
    score_sum = scores.sum(axis=-1, keepdims=True)
    scores = mx.where(score_sum > 0, scores / score_sum, scores)  # safe renormalize
    y = switch_mlp(x, inds)
    y = (y * scores[..., None]).sum(axis=-2)
```

**CRÍTICO**: el `mx.where` guard evita NaN cuando todos los top-K experts de un token no están
en cache (score_sum=0). Sin esto, div-by-zero → NaN → corrupción acumulativa.

Sin esto, el slot 0 (fallback) recibe el peso del score del expert real → contamina la salida.
Con skip-fallback, el expert missing contribuye 0 → degradación limpia vía residual connection.

### Dynamic Cache Update (entre tokens)
**Fuente:** `core.py:592-643`, `server.py:420-453`

Se llama cada N tokens (N decrece con fallback rate):
```python
stats = dynamic_cache_update(model, max_layer_updates=budget)
```

1. Lee indices de `_indices_buffer` (skipping el último, que está in-flight por async_eval)
2. Calcula LCP priority: `frequency × 0.25^(recency/128)`
3. Hasta 10 swaps/layer: cold expert sale, hot expert entra
4. Carga nuevo expert desde SafetensorsMap (mmap byte-level slice)
5. Actualiza tensor stacked in-place: `weights[proj][slot] = new_weight`
6. Rebuild lookup + hit_mask + `mx.eval(lookup, hit_mask)`
7. `mx.clear_cache()` para liberar tensores viejos

**Política adaptiva** (server.py:320-348):
- Tokens 1-5: update cada token, budget=48
- Tokens 5-20: cada 2 tokens, budget=24
- Fallback < 8%: cada 12 tokens, budget=8
- Fallback < 5%: cada 20 tokens, budget=8

### SafetensorsMap (mmap zero-copy)
**Fuente:** `loading.py:252-349`

Implementación propia (NO usa `safetensors.safe_open()`):
- Parsea headers manualmente: `struct.unpack("<Q", ...)` + `json.loads(header)`
- `mmap.mmap(fd, 0, access=ACCESS_READ)` — read-only, OS page cache
- `get_tensor(key)`: numpy frombuffer del mmap slice → `mx.array(np_arr)`
- `get_expert_slices(key, expert_ids)`: byte-level slicing por expert (solo lee los bytes exactos)
- **bfloat16**: loads como uint16 en numpy, luego `result.view(mx.bfloat16)` en MLX

**Esto explica por qué nuestro SafetensorsMap crasheó**: usábamos `safetensors.safe_open(framework="mlx")` que no maneja bfloat16 correctamente. mlx-moe lo bypasea parseando el header manualmente.

### Persistence (warm restart)
**Fuente:** `persistence.py`

- `save_cache_state()`: guarda cached_ids, frequency, last_active per-layer → JSON
- `save_prepacked_weights()`: guarda tensores stacked `(cap, out, in)` → safetensors
- `load_prepacked_weights()`: carga tensores directos, instala PredictiveCache
- Startup con prepacked: 2s vs 15s sin prepacked

### Capacity Auto-Selection
**Fuente:** `loading.py:383-402`

```python
target_gb = recommended_gb * 0.71   # 71% del max_recommended_working_set_size
slot_gb = num_moe_layers * expert_slot_mb / 1024
capacity = int((target_gb - base_memory_gb) / slot_gb)
```
- Target: 71% de Metal recommended limit (el "pressure cliff" empieza a 75%)
- Alinea a múltiplos de 8 si ≥ 16
- Deja headroom para KV cache growth

---

## Diseño para SuperMLX

### Principios

1. **Reutilizar la API de mlx_lm** — `load(lazy=True)` + `stream_generate()` siguen siendo los entry points
2. **Un solo módulo** — `expert_cache.py` (se reescribe con las técnicas correctas)
3. **Integración mínima en server.py** — model load, skip-fallback hook, post-token dynamic update
4. **Compatibilidad total con dense models** — no-op cuando no es MoE
5. **Aprovechar infra existente** — memory_profiler, memory_guard, DPC warmup, session routing
6. **SafetensorsMap manual** — parseo propio de headers, NO `safe_open()` (bfloat16 fix)

### Flujo propuesto

```
SuperMLX startup con MoE:

1. server.py:
   model, tokenizer = load(MODEL_PATH, lazy=True)
   enable_lazy_experts(model, path, capacity, predictive=True)
   mx.eval(model.parameters())                    →  ~1.4 GB
                                                       ↓
2. expert_cache.upgrade_from_profile(model, path, capacity, profile)
   ├─ SafetensorsMap(shards)                       →  mmap manual (bfloat16-safe)
   ├─ Para cada MoE layer:
   │   ├─ PredictiveExpertCache(capacity, 256)
   │   ├─ Cargar experts del profile (byte-level slicing, no full tensor)
   │   ├─ mx.eval(stacked tensors) per-layer
   │   ├─ Build lookup table + hit mask
   │   ├─ PredictiveCachedSwitchLinear(cache)      →  reemplaza CachedQuantizedSwitchLinear
   │   └─ Pin universals
   └─ enable_skip_fallback(model)                  →  monkey-patch MoE blocks
                                                       ↓
3. server.py:  stream_generate() (zero-eval forward pass)
               ↓ (post-token hook, adaptive interval)
4. expert_cache.dynamic_update(model, budget)       →  swap cold↔hot via mmap
```

### Memoria estimada (24 GB machine)

Cada expert de Qwen3.6-35B-A3B tiene ~3.1M params. En 4-bit (0.5 bytes/param): ~1.5 MB/expert.
Con scales + biases: ~1.77 MB/expert (mlx-moe usa exactamente `expert_slot_mb=1.77`).

```
Componente                    GB        Notas
─────────────────────────     ────      ─────
Non-expert params             1.4       attention, embeddings, norms
Expert cache (cap=100)        7.1       100 experts × 40 layers × 3 projs × ~1.77 MB/expert-slot
────────────────────────      ────
Modelo total                  8.5
KV cache (8-bit, 128K tok)    ~2.0      quantized KV
Metal cache + OS              ~5.0
────────────────────────      ────
USADO                        ~15.5
LIBRE                        ~8.5      Suficiente para spikes de generación
```

### Wired Memory

Después del startup, anclar el working set en Metal residency para evitar paging bajo presión:
```python
if hasattr(mx, "set_wired_limit"):
    active = mx.get_active_memory()
    limit = int(mx.device_info()["memory_size"] * 0.75)
    mx.set_wired_limit(min(active, limit))
```
Sin esto, el OS puede paginar los expert tensors bajo presión → throughput collapse.

### Componentes de expert_cache.py (rewrite completo)

#### SafetensorsMap (nuevo, parsing manual)
- Header parsing propio: `struct.unpack("<Q")` + `json.loads`
- mmap read-only
- `get_tensor(key)` → `np.frombuffer(mmap_slice, dtype=np_dtype).reshape(shape)`
- `get_expert_slices(key, expert_ids)` → byte-level per-expert slicing
- bfloat16: load as uint16, `.view(mx.bfloat16)`
- Aliasing automático de `language_model.` prefix

#### PredictiveExpertCache (nuevo, reemplaza ExpertCache)
- `weights/scales/biases`: `Dict[proj_name, mx.array(cap, out, in)]`
- `lookup`: `mx.array(num_experts)` — GPU lookup table
- `hit_mask`: `mx.array(num_experts)` — para skip-fallback
- `cached_ids/cached_set`: tracking de qué experts están en qué slot
- `frequency/last_active/step/pinned_set`: LCP + pinning
- `_indices_buffer`: captura indices durante forward para dynamic update
- `remap(indices)`: `return self.lookup[indices]` (pure GPU, no eval)
- `update()`: procesa buffer, calcula swaps, carga desde SafetensorsMap, rebuild lookup

#### PredictiveCachedSwitchLinear (nuevo, reemplaza CachedSwitchLinear)
- Zero-eval forward: remap + gather_qmm sobre tensores compactos
- Solo up_proj captura indices
- NO tiene `self.weight/self.scales` propios — usa `self._cache.weights[proj]`

#### enable_moe_cache() (modificar)
- Phase 2: instala CachedQuantizedSwitchLinear con ExpertCache
- Si hay profile: inyecta frequencies, llama upgrade_to_predictive
- Si no hay profile: router-only discovery (genera 10 tokens para descubrir)
- Phase 3: upgrade_to_predictive con stacking + lookup

#### enable_skip_fallback() (nuevo)
- Monkey-patch de MoE blocks: zero scores de experts no cacheados + renormalize

#### dynamic_update() (nuevo)
- Wrapper de per-cache update con budget limiting
- Policy adaptiva: fast al inicio, slow cuando fallback < 5%

### server.py — Cambios requeridos

1. **Model load**: `lazy=True` condicional (solo MoE)
2. **Post-load**: `enable_moe_cache()` + `enable_skip_fallback()`
3. **Generate loop**: `dynamic_update()` post-token con policy adaptiva
4. **Memory guard**: ya integrado

### config.py — Sin cambios adicionales
`MOE_EXPERT_CAPACITY` y `MOE_EXPERT_PROFILE` ya existen.

### Diferencias vs mlx-moe

| Aspecto | mlx-moe | SuperMLX (propuesto) |
|---------|---------|---------------------|
| Expert loading | `mx.load(shard)` (materializa full shard) + slice | SafetensorsMap mmap byte-level slice (igual que mlx-moe Phase 3) |
| Dynamic update | En generate loop custom | Hook en `stream_generate` post-token |
| Capacity | Auto (71% recommended) o manual | Config (MOE_EXPERT_CAPACITY) + auto-select posible |
| Persistence | JSON state + prepacked safetensors | Se integra con DPC warmup (futuro) |
| KV cache | 1-2 slots, sin persistence | Dual-slot + canonicalization + persistence + quantization |
| Session routing | No | Multi-session con prompt cache compartido |
| Compressor | No | LLMLingua-2 para prompts > threshold |
| Memory guard | No | Eviction dinámica bajo presión Metal |

### Verificación

1. **Syntax**: `python3 -c "import ast; ast.parse(...)"`
2. **Dense model no-op**: cargar Qwen3.5-9B → `is_moe_model() == False`, sin cambios
3. **MoE lazy load**: cargar Qwen3.6-35B con `lazy=True` → verificar footprint ~1.4 GB
4. **Expert loading**: SafetensorsMap parsea headers, load expert slices sin bfloat16 error
5. **Memory footprint**: con cap=100, modelo total ~8.5 GB (verificar `mx.get_active_memory()`)
6. **Inference**: request simple → respuesta coherente
7. **Skip-fallback**: verificar que scores de experts missing son 0
8. **Telemetría**: hit_rate y fallback_rate en log de fin de request
9. **Dynamic update**: fallback_rate baja progresivamente durante generación
10. **Stability**: 10 requests consecutivos sin OOM ni crash
