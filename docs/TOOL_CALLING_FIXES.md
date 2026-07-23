# Tool Calling Fixes — Qwen3 Family

> Documentación de fixes aplicados al pipeline de tool calling de SuperMLX
> para modelos Qwen3/Qwen3.5/Qwen3.6 (MoE A3B).
>
> **Fecha**: 2026-07-23
> **Archivos modificados**: `supermlx/tool_parsing.py`, `chat_template.jinja`

---

## Fix 1: Extracción de bare `<function=NAME>` sin `<tool_call>` wrapper

**Commit**: `90f0ab9`
**Archivo**: `supermlx/tool_parsing.py` — líneas 399-413

### Problema

Cuando el modelo Qwen3 genera tool calls directamente sin cerrar `</think>` primero,
el THINK_CLEANUP de SuperMLX inyecta `</think>` sintético. Pero en algunos casos,
el modelo emite la tool call **sin** el wrapper `<tool_call>`:

```
<function=read_file>
<parameter=path>/some/file.js</parameter>
</function>
</tool_call>    ← cierre huérfano
</think>        ← inyectado por THINK_CLEANUP
```

El quick-exit de `_extract_openai_tool_calls()` verificaba si existía `<tool_call>`
(apertura) en el texto. Como solo hay `</tool_call>` (cierre), la verificación fallaba
y la función retornaba sin extraer nada → `tool_calls = []` → enviado como texto plano
al cliente → Hermes recibía `finish_reason=stop` sin tool calls ejecutables.

**Efecto visible**: Hermes mostraba
`↻ Empty response after tool calls — using earlier content as final answer`
y desperdiciaba 2 round-trips (~10-14s) en retries.

### Causa raíz

Documentada en el issue [llama.cpp #15012](https://github.com/ggml-org/llama.cpp/issues/15012):
los modelos Qwen3-Coder a veces omiten `<tool_call>` de apertura, especialmente con
cuantización agresiva (3-bit, 4-bit) y en contextos largos (>40K tokens).

### Solución

Cuando no hay `<tool_call>` ni `<|tool_call|>` pero sí hay `<function=NAME>` válido
(detectado por `QWEN_FUNCTION_PATTERN`), envolver automáticamente los bloques bare
con `<tool_call>...</tool_call>` para que el parser estándar los procese:

```python
if not _has_standard and not _has_gemma4:
    # Qwen3 fallback: model sometimes emits <function=NAME>...</function>
    # without <tool_call> wrapper (e.g. when THINK_CLEANUP fires because
    # the model jumped straight to tool calls without closing </think>).
    if model_family == "qwen3" and QWEN_FUNCTION_PATTERN.search(text):
        text = re.sub(
            r'(<function=[^>]+>.*?</function>)',
            r'<tool_call>\n\1\n</tool_call>',
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )
        _has_standard = True
    else:
        return text, []
```

### Protecciones contra falsos positivos

1. **Solo qwen3**: no afecta otros model families (deepseek, gemma4, hermes, glm4)
2. **Solo sin `<tool_call>`**: si el modelo ya emitió el wrapper, se usa el path normal
3. **`QWEN_FUNCTION_PATTERN`**: requiere `<function=NAME>` con nombre real, no bare `<function>`
4. **`allowed_names` validation**: después del parsing, tool calls con nombres no declarados
   se descartan (líneas 539-543)

### Tests de validación

| Caso | Resultado |
|---|---|
| bare `<function=read_file>` (bug original) | ✅ 1 tool call extraído |
| bare `<function=patch>` (bug original) | ✅ 1 tool call con args correctos |
| Normal con `<tool_call>` wrapper | ✅ Sin regresión |
| Texto sin tool calls | ✅ 0 calls, sin falso positivo |
| `<function=fake_tool>` no declarado | ✅ 0 calls, filtrado por allowed_names |
| deepseek (otro family) | ✅ Fallback no se activa |

---

## Fix 2: Desactivar `preserve_thinking` por defecto en chat template

**Archivo**: `chat_template.jinja` (en cache de HuggingFace del modelo)
**Línea**: 7

### Problema

La plantilla froggeric-v20 preservaba **todos** los bloques de thinking históricos
en el prompt por defecto (`_preserve_thinking = true`). En sesiones largas de Hermes
con ~32 mensajes, cada uno con 500-2000 tokens de thinking, esto agregaba
**10-30K tokens** de razonamiento viejo al prompt.

Efectos:
- **Consumo de contexto**: menos espacio para código/datos reales
- **Confusión del modelo**: razonamiento viejo interfiere con el actual
- **NGRAM loops**: el modelo repite patrones de thinking anteriores
- **Repetición de tool calls**: el modelo re-ejecuta planes ya completados

### Referencia

Documentado en [Reddit: Qwen 3.5 27B/35BA3B tool calling fixes](https://reddit.com/r/LocalLLM/):

> *"Bloques de pensamiento históricos se filtraban al contexto y confundían al modelo"*
>
> *"Oculta el razonamiento histórico del contexto pero deja visible el razonamiento actual"*

### Solución

```diff
- {%- set _preserve_thinking = preserve_thinking if preserve_thinking is defined else true %}
+ {%- set _preserve_thinking = preserve_thinking if preserve_thinking is defined else false %}
```

### Comportamiento después del fix

| Escenario | Thinking visible en prompt |
|---|---|
| Mensajes históricos (turnos anteriores) | ❌ Oculto — solo se envía el contenido |
| Turno actual (último assistant antes de generation) | ✅ Preservado — el modelo ve su razonamiento reciente |
| Override explícito `preserve_thinking=True` | ✅ Funciona — se puede forzar desde SuperMLX |

### Notas

- El thinking del turno actual **siempre** se preserva por la condición
  `loop.index0 > ns.last_query_index` en la línea 196 de la plantilla.
- Si se necesita revertir: cambiar `false` a `true` en línea 7 del archivo,
  o pasar `preserve_thinking=True` como kwarg del template desde SuperMLX.
- Este archivo está en el **cache de HuggingFace**, no en el repo de SuperMLX.
  Si el modelo se re-descarga, el cambio se pierde. Considerar mover la plantilla
  al repo de SuperMLX en el futuro.

---

## Contexto: Estado del template froggeric-v20

La plantilla actual ya incluye mejoras significativas sobre la original de Unsloth:

| Feature | Original Unsloth | froggeric-v20 |
|---|---|---|
| Instrucciones de `</think>` antes de tool call | ❌ | ✅ |
| Detección de errores consecutivos en tools | ❌ | ✅ |
| Truncamiento de args/responses largos | ❌ | ✅ |
| Control dinámico de thinking (`<\|think_off\|>`) | ❌ | ✅ |
| Variantes de think tags (`</thinking>`, `</ think>`) | ❌ | ✅ |
| Preserve thinking por defecto | ✅ (implícito) | ❌ (Fix 2) |

---

## Relación con fixplan.md

Estos fixes son **independientes** del [fixplan.md](fixplan.md) que aborda cache/compactación.

| Área | fixplan.md | Estos fixes |
|---|---|---|
| Cache/KV | ✅ | ❌ |
| Compactación | ✅ | ❌ |
| Tool call parsing | ❌ | ✅ Fix 1 |
| Chat template | ❌ | ✅ Fix 2 |
| Thinking management | ❌ | ✅ Fix 2 |

Ambos se pueden aplicar y ejecutar de forma independiente sin conflictos.
