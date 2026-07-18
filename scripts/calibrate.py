#!/usr/bin/env python3
import os
import re
import json
import time
import hashlib
import logging
import argparse
import requests
from dotenv import load_dotenv
import optuna
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

# Importar el SDK de Google GenAI
from google import genai
from google.genai import types

# Cargar variables de entorno del archivo .env
load_dotenv()

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("calibrator")

# Cache de respuestas de referencia (absoluto relativo al script)
CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gold_cache.json")

# =====================================================================
# 1. TEST SUITE (16 Prompts hardcodeados)
# =====================================================================
TEST_SUITE = {
    "TRIVIAL": [
        # T1
        "Generá un commit message conciso (max 72 chars primera línea, opcional cuerpo con bullets) para este diff:\n\n"
        "--- a/server.py\n"
        "+++ b/server.py\n"
        "@@ -226,7 +226,7 @@\n"
        " FEATURE_COMPRESSOR = _env_str(\"FEATURE_COMPRESSOR\", \"false\").lower() in (\"1\", \"true\", \"yes\")\n"
        "-FEATURE_EMERGENCY_COMPRESS = _env_str(\"EMERGENCY_CONTENT_COMPRESS\", \"true\").lower() in (\"1\", \"true\", \"yes\")\n"
        "+FEATURE_EMERGENCY_COMPRESS = _env_str(\"EMERGENCY_CONTENT_COMPRESS\", \"false\").lower() in (\"1\", \"true\", \"yes\")\n\n"
        "@@ -3551,6 +3551,8 @@\n"
        "         sampler, _ = _build_sampler(body)\n"
        "+        if body.get(\"max_thinking_tokens\") is not None:\n"
        "+            _max_thinking = int(body[\"max_thinking_tokens\"])",
        
        # T2
        "Clasificá este mensaje de usuario en exactamente una categoría: CODE_EDIT, QUESTION, DEBUG, ARCHITECTURE, OTHER.\n\n"
        "Respondé solo con la categoría, sin explicación.\n\n"
        "Mensaje: \"el test de integración falla en CI pero pasa local, el error es un timeout en el health check del container — puede ser que el readiness probe tenga un initialDelaySeconds muy bajo?\"",
        
        # T3
        "Corregí este JSON para que sea válido. Devolvé solo el JSON corregido, sin explicación.\n\n"
        "{\n"
        "  \"model\": \"qwen3-35b\",\n"
        "  \"temperature\": 0.6,\n"
        "  \"messages\": [\n"
        "    {\"role\": \"system\", \"content\": \"You are a helpful assistant\"},\n"
        "    {\"role\": \"user\", \"content\": \"explain what's a \"transformer\" in ML\"},\n"
        "  ],\n"
        "  \"max_tokens\": 2048,\n"
        "  \"stream\": true\n"
        "  \"top_p\": 0.9\n"
        "}",
        
        # T4
        "Dado: prefill toma 0.525ms por token, decode genera 38 tokens/segundo.\n\n"
        "Prompt de 12,000 tokens, respuesta esperada de 800 tokens.\n\n"
        "¿Cuánto tarda el request total en segundos? Mostrá la cuenta y el resultado final."
    ],
    "CONVERSATIONAL": [
        # C1
        "Estoy escribiendo un servidor HTTP en Python que hace inferencia con un modelo de ML. Cada request toma entre 5 y 120 segundos. Necesito manejar requests concurrentes.\n\n"
        "Comparame estas 3 opciones y recomendame una:\n"
        "1. threading.Thread por request\n"
        "2. asyncio con un executor para la inferencia\n"
        "3. Queue de requests con un worker thread único\n\n"
        "El modelo solo puede procesar 1 request a la vez (usa toda la GPU). El servidor corre en una Mac con 64GB de unified memory.\n\n"
        "Evaluá cada opción contra: concurrencia, uso de memoria, latencia, complejidad de implementación, y el constraint de single-GPU. Terminá con \"Recomendación: [opción]\" y 1 oración de justificación.",
        
        # C2
        "Explicame qué está pasando en este error y por qué:\n\n"
        "Traceback (most recent call last):\n"
        "  File \"server.py\", line 5432, in _handle_chat_completion\n"
        "    for response in _stream_generate_unified(rest_tokens, max_tokens, sampler, prompt_cache):\n"
        "  File \"server.py\", line 2891, in _stream_generate_unified\n"
        "    yield from mx.generate(model, prompt_tokens, sampler=sampler, cache=cache)\n"
        "  File \"mlx_lm/utils.py\", line 445, in generate\n"
        "    logits = model(tokens, cache=cache)\n"
        "RuntimeError: [metal] Error: Insufficient memory to allocate 2.15 GB buffer\n\n"
        "El modelo es Qwen3 35B en 3-bit quantization corriendo en una Mac con 64GB unified memory. El KV cache tiene 48K tokens. El prompt nuevo tiene 12K tokens.",
        
        # C3
        "En mi servidor de inferencia tengo un KV cache que persiste entre requests para acelerar el prefill. Pero el cache ocupa memoria que podría usar el modelo.\n\n"
        "Hoy el cache usa 8GB de los 64GB disponibles. El modelo usa 22GB. Quedan ~34GB libres.\n\n"
        "¿Tiene sentido reducir el cache a 4GB para dejar más margen? ¿O estoy pensando mal y la memoria libre no se usa para nada útil? Explicame cómo maneja Metal la memoria unificada en este contexto.",
        
        # C4
        "Para un proxy HTTP que intercepta requests de un IDE a un modelo de lenguaje:\n\n"
        "¿Es mejor loguear los requests en archivos JSON individuales (1 archivo por request) o en un archivo append-only tipo JSONL (1 línea por request)?\n\n"
        "Considerá: volumen de ~200 requests/día, necesidad de buscar requests por ID, necesidad de analizar patrones con scripts Python, rotación de logs, y que corre en macOS con SSD."
    ],
    "ANALYTICAL": [
        # A1
        "Escribí una función Python `parse_thinking_tokens` que reciba un string con el output crudo de un modelo de lenguaje y devuelva una tupla (thinking_text: str, response_text: str).\n\n"
        "El modelo puede generar en estos formatos:\n"
        "1. `<think>razonamiento</think>respuesta` — caso normal\n"
        "2. `razonamiento</think>respuesta` — orphan close tag (sin opening)\n"
        "3. `<think>razonamiento` — orphan open tag (generación cortada mid-thinking)\n"
        "4. `respuesta sin thinking` — sin tags\n"
        "5. `<think></think>respuesta` — thinking vacío\n"
        "6. `<think>razón</think>intermedio<think>más razón</think>respuesta` — múltiples bloques\n\n"
        "Manejá los 6 casos. Usá regex. Incluí docstring y type hints.",
        
        # A2
        "Esta función debería cachear respuestas por 60 segundos, pero los usuarios reportan que a veces devuelve respuestas de otro usuario. Encontrá el bug y explicá por qué pasa.\n\n"
        "```python\n"
        "import time\n"
        "from threading import Lock\n\n"
        "_cache = {}\n"
        "_lock = Lock()\n\n"
        "def get_cached_response(user_id, prompt):\n"
        "    key = hash(prompt)  \n"
        "    with _lock:\n"
        "        if key in _cache:\n"
        "            entry = _cache[key]\n"
        "            if time.time() - entry[\"ts\"] < 60:\n"
        "                return entry[\"response\"]\n"
        "    \n"
        "    response = generate_response(prompt)\n"
        "    \n"
        "    with _lock:\n"
        "        _cache[key] = {\"response\": response, \"ts\": time.time()}\n"
        "    \n"
        "    return response\n"
        "```",
        
        # A3
        "Refactoreá esta función para que:\n"
        "1. Soporte timeout configurable (default 30s)\n"
        "2. Haga retry con backoff exponencial (max 3 intentos)\n"
        "3. Loguee cada intento con nivel WARNING\n"
        "4. Devuelva None en vez de crashear si todos los intentos fallan\n\n"
        "```python\n"
        "import urllib.request\n"
        "import json\n\n"
        "def fetch_model_status(host, port):\n"
        "    url = f\"http://{host}:{port}/health\"\n"
        "    resp = urllib.request.urlopen(url)\n"
        "    data = json.loads(resp.read().decode())\n"
        "    return data[\"status\"]\n"
        "```",
        
        # A4
        "Necesito una clase Python `SlidingWindowCounter` que cuente eventos en una ventana de tiempo deslizante.\n\n"
        "Requisitos:\n"
        "- `add()` registra un evento en el timestamp actual\n"
        "- `count(window_seconds)` devuelve cuántos eventos hubo en los últimos N segundos\n"
        "- Debe ser eficiente en memoria (no guardar timestamps indefinidamente)\n"
        "- Thread-safe\n"
        "- Los timestamps viejos se limpian automáticamente\n\n"
        "Ejemplo de uso:\n"
        "```python\n"
        "counter = SlidingWindowCounter()\n"
        "counter.add()\n"
        "time.sleep(0.5)\n"
        "counter.add()\n"
        "counter.add()\n"
        "print(counter.count(1))   # 3\n"
        "print(counter.count(0.3)) # 2\n"
        "```\n\n"
        "Implementá la clase completa con docstring."
    ],
    "COMPACTION": [
        # X1
        "Resumí esta conversación en máximo 200 palabras, preservando: decisiones tomadas, razones, y próximos pasos.\n\n"
        "User: necesito que el servidor detecte cuando Hermes manda un request de compactación\n"
        "Assistant: Hermes manda la compactación como un mensaje de usuario con \"summarization agent\" y \"context checkpoint\". Podemos detectar eso en los primeros 200 chars.\n"
        "User: y si lo detecta qué hace?\n"
        "Assistant: Dos cosas: 1) stripea las tool definitions (30K tokens que no sirven para compactar), 2) reemplaza el prompt genérico de Hermes con el prompt estructurado de Claude Code que tiene 9 secciones.\n"
        "User: pero el prompt de Claude Code tiene analysis tags y summary tags, Hermes no usa eso\n"
        "Assistant: No importa — el modelo genera con esas tags y la respuesta sale mejor estructurada. Probé y el output pasó de texto libre a secciones organizadas.\n"
        "User: dale hacelo pero con feature flag\n"
        "Assistant: Listo, HERMES_COMPACT_SWAP=true en .env. Default apagado.\n"
        "User: funcionó pero no detecta, buscás en system msg y Hermes no manda system\n"
        "Assistant: Corregido — ahora busca en el último user message.\n"
        "User: perfecto, commiteá",
        
        # X2
        "De este texto, extraé las especificaciones técnicas en formato de tabla markdown (columna: spec, columna: valor). Solo datos concretos, no opiniones.\n\n"
        "\"El nuevo MacBook Pro con chip M4 Max trae hasta 128GB de memoria unificada con un ancho de banda de 546 GB/s. El chip tiene 16 núcleos de CPU (12 performance + 4 efficiency) y hasta 40 núcleos de GPU. El neural engine tiene 16 cores. Soporta hasta 4 monitores externos via Thunderbolt 5 (120 Gbps). El SSD alcanza 7.4 GB/s de lectura. La batería dura hasta 24 horas de reproducción de video. La pantalla es Liquid Retina XDR de 16.2 pulgadas con resolución 3456x2234 y brillo SDR de 1000 nits, HDR de 1600 nits. Pesa 2.14 kg.\"",
        
        # X3
        "Condensá estos 8 hallazgos de un análisis de performance en los 3 más importantes. Para cada uno, escribí exactamente 1 oración.\n\n"
        "Priorizá por impacto en latencia total del servidor de inferencia.\n\n"
        "1. El prefill toma 4.2 segundos para prompts de 8K tokens\n"
        "2. El decode genera a 38 tokens/segundo\n"
        "3. El KV cache ocupa 8.1GB para una sesión de 32K tokens\n"
        "4. Los tool definitions agregan 30K tokens al prompt (130K chars)\n"
        "5. El 73% del tiempo de prefill se gasta en las tool definitions\n"
        "6. La compresión con LLMLingua reduce el prompt un 40% sin pérdida de calidad medible\n"
        "7. El reranker agrega 200ms de latencia por request\n"
        "8. Requests de title generation (5 tokens de output) hacen prefill completo del contexto (innecesario)"
    ],
    "AGENTIC": [
        # G1
        "Tengo las siguientes herramientas de base de datos disponibles:\n"
        "- `search_users(query: str) -> list[dict]`\n"
        "- `get_user_orders(user_id: int, status: str = None) -> list[dict]`\n"
        "- `refund_order(order_id: int, amount: float) -> bool`\n\n"
        "Usuario: \"El cliente Juan Pérez (ID 4501) dice que quiere el reembolso total de su última orden de $120.50 porque no llegó. Buscá si es él en la base de datos, mirá sus órdenes completadas, y si coincide el monto de su última orden, procedé con el reembolso.\"\n\n"
        "Generá la secuencia de llamadas a funciones correspondiente en formato JSON."
    ]
}

# =====================================================================
# 2. TEMPLATE DE EVALUACIÓN (de optuna_eval_prompts.md)
# =====================================================================
EVAL_TEMPLATE = """Sos un evaluador experto de calidad de respuestas de modelos de lenguaje.

Tu tarea: comparar una RESPUESTA CANDIDATA (de un modelo local) contra una RESPUESTA DE REFERENCIA (de un modelo más capaz) y producir un score de calidad.

## Contexto de la evaluación

- **Categoría del request**: {category}
- **Parámetros de inferencia usados**: temperature={temperature}, top_p={top_p}, min_p={min_p}, max_thinking_tokens={max_thinking_tokens}, enable_thinking={enable_thinking}

## Input del usuario (el prompt original)

<user_prompt>
{user_prompt}
</user_prompt>

## Respuesta de referencia (gold standard)

<gold_response>
{gold_response}
</gold_response>

## Respuesta candidata (modelo local)

<candidate_response>
{candidate_response}
</candidate_response>

## Razonamiento del modelo local (thinking)

<candidate_thinking>
{candidate_thinking}
</candidate_thinking>

## Instrucciones de evaluación

Evaluá la respuesta candidata en 5 dimensiones. Para cada una, asigná un score de 1 a 10:

### 1. ADHERENCIA (¿respondió lo que se pidió?)
- 10: Responde exactamente lo solicitado, sigue todas las instrucciones del prompt
- 7: Responde la pregunta principal pero omite instrucciones secundarias
- 4: Responde parcialmente, se desvía del pedido
- 1: No responde lo que se pidió, ignora el prompt

### 2. PRECISIÓN (¿es correcto lo que dice?)
- 10: Factualmente correcto, código funcional, razonamiento válido
- 7: Mayormente correcto con errores menores que no afectan la utilidad
- 4: Errores significativos que comprometen la utilidad
- 1: Información incorrecta, código roto, razonamiento inválido

Usá la respuesta de referencia como baseline de corrección. Si la candidata difiere de la referencia, evaluá si la diferencia es un error o una alternativa válida.

### 3. COMPLETITUD (¿cubre todo lo necesario?)
- 10: Cubre todos los aspectos relevantes de la pregunta
- 7: Cubre los aspectos principales, omite detalles menores
- 4: Cubre solo una parte del pedido
- 1: Respuesta incompleta que no es útil

### 4. CONCISIÓN (¿es eficiente en su comunicación?)
- 10: Exactamente la cantidad de información necesaria, sin relleno
- 7: Ligeramente verboso pero sin información irrelevante
- 4: Demasiado verboso, repite ideas, incluye información innecesaria
- 1: La mayor parte del texto es relleno o repetición

### 5. EFICIENCIA DE RAZONAMIENTO (¿el thinking fue productivo?)
Evaluá el bloque <candidate_thinking> en relación al resultado:
- 10: Thinking denso, sin repetición, cada paso contribuye al resultado final
- 7: Thinking mayormente productivo con algo de redundancia
- 4: Thinking circular o repetitivo — llegaría a la misma respuesta con menos
- 1: Thinking desperdiciado — la respuesta no refleja el razonamiento, o el thinking es pura repetición

Si no hay thinking (enable_thinking=false o thinking vacío), asigná 7 por default (neutral — no se puede evaluar).

## Formato de respuesta

Respondé EXCLUSIVAMENTE con un JSON válido, sin texto adicional:

```json
{{
  "adherencia": <1-10>,
  "precision": <1-10>,
  "completitud": <1-10>,
  "concision": <1-10>,
  "eficiencia_thinking": <1-10>,
  "score_compuesto": <1-10>,
  "explicacion": "<1-2 oraciones explicando el score compuesto>",
  "categoria_sugerida": "<TRIVIAL|CONVERSATIONAL|ANALYTICAL|COMPACTION|AGENTIC>"
}}
```

Para `score_compuesto`, usá estos pesos según la categoría:

- **TRIVIAL**: adherencia×0.4 + precision×0.3 + concision×0.3
- **CONVERSATIONAL**: adherencia×0.3 + precision×0.2 + completitud×0.2 + concision×0.3
- **ANALYTICAL**: precision×0.35 + completitud×0.25 + adherencia×0.2 + eficiencia_thinking×0.2
- **COMPACTION**: completitud×0.3 + adherencia×0.3 + concision×0.2 + precision×0.2
- **AGENTIC**: precision×0.35 + adherencia×0.35 + completitud×0.2 + concision×0.1

`categoria_sugerida`: tu clasificación del tipo de request (puede diferir de la categoría asignada).
"""

# Pesos por categoría para calcular score compuesto localmente (D3)
CATEGORY_WEIGHTS = {
    "TRIVIAL":        {"adherencia": 0.4, "precision": 0.3, "concision": 0.3},
    "CONVERSATIONAL": {"adherencia": 0.3, "precision": 0.2, "completitud": 0.2, "concision": 0.3},
    "ANALYTICAL":     {"precision": 0.35, "completitud": 0.25, "adherencia": 0.2, "eficiencia_thinking": 0.2},
    "COMPACTION":     {"completitud": 0.3, "adherencia": 0.3, "concision": 0.2, "precision": 0.2},
    "AGENTIC":        {"precision": 0.35, "adherencia": 0.35, "completitud": 0.2, "concision": 0.1},
}

def compute_composite_score(eval_data: dict, category: str) -> float:
    """Calcula el score compuesto localmente con los pesos por categoría."""
    weights = CATEGORY_WEIGHTS.get(category, CATEGORY_WEIGHTS["CONVERSATIONAL"])
    score = sum(float(eval_data.get(dim, 5.0)) * w for dim, w in weights.items())
    return max(1.0, min(10.0, score))

def escape_braces(text: str) -> str:
    """Escapa { y } para que str.format() no los interprete como placeholders."""
    return text.replace("{", "{{").replace("}", "}}")

# =====================================================================
# 3. EXTRAER THINKING
# =====================================================================
def extract_thinking_and_response(text: str) -> tuple[str, str]:
    """
    Extrae el thinking (contenido entre <think>...</think>) y la respuesta.
    Maneja tags huérfanos o múltiples bloques de forma robusta.
    """
    if not text:
        return "", ""
        
    thinking_blocks = []
    # Buscar bloques completos
    pattern_complete = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    matches = list(pattern_complete.finditer(text))
    
    if matches:
        for match in matches:
            thinking_blocks.append(match.group(1).strip())
        response_text = pattern_complete.sub("", text).strip()
        thinking_text = "\n\n".join(thinking_blocks)
        return thinking_text, response_text
        
    # Caso huérfano de cierre: razonamiento</think>respuesta
    if "</think>" in text and "<think>" not in text:
        parts = text.split("</think>", 1)
        return parts[0].strip(), parts[1].strip()
        
    # Caso huérfano de apertura: <think>razonamiento (cortado)
    if "<think>" in text and "</think>" not in text:
        parts = text.split("<think>", 1)
        return parts[1].strip(), parts[0].strip()
        
    # Caso sin tags
    return "", text.strip()

# =====================================================================
# 4. CACHÉ DE RESPUESTAS GOLD
# =====================================================================
def get_prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()

def load_gold_cache() -> dict:
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error al cargar el caché Gold: {e}")
            return {}
    return {}

def save_gold_cache(cache: dict):
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Error al guardar el caché Gold: {e}")

# =====================================================================
# 5. LLAMADAS CON REINTENTOS
# =====================================================================
def call_gemini_with_retry(client, model, contents, config, max_retries=3, initial_delay=2.0, call_timeout=120):
    """Llama a Gemini con timeout HTTP y backoff exponencial.
    
    call_timeout: segundos máximos para esperar respuesta de Gemini.
    Si la conexión TCP se cuelga, concurrent.futures corta la espera.
    """
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    client.models.generate_content,
                    model=model, contents=contents, config=config
                )
                response = future.result(timeout=call_timeout)
            return response.text
        except FuturesTimeoutError:
            logger.warning(f"Timeout de {call_timeout}s en Gemini (intento {attempt+1}/{max_retries})")
            if attempt == max_retries - 1:
                raise TimeoutError(f"Gemini no respondió en {call_timeout}s después de {max_retries} intentos")
            time.sleep(delay)
            delay *= 2
        except Exception as e:
            logger.warning(f"Error llamando a Gemini (intento {attempt+1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                raise e
            time.sleep(delay)
            delay *= 2

LOCAL_MODEL_BASE = "http://localhost:8080"

def _wait_server_ready(timeout=60):
    """Bloquea hasta que el servidor local responda al health check.
    
    Uso: verificación de arranque y recuperación post-crash.
    NO sirve como barrera durante generación (el health endpoint
    responde 200 en un thread separado mientras el modelo genera).
    La sincronización real la da stream=False: el POST bloquea
    hasta que la generación termina.
    """
    url = f"{LOCAL_MODEL_BASE}/"  # El endpoint principal usa / (no /health)
    deadline = time.time() + timeout
    attempt = 0
    while time.time() < deadline:
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                return True
        except requests.exceptions.ConnectionError:
            if attempt % 10 == 0:  # Loguear cada 5 segundos, no cada 0.5s
                logger.info("Esperando a que el servidor local esté disponible...")
        except Exception:
            pass
        attempt += 1
        time.sleep(0.5)
    return False

def call_local_model_with_retry(prompt, params, max_retries=3, initial_delay=5.0, timeout=180):
    """Llama al modelo local de SuperMLX con reintentos.
    
    Sincronización: stream=False hace que requests.post() bloquee hasta
    que el servidor termine de generar y envíe la respuesta completa.
    No se necesitan barreras adicionales entre requests secuenciales.
    En caso de error de conexión (crash del servidor), espera a que
    vuelva a estar disponible antes de reintentar.
    """
    url = f"{LOCAL_MODEL_BASE}/v1/chat/completions"
    
    payload = {
        "model": "qwen3-35b",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": params["temperature"],
        "top_p": params["top_p"],
        "min_p": params["min_p"],
        "enable_thinking": params["enable_thinking"],
        "max_tokens": 4096,
        "stream": False
    }
    
    if params["enable_thinking"]:
        payload["max_thinking_tokens"] = params["max_thinking_tokens"]

    delay = initial_delay
    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=payload, timeout=timeout)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except requests.exceptions.ConnectionError as e:
            # Servidor se cayó — esperar a que vuelva
            logger.warning(f"Conexión rechazada (intento {attempt+1}/{max_retries}). Esperando recovery...")
            if attempt == max_retries - 1:
                raise e
            if not _wait_server_ready(timeout=120):
                logger.error("Servidor no se recuperó después de 120s")
                raise ConnectionError("Servidor local no se recuperó")
        except requests.exceptions.Timeout:
            logger.warning(f"Timeout de {timeout}s (intento {attempt+1}/{max_retries}). Generación muy larga.")
            if attempt == max_retries - 1:
                raise
            # No esperar — el timeout ya implica que esperamos suficiente
        except Exception as e:
            logger.warning(f"Error llamando al modelo local (intento {attempt+1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                raise e
            time.sleep(delay)
            delay *= 2

# =====================================================================
# 6. CONFIGURACIÓN DEL SEARCH SPACE POR CATEGORÍA
# =====================================================================
def suggest_params(trial, category):
    """Define el search space acotado por categoría de acuerdo a la especificación."""
    top_p = trial.suggest_float("top_p", 0.8, 1.0)
    min_p = trial.suggest_float("min_p", 0.0, 0.1)

    if category == "TRIVIAL":
        # Trivial: poco o nada de thinking, baja temperatura
        enable_thinking = trial.suggest_categorical("enable_thinking", [False, True])
        max_thinking_tokens = trial.suggest_int("max_thinking_tokens", 0, 128, step=32) if enable_thinking else 0
        temperature = trial.suggest_float("temperature", 0.1, 0.5)
        
    elif category == "CONVERSATIONAL":
        # Conversacional: thinking medio, temperatura moderada-alta
        enable_thinking = trial.suggest_categorical("enable_thinking", [True, False])
        max_thinking_tokens = trial.suggest_int("max_thinking_tokens", 512, 1024, step=128) if enable_thinking else 0
        temperature = trial.suggest_float("temperature", 0.5, 0.9)
        
    elif category == "ANALYTICAL":
        # Analítico: requiere razonamiento pesado y preciso, temperatura moderada
        enable_thinking = trial.suggest_categorical("enable_thinking", [True]) # Forzado a True para esta categoría
        max_thinking_tokens = trial.suggest_int("max_thinking_tokens", 2048, 4096, step=256)
        temperature = trial.suggest_float("temperature", 0.3, 0.7)
        
    elif category == "COMPACTION":
        # Compactación: estructuración, temperatura baja
        enable_thinking = trial.suggest_categorical("enable_thinking", [True, False])
        max_thinking_tokens = trial.suggest_int("max_thinking_tokens", 1024, 2048, step=256) if enable_thinking else 0
        temperature = trial.suggest_float("temperature", 0.1, 0.5)
        
    elif category == "AGENTIC":
        # Agéntico: estructuración y tool calls, temperatura moderada-baja
        enable_thinking = trial.suggest_categorical("enable_thinking", [True, False])
        max_thinking_tokens = trial.suggest_int("max_thinking_tokens", 1024, 2048, step=256) if enable_thinking else 0
        temperature = trial.suggest_float("temperature", 0.4, 0.8)
        
    else:
        # Default de fallback general si no coincide
        enable_thinking = trial.suggest_categorical("enable_thinking", [True, False])
        max_thinking_tokens = trial.suggest_int("max_thinking_tokens", 0, 4096, step=128) if enable_thinking else 0
        temperature = trial.suggest_float("temperature", 0.2, 1.0)

    return {
        "enable_thinking": enable_thinking,
        "max_thinking_tokens": max_thinking_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "min_p": min_p
    }

# =====================================================================
# 7. MAIN OBJECTIVE FUNCTION FOR OPTUNA
# =====================================================================
def make_objective(category, prompts, client, gold_cache):
    def objective(trial):
        params = suggest_params(trial, category)
        scores = []
        
        # Ejecutar secuencialmente para los prompts de la categoría seleccionada
        for prompt in prompts:
            prompt_hash = get_prompt_hash(prompt)
            
            # --- 1. GOLD GENERATION / RETRIEVAL ---
            if prompt_hash in gold_cache:
                gold_response = gold_cache[prompt_hash]
            else:
                try:
                    logger.info("Generando Gold Standard usando Gemini...")
                    gold_response = call_gemini_with_retry(
                        client=client,
                        model="gemini-3.5-flash",
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            temperature=0.3,
                            max_output_tokens=8192,
                            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
                        )
                    )
                    gold_cache[prompt_hash] = gold_response
                    save_gold_cache(gold_cache)
                except Exception as e:
                    logger.error(f"Error generando Gold Standard: {e}. Asignando score mínimo.")
                    scores.append(1.0)
                    continue

            # --- 2. LOCAL CANDIDATE GENERATION ---
            try:
                candidate_raw = call_local_model_with_retry(prompt, params)
                candidate_thinking, candidate_response = extract_thinking_and_response(candidate_raw)
            except Exception as e:
                logger.error(f"Error ejecutando modelo local: {e}. Penalizando trial.")
                scores.append(1.0)
                continue

            # --- 3. EVALUATION ---
            try:
                # B2 fix: escapear {} en texto libre para que .format() no explote
                eval_prompt = EVAL_TEMPLATE.format(
                    category=category,
                    temperature=f"{params['temperature']:.3f}",
                    top_p=f"{params['top_p']:.3f}",
                    min_p=f"{params['min_p']:.3f}",
                    max_thinking_tokens=params["max_thinking_tokens"],
                    enable_thinking=str(params["enable_thinking"]).lower(),
                    user_prompt=escape_braces(prompt),
                    gold_response=escape_braces(gold_response),
                    candidate_response=escape_braces(candidate_response),
                    candidate_thinking=escape_braces(candidate_thinking)
                )
                
                # Ejecutar evaluador
                eval_raw = call_gemini_with_retry(
                    client=client,
                    model="gemini-3.5-flash",
                    contents=eval_prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.1,
                        response_mime_type="application/json",
                        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
                    )
                )
                
                # D3 fix: parsear dimensiones individuales y calcular score localmente
                # Fallback robusto: si el JSON está roto (comillas en explicacion),
                # extraer los 5 scores numéricos con regex
                try:
                    eval_data = json.loads(eval_raw)
                except json.JSONDecodeError:
                    # Fallback: extraer scores con regex del JSON parcialmente roto
                    eval_data = {}
                    for dim in ("adherencia", "precision", "completitud", "concision", "eficiencia_thinking"):
                        match = re.search(rf'"{dim}"\s*:\s*(\d+(?:\.\d+)?)', eval_raw)
                        if match:
                            eval_data[dim] = float(match.group(1))
                    if not eval_data:
                        logger.error(f"No se pudieron extraer scores del evaluador. Raw: {eval_raw[:200]}")
                        scores.append(1.0)
                        continue
                    logger.debug(f"JSON roto — scores extraídos con regex: {eval_data}")
                
                try:
                    score = compute_composite_score(eval_data, category)
                    scores.append(score)
                    logger.debug(f"Eval dims: {eval_data} → score local: {score:.2f}")
                except (ValueError, KeyError) as ve:
                    logger.error(f"Error calculando score compuesto: {ve}")
                    scores.append(1.0)
            except Exception as e:
                logger.error(f"Error en el proceso de evaluación: {e}. Score mínimo asignado.")
                scores.append(1.0)

        # Retornamos el score promedio de los prompts
        if not scores:
            return 1.0
        return sum(scores) / len(scores)

    return objective

# =====================================================================
# 8. EXECUTION LOOP & MAIN
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Calibración de Inferencia en SuperMLX usando Optuna.")
    parser.add_argument(
        "--category",
        type=str,
        default="ALL",
        choices=["ALL", "TRIVIAL", "CONVERSATIONAL", "ANALYTICAL", "COMPACTION", "AGENTIC"],
        help="Categoría específica a optimizar o 'ALL' para todas (default: ALL)."
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=60,
        help="Cantidad de trials por categoría (default: 60)."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="calibration_results.json",
        help="Archivo JSON de salida para exportar parámetros óptimos (default: calibration_results.json)."
    )
    
    args = parser.parse_args()
    
    # Validar API key de Gemini
    if not os.environ.get("GEMINI_API_KEY"):
        logger.error("La variable de entorno GEMINI_API_KEY no está seteada.")
        return

    # Verificar que el servidor local esté levantado antes de arrancar
    logger.info("Verificando conexión con servidor local...")
    if not _wait_server_ready(timeout=15):
        logger.error(f"Servidor local en {LOCAL_MODEL_BASE} no responde. Levantalo antes de correr el script.")
        return
    logger.info(f"Servidor local en {LOCAL_MODEL_BASE} OK.")

    # Inicializar cliente Gemini
    logger.info("Inicializando cliente de Gemini con google-genai SDK...")
    client = genai.Client()
    
    # Cargar caché Gold
    gold_cache = load_gold_cache()
    
    # Categorías a procesar
    categories_to_run = []
    if args.category == "ALL":
        categories_to_run = list(TEST_SUITE.keys())
    else:
        categories_to_run = [args.category]
        
    results = {}
    
    # Cargar resultados anteriores si ya existen para no borrarlos en multi-runs
    if os.path.exists(args.output):
        try:
            with open(args.output, "r", encoding="utf-8") as f:
                results = json.load(f)
        except Exception:
            pass

    for cat in categories_to_run:
        logger.info(f"\n=== Iniciando calibración para categoría: {cat} ({args.trials} trials) ===")
        prompts = TEST_SUITE[cat]
        
        # Desactivar logs ruidosos de Optuna para usar tqdm de forma limpia
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        study = optuna.create_study(direction="maximize")
        objective_func = make_objective(cat, prompts, client, gold_cache)
        
        # Ejecutar la optimización con barra de progreso tqdm
        with tqdm(total=args.trials, desc=f"Optimizing {cat}") as pbar:
            def callback(study, trial):
                pbar.update(1)
                pbar.set_postfix({"best_score": f"{study.best_value:.3f}" if study.best_value else "N/A"})
            
            study.optimize(objective_func, n_trials=args.trials, callbacks=[callback])
            
        # B5 fix: proteger contra study sin trials completados
        try:
            best_trial = study.best_trial
        except ValueError:
            logger.error(f"No hay trials completados para {cat}. Saltando.")
            continue
            
        logger.info(f"¡Calibración completada para {cat}!")
        logger.info(f"Mejor Score: {best_trial.value:.4f}")
        logger.info(f"Parámetros óptimos: {best_trial.params}")
        
        # B1 fix: asegurar que max_thinking_tokens siempre esté en el export
        best_params = dict(best_trial.params)
        if "max_thinking_tokens" not in best_params:
            best_params["max_thinking_tokens"] = 0
        
        # Guardar en resultados finales
        results[cat] = {
            "best_score": best_trial.value,
            "params": best_params,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Persistir resultados parciales
        try:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"Resultados guardados temporalmente en: {args.output}")
        except Exception as e:
            logger.error(f"Error escribiendo en {args.output}: {e}")

    logger.info("\n=== Calibración Finalizada exitosamente ===")
    logger.info(f"Resultados consolidados guardados en: {args.output}")

if __name__ == "__main__":
    main()
