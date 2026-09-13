#!/usr/bin/env python3
"""
[AI_DIRECTIVE]
ROL: Calibrador de hiperparámetros de muestreo (Min-P, Temp, Top-P) para SuperMLX
OBJETIVO: Encontrar la combinación óptima de hiperparámetros evaluando la calidad
          de generación contra un modelo juez (Gemini 3.8 Flash) usando Optuna.
ENTRADAS: Servidor local SuperMLX (http://localhost:8080) y Gemini API Key en .env
SALIDAS: Parámetros óptimos por categoría, logs de scoring, cache gold_cache_v2.json
REGLAS INVIOLABLES:
- Prohibido prompts con tokens de reasoning literales (<think>) que confundan al tokenizer
- Prohibido try/except silenciosos — siempre registrar con logger
- Logging exhaustivo con prefijos estándar [INICIO][CONFIG][DATA][CALC][DECISION][RESULT][ERROR]
- Evaluación multidimensional con SSoT en Gemini 3.8 Flash
"""

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv
import optuna
from tqdm import tqdm

from google import genai
from google.genai import types

load_dotenv()

# ── Logging Setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("calibrator_v2")

CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gold_cache_v2.json")
LOCAL_MODEL_BASE = os.getenv("SUPERMLX_URL", "http://localhost:8080")
GEMINI_MODEL = os.getenv("GEMINI_EVAL_MODEL", "gemini-3.8-flash")

# ── Benchmark Prompts (Cleaned & High-Impact) ──────────────────────────────────
BENCHMARK_SUITE: Dict[str, List[Dict[str, str]]] = {
    "CODING": [
        {
            "id": "C1_SEGMENT_TREE",
            "prompt": (
                "Implementá en Python 3 una clase `SegmentTree` con Lazy Propagation para updates "
                "de rango en suma (range sum query + range addition update).\n\n"
                "Requisitos obligatorios:\n"
                "1. Manejo riguroso de propagación lazy para no perder actualizaciones pendientes.\n"
                "2. Métodos: `build(arr)`, `update_range(l, r, val)`, `query_range(l, r)` indexados en 0.\n"
                "3. Tipado estricto (Type hints) y docstrings.\n"
                "4. Un bloque de prueba con `assert` que valide al menos 3 updates y 3 queries consecutivas."
            )
        },
        {
            "id": "C2_SLIDING_WINDOW",
            "prompt": (
                "Implementá una clase Python `SlidingWindowCounter` thread-safe que registre eventos y "
                "cuente ocurrencias en una ventana de tiempo deslizante.\n\n"
                "Requisitos:\n"
                "- `add()` registra un evento en el timestamp actual.\n"
                "- `count(window_seconds: float) -> int` devuelve cuántos eventos ocurrieron en los últimos N segundos.\n"
                "- Eficiente en memoria: expirar y descartar timestamps viejos sin acumulación indefinida.\n"
                "- Thread-safe usando `threading.Lock`.\n"
                "- Docstrings completos y types."
            )
        },
        {
            "id": "C3_RETRY_BACKOFF",
            "prompt": (
                "Refactorizá esta función para que sea resiliente a fallos de red en producción:\n\n"
                "```python\n"
                "import urllib.request\n"
                "import json\n\n"
                "def fetch_model_status(host: str, port: int) -> dict:\n"
                "    url = f'http://{host}:{port}/health'\n"
                "    resp = urllib.request.urlopen(url)\n"
                "    return json.loads(resp.read().decode())\n"
                "```\n\n"
                "Requisitos:\n"
                "1. Timeout configurable (default 10s).\n"
                "2. Reintentos con backoff exponencial y jitter (máx 3 intentos).\n"
                "3. Logging con módulo standard logging a nivel WARNING en cada reintento.\n"
                "4. Devolver None en vez de propagar excepción si todos los intentos fallan.\n"
                "5. Tipado estricto."
            )
        },
        {
            "id": "C4_SCRATCHPAD_PARSER",
            "prompt": (
                "Escribí una función Python `parse_reasoning_blocks(raw_text: str) -> tuple[str, str]` "
                "que reciba un string y devuelva una tupla `(reasoning_text, response_text)`.\n\n"
                "El razonamiento está delimitado por etiquetas `<scratchpad>` y `</scratchpad>`.\n"
                "Manejá estos 6 casos con expresiones regulares (`re`):\n"
                "1. `<scratchpad>razonamiento</scratchpad>respuesta` — caso normal\n"
                "2. `razonamiento</scratchpad>respuesta` — orphan close (sin tag de apertura)\n"
                "3. `<scratchpad>razonamiento` — orphan open (generación cortada mid-reasoning)\n"
                "4. `respuesta sin scratchpad` — sin tags\n"
                "5. `<scratchpad></scratchpad>respuesta` — scratchpad vacío\n"
                "6. `<scratchpad>bloque 1</scratchpad>intermedio<scratchpad>bloque 2</scratchpad>respuesta` — múltiples bloques (concatenar en reasoning)\n\n"
                "Incluí docstring y type hints."
            )
        }
    ],
    "DEBUGGING": [
        {
            "id": "D1_CACHE_RACE",
            "prompt": (
                "Esta función debería cachear respuestas por 60 segundos, pero los usuarios reportan "
                "que a veces devuelve respuestas de otro usuario o datos corruptos. Encontrá todos los bugs "
                "y explicá detalladamente por qué pasan.\n\n"
                "```python\n"
                "import time\n"
                "from threading import Lock\n\n"
                "_cache = {}\n"
                "_lock = Lock()\n\n"
                "def get_cached_response(user_id, prompt):\n"
                "    key = hash(prompt)\n"
                "    with _lock:\n"
                "        if key in _cache:\n"
                "            entry = _cache[key]\n"
                "            if time.time() - entry['ts'] < 60:\n"
                "                return entry['response']\n"
                "    \n"
                "    response = generate_response(prompt)\n"
                "    \n"
                "    with _lock:\n"
                "        _cache[key] = {'response': response, 'ts': time.time()}\n"
                "    \n"
                "    return response\n"
                "```"
            )
        },
        {
            "id": "D2_METAL_OOM",
            "prompt": (
                "Explicá qué está pasando en este error de Metal y qué arquitectura de mitigación debe aplicarse:\n\n"
                "```\n"
                "RuntimeError: [metal] Error: Insufficient memory to allocate 2.15 GB buffer\n"
                "```\n\n"
                "Contexto: Qwen3.6 35B en quant 3-bit corriendo en Apple Silicon con 24GB Unified Memory. "
                "El KV cache tiene 48K tokens acumulados y el nuevo prompt entrante tiene 12K tokens.\n"
                "Explicá cómo maneja Metal la memoria unificada y qué estrategia (eviction, sliding window o chunked prefill) resuelve el fallo."
            )
        },
        {
            "id": "D3_JSON_SYNTAX",
            "prompt": (
                "Corregí este payload JSON para que sea 100% válido según RFC 8259. "
                "Devolvé solo el bloque de código con el JSON corregido, sin explicaciones:\n\n"
                "{\n"
                "  \"model\": \"qwen3-35b\",\n"
                "  \"temperature\": 0.6,\n"
                "  \"messages\": [\n"
                "    {\"role\": \"system\", \"content\": \"You are a coding assistant\"},\n"
                "    {\"role\": \"user\", \"content\": \"explain what's a \"transformer\" in ML\"},\n"
                "  ],\n"
                "  \"max_tokens\": 2048,\n"
                "  \"stream\": true\n"
                "  \"top_p\": 0.9\n"
                "}"
            )
        }
    ],
    "AGENTIC": [
        {
            "id": "G1_TOOL_REPLACE",
            "prompt": (
                "Sos un agente de codificación conectado a una herramienta `replace_file_content`.\n"
                "El esquema de la herramienta es:\n"
                "```json\n"
                "{\n"
                "  \"TargetFile\": \"string (path absoluto)\",\n"
                "  \"Instruction\": \"string\",\n"
                "  \"Description\": \"string\",\n"
                "  \"StartLine\": int,\n"
                "  \"EndLine\": int,\n"
                "  \"TargetContent\": \"string (código exacto a reemplazar)\",\n"
                "  \"ReplacementContent\": \"string (nuevo código)\",\n"
                "  \"AllowMultiple\": bool\n"
                "}\n"
                "```\n\n"
                "Archivo objetivo `/workspace/auth.py`:\n"
                "```python\n"
                "15: def check_token(token: str) -> bool:\n"
                "16:     if not token:\n"
                "17:         return False\n"
                "18:     return token == 'legacy_secret'\n"
                "```\n\n"
                "Tarea: Reemplazar el chequeo hardcodeado para validar contra `os.environ.get('AUTH_TOKEN')`.\n"
                "Generá únicamente la invocación en formato JSON válido respetando espacios de indentación exactos."
            )
        },
        {
            "id": "G2_CONVENTIONAL_COMMIT",
            "prompt": (
                "Generá un commit message convencional (primera línea máx 72 chars con tipo y scope, "
                "seguido de cuerpo con bullets de justificación técnica) para este diff:\n\n"
                "```diff\n"
                "--- a/supermlx/expert_cache.py\n"
                "+++ b/supermlx/expert_cache.py\n"
                "@@ -652,3 +652,3 @@\n"
                "-scores = scores * hit_mask\n"
                "+if not full_residency:\n"
                "+    scores = scores * hit_mask\n"
                "```\n\n"
                "El cambio evita anular las activaciones de los expertos cuando la capacidad del MoE es total (256/256)."
            )
        }
    ],
    "COMPACTION": [
        {
            "id": "X1_SESSION_SUMMARY",
            "prompt": (
                "Condensá esta sesión de pair programming en un resumen técnico de máximo 200 palabras, "
                "organizado en 3 secciones en Markdown:\n"
                "1. `### Decisiones Técnicas`\n"
                "2. `### Causa Raíz Identificada`\n"
                "3. `### Próximos Pasos`\n\n"
                "Texto de la sesión:\n"
                "- Se investigó por qué el modelo corría a 35 TPS en vez de 54 TPS.\n"
                "- Se detectó un error en .env: DEFAULT_MIN_P estaba configurado en 0.8 en vez de 0.05.\n"
                "- Al aplicar Min-P=0.05 la velocidad subió a 54.7 TPS con Metal peak de 18.29GB.\n"
                "- Se constató que el calibrador previo entraba en loops infinitos porque el prompt A1 pedía parsear tags <think>.\n"
                "- Se resolvió crear calibrate_v2 con suite orientada a programación pura y Gemini 3.8 Flash como juez."
            )
        }
    ]
}

# ── Evaluator Prompt Template ─────────────────────────────────────────────────
EVAL_TEMPLATE = """Sos un evaluador experto y riguroso de calidad técnica de modelos de lenguaje.

Tu tarea es evaluar la RESPUESTA CANDIDATA generada por un modelo local comparándola con el INPUT DEL USUARIO y una RESPUESTA GOLD STANDARD de referencia.

## Contexto de inferencia del candidato
- Categoría: {category}
- Hiperparámetros: temperature={temperature}, top_p={top_p}, min_p={min_p}

## Input del usuario
<user_prompt>
{user_prompt}
</user_prompt>

## Respuesta Gold Standard (referencia Gemini 3.8 Flash)
<gold_response>
{gold_response}
</gold_response>

## Respuesta Candidata (modelo local)
<candidate_response>
{candidate_response}
</candidate_response>

## Razonamiento interno del modelo (Thinking)
<candidate_thinking>
{candidate_thinking}
</candidate_thinking>

## Criterios de Evaluación (1.0 a 10.0 cada uno):
1. **precision**: Corrección técnica, matemática y sintáctica. ¿Compila? ¿Tiene bugs sutiles o typos?
2. **adherencia**: Cumplimiento del 100% de los requisitos explícitos del prompt sin saltear nada.
3. **completitud**: Solución integral con type hints, docstrings y pruebas cuando fueron solicitadas.
4. **concision**: Ausencia de texto de relleno innecesario; claridad de código y explicaciones.
5. **calidad_thinking**: Si razonó, ¿el thinking fue lineal, analítico y evitó bucles repetitivos?

Respondé ÚNICAMENTE con un objeto JSON con este schema exacto:
```json
{{
  "precision": 9.0,
  "adherencia": 9.5,
  "completitud": 8.5,
  "concision": 9.0,
  "calidad_thinking": 8.0,
  "justificacion": "Breve explicación de los aciertos y fallos"
}}
```
"""

# ── Helper Functions ──────────────────────────────────────────────────────────

def get_prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.strip().encode("utf-8")).hexdigest()[:16]


def load_gold_cache() -> Dict[str, str]:
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning("[DATA] No se pudo leer cache %s: %s", CACHE_FILE, e)
    return {}


def save_gold_cache(cache: Dict[str, str]) -> None:
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
        logger.info("[DATA] Gold cache actualizado en %s (%d entradas)", CACHE_FILE, len(cache))
    except Exception as e:
        logger.error("[ERROR] Error guardando gold cache: %s", e)


def escape_braces(text: str) -> str:
    return text.replace("{", "{{").replace("}", "}}")


def extract_thinking_and_response(raw_text: str) -> Tuple[str, str]:
    """Extrae bloques <think>...</think> si existen."""
    pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
    match = pattern.search(raw_text)
    if match:
        thinking = match.group(1).strip()
        response = pattern.sub("", raw_text).strip()
        return thinking, response
    # Orphan close tag
    if "</think>" in raw_text:
        parts = raw_text.split("</think>", 1)
        return parts[0].strip(), parts[1].strip()
    return "", raw_text.strip()


def call_gemini_with_retry(
    client: genai.Client,
    model: str,
    contents: str,
    config: types.GenerateContentConfig,
    max_retries: int = 3,
    initial_delay: float = 2.0,
) -> str:
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
            return response.text or ""
        except Exception as e:
            logger.warning("[DATA] Intento %d/%d Gemini falló: %s", attempt + 1, max_retries, e)
            if attempt == max_retries - 1:
                raise
            time.sleep(delay)
            delay *= 2
    return ""


def call_local_model(
    prompt: str,
    params: Dict[str, Any],
    timeout: int = 180,
) -> str:
    url = f"{LOCAL_MODEL_BASE}/v1/chat/completions"
    payload = {
        "model": "qwen3-35b",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": params["temperature"],
        "top_p": params["top_p"],
        "min_p": params["min_p"],
        "enable_thinking": params.get("enable_thinking", True),
        "max_tokens": 4096,
        "stream": False,
    }
    
    start_time = time.time()
    logger.info("[DATA] Requesting Local Model | T=%.2f TopP=%.2f MinP=%.3f",
                params["temperature"], params["top_p"], params["min_p"])
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        elapsed = int((time.time() - start_time) * 1000)
        logger.info("[RESULT] Local Model OK | elapsed=%dms len=%d", elapsed, len(content))
        return content
    except requests.exceptions.Timeout:
        logger.error("[ERROR] Local Model TIMEOUT después de %ds", timeout)
        raise
    except Exception as e:
        logger.error("[ERROR] Local Model request error: %s", e)
        raise


# ── Optuna Objective ──────────────────────────────────────────────────────────

def suggest_params(trial: optuna.Trial, category: str) -> Dict[str, Any]:
    # Min-P search space: 0.01 a 0.10 con granularidad fina
    min_p = trial.suggest_float("min_p", 0.01, 0.10, step=0.01)
    # Top-P centrado en Frontier default
    top_p = trial.suggest_float("top_p", 0.90, 1.00, step=0.05)
    
    if category in ("CODING", "DEBUGGING"):
        temperature = trial.suggest_float("temperature", 0.35, 0.70, step=0.05)
        enable_thinking = True
    elif category == "AGENTIC":
        temperature = trial.suggest_float("temperature", 0.20, 0.50, step=0.05)
        enable_thinking = True
    elif category == "COMPACTION":
        temperature = trial.suggest_float("temperature", 0.15, 0.40, step=0.05)
        enable_thinking = False
    else:
        temperature = trial.suggest_float("temperature", 0.30, 0.70, step=0.05)
        enable_thinking = True

    return {
        "temperature": temperature,
        "top_p": top_p,
        "min_p": min_p,
        "enable_thinking": enable_thinking,
    }


def make_objective(
    category: str,
    prompts: List[Dict[str, str]],
    client: genai.Client,
    gold_cache: Dict[str, str],
):
    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, category)
        trial_scores: List[float] = []

        logger.info("[INICIO] Trial %d | Params=%s", trial.number, params)

        for item in prompts:
            prompt_id = item["id"]
            user_prompt = item["prompt"]
            p_hash = get_prompt_hash(user_prompt)

            # 1. Obtener o generar Gold Standard
            if p_hash in gold_cache:
                gold_response = gold_cache[p_hash]
            else:
                logger.info("[DATA] Generando Gold Standard con %s para %s...", GEMINI_MODEL, prompt_id)
                try:
                    gold_response = call_gemini_with_retry(
                        client=client,
                        model=GEMINI_MODEL,
                        contents=user_prompt,
                        config=types.GenerateContentConfig(
                            temperature=0.2,
                            max_output_tokens=4096,
                        ),
                    )
                    gold_cache[p_hash] = gold_response
                    save_gold_cache(gold_cache)
                except Exception as e:
                    logger.error("[ERROR] Falló generación de Gold Standard: %s", e)
                    trial_scores.append(1.0)
                    continue

            # 2. Generación local
            try:
                candidate_raw = call_local_model(user_prompt, params)
                candidate_thinking, candidate_response = extract_thinking_and_response(candidate_raw)
            except Exception as e:
                logger.error("[ERROR] Falló inferencia local para %s: %s", prompt_id, e)
                trial_scores.append(1.0)
                continue

            # 3. Evaluación con Gemini 3.8 Flash
            eval_input = EVAL_TEMPLATE.format(
                category=category,
                temperature=f"{params['temperature']:.3f}",
                top_p=f"{params['top_p']:.3f}",
                min_p=f"{params['min_p']:.3f}",
                user_prompt=escape_braces(user_prompt),
                gold_response=escape_braces(gold_response),
                candidate_response=escape_braces(candidate_response),
                candidate_thinking=escape_braces(candidate_thinking),
            )

            try:
                eval_raw = call_gemini_with_retry(
                    client=client,
                    model=GEMINI_MODEL,
                    contents=eval_input,
                    config=types.GenerateContentConfig(
                        temperature=0.1,
                        response_mime_type="application/json",
                    ),
                )
                
                # Parsear evaluación
                # Limpiar backticks si los hubiera
                clean_json = eval_raw.strip()
                if clean_json.startswith("```"):
                    clean_json = clean_json.split("\n", 1)[1]
                if clean_json.endswith("```"):
                    clean_json = clean_json.rsplit("\n", 1)[0]
                
                eval_data = json.loads(clean_json)
                precision = float(eval_data.get("precision", 5.0))
                adherencia = float(eval_data.get("adherencia", 5.0))
                completitud = float(eval_data.get("completitud", 5.0))
                concision = float(eval_data.get("concision", 5.0))
                calidad_thinking = float(eval_data.get("calidad_thinking", 5.0))

                # Ponderación calibrada para coding:
                # 35% precisión, 25% adherencia, 20% completitud, 10% concisión, 10% thinking
                composite_score = (
                    precision * 0.35 +
                    adherencia * 0.25 +
                    completitud * 0.20 +
                    concision * 0.10 +
                    calidad_thinking * 0.10
                )
                
                logger.info(
                    "[CALC] %s | P=%.1f A=%.1f C=%.1f Con=%.1f Thk=%.1f -> Score=%.3f | %s",
                    prompt_id, precision, adherencia, completitud, concision, calidad_thinking,
                    composite_score, eval_data.get("justificacion", "")[:80]
                )
                trial_scores.append(composite_score)

            except Exception as e:
                logger.error("[ERROR] Error evaluando con Gemini: %s", e)
                trial_scores.append(2.0)

        avg_score = sum(trial_scores) / max(1, len(trial_scores))
        logger.info("[RESULT] Trial %d Score Promedio = %.3f", trial.number, avg_score)
        return avg_score

    return objective


# ── Main Entrypoint ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrador de muestreo SuperMLX v2")
    parser.add_argument(
        "--category",
        choices=["CODING", "DEBUGGING", "AGENTIC", "COMPACTION", "ALL"],
        default="CODING",
        help="Categoría a calibrar (default: CODING)"
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=12,
        help="Cantidad de trials de Optuna (default: 12)"
    )
    args = parser.parse_args()

    logger.info("[INICIO] Calibrador SuperMLX v2 | Categoria=%s Trials=%d Juez=%s",
                args.category, args.trials, GEMINI_MODEL)

    # Verificar server local
    try:
        r = requests.get(f"{LOCAL_MODEL_BASE}/health", timeout=5)
        if r.status_code != 200:
            logger.warning("[CONFIG] /health retornó %d, verificando /...", r.status_code)
            r = requests.get(LOCAL_MODEL_BASE, timeout=5)
        logger.info("[CONFIG] Servidor local en %s OK.", LOCAL_MODEL_BASE)
    except Exception as e:
        logger.error("[ERROR] No se pudo conectar a %s: %s", LOCAL_MODEL_BASE, e)
        sys.exit(1)

    # Inicializar cliente de Google GenAI
    client = genai.Client()
    gold_cache = load_gold_cache()

    categories_to_run = (
        ["CODING", "DEBUGGING", "AGENTIC", "COMPACTION"]
        if args.category == "ALL"
        else [args.category]
    )

    for cat in categories_to_run:
        prompts = BENCHMARK_SUITE[cat]
        logger.info("\n=== Calibrando Categoría: %s (%d prompts, %d trials) ===",
                    cat, len(prompts), args.trials)

        study = optuna.create_study(
            study_name=f"supermlx_{cat.lower()}_v2",
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )

        obj_func = make_objective(cat, prompts, client, gold_cache)
        study.optimize(obj_func, n_trials=args.trials)

        logger.info("[DECISION] === RESULTADOS ÓPTIMOS PARA %s ===", cat)
        logger.info("  Best Score: %.4f", study.best_value)
        for k, v in study.best_params.items():
            logger.info("  %s: %s", k, v)


if __name__ == "__main__":
    main()
