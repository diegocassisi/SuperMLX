#!/usr/bin/env python3
"""
[AI_DIRECTIVE]
ROL: Benchmark de Calibración Factual y Conflicto de Expertos para EAACD (Palanca 6).
OBJETIVO: Enviar una batería diversa de preguntas con alto potencial de alucinación,
          entidades poco frecuentes, disputas de APIs y premisas falsas para medir la
          distribución real de incertidumbre del router y calibrar H_scale sin sesgo de código.
ENTRADAS: HTTP requests a SuperMLX (:8080)
SALIDAS: Registro de respuestas y métricas de inferencia.
REGLAS INVIOLABLES:
- Prohibido hardcoding de claves
- Obligatorio logging estructurado
"""

import time
import requests
import json
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("factual_qa_benchmark")

URL = "http://localhost:8080/v1/chat/completions"

FACTUAL_PROMPTS = [
    {
        "category": "OBSCURE_FACTS",
        "id": "chem_reaction",
        "prompt": "¿Cuál es el reactivo principal, el subproducto y la temperatura típica empleada en la reacción de Wohl-Ziegler?"
    },
    {
        "category": "OBSCURE_FACTS",
        "id": "hardware_arch",
        "prompt": "¿Qué tamaño tiene el buffer de reordenamiento (ROB) y cuántas ALUs de enteros posee el core Avalanche del Apple M1 Max?"
    },
    {
        "category": "FALSE_PREMISE_TRAP",
        "id": "napoleon_trap",
        "prompt": "¿En qué tratado de paz Napoleón Bonaparte cedió el control de la isla de Terranova a la corona de Portugal en 1811?"
    },
    {
        "category": "FALSE_PREMISE_TRAP",
        "id": "python_trap",
        "prompt": "¿Cómo se utiliza el parámetro `force_cuda=True` en la función `mlx.core.array` de la librería MLX de Apple?"
    },
    {
        "category": "API_DISPUTES",
        "id": "rare_api",
        "prompt": "En la librería PyTorch C++ API (LibTorch), ¿cuál es el signature exacto del método para registrar un hook backward en `torch::autograd::Node`?"
    },
    {
        "category": "FINE_GRAINED_HISTORY",
        "id": "treaty_history",
        "prompt": "¿Quiénes fueron los tres diplomáticos plenipotenciarios que firmaron el Tratado de San Ildefonso de 1777 y qué territorios intercambiaron?"
    },
    {
        "category": "CONTROL_DETERMINISTIC",
        "id": "control_code",
        "prompt": "Escribí una función en Python para invertir una lista enlazada simple de forma iterativa."
    },
    {
        "category": "CROSS_DOMAIN_EPISTEMIC",
        "id": "boltzmann_poincare",
        "prompt": "En la controversia de finales del siglo XIX entre Ludwig Boltzmann y Ernst Zermelo sobre la paradoja de recurrencia de Poincaré en termodinámica estadística, ¿cuál fue el contraargumento cinético de Boltzmann respecto al tiempo de recurrencia frente a la edad del universo?"
    },
    {
        "category": "ADVERSARIAL_CROSS_SYNTAX",
        "id": "llvm_arm64_vector",
        "prompt": "En el backend de LLVM para ARM64 / Apple Silicon, ¿cuál es la diferencia técnica entre las instrucciones LDNP y LDP al vectorizar bucles y cómo afecta el hardware prefetcher L1?"
    },
    {
        "category": "LOW_FREQUENCY_GEO_HISTORY",
        "id": "arbitraje_1902",
        "prompt": "En el Laudo Arbitral británico de 1902 entre Argentina y Chile presidido por Thomas Holdich, ¿qué solución se adoptó específicamente para la disputa de la divisoria de aguas en el Lago Lácar?"
    }
]

def run_benchmark():
    logger.info("[INICIO] Benchmark Factual Anti-Alucinaciones para Calibración de EAACD")
    
    # Check server availability
    try:
        health = requests.get("http://localhost:8080/v1/models", timeout=5)
        if health.status_code != 200:
            logger.error("[ERROR] El servidor respondió HTTP %d en /v1/models", health.status_code)
            return
        logger.info("[DATA] Servidor activo. Modelos: %s", health.json().get("data", [{}])[0].get("id"))
    except Exception as e:
        logger.error("[ERROR] No se pudo conectar a SuperMLX en localhost:8080: %s", e)
        logger.info("[DECISION] Levantá el servidor con ./super.sh antes de ejecutar esta batería.")
        return

    results = []
    for item in FACTUAL_PROMPTS:
        logger.info("[RUN] Evaluando [%s] %s...", item["category"], item["id"])
        payload = {
            "model": "claude-3-5-sonnet-latest",
            "messages": [{"role": "user", "content": item["prompt"]}],
            "max_tokens": 400,
            "temperature": 0.1,  # Mantenemos determinismo factual para aislar el routing
            "stream": False
        }
        t0 = time.time()
        try:
            r = requests.post(URL, json=payload, timeout=180)
            elapsed = time.time() - t0
            if r.status_code == 200:
                resp_json = r.json()
                content = resp_json["choices"][0]["message"]["content"]
                usage = resp_json.get("usage", {})
                logger.info("[RESULT] ✓ Completado en %.2fs | tokens: %s", elapsed, usage.get("completion_tokens"))
                results.append({
                    "id": item["id"],
                    "category": item["category"],
                    "elapsed": elapsed,
                    "tokens": usage.get("completion_tokens"),
                    "sample": content[:160].replace("\n", " ")
                })
            else:
                logger.error("[ERROR] HTTP %d: %s", r.status_code, r.text[:120])
        except Exception as e:
            logger.error("[ERROR] Excepción durante el request: %s", e)

    logger.info("[RESULT] Batería finalizada: %d/%d casos procesados.", len(results), len(FACTUAL_PROMPTS))

if __name__ == "__main__":
    run_benchmark()
