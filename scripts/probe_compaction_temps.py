#!/usr/bin/env python3
"""
[AI_DIRECTIVE]
ROL: Script de sondeo empírico de temperaturas para Context Compaction en SuperMLX
OBJETIVO: Medir fidelidad fáctica, concisión y coherencia estructural de resúmenes
          en T=[0.10, 0.20, 0.30, 0.40] evaluados por Gemini 3.8 Flash como juez.
ENTRADAS: Servidor local SuperMLX en http://localhost:8080 y Gemini API
SALIDAS: Métricas de calidad, recuento de palabras, latencia y tabla resumen
REGLAS INVIOLABLES:
- Prohibido suponer calidad: evaluación ciega y automatizada con Gemini 3.8 Flash
- Min-P fijado en 0.04 y Top-P en 1.0 (calibrados)
- Compaction corre sin thinking (enable_thinking=False)
- Logging estándar con prefijos [INICIO][CONFIG][DATA][CALC][DECISION][RESULT][ERROR]
"""

import json
import logging
import os
import re
import sys
import time
from typing import Any, Dict, List, Tuple

import requests
from dotenv import load_dotenv

from google import genai
from google.genai import types

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("probe_compaction")

LOCAL_URL = os.getenv("SUPERMLX_URL", "http://localhost:8080") + "/v1/chat/completions"
GEMINI_MODEL = os.getenv("GEMINI_EVAL_MODEL", "gemini-3.8-flash")

# Conversación técnica real de prueba para compactar
CONVERSATION_TRANSCRIPT = """
User: El servidor en Metal está corriendo a 35 TPS en vez de 54 TPS. ¿Por qué tan lento y por qué el código a veces comete typos tontos?
Assistant: Revisé el archivo .env y encontré un error crítico: DEFAULT_MIN_P estaba configurado en 0.8 en vez de 0.05. Estaba descartando el 80% de los tokens plausibles, dejando entrar opciones subóptimas.
User: Qué pasa con los expertos del MoE?
Assistant: En supermlx/expert_cache.py línea 652, cuando la capacidad era menor a 256, los expertos faltantes tenían sus scores anulados con scores * hit_mask, amputando el cómputo. Al fijar MOE_EXPERT_CAPACITY=256 en residencia completa, recuperamos el 100% de la calidad.
User: Y la velocidad?
Assistant: Al corregir Min-P a 0.04/0.05, el prefill y decode alcanzaron 54.7 TPS estables con 18.29GB de memoria Metal.
User: Por qué anoche el calibrador se quedó clavado en 50.000 tokens en el prompt de thinking?
Assistant: El prompt pedía parsear etiquetas <think>. Cuando el modelo razonó sobre esa etiqueta, generó el token 248069 (</think>). El ThinkingTracker creyó falsamente que había salido del thinking y pasó a modo respuesta. Como DEFAULT_MAX_TOKENS estaba en 262144, siguió decodificando sin parar.
User: Cuál es el plan para solucionarlo?
Assistant: Adoptar Dual-Phase Sampling: pensar a T=0.50 con Min-P=0.04 para exploración algorítmica profunda, y en cuanto emite </think>, conmutar la respuesta a T=0.10 para precisión quirúrgica en código y tool calls. Para compactación, apagar el thinking y definir COMPACTION_TEMPERATURE en .env.
User: Perfecto, avancemos con eso.
"""

PROMPT = f"""Tu tarea es resumir la siguiente sesión de desarrollo de software en un informe conciso en Markdown.

CONVERSACIÓN A RESUMIR:
{CONVERSATION_TRANSCRIPT}

INSTRUCCIONES OBLIGATORIAS:
1. Usá exactamente estas 3 secciones:
   ### 1. Decisiones y Cambios Arquitectónicos
   ### 2. Bugs Identificados y Causa Raíz
   ### 3. Próximos Pasos Pendientes
2. Límite estricto: Máximo 200 palabras en total.
3. Fidelidad fáctica 100%: Preservá nombres exactos de variables, archivos y números (ej. DEFAULT_MIN_P, 0.04, 54.7 TPS, 18.29GB, MOE_EXPERT_CAPACITY=256, expert_cache.py, token 248069).
4. Cero alucinación o agregados que no estén en la conversación.
"""

TEMPERATURES = [0.10, 0.20, 0.30, 0.40]
MIN_P = 0.04
TOP_P = 1.0

EVAL_JUDGE_PROMPT = """Sos un auditor experto de compresión de contexto para agentes de IA.

Tu tarea es evaluar la calidad de un RESUMEN DE COMPACTACIÓN comparándolo contra la CONVERSACIÓN ORIGINAL.

## Conversación Original:
<original>
{original}
</original>

## Resumen Candidato (generado con temperatura={temp}):
<summary>
{summary}
</summary>

## Instrucciones de Evaluación:
Puntuá de 1.0 a 10.0 en tres dimensiones:
1. **fidelidad_factica**: ¿El resumen contiene SOLO hechos verdaderos de la conversación? ¿Preserva los nombres de archivos, variables y números exactos sin inventar nada?
2. **concision**: ¿Sintetiza la esencia sin rodeos respetando el límite de 200 palabras? (Penalizá severamente si supera 200 palabras).
3. **estructura_coherencia**: ¿Las 3 secciones solicitadas están presentes? ¿El texto fluye lógicamente sin oraciones rotas ni repeticiones?

Devolvé ÚNICAMENTE un JSON válido con este formato:
```json
{{
  "fidelidad_factica": 9.5,
  "concision": 9.0,
  "estructura_coherencia": 9.5,
  "word_count": 145,
  "justificacion": "Explicación breve de fortalezas y debilidades"
}}
```
"""


def count_words(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def run_local_compaction(temp: float) -> Tuple[str, float]:
    payload = {
        "model": "qwen3-35b",
        "messages": [{"role": "user", "content": PROMPT}],
        "temperature": temp,
        "top_p": TOP_P,
        "min_p": MIN_P,
        "enable_thinking": False,
        "max_tokens": 1024,
        "stream": False,
    }
    start_time = time.time()
    resp = requests.post(LOCAL_URL, json=payload, timeout=60)
    resp.raise_for_status()
    elapsed = time.time() - start_time
    data = resp.json()
    summary = data["choices"][0]["message"]["content"].strip()
    return summary, elapsed


def evaluate_with_gemini(client: genai.Client, temp: float, summary: str) -> Dict[str, Any]:
    eval_text = EVAL_JUDGE_PROMPT.format(
        original=CONVERSATION_TRANSCRIPT,
        temp=temp,
        summary=summary,
    )
    res = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=eval_text,
        config=types.GenerateContentConfig(
            temperature=0.1,
            response_mime_type="application/json",
        ),
    )
    raw = res.text.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
    if raw.endswith("```"):
        raw = raw.rsplit("\n", 1)[0]
    return json.loads(raw)


def main() -> None:
    logger.info("[CONFIG] Verificando conexión a SuperMLX en %s...", LOCAL_URL)
    try:
        requests.get(LOCAL_URL.replace("/v1/chat/completions", "/"), timeout=5)
        logger.info("[CONFIG] Servidor local conectado OK.")
    except Exception as e:
        logger.error("[ERROR] No se pudo conectar a SuperMLX: %s", e)
        sys.exit(1)

    client = genai.Client()

    print("\n" + "=" * 85)
    print("🔬 PROBANDO TEMPERATURAS PARA CONTEXT COMPACTION (MIN-P=0.04, SIN THINKING)")
    print("=" * 85 + "\n")

    results = []

    for t in TEMPERATURES:
        logger.info("[INICIO] Probando Compaction con T=%.2f...", t)
        try:
            summary, elapsed = run_local_compaction(t)
            w_count = count_words(summary)
            logger.info("[DATA] T=%.2f generó %d palabras en %.1fs", t, w_count, elapsed)

            logger.info("[DATA] Evaluando con %s...", GEMINI_MODEL)
            eval_data = evaluate_with_gemini(client, t, summary)

            fid = float(eval_data.get("fidelidad_factica", 5.0))
            con = float(eval_data.get("concision", 5.0))
            est = float(eval_data.get("estructura_coherencia", 5.0))

            # Penalización por exceder 200 palabras
            if w_count > 200:
                con = max(1.0, con - (w_count - 200) * 0.1)

            # Score compuesto: 50% fidelidad, 25% concisión, 25% estructura
            composite = fid * 0.50 + con * 0.25 + est * 0.25

            logger.info(
                "[CALC] T=%.2f -> Fid=%.1f Con=%.1f Est=%.1f | Score=%.3f | Palabras=%d | %s",
                t, fid, con, est, composite, w_count, eval_data.get("justificacion", "")[:70]
            )

            results.append({
                "temp": t,
                "elapsed": round(elapsed, 1),
                "words": w_count,
                "fidelidad": fid,
                "concision": con,
                "estructura": est,
                "score": round(composite, 3),
                "summary": summary,
                "justificacion": eval_data.get("justificacion", ""),
            })

            time.sleep(1)

        except Exception as e:
            logger.error("[ERROR] Falló corrida con T=%.2f: %s", t, e)

    # ── Tabla Resumen ─────────────────────────────────────────────────────────
    print("\n" + "=" * 88)
    print("📊 RESULTADOS DEFINITIVOS DE TEMPERATURA PARA COMPACTION")
    print("=" * 88)
    print(f"{'Temp':<6} | {'Score':<8} | {'Fidelidad':<10} | {'Concisión':<10} | {'Estructura':<11} | {'Palabras':<9} | {'Tiempo':<8}")
    print("-" * 88)

    best_item = max(results, key=lambda x: x["score"]) if results else None

    for r in results:
        is_best = " 🏆 (ÓPTIMO)" if r == best_item else ""
        print(
            f"{r['temp']:<6.2f} | {r['score']:<8.3f} | {r['fidelidad']:<10.1f} | "
            f"{r['concision']:<10.1f} | {r['estructura']:<11.1f} | {r['words']:<9} | {r['elapsed']:<6.1f}s{is_best}"
        )
    print("=" * 88)

    if best_item:
        print(f"\n[DECISION] TEMPERATURA GANADORA PARA COMPACTION: T = {best_item['temp']:.2f} (Score: {best_item['score']})")
        print(f"Justificación del juez: {best_item['justificacion']}")
        print("\n--- RESUMEN GENERADO POR LA TEMPERATURA GANADORA ---")
        print(best_item['summary'])
        print("----------------------------------------------------\n")


if __name__ == "__main__":
    main()
