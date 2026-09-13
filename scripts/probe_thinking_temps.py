#!/usr/bin/env python3
"""
[AI_DIRECTIVE]
ROL: Script de sondeo empírico de temperaturas de thinking para SuperMLX
OBJETIVO: Medir tokens de razonamiento, ciclos de duda ("Wait..."), tiempo de inferencia
          y corrección del código (ejecución real de asserts) en varias temperaturas.
ENTRADAS: Servidor local SuperMLX en http://localhost:8080
SALIDAS: Métricas comparativas, logs detallados y tabla resumen en consola
REGLAS INVIOLABLES:
- Prohibido suponer corrección: ejecutar el código con subprocess para verificar asserts reales
- Logging estándar con prefijos [INICIO][CONFIG][DATA][CALC][DECISION][RESULT][ERROR]
- Min-P fijado en 0.04 y Top-P fijado en 1.0 (calibrados en Palanca 2)
"""

import json
import logging
import os
import re
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("probe_thinking")

LOCAL_URL = os.getenv("SUPERMLX_URL", "http://localhost:8080") + "/v1/chat/completions"

PROMPT = (
    "Implementá en Python 3 una clase `SegmentTree` con Lazy Propagation para updates de rango en suma "
    "(range sum query + range addition update).\n\n"
    "Requisitos obligatorios:\n"
    "1. Manejo riguroso de propagación lazy para no perder actualizaciones pendientes.\n"
    "2. Métodos: `build(arr: list[int])`, `update_range(l: int, r: int, val: int)`, `query_range(l: int, r: int) -> int` indexados en 0 (inclusivos [l, r]).\n"
    "3. Tipado estricto (Type hints) y docstrings.\n"
    "4. Un bloque de prueba con `assert` que valide al menos 3 updates y 3 queries consecutivas con solapamiento parcial de rangos.\n\n"
    "El código debe ser auto-contenido y ejecutable directamente. Si todos los asserts pasan, imprimí al final: print('ALL_ASSERTS_PASSED')."
)

TEMPERATURES_TO_TEST = [0.40, 0.50, 0.60, 0.70]
MIN_P = 0.04
TOP_P = 1.0

# Expresiones regulares para detectar ciclos de arrepentimiento / duda en el reasoning
BACKTRACK_REGEX = re.compile(
    r"\b(wait|actually|hold on|let me rethink|let me check|wait a second|wait a minute|no, that's wrong|that's incorrect|correction)\b",
    re.IGNORECASE,
)


def extract_thinking_and_code(raw_text: str) -> Tuple[str, str, str]:
    """Separa el razonamiento, el texto de respuesta y el bloque de código Python."""
    thinking = ""
    response = raw_text

    # Extraer <think>...</think>
    think_match = re.search(r"<think>(.*?)</think>", raw_text, re.DOTALL | re.IGNORECASE)
    if think_match:
        thinking = think_match.group(1).strip()
        response = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL | re.IGNORECASE).strip()
    elif "</think>" in raw_text:
        parts = raw_text.split("</think>", 1)
        thinking = parts[0].replace("<think>", "").strip()
        response = parts[1].strip()

    # Extraer bloque de código python
    code_match = re.search(r"```python\s*(.*?)\s*```", response, re.DOTALL | re.IGNORECASE)
    code = code_match.group(1).strip() if code_match else response

    return thinking, response, code


def test_code_execution(code: str, timeout_sec: int = 10) -> Tuple[bool, str]:
    """Ejecuta el código generado en un subproceso para validar que compile y pasen los asserts."""
    if not code:
        return False, "No code extracted"
    try:
        res = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        if res.returncode == 0:
            if "ALL_ASSERTS_PASSED" in res.stdout:
                return True, "ALL_ASSERTS_PASSED"
            return True, "Executed with code 0 (without explicit print)"
        return False, f"Exit code {res.returncode}: {res.stderr.strip()[:200]}"
    except subprocess.TimeoutExpired:
        return False, f"Execution timed out ({timeout_sec}s)"
    except Exception as e:
        return False, f"Execution error: {str(e)}"


def run_probe(temp: float) -> Dict[str, Any]:
    """Ejecuta una inferencia completa para una temperatura dada y mide métricas."""
    payload = {
        "model": "qwen3-35b",
        "messages": [{"role": "user", "content": PROMPT}],
        "temperature": temp,
        "top_p": TOP_P,
        "min_p": MIN_P,
        "enable_thinking": True,
        "max_tokens": 4096,
        "stream": False,
    }

    logger.info("[INICIO] Iniciando corrida con T=%.2f (Min-P=%.2f, Top-P=%.2f)...", temp, MIN_P, TOP_P)
    start_time = time.time()

    resp = requests.post(LOCAL_URL, json=payload, timeout=300)
    resp.raise_for_status()
    elapsed_total = time.time() - start_time

    data = resp.json()
    raw_content = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})

    thinking, response, code = extract_thinking_and_code(raw_content)

    # Contar dudas en el thinking
    backtrack_matches = BACKTRACK_REGEX.findall(thinking)
    backtrack_count = len(backtrack_matches)

    # Estimar tokens de thinking y response (por aproximación de caracteres o de usage si existe)
    # Un token son ~4 caracteres en inglés/código
    approx_think_tokens = len(thinking) // 4
    approx_resp_tokens = len(response) // 4
    total_tokens = usage.get("completion_tokens", approx_think_tokens + approx_resp_tokens)

    # Ejecutar validación real del código
    code_ok, code_msg = test_code_execution(code)

    logger.info(
        "[RESULT] T=%.2f -> Total=%d tokens (Think~%d, Resp~%d) | Dudas=%d | Duración=%.1fs | Código OK: %s (%s)",
        temp, total_tokens, approx_think_tokens, approx_resp_tokens,
        backtrack_count, elapsed_total, code_ok, code_msg
    )

    return {
        "temp": temp,
        "total_tokens": total_tokens,
        "think_tokens": approx_think_tokens,
        "resp_tokens": approx_resp_tokens,
        "backtrack_count": backtrack_count,
        "backtrack_samples": backtrack_matches[:5],
        "elapsed_sec": round(elapsed_total, 1),
        "code_ok": code_ok,
        "code_msg": code_msg,
        "thinking_snippet": thinking[:300] + ("..." if len(thinking) > 300 else ""),
    }


def main() -> None:
    logger.info("[CONFIG] Verificando conexión a %s...", LOCAL_URL)
    try:
        r = requests.get(LOCAL_URL.replace("/v1/chat/completions", "/"), timeout=5)
        logger.info("[CONFIG] Servidor local conectado OK.")
    except Exception as e:
        logger.error("[ERROR] No se pudo conectar a SuperMLX: %s", e)
        sys.exit(1)

    results = []

    print("\n" + "=" * 80)
    print("🚀 PROBANDO TEMPERATURAS DE THINKING PARA SEGMENT TREE (MIN-P=0.04)")
    print("=" * 80 + "\n")

    for t in TEMPERATURES_TO_TEST:
        try:
            res = run_probe(t)
            results.append(res)
            # Pequeña pausa de 2 segundos para estabilizar Metal
            time.sleep(2)
        except Exception as e:
            logger.error("[ERROR] Falló corrida con T=%.2f: %s", t, e)
            results.append({
                "temp": t,
                "total_tokens": 0,
                "think_tokens": 0,
                "resp_tokens": 0,
                "backtrack_count": -1,
                "elapsed_sec": 0,
                "code_ok": False,
                "code_msg": f"CRASH: {str(e)}",
            })

    # Imprimir Tabla Resumen Final
    print("\n" + "=" * 92)
    print("📊 RESULTADOS COMPARATIVOS DEFINITIVOS DE THINKING")
    print("=" * 92)
    print(f"{'Temp':<8} | {'Think Toks':<12} | {'Dudas (Wait)':<14} | {'Asserts OK':<12} | {'Tiempo':<10} | {'Status'}")
    print("-" * 92)

    for r in results:
        status_icon = "✅ PASÓ" if r["code_ok"] else "❌ FALLÓ"
        dudas_str = str(r['backtrack_count']) if r['backtrack_count'] >= 0 else "N/A"
        print(
            f"{r['temp']:<8.2f} | ~{r['think_tokens']:<11} | {dudas_str:<14} | "
            f"{status_icon:<12} | {r['elapsed_sec']:<8.1f}s | {r['code_msg'][:25]}"
        )
    print("=" * 92 + "\n")


if __name__ == "__main__":
    main()
