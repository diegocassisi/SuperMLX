#!/usr/bin/env python3
"""
stress_test.py — Prueba los 3 bugs corregidos en start-llm4.py

USO:
    # Desde el directorio MLXTurboQuant, con el servidor corriendo:
    python tools/stress_test.py [--test all|ttl|concurrent|memory]

TESTS:
    ttl        → Verifica que el Kripper slot no muera por TTL (Bug #1)
    concurrent → Simula EMBEDDED + MAIN simultáneos (Bug #2)
    memory     → Detecta presión de GPU con muchos tokens (Bug #3)
"""

import argparse
import json
import sys
import threading
import time
from datetime import datetime

import requests

# ── Config ──────────────────────────────────────────────────────────────────
LLM_BASE_URL = "http://127.0.0.1:8080"
MODEL = "mlx-community/Qwen3.5-9B-4bit"
TIMEOUT = 180  # segundos máx por request

# Prompt de sistema minimalista para MAIN (simula context largo)
MAIN_SYSTEM = "You are a helpful assistant. Respond in one sentence."

# Prompt corto para EMBEDDED (simula compaction agent con 2 tools)
EMBEDDED_SYSTEM = "You are a memory agent. Store facts concisely."

COLORS = {
    "green": "\033[92m", "red": "\033[91m", "yellow": "\033[93m",
    "cyan": "\033[96m", "bold": "\033[1m", "reset": "\033[0m",
}

def c(color, text):
    return f"{COLORS[color]}{text}{COLORS['reset']}"

def ts():
    return datetime.now().strftime("%H:%M:%S")

def send_request(
    messages, tools=None, label="req", timeout=TIMEOUT
) -> dict:
    """Envía un chat/completions y devuelve métricas."""
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "max_tokens": 50,
    }
    if tools:
        payload["tools"] = tools

    t0 = time.time()
    try:
        resp = requests.post(
            f"{LLM_BASE_URL}/v1/chat/completions",
            json=payload,
            timeout=timeout,
        )
        elapsed = time.time() - t0
        if resp.status_code != 200:
            return {"ok": False, "label": label, "elapsed": elapsed,
                    "error": f"HTTP {resp.status_code}: {resp.text[:200]}"}
        data = resp.json()
        return {"ok": True, "label": label, "elapsed": elapsed, "data": data}
    except requests.exceptions.Timeout:
        return {"ok": False, "label": label, "elapsed": timeout, "error": "TIMEOUT"}
    except Exception as e:
        return {"ok": False, "label": label, "elapsed": time.time() - t0, "error": str(e)}


def print_result(result, expected_fast=True):
    ok = result["ok"]
    label = result["label"]
    elapsed = result["elapsed"]
    fast = elapsed < 15

    status = c("green", "✅") if ok else c("red", "❌")
    speed = c("green", f"{elapsed:.1f}s") if fast else c("yellow" if elapsed < 40 else "red", f"{elapsed:.1f}s")

    print(f"  {status} [{ts()}] {label:30s} | {speed}", end="")
    if not ok:
        print(f" | {c('red', result.get('error', '?'))}")
    else:
        print()


# ═══════════════════════════════════════════════════════════════════════════
# TEST 1 — TTL: Kripper slot sobrevive inactividad
# ═══════════════════════════════════════════════════════════════════════════
def test_ttl(idle_minutes=2):
    """
    Simula inactividad y verifica que el 2do request sigue siendo rápido.

    Con TTL=1800s y el bug #1, el slot se evictaría a los 30min.
    Para testing rápido, reducir el TTL del servidor a 60s temporalmente.

    NOTA: Para probar el bug real, poner idle_minutes=35 y arrancar el
    servidor con TTL_SECONDS=60 (variable de entorno si está soportada).
    """
    print(c("bold", f"\n{'='*60}"))
    print(c("bold", f"TEST 1 — TTL: Kripper slot sobrevive {idle_minutes}min idle"))
    print(c("bold", f"{'='*60}"))

    msgs_main = [
        {"role": "system", "content": MAIN_SYSTEM},
        {"role": "user", "content": "Hola, ¿cómo estás?"},
    ]

    print(f"\n  [{ts()}] Request inicial (warm-up)...")
    r1 = send_request(msgs_main, label="warm-up request")
    print_result(r1)

    if not r1["ok"]:
        print(c("red", "  FAIL: servidor no responde"))
        return False

    print(f"\n  [{ts()}] Esperando {idle_minutes} minutos (simulando inactividad)...")
    for remaining in range(idle_minutes * 60, 0, -10):
        print(f"  ⏳ {remaining}s restantes...    ", end="\r")
        time.sleep(min(10, remaining))
    print()

    print(f"\n  [{ts()}] Request post-idle (debe ser rápido si Kripper slot sobrevivió)...")
    msgs_main.append({"role": "assistant", "content": "¡Bien, gracias!"})
    msgs_main.append({"role": "user", "content": "¿Cuál es la capital de Francia?"})
    r2 = send_request(msgs_main, label="post-idle request")
    print_result(r2)

    survived = r2["ok"] and r2["elapsed"] < 20
    print(f"\n  {'✅ PASS' if survived else '❌ FAIL'}: Kripper slot {'sobrevivió' if survived else 'fue evictado'} el idle")
    print(f"  Warm: {r1['elapsed']:.1f}s → Post-idle: {r2['elapsed']:.1f}s")
    return survived


# ═══════════════════════════════════════════════════════════════════════════
# TEST 2 — Concurrent: EMBEDDED + MAIN simultáneos no se bloquean
# ═══════════════════════════════════════════════════════════════════════════
def test_concurrent(n_embedded=3):
    """
    Lanza N requests EMBEDDED simultáneos mientras MAIN está procesando.
    Verifica que los embedded no cuelguen indefinidamente.
    """
    print(c("bold", f"\n{'='*60}"))
    print(c("bold", f"TEST 2 — Concurrent: {n_embedded} EMBEDDED + 1 MAIN"))
    print(c("bold", f"{'='*60}"))

    # Tools para simular embedded (2 tools como compaction agent)
    embedded_tools = [
        {"type": "function", "function": {
            "name": "read", "description": "Read a file",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}
        }},
        {"type": "function", "function": {
            "name": "write", "description": "Write a file",
            "parameters": {"type": "object", "properties": {
                "path": {"type": "string"}, "content": {"type": "string"}
            }}
        }},
    ]

    results = {}
    threads = []

    def run(label, messages, tools=None):
        r = send_request(messages, tools=tools, label=label, timeout=120)
        results[label] = r

    # MAIN: request largo
    main_msgs = [
        {"role": "system", "content": MAIN_SYSTEM},
        {"role": "user", "content": "Enumera los 10 países más grandes del mundo."},
    ]

    # EMBEDDED: requests cortos con 2 tools (simulan compaction agent)
    embedded_msgs = [
        {"role": "system", "content": EMBEDDED_SYSTEM},
        {"role": "user", "content": "Store: session started at " + ts()},
    ]

    print(f"\n  [{ts()}] Lanzando {n_embedded + 1} requests concurrentes...")

    t_main = threading.Thread(target=run, args=("MAIN", main_msgs), daemon=True)
    threads.append(t_main)

    for i in range(n_embedded):
        t = threading.Thread(
            target=run,
            args=(f"EMBEDDED-{i+1}", embedded_msgs, embedded_tools),
            daemon=True,
        )
        threads.append(t)

    t0 = time.time()
    for t in threads:
        t.start()
        time.sleep(0.2)  # stagger ligeramente

    for t in threads:
        t.join(timeout=130)

    total = time.time() - t0

    print(f"\n  Resultados ({total:.1f}s total):")
    all_ok = True
    for label, r in sorted(results.items()):
        print_result(r)
        if not r["ok"]:
            all_ok = False

    timeouts = sum(1 for r in results.values() if "TIMEOUT" in r.get("error", ""))
    print(f"\n  {'✅ PASS' if all_ok else '❌ FAIL'}: {len(results)}/{n_embedded+1} completados | {timeouts} timeouts")
    return all_ok


# ═══════════════════════════════════════════════════════════════════════════
# TEST 3 — Memory: muchos requests grandes no causan OOM
# ═══════════════════════════════════════════════════════════════════════════
def test_memory(n_requests=5, padding_kb=2):
    """
    Envía N requests con contexto grande (padded) en secuencia.
    Si hay memory leak, el servidor se cae o se vuelve muy lento.
    """
    print(c("bold", f"\n{'='*60}"))
    print(c("bold", f"TEST 3 — Memory: {n_requests} requests grandes secuenciales"))
    print(c("bold", f"{'='*60}"))

    padding = "A " * (padding_kb * 500)  # ~1KB por 500 palabras "A "
    times = []

    for i in range(n_requests):
        msgs = [
            {"role": "system", "content": MAIN_SYSTEM},
            {"role": "user", "content": f"Request {i+1}: {padding[:200 * (i+1)]} ¿Cuánto es 2+2?"},
        ]
        label = f"large-req-{i+1}"
        print(f"  [{ts()}] {label}...", end=" ", flush=True)
        r = send_request(msgs, label=label, timeout=TIMEOUT)
        times.append(r["elapsed"])
        status = c("green", f"✅ {r['elapsed']:.1f}s") if r["ok"] else c("red", f"❌ {r.get('error','?')}")
        print(status)

        if not r["ok"]:
            print(c("red", f"  FAIL en request {i+1} — posible OOM"))
            return False

    # Si el modelo mantiene velocidad estable, no hay leak
    drift = times[-1] - times[0]
    print(f"\n  Tiempos: {' → '.join(f'{t:.1f}s' for t in times)}")
    print(f"  Drift: {drift:+.1f}s (< 10s = bueno)")
    success = abs(drift) < 10 and all(t < 60 for t in times)
    print(f"\n  {'✅ PASS' if success else '❌ FAIL'}: {'Sin leak de memoria' if success else 'Posible leak o lentitud creciente'}")
    return success


# ═══════════════════════════════════════════════════════════════════════════
# TEST 4 — Restart: cache persiste en disco
# ═══════════════════════════════════════════════════════════════════════════
def test_restart_hint():
    """No automatizable — muestra instrucciones manuales."""
    print(c("bold", f"\n{'='*60}"))
    print(c("bold", "TEST 4 — Restart: cache persiste (manual)"))
    print(c("bold", f"{'='*60}"))
    print("""
  1. Arrancá el servidor con CACHE_PERSIST_PATH=logs/warmup_cache.safetensors
  2. Enviá un request (el servidor guarda el cache al disco)
  3. Parás el servidor (Ctrl+C)
  4. Reiniciás el servidor
  5. Primer request → debe ser < 6s (cache desde disco)

  En los logs debería ver:
    💾 Warmup: disk cache loaded | layers=8 | tokens=XXXX
    📌 Kripper slot PINNED
    [CACHE] match_type=shorter | matched_prefix=XXXX/XXXX
    🟢 Cache: 98%+
""")


# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stress test start-llm4.py")
    parser.add_argument(
        "--test",
        choices=["all", "ttl", "concurrent", "memory", "restart"],
        default="all",
        help="Qué test correr",
    )
    parser.add_argument(
        "--idle-min",
        type=int,
        default=2,
        help="Minutos de idle para test TTL (default: 2, usa 35 para test real de 1800s TTL)",
    )
    args = parser.parse_args()

    # Verificar que el servidor esté up
    try:
        r = requests.get(f"{LLM_BASE_URL}/v1/models", timeout=5)
        print(c("green", f"✅ Servidor online en {LLM_BASE_URL}"))
    except Exception:
        print(c("red", f"❌ Servidor no responde en {LLM_BASE_URL}"))
        print(c("yellow", "   Arrancar con: WARMUP_PROMPT_FILE=warmup_seed.txt CACHE_PERSIST_PATH=logs/warmup_cache.safetensors FORCE_TEXT_MODE=true MAX_KV_SIZE=32768 PROMPT_CACHE_MAX_ENTRIES_GLOBAL=2 python start-llm4.py"))
        sys.exit(1)

    results = {}

    if args.test in ("all", "concurrent"):
        results["concurrent"] = test_concurrent(n_embedded=3)

    if args.test in ("all", "memory"):
        results["memory"] = test_memory(n_requests=4)

    if args.test in ("all", "ttl"):
        results["ttl"] = test_ttl(idle_minutes=args.idle_min)

    if args.test in ("all", "restart"):
        test_restart_hint()

    print(c("bold", f"\n{'='*60}"))
    print(c("bold", "RESUMEN"))
    print(c("bold", f"{'='*60}"))
    for name, passed in results.items():
        icon = c("green", "PASS ✅") if passed else c("red", "FAIL ❌")
        print(f"  {name:20s} → {icon}")
    print()
