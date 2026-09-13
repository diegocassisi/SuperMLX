import time
import requests
import json

URL = "http://localhost:8080/v1/chat/completions"

PROMPTS = [
    # 1. Código estricto (baja entropía esperada en sintaxis)
    {"id": "CODING", "prompt": "Implementá en Python una función recursiva para calcular el factorial de N con validación de tipos y doctests."},
    # 2. Razonamiento lógico / algorítmico (entropía media)
    {"id": "REASONING", "prompt": "¿Por qué un árbol binario de búsqueda balanceado garantiza búsquedas en O(log n) mientras que uno degenerado cae a O(n)? Explicá con un ejemplo concreto."},
    # 3. Hechos y QA factual (donde suele haber dudas de hechos)
    {"id": "FACTUAL", "prompt": "¿Cuáles son las 3 diferencias arquitectónicas fundamentales entre un Transformer denso estándar y una red Mixture of Experts (MoE) con shared experts?"},
    # 4. Texto abierto / creativo (alta entropía de vocabulario legítima)
    {"id": "OPEN_TEXT", "prompt": "Escribí el inicio de una historia de ciencia ficción donde una civilización encuentra una cápsula espacial con tecnología retro de los años 80."}
]

print("=" * 65)
print("CORRIENDO BATERÍA DE CALIBRACIÓN DE ENTROPÍA EN SUPERMLX")
print("Modelo activo:", "unsloth/Qwen3.6-35B-A3B-UD-MLX-3bit")
print("=" * 65)

for p in PROMPTS:
    payload = {
        "model": "claude-3-5-sonnet-latest",
        "messages": [{"role": "user", "content": p["prompt"]}],
        "max_tokens": 120,
        "temperature": 0.5,
        "stream": False
    }
    print(f"\n[TEST] {p['id']}...")
    t0 = time.time()
    try:
        r = requests.post(URL, json=payload, timeout=90)
        t1 = time.time()
        if r.status_code == 200:
            print(f" -> OK en {t1-t0:.2f}s | respuesta recibida.")
        else:
            print(f" -> ERROR: HTTP {r.status_code}: {r.text[:100]}")
    except Exception as e:
        print(f" -> ERROR EXCEPCION: {e}")

print("\n" + "=" * 65)
print("Batería completada. Revisando métricas de entropía...")
