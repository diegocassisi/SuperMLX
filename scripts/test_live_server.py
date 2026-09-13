import time
import requests
import json

URL = "http://localhost:8080/v1/chat/completions"

# Prompt que fuerza decisiones complejas y código estructurado
payload = {
    "model": "claude-3-5-sonnet-latest",
    "messages": [
        {"role": "user", "content": "Escribí una función en Python para invertir un árbol binario y explicá por qué O(n) es óptimo."}
    ],
    "max_tokens": 100,
    "temperature": 0.5,
    "stream": False
}

print("Midiendo llamada real a SuperMLX...")
t0 = time.time()
r = requests.post(URL, json=payload, timeout=60)
t1 = time.time()
print(f"Status: {r.status_code} en {t1-t0:.2f}s")
if r.status_code == 200:
    content = r.json()["choices"][0]["message"]["content"]
    print("Tokens generados exitosamente. Muestra:")
    print(content[:150])
