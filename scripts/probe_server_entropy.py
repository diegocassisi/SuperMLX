import requests
import json
import time

URL = "http://localhost:8080/v1/chat/completions"

prompt = "Explica brevemente en 2 parrafos por que el cielo es azul y que fenomeno optico interviene."

payload = {
    "model": "claude-3-5-sonnet-latest",
    "messages": [
        {"role": "user", "content": prompt}
    ],
    "max_tokens": 150,
    "temperature": 0.6,
    "stream": False
}

print("Enviando request a SuperMLX para recolectar métricas de generación...")
start = time.time()
resp = requests.post(URL, json=payload, timeout=60)
elapsed = time.time() - start

print(f"Status: {resp.status_code} en {elapsed:.2f}s")
if resp.status_code == 200:
    data = resp.json()
    print("Respuesta recibida:", data["choices"][0]["message"]["content"][:200], "...")
else:
    print("Error:", resp.text)
