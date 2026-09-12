#!/usr/bin/env python3
"""
Test de Validación Live para Dual-Phase Sampling en SuperMLX (:8080)
Envía un prompt de razonamiento + código vía streaming SSE y monitorea
la fase de Thinking y la fase de Response en tiempo real.
"""
import json
import time
import requests
import sys

URL = "http://localhost:8080/v1/chat/completions"

PROMPT = """Implementa una función en Python llamada `solve_trapping_rain_water(height: list[int]) -> int` que resuelva el problema de 'Trapping Rain Water' en O(N) tiempo y O(1) memoria usando el enfoque de dos punteros.
Al final, incluye exactamente tres asserts que verifiquen:
assert solve_trapping_rain_water([0,1,0,2,1,0,1,3,2,1,2,1]) == 6
assert solve_trapping_rain_water([4,2,0,3,2,5]) == 9
assert solve_trapping_rain_water([]) == 0
"""

payload = {
    "model": "unsloth/Qwen3.6-35B-A3B-UD-MLX-3bit",
    "messages": [
        {"role": "user", "content": PROMPT}
    ],
    "max_tokens": 4096,
    "stream": True
}

print(f"📡 Conectando a {URL} para test de Dual-Phase...")
start_time = time.time()
resp = requests.post(URL, json=payload, stream=True, timeout=180)

if resp.status_code != 200:
    print(f"❌ Error HTTP {resp.status_code}: {resp.text}")
    sys.exit(1)

in_thinking = False
thinking_chars = 0
response_chars = 0
full_response = []
thinking_chunks = 0
response_chunks = 0

print("🚀 Stream iniciado...\n")

for line in resp.iter_lines():
    if not line:
        continue
    line_str = line.decode("utf-8")
    if line_str.startswith("data: "):
        data_content = line_str[6:].strip()
        if data_content == "[DONE]":
            break
        try:
            chunk = json.loads(data_content)
            delta = chunk["choices"][0]["delta"]
            
            # Check for reasoning_content (Anthropic/DeepSeek/OpenAI style) or text
            reasoning = delta.get("reasoning_content")
            content = delta.get("content")
            
            if reasoning:
                if not in_thinking:
                    in_thinking = True
                    print("\n🧠 === FASE 1: THINKING (T=0.50 | Min-P=0.04) ===")
                sys.stdout.write(reasoning)
                sys.stdout.flush()
                thinking_chars += len(reasoning)
                thinking_chunks += 1
            elif content:
                if in_thinking:
                    in_thinking = False
                    print("\n\n⚡ === FASE 2: RESPONSE DETERMINISTA (T=0.10 | Min-P=0.04) ===")
                elif not full_response:
                    print("\n⚡ === FASE 2: RESPONSE DETERMINISTA (T=0.10 | Min-P=0.04) ===")
                sys.stdout.write(content)
                sys.stdout.flush()
                full_response.append(content)
                response_chars += len(content)
                response_chunks += 1
        except Exception as e:
            continue

elapsed = time.time() - start_time
print("\n" + "=" * 60)
print(f"✅ Generación completada en {elapsed:.2f}s")
print(f"🧠 Thinking: {thinking_chars} caracteres en {thinking_chunks} chunks")
print(f"⚡ Response: {response_chars} caracteres en {response_chunks} chunks")

# Validar que el código generado se ejecute y pase los asserts
code_text = "".join(full_response)
if "def solve_trapping_rain_water" in code_text:
    print("\n🧪 Ejecutando asserts del código generado en Python...")
    # Extraer bloque de código python
    import re
    match = re.search(r"```python\s*(.*?)\s*```", code_text, re.DOTALL)
    code_to_exec = match.group(1) if match else code_text
    
    # Agregar los asserts para verificar
    test_suite = code_to_exec + """
assert solve_trapping_rain_water([0,1,0,2,1,0,1,3,2,1,2,1]) == 6
assert solve_trapping_rain_water([4,2,0,3,2,5]) == 9
assert solve_trapping_rain_water([]) == 0
print("🎉 TODOS LOS ASSERTS PASARON AL 100%!")
"""
    try:
        exec(test_suite, {})
    except Exception as err:
        print(f"❌ Falló la ejecución del código: {err}")
else:
    print("⚠️ No se encontró la función solve_trapping_rain_water en la respuesta.")
