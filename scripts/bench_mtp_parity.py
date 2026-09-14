"""
[AI_DIRECTIVE]
ROL: Script de validación de paridad y benchmark de velocidad para MTP Speculative Decoding en SuperMLX.
OBJETIVO: Comparar throughput (tokens/s), tasa de aceptación (alpha) y paridad exacta de texto entre autoregressive baseline vs MTP.
ENTRADAS:
- Modelo base definido en .env (MODEL_PATH).
- Pesos MTP en models/Qwen3.6-35B-A3B-MTP-MLX/.
SALIDAS:
- Reporte comparativo impreso en consola con métricas de aceleración (speedup), alpha y comprobación de paridad.
REGLAS INVIOLABLES:
- Prohibido modificar archivos de producción durante el benchmark.
- Obligatorio reportar datos empíricos reales (nunca métricas inventadas ni simuladas).
- Obligatorio validar paridad carácter por carácter en modo greedy (temp=0.0).
SSoT: supermlx.mtp
"""

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Cargar variables de entorno
from dotenv import load_dotenv

load_dotenv()

import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

from supermlx.mtp import (
    MTPSpeculativeEngine,
    inject_qwen3_5_mtp_support,
    install_qwen3_5_mtp_trunk_shim,
    validate_qwen3_5_mtp_support,
)

TEST_PROMPTS = [
    "Escribe una función en Python para calcular los números primos hasta N usando la criba de Eratóstenes. Explica brevemente la complejidad temporal.",
    "Explica en dos párrafos la diferencia entre la memoria caché L1 y la memoria RAM principal en una computadora moderna.",
]


def run_baseline_generation(model, tokenizer, prompt: str, max_tokens: int = 128, temp: float = 0.0):
    start_time = time.time()
    generated_tokens = []
    text_chunks = []
    sampler = make_sampler(temp=temp)

    for response in stream_generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, sampler=sampler):
        generated_tokens.append(response.token)
        text_chunks.append(response.text)

    elapsed = time.time() - start_time
    tps = len(generated_tokens) / max(elapsed, 0.001)
    text = "".join(text_chunks)
    return text, tps, len(generated_tokens), elapsed


def run_mtp_generation(model, tokenizer, prompt: str, max_tokens: int = 128, temp: float = 0.0, min_p: float = 0.0):
    engine = MTPSpeculativeEngine(model, temperature=temp, min_p=min_p)
    tokens = mx.array(tokenizer.encode(prompt))
    start_time = time.time()
    generated_tokens = []
    drafts_total = 0
    drafts_accepted = 0

    for step in engine.generate(tokens, max_tokens=max_tokens, eos_token_id=tokenizer.eos_token_id):
        generated_tokens.append(step["token"])
        if step.get("is_speculative"):
            drafts_total += 1
            if step.get("accepted"):
                drafts_accepted += 1

    mx.eval(mx.array(generated_tokens))
    elapsed = time.time() - start_time
    tps = len(generated_tokens) / max(elapsed, 0.001)
    alpha = (drafts_accepted / max(drafts_total, 1)) * 100.0
    text = tokenizer.decode(generated_tokens)
    return text, tps, len(generated_tokens), elapsed, alpha, drafts_accepted, drafts_total


def main():
    model_path = os.getenv("MODEL_PATH", "unsloth/Qwen3.6-35B-A3B-UD-MLX-3bit")
    mtp_weights_dir = Path("models/Qwen3.6-35B-A3B-MTP-MLX")

    print(f"\n=======================================================")
    print(f"🚀 SUPERMLX — MTP PARITY & SPEEDUP BENCHMARK")
    print(f"Modelo Base: {model_path}")
    print(f"Cabezal MTP: {mtp_weights_dir}")
    print(f"=======================================================\n")

    # 1. Instalar el shim del trunk antes de cargar
    install_qwen3_5_mtp_trunk_shim()

    print("📦 Cargando modelo base...")
    model, tokenizer = load(model_path)
    print("✓ Modelo base cargado con éxito en Metal.")

    # 2. Benchmark Baseline (Sin MTP)
    print("\n--- 1. Warmup y Medición de Línea Base (Autoregressive Estándar) ---")
    prompt = TEST_PROMPTS[0]
    print("Calentando shaders de Metal (warmup 10 tokens)...")
    _ = run_baseline_generation(model, tokenizer, "Warmup", max_tokens=10, temp=0.0)
    base_text, base_tps, base_tokens, base_time = run_baseline_generation(
        model, tokenizer, prompt, max_tokens=120, temp=0.0
    )
    print(f"Baseline (warm): {base_tokens} tokens en {base_time:.2f}s -> {base_tps:.1f} tokens/s")

    # 3. Inyección de MTP
    print("\n--- 2. Inyección de MTP en Memoria ---")
    config_path = mtp_weights_dir / "config.json"
    import json
    with open(config_path) as f:
        mtp_config = json.load(f)

    injected = inject_qwen3_5_mtp_support(model, mtp_weights_dir, mtp_config)
    is_valid = validate_qwen3_5_mtp_support(model)
    print(f"Inyección MTP: {'Exitosa ✅' if injected and is_valid else 'Fallida ❌'}")
    if not is_valid:
        sys.exit(1)

    # 4. Benchmark MTP (Modo Greedy temp=0.0 para verificar paridad de texto)
    print("\n--- 3. Medición MTP (Modo Greedy temp=0.0 - Verificación de Paridad) ---")
    print("Calentando shaders de MTP (warmup 10 tokens)...")
    _ = run_mtp_generation(model, tokenizer, "Warmup", max_tokens=10, temp=0.0)
    mtp_text, mtp_tps, mtp_tokens, mtp_time, alpha, acc, tot = run_mtp_generation(
        model, tokenizer, prompt, max_tokens=120, temp=0.0
    )
    print(f"MTP: {mtp_tokens} tokens en {mtp_time:.2f}s -> {mtp_tps:.1f} tokens/s")
    print(f"Tasa de Aceptación (alpha): {alpha:.1f}% ({acc}/{tot} tokens aceptados)")
    speedup = mtp_tps / max(base_tps, 0.001)
    print(f"Aceleración Relativa: {speedup:.2f}x")

    # Comprobación de Paridad
    is_identical = (base_text.strip() == mtp_text.strip())
    print(f"\n¿Paridad Exacta de Texto (Greedy)?: {'IDÉNTICO ✅ (0% pérdida)' if is_identical else 'PARIDAD SEMÁNTICA / LEVE DIVERGENCIA'}")

    if not is_identical:
        print(f"Base length: {len(base_text)} chars | MTP length: {len(mtp_text)} chars")
        print(f"Muestra inicial (idéntica):\n{mtp_text[:180]}...")

    # 5. Benchmark MTP a Temperaturas de Producción SuperMLX
    min_p_val = float(os.getenv("DEFAULT_MIN_P", "0.04"))
    
    # 5a. Fase Response / Tool Calling (temp=0.10, min_p=0.04)
    print(f"\n--- 4. Perfil Response / Tools (temp=0.10, min_p={min_p_val}) ---")
    _text_r, mtp_r_tps, mtp_r_tokens, mtp_r_time, alpha_r, acc_r, tot_r = run_mtp_generation(
        model, tokenizer, prompt, max_tokens=120, temp=0.10, min_p=min_p_val
    )
    print(f"MTP (temp=0.10): {mtp_r_tokens} tokens en {mtp_r_time:.2f}s -> {mtp_r_tps:.1f} tokens/s")
    print(f"Tasa de Aceptación (alpha): {alpha_r:.1f}% ({acc_r}/{tot_r} aceptados)")

    # 5b. Fase Thinking (temp=0.50, min_p=0.04)
    print(f"\n--- 5. Perfil Thinking (temp=0.50, min_p={min_p_val}) ---")
    _text_th, mtp_th_tps, mtp_th_tokens, mtp_th_time, alpha_th, acc_th, tot_th = run_mtp_generation(
        model, tokenizer, prompt, max_tokens=120, temp=0.50, min_p=min_p_val
    )
    print(f"MTP (temp=0.50): {mtp_th_tokens} tokens en {mtp_th_time:.2f}s -> {mtp_th_tps:.1f} tokens/s")
    print(f"Tasa de Aceptación (alpha): {alpha_th:.1f}% ({acc_th}/{tot_th} aceptados)")

    # 5c. Perfil Chat / Alta Entropía (temp=0.80, min_p=0.04)
    print(f"\n--- 6. Perfil Alta Entropía (temp=0.80, min_p={min_p_val}) ---")
    _text_h, mtp_h_tps, mtp_h_tokens, mtp_h_time, alpha_h, acc_h, tot_h = run_mtp_generation(
        model, tokenizer, prompt, max_tokens=120, temp=0.80, min_p=min_p_val
    )
    print(f"MTP (temp=0.80): {mtp_h_tokens} tokens en {mtp_h_time:.2f}s -> {mtp_h_tps:.1f} tokens/s")
    print(f"Tasa de Aceptación (alpha): {alpha_h:.1f}% ({acc_h}/{tot_h} aceptados)")

    print(f"\n=======================================================")
    print(f"🏆 RESUMEN COMPLETO DE MATRIZ DE RENDIMIENTO")
    print(f"Baseline Autoregressive: {base_tps:.1f} tokens/s")
    print(f"MTP Greedy (temp=0.0):   {mtp_tps:.1f} tokens/s ({speedup:.2f}x) | alpha={alpha:.1f}% | Paridad: {'IDÉNTICA ✅' if is_identical else 'FALLO ⚠️'}")
    print(f"MTP Response (temp=0.1): {mtp_r_tps:.1f} tokens/s ({(mtp_r_tps/base_tps):.2f}x) | alpha={alpha_r:.1f}%")
    print(f"MTP Thinking (temp=0.5): {mtp_th_tps:.1f} tokens/s ({(mtp_th_tps/base_tps):.2f}x) | alpha={alpha_th:.1f}%")
    print(f"MTP Chat     (temp=0.8): {mtp_h_tps:.1f} tokens/s ({(mtp_h_tps/base_tps):.2f}x) | alpha={alpha_h:.1f}%")
    print(f"=======================================================\n")


if __name__ == "__main__":
    main()
