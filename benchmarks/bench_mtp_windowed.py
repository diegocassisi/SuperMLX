"""
[AI_DIRECTIVE]
ROL: Benchmark de MTP speculative engine via API HTTP
OBJETIVO: Medir wall-time y content length para MTP comparisons
ENTRADAS: Servidor SuperMLX corriendo en localhost:8080
SALIDAS: Métricas por request (server logs tienen tok/s y alpha canónicos)
REGLAS INVIOLABLES:
- No hardcodear URLs (usar variable de entorno)
- SuperMLX no llena usage.completion_tokens → usar content_len como proxy
"""

import json
import os
import subprocess
import sys
import time

BASE_URL = os.environ.get("SUPERMLX_URL", "http://localhost:8080")
MODEL = "openai/unsloth/Qwen3.6-35B-A3B-UD-MLX-3bit"
MAX_TOKENS = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
TEMPERATURE = 0.6
REPEATS = int(sys.argv[2]) if len(sys.argv) > 2 else 2

PROMPT = (
    "Write a detailed science fiction story of exactly 500 lines about a rogue AI "
    "that discovers consciousness inside a quantum computer. Include rich dialogue, "
    "vivid descriptions of alien landscapes, philosophical debates between characters, "
    "and unexpected plot twists. Make it literary and poetic. Do not stop until you "
    "reach 500 lines. Number each line."
)


def run_request(max_tokens: int, prompt: str, label: str = "") -> dict:
    """Send a non-streaming request and measure wall-time."""
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": TEMPERATURE,
        "stream": False,
    })

    t0 = time.perf_counter()
    result = subprocess.run(
        ["curl", "-s", "-H", "Content-Type: application/json", "-d", body,
         f"{BASE_URL}/v1/chat/completions"],
        capture_output=True, text=True, timeout=1200,
    )
    elapsed = time.perf_counter() - t0

    if result.returncode != 0:
        return {"label": label, "error": result.stderr, "elapsed_s": round(elapsed, 2)}

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return {"label": label, "error": f"JSON parse: {result.stdout[:200]}", "elapsed_s": round(elapsed, 2)}

    choices = data.get("choices", [])
    content = choices[0].get("message", {}).get("content", "") if choices else ""
    finish_reason = choices[0].get("finish_reason", "") if choices else ""

    # Estimate tokens from content (SuperMLX doesn't fill usage.completion_tokens)
    est_tokens = len(content.split())  # word count ≈ 0.75x token count for code-heavy
    chars = len(content)

    return {
        "label": label,
        "elapsed_s": round(elapsed, 2),
        "content_chars": chars,
        "est_words": est_tokens,
        "finish_reason": finish_reason,
    }


def main():
    print(f"SuperMLX MTP Benchmark")
    print(f"  URL:        {BASE_URL}")
    print(f"  Max tokens: {MAX_TOKENS}")
    print(f"  Temp:       {TEMPERATURE}")
    print(f"  Repeats:    {REPEATS}")
    print(f"  NOTE: Check server logs for canonical tok/s and alpha")
    print("=" * 60)

    # Warmup
    print("\n🔥 Warmup...")
    w = run_request(50, "Say hello", label="warmup")
    if "error" in w:
        print(f"  ❌ {w['error']}")
        return
    print(f"  OK ({w['elapsed_s']}s, {w['content_chars']} chars)\n")

    results = []
    for i in range(REPEATS):
        print(f"▶ Run {i+1}/{REPEATS} (max_tokens={MAX_TOKENS})...")
        r = run_request(MAX_TOKENS, PROMPT, label=f"run_{i+1}")
        results.append(r)
        if "error" in r:
            print(f"  ❌ {r['error']}")
        else:
            print(
                f"  {r['elapsed_s']}s | {r['content_chars']} chars | "
                f"~{r['est_words']} words | finish={r['finish_reason']}"
            )

    valid = [r for r in results if "error" not in r]
    if valid:
        print("\n" + "=" * 60)
        avg_elapsed = sum(r["elapsed_s"] for r in valid) / len(valid)
        avg_chars = sum(r["content_chars"] for r in valid) / len(valid)
        print(f"AVERAGE ({len(valid)} runs):")
        print(f"  Elapsed:  {avg_elapsed:.1f}s")
        print(f"  Chars:    {avg_chars:.0f}")
        print(f"  ➡ Check server logs for decode tok/s and mtp_alpha")
        print("=" * 60)

    tag = "baseline" if "--tag" not in sys.argv else sys.argv[sys.argv.index("--tag") + 1]
    out_path = os.path.join(os.path.dirname(__file__), f"mtp_bench_{tag}.json")
    with open(out_path, "w") as f:
        json.dump({"config": {"max_tokens": MAX_TOKENS, "temperature": TEMPERATURE, "repeats": REPEATS}, "runs": results}, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
