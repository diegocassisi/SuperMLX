"""
[AI_DIRECTIVE]
ROL: Benchmark MoE capacity impact on quality and speed.
OBJETIVO: Compare quality/speed/fallback between different expert capacities.
ENTRADAS: Running SuperMLX server on localhost:8080.
SALIDAS: JSON results file + summary to stdout.
REGLAS INVIOLABLES:
- Same prompts every run for fair comparison.
- Capture actual output for quality comparison.
- Use unique session IDs to avoid KV cache hits.
SSoT: Server telemetry via response timing + output length.
"""

import json
import logging
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

SERVER_URL = "http://127.0.0.1:8080/v1/chat/completions"
MODEL = "openai/unsloth/Qwen3.6-35B-A3B-UD-MLX-3bit"

# Diverse coding prompts — exercises different expert regions
PROMPTS = [
    {
        "id": "fibonacci_dp",
        "prompt": "Write a Python function that computes the nth Fibonacci number using dynamic programming with memoization. Include type hints and a docstring.",
        "max_tokens": 600,
    },
    {
        "id": "async_retry",
        "prompt": "Write a Python async retry decorator with exponential backoff, max retries, and jitter. Type hints required.",
        "max_tokens": 700,
    },
    {
        "id": "binary_search",
        "prompt": "Implement binary search in Python that returns the index of the target or -1. Handle edge cases. Type hints.",
        "max_tokens": 500,
    },
    {
        "id": "json_parser",
        "prompt": "Write a Python function that safely parses nested JSON with a dot-notation path accessor, e.g. get_nested(data, 'a.b.c'). Return None on missing keys. Type hints.",
        "max_tokens": 600,
    },
    {
        "id": "sql_explain",
        "prompt": "Explain the difference between INNER JOIN, LEFT JOIN, RIGHT JOIN, and FULL OUTER JOIN in SQL. Give a concrete example with two tables.",
        "max_tokens": 700,
    },
]


def run_prompt(prompt_cfg: dict) -> dict:
    """Send a prompt to the server and capture response + timing."""
    # Unique prefix to bust KV cache
    unique_prefix = f"[request_id={uuid.uuid4().hex[:8]}] "

    payload = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": unique_prefix + prompt_cfg["prompt"]}],
        "max_tokens": prompt_cfg["max_tokens"],
        "temperature": 0.2,
        "stream": False,
    }).encode()

    req = Request(
        SERVER_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    t0 = time.time()
    try:
        with urlopen(req, timeout=180) as resp:
            elapsed = time.time() - t0
            body = json.loads(resp.read())
            diag_headers = {
                k: resp.getheader(k)
                for k in resp.headers
                if k.lower().startswith("x-")
            }
    except Exception as e:
        return {
            "id": prompt_cfg["id"],
            "error": str(e),
            "elapsed": time.time() - t0,
        }

    choice = body.get("choices", [{}])[0]
    message = choice.get("message", {})
    usage = body.get("usage", {})
    output_text = message.get("content") or ""
    output_chars = len(output_text)

    return {
        "id": prompt_cfg["id"],
        "elapsed_s": round(elapsed, 2),
        "output_chars": output_chars,
        "output_lines": output_text.count("\n") + 1,
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "finish_reason": choice.get("finish_reason", "?"),
        "output_preview": output_text[:300],
        "output_full": output_text,
        "diag_headers": diag_headers,
    }


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "default"
    print(f"=== Capacity Benchmark: {tag} ===")
    print(f"Server: {SERVER_URL}")
    print(f"Prompts: {len(PROMPTS)}")
    print()

    results = []
    for i, prompt_cfg in enumerate(PROMPTS):
        print(f"[{i+1}/{len(PROMPTS)}] {prompt_cfg['id']}...", end=" ", flush=True)
        result = run_prompt(prompt_cfg)
        results.append(result)
        if "error" in result:
            print(f"ERROR: {result['error']}")
        else:
            print(
                f"{result['output_chars']} chars | "
                f"{result['elapsed_s']}s | "
                f"{result['finish_reason']}"
            )

    # Summary
    print("\n=== Summary ===")
    total_chars = sum(r.get("output_chars", 0) for r in results)
    total_time = sum(r.get("elapsed_s", 0) for r in results)
    errors = sum(1 for r in results if "error" in r)
    print(f"Total output: {total_chars} chars in {total_time:.1f}s")
    print(f"Errors: {errors}/{len(PROMPTS)}")

    for r in results:
        if "error" not in r:
            print(f"  {r['id']}: {r['output_chars']} chars | {r['elapsed_s']}s")

    # Save
    out_dir = Path(__file__).parent.parent / "logs" / "bench_capacity"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = out_dir / f"{tag}_{ts}.json"

    output = {
        "tag": tag,
        "timestamp": ts,
        "server_url": SERVER_URL,
        "model": MODEL,
        "summary": {
            "total_chars": total_chars,
            "total_time_s": round(total_time, 1),
            "errors": errors,
        },
        "results": results,
    }

    with open(out_file, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nResults: {out_file}")
    print(f"\nServer log: buscá líneas con 'moe_hit' y 'fallback'")


if __name__ == "__main__":
    main()
