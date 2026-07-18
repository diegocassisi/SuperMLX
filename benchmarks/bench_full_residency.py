"""
Benchmark: MoE full-residency optimization — decode_tps before/after.

Usage:
    python benchmarks/bench_full_residency.py [--label BEFORE|AFTER] [--runs 3] [--tokens 512]

Hits the main API endpoint (port 8080) with a fixed prompt at temperature=0.
Measures decode_tps and timing.
"""
import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.error

API_URL = "http://localhost:8080/v1/chat/completions"
PROMPT = (
    "Explain in detail the architecture of a Mixture-of-Experts transformer model, "
    "covering the router mechanism, expert selection, load balancing losses, "
    "capacity factor, and how sparse activation reduces compute. "
    "Then compare the MoE approach with dense transformer scaling, "
    "discussing the trade-offs in terms of memory, inference latency, "
    "and quality. Include concrete numbers where possible."
)


def generate(max_tokens: int, temperature: float = 0.0) -> dict:
    """Send a chat completion request and return timing stats."""
    payload = json.dumps({
        "model": "default",
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }).encode()

    req = urllib.request.Request(
        API_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            body = json.loads(resp.read())
    except urllib.error.URLError as e:
        return {"error": str(e)}
    elapsed = time.perf_counter() - t0

    usage = body.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)

    result = {
        "elapsed_s": round(elapsed, 2),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "decode_tps": round(completion_tokens / elapsed, 2) if elapsed > 0 else 0,
    }

    # Try to extract SuperMLX-specific headers if available
    choices = body.get("choices", [{}])
    if choices:
        content = choices[0].get("message", {}).get("content", "")
        result["response_len"] = len(content)

    return result


def check_server():
    """Check if sidecar is reachable."""
    try:
        req = urllib.request.Request("http://localhost:8080/v1/models")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="UNLABELED", help="BEFORE or AFTER")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--tokens", type=int, default=512)
    args = parser.parse_args()

    if not check_server():
        print("ERROR: Main API not reachable on port 8080")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Benchmark: {args.label}")
    print(f"  Runs: {args.runs} × {args.tokens} tokens")
    print(f"{'='*60}\n")

    # Warmup
    print("Warmup run...")
    w = generate(64)
    if "error" in w:
        print(f"  ERROR: {w['error']}")
        sys.exit(1)
    print(f"  Warmup: {w['decode_tps']} tps, {w['completion_tokens']} tokens\n")

    # Benchmark runs
    results = []
    for i in range(args.runs):
        print(f"Run {i+1}/{args.runs}...")
        r = generate(args.tokens)
        if "error" in r:
            print(f"  ERROR: {r['error']}")
            continue
        results.append(r)
        print(f"  {r['completion_tokens']} tokens in {r['elapsed_s']}s = {r['decode_tps']} tps")

    if not results:
        print("No successful runs!")
        sys.exit(1)

    # Summary
    decode_tps_values = sorted([r["decode_tps"] for r in results])
    median_tps = decode_tps_values[len(decode_tps_values) // 2]
    avg_tps = sum(decode_tps_values) / len(decode_tps_values)
    min_tps = min(decode_tps_values)
    max_tps = max(decode_tps_values)
    avg_tokens = sum(r["completion_tokens"] for r in results) / len(results)

    print(f"\n{'='*60}")
    print(f"  RESULTS: {args.label}")
    print(f"{'='*60}")
    print(f"  Median decode_tps:  {median_tps:.2f}")
    print(f"  Avg decode_tps:     {avg_tps:.2f}")
    print(f"  Min/Max:            {min_tps:.2f} / {max_tps:.2f}")
    print(f"  Avg tokens:         {avg_tokens:.0f}")
    print(f"{'='*60}\n")

    # Save results
    out = {
        "label": args.label,
        "runs": results,
        "summary": {
            "median_tps": median_tps,
            "avg_tps": avg_tps,
            "min_tps": min_tps,
            "max_tps": max_tps,
            "avg_tokens": avg_tokens,
        }
    }
    outpath = f"benchmarks/results_{args.label.lower()}.json"
    os.makedirs("benchmarks", exist_ok=True)
    with open(outpath, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Saved to {outpath}")


if __name__ == "__main__":
    main()
