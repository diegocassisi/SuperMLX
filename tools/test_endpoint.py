#!/usr/bin/env python3
"""
Diagnostic test: hits the MLX server endpoints directly to detect
where the SSE connection dies.

Tests both:
  1. Direct MLX engine (port 8080) — bypasses LiteLLM
  2. LiteLLM proxy (port 4000) — what OpenClaw actually uses

For each, sends a streaming chat completion and logs every SSE event
with precise timestamps.
"""
import os
import sys
import json
import time
import http.client
from datetime import datetime

# --- Config (from env or defaults) ---
MLX_HOST = os.getenv("MLX_HOST", "127.0.0.1")
MLX_PORT = int(os.getenv("MLX_PORT", "8080"))
PROXY_PORT = int(os.getenv("PROXY_PORT", "4000"))
MODEL_ID = os.getenv("MODEL_ID", "openai/mlx-community/Qwen3.5-9B-4bit")

# A realistic OpenClaw-sized prompt (~2k tokens of system + user)
SYSTEM_PROMPT = (
    "You are a helpful coding assistant. You have access to tools for reading files, "
    "editing files, running commands, and browsing the web. Always think step by step "
    "before responding. Analyze the user's request carefully and provide precise, "
    "actionable responses.\n\n"
    + "# Context\n" * 50  # Pad to simulate realistic system prompt
)

USER_MSG = "What is 2+2? Answer in one word."

PAYLOAD = {
    "model": MODEL_ID,
    "stream": True,
    "messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_MSG},
    ],
    "max_tokens": 200,
    "temperature": 0.6,
}


def ts():
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def test_endpoint(label, host, port, path="/v1/chat/completions"):
    print(f"\n{'='*60}")
    print(f"[{ts()}] TEST: {label}")
    print(f"[{ts()}] Target: http://{host}:{port}{path}")
    print(f"{'='*60}")

    try:
        conn = http.client.HTTPConnection(host, port, timeout=300)
        body = json.dumps(PAYLOAD)
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer mlx-local",
        }

        print(f"[{ts()}] Sending POST request...")
        conn.request("POST", path, body=body, headers=headers)

        print(f"[{ts()}] Waiting for response headers...")
        response = conn.getresponse()
        print(f"[{ts()}] Got HTTP {response.status} {response.reason}")
        print(f"[{ts()}] Headers: {dict(response.getheaders())}")

        if response.status != 200:
            print(f"[{ts()}] ERROR: Non-200 status. Body: {response.read().decode()[:500]}")
            return False

        print(f"[{ts()}] --- SSE EVENT STREAM ---")
        event_count = 0
        total_content = ""
        tool_calls_received = 0
        done_received = False
        last_event_at = time.time()

        # Read SSE events line by line
        buffer = b""
        while True:
            try:
                chunk = response.read(1)
                if not chunk:
                    elapsed_since_last = time.time() - last_event_at
                    print(f"[{ts()}] EOF (no more data) | elapsed_since_last_event={elapsed_since_last:.1f}s")
                    break

                buffer += chunk

                # SSE events are separated by double newline
                while b"\n\n" in buffer:
                    event_data, buffer = buffer.split(b"\n\n", 1)
                    event_str = event_data.decode("utf-8", errors="replace").strip()

                    if not event_str:
                        continue

                    last_event_at = time.time()
                    event_count += 1

                    # Handle SSE comments (keepalives)
                    if event_str.startswith(":"):
                        print(f"[{ts()}] SSE comment (keepalive): {event_str[:50]}")
                        continue

                    # Handle data events
                    for line in event_str.split("\n"):
                        line = line.strip()
                        if line == "data: [DONE]":
                            done_received = True
                            print(f"[{ts()}] ✅ [DONE] received")
                            break
                        if line.startswith("data: "):
                            json_str = line[6:]
                            try:
                                obj = json.loads(json_str)
                                choices = obj.get("choices", [])
                                if choices:
                                    delta = choices[0].get("delta", {})
                                    finish = choices[0].get("finish_reason")

                                    if "role" in delta:
                                        print(f"[{ts()}] SSE #{event_count}: role={delta['role']}")
                                    if "content" in delta:
                                        c = delta["content"]
                                        total_content += c
                                        print(f"[{ts()}] SSE #{event_count}: content=[{c[:80]}] ({len(c)} chars)")
                                    if "tool_calls" in delta:
                                        tool_calls_received += 1
                                        tc_list = delta["tool_calls"]
                                        for tc_item in tc_list:
                                            fn = tc_item.get("function", {})
                                            print(f"[{ts()}] SSE #{event_count}: tool_call={fn.get('name','?')}")
                                    if finish:
                                        print(f"[{ts()}] SSE #{event_count}: finish_reason={finish}")
                            except json.JSONDecodeError as e:
                                print(f"[{ts()}] SSE #{event_count}: PARSE ERROR: {e} | raw={json_str[:100]}")

            except Exception as e:
                print(f"[{ts()}] READ ERROR: {type(e).__name__}: {e}")
                break

            if done_received:
                break

        print(f"\n[{ts()}] --- SUMMARY ---")
        print(f"  Events: {event_count}")
        print(f"  Content: [{total_content[:200]}]")
        print(f"  Tool calls: {tool_calls_received}")
        print(f"  [DONE] received: {done_received}")
        print(f"  Success: {done_received}")

        conn.close()
        return done_received

    except Exception as e:
        print(f"[{ts()}] CONNECTION ERROR: {type(e).__name__}: {e}")
        return False


def main():
    print(f"[{ts()}] MLX Endpoint Diagnostic Test")
    print(f"[{ts()}] Model: {MODEL_ID}")
    print(f"[{ts()}] Direct: {MLX_HOST}:{MLX_PORT} | Proxy: {MLX_HOST}:{PROXY_PORT}")

    # Test 1: Direct to MLX engine
    direct_ok = test_endpoint(
        "DIRECT → MLX Engine (bypass LiteLLM)",
        MLX_HOST, MLX_PORT,
        "/v1/chat/completions"
    )

    # Test 2: Through LiteLLM proxy  
    proxy_ok = test_endpoint(
        "PROXY → LiteLLM → MLX Engine",
        MLX_HOST, PROXY_PORT,
        "/v1/chat/completions"
    )

    print(f"\n{'='*60}")
    print(f"RESULTS:")
    print(f"  Direct (:{MLX_PORT}): {'✅ PASS' if direct_ok else '❌ FAIL'}")
    print(f"  Proxy  (:{PROXY_PORT}): {'✅ PASS' if proxy_ok else '❌ FAIL'}")
    print(f"{'='*60}")

    if direct_ok and not proxy_ok:
        print("\n💡 DIAGNOSIS: LiteLLM proxy is dropping the connection.")
        print("   The MLX engine delivers the response, but LiteLLM loses it.")
    elif not direct_ok:
        print("\n💡 DIAGNOSIS: MLX engine itself is failing to deliver SSE events.")
    elif direct_ok and proxy_ok:
        print("\n💡 Both paths work. The issue may be OpenClaw-specific timeout/parsing.")


if __name__ == "__main__":
    main()
