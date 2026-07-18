#!/bin/bash
# Run stress prefill tests in separate processes.
# Each invocation is isolated — if one OOMs, the next still runs.
#
# Usage: ./run_stress.sh
# Results: benchmarks/stress_results.jsonl
#
# NOTE on --diverse: random token IDs don't guarantee diverse expert routing —
# routing depends on learned embeddings, not token ID distribution. The diverse
# runs provide a different activation pattern than single-token repetition, but
# may not represent worst-case MoE memory pressure. Treat as a directional
# signal, not a precise bound.

set -u
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="${SCRIPT_DIR}/../venv/bin/python3"
STRESS="${SCRIPT_DIR}/stress_prefill.py"
RESULTS="${SCRIPT_DIR}/stress_results.jsonl"

# Clear previous results
> "$RESULTS"

echo "═══════════════════════════════════════════════════"
echo "  Stress Prefill Test Suite"
echo "  Results → $RESULTS"
echo "═══════════════════════════════════════════════════"
echo ""

run_test() {
    local tokens=$1
    local chunk=$2
    local extra_args="${3:-}"

    echo "───────────────────────────────────────────────────"
    echo "  TEST: tokens=$tokens chunk=$chunk $extra_args"
    echo "───────────────────────────────────────────────────"

    "$VENV" "$STRESS" --tokens "$tokens" --chunk "$chunk" $extra_args --output "stress_results.jsonl"
    exit_code=$?

    if [ $exit_code -ne 0 ]; then
        echo ""
        echo "  ⚠️  CRASHED (exit code $exit_code) — likely OOM"
        # Log crash to results
        local diverse_flag="false"
        [[ "$extra_args" == *"--diverse"* ]] && diverse_flag="true"
        echo "{\"tokens\":$tokens,\"chunk_size\":$chunk,\"diverse_tokens\":$diverse_flag,\"status\":\"CRASHED\",\"exit_code\":$exit_code}" >> "$RESULTS"
    fi
    echo ""

    # Wait for Metal memory recovery after potential OOM/abort
    echo "  (waiting 10s for Metal recovery...)"
    sleep 10
}

# ── Phase 1: Baseline with synthetic tokens (same expert routing) ──
run_test 20000 512
run_test 25000 512
run_test 28000 512
run_test 31000 512        # expected OOM zone

# ── Phase 2: Smaller chunks at OOM threshold ──
run_test 31000 256
run_test 31000 128
run_test 31000 64

# ── Phase 3: Same sweep with diverse tokens (different MoE routing) ──
run_test 20000 512 "--diverse"
run_test 25000 512 "--diverse"
run_test 28000 512 "--diverse"
run_test 31000 512 "--diverse"   # compare OOM threshold with synthetic
run_test 31000 128 "--diverse"

echo "═══════════════════════════════════════════════════"
echo "  All tests complete. Results:"
echo "═══════════════════════════════════════════════════"
echo ""
# Pretty print results — device_memory_gb comes from each JSON record
"$VENV" -c "
import json
with open('$RESULTS') as f:
    for line in f:
        r = json.loads(line)
        status = r.get('status', '?')
        tokens = r.get('tokens', '?')
        chunk = r.get('chunk_size', '?')
        diverse = '(diverse)' if r.get('diverse_tokens') else '(synth) '
        if status == 'OK':
            peak = r.get('global_peak_gb', 0)
            scratch = r.get('max_chunk_scratch_gb', 0)
            ratio = r.get('max_scratch_ratio', '?')
            dev_mem = r.get('device_memory_gb', 0)
            headroom = dev_mem - peak if dev_mem and peak else 0
            print(f'  {status:>7s}  tok={tokens:>5}  chunk={chunk:>4}  {diverse}  peak={peak:>6.2f}GB  scratch={scratch:.4f}GB  ratio={ratio}  headroom={headroom:.1f}GB')
        else:
            exit_code = r.get('exit_code', '?')
            print(f'  {status:>7s}  tok={tokens:>5}  chunk={chunk:>4}  {diverse}  exit_code={exit_code}')
" 2>/dev/null || cat "$RESULTS"
