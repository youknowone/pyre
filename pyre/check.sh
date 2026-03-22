#!/bin/bash
# pyre pre-merge check: correctness + regression guard
# Usage: ./pyre/check.sh [path/to/pyre]
#
# Runs all benchmark scripts and verifies:
#   1. Correct output (must match expected values)
#   2. No crashes (exit code 0)
#   3. Performance gate: fib_loop must stay fast (JIT-optimized)

set -euo pipefail

PYRE="${1:-./target/release/pyre}"
BENCH=pyre/bench
PASS=0
FAIL=0
RESULTS=()

if [ ! -x "$PYRE" ]; then
    echo "ERROR: $PYRE not found. Run: cargo build --release -p pyrex"
    exit 1
fi

red()   { printf "\033[31m%s\033[0m" "$1"; }
green() { printf "\033[32m%s\033[0m" "$1"; }
bold()  { printf "\033[1m%s\033[0m" "$1"; }

# run_bench NAME SCRIPT EXPECTED TIMEOUT_SEC [MAX_SEC]
# MAX_SEC: if set, fail when wall time exceeds it
run_bench() {
    local name="$1" script="$2" expected="$3" timeout="$4" max_sec="${5:-}"
    printf "  %-20s" "$name"

    local start end elapsed
    start=$(python3 -c "import time; print(time.time())")
    local output
    output=$(timeout "$timeout" "$PYRE" "$script" 2>/dev/null) || {
        local code=$?
        if [ $code -eq 124 ]; then
            RESULTS+=("$(red "FAIL") $name  timeout (>${timeout}s)")
            printf "$(red TIMEOUT)\n"
        else
            RESULTS+=("$(red "FAIL") $name  crash (exit $code)")
            printf "$(red "CRASH") (exit $code)\n"
        fi
        FAIL=$((FAIL + 1))
        return
    }
    end=$(python3 -c "import time; print(time.time())")
    elapsed=$(python3 -c "print(f'{$end - $start:.2f}')")

    # Correctness check
    if [ "$output" != "$expected" ]; then
        RESULTS+=("$(red "FAIL") $name  wrong output")
        printf "$(red WRONG)  got: %s\n" "$(echo "$output" | head -c 60)"
        FAIL=$((FAIL + 1))
        return
    fi

    # Performance check
    if [ -n "$max_sec" ]; then
        local over
        over=$(python3 -c "print('yes' if $elapsed > $max_sec else 'no')")
        if [ "$over" = "yes" ]; then
            RESULTS+=("$(red "FAIL") $name  ${elapsed}s > ${max_sec}s limit")
            printf "$(red SLOW)  ${elapsed}s (limit ${max_sec}s)\n"
            FAIL=$((FAIL + 1))
            return
        fi
    fi

    RESULTS+=("$(green "PASS") $name  ${elapsed}s")
    printf "$(green PASS)  ${elapsed}s\n"
    PASS=$((PASS + 1))
}

echo ""
bold "pyre pre-merge check"; echo ""
echo "binary: $PYRE"
echo ""

# ── Performance gated ───────────────────────────────────────
bold "FAST (JIT-optimized, must stay fast)"; echo ""

run_bench  "int_loop"        "$BENCH/int_loop.py"        "49999995000000"         30  5.0
run_bench  "inline_helper"   "$BENCH/inline_helper.py"   "333333333333000000"     30  5.0

echo ""

# ── Correctness only ────────────────────────────────────────
bold "CORRECTNESS (must not crash or give wrong results)"; echo ""

run_bench  "fib_loop"        "$BENCH/fib_loop.py"        "308061521170129"        10
run_bench  "fib_recursive"   "$BENCH/fib_recursive.py"   "9227465"                60
run_bench  "nbody"           "$BENCH/nbody.py"           "-0.035117363568587606"  120
run_bench  "fannkuch"        "$BENCH/fannkuch.py"        "$(printf '73196\n38')"  120

echo ""

# ── Summary ─────────────────────────────────────────────────
echo "─────────────────────────────────"
for r in "${RESULTS[@]}"; do echo "  $r"; done
echo "─────────────────────────────────"

if [ $FAIL -gt 0 ]; then
    echo "$(red "FAILED"): $FAIL failed, $PASS passed"
    exit 1
else
    echo "$(green "ALL PASSED"): $PASS/$PASS"
    exit 0
fi
