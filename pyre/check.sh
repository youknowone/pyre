#!/bin/bash
# pyre pre-merge check: correctness + regression guard + comparison
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
COMPARISONS=()

echo "Building pyre (release)..."
cargo build --release -p pyrex 2>&1 | tail -1

if [ ! -x "$PYRE" ]; then
    echo "ERROR: build failed"
    exit 1
fi

red()   { printf "\033[31m%s\033[0m" "$1"; }
green() { printf "\033[32m%s\033[0m" "$1"; }
dim()   { printf "\033[2m%s\033[0m" "$1"; }
bold()  { printf "\033[1m%s\033[0m" "$1"; }

time_cmd() {
    local start end
    start=$(python3 -c "import time; print(time.time())")
    "$@" >/dev/null 2>&1 || true
    end=$(python3 -c "import time; print(time.time())")
    python3 -c "print(f'{$end - $start:.2f}')"
}

# run_bench NAME SCRIPT EXPECTED TIMEOUT [MAX_SEC] [beat_cpython_margin]
run_bench() {
    local name="$1" script="$2" expected="$3" timeout="$4" max_sec="${5:-}" beat_cpython="${6:-}"
    printf "  %-20s" "$name"

    # Measure cpython/pypy first
    local t_cpython t_pypy
    t_cpython=$(time_cmd python3 "$script")
    t_pypy=$(time_cmd pypy3 "$script")

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
        COMPARISONS+=("$(printf '  %-20s  cpython %5ss  pypy %5ss  pyre  FAIL' "$name" "$t_cpython" "$t_pypy")")
        return
    }
    end=$(python3 -c "import time; print(time.time())")
    elapsed=$(python3 -c "print(f'{$end - $start:.2f}')")

    # Correctness check
    if [ "$output" != "$expected" ]; then
        RESULTS+=("$(red "FAIL") $name  wrong output")
        printf "$(red WRONG)  got: %s\n" "$(echo "$output" | head -c 60)"
        FAIL=$((FAIL + 1))
        COMPARISONS+=("$(printf '  %-20s  cpython %5ss  pypy %5ss  pyre  WRONG' "$name" "$t_cpython" "$t_pypy")")
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
            COMPARISONS+=("$(printf '  %-20s  cpython %5ss  pypy %5ss  pyre %5ss' "$name" "$t_cpython" "$t_pypy" "$elapsed")")
            return
        fi
    fi

    if [ -n "$beat_cpython" ]; then
        local slower margin="$beat_cpython"
        slower=$(python3 -c "print('yes' if $elapsed > $t_cpython * $margin else 'no')")
        if [ "$slower" = "yes" ]; then
            RESULTS+=("$(red "FAIL") $name  ${elapsed}s > cpython ${t_cpython}s x${margin}")
            printf "$(red "SLOWER")  pyre ${elapsed}s > cpython ${t_cpython}s x${margin}\n"
            FAIL=$((FAIL + 1))
            COMPARISONS+=("$(printf '  %-20s  cpython %5ss  pypy %5ss  pyre %5ss' "$name" "$t_cpython" "$t_pypy" "$elapsed")")
            return
        fi
    fi

    RESULTS+=("$(green "PASS") $name  ${elapsed}s")
    printf "$(green PASS)  ${elapsed}s\n"
    PASS=$((PASS + 1))
    COMPARISONS+=("$(printf '  %-20s  cpython %5ss  pypy %5ss  pyre %5ss' "$name" "$t_cpython" "$t_pypy" "$elapsed")")
}

echo ""
bold "pyre pre-merge check"; echo ""
echo "binary: $PYRE"
echo ""

# Warm up: cold run to prime disk/CPU caches (results discarded)
printf "  %-20s" "(warmup)"
python3 "$BENCH/int_loop.py" >/dev/null 2>&1 || true
pypy3 "$BENCH/int_loop.py" >/dev/null 2>&1 || true
"$PYRE" "$BENCH/int_loop.py" >/dev/null 2>&1 || true
printf "$(dim done)\n"

#                NAME             SCRIPT                         EXPECTED                     TIMEOUT  MAX_SEC  BEAT_CPYTHON
run_bench       "int_loop"       "$BENCH/int_loop.py"           "49999995000000"              30       ""       1
run_bench       "fib_loop"       "$BENCH/fib_loop.py"           "967618232"                   30       ""       1
run_bench       "inline_helper"  "$BENCH/inline_helper.py"      "333333333333000000"          30       ""       1
run_bench       "fib_recursive"  "$BENCH/fib_recursive.py"      "2178309"                     30       ""       2
run_bench       "nbody"          "$BENCH/nbody_50k.py"          "-0.035132020348426815"        30       ""       8
run_bench       "fannkuch"       "$BENCH/fannkuch_9.py"         "$(printf '8629\n30')"          30
run_bench       "raise_catch"   "$BENCH/raise_catch_loop.py"   "1142858"                       30

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
fi

echo "$(green "ALL PASSED"): $PASS/$PASS"
echo ""
bold "Comparison"; echo ""
printf '  %-20s  %10s  %9s  %9s\n' "benchmark" "cpython" "pypy" "pyre"
echo "  ────────────────────────────────────────────────────────"
for c in "${COMPARISONS[@]}"; do echo "$c"; done
echo ""
exit 0
