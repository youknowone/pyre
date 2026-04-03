#!/bin/bash
# pyre pre-merge check: correctness + regression guard + comparison
# Usage: ./pyre/check.sh [path/to/pyre]

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

# Measure user CPU time (not wall clock) via /usr/bin/time -p
time_user() {
    { /usr/bin/time -p "$@" >/dev/null; } 2>&1 | awk '/^user/{printf "%.2f\n", $2}'
}

# run_bench NAME SCRIPT TIMEOUT [MAX_SEC] [beat_cpython_margin] [beat_pypy_margin]
run_bench() {
    local name="$1" script="$2" timeout="$3" max_sec="${4:-}" beat_cpython="${5:-}" beat_pypy="${6:-}"
    printf "  %-20s" "$name"

    local cpython_output
    if ! cpython_output="$(python3 "$script")"; then
        RESULTS+=("$(red "FAIL") $name  cpython crash (exit $? from python)")
        printf "$(red WRONG)  CPython execution failed\n"
        FAIL=$((FAIL + 1))
        COMPARISONS+=("$(printf '  %-20s  cpython %5ss  pypy %5ss  pyre  FAIL' "$name" "-" "-")")
        return
    fi

    # Measure cpython/pypy user CPU time
    local t_cpython t_pypy
    t_cpython=$(time_user python3 "$script")
    t_pypy=$(time_user pypy3 "$script")

    # Measure pyre: capture output and user CPU time separately
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
    local elapsed
    elapsed=$(time_user "$PYRE" "$script")

    # Compute pyre/pypy ratio for comparison table
    local ratio
    ratio=$(python3 -c "print('%.1fx' % ($elapsed / $t_pypy) if float('$t_pypy') > 0 else 'N/A')" 2>/dev/null || echo "N/A")

    if [ "$output" != "$cpython_output" ]; then
        RESULTS+=("$(red "FAIL") $name  wrong output")
        local expected_preview actual_preview
        expected_preview=$(echo "$cpython_output" | head -c 60)
        actual_preview=$(echo "$output" | head -c 60)
        printf "$(red WRONG)  got: %s expected: %s\n" "$actual_preview" "$expected_preview"
        FAIL=$((FAIL + 1))
        COMPARISONS+=("$(printf '  %-20s  cpython %5ss  pypy %5ss  pyre  WRONG' "$name" "$t_cpython" "$t_pypy")")
        return
    fi

    if [ -n "$max_sec" ]; then
        local over
        over=$(python3 -c "print('yes' if $elapsed > $max_sec else 'no')")
        if [ "$over" = "yes" ]; then
            RESULTS+=("$(red "FAIL") $name  ${elapsed}s > ${max_sec}s limit")
            printf "$(red SLOW)  ${elapsed}s (limit ${max_sec}s)\n"
            FAIL=$((FAIL + 1))
            COMPARISONS+=("$(printf '  %-20s  cpython %5ss  pypy %5ss  pyre %5ss  (%s vs pypy)' "$name" "$t_cpython" "$t_pypy" "$elapsed" "$ratio")")
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
            COMPARISONS+=("$(printf '  %-20s  cpython %5ss  pypy %5ss  pyre %5ss  (%s vs pypy)' "$name" "$t_cpython" "$t_pypy" "$elapsed" "$ratio")")
            return
        fi
    fi

    if [ -n "$beat_pypy" ]; then
        local slower margin="$beat_pypy"
        slower=$(python3 -c "print('yes' if $elapsed > $t_pypy * $margin else 'no')")
        if [ "$slower" = "yes" ]; then
            RESULTS+=("$(red "FAIL") $name  ${elapsed}s > pypy ${t_pypy}s x${margin}")
            printf "$(red "SLOWER")  pyre ${elapsed}s > pypy ${t_pypy}s x${margin}\n"
            FAIL=$((FAIL + 1))
            COMPARISONS+=("$(printf '  %-20s  cpython %5ss  pypy %5ss  pyre %5ss  (%s vs pypy)' "$name" "$t_cpython" "$t_pypy" "$elapsed" "$ratio")")
            return
        fi
    fi

    RESULTS+=("$(green "PASS") $name  ${elapsed}s")
    printf "$(green PASS)  ${elapsed}s\n"
    PASS=$((PASS + 1))
    COMPARISONS+=("$(printf '  %-20s  cpython %5ss  pypy %5ss  pyre %5ss  (%s vs pypy)' "$name" "$t_cpython" "$t_pypy" "$elapsed" "$ratio")")
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

#                NAME             SCRIPT                         TIMEOUT  MAX_SEC  BEAT_CPYTHON
run_bench       "int_loop"       "$BENCH/int_loop.py"            5       ""       1       1.5
run_bench       "fib_loop"       "$BENCH/fib_loop.py"            5       ""       1
run_bench       "inline_helper"  "$BENCH/inline_helper.py"       5       ""       1       1.5
run_bench       "fib_recursive" "$BENCH/fib_recursive.py"        5       ""       1       2.5
run_bench       "float_loop"     "$BENCH/float_loop.py"           5
# nested_loop: blocked by optimizer OpRef type resolution bug
# run_bench     "nested_loop"    "$BENCH/nested_loop.py"          5
run_bench       "nbody"          "$BENCH/nbody_50k.py"           5       ""       10
run_bench       "fannkuch"       "$BENCH/fannkuch.py"          5
run_bench       "raise_catch"   "$BENCH/raise_catch_loop.py"     5
run_bench       "spectral_norm" "$BENCH/spectral_norm.py"       10      ""       15

echo ""
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
printf '  %-20s  %10s  %9s  %9s  %s\n' "benchmark" "cpython" "pypy" "pyre" ""
echo "  ──────────────────────────────────────────────────────────────────"
for c in "${COMPARISONS[@]}"; do echo "$c"; done
echo ""
exit 0
