#!/bin/bash
# pyre pre-merge check: correctness + regression guard + comparison
# Usage:
#   ./pyre/check.sh [--backend dynasm|cranelift] [--timeout-scale N]
#                   [--dynasm-timeout-scale N] [--cranelift-timeout-scale N]
#                   [path/to/pyre]

set -euo pipefail

BACKEND=""
DYNASM_TIMEOUT_SCALE=""
CRANELIFT_TIMEOUT_SCALE=""
TIMEOUT_SCALE="1"
PYRE_PATH=""

usage() {
    cat <<'EOF'
Usage:
  ./pyre/check.sh [--backend dynasm|cranelift] [--timeout-scale N]
                  [--dynasm-timeout-scale N] [--cranelift-timeout-scale N]
                  [path/to/pyre]

Options:
  --backend BACKEND              Run only one backend.
  --timeout-scale N             Multiply all benchmark timeouts by N.
  --dynasm-timeout-scale N      Override dynasm timeout multiplier.
  --cranelift-timeout-scale N   Override cranelift timeout multiplier.

Notes:
  - Without --backend, the script runs both dynasm and cranelift.
  - The optional positional binary path is only accepted with --backend.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend)
            BACKEND="$2"; shift 2 ;;
        --backend=*|backend=*)
            BACKEND="${1#*=}"; shift ;;
        --timeout-scale)
            TIMEOUT_SCALE="$2"; shift 2 ;;
        --timeout-scale=*)
            TIMEOUT_SCALE="${1#*=}"; shift ;;
        --dynasm-timeout-scale)
            DYNASM_TIMEOUT_SCALE="$2"; shift 2 ;;
        --dynasm-timeout-scale=*)
            DYNASM_TIMEOUT_SCALE="${1#*=}"; shift ;;
        --cranelift-timeout-scale)
            CRANELIFT_TIMEOUT_SCALE="$2"; shift 2 ;;
        --cranelift-timeout-scale=*)
            CRANELIFT_TIMEOUT_SCALE="${1#*=}"; shift ;;
        -h|--help)
            usage
            exit 0 ;;
        *)
            break ;;
    esac
done

if [[ $# -gt 1 ]]; then
    echo "ERROR: too many positional arguments"
    usage
    exit 1
fi

if [[ $# -eq 1 ]]; then
    PYRE_PATH="$1"
fi

if [[ -n "$PYRE_PATH" && -z "$BACKEND" ]]; then
    echo "ERROR: [path/to/pyre] requires --backend when running a single binary"
    usage
    exit 1
fi

BENCH=pyre/bench
RESULTS=()
COMPARISON_NAMES=()
COMPARISON_CPYTHON=()
COMPARISON_PYPY=()
COMPARISON_DYNASM=()
COMPARISON_CRANELIFT=()
DYNASM_PYRE=""
CRANELIFT_PYRE=""
DYNASM_PASS=0
DYNASM_FAIL=0
CRANELIFT_PASS=0
CRANELIFT_FAIL=0

red()   { printf "\033[31m%s\033[0m" "$1"; }
green() { printf "\033[32m%s\033[0m" "$1"; }
dim()   { printf "\033[2m%s\033[0m" "$1"; }
bold()  { printf "\033[1m%s\033[0m" "$1"; }

# Measure user CPU time (not wall clock) via /usr/bin/time -p
time_user() {
    { /usr/bin/time -p "$@" >/dev/null; } 2>&1 | awk '/^user/{printf "%.2f\n", $2}'
}

comparison_index() {
    local name="$1"
    local i
    for i in "${!COMPARISON_NAMES[@]}"; do
        if [[ "${COMPARISON_NAMES[$i]}" == "$name" ]]; then
            printf '%s\n' "$i"
            return
        fi
    done
    printf '%s\n' "-1"
}

append_comparison() {
    local backend="$1" name="$2" t_cpython="$3" t_pypy="$4" pyre_field="$5" note="${6:-}"
    local idx cell
    idx=$(comparison_index "$name")
    if [[ "$idx" == "-1" ]]; then
        idx="${#COMPARISON_NAMES[@]}"
        COMPARISON_NAMES+=("$name")
        COMPARISON_CPYTHON+=("$t_cpython")
        COMPARISON_PYPY+=("$t_pypy")
        COMPARISON_DYNASM+=("-")
        COMPARISON_CRANELIFT+=("-")
    else
        COMPARISON_CPYTHON[$idx]="$t_cpython"
        COMPARISON_PYPY[$idx]="$t_pypy"
    fi

    cell="$pyre_field"
    if [[ -n "$note" ]]; then
        note="${note#(}"
        note="${note%)}"
        note="${note% vs pypy}"
        printf -v cell '%6s   %5s' "$pyre_field" "$note"
    fi

    case "$backend" in
        dynasm)
            COMPARISON_DYNASM[$idx]="$cell" ;;
        cranelift)
            COMPARISON_CRANELIFT[$idx]="$cell" ;;
        *)
            echo "ERROR: unknown backend '$backend' (use: dynasm, cranelift)" >&2
            exit 1 ;;
    esac
}

backend_timeout_scale() {
    local backend="$1"
    case "$backend" in
        dynasm)
            echo "${DYNASM_TIMEOUT_SCALE:-$TIMEOUT_SCALE}" ;;
        cranelift)
            echo "${CRANELIFT_TIMEOUT_SCALE:-$TIMEOUT_SCALE}" ;;
        *)
            echo "ERROR: unknown backend '$backend' (use: dynasm, cranelift)" >&2
            exit 1 ;;
    esac
}

build_backend() {
    local backend="$1"
    local cargo_extra=""
    local cargo_bin=""
    case "$backend" in
        dynasm)
            cargo_extra="--no-default-features --features dynasm"
            cargo_bin="pyre-dynasm"
            ;;
        cranelift)
            cargo_extra="--no-default-features --features cranelift"
            cargo_bin="pyre-cranelift"
            ;;
        *)
            echo "ERROR: unknown backend '$backend' (use: dynasm, cranelift)"
            exit 1 ;;
    esac

    echo "Building ${cargo_bin} (release, backend=${backend})..."
    cargo build --release -p pyrex --bin "$cargo_bin" $cargo_extra 2>&1 | tail -1
}

default_binary_for_backend() {
    local backend="$1"
    case "$backend" in
        dynasm) echo "./target/release/pyre-dynasm" ;;
        cranelift) echo "./target/release/pyre-cranelift" ;;
        *)
            echo "ERROR: unknown backend '$backend' (use: dynasm, cranelift)" >&2
            exit 1 ;;
    esac
}

backend_binary() {
    local backend="$1"
    case "$backend" in
        dynasm) echo "$DYNASM_PYRE" ;;
        cranelift) echo "$CRANELIFT_PYRE" ;;
        *)
            echo "ERROR: unknown backend '$backend' (use: dynasm, cranelift)" >&2
            exit 1 ;;
    esac
}

backend_enabled() {
    local backend="$1"
    case "$backend" in
        dynasm) [[ -n "$DYNASM_PYRE" ]] ;;
        cranelift) [[ -n "$CRANELIFT_PYRE" ]] ;;
        *)
            echo "ERROR: unknown backend '$backend' (use: dynasm, cranelift)" >&2
            exit 1 ;;
    esac
}

print_backend_config() {
    local parts=()
    if backend_enabled dynasm; then
        parts+=("dynasm=$DYNASM_PYRE(x$(backend_timeout_scale dynasm))")
    fi
    if backend_enabled cranelift; then
        parts+=("cranelift=$CRANELIFT_PYRE(x$(backend_timeout_scale cranelift))")
    fi
    if [ ${#parts[@]} -gt 0 ]; then
        printf 'backend: %s\n' "${parts[*]}"
    fi
}

print_comparison_table() {
    local i
    bold "Comparison"; echo ""
    if backend_enabled dynasm && backend_enabled cranelift; then
        printf '  %-15s %8s %8s %18s %18s\n' "benchmark" "cpython" "pypy" "dynasm" "cranelift"
        echo "  ──────────────────────────────────────────────────────────────────────────────────"
        for i in "${!COMPARISON_NAMES[@]}"; do
            printf '  %-15s %8s %8s %18s %18s\n' \
                "${COMPARISON_NAMES[$i]}" \
                "${COMPARISON_CPYTHON[$i]}" \
                "${COMPARISON_PYPY[$i]}" \
                "${COMPARISON_DYNASM[$i]}" \
                "${COMPARISON_CRANELIFT[$i]}"
        done
    elif backend_enabled dynasm; then
        printf '  %-15s %8s %8s %18s\n' "benchmark" "cpython" "pypy" "dynasm"
        echo "  ────────────────────────────────────────────────────────────"
        for i in "${!COMPARISON_NAMES[@]}"; do
            printf '  %-15s %8s %8s %18s\n' \
                "${COMPARISON_NAMES[$i]}" \
                "${COMPARISON_CPYTHON[$i]}" \
                "${COMPARISON_PYPY[$i]}" \
                "${COMPARISON_DYNASM[$i]}"
        done
    elif backend_enabled cranelift; then
        printf '  %-15s %8s %8s %18s\n' "benchmark" "cpython" "pypy" "cranelift"
        echo "  ────────────────────────────────────────────────────────────"
        for i in "${!COMPARISON_NAMES[@]}"; do
            printf '  %-15s %8s %8s %18s\n' \
                "${COMPARISON_NAMES[$i]}" \
                "${COMPARISON_CPYTHON[$i]}" \
                "${COMPARISON_PYPY[$i]}" \
                "${COMPARISON_CRANELIFT[$i]}"
        done
    fi
}

scaled_timeout() {
    local base_timeout="$1"
    local scale="$2"
    python3 - "$base_timeout" "$scale" <<'PY'
import sys

base = float(sys.argv[1])
scale = float(sys.argv[2])
value = base * scale
if value.is_integer():
    print(int(value))
else:
    print(f"{value:.3f}".rstrip("0").rstrip("."))
PY
}

fmt_time() {
    local t="$1"
    if [[ "$t" == "-" || -z "$t" ]]; then
        printf "%s" "-"
    else
        printf "%ss" "$t"
    fi
}

run_and_capture() {
    local out_var="$1" time_var="$2" status_var="$3"
    shift 3
    local out_file time_file status timed
    out_file=$(mktemp)
    time_file=$(mktemp)
    if /usr/bin/time -p -o "$time_file" "$@" >"$out_file" 2>/dev/null; then
        status=0
    else
        status=$?
    fi
    printf -v "$out_var" '%s' "$(cat "$out_file")"
    timed=$(awk '/^user/{printf "%.2f", $2}' "$time_file")
    if [[ -z "$timed" ]]; then
        timed="-"
    fi
    printf -v "$time_var" '%s' "$timed"
    printf -v "$status_var" '%s' "$status"
    rm -f "$out_file" "$time_file"
}

record_result() {
    local backend="$1" status="$2" name="$3" detail="$4"
    if [[ "$status" == "PASS" ]]; then
        case "$backend" in
            dynasm) DYNASM_PASS=$((DYNASM_PASS + 1)) ;;
            cranelift) CRANELIFT_PASS=$((CRANELIFT_PASS + 1)) ;;
        esac
    else
        RESULTS+=("$(red "FAIL") $backend $name  $detail")
        case "$backend" in
            dynasm) DYNASM_FAIL=$((DYNASM_FAIL + 1)) ;;
            cranelift) CRANELIFT_FAIL=$((CRANELIFT_FAIL + 1)) ;;
        esac
    fi
}

run_backend_bench() {
    local backend="$1" name="$2" script="$3" timeout="$4" vs_cpython="$5" vs_pypy="$6" t_cpython="$7" t_pypy="$8" pypy_output="$9"
    local pyre_bin effective_timeout output elapsed code ratio slower
    pyre_bin=$(backend_binary "$backend")
    effective_timeout=$(scaled_timeout "$timeout" "$(backend_timeout_scale "$backend")")

    printf "    %-10s" "$backend"
    run_and_capture output elapsed code timeout "$effective_timeout" "$pyre_bin" "$script"
    if [ "$code" -ne 0 ]; then
        if [ "$code" -eq 124 ]; then
            record_result "$backend" "FAIL" "$name" "timeout (>${effective_timeout}s)"
            printf "$(red TIMEOUT)  >%ss\n" "$effective_timeout"
        else
            record_result "$backend" "FAIL" "$name" "crash (exit $code)"
            printf "$(red "CRASH") (exit $code)\n"
        fi
        ratio="-"
        if [[ "$t_pypy" != "-" ]]; then
            ratio=$(python3 -c "print('%.1fx' % (${elapsed:-0} / $t_pypy) if float('$t_pypy') > 0 and float('${elapsed:-0}') > 0 else 'N/A')" 2>/dev/null || echo "N/A")
        fi
        append_comparison "$backend" "$name" "$(fmt_time "$t_cpython")" "$(fmt_time "$t_pypy")" "FAIL"
        return
    fi

    if [ "$output" != "$pypy_output" ]; then
        local expected_preview actual_preview
        expected_preview=$(echo "$pypy_output" | head -c 60)
        actual_preview=$(echo "$output" | head -c 60)
        record_result "$backend" "FAIL" "$name" "wrong output"
        printf "$(red WRONG)  got: %s expected(pypy): %s\n" "$actual_preview" "$expected_preview"
        append_comparison "$backend" "$name" "$(fmt_time "$t_cpython")" "$(fmt_time "$t_pypy")" "WRONG"
        return
    fi

    ratio=$(python3 -c "print('%.1fx' % ($elapsed / $t_pypy) if float('$t_pypy') > 0 else 'N/A')" 2>/dev/null || echo "N/A")

    if [ -n "$vs_cpython" ]; then
        slower=$(python3 -c "print('yes' if $elapsed > $t_cpython * $vs_cpython else 'no')")
        if [ "$slower" = "yes" ]; then
            record_result "$backend" "FAIL" "$name" "${elapsed}s > cpython ${t_cpython}s x${vs_cpython}"
            printf "$(red "SLOWER")  pyre ${elapsed}s > cpython ${t_cpython}s x${vs_cpython}\n"
            append_comparison "$backend" "$name" "$(fmt_time "$t_cpython")" "$(fmt_time "$t_pypy")" "$(fmt_time "$elapsed")" "(${ratio} vs pypy)"
            return
        fi
    fi

    if [ -n "$vs_pypy" ]; then
        slower=$(python3 -c "print('yes' if $elapsed > $t_pypy * $vs_pypy else 'no')")
        if [ "$slower" = "yes" ]; then
            record_result "$backend" "FAIL" "$name" "${elapsed}s > pypy ${t_pypy}s x${vs_pypy}"
            printf "$(red "SLOWER")  pyre ${elapsed}s > pypy ${t_pypy}s x${vs_pypy}\n"
            append_comparison "$backend" "$name" "$(fmt_time "$t_cpython")" "$(fmt_time "$t_pypy")" "$(fmt_time "$elapsed")" "(${ratio} vs pypy)"
            return
        fi
    fi

    record_result "$backend" "PASS" "$name" "${elapsed}s"
    printf "$(green PASS)  ${elapsed}s\n"
    append_comparison "$backend" "$name" "$(fmt_time "$t_cpython")" "$(fmt_time "$t_pypy")" "$(fmt_time "$elapsed")" "(${ratio} vs pypy)"
}

warmup_once() {
    local script="$1"
    printf "  %-10s" "warmup"
    python3 "$script" >/dev/null 2>&1 || true
    pypy3 "$script" >/dev/null 2>&1 || true
    if backend_enabled dynasm; then
        "$DYNASM_PYRE" "$script" >/dev/null 2>&1 || true
    fi
    if backend_enabled cranelift; then
        "$CRANELIFT_PYRE" "$script" >/dev/null 2>&1 || true
    fi
    printf "%s\n" "$(dim done)"
}

# run_bench NAME SCRIPT TIMEOUT
#           [dynasm_vs_cpython] [dynasm_vs_pypy]
#           [cranelift_vs_cpython] [cranelift_vs_pypy]
#           [skip_backends]  (space-separated list, e.g. "cranelift")
run_bench() {
    local name="$1" script="$2" timeout="$3"
    local dynasm_vs_cpython="${4:-}" dynasm_vs_pypy="${5:-}"
    local cranelift_vs_cpython="${6:-}" cranelift_vs_pypy="${7:-}"
    local skip_backends="${8:-}"
    local need_cpython="no"
    local cpython_output="" t_cpython="-" cpython_code=0
    local pypy_output="" t_pypy="-" pypy_code=0

    local skip_dynasm="no" skip_cranelift="no"
    for sb in $skip_backends; do
        case "$sb" in
            dynasm) skip_dynasm="yes" ;;
            cranelift) skip_cranelift="yes" ;;
            *)
                echo "ERROR: unknown skip backend '$sb' for bench '$name'" >&2
                exit 1 ;;
        esac
    done

    if { backend_enabled dynasm && [ "$skip_dynasm" != "yes" ] && [ -n "$dynasm_vs_cpython" ]; } || \
       { backend_enabled cranelift && [ "$skip_cranelift" != "yes" ] && [ -n "$cranelift_vs_cpython" ]; }; then
        need_cpython="yes"
    fi

    echo "  $name"

    if [ "$need_cpython" = "yes" ]; then
        printf "    %-10s" "cpython"
        run_and_capture cpython_output t_cpython cpython_code python3 "$script"
        if [ "$cpython_code" -ne 0 ]; then
            printf "$(red "CRASH") (exit $cpython_code)\n"
        else
            printf "$(dim done)  %ss\n" "$t_cpython"
        fi
    fi

    printf "    %-10s" "pypy"
    run_and_capture pypy_output t_pypy pypy_code pypy3 "$script"
    if [ "$pypy_code" -ne 0 ]; then
        printf "$(red "CRASH") (exit $pypy_code)\n"
        if backend_enabled dynasm; then
            record_result "dynasm" "FAIL" "$name" "pypy crash"
            append_comparison "dynasm" "$name" "$(fmt_time "$t_cpython")" "-" "FAIL"
        fi
        if backend_enabled cranelift; then
            record_result "cranelift" "FAIL" "$name" "pypy crash"
            append_comparison "cranelift" "$name" "$(fmt_time "$t_cpython")" "-" "FAIL"
        fi
        return
    fi
    printf "$(dim done)  %ss\n" "$t_pypy"

    if backend_enabled dynasm; then
        if [ "$skip_dynasm" = "yes" ]; then
            printf "    %-10s$(dim "skip")\n" "dynasm"
            append_comparison "dynasm" "$name" "$(fmt_time "$t_cpython")" "$(fmt_time "$t_pypy")" "skip"
        elif [ -n "$dynasm_vs_cpython" ] && [ "$cpython_code" -ne 0 ]; then
            printf "    %-10s$(red "FAIL")  missing cpython baseline\n" "dynasm"
            record_result "dynasm" "FAIL" "$name" "cpython crash"
            append_comparison "dynasm" "$name" "-" "$(fmt_time "$t_pypy")" "FAIL"
        else
            run_backend_bench "dynasm" "$name" "$script" "$timeout" "$dynasm_vs_cpython" "$dynasm_vs_pypy" "$t_cpython" "$t_pypy" "$pypy_output"
        fi
    fi

    if backend_enabled cranelift; then
        if [ "$skip_cranelift" = "yes" ]; then
            printf "    %-10s$(dim "skip")\n" "cranelift"
            append_comparison "cranelift" "$name" "$(fmt_time "$t_cpython")" "$(fmt_time "$t_pypy")" "skip"
        elif [ -n "$cranelift_vs_cpython" ] && [ "$cpython_code" -ne 0 ]; then
            printf "    %-10s$(red "FAIL")  missing cpython baseline\n" "cranelift"
            record_result "cranelift" "FAIL" "$name" "cpython crash"
            append_comparison "cranelift" "$name" "-" "$(fmt_time "$t_pypy")" "FAIL"
        else
            run_backend_bench "cranelift" "$name" "$script" "$timeout" "$cranelift_vs_cpython" "$cranelift_vs_pypy" "$t_cpython" "$t_pypy" "$pypy_output"
        fi
    fi
}

if [[ -n "$BACKEND" ]]; then
    case "$BACKEND" in
        dynasm|cranelift)
            ;;
        *)
            echo "ERROR: unknown backend '$BACKEND' (use: dynasm, cranelift)"
            exit 1 ;;
    esac
fi

if [[ -n "$BACKEND" ]]; then
    build_backend "$BACKEND"
    pyre_bin="${PYRE_PATH:-$(default_binary_for_backend "$BACKEND")}"
    if [ ! -x "$pyre_bin" ]; then
        echo "ERROR: build failed for backend '$BACKEND' (missing executable: $pyre_bin)"
        exit 1
    fi
    case "$BACKEND" in
        dynasm) DYNASM_PYRE="$pyre_bin" ;;
        cranelift) CRANELIFT_PYRE="$pyre_bin" ;;
    esac
else
    for backend in dynasm cranelift; do
        build_backend "$backend"
        pyre_bin="$(default_binary_for_backend "$backend")"
        if [ ! -x "$pyre_bin" ]; then
            echo "ERROR: build failed for backend '$backend' (missing executable: $pyre_bin)"
            exit 1
        fi
        case "$backend" in
            dynasm) DYNASM_PYRE="$pyre_bin" ;;
            cranelift) CRANELIFT_PYRE="$pyre_bin" ;;
        esac
    done
fi

echo ""
bold "pyre pre-merge check"; echo ""
print_backend_config
echo ""
warmup_once "$BENCH/int_loop.py"
echo ""

#                NAME             SCRIPT                         TIMEOUT  DYNASM_VS_CPYTHON  DYNASM_VS_PYPY  CRANELIFT_VS_CPYTHON  CRANELIFT_VS_PYPY
run_bench       "int_loop"       "$BENCH/int_loop.py"            5       ""                   1.5             ""                      1.5
run_bench       "float_loop"     "$BENCH/float_loop.py"          5       ""                   1.0             ""                      2.5
run_bench       "fib_loop"       "$BENCH/fib_loop.py"            5       ""                   1.5             1.0                     ""
run_bench       "inline_helper"  "$BENCH/inline_helper.py"       5       ""                   1.0             ""                      1.0
# Dynasm is already slower than CPython here; keep only a relaxed
# CPython guard and do not add a PyPy criterion.
#
# fib_recursive used to be skipped on cranelift due to a self-recursive
# CALL_ASSEMBLER runaway: the cranelift helper's force_fn fallback
# created a fresh PyFrame with empty locals (PyFrame::new_for_call with
# args=[]) when blackhole returned None, and portal_runner then
# re-entered the JIT with a stale locals_w[0], driving fib into
# unbounded self-recursion and blowing the shadow stack.
# Fixed 2026-04-20 by dropping the force_fn fallback and the
# rebuild_state_after_failure preprocessing in
# call_assembler_guard_failure_inner, matching dynasm's
# call_assembler_helper_trampoline (lib.rs:143) which is
# RPython-orthodox: handle_fail does trace+attach OR blackhole, with
# `assert 0, "unreachable"` after both branches (compile.py:717).
run_bench       "fib_recursive" "$BENCH/fib_recursive.py"        5       1.5                  ""              1                       8
run_bench       "nested_loop"    "$BENCH/nested_loop.py"         5       ""                   2               ""                      2
run_bench       "raise_catch"   "$BENCH/raise_catch_loop.py"     6       ""                   ""              ""                      ""
# Dynasm is slower than CPython on these; do not add a PyPy criterion.
run_bench       "spectral_norm" "$BENCH/spectral_norm.py"       10      10                   ""              10                      ""
run_bench       "nbody"          "$BENCH/nbody_50k.py"           5       15                   ""              15                      ""
run_bench       "fannkuch"       "$BENCH/fannkuch.py"            5       ""                   ""              ""                      ""
# list per-strategy ops (Integer strategy stays without boxing on insert/pop/reverse/setslice)
# PYPYLOG verified: all benchmarks hit guard_class(IntegerListStrategy) + ArrayS 8 ops.
# Keep correctness against PyPy output, but use only CPython performance guards here.
run_bench       "list_reverse"   "$BENCH/list_reverse.py"        5       10                    ""             10                       ""
run_bench       "list_pop_append" "$BENCH/list_pop_append.py"    5       15                   ""              15                      ""
run_bench       "list_insert"    "$BENCH/list_insert.py"         5       ""                   1.5             ""                      1.5
run_bench       "list_setslice"  "$BENCH/list_setslice.py"       5       7                    ""              7                        ""

echo ""
if [ ${#RESULTS[@]} -gt 0 ]; then
    echo "─────────────────────────────────"
    for r in "${RESULTS[@]}"; do echo "  $r"; done
    echo "─────────────────────────────────"
fi

FAILED_BACKEND_RUNS=0
if backend_enabled dynasm && [ $DYNASM_FAIL -gt 0 ]; then
    FAILED_BACKEND_RUNS=$((FAILED_BACKEND_RUNS + 1))
fi
if backend_enabled cranelift && [ $CRANELIFT_FAIL -gt 0 ]; then
    FAILED_BACKEND_RUNS=$((FAILED_BACKEND_RUNS + 1))
fi

ENABLED_BACKEND_RUNS=0
if backend_enabled dynasm; then
    ENABLED_BACKEND_RUNS=$((ENABLED_BACKEND_RUNS + 1))
fi
if backend_enabled cranelift; then
    ENABLED_BACKEND_RUNS=$((ENABLED_BACKEND_RUNS + 1))
fi

print_comparison_table
echo ""
if backend_enabled dynasm; then
    if [ $DYNASM_FAIL -gt 0 ]; then
        echo "$(red "FAILED"): dynasm $DYNASM_FAIL failed, $DYNASM_PASS passed"
    else
        echo "$(green "ALL PASSED"): dynasm $DYNASM_PASS/$DYNASM_PASS"
    fi
fi
if backend_enabled cranelift; then
    if [ $CRANELIFT_FAIL -gt 0 ]; then
        echo "$(red "FAILED"): cranelift $CRANELIFT_FAIL failed, $CRANELIFT_PASS passed"
    else
        echo "$(green "ALL PASSED"): cranelift $CRANELIFT_PASS/$CRANELIFT_PASS"
    fi
fi
if [ $FAILED_BACKEND_RUNS -gt 0 ]; then
    echo "$(red "FAILED"): $FAILED_BACKEND_RUNS backend run(s) failed"
else
    echo "$(green "ALL PASSED"): ${ENABLED_BACKEND_RUNS}/${ENABLED_BACKEND_RUNS} backend run(s)"
fi
if [ $FAILED_BACKEND_RUNS -gt 0 ]; then
    exit 1
fi
exit 0
