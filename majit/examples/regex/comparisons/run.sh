#!/usr/bin/env bash
#
# Build, VERIFY, and run the foreign-language rows of "A JIT for Regular
# Expression Matching" against the same regex, the same input, and the same
# matcher as `majit/examples/regex`.
#
# Runnable from any directory.  Nothing is written inside the repository: the
# compiler output and the per-row logs go to a `mktemp -d` scratch directory,
# removed on exit unless KEEP=1 is set.  Nothing is installed; a toolchain that
# is missing produces a printed reason and a skipped row.
#
#   ./run.sh [length] [repeats] [n]     defaults: 1048576 5 20
#   KEEP=1 ./run.sh                     keep the scratch dir and say where
#   MAXLOAD=1.5 ./run.sh                stamp rows taken above this load
#   PYTHON=... ./run.sh                 use a different interpreter
#
# Two things this harness insists on, because both were learned the hard way:
#
#   * CORRECTNESS BEFORE SPEED.  Every row runs `--verify` first, and its output
#     line must match the other rows character for character.  A row that
#     disagrees is skipped, not timed.  A fast wrong matcher is worse than no
#     row, and a matcher that answers "no match" to everything would sail
#     through a non-matching benchmark and post a number.
#   * A NUMBER WITHOUT ITS LOAD IS NOT A MEASUREMENT.  These rows are single
#     threaded.  Measured on the machine this was written on: the same C++
#     binary on the same 2^20-character input did 8.21-9.13M chars/s at a
#     1-minute load average of 8.75 and 2.70-3.15M at a load average of about
#     33 -- a 3.2x distortion with nothing whatsoever changed in the program,
#     and both endpoints are runs whose load was sampled at the time rather
#     than reconstructed afterwards.  So the 1-minute load average
#     is sampled before and after EVERY timed row, printed beside it, and any
#     row taken above MAXLOAD is stamped PROVISIONAL in the table.

set -uo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
LEN=${1:-1048576}
REPEATS=${2:-5}
N=${3:-20}

# The load average above which a row is stamped PROVISIONAL.
#
# CHOSEN, NOT MEASURED.  The two loads these rows have actually been observed at
# are ~2 (undistorted) and ~33 (3.2x low); the knee between them was never
# measured, because measuring it means deliberately loading a machine other
# people are building on.  4.0 is a conservative guess on a 10-core machine and
# nothing more.  To calibrate it properly, on an otherwise idle machine:
#   for k in 0 1 2 4 8 16; do
#       for i in $(seq $k); do (while :; do :; done) & done
#       ./run.sh 1048576 5 20 | grep '^  cpp'
#       kill $(jobs -p) 2>/dev/null
#   done
MAXLOAD=${MAXLOAD:-4.0}

SCRATCH=$(mktemp -d "${TMPDIR:-/tmp}/regex-comparisons.XXXXXX") || exit 1
cleanup() {
    if [ "${KEEP:-0}" = "1" ]; then
        echo
        echo "scratch kept: $SCRATCH"
    else
        rm -rf "$SCRATCH"
    fi
}
trap cleanup EXIT

PYTHON=${PYTHON:-python3}

# rank|label|status|min|median|max|load_before|load_after|flag|note
# The rank is the post's own table order, minus the rows this example does not
# run (Google re2 and Java -- see the README's Scope section), so the report
# reads next to the post whatever order the rows happened to finish in.
ROWS="$SCRATCH/rows"
: >"$ROWS"
VERIFY="$SCRATCH/verify"
: >"$VERIFY"

rank() {
    case $1 in
        python) echo 1 ;;
        cpp) echo 3 ;;
        re) echo 5 ;;
        *) echo 9 ;;
    esac
}

skip() { printf '%s|%s|skipped|-|-|-|-|-||%s\n' "$(rank "$1")" "$1" "$2" >>"$ROWS"; }

load1() { uptime | sed 's/.*load averages*: *//' | awk '{print $1}'; }

over_maxload() { awk -v a="$1" -v b="$2" -v m="$MAXLOAD" 'BEGIN {exit !(a > m || b > m)}'; }

# stats <file-of-numbers> -> "min median max"
# The median convention is the programs' own: the element at index n/2 of the
# ascending list, so an even count reports the upper of the two middles and the
# script never disagrees with the contract line it just read.
stats() {
    sort -g "$1" | awk '
        {a[NR] = $1}
        END {
            if (NR == 0) { print "- - -"; exit }
            printf "%s %s %s\n", a[1], a[int(NR/2) + 1], a[NR]
        }'
}

# verify_port <label> <command...> -> records the port's verify line
verify_port() {
    local label=$1
    shift
    local out
    if ! out=$("$@" 2>"$SCRATCH/$label.verr"); then
        echo "  $label: VERIFY FAILED: $(tail -n 2 "$SCRATCH/$label.verr" | tr '\n' ' ')"
        return 1
    fi
    case $out in
        verify\ *) ;;
        *)  # e.g. `... unavailable: ...` — nothing to attest to, and nothing to
            # disagree with either.  Do not record it as a verify line.
            echo "  $label: no verify line ($out)"
            return 0 ;;
    esac
    printf '%s %s\n' "$label" "$out" >>"$VERIFY"
    echo "  $label: $(sed -n 's/.*\(input_fnv1a=[0-9a-f]*\).*/\1/p' <<<"$out")"
    return 0
}

# run_row <label> <note> <command...>
run_row() {
    local label=$1 note=$2
    shift 2
    local out="$SCRATCH/$label.out" err="$SCRATCH/$label.err"
    local lb la
    lb=$(load1)
    "$@" >"$out" 2>"$err"
    local rc=$?
    la=$(load1)
    if [ $rc -ne 0 ]; then
        skip "$label" "exited $rc: $(tail -n 2 "$err" | tr '\n' ' ')"
        return
    fi
    grep '^round ' "$err" | awk '{print $3}' >"$SCRATCH/$label.rates"
    if [ ! -s "$SCRATCH/$label.rates" ]; then
        # A program that printed no rounds may still have said something useful
        skip "$label" "$(head -n 1 "$out")"
        return
    fi
    local mn md mx flag=""
    read -r mn md mx <<<"$(stats "$SCRATCH/$label.rates")"
    over_maxload "$lb" "$la" && flag="!"
    printf '%s|%s|ok|%s|%s|%s|%s|%s|%s|%s\n' \
        "$(rank "$label")" "$label" "$mn" "$md" "$mx" "$lb" "$la" "$flag" "$note" >>"$ROWS"
    # The timed run generates the input all over again; this confirms it used the
    # same bytes the verify run attested to.
    local d
    d=$(grep -o 'input_fnv1a=[0-9a-f]*' "$err" | head -n 1)
    if [ -n "$d" ] && ! grep -q "^$label .*$d" "$VERIFY"; then
        echo "  WARNING: $label's timed run used different bytes from its verify run"
    fi
}

echo "=== environment ==================================================="
uname -srm
sw_vers 2>/dev/null | tr '\n' ' ' && echo
sysctl -n machdep.cpu.brand_string 2>/dev/null
echo "cores: $(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null)"
echo "length=$LEN repeats=$REPEATS n=$N   regex=(a|b)*a(a|b){$N}a(a|b)*"
echo "load at start: $(uptime | sed 's/.*load averages*: *//')   MAXLOAD=$MAXLOAD"
echo

echo "=== build ========================================================="

CPP_BIN="$SCRATCH/marked"
CPP_OK=0
if command -v clang++ >/dev/null 2>&1; then
    if clang++ -O2 -std=c++17 -o "$CPP_BIN" "$HERE/marked.cpp" 2>"$SCRATCH/cpp.build"; then
        echo "c++ : clang++ -O2 -std=c++17  ok"
        CPP_OK=1
    else
        echo "c++ : compile failed:"
        sed 's/^/      /' "$SCRATCH/cpp.build"
    fi
else
    echo "c++ : no clang++ on PATH"
fi
[ $CPP_OK = 1 ] || skip cpp "no working clang++ build (see above)"

PY_OK=0
if command -v "$PYTHON" >/dev/null 2>&1; then
    echo "py  : $($PYTHON -VV | head -n 1)"
    PY_OK=1
else
    echo "py  : no $PYTHON on PATH"
fi
[ $PY_OK = 1 ] || skip python "no working $PYTHON (see above)"

echo
echo "=== verify (correctness gate — a row that fails here is not timed) ="
# Every port prints one `verify` line.  The marked-matcher ports print the input
# digest, the answers to a fixed battery, and a digest of all `marked` bits left
# in the tree after scanning the benchmark input; `re` has neither
# marks nor a node tree, so they attest only to the bytes.  The rule is
# agreement, not self-assessment: a port that grades itself can only catch the
# mistakes its author anticipated.
[ $CPP_OK = 1 ] && { verify_port cpp "$CPP_BIN" --verify "$LEN" "$N" || CPP_OK=0; }
if [ $PY_OK = 1 ]; then
    verify_port python "$PYTHON" "$HERE/marked.py" --verify "$LEN" "$N" || PY_OK=0
    verify_port re "$PYTHON" "$HERE/re_module.py" --verify "$LEN" "$N" || true
fi

# The marked-matcher ports must agree on the WHOLE line; `re` only on the
# three input fields, which is all they print.
MARKED_LINES=$(awk '$1 == "cpp" || $1 == "python" {$1 = ""; print}' "$VERIFY" | sort -u)
INPUT_FIELDS=$(sed -n 's/.*\(input_fnv1a=[0-9a-f]*\) \(head=[ab]*\) \(tail=[ab]*\).*/\1 \2 \3/p' "$VERIFY" | sort -u)
VERIFY_OK=1
if [ "$(grep -c . <<<"$MARKED_LINES")" -gt 1 ] && [ -n "$MARKED_LINES" ]; then
    echo
    echo "  MARKED PORTS DISAGREE — they are not the same matcher:"
    sed 's/^/    /' <<<"$MARKED_LINES"
    VERIFY_OK=0
fi
if [ "$(grep -c . <<<"$INPUT_FIELDS")" -gt 1 ]; then
    echo
    echo "  PORTS DISAGREE ON THE INPUT — they did not scan the same bytes:"
    sed 's/^/    /' <<<"$INPUT_FIELDS"
    VERIFY_OK=0
fi
if [ $VERIFY_OK = 1 ] && [ -s "$VERIFY" ]; then
    echo
    # "all agree" counts only the ports that actually produced a line.  A port
    # whose verify FAILED was already dropped, so saying "all agree" without the
    # count would read as if it had passed.
    echo "  $(wc -l <"$VERIFY" | tr -d ' ') attested port(s) agree:"
    sed 's/^/    /' <<<"$MARKED_LINES" | head -n 1
fi
if [ $VERIFY_OK = 0 ]; then
    echo
    echo "  refusing to time anything: fix the disagreement first."
    CPP_OK=0
    PY_OK=0
fi

echo
echo "=== measuring (this is the slow part) ============================="

[ $CPP_OK = 1 ] && run_row cpp "marked matcher, clang++ -O2" "$CPP_BIN" "$LEN" "$REPEATS" "$N"
if [ $PY_OK = 1 ]; then
    run_row python "marked matcher, pure Python" "$PYTHON" "$HERE/marked.py" "$LEN" "$REPEATS" "$N"
    run_row re "NOT COMPARABLE: backtracking, different work" \
        "$PYTHON" "$HERE/re_module.py" "$LEN" "$REPEATS" "$N"
fi

echo
echo "=== results ======================================================="
echo "length $LEN, $REPEATS timed runs per row, chars/s"
if awk -F'|' '$9 == "!" {found = 1} END {exit !found}' "$ROWS"; then
    echo
    echo "  *** PROVISIONAL: rows marked ! were taken above MAXLOAD=$MAXLOAD.   ***"
    echo "  *** These rows are a LOWER BOUND on this machine, not a measurement. ***"
    echo "  *** Rerun on an idle machine before quoting them.                    ***"
fi
echo
awk -F'|' '
    function commas(x,   s, o, i, L) {
        if (x !~ /^[0-9.]+$/) return x
        s = sprintf("%d", x + 0.5); L = length(s); o = ""
        for (i = 1; i <= L; i++) {
            o = o substr(s, i, 1)
            if ((L - i) % 3 == 0 && i < L) o = o ","
        }
        return o
    }
    BEGIN {
        printf "  %-7s %1s %14s %14s %14s  %13s   %s\n", "row", "", "min", "median", "max", "load b->a", "note"
        printf "  %-7s %1s %14s %14s %14s  %13s   %s\n", "-------", "-", "--------------", "--------------", "--------------", "-------------", "----"
    }
    {
        if ($3 == "ok")
            printf "  %-7s %1s %14s %14s %14s  %13s   %s\n", $2, $9, commas($4), commas($5), commas($6), $7 "->" $8, $10
        else
            printf "  %-7s %1s %14s %14s %14s  %13s   %s\n", $2, "", "-", "skipped", "-", "-", $10
    }
' <(sort -t'|' -k1,1n "$ROWS")

if [ -s "$SCRATCH/re.err" ]; then
    echo
    echo "=== the \`re\` row is not comparable; here is the evidence ==========="
    echo "  a flat chars/s across the length sweep means the time is linear in the"
    echo "  input, i.e. it did not stop after a bounded prefix; reaches_last_char"
    echo "  plants a matching pair at the very end, so a True means it read the"
    echo "  final byte. Neither makes the row comparable — a backtracking engine"
    echo "  still does a different amount of work per character."
    grep -E '^(consumed|search_scaling)' "$SCRATCH/re.err" | sed 's/^/  /'
fi

# ── the control: the work must scale with the node count ───────────────────
#
# chars/s alone cannot say whether the optimizer deleted the tree walk.  This
# can: `n = 2` is 21 nodes against `n = 20`'s 93, so the same matcher on the
# same length must come out about 4.4x faster per character.  A ratio near 1
# means the walk is not in the measured loop and the row above is not measuring
# the marked algorithm.  Being a ratio of two runs taken back to back, it is the
# one number here that survives a loaded machine.
echo
echo "=== control: rate must scale as 1 / node count ===================="
echo "  ideal ratio n=2 over n=20 is 93/21 = 4.4x; the VERDICT is >2.0x, because"
echo "  on a loaded machine the ratio is noisy but 'the walk is in the loop' is not"
CTRL_LEN=$((LEN / 4))
[ "$CTRL_LEN" -lt 4096 ] && CTRL_LEN=4096
for label in cpp python; do
    case $label in
        cpp) [ $CPP_OK = 1 ] || continue
             c20=("$CPP_BIN" "$CTRL_LEN" 3 20); c2=("$CPP_BIN" "$CTRL_LEN" 3 2) ;;
        python) [ $PY_OK = 1 ] || continue
             c20=("$PYTHON" "$HERE/marked.py" "$CTRL_LEN" 3 20)
             c2=("$PYTHON" "$HERE/marked.py" "$CTRL_LEN" 3 2) ;;
    esac
    r20=$("${c20[@]}" 2>/dev/null | awk '{print $2}')
    r2=$("${c2[@]}" 2>/dev/null | awk '{print $2}')
    if [ -n "$r20" ] && [ -n "$r2" ]; then
        awk -v a="$r2" -v b="$r20" -v l="$label" -v n="$CTRL_LEN" 'BEGIN {v = (a / b > 2.0) ? "PASS (the tree walk is in the timed loop)" : "FAIL (the walk is NOT in the timed loop)"; printf "  %-7s len=%d  n=2: %.0f  n=20: %.0f  ratio %.2fx  %s\n", l, n, a, b, a / b, v}'
    else
        echo "  $label    control did not run"
    fi
done

echo
echo "load at end: $(uptime | sed 's/.*load averages*: *//')"
echo
echo "The Rust rows of this comparison are not built here — they need the crate:"
echo "  cargo run -p regex --release --no-default-features --features dynasm"
