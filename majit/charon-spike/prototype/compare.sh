#!/usr/bin/env bash
# Regenerate the canonical text for every corpus function and diff against
# the checked-in snapshot under `expected/`.  Exits non-zero on any diff so
# this script can be wired into CI later.
set -euo pipefail

cd "$(dirname "$0")"

CORPUS="${CORPUS:-../corpus.ullbc}"
FNS=(straight_line_add branch_loop_sum strategy_len desugar_mix)

cargo build --quiet --bin charon-spike-lower

mkdir -p expected actual
fail=0
for fn in "${FNS[@]}"; do
    cargo run --quiet --bin charon-spike-lower -- "$CORPUS" "$fn" \
        2>/dev/null > "actual/$fn.txt"
    if ! diff -u "expected/$fn.txt" "actual/$fn.txt"; then
        echo "diff in $fn"
        fail=1
    fi
done

if [[ "$fail" -ne 0 ]]; then
    exit 1
fi
echo "all $((${#FNS[@]})) canonical outputs match"
