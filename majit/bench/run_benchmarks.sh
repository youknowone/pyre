#!/bin/bash
# Run all majit example benchmarks and compare.
set -e

cd "$(dirname "$0")/.."

echo "=== Building all examples in release mode ==="
cargo build --release -p calc -p tlr -p tl 2>/dev/null

echo ""
echo "================================================================"
echo "  majit JIT Benchmark Suite"
echo "================================================================"
echo ""

echo "--- calc: sum(0..10M) ---"
cargo run --release -p calc 2>/dev/null | grep -E "(time|result)"

echo ""
echo "--- tlr: sum(0..10M) ---"
cargo run --release -p tlr 2>/dev/null | grep -E "(time|result)"

echo ""
echo "--- tl: sum(0..10M) ---"
cargo run --release -p tl 2>/dev/null | grep -E "(time|result)"

echo ""
echo "================================================================"

# If RPython binary exists, run it for comparison.
RPYTHON_BIN="../rpython_calc-c"
if [ -x "$RPYTHON_BIN" ]; then
    echo ""
    echo "--- RPython JIT: sum(0..10M) ---"
    $RPYTHON_BIN 10000000
fi
