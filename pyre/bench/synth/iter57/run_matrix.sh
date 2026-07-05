#!/bin/sh
# Usage: sh run_matrix.sh /path/to/pyre-binary
# Compares interp (PYRE_JIT=0) vs default-JIT for each repro.
set -e
PYRE="$1"
DIR="$(dirname "$0")"
for f in countdown_pure stateful_attr global_mutate inner_if real_exception monkeypatch while_dump; do
    base="$(PYRE_JIT=0 "$PYRE" "$DIR/$f.py")"
    jit="$("$PYRE" "$DIR/$f.py")"
    if [ "$base" = "$jit" ]; then
        echo "OK   $f  ->  $base"
    else
        echo "FAIL $f"
        echo "  interp:    $base"
        echo "  jit:       $jit"
        exit 1
    fi
done
echo "MATRIX OK"
