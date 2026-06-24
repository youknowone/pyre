#!/bin/sh
# Usage: sh run_matrix.sh /path/to/pyre-binary
# Compares interp (PYRE_NO_JIT) vs default-JIT vs inline-gate for each repro.
set -e
PYRE="$1"
DIR="$(dirname "$0")"
for f in countdown_pure stateful_attr global_mutate inner_if real_exception monkeypatch while_dump; do
    base="$(PYRE_NO_JIT=1 "$PYRE" "$DIR/$f.py")"
    jit="$("$PYRE" "$DIR/$f.py")"
    inl="$(PYRE_57_INLINE_NEXT=1 "$PYRE" "$DIR/$f.py")"
    if [ "$base" = "$jit" ] && [ "$base" = "$inl" ]; then
        echo "OK   $f  ->  $base"
    else
        echo "FAIL $f"
        echo "  interp:    $base"
        echo "  jit:       $jit"
        echo "  inline:    $inl"
        exit 1
    fi
done
echo "MATRIX OK"
