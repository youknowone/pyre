#!/bin/sh
# Usage: sh run_matrix.sh /path/to/pyre-binary
# Oracle = interp (PYRE_JIT=0); asserts default-JIT matches it.
set -e
if [ "$#" -ne 1 ] || [ ! -x "$1" ]; then
    echo "Usage: sh run_matrix.sh /path/to/pyre-binary" >&2
    exit 2
fi
PYRE="$1"
DIR="$(dirname "$0")"
for f in for_min for_sum for_sum_big for_raise for_gen for_user for_enumerate for_dictkeys for_user_raise for_hotraise for_mutate for_monkeypatch for_dict_abort for_iadd_list_abort for_iadd_bytearray_abort for_finally_loop for_finally_loop_noreturn for_attr_abort for_deref_abort for_prop_abort for_prop_raise_abort for_nested for_nested_kept min_listflip f1_tuple f1_polyonly f1_polyrange iso_subvar poly_v; do
    base="$(PYRE_JIT=0 perl -e 'alarm shift; exec @ARGV' 30 "$PYRE" "$DIR/$f.py" || echo "TIMEOUT/CRASH($?)")"
    jit="$(perl -e 'alarm shift; exec @ARGV' 30 "$PYRE" "$DIR/$f.py" || echo "TIMEOUT/CRASH($?)")"
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
