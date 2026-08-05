# pyre-check: max-pypy-ratio=5.6
# pyre-check: min-pypy-ratio=0.78
# Fused `goto_if_not_<cmp>` with the SAME box on both sides (`b1 is b2`).
#
# `record_or_fold_fused_guard` mirrors `opimpl_goto_if_not_<cmp>`
# (pyjitpl.py:547): `if <not float> and b1 is b2:` skips both the compare op
# and the guard because `x <cmp> x` is statically determined
# (FASTPATHS_SAME_BOXES: eq/le/ge => True, ne/lt/gt => False). Previously the
# Rust path recorded an extra always-passing guard for these self-compares.
#
# A self-compare reaches the fused branch when both operands colour to the same
# register (`i == i`, `obj is obj`). The fast path must follow the branch in
# exactly the concrete direction; a wrong predicate would drop the guard for a
# genuine two-box compare and miscompile. The never-taken arms add a huge
# sentinel so any wrong direction balloons the checksum.
#
# `obj is obj` only reaches that fast path once the walker folds `IS_OP`
# (`try_walker_fold_is_op`); before the fold it left as a `compare_fn`
# may-force residual, and the ptr loop paid a `CALL_MAY_FORCE` plus its
# `GUARD_NOT_FORCED` every iteration.
#
# `N` is sized by the GATE's noise model, not by the loop.  The ratio compares
# startup-subtracted times, so a sample where pypy's execution is smaller than
# pypy's own ~16ms startup is dominated by how far that one startup estimate
# missed: at N=6000000 pypy executes in ~9ms and the gate swings past 6x on
# startup noise alone.  At this N pypy executes in ~21ms, comfortably above its
# startup, and the measured ratio settles near 2.2x.  A regression that puts
# the residuals back is then a twenty-four-million-call blow-up, far outside
# that margin.
N = 12000000


def same_box_int():
    acc = 0
    i = 0
    while i < N:
        if i == i:  # int_eq same-box: always True
            acc += 1
        if i != i:  # int_ne same-box: always False
            acc += 1000000
        if i <= i:  # int_le same-box: always True
            acc += 10
        if i < i:  # int_lt same-box: always False
            acc += 1000000
        if i >= i:  # int_ge same-box: always True
            acc += 100
        if i > i:  # int_gt same-box: always False
            acc += 1000000
        i += 1
    return acc


def same_box_ptr():
    acc = 0
    i = 0
    obj = object()
    while i < N:
        if obj is obj:  # ptr_eq same-box: always True
            acc += 1
        if obj is not obj:  # ptr_ne same-box: always False
            acc += 1000000
        i += 1
    return acc


def main():
    print(same_box_int(), same_box_ptr())


main()
