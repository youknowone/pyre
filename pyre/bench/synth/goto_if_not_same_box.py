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
N = 1000


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
