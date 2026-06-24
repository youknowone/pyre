# Conditional expression whose taken arm loads a heap-int constant (>= 256, a
# LOAD_CONST rather than the inline LoadSmallInt).  That arm value reaches the
# branch-merge block as a flow-graph Constant link arg.
#
# `insert_renamings` skipped any arg whose `as_variable()` equalled
# `last_exception` / `last_exc_value`.  A Constant has `as_variable() == None`,
# and a non-exception link has both fields `None`, so the bare `==` matched and
# dropped the constant before emitting its `ref_copy` -> the merge register was
# left unmaterialised and the heap arm read a stale box.  Small-int arms are
# unaffected because they materialise through box_int_fn into a Variable.
#
# `ce_big` (taken arm const 1000000, a heap int) miscompiled to 333001602
# instead of 666001002; `ce_small` (both arms small ints) stayed correct.  Both
# shapes are kept for the small-vs-heap contrast, and `ce_big` is correct even
# when the branch never diverges (the merge structure alone triggered the
# drop).  Pure arithmetic -> deterministic checksum.
N = 1000


def ce_small():
    acc = 0
    i = 0
    while i < N:
        c = i % 3
        x = 5 if c else 3
        acc = acc + x
        i = i + 1
    return acc


def ce_big():
    acc = 0
    i = 0
    while i < N:
        c = i % 3
        x = 1000000 if c else 3
        acc = acc + x
        i = i + 1
    return acc


def main():
    print(ce_small(), ce_big())


main()
