# pyre-check: selfcheck
# pyre-check: selfcheck-compiles=sites
# pyre-check: spec-folds=builtin_locals
# Self-checking guard for the cell half of the modelled `fast2locals`.
#
# `try_walker_specialize_builtin_locals_in_callee_expand` reads each cell slot
# with one `GETFIELD_GC_R(cell, contents)` instead of handing the frame to a
# residual.  `Cell.contents` is mutable and STORE_DEREF is lowered as
# `PlainCannotRaise`, whose write-descr sets are analyzer-empty, so nothing
# tells `OptHeap` that a store to that cell invalidates the read.  If the two
# were merged, a compiled iteration would report the value the recording
# iteration saw rather than the one just assigned.
#
# The read is recorded through `walker_record_getfield_gc_r_uncached`, so the
# walk never reuses the OpRef across the store; what these sites pin is that
# the answer stays per-iteration all the way through the optimizer.  A counter
# cannot see this — the fold fires either way — and the wrong answer is a
# plausible integer, so it has to be compared against a known sequence.
#
# Sites:
#   A  the store is in the loop that inlines the callee, and the callee reads
#      the cell once per iteration.  The hazard is across the back edge: the
#      read from the previous iteration must not survive into this one.
#   B  two reads of the same cell in one iteration with a store between them,
#      so the hazard is within a single trace body rather than across its
#      edge.  A carries a distinct value per iteration and B a second one, so
#      neither can pass by reporting the other's.
N = 20000


def sites(n):
    cap = -1

    def read_cap():
        d = locals()
        return d["cap"], cap

    bad_a = 0
    bad_b = 0
    mismatched = 0
    for i in range(n):
        # A: one read per iteration, tracking the store above it.
        cap = i
        seen, direct = read_cap()
        if seen != i:
            bad_a += 1
        # The modelled mapping and a plain LOAD_DEREF of the same cell are two
        # different reads of one field; they must agree.
        if seen != direct:
            mismatched += 1

        # B: a second store, then a second read, inside the same iteration.
        cap = i + N
        seen_again, direct_again = read_cap()
        if seen_again != i + N:
            bad_b += 1
        if seen_again != direct_again:
            mismatched += 1
    return bad_a, bad_b, mismatched


def main():
    bad_a, bad_b, mismatched = sites(N)
    if bad_a:
        print(f"FAIL site A: {bad_a} of {N} iterations read a stale cell")
        return 1
    if bad_b:
        print(f"FAIL site B: {bad_b} of {N} iterations read a stale cell")
        return 1
    if mismatched:
        print(f"FAIL mapping and LOAD_DEREF disagreed {mismatched} times")
        return 1
    print("PASS locals cell rebound")
    return 0


import sys

sys.exit(main())
