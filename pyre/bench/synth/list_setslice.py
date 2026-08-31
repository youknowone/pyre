# pyre-check: spec-folds=setslice,newlist
# Integer list setslice (per-strategy ops).  Exercises W_ListObject slice
# assignment: lst[a:b] = [...] stays in Integer strategy when the new items are
# plain ints, rather than falling back to Object.  guard_class on
# IntegerListStrategy and a new_array(3, ArrayS 8) are what the census reads
# back as `setslice` and `newlist`.
#
# No pypy ratio gate: an iteration here is a handful of ops, so the trip count
# that gives pypy a measurable baseline still leaves pyre's own side against the
# timer floor, and a ratio built on the floor constant grades a clock tick.  The
# census decides instead, and does not move with the host.
N = 200000


def main():
    lst = [0] * 10
    i = 0
    while i < N:
        lst[2:5] = [i, i + 1, i + 2]
        i = i + 1
    print(lst)


main()
