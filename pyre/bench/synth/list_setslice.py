# pyre-check: spec-folds=setslice,newlist
# pyre-check: max-pypy-ratio=1.0
# pyre-check: skip-cpython
# Integer list setslice (per-strategy ops).  Exercises W_ListObject slice
# assignment: lst[a:b] = [...] stays in Integer strategy when the new items are
# plain ints, rather than falling back to Object.  guard_class on
# IntegerListStrategy and a new_array(3, ArrayS 8) are what the census reads
# back as `setslice` and `newlist`.
#
# The ceiling is parity, and a ceiling at parity derives no floor: pyre runs
# this in a fraction of what pypy needs, so a floor under it reddens the run
# for beating pypy.  The trip count is set by pypy's end of that comparison --
# below it pypy's own execution drops under FLOOR_GATE_MIN_BASELINE_S and
# neither bound arms at all, which reads exactly like a gate that holds.  It is
# also what cpython is skipped for.
N = 5874966


def main():
    lst = [0] * 10
    i = 0
    while i < N:
        lst[2:5] = [i, i + 1, i + 2]
        i = i + 1
    print(lst)


main()
