# pyre-check: max-pypy-ratio=22
# pyre-check: skip-cpython
# cpython 1.72s vs pyre 0.38s (4.5x on the ubuntu runner), and it is not
# gated on — only pypy is.
# Both `append` and `pop` now fold, so this reads 3.2x on dynasm and 4.3x on
# cranelift where it read 73.5x here before the pop fold -- and the loop that
# reached 188.0x on windows-latest is the same one. The ceiling is twice the
# slowest local reading carried across this fixture's own measured host span
# (188.0/73.5 = 2.6x, taken while it still residualized). That span is the one
# projected quantity here; refit from the runners' own readings once CI has
# reported them rather than widening this again.
# Benchmark: integer list pop/append loop (per-strategy ops)
# Exercises `w_list_append_inner` / `w_list_pop_end_inner` on Integer storage,
# both reached by an orthodox descent, so the compiled loop carries the array
# ops themselves rather than a call to them: the pop's `CallMayForceR` +
# `GuardNotForced` + the 48-byte bound-method allocation are gone, and what is
# left is a length read, a guard, an `int_sub` and a length store.
# The oracle agrees on the shape -- pypy traces the pair to raw
# `setarrayitem_gc` + `setfield_gc` with no boxing.

N = 30000000


def main():
    lst = [0, 1, 2, 3, 4]
    i = 0
    while i < N:
        lst.append(i)
        lst.pop()
        i = i + 1
    print(len(lst), lst[0])


main()
