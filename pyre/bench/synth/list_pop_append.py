# pyre-check: max-pypy-ratio=2.7
# pyre-check: skip-cpython
# cpython is not gated — only pypy is. At this trip count the reference
# run is past a useful ratio sample.
# dynasm 0.8x, cranelift 1.3x; the ceiling is twice the slower,
# rounded up to one decimal place.
# Benchmark: integer list pop/append loop (per-strategy ops)
# Exercises `w_list_append_inner` / `w_list_pop_end_inner` on Integer storage,
# both reached by an orthodox descent, so the compiled loop carries the array
# ops themselves rather than a call to them: the pop's `CallMayForceR` +
# `GuardNotForced` + the 48-byte bound-method allocation are gone, and what is
# left is a length read, a guard, an `int_sub` and a length store.
# The oracle agrees on the shape -- pypy traces the pair to raw
# `setarrayitem_gc` + `setfield_gc` with no boxing.

N = 146341464


def main():
    lst = [0, 1, 2, 3, 4]
    i = 0
    while i < N:
        lst.append(i)
        lst.pop()
        i = i + 1
    print(len(lst), lst[0])


main()
