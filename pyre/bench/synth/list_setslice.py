# pyre-check: max-pypy-ratio=8
# Benchmark: integer list setslice (per-strategy ops)
# Exercises W_ListObject slice assignment: lst[a:b] = [...] on Integer strategy.
# PYPYLOG confirms: guard_class(IntegerListStrategy) + new_array(3, ArrayS 8).
# On main, there was no setslice op (Object-only fallback).
# On this branch, setslice stays in Integer strategy when new items are plain ints.
#
# Sized so pypy's own execution clears the measurement floor: below it the
# ratio gate divides by the floor and reads startup rather than this loop.
N = 5874966

def main():
    lst = [0] * 10
    i = 0
    while i < N:
        lst[2:5] = [i, i + 1, i + 2]
        i = i + 1
    print(lst)

main()
