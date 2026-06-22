# Benchmark: integer list reverse (per-strategy ops)
# Exercises W_ListObject.reverse() on Integer strategy.
# PYPYLOG confirms: guard_class(IntegerListStrategy) + setarrayitem(ArrayS 8) swaps.
# On main, reverse() called items_to_vec() → Object strategy first.
# On this branch, reverse() stays in Integer strategy (no boxing overhead).
# REPS is large and odd: the one-time build loop and the trace warmup are
# amortised over many reverse() iterations so the measurement reflects
# reverse() itself, and the final list is reversed (lst[0]=N-1, lst[-1]=0).
#
def main():
    N = 700000
    REPS = 401

    lst = []
    i = 0
    while i < N:
        lst.append(i)
        i = i + 1

    r = 0
    while r < REPS:
        lst.reverse()
        r = r + 1

    print(lst[0], lst[-1])


main()
