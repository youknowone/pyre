from functools import cmp_to_key


descending = cmp_to_key(lambda a, b: b - a)
ascending = cmp_to_key(lambda a, b: a - b)

assert sorted([3, 1, 2], key=descending) == [3, 2, 1]
assert sorted([3, 1, 2], key=ascending) == [1, 2, 3]
assert min([3, 1, 2], key=descending) == 3
assert max([3, 1, 2], key=descending) == 1
assert sorted(
    [(1, "first"), (1, "second"), (0, "third")],
    key=cmp_to_key(lambda a, b: a[0] - b[0]),
) == [(0, "third"), (1, "first"), (1, "second")]
