from functools import cmp_to_key
from inspect import signature


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


class CmpOwner:
    cmp_to_key = cmp_to_key


# The accelerator callable is descriptor-neutral when stored on a class.
assert str(signature(CmpOwner().cmp_to_key)) == "(mycmp)"
assert CmpOwner().cmp_to_key(lambda a, b: a - b)(1) < CmpOwner().cmp_to_key(
    lambda a, b: a - b
)(2)
