# pyre-check: gate=1
"""`types.GenericAlias` keeps the tuples it walks current across a user `__eq__`.

Building `list[...]` gathers the free type variables of the arguments, and
every turn of that walk runs Python: the `__typing_subst__` and
`__parameters__` lookups, and the `__eq__` behind `if item not in parameters`.
The containers being walked -- the argument tuple, any tuple or list nested in
it, and each `__parameters__` tuple pulled off an argument -- are all
nursery-allocated, so a collection during any of those relocates them while the
walk is still indexing the word it read first.

The parameters gathered so far have the same problem from the other side: they
were held in a native column that no collector scans, so both operands of the
uniqueness test had to be published before the comparison and read back after.

Substitution (`alias[...]`) repeats the shape a second time, in the
`params.index(param)` search that also runs `__eq__`.
"""

import gc


class Collects:
    """Plays a type variable, and collects from `__eq__`.

    `__typing_subst__` is what marks an argument as a parameter in its own
    right; without it an object contributes only through `__parameters__`.
    """

    def __init__(self, i):
        self.i = i

    def __typing_subst__(self, arg):
        return arg

    def __eq__(self, other):
        gc.collect()
        # Churn so a relocated block is handed straight back out.
        self.junk = [tuple(range(4)) for _ in range(500)]
        return isinstance(other, Collects) and other.i == self.i

    def __hash__(self):
        return 0

    def __repr__(self):
        return "Collects(%d)" % self.i


class Holder:
    """Exposes a `__parameters__` tuple born just before the walk reads it."""

    def __init__(self, params):
        self.__parameters__ = params


def nested():
    # The argument tuple, the nested tuple and list, and the `__parameters__`
    # tuple are each allocated inside the expression, so nothing but the walk's
    # own roots refers to them once `list[...]` has popped its subscript.
    return list[
        Holder(tuple(Collects(i) for i in range(6))),
        (Collects(6), Collects(7)),
        [Collects(8)],
        Collects(9),
    ]


got = [p.i for p in nested().__parameters__]
assert got == list(range(10)), got

# The same parameter appearing twice takes the `in` test all the way through
# the gathered column, so every comparison reads a pair back out of the slots.
shared = Collects(0)
alias = list[Holder((shared, Collects(1), shared)), shared]
assert [p.i for p in alias.__parameters__] == [0, 1], alias.__parameters__


class Subst(Collects):
    def __typing_subst__(self, arg):
        gc.collect()
        return arg


ts = Subst(0)
generic = dict[ts, Subst(1)]
assert len(generic.__parameters__) == 2, generic.__parameters__
substituted = generic[str, int]
assert substituted == dict[str, int], substituted

print("ok")
