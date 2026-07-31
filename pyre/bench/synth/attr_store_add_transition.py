# pyre-check: max-pypy-ratio=12
# STORE_ATTR that ADDS an attribute not yet in the instance's map: the
# `map -> PlainAttribute` transition plus the grow-by-one storage rewrite
# (mapdict.py:942-959 `_set_mapdict_increase_storage1`).  The values are
# strings so every slot stays boxed, which is the shape the JIT emits inline;
# an unboxed slot keeps the general residual.
#
# The hot loop holds the two folded shapes — the first attribute of a fresh
# instance (empty storage block) and a second attribute (the live slot is
# copied into the wider block) — so the ratio measures the fold itself.  The
# shapes the fold declines or that force materialization are exercised once
# each in `edges()`, for agreement with the interpreter rather than for speed.
N = 200000


class Fresh:
    pass


class Pair:
    def __init__(self, a, b):
        self.a = a
        self.b = b


class Either:
    pass


def edges():
    out = []
    # `_reorder_and_add` (mapdict.py:204-258): both orders off one class, so
    # the second instance finds the attribute above its own map.
    for first in (0, 1):
        obj = Either()
        if first:
            obj.p = "p"
            obj.q = "qq"
        else:
            obj.q = "qq"
            obj.p = "p"
        out.append(sorted(vars(obj).items()))
    # An instance the loop keeps has to materialize instead of staying virtual.
    kept = []
    for i in range(4):
        obj = Pair("a" * i, "b")
        kept.append(obj)
    out.append([(o.a, o.b) for o in kept])
    # Add, delete, re-add: the second add starts from the popped-back map.
    obj = Fresh()
    obj.x = "1"
    del obj.x
    obj.x = "2"
    out.append((obj.x, sorted(vars(obj).items())))
    return out


def main():
    total = 0
    i = 0
    while i < N:
        # first attribute: storage grows 0 -> 1
        one = Fresh()
        one.x = "x"
        total = total + len(one.x)

        # second attribute: the existing slot is copied into the wider block
        two = Pair("ab", "cde")
        total = total + len(two.a) + len(two.b)

        i = i + 1
    print(total, edges())


main()
