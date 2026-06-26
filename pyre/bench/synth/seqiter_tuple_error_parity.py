# PR271 review parity guard.  check.py's correctness oracle is PyPy, so this
# bench only asserts behaviour where 3.14 and PyPy AGREE; it guards the
# tuple-subclass override (P1), the live `__length_hint__` (P2) and the
# type-slot `__getitem__` iteration check (3b).  The 3.14-specific seq-iter
# `next` behaviour (StopIteration treated as exhaustion, retryable after a
# non-IndexError) diverges from PyPy and is documented at baseobjspace.rs
# `next` / `iter_iternext`, so it cannot be asserted against the PyPy oracle
# here.
import operator


class LenSeq:
    def __len__(self):
        return 7

    def __getitem__(self, i):
        if i >= 7:
            raise IndexError
        return i


class NoLenSeq:
    def __getitem__(self, i):
        if i >= 4:
            raise IndexError
        return i


class TupOverride(tuple):
    def __getitem__(self, i):
        return 1000 + tuple.__getitem__(self, i)


class InstanceGetitem:
    pass


def drive():
    out = []

    # A generic `__getitem__` cursor iterates lazily to the IndexError (it
    # ignores `__len__`).  Driven hot so the FOR_ITER over the cursor compiles.
    total = 0
    n = 0
    while n < 20000:
        for x in LenSeq():
            total += x
        n += 1
    out.append(("lazy_iter", total))

    # (P2) `__length_hint__` recomputed from the live sequence (len - index),
    # reachable through operator.length_hint as well as the direct call.
    it = iter(LenSeq())
    lh0 = operator.length_hint(it)
    next(it)
    lh1 = operator.length_hint(it)
    out.append(("len_hint", lh0, lh1, it.__length_hint__()))
    out.append(("len_hint_nolen", operator.length_hint(iter(NoLenSeq()))))

    # (P1) tuple-subclass `__getitem__` override honoured in a hot trace (the
    # JIT must not read the raw `wrappeditems` block).
    t = TupOverride([10, 20, 30])
    acc = 0
    n = 0
    while n < 20000:
        acc = t[1]
        n += 1
    out.append(("tuple_override", acc))

    # (3b) an instance-dict `__getitem__` does not enable iteration (special
    # methods resolve on the type, not the instance).
    c = InstanceGetitem()
    c.__getitem__ = lambda i: i
    try:
        list(c)
        out.append(("instance_getitem", "iterable"))
    except TypeError:
        out.append(("instance_getitem", "not_iterable"))

    return out


for row in drive():
    print(row)
