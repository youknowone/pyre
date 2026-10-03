# pyre-check: no-cpython
# Every assertion here is a place where PyPy remains pyre's reference rather
# than CPython.  CPython 3.14 layout members and `__sizeof__` are deliberate
# pyre compatibility surfaces and so do not belong in this fixture.
N = 20000

class Heap:
    pass


class Derived(Heap):
    pass


# The bits `W_TypeObject.get_flags` publishes, and only those: `_CPYTYPE` (1,
# cpyext-defined static types, which have no owner here), PATMA_SEQUENCE
# (1 << 5), PATMA_MAPPING (1 << 6), `_HEAPTYPE` (1 << 9),
# Py_TPFLAGS_METHOD_DESCRIPTOR (1 << 17) and `_ABSTRACT` (1 << 20).
#
# Comparing against a positive mask rather than the whole word is what keeps
# this an oracle comparison.  pyre also publishes a 3.14 `tp_flags` surface —
# IMMUTABLETYPE, BASETYPE, HAVE_GC, READY, DISALLOW_INSTANTIATION, the managed
# dict/weakref pair, the fast-subclass family — and every one of those bits
# falls outside this mask precisely because `get_flags` has no opinion to
# compare it to.  Each carries its own justification where it is published; a
# bit with no oracle does not belong in a fixture whose oracle is PyPy.
#
# What the masked-out bits name is still asserted where it is observable:
# IMMUTABLETYPE by `check_exception_group_immutable`, which makes the store and
# fails if it succeeds.
PYPY_FLAGS = 1 | (1 << 5) | (1 << 6) | (1 << 9) | (1 << 17) | (1 << 20)


def check_flags():
    # 0x20 is PATMA_SEQUENCE, 0x40 PATMA_MAPPING, 0x200 `_HEAPTYPE`.
    #
    # str, bytes and bytearray carry no row.  `get_flags` answers 0x20 for
    # them, but a sequence pattern must not match a string, so pyre publishes
    # 0 and keeps the `S` marker only where `issequence_w` reads it.  That
    # makes them a place where PyPy is not the reference, which is what this
    # fixture's rows are for; `bench/synth/match_sequence_str.py` is where the
    # observable half is pinned.
    expected = {
        object: 0x0,
        type: 0x0,
        int: 0x0,
        set: 0x0,
        frozenset: 0x0,
        BaseException: 0x0,
        type(None): 0x0,
        list: 0x20,
        tuple: 0x20,
        range: 0x20,
        memoryview: 0x20,
        dict: 0x40,
        Heap: 0x200,
        Derived: 0x200,
    }
    for cls, want in expected.items():
        got = cls.__flags__ & PYPY_FLAGS
        if got != want:
            raise AssertionError((cls.__name__, hex(got), hex(want)))
    return len(expected)


def check_descriptor_kinds():
    for name in ("__flags__", "__base__"):
        kind = type(type.__dict__[name]).__name__
        if kind != "getset_descriptor":
            raise AssertionError((name, kind))
    return 2


def check_exception_group_immutable():
    # `moduledef.py` names `W_ExceptionGroup` under `interpleveldefs`, so
    # `ExceptionGroup` is a static builtin beside `BaseExceptionGroup` rather
    # than a heap type over it.
    group = ExceptionGroup.__flags__ & PYPY_FLAGS
    base = BaseExceptionGroup.__flags__ & PYPY_FLAGS
    if group != 0x0 or base != 0x0:
        raise AssertionError((hex(group), hex(base)))
    try:
        ExceptionGroup._probe = 1
    except TypeError:
        return 1
    del ExceptionGroup._probe
    raise AssertionError("ExceptionGroup accepted an attribute assignment")


def main():
    acc = 0
    i = 0
    while i < N:
        acc += check_flags()
        acc += check_descriptor_kinds()
        acc += check_exception_group_immutable()
        i += 1
    print(acc)


main()
