# pyre-check: no-cpython
# Every assertion here is a place where PyPy remains pyre's reference rather
# than CPython.  CPython 3.14 layout members and `__sizeof__` are now deliberate
# pyre compatibility surfaces and therefore no longer belong in this fixture.
N = 20000

class Heap:
    pass


class Derived(Heap):
    pass


def check_flags():
    # `get_flags`: _HEAPTYPE 1<<9, PATMA_SEQUENCE 1<<5, PATMA_MAPPING 1<<6,
    # _ABSTRACT 1<<20, Py_TPFLAGS_METHOD_DESCRIPTOR 1<<17.  _CPYTYPE marks
    # cpyext-defined static types and has no owner here.
    expected = {
        object: 0x0,
        type: 0x0,
        int: 0x0,
        set: 0x0,
        frozenset: 0x0,
        BaseException: 0x0,
        type(None): 0x0,
        str: 0x20,
        bytes: 0x20,
        bytearray: 0x20,
        list: 0x20,
        tuple: 0x20,
        range: 0x20,
        memoryview: 0x20,
        dict: 0x40,
        Heap: 0x200,
        Derived: 0x200,
    }
    for cls, want in expected.items():
        if cls.__flags__ != want:
            raise AssertionError((cls.__name__, hex(cls.__flags__), hex(want)))
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
    if ExceptionGroup.__flags__ != 0x0 or BaseExceptionGroup.__flags__ != 0x0:
        raise AssertionError(
            (hex(ExceptionGroup.__flags__), hex(BaseExceptionGroup.__flags__))
        )
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
