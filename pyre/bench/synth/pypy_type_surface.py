# pyre-check: no-cpython
# Every assertion here is a place the reference has something pyre deliberately
# does not, so the reference cannot be an oracle for this fixture: it is the
# side being diverged from. pypy is.
#
# `type.__basicsize__` and its three siblings describe a C struct behind a type
# and `W_TypeObject.typedef` (objspace/std/typeobject.py) exposes neither them
# nor a descriptor for them; `__flags__` there is a six-bit mask built by
# `get_flags` and reached through a `GetSetProperty`, not a full `tp_flags`
# word behind a member descriptor; `__sizeof__` exists on no typedef at all and
# `vm.py getsizeof` returns its default or raises.
import sys

N = 20000

LAYOUT_MEMBERS = ("__basicsize__", "__itemsize__", "__weakrefoffset__", "__dictoffset__")


class Heap:
    pass


class Derived(Heap):
    pass


def check_layout_members_absent():
    seen = 0
    for name in LAYOUT_MEMBERS:
        if name in type.__dict__:
            raise AssertionError(f"type.__dict__ carries {name}")
        for owner in (object, type, int, str, tuple, BaseException, Heap):
            if hasattr(owner, name):
                raise AssertionError(f"{owner.__name__} carries {name}")
            seen += 1
    return seen


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


def check_sizeof_absent():
    owners = (object, type, int, str, list, dict, bytearray, set, frozenset)
    seen = 0
    for owner in owners:
        if "__sizeof__" in owner.__dict__:
            raise AssertionError(f"{owner.__name__} carries __sizeof__")
        seen += 1
    if hasattr(object, "__sizeof__"):
        raise AssertionError("object exposes __sizeof__")
    return seen


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


def check_getsizeof():
    sentinel = object()
    if sys.getsizeof(sentinel, sentinel) is not sentinel:
        raise AssertionError("getsizeof did not return its default by identity")
    try:
        sys.getsizeof(sentinel)
    except TypeError as error:
        if str(error) != sys.getsizeof.__doc__:
            raise AssertionError("getsizeof message is not its docstring")
        return 1
    raise AssertionError("getsizeof accepted a single argument")


def main():
    acc = 0
    i = 0
    while i < N:
        acc += check_layout_members_absent()
        acc += check_flags()
        acc += check_descriptor_kinds()
        acc += check_sizeof_absent()
        acc += check_exception_group_immutable()
        acc += check_getsizeof()
        i += 1
    print(acc)


main()
