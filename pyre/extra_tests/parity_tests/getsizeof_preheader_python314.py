"""Python 3.14 ``sys.getsizeof`` pre-header, isolated from ``__sizeof__``.

``_PySys_GetSizeOf`` calls the type's ``__sizeof__`` and then adds
``_PyType_PreHeaderSize(Py_TYPE(o))``, which is two independent terms: a
two-word ``PyGC_Head`` when ``_PyObject_IS_GC(o)``, plus two words of managed
dict/weakref prefix when the instance type asks for one.

Every assertion below is on ``getsizeof(o) - type(o).__sizeof__(o)``, so what
it pins is the pre-header alone; a type whose ``__sizeof__`` still models its
payload approximately does not make this file fail for the wrong reason.

The discriminating rows are the ones where the pre-header does not follow the
object's live collector state:

* ``()`` is untracked -- ``gc.is_tracked(()) is False`` -- and is still charged
  a ``PyGC_Head``, because the term is ``tuple``'s ``Py_TPFLAGS_HAVE_GC``, not
  the object's tracking bit.
* ``object()``, ``2 ** 300``, ``b""``, ``bytearray()`` and ``range(3)`` are
  charged nothing, however the runtime happens to allocate them, because those
  types do not carry the flag.
* ``type`` is charged nothing while a class built by ``type_new`` is charged,
  which is ``type_is_gc`` -- ``tp_is_gc`` answering with
  ``Py_TPFLAGS_HEAPTYPE`` -- rather than anything about the flag.
"""

import sys
import weakref

WORD = 8
GC_HEAD = 2 * WORD
MANAGED = 2 * WORD


class Plain:
    pass


class Slotted:
    __slots__ = ("a",)


class WeakSlot:
    __slots__ = ("__weakref__",)


def function():
    pass


def generator():
    yield 1


# No pre-header: the type declares neither Py_TPFLAGS_HAVE_GC nor a managed
# dict/weakref prefix.
untracked = [
    object(),
    None,
    Ellipsis,
    NotImplemented,
    5,
    2**300,
    True,
    1.5,
    1j,
    "abc",
    b"abc",
    bytearray(b"a"),
    range(3),
    function.__code__,
    int,
    type,
    object,
]

# Py_TPFLAGS_HAVE_GC only. `()` is here while untracked, and the static type
# objects above are absent while `Plain` is here, for the two reasons in the
# module docstring.
gc_only = [
    [],
    (),
    (1,),
    {1: 2},
    {1},
    frozenset({1}),
    slice(1),
    memoryview(b"ab"),
    map(str, []),
    filter(None, []),
    zip(),
    enumerate([]),
    reversed([]),
    sys,
    function,
    generator(),
    len,
    list.append,
    object.__init__,
    property(),
    staticmethod(function),
    classmethod(function),
    weakref.ref(Plain()),
    Plain,
    Slotted(),
]

# Both terms: a heap instance that keeps a managed dict, or a managed weakref
# slot, on top of the header its heap type's flag charges.
gc_and_managed = [
    Plain(),
    WeakSlot(),
]

for expected, group in (
    (0, untracked),
    (GC_HEAD, gc_only),
    (GC_HEAD + MANAGED, gc_and_managed),
):
    for obj in group:
        actual = sys.getsizeof(obj) - type(obj).__sizeof__(obj)
        assert actual == expected, (type(obj).__name__, actual, expected)

# `sys.getsizeof` reads `__sizeof__` off the type and refuses a negative or
# non-integer answer.
class Negative:
    def __sizeof__(self):
        return -1


class NotAnInt:
    def __sizeof__(self):
        return "x"


try:
    sys.getsizeof(Negative())
except ValueError as exc:
    assert str(exc) == "__sizeof__() should return >= 0", str(exc)
else:
    raise AssertionError("negative __sizeof__ must raise ValueError")

try:
    sys.getsizeof(NotAnInt())
except TypeError:
    pass
else:
    raise AssertionError("non-integer __sizeof__ must raise TypeError")

# The default argument is returned only when the type has no `__sizeof__`.
assert sys.getsizeof([], 123) == sys.getsizeof([])

print("OK")
