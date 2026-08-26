# pyre-check: gate=1
# TypeDef-member census lock: every PyPy builtins-visible TypeDef either
# has its members on the shipped type dict, or the member is a 3.14
# SPEC-omit (absent from CPython 3.14's type dict). Drives the real
# type objects, not a reimplementation.
import sys
import types
import weakref
from array import array
from collections import deque

from testutils import assert_raises


def _absent(tp, name):
    assert name not in tp.__dict__, (tp, name)


# --- members that CPython 3.14 and PyPy both publish, exercised here ---

assert (1.5).real == 1.5
assert (1.5).imag == 0.0
assert type((1.5).real) is float
assert (3 + 4j).real == 3.0
assert (3 + 4j).imag == 4.0

assert object.__class__ is type
o = object()
assert o.__class__ is object

assert type(float.real).__name__ == "getset_descriptor"
assert float.real.__name__ == "real"
assert float.real.__objclass__ is float
assert type(int.real).__name__ == "getset_descriptor"
assert int.real.__name__ == "real"
assert int.real.__objclass__ is int
assert type(int.imag).__name__ == "getset_descriptor"
assert int.imag.__name__ == "imag"
assert int.imag.__objclass__ is int
assert type(int.numerator).__name__ == "getset_descriptor"
assert int.numerator.__name__ == "numerator"
assert int.numerator.__objclass__ is int
assert type(int.denominator).__name__ == "getset_descriptor"
assert int.denominator.__objclass__ is int
assert type(range.start).__name__ == "member_descriptor"
assert range.start.__name__ == "start"
assert range.start.__objclass__ is range
for name in (
    "format",
    "itemsize",
    "ndim",
    "nbytes",
    "readonly",
    "shape",
    "strides",
    "suboffsets",
    "obj",
    "c_contiguous",
    "f_contiguous",
    "contiguous",
):
    descr = getattr(memoryview, name)
    assert type(descr).__name__ == "getset_descriptor", (name, type(descr).__name__)
    assert descr.__name__ == name
    assert descr.__objclass__ is memoryview

assert (True & True) is True
assert (True | False) is True
assert (True ^ True) is False
assert str(True) == "True"
assert True.__bool__() is True
assert str(7) == "7"

assert isinstance(memoryview.__doc__, str)
assert "memoryview" in memoryview.__doc__

for view_type in (type({}.keys()), type({}.values()), type({}.items())):
    assert "__doc__" in view_type.__dict__
    assert view_type.__dict__["__doc__"] is None

# `typeobject.py ensure_common_attributes` gives every TypeDef an own doc
# entry; `ensure_hash` suppresses an inherited object hash when equality is
# defined locally.
assert "__doc__" in array.__dict__
assert array.__doc__.startswith("array(typecode [, initializer]) -> array\n")
assert "itemsize -- the length in bytes of one array item" in array.__doc__
assert array.__dict__["__hash__"] is None
assert_raises(TypeError, hash, array("i"))

assert deque.__dict__["__doc__"] == (
    "A list-like sequence optimized for data accesses near its endpoints."
)
for weak_type in (
    weakref.ReferenceType,
    weakref.ProxyType,
    weakref.CallableProxyType,
):
    assert "__doc__" in weak_type.__dict__
    assert weak_type.__dict__["__doc__"] is None

wrapper_descriptor = type(object.__str__)
assert wrapper_descriptor.__name__ == "wrapper_descriptor"
assert "__repr__" in wrapper_descriptor.__dict__
assert wrapper_descriptor.__dict__["__repr__"](object.__str__) == repr(object.__str__)

# --- SPEC-omit: PyPy TypeDef key, CPython 3.14 type dict has no such key ---

_absent(bool, "__str__")
_absent(bool, "__bool__")
_absent(int, "__str__")
_absent(memoryview, "__weakref__")
_absent(type, "__weakref__")
_absent(type({}.keys()), "__new__")
_absent(type({}.values()), "__new__")
_absent(type({}.items()), "__new__")
_absent(type({}.keys()), "_dict")
_absent(type(type.__dict__), "__init__")
_absent(type((lambda: None).__code__), "__reduce__")
_absent(type((lambda: None).__code__), "__weakref__")
_absent(type(sys), "__reduce__")
_absent(type(sys), "__weakref__")
_absent(types.FunctionType, "__weakref__")
_absent(types.MethodType, "__weakref__")
_absent(types.GeneratorType, "__weakref__")
_absent(types.CoroutineType, "__weakref__")
_absent(types.AsyncGeneratorType, "__weakref__")
_absent(property, "__reduce__")

assert_raises(TypeError, type({}.keys()))
assert_raises(TypeError, type({}.keys()), {})

# Types stay weak-referenceable without publishing the descriptor.
view = memoryview(b"ab")
assert weakref.ref(view)() is view
assert_raises(AttributeError, lambda: view.__weakref__)

assert weakref.ref(type)() is type
assert_raises(AttributeError, lambda: type.__weakref__)


def _fn():
    pass


assert weakref.ref(_fn)() is _fn


class _C:
    def m(self):
        pass


bound = _C().m
assert weakref.ref(bound)() is bound


def _gen():
    yield 1


g = _gen()
assert weakref.ref(g)() is g


async def _co():
    return 1


co = _co()
assert weakref.ref(co)() is co
co.close()


async def _ag():
    yield 1


ag = _ag()
assert weakref.ref(ag)() is ag

code = _fn.__code__
assert weakref.ref(code)() is code
assert weakref.ref(sys)() is sys

try:
    raise ValueError("census")
except ValueError:
    tb = sys.exc_info()[2]
_absent(type(tb), "__reduce__")
_absent(type(tb), "__setstate__")


def _outer():
    x = 1

    def _inner():
        return x

    return _inner


cell = _outer().__closure__[0]
_absent(type(cell), "__reduce__")
_absent(type(cell), "__setstate__")
assert cell.cell_contents == 1

print("typedef-census-ok")
