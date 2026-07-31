"""Parity test: a builtin type's `__new__` entry IS the bound carrier.

`add_tp_new_wrapper` stores `PyCFunction_NewEx(tp_new_methoddef, type)`
straight into the type dict, so `T.__dict__['__new__']` is a
`builtin_function_or_method` whose `m_self` is `T`, and
`T.__new__ is T.__dict__['__new__']` — a `PyCFunction` is not a descriptor,
so nothing rebinds it on access.  That non-binding is the whole point:
`cls.__new__(other, ...)` has to pass `other` as the first argument, not
`cls`.

Two neighbours are deliberately pinned here as contrasts, because the
obvious way to get `__new__` non-binding is a `staticmethod` wrapper and
that spelling IS correct for both of them:
  * a `METH_STATIC` `tp_methods` entry (`str.maketrans`) really is a
    `staticmethod` wrapping a builtin carrier;
  * a Python-level `def __new__` really is auto-wrapped at class creation.

`meth_repr` keys off `m_self` rather than the carrier's spelling, so the
same object reports as a *method* once it carries a receiver — which is why
`repr(tuple.__new__)` names `type` (the type of the receiver, not the
receiver itself) while `repr(len)` stays a plain function.
"""

import os
import pickle
import sys

TYPES = [int, str, bytes, bytearray, list, tuple, dict, set, frozenset,
         float, complex, range, object, type, super, memoryview,
         Exception, ValueError, OSError, os.stat_result]

for owner in TYPES:
    entry = owner.__dict__["__new__"]
    assert type(entry).__name__ == "builtin_function_or_method", (owner, type(entry))
    assert type(entry) is type(len), owner
    # The stored object IS what attribute access hands back; no rebinding,
    # and no fresh carrier minted per access.
    assert owner.__new__ is entry, owner
    assert entry.__self__ is owner, owner
    assert entry.__qualname__ == f"{owner.__name__}.__new__", owner
    assert repr(entry).startswith("<built-in method __new__ of type object at 0x"), repr(entry)
    # `__objclass__` belongs to the descriptor kinds; a carrier has none.
    assert not hasattr(entry, "__objclass__"), owner


# A subclass that does not define its own inherits the identical object.
class _MyInt(int):
    pass


assert _MyInt.__new__ is int.__new__
assert "__new__" not in _MyInt.__dict__
# `bool` defines its own, so it is a different object with its own receiver.
assert bool.__new__ is not int.__new__
assert bool.__new__.__self__ is bool

# Instance access does not bind the instance either.
_five = 5
assert type(_five.__new__).__name__ == "builtin_function_or_method"
assert _five.__new__ is int.__new__
assert _five.__new__.__self__ is int

# Calling still routes the explicit class argument.
assert int.__new__(int) == 0
assert str.__new__(str, "x") == "x"
assert type(int.__new__(_MyInt)) is _MyInt
try:
    int.__new__(bool)
except TypeError as exc:
    assert str(exc) == "int.__new__(bool) is not safe, use bool.__new__()", exc
else:
    raise AssertionError("an unsafe cross-type __new__ must be rejected")


# `meth_repr`: a null or module receiver keeps the plain-function wording.
assert repr(len) == "<built-in function len>"
assert repr(sys.exit) == "<built-in function exit>"
# ... and an ordinary bound builtin names its receiver's own type.
assert repr([].append).startswith("<built-in method append of list object at 0x")


# Contrast 1: a METH_STATIC `tp_methods` entry keeps the staticmethod wrapper.
assert type(str.__dict__["maketrans"]).__name__ == "staticmethod"
assert type(bytes.__dict__["maketrans"]).__name__ == "staticmethod"
assert str.maketrans({"a": "b"}) == {97: "b"}


# Contrast 2: a Python-level `__new__` is still auto-wrapped at class creation.
class _User:
    def __new__(cls):
        return super().__new__(cls)


assert type(_User.__dict__["__new__"]).__name__ == "staticmethod"
assert type(_User()) is _User


# The carrier swap must not disturb the reduce protocol, which reaches
# `cls.__new__` through `copyreg`.
assert pickle.loads(pickle.dumps(_MyInt(5))) == 5
assert pickle.loads(pickle.dumps(OSError("x"))).args == ("x",)

print("OK")
