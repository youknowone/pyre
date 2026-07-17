"""CPython 3.14 surface and protocol parity for builtin functions."""

import builtins
import math


def raises(exc_type, action):
    try:
        action()
    except exc_type:
        return
    raise AssertionError(f"expected {exc_type.__name__}")


t = type(len)
assert t.__name__ == "builtin_function_or_method"
assert sorted(t.__dict__) == [
    "__call__",
    "__doc__",
    "__eq__",
    "__ge__",
    "__gt__",
    "__hash__",
    "__le__",
    "__lt__",
    "__module__",
    "__name__",
    "__ne__",
    "__qualname__",
    "__reduce__",
    "__repr__",
    "__self__",
    "__text_signature__",
]

assert not hasattr(len, "__dict__")
assert len.__module__ == "builtins"
assert len.__name__ == "len"
assert len.__qualname__ == "len"
assert len.__self__ is builtins
assert math.sqrt.__self__ is math
assert len.__reduce__() == "len"
assert len.__reduce_ex__(4) == "len"
for bound_builtin, owner, name in (
    (int.__new__, int, "__new__"),
    (dict.fromkeys, dict, "fromkeys"),
):
    reconstructor, reduce_args = bound_builtin.__reduce__()
    assert reconstructor is getattr
    assert reduce_args == (owner, name)
    rebuilt = reconstructor(*reduce_args)
    assert rebuilt.__self__ is owner
    assert rebuilt.__name__ == name
assert len.__repr__() == "<built-in function len>"
assert t.__dict__["__call__"](len, [1, 2, 3]) == 3

for method_name, args in (
    ("__call__", ()),
    ("__eq__", (len,)),
    ("__ne__", (len,)),
    ("__lt__", (len,)),
    ("__le__", (len,)),
    ("__gt__", (len,)),
    ("__ge__", (len,)),
    ("__hash__", ()),
    ("__repr__", ()),
    ("__reduce__", ()),
):
    raises(TypeError, lambda n=method_name, a=args: t.__dict__[n](object(), *a))

assert len == len
assert len != abs
assert isinstance(hash(len), int)
raises(TypeError, lambda: len < abs)
raises(TypeError, t)

descriptor_kinds = {
    "__doc__": "getset_descriptor",
    "__module__": "member_descriptor",
    "__name__": "getset_descriptor",
    "__qualname__": "getset_descriptor",
    "__self__": "getset_descriptor",
    "__text_signature__": "getset_descriptor",
}
for name, kind in descriptor_kinds.items():
    descriptor = t.__dict__[name]
    assert type(descriptor).__name__ == kind
    assert descriptor.__objclass__ is t
    raises(TypeError, lambda d=descriptor: d.__get__(1, int))
    if name == "__module__":
        continue
    raises(AttributeError, lambda n=name: setattr(len, n, "changed"))
    raises(AttributeError, lambda n=name: delattr(len, n))

len.__module__ = "changed"
assert len.__module__ == "changed"
del len.__module__
assert len.__module__ is None
len.__module__ = "builtins"

print("builtin_function_or_method Python 3.14 parity: ok")
