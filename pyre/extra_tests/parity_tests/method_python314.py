import types


EXPECTED = {
    "__call__",
    "__doc__",
    "__eq__",
    "__func__",
    "__ge__",
    "__get__",
    "__getattribute__",
    "__gt__",
    "__hash__",
    "__le__",
    "__lt__",
    "__ne__",
    "__new__",
    "__reduce__",
    "__repr__",
    "__self__",
}

assert set(types.MethodType.__dict__) == EXPECTED
assert types.MethodType.__doc__ == "Create a bound instance method object."
assert types.FunctionType.__doc__.startswith("Create a function object.\n\n  code\n")


class C:
    def method(self, value=1, *, scale=1):
        "method documentation"
        return self.base + value * scale


c = C()
c.base = 10
function = C.__dict__["method"]

# Python 3.14 function descriptor behavior, including omitted owner.
try:
    function.__get__(None)
except TypeError as exc:
    assert str(exc) == "__get__(None, None) is invalid"
else:
    raise AssertionError("function.__get__(None) did not fail")

assert function.__get__(None, C) is function
bound = function.__get__(c)
assert isinstance(bound, types.MethodType)
assert bound.__func__ is function
assert bound.__self__ is c
assert bound(2, scale=3) == 16
assert types.MethodType.__call__(bound, 4, scale=2) == 18

# An already-bound method stays bound under descriptor access.
try:
    bound.__get__(None)
except TypeError as exc:
    assert str(exc) == "__get__(None, None) is invalid"
else:
    raise AssertionError("method.__get__(None) did not fail")

assert bound.__get__(None, C) is bound
assert bound.__get__(object()) is bound

# Method attributes are forwarded to the wrapped function after the
# method type's own namespace has been checked.
assert bound.__doc__ == "method documentation"
assert bound.__name__ == "method"
assert bound.__qualname__ == "C.method"
assert bound.__annotations__ == function.__annotations__
assert types.MethodType.__getattribute__(bound, "__name__") == "method"

constructed = types.MethodType(function, c)
assert constructed.__func__ is function
assert constructed.__self__ is c
assert constructed() == 11

for args in ((), (function,), (function, c, C)):
    try:
        types.MethodType(*args)
    except TypeError as exc:
        assert str(exc) == f"method expected 2 arguments, got {len(args)}"
    else:
        raise AssertionError(f"MethodType accepted {len(args)} arguments")

try:
    types.MethodType(1, c)
except TypeError as exc:
    assert str(exc) == "first argument must be callable"
else:
    raise AssertionError("MethodType accepted a non-callable")

try:
    types.MethodType(function, None)
except TypeError as exc:
    assert str(exc) == "instance must not be None"
else:
    raise AssertionError("MethodType accepted None as the instance")

same = types.MethodType(function, c)
other = C()
other.base = 10
different_self = types.MethodType(function, other)
assert bound == same
assert bound != different_self
assert types.MethodType.__eq__(bound, object()) is NotImplemented
assert types.MethodType.__ne__(bound, object()) is NotImplemented
for name in ("__lt__", "__le__", "__gt__", "__ge__"):
    assert getattr(types.MethodType, name)(bound, same) is NotImplemented

assert hash(bound) == hash(same)
assert repr(bound).startswith("<bound method C.method of <")
assert repr(bound).endswith(">>")

reconstructor, reduce_args = bound.__reduce__()
assert reconstructor is getattr
assert reduce_args == (c, "method")
assert reconstructor(*reduce_args).__func__ is function

try:
    types.MethodType.__getattribute__(bound, 123)
except TypeError:
    pass
else:
    raise AssertionError("method.__getattribute__ accepted a non-string name")

for name, args in (
    ("__call__", ()),
    ("__get__", (None, C)),
    ("__getattribute__", ("__name__",)),
    ("__eq__", (bound,)),
    ("__ne__", (bound,)),
    ("__lt__", (bound,)),
    ("__le__", (bound,)),
    ("__gt__", (bound,)),
    ("__ge__", (bound,)),
    ("__hash__", ()),
    ("__repr__", ()),
    ("__reduce__", ()),
):
    try:
        types.MethodType.__dict__[name](object(), *args)
    except TypeError:
        pass
    else:
        raise AssertionError(f"method {name} accepted a foreign receiver")

for name in ("__doc__", "__func__", "__self__"):
    try:
        types.MethodType.__dict__[name].__get__(object())
    except TypeError:
        pass
    else:
        raise AssertionError(f"method {name} accepted a foreign receiver")

print("OK")
