import types


EXPECTED = {
    "__annotate__",
    "__annotations__",
    "__builtins__",
    "__call__",
    "__closure__",
    "__code__",
    "__defaults__",
    "__dict__",
    "__doc__",
    "__get__",
    "__globals__",
    "__kwdefaults__",
    "__module__",
    "__name__",
    "__new__",
    "__qualname__",
    "__repr__",
    "__type_params__",
}


assert set(types.FunctionType.__dict__) == EXPECTED


def f(value=1):
    return value


assert types.FunctionType.__call__(f, 7) == 7


def keyword_only(*, value):
    return value


assert types.FunctionType.__call__(keyword_only, value=8) == 8
assert types.FunctionType.__repr__(f).startswith("<function f at 0x")
assert types.FunctionType.__repr__(f).endswith(">")

first_dict = f.__dict__
assert first_dict == {}
assert f.__dict__ is first_dict
f.marker = 42
assert f.__dict__ == {"marker": 42}
replacement = {"other": 3}
f.__dict__ = replacement
assert f.__dict__ is replacement

try:
    f.__dict__ = []
except TypeError:
    pass
else:
    raise AssertionError("function.__dict__ accepted a non-dict")

assert f.__annotate__ is None


def annotate(format):
    assert format == 1
    return {"value": int}


f.__annotate__ = annotate
assert f.__annotate__ is annotate
assert f.__annotations__ == {"value": int}
f.__annotations__ = {"result": str}
assert f.__annotate__ is None
assert f.__annotations__ == {"result": str}

try:
    f.__annotate__ = 1
except TypeError as exc:
    assert str(exc) == "__annotate__ must be callable or None"
else:
    raise AssertionError("function.__annotate__ accepted a non-callable")

try:
    del f.__annotate__
except TypeError as exc:
    assert str(exc) == "__annotate__ cannot be deleted"
else:
    raise AssertionError("function.__annotate__ was deletable")

assert f.__type_params__ == ()
params = (int, str)
f.__type_params__ = params
assert f.__type_params__ is params

try:
    f.__type_params__ = []
except TypeError as exc:
    assert str(exc) == "__type_params__ must be set to a tuple"
else:
    raise AssertionError("function.__type_params__ accepted a non-tuple")

try:
    del f.__type_params__
except TypeError as exc:
    assert str(exc) == "__type_params__ must be set to a tuple"
else:
    raise AssertionError("function.__type_params__ was deletable")


def identity[T](value: T) -> T:
    return value


assert len(identity.__type_params__) == 1
assert identity.__type_params__[0].__name__ == "T"
assert identity(5) == 5
