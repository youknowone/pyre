import sys
import types
import warnings


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
if sys.implementation.name != "cpython":
    # PyPy typedef.py:805-806 exposes these interpreter-level Function slots.
    EXPECTED.update({"__objclass__", "__text_signature__"})


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
assert "other" in dir(f)

try:
    f.__dict__ = []
except TypeError:
    pass
else:
    raise AssertionError("function.__dict__ accepted a non-dict")

for name, message in (
    ("__dict__", "cannot delete __dict__"),
    ("__name__", "__name__ must be set to a string object"),
    ("__qualname__", "__qualname__ must be set to a string object"),
    ("__code__", "__code__ must be set to a code object"),
):
    try:
        delattr(f, name)
    except TypeError as exc:
        assert str(exc) == message
    else:
        raise AssertionError(f"function {name} was deletable")


def generator():
    yield 1


original_code = f.__code__
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    f.__code__ = generator.__code__
assert len(caught) == 1
assert caught[0].category is DeprecationWarning
assert "code object of non-matching type" in str(caught[0].message)
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    f.__code__ = original_code

assert f.__annotate__ is None


def annotate(annotation_format):
    assert annotation_format == 1
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

for descriptor_name in (
    "__name__",
    "__qualname__",
    "__defaults__",
    "__kwdefaults__",
    "__code__",
    "__annotations__",
    "__annotate__",
    "__dict__",
    "__type_params__",
):
    descriptor = type(f).__dict__[descriptor_name]
    try:
        descriptor.__get__(42, int)
    except TypeError:
        pass
    else:
        raise AssertionError(f"function {descriptor_name} accepted a foreign receiver")

for method_name, args in (
    ("__call__", ()),
    ("__get__", (None, type(None))),
    ("__repr__", ()),
):
    # PyPy function.py deliberately represents builtins and Python functions
    # with the same Function class, so `len` is a valid receiver there. Use an
    # object foreign to both models while retaining CPython's stricter check.
    foreign_receiver = len if sys.implementation.name == "cpython" else 42
    try:
        type(f).__dict__[method_name](foreign_receiver, *args)
    except TypeError:
        pass
    else:
        raise AssertionError(f"function {method_name} accepted a foreign receiver")

print("OK")
