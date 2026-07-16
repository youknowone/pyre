"""Python 3.14 / PyPy structural parity for ``staticmethod``.

PyPy ``interpreter/function.py:671-716 StaticMethod`` supplies the wrapped
callable, lazy instance dictionary, descriptor, call and repr operations.
``interpreter/typedef.py:852-877`` exposes them in the type dictionary.
Python 3.14 additionally proxies PEP 649's ``__annotations__`` and
``__annotate__`` attributes and exposes ``__class_getitem__``; it omits
PyPy 3.11's ``__reduce_ex__`` entry.
"""


EXPECTED_SURFACE = {
    "__annotate__",
    "__annotations__",
    "__call__",
    "__class_getitem__",
    "__dict__",
    "__doc__",
    "__func__",
    "__get__",
    "__init__",
    "__isabstractmethod__",
    "__new__",
    "__repr__",
    "__wrapped__",
}

assert set(staticmethod.__dict__) == EXPECTED_SURFACE
assert staticmethod.__doc__.startswith("Convert a function to be a static method.\n")
assert staticmethod.__doc__.endswith("see the classmethod builtin.")


def wrapped(a: int, *, b: int = 2) -> str:
    "wrapped doc"
    return str(a + b)


sm = staticmethod(wrapped)
assert sm.__func__ is wrapped
assert sm.__wrapped__ is wrapped
assert sm.__dict__ == {
    "__module__": __name__,
    "__name__": "wrapped",
    "__qualname__": "wrapped",
    "__doc__": "wrapped doc",
}
assert repr(sm).startswith("<staticmethod(<function wrapped")
assert repr(sm).endswith(")>")
assert sm(3, b=4) == "7"
assert sm.__call__(3, b=4) == "7"
assert sm.__get__(None, object) is wrapped
assert sm.__get__(object(), object) is wrapped

# PyPy StaticMethod.getdict/setdict: arbitrary attributes and wholesale dict
# replacement use the object's w_dict field, including dict subclasses.
sm.extra = 9
assert sm.extra == 9
assert sm.__dict__["extra"] == 9


class DictSubclass(dict):
    pass


replacement = DictSubclass(marker=11)
sm.__dict__ = replacement
assert sm.__dict__ is replacement
assert sm.marker == 11
sm.more = 12
assert replacement["more"] == 12

try:
    sm.__dict__ = 1
except TypeError as exc:
    assert str(exc) == "__dict__ must be set to a dictionary, not a 'int'"
else:
    assert False, "staticmethod.__dict__ must reject non-dicts"

try:
    del sm.__dict__
except TypeError as exc:
    assert str(exc) == "cannot delete __dict__"
else:
    assert False, "staticmethod.__dict__ cannot be deleted"

# CPython 3.14 descriptor_get_wrapped_attribute: first read obtains the
# wrapped function's value and caches it locally. Setting/deleting affects the
# wrapper dictionary, never the wrapped function.
sm = staticmethod(wrapped)
assert "__annotations__" not in sm.__dict__
assert "__annotate__" not in sm.__dict__
assert sm.__annotations__ is wrapped.__annotations__
assert sm.__annotate__ is wrapped.__annotate__
assert sm.__dict__["__annotations__"] is wrapped.__annotations__
assert sm.__dict__["__annotate__"] is wrapped.__annotate__

wrapped_annotations = wrapped.__annotations__
wrapped_annotate = wrapped.__annotate__
sm.__annotations__ = 42
sm.__annotate__ = 43
assert sm.__annotations__ == 42
assert sm.__annotate__ == 43
assert wrapped.__annotations__ is wrapped_annotations
assert wrapped.__annotate__ is wrapped_annotate
del sm.__annotations__
del sm.__annotate__
assert sm.__annotations__ is wrapped_annotations
assert sm.__annotate__ is wrapped_annotate

fresh = staticmethod(wrapped)
for name in ("__annotations__", "__annotate__"):
    try:
        delattr(fresh, name)
    except AttributeError as exc:
        assert str(exc) == f"'staticmethod' object has no attribute '{name}'"
    else:
        assert False, f"deleting uncached {name} must fail"

try:
    staticmethod()
except TypeError as exc:
    assert str(exc) == "staticmethod expected 1 argument, got 0"
else:
    assert False, "staticmethod requires one argument"

try:
    staticmethod(wrapped, wrapped)
except TypeError as exc:
    assert str(exc) == "staticmethod expected 1 argument, got 2"
else:
    assert False, "staticmethod accepts exactly one argument"

try:
    staticmethod(function=wrapped)
except TypeError as exc:
    assert str(exc) == "staticmethod() takes no keyword arguments"
else:
    assert False, "staticmethod's argument is positional-only"


class StaticMethodSubclass(staticmethod):
    pass


sub = StaticMethodSubclass(wrapped)
assert type(sub) is StaticMethodSubclass
assert sub(5, b=6) == "11"
assert sub.__func__ is wrapped

alias = staticmethod[int]
assert type(alias).__name__ == "GenericAlias"
assert alias.__origin__ is staticmethod
assert alias.__args__ == (int,)

print("OK")
