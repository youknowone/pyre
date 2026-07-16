"""Python 3.14 / PyPy structural parity for ``classmethod``.

PyPy ``interpreter/function.py:718-768 ClassMethod`` supplies the wrapped
callable, lazy instance dictionary, descriptor and repr operations.
``interpreter/typedef.py:878-908`` exposes them in the type dictionary.
Python 3.14 additionally proxies PEP 649's ``__annotations__`` and
``__annotate__`` attributes and exposes ``__class_getitem__``; it omits
PyPy 3.11's ``__reduce_ex__`` entry and does not make the wrapper callable.
"""


EXPECTED_SURFACE = {
    "__annotate__",
    "__annotations__",
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

assert set(classmethod.__dict__) == EXPECTED_SURFACE
assert classmethod.__doc__.startswith("Convert a function to be a class method.\n")
assert classmethod.__doc__.endswith("see the staticmethod builtin.")


def wrapped(cls, value: int = 2) -> str:
    "wrapped doc"
    return f"{cls.__name__}:{value}"


cm = classmethod(wrapped)
assert cm.__func__ is wrapped
assert cm.__wrapped__ is wrapped
assert cm.__dict__ == {
    "__module__": __name__,
    "__name__": "wrapped",
    "__qualname__": "wrapped",
    "__doc__": "wrapped doc",
}
assert repr(cm).startswith("<classmethod(<function wrapped")
assert repr(cm).endswith(")>")
assert not callable(cm)
try:
    cm()
except TypeError as exc:
    assert str(exc) == "'classmethod' object is not callable"
else:
    assert False, "a raw classmethod wrapper must not be callable"


class Base:
    method = cm


class Derived(Base):
    pass


assert Base.method(3) == "Base:3"
assert Base().method(4) == "Base:4"
assert Derived.method(5) == "Derived:5"
assert Derived().method(6) == "Derived:6"
bound = cm.__get__(None, Derived)
assert bound.__func__ is wrapped
assert bound.__self__ is Derived
assert bound(7) == "Derived:7"

# Python 3.14 cm_descr_get binds the stored object directly rather than
# invoking a descriptor nested inside classmethod.
nested = property(lambda obj: "property result")
nested_bound = classmethod(nested).__get__(None, Base)
assert nested_bound.__func__ is nested
assert nested_bound.__self__ is Base

# PyPy ClassMethod.getdict/setdict: arbitrary attributes and wholesale dict
# replacement use the object's w_dict field, including dict subclasses.
cm.extra = 9
assert cm.extra == 9
assert cm.__dict__["extra"] == 9


class DictSubclass(dict):
    pass


replacement = DictSubclass(marker=11)
cm.__dict__ = replacement
assert cm.__dict__ is replacement
assert cm.marker == 11
cm.more = 12
assert replacement["more"] == 12

try:
    cm.__dict__ = 1
except TypeError as exc:
    assert str(exc) == "__dict__ must be set to a dictionary, not a 'int'"
else:
    assert False, "classmethod.__dict__ must reject non-dicts"

try:
    del cm.__dict__
except TypeError as exc:
    assert str(exc) == "cannot delete __dict__"
else:
    assert False, "classmethod.__dict__ cannot be deleted"

# CPython 3.14 descriptor_get_wrapped_attribute: first read obtains the
# wrapped function's value and caches it locally. Setting/deleting affects the
# wrapper dictionary, never the wrapped function.
cm = classmethod(wrapped)
assert "__annotations__" not in cm.__dict__
assert "__annotate__" not in cm.__dict__
assert cm.__annotations__ is wrapped.__annotations__
assert cm.__annotate__ is wrapped.__annotate__
assert cm.__dict__["__annotations__"] is wrapped.__annotations__
assert cm.__dict__["__annotate__"] is wrapped.__annotate__

wrapped_annotations = wrapped.__annotations__
wrapped_annotate = wrapped.__annotate__
cm.__annotations__ = 42
cm.__annotate__ = 43
assert cm.__annotations__ == 42
assert cm.__annotate__ == 43
assert wrapped.__annotations__ is wrapped_annotations
assert wrapped.__annotate__ is wrapped_annotate
del cm.__annotations__
del cm.__annotate__
assert cm.__annotations__ is wrapped_annotations
assert cm.__annotate__ is wrapped_annotate

fresh = classmethod(wrapped)
for name in ("__annotations__", "__annotate__"):
    try:
        delattr(fresh, name)
    except AttributeError as exc:
        assert str(exc) == f"'classmethod' object has no attribute '{name}'"
    else:
        assert False, f"deleting uncached {name} must fail"

try:
    classmethod()
except TypeError as exc:
    assert str(exc) == "classmethod expected 1 argument, got 0"
else:
    assert False, "classmethod requires one argument"

try:
    classmethod(wrapped, wrapped)
except TypeError as exc:
    assert str(exc) == "classmethod expected 1 argument, got 2"
else:
    assert False, "classmethod accepts exactly one argument"

try:
    classmethod(function=wrapped)
except TypeError as exc:
    assert str(exc) == "classmethod() takes no keyword arguments"
else:
    assert False, "classmethod's argument is positional-only"


class ClassMethodSubclass(classmethod):
    pass


sub = ClassMethodSubclass(wrapped)
assert type(sub) is ClassMethodSubclass
assert sub.__func__ is wrapped


class CallableClassMethod(classmethod):
    def __call__(self, value):
        return self.__func__(Base, value)


callable_sub = CallableClassMethod(wrapped)
assert callable(callable_sub)
assert callable_sub(8) == "Base:8"
assert callable_sub(value=9) == "Base:9"

alias = classmethod[int]
assert type(alias).__name__ == "GenericAlias"
assert alias.__origin__ is classmethod
assert alias.__args__ == (int,)

print("OK")
