doc_descriptor = type.__dict__["__doc__"]

assert doc_descriptor.__objclass__ is type
assert doc_descriptor.__get__(list, type) == list.__doc__
assert type.__doc__ == (
    "type(object) -> the object's type\n"
    "type(name, bases, dict, **kwds) -> a new type"
)


class Mutable:
    "before"


Mutable.__doc__ = "after"
assert Mutable.__doc__ == "after"
assert Mutable.__dict__["__doc__"] == "after"

try:
    doc_descriptor.__set__(list, "changed")
except TypeError as exc:
    assert "cannot set '__doc__' attribute of immutable type 'list'" in str(exc)
else:
    raise AssertionError("type.__doc__ descriptor changed an immutable type")

try:
    doc_descriptor.__delete__(Mutable)
except TypeError as exc:
    assert "cannot delete '__doc__' attribute of immutable type 'Mutable'" in str(exc)
else:
    raise AssertionError("type.__doc__ descriptor allowed deletion")
