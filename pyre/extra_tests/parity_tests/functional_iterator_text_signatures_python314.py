"""CPython 3.14 signatures for the builtin functional iterator types."""

import inspect


METHODS = {
    enumerate: ("__iter__", "__next__", "__reduce__", "__class_getitem__"),
    reversed: (
        "__iter__",
        "__next__",
        "__length_hint__",
        "__reduce__",
        "__setstate__",
    ),
    map: ("__iter__", "__next__", "__reduce__", "__setstate__"),
    filter: ("__iter__", "__next__", "__reduce__"),
    zip: ("__iter__", "__next__", "__reduce__", "__setstate__"),
}

for typ, names in METHODS.items():
    assert typ.__dict__["__new__"].__text_signature__ == "($type, *args, **kwargs)"
    for name in names:
        expected = (
            "($type, object, /)"
            if name == "__class_getitem__"
            else "($self, object, /)"
            if name == "__setstate__"
            else "($self, /)"
        )
        assert typ.__dict__[name].__text_signature__ == expected, (typ, name)

assert str(inspect.signature(enumerate.__next__)) == "(self, /)"
assert str(inspect.signature(reversed.__setstate__)) == "(self, object, /)"
assert str(inspect.signature(enumerate.__class_getitem__)) == "(object, /)"

print("OK")
