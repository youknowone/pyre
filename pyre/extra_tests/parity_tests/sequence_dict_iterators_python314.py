"""PyPy iterator typedef protocols with Python 3.14 concrete surfaces."""

import operator


class Sequence:
    def __getitem__(self, index):
        if index >= 3:
            raise IndexError
        return index * 2


iterator = iter(Sequence())
iterator_type = type(iterator)
assert iterator_type.__name__ == "iterator"
assert iterator_type.__doc__ is None
assert {
    "__doc__",
    "__iter__",
    "__next__",
    "__reduce__",
    "__length_hint__",
    "__setstate__",
} <= set(iterator_type.__dict__)
assert iter(iterator) is iterator
assert next(iterator) == 0
iterator.__setstate__(-10)
assert next(iterator) == 0
iterator.__setstate__(2)
assert list(iterator) == [4]

try:
    iterator_type()
except TypeError:
    pass
else:
    raise AssertionError("iterator must not be directly instantiable")

source = {"a": 1, "b": 2, "c": 3}
factories = (
    (iter, "dict_keyiterator", ["a", "b", "c"]),
    (lambda d: iter(d.values()), "dict_valueiterator", [1, 2, 3]),
    (lambda d: iter(d.items()), "dict_itemiterator", [("a", 1), ("b", 2), ("c", 3)]),
)
for factory, expected_name, expected in factories:
    iterator = factory(source)
    iterator_type = type(iterator)
    assert iterator_type.__name__ == expected_name
    assert iterator_type.__doc__ is None
    assert {
        "__doc__",
        "__iter__",
        "__next__",
        "__length_hint__",
        "__reduce__",
    } <= set(iterator_type.__dict__)
    assert operator.length_hint(iterator) == 3
    assert next(iterator) == expected[0]
    assert operator.length_hint(iterator) == 2
    reduced = iterator.__reduce__()
    assert reduced[0] is iter and list(reduced[1][0]) == expected[1:]
    foreign = iter([])
    for name in ("__iter__", "__next__", "__length_hint__", "__reduce__"):
        descriptor = iterator_type.__dict__[name]
        try:
            descriptor(foreign)
        except TypeError as exc:
            if name in ("__iter__", "__next__"):
                message = (
                    f"descriptor '{name}' requires a '{expected_name}' object "
                    "but received a 'list_iterator'"
                )
            else:
                message = (
                    f"descriptor '{name}' for '{expected_name}' objects "
                    "doesn't apply to a 'list_iterator' object"
                )
            assert str(exc) == message
        else:
            raise AssertionError("dict iterator accepted a foreign receiver")
    try:
        iterator_type()
    except TypeError:
        pass
    else:
        raise AssertionError("dict iterator must not be directly instantiable")

reverse_factories = (
    (reversed, "dict_reversekeyiterator", ["c", "b", "a"]),
    (lambda d: reversed(d.values()), "dict_reversevalueiterator", [3, 2, 1]),
    (
        lambda d: reversed(d.items()),
        "dict_reverseitemiterator",
        [("c", 3), ("b", 2), ("a", 1)],
    ),
)
iterator_surface = {
    "__doc__",
    "__iter__",
    "__next__",
    "__length_hint__",
    "__reduce__",
}
for factory, expected_name, expected in reverse_factories:
    source = {"a": 1, "b": 2, "c": 3}
    iterator = factory(source)
    iterator_type = type(iterator)
    assert iterator_type.__name__ == expected_name
    assert set(iterator_type.__dict__) == iterator_surface
    assert iter(iterator) is iterator
    assert operator.length_hint(iterator) == 3
    assert next(iterator) == expected[0]
    assert operator.length_hint(iterator) == 2
    reduced = iterator.__reduce__()
    assert reduced[0] is iter and reduced[1] == (expected[1:],)

    foreign = iter({"x": 1})
    for name in ("__iter__", "__next__", "__length_hint__", "__reduce__"):
        descriptor = iterator_type.__dict__[name]
        try:
            descriptor(foreign)
        except TypeError as exc:
            if name in ("__iter__", "__next__"):
                message = (
                    f"descriptor '{name}' requires a '{expected_name}' object "
                    "but received a 'dict_keyiterator'"
                )
            else:
                message = (
                    f"descriptor '{name}' for '{expected_name}' objects "
                    "doesn't apply to a 'dict_keyiterator' object"
                )
            assert str(exc) == message
        else:
            raise AssertionError("reverse dict iterator accepted a foreign receiver")

    source["d"] = 4
    assert operator.length_hint(iterator) == 0
    try:
        next(iterator)
    except RuntimeError as exc:
        assert str(exc) == "dictionary changed size during iteration"
    else:
        raise AssertionError("reverse dict iterator missed a size mutation")

    try:
        iterator_type()
    except TypeError:
        pass
    else:
        raise AssertionError("reverse dict iterator must not be directly instantiable")

reverse_owners = (
    (dict, "dict"),
    (type({}.keys()), "dict_keys"),
    (type({}.values()), "dict_values"),
    (type({}.items()), "dict_items"),
    (type(type.__dict__), "mappingproxy"),
)
for owner, owner_name in reverse_owners:
    descriptor = owner.__dict__["__reversed__"]
    try:
        descriptor(None)
    except TypeError as exc:
        assert str(exc) == (
            f"descriptor '__reversed__' for '{owner_name}' objects "
            "doesn't apply to a 'NoneType' object"
        )
    else:
        raise AssertionError(f"{owner_name}.__reversed__ accepted a foreign receiver")

print("OK")
