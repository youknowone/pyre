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
    try:
        iterator_type()
    except TypeError:
        pass
    else:
        raise AssertionError("dict iterator must not be directly instantiable")

print("OK")
