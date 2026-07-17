EXPECTED = {
    "__add__",
    "__class_getitem__",
    "__contains__",
    "__delitem__",
    "__doc__",
    "__eq__",
    "__ge__",
    "__getitem__",
    "__gt__",
    "__hash__",
    "__iadd__",
    "__imul__",
    "__init__",
    "__iter__",
    "__le__",
    "__len__",
    "__lt__",
    "__mul__",
    "__ne__",
    "__new__",
    "__repr__",
    "__reversed__",
    "__rmul__",
    "__setitem__",
    "__sizeof__",
    "append",
    "clear",
    "copy",
    "count",
    "extend",
    "index",
    "insert",
    "pop",
    "remove",
    "reverse",
    "sort",
}

assert set(list.__dict__) == EXPECTED

WRAPPER_DESCRIPTORS = {
    "__add__",
    "__contains__",
    "__delitem__",
    "__eq__",
    "__ge__",
    "__gt__",
    "__iadd__",
    "__imul__",
    "__init__",
    "__iter__",
    "__le__",
    "__len__",
    "__lt__",
    "__mul__",
    "__ne__",
    "__repr__",
    "__rmul__",
    "__setitem__",
}

METHOD_DESCRIPTORS = {
    "__getitem__",
    "__reversed__",
    "__sizeof__",
    "append",
    "clear",
    "copy",
    "count",
    "extend",
    "index",
    "insert",
    "pop",
    "remove",
    "reverse",
    "sort",
}

for name in WRAPPER_DESCRIPTORS:
    descriptor = list.__dict__[name]
    try:
        descriptor(())
    except TypeError as exc:
        assert str(exc) == (
            f"descriptor '{name}' requires a 'list' object but received a 'tuple'"
        )
    else:
        raise AssertionError(f"list.{name} accepted a tuple receiver")

    try:
        descriptor()
    except TypeError as exc:
        assert str(exc) == f"descriptor '{name}' of 'list' object needs an argument"
    else:
        raise AssertionError(f"list.{name} accepted a missing receiver")

for name in METHOD_DESCRIPTORS:
    descriptor = list.__dict__[name]
    try:
        descriptor(())
    except TypeError as exc:
        assert str(exc) == (
            f"descriptor '{name}' for 'list' objects doesn't apply to a 'tuple' object"
        )
    else:
        raise AssertionError(f"list.{name} accepted a tuple receiver")

    try:
        descriptor()
    except TypeError as exc:
        assert str(exc) == f"unbound method list.{name}() needs an argument"
    else:
        raise AssertionError(f"list.{name} accepted a missing receiver")


class ListSubclass(list):
    pass


value = ListSubclass([3, 1, 2])
list.append(value, 4)
list.sort(value)
assert value == [1, 2, 3, 4]
assert list.copy(value) == [1, 2, 3, 4]

values = list(range(6))
values[1:0] = ["a", "b"]
assert values == [0, "a", "b", 1, 2, 3, 4, 5]
values[6:2] = []
assert values == [0, "a", "b", 1, 2, 3, 4, 5]

for base in (object, list, tuple):
    side_effects = [1]

    class IterableSubclass(base):
        def __iter__(self):
            side_effects.append(2)

            def inner():
                yield 3
                side_effects.append(4)

            return inner()

    unpacked = [*side_effects, *IterableSubclass(), *side_effects.copy()]
    assert unpacked == [1, 3, 1, 2, 4]

print("OK")
