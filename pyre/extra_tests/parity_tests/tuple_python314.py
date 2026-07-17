EXPECTED = {
    "__add__",
    "__class_getitem__",
    "__contains__",
    "__doc__",
    "__eq__",
    "__ge__",
    "__getitem__",
    "__getnewargs__",
    "__gt__",
    "__hash__",
    "__iter__",
    "__le__",
    "__len__",
    "__lt__",
    "__mul__",
    "__ne__",
    "__new__",
    "__repr__",
    "__rmul__",
    "count",
    "index",
}

assert set(tuple.__dict__) == EXPECTED

WRAPPER_DESCRIPTORS = {
    "__add__",
    "__contains__",
    "__eq__",
    "__ge__",
    "__getitem__",
    "__gt__",
    "__hash__",
    "__iter__",
    "__le__",
    "__len__",
    "__lt__",
    "__mul__",
    "__ne__",
    "__repr__",
    "__rmul__",
}

METHOD_DESCRIPTORS = {"__getnewargs__", "count", "index"}

for name in WRAPPER_DESCRIPTORS:
    descriptor = tuple.__dict__[name]
    try:
        descriptor([])
    except TypeError as exc:
        assert str(exc) == (
            f"descriptor '{name}' requires a 'tuple' object but received a 'list'"
        )
    else:
        raise AssertionError(f"tuple.{name} accepted a list receiver")

    try:
        descriptor()
    except TypeError as exc:
        assert str(exc) == f"descriptor '{name}' of 'tuple' object needs an argument"
    else:
        raise AssertionError(f"tuple.{name} accepted a missing receiver")

for name in METHOD_DESCRIPTORS:
    descriptor = tuple.__dict__[name]
    try:
        descriptor([])
    except TypeError as exc:
        assert str(exc) == (
            f"descriptor '{name}' for 'tuple' objects doesn't apply to a 'list' object"
        )
    else:
        raise AssertionError(f"tuple.{name} accepted a list receiver")

    try:
        descriptor()
    except TypeError as exc:
        assert str(exc) == f"unbound method tuple.{name}() needs an argument"
    else:
        raise AssertionError(f"tuple.{name} accepted a missing receiver")


class EqualToNeedle:
    def __init__(self, label):
        self.label = label

    def __eq__(self, other):
        return other == "needle"


values = (EqualToNeedle("first"), 0, EqualToNeedle("second"))
assert values.count("needle") == 2
assert values.index("needle") == 0
assert values.index("needle", 1) == 2
assert values.index("needle", -1) == 2


class Index:
    def __init__(self, value):
        self.value = value

    def __index__(self):
        return self.value


assert values.index("needle", Index(1), Index(3)) == 2
try:
    values.index("needle", 1, 2)
except ValueError as exc:
    assert str(exc) == "tuple.index(x): x not in tuple"
else:
    raise AssertionError("tuple.index ignored stop")


class TupleSubclass(tuple):
    pass


subclass = TupleSubclass((1, 2, 1))
assert tuple.count(subclass, 1) == 2
assert tuple.index(subclass, 2) == 1
assert tuple.__getnewargs__(subclass) == ((1, 2, 1),)

print("OK")
