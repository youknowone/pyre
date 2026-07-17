"""PyPy slice port with CPython 3.14 richcompare/hash additions."""


def check(condition, message):
    if not condition:
        raise AssertionError(message)


surface = {
    "__doc__", "__eq__", "__ge__", "__gt__", "__hash__", "__le__", "__lt__",
    "__ne__", "__new__", "__reduce__", "__repr__", "indices", "start", "step",
    "stop",
}
check(set(slice.__dict__) == surface, "slice TypeDef surface")

a = slice(1, 2)
b = slice(1, 3)
check(a < b and a <= b and not (a > b) and not (a >= b), "slice ordering")
check(not (a < a) and a <= a and not (a > a) and a >= a, "slice identity ordering")
check(slice.__lt__(a, 1) is NotImplemented, "slice foreign comparison")
check(hash(a) == slice.__hash__(a), "slice hash descriptor")
check(hash(slice(1, 2)) == hash(slice(1, 2)), "equal slices hash equally")


class Unhashable:
    __hash__ = None


try:
    hash(slice(Unhashable()))
except TypeError:
    pass
else:
    raise AssertionError("slice component hash error")

for name, call_args in (
    ("__repr__", (42,)),
    ("__hash__", (42,)),
    ("__reduce__", (42,)),
    ("__eq__", (42, a)),
    ("__ne__", (42, a)),
    ("__lt__", (42, a)),
    ("__le__", (42, a)),
    ("__gt__", (42, a)),
    ("__ge__", (42, a)),
    ("indices", (42, 3)),
):
    try:
        getattr(slice, name)(*call_args)
    except TypeError:
        pass
    else:
        raise AssertionError(f"slice.{name} accepted a foreign receiver")

for name in ("__repr__", "__hash__", "__reduce__"):
    try:
        getattr(slice, name)()
    except TypeError:
        pass
    else:
        raise AssertionError(f"slice.{name} accepted a missing receiver")

try:
    slice.__new__(42, 1)
except TypeError:
    pass
else:
    raise AssertionError("slice.__new__ accepted a non-type class")

print("OK")
