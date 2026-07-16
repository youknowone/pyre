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

print("OK")
