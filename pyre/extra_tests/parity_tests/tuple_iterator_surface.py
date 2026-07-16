"""PyPy tupleobject/iterobject parity with Python 3.14 type identity."""


def check(condition, message):
    if not condition:
        raise AssertionError(message)


tuple_surface = {
    "__add__", "__class_getitem__", "__contains__", "__doc__", "__eq__",
    "__ge__", "__getitem__", "__getnewargs__", "__gt__", "__hash__",
    "__iter__", "__le__", "__len__", "__lt__", "__mul__", "__ne__",
    "__new__", "__repr__", "__rmul__", "count", "index",
}
iterator_surface = {
    "__doc__", "__iter__", "__length_hint__", "__next__", "__reduce__",
    "__setstate__",
}

check(set(tuple.__dict__) == tuple_surface, "tuple TypeDef surface")
check(hash((1, 2)) == tuple.__hash__((1, 2)), "tuple hash descriptor")


class T(tuple):
    def __repr__(self):
        return "T(" + super().__repr__() + ")"


check(repr(T((1, 2))) == "T((1, 2))", "tuple subclass/base repr dispatch")

values = (10, 20, 30)
it = iter(values)
check(type(it).__name__ == "tuple_iterator", "tuple iterator identity")
check(set(type(it).__dict__) == iterator_surface, "tuple iterator surface")
check(iter(it) is it and it.__length_hint__() == 3, "tuple iterator initial state")
check(next(it) == 10, "tuple iterator first item")
reduced = it.__reduce__()
check(reduced[0] is iter and reduced[1][0] is values and reduced[2] == 1,
      "tuple iterator reduce state")
it.__setstate__(-9)
check(next(it) == 10, "tuple iterator negative state clamps to zero")
it.__setstate__(999)
check(next(it, "STOP") == "STOP", "tuple iterator high state exhausts")

print("OK")
