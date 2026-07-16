"""PyPy listobject/iterobject parity with Python 3.14 type identities."""


def check(condition, message):
    if not condition:
        raise AssertionError(message)


expected_list_surface = {
    "__add__", "__class_getitem__", "__contains__", "__delitem__", "__doc__",
    "__eq__", "__ge__", "__getitem__", "__gt__", "__hash__", "__iadd__",
    "__imul__", "__init__", "__iter__", "__le__", "__len__", "__lt__",
    "__mul__", "__ne__", "__new__", "__repr__", "__reversed__", "__rmul__",
    "__setitem__", "__sizeof__", "append", "clear", "copy", "count", "extend",
    "index", "insert", "pop", "remove", "reverse", "sort",
}
expected_iterator_surface = {
    "__doc__", "__iter__", "__length_hint__", "__next__", "__reduce__",
    "__setstate__",
}

check(set(list.__dict__) == expected_list_surface, "list TypeDef surface")
check(list.__hash__ is None, "list must be unhashable")
check([].__sizeof__() == 40, "empty list sizeof")


class L(list):
    def __repr__(self):
        return "L(" + super().__repr__() + ")"


check(repr(L([1, 2])) == "L([1, 2])", "base list repr must not redispatch")

values = [1]
it = iter(values)
check(type(it).__name__ == "list_iterator", "forward iterator identity")
check(set(type(it).__dict__) == expected_iterator_surface, "forward iterator surface")
check(iter(it) is it, "forward iterator self")
check(it.__length_hint__() == 1, "initial forward hint")
check(next(it) == 1, "first forward item")
values.append(2)
check(it.__length_hint__() == 1, "forward iterator observes growth")
check(next(it) == 2, "forward iterator yields appended item")
try:
    next(it)
except StopIteration:
    pass
else:
    raise AssertionError("forward iterator exhaustion")
check(it.__length_hint__() == 0, "exhausted forward hint")

values = [10, 20, 30]
it = iter(values)
check(next(it) == 10, "reduce cursor setup")
reduced = it.__reduce__()
check(reduced[0] is iter and reduced[1][0] is values and reduced[2] == 1,
      "forward reduce state")
it.__setstate__(-5)
check(next(it, "STOP") == "STOP", "forward negative setstate exhausts in 3.14")

values = [1, 2, 3]
rit = reversed(values)
check(type(rit).__name__ == "list_reverseiterator", "reverse iterator identity")
check(set(type(rit).__dict__) == expected_iterator_surface, "reverse iterator surface")
check(iter(rit) is rit, "reverse iterator self")
check(rit.__length_hint__() == 3, "initial reverse hint")
check(next(rit) == 3, "first reverse item")
values.append(4)
check(next(rit) == 2, "reverse iterator ignores appended tail")
reduced = rit.__reduce__()
check(reduced[0] is reversed and reduced[1][0] is values and reduced[2] == 0,
      "reverse reduce state")
rit.__setstate__(999)
check(next(rit) == 4, "reverse setstate upper clamp")

print("OK")
