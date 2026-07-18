"""PyPy W_Compress structure with Python 3.14 public surface."""

import itertools


assert isinstance(itertools.compress, type)
assert itertools.compress.__module__ == "itertools"
assert itertools.compress.__name__ == "compress"
assert itertools.compress.__doc__ == (
    "Return data elements corresponding to true selector elements.\n\n"
    "Forms a shorter iterator from selected data elements using the selectors to\n"
    "choose the data elements."
)
assert {"__new__", "__iter__", "__next__", "__doc__"} <= set(
    itertools.compress.__dict__
)
assert "__reduce__" not in itertools.compress.__dict__


events = []


class Source:
    def __init__(self, name, values):
        self.name = name
        self.values = iter(values)

    def __iter__(self):
        events.append(("iter", self.name))
        return self

    def __next__(self):
        value = next(self.values)
        events.append(("next", self.name, value))
        return value


obj = itertools.compress(Source("data", ["a", "b", "c"]), Source("selectors", [0, 1, 1]))
assert type(obj) is itertools.compress
assert iter(obj) is obj
assert events == [("iter", "data"), ("iter", "selectors")]
assert next(obj) == "b"
assert events == [
    ("iter", "data"),
    ("iter", "selectors"),
    ("next", "data", "a"),
    ("next", "selectors", 0),
    ("next", "data", "b"),
    ("next", "selectors", 1),
]
assert list(obj) == ["c"]

# The shortest input stops first, and data is pulled before the corresponding
# selector exactly as W_Compress.next_w specifies.
events = []
obj = itertools.compress(Source("data", [1, 2]), Source("selectors", []))
try:
    next(obj)
except StopIteration:
    pass
else:
    raise AssertionError("compress did not stop with its selectors")
assert events == [
    ("iter", "data"),
    ("iter", "selectors"),
    ("next", "data", 1),
]

# Python 3.14 accepts the two Argument Clinic keyword names.
assert list(itertools.compress(data=[1, 2], selectors=[True, False])) == [1]
assert list(itertools.compress([1, 2], selectors=[False, True])) == [2]

for args in ((), ([],), ([], [], None)):
    try:
        itertools.compress(*args)
    except TypeError:
        pass
    else:
        raise AssertionError(args)

for name in ("__iter__", "__next__"):
    try:
        getattr(itertools.compress, name)(iter(()))
    except TypeError as exc:
        assert str(exc) == (
            f"descriptor '{name}' requires a 'itertools.compress' object "
            "but received a 'tuple_iterator'"
        )
    else:
        raise AssertionError(f"compress.{name} accepted a foreign receiver")


class CompressSubclass(itertools.compress):
    pass


sub = CompressSubclass([1, 2], [0, 1])
assert type(sub) is CompressSubclass
assert list(sub) == [2]

try:
    itertools.compress([], []).__reduce__()
except TypeError:
    pass
else:
    raise AssertionError("Python 3.14 compress unexpectedly became picklable")

print("itertools.compress Python 3.14 parity ok")
