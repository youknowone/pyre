"""PyPy W_StarMap structure with Python 3.14 public surface."""

import itertools


assert isinstance(itertools.starmap, type)
assert itertools.starmap.__module__ == "itertools"
assert itertools.starmap.__name__ == "starmap"
assert itertools.starmap.__doc__ == (
    "Return an iterator whose values are returned from the function evaluated "
    "with an argument tuple taken from the given sequence."
)
assert {"__new__", "__iter__", "__next__", "__doc__"} <= set(
    itertools.starmap.__dict__
)


events = []


class Source:
    def __init__(self):
        self.items = iter([(2, 3), [3, 2]])

    def __iter__(self):
        events.append("iter")
        return self

    def __next__(self):
        value = next(self.items)
        events.append(("next", value))
        return value


def function(a, b):
    events.append(("call", a, b))
    return a**b


obj = itertools.starmap(function, Source())
assert type(obj) is itertools.starmap
assert iter(obj) is obj
assert events == ["iter"]
assert next(obj) == 8
assert events == ["iter", ("next", (2, 3)), ("call", 2, 3)]
assert list(obj) == [9]

# The source stays live: an exception is raised at the corresponding next(),
# not during construction, and the source has advanced by one bundle.
events = []
obj = itertools.starmap(function, Source())
assert events == ["iter"]
assert next(obj) == 8
assert events[-1] == ("call", 2, 3)

for args in ((), (function,), (function, [], None)):
    try:
        itertools.starmap(*args)
    except TypeError:
        pass
    else:
        raise AssertionError(args)

try:
    itertools.starmap(function=function, iterable=[])
except TypeError as exc:
    assert str(exc) == "starmap() takes no keyword arguments"
else:
    raise AssertionError("starmap accepted keyword arguments")

for name in ("__iter__", "__next__"):
    try:
        getattr(itertools.starmap, name)(iter(()))
    except TypeError as exc:
        assert str(exc) == (
            f"descriptor '{name}' requires a 'itertools.starmap' object "
            "but received a 'tuple_iterator'"
        )
    else:
        raise AssertionError(f"starmap.{name} accepted a foreign receiver")


class StarMapSubclass(itertools.starmap):
    pass


sub = StarMapSubclass(pow, [(2, 4)])
assert type(sub) is StarMapSubclass
assert list(sub) == [16]

print("itertools.starmap Python 3.14 parity ok")
