import itertools


CASES = (
    (
        itertools.takewhile,
        "Return successive entries from an iterable as long as the predicate evaluates to true for each entry.",
        [1, 2],
    ),
    (
        itertools.dropwhile,
        "Drop items from the iterable while predicate(item) is true.\n\n"
        "Afterwards, return every element until the iterable is exhausted.",
        [0, 3],
    ),
    (
        itertools.filterfalse,
        "Return those items of iterable for which function(item) is false.\n\n"
        "If function is None, return the items that are false.",
        [0],
    ),
)


for iterator_type, doc, expected in CASES:
    assert isinstance(iterator_type, type)
    assert iterator_type.__module__ == "itertools"
    assert iterator_type.__doc__ == doc
    assert {"__new__", "__iter__", "__next__", "__doc__"} <= set(
        iterator_type.__dict__
    )
    assert "__reduce__" not in iterator_type.__dict__
    assert "__setstate__" not in iterator_type.__dict__

    obj = iterator_type(bool, [1, 2, 0, 3])
    assert type(obj) is iterator_type
    assert iter(obj) is obj
    assert list(obj) == expected

    for args in ((), (bool,), (bool, [], None)):
        try:
            iterator_type(*args)
        except TypeError:
            pass
        else:
            raise AssertionError((iterator_type, args))

    try:
        iterator_type(predicate=bool, iterable=[])
    except TypeError as exc:
        assert str(exc) == f"{iterator_type.__name__}() takes no keyword arguments"
    else:
        raise AssertionError("keyword arguments accepted")

    try:
        iterator_type(bool, []).__reduce__()
    except TypeError:
        pass
    else:
        raise AssertionError("Python 3.14 non-picklable iterator became picklable")

    class Subclass(iterator_type):
        def __init__(self, predicate, iterable, *, marker=None):
            self.marker = marker

    sub = Subclass(bool, [1, 0], marker="kept")
    assert type(sub) is Subclass
    assert sub.marker == "kept"
    assert list(sub) == ([1] if iterator_type is itertools.takewhile else [0])


assert list(itertools.filterfalse(None, [0, 1, "", "x", [], [1]])) == [0, "", []]

# Construction is lazy: only iter(iterable) runs before the first __next__.
events = []


def predicate(value):
    events.append(value)
    return value < 2


obj = itertools.takewhile(predicate, iter([1, 2, 3]))
assert events == []
assert next(obj) == 1
assert events == [1]
try:
    next(obj)
except StopIteration:
    pass
else:
    raise AssertionError("takewhile did not stop")
assert events == [1, 2]

print("itertools predicate type Python 3.14 parity ok")
