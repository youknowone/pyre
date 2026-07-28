import gc
import itertools


assert isinstance(itertools.islice, type)
assert "__reduce__" not in itertools.islice.__dict__
assert "__setstate__" not in itertools.islice.__dict__


class Source:
    def __init__(self):
        self.value = 0
        self.pulls = 0

    def __iter__(self):
        return self

    def __next__(self):
        value = self.value
        if value == 20:
            raise StopIteration
        self.value += 1
        self.pulls += 1
        return value


# W_ISlice keeps the source iterator and advances only far enough to produce
# the next selected item.
source = Source()
selected = itertools.islice(source, 2, 10, 3)
assert source.pulls == 0
assert iter(selected) is selected
assert next(selected) == 2
assert source.pulls == 3
assert next(selected) == 5
assert source.pulls == 6
assert list(selected) == [8]
assert source.pulls == 10

assert list(itertools.islice(range(5), None)) == [0, 1, 2, 3, 4]
assert list(itertools.islice(range(5), None, None, None)) == [0, 1, 2, 3, 4]
assert list(itertools.islice(range(10), 3, 1)) == []
assert list(itertools.islice(range(10), None, 8, 3)) == [0, 3, 6]

# The source is owned solely through W_ISlice and must survive a moving
# collection via its generated pointer-offset trace.
owned_source = itertools.islice(iter(range(8)), 1, 7, 2)
gc.collect()
assert list(owned_source) == [1, 3, 5]


class Index:
    def __init__(self, value):
        self.value = value

    def __index__(self):
        return self.value


assert list(itertools.islice(range(10), Index(2), Index(8), Index(3))) == [2, 5]

for args in [
    (range(3), -1),
    (range(3), -1, 2),
    (range(3), 0, 2, 0),
    (range(3), 1.5),
    (range(3), 0, 2**100),
]:
    try:
        itertools.islice(*args)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid islice index accepted: %r" % (args,))

try:
    itertools.islice(iterable=range(3), stop=2)
except TypeError:
    pass
else:
    raise AssertionError("islice accepted keyword-only construction")


class KeywordISlice(itertools.islice):
    def __init__(self, iterable, stop, *, marker):
        self.marker = marker


keyword = KeywordISlice(range(3), 2, marker=42)
assert keyword.marker == 42
assert list(keyword) == [0, 1]


class IterOverride(itertools.islice):
    def __iter__(self):
        return iter((99,))


class NextOverride(itertools.islice):
    def __next__(self):
        return 88


assert list(IterOverride(range(3), 2)) == [99]
assert next(NextOverride(range(3), 2)) == 88

finalized = []


def make_subtype():
    def finalize(self):
        finalized.append("islice")

    return type("FinalizingISlice", (itertools.islice,), {"__del__": finalize})


subtype = make_subtype()
obj = subtype(range(3), 2)
assert type(obj) is subtype
assert next(obj) == 0
del obj, subtype
gc.collect()
assert finalized == ["islice"]

print("OK")
