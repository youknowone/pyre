import gc
import itertools


assert isinstance(itertools.product, type)
assert "__reduce__" not in itertools.product.__dict__
assert "__setstate__" not in itertools.product.__dict__
assert "__sizeof__" in itertools.product.__dict__

assert list(itertools.product()) == [()]
assert list(itertools.product("AB", "xy")) == [
    ("A", "x"),
    ("A", "y"),
    ("B", "x"),
    ("B", "y"),
]
assert list(itertools.product("AB", repeat=2)) == [
    ("A", "A"),
    ("A", "B"),
    ("B", "A"),
    ("B", "B"),
]
assert list(itertools.product([], "AB")) == []


class Source:
    def __init__(self, values):
        self.values = values
        self.index = 0
        self.iter_calls = 0

    def __iter__(self):
        self.iter_calls += 1
        return self

    def __next__(self):
        if self.index == len(self.values):
            raise StopIteration
        value = self.values[self.index]
        self.index += 1
        return value


# Product snapshots each non-zero-repeat source once, but does not build the
# Cartesian result eagerly.
left = Source(range(100))
right = Source(range(100))
product = itertools.product(left, right)
assert left.index == 100 and right.index == 100
assert next(product) == (0, 0)
assert next(product) == (0, 1)
assert left.iter_calls == 1 and right.iter_calls == 1

# Python 3.14 sets nargs=0 before touching inputs when repeat is zero.
untouched = Source([1, 2, 3])
assert list(itertools.product(untouched, repeat=0)) == [()]
assert untouched.iter_calls == 0
assert untouched.index == 0


class Index:
    def __index__(self):
        return 2


assert list(itertools.product("x", repeat=Index())) == [("x", "x")]
for repeat in (-1,):
    try:
        itertools.product("x", repeat=repeat)
    except ValueError as exc:
        assert str(exc) == "repeat argument cannot be negative"
    else:
        raise AssertionError("negative product repeat accepted")
try:
    itertools.product("x", repeat=1.5)
except TypeError:
    pass
else:
    raise AssertionError("non-index product repeat accepted")
try:
    itertools.product("x", unknown=True)
except TypeError:
    pass
else:
    raise AssertionError("unknown product keyword accepted")

# Pool objects and their contents are reachable only through W_Product.
owned = itertools.product([object(), object()], [object()])
gc.collect()
first = next(owned)
assert len(first) == 2
assert next(owned)[0] is not first[0]
try:
    next(owned)
except StopIteration:
    pass
else:
    raise AssertionError("product did not stop after its final tuple")

assert itertools.product().__sizeof__() < itertools.product([1], [2]).__sizeof__()


class IterOverride(itertools.product):
    def __iter__(self):
        return iter(((99,),))


class NextOverride(itertools.product):
    def __next__(self):
        return (88,)


assert list(IterOverride("ab")) == [(99,)]
assert next(NextOverride("ab")) == (88,)

finalized = []


def make_subtype():
    def finalize(self):
        finalized.append("product")

    return type("FinalizingProduct", (itertools.product,), {"__del__": finalize})


subtype = make_subtype()
obj = subtype("ab")
assert type(obj) is subtype
assert next(obj) == ("a",)
del obj, subtype
gc.collect()
assert finalized == ["product"]

print("OK")
