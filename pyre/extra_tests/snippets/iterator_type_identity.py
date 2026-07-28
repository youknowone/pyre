# Python 3.14 gives str / bytes / bytearray / memoryview iteration its own
# concrete type per producer where PyPy serves all of them from one abstract
# `sequenceiterator`. Pyre keeps PyPy's single payload and gives each producer
# the 3.14 identity, so this pins both the reported type name and the surface
# that type exposes. (These assertions deliberately disagree with PyPy, so they
# cannot live in the synthetic suite, which requires cpython == pypy output.)
import pickle


class Seq:
    def __getitem__(self, i):
        if i > 2:
            raise IndexError
        return i


NAMES = [
    (iter(Seq()), "iterator"),
    (iter("abc"), "str_ascii_iterator"),
    (iter("aéc"), "str_iterator"),
    (iter(b"abc"), "bytes_iterator"),
    (iter(bytearray(b"abc")), "bytearray_iterator"),
    (iter(memoryview(b"abc")), "memory_iterator"),
    (iter([1]), "list_iterator"),
    (iter((1,)), "tuple_iterator"),
    (iter(range(3)), "range_iterator"),
    (reversed([1]), "list_reverseiterator"),
]
for it, name in NAMES:
    assert type(it).__name__ == name, (type(it).__name__, name)

# The split is per storage kind, not per object.
assert type(iter("a")) is type(iter("bb"))
assert type(iter("a")) is not type(iter("é"))
assert type(iter(b"a")) is not type(iter(bytearray(b"a")))

# None of them is instantiable or subclassable.
for it, name in NAMES:
    try:
        type(it)()
        raise AssertionError("expected TypeError for " + name)
    except TypeError:
        pass

# `memory_iterator` carries the iteration protocol only: no `__length_hint__`,
# no `__setstate__`, and pickling one is refused.
mem = iter(memoryview(b"abc"))
assert sorted(set(dir(type(mem))) - set(dir(object))) == ["__iter__", "__next__"]
assert not hasattr(mem, "__length_hint__")
assert not hasattr(mem, "__setstate__")
try:
    mem.__reduce__()
    raise AssertionError("expected TypeError pickling memory_iterator")
except TypeError:
    pass
assert list(mem) == [97, 98, 99]

# Every other flavour keeps the pickle protocol its 3.14 type declares.
for make in (
    lambda: iter("abc"),
    lambda: iter("aéc"),
    lambda: iter(b"abc"),
    lambda: iter(bytearray(b"abc")),
    lambda: iter(Seq()),
):
    it = make()
    next(it)
    revived = pickle.loads(pickle.dumps(it))
    assert type(revived) is type(it), make
    assert list(revived) == list(make())[1:], make

# `__length_hint__` reports the remaining count; over a bare `__getitem__`
# sequence with no `__len__` there is nothing to report.
for make, expected in (
    (lambda: iter("abc"), 2),
    (lambda: iter("aéc"), 2),
    (lambda: iter(b"abc"), 2),
    (lambda: iter(bytearray(b"abc")), 2),
    (lambda: iter(Seq()), NotImplemented),
):
    it = make()
    next(it)
    assert it.__length_hint__() == expected, (make, it.__length_hint__())

# Iterating a str through a hot loop must not collapse the identity back to
# the shared type when the loop is traced.
seen = set()
total = 0
for _ in range(2000):
    it = iter("abc")
    seen.add(type(it).__name__)
    for ch in it:
        total += ord(ch)
assert seen == {"str_ascii_iterator"}, seen
assert total == 2000 * (97 + 98 + 99), total

print("OK")
