# CPython-suite gap: no test stores attributes, slots, weakrefs or __del__ on itertools subclasses.
# parity-tests reason: this targets the typedef.py _getusercls mapdict storage on those layouts.

"""Subclass instances of the itertools types that accept subclassing.

`typedef.py` `_getusercls` allocates them with mapdict storage: attributes,
`__dict__`, slots, weakrefs and `__del__` live on that layout. `batched` is
absent: pypy3 is 3.11. `_grouper`, `_tee` and `_tee_dataobject` reject
subclassing.
"""
import gc
import itertools
import weakref
from itertools import islice


def exc_name(fn):
    try:
        fn()
    except Exception as e:
        return type(e).__name__
    else:
        return "ok"


def plain(label, base, make, iterate):
    class S(base):
        pass

    x = make(S)
    object.__setattr__(x, "attr", 1)
    got = object.__getattribute__(x, "attr")
    reflected = object.__getattribute__(x, "__dict__")["attr"]
    object.__delattr__(x, "attr")
    missing = exc_name(lambda: object.__getattribute__(x, "attr"))
    gone = "attr" in object.__getattribute__(x, "__dict__")
    alive = weakref.ref(x)() is x
    if iterate is None:
        produced, same = "skip", "skip"
    else:
        produced = list(iterate(S))
        same = produced == list(iterate(base))
    print(
        "plain",
        label,
        type(x).__name__,
        isinstance(x, base),
        got,
        reflected,
        missing,
        gone,
        alive,
        produced,
        same,
    )


def slots(label, base, make):
    class S(base):
        __slots__ = ("a",)

    x = make(S)
    object.__setattr__(x, "a", 2)
    got = object.__getattribute__(x, "a")
    object.__delattr__(x, "a")
    missing = exc_name(lambda: object.__getattribute__(x, "a"))
    extra = exc_name(lambda: object.__setattr__(x, "z", 1))
    dic = exc_name(lambda: object.__getattribute__(x, "__dict__"))
    print(
        "slots",
        label,
        type(x).__name__,
        isinstance(x, base),
        got,
        missing,
        extra,
        dic,
    )


def finalizer(label, base, make):
    seen = []

    class D(base):
        def __del__(self):
            seen.append(label)

    def spawn():
        obj = make(D)
        return weakref.ref(obj)

    ref = spawn()
    gc.collect()
    print("del", label, ref() is None, seen)


def make_count(cls):
    return cls()


def iter_count(cls):
    return islice(cls(), 4)


def make_repeat(cls):
    return cls(7)


def iter_repeat(cls):
    return islice(cls(7), 3)


def make_takewhile(cls):
    return cls(lambda x: x < 3, [1, 2, 3, 4])


def make_dropwhile(cls):
    return cls(lambda x: x < 3, [1, 2, 3, 4])


def make_filterfalse(cls):
    return cls(None, [0, 1, 2])


def make_islice(cls):
    return cls(range(10), 2, 6, 2)


def make_product(cls):
    return cls([1, 2], [3])


def make_combinations(cls):
    return cls([1, 2, 3], 2)


def make_cwr(cls):
    return cls([1, 2], 2)


def make_permutations(cls):
    return cls([1, 2, 3], 2)


def make_groupby(cls):
    return cls([1, 1, 2])


def iter_groupby(cls):
    return [(k, list(g)) for k, g in cls([1, 1, 2])]


def make_compress(cls):
    return cls([1, 2, 3], [1, 0, 1])


def make_starmap(cls):
    return cls(lambda a, b: a + b, [(1, 2), (3, 4)])


def make_accumulate(cls):
    return cls([1, 2, 3])


def make_zip_longest(cls):
    return cls([1], [2, 3], fillvalue=0)


def make_pairwise(cls):
    return cls([1, 2, 3])


def make_cycle(cls):
    return cls([1, 2])


def iter_cycle(cls):
    return islice(cls([1, 2]), 4)


def make_chain(cls):
    return cls([1, 2], [3])


cases = [
    ("count", itertools.count, make_count, iter_count),
    ("repeat", itertools.repeat, make_repeat, iter_repeat),
    ("takewhile", itertools.takewhile, make_takewhile, make_takewhile),
    ("dropwhile", itertools.dropwhile, make_dropwhile, make_dropwhile),
    ("filterfalse", itertools.filterfalse, make_filterfalse, make_filterfalse),
    ("islice", itertools.islice, make_islice, make_islice),
    ("product", itertools.product, make_product, make_product),
    ("combinations", itertools.combinations, make_combinations, make_combinations),
    (
        "combinations_with_replacement",
        itertools.combinations_with_replacement,
        make_cwr,
        make_cwr,
    ),
    ("permutations", itertools.permutations, make_permutations, make_permutations),
    ("groupby", itertools.groupby, make_groupby, iter_groupby),
    ("compress", itertools.compress, make_compress, make_compress),
    ("starmap", itertools.starmap, make_starmap, make_starmap),
    ("accumulate", itertools.accumulate, make_accumulate, make_accumulate),
    ("zip_longest", itertools.zip_longest, make_zip_longest, make_zip_longest),
    ("pairwise", itertools.pairwise, make_pairwise, make_pairwise),
    ("cycle", itertools.cycle, make_cycle, iter_cycle),
    ("chain", itertools.chain, make_chain, make_chain),
]
for label, base, make, iterate in cases:
    plain(label, base, make, iterate)
    slots(label, base, make)
    finalizer(label, base, make)

print("OK")
