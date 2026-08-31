# CPython-suite gap: builtin-subclass tests omit user-finalizer lifetime.
# parity-tests reason: this guards PyPy-style subtype allocation and moving-GC ownership.

"""Every builtin subtype allocated through ``space.allocate_instance`` finalizes."""

import gc
import itertools
import weakref
from types import ModuleType


finalized = []
expected = []


def make(label, base, args, snapshot=lambda obj: None):
    def finalize(self):
        finalized.append((label, snapshot(self)))

    subtype = type("Finalizing_" + label, (base,), {"__del__": finalize})
    obj = subtype(*args)
    expected.append((label, snapshot(obj)))
    if base in (bytes, bytearray, int, float, complex):
        assert type(base(obj)) is base, (label, type(base(obj)))
    return obj


objects = [
    make("bytes", bytes, (b"abc",), bytes),
    make("bytearray", bytearray, (b"xyz",), bytes),
    make("int-small", int, (42,), int),
    make("int-big", int, (1 << 100,), int),
    make("float", float, (1.25,), float),
    make("complex", complex, (1.25, -2.5), complex),
    make("staticmethod", staticmethod, (lambda: None,)),
    make("classmethod", classmethod, (lambda: None,)),
    make("property", property, (lambda self: None,)),
    make("module", ModuleType, ("finalizing_module",)),
    make("enumerate", enumerate, ([],)),
    make("map", map, (int, [])),
    make("filter", filter, (None, [])),
    make("zip", zip, ([],)),
]


ITERTOOLS_CASES = [
    ("count", itertools.count, ()),
    ("repeat", itertools.repeat, (1,)),
    ("takewhile", itertools.takewhile, (bool, [])),
    ("dropwhile", itertools.dropwhile, (bool, [])),
    ("filterfalse", itertools.filterfalse, (None, [])),
    ("groupby", itertools.groupby, ([],)),
    ("islice", itertools.islice, ([], None)),
    ("product", itertools.product, ([],)),
    ("combinations", itertools.combinations, ([], 0)),
    ("combinations_with_replacement", itertools.combinations_with_replacement, ([], 0)),
    ("permutations", itertools.permutations, ([], 0)),
    ("compress", itertools.compress, ([], [])),
    ("starmap", itertools.starmap, (pow, [])),
    ("accumulate", itertools.accumulate, ([],)),
    ("zip_longest", itertools.zip_longest, ([], [])),
    ("chain", itertools.chain, ([],)),
    ("cycle", itertools.cycle, ([],)),
    ("pairwise", itertools.pairwise, ([],)),
]
if hasattr(itertools, "batched"):
    ITERTOOLS_CASES.append(("batched", itertools.batched, ([], 2)))

objects.extend(make(label, base, args) for label, base, args in ITERTOOLS_CASES)
del objects
gc.collect()

assert sorted(finalized, key=repr) == sorted(expected, key=repr), (finalized, expected)


class DictList(list):
    pass


class DictStr(str):
    pass


def churn_dead(count):
    for index in range(count):
        value = DictList([1, 2, 3])
        value.foo = index
        weakref.ref(value)


def inherited_dicts(count):
    inherited = 0
    for index in range(count):
        for value, key in ((DictList(), "foo"), (DictStr("x"), "tag")):
            inherited += bool(value.__dict__)
            setattr(value, key, index)
            inherited += value.__dict__ != {key: index}
    return inherited


# A new owner must not inherit an address-keyed dict or weakref lifeline from
# a collected builtin-subclass instance.
inherited = 0
for _ in range(40):
    churn_dead(300)
    inherited += inherited_dicts(300)
assert inherited == 0, inherited

alive = []
for _ in range(20):
    churn_dead(300)
    for _ in range(300):
        value = DictList()
        reference = weakref.ref(value)
        assert reference() is value
        alive.append((value, reference))
    alive.clear()

print("OK")
