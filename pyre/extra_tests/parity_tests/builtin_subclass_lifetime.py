# CPython-suite gap: builtin-subclass tests omit user-finalizer lifetime.
# parity-tests reason: this guards PyPy-style subtype allocation and moving-GC ownership.

"""Every builtin subtype allocated through ``space.allocate_instance`` finalizes."""

import gc
import itertools
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
print("OK")
