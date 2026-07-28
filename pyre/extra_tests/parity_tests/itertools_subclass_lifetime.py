import gc
import itertools


# The corresponding constructors in interp_itertools.py all allocate their
# requested subtype through space.allocate_instance.
finalized = []


def make_finalizing(name, base, args, kwargs=None):
    def finalize(self):
        finalized.append(name)

    subtype = type("Finalizing_" + name, (base,), {"__del__": finalize})
    return subtype(*args, **(kwargs or {}))


objects = [
    make_finalizing("count", itertools.count, ()),
    make_finalizing("repeat", itertools.repeat, (1,)),
    make_finalizing("takewhile", itertools.takewhile, (bool, [])),
    make_finalizing("dropwhile", itertools.dropwhile, (bool, [])),
    make_finalizing("filterfalse", itertools.filterfalse, (None, [])),
    make_finalizing("compress", itertools.compress, ([], [])),
    make_finalizing("starmap", itertools.starmap, (pow, [])),
    make_finalizing("accumulate", itertools.accumulate, ([],)),
    make_finalizing("zip_longest", itertools.zip_longest, ([], [])),
]
del objects
gc.collect()

assert sorted(finalized) == [
    "accumulate",
    "compress",
    "count",
    "dropwhile",
    "filterfalse",
    "repeat",
    "starmap",
    "takewhile",
    "zip_longest",
]

print("OK")
