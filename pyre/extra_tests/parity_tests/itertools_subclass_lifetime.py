# CPython-suite gap: itertools subclass tests omit user-finalizer lifetime.
# parity-tests reason: this guards PyPy-style allocation and moving-GC ownership.

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
    make_finalizing("groupby", itertools.groupby, ([],)),
    make_finalizing("islice", itertools.islice, ([], None)),
    make_finalizing("batched", itertools.batched, ([], 2)),
    make_finalizing("product", itertools.product, ([],)),
    make_finalizing("combinations", itertools.combinations, ([], 0)),
    make_finalizing(
        "combinations_with_replacement",
        itertools.combinations_with_replacement,
        ([], 0),
    ),
    make_finalizing("permutations", itertools.permutations, ([], 0)),
    make_finalizing("compress", itertools.compress, ([], [])),
    make_finalizing("starmap", itertools.starmap, (pow, [])),
    make_finalizing("accumulate", itertools.accumulate, ([],)),
    make_finalizing("zip_longest", itertools.zip_longest, ([], [])),
    make_finalizing("chain", itertools.chain, ([],)),
    make_finalizing("cycle", itertools.cycle, ([],)),
    make_finalizing("pairwise", itertools.pairwise, ([],)),
]
del objects
gc.collect()

assert sorted(finalized) == [
    "accumulate",
    "batched",
    "chain",
    "combinations",
    "combinations_with_replacement",
    "compress",
    "count",
    "cycle",
    "dropwhile",
    "filterfalse",
    "groupby",
    "islice",
    "pairwise",
    "permutations",
    "product",
    "repeat",
    "starmap",
    "takewhile",
    "zip_longest",
]

print("OK")
