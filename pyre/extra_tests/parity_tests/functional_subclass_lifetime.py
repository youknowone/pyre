# CPython-suite gap: functional builtin subclasses omit user-finalizer lifetime.
# parity-tests reason: this guards PyPy-style allocation and moving-GC ownership.

import gc


# functional.py allocates each strict subtype through space.allocate_instance,
# so enumerate, map, filter, and zip instances all enter the user finalizer
# queue.
finalized = []


class FinalizingEnumerate(enumerate):
    def __del__(self):
        finalized.append("enumerate")


class FinalizingMap(map):
    def __del__(self):
        finalized.append("map")


class FinalizingFilter(filter):
    def __del__(self):
        finalized.append("filter")


class FinalizingZip(zip):
    def __del__(self):
        finalized.append("zip")


objects = [
    FinalizingEnumerate([]),
    FinalizingMap(int, []),
    FinalizingFilter(None, []),
    FinalizingZip([]),
]
del objects
gc.collect()

assert sorted(finalized) == ["enumerate", "filter", "map", "zip"]

print("OK")
