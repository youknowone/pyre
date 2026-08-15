"""``f(**mapping)`` when the mapping runs a collection while it is read.

``DICT_MERGE`` calls the mapping's ``keys()``, then ``__getitem__`` per key, and
tests each key against the accumulated target.  Every one of those runs Python,
so a mapping that collects between them exercises whether the merge still reads
the objects it started with.
"""

import gc

from testutils import assert_raises


def f(*args, **kwargs):
    return args, sorted(kwargs.items())


class CollectingMapping:
    def keys(self):
        gc.collect()
        return ["w", "x"]

    def __getitem__(self, key):
        gc.collect()
        return key.upper()


class CollectingKeys(dict):
    def keys(self):
        gc.collect()
        return super().keys()


class CollectingGetItem(dict):
    def keys(self):
        return super().keys()

    def __getitem__(self, key):
        gc.collect()
        return super().__getitem__(key)


assert f(**CollectingMapping()) == ((), [("w", "W"), ("x", "X")])
assert f(**CollectingKeys(w=1, x=2)) == ((), [("w", 1), ("x", 2)])
assert f(**CollectingGetItem(w=1, x=2)) == ((), [("w", 1), ("x", 2)])

# A `*` unpack ahead of the merge is a collection point of its own, and the
# callable and the already-merged names have to survive it.
assert f(*[1, 2], **CollectingMapping()) == ((1, 2), [("w", "W"), ("x", "X")])
assert f(*[1, 2], **CollectingKeys(w=1)) == ((1, 2), [("w", 1)])

# Two mappings merged into one call: the second sees a non-empty target, which
# is the arm that tests each key for membership.
assert f(**CollectingMapping(), **CollectingKeys(y=3)) == (
    (),
    [("w", "W"), ("x", "X"), ("y", 3)],
)

# The duplicate report names the key and the callable, and reaches the same
# membership test.
assert_raises(
    TypeError,
    lambda: f(**CollectingMapping(), **CollectingKeys(w=1)),
    _msg="f() got multiple values for keyword argument 'w'",
)

# A non-mapping after ** is still rejected by its own message.
assert_raises(
    TypeError,
    lambda: f(**[1, 2]),
    _msg="f() argument after ** must be a mapping, not list",
)
