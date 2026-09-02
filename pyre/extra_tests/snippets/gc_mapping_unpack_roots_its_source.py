# pyre-check: gate=1
"""`{**mapping}` keeps the destination dict current across the `keys()` call.

`PyDict_Update`'s generic branch fetches `mapping.keys()` and iterates the
result before it touches the destination, and all three of those steps run
Python: the `keys` attribute lookup, the call, and the iteration.  The
destination is a `W_DictObject`, which a minor collection relocates, so the
word the caller handed in is the pre-move one by the time the store loop starts
-- and the loop publishes it as a root, so the whole loop stores into the
corpse.

The binary operators reach the same shape from the other side: `a + b` and
`a * b` dispatch a reflected special method with both operands live, and the
set and dict displays drain a user iterable before storing.
"""

import gc

CHURN = None


def collect():
    global CHURN
    gc.collect()
    # Churn so a relocated block is handed straight back out.
    CHURN = [tuple(range(4)) for _ in range(400)]


class Mapping:
    """Collects from both halves of the `PyMapping_Keys` protocol."""

    def keys(self):
        collect()
        return ["a", "b", "c"]

    def __getitem__(self, key):
        collect()
        return key * 2


assert {**Mapping(), "z": 1} == {"a": "aa", "b": "bb", "c": "cc", "z": 1}

merged = {"pre": 0}
merged.update(Mapping())
assert merged == {"pre": 0, "a": "aa", "b": "bb", "c": "cc"}, merged


class RAdd:
    def __radd__(self, other):
        collect()
        return list(other) + ["r"]


class RMul:
    def __rmul__(self, other):
        collect()
        return list(other) * 2


assert ["p", "q"] + RAdd() == ["p", "q", "r"]
assert ["m", "n"] * RMul() == ["m", "n", "m", "n"]


class Iterable:
    def __iter__(self):
        collect()
        return iter(["s1", "s2"])


assert {*Iterable(), "s3"} == {"s1", "s2", "s3"}
assert [*Iterable(), "s3"] == ["s1", "s2", "s3"]

print("ok")
