# CPython-suite gap: no suite test collects inside the key comparison a
# FrameLocalsProxy operation performs.
# parity-tests reason: this is a pyre/PyPy moving-GC root-liveness regression.

"""``frame.f_locals`` operations must survive a collection mid-lookup.

Every proxy operation materializes a fresh snapshot dict, or scans the code
object's names allocating a string per candidate, before it reads the key and
value it was handed.  A key whose ``__eq__`` collects relocates those arguments
while the operation still holds them.  The payloads below are lists and dicts
because those relocate; an instance of a Python class is born at a stable
address and would never exercise this.
"""

import gc
import sys


class CollectingName(str):
    """A name that forces a moving collection before answering a compare."""

    def __eq__(self, other):
        garbage = [[index] for index in range(4000)]
        assert len(garbage) == 4000
        gc.collect()
        return str.__eq__(str(self), other)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return str.__hash__(self)


def write_fast_local():
    """A proxy store resolving to a fast slot must store the live value."""
    alpha = None
    proxy = sys._getframe().f_locals
    replacement = ["replacement"]
    proxy[CollectingName("alpha")] = replacement
    assert alpha is replacement, alpha
    assert alpha == ["replacement"], alpha
    return alpha


def write_extra_key():
    """A key naming no fast local lands in the frame's extras dict."""
    beta = ["beta"]
    proxy = sys._getframe().f_locals
    stored = {"stored": True}
    proxy[CollectingName("absent")] = stored
    assert beta == ["beta"], beta
    assert proxy[CollectingName("absent")] is stored
    assert CollectingName("absent") in proxy
    assert proxy.get(CollectingName("absent")) is stored
    return stored


def read_and_default():
    """``get`` / ``setdefault`` route through the same snapshot and scan."""
    gamma = ["gamma"]
    proxy = sys._getframe().f_locals
    fallback = ["fallback"]
    assert proxy.get(CollectingName("gamma")) is gamma
    assert proxy.setdefault(CollectingName("gamma"), fallback) is gamma
    assert proxy.setdefault(CollectingName("fresh"), fallback) is fallback
    assert proxy[CollectingName("fresh")] is fallback
    assert gamma == ["gamma"], gamma
    return gamma


def merge_and_snapshot():
    """``update`` and ``items`` rebuild their pairs across allocations."""
    delta = None
    proxy = sys._getframe().f_locals
    incoming = ["incoming"]
    proxy.update({CollectingName("delta"): incoming})
    assert delta is incoming, delta
    pairs = dict(proxy.items())
    assert pairs["delta"] is incoming
    assert pairs["incoming"] is incoming
    return delta


for _ in range(5):
    assert write_fast_local() == ["replacement"]
    assert write_extra_key() == {"stored": True}
    assert read_and_default() == ["gamma"]
    assert merge_and_snapshot() == ["incoming"]

print("OK")
