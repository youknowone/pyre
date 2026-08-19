# CPython-suite gap: no match statement uses a mapping whose `get` runs Python.
# parity-tests reason: this is a pyre/PyPy moving-GC root-liveness regression.

"""``MATCH_KEYS`` must survive a mapping ``get`` that moves the subject.

The opcode probes the subject once per key pattern, and each probe is an
arbitrary Python call.  The subject, the seen-key set, the sentinel and every
value collected so far all have to outlive it.
"""

import gc


def churn():
    garbage = [[index, index + 1] for index in range(20000)]
    assert len(garbage) == 20000
    gc.collect()


class Probed(dict):
    def get(self, key, default=None):
        churn()
        return dict.get(self, key, default)


def classify(subject):
    match subject:
        case {"kind": kind, "left": left, "right": right}:
            return ("triple", kind, left, right)
        case {"kind": kind}:
            return ("single", kind)
    return ("none",)


for _ in range(10):
    full = Probed(kind=["op"], left=["l"], right=["r"])
    assert classify(full) == ("triple", ["op"], ["l"], ["r"]), classify(full)

    partial = Probed(kind=["op"])
    assert classify(partial) == ("single", ["op"]), classify(partial)

    assert classify(Probed(other=1)) == ("none",)

print("OK")
