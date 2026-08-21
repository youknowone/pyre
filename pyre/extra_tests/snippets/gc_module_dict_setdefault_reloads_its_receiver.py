# pyre-check: gate=1
"""A module dict's `setdefault` addresses the live dictionary after its lookup.

A key that is not a `str` sends the lookup through the object strategy, which
hashes it, and a `__hash__` written in Python is a collection point that moves
the module dict.  The store below the lookup was handed the receiver, the key
and the default as they stood before it.

The dictionary has to be EMPTY when the collecting hash runs: a module dict
that already survived one collection is promoted, and an immobile receiver
hides the defect.
"""

import gc
import types

KEEP = None


def churn():
    global KEEP
    KEEP = [[i] * 24 for i in range(60)] + [bytearray(b"Q" * 96) for _ in range(30)]
    gc.collect()
    KEEP = [[i] * 24 for i in range(60)] + [bytearray(b"Q" * 96) for _ in range(30)]


class CollectingKey:
    """Not a `str`, so the lookup switches strategy and hashes it."""

    def __hash__(self):
        churn()
        return 7

    def __eq__(self, other):
        return self is other


held = []
for i in range(30):
    module = types.ModuleType("probe%d" % i)
    key = CollectingKey()
    stored = vars(module).setdefault(key, [])
    stored.append(i)
    held.append((module, key, stored))

for module, key, stored in held:
    assert vars(module)[key] is stored, "setdefault stored into a stale receiver"
    print(stored)
