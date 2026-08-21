# pyre-check: gate=1
"""`dict.setdefault` hands back the default at its live address.

The store hashes the key, and a `__hash__` written in Python is a collection
point that moves a nursery-allocated default.  The empty-dict arms of the
checked setdefault returned the by-value word taken before the store, so the
caller was handed a pre-move header.

The caller has to KEEP what comes back: a run that discards the return value
never dereferences the stale word and stays clean either way.
"""

import gc

KEEP = None


def churn():
    global KEEP
    KEEP = [[i] * 24 for i in range(60)] + [bytearray(b"Q" * 96) for _ in range(30)]
    gc.collect()
    KEEP = [[i] * 24 for i in range(60)] + [bytearray(b"Q" * 96) for _ in range(30)]


class CollectingKey:
    """Collects while being hashed, which is what moves the default."""

    def __hash__(self):
        churn()
        return 7

    def __eq__(self, other):
        return self is other


held = []
for _ in range(30):
    victim = {}
    key = CollectingKey()
    stored = dict.setdefault(victim, key, [])
    stored.append(1)
    held.append((victim, key, stored))

for victim, key, stored in held:
    assert victim[key] is stored, "setdefault returned a value the dict does not hold"
    print(len(victim), stored)
