# pyre-check: gate=1
"""`%` formatting keeps its operands alive and current across a conversion.

Two separate holes, both reached through the same operand column:

* `BINARY_OP %` pops the format string and the operand off the value stack
  before dispatching, so on `b"%a" % (Collects(),)` nothing refers to the
  operand tuple while `__repr__` runs.  A tuple never moves, but an
  unreachable one is still collected.
* Every conversion runs Python, and the operands are drained into a plain
  native column first.  A list or dict still waiting its turn moves under a
  collection triggered by an earlier conversion, so it would be formatted at
  a pre-move address.  This one bites on the method form too, where the
  argument slice is already pinned.
"""

import gc

KEEP = None


def churn():
    global KEEP
    KEEP = [[i] * 24 for i in range(60)] + [bytearray(b"Q" * 96) for _ in range(30)]
    gc.collect()
    KEEP = [[i] * 24 for i in range(60)] + [bytearray(b"Q" * 96) for _ in range(30)]


class Collects:
    def __str__(self):
        churn()
        return "s"

    def __repr__(self):
        churn()
        return "r"

    def __format__(self, spec):
        churn()
        return "f"


# Liveness: the operand tuple is a temporary the operator has already popped.
for _ in range(30):
    print(b"%a-%a" % (Collects(), Collects()))
    print("%s-%s" % (Collects(), Collects()))
    print("{}-{}".format(Collects(), Collects()))

# Staleness: a movable operand queued behind a collecting one, through the
# operator and through the method form whose argument slice is pinned.
for _ in range(30):
    print(b"%a|%a|%a" % (Collects(), [1, 2, 3], {4: 5}))
    print("%s|%s|%s" % (Collects(), [1, 2, 3], {4: 5}))
    print(bytes.__mod__(b"%a|%a|%a", (Collects(), [1, 2, 3], {4: 5})))
    print(str.__mod__("%s|%s|%s", (Collects(), [1, 2, 3], {4: 5})))

# A `*` width reads its own operand through the same column.
for _ in range(30):
    print("%*s|%s" % (4, Collects(), [1, 2, 3]))

# The keyed form holds the mapping itself across every lookup.
for _ in range(30):
    print("%(a)s|%(b)s" % {"a": Collects(), "b": [1, 2, 3]})
