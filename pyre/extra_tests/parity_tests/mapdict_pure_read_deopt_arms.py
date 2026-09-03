# CPython-suite gap: attribute tests never hot-loop a constant receiver whose
# attribute is written, deleted, or devolved out from under the compiled read.
# parity-tests reason: pins the `_pure_direct_read` fold's deopt arms without
# adding a throughput fixture's guard noise to the synth gate.

"""`AbstractAttribute.read` answers a green receiver's never-written attribute
from the elidable read; every way of making it written must be observed."""


try:
    import pypyjit

    pypyjit.set_param("threshold=20,function_threshold=20")
except ImportError:
    pass

N = 30000


# The fold's own shape: a module-level instance, so the receiver is a constant,
# and an attribute written once during construction and never again.
class Held:
    def __init__(self):
        self.value = 11
        self.tag = "t"


held = Held()
total = 0
for _ in range(N):
    total += held.value
assert total == 11 * N, total
assert held.tag == "t"


# A write mid-loop marks the attribute mutated; the very next iteration must
# read the new value, not the one the trace folded in.
class Rebound:
    def __init__(self):
        self.value = 1


rebound = Rebound()
seen = 0
for i in range(N):
    seen = rebound.value
    if i == N // 2:
        rebound.value = 2
assert seen == 2, seen


# Deleting the attribute marks it mutated too, and the read that follows must
# raise instead of answering from the folded value.
class Dropped:
    def __init__(self):
        self.value = 7


dropped = Dropped()
misses = 0
for i in range(N):
    try:
        assert dropped.value == 7
    except AttributeError:
        misses += 1
    if i == N // 2:
        del dropped.value
assert misses == N - N // 2 - 1, misses


# Devolving the instance dict moves the attribute out of the map chain without
# touching the attribute node, so the map pin is what has to notice.
class Devolved:
    def __init__(self):
        self.value = 3


devolved = Devolved()
seen = 0
for i in range(N):
    seen = devolved.value
    if i == N // 2:
        devolved.__dict__[1] = "devolve"
        devolved.__dict__["value"] = 4
assert seen == 4, seen


# The attribute node is shared by every instance of the class, so a write
# through a sibling instance marks it mutated for the folded receiver too.
class Shared:
    def __init__(self, v):
        self.value = v


pinned = Shared(5)
other = Shared(5)
seen = 0
for i in range(N):
    seen = pinned.value
    if i == N // 2:
        other.value = 6
assert seen == 5, seen


# A slot resolves to a `("slot", SLOTS_STARTING_FROM + index)` attribute node
# on the same map chain, and its write marks the same flag.
class Slotted:
    __slots__ = ("value",)

    def __init__(self):
        self.value = 8


slotted = Slotted()
seen = 0
for i in range(N):
    seen = slotted.value
    if i == N // 2:
        slotted.value = 9
assert seen == 9, seen


# An unboxed float slot takes the `_pure_unboxed_read` half of the same fold.
class Floated:
    def __init__(self):
        self.value = 1.5


floated = Floated()
acc = 0.0
for _ in range(N):
    acc += floated.value
assert acc == 1.5 * N, acc

print("OK")
