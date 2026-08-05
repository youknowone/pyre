# pyre-check: max-pypy-ratio=62
# pyre-check: min-pypy-ratio=10
# The frame-identity read the multi-frame blackhole adopt commits, and the
# regression guard for the path that commits it.
#
# The walk executes the forcing residual CONCRETELY, and an inline push never
# runs the interpreter's call sequence. Before `walker_ec_enter` /
# `walker_ec_leave` published the callee frame on the execution context,
# `ec.topframeref` still named the CALLER while the inlined callee body ran, so
# a `sys._getframe` that is itself the escaping call read the caller's frame at
# walk time and the adopt committed that answer instead of discarding it the way
# the legacy escape/replay path does. Measured then as one wrong iteration per
# multi-frame adopt, in each part:
#
#   part_a  `_gf()`  named `main`, not `leaf`
#   part_b  `_gf(1)` named `<module>`, not `main` -- one level too far up, which
#           is the same error seen through the argument
#
# A `_gf(1)` reading `f_locals` on that shape raised `KeyError` for any caller
# local, for the same reason and not because outer locals go unmaterialized.
#
# Both answers are correct now, and they come from the adopt: with every level
# resumable from the concrete frame it owns, the chain is adopted rather than
# replayed
# (`PYRE_FBW_DEBUG_ABORT=1` prints 10 `BUILT multi-frame` and 10 `adopted
# multi-frame terminal`, with no decline of any kind), so this fixture is once
# again the discriminator for the identity answer rather than a guard on the
# replay that stood in for it. Note the read has to be the ESCAPING call: once
# the escape has happened, a `sys._getframe(1)` executed inside the blackhole was
# always correct, because the chain publishes each level's frame as it runs.
import sys

_gf = sys._getframe

wrong_a = []
wrong_b = []


def leaf_a(x):
    name = _gf().f_code.co_name
    if name != "leaf_a":
        wrong_a.append(name)
    return x + 1


def part_a():
    total = 0
    i = 0
    while i < 30000:
        total = leaf_a(total)
        i = i + 1
    return total


def leaf_b(x):
    name = _gf(1).f_code.co_name
    if name != "part_b":
        wrong_b.append(name)
    return x + 1


def part_b():
    total = 0
    i = 0
    while i < 30000:
        total = leaf_b(total)
        i = i + 1
    return total


print(part_a(), part_b(), len(wrong_a), len(wrong_b))
