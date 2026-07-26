# The frame-identity read that the multi-frame blackhole adopt gets wrong, and
# the acceptance test for flipping `PYRE_FBW_MULTIFRAME` default-ON.
#
# The walk executes the forcing residual CONCRETELY, and an inline push never
# runs the interpreter's call sequence, so `ec.topframeref` still names the
# CALLER while the inlined callee body runs. A `sys._getframe` that is itself
# the escaping call therefore reads the caller's frame at walk time, and the
# adopt commits that answer instead of discarding it the way the legacy
# escape/replay path does.
#
# Measured 2026-07-26 with the gate forced on -- one wrong iteration per
# multi-frame adopt, 5 adopts and 5 wrong in each part:
#
#   part_a  `_gf()`  names `main`, not `leaf`
#   part_b  `_gf(1)` names `<module>`, not `main` -- one level too far up, which
#           is the same error seen through the argument
#
# A `_gf(1)` reading `f_locals` on that shape raises `KeyError` for any caller
# local, for the same reason and not because outer locals go unmaterialized.
#
# Both are correct with the gate off, which is the default, so this fixture
# passes today. It exists to fail loudly if the gate is flipped before the
# inlined-call push publishes the callee frame on the execution context. Note
# the read has to be the ESCAPING call: once the escape has happened, a
# `sys._getframe(1)` executed inside the blackhole is correct, because the chain
# publishes each level's frame as it runs.
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
