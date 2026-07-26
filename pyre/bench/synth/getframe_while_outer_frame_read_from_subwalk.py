# The inlined callee reads its CALLER's frame, from inside an inline sub-walk.
#
# This is the acceptance test for `PYRE_FBW_MULTIFRAME`. The multi-frame
# blackhole runs the recovered chain innermost-first, and each level runs against
# its own live frame -- but an OUTER level's recovered locals are still only in
# its blackhole registers at that point, not written back to its live frame. An
# inner `sys._getframe(1)` read therefore observes a caller frame that has not
# been restored yet.
#
# Measured 2026-07-26 with the gate forced on: `part_a` raises
# `KeyError: 'bias'`, and `part_b` returns 29995 instead of 30000 -- exactly one
# lost iteration per multi-frame adopt (5 adopts, 5 missing). Both are correct
# with the gate off, which is the default, so this fixture passes today and is
# here to fail loudly if the gate is flipped before the outer-frame
# materialization lands.
import sys

_gf = sys._getframe


def leaf_a(x):
    f = _gf(1)
    return x + f.f_locals["bias"]


def part_a():
    total = 0
    bias = 1
    i = 0
    while i < 30000:
        total = leaf_a(total)
        bias = 1
        i = i + 1
    return total, bias


def leaf_b(x):
    f = _gf(1)
    return x + (1 if f.f_code.co_name == "part_b" else 0)


def part_b():
    total = 0
    i = 0
    while i < 30000:
        total = leaf_b(total)
        i = i + 1
    return total


print(part_a(), part_b())
