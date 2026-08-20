# pyre-check: max-pypy-ratio=86
# The ceiling is a function of N, so raising N refits it. pypy's execution here
# is almost all fixed cost -- doubling N moved it 0.035s to 0.039s -- while this
# backend pays roughly 27us per iteration, so the ratio tracks N nearly one for
# one. At N = 64000 the local cranelift ratio reads 44.9x, and the ceiling sits
# just under twice it.
# An inlined closure keeps its freevar cells in MIFrame.registers_r across a
# may-force call.  The `Fraction` arithmetic in `forward` is that may-force
# call and invalidates heap-cache facts; the following LOAD_DEREF must recover
# `adjust` from the callee frame's own shadow instead of becoming an unstamped
# GetarrayitemGcR and aborting at the result branch.
#
# The abort is what this fixture guards, and check.py's regression floor gates
# `loops_aborted` at 0 independently of the ratio below.  Bridge resume must
# retain the callee frame's own globals identity: treating its valid pc=0 as a
# failed decode leaves two guards failing on nearly every iteration, while
# guarding the callee namespace through the portal/root frame compiles an
# endless chain of equally failing bridges.  With both frame properties
# preserved, guard failures stay at 471 and local native ratios are 10.3x
# dynasm / 15.9x cranelift versus PyPy.
from fractions import Fraction


# Sized so pypy's own execution clears the measurement floor: below it the
# ratio gate divides by the floor and reads startup rather than this loop.
#
# The upper bound is what fixes the counters. At 32176 the loop ended while the
# JIT was still converging, so the gated totals recorded how far that had got
# and moved between two runs of one binary -- 923/6 loops against 925/7 here,
# 923/6 against 938/8 on ubuntu and 922 against 923 on windows. Convergence
# completes by 48000 on both native backends, and past it every gated counter is
# independent of N: dynasm holds 1004 guard failures and cranelift 1010, with
# six loops and five bridges, unchanged from 48000 through 96000. This sits far
# enough above that point to keep the fixed point on a host that needs a few
# more iterations to reach it.
N = 64000


def make_forwarder():
    def adjust(value):
        return value + Fraction(3, 97)

    def forward(value):
        scaled = value / Fraction(2, 89)
        return adjust(scaled)

    return forward


forward = make_forwarder()
count = 0
for i in range(N):
    value = forward(Fraction(i % 97 + 1, 97))
    if value > 1:
        count += 1

print(count)
