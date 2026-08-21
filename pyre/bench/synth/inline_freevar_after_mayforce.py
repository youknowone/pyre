# pyre-check: max-pypy-ratio=86
# pyre-check: jitstats-band=guard_failures=1
# One tree, three runners, one CI run (`1d212895c6b`): macOS and ubuntu read
# 1003 dynasm / 1008 cranelift, windows 1004 / 1009. The loop and bridge counts
# agreed at six and five everywhere, so the split is carried entirely by this
# counter and is not a function of the tree. The baseline holds the pair the two
# agreeing runners read; the band is exactly the measured width, so anything
# wider than the split still gates.
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
# independent of N: dynasm holds 1003 guard failures and cranelift 1008, with
# six loops and five bridges, unchanged from 48000 through 96000. This sits far
# enough above that point to keep the fixed point on a host that needs a few
# more iterations to reach it.
#
# The fixed point was seven loops and 1007/1012 guard failures for as long as
# the blackhole recognized `catch_exception/L`: once jd0's staticdata carries
# the assembler's real opcode ids instead of the 255 `op_live` sentinel, an
# exception it used to let escape is caught, and the extra loop plus three
# guard failures are that arm being compiled. `driver_finish_setup` still
# installs those ids, but the arm is no longer reached, so the counters sit
# below that pair on both native backends. Whatever stopped reaching it is
# unattributed; the wasm backend never reached the arm at all, which is why its
# baseline never moved off six.
#
# At six loops the guard counts have alternated between 1004/1009 and 1003/1008
# across runs while the loop and bridge counts held. Only the loop count answers
# whether the arm compiles, so treat a one-count move here as the unattributed
# remainder rather than as this fixture's subject.
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
