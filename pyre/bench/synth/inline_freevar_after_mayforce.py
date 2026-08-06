# pyre-check: max-pypy-ratio=25
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
# preserved, guard failures fall from 16898 to 471 and local native ratios are
# 10.3x dynasm / 15.9x cranelift versus PyPy (formerly 22-29x on macOS).
from fractions import Fraction


N = 10000


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
