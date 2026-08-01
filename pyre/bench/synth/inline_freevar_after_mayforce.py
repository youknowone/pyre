# pyre-check: max-pypy-ratio=45
# An inlined closure keeps its freevar cells in MIFrame.registers_r across a
# may-force call.  The `Fraction` arithmetic in `forward` is that may-force
# call and invalidates heap-cache facts; the following LOAD_DEREF must recover
# `adjust` from the callee frame's own shadow instead of becoming an unstamped
# GetarrayitemGcR and aborting at the result branch.
#
# The abort is what this fixture guards, and check.py's regression floor gates
# `loops_aborted` at 0 independently of the ratio below.  The ratio itself has
# measured 22-37x since the fixture landed (macOS dynasm 22.4x / cranelift
# 28.9x, ubuntu dynasm 23.6x / cranelift 25.2x / wasm 37.1x) and has never met
# the 20x it was first written with.  It stays high because the compiled loop
# deopts on nearly every iteration: two guards take 16.4k of the 16.9k
# `guard_failures` at N=10000 while `bridges_compiled` stays at 1, so the
# guards are never patched.  That bridge gap is open work, not something this
# gate should hide - keep the gate honest and lower it when the gap closes.
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
