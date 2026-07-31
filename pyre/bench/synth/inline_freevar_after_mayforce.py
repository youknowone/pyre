# pyre-check: max-pypy-ratio=20
# An inlined closure keeps its freevar cells in MIFrame.registers_r across a
# may-force call.  The append invalidates heap-cache facts; the following
# LOAD_DEREF must recover `adjust` from the callee frame's own shadow instead
# of becoming an unstamped GetarrayitemGcR and aborting at the result branch.
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
