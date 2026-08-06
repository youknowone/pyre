# No `max-pypy-ratio`: this fixture compiles no loop -- its jitstats record
# `loops_compiled=0` -- so a pypy ratio compares two interpreters' startup
# rather than any generated code, and reads whatever the host's process
# spawn cost happens to be that run. The jitstats baselines gate it.
# Regression guard: a callee stores its OWN frame (sys._getframe(0)); the loop
# body later walks .f_back.f_locals as a SEPARATE residual. That read forces
# the traced caller mid-expression; the escape flush must commit with the
# operand-stack mirror (the vable shadow's stack region is NULL there) and
# resume forward AT the escaping opcode. Before the latched-stack escape
# flush, the flush declined, the FOR_ITER inflight deliver was refused, and
# the consumed iteration was DROPPED -- total came up short (JIT-only).
import sys

box = [None]


def bump(x):
    box[0] = sys._getframe(0)
    return x


def main():
    total = 0
    stale = 0
    for i in range(20000):
        total += bump(i)
        if box[0].f_back.f_locals['total'] != total:
            stale += 1
    print(total, stale)


main()
