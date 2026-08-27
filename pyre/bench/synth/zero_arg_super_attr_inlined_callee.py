# pyre-check: max-pypy-ratio=2
# pyre-check: skip-cpython
# The loop folds to one add per iteration on both JITs, so N has to be in
# the hundreds of millions before pypy's execution time is a measurement
# rather than a clock tick — and at that size cpython is minutes behind.
# Zero-argument `super().m(x)` in a callee the FOR_ITER-driven trace inlines —
# the zero-argument twin of `super_attr_inlined_callee.py`.
#
# One bytecode separates them, and it decided whether the callee inlined at
# all: `super()` reads `__class__` out of the closure, so the callee body holds
# a LOAD_DEREF.  LOAD_DEREF lowers to a residual, and
# `fbw_callee_body_replay_scan` listed `RuntimeHelperKind::LoadDeref` in
# neither `replay_safe_read` nor `defer_helper`, so it poisoned that pc and
# refused the whole callee.  The cost was not specific to super — `def m(self,
# x): return x + CAP` over a closed-over CAP ran ~780ns/iteration against ~1ns
# for the same body reading a global — but zero-argument super always carries
# the read, so it always paid.
#
# Sized so pypy's own execution clears the measurement floor: below it the
# ratio gate divides by the floor and reads startup rather than this loop.
N = 500000000


class Base:
    def val(self, x):
        return x + 1


class Child(Base):
    def val(self, x):
        return super().val(x)


def run(o, n):
    acc = 0
    for _ in range(n):
        acc = o.val(acc)
    return acc


print(run(Child(), N))
