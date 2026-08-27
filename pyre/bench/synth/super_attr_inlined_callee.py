# pyre-check: max-pypy-ratio=2
# pyre-check: skip-cpython
# The loop folds to one add per iteration on both JITs, so N has to be in
# the hundreds of millions before pypy's execution time is a measurement
# rather than a clock tick — and at that size cpython is minutes behind.
# `super(C, self).m(x)` in a callee the FOR_ITER-driven trace inlines.
# `load_super_attr.py` keeps its loop inside the super-bearing method, so the
# callee replay scan never sees the LOAD_SUPER_ATTR residuals; this shape does.
#
# Sized so pypy's own execution clears the measurement floor: below it the
# ratio gate divides by the floor and reads startup rather than this loop.
N = 500000000


class Base:
    def val(self, x):
        return x + 1


class Child(Base):
    def val(self, x):
        return super(Child, self).val(x)


def run(o, n):
    acc = 0
    for _ in range(n):
        acc = o.val(acc)
    return acc


print(run(Child(), N))
