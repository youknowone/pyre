# pyre-check: max-pypy-ratio=4
# pyre-check: skip-cpython
# pyre-check: spec-folds=load_super_attr,super_attr_unwrap
# `spec-folds` is the invariant here: those two folds fire or the fixture
# fails, on every host and backend.  The ratio is a backstop sized to the
# slowest reading -- 3.1x under cranelift on x86_64 ubuntu against 1.9x under
# dynasm on that same runner, where the whole macro suite shows the same
# backend spread with no super in it.  See `name_bound_super_attr.py` for the
# measurement.
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
