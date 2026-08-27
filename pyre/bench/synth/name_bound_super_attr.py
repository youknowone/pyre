# pyre-check: max-pypy-ratio=2
# pyre-check: skip-cpython
# The loop folds to one add per iteration on both JITs, so N has to be in
# the hundreds of millions before pypy's execution time is a measurement
# rather than a clock tick — and at that size cpython is minutes behind.
# `super(C, self)` bound to a name, then read — the spelling LOAD_SUPER_ATTR
# does not cover, because the name binding splits the proxy construction and
# the attribute load into two opcodes the compiler never fuses.
#
# The construction arrives as an opaque may-force `bh_call_fn`, which costs
# more than the four-word allocation it performs: the walk publishes a vref
# for the executing frame ahead of it and re-checks it after, nine ops around
# one.  The read arrives as an opaque MRO walk that also wipes the trace's
# heap-field cache.  Folded, the proxy is `New` + `SetfieldGc` whose reads the
# optimizer answers from the emission itself, so the allocation dies and this
# shape costs what `super(C, self).val(x)` costs.
#
# Sized so pypy's own execution clears the measurement floor: below it the
# ratio gate divides by the floor and reads startup rather than this loop.
N = 500000000


class Base:
    def val(self, x):
        return x + 1


class Child(Base):
    def val(self, x):
        su = super(Child, self)
        return su.val(x)


def run(o, n):
    acc = 0
    for _ in range(n):
        acc = o.val(acc)
    return acc


print(run(Child(), N))
