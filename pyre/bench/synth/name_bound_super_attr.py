# pyre-check: max-pypy-ratio=4
# pyre-check: skip-cpython
# The ceiling is 4 rather than 2 because this ratio is host-dependent and the
# gate has to hold on the worst host, not the best.  Measured on the same
# commit: 1.8x dynasm / 1.9x cranelift on an arm64 mac runner, 2.1x dynasm on
# a windows runner, 2.0x dynasm on an x86_64 ubuntu runner -- and 3.0x
# cranelift on that same ubuntu runner.  The excess is cranelift's x86_64
# code for the proxy ops, not a decline: the fold fires on every host.
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
