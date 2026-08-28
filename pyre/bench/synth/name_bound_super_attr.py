# pyre-check: max-pypy-ratio=4.5
# pyre-check: skip-cpython
# pyre-check: spec-folds=two_arg_super_call,load_attr_on_super
# The two gates answer different questions.  `spec-folds` is the invariant:
# these two folds fire or the fixture fails, on every host and backend, which
# is what a decline would break.  The ratio is only a backstop against a gross
# regression, and it is sized to the slowest reading rather than the best,
# because that reading is not about super.
#
# On x86_64 ubuntu this shape reads 1.8x under dynasm and 3.6x under cranelift.
# The trace is the same on both -- 60 ops, 19 guards, no residual call and no
# allocation in the loop body -- and on arm64 the two backends agree (1.6x
# dynasm, 1.9x cranelift).  The excess is cranelift's x86_64 code generation,
# which the same runner shows across the macro suite with no super in it:
# spectral_norm 1.6x dynasm against 2.4x cranelift, nbody 1.6x against 2.4x,
# fannkuch 2.9x against 5.4x.  4.5 clears the 3.6x reading by the 15% this
# instrument's own run-to-run spread asks for (`check.py wasm_ratio_gate`
# records the distribution: median 4.7%, p75 8.8%, p90 14.9%).
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
