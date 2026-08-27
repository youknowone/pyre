# pyre-check: max-pypy-ratio=2
# pyre-check: skip-cpython
# The loop folds to one add per iteration on both JITs, so N has to be in
# the hundreds of millions before pypy's execution time is a measurement
# rather than a clock tick — and at that size cpython is minutes behind.
# Zero-argument `super()` bound to a name, then read, in a callee the
# FOR_ITER-driven trace inlines — the zero-argument twin of
# `name_bound_super_attr.py`, and the last super spelling that carried a
# residual.  `zero_arg_name_bound_super_attr.py` keeps the loop inside the
# super-bearing method, where the same two slots reach the walk through the
# standard virtualizable rather than through the callee slot shadow.
#
# `super()` with no arguments reads two frame slots, so it cannot be emitted
# from its operand list the way the two-argument call can.  It was re-routed
# instead: a may-force residual taking the walk's own frame box, which moved
# the frame force onto a channel the walker models — enough to stop the loop
# aborting, not enough to make the call cheap.  What was left published a vref
# for the frame, wiped the trace's heap-field cache and was re-checked by two
# guards, once per iteration.
#
# An inlined callee's walk already holds both slots as SSA values: the inline
# seeds the callee slot shadow with the argument operands and with the live
# closure-cell reads it threaded into the new frame.  Reading them there
# leaves the same `New` + `SetfieldGc` the two-argument spelling emits.
# Measured at 2,000,000 iterations on this shape: ~76ns per iteration before,
# ~0.8 after, against ~1.1 for pypy.
#
# `bare_super_frame_escape.py` carries the correctness half, including the
# portal-frame site this fold declines.
#
# Sized so pypy's own execution clears the measurement floor: below it the
# ratio gate divides by the floor and reads startup rather than this loop.
N = 500000000


class Base:
    def val(self, x):
        return x + 1


class Child(Base):
    def val(self, x):
        su = super()
        return su.val(x)


def run(o, n):
    acc = 0
    for _ in range(n):
        acc = o.val(acc)
    return acc


print(run(Child(), N))
