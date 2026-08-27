# pyre-check: max-pypy-ratio=2
# pyre-check: skip-cpython
# The loop folds to one add per iteration on both JITs, so N has to be in
# the hundreds of millions before pypy's execution time is a measurement
# rather than a clock tick — and at that size cpython is minutes behind.
# Zero-argument `super()` bound to a name, with the loop inside the
# super-bearing method — the portal-frame twin of
# `zero_arg_name_bound_super_attr_inlined_callee.py`.
#
# One thing separates them and it is not visible in the Python: the two slots
# `super()` reads reach the walk on different channels.  A callee the trace
# inlines owns a slot shadow the inline seeded; a method that carries its own
# loop is the frame the portal traces, and its slots come out of the standard
# virtualizable.  A fold reading only the first channel leaves this spelling on
# the re-routed may-force residual, which is where it was: ~62ns per iteration
# at 2,000,000 iterations, against ~3 for the same body with no `super()` in
# it at all.
#
# At this size the two are indistinguishable — this fixture and the same loop
# calling `self.val(acc)` both run 0.53s against pypy's 0.35 — so what is left
# is the loop shape, not the proxy.
#
# Sized so pypy's own execution clears the measurement floor: below it the
# ratio gate divides by the floor and reads startup rather than this loop.
N = 500000000


class Base:
    def val(self, x):
        return x + 1


class Child(Base):
    def loop(self, n):
        acc = 0
        for _ in range(n):
            su = super()
            acc = su.val(acc)
        return acc


print(Child().loop(N))
