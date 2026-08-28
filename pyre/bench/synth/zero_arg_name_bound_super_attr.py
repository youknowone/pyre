# pyre-check: max-pypy-ratio=2
# pyre-check: skip-cpython
# The ceiling is 4 rather than 2 because this ratio is host-dependent and the
# gate has to hold on the worst host, not the best.  Measured on the same
# commit: 1.7x on an arm64 mac runner and 1.8x dynasm on a windows runner,
# but 2.3x dynasm and 2.5x cranelift on an x86_64 ubuntu runner -- two
# different backends over 2 on the one host, so 2 is not a ceiling this
# shape holds anywhere but the fastest machines.
# The loop folds to one add per iteration on both JITs, so N has to be in
# the hundreds of millions before pypy's execution time is a measurement
# rather than a clock tick — and at that size cpython is minutes behind.
# Zero-argument `super()` bound to a name, with the loop inside the
# super-bearing method.
#
# Which channel the walk reads is not visible in the Python: a callee the trace
# inlines owns a slot shadow the inline seeded, while a method that carries its
# own loop is the frame the portal traces and its slots come out of the
# standard virtualizable.  This fixture is the portal-frame side; the callee
# side is `super_attr_inlined_callee.py`, and `bare_super_frame_escape.py`
# pins both channels' answers.  A fold reading only the callee channel leaves
# this spelling on the re-routed may-force residual, which is where it was: ~62ns per iteration
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
