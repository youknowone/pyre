# pyre-check: spec-folds=bare_super_virtual,load_attr_on_super
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
try:
    import pypyjit

    pypyjit.set_param("threshold=20,function_threshold=20")
except ImportError:
    pass

N = 20000


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
