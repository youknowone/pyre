# pyre-check: max-pypy-ratio=3
# pyre-check: skip-cpython
# pyre-check: spec-folds=load_deref,load_super_attr,load_super_attr_descent,super_attr_unwrap
# Zero-argument `super().val()` with the loop INSIDE the super-bearing method,
# in the `for` spelling — the `for` twin of `load_super_attr.py`.
#
# The loop form is the point.  `eval.rs for_iter_body_op_is_jit_safe` is an
# allow-list over the opcodes a FOR_ITER body may hold, and `LOAD_SUPER_ATTR`
# was not on it, so the whole `run` frame was refused at the back edge:
# `gate_declined_for_iter_region` counted once per iteration and the only
# compiled loop was `val`'s own function-entry trace.  At 2,000,000 iterations
# that ran 2.28s, against 0.03s for the identical body calling a plain
# `self.plain()`.  The `while` twin never reaches the gate and so never showed
# it.
#
# The body is also the LOAD_DEREF carrier: `super()` with no arguments reads
# `__class__` out of the closure, so the compiler emits LOAD_GLOBAL super,
# LOAD_DEREF __class__, LOAD_FAST self, LOAD_SUPER_ATTR — three of the four
# were already admitted.
#
# N is sized for pypy, not for cpython.  This file used to run 10,000
# iterations, at which pypy's execution-only time is a few microseconds --
# under `EXEC_TIME_FLOOR_S`, so check.py printed the ratio with a `~`, whose
# legend says "ratio is not a measurement, and no ratio gate is applied to
# it".  What that column showed was warmup: the same file reads 12.6ns per
# iteration at 1,000,000 and 0.9ns at 20,000,000.  pypy settles at ~0.6ns per
# iteration here, so 250,000,000 is what puts its execution time (~0.17s) far
# enough above `FLOOR_GATE_MIN_BASELINE_S` for the ratio to be a measurement
# and for the ceiling below to be enforced.  cpython cannot usefully run that
# many, hence `skip-cpython`; pypy stays the oracle the backends' output is
# compared against.
#
# The ceiling is fitted to what that measurement reads: 1.5x on dynasm and
# 1.7x on cranelift, carried with room for a slower host.
# `load_super_attr.py` carries the same 3 for the same family.
#
try:
    import pypyjit

    pypyjit.set_param("threshold=20,function_threshold=20")
except ImportError:
    pass

N = 250000000


class Base:
    def val(self):
        return 1


class Child(Base):
    def run(self, n):
        acc = 0
        for _ in range(n):
            acc = acc + super().val()
        return acc


print(Child().run(N))
