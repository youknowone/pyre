# pyre-check: max-pypy-ratio=4
# pyre-check: skip-cpython
# The ceiling is 4 rather than 2 because this ratio is host-dependent and the
# gate has to hold on the worst host, not the best.  Measured on the same
# commit: 1.5x dynasm / 1.5x cranelift on an arm64 mac runner, 1.9x dynasm on
# a windows runner, 2.0x dynasm on an x86_64 ubuntu runner -- and 3.0x
# cranelift on that same ubuntu runner.  The excess is cranelift's x86_64
# code for the proxy ops, not a decline: the fold fires on every host.
# The loop folds to one add per iteration on both JITs, so N has to be in
# the hundreds of millions before pypy's execution time is a measurement
# rather than a clock tick — and at that size cpython is minutes behind.
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
# Sized so pypy's own execution clears the measurement floor: below it the
# ratio gate divides by the floor and reads startup rather than this loop.
N = 500000000


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
