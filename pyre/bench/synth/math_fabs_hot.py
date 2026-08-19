# pyre-check: max-pypy-ratio=14
# pyre-check: skip-cpython
# A hot `math.fabs(x)` loop.  `interp_math.py:386` is `math1(space, math.fabs,
# w_x)` and RPython lowers `ll_math_fabs` to a sign mask, so
# `try_walker_specialize_math_fabs` emits a single `FloatAbs` on the unboxed
# operand plus an inline `wrapfloat` rather than the opaque
# `bh_call_fn(fabs_builtin, NULL, x)` residual.  `fabs` raises for no input, so
# the fold carries no domain guard; only the operand's class and exact-w_class
# guards remain.  A numeric subclass or a rebound `math.fabs` declines.
import math

# Sized so pypy's own execution clears the measurement floor: below it the
# ratio gate divides by the floor and declines the baseline as too small.
N = 40000000


def run():
    total = 0.0
    for i in range(N):
        total += math.fabs(float(i) - 6000000.0)
    return total


print(round(run(), 6))
