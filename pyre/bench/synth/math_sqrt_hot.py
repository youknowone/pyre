# pyre-check: max-pypy-ratio=1
# The ceiling sits between the two measured states: folded this runs 0.1x
# pypy, and with `math_isqrt` suppressed it runs about 2.1x.
# pyre-check: skip-cpython
# cpython 3.46s vs pyre 0.29s (11.9x on the ubuntu runner), and it is not
# gated on — only pypy is.
# Two hot math-builtin loops whose walker specializations this ratio gates.
#
# `math.sqrt(x)`: the walker specializes the call
# (`try_walker_specialize_math_sqrt`) to a domain-guarded pure
# `CALL_F(sqrt_nonneg_jit)` (ll_math.rs `ll_math_sqrt` -> `sqrt_nonneg`,
# EF_ELIDABLE_CANNOT_RAISE) plus inline `wrapfloat`, instead of the opaque
# `bh_call_fn(sqrt_builtin, NULL, x)` residual.  Two guards pin the
# `ll_math_sqrt` branches — `x >= 0` (the ValueError direction) and
# `isfinite(x)` — so the result `W_FloatObject` virtualizes.  A negative /
# non-finite argument or a rebound `math.sqrt` falls through to the residual.
#
# `math.isqrt(i)`: `try_walker_specialize_math_isqrt` turns the call into the
# elidable integer body.  Suppressing that one fold alone measures 20.2x here
# (0.097s -> 1.961s), so a regression that loses it walks straight through the
# ceiling below.  Loosen the ceiling if the machine gets slower; do not drop
# the loop.
import math

# Sized so pypy's own execution clears the measurement floor: below it the
# ratio gate divides by the floor and reads startup rather than these loops.
N_SQRT = 32645190
N_ISQRT = 20000000


def run_sqrt():
    total = 0.0
    for i in range(N_SQRT):
        total += math.sqrt(float(i))
    return total


def run_isqrt():
    total = 0
    i = 0
    while i < N_ISQRT:
        total += math.isqrt(i)
        i += 1
    return total


print(round(run_sqrt(), 6))
print(run_isqrt())
