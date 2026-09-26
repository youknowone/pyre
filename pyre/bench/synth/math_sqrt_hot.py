# pyre-check: max-pypy-ratio=0.6
# Ubuntu run 33279264115: 0.2-0.3x; the ceiling is twice the slowest,
# rounded up to one decimal place.
# The ceiling sits between the two measured states: folded this runs 0.1x
# pypy, and with the `isqrt` gateway's fast arm suppressed it runs about 2.1x.
# pyre-check: skip-cpython
# cpython 3.46s vs pyre 0.29s (11.9x on the ubuntu runner), and it is not
# gated on — only pypy is.
# Two hot math-builtin loops whose walker specializations this ratio gates.
#
# `math.sqrt(x)`: the builtin-call descent walks the `__majit_wrap_math_sqrt`
# gateway (interp_math.py `math1` + ll_math.py `ll_math_sqrt`) and records a
# pure `CALL_F(sqrt_nonneg_jit)` (`sqrt_nonneg`, EF_ELIDABLE_CANNOT_RAISE)
# plus the boxed float, instead of the opaque `bh_call_fn(sqrt_builtin, NULL,
# x)` residual.  The gateway's `x >= 0` and `isfinite(x)` branches become the
# two guards, so the result `W_FloatObject` virtualizes.  A negative /
# non-finite argument takes the `dont_look_inside` slow path.
#
# `math.isqrt(i)`: `__majit_wrap_math_isqrt` walks `_int_isqrt` for an exact
# positive machine int below `2**53`.  Suppressing that fast arm measures 20.2x here
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
