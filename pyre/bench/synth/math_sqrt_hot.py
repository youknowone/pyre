# pyre-check: max-pypy-ratio=11
# A hot `math.sqrt(x)` loop.  The walker specializes the call
# (`try_walker_specialize_math_sqrt`) to a domain-guarded pure
# `CALL_F(sqrt_nonneg_jit)` (ll_math.rs `ll_math_sqrt` -> `sqrt_nonneg`,
# EF_ELIDABLE_CANNOT_RAISE) plus inline `wrapfloat`, instead of the opaque
# `bh_call_fn(sqrt_builtin, NULL, x)` residual.  Two guards pin the
# `ll_math_sqrt` branches — `x >= 0` (the ValueError direction) and
# `isfinite(x)` — so the result `W_FloatObject` virtualizes.  A negative /
# non-finite argument or a rebound `math.sqrt` falls through to the residual.
import math

N = 300000


def run():
    total = 0.0
    for i in range(N):
        total += math.sqrt(float(i))
    return total


print(round(run(), 6))
