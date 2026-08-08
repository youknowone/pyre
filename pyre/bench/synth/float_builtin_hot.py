# pyre-check: max-pypy-ratio=27
# The ceiling is twice the slowest ratio observed, 13.5x on the macos
# runner. pypy runs this in under the execution floor, so every reading is
# a lower bound on the true ratio and the ceiling is the conservative side
# of one: the derived floor stays disarmed here for the same reason.
# A hot `float(x)` builtin-call loop over int and float arguments.  The walker
# specializes the call (`try_walker_specialize_float_call`) to an inline
# conversion — `CastIntToFloat` + `wrapfloat` for an int/bool argument, or the
# `float(f) is f` identity for an exact float — instead of the opaque
# `bh_call_fn(float_type, NULL, x)` residual, so the result virtualizes.  A
# rebound `float` name or a float subclass (which reboxes) falls through to the
# residual.
N = 300000


def run():
    total = 0.0
    for i in range(N):
        f = float(i)             # int -> CastIntToFloat
        total += float(f) * 0.5  # exact float -> identity forward, then halve
    return total


print(round(run(), 6))
