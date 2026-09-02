# pyre-check: max-pypy-ratio=4
# The 27 this carried was twice a 13.5x reading taken while pypy ran the loop
# in under the execution floor, so every reading then was a lower bound and
# the derived floor stayed disarmed.  The trip count below now puts pypy's
# execution over the floor-gate minimum, which arms that floor -- and 27
# derives a floor of 1x, which pyre is under.  Refitted to the readings the
# armed baseline gives, 0.9x on a darwin box to 2.2x on the ubuntu runner:
# 4 clears the widest with headroom and its 0.667x floor stays under the
# narrowest.
# A hot `float(x)` builtin-call loop over int and float arguments.  The walker
# specializes the call (`try_walker_specialize_float_call`) to an inline
# conversion — `CastIntToFloat` + `wrapfloat` for an int/bool argument, or the
# `float(f) is f` identity for an exact float — instead of the opaque
# `bh_call_fn(float_type, NULL, x)` residual, so the result virtualizes.  A
# rebound `float` name or a float subclass (which reboxes) falls through to the
# residual.
N = 80008700


def run():
    total = 0.0
    for i in range(N):
        f = float(i)             # int -> CastIntToFloat
        total += float(f) * 0.5  # exact float -> identity forward, then halve
    return total


print(round(run(), 6))
