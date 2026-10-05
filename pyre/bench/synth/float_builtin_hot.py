# pyre-check: max-pypy-ratio=3
# Root brackets erased: macos cranelift 0.8x-0.9x, ubuntu 2.1x-2.5x before. Floor 0.75x.
# A hot `float(x)` builtin-call loop over int and float arguments.  An exact
# int walks `floatobject.py newfloat` after `CastIntToFloat`.  An exact float
# is `float(f) is f`.  A rebound `float` name or a float subclass (which
# reboxes) falls through to the `bh_call_fn(float_type, NULL, x)` residual.
N = 80008700


def run():
    total = 0.0
    for i in range(N):
        f = float(i)             # exact int -> newfloat
        total += float(f) * 0.5  # exact float -> identity forward, then halve
    return total


print(round(run(), 6))
