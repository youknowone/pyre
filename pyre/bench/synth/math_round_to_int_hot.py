# pyre-check: max-pypy-ratio=14
# pyre-check: skip-cpython
# A hot `math.floor` / `math.ceil` / `math.trunc` loop over exact floats.
# The walker specializes all three through
# `try_walker_specialize_math_round_to_int`: unbox the operand, guard it into
# the signed machine range, apply the rounding, and `CastFloatToInt`, instead
# of the opaque `bh_call_fn` residual whose interpreter body looks the dunder
# up on the type and calls it.  `floor` and `ceil` emit a pure elidable
# `CALL_F`; `trunc` needs none, because the cast already truncates toward zero.
# NaN, either infinity, an operand outside the signed range, an int operand
# (whose `int.__floor__` returns the argument object itself), a float subclass
# and a rebound callable all keep the residual.
import math

# Sized so pypy's own execution clears the measurement floor: below it the
# ratio gate divides by the floor and declines the baseline as too small.
N = 40000000


def run():
    total = 0
    for i in range(N):
        x = float(i) * 0.5 - 1000.0
        total += math.floor(x) + math.ceil(x) + math.trunc(x)
    return total


print(run())
