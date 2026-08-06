# pyre-check: max-pypy-ratio=14
# RPython lowers the guarded `ll_math_log/cos/sin` bodies to raw float calls.
# Keep all inputs in their hot domains so the walker emits those CALL_F ops
# and the temporary W_FloatObject results can virtualize.
from math import cos, log, sin

N = 200000


def run():
    total = 0.0
    for i in range(N):
        x = 1.0 + float(i % 97) / 97.0
        total += log(x) + cos(x) + sin(x)
    return total


print(round(run(), 6))
