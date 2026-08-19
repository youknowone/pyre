# pyre-check: max-pypy-ratio=14
# pyre-check: skip-cpython
# cpython 1.70s vs pyre 0.46s (3.7x on the ubuntu runner), and it is not
# gated on — only pypy is.
# RPython lowers the guarded `ll_math_log/cos/sin` bodies to raw float calls.
# Keep all inputs in their hot domains so the walker emits those CALL_F ops
# and the temporary W_FloatObject results can virtualize.
from math import cos, log, sin

N = 6600000


def run():
    total = 0.0
    for i in range(N):
        x = 1.0 + float(i % 97) / 97.0
        total += log(x) + cos(x) + sin(x)
    return total


print(round(run(), 6))
