# CPython-suite gap: math tests do not collect while retaining numeric operands.
# parity-tests reason: this is a pyre/PyPy moving-GC root-liveness regression.
# parity-env: PYPY_GC_NURSERY=4096

"""Math reductions must keep collected Python numeric operands alive."""

import math
from decimal import Decimal
from fractions import Fraction


PROD_LIMIT = 1000
DIST_LIMIT = 300
SUMPROD_LIMIT = 200


expected_product = Decimal(3)
for factor in map(Decimal, range(1, PROD_LIMIT)):
    expected_product *= factor
actual_product = math.prod(
    map(Decimal, range(1, PROD_LIMIT)),
    start=Decimal(3),
)
assert actual_product == expected_product


point = list(map(Decimal, range(DIST_LIMIT)))
assert math.dist(point, map(Decimal, range(DIST_LIMIT))) == 0.0


left_values = range(1, SUMPROD_LIMIT)
right_values = range(SUMPROD_LIMIT, 1, -1)
expected_sumprod = 0
for left, right in zip(map(Fraction, left_values), map(Fraction, right_values)):
    expected_sumprod += left * right
actual_sumprod = math.sumprod(
    map(Fraction, range(1, SUMPROD_LIMIT)),
    map(Fraction, range(SUMPROD_LIMIT, 1, -1)),
)
assert actual_sumprod == expected_sumprod


print("OK")
