import math
import random


assert repr(sum([-0.0])) == "0.0"
assert repr(sum([-0.0], -0.0)) == "-0.0"
assert repr(sum([], -0.0)) == "-0.0"

assert sum([0.1] * 10) == 1.0
assert math.isinf(sum([float("inf"), float("inf")]))
assert math.isinf(sum([1e308, 1e308]))

random.seed(0)
values = [
    complex(random.random() - 0.5, random.random() - 0.5)
    for _ in range(10_000)
]
assert sum(values) == complex(
    sum(value.real for value in values),
    sum(value.imag for value in values),
)

for values in (
    [complex(1, -0.0), 1],
    [1, complex(1, -0.0)],
    [complex(1, -0.0), 1.0],
    [1.0, complex(1, -0.0)],
):
    result = sum(values)
    assert result == complex(2, -0.0)
    assert math.copysign(1.0, result.imag) == -1.0

print("OK")
