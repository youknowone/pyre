"""Counter in-place operations remain correct after their loops become hot."""

from collections import Counter
from random import randrange, seed
import sys


OPERATIONS = (
    (Counter.__iadd__, Counter.__add__),
    (Counter.__isub__, Counter.__sub__),
    (Counter.__ior__, Counter.__or__),
    (Counter.__iand__, Counter.__and__),
)

seed(0)
operations = OPERATIONS
if len(sys.argv) > 1:
    operations = (OPERATIONS[int(sys.argv[1])],)
iterations = int(sys.argv[2]) if len(sys.argv) > 2 else 1000

for inplace_op, regular_op in operations:
    for _ in range(iterations):
        left = Counter({key: randrange(-2, 4) for key in "abcd"})
        left.update(e=1, f=-1, g=0)
        right = Counter({key: randrange(-2, 4) for key in "abcd"})
        right.update(h=1, i=-1, j=0)
        result = left.copy()
        identity = id(result)
        expected = regular_op(result, right)
        actual = inplace_op(result, right)
        assert actual == expected
        assert id(actual) == identity

print("OK")
