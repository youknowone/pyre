# CPython-suite gap: `test_dict` runs `items ^ items` on random dicts only a
# few times, so whether the loop gets hot enough to trace depends on what ran
# before it in the same process; the suite crashes on one runner layout and
# passes on another.
#
# parity-tests reason: tracing `dict.items() ^ dict.items()` records a helper
# that builds a fixed-size array and hands it on as a slice.  The callee reads
# the slice's length and items, so the array must be allocated with the length
# header that reader expects.  A fixed seed and enough iterations make the loop
# hot deterministically.
import random

random.seed(6)
rr = random.randrange
for _ in range(300):
    left = {x: rr(3) for x in range(20) if rr(2)}
    right = {x: rr(3) for x in range(20) if rr(2)}
    expected = set(left.items()) ^ set(right.items())
    actual = left.items() ^ right.items()
    assert actual == expected, (left, right)
print("OK")
