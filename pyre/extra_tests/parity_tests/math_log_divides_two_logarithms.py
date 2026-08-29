# pyre-check: pypy-diverges: pypy3's `math` states `math domain error` for
# every refusal here and computes `log(x, base)` with its own division, so
# neither the wording nor the base-1 refusal is expressible there.
#
# CPython-suite gap: `test_math` checks `log(x, base)` against a tolerance and
# the refusals by exception type, so a runtime that takes the logarithm in
# base `base` directly, or refuses a base of 1 as a domain error, passes it.
#
# parity-tests reason: `math_log_impl` is two `loghelper` calls and a division.
# That is observable three ways: the argument is what a refusal names first, a
# base of 1 is a zero denominator rather than a domain error, and the result is
# `log(x) / log(base)` -- not `log2`/`log10`, which round differently.
# `loghelper` also splits its refusal in two: an integer operand can be
# arbitrarily large, so its message carries no value, while a float's does.
import math


def refusal(fn):
    try:
        fn()
    except BaseException as exc:
        return "%s: %s" % (type(exc).__name__, exc)
    raise AssertionError("accepted")


# Two logarithms and a division, so a base is never folded into the argument's
# own logarithm.
for x, base in [(1e300, 10), (100, 10), (8, 2), (1 << 60, 2), (3, 7), (1e-300, 10)]:
    assert math.log(x, base) == math.log(x) / math.log(base), (x, base)

# A base of 1 makes the denominator zero, which is a division, not a domain.
for base in (1, 1.0, True):
    assert refusal(lambda b=base: math.log(2, b)) == "ZeroDivisionError: division by zero", base

# The argument is read first, so it is what a refusal names even when the base
# is also out of the domain.
assert refusal(lambda: math.log(0, -1.0)) == "ValueError: expected a positive input"
assert refusal(lambda: math.log(0.0, -1.0)) == (
    "ValueError: expected a positive input, got 0.0"
), refusal(lambda: math.log(0.0, -1.0))
# A NaN argument is not a refusal, so the base is what fails.
assert refusal(lambda: math.log(float("nan"), 0)) == "ValueError: expected a positive input"
assert refusal(lambda: math.log(2, 0.0)) == "ValueError: expected a positive input, got 0.0"
assert refusal(lambda: math.log(2, 0)) == "ValueError: expected a positive input"

# An integer that a float can hold is converted and the logarithm taken of the
# conversion, so an int and the float beside it answer alike.
for n in [10**k for k in range(1, 300)] + [1 << k for k in range(1, 1000)]:
    assert math.log(n) == math.log(float(n)), n
    assert math.log2(n) == math.log2(float(n)), n
    assert math.log10(n) == math.log10(float(n)), n

# Each restricted domain names its own constraint and the value it read.
assert refusal(lambda: math.log1p(-1.0)) == (
    "ValueError: expected argument value > -1, got -1.0"
), refusal(lambda: math.log1p(-1.0))
assert refusal(lambda: math.log1p(-2)) == "ValueError: expected argument value > -1, got -2.0"
assert refusal(lambda: math.acosh(0.5)) == (
    "ValueError: expected argument value not less than 1, got 0.5"
), refusal(lambda: math.acosh(0.5))
assert refusal(lambda: math.acosh(0)) == (
    "ValueError: expected argument value not less than 1, got 0.0"
)
assert refusal(lambda: math.lgamma(-1.0)) == (
    "ValueError: expected a noninteger or positive integer, got -1.0"
), refusal(lambda: math.lgamma(-1.0))
assert refusal(lambda: math.lgamma(0.0)) == (
    "ValueError: expected a noninteger or positive integer, got 0.0"
)
assert refusal(lambda: math.gamma(-1.0)) == (
    "ValueError: expected a noninteger or positive integer, got -1.0"
)

# Nothing above disturbed the ordinary answers.
assert math.log(1) == 0.0
assert math.log2(1024) == 10.0
assert math.log10(1000) == 3.0
assert math.log1p(0.0) == 0.0
assert math.acosh(1.0) == 0.0
assert math.lgamma(1.0) == 0.0

print("OK")
