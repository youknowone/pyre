# pyre-check: selfcheck
# pyre-check: selfcheck-compiles=hot1,hot2
# A `math` fold recorded on an exact `int` must stop answering once the operand
# is an `int` subclass.
#
# The coercion every float fold shares unboxes behind a `guard_class` on
# `ob_type`, which a numeric subclass shares with its builtin; what separates
# the two is `w_class`, and `w_class` is what decides whether `try_get_double`
# finds an overridden `__float__`.  Four of the eleven arms that coerce an
# operand to a float used to leave that pin to the caller and never emit it, so
# a loop compiled on `2` kept computing `tan(2.0)` for a `MyInt(2)` whose
# `__float__` answers `100.0`.  The pin now belongs to the coercion, so no arm
# can be written without it.
#
# THE ORACLE IS THIS INTERPRETER, not another implementation.  `math.log` reads
# its argument without consulting `__float__` here and in CPython, while pypy
# does consult it; the invariant that holds regardless is that the compiled
# loop answers what this build's own interpreter answers, and calls `__float__`
# as often as it does.
#
# THE COUNT IS THE TEST.  A fold that keeps answering calls `__float__` once —
# for the single interpreted iteration before the loop is entered — and then
# never again, whatever the value it produces.  The values are checked too, but
# a row whose two answers happen to agree would still be caught by its count.
import math

WARM = 6000


class MyInt(int):
    def __float__(self):
        CALLS[0] += 1
        return 100.0


CALLS = [0]


def hot1(n, fn, x):
    total = 0.0
    for _ in range(n):
        total += fn(x)
    return total


def hot2(n, fn, x, y):
    total = 0.0
    for _ in range(n):
        total += fn(x, y)
    return total


# (name, warm-up operands, operands whose first is the subclass).  Every row is
# a fold arm: sqrt / log+cos+sin / fabs / isclose have their own, the rest are
# rows of the generic one- and two-argument tables.
CASES = (
    ("sqrt", (2,), (MyInt(2),)),
    ("log", (2,), (MyInt(2),)),
    ("cos", (2,), (MyInt(2),)),
    ("sin", (2,), (MyInt(2),)),
    ("fabs", (2,), (MyInt(2),)),
    ("tan", (2,), (MyInt(2),)),
    ("exp", (2,), (MyInt(2),)),
    ("atan", (2,), (MyInt(2),)),
    ("degrees", (2,), (MyInt(2),)),
    ("pow", (2, 3), (MyInt(2), 3)),
    ("fmod", (2, 3), (MyInt(2), 3)),
    ("copysign", (2, 3), (MyInt(2), 3)),
    ("remainder", (2, 3), (MyInt(2), 3)),
    ("atan2", (2, 3), (MyInt(2), 3)),
    # `isclose` answers False either way on `(2, 3)`, so give it a pair whose
    # two readings disagree.
    ("isclose", (2, 100), (MyInt(2), 100)),
)


def measure(name, warm_args, test_args):
    fn = getattr(math, name)
    runner = hot1 if len(test_args) == 1 else hot2

    # What one interpreted call does on this build, before anything is warm.
    CALLS[0] = 0
    want = float(fn(*test_args))
    per_call = CALLS[0]

    runner(WARM, fn, *warm_args)
    CALLS[0] = 0
    total = runner(WARM, fn, *test_args)
    return total / WARM, want, CALLS[0], per_call * WARM


def main():
    failures = []
    for name, warm_args, test_args in CASES:
        got, want, calls, owed = measure(name, warm_args, test_args)
        if calls != owed:
            failures.append(
                "math.%s called __float__ %d times over %d iterations, owed %d "
                "— the compiled loop is answering for the subclass it was not "
                "recorded on" % (name, calls, WARM, owed)
            )
        if abs(got - want) > abs(want) * 1e-12:
            failures.append(
                "math.%s compiled to %r, this interpreter answers %r"
                % (name, got, want)
            )
    if failures:
        for line in failures:
            print("FAIL", line)
        return 1
    print("PASS every math float fold deopts for an int subclass")
    return 0


import sys

sys.exit(main())
