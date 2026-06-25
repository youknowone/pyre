# A try/except INSIDE a hot loop whose body raises through a may-force
# residual call (`//`, `%`, `int("z")`) on some iterations and is CAUGHT,
# with the loop CONTINUING afterwards.  The raising op is int-specialized
# (`GuardFalse(int_eq(divisor, 0))` + bare `IntFloorDiv`); when the divisor
# is 0 the precondition guard deopts, the blackhole re-executes the op, it
# raises, and the in-frame `catch_exception` consumes the exception (v=7).
# The blackhole then reaches the loop merge point and ContinueRunningNormally
# re-enters compiled code.
#
# The raising helper publishes to BOTH the blackhole `last_exc_value` and the
# backend `_store_exception` cells; the in-frame catch cleared only the
# former, so the re-entered loop's first `GUARD_NO_EXCEPTION` read the stale
# backend cell as a spurious pending exception and deopted at the loop header
# (no handler), escaping the try-block — an uncaught ZeroDivisionError.  This
# bench pins that a residual-call raise caught on a resume path that RE-ENTERS
# the loop returns byte-identically (the merge-point re-entry drains the
# backend exception cells).
N = 120000
M = 997


def divzero_in_loop():
    total = 0
    i = 1
    while i < N:
        try:
            v = i // (i % M)
        except ZeroDivisionError:
            v = 7
        total = total + v
        i = i + 1
    return total


def modzero_in_loop():
    total = 0
    i = 1
    while i < N:
        try:
            v = 100 % (i % M)
        except ZeroDivisionError:
            v = 3
        total = total + v
        i = i + 1
    return total


def residual_raise_in_loop():
    total = 0
    i = 1
    while i < N:
        try:
            v = int("z") if (i % M == 0) else i
        except ValueError:
            v = 7
        total = total + v
        i = i + 1
    return total


def main():
    print(divzero_in_loop())
    print(modzero_in_loop())
    print(residual_raise_in_loop())


main()
