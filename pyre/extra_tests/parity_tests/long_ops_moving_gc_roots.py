# CPython-suite gap: longobject tests do not collect between digit-array
# allocations of a still-live operand.
# parity-tests reason: nursery-born rbigint digits move; host locals that
# read an operand after with_size/Digits::new must hold it on the shadow
# stack (live_rbigint / RBigIntGcRoot).
# parity-env: PYPY_GC_NURSERY=1
# parity-env: PYPY_GC_NURSERY_DEBUG=1

"""Long operations must keep both operands alive across collecting allocs."""

LEFT = 10**50 + 7
RIGHT = 10**25 + 3

EXPECTED_TRUEDIV = LEFT / RIGHT
EXPECTED_FLOORDIV = LEFT // RIGHT
EXPECTED_MOD = LEFT % RIGHT
EXPECTED_DIVMOD = divmod(LEFT, RIGHT)
EXPECTED_POWMOD = pow(LEFT, 5, RIGHT)
EXPECTED_LSHIFT = LEFT << 7
EXPECTED_RSHIFT = LEFT >> 7
EXPECTED_FLOAT = float(LEFT)
EXPECTED_HASH = hash(LEFT)
EXPECTED_STR = str(LEFT)

ROUNDS = 200


def check_once():
    assert LEFT / RIGHT == EXPECTED_TRUEDIV
    assert LEFT // RIGHT == EXPECTED_FLOORDIV
    assert LEFT % RIGHT == EXPECTED_MOD
    assert divmod(LEFT, RIGHT) == EXPECTED_DIVMOD
    assert pow(LEFT, 5, RIGHT) == EXPECTED_POWMOD
    assert (LEFT << 7) == EXPECTED_LSHIFT
    assert (LEFT >> 7) == EXPECTED_RSHIFT
    assert float(LEFT) == EXPECTED_FLOAT
    assert hash(LEFT) == EXPECTED_HASH
    assert str(LEFT) == EXPECTED_STR


i = 0
while i < ROUNDS:
    check_once()
    # Force nursery traffic between iterations so a missed root is not
    # hidden by a still-live bump pointer.
    _ = [0] * 32
    i += 1


def f_truediv_long(n):
    big = 1 << 200
    s = 0.0
    i = 1
    while i <= n:
        s += big / (big + i)
        i += 1
    return s


TRUEDIV_N = 5000
EXPECTED_TRUEDIV_CHECKSUM = 5000000000
repeat = 0
while repeat < 8:
    got = int(round(f_truediv_long(TRUEDIV_N) * 1e6))
    assert got == EXPECTED_TRUEDIV_CHECKSUM, (repeat, got)
    repeat += 1


# Machine-int modulus: int_pow's loop must reread the base guard, not a
# `&RBigInt` taken once before the collecting `_help_mult`s.
BIG = (1 << 200) + 12345
EXPECTED_POWMOD_INT = 659451
i = 0
while i < ROUNDS:
    assert pow(BIG, 3, 1000003) == EXPECTED_POWMOD_INT
    _ = [0] * 32
    i += 1

print("OK")
