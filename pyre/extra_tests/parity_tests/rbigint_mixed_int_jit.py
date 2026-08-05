"""Mixed machine-int/rbigint paths used by PyPy's longobject descriptors."""

import gc


BIG = (1 << 4095) + (1 << 2001) + 0x123456789
MASK = (1 << 127) - 1


def arithmetic(rounds):
    x = BIG
    checksum = 0
    for i in range(rounds):
        small = (i & 255) - 127
        x = ((x + small) ^ small) | 1
        x = x - small
        checksum ^= x & MASK
    return x, checksum


x, checksum = arithmetic(20_000)
assert (x & MASK, checksum) == (4_886_718_409, 96)


# Both operand orders of the commutative int_* descriptor path.
for small in (-257, -1, 0, 1, 257):
    assert BIG + small == small + BIG
    assert BIG * small == small * BIG
    assert BIG & small == small & BIG
    assert BIG | small == small | BIG
    assert BIG ^ small == small ^ BIG
    assert BIG - small == BIG + (-small)


# Reflected comparisons select the inverse rbigint.int_<cmp> operation.
for small in (-(1 << 63), -1, 0, 1, (1 << 63) - 1):
    assert (BIG > small) and (small < BIG)
    assert (BIG >= small) and (small <= BIG)
    assert not (BIG < small) and not (small > BIG)
    assert not (BIG <= small) and not (small >= BIG)
    assert BIG != small and small != BIG
    assert not (BIG == small) and not (small == BIG)


def comparisons(rounds):
    score = 0
    for i in range(rounds):
        small = (i & 255) - 127
        score += BIG > small
        score += small < BIG
        score += BIG >= small
        score += small <= BIG
        score += BIG != small
        score += small != BIG
    return score


assert comparisons(20_000) == 120_000


def clone_prebuilt_zero(rounds):
    checksum = 0
    for i in range(rounds):
        # The long-long subtraction stays on the rbigint path even though its
        # numeric result is the translated prebuilt zero. Unary negation then
        # exercises RBigInt::clone's fresh-handle residual. Collections make a
        # stale alias or an unrooted shared digit edge fail deterministically.
        zero = BIG - BIG
        checksum += -zero
        if i & 255 == 0:
            gc.collect()
    return checksum


assert clone_prebuilt_zero(20_000) == 0


def shift_guards(rounds):
    x = -BIG
    negative_count_seen = False
    demoted = None
    for i in range(rounds):
        # The exceptional and bigint→machine-int result cases occur after the
        # hot long/int trace has formed, exercising its side-exit guards.
        count = -1 if i == rounds - 2 else i & 63
        try:
            shifted = x << count
            right_count = 5000 if i == rounds - 1 else count
            y = shifted >> right_count
        except ValueError:
            negative_count_seen = True
            continue
        if i == rounds - 1:
            demoted = y
            continue
        x ^= i
        assert y == x ^ i
    return negative_count_seen, demoted


assert shift_guards(20_000) == (True, -1)


def fitting_results(rounds):
    """Long-typed operands whose results land back inside a machine word.

    Nothing in `_make_generic_descr_binop`, `_int_lshift`/`_int_rshift` or
    `_int_floordiv` demotes such a result to a compact int, so a traced arm has
    to box it the same way the interpreter's `w_long_new` does. Every operator
    below is driven from `tiny`, which is a long by construction, while its
    result stays machine-sized; the final term keeps a non-fitting shift at the
    same site.
    """
    tiny = BIG >> 4090
    checksum = 0
    for i in range(rounds):
        small = 1 + (i & 7)
        checksum += tiny & 0xFF
        checksum += tiny | small
        checksum += tiny ^ small
        checksum += tiny + small
        checksum += tiny - small
        checksum += tiny * small
        checksum += tiny // small
        checksum += tiny << (i & 3)
        checksum += BIG >> (4000 + (i & 7))
        checksum %= 1000000007
    return checksum


assert fitting_results(20_000) == 489_509_669

print("OK")
