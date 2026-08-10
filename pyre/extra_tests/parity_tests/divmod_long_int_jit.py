# CPython-suite gap: divmod tests do not trace mixed bigint/int results under GC.
# parity-tests reason: this targets pyre's JIT bigint lowering and GC-rooted pair.

"""`divmod(long, int)` under a JIT-hot loop.

`_int_divmod` keeps a `W_IntObject` divisor unwrapped and calls
`rbigint.int_divmod`, whose rtyped return is a two-item `GcStruct`.  The
walker emits that as one elidable call plus two `getfield_gc_r`, and boxes
both halves and the result pair itself, so this exercises three fresh GC
allocations per call as well as the arithmetic.  Run it under
`MAJIT_GC_NURSERY_POISON=1` (`run.py --gc-poison`) — a missing root around
those allocations shows up as a wrong digit, not a crash.

`makespecialisedtuple2` only reaches the object pair when the two halves are
not both machine-word sized, so the small-quotient cases below have to leave
the specialised arm and still come out exact.
"""

ROUNDS = 3000
BIG = (1 << 200) + 12345
NEG = -BIG


def floored_pairs():
    """All four sign combinations must follow Python's floored division."""
    out = []
    for _ in range(ROUNDS):
        out = []
        for a in (BIG, NEG):
            for d in (97, -97, 1, -1):
                q, r = divmod(a, d)
                out.append((q == a // d, r == a % d, q * d + r == a, 0 <= abs(r) < abs(d)))
    return out


def exact_values():
    q = r = None
    for _ in range(ROUNDS):
        q, r = divmod(BIG, 97)
    return q, r, type(q) is int, type(r) is int, q * 97 + r == BIG


def churn_keeps_earlier_results():
    """Hold both halves live across later allocations of the same shape.

    A quotient reclaimed while a subsequent `int_divmod` allocates would only
    be visible as a wrong sum at the end.
    """
    held = []
    total = 0
    for i in range(ROUNDS):
        q, r = divmod(BIG + i, 97)
        held.append(q)
        total += r
        if len(held) > 16:
            head = held.pop(0)
            total += head % 1000003
    return total, len(held), sum(x % 7 for x in held)


def small_quotient():
    """A quotient that fits a machine word leaves the specialised arm."""
    out = None
    small = (1 << 70) >> 70
    for _ in range(ROUNDS):
        out = (divmod(small, 3), divmod(small * 5, 7), divmod(-small, 3))
    return out


def zero_divisor_after_warmup():
    """The trace is built with a nonzero divisor, then replayed with 0."""
    acc = 0
    for i in range(ROUNDS):
        d = 3 if i < ROUNDS - 1 else 0
        try:
            q, r = divmod(BIG, d)
            acc += r
        except ZeroDivisionError:
            acc = -1
    return acc


def bool_divisor():
    out = None
    for _ in range(ROUNDS):
        out = divmod(BIG, True)
    try:
        divmod(BIG, False)
    except ZeroDivisionError:
        caught = "ZeroDivisionError"
    else:
        caught = "no raise"
    return out[0] == BIG, out[1], caught


def alternating_divisor():
    """One site alternating between a huge and a machine-sized quotient."""
    acc = 0
    tiny = (1 << 70) >> 70
    for i in range(ROUNDS):
        a = BIG if (i & 1) else tiny
        q, r = divmod(a, 5)
        acc += r
        acc ^= q & 0xFF
    return acc


def reflected_and_long_long():
    """`int // long` and `long // long` take other legs at the same site."""
    out = None
    for _ in range(ROUNDS):
        out = (divmod(12345, BIG), divmod(BIG, BIG), divmod(BIG, BIG // 7))
    return out


class MyInt(int):
    pass


def int_subclass_divisor():
    out = None
    d = MyInt(97)
    for _ in range(ROUNDS):
        out = divmod(BIG, d)
    return out[0] * 97 + out[1] == BIG, out[1]


def tuple_is_a_real_tuple():
    out = None
    for _ in range(ROUNDS):
        t = divmod(BIG, 97)
        out = (len(t), t[0] * 97 + t[1] == BIG, type(t) is tuple, t == (t[0], t[1]))
    return out


def int_min_divisor():
    """`INT_MIN` is the one divisor `int_in_valid_range` rejects.

    It leaves `_divrem1` for the full `divmod` against a converted operand, as
    does every negative divisor of a non-negative dividend.
    """
    out = None
    for _ in range(ROUNDS):
        out = divmod(BIG, -(1 << 63))
    q, r = out
    return q * -(1 << 63) + r == BIG, r


def negative_one_quotient():
    """The correction arm that turns a zero quotient into the prebuilt -1.

    Opposite signs with the dividend's magnitude below the divisor's, which
    needs a long-typed dividend whose value is small.
    """
    tiny = (1 << 70) >> 70
    out = None
    for _ in range(ROUNDS):
        out = (divmod(tiny, -7), divmod(tiny, -1), divmod(-tiny, 7))
    return out


def collects_between_halves():
    """A collection lands while the halves and their pair are being built."""
    import gc

    held = []
    for i in range(ROUNDS):
        q, r = divmod(BIG + i, 97)
        held.append((q, r))
        if len(held) > 8:
            held = held[-4:]
        if i % 256 == 0:
            gc.collect()
    return [q * 97 + r == BIG + i for i, (q, r) in enumerate(held, ROUNDS - len(held))]


def shadowed_divmod():
    """A rebound global `divmod` must not keep the builtin's trace."""
    global divmod
    builtin_divmod = divmod
    out = []
    for i in range(ROUNDS):
        if i == ROUNDS - 1:
            divmod = lambda a, b: ("shadowed", b)  # noqa: E731
        out = divmod(BIG, 97)
    divmod = builtin_divmod
    return out


print(floored_pairs())
print(exact_values())
print(churn_keeps_earlier_results())
print(small_quotient())
print(zero_divisor_after_warmup())
print(bool_divisor())
print(alternating_divisor())
print(reflected_and_long_long())
print(int_subclass_divisor())
print(tuple_is_a_real_tuple())
print(int_min_divisor())
print(negative_one_quotient())
print(collects_between_halves())
print(shadowed_divmod())
print("OK")
