"""`divmod(a, b)` on two exact ints under a JIT-hot loop.

The walker recognises the `divmod` builtin and emits the guarded
`int_py_div` / `int_py_mod` pair straight into a virtual `Cls_ii` specialised
tuple, so a `q, r = divmod(...)` site keeps no tuple at all.  Everything the
recognition declines — a zero divisor, `INT_MIN // -1`, a bool / long / float
operand, an `int` subclass, a shadowed `divmod` name — has to keep producing
the interpreted answer at a site that has already been traced.
"""

ROUNDS = 3000
INT_MIN = -(1 << 63)


def floored_pairs():
    """All four sign combinations must follow Python's floored division."""
    out = []
    for _ in range(ROUNDS):
        out = []
        for a in (98765, -98765):
            for d in (60, -60, 1, -1):
                q, r = divmod(a, d)
                out.append((q, r, q * d + r == a, type(q) is int, type(r) is int))
    return out


def remainder_hits_zero():
    """The `r == 0` iterations must not diverge from the nonzero ones."""
    acc = []
    for i in range(ROUNDS):
        q, r = divmod(i, 60)
        acc.append(q * 60 + r == i)
    return all(acc), divmod(ROUNDS - 1, 60)


def tuple_is_a_real_tuple():
    """A result that is not unpacked still has to materialise correctly."""
    out = None
    for i in range(ROUNDS):
        t = divmod(i + 7, 5)
        out = (t, len(t), t[0], t[1], type(t) is tuple, t == (t[0], t[1]))
    return out


def zero_divisor_after_warmup():
    """The trace is built with a nonzero divisor, then replayed with 0."""
    acc = 0
    for i in range(ROUNDS):
        d = 3 if i < ROUNDS - 1 else 0
        try:
            q, r = divmod(i, d)
            acc += r
        except ZeroDivisionError:
            acc = -1
    return acc


def int_min_overflow():
    """`divmod(INT_MIN, -1)` leaves the machine-int domain."""
    out = None
    for i in range(ROUNDS):
        d = -1 if i == ROUNDS - 1 else 7
        out = divmod(INT_MIN, d)
    return out, INT_MIN // -1 == -INT_MIN


def bool_operands():
    """Bools are not exact ints; the divisor `False` is a zero."""
    out = None
    for _ in range(ROUNDS):
        out = (divmod(True, 2), divmod(7, True), divmod(True, True))
    try:
        divmod(5, False)
    except ZeroDivisionError:
        caught = "ZeroDivisionError"
    else:
        caught = "no raise"
    return out, caught


def mixed_operand_kinds():
    """long / float operands take other interpreter legs at the same site."""
    big = (1 << 200) + 12345
    out = None
    for _ in range(ROUNDS):
        out = (divmod(big, 97)[1], divmod(7.5, 2.0), divmod(-7.5, 2.0), divmod(big, big))
    return out


class MyInt(int):
    pass


def int_subclass_operands():
    """An int subclass must not take the exact-builtin fast path."""
    out = None
    a = MyInt(98765)
    for _ in range(ROUNDS):
        out = (divmod(a, 60), divmod(98765, MyInt(60)), type(divmod(a, 60)[0]) is int)
    return out


def subclass_overrides_divmod():
    """`__divmod__` on a subclass wins over the int leg."""

    class Weird(int):
        def __divmod__(self, other):
            return ("weird", int(self), other)

    out = None
    w = Weird(9)
    for _ in range(ROUNDS):
        out = divmod(w, 4)
    return out


def alternating_divisor_sign():
    """One site alternating between a positive and a negative divisor."""
    acc = 0
    for i in range(ROUNDS):
        d = 7 if (i & 1) else -7
        q, r = divmod(i, d)
        acc += q * d + r - i
        acc ^= r & 0xFF
    return acc


def shadowed_divmod():
    """A rebound global `divmod` must not keep the builtin's trace."""
    global divmod
    builtin_divmod = divmod
    out = []
    for i in range(ROUNDS):
        if i == ROUNDS - 1:
            divmod = lambda a, b: ("shadowed", a, b)  # noqa: E731
        out = divmod(i, 60)
    divmod = builtin_divmod
    return out


print(floored_pairs())
print(remainder_hits_zero())
print(tuple_is_a_real_tuple())
print(zero_divisor_after_warmup())
print(int_min_overflow())
print(bool_operands())
print(mixed_operand_kinds())
print(int_subclass_operands())
print(subclass_overrides_divmod())
print(alternating_divisor_sign())
print(shadowed_divmod())
print("OK")
