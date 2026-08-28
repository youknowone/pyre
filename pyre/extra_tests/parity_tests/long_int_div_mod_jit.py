# CPython-suite gap: arithmetic tests do not trace mixed bigint/int division paths.
# parity-tests reason: this guards pyre's JIT bigint descriptors and side exits.

"""`long // int` and `long % int` under a JIT-hot loop.

The walker specialises this operand pair apart from the bigint/bigint leg:
`_int_floordiv` keeps a long quotient while `_int_mod` returns a machine int
(`space.newint`), and the zero-divisor branch is traced as a guard rather than
raised inside the trace. Each function runs long enough to be traced, so a
divergence between the traced and interpreted shapes shows up as a wrong value,
a wrong result type, or a missing exception.
"""

ROUNDS = 3000
BIG = (1 << 200) + 12345


def floored_pairs():
    """All four sign combinations must follow Python's floored division."""
    out = []
    for _ in range(ROUNDS):
        out = []
        for a in (BIG, -BIG):
            for d in (97, -97, 1, -1):
                q = a // d
                r = a % d
                out.append((q, r, q * d + r == a, type(r) is int))
    return out


def result_types():
    """`%` yields a plain int; `//` keeps the long value exactly."""
    q = r = None
    for _ in range(ROUNDS):
        q = BIG // 7
        r = BIG % 7
    return q, r, type(q) is int, type(r) is int, q * 7 + r == BIG


def bool_operand():
    """A bool divisor unboxes through the same int path; False is a zero."""
    total = 0
    for _ in range(ROUNDS):
        total += BIG % True
        total += BIG // True & 1
    caught = []
    for expr in (lambda: BIG // False, lambda: BIG % False):
        try:
            expr()
        except ZeroDivisionError:
            caught.append("ZeroDivisionError")
        else:
            caught.append("no raise")
    return total, caught


def zero_divisor_after_warmup():
    """The trace is built with a nonzero divisor, then replayed with 0."""
    acc = 0
    for i in range(ROUNDS):
        d = 3 if i < ROUNDS - 1 else 0
        try:
            acc += BIG % d
            acc += BIG // d & 1
        except ZeroDivisionError:
            acc = -1
    return acc


def inplace_forms():
    a = b = None
    for _ in range(ROUNDS):
        a = BIG
        a //= 31
        b = BIG
        b %= 31
    return a, b, type(b) is int


def reflected_order():
    """`int // long` / `int % long` are `descr_rfloordiv` / `descr_rmod`."""
    out = None
    for _ in range(ROUNDS):
        out = (12345 // BIG, 12345 % BIG, (-12345) // BIG, (-12345) % BIG)
    return out


def quotient_fits_machine_int():
    """A quotient that fits a machine int still has to come out exact."""
    out = None
    for _ in range(ROUNDS):
        out = (BIG // BIG, BIG % BIG, (1 << 64) // 3, (1 << 64) % 3)
    return out


def alternating_divisor():
    """Alternate between a huge and a tiny quotient at one call site."""
    acc = 0
    for i in range(ROUNDS):
        d = 3 if (i & 1) else 5
        acc += BIG % d
        acc ^= (BIG // d) & 0xFF
        small = 1 << 62
        acc += small % d
        acc ^= (small // d) & 0xFF
    return acc


class MyInt(int):
    pass


def int_subclass_divisor():
    """An int subclass must not take the exact-builtin fast path."""
    out = None
    d = MyInt(97)
    for _ in range(ROUNDS):
        out = (BIG // d, BIG % d, type(BIG % d) is int)
    return out


print(floored_pairs())
print(result_types())
print(bool_operand())
print(zero_divisor_after_warmup())
print(inplace_forms())
print(reflected_order())
print(quotient_fits_machine_int())
print(alternating_divisor())
print(int_subclass_divisor())
print("OK")
