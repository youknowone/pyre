"""`long ** int` under a JIT-hot loop.

`descr_pow` keeps a `W_IntObject` exponent unwrapped and calls
`rbigint.int_pow`, but only after four short-circuits: a negative exponent
goes to the float path, a zero exponent returns 1, and a base of 0 / 1 / -1
returns a constant. The walker specialises the remaining case, so each of the
short-circuits has to keep working at a site that has already been traced.
"""

ROUNDS = 3000
BIG = (1 << 200) + 12345
NEG = -BIG


def positive_exponents():
    out = None
    for _ in range(ROUNDS):
        out = (BIG**1, BIG**2, BIG**3, NEG**2, NEG**3)
    return [x % (10**30) for x in out]


def zero_and_negative():
    out = None
    for _ in range(ROUNDS):
        out = (BIG**0, NEG**0, type(BIG**0) is int)
    neg = BIG**-1
    return out, type(neg) is float, round(neg, 70) == 0.0


def small_bases():
    """A long whose payload fits a machine word must not take the fast arm."""
    out = None
    tiny = (1 << 70) >> 70
    for _ in range(ROUNDS):
        out = (tiny**5, (tiny - 1) ** 7, (-tiny) ** 3, (-tiny) ** 4)
    return out


def bool_exponent():
    out = None
    for _ in range(ROUNDS):
        out = (BIG**True, BIG**False)
    return [x % (10**20) for x in out]


def alternating_exponent():
    """One site alternating between the fast arm and the zero short-circuit."""
    acc = 0
    for i in range(ROUNDS):
        e = 2 if (i & 1) else 0
        acc ^= (BIG**e) & 0xFFFF
    return acc


def inplace_pow():
    a = None
    for _ in range(ROUNDS):
        a = BIG
        a **= 2
    return a % (10**25)


def three_arg():
    """`pow(a, b, m)` is not a BINARY_OP and must keep its modulus."""
    out = None
    for _ in range(ROUNDS):
        out = pow(BIG, 3, 1000003)
    return out


print(positive_exponents())
print(zero_and_negative())
print(small_bases())
print(bool_exponent())
print(alternating_exponent())
print(inplace_pow())
print(three_arg())
print("OK")
