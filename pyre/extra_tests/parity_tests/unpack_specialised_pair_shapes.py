# Arity-2 unpacks across every makespecialisedtuple2 arm, plus the shapes the
# fold must decline.
BIG = (1 << 4095) + (1 << 2001) + 0x123456789


def oo_longs(rounds):
    acc = 0
    for i in range(rounds):
        q, r = divmod(BIG + i, (1 << 755) + 0xABCDEF)
        acc ^= (q ^ r) & 0xFFFF
    return acc


def oo_strings(rounds):
    acc = 0
    for i in range(rounds):
        a, b = ("left", "right")
        acc += len(a) + len(b) + (i & 1)
    return acc


def oo_mixed(rounds):
    acc = 0
    for i in range(rounds):
        a, b = (i, BIG)
        acc ^= (a ^ (b & 0xFF))
    return acc


def ii_ints(rounds):
    acc = 0
    for i in range(rounds):
        q, r = divmod(i + 7919, 97)
        acc ^= q ^ r
    return acc


def ff_floats(rounds):
    acc = 0.0
    for i in range(rounds):
        a, b = (1.5, 2.5)
        acc += a * b + i
    return acc


def arity3(rounds):
    acc = 0
    for i in range(rounds):
        a, b, c = (i, i + 1, BIG)
        acc ^= a ^ b ^ (c & 0xFF)
    return acc


def from_list(rounds):
    acc = 0
    for i in range(rounds):
        a, b = [i, BIG]
        acc ^= a ^ (b & 0xFF)
    return acc


def wrong_arity_raises(rounds):
    seen = 0
    for i in range(rounds):
        try:
            a, b = (1, 2, 3)
        except ValueError:
            seen += 1
    return seen


def alternating(rounds):
    # The same UNPACK_SEQUENCE site sees ii then oo, so the class guard must
    # deopt rather than read oo fields off an ii layout.
    acc = 0
    for i in range(rounds):
        pair = (i, i + 1) if i % 2 else (BIG, "s")
        a, b = pair
        acc ^= (a & 0xFF) if isinstance(a, int) else 0
    return acc


for fn in (oo_longs, oo_strings, oo_mixed, ii_ints, ff_floats, arity3,
           from_list, wrong_arity_raises, alternating):
    print(fn.__name__, fn(3000))
print("OK")
