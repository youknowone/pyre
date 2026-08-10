# CPython-suite gap: tuple subscription tests lack PyPy specialised-pair variants.
# parity-tests reason: this guards pyre JIT layout guards and deoptimization.

# `len(pair)` and `pair[i]` across every makespecialisedtuple2 arm, plus the
# shapes the fold must decline: a slice key, an out-of-range index, an
# __index__ key, a mixed class site, and a tuple subclass overriding both.
#
# Every accumulator is position-sensitive (`acc * k + item`), so reading the
# wrong slot changes the printed value instead of cancelling out.
BIG = (1 << 4095) + (1 << 2001) + 0x123456789
MOD = 1000000007


def oo_longs(rounds):
    acc = 0
    for i in range(rounds):
        t = divmod(BIG + i, (1 << 755) + 0xABCDEF)
        acc = (acc * 3 + len(t) + (t[0] & 0xFFFF) * 5 + (t[1] & 0xFFFF)) % MOD
    return acc


def ii_ints(rounds):
    acc = 0
    for i in range(rounds):
        t = divmod(i + 7919, 97)
        acc = (acc * 3 + len(t) + t[0] * 5 + t[1]) % MOD
    return acc


def float_pair(rounds):
    # `w_tuple_new` sends a plain-float pair to `Cls_oo` rather than `Cls_ff`
    # so that `(x, x)` keeps the 3.14 pointer identity of `x`, so this is an
    # object pair whose slots happen to hold floats.
    acc = 0.0
    for i in range(rounds):
        t = (1.5, i * 0.25)
        acc = (acc + len(t) + t[0] * 5 + t[1]) % 65536.0
    return acc


def negative_index(rounds):
    # `getitem` adds `typelen` before the unrolled slot comparison, so -1 and
    # -2 must reach the same slots as 1 and 0.
    acc = 0
    for i in range(rounds):
        t = (i, i * 7 + 1)
        acc = (acc * 3 + t[-1] * 5 + t[-2] + t[1] * 11 + t[0]) % MOD
    return acc


def alternating_index(rounds):
    # One subscript site sees 0, 1, -1 and -2 in turn; the recorded slot is
    # pinned per trace, so the other three arrive as side exits.
    acc = 0
    for i in range(rounds):
        t = (i, BIG)
        item = t[i & 1] if i & 2 else t[(i & 1) - 2]
        acc = (acc * 3 + (item & 0xFFFF)) % MOD
    return acc


def bool_index(rounds):
    acc = 0
    for i in range(rounds):
        t = (i, i * 7 + 3)
        acc = (acc * 3 + t[True] * 5 + t[False]) % MOD
    return acc


def alternating_class(rounds):
    # The same site sees ii, ff, oo and a plain arity-3 tuple, so the class
    # guard has to deopt rather than read one layout's slots off another.
    acc = 0
    for i in range(rounds):
        kind = i & 3
        if kind == 0:
            t = (i, i + 1)
        elif kind == 1:
            t = (0.5, i * 1.5)
        elif kind == 2:
            t = (BIG + i, "s")
        else:
            t = (i, i + 1, i + 2)
        acc = (acc * 3 + len(t) + int(t[0]) + int(t[-1] != "s")) % MOD
    return acc


def slice_key(rounds):
    acc = 0
    for i in range(rounds):
        t = (i, BIG + i)
        acc = (acc * 3 + len(t[:]) + len(t[0:1]) + (t[1:][0] & 0xFFFF)) % MOD
    return acc


class Index:
    def __init__(self, value):
        self.value = value

    def __index__(self):
        return self.value


def index_protocol_key(rounds):
    acc = 0
    for i in range(rounds):
        t = (i, i * 7 + 5)
        acc = (acc * 3 + t[Index(i & 1)]) % MOD
    return acc


def out_of_range_raises(rounds):
    seen = 0
    for i in range(rounds):
        t = (i, i + 1)
        for bad in (2, -3):
            try:
                t[bad]
            except IndexError:
                seen += 1
    return seen


class Pair(tuple):
    def __len__(self):
        return 99

    def __getitem__(self, index):
        return 7


def subclass_overrides(rounds):
    # A subclass keeps the canonical layout and overrides both dunders; the
    # exact-`w_class` guard must send it to the generic residual.
    acc = 0
    for i in range(rounds):
        t = Pair((i, i + 1))
        acc = (acc * 3 + len(t) + t[0] * 5 + t[1]) % MOD
    return acc


for fn in (oo_longs, ii_ints, float_pair, negative_index, alternating_index,
           bool_index, alternating_class, slice_key, index_protocol_key,
           out_of_range_raises, subclass_overrides):
    print(fn.__name__, fn(3000))
print("OK")
