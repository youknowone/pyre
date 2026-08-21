# pyre-check: max-pypy-ratio=4
# The ceiling sits between the two measured states: folded this runs 1.4x
# pypy, and with `builtin_divmod` / `binary_op_long_int_div` suppressed
# about 6.2x.
# `divmod(long, int)` at a traced call site. `_int_divmod` keeps the divisor
# unwrapped and calls `rbigint.int_divmod`, whose rtyped return is a two-item
# `GcStruct`: the walker emits one elidable call plus two `getfield_gc_r`, then
# boxes both halves and builds the specialised object pair. That is three GC
# allocations per iteration, all inside the loop — an operand varies at every
# site so the elidable cannot be hoisted out, and the halves stay live across
# the following iteration's allocations.
# Two hot loops ride along so the ratio also answers for the folds behind
# them: `builtin_divmod` on two exact ints (11.7x on its own, 0.100s ->
# 1.169s) and `binary_op_long_int_div`, the bigint `//`/`%` by a machine int
# (2.2x, 0.255s -> 0.566s -- the thinnest margin of the group; loosen the
# ceiling rather than delete the loop if a slower host proves it flaky).


BIG = (1 << 200) + 12345
N = 250000


def unpacked(n):
    """Both halves are consumed at once, so the pair itself never escapes."""
    acc = 0
    for i in range(n):
        q, r = divmod(BIG, 97 + (i & 7))
        acc = (acc + r) ^ (q & 0xFFFF)
    return acc


def held_across_calls(n):
    """The quotient outlives the next iteration's three allocations."""
    prev = 0
    acc = 0
    for i in range(n):
        q, r = divmod(BIG + i, 1000003)
        acc += (prev ^ q) & 0xFFFFFF
        prev = q + r
    return acc % 1000000007


def pair_escapes(n):
    """The tuple is read as a tuple, so the object pair is materialised."""
    out = None
    acc = 0
    for i in range(n):
        t = divmod(BIG, 3 + (i & 15))
        acc = (acc + len(t) + (t[1] & 7)) % 999983
        out = t
    return acc, out[0] * (3 + ((n - 1) & 15)) + out[1] == BIG


def negative_divisor(n):
    """`othersign == -1 and selfsign != othersign` sends `int_divmod` past
    `_divrem1` to the full `divmod` against a converted operand, so the same
    residual returns its pair from the other leg of its own body."""
    acc = 0
    for i in range(n):
        q, r = divmod(BIG, -(97 + (i & 7)))
        acc = (acc + r + (q & 0xFFFF)) & 0xFFFFFFFF
    return acc


print(unpacked(N))
print(held_across_calls(N))
print(pair_escapes(N))
print(negative_divisor(N))


def hot_divmod_int(n):
    """`divmod` on two exact ints, the pair consumed by UNPACK."""
    s = 0
    i = 1
    while i <= n:
        q, r = divmod(i, 7)
        s += q + r
        i += 1
    return s


def hot_long_int_floordiv_mod(n):
    """Bigint `//` and `%` by a machine int, the `_int_floordiv` / `_int_mod`
    family."""
    big = 1 << 200
    s = 0
    i = 1
    while i <= n:
        s += (big + i) // 1000003 % 7 + (big + i) % 1000003 % 7
        i += 1
    return s


print(hot_divmod_int(20000000))
print(hot_long_int_floordiv_mod(6000000))
