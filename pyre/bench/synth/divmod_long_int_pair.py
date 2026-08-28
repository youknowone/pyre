# pyre-check: spec-folds=builtin_divmod,builtin_divmod_long_int,binary_op_long_int_div
# `divmod(long, int)` at a traced call site. `_int_divmod` keeps the divisor
# unwrapped and calls `rbigint.int_divmod`, whose rtyped return is a two-item
# `GcStruct`: the walker emits one elidable call plus two `getfield_gc_r`, then
# boxes both halves and builds the specialised object pair. That is three GC
# allocations per iteration, all inside the loop — an operand varies at every
# site so the elidable cannot be hoisted out, and the halves stay live across
# the following iteration's allocations.
# Two hot loops cover the sibling folds: `builtin_divmod` on exact ints and
# `binary_op_long_int_div` for bigint `//` and `%` by a machine int.  The fold
# census verifies all three directly, so the loops need only compile and check
# their results.


try:
    import pypyjit

    pypyjit.set_param("threshold=20,function_threshold=20")
except ImportError:
    pass

BIG = (1 << 200) + 12345
N = 5000


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


# Both loops remain far beyond the tracing threshold; the fold census, rather
# than low-resolution wall-clock subtraction, proves that each arm fired.
print(hot_divmod_int(100000))
print(hot_long_int_floordiv_mod(100000))
