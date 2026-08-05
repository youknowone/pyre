# pyre-check: max-pypy-ratio=8
# Sized for headroom, not sensitivity: the worst ratio measured across
# platforms is 3.7x (macOS cranelift; macOS dynasm 2.9x, linux/aarch64 2.4x
# dynasm and 3.2x cranelift), and CI runners swing this fixture by ~20% under
# load. Losing either fold this file covers also moves `guard_failures`, which
# the jit-stats baselines gate in both directions, so that is the sensitive
# guard and this one is the net beneath it.
# `divmod(long, int)` at a traced call site. `_int_divmod` keeps the divisor
# unwrapped and calls `rbigint.int_divmod`, whose rtyped return is a two-item
# `GcStruct`: the walker emits one elidable call plus two `getfield_gc_r`, then
# boxes both halves and builds the specialised object pair. That is three GC
# allocations per iteration, all inside the loop — an operand varies at every
# site so the elidable cannot be hoisted out, and the halves stay live across
# the following iteration's allocations.

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
