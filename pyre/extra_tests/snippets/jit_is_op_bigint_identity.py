# pyre-check: gate=1
"""`is` between two BigInt-backed ints, in a loop long enough to compile.

Whether two equal-valued bigints are one object is an interpreter's own
choice, so the oracle is the interpreter: the answer is taken once on a cold
path and the hot loop must agree with it every iteration.

What can break that agreement is the layout: the `IS_OP` fold guards the
object's layout and then emits a pointer compare, which is only sound where
no equal-valued pair of that layout is `is`.  A bigint has a layout of its
own, so a fold that reads only the machine-word integer layout as
value-comparing emits the pointer compare here and loses precisely the equal
half of the loop.
"""

BIG = 1 << 70
ROUNDS = 4000

# Cold, single-shot, and therefore never compiled: this is the interpreter's
# own answer for the pair the loop below builds.
EQUAL_VALUES_ARE_ONE_OBJECT = (BIG + 1) is (BIG + 1)


def equal_half(rounds):
    other = BIG + 1
    hits = 0
    i = 0
    while i < rounds:
        this = BIG + (i & 1)
        if this is other:
            hits += 1
        i += 1
    return hits


def unequal_half(rounds):
    other = BIG + 1
    misses = 0
    i = 0
    while i < rounds:
        this = BIG + (i & 1)
        if this is not other:
            misses += 1
        i += 1
    return misses


expected_hits = ROUNDS // 2 if EQUAL_VALUES_ARE_ONE_OBJECT else 0
assert equal_half(ROUNDS) == expected_hits, (equal_half(ROUNDS), expected_hits)
assert unequal_half(ROUNDS) == ROUNDS - expected_hits, (
    unequal_half(ROUNDS),
    ROUNDS - expected_hits,
)
