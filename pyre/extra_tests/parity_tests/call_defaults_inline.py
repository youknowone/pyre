# Argument binding for calls that leave parameters to their defaults, run hot
# enough to be traced.
#
# `w_tuple_new` routes every arity-2 tuple through `makespecialisedtuple2`, so
# `__defaults__` for a two-defaulted signature is a `W_SpecialisedTupleObject_*`
# rather than the array-backed layout — a different element read, and for the
# int pair an unboxed one.  A keyword argument can also leave a hole anywhere in
# the parameter list rather than a missing tail, so the default that fills slot
# `p` is `defs_w[p - (co_argcount - len(defs_w))]`, not "the tail of defs_w".
#
# Every case below is a fold whose result depends on each parameter, so a
# default read from the wrong slot changes the number.

N = 300


def check(got, want, label):
    assert got == want, "%s: %r != %r" % (label, got, want)


# ── one default: array-backed `defs_w` ──
def one(a, b=3):
    return a * 100 + b


# ── two defaults: the int pair, `Cls_ii`, unboxed slots ──
def two_ints(a, b=3, c=5):
    return a * 100 + b * 10 + c


# ── two defaults that are not both plain ints: `Cls_oo`, object slots ──
def two_objs(a, b="x", c=None):
    return "%d-%s-%s" % (a, b, c)


def two_bools(a, b=True, c=False):
    # `is_plain_int1` rejects bool, so this pair is `Cls_oo`, not `Cls_ii`.
    return a * 100 + (10 if b else 0) + (1 if c else 0)


# ── two float defaults: `Cls_ff`, which stays on the residual path ──
def two_floats(a, b=0.5, c=0.25):
    return a + b * 10 + c * 100


# ── three defaults: array-backed again ──
def three(a, b=3, c=5, d=7):
    return a * 1000 + b * 100 + c * 10 + d


# ── a default the callee mutates: the SAME object every call ──
def accumulating(a, acc=[]):
    acc.append(a)
    return len(acc)


# ── positional-only: a keyword may not bind `a` ──
def posonly(a, /, b=3, c=5):
    return a * 100 + b * 10 + c


for i in range(N):
    check(one(1), 103, "one/default")
    check(one(1, 4), 104, "one/positional")
    check(one(1, b=4), 104, "one/keyword")

    check(two_ints(1), 135, "ii/both defaults")
    check(two_ints(1, 4), 145, "ii/one positional")
    check(two_ints(1, 4, 6), 146, "ii/none defaulted")
    # a keyword leaving a HOLE at `b` rather than a missing tail
    check(two_ints(1, c=7), 137, "ii/hole at b")
    check(two_ints(1, b=4), 145, "ii/keyword b")
    check(two_ints(1, b=4, c=6), 146, "ii/both keyword")
    check(two_ints(1, c=6, b=4), 146, "ii/both keyword reordered")

    check(two_objs(1), "1-x-None", "oo/both defaults")
    check(two_objs(1, "y"), "1-y-None", "oo/one positional")
    check(two_objs(1, c="z"), "1-x-z", "oo/hole at b")

    check(two_bools(1), 110, "bool/both defaults")
    check(two_bools(1, c=True), 111, "bool/hole at b")

    check(two_floats(1.0), 31.0, "ff/both defaults")
    check(two_floats(1.0, c=0.5), 56.0, "ff/hole at b")

    check(three(1), 1357, "three/all defaults")
    check(three(1, d=8), 1358, "three/hole at b,c")
    check(three(1, 4, d=8), 1458, "three/hole at c")

    check(posonly(1), 135, "posonly/defaults")
    check(posonly(1, c=7), 137, "posonly/hole at b")

    check(accumulating(i), i + 1, "shared mutable default")

# A keyword aimed at a positional-only parameter is a TypeError, inlined or not.
try:
    posonly(a=1)
except TypeError:
    pass
else:
    raise AssertionError("posonly(a=1) must raise TypeError")

# Missing a parameter that has no default is a TypeError too.
def needs_two(a, b):
    return a + b


try:
    needs_two(1)
except TypeError:
    pass
else:
    raise AssertionError("needs_two(1) must raise TypeError")


# `__defaults__` replaced mid-loop must be observed: the tuple identity the
# trace pinned is gone, so binding has to re-derive which element fills which
# parameter — including when the replacement changes the tuple's length AND its
# representation (3-tuple array-backed -> 2-int pair).
def swapped(a, b=3, c=5):
    return a * 100 + b * 10 + c


seen = []
for i in range(N):
    if i == N // 3:
        swapped.__defaults__ = (4, 6)
    elif i == 2 * N // 3:
        swapped.__defaults__ = (7, 8, 9)
    seen.append(swapped(1))
check(seen[0], 135, "swap/before")
check(seen[N // 3], 146, "swap/after two-int pair")
# A defaults tuple LONGER than the parameter list keeps its tail: `def_first`
# goes negative and `b`/`c` take `defs_w[1]` / `defs_w[2]`.
check(seen[2 * N // 3], 189, "swap/after over-long tuple")

print("OK")
