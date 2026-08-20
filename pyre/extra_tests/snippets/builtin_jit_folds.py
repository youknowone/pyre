"""The builtins the JIT walker folds out of their residual call.

One file for the whole table rather than one per builtin: what is under test
is a single mechanism, and every row shares the same two obligations — the
folded answer equals the builtin's, and every operand the raw helper does not
implement still reaches the builtin.

Each case runs in a loop long enough for the loop to compile, and the value is
read inside that loop; a check that only inspects the result afterwards never
gets the fold compiled at all.  The assert is outside so the reads stay
ordinary consumers of the folded value.
"""

ROUNDS = 400


def _agrees(got, want):
    # NaN is its own witness here: the float rows decline on it, so a NaN that
    # survives means the residual answered, which is still correct.
    if isinstance(got, float) and isinstance(want, float):
        if got != got and want != want:
            return True
    return type(got) is type(want) and got == want


def _stable(fn, args):
    """`fn(*args)` agrees with itself across a compiled loop, or raises the
    same exception every time."""
    try:
        want = fn(*args)
    except Exception as exc:  # noqa: BLE001 - the raising direction is under test
        want = (type(exc), str(exc))
        bad = 0
        for _ in range(ROUNDS):
            try:
                fn(*args)
            except Exception as again:  # noqa: BLE001
                if (type(again), str(again)) != want:
                    bad += 1
            else:
                bad += 1
        assert bad == 0, (fn, args, want)
        return want
    bad = 0
    for _ in range(ROUNDS):
        if not _agrees(fn(*args), want):
            bad += 1
    assert bad == 0, (fn, args, want)
    return want


# --- hash -------------------------------------------------------------------
# The exact scalar types the raw helper answers for, plus the operands that
# send it back to the builtin.
class _Hashable:
    def __hash__(self):
        return 4242


class _StrSub(str):
    pass


class _IntSub(int):
    pass


# One NaN object, reused: its hash is the identity hash, so a fresh NaN per
# iteration would not be stable to compare against.  This is the arm the raw
# helper declines rather than answers, because reaching the identity hash means
# wrapping a fresh int under a call that is emitted as unable to collect.
_NAN = float("nan")

for _v in [0, 1, -1, 7, -7, 2**62, -(2**62), 2**70, -(2**70),
           True, False, 1.5, -0.0, 0.0, float("inf"), _NAN,
           "", "a", "abcdefgh", b"", b"a", b"abcdefgh"]:
    _stable(hash, (_v,))

assert _stable(hash, (_Hashable(),)) == 4242
assert _stable(hash, (_StrSub("abc"),)) == hash("abc")
assert _stable(hash, (_IntSub(9),)) == hash(9)
# Numeric equality still implies hash equality through the fold.
assert _stable(hash, (1,)) == _stable(hash, (1.0,)) == _stable(hash, (True,))
_stable(hash, ([1, 2],))  # unhashable: the raising direction

# --- ord --------------------------------------------------------------------
for _v in ["a", "é", "中", "\U0001f600", chr(0xD800), b"\x00", b"\xff"]:
    _stable(ord, (_v,))

_stable(ord, ("ab",))       # length != 1 raises
_stable(ord, ("",))
_stable(ord, (65,))         # not a string at all
assert _stable(ord, (_StrSub("z"),)) == ord("z")

# --- abs --------------------------------------------------------------------
for _v in [0, 1, -1, 7, -7, 2**62, -(2**62), -(2**63), 2**63, -(2**70),
           True, False, 0.0, -0.0, 1.5, -1.5,
           float("inf"), float("-inf"), float("nan")]:
    _stable(abs, (_v,))

# `-(2**63)` is exactly the int channel's decline sentinel *and* the operand
# whose absolute value does not fit a machine word; both directions leave
# through the same side exit, and the builtin promotes it to a long.
assert _stable(abs, (-(2**63),)) == 2**63
assert type(_stable(abs, (-7,))) is int
assert type(_stable(abs, (True,))) is int and _stable(abs, (True,)) == 1
assert _stable(abs, (_IntSub(-5),)) == 5


class _Abs:
    def __abs__(self):
        return "custom"


assert _stable(abs, (_Abs(),)) == "custom"

# --- min / max --------------------------------------------------------------
for _pair in [(1, 2), (2, 1), (1, 1), (-3, 3), (2**62, 2**62 + 1),
              (1.5, 2.5), (2.5, 1.5), (1.5, 1.5), (-0.0, 0.0),
              (float("inf"), 1.0), (float("nan"), 1.0), (1.0, float("nan")),
              (1, 1.0), (1.0, 1), (True, 0), (2**70, 1)]:
    _stable(min, _pair)
    _stable(max, _pair)

# A tie keeps the first argument, which is the object identity the scan order
# produces; the fold returns one of its own operands rather than a fresh box.
_a, _b = 10**3, 10**3
assert _stable(min, (_a, _b)) is _a
assert _stable(max, (_a, _b)) is _a
_x, _y = 1e3, 1e3
assert _stable(min, (_x, _y)) is _x
assert _stable(max, (_x, _y)) is _x
_stable(min, ("b", "a"))
_stable(max, ([1], [2]))
_stable(min, (1,))          # a single iterable argument, not the pair form
_stable(max, ([3, 1, 2],))


class _Cmp:
    def __init__(self, v):
        self.v = v

    def __lt__(self, other):
        return self.v < other.v

    def __gt__(self, other):
        return self.v > other.v


assert _stable(min, (_Cmp(1), _Cmp(2))).v == 1
assert _stable(max, (_Cmp(1), _Cmp(2))).v == 2

# --- rebound names ----------------------------------------------------------
# The fold keys on the wrapped builtin code, not the name it is reachable
# under, so a shadowing definition must win.
_real_abs, _real_hash, _real_min = abs, hash, min


def abs(x):  # noqa: A001 - shadowing is the point
    return "shadowed"


def hash(x):  # noqa: A001
    return -12345


def min(*a):  # noqa: A001
    return "smallest"


_bad = 0
for _ in range(ROUNDS):
    if abs(-1) != "shadowed" or hash(1) != -12345 or min(1, 2) != "smallest":
        _bad += 1
assert _bad == 0

abs, hash, min = _real_abs, _real_hash, _real_min
assert abs(-1) == 1 and min(1, 2) == 1

print("OK")
