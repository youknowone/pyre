# CPython-suite gap: unary tests do not trace INT_MIN promotion and subclass guards.
# parity-tests reason: this targets pyre's overflow-aware JIT integer fold.

"""`-x` on an exact int under a JIT-hot loop, across the INT_MIN promote.

`descr_neg` (intobject.py:628) answers `-x` with a machine int for every
operand except `INT_MIN`, which `_make_ovf2long` promotes to a `W_LongObject`.
The walker fold spells `-x` as `IntSubOvf(0, x)` and picks its guard from the
recorded operand -- `GUARD_NO_OVERFLOW` for a plain one, `GUARD_VALUE INT_MIN`
plus the bigint tail for the promoting one -- so each loop below pins one
direction of that split: a trace recorded on the promoting operand still has to
answer plain ones, and a trace recorded on a plain operand must not wrap
`INT_MIN` back to itself.

The unary folds read the operand's exact builtin class at record time, which
settles only the object the trace saw.  An `int` subclass keeps the builtin
`ob_type`, so the fold's `GUARD_CLASS INT` does not stop one from entering the
trace later; the subclass loops below hand the fold a plain int first and the
subclass only after the loop is hot, which is the arrival the `w_class` guard
has to side-exit.
"""

ROUNDS = 3000
INT_MIN = -9223372036854775807 - 1
INT_MAX = 9223372036854775807
TWO_63 = 9223372036854775808


def promoting_only():
    """Every iteration promotes, so the trace records the INT_MIN operand."""
    out = []
    for _ in range(ROUNDS):
        m = INT_MIN
        n = -m
        out = [n, type(n) is int, n == TWO_63, n - 1 == INT_MAX, n // 2]
    return out


def promoting_then_plain():
    """A trace recorded on INT_MIN must still answer a plain operand."""
    out = []
    for i in range(ROUNDS):
        m = INT_MIN if i < ROUNDS // 2 else 7
        n = -m
        out.append((n, type(n) is int))
    return out[0], out[-1], len(out)


def plain_then_promoting():
    """A trace recorded on a plain operand must not wrap INT_MIN."""
    seen = []
    promoted = 0
    for i in range(ROUNDS):
        m = 7 if i < ROUNDS // 2 else INT_MIN
        n = -m
        if n == TWO_63:
            promoted += 1
        seen = [n, type(n) is int]
    return seen, promoted, promoted == ROUNDS - ROUNDS // 2


def alternating():
    """Both arms live in the same trace, taken every other iteration."""
    acc = 0
    wrapped = 0
    for i in range(ROUNDS):
        m = INT_MIN if i % 2 else -1
        n = -m
        if n == TWO_63:
            acc += 1
        elif n == INT_MIN:
            wrapped += 1
    return acc, wrapped


class Derived(int):
    def __neg__(self):
        return "NEG"

    def __pos__(self):
        return "POS"

    def __invert__(self):
        return "INV"


def _late_subclass(op, plain):
    """Iterate plain ints until the loop is hot, then hand `op` a subclass.

    The arrival has to be data-driven: selecting the subclass through a branch
    (`x = plain if i < N else Derived(plain)`) side-exits on the branch guard
    before the fold is reached, so it answers correctly whether or not the
    operand's class is pinned.
    """
    items = [plain] * ROUNDS + [Derived(plain)]
    out = None
    for x in items:
        out = op(x)
    return out


def subclass_neg():
    """`-Derived(...)` after a plain-int trace must reach `Derived.__neg__`."""
    return (
        _late_subclass(lambda x: -x, 7),
        _late_subclass(lambda x: -x, INT_MIN),
    )


def subclass_pos():
    """`+Derived(...)` must reach `Derived.__pos__`, not forward the operand."""
    return _late_subclass(lambda x: +x, 7)


def subclass_invert():
    """`~Derived(...)` must reach `Derived.__invert__`."""
    return _late_subclass(lambda x: ~x, 7)


def bool_operand():
    """`-True` is `-1`; a bool operand never reaches the promote."""
    out = []
    for _ in range(ROUNDS):
        t = True
        f = False
        out = [-t, -f, type(-t) is int]
    return out


def round_trip():
    """`-(-INT_MIN)` negates the promoted long straight back."""
    out = []
    for _ in range(ROUNDS):
        m = INT_MIN
        n = -(-m)
        out = [n, n == INT_MIN, type(n) is int, n + 1 == -INT_MAX]
    return out


print(promoting_only())
print(promoting_then_plain())
print(plain_then_promoting())
print(alternating())
print(subclass_neg())
print(subclass_pos())
print(subclass_invert())
print(bool_operand())
print(round_trip())
print("OK")
