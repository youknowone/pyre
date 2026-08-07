# Locals of an inlined callee must survive a guard that fails inside the
# callee's own body.  A callee's `LOAD_FAST`/`STORE_FAST` lower to
# `getarrayitem_vable_*`/`setarrayitem_vable_*` on its own frame array, and the
# split between the `locals_cells_stack_w` prefix and the operand-stack region
# is that frame's own local count.  Both shapes are covered: a callee with MORE
# locals than its caller (a `STORE_FAST` mistaken for an operand-stack push
# folds away, and the resumed frame reads the slot back as NULL) and one with
# FEWER (an operand-stack cell mistaken for a local).
#
# The guard is an attribute read off a receiver whose class alternates between
# rounds, so every round after the first resumes inside the callee.

N = 400
ROUNDS = 60


class Shape:
    def __init__(self, v):
        self.v = v


class Other:
    def __init__(self, v):
        self.v = v


def sink(a, b, c):
    return (a * 7 + b * 3 + c) % 1000003


def wide_callee(o, k):
    # Six locals — more than `narrow_driver` has.  `t0`/`t2` are stored before
    # the guarded `o.v` read and consumed after it.
    t0 = k * 2 + 1
    t1 = t0 + 5
    t2 = t1 * 3
    v = o.v
    return sink(t0, t2, v)


def narrow_driver(n, o):
    acc = 0
    for i in range(n):
        acc = (acc + wide_callee(o, i)) % 1000003
    return acc


def narrow_callee(o):
    # Two locals — fewer than `wide_driver` has.
    v = o.v
    return (v, v + 1, v + 2)


def wide_driver(n, o):
    a = 0
    b = 1
    c = 2
    d = 3
    e = 4
    f = 5
    g = 6
    acc = 0
    for i in range(n):
        x, y, z = narrow_callee(o)
        acc = (acc + x + y + z + a + b + c + d + e + f + g + i) % 1000003
    return acc


# sum over i in 0..N-1 of (32 * i + 72), taken mod 1000003
NARROW_EXPECTED = (32 * (N * (N - 1) // 2) + 72 * N) % 1000003
# per iteration: (11 + 12 + 13) + (0 + 1 + ... + 6) + i
WIDE_EXPECTED = ((36 + 21) * N + N * (N - 1) // 2) % 1000003

warm = Shape(11)
flip = Other(11)

for round_ in range(ROUNDS):
    for receiver in (warm, flip):
        assert narrow_driver(N, receiver) == NARROW_EXPECTED
        assert wide_driver(N, receiver) == WIDE_EXPECTED

print("OK")
