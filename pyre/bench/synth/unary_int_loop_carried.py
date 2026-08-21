# pyre-check: max-pypy-ratio=6
# The ceiling sits between the two measured states: folded this runs 2.5x
# pypy, and with `unary_invert_int` suppressed about 15.8x.
# Unary operations must observe the current loop-carried integer. Exercise
# both ordinary values and the large-integer boundary, plus neighboring
# operations that serve as controls. Deterministic.
# A hot `~i` loop is folded here too: without the `unary_invert_int` fold each
# iteration leaves a `CallMayForce` residual instead of an `IntInvert`, which
# measures 6.9x on its own (0.095s -> 0.653s).


def loop_carried_neg(n):
    s = 1000
    acc = 0
    for i in range(n):
        s -= 1
        acc += -s
    return s


def large_neg(n):
    t = 0
    s = -(1 << 62)
    for i in range(n):
        s += i
        t += -s
    return t


def controls(n):
    s = 1000
    acc = 0
    for i in range(n):
        s -= 1
        acc += s
        acc += 0 - s
        acc += abs(i % 17)
    return s, acc, -7


print(loop_carried_neg(30000))
print(large_neg(20000))
print(controls(30000))


def hot_invert(n):
    """Hot `~int`, the `unary_invert_int` fold."""
    s = 0
    i = 0
    while i < n:
        s += ~i
        i += 1
    return s


print(hot_invert(30000000))
