# pyre-check: max-pypy-ratio=6
# Negative and invert are emitted as canonical codewriter `inline_call`s and
# carry no residual-call fold gate.
# Unary operations must observe the current loop-carried integer. Exercise
# both ordinary values and the large-integer boundary, plus neighboring
# operations that serve as controls. Deterministic.
# A hot `~i` loop proves that generation now reaches the interpreter body's
# `IntInvert` directly instead of leaving a `CallMayForce` residual.


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
    """Hot `~int`, served by the interpreter-body invert descent."""
    s = 0
    i = 0
    while i < n:
        s += ~i
        i += 1
    return s


print(hot_invert(30000000))
