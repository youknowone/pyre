# Unary operations must observe the current loop-carried integer. Exercise
# both ordinary values and the large-integer boundary, plus neighboring
# operations that serve as controls. Deterministic.


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
