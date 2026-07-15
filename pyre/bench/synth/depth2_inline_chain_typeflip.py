N = 300000
FLIP_AT = 200000


def h(x):
    if x < 3:
        return x + 1
    return x * 2


def g(x):
    return h(x) + 1


def main():
    acc = 0.0
    i = 0
    while i < N:
        v = (i % 5) if i < FLIP_AT else float(i % 5)
        acc += g(v)
        i += 1
    return acc


print(main())
