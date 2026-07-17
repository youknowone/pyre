# Regression guard for P2 root result seeding when the residual-call result is
# stored into a root local slot.  The late type flip forces a guard failure in
# the inlined callee chain; the root stores the returned value in `x` before
# using it so the P2 drain must keep the local-slot result live for the bridge
# walk.
N = 300000
FLIP_AT = 200000


def leaf(x):
    if x < 3:
        return x + 10
    return x * 2


def middle(x):
    return leaf(x) + 1


def main():
    acc = 0.0
    i = 0
    while i < N:
        arg = i % 5
        if i >= FLIP_AT:
            arg = float(arg)
        x = middle(arg)
        acc = acc + x
        i = i + 1
    return acc


print(main())
