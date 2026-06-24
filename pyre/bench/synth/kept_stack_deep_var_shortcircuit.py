# Deep operand-stack Variables kept across a short-circuit guard.
# `g(i)` and `h(i)` are computed Variables held deep on the stack while the
# `p or q` short-circuit guard decides the third tuple element. A guard resume
# must restore the deep-stack g(i)/h(i).
def g(i):
    return i * 2 + 1


def h(i):
    return i * 3 - 1


def f(n):
    s = 0
    for i in range(n):
        p = (i % 4) != 0
        q = (i % 5) != 0
        t = (g(i), h(i), (g(i) if (p or q) else h(i)))
        s += t[0] - t[1] + t[2]
    return s


print(f(40000))
