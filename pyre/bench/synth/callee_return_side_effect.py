# pyre-check: max-pypy-ratio=22
# gh#495 guard: a callee's mutation must not be replayed or dropped, for a
# constant-int and for a float return value.  Each return kind keeps its own
# driver so the call stays a direct global call rather than an indirect one.
N = 30000


class C:
    def __init__(self):
        self.pos = 0


def step_int(c):
    c.pos = c.pos + 1
    return 7


def step_float(c):
    c.pos = c.pos + 1
    return 1.5


def run_int():
    c = C()
    acc = 0
    i = 0
    while i < N:
        acc = acc + step_int(c)
        i = i + 1
    return acc, c.pos


def run_float():
    c = C()
    acc = 0.0
    i = 0
    while i < N:
        acc = acc + step_float(c)
        i = i + 1
    return acc, c.pos


print(run_int())
print(run_float())
