# gh#495 guard: inlined subwalk with mutating residual call and no exception replay.
# same as adv_3_a but NO exception: the miss branch avoids KeyError entirely
N = 60000


class Counter:
    def __init__(self):
        self.pos = 0

    def bump(self):
        self.pos = self.pos + 1


def step(c, d, k):
    c.bump()
    if k < 0:
        return 0
    if k in d:
        return d[k]
    return -1


def run():
    d = {1: 100, 2: 200, 3: 300}
    acc = 0
    c = Counter()
    i = 0
    while i < N:
        k = i % 5
        v = step(c, d, k)
        if v == -1:
            acc -= 1
        else:
            acc += v
        i = i + 1
    return acc, c.pos


print(run())
