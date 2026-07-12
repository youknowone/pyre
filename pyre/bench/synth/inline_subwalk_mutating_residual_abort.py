# gh#495 guard: inlined subwalk with mutating residual call and local abort path.
N = 60000


class Counter:
    def __init__(self):
        self.pos = 0

    def bump(self):
        self.pos = self.pos + 1


def step(c, d, k):
    c.bump()                 # unjournaled SetfieldGc via nested residual CALL
    if k < 0:                # branch to force branch-bearing (multiframe) shape
        return 0
    try:
        return d[k]          # residual dict-lookup; raises KeyError for misses
    except KeyError:
        return -1            # caught LOCALLY in the callee frame


def run():
    d = {1: 100, 2: 200, 3: 300}
    acc = 0
    c = Counter()
    i = 0
    while i < N:
        k = i % 5            # 0 and 4 miss -> KeyError caught -> -1
        v = step(c, d, k)
        if v == -1:
            acc -= 1
        else:
            acc += v
        i = i + 1
    return acc, c.pos


print(run())
