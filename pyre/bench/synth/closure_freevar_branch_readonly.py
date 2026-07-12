# gh#498 guard: branch resume keeps the current closure-call result local.
N = 40000


def run():
    base = 10
    acc = 0

    def step(k):
        if k == 2:
            return base + 190
        return -1

    i = 0
    while i < N:
        v = step(i % 5)
        acc += v if v != -1 else 0
        i += 1
    return acc


print(run())
