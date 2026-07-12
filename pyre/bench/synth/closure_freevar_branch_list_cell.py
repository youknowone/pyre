# gh#498 guard: branch resume keeps local result across list-cell mutation.
N = 40000


def run():
    buf = []
    acc = 0

    def step(k):
        buf.append(k)
        if k < 0:
            return 0
        if k == 2:
            return 200
        return -1

    i = 0
    while i < N:
        v = step(i % 5)
        acc += v if v != -1 else 0
        i += 1
    return acc, len(buf)


print(run())
