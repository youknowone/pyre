# gh#498 guard: branch resume keeps local result across nonlocal mutation.
N = 40000


def run():
    n = 0
    acc = 0

    def step(k):
        nonlocal n
        n = n + 1
        if k < 0:
            return 0
        if k == 2:
            return 200
        return -1

    i = 0
    while i < N:
        k = i % 5
        v = step(k)
        acc += v if v != -1 else 0
        i += 1
    return acc, n


print(run())
