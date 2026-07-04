N = 3000
ITERS = 2000


def run(n, extra):
    # `{"k": i, **extra}` compiles to BUILD_MAP + DICT_UPDATE in a
    # while-loop body.  Before DICT_UPDATE was lowered, its abort_permanent
    # marker declined the whole loop.
    total = 0
    i = 0
    while i < n:
        d = {"k": i, **extra}
        total += d["k"] + d["x"]
        i += 1
    return total


def main():
    extra = {"x": 7}
    total = 0
    for _ in range(ITERS):
        total += run(N, extra)
    print(total)


main()
