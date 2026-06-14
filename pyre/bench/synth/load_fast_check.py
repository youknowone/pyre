N = 1400000


def main():
    acc = 0
    i = 0
    while i < N:
        # `i >= 0` is always true, so `x` is bound on every iteration, but
        # the compiler's definite-assignment analysis cannot prove the branch
        # is always taken — it emits LOAD_FAST_CHECK for the `acc + x` read.
        if i >= 0:
            x = i
        acc = acc + x
        i = i + 1
    print(acc)


main()
