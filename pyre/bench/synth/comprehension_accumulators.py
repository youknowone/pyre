N = 100000


def main():
    # SET_ADD (set comprehension) and MAP_ADD (dict comprehension) are
    # lowered but latent — the comprehension FOR_ITER loop never gets a JIT
    # token (a separate no-token cliff), so no demonstrable speedup; this
    # bench guards their output correctness.  DICT_MERGE via `f(**a, **b)`.
    s = {x % 7 for x in range(N)}
    d = {x: x * 2 for x in range(N // 100)}

    def f(**kw):
        return sum(kw.values())

    m = f(**{"a": 1, "b": 2}, **{"c": 3})
    print(len(s), len(d), d[N // 100 - 1], m)


main()
