# pyre-check: max-pypy-ratio=40
N = 100000


def main():
    # SET_ADD (set comprehension) and MAP_ADD (dict comprehension) accumulate
    # into the container the preceding BUILD_SET / BUILD_MAP left on the value
    # stack; both comprehension loops take a JIT token here.  DICT_MERGE via
    # `f(**a, **b)`.
    s = {x % 7 for x in range(N)}
    d = {x: x * 2 for x in range(N // 100)}

    def f(**kw):
        return sum(kw.values())

    m = f(**{"a": 1, "b": 2}, **{"c": 3})
    print(len(s), len(d), d[N // 100 - 1], m)


main()
