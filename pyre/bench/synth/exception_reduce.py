# pyre-check: max-pypy-ratio=32
# pyre-check: min-pypy-ratio=3.38
N = 50000


def main():
    # BaseException.__reduce__ / __setstate__ exercised under the JIT.
    # Restricted to the surface the CPython and PyPy baselines agree on:
    # the ImportError name_from and OSError filename reduce slots only
    # exist on the 3.14 baseline, so they live in standalone parity checks,
    # not in this baseline-gated bench.
    count = 0
    i = 0
    while i < N:
        r = ValueError(i).__reduce__()
        if len(r) == 2:
            count = count + 1
        i = i + 1
    print(count)
    print(ValueError('a', 'b').__reduce__())
    print(ValueError().__reduce__())
    e = ValueError('a')
    e.foo = 1
    print(e.__reduce__())
    v = ValueError('z')
    v.__setstate__({'k': 9})
    print(v.k, v.__reduce__())


main()
