# CPython-suite gap: test_itertools TestBasicOps.test_combinations_with_replacement
# panics in compile_bridge rather than asserting a Python result.
# parity-tests reason: a nested combinations_with_replacement helper that
# walks product and groupby must close a crossed bridge JUMP with the dest
# loop LABEL's arity (virtualizable.py get_array_length / read_boxes on the
# current frame). One fewer JUMP arg than the target LABEL is the panic.
#
# parity-env: MAJIT_STRICT=1

from itertools import combinations, combinations_with_replacement, groupby, product


def fact(n):
    r = 1
    for i in range(1, n + 1):
        r *= i
    return r


def main():
    def cwr1(iterable, r):
        pool = tuple(iterable)
        n = len(pool)
        if not n and r:
            return
        indices = [0] * r
        yield tuple(pool[i] for i in indices)
        while 1:
            for i in reversed(range(r)):
                if indices[i] != n - 1:
                    break
            else:
                return
            indices[i:] = [indices[i] + 1] * (r - i)
            yield tuple(pool[i] for i in indices)

    def cwr2(iterable, r):
        pool = tuple(iterable)
        n = len(pool)
        for indices in product(range(n), repeat=r):
            if sorted(indices) == list(indices):
                yield tuple(pool[i] for i in indices)

    def numcombs(n, r):
        if not n:
            return 0 if r else 1
        return fact(n + r - 1) / fact(r) / fact(n - 1)

    cwr = combinations_with_replacement
    for n in range(7):
        values = [5 * x - 12 for x in range(n)]
        for r in range(n + 2):
            result = list(cwr(values, r))
            assert len(result) == numcombs(n, r)
            assert len(result) == len(set(result))
            assert result == sorted(result)
            regular_combs = list(combinations(values, r))
            if n == 0 or r <= 1:
                assert result == regular_combs
            else:
                assert set(result) >= set(regular_combs)
            for c in result:
                assert len(c) == r
                noruns = [k for k, v in groupby(c)]
                assert len(noruns) == len(set(noruns))
                assert list(c) == sorted(c)
                assert all(e in values for e in c)
                assert noruns == [e for e in values if e in c]
            assert result == list(cwr1(values, r))
            assert result == list(cwr2(values, r))


main()
print("OK")
