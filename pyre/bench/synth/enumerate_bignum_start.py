# enumerate(iterable, start) accepts an arbitrary-precision start past i64,
# activating the bigint index slot instead of raising OverflowError. Output
# verified against CPython/PyPy.
N = 20000
BIG = 2**63


def main():
    last = None
    for _ in range(N):
        last = list(enumerate(["a", "b", "c"], BIG))
    print(last)


main()
