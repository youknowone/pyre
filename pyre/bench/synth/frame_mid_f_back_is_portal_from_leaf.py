# pyre-check: selfcheck
# pyre-check: selfcheck-compiles=hot
# pyre-check: skip-backends=wasm
# Same wasm compile-shape skip as the mid lasti sibling.
# `sys._getframe(1).f_back` from an inlined leaf is the portal. The fold
# is `pyframe.py fget_f_back` → `getnextframe_nohidden` when the hop
# names the standard virtualizable.
import sys

N = 20000
ROUNDS = 4


def leaf(i):
    mid = sys._getframe(1)
    back = mid.f_back
    return (mid.f_code.co_name, back.f_code.co_name)


def mid(i):
    return leaf(i)


def hot(n):
    rows = {}
    i = 0
    while i < n:
        row = mid(i)
        rows[row] = rows.get(row, 0) + 1
        i += 1
    return rows


def main():
    cold_rows = hot(1)
    if len(cold_rows) != 1:
        print("FAIL cold rows:", sorted(cold_rows.items()))
        return 1
    cold_row = next(iter(cold_rows))
    if cold_row != ("mid", "hot"):
        print("FAIL expected (mid, hot), got:", cold_row)
        return 1

    observed = {}
    for _ in range(ROUNDS):
        for row, count in hot(N).items():
            observed[row] = observed.get(row, 0) + count

    expected = {cold_row: N * ROUNDS}
    if observed != expected:
        print("FAIL mid.f_back is portal from inlined leaf")
        print("expected:", sorted(expected.items()))
        print("observed:", sorted(observed.items()))
        return 1

    print("PASS mid.f_back is portal from inlined leaf")
    return 0


sys.exit(main())
