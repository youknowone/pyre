# pyre-check: selfcheck
# pyre-check: selfcheck-compiles=hot
# `sys._getframe(2)` from `leaf` lands on `grand`, not `mid` and not the
# portal. lasti/lineno fold at the CALL `grand` is suspended at
# (`InlineParentFrame.caller_py_pc`).
import sys

N = 20000
ROUNDS = 4


def leaf(i):
    frame = sys._getframe(2)
    return (
        frame.f_code.co_name,
        frame.f_lasti,
        frame.f_lineno,
        frame.f_code.co_firstlineno,
    )


def mid(i):
    return leaf(i)


def grand(i):
    return mid(i)


def hot(n):
    rows = {}
    i = 0
    while i < n:
        row = grand(i)
        rows[row] = rows.get(row, 0) + 1
        i += 1
    return rows


def main():
    cold_rows = hot(1)
    if len(cold_rows) != 1:
        print("FAIL cold rows:", sorted(cold_rows.items()))
        return 1
    cold_row = next(iter(cold_rows))
    if cold_row[0] != "grand":
        print("FAIL expected grand caller, got:", cold_row)
        return 1

    observed = {}
    for _ in range(ROUNDS):
        for row, count in hot(N).items():
            observed[row] = observed.get(row, 0) + count

    expected = {cold_row: N * ROUNDS}
    if observed != expected:
        print("FAIL grand frame image from inlined leaf")
        print("expected:", sorted(expected.items()))
        print("observed:", sorted(observed.items()))
        return 1

    print("PASS grand frame image from inlined leaf")
    return 0


sys.exit(main())
