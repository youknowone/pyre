# pyre-check: selfcheck
# Self-checking regression guard for a caller frame read from inside an inlined
# callee while the caller's compiled loop is still running.
#
# The callee reads non-forcing coordinate fields from `sys._getframe(1)`.  A
# stale caller `last_instr` shows up as an extra pre-loop row during the first
# compiled survey rounds, before the final steady row hides it in a set-only
# check.
import sys

N = 20000
ROUNDS = 4


def callee_d1(i):
    frame = sys._getframe(1)
    return (
        frame.f_code.co_name,
        frame.f_lasti,
        frame.f_lineno,
        frame.f_code.co_firstlineno,
    )


def hot_d1(n):
    rows = {}
    i = 0
    while i < n:
        row = callee_d1(i)
        rows[row] = rows.get(row, 0) + 1
        i += 1
    return rows


def main():
    cold_rows = hot_d1(1)
    if len(cold_rows) != 1:
        print("FAIL cold rows:", sorted(cold_rows.items()))
        return 1
    cold_row = next(iter(cold_rows))

    observed = {}
    for _ in range(ROUNDS):
        for row, count in hot_d1(N).items():
            observed[row] = observed.get(row, 0) + count

    expected = {cold_row: N * ROUNDS}
    if observed != expected:
        print("FAIL caller frame image from inlined callee")
        print("expected:", sorted(expected.items()))
        print("observed:", sorted(observed.items()))
        return 1

    print("PASS caller frame image from inlined callee")
    return 0


sys.exit(main())
