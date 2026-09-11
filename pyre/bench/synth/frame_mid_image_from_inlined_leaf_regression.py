# pyre-check: selfcheck
# pyre-check: selfcheck-compiles=hot
# pyre-check: skip-backends=wasm
# wasm still prints PASS but compiles a root trace after five loop aborts,
# so it cannot declare the native `hot` loop.
# Self-checking regression guard for a mid inlined caller's frame read from
# inside its inlined leaf while the portal loop is still running.
#
# `sys._getframe(1)` lands on `mid`, not the portal.  The walk answers with
# the virtual box `walker_ec_enter` published for that ancestor, and
# `f_lasti` / `f_lineno` fold at the CALL `mid` is suspended at
# (`fbw_mode.immediate_inline_caller_py_pc`), not the portal CALL
# `inline_caller_py_pc` inherits.
import sys

N = 20000
ROUNDS = 4


def leaf(i):
    frame = sys._getframe(1)
    return (
        frame.f_code.co_name,
        frame.f_lasti,
        frame.f_lineno,
        frame.f_code.co_firstlineno,
    )


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
    if cold_row[0] != "mid":
        print("FAIL expected mid caller, got:", cold_row)
        return 1

    observed = {}
    for _ in range(ROUNDS):
        for row, count in hot(N).items():
            observed[row] = observed.get(row, 0) + count

    expected = {cold_row: N * ROUNDS}
    if observed != expected:
        print("FAIL mid frame image from inlined leaf")
        print("expected:", sorted(expected.items()))
        print("observed:", sorted(observed.items()))
        return 1

    print("PASS mid frame image from inlined leaf")
    return 0


sys.exit(main())
