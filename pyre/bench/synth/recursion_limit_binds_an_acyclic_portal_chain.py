# pyre-check: selfcheck
# pyre-check: selfcheck-compiles=hot,level0,root:level0,root:level1,root:level2,root:level3,root:level4,root:level5,entry-bridge:level0
# Every Python activation consumes recursion budget, not only an activation
# whose code object is already present in the live call chain.  The six
# distinct callees below each carry a loop so a warmed caller can cross their
# entries through compiled portal/CALL_ASSEMBLER paths.
#
# The two limits pin both sides of `PyFrame.execute_frame`'s pre-entry check.
# `frame_depth` includes its own transient frame, so `base + 5` leaves six
# units from its caller: all six callees are admitted. `base + 4` leaves five,
# so the sixth must raise. Checking the charged depth with `>=` rejects the
# first case one level early, while
# charging recursive code objects only lets the second case return normally.
#
# This is the same intentional 3.14-spec split as
# `recursion_limit_binds_a_portal_driven_recursion`: measured CPython 3.14.6
# raises in the `base + 4` arm, while pypy3 7.3.22 returns. The implementation
# still follows PyPy's `PyFrame.execute_frame` activation seam; the observable
# boundary follows the pinned CPython build.
import sys

WARM = 3000
INNER = 20


def level5():
    total = 0
    for i in range(INNER):
        total += i
    return total


def level4():
    total = 0
    for i in range(INNER):
        total += i
    return total + level5()


def level3():
    total = 0
    for i in range(INNER):
        total += i
    return total + level4()


def level2():
    total = 0
    for i in range(INNER):
        total += i
    return total + level3()


def level1():
    total = 0
    for i in range(INNER):
        total += i
    return total + level2()


def level0():
    total = 0
    for i in range(INNER):
        total += i
    return total + level1()


def hot(n):
    result = 0
    for _ in range(n):
        result = level0()
    return result


def frame_depth():
    frame = sys._getframe()
    depth = 0
    while frame is not None:
        depth += 1
        frame = frame.f_back
    return depth


def main():
    expected = 6 * sum(range(INNER))
    assert hot(WARM) == expected

    base = frame_depth()
    saved = sys.getrecursionlimit()
    try:
        sys.setrecursionlimit(base + 5)
        exact = level0()
        sys.setrecursionlimit(base + 4)
        try:
            level0()
        except RecursionError:
            over = "raised"
        else:
            over = "returned"
    finally:
        sys.setrecursionlimit(saved)

    assert exact == expected, exact
    assert over == "raised", over
    print("PASS")


main()
