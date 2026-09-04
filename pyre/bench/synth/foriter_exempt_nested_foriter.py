# pyre-check: max-pypy-ratio=20
# The function-entry door reading its own cell took this off the 44 it needed
# while the door read another cell's answer, asked to trace at every call and
# never entered the compiled loop.
#
# The ceiling is NOT the measured ratio.  pypy's execution-only time here lands
# either side of EXEC_TIME_FLOOR_S, and check.py gates the ratio whenever it
# lands above (`?`) and skips it whenever it is clamped to the floor (`~`), so
# the same binary reads 17.7x on one runner and 27.9x on the next.  Size the
# ceiling for the worst denominator in the gated band instead: dynasm's
# execution-only time over `2 * EXEC_TIME_FLOOR_S` -- the floor plus the grace
# `_compare_buffer` adds for a floor-sized baseline -- which is 0.14s / 0.01s,
# plus room for the run-to-run spread of that numerator.
# gh#495 guard: fbw_abort_nested_unjournaled_residual prevents the ForIterNext exemption double-advance.
# branch-bearing callee with a SECOND FOR_ITER (nested), not the loop header.
# Two shared generators; inner FOR_ITER advance is a non-header foriter (Finding #2).
# Post-inner declining residual forces abort while inner item in-flight.
N = 20000


class Shared:
    def __init__(self):
        self.a = 0
        self.b = 0
        self.t = 0


def gouter(sh, m):
    j = 0
    while j < m:
        sh.a += 1
        yield j
        j += 1


def ginner(sh, m):
    j = 0
    while j < m:
        sh.b += 1
        yield j * 10
        j += 1


def tail(sh):
    sh.t += 1
    return sh.t


def step(go, gi, sh, k):
    if k < 0:
        return 0
    s = 0
    for x in go:
        s += x
        for y in gi:
            s += y
            break
        t = tail(sh)
        s += t & 0
        break
    return s


def run(N):
    sh = Shared()
    go = gouter(sh, N * 10)
    gi = ginner(sh, N * 10)
    acc = 0
    i = 0
    while i < N:
        k = i % 5
        acc += step(go, gi, sh, k)
        i += 1
    return acc, sh.a, sh.b, sh.t


print(run(N))
