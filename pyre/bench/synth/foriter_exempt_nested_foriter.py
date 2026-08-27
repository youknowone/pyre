# pyre-check: max-pypy-ratio=10
# Measured ~9.0x on dynasm once the function-entry door resolved its
# bucket hash to a cell key: before that the door read another cell's answer,
# asked to trace at every call and never entered the compiled loop, and the
# ceiling here was 44.  pypy's execution time is clamped to the runner's
# floor for this fixture, so check.py marks the ratio `~` and applies no gate
# to it; the ceiling records the level rather than enforcing it, and becomes
# enforceable if the fixture is ever sized past that floor.
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
