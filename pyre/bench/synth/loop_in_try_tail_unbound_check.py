# pyre-check: max-pypy-ratio=23
# pyre-check: min-pypy-ratio=2.45
# A hot loop whose header is covered by an exception handler, with a post-loop
# tail that reads the loop variable through LOAD_FAST_CHECK.
#
# The two halves are structurally coupled, which is what makes this shape worth
# pinning.  Putting the loop inside a `try` is exactly what forces the compiler
# to spell the tail's read of `i` as LOAD_FAST_CHECK instead of LOAD_FAST: a
# raise could skip `i = 0`, so the slot is only conditionally bound on the join.
# The walker has no SSA value for a conditionally-bound slot, so LOAD_FAST_CHECK
# splits — the bound arm compiles, and the null arm emits `abort_permanent` and
# dead-ends.  That marker is correct and pc-scoped; it is reached only when the
# local really is unbound.
#
# `unbound_tail` is the arm that reaches it: the raise happens before `i = 0`,
# so the tail's read must raise UnboundLocalError.  `bound_tail` and
# `raised_tail` are the arms that must NOT: the loop ran, `i` is bound, and the
# tail prints it exactly once.  A tail that runs twice, exits early, or reads a
# mis-seeded `i` shows up as a different tuple.
#
# Only the exception type is printed, never the UnboundLocalError message: the
# pypy leg check.py compares against spells it differently from 3.14.
#
# The ratio gate is loose because it is sizing a decline, not a code
# generator: this loop does not compile today, so pyre runs it interpreted
# against a pypy that traces it.  It tightens on its own if the decline is
# ever lifted.
N = 40000
R = 20


def bound_tail(n):
    total = 0
    log = []
    try:
        i = 0
        while i < n:
            total += i
            i += 1
    except ValueError as e:
        log.append(("caught", str(e)))
    log.append(("tail", total, i))
    return log


def raised_tail(n):
    total = 0
    log = []
    try:
        i = 0
        while i < n:
            total += i
            if i == n - 1:
                raise ValueError(i)
            i += 1
    except ValueError as e:
        log.append(("caught", str(e), total))
    log.append(("tail", total, i))
    return log


def unbound_tail(n, boom):
    total = 0
    log = []
    try:
        if boom:
            raise ValueError("early")
        i = 0
        while i < n:
            total += i
            i += 1
    except ValueError as e:
        log.append(("caught", str(e)))
    try:
        log.append(("tail", total, i))
    except UnboundLocalError as e:
        log.append(("unbound", type(e).__name__))
    return log


def survey(name, fn):
    seen = set()
    for _ in range(R):
        seen.add(tuple(fn(N)))
    print(name, sorted(seen))


def main():
    survey("bound_tail  ", bound_tail)
    survey("raised_tail ", raised_tail)
    survey("unbound_hot ", lambda n: unbound_tail(n, False))
    survey("unbound_cold", lambda n: unbound_tail(n, True))


main()
