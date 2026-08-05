# pyre-check: max-pypy-ratio=26
# pyre-check: min-pypy-ratio=2.02
# The post-loop tail itself is the raising region, and a second hot loop lives
# after the handler.
#
# `raise_in_tail` completes its hot loop and only then raises, so the exception
# is delivered into the handler from the region the widened `abort_permanent`
# scan covers rather than from inside the loop.  `loop_after_handler` puts
# compilable work AFTER the handler, which pins that a decline aimed at the
# try-covered loop does not spill onto a sibling loop that is not covered by
# any handler.  `two_loops_one_try` keeps both loops inside the same `try`, so
# a marker found in the second must not decline the first.
#
# Every tail reads the loop variable, so it is a LOAD_FAST_CHECK read of a
# conditionally-bound slot in all three.
#
# The ratio gate is loose for the same reason as the sibling fixtures: it is
# sizing a decline, not a code generator.
N = 40000
R = 20


def raise_in_tail(n):
    seen, total = [], 0
    try:
        i = 0
        while i < n:
            total += i
            i += 1
        raise ValueError(total)
    except ValueError as e:
        seen.append(("caught", str(e)))
    seen.append(("tail", total, i))
    return seen


def loop_after_handler(n):
    seen, total = [], 0
    try:
        i = 0
        while i < n:
            total += i
            if i == n - 1:
                raise ValueError(i)
            i += 1
    except ValueError:
        seen.append(("caught", total))
    j, tail_total = 0, 0
    while j < n:
        tail_total += j
        j += 1
    seen.append(("tail", tail_total, j, i))
    return seen


def two_loops_one_try(n):
    seen, total = [], 0
    try:
        i = 0
        while i < n:
            total += i
            i += 1
        k = 0
        while k < n:
            total += k
            if k == n - 1:
                raise ValueError(k)
            k += 1
    except ValueError as e:
        seen.append(("caught", str(e)))
    seen.append(("tail", total, i, k))
    return seen


def survey(name, fn):
    shapes = set()
    for _ in range(R):
        shapes.add(tuple(fn(N)))
    print(name, sorted(shapes))


def main():
    survey("raise_in_tail    ", raise_in_tail)
    survey("loop_after_handler", loop_after_handler)
    survey("two_loops_one_try", two_loops_one_try)


main()
