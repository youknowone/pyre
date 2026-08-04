# pyre-check: max-pypy-ratio=600
# A hot loop inside a `try` that raises out of the loop into a handler the SAME
# frame owns, across the delivery flavours that reach the handler differently:
# an operation that raises from a builtin container, an int-specialized
# arithmetic guard, an inner `try` whose handler does not match, and a raise
# from an inlined callee.
#
# Each shape's post-loop tail reads the loop variable, so it is spelled
# LOAD_FAST_CHECK — the slot is only conditionally bound, because the raise can
# reach the handler before the loop ever assigns it.  The handler appends to a
# list the tail then reads, so a tail that runs twice, or a handler entered
# twice, is a longer list rather than a silent duplicate.
#
# Surveying over R repeats is what makes this catch anything: the pre-compile
# repeats are correct by construction, so a single sample proves nothing and a
# miscount shows up only as a SECOND tuple in the shape set.
#
# The ratio gate is loose because it is measuring a decline, not a code
# generator: none of these loops compiles today, so pyre runs them interpreted
# against a pypy that traces them.  It tightens on its own if the decline is
# ever lifted.
N = 25000
R = 20


def from_builtin_container(n):
    seen, total, d = [], 0, {}
    try:
        i = 0
        while i < n:
            total += i
            if i == n - 1:
                total += d["missing"]
            i += 1
    except KeyError as e:
        seen.append(("caught", str(e)))
    seen.append(("tail", total, i))
    return seen


def from_int_guard(n):
    seen, total = [], 0
    try:
        i = 0
        while i < n:
            total += i
            total += i // (n - 1 - i)
            i += 1
    except ZeroDivisionError:
        seen.append("caught")
    seen.append(("tail", total, i))
    return seen


def past_inner_handler(n):
    seen, total = [], 0
    try:
        i = 0
        while i < n:
            try:
                total += i
                if i == n - 1:
                    raise ValueError(i)
            except TypeError:
                seen.append("wrong-handler")
            i += 1
    except ValueError as e:
        seen.append(("caught", str(e)))
    seen.append(("tail", total, i))
    return seen


def raiser(i, limit):
    if i == limit:
        raise ValueError(i)
    return i


def from_inlined_callee(n):
    seen, total = [], 0
    try:
        i = 0
        while i < n:
            total += raiser(i, n - 1)
            i += 1
    except ValueError as e:
        seen.append(("caught", str(e)))
    seen.append(("tail", total, i))
    return seen


def survey(name, fn):
    shapes = set()
    for _ in range(R):
        shapes.add(tuple(fn(N)))
    print(name, sorted(shapes))


def main():
    survey("builtin_container", from_builtin_container)
    survey("int_guard        ", from_int_guard)
    survey("past_inner       ", past_inner_handler)
    survey("inlined_callee   ", from_inlined_callee)


main()
