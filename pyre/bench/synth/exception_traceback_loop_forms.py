# pyre-check: max-pypy-ratio=36
# pyre-check: min-pypy-ratio=4.6
# Traceback nodes recorded by a compiled loop, driven by BOTH loop forms.
#
# The sibling exception-traceback fixtures all drive their workload with
# `for i in range(N)`. The two loop forms compile through different paths and
# have failed independently: the same callee raising into a `while` loop lost
# every node while the `for` twin lost exactly one, so a `for`-only fixture
# reports the milder symptom or none at all.
#
# Each shape reports the SET of chain depths it observed rather than one
# sample, so a compiled iteration that disagrees with the interpreted ones
# shows up as a second element instead of being averaged away. A node built on
# both sides of a frame boundary shows up the same way.
#
# The traceback walk is repeated inline in each shape on purpose: moving it
# into a helper adds a call to the loop body and changes what the body traces,
# which is enough to hide the divergence.

N = 300000


def out(key, value):
    print(f"{key} = {value}")


def thrower(i):
    raise KeyError(i)


def while_callee(n):
    depths, missing, i = set(), 0, 0
    while i < n:
        try:
            thrower(i)
        except KeyError as e:
            tb = e.__traceback__
            if tb is None:
                missing += 1
            else:
                d = 0
                while tb is not None:
                    d += 1
                    tb = tb.tb_next
                depths.add(d)
        i += 1
    return sorted(depths), missing


def for_callee(n):
    depths, missing = set(), 0
    for i in range(n):
        try:
            thrower(i)
        except KeyError as e:
            tb = e.__traceback__
            if tb is None:
                missing += 1
            else:
                d = 0
                while tb is not None:
                    d += 1
                    tb = tb.tb_next
                depths.add(d)
    return sorted(depths), missing


def while_same_frame(n):
    depths, missing, i = set(), 0, 0
    while i < n:
        try:
            raise ValueError(i)
        except ValueError as e:
            tb = e.__traceback__
            if tb is None:
                missing += 1
            else:
                d = 0
                while tb is not None:
                    d += 1
                    tb = tb.tb_next
                depths.add(d)
        i += 1
    return sorted(depths), missing


def for_same_frame(n):
    depths, missing = set(), 0
    for i in range(n):
        try:
            raise ValueError(i)
        except ValueError as e:
            tb = e.__traceback__
            if tb is None:
                missing += 1
            else:
                d = 0
                while tb is not None:
                    d += 1
                    tb = tb.tb_next
                depths.add(d)
    return sorted(depths), missing


def while_residual_raise(n):
    """A builtin operation raises for the frame -- no `raise` statement."""
    depths, missing, i, zero = set(), 0, 0, 0
    while i < n:
        try:
            _ = i // zero
        except ZeroDivisionError as e:
            tb = e.__traceback__
            if tb is None:
                missing += 1
            else:
                d = 0
                while tb is not None:
                    d += 1
                    tb = tb.tb_next
                depths.add(d)
        i += 1
    return sorted(depths), missing


def while_bare_reraise(n):
    depths, missing, i = set(), 0, 0
    while i < n:
        try:
            try:
                raise ValueError(i)
            except ValueError:
                raise
        except ValueError as e:
            tb = e.__traceback__
            if tb is None:
                missing += 1
            else:
                d = 0
                while tb is not None:
                    d += 1
                    tb = tb.tb_next
                depths.add(d)
        i += 1
    return sorted(depths), missing


def while_innermost_lineno(n):
    """The innermost node's line is the raising line, not the helper's `def`."""
    linenos, missing, i = set(), 0, 0
    while i < n:
        try:
            thrower(i)
        except KeyError as e:
            tb = e.__traceback__
            if tb is None:
                missing += 1
            else:
                while tb.tb_next is not None:
                    tb = tb.tb_next
                linenos.add(tb.tb_lineno)
        i += 1
    return sorted(linenos), missing


out("while_callee_tb", while_callee(N))
out("for_callee_tb", for_callee(N))
out("while_same_frame_tb", while_same_frame(N))
out("for_same_frame_tb", for_same_frame(N))
out("while_residual_raise_tb", while_residual_raise(N))
out("while_bare_reraise_tb", while_bare_reraise(N))
out("while_innermost_lineno", while_innermost_lineno(N))
