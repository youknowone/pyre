# pyre-check: max-pypy-ratio=73
# pyre-check: min-pypy-ratio=3.22
# A hot compiled CALLEE holding a try block the exception does not stay in used
# to record its own traceback node TWICE.
#
# Every iteration of every driver raises and unwinds, so the ratio measures
# exception throughput rather than the contract under test, and it varies 2.5x
# across hosts -- hence the explicit floor beside the ceiling.
#
# N is 4000 because the witness needs the callee compiled: the duplicate node
# first appears at N=2500 and never at N=1500, and 4000 reproduces every one of
# the five shapes below with the same compiled structure N=20000 gave.
#
# `handle_operation_error` attaches one node per frame per delivery, at the
# RAISE instruction, and only then looks the handler table up -- a frame whose
# handler declines propagates with no second attach.  Once the callee ran
# compiled, the blackhole frame-exit walk recorded a node of its own at the
# position the handler dispatch stopped at.  That is a different instruction
# from the raise, so the screen that drops an already-recorded frame, which
# compared the instruction as well as the frame, let the pair through.
#
# The line numbers are load-bearing: the duplicate carries the handler's line,
# not the raise's, so a frame-name-only probe (the sibling
# `exception_escape_caller_frame_tb_node`) cannot see it.  Each shape gets its
# own driver frame, because a driver whose loop first compiled around a
# try-free callee does not reproduce it.
#
# `d_no_try` and `d_finally` are the controls -- neither reaches an `except`
# clause to stop the dispatch at, and both chains were already right.
# `d_two_excepts` pins that the duplicate carries the SECOND clause's line,
# `d_bare_reraise` covers the arm that preserves the existing chain, and
# `d_deep` pins that the raise may come from a frame below the one holding the
# try.  `d_explicit_reraise` is the guard on the screen itself: re-raising the
# caught object by name is the one shape whose frame legitimately owns two
# adjacent nodes, so a screen that dropped every repeat of an already-recorded
# frame instead of only the ones that follow it would lose the outer node here.
N = 4000


def chain(e):
    out = []
    tb = e.__traceback__
    while tb is not None:
        out.append((tb.tb_frame.f_code.co_name, tb.tb_lineno))
        tb = tb.tb_next
    return tuple(out)


def no_try(i):
    raise KeyError(i)


def nonmatching(i):
    try:
        raise KeyError(i)
    except ValueError:
        return "no"


def finally_only(i):
    try:
        raise KeyError(i)
    finally:
        pass


def two_excepts(i):
    try:
        raise KeyError(i)
    except ValueError:
        return "no"
    except IndexError:
        return "no2"


def bare_reraise(i):
    try:
        raise KeyError(i)
    except KeyError:
        raise


def explicit_reraise(i):
    try:
        raise KeyError(i)
    except KeyError as inner:
        raise inner


def deep_leaf(i):
    raise KeyError(i)


def deep(i):
    try:
        return deep_leaf(i)
    except ValueError:
        return "no"


def d_no_try(n):
    last = None
    for i in range(n):
        try:
            no_try(i)
        except KeyError as e:
            last = chain(e)
    return last


def d_nonmatching(n):
    last = None
    for i in range(n):
        try:
            nonmatching(i)
        except KeyError as e:
            last = chain(e)
    return last


def d_finally(n):
    last = None
    for i in range(n):
        try:
            finally_only(i)
        except KeyError as e:
            last = chain(e)
    return last


def d_two_excepts(n):
    last = None
    for i in range(n):
        try:
            two_excepts(i)
        except KeyError as e:
            last = chain(e)
    return last


def d_bare_reraise(n):
    last = None
    for i in range(n):
        try:
            bare_reraise(i)
        except KeyError as e:
            last = chain(e)
    return last


def d_explicit_reraise(n):
    last = None
    for i in range(n):
        try:
            explicit_reraise(i)
        except KeyError as e:
            last = chain(e)
    return last


def d_deep(n):
    last = None
    for i in range(n):
        try:
            deep(i)
        except KeyError as e:
            last = chain(e)
    return last


for driver in (
    d_no_try,
    d_nonmatching,
    d_finally,
    d_two_excepts,
    d_bare_reraise,
    d_explicit_reraise,
    d_deep,
):
    print(driver.__name__, driver(N))
