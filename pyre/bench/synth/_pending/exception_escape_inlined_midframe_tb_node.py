# KNOWN FAILING: an exception that escapes a compiled frame loses the traceback
# node of every INLINED intermediate frame it passed through.
#
#   pypy3 / CPython / PYRE_JIT=0:  ('<module>', 'driver', 'mid_pass', 'leaf')
#   pyre JIT, both backends:       ('<module>', 'driver', 'leaf')
#
# `driver`'s loop compiles, `mid_pass` is inlined into the trace and `leaf`
# stays residual.  When `leaf` raises, the interpreter records `leaf`'s node and
# hands the exception back to the trace, whose `GuardNoException` fails; the
# blackhole exit-guard-exception path records the ROOT frame (`driver`) and
# leaves.  `mid_pass` exists only as an inlined level of the trace, so no
# blackhole frame is ever built for it and nothing records its node.
#
# Discriminators, each verified:
#   * the walk-time path is FINE - catching the same exception inside the hot
#     loop gives ('caught_in_loop', 'mid', 'leaf') on every runtime, because
#     the walker's inline SubRaise+handler arm records the level;
#   * giving the middle function ANY try block (matching or not) restores the
#     node - the inline declines on the try-block marker, so the frame becomes
#     residual and the interpreter records it;
#   * three levels restore it too, for the same reason;
#   * PYRE_JIT=0 is clean, so it is not a bytecode or interpreter issue.
#
# Blocked on multi-frame resume data: the guard's `ResumeData.frames` is empty,
# so the blackhole has no inlined levels to unwind one frame at a time the way
# `record_caught_blackhole_traceback` already does for real frames.
N = 4000


def leaf(i, n):
    if i == n - 1:
        raise ValueError("escape")
    return i


def mid_pass(i, n):
    return leaf(i, n)


def driver(n):
    acc = 0
    i = 0
    while i < n:
        acc += mid_pass(i, n)
        i += 1
    return acc


try:
    driver(N)
except ValueError as e:
    names = []
    tb = e.__traceback__
    while tb is not None:
        names.append(tb.tb_frame.f_code.co_name)
        tb = tb.tb_next
    print(tuple(names))
