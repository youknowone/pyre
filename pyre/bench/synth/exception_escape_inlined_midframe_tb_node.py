# An exception escaping a compiled frame must keep a traceback node for every
# INLINED intermediate frame it passed through.
#
# `driver`'s loop compiles with `mid_pass` inlined into the trace and `leaf`
# residual.  When `leaf` raises, the exception surfaces at the trace's
# GuardNoException and the blackhole chain - two frames, `mid_pass` then
# `driver` - is walked looking for a handler.  That walk recorded a node only at
# the bottommost frame, so `mid_pass` was propagated over silently and vanished
# from the traceback: it has no interpreter frame of its own, the trace is its
# whole presence.
#
#   before: ('<module>', 'driver', 'leaf')
#   after:  ('<module>', 'driver', 'mid_pass', 'leaf')
#
# Discriminators, each verified against pypy3 and PYRE_JIT=0:
#   * the walk-time path was already correct - catching the same exception
#     inside the hot loop keeps every level, because a different arm records it;
#   * giving the middle function any try block, matching or not, or a bare
#     `finally`, hid the loss: the inline declines on the try-block marker, so
#     the frame becomes residual and the interpreter records it;
#   * three call levels hid it for the same reason;
#   * the middle frame had to be INLINED - a residual one was never affected.
#
# Expected output: ('<module>', 'driver', 'mid_pass', 'leaf').
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
