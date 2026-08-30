# CPython-suite gap: nothing in the suite walks `f_back` from inside a frame the
# JIT has taken over while a bridge is being traced, so a frame linked into the
# chain twice reads as a hang or a RecursionError somewhere else entirely.
# parity-tests reason: the JIT activation seam linked the frame itself
# (`install_current_frame`) and then handed the frame to the plain evaluator on
# the bridge-tracing decline, which links it a second time through
# `ExecutionContext.enter` — and the second link reads `topframeref`, which the
# first one had already set to this frame, so `f_backref` ended up naming the
# frame it is stored on.

"""A frame the portal activates is linked into the caller chain exactly once."""

import sys


def walk_back():
    """Walk `f_back` to the root; raise rather than spin if the chain loops."""
    seen = set()
    frame = sys._getframe()
    depth = 0
    while frame is not None:
        if id(frame) in seen:
            raise RuntimeError(
                "f_back cycle at %s after %d frames" % (frame.f_code.co_name, depth)
            )
        seen.add(id(frame))
        depth += 1
        frame = frame.f_back
    return depth


def recurse(n):
    if n:
        return recurse(n - 1)
    return walk_back()


# The accumulating `t +=` is load-bearing: it is what puts a guard on the loop
# body whose failure starts the bridge the decline is taken on.  A loop that
# only rebinds the result does not reach it.
def hot(n):
    t = 0
    for i in range(n):
        t += recurse(3)
    return t


total = hot(200000)
assert total > 0, total

# The same walk from a cold frame must agree with itself across repeats.
first = recurse(3)
for _ in range(5):
    again = recurse(3)
    assert again == first, (again, first)

print("OK")
