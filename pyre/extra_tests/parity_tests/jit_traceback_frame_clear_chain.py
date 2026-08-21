# CPython-suite gap: test.test_userstring reaches this shape only through a
# stdlib helper, so no suite test names the frame-clear chain directly.
# parity-tests reason: this guards the executing caller's operand stack when a
# JIT walk clears a whole traceback chain from inside a loop-bearing callee.

"""Clearing every frame of a traceback leaves the executing caller runnable.

`jit_inline_traceback_frame_clear.py` clears only the DEEPEST traceback
frame, which has finished and owns nothing live.  Walking the chain also
reaches the frame that is still executing the `except` block; `frame.clear()`
must refuse that one with RuntimeError and leave its operand stack — here the
`for` loop's iterator — untouched.
"""

import sys


def raises():
    raise ValueError("expected")


def clear_chain(tb):
    while tb is not None:
        try:
            tb.tb_frame.clear()
        except RuntimeError:
            pass
        tb = tb.tb_next


for _ in range(4_000):
    try:
        raises()
    except ValueError:
        clear_chain(sys.exc_info()[2])

print("OK")
