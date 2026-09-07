# CPython-suite gap: a frame returned after a compiled loop exit must be
# finished and clearable after resuming normal execution.
# parity-tests reason: cover frame lifecycle across a JIT loop exit.

import sys


def frame_after_loop(n):
    frame = sys._getframe()
    total = 0
    for i in range(n):
        total += i
    return frame


for _ in range(30):
    frame = frame_after_loop(500)
    assert frame.f_code is frame_after_loop.__code__
    assert frame.f_locals["total"] == 124750
    frame.clear()
print("OK")
