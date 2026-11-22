# CPython-suite gap: test_enum / test_trace reach a sort-key walk abort only
# as a leftover recursion-guard panic, never as a counted key-function call.
# parity-tests reason: a sortkey walk that leftover-executes the key function
# and then aborts must continue that iteration in the blackhole; replaying the
# portal body would run the key twice for the same item.

"""A sort key function runs once per item when the sortkey walk aborts.

`f` increments a counter, then copies the flocals write-through shape
(`sys._getframe` + `f_locals` store) that forces a residual and aborts the
sortkey walk. The abort must not re-call `f` for the same item.
"""

import sys

N = 400
counter = 0


def f(v):
    global counter
    counter += 1
    # flocals_writethrough_forcing_residual.py `setter`: FrameLocalsProxy
    # store that leftover-executes `enter_recursive_frame` then aborts.
    frame = sys._getframe(1)
    frame.f_locals["x"] = 0
    return v


x = 0
sorted(range(N), key=f)
assert counter == N, (counter, N)
print("OK")
