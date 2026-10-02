# pyre-check: max-pypy-ratio=5
# Fitted 2026-09-23, darwin-arm64, dynasm and cranelift, execution-only
# ratios as check.py printed them:
#   run 1, 1-min load 14.63 at start: dynasm 3.7x (0.37s vs pypy 0.11s),
#     cranelift 3.8x (0.39s vs pypy 0.11s)
#   run 2, 1-min load 31.96 at start: dynasm 2.4x (0.35s vs pypy 0.16s),
#     cranelift 2.7x (0.39s vs pypy 0.16s)
# 5 is above the slowest reading, 3.8x. 6 times the fastest, 2.4x, is 14.4,
# and the derived floor 5/6 is 0.833, under 2.4x.
# Module-scope `for i in range(N)` whose body raises, catches, and reads
# sys.exc_info() both inside and after the handler.  At module scope the loop
# variable `i` is a STORE_NAME (a global-dict residual), not a STORE_FAST frame
# local.  When an escaping residual in the handler body aborts the recording
# walk, the in-flight FOR_ITER item must still be re-delivered so the iteration
# runs exactly once: the loop-variable store re-binds the SAME re-delivered
# item, so it is not an accumulating body effect and must not refuse-drop the
# iteration.  A drop loses that iteration's `exc_info_inside` / `exc_info_after`
# increments (both would read N-k instead of N).

import sys

N = 15_000_000

exc_info_inside = 0
exc_info_after_none = 0
for i in range(N):
    try:
        raise ValueError(i)
    except ValueError:
        info = sys.exc_info()
        exc_info_inside += info[0] is ValueError and info[1].args == (i,) and info[2] is not None
    exc_info_after_none += sys.exc_info() == (None, None, None)

print("exc_info_inside", exc_info_inside)
print("exc_info_after_none", exc_info_after_none)
