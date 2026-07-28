# pyre-check: max-pypy-ratio=12
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

N = 3000

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
