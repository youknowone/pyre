# pyre-check: max-pypy-ratio=25
# pyre-check: max-wasm-ratio=9.4
# The wasm allowance is a native-side improvement, not a wasm regression:
# `linkme` has no wasm32 arm (wasm32 rejects its link section), so
# `BUILTIN_WRAPPER_DESCRIPTORS` is empty there, every builtin wrapper lookup
# ends at `no jitcode for address`, and `try_walker_inline_builtin_call`
# cannot descend at all.  `PYRE_WASM_CALL_HIST` shows what is left: 66.7% of
# this fixture's wasm residual calls are `bh_call_fn_0`, two per iteration --
# the two `sys.exc_info()` calls dynasm folds away.  A fold-based builtin
# such as `math_sqrt_hot` stays clean on wasm (2 residual calls total), so
# this is the descent path alone.
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
