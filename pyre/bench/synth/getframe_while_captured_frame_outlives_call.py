# No `max-pypy-ratio`: the only loop here is the two-statement `while` driver,
# so a pypy ratio compares two interpreters' startup rather than any generated
# code, and reads whatever the host's process spawn cost happens to be that
# run. The jitstats baselines gate it.
# The callee's frame OUTLIVES the call and its f_back is read after the loop.
#
# This is the discriminator for the root `f_backref` operand in the multi-frame
# blackhole adopt. The adopt relinks the resumed chain before running it; the
# walked frame is represented twice, by the live frame the compiled loop runs on
# and by the `snapshot_for_tracing` copy, and the snapshot is freed at the end of
# the walk. Linking the chain root to the snapshot therefore leaves a dangling
# `f_back` that only a reader outliving the walk can observe -- which is what
# `kept.f_back` below does. A run that prints the right names proves nothing
# unless the fixture actually reaches the path, so keep the `while` drive.
#
# The escape is the SECOND getframe call, at the CALLER's depth.  The captured
# `kept = _gf()` cannot be it: `try_walker_specialize_sys_getframe` takes depth
# 0 at the top walk level, where getframe's answer IS the portal virtualizable
# and no force is needed, so that call escapes nothing -- and the depth-0
# capture is what makes `kept.f_back` name `main`, so it has to stay.  `_gf(1)`
# names a frame the specialization refuses, keeping one forcing residual per
# iteration.  Delete it and the adopt goes back to zero while the printed line
# stays right.
import sys

_gf = sys._getframe

kept = None


def leaf(x):
    global kept
    kept = _gf()
    _gf(1)
    return x + 1


def main():
    total = 0
    i = 0
    while i < 30000:
        total = leaf(total)
        i = i + 1
    return total


t = main()
print(t, kept.f_back.f_code.co_name, kept.f_code.co_name)
