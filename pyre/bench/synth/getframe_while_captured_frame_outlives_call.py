# No `max-pypy-ratio`: this fixture compiles no loop -- its jitstats record
# `loops_compiled=0` -- so a pypy ratio compares two interpreters' startup
# rather than any generated code, and reads whatever the host's process
# spawn cost happens to be that run. The jitstats baselines gate it.
# The callee's frame OUTLIVES the call and its f_back is read after the loop.
#
# This is the discriminator for the root `f_backref` operand in the multi-frame
# blackhole adopt. The adopt relinks the resumed chain before running it; the
# walked frame is represented twice, by the live frame the compiled loop runs on
# and by the `snapshot_for_tracing` copy, and the snapshot is freed at the end of
# the walk. Linking the chain root to the snapshot therefore leaves a dangling
# `f_back` that only a reader outliving the walk can observe -- which is what
# `kept.f_back` below does. A run that prints the right names proves nothing
# unless the fixture actually reaches the path, so keep the `while` drive and the
# zero-argument sys._getframe (see getframe_while_inlined_callee_subwalk).
import sys

_gf = sys._getframe

kept = None


def leaf(x):
    global kept
    kept = _gf()
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
