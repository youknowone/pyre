# Companion to getframe_root_loop_force_while_merge,
# carrying that file's shape at a force the constant-depth
# `sys._getframe` arm DECLINES, so the machinery it documents stays
# covered now that its own call site folds.
#
# `try_walker_specialize_sys_getframe` (jitcode_dispatch/specialize.rs) takes
# depth 0 at the top walk level, where `getframe`'s answer IS the portal
# virtualizable and no force is needed, so the CALL folds here as well. The
# force this file needs comes from the `.f_locals` getset on the folded
# result: it reads the virtualizable's own fields, so it routes through
# `force_frame_before_locals_read` on the portal and escapes it.
#
# A nonzero depth does NOT serve. `getframe` forces only the frame it RETURNS,
# and one below the portal is not the traced virtualizable, so nothing escapes.
#
# The counters recorded for this file are the ones its original
# carried before the fold; a diff against them is a real change in the escape
# machinery, not in the arm.
#
# Reachability fixture: a force whose blackhole drive REACHES a jit_merge_point.
#
# Companion to getframe_root_loop_force_blackhole_crn. The walk roots at `main`,
# whose `while` loop gives the portal jitcode a merge point, so driving the
# latched image from just past the forcing residual runs to the back edge and
# hands back ContinueRunningNormally rather than a frame terminal.
#
# Every getframe_* fixture that predates this one forces inside a short leaf
# callee whose jitcode has nothing after the residual but a return, so all of
# them ended in DoneWithThisFrameRef and the CRN arm -- the only arm that can
# reject the image -- was never exercised at all.
import sys
_gf = sys._getframe
kept = None
def main():
    global kept
    total = 0
    i = 0
    while i < 30000:
        kept = _gf(0)
        total = total + len(kept.f_locals)
        i = i + 1
    return total
t = main()
print(t, kept.f_code.co_name)
