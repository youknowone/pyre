# Companion to getframe_root_loop_force_blackhole_crn,
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
# Regression guard: SIGSEGV from a NULL operand-stack slot written by the
# single-frame blackhole's ContinueRunningNormally handoff.
#
# The walk roots at `main`, which HAS a Python loop, so its portal jitcode
# carries a jit_merge_point at the loop header. A sys._getframe force inside the
# loop latched a blackhole image built at the resume pc just past the residual;
# driving it ran to the loop back edge and raised ContinueRunningNormally at the
# merge point. The MIFrame is seeded from the live colors at the BUILD pc, but
# the merge point has its own live set, so a Ref color live at the merge but not
# at the build read back NULL -- and apply_blackhole_crn had no NULL guard, so it
# wrote a null into a live operand-stack slot. Resuming there faulted the
# interpreter (EXC_BAD_ACCESS at 0x0 in baseobjspace::next), exit 139, JIT-only.
#
# The walk's own flush declines exactly this case ("NULL operand-stack shadow
# slot (mid-expression)"); the blackhole path did not.
#
# Guarding it post-drive is NOT sufficient and this fixture also pins that: a
# post-drive decline discards a region the drive already executed and hands it
# back to the replay, which turned 199990000 into 200005595.
#
# This shape is also the one that exposed the two defects behind that NULL. The
# image seeded only the colors live at the build pc rather than the whole
# concrete bank; and the walk synchronized the virtualizable into the
# snapshot_for_tracing copy while the image's vable identity pointed at the live
# frame, so the drive read locals one Python iteration stale and accumulated
# total += 1039 where i was already 1040.
#
# None of that is reachable here: the CRN handoff does not rebuild the frame
# from the terminal register banks, so there is no NULL to write and nothing
# to decline. The drive runs and this file adopts it five times. Its effects are idempotent, though, which is exactly
# what hid the residual heap half of a post-drive decline — see the
# `_nonidempotent` sibling.
import sys


def main():
    total = 0
    names = set()
    for i in range(20000):
        fr = sys._getframe(0)
        names.add(len(fr.f_locals))
        total += i
    print(total, sorted(names))


main()
