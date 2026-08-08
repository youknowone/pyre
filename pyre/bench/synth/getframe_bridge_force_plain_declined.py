# Companion to getframe_bridge_force_plain,
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
# Coverage for the forced-vable escape on a BRIDGE walk, which the rest of the
# corpus never produces: of 138 forced escapes across the synth fixtures, zero
# are `bridge=true`.
#
# Shape, each clause load-bearing:
#   * the `for` loop compiles on the common arm;
#   * `i % 97 == 0` is the rare arm, so its guard fails ~4124 times -- past
#     `DEFAULT_TRACE_EAGERNESS` -- and `start_bridge_tracing` sets
#     `ctx.is_bridge_trace`, making the walk over the rare arm a bridge walk;
#   * `_gf(0).f_locals` forces the virtualizable through a residual that stays
#     in the portal frame: the `_gf(0)` call folds, and the getset behind
#     `.f_locals` is a builtin, so `frame_entry_count()` does not move and no
#     user Python frame is entered;
#   * the call sits directly in the portal frame's loop body, so the framestack
#     is empty and this is not an inline sub-walk.
#
# With nothing else in the rare arm the escape's mirror image resolves and the
# walk adopts a single-frame blackhole terminal. Its sibling
# `getframe_bridge_force_after_store` puts an un-journaled store ahead of the
# forcing call and takes the replay path instead.
import sys

_gf = sys._getframe


def main():
    total = 0
    names = 0
    for i in range(400000):
        if i % 97 == 0:
            fr = _gf(0)
            names += len(fr.f_locals)
        total += i
    return total, names


print(main())
