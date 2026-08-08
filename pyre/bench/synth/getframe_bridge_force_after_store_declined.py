# Companion to getframe_bridge_force_after_store,
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
# The bridge forced-vable escape of `getframe_bridge_force_plain`, with one
# un-journaled store ahead of the forcing call.
#
# `box.n = i` lowers to a Void `store_attr_fn` residual: it writes live heap, so
# it bumps the executed-effect odometer, and no journal covers it. Rolling the
# walk back therefore cannot undo the store, and the legacy entry replay would
# apply it a second time. The escape has to capture its operand-stack mirror and
# resume forward instead, which is what the recorded
# `fbw_blackhole_adopted_single_frame` pins; `fbw_rolled_back_with_effects` back
# above zero means the capture broke and the store is running twice again.
#
# The forcing residual itself never contributes an effect: the force branch
# returns before the odometer bump, so a bridge escape needs a second, earlier
# effectful op to register at all -- which is exactly what this file adds.
import sys

_gf = sys._getframe


class Box:
    n = 0


box = Box()


def main():
    total = 0
    names = 0
    for i in range(400000):
        if i % 97 == 0:
            box.n = i
            fr = _gf(0)
            names += len(fr.f_locals)
        total += i
    return total, names, box.n


print(main())
