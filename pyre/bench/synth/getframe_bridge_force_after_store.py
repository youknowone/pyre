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
#
# !! THE ESCAPE IS GONE.  `_gf(0)` no longer produces a forcing residual --
# `try_walker_specialize_sys_getframe` answers depth 0 at the top walk level out
# of the portal virtualizable -- so the shape below still compiles its loop and
# its bridge while forcing nothing.  The recorded baseline gates only part of
# that: it carries `fbw_blackhole_adopted_single_frame=0`, `..._multi_frame=0`,
# `loops_compiled=1` and `bridges_compiled=1`.  It carries NO `fbw_escape_*`,
# `fbw_force_*` or `sbt_entered` key at all, and an absent key is UNGATED, not
# zero -- those read 0 (and `sbt_entered=1`) only under `MAJIT_STATS=1`, which
# is where the claim above was measured.
#
# !!Raising the depth is NOT the repair here.  `_gf(1)` is what restored the
# escape in the `getframe_while_*` fixtures, but measured on this shape it
# leaves every `fbw_escape_*` at zero as well, so what a bridge walk does with a
# may-force residual is a separate question from what the depth argument folds.
# Until that is answered, the bridge-walk forced escape has no fixture and the
# claim below about "138 forced escapes, zero bridge=true" is a historical note,
# not a live count.
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
            names += len(fr.f_code.co_name)
        total += i
    return total, names, box.n


print(main())
