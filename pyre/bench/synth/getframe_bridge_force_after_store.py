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
            names += len(fr.f_code.co_name)
        total += i
    return total, names, box.n


print(main())
