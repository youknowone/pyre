# pyre-check: max-pypy-ratio=35
# pyre-check: min-pypy-ratio=3.75
# Coverage guard for the unconditional multi-frame blackhole path.
#
# A vable escape inside an INLINE sub-walk is what latches a multi-frame
# blackhole image. The rest of the corpus never produces one: every other
# getframe_* fixture drives with `for`, and with a FOR_ITER item in flight the
# callee's nested residual is declined by fbw_abort_nested_unjournaled_residual
# before execute_residual_call runs, so the force happens outside the sub-walk
# and the single-frame arm takes it. Driving with `while` is what reaches the
# site, and build_multi_frame_miframe then produces a depth-2 image the adopt
# takes.
#
# The shape below is load-bearing, not incidental:
#   - `while`, not `for`, per the decline above;
#   - sys._getframe called with NO argument, because the executor declines a
#     non-void residual whose arguments are not all concrete and the generic
#     LoadConst path is hard-declined as symbolic inside a sub-walk, so a
#     literal argument would make this depend on a dedicated fold rather than
#     on the loop form;
#   - nothing read off the returned frame, which would reintroduce that
#     constant-fold dependency.
# Changing any of the three can silently stop exercising the path.
#
# The printed total counts one callee entry per iteration, so a resume that
# replays the region or re-delivers an iteration prints something other than
# 30000.
import sys

_gf = sys._getframe


def leaf(x):
    _gf()
    return x + 1


def main():
    total = 0
    i = 0
    while i < 30000:
        total = leaf(total)
        i = i + 1
    return total


print(main())
