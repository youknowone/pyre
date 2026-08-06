# Coverage for the forced-vable escape on a BRIDGE walk, which the rest of the
# corpus never produces: of 138 forced escapes across the synth fixtures, zero
# are `bridge=true`.
#
# Shape, each clause load-bearing:
#   * the `for` loop compiles on the common arm;
#   * `i % 97 == 0` is the rare arm, so its guard fails ~4124 times -- past
#     `DEFAULT_TRACE_EAGERNESS` -- and `start_bridge_tracing` sets
#     `ctx.is_bridge_trace`, making the walk over the rare arm a bridge walk;
#   * `_gf(0)` is a `CallFn` residual returning a Ref that forces the
#     virtualizable, and it is a builtin, so `frame_entry_count()` does not move
#     and no user Python frame is entered;
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
            names += len(fr.f_code.co_name)
        total += i
    return total, names


print(main())
