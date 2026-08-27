# No `max-pypy-ratio`: the only loop here is the two-statement `while` driver,
# so a pypy ratio compares two interpreters' startup rather than any generated
# code, and reads whatever the host's process spawn cost happens to be that
# run. The jitstats baselines gate it.
# Coverage guard for the unconditional multi-frame blackhole path.
#
# A vable escape inside an INLINE sub-walk is what latches a multi-frame
# blackhole image: driving with `while` reaches the site, and
# build_multi_frame_miframe then produces a depth-2 image the adopt takes.
# Other fixtures reach a multi-frame adopt by other routes
# (`getframe_caller_resume_coord_two_call_sites`,
# `getframe_while_escaping_read_frame_identity`,
# `trace_too_long_inline_multiframe`); this one pins the sub-walk route.
#
# The shape below is load-bearing, not incidental:
#   - `while`, not `for`: with a FOR_ITER item in flight the callee's nested
#     residual is declined by fbw_abort_nested_unjournaled_residual before
#     execute_residual_call runs, so the force would happen outside the
#     sub-walk and the single-frame arm would take it;
#   - the CALLER's depth, `_gf(1)`.  `try_walker_specialize_sys_getframe` takes
#     only depth 0 at the top walk level, where getframe's answer IS the portal
#     virtualizable and no force is needed -- so a zero-argument call, which is
#     what this fixture used to carry, now escapes nothing at all and the
#     baselines went to `fbw_blackhole_adopted_multi_frame=0`.  Depth 1 names a
#     frame the specialization refuses, so the call stays residual and forces;
#   - nothing read off the returned frame: the read accessors force nothing
#     either -- `fast2locals` is `@jit.unroll_safe` -- so a read would add
#     opcodes without adding an escape.
# Changing any of the three can silently stop exercising the path.
#
# The printed total counts one callee entry per iteration, so a resume that
# replays the region or re-delivers an iteration prints something other than
# 30000.
import sys

_gf = sys._getframe


def leaf(x):
    _gf(1)
    return x + 1


def main():
    total = 0
    i = 0
    while i < 30000:
        total = leaf(total)
        i = i + 1
    return total


print(main())
