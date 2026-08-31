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
#     virtualizable, so a zero-argument call names a frame the walk already
#     holds and the baselines went to `fbw_blackhole_adopted_multi_frame=0`.
#     Depth 1 names a frame the specialization refuses, so the call stays
#     residual and hands the traced virtualizable to the interpreter;
#   - a read of `f_lasti` off the frame it returns.  `getframe` takes no
#     virtualizable force of its own; `rvirtualizable.py hook_access_field`
#     places one at each REDIRECTED field access, and `virtualizable_gen.rs`
#     declares that set as `last_instr`, `pycode`, `valuestackdepth`,
#     `debugdata` and `locals_cells_stack_w`.  `f_lasti` reads `last_instr`, so
#     its gateway forces.  A bare call forces nothing, and neither does a read
#     of `f_code` or `f_back`: `pyframe.rs`
#     `__majit_wrap_descr_typecheck_fget_f_code` and `..._fget_f_back` each say
#     at their own definition why they carry no marker.
# Changing any of the three can silently stop exercising the path.
#
# The printed total counts one callee entry per iteration, so a resume that
# replays the region or re-delivers an iteration prints something other than
# 30000.
import sys

_gf = sys._getframe


def leaf(x):
    _ = _gf(1).f_lasti
    return x + 1


def main():
    total = 0
    i = 0
    while i < 30000:
        total = leaf(total)
        i = i + 1
    return total


print(main())
