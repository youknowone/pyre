# pyre-check: max-pypy-ratio=60
# The forced-vable escape on a BRIDGE walk -- the case its two older siblings
# `getframe_bridge_force_plain` and `getframe_bridge_force_after_store` were
# written for and can no longer produce.
#
# What those two are missing is the lever, not the bridge. A residual forces the
# tracing token only when it is handed the TRACED virtualizable itself, and from
# inside the portal frame no `sys._getframe` call names it:
#   * `_gf(0)` / `_gf()` is folded -- `try_walker_specialize_sys_getframe`
#     answers depth 0 at the top walk level out of the portal virtualizable, so
#     there is no residual at all;
#   * `_gf(1)` from the portal frame names its CALLER, the module frame, which
#     is not the traced one, so the escape test returns false.
# Measured at zero on this shape as well: `repr(_gf(0).f_code)`, storing
# `_gf(0)` into a module list, and `str(_gf(0).f_lasti)`.
#
# One frame down, depth 1 lands on the portal frame, and that is the whole
# shape: `peek()` inlines into the rare arm and its `_gf(1)` is a `CallFn`
# residual that hands `main`'s own virtualizable to Python.
#
# Each clause is load-bearing:
#   * the `for` loop compiles on the common arm;
#   * `i % 97 == 0` is the rare arm, so its guard fails past
#     `DEFAULT_TRACE_EAGERNESS` and `start_bridge_tracing` sets
#     `ctx.is_bridge_trace` -- the walk over the rare arm is a bridge walk,
#     which `PYRE_FBW_CENSUS=1` reports as
#     `end=VableEscapedDuringResidualCall committed=true leg=2 bridge=true`;
#   * `peek` is a Python callee, so the escape's blackhole terminal is
#     MULTI-frame. A single-frame one is not reachable here for the reason
#     above -- the escape needs a frame below the portal to be called from.
#
# The baseline pins what that produces: `fbw_blackhole_adopted_multi_frame=20`
# with `fbw_rolled_back_with_effects=0` and `fbw_store_journal_rollback_failed=0`
# -- the escape captured its image and resumed forward rather than rolling back
# onto a legacy replay. Under `MAJIT_STATS=1` the same run reports
# `fbw_escape_portal_only=20` and `fbw_force_by_portal=20`; those keys are not in
# the baseline, and an absent key is UNGATED rather than zero.
#
# `loops_aborted=20` and `bridges_compiled=0` are the designed values, not a
# regression to drive down: an escape ends the walk, so every bridge trace over
# the rare arm is abandoned and the rare arm never gets compiled code. That is
# what a forced escape on a bridge walk costs, and pinning it is the point.
#
# The ceiling is a level record, not a fitted number. Locally this reads ~9.4x
# on dynasm and ~11.2x on cranelift, both `~`-clamped so no gate is applied --
# but `~` is not a property of the fixture, it is a property of the run, and the
# same family has been seen to read 5.6x locally and 25-30x on the ubuntu
# cranelift leg. 60 is that local cranelift number carried across that observed
# spread with room; tighten it once a CI reading exists, and set it from the leg
# that ENFORCES it rather than from a dynasm measurement.
import sys

_gf = sys._getframe


def peek():
    # Depth 1 from inside the inlined callee names `main` -- the traced
    # virtualizable -- so this stays a residual and forces the token.
    return _gf(1).f_code.co_name


def main():
    total = 0
    names = 0
    for i in range(400000):
        if i % 97 == 0:
            names += len(peek())
        total += i
    return total, names


print(main())
