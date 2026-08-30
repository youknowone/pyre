# pyre-check: max-pypy-ratio=800
# The declining twin of `zero_arg_name_bound_super_attr.py`: same base class,
# same method, same loop, same `s = super()` name binding.  The one difference
# is the receiver, and it is worth 29x.
#
# `super_check_python_free` settles only the two arms that need no Python --
# `obj` is itself a subtype of the start type, or `type(obj)` is.  A receiver
# whose `type()` is unrelated and whose `__class__` alone names the class, which
# is what a proxy object is, falls through to `super_check`'s
# `getattr_str(obj, "__class__")` arm, so BOTH super folds decline:
# `PYRE_FBW_SPEC_CENSUS=1` reports `bare_super_call consulted=5 fired=0` and
# `bare_super_virtual consulted=5 fired=0` here, against `bare_super_virtual
# consulted=1 fired=1` for the same file with `Child()` as the receiver.
#
# What the decline costs is not the extra `getattr`.  `s = super()` compiles to
# LOAD_GLOBAL super + CALL, so declining falls through to the generic FRAMELESS
# residual, which reaches `builtin_super`'s zero-argument tail and its
# `ExecutionContext::gettopframe()`.  That `force_frame` runs inside an opaque
# `bh_call_fn` and clears TOKEN_TRACING_RESCALL, which
# `tracing_after_residual_call` reads back as
# `VableEscapedDuringResidualCall` -- so the walk aborts, every time, and after
# `MAX_TRACE_ABORT_COUNT` of them (`warmstate.rs`) the merge point is stamped
# `JC_DONT_TRACE_HERE` and is never traced again in this process.  The loop is
# not slow; it is permanently uncompilable.
#
# So `loops_aborted=5` is a defect to drive to zero, not a designed value --
# the opposite of `getframe_bridge_force_from_inlined_callee`'s 20, where an
# escape on a bridge walk is what the fixture exists to pin.
#
# `loops_compiled=1` is the reason this needs its own baseline rather than a
# clause in that fixture: it is 1 in BOTH shapes and names a different trace in
# each.  `PYRE_LOOP_CENSUS=1` reads `loop loop #15 ForIter` on the folding
# receiver -- the loop below, compiled -- and `root val #0 Resume` here, a root
# trace over the two-line `Base.val` that the tracer settles for once the loop's
# own merge point is banned.  A gate on the count alone would see no change.
#
# Under `MAJIT_STATS=1` the same run reports `abrt_escape=5`,
# `fbw_escape_portal_only=5`, `fbw_force_by_portal=5`, `abort_ceiling_banned=1`
# and `abort_ceiling_refused=1`.  None of those keys is on the recorded surface
# (`JITSTATS_SNAPSHOT_FIELDS`), so `loops_aborted` is what gates them here:
# `aborted_tracing` is the only producer of both, and nothing else in this
# fixture aborts a trace.
#
# It has to be a bench.  `bare_super_frame_escape.py` already covers this route
# and covers it better -- it pins the ANSWER, which a counter cannot -- but it
# is a selfcheck, and `_apply_snapshot_gate` is called from `_run_backend_bench`
# alone, so no `.jitstats` baseline is read or written for it.  The whole
# decline region was jit-stats-invisible; this is the gated half.
#
# The ceiling is a level record, not a fitted number, and it is loose on
# purpose while `loops_aborted` is 5.  pypy's execution-only time here lands
# right at the measurement floor, so which gate applies is a property of the run
# rather than of the fixture: two runs minutes apart read `?368.2x` on dynasm
# (above the floor, under `FLOOR_GATE_MIN_BASELINE_S`, so the ceiling applies
# and the floor gate declines the baseline as too small to judge) and `~348.4x`
# on cranelift (clamped, so neither applies).  The whole of that figure is the
# abort: the same file with `Child()` in place of `Proxy()` is
# `zero_arg_name_bound_super_attr.py`'s loop, and it runs the same N in 0.06s.
# Tighten this the run after `loops_aborted` reaches 0, from the leg that
# ENFORCES it rather than from a dynasm reading.
N = 1000000


class Base:
    def val(self, x):
        return x + 1


class Child(Base):
    def loop(self, n):
        acc = 0
        for _ in range(n):
            s = super()
            acc = s.val(acc)
        return acc


class Proxy:
    """`type(self)` is not a `Child`.  Only `self.__class__` says so."""

    __class__ = Child


print(Child.loop(Proxy(), N))
