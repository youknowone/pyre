# pyre-check: max-pypy-ratio=150
# pyre-check: spec-folds=bare_super_call,load_attr_on_super
# The declining twin of `zero_arg_name_bound_super_attr.py`: same base class,
# same method, same loop, same `s = super()` name binding.  The one difference
# is the receiver.
#
# `super_check_python_free` settles only the two arms that need no Python --
# `obj` is itself a subtype of the start type, or `type(obj)` is.  A receiver
# whose `type()` is unrelated and whose `__class__` alone names the class,
# which is what a proxy object is, falls through to `super_check`'s
# `getattr_str(obj, "__class__")` arm.  `PYRE_FBW_SPEC_CENSUS=1` reads
# `bare_super_virtual consulted=1 fired=0` here against `fired=1` for the same
# file with `Child()` as the receiver, so this is the fixture that keeps
# `bare_super_call` -- the re-route the `spec-folds` header names -- reachable
# from a receiver the virtual fold cannot answer for.
#
# What the re-route is worth is the whole trace, not the `getattr`.
# `s = super()` compiles to LOAD_GLOBAL super + CALL, and the generic call it
# would otherwise reach is FRAMELESS: it enters `builtin_super`'s zero-argument
# tail, whose `ExecutionContext::gettopframe()` runs `force_frame` inside an
# opaque `bh_call_fn`, clearing TOKEN_TRACING_RESCALL for
# `tracing_after_residual_call` to read back as
# `VableEscapedDuringResidualCall`.  Every walk over this loop aborted that
# way, and after `MAX_TRACE_ABORT_COUNT` of them (`warmstate.rs`) the merge
# point was stamped `JC_DONT_TRACE_HERE` for the rest of the process.
#
# `loops_aborted=0` is what this fixture exists to hold.  Recorded at 5 on the
# commit that added it -- with `fbw_blackhole_adopted_single_frame=5`, and
# `abrt_escape=5 fbw_escape_portal_only=5 fbw_force_by_portal=5
# abort_ceiling_banned=1` under `MAJIT_STATS=1`, none of which is on the
# recorded surface -- against 1.91s of execution on dynasm.  Since the re-route
# admits this receiver the loop compiles: `PYRE_LOOP_CENSUS=1` reads
# `loop loop #15 ForIter` where it read `root val #0 Resume`, the consolation
# root trace over `Base.val` the tracer settled for once the loop's own merge
# point was banned.
#
# It has to be a bench.  `bare_super_frame_escape.py` covers this route and
# covers it better -- it pins the ANSWER, which a counter cannot -- but it is a
# selfcheck, and `_apply_snapshot_gate` is called from `_run_backend_bench`
# alone, so no `.jitstats` baseline is read or written for it.  This is the
# gated half.
#
# The `s.val(acc)` read off the proxy is the second fold in the header, and it
# reaches this receiver for a reason the first one does not: `W_Super`
# getattribute walks the `w_objtype` `_super_check` stored, and the proxy is
# holding it, so nothing has to re-derive a class from `type(self)`.  It cost
# 0.32s here to decline -- against 0.17s now, and 0.06s for the same file with
# `Child()` as the receiver.
#
# What separates those last two is not a fold.  `_super_check` on this receiver
# IS a `getattr(obj, "__class__")`, once per iteration, and the re-route runs it
# concretely because that is what the semantics are; the `Child()` receiver
# settles the same question by walking installed MROs and pays nothing.
#
# The ceiling is a level record, not a fitted number.  pypy's execution-only
# time here lands at the measurement floor, so it clamps and no ratio gate is
# applied: locally that reads `~22.5x` on dynasm and `~39.3x` on cranelift,
# from `~52.7x` and `~72.9x` when the read declined.  150 is the cranelift
# number carried with room, for the host where the clamp does not apply.
# Re-fit it from the leg that ENFORCES it if one ever does.
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
