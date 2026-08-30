# pyre-check: max-pypy-ratio=150
# pyre-check: spec-folds=bare_super_call
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
# What is left is a different fold: `load_attr_on_super consulted=1 fired=0`.
# The proxy is built, but the `s.val(acc)` read off it declines because
# `super_attr_fast_path` is asked about a receiver whose `type()` is not the
# `objtype` the proxy resolved -- and that is the whole remaining gap, 0.32s
# here against 0.06s for the `Child()` receiver, which folds both.  The
# `Rebound` shape in `bare_super_frame_escape.py` takes this same re-route with
# an ordinary receiver and reads 0.07s.
#
# The ceiling is a level record, not a fitted number.  pypy's execution-only
# time here lands at the measurement floor, so it clamps and no ratio gate is
# applied: locally that reads `~52.7x` on dynasm and `~72.9x` on cranelift.
# 150 is the cranelift number carried with room, for the host where the clamp
# does not apply.  Re-fit it from the leg that ENFORCES it if one ever does.
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
