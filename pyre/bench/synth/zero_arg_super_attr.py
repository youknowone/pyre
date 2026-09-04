# pyre-check: max-pypy-ratio=3
# pyre-check: skip-cpython
# pyre-check: spec-folds=load_deref,load_super_attr,super_attr_unwrap
# Zero-argument `super().val()` with the loop INSIDE the super-bearing method,
# in the `for` spelling — the `for` twin of `load_super_attr.py`.
#
# The loop form is the point.  `eval.rs for_iter_body_op_is_jit_safe` is an
# allow-list over the opcodes a FOR_ITER body may hold, and `LOAD_SUPER_ATTR`
# was not on it, so the whole `run` frame was refused at the back edge:
# `gate_declined_for_iter_region` counted once per iteration and the only
# compiled loop was `val`'s own function-entry trace.  At 2,000,000 iterations
# that ran 2.28s, against 0.03s for the identical body calling a plain
# `self.plain()`.  The `while` twin never reaches the gate and so never showed
# it.
#
# The body is also the LOAD_DEREF carrier: `super()` with no arguments reads
# `__class__` out of the closure, so the compiler emits LOAD_GLOBAL super,
# LOAD_DEREF __class__, LOAD_FAST self, LOAD_SUPER_ATTR — three of the four
# were already admitted.
#
# N is sized for pypy, not for cpython.  This file used to run 10,000
# iterations, at which pypy's execution-only time is a few microseconds --
# under `EXEC_TIME_FLOOR_S`, so check.py printed the ratio with a `~`, whose
# legend says "ratio is not a measurement, and no ratio gate is applied to
# it".  What that column showed was warmup: the same file reads 12.6ns per
# iteration at 1,000,000 and 0.9ns at 20,000,000.  pypy settles at ~0.6ns per
# iteration here, so 250,000,000 is what puts its execution time (~0.17s) far
# enough above `FLOOR_GATE_MIN_BASELINE_S` for the ratio to be a measurement
# and for the ceiling below to be enforced.  cpython cannot usefully run that
# many, hence `skip-cpython`; pypy stays the oracle the backends' output is
# compared against.
#
# The ceiling is fitted to what that measurement reads: 1.5x on dynasm and
# 1.7x on cranelift, carried with room for a slower host.
# `load_super_attr.py` carries the same 3 for the same family.
#
# `load_super_attr_descent` is not among them, and adding it is not the
# improvement it looks like.  The descent walks the MRO suffix and declines at
# `pyre_object::function::w_method_new`, the unpublished descriptor bind that
# ends the walk, so it is consulted once per trace and fires zero times.  The
# earlier `wtf8_key_is_utf8` wall this comment used to name is gone; that one
# moved rather than resolved anything.
#
# The wall has since been lifted experimentally.  What was behind it was not a
# constant factor but a regression that got worse with N, and the cause was
# the GC root bracket: the descended body pinned 10 roots per iteration and
# truncated none, because only `RootScope::drop` shrinks the shadow stack and
# that is a Rust `Drop`, which the MIR lowering's `TermKind::Drop` arm erased.
# The bracket was push-only in jitcode, so the stack grew 10 slots per
# iteration and every collection walked all of it.
#
# That arm now lowers the bracket's glue as the call it is.  Reading the
# firing arm still means publishing `w_method_new`, which is the wall, so
# these figures come from a throwaway binary carrying that one registration.
# A/B'd with `PYRE_FBW_NO_SPECIALIZE`, identical output on every arm, best of
# 3, one script and one host.  Per ITERATION, so a flat column is linear:
#
#                  descent           fold          `PYRE_NO_JIT=1`
#         N     before  after     before  after     before  after
#   250,000      3131    265        353    182       1516    860
#   500,000      3445    169        170     91       1315    774
# 1,000,000      4372    119         87     46       1243    730
# 2,000,000      8009     95         45     23       1181    709
#
# "before" is `c827bea5aaa`, "after" is `7d57e55a2dc` with the close emitted.
# The two binaries do not share a base, so the fold and `PYRE_NO_JIT` columns
# are the free control: the base moved them by at most 2.0x, and no base bump
# turns a rising column into a falling one.  The descent's total is now 0.049s
# fixed plus 71ns per iteration, holding to within 0.5% across the four N, and
# "slower than not running the JIT at all" is gone -- at N=2,000,000 it is
# 7.4x faster than `PYRE_NO_JIT=1` where it had been 6.8x slower.
#
# It is not deopt, and never was: both arms report `loops_compiled=2
# loops_aborted=0 bridges_compiled=0 guard_failures=1` with
# `MAJIT_GUARD_CENSUS` at `distinct=1 total=1` on each.  `MAJIT_LOG_OPT`
# measures 528 ops / 101 guards against the fold's 72 / 18, with 52 residual
# calls per iteration where the fold has none, 36 of them the bracket -- four
# opens and four closes now, not four opens alone.  The gate comment above
# `try_walker_orthodox_load_super_attr` carries the full reading, including
# why the wall stays even though the curve no longer bends: at this site the
# descent is still 4.2x the fold, and the gate consults it first.
#
# `try_walker_specialize_load_super_attr` consults the descent first and
# returns on success, so a firing descent can take a site away from the fold
# rather than add one.  That is a property of the ordering, not something this
# snippet currently exhibits: on
# `extra_tests/snippets/class_super_zero_arg_inlined_callee.py` the fold fires
# at 5 of its 6 consulted sites in BOTH arms, the descent takes the sixth, and
# `loops_compiled=8 loops_aborted=0` either way.  An earlier reading on base
# `9f81966a6eb` had coverage split 3+2 with `loops_aborted` rising 0 -> 6;
# that no longer reproduces.
#
try:
    import pypyjit

    pypyjit.set_param("threshold=20,function_threshold=20")
except ImportError:
    pass

N = 250000000


class Base:
    def val(self):
        return 1


class Child(Base):
    def run(self, n):
        acc = 0
        for _ in range(n):
            acc = acc + super().val()
        return acc


print(Child().run(N))
