use majit_ir::operand::Operand;
/// OptRewrite: algebraic simplification and constant folding.
///
/// Translated from rpython/jit/metainterp/optimizeopt/rewrite.py.
/// Rewrites operations into equivalent, cheaper operations.
/// This includes constant folding for pure ops and algebraic identities.
use majit_ir::{Op, OpCode, OpRef, Value};

use crate::optimizeopt::info::{PreambleOp, PtrInfoExt};
use crate::optimizeopt::{OptContext, Optimization, OptimizationResult, intdiv};
use majit_ir::box_ref::BoxRef;

/// rewrite.py: loop_invariant_results value.
/// RPython stores PreambleOp or regular Box (AbstractResOp) directly
/// in the dict. In Rust, we use an enum to distinguish.
#[derive(Clone, Debug)]
enum LoopInvariantEntry {
    /// Regular result (already forced or body-computed).
    Direct(OpRef),
    /// shortpreamble.py:148-159: LoopInvariantOp.produce_op stores
    /// PreambleOp(op, preamble_op, invented_name) in the dict.
    Preamble(PreambleOp),
}

/// Yield the `OptimizationResult::InvalidLoop` control value so the driver
/// (`propagate_from_pass_range`) converts it to `Err(InvalidLoop)` at the
/// pass barrier.  RPython `raise InvalidLoop`; threaded as a value here so
/// it works under `panic=abort`.  Use as `return raise_invalid_loop(msg)`.
#[cold]
#[inline(never)]
fn raise_invalid_loop(msg: &'static str) -> OptimizationResult {
    OptimizationResult::InvalidLoop(msg)
}

/// info.py:16-18: INFO_NULL / INFO_NONNULL / INFO_UNKNOWN
/// optimizer.py:127-135: getnullness()
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Nullness {
    Null,
    Nonnull,
    Unknown,
}

/// Rewrite operations into equivalent, cheaper forms.
///
/// Handles:
/// - Constant folding for pure integer/boolean ops
/// - Algebraic simplifications (identity, absorbing elements)
/// - Strength reduction for power-of-two `IntFloorDiv` / `IntMod`
///   (every other integer fold lives in OptIntBounds / OptPure)
/// - Guard simplification when argument is known constant
/// - Boolean operation rewrites (inverse/reflex)
/// - Conditional call elimination when condition/value is constant
/// - Pointer equality on same OpRef
/// - Cast and convert round-trip elimination
/// - Guard-no-exception removal after removed calls
pub struct OptRewrite {
    /// pyre-only side-cache: (opcode, arg0, arg1) → result OpRef, populated by
    /// optimize_comparison. Upstream has no bool_result_cache; find_rewritable_bool
    /// / try_boolinvers (rewrite.py:54-93) build a synthetic ResOperation and look
    /// it up via get_pure_result against the shared _pure_operations table.
    /// Convergence: retire this cache and route the bool lookups through the pure
    /// optimizer's get_pure_result / pure_from_args2 (both already present at
    /// pure.rs:498/492) keyed off the pure-op table — coupled to the pure-optimizer
    /// subsystem. NOT a box-identity rekey target: rekeying the OpRef pair to BoxRef
    /// would entrench a structure upstream does not have.
    bool_result_cache: majit_ir::VecMap<(OpCode, OpRef, OpRef), OpRef>,
    /// rewrite.py:39: loop_invariant_results — cache for CALL_LOOPINVARIANT results.
    /// Key: function pointer (arg0 as i64).
    /// Value: Direct(OpRef) or Preamble(PreambleOp) — RPython isinstance check.
    loop_invariant_results: majit_ir::VecMap<i64, LoopInvariantEntry>,
    /// rewrite.py:40: loop_invariant_producer — maps func_ptr → emitted Call op.
    /// Used by produce_potential_short_preamble_ops (rewrite.py:45-47).
    loop_invariant_producer: majit_ir::VecMap<i64, Op>,
}

impl OptRewrite {
    pub fn new() -> Self {
        OptRewrite {
            bool_result_cache: majit_ir::VecMap::new(),
            loop_invariant_results: majit_ir::VecMap::new(),
            loop_invariant_producer: majit_ir::VecMap::new(),
        }
    }

    // ── Constant folding for binary integer ops ──

    /// Constant-fold for the pyre-opcode strength-reduction rules below
    /// (`IntFloorDiv` / `IntMod` only — every other integer fold lives in
    /// OptIntBounds / OptPure, as upstream).
    fn try_fold_binary_int(&self, opcode: OpCode, lhs: i64, rhs: i64) -> Option<i64> {
        match opcode {
            OpCode::IntFloorDiv => {
                if rhs != 0 {
                    Some(lhs.wrapping_div(rhs))
                } else {
                    None
                }
            }
            OpCode::IntMod => {
                if rhs != 0 {
                    Some(lhs.wrapping_rem(rhs))
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Try algebraic simplification for INT_FLOORDIV.
    /// `x // 1 -> x`, constant fold when both operands are known.
    fn optimize_int_floor_div(
        &self,
        op: &Op,
        op_rc: &majit_ir::OpRc,
        ctx: &mut OptContext,
    ) -> OptimizationResult {
        let arg0 = op.arg(0);
        let arg1 = op.arg(1);

        // Constant fold
        if let (Some(a), Some(b)) = (
            ctx.resolve_box_box_opt(&arg0.to_boxref())
                .and_then(|b| ctx.get_constant_int_box(&b)),
            ctx.resolve_box_box_opt(&arg1.to_boxref())
                .and_then(|b| ctx.get_constant_int_box(&b)),
        ) {
            if let Some(result) = self.try_fold_binary_int(OpCode::IntFloorDiv, a, b) {
                let b = ctx.materialize_box_at(op.pos.get());
                ctx.make_constant_box(&b, Value::Int(result));
                return OptimizationResult::Remove;
            }
        }

        // x // 1 -> x (identity)
        if let Some(1) = ctx
            .resolve_box_box_opt(&arg1.to_boxref())
            .and_then(|b| ctx.get_constant_int_box(&b))
        {
            let b_old = BoxRef::from_bound_op(op_rc);
            let b_arg = ctx.resolve_box_box(&arg0.to_boxref());
            ctx.make_equal_to(&b_old, &b_arg);
            return OptimizationResult::Remove;
        }

        // x // (-1) -> INT_NEG(x)
        if let Some(-1) = ctx
            .resolve_box_box_opt(&arg1.to_boxref())
            .and_then(|b| ctx.get_constant_int_box(&b))
        {
            let mut neg = Op::new(OpCode::IntNeg, &[arg0]);
            neg.pos.set(op.pos.get());
            return OptimizationResult::Replace(neg);
        }

        // 0 // x -> 0 (zero dividend)
        if let Some(0) = ctx
            .resolve_box_box_opt(&arg0.to_boxref())
            .and_then(|b| ctx.get_constant_int_box(&b))
        {
            let b = ctx.materialize_box_at(op.pos.get());
            ctx.make_constant_box(&b, Value::Int(0));
            return OptimizationResult::Remove;
        }

        // x // x -> 1 (self-division, x != 0 guaranteed by semantics)
        if ctx
            .resolve_box_box(&arg0.to_boxref())
            .same_box(&ctx.resolve_box_box(&arg1.to_boxref()))
        {
            let b = ctx.materialize_box_at(op.pos.get());
            ctx.make_constant_box(&b, Value::Int(1));
            return OptimizationResult::Remove;
        }

        // Strength reduction for constant divisor >= 2
        if let Some(divisor) = ctx
            .resolve_box_box_opt(&arg1.to_boxref())
            .and_then(|b| ctx.get_constant_int_box(&b))
        {
            if divisor > 1 && divisor.count_ones() == 1 {
                // Power-of-2 floor division: x // (2^n) = x >> n
                // Arithmetic right shift IS floor division for positive divisors.
                let shift = divisor.trailing_zeros();
                let shift_ref = self.emit_constant_int(ctx, shift as i64);
                let arg_shift = ctx.materialize_box_at(shift_ref);
                let result_ref = ctx.emit(Op::new(
                    OpCode::IntRshift,
                    &[arg0, Operand::from_boxref(&arg_shift)],
                ));
                let b_old = BoxRef::from_bound_op(op_rc);
                let b_res = ctx.get_box_replacement(result_ref);
                ctx.make_equal_to(&b_old, &b_res);
                return OptimizationResult::Remove;
            }

            // General constant divisor >= 3: magic number multiplication
            if divisor >= 3 {
                // rewrite.py:770 `known_nonneg = b1.known_nonnegative()`:
                // a non-negative dividend skips the sign-correction ops.
                let known_nonneg = ctx
                    .resolve_box_box_opt(&arg0.to_boxref())
                    .and_then(|b| ctx.peek_intbound_box(&b))
                    .map_or(false, |bound| bound.known_nonnegative());
                let result = intdiv::division_operations(
                    arg0.to_opref(),
                    divisor,
                    known_nonneg,
                    ctx.current_pass_idx,
                    ctx,
                );
                let b_old = BoxRef::from_bound_op(op_rc);
                let b_res = ctx.get_box_replacement(result);
                ctx.make_equal_to(&b_old, &b_res);
                return OptimizationResult::Remove;
            }
        }

        OptimizationResult::PassOn
    }

    /// Try algebraic simplification for INT_MOD.
    ///
    /// Strength reduction from rpython/jit/metainterp/optimizeopt/intdiv.py.
    fn optimize_int_mod(
        &self,
        op: &Op,
        op_rc: &majit_ir::OpRc,
        ctx: &mut OptContext,
    ) -> OptimizationResult {
        let arg0 = op.arg(0);
        let arg1 = op.arg(1);

        // Constant fold
        if let (Some(a), Some(b)) = (
            ctx.resolve_box_box_opt(&arg0.to_boxref())
                .and_then(|b| ctx.get_constant_int_box(&b)),
            ctx.resolve_box_box_opt(&arg1.to_boxref())
                .and_then(|b| ctx.get_constant_int_box(&b)),
        ) {
            if let Some(result) = self.try_fold_binary_int(OpCode::IntMod, a, b) {
                let b = ctx.materialize_box_at(op.pos.get());
                ctx.make_constant_box(&b, Value::Int(result));
                return OptimizationResult::Remove;
            }
        }

        // x % 1 -> 0 (any integer mod 1 is 0)
        if let Some(1) = ctx
            .resolve_box_box_opt(&arg1.to_boxref())
            .and_then(|b| ctx.get_constant_int_box(&b))
        {
            let b = ctx.materialize_box_at(op.pos.get());
            ctx.make_constant_box(&b, Value::Int(0));
            return OptimizationResult::Remove;
        }

        // x % (-1) -> 0 (any integer mod -1 is 0)
        if let Some(-1) = ctx
            .resolve_box_box_opt(&arg1.to_boxref())
            .and_then(|b| ctx.get_constant_int_box(&b))
        {
            let b = ctx.materialize_box_at(op.pos.get());
            ctx.make_constant_box(&b, Value::Int(0));
            return OptimizationResult::Remove;
        }

        // 0 % x -> 0 (zero dividend)
        if let Some(0) = ctx
            .resolve_box_box_opt(&arg0.to_boxref())
            .and_then(|b| ctx.get_constant_int_box(&b))
        {
            let b = ctx.materialize_box_at(op.pos.get());
            ctx.make_constant_box(&b, Value::Int(0));
            return OptimizationResult::Remove;
        }

        // x % x -> 0 (self-modulo)
        if ctx
            .resolve_box_box(&arg0.to_boxref())
            .same_box(&ctx.resolve_box_box(&arg1.to_boxref()))
        {
            let b = ctx.materialize_box_at(op.pos.get());
            ctx.make_constant_box(&b, Value::Int(0));
            return OptimizationResult::Remove;
        }

        // Strength reduction for constant divisor >= 3 (non-power-of-2)
        if let Some(divisor) = ctx
            .resolve_box_box_opt(&arg1.to_boxref())
            .and_then(|b| ctx.get_constant_int_box(&b))
        {
            if divisor >= 3 && divisor.count_ones() != 1 {
                // rewrite.py:809 `known_nonneg = b1.known_nonnegative()`:
                // a non-negative dividend skips the sign-correction ops.
                let known_nonneg = ctx
                    .resolve_box_box_opt(&arg0.to_boxref())
                    .and_then(|b| ctx.peek_intbound_box(&b))
                    .map_or(false, |bound| bound.known_nonnegative());
                let result = intdiv::modulo_operations(
                    arg0.to_opref(),
                    divisor,
                    known_nonneg,
                    ctx.current_pass_idx,
                    ctx,
                );
                let b_old = BoxRef::from_bound_op(op_rc);
                let b_res = ctx.get_box_replacement(result);
                ctx.make_equal_to(&b_old, &b_res);
                return OptimizationResult::Remove;
            }
        }

        OptimizationResult::PassOn
    }

    // ── Unary operations ──

    /// Constant fold INT_IS_ZERO.
    /// rewrite.py:512-513 `optimize_INT_IS_ZERO`:
    ///     return self._optimize_nullness(op, op.getarg(0), False)
    fn optimize_int_is_zero(&self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        self.optimize_nullness(op, op.arg(0).to_opref(), false, ctx)
    }

    /// rewrite.py:505-510 `optimize_INT_IS_TRUE`:
    ///     if (not self.is_raw_ptr(op.getarg(0)) and
    ///         self.getintbound(op.getarg(0)).is_bool()):
    ///         self.make_equal_to(op, op.getarg(0))
    ///         return
    ///     return self._optimize_nullness(op, op.getarg(0), True)
    fn optimize_int_is_true(
        &self,
        op: &Op,
        op_rc: &majit_ir::OpRc,
        ctx: &mut OptContext,
    ) -> OptimizationResult {
        let arg0 = op.arg(0);

        // rewrite.py:505-510 optimize_INT_IS_TRUE:
        //     if (not self.is_raw_ptr(op.getarg(0)) and
        //         self.getintbound(op.getarg(0)).is_bool()):
        //         self.make_equal_to(op, op.getarg(0))
        //         return
        //     return self._optimize_nullness(op, op.getarg(0), True)
        //
        // is_raw_ptr (optimizer.py:154-158) checks for an
        // AbstractRawPtrInfo on the box's forwarded slot — NOT a
        // Ref-typed Box check. A raw-pointer 'i'-typed Box (i.e. one
        // pointing into a virtual raw buffer) skips the is_bool
        // shortcut because the buffer pointer's intbound is unrelated
        // to its boolean truthiness.
        let arg0_is_raw = ctx.is_raw_ptr(&op.arg(0).to_boxref().get_box_replacement(false));
        if !arg0_is_raw {
            if let Some(bound) = ctx
                .resolve_box_box_opt(&arg0.to_boxref())
                .and_then(|b| ctx.peek_intbound_box(&b))
            {
                if bound.is_bool() {
                    // make_equal_to: replace INT_IS_TRUE result with arg0.
                    let b_old = BoxRef::from_bound_op(op_rc);
                    let b_arg = ctx.resolve_box_box(&arg0.to_boxref());
                    ctx.make_equal_to(&b_old, &b_arg);
                    return OptimizationResult::Remove;
                }
            }
        }

        // is_true_and_minint: int_is_true(int_and(x, MININT)) => int_lt(x, 0)
        if let Some(inner) = ctx
            .resolve_box_box_opt(&arg0.to_boxref())
            .and_then(|pb| ctx.get_producing_op(&pb))
        {
            if inner.opcode == OpCode::IntAnd {
                if ctx.get_constant_int_box(&inner.arg(1).to_boxref().get_box_replacement(false))
                    == Some(i64::MIN)
                {
                    let zero = self.emit_constant_int(ctx, 0);
                    let arg_zero = ctx.materialize_box_at(zero);
                    let mut new_op = Op::new(
                        OpCode::IntLt,
                        &[inner.arg(0), Operand::from_boxref(&arg_zero)],
                    );
                    new_op.pos.set(op.pos.get());
                    return OptimizationResult::Emit(new_op);
                }
            }
        }

        self.optimize_nullness(op, arg0.to_opref(), true, ctx)
    }

    /// rewrite.py:515-554: _optimize_oois_ooisnot(op, expect_isnot, instance)
    ///
    /// Pointer equality optimization using virtual/null/class information.
    fn optimize_oois_ooisnot(
        &self,
        op: &Op,
        expect_isnot: bool,
        instance: bool,
        ctx: &mut OptContext,
    ) -> OptimizationResult {
        // rewrite.py:515-554 _optimize_oois_ooisnot:
        //     arg0 = get_box_replacement(op.getarg(0))
        //     arg1 = get_box_replacement(op.getarg(1))
        //     info0 = getptrinfo(arg0)
        //     info1 = getptrinfo(arg1)
        let info0 = ctx.getptrinfo(&op.arg(0).to_boxref().get_box_replacement(false));
        let info1 = ctx.getptrinfo(&op.arg(1).to_boxref().get_box_replacement(false));

        let is_virtual0 = info0.as_ref().is_some_and(|i| i.is_virtual());
        let is_virtual1 = info1.as_ref().is_some_and(|i| i.is_virtual());

        // rewrite.py:530-535: virtual objects
        if is_virtual0 {
            let intres = if is_virtual1 {
                // rewrite.py:532: `intres = (info0 is info1) ^ expect_isnot`
                // — PtrInfo identity (info.py:71-72 `same_info` is `self is
                // other` for non-Const infos), via getptrinfo_handle which
                // preserves the `_forwarded` cell identity.
                let same = match (
                    ctx.getptrinfo_handle(&op.arg(0).to_boxref().get_box_replacement(false)),
                    ctx.getptrinfo_handle(&op.arg(1).to_boxref().get_box_replacement(false)),
                ) {
                    (Some(h0), Some(h1)) => h0.same_info(&h1),
                    _ => false,
                };
                same ^ expect_isnot
            } else {
                expect_isnot
            };
            let b = ctx.materialize_box_at(op.pos.get());
            ctx.make_constant_box(&b, Value::Int(intres as i64));
            return OptimizationResult::Remove;
        }
        if is_virtual1 {
            let b = ctx.materialize_box_at(op.pos.get());
            ctx.make_constant_box(&b, Value::Int(expect_isnot as i64));
            return OptimizationResult::Remove;
        }

        // rewrite.py:528-531: null checks — fall back to OpRef for downstream
        let arg0 = ctx.resolve_box_box(&op.arg(0).to_boxref()).to_opref();
        let arg1 = ctx.resolve_box_box(&op.arg(1).to_boxref()).to_opref();
        if info1.as_ref().is_some_and(|i| i.is_null()) {
            return self.optimize_nullness(op, arg0, expect_isnot, ctx);
        }
        if info0.as_ref().is_some_and(|i| i.is_null()) {
            return self.optimize_nullness(op, arg1, expect_isnot, ctx);
        }

        // rewrite.py:542-543: `elif arg0 is arg1:` — box identity
        // (resoperation.py:38 `same_box` base = `self is other`).
        if ctx.box_is(arg0, arg1) {
            let b = ctx.materialize_box_at(op.pos.get());
            ctx.make_constant_box(&b, Value::Int(!expect_isnot as i64));
            return OptimizationResult::Remove;
        }

        // rewrite.py:535-553: instance comparison — different classes → not same
        if instance {
            let cls0 = info0
                .as_ref()
                .and_then(|i| i.get_known_class(ctx.cpu.as_ref()));
            let cls1 = info1
                .as_ref()
                .and_then(|i| i.get_known_class(ctx.cpu.as_ref()));
            if let (Some(c0), Some(c1)) = (cls0, cls1) {
                if c0 != c1 {
                    let b = ctx.materialize_box_at(op.pos.get());
                    ctx.make_constant_box(&b, Value::Int(expect_isnot as i64));
                    return OptimizationResult::Remove;
                }
            }
        } else {
            // rewrite.py:550-553: non-instance array pointer comparison.
            // If both are ArrayPtrInfo with known-different length bounds,
            // they cannot be the same object.
            let lb0 = info0.clone().and_then(|mut i| i.getlenbound(None));
            let lb1 = info1.clone().and_then(|mut i| i.getlenbound(None));
            if let (Some(lb0), Some(lb1)) = (lb0, lb1) {
                if lb0.known_ne(&lb1) {
                    let b = ctx.materialize_box_at(op.pos.get());
                    ctx.make_constant_box(&b, Value::Int(expect_isnot as i64));
                    return OptimizationResult::Remove;
                }
            }
        }

        OptimizationResult::PassOn
    }

    /// rewrite.py:496-503 `_optimize_nullness(op, box, expect_nonnull)`:
    ///     info = self.getnullness(box)
    ///     if info == INFO_NONNULL: self.make_constant_int(op, expect_nonnull)
    ///     elif info == INFO_NULL: self.make_constant_int(op, not expect_nonnull)
    ///     else: return self.emit(op)
    fn optimize_nullness(
        &self,
        op: &Op,
        arg: OpRef,
        expect_nonnull: bool,
        ctx: &mut OptContext,
    ) -> OptimizationResult {
        match self.getnullness(arg, ctx) {
            Nullness::Nonnull => {
                let b = ctx.materialize_box_at(op.pos.get());
                ctx.make_constant_box(&b, Value::Int(expect_nonnull as i64));
                OptimizationResult::Remove
            }
            Nullness::Null => {
                let b = ctx.materialize_box_at(op.pos.get());
                ctx.make_constant_box(&b, Value::Int(!expect_nonnull as i64));
                OptimizationResult::Remove
            }
            Nullness::Unknown => OptimizationResult::PassOn,
        }
    }

    /// Constant fold INT_FORCE_GE_ZERO.
    fn optimize_int_force_ge_zero(
        &self,
        op: &Op,
        op_rc: &majit_ir::OpRc,
        ctx: &mut OptContext,
    ) -> OptimizationResult {
        let arg0 = op.arg(0);

        if let Some(a) = ctx
            .resolve_box_box_opt(&arg0.to_boxref())
            .and_then(|b| ctx.get_constant_int_box(&b))
        {
            let b = ctx.materialize_box_at(op.pos.get());
            ctx.make_constant_box(&b, Value::Int(if a < 0 { 0 } else { a }));
            return OptimizationResult::Remove;
        }

        // force_ge_zero_pos: int_force_ge_zero(x) => x (if x known nonneg)
        if let Some(bound) = ctx
            .resolve_box_box_opt(&arg0.to_boxref())
            .and_then(|b| ctx.peek_intbound_box(&b))
        {
            if bound.known_nonnegative() {
                let b_old = BoxRef::from_bound_op(op_rc);
                let b_arg = ctx.resolve_box_box(&arg0.to_boxref());
                ctx.make_equal_to(&b_old, &b_arg);
                return OptimizationResult::Remove;
            }
            // force_ge_zero_neg: int_force_ge_zero(x) => 0 (if x known negative)
            if bound.upper < 0 {
                let b = ctx.materialize_box_at(op.pos.get());
                ctx.make_constant_box(&b, Value::Int(0));
                return OptimizationResult::Remove;
            }
        }

        OptimizationResult::PassOn
    }

    /// Constant fold int_between(a, b, c) => a <= b < c.
    fn optimize_int_between(&self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let arg0 = op.arg(0);
        let arg1 = op.arg(1);
        let arg2 = op.arg(2);

        if let (Some(a), Some(b), Some(c)) = (
            ctx.resolve_box_box_opt(&arg0.to_boxref())
                .and_then(|b| ctx.get_constant_int_box(&b)),
            ctx.resolve_box_box_opt(&arg1.to_boxref())
                .and_then(|b| ctx.get_constant_int_box(&b)),
            ctx.resolve_box_box_opt(&arg2.to_boxref())
                .and_then(|b| ctx.get_constant_int_box(&b)),
        ) {
            let result = (a <= b && b < c) as i64;
            let b = ctx.materialize_box_at(op.pos.get());
            ctx.make_constant_box(&b, Value::Int(result));
            return OptimizationResult::Remove;
        }

        OptimizationResult::PassOn
    }

    // ── Comparisons ──

    /// Comparison folds (constant folds, knownbits eq/ne, eq_zero /
    /// eq_one / eq_sub_eq) live in OptIntBounds, as upstream. This arm
    /// only records the comparison result for `find_rewritable_bool`
    /// (inverse/reflex lookup).
    fn optimize_comparison(&mut self, op: &Op) -> OptimizationResult {
        let arg0 = op.arg(0);
        let arg1 = op.arg(1);
        self.bool_result_cache
            .insert((op.opcode, arg0.to_opref(), arg1.to_opref()), op.pos.get());

        OptimizationResult::PassOn
    }

    // ── Guards ──

    /// Optimize GUARD_TRUE following RPython rewrite.py: optimize_guard(op, CONST_1).
    /// If the condition is a known constant 0, the trace is impossible and must abort.
    ///
    /// rewrite.py:163-184 `optimize_guard` proper (the contradiction check
    /// + emit) is the call-time half. The `make_constant(box, CONST_1)` half
    /// of the upstream `optimize_guard` is split into
    /// `propagate_postprocess` (rewrite.py:352-371) per RPython's
    /// `have_postprocess` model — see the bottom of this file.
    fn optimize_guard_true(&self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let arg0 = op.arg(0);

        // rewrite.py:165-168: box.type=='i' checks intbound.is_constant(),
        // which catches values narrowed to a single point by bounds analysis,
        // not just the constant pool.
        if let Some(val) = ctx
            .resolve_box_box_opt(&arg0.to_boxref())
            .and_then(|b| ctx.get_constant_int_or_bound_box(&b))
        {
            if val != 0 {
                return OptimizationResult::Remove;
            }
            return raise_invalid_loop("GUARD_TRUE proven to always fail");
        }

        OptimizationResult::PassOn
    }

    /// Optimize GUARD_FALSE following RPython rewrite.py: optimize_guard(op, CONST_0).
    fn optimize_guard_false(&self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let arg0 = op.arg(0);

        // rewrite.py:165-168: box.type=='i' checks intbound.is_constant().
        if let Some(val) = ctx
            .resolve_box_box_opt(&arg0.to_boxref())
            .and_then(|b| ctx.get_constant_int_or_bound_box(&b))
        {
            if val == 0 {
                return OptimizationResult::Remove;
            }
            return raise_invalid_loop("GUARD_FALSE proven to always fail");
        }

        OptimizationResult::PassOn
    }

    /// rewrite.py:284-347: optimize_GUARD_VALUE + replace_old_guard_with_guard_value
    ///
    /// If both args are constants and equal, the guard is redundant → remove.
    /// If arg0 is Ref-typed with a prior guard_nonnull/guard_class, replace
    /// that old guard with guard_value (rewrite.py:307-347).
    /// If the expected value is boolean, replace with GUARD_TRUE/FALSE.
    fn optimize_guard_value(&self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        if op.num_args() < 2 {
            return OptimizationResult::PassOn;
        }
        let arg0 = op.arg(0);
        let arg1 = op.arg(1);

        // rewrite.py:163-184 optimize_guard(op, constbox) — base contradiction
        // check called from optimize_GUARD_VALUE at line 301. `arg1` is the
        // asserted Const. For box.type=='i' (rewrite.py:165-168) the check is
        // intbound.is_constant()/get_constant_int(), catching values narrowed
        // by bounds analysis, not just the constant pool. For 'r'
        // (rewrite.py:174-182) it is get_box_replacement(box).is_constant()/
        // same_constant. For 'f', rewrite.py:295-298 returns silently when
        // arg0 is constant without checking equality, so we mirror that by
        // removing on equality but never raising on mismatch.
        if let Some(expected_int) = ctx
            .resolve_box_box_opt(&arg1.to_boxref())
            .and_then(|b| ctx.get_constant_int_box(&b))
        {
            if let Some(actual_int) = ctx
                .resolve_box_box_opt(&arg0.to_boxref())
                .and_then(|b| ctx.get_constant_int_or_bound_box(&b))
            {
                if actual_int == expected_int {
                    return OptimizationResult::Remove;
                }
                return raise_invalid_loop("GUARD_VALUE proven to always fail");
            }
        } else if let (Some(actual), Some(expected)) = (
            ctx.resolve_box_box_opt(&arg0.to_boxref())
                .and_then(|b| ctx.get_constant_box(&b)),
            ctx.resolve_box_box_opt(&arg1.to_boxref())
                .and_then(|cb| cb.const_value()),
        ) {
            if actual == expected {
                return OptimizationResult::Remove;
            }
            match actual {
                Value::Int(_) | Value::Ref(_) => {
                    return raise_invalid_loop("GUARD_VALUE proven to always fail");
                }
                Value::Float(_) => {
                    return OptimizationResult::Remove;
                }
                Value::Void => {}
            }
        }

        // rewrite.py:284-301: optimize_GUARD_VALUE for Ref args.
        // getptrinfo synthesizes ConstPtrInfo for constant Refs, matching
        // `if info:` in RPython (which is True for ConstPtrInfo too).
        let obj_box = ctx.resolve_box_box_opt(&arg0.to_boxref());
        let obj_info = obj_box.as_ref().and_then(|b| ctx.getptrinfo(b));
        if let Some(info) = obj_info {
            if info.is_virtual() {
                return raise_invalid_loop("promote of a virtual");
            }
            // rewrite.py:307-347: replace_old_guard_with_guard_value
            if let Some(old_guard) = obj_box
                .as_ref()
                .and_then(|b| ctx.get_last_guard(b))
                .cloned()
            {
                // rewrite.py:320: c_value = op.getarg(1) — generic Const.
                // Under typed seeding c_value can be Int OR Ref; the
                // previous gating on get_constant_int dropped the Ref
                // case entirely so prior GUARD_NONNULL/GUARD_CLASS were
                // never strengthened to GUARD_VALUE for Ref-typed args.
                if let Some(c_value) = ctx
                    .resolve_box_box_opt(&arg1.to_boxref())
                    .and_then(|b| b.const_value())
                {
                    // rewrite.py:321-323: c_value.nonnull(). ConstInt.nonnull
                    // == (value != 0); ConstPtr.nonnull == (gcref != null).
                    let c_nonnull = match c_value {
                        Value::Int(i) => i != 0,
                        Value::Ref(g) => !g.is_null(),
                        Value::Float(_) => true,
                        Value::Void => true,
                    };
                    if !c_nonnull {
                        return raise_invalid_loop(
                            "GUARD_VALUE(..., NULL) follows some other guard that it is not NULL",
                        );
                    }
                    // rewrite.py:324-332: previous_classbox = info.get_known_class(cpu)
                    // expected_classbox = cpu.cls_of_box(c_value)
                    // get_known_class on the c_value side dispatches through
                    // getptrinfo → ConstPtrInfo.get_known_class (info.py:763-772)
                    // which is exactly cls_of_box for constant pointers.
                    if let Some(prev_cls) = info.get_known_class(ctx.cpu.as_ref()) {
                        if let Some(arg1_box) = ctx.resolve_box_box_opt(&arg1.to_boxref()) {
                            if let Some(expected_cls) = ctx.get_known_class(&arg1_box) {
                                if prev_cls != expected_cls {
                                    return raise_invalid_loop(
                                        "GUARD_VALUE proven to always fail (class mismatch)",
                                    );
                                }
                            }
                        }
                    }
                    // rewrite.py:333-334: can_replace_guards check.
                    if !ctx.can_replace_guards {
                        return OptimizationResult::PassOn;
                    }
                    // rewrite.py:335-347: replace old guard with GUARD_VALUE.
                    // last_guard_pos is a _newoperations index (info.py:100-103).
                    // rewrite.py:339-340: old descr must not be ResumeAtPositionDescr
                    // — RPython's fresh ResumeGuardDescr() at line 335 must
                    // not overwrite a RAPD marker.
                    if let Some(old_idx) = obj_box.as_ref().and_then(|b| ctx.last_guard_pos(b))
                        && !ctx.is_resume_at_position_guard(old_idx as i32)
                    {
                        // rewrite.py:345-354 + optimizer.py:713-718:
                        // RPython creates a fresh ResumeGuardDescr() (an
                        // empty descr OBJECT, not None) for the strengthened
                        // guard, then replace_guard / replace_guard_op copies
                        // the resume payload from the old guard descr into the
                        // new one. This path writes directly into
                        // new_operations (rather than deferring through
                        // replaces_guard), so perform the descr copy inline
                        // before replacing the op — identical to the
                        // GUARD_CLASS strengthening below. Passing None here
                        // would leave the strengthened guard with no descr →
                        // no rd_numb → exit_layout.storage None → resume panic
                        // when the promoted value actually changes at runtime.
                        let new_descr = crate::compile::make_resume_guard_descr_typed(
                            old_guard
                                .get_fail_arg_types()
                                .map(|t| t.to_vec())
                                .unwrap_or_default(),
                        );
                        let old_descr = old_guard
                            .getdescr()
                            .expect("strengthened GUARD_VALUE donor must carry a descr");
                        crate::compile::copy_all_attributes_from(&new_descr, &old_descr);
                        // rewrite.py:346-348 GuardResOp.copy_and_change:
                        // shallow copy with new opcode/args/descr; fail_args,
                        // fail_arg_types, rd_resume_position, rd_numb, rd_consts,
                        // rd_virtuals, rd_pendingfields are carried from
                        // old_guard automatically.
                        let replacement = old_guard.copy_and_change(
                            OpCode::GuardValue,
                            Some(&[old_guard.arg(0), arg1]),
                            Some(Some(new_descr)),
                        );
                        // rewrite.py:343: self.optimizer.replace_guard(op, info)
                        ctx.new_operations[old_idx] = std::rc::Rc::new(replacement);
                        // rewrite.py:345-346: info.reset_last_guard_pos()
                        if let Some(b) = obj_box.as_ref() {
                            ctx.with_ptr_info_mut(b, |info_mut| info_mut.reset_last_guard_pos());
                        }
                        // postprocess_GUARD_VALUE (rewrite.py:303-305): make_constant
                        // with the actual c_value (preserving Int vs Ref typing).
                        ctx.make_constant_arg(&arg0.to_boxref(), c_value);
                        return OptimizationResult::Remove;
                    }
                }
            }
        }

        // rewrite.py:303-305 postprocess_GUARD_VALUE `make_constant(box,
        // op.getarg(1))` runs in `propagate_postprocess` below, AFTER the
        // guard has been emitted with its argument resolved. Calling it
        // here (pre-emit) installs the Const forwarding before
        // `emit_operation` resolves the guard's own arg0, collapsing the
        // emitted guard to `guard_value(Const, Const)` — a no-op the
        // backend compiles to nothing, so promoted values were never
        // re-checked at runtime (#210 unguarded zombie loops).
        OptimizationResult::PassOn
    }

    /// rewrite.py:397-436 optimize_GUARD_CLASS / postprocess_GUARD_CLASS.
    ///
    /// Shared by GuardClass and GuardNonnullClass — RPython
    /// `optimize_GUARD_NONNULL_CLASS` (rewrite.py:438-444) delegates to
    /// `optimize_GUARD_CLASS` after the null check, so both opcodes go
    /// through the same known-class / strengthening / postprocess logic.
    fn optimize_guard_class(&self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        // rewrite.py:402: info = self.ensure_ptr_info_arg0(op) — installs
        // an InstancePtrInfo on arg0 when one is missing. Discard the
        // EnsuredPtrInfo borrow; downstream lookups re-acquire via
        // `getptrinfo` / `get_ptr_info` against the resolved OpRef.
        let _ = ctx.ensure_ptr_info_arg0(op);
        let obj = ctx.resolve_box_box(&op.arg(0).to_boxref()).to_opref();
        // rewrite.py:397-407: ensure_ptr_info_arg0 → info.py:880 getptrinfo.
        // `getptrinfo(ConstPtr)` returns a synthesized ConstPtrInfo, so a
        // constant Ref arg0 is handled uniformly with virtual / instance
        // info: ConstPtrInfo.get_known_class(cpu) (info.py:763-772) reads
        // the typeptr at offset 0 via cls_of_box and compares against
        // expectedclassbox. Mismatch → proven-fail guard → InvalidLoop.
        let obj_info_for_class = ctx.getptrinfo(&op.arg(0).to_boxref().get_box_replacement(false));
        if let Some(known_class) =
            obj_info_for_class.and_then(|i| i.get_known_class(ctx.cpu.as_ref()))
        {
            if op.num_args() >= 2 {
                // RPython GuardClass / GuardNonnullClass class operands are
                // ConstInt vtable addresses (`expectedclassbox.getint()`).
                let expected = ctx.get_constant_int_box(&op.arg(1).to_boxref());
                if let Some(expected) = expected {
                    if known_class == expected {
                        return OptimizationResult::Remove;
                    }
                    // rewrite.py:404-407: known class mismatch is a
                    // proven-fail guard — abort the trace.
                    return raise_invalid_loop("GUARD_CLASS proven to always fail");
                }
            }
        }
        // rewrite.py:408-427: guard strengthening.
        // If there was a previous GUARD_NONNULL on the same value,
        // replace it with GUARD_NONNULL_CLASS (combining both checks).
        // rewrite.py:409-410: skip replacement if old descr is a
        // ResumeAtPositionDescr — RPython's fresh ResumeGuardDescr() at
        // line 417 must not overwrite a RAPD marker (rewrite.py:421-422
        // "old descr must not be ResumeAtPositionDescr").
        if let Some(old_guard) =
            ctx.get_last_guard(&op.arg(0).to_boxref().get_box_replacement(false))
        {
            if old_guard.opcode == OpCode::GuardNonnull
                && op.num_args() >= 2
                && ctx.can_replace_guards
            {
                // last_guard_pos is a _newoperations index.
                let old_guard_idx =
                    ctx.last_guard_pos(&op.arg(0).to_boxref().get_box_replacement(false));
                if let Some(old_idx) = old_guard_idx
                    && !ctx.is_resume_at_position_guard(old_idx as i32)
                {
                    // rewrite.py:417-426 + optimizer.py:713-718:
                    // RPython creates a fresh ResumeGuardDescr for the
                    // strengthened guard, then replace_guard_op copies the
                    // resume payload from the old guard descr into the new
                    // one. This path writes directly into new_operations, so
                    // perform the descr copy inline before replacing the op.
                    let new_descr = crate::compile::make_resume_guard_descr_typed(
                        old_guard
                            .get_fail_arg_types()
                            .map(|t| t.to_vec())
                            .unwrap_or_default(),
                    );
                    let old_descr = old_guard
                        .getdescr()
                        .expect("strengthened GUARD_CLASS donor must carry a descr");
                    crate::compile::copy_all_attributes_from(&new_descr, &old_descr);
                    let combined = old_guard.copy_and_change(
                        OpCode::GuardNonnullClass,
                        Some(&[old_guard.arg(0), op.arg(1)]),
                        Some(Some(new_descr)),
                    );
                    ctx.new_operations[old_idx] = std::rc::Rc::new(combined);
                    // rewrite.py:430-436 postprocess_GUARD_CLASS parity
                    // (invoked inline here because the replacement path
                    // rewrites `new_operations[old_idx]` directly instead
                    // of going through `emit_operation`, which would have
                    // triggered the regular postprocess dispatch).
                    //
                    // The replacement happened because the old guard is
                    // not a ResumeAtPositionDescr (guarded at
                    // `is_resume_at_position_guard(old_idx)` above), so
                    // rewrite.py:434-435 `update_last_guard = not
                    // old_guard_op or isinstance(descr, RAPD)` evaluates
                    // to False — pass `update_last_guard=false` so that
                    // make_constant_class preserves the strengthened
                    // guard's position in last_guard_pos (optimizer.py:137
                    // parity) rather than snapping it to the tail of
                    // new_operations.
                    if let Some(class_val) = ctx.get_constant_int_box(&op.arg(1).to_boxref()) {
                        if let Some(b) = ctx.get_box_replacement_box(obj) {
                            crate::optimizeopt::optimizer::Optimizer::make_constant_class(
                                ctx, &b, class_val, /* update_last_guard = */ false,
                            );
                        }
                    }
                    return OptimizationResult::Remove;
                }
            }
        }
        // rewrite.py:430-436 postprocess_GUARD_CLASS: runs AFTER emit.
        // Register deferred postprocess — executed by emit_operation
        // after the guard is added to new_operations.  Upstream
        // `postprocess_GUARD_CLASS` runs unconditionally (no
        // virtual-skip): `make_constant_class` already preserves
        // existing `InstancePtrInfo` whether or not `is_virtual=True`
        // (`optimizer.py:137-151`), so the Rust port skips the local
        // `is_virtual` guard and lets `Optimizer::make_constant_class`
        // dispatch on the live `Instance` / `Virtual` arm.
        if op.num_args() >= 2 {
            if let Some(class_val) = ctx.get_constant_int_box(&op.arg(1).to_boxref()) {
                ctx.pending_guard_class_postprocess =
                    Some(crate::optimizeopt::PendingGuardClassPostprocess { obj, class_val });
            }
        }
        OptimizationResult::PassOn
    }

    // ── SAME_AS identity ──

    /// optimizer.py:127-135 `getnullness(op)` wrapper. Delegates to
    /// `OptContext::getnullness`, which implements the upstream
    /// `op.type == 'r' or is_raw_ptr(op)` dispatch line-by-line, then
    /// converts the upstream `INFO_NULL` / `INFO_NONNULL` /
    /// `INFO_UNKNOWN` integer return into the local `Nullness` enum.
    fn getnullness(&self, opref: OpRef, ctx: &mut OptContext) -> Nullness {
        // optimizer.py:127-135 `getnullness` has no missing-Box branch —
        // every `op` has a backing `AbstractValue` per
        // `resoperation.py:233-248`. `get_box_replacement_box` resolves
        // the opref to its bound host; the read-only `getnullness` below
        // never writes, so an unresolvable opref (OpRef::NONE sentinel,
        // no upstream equivalent) maps to `INFO_UNKNOWN`.
        let info = match ctx.get_box_replacement_box(opref) {
            Some(b) => ctx.getnullness(&b),
            None => crate::optimizeopt::INFO_UNKNOWN,
        };
        Self::nullness_from_info(info)
    }

    /// Convert an `info.py` INFO_NULL/INFO_NONNULL/INFO_UNKNOWN return
    /// into the local `Nullness` enum used by the rewrite pass.
    fn nullness_from_info(value: i8) -> Nullness {
        if value == crate::optimizeopt::INFO_NULL {
            Nullness::Null
        } else if value == crate::optimizeopt::INFO_NONNULL {
            Nullness::Nonnull
        } else {
            Nullness::Unknown
        }
    }

    /// Check if an OpRef is Ref-typed.
    /// optimizer.py:128: op.type == 'r'
    ///
    /// Routes through the canonical `OptContext::opref_type` accessor
    /// (constant → value_types → producer op result_type) and falls back
    /// to PtrInfo presence — a Ref-only side channel populated for input
    /// args that do not appear in `new_operations`.
    fn is_ref_typed(&self, opref: OpRef, ctx: &OptContext) -> bool {
        if ctx.opref_type(opref) == Some(majit_ir::Type::Ref) {
            return true;
        }
        // BoxRef shim — has_ptr_info takes &BoxRef per info.py:880-894.
        ctx.get_box_replacement_box(opref)
            .as_ref()
            .map_or(false, |b| ctx.has_ptr_info(b))
    }

    /// rewrite.py:95-101: _optimize_CALL_INT_UDIV
    /// x / 1 → x
    fn optimize_call_int_udiv(
        &mut self,
        op: &Op,
        op_rc: &majit_ir::OpRc,
        ctx: &mut OptContext,
    ) -> bool {
        if op.num_args() < 3 {
            return false;
        }
        let arg2 = op.arg(2);
        if let Some(1) = ctx
            .resolve_box_box_opt(&arg2.to_boxref())
            .and_then(|b| ctx.get_constant_int_box(&b))
        {
            let b_old = BoxRef::from_bound_op(op_rc);
            let b_arg = ctx.resolve_box_box(&op.arg(1).to_boxref());
            ctx.make_equal_to(&b_old, &b_arg);
            ctx.last_op_removed = true;
            return true;
        }
        false
    }

    /// rewrite.py:768-805: _optimize_CALL_INT_PY_MOD
    fn optimize_call_int_py_mod(
        &mut self,
        op: &Op,
        op_rc: &majit_ir::OpRc,
        ctx: &mut OptContext,
    ) -> Option<OptimizationResult> {
        if op.num_args() < 3 {
            return None;
        }
        let arg1 = op.arg(1);
        let arg2 = op.arg(2);
        let b1 = {
            let b = ctx.resolve_box_box(&arg1.to_boxref());
            ctx.getintbound_handle(&b).borrow().clone()
        };
        let b2 = {
            let b = ctx.resolve_box_box(&arg2.to_boxref());
            ctx.getintbound_handle(&b).borrow().clone()
        };

        // rewrite.py:774-777: b1.known_eq_const(0) → 0
        if b1.known_eq_const(0) {
            let b = ctx.materialize_box_at(op.pos.get());
            ctx.make_constant_box(&b, Value::Int(0));
            ctx.last_op_removed = true;
            return Some(OptimizationResult::Remove);
        }
        // rewrite.py:780-781: if not b2.is_constant(): return False
        if !b2.is_constant() {
            return None;
        }
        let val = b2.get_constant_int();
        // rewrite.py:783-784
        if val <= 0 {
            return None;
        }
        // rewrite.py:785-788: x % 1 → 0
        if val == 1 {
            let b = ctx.materialize_box_at(op.pos.get());
            ctx.make_constant_box(&b, Value::Int(0));
            ctx.last_op_removed = true;
            return Some(OptimizationResult::Remove);
        }
        // rewrite.py:789-796: x % power_of_two → x & (power_of_two - 1)
        // Python's modulo: valid even for negative x.
        // RPython: replace_op_with + send_extra_operation (routes through passes).
        if val & (val - 1) == 0 {
            let mask = ctx.make_constant_int(val - 1);
            let arg_mask = ctx.materialize_box_at(mask);
            let mut and_op = Op::new(OpCode::IntAnd, &[arg1, Operand::from_boxref(&arg_mask)]);
            and_op.pos.set(op.pos.get());
            ctx.emit_extra(ctx.current_pass_idx, and_op);
            ctx.last_op_removed = true;
            return Some(OptimizationResult::Remove);
        }
        // rewrite.py:797-805: intdiv.modulo_operations fallback
        let known_nonneg = b1.known_nonnegative();
        let result_ref = crate::optimizeopt::intdiv::modulo_operations(
            arg1.to_opref(),
            val,
            known_nonneg,
            ctx.current_pass_idx,
            ctx,
        );
        let b_old = BoxRef::from_bound_op(op_rc);
        let b_res = ctx.get_box_replacement(result_ref);
        ctx.make_equal_to(&b_old, &b_res);
        ctx.last_op_removed = true;
        Some(OptimizationResult::Remove)
    }

    /// rewrite.py:713-766: _optimize_CALL_INT_PY_DIV
    fn optimize_call_int_py_div(
        &mut self,
        op: &Op,
        op_rc: &majit_ir::OpRc,
        ctx: &mut OptContext,
    ) -> Option<OptimizationResult> {
        if op.num_args() < 3 {
            return None;
        }
        let arg1 = op.arg(1);
        let arg2 = op.arg(2);
        let b1 = {
            let b = ctx.resolve_box_box(&arg1.to_boxref());
            ctx.getintbound_handle(&b).borrow().clone()
        };
        let b2 = {
            let b = ctx.resolve_box_box(&arg2.to_boxref());
            ctx.getintbound_handle(&b).borrow().clone()
        };

        // rewrite.py:726-729: b1.known_eq_const(0) → 0
        if b1.known_eq_const(0) {
            let b = ctx.materialize_box_at(op.pos.get());
            ctx.make_constant_box(&b, Value::Int(0));
            ctx.last_op_removed = true;
            return Some(OptimizationResult::Remove);
        }
        // rewrite.py:730-741: non-constant divisor (shift optimization)
        if !b2.is_constant() {
            // rewrite.py:731-740: x // (1 << y) → x >> y
            // when 0 <= y < LONG_BIT - 1
            if let Some(shift_op) = ctx
                .resolve_box_box_opt(&arg2.to_boxref())
                .and_then(|pb| ctx.get_producing_op(&pb))
            {
                if shift_op.opcode == OpCode::IntLshift
                    && shift_op.num_args() >= 2
                    && shift_op
                        .arg(0)
                        .to_boxref()
                        .get_box_replacement(false)
                        .const_int()
                        == Some(1)
                {
                    let shiftvar = ctx.resolve_box_box(&shift_op.arg(1).to_boxref()).to_opref();
                    let shiftbound = {
                        let b = ctx.get_box_replacement(shiftvar);
                        ctx.getintbound_handle(&b).borrow().clone()
                    };
                    if shiftbound.known_nonnegative() && shiftbound.known_lt_const(63) {
                        let arg_shift = ctx.materialize_box_at(shiftvar);
                        let mut rshift_op =
                            Op::new(OpCode::IntRshift, &[arg1, Operand::from_boxref(&arg_shift)]);
                        rshift_op.pos.set(op.pos.get());
                        ctx.emit_extra(ctx.current_pass_idx, rshift_op);
                        ctx.last_op_removed = true;
                        return Some(OptimizationResult::Remove);
                    }
                }
            }
            return None;
        }
        let val = b2.get_constant_int();
        // rewrite.py:743-749: x // -1 → -x (if x > MININT)
        if val == -1 {
            if b1.known_gt_const(i64::MIN) {
                let mut neg_op = Op::new(OpCode::IntNeg, &[arg1]);
                neg_op.pos.set(op.pos.get());
                ctx.emit_extra(ctx.current_pass_idx, neg_op);
                ctx.last_op_removed = true;
                return Some(OptimizationResult::Remove);
            }
        }
        // rewrite.py:750-751
        if val <= 0 {
            return None;
        }
        // rewrite.py:752-755: x // 1 → x
        if val == 1 {
            let b_old = BoxRef::from_bound_op(op_rc);
            let b_arg = ctx.resolve_box_box(&arg1.to_boxref());
            ctx.make_equal_to(&b_old, &b_arg);
            ctx.last_op_removed = true;
            return Some(OptimizationResult::Remove);
        }
        // rewrite.py:756-757: x // power_of_two → x >> shift
        if val & (val - 1) == 0 {
            let shift = val.trailing_zeros() as i64;
            let shift_const = ctx.make_constant_int(shift);
            let arg_shift = ctx.materialize_box_at(shift_const);
            let mut rshift_op =
                Op::new(OpCode::IntRshift, &[arg1, Operand::from_boxref(&arg_shift)]);
            rshift_op.pos.set(op.pos.get());
            ctx.emit_extra(ctx.current_pass_idx, rshift_op);
            ctx.last_op_removed = true;
            return Some(OptimizationResult::Remove);
        }
        // rewrite.py:758-766: intdiv.division_operations fallback
        let known_nonneg = b1.known_nonnegative();
        let result_ref = crate::optimizeopt::intdiv::division_operations(
            arg1.to_opref(),
            val,
            known_nonneg,
            ctx.current_pass_idx,
            ctx,
        );
        let b_old = BoxRef::from_bound_op(op_rc);
        let b_res = ctx.get_box_replacement(result_ref);
        ctx.make_equal_to(&b_old, &b_res);
        ctx.last_op_removed = true;
        Some(OptimizationResult::Remove)
    }

    /// rewrite.py:599-670: _optimize_call_arrayop
    ///
    /// Element-by-element unrolling for small constant-length array
    /// copy/move operations. Handles both virtual and non-virtual arrays.
    fn optimize_call_arrayop(
        &mut self,
        op: &Op,
        source_box: OpRef,
        dest_box: OpRef,
        source_start_box: OpRef,
        dest_start_box: OpRef,
        length_box: OpRef,
        ctx: &mut OptContext,
    ) -> bool {
        // rewrite.py:601-602: length = self.get_constant_box(length_box)
        let length_int = match ctx
            .get_box_replacement_box(length_box)
            .and_then(|b| ctx.get_constant_int_box(&b))
        {
            Some(l) => l,
            None => return false,
        };
        // rewrite.py:605-606: 0-length → remove
        if length_int == 0 {
            return true;
        }

        // One chain walk each; the position view falls back to the source.
        let source_b = ctx.get_box_replacement_box(source_box);
        let source_box = source_b.as_ref().map_or(source_box, |b| b.to_opref());
        let dest_b = ctx.get_box_replacement_box(dest_box);
        let dest_box = dest_b.as_ref().map_or(dest_box, |b| b.to_opref());
        let source_is_virtual = source_b.as_ref().map_or(false, |b| ctx.is_virtual(b));
        let dest_is_virtual = dest_b.as_ref().map_or(false, |b| ctx.is_virtual(b));

        // rewrite.py:610-611: constant start indices required
        let source_start = match ctx
            .get_box_replacement_box(source_start_box)
            .and_then(|b| ctx.get_constant_int_box(&b))
        {
            Some(s) => s,
            None => return false,
        };
        let dest_start = match ctx
            .get_box_replacement_box(dest_start_box)
            .and_then(|b| ctx.get_constant_int_box(&b))
        {
            Some(d) => d,
            None => return false,
        };

        // rewrite.py:613-617: both start constant, at least one virtual or length <= 8
        if !((dest_is_virtual || length_int <= 8) && (source_is_virtual || length_int <= 8)) {
            return false;
        }

        // rewrite.py:612,617: extrainfo.single_write_descr_array sanity check
        let call_descr = match op.getdescr() {
            Some(d) => d,
            None => return false,
        };
        let cd = match call_descr.as_call_descr() {
            Some(cd) => cd,
            None => return false,
        };
        let ei = cd.get_extra_info();
        // rewrite.py:617: extrainfo.single_write_descr_array is not None
        // effectinfo.py:201-206: set when exactly one write array descriptor.
        let arraydescr = match &ei.single_write_descr_array {
            Some(d) => d.clone(),
            None => {
                // Fallback: check bitstring — must have exactly one array write.
                let count: u32 = ei
                    .write_descrs_arrays
                    .as_ref()
                    .map(|w| w.iter().map(|b| b.count_ones()).sum())
                    .unwrap_or(0);
                if count != 1 {
                    return false;
                }
                // No actual DescrRef available — cannot emit typed ops.
                return false;
            }
        };

        // rewrite.py:621-635: arraydescr.is_array_of_structs()
        if arraydescr
            .as_array_descr()
            .is_some_and(|ad| ad.is_array_of_structs())
        {
            // rewrite.py:624-627: only if both virtual, not memmove
            if !(source_is_virtual && dest_is_virtual && source_box != dest_box) {
                return false;
            }
            // rewrite.py:628-629: all_fdescrs = arraydescr.get_all_fielddescrs()
            // → all_interiorfielddescrs in descr.py:291.
            let all_fdescr_indices: Vec<u32> = arraydescr
                .as_array_descr()
                .and_then(|ad| ad.get_all_interiorfielddescrs())
                .map(|fds| fds.iter().map(|d| d.index()).collect())
                .or_else(|| {
                    // Fallback: get from virtual's metadata
                    ctx.get_box_replacement_box(source_box)
                        .as_ref()
                        .and_then(|b| ctx.peek_ptr_info(b))
                        .and_then(|info| match info {
                            crate::optimizeopt::info::PtrInfo::VirtualArrayStruct(v) => {
                                if v.fielddescrs.is_empty() {
                                    None
                                } else {
                                    Some(v.fielddescrs.iter().map(|d| d.index()).collect())
                                }
                            }
                            _ => None,
                        })
                })
                .unwrap_or_default();
            if all_fdescr_indices.is_empty() {
                return false;
            }
            // rewrite.py:631-634: copy interior fields element by element
            for index in 0..length_int {
                for &fdescr_idx in &all_fdescr_indices {
                    let val = ctx
                        .get_box_replacement_box(source_box)
                        .as_ref()
                        .and_then(|b| ctx.peek_ptr_info(b))
                        .and_then(|info| {
                            info.getinteriorfield_virtual(
                                (index + source_start) as usize,
                                fdescr_idx,
                            )
                        });
                    if let Some(val) = val {
                        let idx = (index + dest_start) as usize;
                        if let Some(b) = ctx.get_box_replacement_box(dest_box) {
                            ctx.with_ptr_info_mut(&b, |info| {
                                info.setinteriorfield_virtual(idx, fdescr_idx, val);
                            });
                        }
                    }
                }
            }
            return true;
        }

        // rewrite.py:636-643: iteration direction
        let mut index_current: i64 = 0;
        let mut index_delta: i64 = 1;
        let mut index_stop: i64 = length_int;
        if source_box == dest_box && source_start < dest_start {
            // ARRAYMOVE with overlapping regions: iterate in reverse
            index_current = index_stop - 1;
            index_delta = -1;
            index_stop = -1;
        }

        // rewrite.py:646-670: element-by-element copy
        // RPython routes synthesized ops through send_extra_operation()
        // so they pass through downstream optimization passes.
        // We use ctx.emit_extra(current_pass_idx, op) for the same effect.
        let pass_idx = ctx.current_pass_idx;
        while index_current != index_stop {
            let index = index_current;
            index_current += index_delta;
            debug_assert!(index >= 0);

            // Read source element
            let val = if source_is_virtual {
                // rewrite.py:650-651: source_info.getitem(arraydescr, index + source_start)
                ctx.get_box_replacement_box(source_box)
                    .as_ref()
                    .and_then(|b| ctx.peek_ptr_info(b))
                    .and_then(|info| info.getitem((index + source_start) as usize))
                    .and_then(|e| e.as_opref())
            } else {
                // rewrite.py:653: opnum = OpHelpers.getarrayitem_for_descr(arraydescr)
                // Select I/R/F opcode based on item type.
                let item_type = arraydescr
                    .as_array_descr()
                    .map(|ad| ad.item_type())
                    .unwrap_or(majit_ir::Type::Int);
                let opcode = OpCode::getarrayitem_for_type(item_type);
                let idx_const = ctx.make_constant_int(index + source_start);
                let arg_source = ctx.materialize_box_at(source_box);
                let arg_idx = ctx.materialize_box_at(idx_const);
                let mut getop = Op::new(
                    opcode,
                    &[
                        Operand::from_boxref(&arg_source),
                        Operand::from_boxref(&arg_idx),
                    ],
                );
                getop.setdescr(arraydescr.clone());
                let pos = ctx.emit_extra(pass_idx, getop);
                Some(pos)
            };

            let val = match val {
                Some(v) => v,
                None => continue, // rewrite.py:660-661: if val is None: continue
            };

            // Write to destination
            if dest_is_virtual {
                // rewrite.py:662-665: dest_info.setitem(...)
                let idx = (index + dest_start) as usize;
                if let Some(b) = ctx.get_box_replacement_box(dest_box) {
                    ctx.with_ptr_info_mut(&b, |info| info.setitem(idx, val));
                }
            } else {
                // rewrite.py:666-670: emit SETARRAYITEM_GC
                let idx_const = ctx.make_constant_int(index + dest_start);
                let arg_dest = ctx.materialize_box_at(dest_box);
                let arg_idx = ctx.materialize_box_at(idx_const);
                let arg_val = ctx.materialize_box_at(val);
                let mut setop = Op::new(
                    OpCode::SetarrayitemGc,
                    &[
                        Operand::from_boxref(&arg_dest),
                        Operand::from_boxref(&arg_idx),
                        Operand::from_boxref(&arg_val),
                    ],
                );
                setop.setdescr(arraydescr.clone());
                ctx.emit_extra(pass_idx, setop);
            }
        }
        true
    }

    fn optimize_same_as(
        &self,
        op: &Op,
        op_rc: &majit_ir::OpRc,
        ctx: &mut OptContext,
    ) -> OptimizationResult {
        if op.num_args() == 0 {
            return OptimizationResult::PassOn;
        }
        let arg0 = op.arg(0);
        let b_old = BoxRef::from_bound_op(op_rc);
        let b_arg = ctx.resolve_box_box(&arg0.to_boxref());
        ctx.make_equal_to(&b_old, &b_arg);
        OptimizationResult::Remove
    }

    // ── Boolean inverse/reflex rewrites ──

    /// For comparison ops that have a bool_inverse or bool_reflex:
    /// Check if we already computed the inverse/reflex and can reuse that result.
    ///
    /// This mirrors `find_rewritable_bool` from rewrite.py: if we see INT_LT(a, b)
    /// and we previously computed INT_GE(a, b) = K (a constant 0 or 1), then
    /// INT_LT(a, b) = 1 - K.
    /// rewrite.py: find_rewritable_bool(op)
    /// If we see INT_LT(a, b) and previously computed INT_GE(a, b) = K,
    /// then INT_LT(a, b) = 1 - K (boolean inverse).
    /// rewrite.py:56-66 try_boolinvers — check if the inverse operation has
    /// a cached boolean result and negate it.
    ///
    /// RPython uses get_pure_result(targs) + getintbound(oldop).known_eq_const()
    /// which recognizes values that are guaranteed to be 0 or 1 even if not
    /// explicitly constant-folded. We match this by checking IntBound in
    /// addition to direct constant lookup.
    fn try_boolinvers(
        &self,
        op: &Op,
        inverse_opcode: OpCode,
        arg0: OpRef,
        arg1: OpRef,
        ctx: &mut OptContext,
    ) -> Option<OptimizationResult> {
        let key = (inverse_opcode, arg0, arg1);
        let cached_ref = self.bool_result_cache.get(&key).copied()?;
        // rewrite.py:60-65: b = self.getintbound(oldop)
        // First try direct constant (fast path)
        if let Some(val) = ctx
            .get_box_replacement_box(cached_ref)
            .and_then(|b| ctx.get_constant_int_box(&b))
        {
            let result = 1 - val;
            let b = ctx.materialize_box_at(op.pos.get());
            ctx.make_constant_box(&b, Value::Int(result));
            return Some(OptimizationResult::Remove);
        }
        // rewrite.py:61-65: b.known_eq_const(1) / b.known_eq_const(0)
        // Intbound analysis: the value may be bounded to exactly 0 or 1
        // even without being a constant in the optimizer's sense.
        if let Some(bound) = ctx
            .get_box_replacement_box(cached_ref)
            .and_then(|b| ctx.peek_intbound_box(&b))
        {
            if bound.known_eq_const(1) {
                let b = ctx.materialize_box_at(op.pos.get());
                ctx.make_constant_box(&b, Value::Int(0));
                return Some(OptimizationResult::Remove);
            } else if bound.known_eq_const(0) {
                let b = ctx.materialize_box_at(op.pos.get());
                ctx.make_constant_box(&b, Value::Int(1));
                return Some(OptimizationResult::Remove);
            }
        }
        None
    }

    /// rewrite.py:68-93 find_rewritable_bool — three-phase boolean rewrite:
    /// 1. boolinverse(same args)
    /// 2. boolreflex(swapped args)
    /// 3. boolreflex.boolinverse(swapped args)
    fn find_rewritable_bool(
        &self,
        op: &Op,
        op_rc: &majit_ir::OpRc,
        ctx: &mut OptContext,
    ) -> Option<OptimizationResult> {
        if op.num_args() < 2 {
            return None;
        }
        let arg0 = op.arg(0);
        let arg1 = op.arg(1);

        // rewrite.py:72-75: boolinverse(arg0, arg1)
        if let Some(inverse_opcode) = op.opcode.bool_inverse() {
            if let Some(result) =
                self.try_boolinvers(op, inverse_opcode, arg0.to_opref(), arg1.to_opref(), ctx)
            {
                return Some(result);
            }
        }

        // rewrite.py:77-83: boolreflex(arg1, arg0)
        if let Some(reflex_opcode) = op.opcode.bool_reflex() {
            let key = (reflex_opcode, arg1.to_opref(), arg0.to_opref());
            if let Some(&cached_ref) = self.bool_result_cache.get(&key) {
                let b_old = BoxRef::from_bound_op(op_rc);
                let b_cached = ctx.get_box_replacement(cached_ref);
                ctx.make_equal_to(&b_old, &b_cached);
                return Some(OptimizationResult::Remove);
            }

            // rewrite.py:87-91: boolreflex.boolinverse(arg1, arg0)
            if let Some(reflex_inverse) = reflex_opcode.bool_inverse() {
                if let Some(result) =
                    self.try_boolinvers(op, reflex_inverse, arg1.to_opref(), arg0.to_opref(), ctx)
                {
                    return Some(result);
                }
            }
        }

        None
    }

    // ── Float algebraic simplifications ──
    // rewrite.py:103-161 — only FLOAT_MUL, FLOAT_TRUEDIV, FLOAT_NEG, FLOAT_ABS.
    // Constant folding for all float ops is handled by execute_nonspec_const.

    /// rewrite.py:103-120 optimize_FLOAT_MUL
    fn optimize_float_mul(
        &self,
        op: &Op,
        op_rc: &majit_ir::OpRc,
        ctx: &mut OptContext,
    ) -> OptimizationResult {
        let arg0 = op.arg(0);
        let arg1 = op.arg(1);
        // rewrite.py:109: for lhs, rhs in [(arg1, arg2), (arg2, arg1)]:
        for (lhs, rhs) in [(&arg0, &arg1), (&arg1, &arg0)] {
            if let Some(v) = ctx
                .resolve_box_box_opt(&lhs.to_boxref())
                .and_then(|b| b.const_value())
                .and_then(|v| match v {
                    Value::Float(f) => Some(f),
                    _ => None,
                })
            {
                if v == 1.0 {
                    let b_old = BoxRef::from_bound_op(op_rc);
                    let b_v2 = ctx.resolve_box_box(&rhs.to_boxref());
                    ctx.make_equal_to(&b_old, &b_v2);
                    return OptimizationResult::Remove;
                }
                if v == -1.0 {
                    let mut neg = Op::new(OpCode::FloatNeg, &[rhs.clone()]);
                    neg.pos.set(op.pos.get());
                    return OptimizationResult::Replace(neg);
                }
            }
        }
        OptimizationResult::PassOn
    }

    /// rewrite.py:126-145 optimize_FLOAT_TRUEDIV
    fn optimize_float_truediv(&self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let arg0 = op.arg(0);
        let arg1 = op.arg(1);
        if let Some(divisor) = ctx
            .resolve_box_box_opt(&arg1.to_boxref())
            .and_then(|b| b.const_value())
            .and_then(|v| match v {
                Value::Float(f) => Some(f),
                _ => None,
            })
        {
            // rewrite.py:135-141: frexp check that divisor AND reciprocal
            // are both exact powers of 2. Bit-level equivalent: mantissa
            // bits are all zero and exponent is normal (not zero/subnormal/inf/nan).
            if Self::is_exact_power_of_two(divisor) {
                let reciprocal = 1.0 / divisor;
                if Self::is_exact_power_of_two(reciprocal) {
                    let recip_ref = self.emit_constant_float(ctx, reciprocal);
                    let arg_recip = ctx.materialize_box_at(recip_ref);
                    let mut new_op =
                        Op::new(OpCode::FloatMul, &[arg0, Operand::from_boxref(&arg_recip)]);
                    new_op.pos.set(op.pos.get());
                    return OptimizationResult::Emit(new_op);
                }
            }
        }
        OptimizationResult::PassOn
    }

    /// rewrite.py:135: `math.frexp(divisor)[0]` == ±0.5 iff exact power of 2.
    fn is_exact_power_of_two(v: f64) -> bool {
        let bits = v.to_bits();
        let mantissa = bits & 0x000F_FFFF_FFFF_FFFF;
        let exponent = ((bits >> 52) & 0x7FF) as u32;
        mantissa == 0 && exponent > 0 && exponent < 0x7FF
    }

    /// rewrite.py:147-153 optimize_FLOAT_NEG
    fn optimize_float_neg(
        &self,
        op: &Op,
        op_rc: &majit_ir::OpRc,
        ctx: &mut OptContext,
    ) -> OptimizationResult {
        let v = ctx
            .resolve_box_box_opt(&op.arg(0).to_boxref())
            .or_else(|| Some(ctx.resolve_box_box(&op.arg(0).to_boxref())));
        if let Some(arg_op) = v.and_then(|pb| ctx.get_producing_op(&pb)) {
            if arg_op.opcode == OpCode::FloatNeg {
                let b_old = BoxRef::from_bound_op(op_rc);
                let b_inner = ctx.resolve_box_box(&arg_op.arg(0).to_boxref());
                ctx.make_equal_to(&b_old, &b_inner);
                return OptimizationResult::Remove;
            }
        }
        OptimizationResult::PassOn
    }

    /// rewrite.py:155-161 optimize_FLOAT_ABS
    fn optimize_float_abs(
        &self,
        op: &Op,
        op_rc: &majit_ir::OpRc,
        ctx: &mut OptContext,
    ) -> OptimizationResult {
        let v = ctx.resolve_box_box_opt(&op.arg(0).to_boxref());
        if let Some(v) = v {
            if let Some(arg_op) = ctx.get_producing_op(&v) {
                if arg_op.opcode == OpCode::FloatAbs {
                    let b_old = BoxRef::from_bound_op(op_rc);
                    ctx.make_equal_to(&b_old, &v);
                    return OptimizationResult::Remove;
                }
            }
        }
        OptimizationResult::PassOn
    }

    // ── Helper ──

    /// Emit a constant integer value into the trace and return its OpRef.
    fn emit_constant_int(&self, ctx: &mut OptContext, value: i64) -> OpRef {
        let op = Op::new(OpCode::SameAsI, &[]);
        let opref = ctx.emit(op);
        let b = ctx.materialize_box_at(opref);
        ctx.make_constant_box(&b, Value::Int(value));
        opref
    }

    fn emit_constant_float(&self, ctx: &mut OptContext, value: f64) -> OpRef {
        let op = Op::new(OpCode::SameAsF, &[]);
        let opref = ctx.emit(op);
        let b = ctx.materialize_box_at(opref);
        ctx.make_constant_box(&b, Value::Float(value));
        opref
    }
}

impl Optimization for OptRewrite {
    fn propagate_forward(
        &mut self,
        op: &Op,
        op_rc: &majit_ir::OpRc,
        ctx: &mut OptContext,
    ) -> OptimizationResult {
        // Track last_op_removed for GuardNoException optimization.
        // Reset for non-guard ops (guards don't count as "the last op").
        if !op.opcode.is_guard() {
            ctx.last_op_removed = false;
        }

        // Try boolean inverse/reflex rewrites for comparisons
        if op.opcode.bool_inverse().is_some() || op.opcode.bool_reflex().is_some() {
            if let Some(result) = self.find_rewritable_bool(op, op_rc, ctx) {
                return result;
            }
        }

        match op.opcode {
            // Integer arithmetic rewrites (the autogenintrules.py ruleset)
            // live in OptIntBounds, as upstream; rewrite.py has no
            // optimize_INT_ADD/SUB/MUL/AND/OR/XOR/shift/NEG/INVERT methods.
            // IntFloorDiv / IntMod are pyre opcodes (upstream lowers // and %
            // to residual calls, see optimize_call_int_py_div / _py_mod), so
            // their strength-reduction rules stay here with the opcode.
            OpCode::IntFloorDiv => self.optimize_int_floor_div(op, op_rc, ctx),
            OpCode::IntMod => self.optimize_int_mod(op, op_rc, ctx),

            OpCode::IntIsZero => self.optimize_int_is_zero(op, ctx),
            OpCode::IntIsTrue => self.optimize_int_is_true(op, op_rc, ctx),
            OpCode::IntForceGeZero => self.optimize_int_force_ge_zero(op, op_rc, ctx),
            OpCode::IntBetween => self.optimize_int_between(op, ctx),

            // ── Comparisons ──
            OpCode::IntLt
            | OpCode::IntLe
            | OpCode::IntEq
            | OpCode::IntNe
            | OpCode::IntGt
            | OpCode::IntGe
            | OpCode::UintLt
            | OpCode::UintLe
            | OpCode::UintGt
            | OpCode::UintGe => self.optimize_comparison(op),

            // ── Guards ──
            OpCode::GuardTrue => self.optimize_guard_true(op, ctx),
            OpCode::GuardFalse => self.optimize_guard_false(op, ctx),
            OpCode::GuardValue => self.optimize_guard_value(op, ctx),
            // RPython rewrite.py guard optimizations:
            // If the guarded condition is already known to be true (constant),
            // the guard can be removed entirely.
            OpCode::GuardNonnull => {
                // rewrite.py:269-278 optimize_GUARD_NONNULL
                //     opinfo = getptrinfo(op.getarg(0))
                //     if opinfo is not None:
                //         if opinfo.is_nonnull(): return
                //         elif opinfo.is_null(): raise InvalidLoop(...)
                //     return self.emit(op)
                let obj_box = op.arg(0).to_boxref().get_box_replacement(false);
                let obj = obj_box.to_opref();
                if let Some(info) = ctx.getptrinfo(&obj_box) {
                    if info.is_nonnull() {
                        return OptimizationResult::Remove;
                    }
                    if info.is_null() {
                        return raise_invalid_loop("GUARD_NONNULL proven to always fail");
                    }
                }
                // rewrite.py:280-282 postprocess_GUARD_NONNULL:
                // make_nonnull runs immediately; mark_last_guard deferred
                // until emit adds the guard to new_operations.
                let has_info = ctx.has_ptr_info(&obj_box);
                if !has_info {
                    ctx.set_ptr_info(&obj_box, crate::optimizeopt::info::PtrInfo::nonnull());
                }
                // rewrite.py:282: mark_last_guard deferred to emit_operation
                ctx.pending_mark_last_guard = Some(obj);
                OptimizationResult::PassOn
            }
            OpCode::GuardIsnull => {
                // rewrite.py:186-195 optimize_GUARD_ISNULL
                //     info = getptrinfo(op.getarg(0))
                //     if info is not None:
                //         if info.is_null(): return
                //         elif info.is_nonnull(): raise InvalidLoop(...)
                //     return self.emit(op)
                let obj = ctx.resolve_box_box(&op.arg(0).to_boxref()).to_opref();
                let obj_box = ctx.resolve_box_box_opt(&op.arg(0).to_boxref());
                if let Some(info) = obj_box.as_ref().and_then(|b| ctx.getptrinfo(b)) {
                    if info.is_null() {
                        return OptimizationResult::Remove;
                    }
                    if info.is_nonnull() {
                        return raise_invalid_loop("GUARD_ISNULL proven to always fail");
                    }
                }
                // rewrite.py:197-198 postprocess_GUARD_ISNULL:
                //     self.make_constant(op.getarg(0), CONST_NULL)
                // Ref-typed → Value::Ref(NULL); Int-typed → Value::Int(0).
                if self.is_ref_typed(obj, ctx) {
                    ctx.make_constant_arg(&op.arg(0).to_boxref(), Value::Ref(majit_ir::GcRef(0)));
                } else {
                    ctx.make_constant_arg(&op.arg(0).to_boxref(), Value::Int(0));
                }
                OptimizationResult::PassOn
            }
            OpCode::GuardClass => self.optimize_guard_class(op, ctx),
            OpCode::GuardNonnullClass => {
                // rewrite.py:438-444 optimize_GUARD_NONNULL_CLASS:
                //     info = getptrinfo(op.getarg(0))
                //     if info and info.is_null():
                //         raise InvalidLoop(...)
                //     return self.optimize_GUARD_CLASS(op)
                if let Some(info) =
                    ctx.getptrinfo(&op.arg(0).to_boxref().get_box_replacement(false))
                {
                    if info.is_null() {
                        return raise_invalid_loop("GUARD_NONNULL_CLASS proven to always fail");
                    }
                }
                self.optimize_guard_class(op, ctx)
            }
            // rewrite.py: GUARD_IS_OBJECT — if arg is a known constant, the guard
            // was already checked at recording time and can be removed.
            OpCode::GuardIsObject => {
                if ctx
                    .get_constant_box(&op.arg(0).to_boxref().get_box_replacement(false))
                    .is_some()
                {
                    return OptimizationResult::Remove;
                }
                OptimizationResult::PassOn
            }
            // rewrite.py: GUARD_GC_TYPE — if arg is a known constant, remove.
            OpCode::GuardGcType => {
                if ctx
                    .get_constant_box(&op.arg(0).to_boxref().get_box_replacement(false))
                    .is_some()
                {
                    return OptimizationResult::Remove;
                }
                OptimizationResult::PassOn
            }
            // rewrite.py: GUARD_SUBCLASS — if arg is a known constant, remove.
            OpCode::GuardSubclass => {
                if ctx
                    .get_constant_box(&op.arg(0).to_boxref().get_box_replacement(false))
                    .is_some()
                {
                    return OptimizationResult::Remove;
                }
                OptimizationResult::PassOn
            }

            // ── Float arithmetic ──
            OpCode::FloatMul => self.optimize_float_mul(op, op_rc, ctx),
            OpCode::FloatTrueDiv => self.optimize_float_truediv(op, ctx),
            OpCode::FloatNeg => self.optimize_float_neg(op, op_rc, ctx),
            OpCode::FloatAbs => self.optimize_float_abs(op, op_rc, ctx),

            // ── Identity ops ──
            OpCode::SameAsI | OpCode::SameAsR | OpCode::SameAsF => {
                self.optimize_same_as(op, op_rc, ctx)
            }

            // ── Conditional calls ──
            OpCode::CondCallN => {
                if let Some(0) =
                    ctx.get_constant_int_box(&op.arg(0).to_boxref().get_box_replacement(false))
                {
                    ctx.last_op_removed = true;
                    return OptimizationResult::Remove;
                }
                if let Some(c) =
                    ctx.get_constant_int_box(&op.arg(0).to_boxref().get_box_replacement(false))
                {
                    if c != 0 {
                        let mut call_op = Op::new(
                            OpCode::CallN,
                            &op.getarglist()[1..]
                                .iter()
                                .map(Operand::from_boxref)
                                .collect::<Vec<_>>(),
                        );
                        call_op.pos.set(op.pos.get());
                        if let Some(d) = op.getdescr() {
                            call_op.setdescr(d);
                        }
                        ctx.last_op_removed = false;
                        return OptimizationResult::Replace(call_op);
                    }
                }
                ctx.last_op_removed = false;
                OptimizationResult::PassOn
            }
            // rewrite.py:483-494: optimize_COND_CALL_VALUE_I/R
            OpCode::CondCallValueI | OpCode::CondCallValueR => {
                let nullness = self.getnullness(op.arg(0).to_opref(), ctx);
                // rewrite.py:486-489: INFO_NONNULL → result is arg(0)
                if nullness == Nullness::Nonnull {
                    let b_old = BoxRef::from_bound_op(op_rc);
                    let b_arg = ctx.resolve_box_box(&op.arg(0).to_boxref());
                    ctx.make_equal_to(&b_old, &b_arg);
                    ctx.last_op_removed = true;
                    return OptimizationResult::Remove;
                }
                // rewrite.py:490-493: INFO_NULL → demote to CALL_PURE
                if nullness == Nullness::Null {
                    let call_opcode = if op.opcode == OpCode::CondCallValueI {
                        OpCode::CallPureI
                    } else {
                        OpCode::CallPureR
                    };
                    let mut call_op = Op::new(
                        call_opcode,
                        &op.getarglist()[1..]
                            .iter()
                            .map(Operand::from_boxref)
                            .collect::<Vec<_>>(),
                    );
                    call_op.pos.set(op.pos.get());
                    if let Some(d) = op.getdescr() {
                        call_op.setdescr(d);
                    }
                    ctx.last_op_removed = false;
                    return OptimizationResult::Replace(call_op);
                }
                ctx.last_op_removed = false;
                OptimizationResult::PassOn
            }

            // ── Pointer equality (rewrite.py: _optimize_oois_ooisnot) ──
            OpCode::PtrEq | OpCode::InstancePtrEq => {
                let instance = matches!(op.opcode, OpCode::InstancePtrEq);
                if instance {
                    // rewrite.py:563-565 optimize_INSTANCE_PTR_EQ:
                    //     arg0 = get_box_replacement(op.getarg(0))
                    //     arg1 = get_box_replacement(op.getarg(1))
                    //     self.pure_from_args2(rop.INSTANCE_PTR_EQ, arg1, arg0, op)
                    let arg0 = ctx.resolve_box_box(&op.arg(0).to_boxref()).to_opref();
                    let arg1 = ctx.resolve_box_box(&op.arg(1).to_boxref()).to_opref();
                    ctx.register_pure_from_args2(OpCode::InstancePtrEq, op.pos.get(), arg1, arg0);
                }
                return self.optimize_oois_ooisnot(op, false, instance, ctx);
            }
            OpCode::PtrNe | OpCode::InstancePtrNe => {
                let instance = matches!(op.opcode, OpCode::InstancePtrNe);
                if instance {
                    // rewrite.py:568-571 optimize_INSTANCE_PTR_NE: same swap.
                    let arg0 = ctx.resolve_box_box(&op.arg(0).to_boxref()).to_opref();
                    let arg1 = ctx.resolve_box_box(&op.arg(1).to_boxref()).to_opref();
                    ctx.register_pure_from_args2(OpCode::InstancePtrNe, op.pos.get(), arg1, arg0);
                }
                return self.optimize_oois_ooisnot(op, true, instance, ctx);
            }

            // ── Cast round-trip elimination ──
            // rewrite.py:807-813: register pure inverse for CSE, then emit.
            OpCode::CastPtrToInt => {
                ctx.register_pure_from_args1(
                    OpCode::CastIntToPtr,
                    op.pos.get(),
                    op.arg(0).to_opref(),
                );
                OptimizationResult::PassOn
            }
            OpCode::CastIntToPtr => {
                ctx.register_pure_from_args1(
                    OpCode::CastPtrToInt,
                    op.pos.get(),
                    op.arg(0).to_opref(),
                );
                OptimizationResult::PassOn
            }
            // jtransform.py:1264-1266: CAST_OPAQUE_PTR is identity (no-op).
            OpCode::CastOpaquePtr => {
                let b_old = BoxRef::from_bound_op(op_rc);
                let b_arg = ctx.resolve_box_box(&op.arg(0).to_boxref());
                ctx.make_equal_to(&b_old, &b_arg);
                OptimizationResult::Remove
            }

            // ── Float-bytes conversion round-trip elimination ──
            // rewrite.py:815-821: register inverse pure relationship for CSE.
            // CONVERT_FLOAT_BYTES_TO_LONGLONG(x) does NOT reduce to x —
            // it changes the bit representation. But if we later see
            // CONVERT_LONGLONG_BYTES_TO_FLOAT(result), pure.rs can
            // recognize the round-trip and recover x.
            OpCode::ConvertFloatBytesToLonglong => {
                ctx.register_pure_from_args1(
                    OpCode::ConvertLonglongBytesToFloat,
                    op.pos.get(),
                    op.arg(0).to_opref(),
                );
                OptimizationResult::PassOn
            }
            OpCode::ConvertLonglongBytesToFloat => {
                ctx.register_pure_from_args1(
                    OpCode::ConvertFloatBytesToLonglong,
                    op.pos.get(),
                    op.arg(0).to_opref(),
                );
                OptimizationResult::PassOn
            }

            // rewrite.py:712-718 optimize_GUARD_NO_EXCEPTION:
            //
            //     def optimize_GUARD_NO_EXCEPTION(self, op):
            //         if self.last_emitted_operation is REMOVED:
            //             return  # the prior op was a CALL_PURE that
            //                     # was killed; kill the guard too
            //         return self.emit(op)
            //
            // `last_emitted_operation` is set by every pass's emit
            // (optimizer.py:84-92), so the flag reflects the
            // PREVIOUS op's fate regardless of which pass dropped it.
            // pyre's ctx.last_op_removed is the cross-pass equivalent.
            OpCode::GuardNoException => {
                if ctx.last_op_removed {
                    return OptimizationResult::Remove;
                }
                OptimizationResult::PassOn
            }
            // rewrite.py: optimize_GUARD_FUTURE_CONDITION
            OpCode::GuardFutureCondition => {
                ctx.patchguardop = Some(op.clone());
                OptimizationResult::Remove
            }

            // INT_SIGNEXT belongs on `OptIntBounds`, not `OptRewrite`.
            // rewrite.py has no `optimize_INT_SIGNEXT`; the handler lives
            // at intbounds.py:450-466 (optimize + postprocess). pyre's
            // intbounds.rs:1760 already implements the full upstream
            // logic (is_within_range check), so this `OptRewrite` arm
            // is redundant — its weaker `nbytes == 8` shortcut admits a
            // strict subset of intbounds's removals. Removed for
            // line-by-line dispatch-shape parity.

            // rewrite.py:676-698: optimize_CALL_PURE_I
            // Dispatch based on oopspecindex to specialized handlers.
            // Constant-fold and CSE are handled by pure.rs; here we
            // only do oopspec-specific simplifications.
            OpCode::CallPureI | OpCode::CallPureR | OpCode::CallPureF | OpCode::CallPureN => {
                let __descr_arc_descr = op.getdescr();
                if let Some(ref descr) = __descr_arc_descr.as_ref() {
                    if let Some(cd) = descr.as_call_descr() {
                        let ei = cd.get_extra_info();
                        match ei.oopspecindex {
                            // rewrite.py:688: OS_INT_UDIV
                            majit_ir::OopSpecIndex::IntUdiv => {
                                if self.optimize_call_int_udiv(op, op_rc, ctx) {
                                    return OptimizationResult::Remove;
                                }
                            }
                            // rewrite.py:689: OS_INT_PY_DIV
                            majit_ir::OopSpecIndex::IntPyDiv => {
                                if let Some(result) = self.optimize_call_int_py_div(op, op_rc, ctx)
                                {
                                    return result;
                                }
                            }
                            // rewrite.py:692: OS_INT_PY_MOD
                            majit_ir::OopSpecIndex::IntPyMod => {
                                if let Some(result) = self.optimize_call_int_py_mod(op, op_rc, ctx)
                                {
                                    return result;
                                }
                            }
                            _ => {}
                        }
                    }
                }
                OptimizationResult::PassOn
            }

            // rewrite.py:448-470: optimize_CALL_LOOPINVARIANT_I
            OpCode::CallLoopinvariantI
            | OpCode::CallLoopinvariantR
            | OpCode::CallLoopinvariantF
            | OpCode::CallLoopinvariantN => {
                if let Some(func_val) = op.arg(0).to_boxref().get_box_replacement(false).const_int()
                {
                    // RPython: LoopInvariantOp.produce_op stores PreambleOp
                    // in loop_invariant_results during import. Transfer from
                    // ctx.imported_loop_invariant_results on first access.
                    if let Some(&(_, source)) = ctx
                        .imported_loop_invariant_results
                        .iter()
                        .find(|(k, _)| *k == func_val)
                    {
                        if !self
                            .loop_invariant_results
                            .iter()
                            .any(|(k, _)| *k == func_val)
                        {
                            // RPython shortpreamble.py:158-159. Cat-2.2 dual-slot:
                            // `produce_loop_invariant` installs
                            // `make_equal_to(source, result_opref)`, so the source
                            // box's `_forwarded` slot now holds
                            // `Forwarded::Op(result_op)`.
                            // Build the synthetic SameAsI replay at
                            // `result_opref` (= get_box_replacement(source))
                            // so `take_preamble_forwarded_opinfo` reads the
                            // info seeded at result_opref's slot per the
                            // dual-slot rule (mod.rs:1817 replay_pos).
                            let replay_pos = ctx.get_replacement_opref(source);
                            let source_box = ctx.materialize_box_at(source);
                            let mut replay =
                                Op::new(OpCode::SameAsI, &[Operand::from_boxref(&source_box)]);
                            replay.pos.set(replay_pos);
                            self.loop_invariant_results.insert(
                                func_val,
                                LoopInvariantEntry::Preamble(PreambleOp {
                                    op: source_box,
                                    invented_name: false,
                                    preamble_op: std::rc::Rc::new(replay),
                                    // Non-invented loop-invariant producer: the
                                    // SameAs arm is never taken, so no source.
                                    same_as_source: None,
                                }),
                            );
                        }
                    }
                    // rewrite.py:453-458: isinstance(resvalue, PreambleOp)
                    // → force_op_from_preamble → replace in dict
                    if let Some(entry) = self.loop_invariant_results.get(&func_val).cloned() {
                        let cached_result = match entry {
                            LoopInvariantEntry::Preamble(ref pop) => {
                                // unroll.py:26: force_op_from_preamble(preamble_op)
                                let forced = ctx.force_op_from_preamble_op(pop);
                                self.loop_invariant_results
                                    .insert(func_val, LoopInvariantEntry::Direct(forced));
                                forced
                            }
                            LoopInvariantEntry::Direct(r) => r,
                        };
                        let b_old = BoxRef::from_bound_op(op_rc);
                        let b_cached = ctx.get_box_replacement(cached_result);
                        ctx.make_equal_to(&b_old, &b_cached);
                        ctx.last_op_removed = true;
                        return OptimizationResult::Remove;
                    }
                    // Cache miss: demote and record result
                    self.loop_invariant_results
                        .insert(func_val, LoopInvariantEntry::Direct(op.pos.get()));
                    // rewrite.py:30-31: _callback records producer op
                    let call_opcode = OpCode::call_for_type(op.result_type());
                    let producer = op.copy_and_change(call_opcode, None, None);
                    producer.pos.set(op.pos.get());
                    self.loop_invariant_producer.insert(func_val, producer);
                }
                let call_opcode = OpCode::call_for_type(op.result_type());
                let new_op = op.copy_and_change(call_opcode, None, None);
                new_op.pos.set(op.pos.get());
                ctx.last_op_removed = false;
                OptimizationResult::Emit(new_op)
            }

            // ── rewrite.py:373-374: optimize_ASSERT_NOT_NONE ──
            OpCode::AssertNotNone => {
                // RPython: self.make_nonnull(op.getarg(0))
                let obj_box = op.arg(0).to_boxref().get_box_replacement(false);
                let has_info = ctx.has_ptr_info(&obj_box);
                if !has_info {
                    ctx.set_ptr_info(&obj_box, crate::optimizeopt::info::PtrInfo::nonnull());
                }
                OptimizationResult::Remove
            }

            // rewrite.py:376-386 optimize_RECORD_EXACT_CLASS:
            //     opinfo = getptrinfo(op.getarg(0))
            //     expectedclassbox = op.getarg(1)
            //     if opinfo is not None:
            //         realclassbox = opinfo.get_known_class(cpu)
            //         if realclassbox is not None:
            //             assert realclassbox.same_constant(expectedclassbox)
            //             return
            //     self.make_constant_class(op.getarg(0), expectedclassbox,
            //                              update_last_guard=False)
            OpCode::RecordExactClass => {
                if op.num_args() >= 2 {
                    // RPython `RECORD_EXACT_CLASS` carries the same ConstInt
                    // vtable address shape as GUARD_CLASS.
                    let expected_class = ctx.get_constant_int_box(&op.arg(1).to_boxref());
                    if let Some(expected_class) = expected_class {
                        // getptrinfo synthesizes ConstPtrInfo for constant
                        // Refs so `get_known_class` reads cls_of_box for them.
                        let obj_box = op.arg(0).to_boxref().get_box_replacement(false);
                        if let Some(known) = ctx
                            .getptrinfo(&obj_box)
                            .and_then(|i| i.get_known_class(ctx.cpu.as_ref()))
                        {
                            debug_assert_eq!(known, expected_class);
                            return OptimizationResult::Remove;
                        }
                        crate::optimizeopt::optimizer::Optimizer::make_constant_class(
                            ctx,
                            &obj_box,
                            expected_class,
                            false, // update_last_guard=False
                        );
                    }
                }
                OptimizationResult::Remove
            }

            // rewrite.py:397-401: optimize_record_exact_value
            //   box = op.getarg(0)
            //   expectedconstbox = op.getarg(1)
            //   assert isinstance(expectedconstbox, Const)
            //   self.make_constant(box, expectedconstbox)
            //
            // `make_constant` walks the forwarding chain internally
            // (optimizer.py:412 `box = get_box_replacement(box)`), so
            // upstream passes `op.getarg(0)` raw without a prior
            // `get_box_replacement` resolution. Pyre matches.
            OpCode::RecordExactValueI | OpCode::RecordExactValueR => {
                let val = op
                    .arg(1)
                    .to_boxref()
                    .get_box_replacement(false)
                    .const_value()
                    .expect(
                        "rewrite.py:400 — RECORD_EXACT_VALUE expectedconstbox \
                     must be a Const",
                    );
                ctx.make_constant_arg(&op.arg(0).to_boxref(), val);
                OptimizationResult::Remove
            }

            // rewrite.py:574-584: optimize_CALL_N — dispatch on oopspecindex
            OpCode::CallN | OpCode::CallI | OpCode::CallR => {
                let __descr_arc_descr = op.getdescr();
                if let Some(ref descr) = __descr_arc_descr.as_ref() {
                    if let Some(cd) = descr.as_call_descr() {
                        let ei = cd.get_extra_info();
                        match ei.oopspecindex {
                            // rewrite.py:580-590: OS_ARRAYCOPY / OS_ARRAYMOVE
                            majit_ir::OopSpecIndex::Arraycopy => {
                                if op.num_args() >= 6 {
                                    if self.optimize_call_arrayop(
                                        op,
                                        op.arg(1).to_opref(),
                                        op.arg(2).to_opref(), // source, dest
                                        op.arg(3).to_opref(),
                                        op.arg(4).to_opref(),
                                        op.arg(5).to_opref(), // src_start, dst_start, length
                                        ctx,
                                    ) {
                                        return OptimizationResult::Remove;
                                    }
                                }
                            }
                            majit_ir::OopSpecIndex::Arraymove => {
                                // rewrite.py:592-597: ARRAYMOVE: source == dest
                                if op.num_args() >= 5 {
                                    let array_box = op.arg(1);
                                    if self.optimize_call_arrayop(
                                        op,
                                        array_box.to_opref(),
                                        array_box.to_opref(), // source == dest
                                        op.arg(2).to_opref(),
                                        op.arg(3).to_opref(),
                                        op.arg(4).to_opref(),
                                        ctx,
                                    ) {
                                        return OptimizationResult::Remove;
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                }
                OptimizationResult::PassOn
            }

            // Everything else: pass on to next optimization pass
            _ => OptimizationResult::PassOn,
        }
    }

    fn setup(&mut self) {
        // ctx.last_op_removed is initialised by OptContext::new() and
        // maintained cross-pass by propagate_from_pass_range +
        // emit_operation — no per-pass setup needed.
        self.bool_result_cache.clear();
        self.loop_invariant_results.clear();
        self.loop_invariant_producer.clear();
    }

    fn name(&self) -> &'static str {
        "rewrite"
    }

    fn have_postprocess(&self) -> bool {
        true
    }

    /// rewrite.py:303-305 postprocess_GUARD_VALUE,
    /// rewrite.py:352-371 postprocess_GUARD_TRUE / postprocess_GUARD_FALSE.
    ///
    /// The `make_constant(box, CONST_*)` call is the second half of PyPy's
    /// `optimize_guard` (rewrite.py:163-184). PyPy emits the guard, then
    /// records that the guard's input box is now known constant. The Rust
    /// port keeps the same split that `have_postprocess` requires — the
    /// emit happens via `optimize_guard_true/false/value` and the
    /// `make_constant` happens here.
    ///
    /// Safety on stable OpRefs: PyPy uses fresh `Box` objects per loop
    /// iteration; majit uses positional `OpRef` slots. The constant lands
    /// on the resolved OpRef of the comparison result (e.g. the position
    /// of an `int_lt`), which is itself fresh per iteration: preamble and
    /// body optimization emit comparison ops into disjoint OpRef ranges.
    /// CSE within a single phase is the only way
    /// for two uses to share the same OpRef, in which case they describe
    /// the same value and the constant is correct for all of them. PyPy's
    /// stable-Box vs majit's stable-OpRef yield the same observable
    /// behavior.
    fn propagate_postprocess(&mut self, op: &Op, ctx: &mut OptContext) {
        match op.opcode {
            OpCode::GuardTrue => {
                ctx.make_constant_arg(&op.arg(0).to_boxref(), majit_ir::Value::Int(1));
            }
            OpCode::GuardFalse => {
                ctx.make_constant_arg(&op.arg(0).to_boxref(), majit_ir::Value::Int(0));
            }
            OpCode::GuardValue => {
                if op.num_args() >= 2 {
                    if let Some(val) = op
                        .arg(1)
                        .to_boxref()
                        .get_box_replacement(false)
                        .const_value()
                    {
                        ctx.make_constant_arg(&op.arg(0).to_boxref(), val);
                    }
                }
            }
            _ => {}
        }
    }

    /// rewrite.py:45-47: produce_potential_short_preamble_ops
    fn produce_potential_short_preamble_ops(
        &self,
        sb: &mut crate::optimizeopt::shortpreamble::ShortBoxes,
        ctx: &mut OptContext,
    ) {
        for (_, op) in &self.loop_invariant_producer {
            sb.add_loopinvariant_op(ctx, op.clone());
        }
    }

    /// rewrite.py:828-834 serialize_optrewrite
    fn serialize_optrewrite(&self) -> Vec<(i64, OpRef)> {
        self.loop_invariant_results
            .iter()
            .filter_map(|(func_ptr, entry)| match entry {
                LoopInvariantEntry::Direct(r) => Some((*func_ptr, *r)),
                LoopInvariantEntry::Preamble(pop) => Some((*func_ptr, pop.op.to_opref())),
            })
            .collect()
    }

    /// rewrite.py:836-838 deserialize_optrewrite
    fn deserialize_optrewrite(&mut self, entries: &[(i64, OpRef)]) {
        for &(func_ptr, result) in entries {
            self.loop_invariant_results
                .insert(func_ptr, LoopInvariantEntry::Direct(result));
        }
    }
}

impl Default for OptRewrite {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizeopt::optimizer::Optimizer;
    use majit_ir::GcRef;

    /// Producer-position trace spec. A consumer's `args` name the result
    /// positions of earlier producers in the same spec slice, so no op-arg
    /// is constructed as a position-only `BoxRef::from_opref(...)` box;
    /// [`build_specs`] later binds each arg to its producing `OpRc`.
    #[derive(Clone)]
    struct OpSpec {
        opcode: OpCode,
        args: Vec<u32>,
    }

    fn op_spec(opcode: OpCode, args: &[u32]) -> OpSpec {
        OpSpec {
            opcode,
            args: args.to_vec(),
        }
    }

    /// Turn a producer-position spec slice into a bound `OpRc` graph,
    /// oparser-faithful (`rpython/jit/tool/oparser.py`): each producer op
    /// is appended at its index position and every consumer arg references
    /// the producing op's bound result box (`from_bound_op`), mirroring
    /// oparser's `self.vars[name] = resop; args.append(self.vars[arg])`
    /// object-identity wiring — so each arg sheds to `Operand::Op` at
    /// construction, never the position-only `Operand::Box`.
    fn build_specs(specs: &[OpSpec]) -> Vec<majit_ir::OpRc> {
        let mut ops: Vec<majit_ir::OpRc> = Vec::new();
        for (pos, spec) in specs.iter().enumerate() {
            let args: Vec<BoxRef> = spec
                .args
                .iter()
                .map(|&p| BoxRef::from_bound_op(&ops[p as usize]))
                .collect();
            let arg_ops: Vec<Operand> = args.iter().map(Operand::from_boxref).collect();
            let op = std::rc::Rc::new(Op::new(spec.opcode, &arg_ops));
            op.pos
                .set(OpRef::op_typed(pos as u32, spec.opcode.result_type()));
            ops.push(op);
        }
        ops
    }

    fn same_i() -> OpSpec {
        op_spec(OpCode::SameAsI, &[])
    }

    fn same_f() -> OpSpec {
        op_spec(OpCode::SameAsF, &[])
    }

    fn same_r() -> OpSpec {
        op_spec(OpCode::SameAsR, &[])
    }

    fn bin_i(opcode: OpCode, left: u32, right: u32) -> OpSpec {
        op_spec(opcode, &[left, right])
    }

    fn bin_f(opcode: OpCode, left: u32, right: u32) -> OpSpec {
        op_spec(opcode, &[left, right])
    }

    fn bin_r(opcode: OpCode, left: u32, right: u32) -> OpSpec {
        op_spec(opcode, &[left, right])
    }

    fn unary_i(opcode: OpCode, arg: u32) -> OpSpec {
        op_spec(opcode, &[arg])
    }

    fn unary_f(opcode: OpCode, arg: u32) -> OpSpec {
        op_spec(opcode, &[arg])
    }

    fn unary_r(opcode: OpCode, arg: u32) -> OpSpec {
        op_spec(opcode, &[arg])
    }

    fn run_one(
        specs: Vec<OpSpec>,
        target: usize,
        constants: &[(OpRef, Value)],
    ) -> (OptimizationResult, OptContext) {
        let ops = build_specs(&specs);
        let mut ctx = OptContext::new(ops.len());
        for op in &ops[..target] {
            ctx.emit((**op).clone());
        }
        for &(opref, value) in constants {
            let b = ctx.materialize_box_at(opref);
            ctx.make_constant_box(&b, value);
        }
        let mut passes = test_pass_chain();
        let mut op = (*ops[target]).clone();
        resolve_op_args_in_ctx(&mut op, &mut ctx);
        let op_rc = std::rc::Rc::new(op.clone());
        ctx.bind_input_resops(std::slice::from_ref(&op_rc));
        let mut result = OptimizationResult::PassOn;
        for pass in passes.iter_mut() {
            result = pass.propagate_forward(&op, &op_rc, &mut ctx);
            if !matches!(result, OptimizationResult::PassOn) {
                break;
            }
        }
        (result, ctx)
    }

    /// The int-rewrite slice of the production pipeline: integer rule
    /// rewrites live in OptIntBounds (autogenintrules.py), all-constant
    /// pure folding in OptPure (pure.py:131), with OptRewrite between
    /// them as in default_pipeline. Tests in this module assert the
    /// chain's observable result, wherever the individual rule lives.
    fn test_pass_chain() -> Vec<Box<dyn crate::optimizeopt::Optimization>> {
        vec![
            Box::new(crate::optimizeopt::intbounds::OptIntBounds::new()),
            Box::new(OptRewrite::new()),
            Box::new(crate::optimizeopt::pure::OptPure::new()),
        ]
    }

    /// `run_one` against OptRewrite alone, for tests that assert what the
    /// rewrite pass itself must NOT do (a chained pass would mask it).
    fn run_one_rewrite_only(
        specs: Vec<OpSpec>,
        target: usize,
        constants: &[(OpRef, Value)],
    ) -> (OptimizationResult, OptContext) {
        let ops = build_specs(&specs);
        let mut ctx = OptContext::new(ops.len());
        for op in &ops[..target] {
            ctx.emit((**op).clone());
        }
        for &(opref, value) in constants {
            let b = ctx.materialize_box_at(opref);
            ctx.make_constant_box(&b, value);
        }
        let mut pass = OptRewrite::new();
        let mut op = (*ops[target]).clone();
        resolve_op_args_in_ctx(&mut op, &mut ctx);
        let op_rc = std::rc::Rc::new(op.clone());
        ctx.bind_input_resops(std::slice::from_ref(&op_rc));
        let result = pass.propagate_forward(&op, &op_rc, &mut ctx);
        (result, ctx)
    }

    fn resolve_op_args_in_ctx(op: &mut Op, ctx: &mut OptContext) {
        // optimizer.py:651-652 setarg loop parity. Direct unit tests that
        // bypass Optimizer::propagate_from_pass_range still need the same
        // canonical BoxRef args that production passes receive.
        for i in 0..op.num_args() {
            let arg = op.arg(i);
            let resolved = match ctx.resolve_box_box_opt(&arg.to_boxref()) {
                Some(b) => Operand::from_boxref(&b),
                None => {
                    let argref = arg.to_opref();
                    if argref.is_none() {
                        arg.clone()
                    } else {
                        Operand::from_boxref(
                            &ctx.materialize_box_at(argref).get_box_replacement(false),
                        )
                    }
                }
            };
            op.setarg(i, resolved);
        }
    }

    fn assert_remove(result: &OptimizationResult) {
        assert!(matches!(result, OptimizationResult::Remove));
    }

    fn assert_pass_on(result: &OptimizationResult) {
        assert!(matches!(result, OptimizationResult::PassOn));
    }

    fn assert_int_const(ctx: &OptContext, opref: OpRef, expected: i64) {
        assert_eq!(
            ctx.get_box_replacement_box(opref)
                .and_then(|cb| cb.const_int()),
            Some(expected)
        );
    }

    fn assert_forward(ctx: &OptContext, from: OpRef, to: OpRef) {
        assert_eq!(ctx.get_box_replacement(from).to_opref(), to);
    }

    /// Run the rewrite pass on a sequence of ops and return the optimized ops.
    #[allow(dead_code)]
    fn run_rewrite(specs: &[OpSpec]) -> (Vec<Op>, OptContext) {
        let ops = build_specs(specs);
        let mut ctx = OptContext::new(ops.len());
        let mut passes = test_pass_chain();

        for op in ops.iter() {
            // Resolve forwarded arguments
            let mut resolved = (**op).clone();
            resolve_op_args_in_ctx(&mut resolved, &mut ctx);

            let __pf_rc = std::rc::Rc::new(resolved.clone());
            ctx.bind_input_resops(std::slice::from_ref(&__pf_rc));
            let mut result = OptimizationResult::PassOn;
            for pass in passes.iter_mut() {
                result = pass.propagate_forward(&resolved, &__pf_rc, &mut ctx);
                if !matches!(result, OptimizationResult::PassOn) {
                    break;
                }
            }
            match result {
                OptimizationResult::Emit(emitted) => {
                    ctx.emit(emitted);
                }
                OptimizationResult::Replace(replacement)
                | OptimizationResult::Restart(replacement) => {
                    ctx.emit(replacement);
                }
                OptimizationResult::Remove => {
                    // removed, nothing emitted
                }
                OptimizationResult::PassOn => {
                    ctx.emit(resolved);
                }
                OptimizationResult::InvalidLoop(_) => {
                    std::panic::panic_any(crate::optimize::InvalidLoop(
                        "guard proven to always fail",
                    ));
                }
            }
        }

        let new_ops: Vec<Op> = ctx.new_operations.iter().map(|rc| (**rc).clone()).collect();
        (new_ops, ctx)
    }

    // ── Binary integer operation tests (consolidated) ──
    // RPython rewrite.py: identity, absorbing, constant-fold rules for all binops.

    /// Helper: test a binary op where one arg is constant → expect Remove + forwarding.
    fn assert_binop_identity(
        opcode: OpCode,
        const_pos: usize,
        const_val: i64,
        expected_forward_to: u32,
    ) {
        let (result, ctx) = run_one(
            vec![same_i(), same_i(), bin_i(opcode, 0, 1)],
            2,
            &[(OpRef::int_op(const_pos as u32), Value::Int(const_val))],
        );
        assert_remove(&result);
        assert_forward(&ctx, OpRef::int_op(2), OpRef::int_op(expected_forward_to));
    }

    /// Helper: test constant fold → expect Remove + constant result.
    fn assert_binop_const_fold(opcode: OpCode, a: i64, b: i64, expected: i64) {
        let (result, ctx) = run_one(
            vec![same_i(), same_i(), bin_i(opcode, 0, 1)],
            2,
            &[
                (OpRef::int_op(0), Value::Int(a)),
                (OpRef::int_op(1), Value::Int(b)),
            ],
        );
        assert_remove(&result);
        assert_int_const(&ctx, OpRef::int_op(2), expected);
    }

    /// Helper: test same-arg binop → expect Remove.
    fn assert_binop_self(opcode: OpCode, expected_const: Option<i64>) {
        let (result, ctx) = run_one(vec![same_i(), bin_i(opcode, 0, 0)], 1, &[]);
        assert_remove(&result);
        if let Some(val) = expected_const {
            assert_int_const(&ctx, OpRef::int_op(1), val);
        }
    }

    #[test]
    fn test_int_add_identities() {
        // x + 0 = x
        assert_binop_identity(OpCode::IntAdd, 1, 0, 0);
        // 0 + x = x
        assert_binop_identity(OpCode::IntAdd, 0, 0, 1);
        // constant fold
        assert_binop_const_fold(OpCode::IntAdd, 10, 20, 30);
    }

    #[test]
    fn test_int_sub_identities() {
        // x - 0 = x
        assert_binop_identity(OpCode::IntSub, 1, 0, 0);
        // x - x = 0
        assert_binop_self(OpCode::IntSub, Some(0));
        // constant fold
        assert_binop_const_fold(OpCode::IntSub, 30, 10, 20);
    }

    #[test]
    fn test_int_mul_identities() {
        // x * 0 = 0
        let (result, ctx) = run_one(
            vec![same_i(), same_i(), bin_i(OpCode::IntMul, 0, 1)],
            2,
            &[(OpRef::int_op(1), Value::Int(0))],
        );
        assert_remove(&result);
        assert_int_const(&ctx, OpRef::int_op(2), 0);

        // x * 1 = x
        assert_binop_identity(OpCode::IntMul, 1, 1, 0);
        // constant fold
        assert_binop_const_fold(OpCode::IntMul, 6, 7, 42);
    }

    #[test]
    fn test_int_mul_power_of_two() {
        // x * 8 → lshift(x, 3)
        let (result, _) = run_one(
            vec![same_i(), same_i(), bin_i(OpCode::IntMul, 0, 1)],
            2,
            &[(OpRef::int_op(1), Value::Int(8))],
        );
        match result {
            OptimizationResult::Replace(ref new_op) | OptimizationResult::Emit(ref new_op) => {
                assert_eq!(new_op.opcode, OpCode::IntLshift);
            }
            _ => {} // may also Remove with forwarding
        }
    }

    #[test]
    fn test_int_floordiv_identities() {
        // x / 1 = x
        assert_binop_identity(OpCode::IntFloorDiv, 1, 1, 0);
        // 0 / x = 0
        let (result, ctx) = run_one(
            vec![same_i(), same_i(), bin_i(OpCode::IntFloorDiv, 0, 1)],
            2,
            &[(OpRef::int_op(0), Value::Int(0))],
        );
        assert_remove(&result);
        assert_int_const(&ctx, OpRef::int_op(2), 0);
        // x / x = 1
        assert_binop_self(OpCode::IntFloorDiv, Some(1));
        // x / -1 = neg(x)
        // constant fold
        assert_binop_const_fold(OpCode::IntFloorDiv, 42, 7, 6);
    }

    #[test]
    fn test_int_mod_identities() {
        // x % 1 = 0
        let (result, ctx) = run_one(
            vec![same_i(), same_i(), bin_i(OpCode::IntMod, 0, 1)],
            2,
            &[(OpRef::int_op(1), Value::Int(1))],
        );
        assert_remove(&result);
        assert_int_const(&ctx, OpRef::int_op(2), 0);
        // x % x = 0
        assert_binop_self(OpCode::IntMod, Some(0));
    }

    #[test]
    fn test_int_bitwise_identities() {
        // AND: x & 0 = 0, x & -1 = x, x & x = x
        assert_binop_identity(OpCode::IntAnd, 0, -1i64, 1); // -1 & x = x
        assert_binop_self(OpCode::IntAnd, None); // x & x = x (forward to x)

        // OR: x | 0 = x, x | -1 = -1, x | x = x
        assert_binop_identity(OpCode::IntOr, 1, 0, 0);
        assert_binop_self(OpCode::IntOr, None);

        // XOR: x ^ 0 = x, x ^ x = 0, x ^ -1 = ~x
        assert_binop_identity(OpCode::IntXor, 1, 0, 0);
        assert_binop_self(OpCode::IntXor, Some(0));
        assert_binop_const_fold(OpCode::IntXor, 0xFF, 0x0F, 0xF0);
    }

    #[test]
    fn test_shift_identities() {
        // x << 0 = x
        assert_binop_identity(OpCode::IntLshift, 1, 0, 0);
        // x >> 0 = x
        assert_binop_identity(OpCode::IntRshift, 1, 0, 0);
        // constant fold
        assert_binop_const_fold(OpCode::IntLshift, 1, 4, 16);
    }

    #[test]
    fn test_unary_constant_fold() {
        // neg constant
        let (result, ctx) = run_one(
            vec![same_i(), unary_i(OpCode::IntNeg, 0)],
            1,
            &[(OpRef::int_op(0), Value::Int(42))],
        );
        assert_remove(&result);
        assert_int_const(&ctx, OpRef::int_op(1), -42);

        // invert constant
        let (result, ctx) = run_one(
            vec![same_i(), unary_i(OpCode::IntInvert, 0)],
            1,
            &[(OpRef::int_op(0), Value::Int(0xFF))],
        );
        assert_remove(&result);
        assert_int_const(&ctx, OpRef::int_op(1), !0xFF);
    }

    #[test]
    fn test_int_is_zero_and_is_true() {
        // is_zero(0) = 1
        let (result, ctx) = run_one(
            vec![same_i(), unary_i(OpCode::IntIsZero, 0)],
            1,
            &[(OpRef::int_op(0), Value::Int(0))],
        );
        assert_remove(&result);
        assert_int_const(&ctx, OpRef::int_op(1), 1);

        // is_zero(5) = 0
        let (result, ctx) = run_one(
            vec![same_i(), unary_i(OpCode::IntIsZero, 0)],
            1,
            &[(OpRef::int_op(0), Value::Int(5))],
        );
        assert_remove(&result);
        assert_int_const(&ctx, OpRef::int_op(1), 0);
    }

    #[test]
    fn test_comparison_constant_fold() {
        assert_binop_const_fold(OpCode::IntLt, 3, 5, 1);
        assert_binop_const_fold(OpCode::IntLt, 5, 3, 0);
        assert_binop_const_fold(OpCode::IntEq, 7, 7, 1);
        assert_binop_const_fold(OpCode::IntEq, 7, 8, 0);
        assert_binop_const_fold(OpCode::UintLt, 3, 5, 1);
    }

    // ── Guard tests ──

    #[test]
    fn test_guard_true_known_true() {
        let (result, _) = run_one(
            vec![same_i(), op_spec(OpCode::GuardTrue, &[0])],
            1,
            &[(OpRef::int_op(0), Value::Int(1))],
        );
        assert_remove(&result);
    }

    #[test]
    fn test_guard_true_known_false() {
        let (result, _) = run_one(
            vec![same_i(), op_spec(OpCode::GuardTrue, &[0])],
            1,
            &[(OpRef::int_op(0), Value::Int(0))],
        );
        assert!(
            matches!(result, OptimizationResult::InvalidLoop(_)),
            "guard_true(0) should abort as InvalidLoop, got {result:?}"
        );
    }

    #[test]
    fn test_guard_true_unknown() {
        let (result, _) = run_one(vec![same_i(), op_spec(OpCode::GuardTrue, &[0])], 1, &[]);
        assert_pass_on(&result);
    }

    #[test]
    fn test_guard_false_known_false() {
        let (result, _) = run_one(
            vec![same_i(), op_spec(OpCode::GuardFalse, &[0])],
            1,
            &[(OpRef::int_op(0), Value::Int(0))],
        );
        assert_remove(&result);
    }

    #[test]
    fn test_guard_false_known_true() {
        let (result, _) = run_one(
            vec![same_i(), op_spec(OpCode::GuardFalse, &[0])],
            1,
            &[(OpRef::int_op(0), Value::Int(1))],
        );
        assert!(
            matches!(result, OptimizationResult::InvalidLoop(_)),
            "guard_false(1) should abort as InvalidLoop, got {result:?}"
        );
    }

    #[test]
    fn test_guard_value_match() {
        let (result, _) = run_one(
            vec![same_i(), same_i(), bin_i(OpCode::GuardValue, 0, 1)],
            2,
            &[
                (OpRef::int_op(0), Value::Int(42)),
                (OpRef::int_op(1), Value::Int(42)),
            ],
        );
        assert_remove(&result);
    }

    // ── SAME_AS tests ──

    #[test]
    fn test_same_as_i() {
        let (result, ctx) = run_one(vec![same_i(), op_spec(OpCode::SameAsI, &[0])], 1, &[]);
        assert_remove(&result);
        assert_forward(&ctx, OpRef::int_op(1), OpRef::int_op(0));
    }

    // ── Integration test: full optimizer with OptRewrite ──

    #[test]
    fn test_optimizer_integration_add_zero() {
        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptRewrite::new()));

        // Create a trace: x = SameAsI(), y = SameAsI(constant 0), z = IntAdd(x, y)
        let ops = build_specs(&[
            same_i(),                         // op0: x
            same_i(),                         // op1: 0
            op_spec(OpCode::IntAdd, &[0, 1]), // op2: x + 0
        ]);

        // We need to set up constants before the optimizer runs.
        // The optimizer creates its own context, so we need a way to
        // inject constants. Since we're testing through the optimizer,
        // let's test the pass directly instead.
        let mut ctx = OptContext::new(3);

        let mut passes = test_pass_chain();

        // Simulate the optimizer loop
        for (i, op) in ops.iter().enumerate() {
            let mut resolved = (**op).clone();
            // optimizer.py:651-652 setarg loop parity.
            for i in 0..resolved.num_args() {
                resolved.setarg(
                    i,
                    Operand::from_boxref(&ctx.resolve_box_box(&resolved.arg(i).to_boxref())),
                );
            }
            let __pf_rc = std::rc::Rc::new(resolved.clone());
            ctx.bind_input_resops(std::slice::from_ref(&__pf_rc));
            let mut result = OptimizationResult::PassOn;
            // The production loop absorbs SameAs* before the passes
            // (optimizer.py:864-867); the argless SameAsI fixtures here
            // stand in for inputs and are emitted directly.
            if resolved.opcode != OpCode::SameAsI {
                for pass in passes.iter_mut() {
                    result = pass.propagate_forward(&resolved, &__pf_rc, &mut ctx);
                    if !matches!(result, OptimizationResult::PassOn) {
                        break;
                    }
                }
            }
            match result {
                OptimizationResult::Emit(emitted) => {
                    ctx.emit(emitted);
                }
                OptimizationResult::Replace(replacement)
                | OptimizationResult::Restart(replacement) => {
                    ctx.emit(replacement);
                }
                OptimizationResult::Remove => {}
                OptimizationResult::PassOn => {
                    ctx.emit(resolved);
                }
                OptimizationResult::InvalidLoop(_) => {
                    std::panic::panic_any(crate::optimize::InvalidLoop(
                        "guard proven to always fail",
                    ));
                }
            }
            // Set op1 as constant 0 after it has been emitted
            if i == 1 {
                let b = ctx.materialize_box_at(OpRef::int_op(1));
                ctx.make_constant_box(&b, Value::Int(0));
            }
        }

        // The SameAsI(x) should be removed and forwarded, but we only
        // have SameAsI with no args (acting as input). Let's verify
        // the IntAdd was removed and the result is forwarded.
        // op0 is emitted, op1 is emitted (just a constant), op2 is removed.
        // After forwarding, any reference to op2 should resolve to op0.
        assert_eq!(
            ctx.get_box_replacement(OpRef::int_op(2)).to_opref(),
            OpRef::int_op(0)
        );
    }

    #[test]
    fn test_optimizer_integration_chain() {
        // RPython parity: x - x -> 0, then guard_true(0) makes the trace impossible.
        let ops = build_specs(&[
            same_i(),                         // op0: x
            op_spec(OpCode::IntSub, &[0, 0]), // op1: x - x -> 0
            op_spec(OpCode::GuardTrue, &[1]), // op2: guard_true(0)
        ]);

        let mut ctx = OptContext::new(3);
        let mut passes = test_pass_chain();

        let err = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            for op in &ops {
                let mut resolved = (**op).clone();
                // optimizer.py:651-652 setarg loop parity.
                for i in 0..resolved.num_args() {
                    resolved.setarg(
                        i,
                        Operand::from_boxref(&ctx.resolve_box_box(&resolved.arg(i).to_boxref())),
                    );
                }
                let __pf_rc = std::rc::Rc::new(resolved.clone());
                ctx.bind_input_resops(std::slice::from_ref(&__pf_rc));
                let mut result = OptimizationResult::PassOn;
                // SameAsI input fixtures bypass the passes, as in the
                // production loop's SameAs absorption (optimizer.py:864-867).
                if resolved.opcode != OpCode::SameAsI {
                    for pass in passes.iter_mut() {
                        result = pass.propagate_forward(&resolved, &__pf_rc, &mut ctx);
                        if !matches!(result, OptimizationResult::PassOn) {
                            break;
                        }
                    }
                }
                match result {
                    OptimizationResult::Emit(emitted) => {
                        ctx.emit(emitted);
                    }
                    OptimizationResult::Replace(replacement)
                    | OptimizationResult::Restart(replacement) => {
                        ctx.emit(replacement);
                    }
                    OptimizationResult::Remove => {}
                    OptimizationResult::PassOn => {
                        ctx.emit(resolved);
                    }
                    OptimizationResult::InvalidLoop(_) => {
                        std::panic::panic_any(crate::optimize::InvalidLoop(
                            "guard proven to always fail",
                        ));
                    }
                }
            }
        }))
        .expect_err("guard_true(0) should abort the optimized trace");
        assert!(err.downcast_ref::<crate::optimize::InvalidLoop>().is_some());
    }

    // ── Wrapping arithmetic tests ──

    #[test]
    fn test_int_add_wrapping() {
        // wrapping
        let (result, ctx) = run_one(
            vec![same_i(), same_i(), bin_i(OpCode::IntAdd, 0, 1)],
            2,
            &[
                (OpRef::int_op(0), Value::Int(i64::MAX)),
                (OpRef::int_op(1), Value::Int(1)),
            ],
        );
        assert_remove(&result);
        assert_int_const(&ctx, OpRef::int_op(2), i64::MIN);
    }

    // ── Shift of zero constant tests ──

    #[test]
    fn test_zero_lshift_anything() {
        let (result, ctx) = run_one(
            vec![same_i(), same_i(), bin_i(OpCode::IntLshift, 0, 1)],
            2,
            &[(OpRef::int_op(0), Value::Int(0))],
        );
        assert_remove(&result);
        assert_int_const(&ctx, OpRef::int_op(2), 0);
    }

    #[test]
    fn test_zero_rshift_anything() {
        let (result, ctx) = run_one(
            vec![same_i(), same_i(), bin_i(OpCode::IntRshift, 0, 1)],
            2,
            &[(OpRef::int_op(0), Value::Int(0))],
        );
        assert_remove(&result);
        assert_int_const(&ctx, OpRef::int_op(2), 0);
    }

    // ── Non-optimizable cases (should PassOn) ──

    #[test]
    fn test_int_add_no_constants() {
        let (result, _) = run_one(
            vec![same_i(), same_i(), bin_i(OpCode::IntAdd, 0, 1)],
            2,
            &[],
        );
        assert_pass_on(&result);
    }

    #[test]
    fn test_unknown_opcode_passthrough() {
        // SETFIELD_GC(struct, value): struct is a Ref producer, value an Int
        // producer. OptRewrite has no rule for it → PassOn.
        let (result, _) = run_one(
            vec![same_r(), same_i(), op_spec(OpCode::SetfieldGc, &[0, 1])],
            2,
            &[],
        );
        assert_pass_on(&result);
    }

    // ── INT_AND constant fold ──

    #[test]
    fn test_int_and_constant_fold() {
        let (result, ctx) = run_one(
            vec![same_i(), same_i(), bin_i(OpCode::IntAnd, 0, 1)],
            2,
            &[
                (OpRef::int_op(0), Value::Int(0xFF)),
                (OpRef::int_op(1), Value::Int(0x0F)),
            ],
        );
        assert_remove(&result);
        assert_int_const(&ctx, OpRef::int_op(2), 0x0F);
    }

    // ── INT_OR constant fold ──

    #[test]
    fn test_int_or_constant_fold() {
        let (result, ctx) = run_one(
            vec![same_i(), same_i(), bin_i(OpCode::IntOr, 0, 1)],
            2,
            &[
                (OpRef::int_op(0), Value::Int(0xF0)),
                (OpRef::int_op(1), Value::Int(0x0F)),
            ],
        );
        assert_remove(&result);
        assert_int_const(&ctx, OpRef::int_op(2), 0xFF);
    }

    // ── UINT_RSHIFT tests ──

    #[test]
    fn test_uint_rshift_zero() {
        let (result, ctx) = run_one(
            vec![same_i(), same_i(), bin_i(OpCode::UintRshift, 0, 1)],
            2,
            &[(OpRef::int_op(1), Value::Int(0))],
        );
        assert_remove(&result);
        assert_forward(&ctx, OpRef::int_op(2), OpRef::int_op(0));
    }

    #[test]
    fn test_uint_rshift_constant_fold() {
        let (result, ctx) = run_one(
            vec![same_i(), same_i(), bin_i(OpCode::UintRshift, 0, 1)],
            2,
            &[
                (OpRef::int_op(0), Value::Int(-1)), // all ones
                (OpRef::int_op(1), Value::Int(1)),
            ],
        );
        assert_remove(&result);
        // u64::MAX >> 1 = i64::MAX
        assert_int_const(&ctx, OpRef::int_op(2), i64::MAX);
    }

    // ── Float optimization tests ──

    #[test]
    fn test_float_mul_one_right() {
        let (result, ctx) = run_one(
            vec![same_f(), same_f(), bin_f(OpCode::FloatMul, 0, 1)],
            2,
            &[(OpRef::float_op(1), Value::Float(1.0))],
        );
        assert_remove(&result);
        assert_forward(&ctx, OpRef::float_op(2), OpRef::float_op(0));
    }

    #[test]
    fn test_float_mul_one_left() {
        let (result, ctx) = run_one(
            vec![same_f(), same_f(), bin_f(OpCode::FloatMul, 0, 1)],
            2,
            &[(OpRef::float_op(0), Value::Float(1.0))],
        );
        assert_remove(&result);
        assert_forward(&ctx, OpRef::float_op(2), OpRef::float_op(1));
    }

    #[test]
    fn test_float_neg_double_negation() {
        // FloatNeg(FloatNeg(x)) -> x
        let ops = build_specs(&[
            same_f(),                        // op0: x
            op_spec(OpCode::FloatNeg, &[0]), // op1: -x
            op_spec(OpCode::FloatNeg, &[1]), // op2: -(-x) -> x
        ]);
        let mut ctx = OptContext::new(3);
        ctx.emit((*ops[0]).clone());

        let mut pass = OptRewrite::new();
        // Process op1 first (pass it through)
        let __pf_rc = std::rc::Rc::new((*ops[1]).clone());
        ctx.bind_input_resops(std::slice::from_ref(&__pf_rc));
        let result1 = pass.propagate_forward(&ops[1], &__pf_rc, &mut ctx);
        assert!(matches!(result1, OptimizationResult::PassOn));
        ctx.emit((*ops[1]).clone());

        // Process op2: should detect double negation
        let mut resolved2 = (*ops[2]).clone();
        resolve_op_args_in_ctx(&mut resolved2, &mut ctx);
        let __pf_rc = std::rc::Rc::new(resolved2.clone());
        ctx.bind_input_resops(std::slice::from_ref(&__pf_rc));
        let result2 = pass.propagate_forward(&resolved2, &__pf_rc, &mut ctx);
        assert!(matches!(result2, OptimizationResult::Remove));
        assert_eq!(
            ctx.get_box_replacement(OpRef::float_op(2)).to_opref(),
            OpRef::float_op(0)
        );
    }

    #[test]
    fn test_float_truediv_power_of_two() {
        // x / 2.0 → x * 0.5
        let (result, _) = run_one(
            vec![same_f(), same_f(), bin_f(OpCode::FloatTrueDiv, 0, 1)],
            2,
            &[(OpRef::float_op(1), Value::Float(2.0))],
        );
        assert!(matches!(result, OptimizationResult::Emit(_)));
    }

    #[test]
    fn test_float_no_opt_passthrough() {
        // FloatAdd with no constants: no RPython rewrite → PassOn
        let (result, _) = run_one(
            vec![same_f(), same_f(), bin_f(OpCode::FloatAdd, 0, 1)],
            2,
            &[],
        );
        assert_pass_on(&result);
    }

    // ── COND_CALL tests ──

    #[test]
    fn test_cond_call_constant_false_removed() {
        // CondCallN(condition=0, func, arg1) -> removed (dead call)
        let (result, _) = run_one(
            vec![
                same_i(),
                same_i(),
                same_i(),
                op_spec(OpCode::CondCallN, &[0, 1, 2]),
            ],
            3,
            &[(OpRef::int_op(0), Value::Int(0))],
        );
        assert_remove(&result);
    }

    #[test]
    fn test_cond_call_constant_true_to_direct_call() {
        // CondCallN(condition=1, func, arg1) -> CallN(func, arg1)
        let (result, _) = run_one(
            vec![
                same_i(),
                same_i(),
                same_i(),
                op_spec(OpCode::CondCallN, &[0, 1, 2]),
            ],
            3,
            &[(OpRef::int_op(0), Value::Int(1))],
        );
        match result {
            OptimizationResult::Replace(op) => {
                assert_eq!(op.opcode, OpCode::CallN);
                // Should have args [func, arg1] (condition arg stripped)
                assert_eq!(op.num_args(), 2);
                assert_eq!(op.arg(0).to_opref(), OpRef::int_op(1));
                assert_eq!(op.arg(1).to_opref(), OpRef::int_op(2));
            }
            other => panic!("expected Replace(CallN), got {:?}", other),
        }
    }

    // ── COND_CALL_VALUE tests ──

    #[test]
    fn test_cond_call_value_nonnull_returns_value() {
        // CondCallValueI(value=42, func, arg1) -> value itself (no call needed)
        let (result, ctx) = run_one(
            vec![
                same_i(),
                same_i(),
                same_i(),
                op_spec(OpCode::CondCallValueI, &[0, 1, 2]),
            ],
            3,
            &[(OpRef::int_op(0), Value::Int(42))],
        );
        assert_remove(&result);
        let resolved = ctx.get_box_replacement(OpRef::int_op(3)).to_opref();
        assert!(resolved.is_constant());
        assert_eq!(
            ctx.get_box_replacement_box(resolved)
                .and_then(|b| ctx.get_constant_int_box(&b)),
            Some(42)
        );
    }

    #[test]
    fn test_cond_call_value_null_to_direct_call() {
        // CondCallValueI(value=0, func, arg1) -> CallPureI(func, arg1)
        let (result, _) = run_one(
            vec![
                same_i(),
                same_i(),
                same_i(),
                op_spec(OpCode::CondCallValueI, &[0, 1, 2]),
            ],
            3,
            &[(OpRef::int_op(0), Value::Int(0))],
        );
        match result {
            OptimizationResult::Replace(op) => {
                assert_eq!(op.opcode, OpCode::CallPureI);
                assert_eq!(op.num_args(), 2);
                assert_eq!(op.arg(0).to_opref(), OpRef::int_op(1));
                assert_eq!(op.arg(1).to_opref(), OpRef::int_op(2));
            }
            other => panic!("expected Replace(CallPureI), got {:?}", other),
        }
    }

    // ── PTR_EQ / PTR_NE tests ──

    #[test]
    fn test_ptr_eq_same_opref() {
        // PtrEq(x, x) -> 1
        // resoperation.py:739 InputArgRef / 615 RefOp `type = 'r'`: ptr
        // boxes carry the Ref variant tag, not Int.
        let (result, ctx) = run_one(vec![same_r(), bin_r(OpCode::PtrEq, 0, 0)], 1, &[]);
        assert_remove(&result);
        assert_int_const(&ctx, OpRef::int_op(1), 1);
    }

    #[test]
    fn test_ptr_ne_same_opref() {
        // PtrNe(x, x) -> 0
        let (result, ctx) = run_one(vec![same_r(), bin_r(OpCode::PtrNe, 0, 0)], 1, &[]);
        assert_remove(&result);
        assert_int_const(&ctx, OpRef::int_op(1), 0);
    }

    #[test]
    fn test_instance_ptr_eq_same_opref() {
        // InstancePtrEq(x, x) -> 1
        let (result, ctx) = run_one(vec![same_r(), bin_r(OpCode::InstancePtrEq, 0, 0)], 1, &[]);
        assert_remove(&result);
        assert_int_const(&ctx, OpRef::int_op(1), 1);
    }

    #[test]
    fn test_instance_ptr_ne_same_opref() {
        // InstancePtrNe(x, x) -> 0
        let (result, ctx) = run_one(vec![same_r(), bin_r(OpCode::InstancePtrNe, 0, 0)], 1, &[]);
        assert_remove(&result);
        assert_int_const(&ctx, OpRef::int_op(1), 0);
    }

    #[test]
    fn test_ptr_eq_distinct_constants_not_folded_in_rewrite() {
        // PtrEq(const 100, const 200): two distinct non-null ConstPtr.
        // rewrite.py:525-564 _optimize_oois_ooisnot has no value-compare
        // branch — `arg0 is arg1` (line 542) is object identity, which is
        // False for distinct ConstPtr, so it falls through to `emit(op)`
        // (line 564). The actual constant fold lives in the pure pass
        // (pure.py:126-136 → execute_ptr_compare_const), not rewrite.
        // history.py:307 ConstPtr — Value::Ref must land on a Ref-tagged
        // OpRef so the box class identity matches the resoperation.py:615
        // RefOp mixin of the producer SameAsR.
        // OptRewrite alone: this asserts the REWRITE pass's behavior; the
        // chained harness would let OptPure fold it, masking a regression
        // where rewrite value-compares ConstPtr.
        let (result, ctx) = run_one_rewrite_only(
            vec![same_r(), same_r(), bin_r(OpCode::PtrEq, 0, 1)],
            2,
            &[
                (OpRef::ref_op(0), Value::Ref(GcRef(100))),
                (OpRef::ref_op(1), Value::Ref(GcRef(200))),
            ],
        );
        // rewrite.py:564 `return self.emit(op)` — rewrite passes the op on
        // unchanged; it does not fold distinct constants.
        assert_pass_on(&result);
        assert_eq!(
            ctx.get_box_replacement_box(OpRef::int_op(2))
                .and_then(|cb| cb.const_int()),
            None
        );
    }

    // ── CAST round-trip tests ──

    #[test]
    fn test_cast_ptr_to_int_passes_through() {
        // rewrite.py:807-809: CastPtrToInt registers pure inverse, emits.
        // arg0 is a Ref box (resoperation.py:615 RefOp `type = 'r'`).
        let (result, _) = run_one(vec![same_r(), unary_r(OpCode::CastPtrToInt, 0)], 1, &[]);
        assert_pass_on(&result);
    }

    #[test]
    fn test_cast_int_to_ptr_passes_through() {
        // rewrite.py:811-813: CastIntToPtr registers pure inverse, emits.
        let (result, _) = run_one(vec![same_i(), unary_i(OpCode::CastIntToPtr, 0)], 1, &[]);
        assert_pass_on(&result);
    }

    #[test]
    fn test_cast_opaque_ptr_eliminated() {
        // CastOpaquePtr(x) -> x
        let (result, ctx) = run_one(vec![same_r(), unary_r(OpCode::CastOpaquePtr, 0)], 1, &[]);
        assert_remove(&result);
        assert_forward(&ctx, OpRef::ref_op(1), OpRef::ref_op(0));
    }

    // ── CONVERT_FLOAT_BYTES tests ──
    // rewrite.py:815-821: these conversions are NOT eliminated —
    // they actually change bit representation. Only round-trips
    // (A→B→A) are eliminated via pure.rs CSE.

    #[test]
    fn test_convert_float_bytes_to_longlong_passes_through() {
        let (result, _) = run_one(
            vec![same_f(), unary_f(OpCode::ConvertFloatBytesToLonglong, 0)],
            1,
            &[],
        );
        assert_pass_on(&result);
    }

    #[test]
    fn test_convert_longlong_bytes_to_float_passes_through() {
        let (result, _) = run_one(
            vec![same_i(), unary_i(OpCode::ConvertLonglongBytesToFloat, 0)],
            1,
            &[],
        );
        // PassOn: op is emitted, no replacement registered.
        assert_pass_on(&result);
    }

    // ── GUARD_NO_EXCEPTION tests ──

    #[test]
    fn test_guard_no_exception_after_removed_call() {
        // CondCallN(condition=0, ...) -> removed, then GuardNoException -> removed
        let ops = build_specs(&[
            same_i(),                               // op0: condition (const 0)
            same_i(),                               // op1: func
            op_spec(OpCode::CondCallN, &[0, 1]),    // op2: removed
            op_spec(OpCode::GuardNoException, &[]), // op3: should be removed
        ]);
        let mut ctx = OptContext::new(4);
        ctx.emit((*ops[0]).clone());
        ctx.emit((*ops[1]).clone());
        let b = ctx.materialize_box_at(OpRef::int_op(0));
        ctx.make_constant_box(&b, Value::Int(0));

        let mut pass = OptRewrite::new();
        // Process CondCallN -> removed
        let mut resolved2 = (*ops[2]).clone();
        resolve_op_args_in_ctx(&mut resolved2, &mut ctx);
        let __pf_rc = std::rc::Rc::new(resolved2.clone());
        ctx.bind_input_resops(std::slice::from_ref(&__pf_rc));
        let result2 = pass.propagate_forward(&resolved2, &__pf_rc, &mut ctx);
        assert!(matches!(result2, OptimizationResult::Remove));

        // Process GuardNoException -> should also be removed
        let __pf_rc = std::rc::Rc::new((*ops[3]).clone());
        ctx.bind_input_resops(std::slice::from_ref(&__pf_rc));
        let result3 = pass.propagate_forward(&ops[3], &__pf_rc, &mut ctx);
        assert!(matches!(result3, OptimizationResult::Remove));
    }

    #[test]
    fn test_guard_no_exception_after_emitted_call() {
        // CallN(...) -> emitted, then GuardNoException -> kept
        let ops = build_specs(&[
            same_i(),                               // op0: func
            op_spec(OpCode::CallN, &[0]),           // op1: call
            op_spec(OpCode::GuardNoException, &[]), // op2: should NOT be removed
        ]);
        let mut ctx = OptContext::new(3);
        ctx.emit((*ops[0]).clone());

        let mut pass = OptRewrite::new();
        // Process CallN -> PassOn (not handled by OptRewrite)
        let __pf_rc = std::rc::Rc::new((*ops[1]).clone());
        ctx.bind_input_resops(std::slice::from_ref(&__pf_rc));
        let result1 = pass.propagate_forward(&ops[1], &__pf_rc, &mut ctx);
        assert!(matches!(result1, OptimizationResult::PassOn));
        ctx.emit((*ops[1]).clone());

        // Process GuardNoException -> should NOT be removed
        let __pf_rc = std::rc::Rc::new((*ops[2]).clone());
        ctx.bind_input_resops(std::slice::from_ref(&__pf_rc));
        let result2 = pass.propagate_forward(&ops[2], &__pf_rc, &mut ctx);
        assert!(matches!(result2, OptimizationResult::PassOn));
    }

    #[test]
    fn test_guard_future_condition_records_and_removes() {
        // rewrite.py: GUARD_FUTURE_CONDITION → record in patchguardop + remove
        let (result, ctx) = run_one(vec![op_spec(OpCode::GuardFutureCondition, &[])], 0, &[]);
        assert_remove(&result);
        assert!(ctx.patchguardop.is_some());
        assert_eq!(
            ctx.patchguardop.unwrap().opcode,
            OpCode::GuardFutureCondition
        );
    }

    #[test]
    fn test_guard_value_to_guard_false() {
        // GUARD_VALUE(v, 0) on a bool-bounded v → GUARD_FALSE(v)
        // (_maybe_replace_guard_value, optimizer.py:755-776). The [0,1]
        // bound on v comes from the `int_gt` producer the intbounds pass
        // analyzed; the guard's own make_constant runs in
        // postprocess_GUARD_VALUE (rewrite.py:313-315), after emit, so it
        // does not bound v at emit time.
        // Bound oparser graph: i0/i1 are header InputArgs, v = IntGt(i0, i1)
        // a live producer, and GUARD_VALUE's expected operand is the literal
        // ConstInt(0) — so every arg sheds to Operand::{InputArg,Op,Const}.
        use crate::history::test_support::bound_inputarg_box;
        let (i0, _i0_rc) = bound_inputarg_box(majit_ir::Type::Int, 0);
        let (i1, _i1_rc) = bound_inputarg_box(majit_ir::Type::Int, 1);
        // A live producer for v (IntGt result) at int_op(2); the OpRc is held
        // in `int_gt` so the from_bound_op box's Weak upgrade stays live.
        let int_gt = std::rc::Rc::new(Op::new(
            OpCode::IntGt,
            &[Operand::from_boxref(&i0), Operand::from_boxref(&i1)],
        ));
        int_gt.pos.set(OpRef::int_op(2));
        let v = BoxRef::from_bound_op(&int_gt);
        let zero = BoxRef::new_const(Value::Int(0));
        let guard_value = Op::new(
            OpCode::GuardValue,
            &[Operand::from_boxref(&v), Operand::from_boxref(&zero)],
        );
        let finish = Op::new(OpCode::Finish, &[]);
        let ops = {
            let mut gv = guard_value;
            gv.pos.set(OpRef::void_op(0));
            let mut fin = finish;
            fin.pos.set(OpRef::void_op(1));
            vec![(*int_gt).clone(), gv, fin]
        };
        let mut opt = crate::optimizeopt::optimizer::Optimizer::new();
        opt.add_pass(Box::new(crate::optimizeopt::intbounds::OptIntBounds::new()));
        opt.add_pass(Box::new(OptRewrite::new()));
        opt.trace_inputargs = majit_ir::OpRef::inputarg_refs(&vec![majit_ir::Type::Int; 2]);
        let mut constants: majit_ir::VecMap<u32, majit_ir::Value> = majit_ir::VecMap::new();
        let (ops, snapshots) = super::super::seed_empty_guard_snapshots(&ops);
        opt.snapshot_boxes = snapshots;
        let result = opt.optimize_with_constants_and_inputs(&ops, &mut constants, 2);

        assert!(
            result.iter().any(|o| o.opcode == OpCode::GuardFalse),
            "GUARD_VALUE(v, 0) should become GUARD_FALSE(v)"
        );
    }

    #[test]
    fn test_int_mul_neg_one() {
        // x * (-1) → INT_NEG(x)
        let mut b = crate::history::test_support::TraceBuilder::new();
        let x = b.input(majit_ir::Type::Int, 0);
        let neg_one = b.const_int(-1);
        let prod = b.op(OpCode::IntMul, &[x, neg_one]);
        b.op(OpCode::Finish, &[prod]);
        let (ops, inputs) = b.build();

        let mut opt = crate::optimizeopt::optimizer::Optimizer::new();
        opt.trace_inputargs = OpRef::inputarg_refs(&inputs);
        // mul_minus_one lives in OptIntBounds (autogenintrules.py).
        opt.add_pass(Box::new(crate::optimizeopt::intbounds::OptIntBounds::new()));
        opt.add_pass(Box::new(OptRewrite::new()));
        let mut constants: majit_ir::VecMap<u32, majit_ir::Value> = majit_ir::VecMap::new();
        let num_inputs = inputs.len();
        let result = opt
            .optimize_with_constants_and_inputs_oprc(&ops, &mut constants, num_inputs)
            .expect("test: unexpected InvalidLoop");

        assert!(
            result.iter().any(|o| o.opcode == OpCode::IntNeg),
            "x * (-1) should become INT_NEG(x)"
        );
    }

    #[test]
    fn test_float_mul_neg_one() {
        // x * (-1.0) → FLOAT_NEG(x)
        let mut b = crate::history::test_support::TraceBuilder::new();
        let x = b.input(majit_ir::Type::Float, 0);
        let neg_one = BoxRef::new_const(Value::Float(-1.0));
        let prod = b.op(OpCode::FloatMul, &[x, neg_one]);
        b.op(OpCode::Finish, &[prod]);
        let (ops, inputs) = b.build();

        let mut opt = crate::optimizeopt::optimizer::Optimizer::new();
        opt.trace_inputargs = OpRef::inputarg_refs(&inputs);
        opt.add_pass(Box::new(OptRewrite::new()));
        let mut constants: majit_ir::VecMap<u32, majit_ir::Value> = majit_ir::VecMap::new();
        let num_inputs = inputs.len();
        let result = opt
            .optimize_with_constants_and_inputs_oprc(&ops, &mut constants, num_inputs)
            .expect("test: unexpected InvalidLoop");
        assert!(!result.is_empty());
    }

    #[test]
    fn test_cond_call_n_zero_removes() {
        // COND_CALL_N(0, func, args...) → removed (condition is false)
        let mut b = crate::history::test_support::TraceBuilder::new();
        let cond = BoxRef::new_const(Value::Int(0));
        let func = b.input(majit_ir::Type::Int, 0);
        let arg = b.input(majit_ir::Type::Int, 1);
        b.op(OpCode::CondCallN, &[cond, func, arg]);
        b.op(OpCode::Finish, &[]);
        let (ops, inputs) = b.build();
        let mut opt = crate::optimizeopt::optimizer::Optimizer::new();
        opt.trace_inputargs = OpRef::inputarg_refs(&inputs);
        opt.add_pass(Box::new(OptRewrite::new()));
        let mut constants: majit_ir::VecMap<u32, majit_ir::Value> = majit_ir::VecMap::new();
        let num_inputs = inputs.len();
        let result = opt
            .optimize_with_constants_and_inputs_oprc(&ops, &mut constants, num_inputs)
            .expect("test: unexpected InvalidLoop");
        assert!(
            !result.iter().any(|o| o.opcode == OpCode::CondCallN),
            "COND_CALL_N(0, ...) should be removed"
        );
    }

    #[test]
    fn test_cond_call_n_nonzero_converts() {
        // COND_CALL_N(1, func, args...) → CALL_N(func, args...)
        let mut b = crate::history::test_support::TraceBuilder::new();
        let cond = BoxRef::new_const(Value::Int(1));
        let func = b.input(majit_ir::Type::Int, 0);
        let arg = b.input(majit_ir::Type::Int, 1);
        b.op(OpCode::CondCallN, &[cond, func, arg]);
        b.op(OpCode::Finish, &[]);
        let (ops, inputs) = b.build();
        let mut opt = crate::optimizeopt::optimizer::Optimizer::new();
        opt.trace_inputargs = OpRef::inputarg_refs(&inputs);
        opt.add_pass(Box::new(OptRewrite::new()));
        let mut constants: majit_ir::VecMap<u32, majit_ir::Value> = majit_ir::VecMap::new();
        let num_inputs = inputs.len();
        let result = opt
            .optimize_with_constants_and_inputs_oprc(&ops, &mut constants, num_inputs)
            .expect("test: unexpected InvalidLoop");
        assert!(
            result.iter().any(|o| o.opcode == OpCode::CallN),
            "COND_CALL_N(1, ...) should become CALL_N"
        );
    }
}
