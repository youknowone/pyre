/// Integer bounds optimization pass.
///
/// Translated from rpython/jit/metainterp/optimizeopt/intbounds.py.
///
/// Propagates integer bounds information through the trace. When a guard tests
/// a condition that is already known true from integer bounds, the guard can be
/// removed. It also narrows bounds after guards and arithmetic operations.
use majit_ir::{Op, OpCode, OpRef, Value};

use crate::optimizeopt::intutils::IntBound;
use crate::optimizeopt::{OptContext, Optimization, OptimizationResult};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PendingOverflowGuard {
    Present,
    ProvenSafeRemoved,
}

/// Integer bounds optimization pass.
///
/// Keeps track of the bounds placed on integers by guards and removes
/// redundant guards.
pub struct OptIntBounds {
    /// Per-operation IntBound storage, indexed by OpRef.
    bounds: Vec<Option<IntBound>>,
    /// The last emitted operation's opcode (for overflow guard handling).
    last_emitted_opcode: Option<OpCode>,
    /// The last emitted operation's args (for overflow guard handling).
    last_emitted_args: Vec<OpRef>,
    /// The last emitted operation's OpRef result.
    last_emitted_ref: OpRef,
    /// Tracks whether the next overflow guard still has a live overflow-producing
    /// source op, or whether that source was optimized away as provably safe.
    pending_overflow_guard: Option<PendingOverflowGuard>,
    /// intbounds.py: pure_from_args synthesis cache.
    /// Records equivalent pure operations discovered through bounds analysis
    /// (e.g., INT_OR with non-overlapping ranges = INT_ADD).
    /// Key: (opcode, arg0, arg1), Value: result OpRef.
    pure_from_args_cache: Vec<(OpCode, OpRef, OpRef, OpRef)>,
    /// Deferred guard bounds propagation. When a guard's comparison op
    /// is postponed by the heap pass, find_producing_op can't find it.
    /// Store (cond_ref, is_guard_true) and retry on next op.
    pending_guard_bounds: Option<(OpRef, bool)>,
}

impl OptIntBounds {
    pub fn new() -> Self {
        OptIntBounds {
            bounds: Vec::new(),
            pure_from_args_cache: Vec::new(),
            last_emitted_opcode: None,
            last_emitted_args: Vec::new(),
            last_emitted_ref: OpRef::NONE,
            pending_overflow_guard: None,
            pending_guard_bounds: None,
        }
    }

    /// Retry deferred guard bounds propagation. By the time the next op
    /// arrives, the heap pass has flushed the postponed comparison.
    fn flush_pending_guard_bounds(&mut self, ctx: &OptContext) {
        let pending = match self.pending_guard_bounds.take() {
            Some(p) => p,
            None => return,
        };
        let (cond_ref, is_true) = pending;
        if let Some(producing_op) = self.find_producing_op(cond_ref, ctx) {
            if producing_op.opcode.returns_bool() {
                if is_true {
                    self.setintbound(cond_ref, IntBound::from_constant(1));
                }
                let producing_op = producing_op.clone();
                self.propagate_bounds_from_comparison(&producing_op, is_true, ctx);
            }
        }
    }

    /// Get or create bounds for an operation.
    fn getintbound(&self, opref: OpRef, ctx: &OptContext) -> IntBound {
        let opref = ctx.get_box_replacement(opref);
        // Check if there is a known constant
        if let Some(val) = ctx.get_constant_int(opref) {
            return IntBound::from_constant(val);
        }
        let imported = ctx.imported_int_bounds.get(&opref).cloned();
        let idx = opref.0 as usize;
        if idx < self.bounds.len() {
            if let Some(ref b) = self.bounds[idx] {
                let mut merged = if let Some(imported) = imported {
                    let mut merged = b.clone();
                    let _ = merged.intersect(&imported);
                    merged
                } else {
                    b.clone()
                };
                // heap.py: merge with cross-pass lower bounds (array lengths)
                if let Some(&lower) = ctx.int_lower_bounds.get(&opref) {
                    let _ = merged.intersect(&IntBound::bounded(lower, i64::MAX));
                }
                return merged;
            }
        }
        if let Some(mut imported) = imported {
            if let Some(&lower) = ctx.int_lower_bounds.get(&opref) {
                let _ = imported.intersect(&IntBound::bounded(lower, i64::MAX));
            }
            return imported;
        }
        // heap.py: check cross-pass lower bounds even without local bound
        if let Some(&lower) = ctx.int_lower_bounds.get(&opref) {
            return IntBound::bounded(lower, i64::MAX);
        }
        IntBound::unbounded()
    }

    /// Store bounds for an operation.
    fn setintbound(&mut self, opref: OpRef, bound: IntBound) {
        let idx = opref.0 as usize;
        if idx >= self.bounds.len() {
            self.bounds.resize(idx + 1, None);
        }
        self.bounds[idx] = Some(bound);
    }

    /// Get a mutable reference to the bound for an opref, creating unbounded if needed.
    fn get_bound_mut(&mut self, opref: OpRef) -> &mut IntBound {
        let idx = opref.0 as usize;
        if idx >= self.bounds.len() {
            self.bounds.resize(idx + 1, None);
        }
        self.bounds[idx].get_or_insert_with(IntBound::unbounded)
    }

    /// Intersect a bound into the stored bound for opref.
    fn intersect_bound(&mut self, opref: OpRef, bound: &IntBound) {
        let b = self.get_bound_mut(opref);
        let _ = b.intersect(bound);
    }

    /// optimizer.py:434: make_constant_int(box, intvalue)
    /// Validates that the constant value is within the existing IntBound
    /// range before making the box constant. Raises InvalidLoop if not.
    fn make_constant_int(&mut self, op: &Op, value: i64, ctx: &mut OptContext) {
        // optimizer.py:415-426: safety check — if the box already has an
        // IntBound, verify the constant is within that range.
        let replaced = ctx.get_box_replacement(op.pos);
        let existing = self.getintbound(replaced, ctx);
        if !existing.is_unbounded() {
            if !existing.contains(value) {
                std::panic::panic_any(crate::optimize::InvalidLoop(
                    "constant int is outside the range allowed for that box",
                ));
            }
            // intutils.py:412-423: make_eq_const — narrow the shared bound
            // to the constant value. Important when the bound is shared
            // (e.g., with an array length).
            let idx = replaced.0 as usize;
            if idx < self.bounds.len() {
                if let Some(ref mut b) = self.bounds[idx] {
                    let _ = b.make_eq_const(value);
                }
            }
        }
        ctx.make_constant(op.pos, Value::Int(value));
        self.setintbound(op.pos, IntBound::from_constant(value));
    }

    /// Record a pure_from_args entry for CSE.
    /// intbounds.py: self.optimizer.pure_from_args(opnum, args, result)
    fn record_pure_from_args(&mut self, opcode: OpCode, arg0: OpRef, arg1: OpRef, result: OpRef) {
        self.pure_from_args_cache.push((opcode, arg0, arg1, result));
    }

    /// Get or create a constant OpRef for the given value.
    fn get_or_make_const(&self, value: i64, ctx: &mut OptContext) -> OpRef {
        // Search existing constants
        for (idx, slot) in ctx.constants.iter().enumerate() {
            if let Some(Value::Int(v)) = slot {
                if *v == value {
                    return OpRef(idx as u32);
                }
            }
        }
        // Create a new constant via a SameAs op
        let op = Op::new(OpCode::SameAsI, &[]);
        let opref = ctx.emit(op);
        ctx.make_constant(opref, Value::Int(value));
        opref
    }

    /// Get the pure_from_args synthesis cache (for integration with OptPure).
    pub fn get_pure_from_args_cache(&self) -> &[(OpCode, OpRef, OpRef, OpRef)] {
        &self.pure_from_args_cache
    }

    // ── Comparison optimizations ──

    fn optimize_int_lt(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let arg0 = ctx.get_box_replacement(op.arg(0));
        let arg1 = ctx.get_box_replacement(op.arg(1));
        let b0 = self.getintbound(arg0, ctx);
        let b1 = self.getintbound(arg1, ctx);
        if b0.known_lt(&b1) {
            self.make_constant_int(op, 1, ctx);
            OptimizationResult::Remove
        } else if b0.known_ge(&b1) || arg0 == arg1 {
            self.make_constant_int(op, 0, ctx);
            OptimizationResult::Remove
        } else {
            self.postprocess_bool_result(op);
            OptimizationResult::PassOn
        }
    }

    fn optimize_int_gt(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let arg0 = ctx.get_box_replacement(op.arg(0));
        let arg1 = ctx.get_box_replacement(op.arg(1));
        let b0 = self.getintbound(arg0, ctx);
        let b1 = self.getintbound(arg1, ctx);
        if b0.known_gt(&b1) {
            self.make_constant_int(op, 1, ctx);
            OptimizationResult::Remove
        } else if b0.known_le(&b1) || arg0 == arg1 {
            self.make_constant_int(op, 0, ctx);
            OptimizationResult::Remove
        } else {
            self.postprocess_bool_result(op);
            OptimizationResult::PassOn
        }
    }

    fn optimize_int_le(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let arg0 = ctx.get_box_replacement(op.arg(0));
        let arg1 = ctx.get_box_replacement(op.arg(1));
        let b0 = self.getintbound(arg0, ctx);
        let b1 = self.getintbound(arg1, ctx);
        if b0.known_le(&b1) || arg0 == arg1 {
            self.make_constant_int(op, 1, ctx);
            OptimizationResult::Remove
        } else if b0.known_gt(&b1) {
            self.make_constant_int(op, 0, ctx);
            OptimizationResult::Remove
        } else {
            self.postprocess_bool_result(op);
            OptimizationResult::PassOn
        }
    }

    fn optimize_int_ge(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let arg0 = ctx.get_box_replacement(op.arg(0));
        let arg1 = ctx.get_box_replacement(op.arg(1));
        let b0 = self.getintbound(arg0, ctx);
        let b1 = self.getintbound(arg1, ctx);
        if b0.known_ge(&b1) || arg0 == arg1 {
            self.make_constant_int(op, 1, ctx);
            OptimizationResult::Remove
        } else if b0.known_lt(&b1) {
            self.make_constant_int(op, 0, ctx);
            OptimizationResult::Remove
        } else {
            self.postprocess_bool_result(op);
            OptimizationResult::PassOn
        }
    }

    fn optimize_int_eq(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let arg0 = ctx.get_box_replacement(op.arg(0));
        let arg1 = ctx.get_box_replacement(op.arg(1));
        let b0 = self.getintbound(arg0, ctx);
        let b1 = self.getintbound(arg1, ctx);
        if arg0 == arg1 {
            self.make_constant_int(op, 1, ctx);
            OptimizationResult::Remove
        } else if b0.known_ne(&b1) {
            self.make_constant_int(op, 0, ctx);
            OptimizationResult::Remove
        } else {
            self.postprocess_bool_result(op);
            OptimizationResult::PassOn
        }
    }

    fn optimize_int_ne(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let arg0 = ctx.get_box_replacement(op.arg(0));
        let arg1 = ctx.get_box_replacement(op.arg(1));
        let b0 = self.getintbound(arg0, ctx);
        let b1 = self.getintbound(arg1, ctx);
        if arg0 == arg1 {
            self.make_constant_int(op, 0, ctx);
            OptimizationResult::Remove
        } else if b0.known_ne(&b1) {
            self.make_constant_int(op, 1, ctx);
            OptimizationResult::Remove
        } else {
            self.postprocess_bool_result(op);
            OptimizationResult::PassOn
        }
    }

    // ── Unsigned comparison optimizations ──

    fn optimize_uint_lt(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let arg0 = ctx.get_box_replacement(op.arg(0));
        let arg1 = ctx.get_box_replacement(op.arg(1));
        let b0 = self.getintbound(arg0, ctx);
        let b1 = self.getintbound(arg1, ctx);
        if b0.known_unsigned_lt(&b1) {
            self.make_constant_int(op, 1, ctx);
            OptimizationResult::Remove
        } else if b0.known_unsigned_ge(&b1) || arg0 == arg1 {
            self.make_constant_int(op, 0, ctx);
            OptimizationResult::Remove
        } else {
            self.postprocess_bool_result(op);
            OptimizationResult::PassOn
        }
    }

    fn optimize_uint_gt(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let arg0 = ctx.get_box_replacement(op.arg(0));
        let arg1 = ctx.get_box_replacement(op.arg(1));
        let b0 = self.getintbound(arg0, ctx);
        let b1 = self.getintbound(arg1, ctx);
        if b0.known_unsigned_gt(&b1) {
            self.make_constant_int(op, 1, ctx);
            OptimizationResult::Remove
        } else if b0.known_unsigned_le(&b1) || arg0 == arg1 {
            self.make_constant_int(op, 0, ctx);
            OptimizationResult::Remove
        } else {
            self.postprocess_bool_result(op);
            OptimizationResult::PassOn
        }
    }

    fn optimize_uint_le(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let arg0 = ctx.get_box_replacement(op.arg(0));
        let arg1 = ctx.get_box_replacement(op.arg(1));
        let b0 = self.getintbound(arg0, ctx);
        let b1 = self.getintbound(arg1, ctx);
        if b0.known_unsigned_le(&b1) || arg0 == arg1 {
            self.make_constant_int(op, 1, ctx);
            OptimizationResult::Remove
        } else if b0.known_unsigned_gt(&b1) {
            self.make_constant_int(op, 0, ctx);
            OptimizationResult::Remove
        } else {
            self.postprocess_bool_result(op);
            OptimizationResult::PassOn
        }
    }

    fn optimize_uint_ge(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let arg0 = ctx.get_box_replacement(op.arg(0));
        let arg1 = ctx.get_box_replacement(op.arg(1));
        let b0 = self.getintbound(arg0, ctx);
        let b1 = self.getintbound(arg1, ctx);
        if b0.known_unsigned_ge(&b1) || arg0 == arg1 {
            self.make_constant_int(op, 1, ctx);
            OptimizationResult::Remove
        } else if b0.known_unsigned_lt(&b1) {
            self.make_constant_int(op, 0, ctx);
            OptimizationResult::Remove
        } else {
            self.postprocess_bool_result(op);
            OptimizationResult::PassOn
        }
    }

    /// Set result bounds to [0, 1] for comparison/boolean-result operations.
    fn postprocess_bool_result(&mut self, op: &Op) {
        self.setintbound(op.pos, IntBound::bounded(0, 1));
    }

    // ── Arithmetic postprocessing ──

    /// intbounds.py:114-146 postprocess_INT_ADD
    fn postprocess_int_add(&mut self, op: &Op, ctx: &mut OptContext) {
        let arg0 = ctx.get_box_replacement(op.arg(0));
        let arg1 = ctx.get_box_replacement(op.arg(1));
        let b0 = self.getintbound(arg0, ctx);
        // intbounds.py:119-123: if arg0 is arg1: b = b0.lshift_bound(1) (x+x is even)
        let b = if arg0 == arg1 {
            b0.lshift_bound(&IntBound::from_constant(1))
        } else {
            let b1 = self.getintbound(arg1, ctx);
            b0.add_bound(&b1)
        };
        self.intersect_bound(op.pos, &b);
        // intbounds.py:125-127:
        //   self.optimizer.pure_from_args2(rop.INT_SUB, op, arg1, arg0)
        //   self.optimizer.pure_from_args2(rop.INT_SUB, op, arg0, arg1)
        self.record_pure_from_args(OpCode::IntSub, op.pos, arg1, arg0);
        self.record_pure_from_args(OpCode::IntSub, op.pos, arg0, arg1);
        // intbounds.py:128-142: pick the constant arg, fall back to commutative
        // swap so `arg1` ends up holding the non-const operand.
        let (inv_const, other) = if let Some(c) = ctx.get_constant_int(arg0) {
            if c == i64::MIN {
                return;
            }
            (c, arg1)
        } else if let Some(c) = ctx.get_constant_int(arg1) {
            if c == i64::MIN {
                return;
            }
            (c, arg0)
        } else {
            return;
        };
        let neg_ref = self.get_or_make_const(-inv_const, ctx);
        // intbounds.py:143-146:
        //   self.optimizer.pure_from_args2(rop.INT_SUB, arg1, inv_arg0, op)
        //   self.optimizer.pure_from_args2(rop.INT_SUB, arg1, op, inv_arg0)
        //   self.optimizer.pure_from_args2(rop.INT_ADD, op, inv_arg0, arg1)
        //   self.optimizer.pure_from_args2(rop.INT_ADD, inv_arg0, op, arg1)
        self.record_pure_from_args(OpCode::IntSub, other, neg_ref, op.pos);
        self.record_pure_from_args(OpCode::IntSub, other, op.pos, neg_ref);
        self.record_pure_from_args(OpCode::IntAdd, op.pos, neg_ref, other);
        self.record_pure_from_args(OpCode::IntAdd, neg_ref, op.pos, other);
    }

    /// intbounds.py: INT_SUB postprocess with constant inversion synthesis.
    fn postprocess_int_sub(&mut self, op: &Op, ctx: &mut OptContext) {
        let arg0 = ctx.get_box_replacement(op.arg(0));
        let arg1 = ctx.get_box_replacement(op.arg(1));
        let b0 = self.getintbound(arg0, ctx);
        let b1 = self.getintbound(arg1, ctx);
        let b = b0.sub_bound(&b1);
        self.intersect_bound(op.pos, &b);
        // Synthesis: INT_SUB(a,b)=res → INT_ADD(res,b)=a, INT_SUB(a,res)=b
        self.record_pure_from_args(OpCode::IntAdd, op.pos, arg1, arg0);
        self.record_pure_from_args(OpCode::IntSub, arg0, op.pos, arg1);
        // intbounds.py: constant inversion for INT_SUB
        if let Some(c1) = ctx.get_constant_int(arg1) {
            if c1 != i64::MIN {
                let neg_ref = self.get_or_make_const(-c1, ctx);
                self.record_pure_from_args(OpCode::IntAdd, arg0, neg_ref, op.pos);
                self.record_pure_from_args(OpCode::IntAdd, neg_ref, arg0, op.pos);
                self.record_pure_from_args(OpCode::IntSub, op.pos, neg_ref, arg0);
                self.record_pure_from_args(OpCode::IntSub, op.pos, arg0, neg_ref);
            }
        }
    }

    fn postprocess_int_mul(&mut self, op: &Op, ctx: &OptContext) {
        let b0 = self.getintbound(op.arg(0), ctx);
        let b1 = self.getintbound(op.arg(1), ctx);
        let b = b0.mul_bound(&b1);
        self.intersect_bound(op.pos, &b);
    }

    fn postprocess_int_and(&mut self, op: &Op, ctx: &OptContext) {
        let b0 = self.getintbound(op.arg(0), ctx);
        let b1 = self.getintbound(op.arg(1), ctx);
        let b = b0.and_bound(&b1);
        self.intersect_bound(op.pos, &b);
    }

    /// intbounds.py:60-71 postprocess_INT_OR
    fn postprocess_int_or(&mut self, op: &Op, ctx: &OptContext) {
        let arg0 = ctx.get_box_replacement(op.arg(0));
        let arg1 = ctx.get_box_replacement(op.arg(1));
        let b0 = self.getintbound(arg0, ctx);
        let b1 = self.getintbound(arg1, ctx);
        // intbounds.py:65: if b0.and_bound(b1).known_eq_const(0):
        if b0.and_bound(&b1).known_eq_const(0) {
            // intbounds.py:66-69:
            //   pure_from_args2(rop.INT_ADD, arg0, arg1, op)
            //   pure_from_args2(rop.INT_XOR, arg0, arg1, op)
            self.record_pure_from_args(OpCode::IntAdd, arg0, arg1, op.pos);
            self.record_pure_from_args(OpCode::IntXor, arg0, arg1, op.pos);
        }
        let b = b0.or_bound(&b1);
        self.intersect_bound(op.pos, &b);
    }

    /// intbounds.py:73-84 postprocess_INT_XOR
    fn postprocess_int_xor(&mut self, op: &Op, ctx: &OptContext) {
        let arg0 = ctx.get_box_replacement(op.arg(0));
        let arg1 = ctx.get_box_replacement(op.arg(1));
        let b0 = self.getintbound(arg0, ctx);
        let b1 = self.getintbound(arg1, ctx);
        // intbounds.py:78: if b0.and_bound(b1).known_eq_const(0):
        if b0.and_bound(&b1).known_eq_const(0) {
            // intbounds.py:79-82:
            //   pure_from_args2(rop.INT_ADD, arg0, arg1, op)
            //   pure_from_args2(rop.INT_OR, arg0, arg1, op)
            self.record_pure_from_args(OpCode::IntAdd, arg0, arg1, op.pos);
            self.record_pure_from_args(OpCode::IntOr, arg0, arg1, op.pos);
        }
        let b = b0.xor_bound(&b1);
        self.intersect_bound(op.pos, &b);
    }

    /// intbounds.py: INT_LSHIFT pure_from_args synthesis.
    /// If res = INT_LSHIFT(a, b), then a = INT_RSHIFT(res, b).
    fn postprocess_int_lshift(&mut self, op: &Op, ctx: &OptContext) {
        let arg0 = ctx.get_box_replacement(op.arg(0));
        let arg1 = ctx.get_box_replacement(op.arg(1));
        let b0 = self.getintbound(arg0, ctx);
        let b1 = self.getintbound(arg1, ctx);
        let b = b0.lshift_bound(&b1);
        self.intersect_bound(op.pos, &b);
        // intbounds.py:185: only synthesize reverse if lshift cannot overflow
        if b0.lshift_bound_cannot_overflow(&b1) {
            self.record_pure_from_args(OpCode::IntRshift, op.pos, arg1, arg0);
        }
    }

    fn postprocess_int_rshift(&mut self, op: &Op, ctx: &OptContext) {
        let b0 = self.getintbound(op.arg(0), ctx);
        let b1 = self.getintbound(op.arg(1), ctx);
        let b = b0.rshift_bound(&b1);
        self.intersect_bound(op.pos, &b);
    }

    fn postprocess_uint_rshift(&mut self, op: &Op, ctx: &OptContext) {
        let b0 = self.getintbound(op.arg(0), ctx);
        let b1 = self.getintbound(op.arg(1), ctx);
        let b = b0.urshift_bound(&b1);
        self.intersect_bound(op.pos, &b);
    }

    fn postprocess_int_floordiv(&mut self, op: &Op, ctx: &OptContext) {
        let b0 = self.getintbound(op.arg(0), ctx);
        let b1 = self.getintbound(op.arg(1), ctx);
        let b = b0.floordiv_bound(&b1);
        self.intersect_bound(op.pos, &b);
    }

    fn postprocess_int_mod(&mut self, op: &Op, ctx: &OptContext) {
        let b0 = self.getintbound(op.arg(0), ctx);
        let b1 = self.getintbound(op.arg(1), ctx);
        let b = b0.mod_bound(&b1);
        self.intersect_bound(op.pos, &b);
    }

    fn postprocess_int_neg(&mut self, op: &Op, ctx: &OptContext) {
        let b = self.getintbound(op.arg(0), ctx);
        let result = b.neg_bound();
        self.intersect_bound(op.pos, &result);
    }

    fn postprocess_int_invert(&mut self, op: &Op, ctx: &OptContext) {
        let b = self.getintbound(op.arg(0), ctx);
        let result = b.invert_bound();
        self.intersect_bound(op.pos, &result);
    }

    fn postprocess_int_force_ge_zero(&mut self, op: &Op, ctx: &OptContext) {
        let b_arg = self.getintbound(op.arg(0), ctx);
        let mut result = IntBound::nonnegative();
        if b_arg.upper >= 0 {
            let _ = result.make_le(&b_arg);
        }
        self.intersect_bound(op.pos, &result);
    }

    fn postprocess_arraylen_gc(&mut self, op: &Op) {
        // Array length is always non-negative
        self.intersect_bound(op.pos, &IntBound::nonnegative());
    }

    fn postprocess_strlen(&mut self, op: &Op) {
        // String length is always non-negative
        self.intersect_bound(op.pos, &IntBound::nonnegative());
    }

    fn postprocess_unicodelen(&mut self, op: &Op) {
        self.intersect_bound(op.pos, &IntBound::nonnegative());
    }

    // ── INT_SIGNEXT optimization ──

    fn optimize_int_signext(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let b = self.getintbound(op.arg(0), ctx);
        let b1 = self.getintbound(op.arg(1), ctx);
        if b1.is_constant() {
            let byte_size = b1.get_constant();
            let numbits = byte_size * 8;
            let start = -(1i64 << (numbits - 1));
            let stop = 1i64 << (numbits - 1);
            if b.is_within_range(start, stop - 1) {
                // The value already fits; replace with the input.
                ctx.replace_op(op.pos, op.arg(0));
                return OptimizationResult::Remove;
            }
        }
        self.postprocess_int_signext(op, ctx);
        OptimizationResult::PassOn
    }

    fn postprocess_int_signext(&mut self, op: &Op, ctx: &OptContext) {
        let b1 = self.getintbound(op.arg(1), ctx);
        if b1.is_constant() {
            let byte_size = b1.get_constant();
            let numbits = byte_size * 8;
            let start = -(1i64 << (numbits - 1));
            let stop = 1i64 << (numbits - 1);
            let _ = self.get_bound_mut(op.pos).intersect_const(start, stop - 1);
        }
    }

    // ── Overflow operations ──

    fn optimize_int_add_ovf(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        if let (Some(a), Some(b)) = (
            ctx.get_constant_int(op.arg(0)),
            ctx.get_constant_int(op.arg(1)),
        ) {
            if let Some(result) = a.checked_add(b) {
                self.make_constant_int(op, result, ctx);
                self.pending_overflow_guard = Some(PendingOverflowGuard::ProvenSafeRemoved);
                return OptimizationResult::Remove;
            }
        }
        let b0 = self.getintbound(op.arg(0), ctx);
        let b1 = self.getintbound(op.arg(1), ctx);
        if b0.add_bound_cannot_overflow(&b1) {
            // Transform to non-overflow INT_ADD and mark the following
            // GUARD_NO_OVERFLOW as redundant.
            self.pending_overflow_guard = Some(PendingOverflowGuard::ProvenSafeRemoved);
            let mut new_op = Op::new(OpCode::IntAdd, &op.args);
            new_op.descr = op.descr.clone();
            new_op.pos = op.pos;
            self.postprocess_int_add(&new_op, ctx);
            OptimizationResult::Emit(new_op)
        } else {
            self.pending_overflow_guard = Some(PendingOverflowGuard::Present);
            self.postprocess_int_add_ovf(op, ctx);
            OptimizationResult::PassOn
        }
    }

    fn postprocess_int_add_ovf(&mut self, op: &Op, ctx: &OptContext) {
        let arg0 = ctx.get_box_replacement(op.arg(0));
        let arg1 = ctx.get_box_replacement(op.arg(1));
        let b0 = self.getintbound(arg0, ctx);
        let b = if arg0 == arg1 {
            b0.mul2_bound_no_overflow()
        } else {
            let b1 = self.getintbound(arg1, ctx);
            b0.add_bound_no_overflow(&b1)
        };
        self.intersect_bound(op.pos, &b);
    }

    fn optimize_int_sub_ovf(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        if let (Some(a), Some(b)) = (
            ctx.get_constant_int(op.arg(0)),
            ctx.get_constant_int(op.arg(1)),
        ) {
            if let Some(result) = a.checked_sub(b) {
                self.make_constant_int(op, result, ctx);
                self.pending_overflow_guard = Some(PendingOverflowGuard::ProvenSafeRemoved);
                return OptimizationResult::Remove;
            }
        }
        let arg0 = ctx.get_box_replacement(op.arg(0));
        let arg1 = ctx.get_box_replacement(op.arg(1));
        let b0 = self.getintbound(arg0, ctx);
        let b1 = self.getintbound(arg1, ctx);
        if arg0 == arg1 {
            // x - x = 0
            self.make_constant_int(op, 0, ctx);
            return OptimizationResult::Remove;
        }
        if b0.sub_bound_cannot_overflow(&b1) {
            self.pending_overflow_guard = Some(PendingOverflowGuard::ProvenSafeRemoved);
            let mut new_op = Op::new(OpCode::IntSub, &op.args);
            new_op.descr = op.descr.clone();
            new_op.pos = op.pos;
            self.postprocess_int_sub(&new_op, ctx);
            OptimizationResult::Emit(new_op)
        } else {
            self.pending_overflow_guard = Some(PendingOverflowGuard::Present);
            self.postprocess_int_sub_ovf(op, ctx);
            OptimizationResult::PassOn
        }
    }

    fn postprocess_int_sub_ovf(&mut self, op: &Op, ctx: &OptContext) {
        let b0 = self.getintbound(op.arg(0), ctx);
        let b1 = self.getintbound(op.arg(1), ctx);
        let b = b0.sub_bound_no_overflow(&b1);
        self.intersect_bound(op.pos, &b);
    }

    fn optimize_int_mul_ovf(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        if let (Some(a), Some(b)) = (
            ctx.get_constant_int(op.arg(0)),
            ctx.get_constant_int(op.arg(1)),
        ) {
            if let Some(result) = a.checked_mul(b) {
                self.make_constant_int(op, result, ctx);
                self.pending_overflow_guard = Some(PendingOverflowGuard::ProvenSafeRemoved);
                return OptimizationResult::Remove;
            }
        }
        let b0 = self.getintbound(op.arg(0), ctx);
        let b1 = self.getintbound(op.arg(1), ctx);
        if b0.mul_bound_cannot_overflow(&b1) {
            self.pending_overflow_guard = Some(PendingOverflowGuard::ProvenSafeRemoved);
            let mut new_op = Op::new(OpCode::IntMul, &op.args);
            new_op.descr = op.descr.clone();
            new_op.pos = op.pos;
            self.postprocess_int_mul(&new_op, ctx);
            OptimizationResult::Emit(new_op)
        } else {
            self.pending_overflow_guard = Some(PendingOverflowGuard::Present);
            self.postprocess_int_mul_ovf(op, ctx);
            OptimizationResult::PassOn
        }
    }

    fn postprocess_int_mul_ovf(&mut self, op: &Op, ctx: &OptContext) {
        let arg0 = ctx.get_box_replacement(op.arg(0));
        let arg1 = ctx.get_box_replacement(op.arg(1));
        let b0 = self.getintbound(arg0, ctx);
        let b = if arg0 == arg1 {
            b0.square_bound_no_overflow()
        } else {
            let b1 = self.getintbound(arg1, ctx);
            b0.mul_bound_no_overflow(&b1)
        };
        self.intersect_bound(op.pos, &b);
    }

    /// intbounds.py:209-229 optimize_GUARD_NO_OVERFLOW
    fn optimize_guard_no_overflow(&mut self, op: &Op) -> OptimizationResult {
        let _ = op;
        // intbounds.py:210-220: lastop = self.last_emitted_operation
        //   if lastop is not None:
        //     if opnum not in (INT_ADD_OVF, INT_SUB_OVF, INT_MUL_OVF): return
        let last_opcode = self.last_emitted_opcode;
        let last_is_ovf = matches!(
            last_opcode,
            Some(OpCode::IntAddOvf | OpCode::IntSubOvf | OpCode::IntMulOvf)
        );
        if !last_is_ovf {
            self.pending_overflow_guard = None;
            return OptimizationResult::Remove;
        }
        // intbounds.py:222-228: synthesize the non-overflowing inverse for
        // optimize_default to reuse, plus the reverse op.
        //   if INT_ADD_OVF:
        //     pure_from_args2(INT_SUB, result, args[1], args[0])
        //     pure_from_args2(INT_SUB, result, args[0], args[1])
        //   elif INT_SUB_OVF:
        //     pure_from_args2(INT_ADD, result, args[1], args[0])
        //     pure_from_args2(INT_SUB, args[0], result, args[1])
        let result = self.last_emitted_ref;
        if self.last_emitted_args.len() >= 2 && !result.is_none() {
            let arg0 = self.last_emitted_args[0];
            let arg1 = self.last_emitted_args[1];
            match last_opcode {
                Some(OpCode::IntAddOvf) => {
                    self.record_pure_from_args(OpCode::IntSub, result, arg1, arg0);
                    self.record_pure_from_args(OpCode::IntSub, result, arg0, arg1);
                }
                Some(OpCode::IntSubOvf) => {
                    self.record_pure_from_args(OpCode::IntAdd, result, arg1, arg0);
                    self.record_pure_from_args(OpCode::IntSub, arg0, result, arg1);
                }
                _ => {}
            }
        }
        // intbounds.py:229: return self.emit(op)
        match self.pending_overflow_guard.take() {
            Some(PendingOverflowGuard::Present) => OptimizationResult::PassOn,
            Some(PendingOverflowGuard::ProvenSafeRemoved) | None => OptimizationResult::Remove,
        }
    }

    fn optimize_guard_overflow(&mut self, op: &Op) -> OptimizationResult {
        self.pending_overflow_guard = None;
        let _ = op;
        OptimizationResult::PassOn
    }

    // ── Guard optimizations ──

    fn optimize_guard_true(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let cond_ref = ctx.get_box_replacement(op.arg(0));

        // If the condition is a known constant, we can determine the guard outcome
        if let Some(val) = ctx.get_constant_int(cond_ref) {
            if val != 0 {
                // Guard always passes, remove it
                return OptimizationResult::Remove;
            }
            // Guard always fails - still emit it (will fail at runtime)
        }

        // Check if the bound on the condition tells us it's always true
        let b = self.getintbound(cond_ref, ctx);
        if b.known_gt_const(0) {
            // known nonzero, guard always passes
            return OptimizationResult::Remove;
        }

        // After emitting, propagate bounds backward from the guard
        // The condition is known to be true (nonzero) after this guard
        self.propagate_bounds_from_guard_true(cond_ref, ctx);
        OptimizationResult::PassOn
    }

    fn optimize_guard_false(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let cond_ref = ctx.get_box_replacement(op.arg(0));

        if let Some(val) = ctx.get_constant_int(cond_ref) {
            if val == 0 {
                return OptimizationResult::Remove;
            }
        }

        let b = self.getintbound(cond_ref, ctx);
        if b.known_eq_const(0) {
            return OptimizationResult::Remove;
        }

        self.propagate_bounds_from_guard_false(cond_ref, ctx);
        OptimizationResult::PassOn
    }

    /// optimizer.py:366 as_operation parity:
    /// Find the operation that produced cond_ref by searching new_operations.
    /// RPython's `as_operation(box)` checks `_emittedoperations` directly;
    /// majit's flat OpRef model requires a positional lookup.
    fn find_producing_op<'a>(&self, cond_ref: OpRef, ctx: &'a OptContext) -> Option<&'a Op> {
        // First try direct index (when OpRef matches new_operations index)
        let idx = cond_ref.0 as usize;
        if idx < ctx.new_operations.len() && ctx.new_operations[idx].pos == cond_ref {
            return Some(&ctx.new_operations[idx]);
        }
        // Otherwise search by pos field
        ctx.new_operations.iter().rfind(|op| op.pos == cond_ref)
    }

    /// Propagate bounds backward after GUARD_TRUE.
    /// The condition (cond_ref) is known to produce a nonzero value (true).
    fn propagate_bounds_from_guard_true(&mut self, cond_ref: OpRef, ctx: &OptContext) {
        if let Some(producing_op) = self.find_producing_op(cond_ref, ctx) {
            if producing_op.opcode.returns_bool() {
                self.setintbound(cond_ref, IntBound::from_constant(1));
                let producing_op = producing_op.clone();
                self.propagate_bounds_from_comparison(&producing_op, true, ctx);
                return;
            }
        } else if ctx.imported_label_args.is_some() {
            // Defer only in Phase 2 (has imported_label_args) where the
            // comparison is postponed by the heap pass and bounds
            // propagation is needed for overflow elimination in loop bodies.
            // Phase 1 (preamble) also has skip_flush but doesn't need
            // deferred propagation.
            self.pending_guard_bounds = Some((cond_ref, true));
        }

        let b0 = self.getintbound(cond_ref, ctx);
        if b0.known_nonnegative() {
            let _ = self.get_bound_mut(cond_ref).make_gt_const(0);
        } else if b0.known_le_const(0) {
            let _ = self.get_bound_mut(cond_ref).make_lt_const(0);
        }
    }

    /// Propagate bounds backward after GUARD_FALSE.
    /// The condition is known to produce 0 (false).
    fn propagate_bounds_from_guard_false(&mut self, cond_ref: OpRef, ctx: &OptContext) {
        self.setintbound(cond_ref, IntBound::from_constant(0));

        if let Some(producing_op) = self.find_producing_op(cond_ref, ctx) {
            if producing_op.opcode.returns_bool() {
                let producing_op = producing_op.clone();
                self.propagate_bounds_from_comparison(&producing_op, false, ctx);
            }
        } else if ctx.imported_label_args.is_some() {
            self.pending_guard_bounds = Some((cond_ref, false));
        }
    }

    /// Given that a comparison op produced `is_true`, narrow the bounds of its arguments.
    fn propagate_bounds_from_comparison(&mut self, op: &Op, is_true: bool, ctx: &OptContext) {
        // RPython intbounds.py parity: resolve through get_box_replacement
        // so bounds are stored on the canonical OpRef. Without this, SameAs
        // forwarding (e.g. OpRef(44) → OpRef(39)) causes bounds to be set
        // on the stale OpRef while subsequent lookups resolve to the
        // canonical one, losing the bound information.
        let arg0 = ctx.get_box_replacement(op.arg(0));

        // Handle unary ops
        match op.opcode {
            OpCode::IntIsTrue => {
                if is_true {
                    // nonzero
                    let b0 = self.getintbound(arg0, ctx);
                    if b0.known_nonnegative() {
                        let _ = self.get_bound_mut(arg0).make_gt_const(0);
                    } else if b0.known_le_const(0) {
                        let _ = self.get_bound_mut(arg0).make_lt_const(0);
                    }
                } else {
                    // zero
                    let _ = self.get_bound_mut(arg0).make_eq_const(0);
                }
                return;
            }
            OpCode::IntIsZero => {
                if is_true {
                    // the value is zero
                    let _ = self.get_bound_mut(arg0).make_eq_const(0);
                } else {
                    // the value is nonzero
                    let b0 = self.getintbound(arg0, ctx);
                    if b0.known_nonnegative() {
                        let _ = self.get_bound_mut(arg0).make_gt_const(0);
                    } else if b0.known_le_const(0) {
                        let _ = self.get_bound_mut(arg0).make_lt_const(0);
                    }
                }
                return;
            }
            _ => {}
        }

        let arg1 = if op.num_args() > 1 {
            ctx.get_box_replacement(op.arg(1))
        } else {
            return;
        };

        match op.opcode {
            OpCode::IntLt => {
                if is_true {
                    self.make_int_lt(arg0, arg1, ctx);
                } else {
                    self.make_int_ge(arg0, arg1, ctx);
                }
            }
            OpCode::IntLe => {
                if is_true {
                    self.make_int_le(arg0, arg1, ctx);
                } else {
                    self.make_int_gt(arg0, arg1, ctx);
                }
            }
            OpCode::IntGt => {
                if is_true {
                    self.make_int_gt(arg0, arg1, ctx);
                } else {
                    self.make_int_le(arg0, arg1, ctx);
                }
            }
            OpCode::IntGe => {
                if is_true {
                    self.make_int_ge(arg0, arg1, ctx);
                } else {
                    self.make_int_lt(arg0, arg1, ctx);
                }
            }
            OpCode::IntEq => {
                if is_true {
                    self.make_eq(arg0, arg1, ctx);
                } else {
                    self.make_ne(arg0, arg1, ctx);
                }
            }
            OpCode::IntNe => {
                if is_true {
                    self.make_ne(arg0, arg1, ctx);
                } else {
                    self.make_eq(arg0, arg1, ctx);
                }
            }
            OpCode::UintLt => {
                if is_true {
                    self.make_unsigned_lt(arg0, arg1, ctx);
                } else {
                    self.make_unsigned_ge(arg0, arg1, ctx);
                }
            }
            OpCode::UintLe => {
                if is_true {
                    self.make_unsigned_le(arg0, arg1, ctx);
                } else {
                    self.make_unsigned_gt(arg0, arg1, ctx);
                }
            }
            OpCode::UintGt => {
                if is_true {
                    self.make_unsigned_gt(arg0, arg1, ctx);
                } else {
                    self.make_unsigned_le(arg0, arg1, ctx);
                }
            }
            OpCode::UintGe => {
                if is_true {
                    self.make_unsigned_ge(arg0, arg1, ctx);
                } else {
                    self.make_unsigned_lt(arg0, arg1, ctx);
                }
            }
            _ => {}
        }
    }

    // ── Bound narrowing helpers ──

    fn make_int_lt(&mut self, box1: OpRef, box2: OpRef, ctx: &OptContext) {
        let b2 = self.getintbound(box2, ctx);
        let b1 = self.get_bound_mut(box1);
        let _ = b1.make_lt(&b2);
        let b1 = self.getintbound(box1, ctx);
        let b2_mut = self.get_bound_mut(box2);
        let _ = b2_mut.make_gt(&b1);
    }

    fn make_int_le(&mut self, box1: OpRef, box2: OpRef, ctx: &OptContext) {
        let b2 = self.getintbound(box2, ctx);
        let b1 = self.get_bound_mut(box1);
        let _ = b1.make_le(&b2);
        let b1 = self.getintbound(box1, ctx);
        let b2_mut = self.get_bound_mut(box2);
        let _ = b2_mut.make_ge(&b1);
    }

    fn make_int_gt(&mut self, box1: OpRef, box2: OpRef, ctx: &OptContext) {
        self.make_int_lt(box2, box1, ctx);
    }

    fn make_int_ge(&mut self, box1: OpRef, box2: OpRef, ctx: &OptContext) {
        self.make_int_le(box2, box1, ctx);
    }

    fn make_unsigned_lt(&mut self, box1: OpRef, box2: OpRef, ctx: &OptContext) {
        let b2 = self.getintbound(box2, ctx);
        let b1 = self.get_bound_mut(box1);
        let _ = b1.make_unsigned_lt(&b2);
        let b1 = self.getintbound(box1, ctx);
        let b2_mut = self.get_bound_mut(box2);
        let _ = b2_mut.make_unsigned_gt(&b1);
    }

    fn make_unsigned_le(&mut self, box1: OpRef, box2: OpRef, ctx: &OptContext) {
        let b2 = self.getintbound(box2, ctx);
        let b1 = self.get_bound_mut(box1);
        let _ = b1.make_unsigned_le(&b2);
        let b1 = self.getintbound(box1, ctx);
        let b2_mut = self.get_bound_mut(box2);
        let _ = b2_mut.make_unsigned_ge(&b1);
    }

    fn make_unsigned_gt(&mut self, box1: OpRef, box2: OpRef, ctx: &OptContext) {
        self.make_unsigned_lt(box2, box1, ctx);
    }

    fn make_unsigned_ge(&mut self, box1: OpRef, box2: OpRef, ctx: &OptContext) {
        self.make_unsigned_le(box2, box1, ctx);
    }

    fn make_eq(&mut self, arg0: OpRef, arg1: OpRef, ctx: &OptContext) {
        let b1 = self.getintbound(arg1, ctx);
        let b0 = self.get_bound_mut(arg0);
        let _ = b0.intersect(&b1);
        let b0 = self.getintbound(arg0, ctx);
        let b1_mut = self.get_bound_mut(arg1);
        let _ = b1_mut.intersect(&b0);
    }

    fn make_ne(&mut self, arg0: OpRef, arg1: OpRef, ctx: &OptContext) {
        let b1 = self.getintbound(arg1, ctx);
        if b1.is_constant() {
            let v1 = b1.get_constant();
            let b0 = self.get_bound_mut(arg0);
            b0.make_ne_const(v1);
        } else {
            let b0 = self.getintbound(arg0, ctx);
            if b0.is_constant() {
                let v0 = b0.get_constant();
                let b1_mut = self.get_bound_mut(arg1);
                b1_mut.make_ne_const(v0);
            }
        }
    }

    // ── Backward propagation after constant discovery ──

    #[allow(dead_code)]
    fn propagate_bounds_backward(&mut self, opref: OpRef, ctx: &OptContext) {
        let b = self.getintbound(opref, ctx);
        if b.is_constant() {
            // Already a constant - nothing more to propagate
            return;
        }
        // Look at the producing operation for backward propagation
        if let Some(producing_op) = self.find_producing_op(opref, ctx) {
            let producing_op = producing_op.clone();
            self.propagate_bounds_backward_op(&producing_op, ctx);
        }
    }

    #[allow(dead_code)]
    fn propagate_bounds_backward_op(&mut self, op: &Op, ctx: &OptContext) {
        match op.opcode {
            OpCode::IntAdd | OpCode::IntAddOvf => {
                let b1 = self.getintbound(op.arg(0), ctx);
                let b2 = self.getintbound(op.arg(1), ctx);
                let r = self.getintbound(op.pos, ctx);
                let b = r.sub_bound(&b2);
                let b1_mut = self.get_bound_mut(op.arg(0));
                let _ = b1_mut.intersect(&b);
                let b = r.sub_bound(&b1);
                let b2_mut = self.get_bound_mut(op.arg(1));
                let _ = b2_mut.intersect(&b);
            }
            OpCode::IntSub | OpCode::IntSubOvf => {
                let b1 = self.getintbound(op.arg(0), ctx);
                let b2 = self.getintbound(op.arg(1), ctx);
                let r = self.getintbound(op.pos, ctx);
                let b = r.add_bound(&b2);
                let b1_mut = self.get_bound_mut(op.arg(0));
                let _ = b1_mut.intersect(&b);
                let b = r.sub_bound(&b1).neg_bound();
                let b2_mut = self.get_bound_mut(op.arg(1));
                let _ = b2_mut.intersect(&b);
            }
            OpCode::IntMul | OpCode::IntMulOvf => {
                let b1 = self.getintbound(op.arg(0), ctx);
                let b2 = self.getintbound(op.arg(1), ctx);
                if op.opcode != OpCode::IntMulOvf && !b1.mul_bound_cannot_overflow(&b2) {
                    return;
                }
                let r = self.getintbound(op.pos, ctx);
                let b = r.py_div_bound(&b2);
                let _ = self.get_bound_mut(op.arg(0)).intersect(&b);
                let b = r.py_div_bound(&b1);
                let _ = self.get_bound_mut(op.arg(1)).intersect(&b);
            }
            OpCode::IntLshift => {
                let b1 = self.getintbound(op.arg(0), ctx);
                let b2 = self.getintbound(op.arg(1), ctx);
                if !b1.lshift_bound_cannot_overflow(&b2) {
                    return;
                }
                let r = self.getintbound(op.pos, ctx);
                if let Ok(b) = r.lshift_bound_backwards(&b2) {
                    let _ = self.get_bound_mut(op.arg(0)).intersect(&b);
                }
            }
            OpCode::IntRshift => {
                let b2 = self.getintbound(op.arg(1), ctx);
                if !b2.is_constant() {
                    return;
                }
                let r = self.getintbound(op.pos, ctx);
                let b = r.rshift_bound_backwards(&b2);
                let _ = self.get_bound_mut(op.arg(0)).intersect(&b);
            }
            OpCode::UintRshift => {
                let b2 = self.getintbound(op.arg(1), ctx);
                if !b2.is_constant() {
                    return;
                }
                let r = self.getintbound(op.pos, ctx);
                let b = r.urshift_bound_backwards(&b2);
                let _ = self.get_bound_mut(op.arg(0)).intersect(&b);
            }
            OpCode::IntAnd => {
                let r = self.getintbound(op.pos, ctx);
                let b0 = self.getintbound(op.arg(0), ctx);
                let b1 = self.getintbound(op.arg(1), ctx);
                if let Ok(b) = b0.and_bound_backwards(&r) {
                    let _ = self.get_bound_mut(op.arg(1)).intersect(&b);
                }
                if let Ok(b) = b1.and_bound_backwards(&r) {
                    let _ = self.get_bound_mut(op.arg(0)).intersect(&b);
                }
            }
            OpCode::IntOr => {
                let r = self.getintbound(op.pos, ctx);
                let b0 = self.getintbound(op.arg(0), ctx);
                let b1 = self.getintbound(op.arg(1), ctx);
                if let Ok(b) = b0.or_bound_backwards(&r) {
                    let _ = self.get_bound_mut(op.arg(1)).intersect(&b);
                }
                if let Ok(b) = b1.or_bound_backwards(&r) {
                    let _ = self.get_bound_mut(op.arg(0)).intersect(&b);
                }
            }
            OpCode::IntXor => {
                let r = self.getintbound(op.pos, ctx);
                let b0 = self.getintbound(op.arg(0), ctx);
                let b1 = self.getintbound(op.arg(1), ctx);
                // xor is its own inverse
                let b = b0.xor_bound(&r);
                let _ = self.get_bound_mut(op.arg(1)).intersect(&b);
                let b = b1.xor_bound(&r);
                let _ = self.get_bound_mut(op.arg(0)).intersect(&b);
            }
            OpCode::IntInvert => {
                let bres = self.getintbound(op.pos, ctx);
                let bounds = bres.invert_bound();
                let _ = self.get_bound_mut(op.arg(0)).intersect(&bounds);
            }
            OpCode::IntNeg => {
                let bres = self.getintbound(op.pos, ctx);
                let bounds = bres.neg_bound();
                let _ = self.get_bound_mut(op.arg(0)).intersect(&bounds);
            }
            // intbounds.py: propagate_bounds_INT_EQ
            // If result is known 1 (true): arg0 == arg1 → intersect bounds
            // If result is known 0 (false): no bounds propagation for !=
            OpCode::IntEq => {
                let r = self.getintbound(op.pos, ctx);
                if r.is_constant() && r.lower == 1 {
                    // make_eq: intersect arg0 with arg1's bounds and vice versa
                    let b0 = self.getintbound(op.arg(0), ctx);
                    let b1 = self.getintbound(op.arg(1), ctx);
                    let _ = self.get_bound_mut(op.arg(0)).intersect(&b1);
                    let _ = self.get_bound_mut(op.arg(1)).intersect(&b0);
                }
            }
            // intbounds.py: propagate_bounds_INT_NE
            // If result is known 0 (false): arg0 == arg1 → intersect bounds
            OpCode::IntNe => {
                let r = self.getintbound(op.pos, ctx);
                if r.is_constant() && r.lower == 0 {
                    let b0 = self.getintbound(op.arg(0), ctx);
                    let b1 = self.getintbound(op.arg(1), ctx);
                    let _ = self.get_bound_mut(op.arg(0)).intersect(&b1);
                    let _ = self.get_bound_mut(op.arg(1)).intersect(&b0);
                }
            }
            _ => {}
        }
    }

    /// Record what we last emitted (for overflow guard removal).
    fn record_emitted(&mut self, op: &Op) {
        self.last_emitted_opcode = Some(op.opcode);
        self.last_emitted_args = op.args.to_vec();
        self.last_emitted_ref = op.pos;
        if !matches!(
            op.opcode,
            OpCode::IntAdd
                | OpCode::IntSub
                | OpCode::IntMul
                | OpCode::IntAddOvf
                | OpCode::IntSubOvf
                | OpCode::IntMulOvf
                | OpCode::GuardNoOverflow
                | OpCode::GuardOverflow
        ) {
            self.pending_overflow_guard = None;
        }
    }
}

impl Default for OptIntBounds {
    fn default() -> Self {
        Self::new()
    }
}

impl Optimization for OptIntBounds {
    fn propagate_forward(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        // RPython emit-then-propagate parity: retry deferred guard bounds.
        self.flush_pending_guard_bounds(ctx);

        let result = match op.opcode {
            // ── Comparisons ──
            OpCode::IntLt => self.optimize_int_lt(op, ctx),
            OpCode::IntLe => self.optimize_int_le(op, ctx),
            OpCode::IntGt => self.optimize_int_gt(op, ctx),
            OpCode::IntGe => self.optimize_int_ge(op, ctx),
            OpCode::IntEq => self.optimize_int_eq(op, ctx),
            OpCode::IntNe => self.optimize_int_ne(op, ctx),
            OpCode::UintLt => self.optimize_uint_lt(op, ctx),
            OpCode::UintLe => self.optimize_uint_le(op, ctx),
            OpCode::UintGt => self.optimize_uint_gt(op, ctx),
            OpCode::UintGe => self.optimize_uint_ge(op, ctx),

            // ── Signext ──
            OpCode::IntSignext => self.optimize_int_signext(op, ctx),

            // ── Overflow arithmetic ──
            OpCode::IntAddOvf => self.optimize_int_add_ovf(op, ctx),
            OpCode::IntSubOvf => self.optimize_int_sub_ovf(op, ctx),
            OpCode::IntMulOvf => self.optimize_int_mul_ovf(op, ctx),

            // ── Overflow guards ──
            OpCode::GuardNoOverflow => self.optimize_guard_no_overflow(op),
            OpCode::GuardOverflow => self.optimize_guard_overflow(op),

            // ── Guards on conditions ──
            OpCode::GuardTrue => self.optimize_guard_true(op, ctx),
            OpCode::GuardFalse => self.optimize_guard_false(op, ctx),

            // ── Arithmetic (postprocess to set bounds, then pass on) ──
            OpCode::IntAdd => {
                self.postprocess_int_add(op, ctx);
                OptimizationResult::PassOn
            }
            OpCode::IntSub => {
                self.postprocess_int_sub(op, ctx);
                OptimizationResult::PassOn
            }
            OpCode::IntMul => {
                self.postprocess_int_mul(op, ctx);
                OptimizationResult::PassOn
            }
            OpCode::IntAnd => {
                self.postprocess_int_and(op, ctx);
                OptimizationResult::PassOn
            }
            OpCode::IntOr => {
                self.postprocess_int_or(op, ctx);
                OptimizationResult::PassOn
            }
            OpCode::IntXor => {
                self.postprocess_int_xor(op, ctx);
                OptimizationResult::PassOn
            }
            OpCode::IntLshift => {
                self.postprocess_int_lshift(op, ctx);
                OptimizationResult::PassOn
            }
            OpCode::IntRshift => {
                self.postprocess_int_rshift(op, ctx);
                OptimizationResult::PassOn
            }
            OpCode::UintRshift => {
                self.postprocess_uint_rshift(op, ctx);
                OptimizationResult::PassOn
            }
            OpCode::IntFloorDiv => {
                self.postprocess_int_floordiv(op, ctx);
                OptimizationResult::PassOn
            }
            OpCode::IntMod => {
                self.postprocess_int_mod(op, ctx);
                OptimizationResult::PassOn
            }
            OpCode::IntNeg => {
                self.postprocess_int_neg(op, ctx);
                OptimizationResult::PassOn
            }
            OpCode::IntInvert => {
                self.postprocess_int_invert(op, ctx);
                OptimizationResult::PassOn
            }
            OpCode::IntForceGeZero => {
                self.postprocess_int_force_ge_zero(op, ctx);
                OptimizationResult::PassOn
            }
            OpCode::IntIsZero | OpCode::IntIsTrue | OpCode::IntBetween => {
                self.postprocess_bool_result(op);
                OptimizationResult::PassOn
            }

            // ── Lengths (always non-negative) ──
            OpCode::ArraylenGc => {
                self.postprocess_arraylen_gc(op);
                OptimizationResult::PassOn
            }
            OpCode::Strlen => {
                self.postprocess_strlen(op);
                OptimizationResult::PassOn
            }
            OpCode::Unicodelen => {
                self.postprocess_unicodelen(op);
                OptimizationResult::PassOn
            }

            // intbounds.py: CALL_PURE_I/CALL_I — propagate bounds from oopspec.
            // OS_INT_PY_DIV/OS_INT_PY_MOD: result bounded by divisor.
            OpCode::CallPureI | OpCode::CallI => {
                if let Some(ref d) = op.descr {
                    if let Some(cd) = d.as_call_descr() {
                        let ei = cd.effect_info();
                        match ei.oopspec_index {
                            majit_ir::OopSpecIndex::IntPyDiv => {
                                // Python integer division: result sign = sign(divisor)
                                // |result| <= |dividend|
                                // For positive divisor: 0 <= result < divisor
                                if op.num_args() >= 3 {
                                    let divisor_bound = self.getintbound(op.arg(2), ctx);
                                    if divisor_bound.known_positive() {
                                        let result_bound = IntBound::new(
                                            0,
                                            divisor_bound.upper.saturating_sub(1),
                                            0,
                                            u64::MAX,
                                        );
                                        self.intersect_bound(op.pos, &result_bound);
                                    }
                                }
                            }
                            majit_ir::OopSpecIndex::IntPyMod => {
                                // Python modulo: 0 <= result < |divisor| for positive divisor
                                if op.num_args() >= 3 {
                                    let divisor_bound = self.getintbound(op.arg(2), ctx);
                                    if divisor_bound.known_positive() {
                                        let result_bound = IntBound::new(
                                            0,
                                            divisor_bound.upper.saturating_sub(1),
                                            0,
                                            u64::MAX,
                                        );
                                        self.intersect_bound(op.pos, &result_bound);
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                }
                OptimizationResult::PassOn
            }

            // intbounds.py: postprocess_STRGETITEM — result in [0, 255].
            OpCode::Strgetitem => {
                self.intersect_bound(op.pos, &IntBound::bounded(0, 255));
                OptimizationResult::PassOn
            }

            // intbounds.py: postprocess_UNICODEGETITEM — result >= 0.
            OpCode::Unicodegetitem => {
                self.intersect_bound(op.pos, &IntBound::nonnegative());
                OptimizationResult::PassOn
            }

            // intbounds.py: postprocess_GETFIELD_RAW_I — integer-bounded fields.
            // If the descriptor indicates a bounded integer field (e.g. u8, u16),
            // narrow the result bound to [min, max].
            // RPython aliases all GETFIELD/GETINTERIORFIELD variants (I/R/F)
            // to the same handler; is_integer_bounded() returns false for
            // non-integer fields so the intersect is effectively a no-op.
            OpCode::GetfieldRawI
            | OpCode::GetfieldGcI
            | OpCode::GetinteriorfieldGcI
            | OpCode::GetfieldRawR
            | OpCode::GetfieldGcR
            | OpCode::GetinteriorfieldGcR
            | OpCode::GetfieldRawF
            | OpCode::GetfieldGcF
            | OpCode::GetinteriorfieldGcF => {
                if let Some(ref d) = op.descr {
                    let (field_size, signed) = d.field_size_and_sign();
                    if field_size > 0 && field_size < 8 {
                        let (lo, hi) = if signed {
                            let half = 1i64 << (field_size * 8 - 1);
                            (-half, half - 1)
                        } else {
                            (0, (1i64 << (field_size * 8)) - 1)
                        };
                        self.intersect_bound(op.pos, &IntBound::bounded(lo, hi));
                    }
                }
                OptimizationResult::PassOn
            }

            // intbounds.py: postprocess_GETARRAYITEM_RAW_I — bounded array items.
            // RPython aliases all GETARRAYITEM variants to the same handler.
            OpCode::GetarrayitemRawI | OpCode::GetarrayitemGcI => {
                if let Some(ref d) = op.descr {
                    if let Some(ad) = d.as_array_descr() {
                        let item_size = ad.item_size();
                        if item_size > 0 && item_size < 8 {
                            let signed = ad.is_item_signed();
                            let (lo, hi) = if signed {
                                let half = 1i64 << (item_size * 8 - 1);
                                (-half, half - 1)
                            } else {
                                (0, (1i64 << (item_size * 8)) - 1)
                            };
                            self.intersect_bound(op.pos, &IntBound::bounded(lo, hi));
                        }
                    }
                }
                OptimizationResult::PassOn
            }

            // intbounds.py: postprocess_STRLEN — result >= 0.
            OpCode::Strlen | OpCode::Unicodelen => {
                self.intersect_bound(op.pos, &IntBound::nonnegative());
                OptimizationResult::PassOn
            }

            _ => OptimizationResult::PassOn,
        };

        // Track last emitted for overflow guard handling
        match &result {
            OptimizationResult::Emit(emitted_op) => self.record_emitted(emitted_op),
            OptimizationResult::PassOn => self.record_emitted(op),
            _ => {
                // Remove: don't update last_emitted
            }
        }

        // optimizer.py:415-426 parity: sync bounds to OptContext so that
        // later passes calling make_constant can validate int ranges.
        ctx.int_bounds.clone_from(&self.bounds);

        result
    }

    fn setup(&mut self) {
        self.bounds.clear();
        self.last_emitted_opcode = None;
        self.last_emitted_args.clear();
        self.last_emitted_ref = OpRef::NONE;
        self.pending_overflow_guard = None;
    }

    fn name(&self) -> &'static str {
        "intbounds"
    }

    /// intbounds.py: produce_potential_short_preamble_ops(sb)
    /// Contribute bounds guards to the short preamble.
    fn produce_potential_short_preamble_ops(
        &self,
        _sb: &mut crate::optimizeopt::shortpreamble::ShortBoxes,
        _ctx: &OptContext,
    ) {
        // In RPython, this adds INT_GE/INT_LE guards for known bounds
        // that the loop body depends on. The bounds are discovered during
        // preamble optimization and must be re-checked on bridge entry.
        // The actual bounds are in self.bounds — a real implementation
        // would iterate non-trivial bounds and generate guard ops.
    }

    fn export_arg_int_bounds(
        &self,
        args: &[OpRef],
        ctx: &OptContext,
    ) -> std::collections::HashMap<OpRef, IntBound> {
        let mut exported = std::collections::HashMap::new();
        for &arg in args {
            let resolved = ctx.get_box_replacement(arg);
            let bound = self.getintbound(resolved, ctx);
            if !bound.is_unbounded() {
                exported.insert(resolved, bound);
            }
        }
        exported
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizeopt::optimizer::Optimizer;

    fn run_pass(ops: &[Op]) -> Vec<Op> {
        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptIntBounds::new()));
        opt.propagate_all_forward(ops)
    }

    /// Create an OptIntBounds with specific bounds pre-set and run it on ops.
    fn run_pass_with_bounds(
        ops: &[Op],
        initial_bounds: &[(OpRef, IntBound)],
    ) -> (Vec<Op>, OptIntBounds, OptContext) {
        let mut pass = OptIntBounds::new();
        let mut ctx = OptContext::new(ops.len());

        pass.setup();
        for (opref, bound) in initial_bounds {
            pass.setintbound(*opref, bound.clone());
        }

        for op in ops.iter() {
            let mut resolved_op = op.clone();
            // Keep the op.pos as set by the test (not overriding with index)
            for arg in &mut resolved_op.args {
                *arg = ctx.get_box_replacement(*arg);
            }
            match pass.propagate_forward(&resolved_op, &mut ctx) {
                OptimizationResult::Emit(emit_op) => {
                    ctx.emit(emit_op);
                }
                OptimizationResult::Replace(rep_op) => {
                    ctx.emit(rep_op);
                }
                OptimizationResult::Remove => {}
                OptimizationResult::PassOn => {
                    ctx.emit(resolved_op);
                }
                OptimizationResult::InvalidLoop => {
                    panic!("unexpected InvalidLoop in test");
                }
            }
        }

        let result = ctx.new_operations.clone();
        (result, pass, ctx)
    }

    fn make_op(opcode: OpCode, args: &[OpRef], pos: u32) -> Op {
        let mut op = Op::new(opcode, args);
        op.pos = OpRef(pos);
        op
    }

    // ── Test: INT_ADD narrows bounds ──

    #[test]
    fn test_int_add_narrows_bounds() {
        // Set up two operands with known bounds and add them
        let initial_bounds = vec![
            (OpRef(0), IntBound::bounded(0, 10)),
            (OpRef(1), IntBound::bounded(5, 20)),
        ];
        let ops = vec![make_op(OpCode::IntAdd, &[OpRef(0), OpRef(1)], 2)];

        let (result, pass, _ctx) = run_pass_with_bounds(&ops, &initial_bounds);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].opcode, OpCode::IntAdd);

        // The result should have bounds [5, 30]
        let b = pass.bounds[2].as_ref().unwrap();
        assert_eq!(b.lower, 5);
        assert_eq!(b.upper, 30);
    }

    // ── Test: Guard removal when bounds prove the condition ──

    #[test]
    fn test_guard_removal_known_true() {
        // i0 in [10, 20], i1 in [0, 5]
        // INT_LT(i1, i0) is always true (5 < 10)
        // GUARD_TRUE on it should be removable
        let initial_bounds = vec![
            (OpRef(0), IntBound::bounded(10, 20)),
            (OpRef(1), IntBound::bounded(0, 5)),
        ];
        let ops = vec![
            // i2 = INT_LT(i1, i0)  -- i1 < i0 is always true
            make_op(OpCode::IntLt, &[OpRef(1), OpRef(0)], 2),
            // GUARD_TRUE(i2)
            make_op(OpCode::GuardTrue, &[OpRef(2)], 3),
        ];

        let (result, _pass, _ctx) = run_pass_with_bounds(&ops, &initial_bounds);
        // INT_LT should be removed (replaced by constant 1)
        // GUARD_TRUE on a known constant 1 should also be removed
        assert!(
            result.is_empty()
                || result
                    .iter()
                    .all(|op| op.opcode != OpCode::GuardTrue && op.opcode != OpCode::IntLt),
            "Guard and comparison should both be removed, got: {:?}",
            result.iter().map(|o| o.opcode).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_guard_removal_known_false() {
        // i0 in [10, 20], i1 in [0, 5]
        // INT_GE(i1, i0) is always false (5 < 10, so not >=)
        // GUARD_FALSE on it should be removable
        let initial_bounds = vec![
            (OpRef(0), IntBound::bounded(10, 20)),
            (OpRef(1), IntBound::bounded(0, 5)),
        ];
        let ops = vec![
            // i2 = INT_GE(i1, i0)  -- always false
            make_op(OpCode::IntGe, &[OpRef(1), OpRef(0)], 2),
            // GUARD_FALSE(i2)
            make_op(OpCode::GuardFalse, &[OpRef(2)], 3),
        ];

        let (result, _pass, _ctx) = run_pass_with_bounds(&ops, &initial_bounds);
        assert!(
            result.is_empty()
                || result
                    .iter()
                    .all(|op| op.opcode != OpCode::GuardFalse && op.opcode != OpCode::IntGe),
            "Guard and comparison should both be removed, got: {:?}",
            result.iter().map(|o| o.opcode).collect::<Vec<_>>()
        );
    }

    // ── Test: Guard narrowing (bounds updated after guard) ──

    #[test]
    fn test_guard_narrowing_int_lt() {
        // i0 unbounded, i1 in [10, 10] (constant 10)
        // After GUARD_TRUE on INT_LT(i0, i1), i0's upper should be < 10
        let initial_bounds = vec![
            (OpRef(0), IntBound::unbounded()),
            (OpRef(1), IntBound::from_constant(10)),
        ];
        let ops = vec![
            // i2 = INT_LT(i0, i1)
            make_op(OpCode::IntLt, &[OpRef(0), OpRef(1)], 2),
            // GUARD_TRUE(i2)
            make_op(OpCode::GuardTrue, &[OpRef(2)], 3),
        ];

        let (_result, pass, _ctx) = run_pass_with_bounds(&ops, &initial_bounds);

        // After the guard, i0 should be < 10, meaning upper <= 9
        let b0 = pass.bounds[0].as_ref().unwrap();
        assert!(
            b0.upper <= 9,
            "After GUARD_TRUE(INT_LT(i0, 10)), i0.upper should be <= 9, got {}",
            b0.upper
        );
    }

    #[test]
    fn test_guard_narrowing_int_ge() {
        // i0 unbounded, i1 = 5
        // After GUARD_TRUE on INT_GE(i0, i1), i0.lower >= 5
        let initial_bounds = vec![
            (OpRef(0), IntBound::unbounded()),
            (OpRef(1), IntBound::from_constant(5)),
        ];
        let ops = vec![
            make_op(OpCode::IntGe, &[OpRef(0), OpRef(1)], 2),
            make_op(OpCode::GuardTrue, &[OpRef(2)], 3),
        ];

        let (_result, pass, _ctx) = run_pass_with_bounds(&ops, &initial_bounds);

        let b0 = pass.bounds[0].as_ref().unwrap();
        assert!(
            b0.lower >= 5,
            "After GUARD_TRUE(INT_GE(i0, 5)), i0.lower should be >= 5, got {}",
            b0.lower
        );
    }

    // ── Test: Overflow guard removal ──

    #[test]
    fn test_add_ovf_cannot_overflow() {
        // i0 in [0, 100], i1 in [0, 100]
        // INT_ADD_OVF(i0, i1) cannot overflow, should be replaced by INT_ADD
        // GUARD_NO_OVERFLOW should be removed
        let initial_bounds = vec![
            (OpRef(0), IntBound::bounded(0, 100)),
            (OpRef(1), IntBound::bounded(0, 100)),
        ];
        let ops = vec![
            make_op(OpCode::IntAddOvf, &[OpRef(0), OpRef(1)], 2),
            make_op(OpCode::GuardNoOverflow, &[], 3),
        ];

        let (result, _pass, _ctx) = run_pass_with_bounds(&ops, &initial_bounds);
        // The OVF should be replaced with IntAdd, and the guard removed
        assert!(
            result.iter().any(|op| op.opcode == OpCode::IntAdd),
            "INT_ADD_OVF should be transformed to INT_ADD"
        );
        assert!(
            !result.iter().any(|op| op.opcode == OpCode::IntAddOvf),
            "INT_ADD_OVF should not remain"
        );
        assert!(
            !result.iter().any(|op| op.opcode == OpCode::GuardNoOverflow),
            "GUARD_NO_OVERFLOW should be removed"
        );
    }

    #[test]
    fn test_add_ovf_may_overflow() {
        // i0 in [0, i64::MAX - 1], i1 in [0, i64::MAX - 1]
        // INT_ADD_OVF(i0, i1) may overflow, should NOT be replaced
        let initial_bounds = vec![
            (OpRef(0), IntBound::bounded(0, i64::MAX - 1)),
            (OpRef(1), IntBound::bounded(0, i64::MAX - 1)),
        ];
        let ops = vec![
            make_op(OpCode::IntAddOvf, &[OpRef(0), OpRef(1)], 2),
            make_op(OpCode::GuardNoOverflow, &[], 3),
        ];

        let (result, _pass, _ctx) = run_pass_with_bounds(&ops, &initial_bounds);
        assert!(
            result.iter().any(|op| op.opcode == OpCode::IntAddOvf),
            "INT_ADD_OVF should remain when overflow is possible"
        );
        assert!(
            result.iter().any(|op| op.opcode == OpCode::GuardNoOverflow),
            "GUARD_NO_OVERFLOW should remain when overflow is possible"
        );
    }

    #[test]
    fn test_sub_ovf_cannot_overflow() {
        let initial_bounds = vec![
            (OpRef(0), IntBound::bounded(0, 100)),
            (OpRef(1), IntBound::bounded(0, 50)),
        ];
        let ops = vec![
            make_op(OpCode::IntSubOvf, &[OpRef(0), OpRef(1)], 2),
            make_op(OpCode::GuardNoOverflow, &[], 3),
        ];

        let (result, _pass, _ctx) = run_pass_with_bounds(&ops, &initial_bounds);
        assert!(
            result.iter().any(|op| op.opcode == OpCode::IntSub),
            "INT_SUB_OVF should be transformed to INT_SUB"
        );
        assert!(
            !result.iter().any(|op| op.opcode == OpCode::GuardNoOverflow),
            "GUARD_NO_OVERFLOW should be removed"
        );
    }

    #[test]
    fn test_mul_ovf_cannot_overflow() {
        let initial_bounds = vec![
            (OpRef(0), IntBound::bounded(0, 10)),
            (OpRef(1), IntBound::bounded(0, 10)),
        ];
        let ops = vec![
            make_op(OpCode::IntMulOvf, &[OpRef(0), OpRef(1)], 2),
            make_op(OpCode::GuardNoOverflow, &[], 3),
        ];

        let (result, _pass, _ctx) = run_pass_with_bounds(&ops, &initial_bounds);
        assert!(
            result.iter().any(|op| op.opcode == OpCode::IntMul),
            "INT_MUL_OVF should be transformed to INT_MUL"
        );
    }

    #[test]
    fn test_second_overflow_guard_survives_after_first_guard() {
        let initial_bounds = vec![
            (OpRef(0), IntBound::unbounded()),
            (OpRef(1), IntBound::unbounded()),
            (OpRef(2), IntBound::unbounded()),
        ];
        let ops = vec![
            make_op(OpCode::GuardTrue, &[OpRef(1)], 3),
            make_op(OpCode::IntSubOvf, &[OpRef(0), OpRef(1)], 4),
            make_op(OpCode::GuardNoOverflow, &[], 5),
            make_op(OpCode::IntMulOvf, &[OpRef(2), OpRef(1)], 6),
            make_op(OpCode::GuardNoOverflow, &[], 7),
            make_op(OpCode::Jump, &[OpRef(4), OpRef(4), OpRef(6)], 8),
        ];

        let (result, _pass, _ctx) = run_pass_with_bounds(&ops, &initial_bounds);
        let guard_count = result
            .iter()
            .filter(|op| op.opcode == OpCode::GuardNoOverflow)
            .count();
        let opcodes: Vec<_> = result.iter().map(|op| op.opcode).collect();
        assert_eq!(
            guard_count, 2,
            "both overflow guards must remain, got {opcodes:?}"
        );
    }

    // ── Test: Comparison result when bounds determine outcome ──

    #[test]
    fn test_int_lt_known_true() {
        let initial_bounds = vec![
            (OpRef(0), IntBound::bounded(0, 5)),
            (OpRef(1), IntBound::bounded(10, 20)),
        ];
        let ops = vec![make_op(OpCode::IntLt, &[OpRef(0), OpRef(1)], 2)];

        let (result, pass, ctx) = run_pass_with_bounds(&ops, &initial_bounds);
        // Should be removed (replaced by constant 1)
        assert!(
            result.is_empty(),
            "INT_LT should be removed when known true"
        );
        // The constant should be set
        let b = pass.getintbound(OpRef(2), &ctx);
        assert!(b.is_constant() && b.get_constant() == 1);
    }

    #[test]
    fn test_int_lt_known_false() {
        let initial_bounds = vec![
            (OpRef(0), IntBound::bounded(10, 20)),
            (OpRef(1), IntBound::bounded(0, 5)),
        ];
        let ops = vec![make_op(OpCode::IntLt, &[OpRef(0), OpRef(1)], 2)];

        let (result, pass, ctx) = run_pass_with_bounds(&ops, &initial_bounds);
        assert!(
            result.is_empty(),
            "INT_LT should be removed when known false"
        );
        let b = pass.getintbound(OpRef(2), &ctx);
        assert!(b.is_constant() && b.get_constant() == 0);
    }

    #[test]
    fn test_int_eq_same_arg() {
        let ops = vec![make_op(OpCode::IntEq, &[OpRef(0), OpRef(0)], 1)];

        let (result, pass, ctx) = run_pass_with_bounds(&ops, &[]);
        assert!(
            result.is_empty(),
            "INT_EQ(x, x) should be removed (always 1)"
        );
        let b = pass.getintbound(OpRef(1), &ctx);
        assert!(b.is_constant() && b.get_constant() == 1);
    }

    #[test]
    fn test_int_ne_same_arg() {
        let ops = vec![make_op(OpCode::IntNe, &[OpRef(0), OpRef(0)], 1)];

        let (result, pass, ctx) = run_pass_with_bounds(&ops, &[]);
        assert!(
            result.is_empty(),
            "INT_NE(x, x) should be removed (always 0)"
        );
        let b = pass.getintbound(OpRef(1), &ctx);
        assert!(b.is_constant() && b.get_constant() == 0);
    }

    #[test]
    fn test_int_le_known_true() {
        let initial_bounds = vec![
            (OpRef(0), IntBound::bounded(0, 5)),
            (OpRef(1), IntBound::bounded(5, 20)),
        ];
        let ops = vec![make_op(OpCode::IntLe, &[OpRef(0), OpRef(1)], 2)];

        let (result, _pass, _ctx) = run_pass_with_bounds(&ops, &initial_bounds);
        assert!(
            result.is_empty(),
            "INT_LE should be removed when known true"
        );
    }

    #[test]
    fn test_int_ge_known_true() {
        let initial_bounds = vec![
            (OpRef(0), IntBound::bounded(10, 20)),
            (OpRef(1), IntBound::bounded(0, 10)),
        ];
        let ops = vec![make_op(OpCode::IntGe, &[OpRef(0), OpRef(1)], 2)];

        let (result, _pass, _ctx) = run_pass_with_bounds(&ops, &initial_bounds);
        assert!(
            result.is_empty(),
            "INT_GE should be removed when known true"
        );
    }

    // ── Test: Arithmetic bounds propagation ──

    #[test]
    fn test_int_sub_bounds() {
        let initial_bounds = vec![
            (OpRef(0), IntBound::bounded(10, 20)),
            (OpRef(1), IntBound::bounded(0, 5)),
        ];
        let ops = vec![make_op(OpCode::IntSub, &[OpRef(0), OpRef(1)], 2)];

        let (result, pass, _ctx) = run_pass_with_bounds(&ops, &initial_bounds);
        assert_eq!(result.len(), 1);
        let b = pass.bounds[2].as_ref().unwrap();
        // [10, 20] - [0, 5] = [5, 20]
        assert_eq!(b.lower, 5);
        assert_eq!(b.upper, 20);
    }

    #[test]
    fn test_int_mul_bounds() {
        let initial_bounds = vec![
            (OpRef(0), IntBound::bounded(2, 5)),
            (OpRef(1), IntBound::bounded(3, 7)),
        ];
        let ops = vec![make_op(OpCode::IntMul, &[OpRef(0), OpRef(1)], 2)];

        let (result, pass, _ctx) = run_pass_with_bounds(&ops, &initial_bounds);
        assert_eq!(result.len(), 1);
        let b = pass.bounds[2].as_ref().unwrap();
        // [2, 5] * [3, 7] = [6, 35]
        assert_eq!(b.lower, 6);
        assert_eq!(b.upper, 35);
    }

    #[test]
    fn test_int_and_bounds() {
        let initial_bounds = vec![
            (OpRef(0), IntBound::bounded(0, 255)),
            (OpRef(1), IntBound::bounded(0, 15)),
        ];
        let ops = vec![make_op(OpCode::IntAnd, &[OpRef(0), OpRef(1)], 2)];

        let (result, pass, _ctx) = run_pass_with_bounds(&ops, &initial_bounds);
        assert_eq!(result.len(), 1);
        let b = pass.bounds[2].as_ref().unwrap();
        // AND of [0, 255] and [0, 15] -> [0, 15]
        assert!(b.lower >= 0);
        assert!(b.upper <= 15);
    }

    #[test]
    fn test_int_force_ge_zero() {
        let initial_bounds = vec![(OpRef(0), IntBound::bounded(-10, 20))];
        let ops = vec![make_op(OpCode::IntForceGeZero, &[OpRef(0)], 1)];

        let (_result, pass, _ctx) = run_pass_with_bounds(&ops, &initial_bounds);
        let b = pass.bounds[1].as_ref().unwrap();
        assert!(
            b.lower >= 0,
            "INT_FORCE_GE_ZERO result should be >= 0, got {}",
            b.lower
        );
        assert!(
            b.upper <= 20,
            "INT_FORCE_GE_ZERO result upper should be <= 20, got {}",
            b.upper
        );
    }

    #[test]
    fn test_arraylen_nonneg() {
        let ops = vec![make_op(OpCode::ArraylenGc, &[OpRef(0)], 1)];

        let (_result, pass, _ctx) = run_pass_with_bounds(&ops, &[]);
        let b = pass.bounds[1].as_ref().unwrap();
        assert!(b.lower >= 0, "ARRAYLEN_GC result should be non-negative");
    }

    #[test]
    fn test_strlen_nonneg() {
        let ops = vec![make_op(OpCode::Strlen, &[OpRef(0)], 1)];

        let (_result, pass, _ctx) = run_pass_with_bounds(&ops, &[]);
        let b = pass.bounds[1].as_ref().unwrap();
        assert!(b.lower >= 0, "STRLEN result should be non-negative");
    }

    #[test]
    fn test_int_neg_bounds() {
        let initial_bounds = vec![(OpRef(0), IntBound::bounded(3, 10))];
        let ops = vec![make_op(OpCode::IntNeg, &[OpRef(0)], 1)];

        let (_result, pass, _ctx) = run_pass_with_bounds(&ops, &initial_bounds);
        let b = pass.bounds[1].as_ref().unwrap();
        // neg([3, 10]) = [-10, -3]
        assert_eq!(b.lower, -10);
        assert_eq!(b.upper, -3);
    }

    #[test]
    fn test_int_invert_bounds() {
        let initial_bounds = vec![(OpRef(0), IntBound::bounded(3, 10))];
        let ops = vec![make_op(OpCode::IntInvert, &[OpRef(0)], 1)];

        let (_result, pass, _ctx) = run_pass_with_bounds(&ops, &initial_bounds);
        let b = pass.bounds[1].as_ref().unwrap();
        // invert([3, 10]) = [!10, !3] = [-11, -4]
        assert_eq!(b.lower, -11);
        assert_eq!(b.upper, -4);
    }

    #[test]
    fn test_sub_ovf_same_arg() {
        // INT_SUB_OVF(x, x) should be replaced by constant 0
        let initial_bounds = vec![(OpRef(0), IntBound::unbounded())];
        let ops = vec![make_op(OpCode::IntSubOvf, &[OpRef(0), OpRef(0)], 1)];

        let (result, pass, ctx) = run_pass_with_bounds(&ops, &initial_bounds);
        assert!(result.is_empty(), "INT_SUB_OVF(x, x) should be removed");
        let b = pass.getintbound(OpRef(1), &ctx);
        assert!(b.is_constant() && b.get_constant() == 0);
    }

    // ── Test: Unsigned comparison optimization ──

    #[test]
    fn test_uint_lt_known_true() {
        let initial_bounds = vec![
            (OpRef(0), IntBound::bounded(0, 5)),
            (OpRef(1), IntBound::bounded(10, 20)),
        ];
        let ops = vec![make_op(OpCode::UintLt, &[OpRef(0), OpRef(1)], 2)];

        let (result, _pass, _ctx) = run_pass_with_bounds(&ops, &initial_bounds);
        assert!(
            result.is_empty(),
            "UINT_LT should be removed when known true"
        );
    }

    // ── Test: Lshift bounds ──

    #[test]
    fn test_int_lshift_bounds() {
        let initial_bounds = vec![
            (OpRef(0), IntBound::bounded(1, 4)),
            (OpRef(1), IntBound::from_constant(2)),
        ];
        let ops = vec![make_op(OpCode::IntLshift, &[OpRef(0), OpRef(1)], 2)];

        let (_result, pass, _ctx) = run_pass_with_bounds(&ops, &initial_bounds);
        let b = pass.bounds[2].as_ref().unwrap();
        // [1, 4] << 2 = [4, 16]
        assert_eq!(b.lower, 4);
        assert_eq!(b.upper, 16);
    }

    // ── Test: Rshift bounds ──

    #[test]
    fn test_int_rshift_bounds() {
        let initial_bounds = vec![
            (OpRef(0), IntBound::bounded(8, 20)),
            (OpRef(1), IntBound::from_constant(2)),
        ];
        let ops = vec![make_op(OpCode::IntRshift, &[OpRef(0), OpRef(1)], 2)];

        let (_result, pass, _ctx) = run_pass_with_bounds(&ops, &initial_bounds);
        let b = pass.bounds[2].as_ref().unwrap();
        // [8, 20] >> 2 = [2, 5]
        assert_eq!(b.lower, 2);
        assert_eq!(b.upper, 5);
    }

    // ── Test: INT_IS_TRUE and INT_IS_ZERO produce bool bounds ──

    #[test]
    fn test_int_is_true_bool_bounds() {
        let ops = vec![make_op(OpCode::IntIsTrue, &[OpRef(0)], 1)];

        let (_result, pass, _ctx) = run_pass_with_bounds(&ops, &[]);
        let b = pass.bounds[1].as_ref().unwrap();
        assert_eq!(b.lower, 0);
        assert_eq!(b.upper, 1);
    }

    #[test]
    fn test_int_is_zero_bool_bounds() {
        let ops = vec![make_op(OpCode::IntIsZero, &[OpRef(0)], 1)];

        let (_result, pass, _ctx) = run_pass_with_bounds(&ops, &[]);
        let b = pass.bounds[1].as_ref().unwrap();
        assert_eq!(b.lower, 0);
        assert_eq!(b.upper, 1);
    }

    // ── Test: Comparison with unknown bounds stays ──

    #[test]
    fn test_int_lt_unknown_not_removed() {
        let initial_bounds = vec![
            (OpRef(0), IntBound::unbounded()),
            (OpRef(1), IntBound::unbounded()),
        ];
        let ops = vec![make_op(OpCode::IntLt, &[OpRef(0), OpRef(1)], 2)];

        let (result, pass, _ctx) = run_pass_with_bounds(&ops, &initial_bounds);
        assert_eq!(
            result.len(),
            1,
            "INT_LT should remain when bounds are unknown"
        );
        assert_eq!(result[0].opcode, OpCode::IntLt);
        // But the result should have bool bounds [0, 1]
        let b = pass.bounds[2].as_ref().unwrap();
        assert_eq!(b.lower, 0);
        assert_eq!(b.upper, 1);
    }

    // ── Test: INT_SIGNEXT ──

    #[test]
    fn test_int_signext_eliminated() {
        // x in [-100, 100], signext to 2 bytes (-32768..32767) -> identity
        let initial_bounds = vec![
            (OpRef(0), IntBound::bounded(-100, 100)),
            (OpRef(1), IntBound::from_constant(2)), // byte_size = 2
        ];
        let ops = vec![make_op(OpCode::IntSignext, &[OpRef(0), OpRef(1)], 2)];

        let (result, _pass, _ctx) = run_pass_with_bounds(&ops, &initial_bounds);
        assert!(
            result.is_empty(),
            "signext should be eliminated when value fits"
        );
    }

    #[test]
    fn test_int_signext_kept() {
        // x in [-50000, 50000], signext to 1 byte (-128..127) -> can't eliminate
        let initial_bounds = vec![
            (OpRef(0), IntBound::bounded(-50000, 50000)),
            (OpRef(1), IntBound::from_constant(1)), // byte_size = 1
        ];
        let ops = vec![make_op(OpCode::IntSignext, &[OpRef(0), OpRef(1)], 2)];

        let (result, pass, _ctx) = run_pass_with_bounds(&ops, &initial_bounds);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].opcode, OpCode::IntSignext);
        // Result should have bounds [-128, 127]
        let b = pass.bounds[2].as_ref().unwrap();
        assert!(b.lower >= -128);
        assert!(b.upper <= 127);
    }

    // ── Test: Chain of operations with bound propagation ──

    #[test]
    fn test_chain_add_then_compare() {
        // x in [0, 5], y in [0, 5]
        // z = x + y   -> z in [0, 10]
        // z < 100     -> always true
        let initial_bounds = vec![
            (OpRef(0), IntBound::bounded(0, 5)),
            (OpRef(1), IntBound::bounded(0, 5)),
            (OpRef(3), IntBound::from_constant(100)),
        ];
        let ops = vec![
            make_op(OpCode::IntAdd, &[OpRef(0), OpRef(1)], 2),
            make_op(OpCode::IntLt, &[OpRef(2), OpRef(3)], 4),
        ];

        let (result, _pass, ctx) = run_pass_with_bounds(&ops, &initial_bounds);
        // INT_ADD should remain, INT_LT should be eliminated as constant true
        assert_eq!(result.len(), 1, "only INT_ADD should remain");
        assert_eq!(result[0].opcode, OpCode::IntAdd);
        assert_eq!(ctx.get_constant_int(OpRef(4)), Some(1));
    }

    // ── Test: Guard narrowing with INT_IS_TRUE ──

    #[test]
    fn test_guard_true_on_int_is_true() {
        // i0 in [0, 100]
        // i1 = INT_IS_TRUE(i0)
        // GUARD_TRUE(i1)
        // After guard, i0 should have lower > 0
        let initial_bounds = vec![(OpRef(0), IntBound::bounded(0, 100))];
        let ops = vec![
            make_op(OpCode::IntIsTrue, &[OpRef(0)], 1),
            make_op(OpCode::GuardTrue, &[OpRef(1)], 2),
        ];

        let (_result, pass, _ctx) = run_pass_with_bounds(&ops, &initial_bounds);
        let b0 = pass.bounds[0].as_ref().unwrap();
        assert!(
            b0.lower >= 1,
            "After GUARD_TRUE(INT_IS_TRUE(i0)), i0.lower should be >= 1, got {}",
            b0.lower
        );
    }

    #[test]
    fn test_guard_false_on_int_is_true() {
        // i0 in [0, 100]
        // i1 = INT_IS_TRUE(i0)
        // GUARD_FALSE(i1)
        // After guard, i0 should be 0
        let initial_bounds = vec![(OpRef(0), IntBound::bounded(0, 100))];
        let ops = vec![
            make_op(OpCode::IntIsTrue, &[OpRef(0)], 1),
            make_op(OpCode::GuardFalse, &[OpRef(1)], 2),
        ];

        let (_result, pass, _ctx) = run_pass_with_bounds(&ops, &initial_bounds);
        let b0 = pass.bounds[0].as_ref().unwrap();
        assert!(
            b0.is_constant() && b0.get_constant() == 0,
            "After GUARD_FALSE(INT_IS_TRUE(i0)), i0 should be 0, got [{}, {}]",
            b0.lower,
            b0.upper
        );
    }

    // ── Test: x + x bounds ──

    #[test]
    fn test_int_add_x_plus_x() {
        // x in [3, 5], x + x -> should be [6, 10]
        let initial_bounds = vec![(OpRef(0), IntBound::bounded(3, 5))];
        let ops = vec![make_op(OpCode::IntAdd, &[OpRef(0), OpRef(0)], 1)];

        let (_result, pass, _ctx) = run_pass_with_bounds(&ops, &initial_bounds);
        let b = pass.bounds[1].as_ref().unwrap();
        assert!(b.lower >= 6, "lower should be >= 6, got {}", b.lower);
        assert!(b.upper <= 10, "upper should be <= 10, got {}", b.upper);
    }

    // ── Test: Backward propagation through arithmetic ──

    #[test]
    fn test_backward_prop_int_neg() {
        // i0 unbounded
        // i1 = INT_NEG(i0)  -- i1 in [-5, -1] initially
        // After propagation, i0 should have bounds [1, 5]
        let initial_bounds = vec![(OpRef(0), IntBound::unbounded())];
        let ops = vec![make_op(OpCode::IntNeg, &[OpRef(0)], 1)];

        let (_result, mut pass, ctx) = run_pass_with_bounds(&ops, &initial_bounds);
        // Manually tighten the result and trigger backward prop
        pass.setintbound(OpRef(1), IntBound::bounded(-5, -1));
        pass.propagate_bounds_backward(OpRef(1), &ctx);
        let b0 = pass.bounds[0].as_ref().unwrap();
        assert!(
            b0.lower >= 1,
            "backward neg: lower should be >= 1, got {}",
            b0.lower
        );
        assert!(
            b0.upper <= 5,
            "backward neg: upper should be <= 5, got {}",
            b0.upper
        );
    }

    #[test]
    fn test_strgetitem_bounds() {
        // STRGETITEM result should be bounded to [0, 255].
        let mut opt = OptIntBounds::new();
        let mut op = Op::new(OpCode::Strgetitem, &[OpRef(100), OpRef(101)]);
        op.pos = OpRef(102);
        let mut ctx = OptContext::new(10);
        let result = opt.propagate_forward(&op, &mut ctx);
        assert!(matches!(result, OptimizationResult::PassOn));
        // After processing, the bounds for op.pos should be [0, 255].
        let b = opt.getintbound(op.pos, &ctx);
        assert!(b.lower >= 0, "STRGETITEM lower should be >= 0");
        assert!(b.upper <= 255, "STRGETITEM upper should be <= 255");
    }
}
