/// Integer bounds optimization pass.
///
/// Translated from rpython/jit/metainterp/optimizeopt/intbounds.py.
///
/// Propagates integer bounds information through the trace. When a guard tests
/// a condition that is already known true from integer bounds, the guard can be
/// removed. It also narrows bounds after guards and arithmetic operations.

use majit_ir::{Op, OpCode, OpRef, Value};

use crate::intutils::IntBound;
use crate::{OptContext, OptimizationPass, PassResult};

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
}

impl OptIntBounds {
    pub fn new() -> Self {
        OptIntBounds {
            bounds: Vec::new(),
            last_emitted_opcode: None,
            last_emitted_args: Vec::new(),
            last_emitted_ref: OpRef::NONE,
        }
    }

    /// Get or create bounds for an operation.
    fn get_bound(&self, opref: OpRef, ctx: &OptContext) -> IntBound {
        let opref = ctx.get_replacement(opref);
        // Check if there is a known constant
        if let Some(val) = ctx.get_constant_int(opref) {
            return IntBound::from_constant(val);
        }
        let idx = opref.0 as usize;
        if idx < self.bounds.len() {
            if let Some(ref b) = self.bounds[idx] {
                return b.clone();
            }
        }
        IntBound::unbounded()
    }

    /// Store bounds for an operation.
    fn set_bound(&mut self, opref: OpRef, bound: IntBound) {
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

    /// Make an op a known constant integer.
    fn make_constant_int(&mut self, op: &Op, value: i64, ctx: &mut OptContext) {
        ctx.make_constant(op.pos, Value::Int(value));
        self.set_bound(op.pos, IntBound::from_constant(value));
    }

    // ── Comparison optimizations ──

    fn optimize_int_lt(&mut self, op: &Op, ctx: &mut OptContext) -> PassResult {
        let arg0 = ctx.get_replacement(op.arg(0));
        let arg1 = ctx.get_replacement(op.arg(1));
        let b0 = self.get_bound(arg0, ctx);
        let b1 = self.get_bound(arg1, ctx);
        if b0.known_lt(&b1) {
            self.make_constant_int(op, 1, ctx);
            PassResult::Remove
        } else if b0.known_ge(&b1) || arg0 == arg1 {
            self.make_constant_int(op, 0, ctx);
            PassResult::Remove
        } else {
            self.postprocess_bool_result(op);
            PassResult::PassOn
        }
    }

    fn optimize_int_gt(&mut self, op: &Op, ctx: &mut OptContext) -> PassResult {
        let arg0 = ctx.get_replacement(op.arg(0));
        let arg1 = ctx.get_replacement(op.arg(1));
        let b0 = self.get_bound(arg0, ctx);
        let b1 = self.get_bound(arg1, ctx);
        if b0.known_gt(&b1) {
            self.make_constant_int(op, 1, ctx);
            PassResult::Remove
        } else if b0.known_le(&b1) || arg0 == arg1 {
            self.make_constant_int(op, 0, ctx);
            PassResult::Remove
        } else {
            self.postprocess_bool_result(op);
            PassResult::PassOn
        }
    }

    fn optimize_int_le(&mut self, op: &Op, ctx: &mut OptContext) -> PassResult {
        let arg0 = ctx.get_replacement(op.arg(0));
        let arg1 = ctx.get_replacement(op.arg(1));
        let b0 = self.get_bound(arg0, ctx);
        let b1 = self.get_bound(arg1, ctx);
        if b0.known_le(&b1) || arg0 == arg1 {
            self.make_constant_int(op, 1, ctx);
            PassResult::Remove
        } else if b0.known_gt(&b1) {
            self.make_constant_int(op, 0, ctx);
            PassResult::Remove
        } else {
            self.postprocess_bool_result(op);
            PassResult::PassOn
        }
    }

    fn optimize_int_ge(&mut self, op: &Op, ctx: &mut OptContext) -> PassResult {
        let arg0 = ctx.get_replacement(op.arg(0));
        let arg1 = ctx.get_replacement(op.arg(1));
        let b0 = self.get_bound(arg0, ctx);
        let b1 = self.get_bound(arg1, ctx);
        if b0.known_ge(&b1) || arg0 == arg1 {
            self.make_constant_int(op, 1, ctx);
            PassResult::Remove
        } else if b0.known_lt(&b1) {
            self.make_constant_int(op, 0, ctx);
            PassResult::Remove
        } else {
            self.postprocess_bool_result(op);
            PassResult::PassOn
        }
    }

    fn optimize_int_eq(&mut self, op: &Op, ctx: &mut OptContext) -> PassResult {
        let arg0 = ctx.get_replacement(op.arg(0));
        let arg1 = ctx.get_replacement(op.arg(1));
        let b0 = self.get_bound(arg0, ctx);
        let b1 = self.get_bound(arg1, ctx);
        if arg0 == arg1 {
            self.make_constant_int(op, 1, ctx);
            PassResult::Remove
        } else if b0.known_ne(&b1) {
            self.make_constant_int(op, 0, ctx);
            PassResult::Remove
        } else {
            self.postprocess_bool_result(op);
            PassResult::PassOn
        }
    }

    fn optimize_int_ne(&mut self, op: &Op, ctx: &mut OptContext) -> PassResult {
        let arg0 = ctx.get_replacement(op.arg(0));
        let arg1 = ctx.get_replacement(op.arg(1));
        let b0 = self.get_bound(arg0, ctx);
        let b1 = self.get_bound(arg1, ctx);
        if arg0 == arg1 {
            self.make_constant_int(op, 0, ctx);
            PassResult::Remove
        } else if b0.known_ne(&b1) {
            self.make_constant_int(op, 1, ctx);
            PassResult::Remove
        } else {
            self.postprocess_bool_result(op);
            PassResult::PassOn
        }
    }

    // ── Unsigned comparison optimizations ──

    fn optimize_uint_lt(&mut self, op: &Op, ctx: &mut OptContext) -> PassResult {
        let arg0 = ctx.get_replacement(op.arg(0));
        let arg1 = ctx.get_replacement(op.arg(1));
        let b0 = self.get_bound(arg0, ctx);
        let b1 = self.get_bound(arg1, ctx);
        if b0.known_unsigned_lt(&b1) {
            self.make_constant_int(op, 1, ctx);
            PassResult::Remove
        } else if b0.known_unsigned_ge(&b1) || arg0 == arg1 {
            self.make_constant_int(op, 0, ctx);
            PassResult::Remove
        } else {
            self.postprocess_bool_result(op);
            PassResult::PassOn
        }
    }

    fn optimize_uint_gt(&mut self, op: &Op, ctx: &mut OptContext) -> PassResult {
        let arg0 = ctx.get_replacement(op.arg(0));
        let arg1 = ctx.get_replacement(op.arg(1));
        let b0 = self.get_bound(arg0, ctx);
        let b1 = self.get_bound(arg1, ctx);
        if b0.known_unsigned_gt(&b1) {
            self.make_constant_int(op, 1, ctx);
            PassResult::Remove
        } else if b0.known_unsigned_le(&b1) || arg0 == arg1 {
            self.make_constant_int(op, 0, ctx);
            PassResult::Remove
        } else {
            self.postprocess_bool_result(op);
            PassResult::PassOn
        }
    }

    fn optimize_uint_le(&mut self, op: &Op, ctx: &mut OptContext) -> PassResult {
        let arg0 = ctx.get_replacement(op.arg(0));
        let arg1 = ctx.get_replacement(op.arg(1));
        let b0 = self.get_bound(arg0, ctx);
        let b1 = self.get_bound(arg1, ctx);
        if b0.known_unsigned_le(&b1) || arg0 == arg1 {
            self.make_constant_int(op, 1, ctx);
            PassResult::Remove
        } else if b0.known_unsigned_gt(&b1) {
            self.make_constant_int(op, 0, ctx);
            PassResult::Remove
        } else {
            self.postprocess_bool_result(op);
            PassResult::PassOn
        }
    }

    fn optimize_uint_ge(&mut self, op: &Op, ctx: &mut OptContext) -> PassResult {
        let arg0 = ctx.get_replacement(op.arg(0));
        let arg1 = ctx.get_replacement(op.arg(1));
        let b0 = self.get_bound(arg0, ctx);
        let b1 = self.get_bound(arg1, ctx);
        if b0.known_unsigned_ge(&b1) || arg0 == arg1 {
            self.make_constant_int(op, 1, ctx);
            PassResult::Remove
        } else if b0.known_unsigned_lt(&b1) {
            self.make_constant_int(op, 0, ctx);
            PassResult::Remove
        } else {
            self.postprocess_bool_result(op);
            PassResult::PassOn
        }
    }

    /// Set result bounds to [0, 1] for comparison/boolean-result operations.
    fn postprocess_bool_result(&mut self, op: &Op) {
        self.set_bound(op.pos, IntBound::bounded(0, 1));
    }

    // ── Arithmetic postprocessing ──

    fn postprocess_int_add(&mut self, op: &Op, ctx: &OptContext) {
        let arg0 = ctx.get_replacement(op.arg(0));
        let arg1 = ctx.get_replacement(op.arg(1));
        let b0 = self.get_bound(arg0, ctx);
        let b = if arg0 == arg1 {
            // x + x is even, equivalent to x << 1
            b0.lshift_bound(&IntBound::from_constant(1))
        } else {
            let b1 = self.get_bound(arg1, ctx);
            b0.add_bound(&b1)
        };
        self.intersect_bound(op.pos, &b);
    }

    fn postprocess_int_sub(&mut self, op: &Op, ctx: &OptContext) {
        let b0 = self.get_bound(op.arg(0), ctx);
        let b1 = self.get_bound(op.arg(1), ctx);
        let b = b0.sub_bound(&b1);
        self.intersect_bound(op.pos, &b);
    }

    fn postprocess_int_mul(&mut self, op: &Op, ctx: &OptContext) {
        let b0 = self.get_bound(op.arg(0), ctx);
        let b1 = self.get_bound(op.arg(1), ctx);
        let b = b0.mul_bound(&b1);
        self.intersect_bound(op.pos, &b);
    }

    fn postprocess_int_and(&mut self, op: &Op, ctx: &OptContext) {
        let b0 = self.get_bound(op.arg(0), ctx);
        let b1 = self.get_bound(op.arg(1), ctx);
        let b = b0.and_bound(&b1);
        self.intersect_bound(op.pos, &b);
    }

    fn postprocess_int_or(&mut self, op: &Op, ctx: &OptContext) {
        let b0 = self.get_bound(op.arg(0), ctx);
        let b1 = self.get_bound(op.arg(1), ctx);
        let b = b0.or_bound(&b1);
        self.intersect_bound(op.pos, &b);
    }

    fn postprocess_int_xor(&mut self, op: &Op, ctx: &OptContext) {
        let b0 = self.get_bound(op.arg(0), ctx);
        let b1 = self.get_bound(op.arg(1), ctx);
        let b = b0.xor_bound(&b1);
        self.intersect_bound(op.pos, &b);
    }

    fn postprocess_int_lshift(&mut self, op: &Op, ctx: &OptContext) {
        let b0 = self.get_bound(op.arg(0), ctx);
        let b1 = self.get_bound(op.arg(1), ctx);
        let b = b0.lshift_bound(&b1);
        self.intersect_bound(op.pos, &b);
    }

    fn postprocess_int_rshift(&mut self, op: &Op, ctx: &OptContext) {
        let b0 = self.get_bound(op.arg(0), ctx);
        let b1 = self.get_bound(op.arg(1), ctx);
        let b = b0.rshift_bound(&b1);
        self.intersect_bound(op.pos, &b);
    }

    fn postprocess_uint_rshift(&mut self, op: &Op, ctx: &OptContext) {
        let b0 = self.get_bound(op.arg(0), ctx);
        let b1 = self.get_bound(op.arg(1), ctx);
        let b = b0.urshift_bound(&b1);
        self.intersect_bound(op.pos, &b);
    }

    fn postprocess_int_neg(&mut self, op: &Op, ctx: &OptContext) {
        let b = self.get_bound(op.arg(0), ctx);
        let result = b.neg_bound();
        self.intersect_bound(op.pos, &result);
    }

    fn postprocess_int_invert(&mut self, op: &Op, ctx: &OptContext) {
        let b = self.get_bound(op.arg(0), ctx);
        let result = b.invert_bound();
        self.intersect_bound(op.pos, &result);
    }

    fn postprocess_int_force_ge_zero(&mut self, op: &Op, ctx: &OptContext) {
        let b_arg = self.get_bound(op.arg(0), ctx);
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

    fn optimize_int_signext(&mut self, op: &Op, ctx: &mut OptContext) -> PassResult {
        let b = self.get_bound(op.arg(0), ctx);
        let b1 = self.get_bound(op.arg(1), ctx);
        if b1.is_constant() {
            let byte_size = b1.get_constant();
            let numbits = byte_size * 8;
            let start = -(1i64 << (numbits - 1));
            let stop = 1i64 << (numbits - 1);
            if b.is_within_range(start, stop - 1) {
                // The value already fits; replace with the input.
                ctx.replace_op(op.pos, op.arg(0));
                return PassResult::Remove;
            }
        }
        self.postprocess_int_signext(op, ctx);
        PassResult::PassOn
    }

    fn postprocess_int_signext(&mut self, op: &Op, ctx: &OptContext) {
        let b1 = self.get_bound(op.arg(1), ctx);
        if b1.is_constant() {
            let byte_size = b1.get_constant();
            let numbits = byte_size * 8;
            let start = -(1i64 << (numbits - 1));
            let stop = 1i64 << (numbits - 1);
            let _ = self.get_bound_mut(op.pos).intersect_const(start, stop - 1);
        }
    }

    // ── Overflow operations ──

    fn optimize_int_add_ovf(&mut self, op: &Op, ctx: &mut OptContext) -> PassResult {
        let b0 = self.get_bound(op.arg(0), ctx);
        let b1 = self.get_bound(op.arg(1), ctx);
        if b0.add_bound_cannot_overflow(&b1) {
            // Transform to non-overflow INT_ADD. The following GUARD_NO_OVERFLOW
            // will be removed because last_emitted_opcode will be IntAdd.
            let mut new_op = Op::new(OpCode::IntAdd, &op.args);
            new_op.descr = op.descr.clone();
            new_op.pos = op.pos;
            self.postprocess_int_add(&new_op, ctx);
            PassResult::Emit(new_op)
        } else {
            self.postprocess_int_add_ovf(op, ctx);
            PassResult::PassOn
        }
    }

    fn postprocess_int_add_ovf(&mut self, op: &Op, ctx: &OptContext) {
        let arg0 = ctx.get_replacement(op.arg(0));
        let arg1 = ctx.get_replacement(op.arg(1));
        let b0 = self.get_bound(arg0, ctx);
        let b = if arg0 == arg1 {
            b0.mul2_bound_no_overflow()
        } else {
            let b1 = self.get_bound(arg1, ctx);
            b0.add_bound_no_overflow(&b1)
        };
        self.intersect_bound(op.pos, &b);
    }

    fn optimize_int_sub_ovf(&mut self, op: &Op, ctx: &mut OptContext) -> PassResult {
        let arg0 = ctx.get_replacement(op.arg(0));
        let arg1 = ctx.get_replacement(op.arg(1));
        let b0 = self.get_bound(arg0, ctx);
        let b1 = self.get_bound(arg1, ctx);
        if arg0 == arg1 {
            // x - x = 0
            self.make_constant_int(op, 0, ctx);
            return PassResult::Remove;
        }
        if b0.sub_bound_cannot_overflow(&b1) {
            let mut new_op = Op::new(OpCode::IntSub, &op.args);
            new_op.descr = op.descr.clone();
            new_op.pos = op.pos;
            self.postprocess_int_sub(&new_op, ctx);
            PassResult::Emit(new_op)
        } else {
            self.postprocess_int_sub_ovf(op, ctx);
            PassResult::PassOn
        }
    }

    fn postprocess_int_sub_ovf(&mut self, op: &Op, ctx: &OptContext) {
        let b0 = self.get_bound(op.arg(0), ctx);
        let b1 = self.get_bound(op.arg(1), ctx);
        let b = b0.sub_bound_no_overflow(&b1);
        self.intersect_bound(op.pos, &b);
    }

    fn optimize_int_mul_ovf(&mut self, op: &Op, ctx: &mut OptContext) -> PassResult {
        let b0 = self.get_bound(op.arg(0), ctx);
        let b1 = self.get_bound(op.arg(1), ctx);
        if b0.mul_bound_cannot_overflow(&b1) {
            let mut new_op = Op::new(OpCode::IntMul, &op.args);
            new_op.descr = op.descr.clone();
            new_op.pos = op.pos;
            self.postprocess_int_mul(&new_op, ctx);
            PassResult::Emit(new_op)
        } else {
            self.postprocess_int_mul_ovf(op, ctx);
            PassResult::PassOn
        }
    }

    fn postprocess_int_mul_ovf(&mut self, op: &Op, ctx: &OptContext) {
        let arg0 = ctx.get_replacement(op.arg(0));
        let arg1 = ctx.get_replacement(op.arg(1));
        let b0 = self.get_bound(arg0, ctx);
        let b = if arg0 == arg1 {
            b0.square_bound_no_overflow()
        } else {
            let b1 = self.get_bound(arg1, ctx);
            b0.mul_bound_no_overflow(&b1)
        };
        self.intersect_bound(op.pos, &b);
    }

    fn optimize_guard_no_overflow(&mut self, op: &Op) -> PassResult {
        // If the INT_xxx_OVF was replaced with INT_xxx, remove the guard.
        match self.last_emitted_opcode {
            Some(OpCode::IntAddOvf | OpCode::IntSubOvf | OpCode::IntMulOvf) => {
                // The OVF op is still present, keep the guard
                PassResult::PassOn
            }
            _ => {
                // The OVF was replaced with a non-overflow op or removed.
                // The guard is redundant.
                let _ = op;
                PassResult::Remove
            }
        }
    }

    fn optimize_guard_overflow(&mut self, op: &Op) -> PassResult {
        match self.last_emitted_opcode {
            Some(OpCode::IntAddOvf | OpCode::IntSubOvf | OpCode::IntMulOvf) => {
                // The OVF is still present, keep the guard
                let _ = op;
                PassResult::PassOn
            }
            _ => {
                // The OVF was proven not to overflow; this GUARD_OVERFLOW
                // means the loop is invalid. We can't raise here, so just
                // pass it on.
                PassResult::PassOn
            }
        }
    }

    // ── Guard optimizations ──

    fn optimize_guard_true(&mut self, op: &Op, ctx: &mut OptContext) -> PassResult {
        let cond_ref = ctx.get_replacement(op.arg(0));

        // If the condition is a known constant, we can determine the guard outcome
        if let Some(val) = ctx.get_constant_int(cond_ref) {
            if val != 0 {
                // Guard always passes, remove it
                return PassResult::Remove;
            }
            // Guard always fails - still emit it (will fail at runtime)
        }

        // Check if the bound on the condition tells us it's always true
        let b = self.get_bound(cond_ref, ctx);
        if b.known_gt_const(0) {
            // known nonzero, guard always passes
            return PassResult::Remove;
        }

        // After emitting, propagate bounds backward from the guard
        // The condition is known to be true (nonzero) after this guard
        self.propagate_bounds_from_guard_true(cond_ref, ctx);
        PassResult::PassOn
    }

    fn optimize_guard_false(&mut self, op: &Op, ctx: &mut OptContext) -> PassResult {
        let cond_ref = ctx.get_replacement(op.arg(0));

        if let Some(val) = ctx.get_constant_int(cond_ref) {
            if val == 0 {
                return PassResult::Remove;
            }
        }

        let b = self.get_bound(cond_ref, ctx);
        if b.known_eq_const(0) {
            return PassResult::Remove;
        }

        self.propagate_bounds_from_guard_false(cond_ref, ctx);
        PassResult::PassOn
    }

    /// Find the operation that produced cond_ref by searching new_operations.
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
        // Set the condition's bound to 1 (we know it's true)
        self.set_bound(cond_ref, IntBound::from_constant(1));

        if let Some(producing_op) = self.find_producing_op(cond_ref, ctx) {
            let producing_op = producing_op.clone();
            self.propagate_bounds_from_comparison(&producing_op, true, ctx);
        }
    }

    /// Propagate bounds backward after GUARD_FALSE.
    /// The condition is known to produce 0 (false).
    fn propagate_bounds_from_guard_false(&mut self, cond_ref: OpRef, ctx: &OptContext) {
        self.set_bound(cond_ref, IntBound::from_constant(0));

        if let Some(producing_op) = self.find_producing_op(cond_ref, ctx) {
            let producing_op = producing_op.clone();
            self.propagate_bounds_from_comparison(&producing_op, false, ctx);
        }
    }

    /// Given that a comparison op produced `is_true`, narrow the bounds of its arguments.
    fn propagate_bounds_from_comparison(&mut self, op: &Op, is_true: bool, ctx: &OptContext) {
        let arg0 = op.arg(0);

        // Handle unary ops
        match op.opcode {
            OpCode::IntIsTrue => {
                if is_true {
                    // nonzero
                    let b0 = self.get_bound(arg0, ctx);
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
                    let b0 = self.get_bound(arg0, ctx);
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

        let arg1 = if op.num_args() > 1 { op.arg(1) } else { return };

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
        let b2 = self.get_bound(box2, ctx);
        let b1 = self.get_bound_mut(box1);
        let _ = b1.make_lt(&b2);
        let b1 = self.get_bound(box1, ctx);
        let b2_mut = self.get_bound_mut(box2);
        let _ = b2_mut.make_gt(&b1);
    }

    fn make_int_le(&mut self, box1: OpRef, box2: OpRef, ctx: &OptContext) {
        let b2 = self.get_bound(box2, ctx);
        let b1 = self.get_bound_mut(box1);
        let _ = b1.make_le(&b2);
        let b1 = self.get_bound(box1, ctx);
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
        let b2 = self.get_bound(box2, ctx);
        let b1 = self.get_bound_mut(box1);
        let _ = b1.make_unsigned_lt(&b2);
        let b1 = self.get_bound(box1, ctx);
        let b2_mut = self.get_bound_mut(box2);
        let _ = b2_mut.make_unsigned_gt(&b1);
    }

    fn make_unsigned_le(&mut self, box1: OpRef, box2: OpRef, ctx: &OptContext) {
        let b2 = self.get_bound(box2, ctx);
        let b1 = self.get_bound_mut(box1);
        let _ = b1.make_unsigned_le(&b2);
        let b1 = self.get_bound(box1, ctx);
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
        let b1 = self.get_bound(arg1, ctx);
        let b0 = self.get_bound_mut(arg0);
        let _ = b0.intersect(&b1);
        let b0 = self.get_bound(arg0, ctx);
        let b1_mut = self.get_bound_mut(arg1);
        let _ = b1_mut.intersect(&b0);
    }

    fn make_ne(&mut self, arg0: OpRef, arg1: OpRef, ctx: &OptContext) {
        let b1 = self.get_bound(arg1, ctx);
        if b1.is_constant() {
            let v1 = b1.get_constant();
            let b0 = self.get_bound_mut(arg0);
            b0.make_ne_const(v1);
        } else {
            let b0 = self.get_bound(arg0, ctx);
            if b0.is_constant() {
                let v0 = b0.get_constant();
                let b1_mut = self.get_bound_mut(arg1);
                b1_mut.make_ne_const(v0);
            }
        }
    }

    // ── Backward propagation after constant discovery ──

    fn propagate_bounds_backward(&mut self, opref: OpRef, ctx: &OptContext) {
        let b = self.get_bound(opref, ctx);
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

    fn propagate_bounds_backward_op(&mut self, op: &Op, ctx: &OptContext) {
        match op.opcode {
            OpCode::IntAdd | OpCode::IntAddOvf => {
                let b1 = self.get_bound(op.arg(0), ctx);
                let b2 = self.get_bound(op.arg(1), ctx);
                let r = self.get_bound(op.pos, ctx);
                let b = r.sub_bound(&b2);
                let b1_mut = self.get_bound_mut(op.arg(0));
                let _ = b1_mut.intersect(&b);
                let b = r.sub_bound(&b1);
                let b2_mut = self.get_bound_mut(op.arg(1));
                let _ = b2_mut.intersect(&b);
            }
            OpCode::IntSub | OpCode::IntSubOvf => {
                let b1 = self.get_bound(op.arg(0), ctx);
                let b2 = self.get_bound(op.arg(1), ctx);
                let r = self.get_bound(op.pos, ctx);
                let b = r.add_bound(&b2);
                let b1_mut = self.get_bound_mut(op.arg(0));
                let _ = b1_mut.intersect(&b);
                let b = r.sub_bound(&b1).neg_bound();
                let b2_mut = self.get_bound_mut(op.arg(1));
                let _ = b2_mut.intersect(&b);
            }
            OpCode::IntMul | OpCode::IntMulOvf => {
                let b1 = self.get_bound(op.arg(0), ctx);
                let b2 = self.get_bound(op.arg(1), ctx);
                if op.opcode != OpCode::IntMulOvf && !b1.mul_bound_cannot_overflow(&b2) {
                    return;
                }
                let r = self.get_bound(op.pos, ctx);
                let b = r.py_div_bound(&b2);
                let _ = self.get_bound_mut(op.arg(0)).intersect(&b);
                let b = r.py_div_bound(&b1);
                let _ = self.get_bound_mut(op.arg(1)).intersect(&b);
            }
            OpCode::IntLshift => {
                let b1 = self.get_bound(op.arg(0), ctx);
                let b2 = self.get_bound(op.arg(1), ctx);
                if !b1.lshift_bound_cannot_overflow(&b2) {
                    return;
                }
                let r = self.get_bound(op.pos, ctx);
                if let Ok(b) = r.lshift_bound_backwards(&b2) {
                    let _ = self.get_bound_mut(op.arg(0)).intersect(&b);
                }
            }
            OpCode::IntRshift => {
                let b2 = self.get_bound(op.arg(1), ctx);
                if !b2.is_constant() {
                    return;
                }
                let r = self.get_bound(op.pos, ctx);
                let b = r.rshift_bound_backwards(&b2);
                let _ = self.get_bound_mut(op.arg(0)).intersect(&b);
            }
            OpCode::UintRshift => {
                let b2 = self.get_bound(op.arg(1), ctx);
                if !b2.is_constant() {
                    return;
                }
                let r = self.get_bound(op.pos, ctx);
                let b = r.urshift_bound_backwards(&b2);
                let _ = self.get_bound_mut(op.arg(0)).intersect(&b);
            }
            OpCode::IntAnd => {
                let r = self.get_bound(op.pos, ctx);
                let b0 = self.get_bound(op.arg(0), ctx);
                let b1 = self.get_bound(op.arg(1), ctx);
                if let Ok(b) = b0.and_bound_backwards(&r) {
                    let _ = self.get_bound_mut(op.arg(1)).intersect(&b);
                }
                if let Ok(b) = b1.and_bound_backwards(&r) {
                    let _ = self.get_bound_mut(op.arg(0)).intersect(&b);
                }
            }
            OpCode::IntOr => {
                let r = self.get_bound(op.pos, ctx);
                let b0 = self.get_bound(op.arg(0), ctx);
                let b1 = self.get_bound(op.arg(1), ctx);
                if let Ok(b) = b0.or_bound_backwards(&r) {
                    let _ = self.get_bound_mut(op.arg(1)).intersect(&b);
                }
                if let Ok(b) = b1.or_bound_backwards(&r) {
                    let _ = self.get_bound_mut(op.arg(0)).intersect(&b);
                }
            }
            OpCode::IntXor => {
                let r = self.get_bound(op.pos, ctx);
                let b0 = self.get_bound(op.arg(0), ctx);
                let b1 = self.get_bound(op.arg(1), ctx);
                // xor is its own inverse
                let b = b0.xor_bound(&r);
                let _ = self.get_bound_mut(op.arg(1)).intersect(&b);
                let b = b1.xor_bound(&r);
                let _ = self.get_bound_mut(op.arg(0)).intersect(&b);
            }
            OpCode::IntInvert => {
                let bres = self.get_bound(op.pos, ctx);
                let bounds = bres.invert_bound();
                let _ = self.get_bound_mut(op.arg(0)).intersect(&bounds);
            }
            OpCode::IntNeg => {
                let bres = self.get_bound(op.pos, ctx);
                let bounds = bres.neg_bound();
                let _ = self.get_bound_mut(op.arg(0)).intersect(&bounds);
            }
            _ => {}
        }
    }

    /// Record what we last emitted (for overflow guard removal).
    fn record_emitted(&mut self, op: &Op) {
        self.last_emitted_opcode = Some(op.opcode);
        self.last_emitted_args = op.args.to_vec();
        self.last_emitted_ref = op.pos;
    }
}

impl Default for OptIntBounds {
    fn default() -> Self {
        Self::new()
    }
}

impl OptimizationPass for OptIntBounds {
    fn propagate_forward(&mut self, op: &Op, ctx: &mut OptContext) -> PassResult {
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
                PassResult::PassOn
            }
            OpCode::IntSub => {
                self.postprocess_int_sub(op, ctx);
                PassResult::PassOn
            }
            OpCode::IntMul => {
                self.postprocess_int_mul(op, ctx);
                PassResult::PassOn
            }
            OpCode::IntAnd => {
                self.postprocess_int_and(op, ctx);
                PassResult::PassOn
            }
            OpCode::IntOr => {
                self.postprocess_int_or(op, ctx);
                PassResult::PassOn
            }
            OpCode::IntXor => {
                self.postprocess_int_xor(op, ctx);
                PassResult::PassOn
            }
            OpCode::IntLshift => {
                self.postprocess_int_lshift(op, ctx);
                PassResult::PassOn
            }
            OpCode::IntRshift => {
                self.postprocess_int_rshift(op, ctx);
                PassResult::PassOn
            }
            OpCode::UintRshift => {
                self.postprocess_uint_rshift(op, ctx);
                PassResult::PassOn
            }
            OpCode::IntNeg => {
                self.postprocess_int_neg(op, ctx);
                PassResult::PassOn
            }
            OpCode::IntInvert => {
                self.postprocess_int_invert(op, ctx);
                PassResult::PassOn
            }
            OpCode::IntForceGeZero => {
                self.postprocess_int_force_ge_zero(op, ctx);
                PassResult::PassOn
            }
            OpCode::IntIsZero | OpCode::IntIsTrue => {
                self.postprocess_bool_result(op);
                PassResult::PassOn
            }

            // ── Lengths (always non-negative) ──
            OpCode::ArraylenGc => {
                self.postprocess_arraylen_gc(op);
                PassResult::PassOn
            }
            OpCode::Strlen => {
                self.postprocess_strlen(op);
                PassResult::PassOn
            }
            OpCode::Unicodelen => {
                self.postprocess_unicodelen(op);
                PassResult::PassOn
            }

            _ => PassResult::PassOn,
        };

        // Track last emitted for overflow guard handling
        match &result {
            PassResult::Emit(emitted_op) => self.record_emitted(emitted_op),
            PassResult::PassOn => self.record_emitted(op),
            _ => {
                // Remove: don't update last_emitted
            }
        }

        result
    }

    fn setup(&mut self) {
        self.bounds.clear();
        self.last_emitted_opcode = None;
        self.last_emitted_args.clear();
        self.last_emitted_ref = OpRef::NONE;
    }

    fn name(&self) -> &'static str {
        "intbounds"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizer::Optimizer;

    fn run_pass(ops: &[Op]) -> Vec<Op> {
        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptIntBounds::new()));
        opt.optimize(ops)
    }

    /// Create an OptIntBounds with specific bounds pre-set and run it on ops.
    fn run_pass_with_bounds(ops: &[Op], initial_bounds: &[(OpRef, IntBound)]) -> (Vec<Op>, OptIntBounds, OptContext) {
        let mut pass = OptIntBounds::new();
        let mut ctx = OptContext::new(ops.len());

        pass.setup();
        for (opref, bound) in initial_bounds {
            pass.set_bound(*opref, bound.clone());
        }

        for op in ops.iter() {
            let mut resolved_op = op.clone();
            // Keep the op.pos as set by the test (not overriding with index)
            for arg in &mut resolved_op.args {
                *arg = ctx.get_replacement(*arg);
            }
            match pass.propagate_forward(&resolved_op, &mut ctx) {
                PassResult::Emit(emit_op) => { ctx.emit(emit_op); }
                PassResult::Replace(rep_op) => { ctx.emit(rep_op); }
                PassResult::Remove => {}
                PassResult::PassOn => { ctx.emit(resolved_op); }
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
        let ops = vec![
            make_op(OpCode::IntAdd, &[OpRef(0), OpRef(1)], 2),
        ];

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
        assert!(result.is_empty() || result.iter().all(|op|
            op.opcode != OpCode::GuardTrue && op.opcode != OpCode::IntLt
        ), "Guard and comparison should both be removed, got: {:?}", result.iter().map(|o| o.opcode).collect::<Vec<_>>());
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
        assert!(result.is_empty() || result.iter().all(|op|
            op.opcode != OpCode::GuardFalse && op.opcode != OpCode::IntGe
        ), "Guard and comparison should both be removed, got: {:?}", result.iter().map(|o| o.opcode).collect::<Vec<_>>());
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
        assert!(b0.upper <= 9, "After GUARD_TRUE(INT_LT(i0, 10)), i0.upper should be <= 9, got {}", b0.upper);
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
        assert!(b0.lower >= 5, "After GUARD_TRUE(INT_GE(i0, 5)), i0.lower should be >= 5, got {}", b0.lower);
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
        assert!(result.iter().any(|op| op.opcode == OpCode::IntAdd),
            "INT_ADD_OVF should be transformed to INT_ADD");
        assert!(!result.iter().any(|op| op.opcode == OpCode::IntAddOvf),
            "INT_ADD_OVF should not remain");
        assert!(!result.iter().any(|op| op.opcode == OpCode::GuardNoOverflow),
            "GUARD_NO_OVERFLOW should be removed");
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
        assert!(result.iter().any(|op| op.opcode == OpCode::IntAddOvf),
            "INT_ADD_OVF should remain when overflow is possible");
        assert!(result.iter().any(|op| op.opcode == OpCode::GuardNoOverflow),
            "GUARD_NO_OVERFLOW should remain when overflow is possible");
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
        assert!(result.iter().any(|op| op.opcode == OpCode::IntSub),
            "INT_SUB_OVF should be transformed to INT_SUB");
        assert!(!result.iter().any(|op| op.opcode == OpCode::GuardNoOverflow),
            "GUARD_NO_OVERFLOW should be removed");
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
        assert!(result.iter().any(|op| op.opcode == OpCode::IntMul),
            "INT_MUL_OVF should be transformed to INT_MUL");
    }

    // ── Test: Comparison result when bounds determine outcome ──

    #[test]
    fn test_int_lt_known_true() {
        let initial_bounds = vec![
            (OpRef(0), IntBound::bounded(0, 5)),
            (OpRef(1), IntBound::bounded(10, 20)),
        ];
        let ops = vec![
            make_op(OpCode::IntLt, &[OpRef(0), OpRef(1)], 2),
        ];

        let (result, pass, ctx) = run_pass_with_bounds(&ops, &initial_bounds);
        // Should be removed (replaced by constant 1)
        assert!(result.is_empty(), "INT_LT should be removed when known true");
        // The constant should be set
        let b = pass.get_bound(OpRef(2), &ctx);
        assert!(b.is_constant() && b.get_constant() == 1);
    }

    #[test]
    fn test_int_lt_known_false() {
        let initial_bounds = vec![
            (OpRef(0), IntBound::bounded(10, 20)),
            (OpRef(1), IntBound::bounded(0, 5)),
        ];
        let ops = vec![
            make_op(OpCode::IntLt, &[OpRef(0), OpRef(1)], 2),
        ];

        let (result, pass, ctx) = run_pass_with_bounds(&ops, &initial_bounds);
        assert!(result.is_empty(), "INT_LT should be removed when known false");
        let b = pass.get_bound(OpRef(2), &ctx);
        assert!(b.is_constant() && b.get_constant() == 0);
    }

    #[test]
    fn test_int_eq_same_arg() {
        let ops = vec![
            make_op(OpCode::IntEq, &[OpRef(0), OpRef(0)], 1),
        ];

        let (result, pass, ctx) = run_pass_with_bounds(&ops, &[]);
        assert!(result.is_empty(), "INT_EQ(x, x) should be removed (always 1)");
        let b = pass.get_bound(OpRef(1), &ctx);
        assert!(b.is_constant() && b.get_constant() == 1);
    }

    #[test]
    fn test_int_ne_same_arg() {
        let ops = vec![
            make_op(OpCode::IntNe, &[OpRef(0), OpRef(0)], 1),
        ];

        let (result, pass, ctx) = run_pass_with_bounds(&ops, &[]);
        assert!(result.is_empty(), "INT_NE(x, x) should be removed (always 0)");
        let b = pass.get_bound(OpRef(1), &ctx);
        assert!(b.is_constant() && b.get_constant() == 0);
    }

    #[test]
    fn test_int_le_known_true() {
        let initial_bounds = vec![
            (OpRef(0), IntBound::bounded(0, 5)),
            (OpRef(1), IntBound::bounded(5, 20)),
        ];
        let ops = vec![
            make_op(OpCode::IntLe, &[OpRef(0), OpRef(1)], 2),
        ];

        let (result, _pass, _ctx) = run_pass_with_bounds(&ops, &initial_bounds);
        assert!(result.is_empty(), "INT_LE should be removed when known true");
    }

    #[test]
    fn test_int_ge_known_true() {
        let initial_bounds = vec![
            (OpRef(0), IntBound::bounded(10, 20)),
            (OpRef(1), IntBound::bounded(0, 10)),
        ];
        let ops = vec![
            make_op(OpCode::IntGe, &[OpRef(0), OpRef(1)], 2),
        ];

        let (result, _pass, _ctx) = run_pass_with_bounds(&ops, &initial_bounds);
        assert!(result.is_empty(), "INT_GE should be removed when known true");
    }

    // ── Test: Arithmetic bounds propagation ──

    #[test]
    fn test_int_sub_bounds() {
        let initial_bounds = vec![
            (OpRef(0), IntBound::bounded(10, 20)),
            (OpRef(1), IntBound::bounded(0, 5)),
        ];
        let ops = vec![
            make_op(OpCode::IntSub, &[OpRef(0), OpRef(1)], 2),
        ];

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
        let ops = vec![
            make_op(OpCode::IntMul, &[OpRef(0), OpRef(1)], 2),
        ];

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
        let ops = vec![
            make_op(OpCode::IntAnd, &[OpRef(0), OpRef(1)], 2),
        ];

        let (result, pass, _ctx) = run_pass_with_bounds(&ops, &initial_bounds);
        assert_eq!(result.len(), 1);
        let b = pass.bounds[2].as_ref().unwrap();
        // AND of [0, 255] and [0, 15] -> [0, 15]
        assert!(b.lower >= 0);
        assert!(b.upper <= 15);
    }

    #[test]
    fn test_int_force_ge_zero() {
        let initial_bounds = vec![
            (OpRef(0), IntBound::bounded(-10, 20)),
        ];
        let ops = vec![
            make_op(OpCode::IntForceGeZero, &[OpRef(0)], 1),
        ];

        let (_result, pass, _ctx) = run_pass_with_bounds(&ops, &initial_bounds);
        let b = pass.bounds[1].as_ref().unwrap();
        assert!(b.lower >= 0, "INT_FORCE_GE_ZERO result should be >= 0, got {}", b.lower);
        assert!(b.upper <= 20, "INT_FORCE_GE_ZERO result upper should be <= 20, got {}", b.upper);
    }

    #[test]
    fn test_arraylen_nonneg() {
        let ops = vec![
            make_op(OpCode::ArraylenGc, &[OpRef(0)], 1),
        ];

        let (_result, pass, _ctx) = run_pass_with_bounds(&ops, &[]);
        let b = pass.bounds[1].as_ref().unwrap();
        assert!(b.lower >= 0, "ARRAYLEN_GC result should be non-negative");
    }

    #[test]
    fn test_strlen_nonneg() {
        let ops = vec![
            make_op(OpCode::Strlen, &[OpRef(0)], 1),
        ];

        let (_result, pass, _ctx) = run_pass_with_bounds(&ops, &[]);
        let b = pass.bounds[1].as_ref().unwrap();
        assert!(b.lower >= 0, "STRLEN result should be non-negative");
    }

    #[test]
    fn test_int_neg_bounds() {
        let initial_bounds = vec![
            (OpRef(0), IntBound::bounded(3, 10)),
        ];
        let ops = vec![
            make_op(OpCode::IntNeg, &[OpRef(0)], 1),
        ];

        let (_result, pass, _ctx) = run_pass_with_bounds(&ops, &initial_bounds);
        let b = pass.bounds[1].as_ref().unwrap();
        // neg([3, 10]) = [-10, -3]
        assert_eq!(b.lower, -10);
        assert_eq!(b.upper, -3);
    }

    #[test]
    fn test_int_invert_bounds() {
        let initial_bounds = vec![
            (OpRef(0), IntBound::bounded(3, 10)),
        ];
        let ops = vec![
            make_op(OpCode::IntInvert, &[OpRef(0)], 1),
        ];

        let (_result, pass, _ctx) = run_pass_with_bounds(&ops, &initial_bounds);
        let b = pass.bounds[1].as_ref().unwrap();
        // invert([3, 10]) = [!10, !3] = [-11, -4]
        assert_eq!(b.lower, -11);
        assert_eq!(b.upper, -4);
    }

    #[test]
    fn test_sub_ovf_same_arg() {
        // INT_SUB_OVF(x, x) should be replaced by constant 0
        let initial_bounds = vec![
            (OpRef(0), IntBound::unbounded()),
        ];
        let ops = vec![
            make_op(OpCode::IntSubOvf, &[OpRef(0), OpRef(0)], 1),
        ];

        let (result, pass, ctx) = run_pass_with_bounds(&ops, &initial_bounds);
        assert!(result.is_empty(), "INT_SUB_OVF(x, x) should be removed");
        let b = pass.get_bound(OpRef(1), &ctx);
        assert!(b.is_constant() && b.get_constant() == 0);
    }

    // ── Test: Unsigned comparison optimization ──

    #[test]
    fn test_uint_lt_known_true() {
        let initial_bounds = vec![
            (OpRef(0), IntBound::bounded(0, 5)),
            (OpRef(1), IntBound::bounded(10, 20)),
        ];
        let ops = vec![
            make_op(OpCode::UintLt, &[OpRef(0), OpRef(1)], 2),
        ];

        let (result, _pass, _ctx) = run_pass_with_bounds(&ops, &initial_bounds);
        assert!(result.is_empty(), "UINT_LT should be removed when known true");
    }

    // ── Test: Lshift bounds ──

    #[test]
    fn test_int_lshift_bounds() {
        let initial_bounds = vec![
            (OpRef(0), IntBound::bounded(1, 4)),
            (OpRef(1), IntBound::from_constant(2)),
        ];
        let ops = vec![
            make_op(OpCode::IntLshift, &[OpRef(0), OpRef(1)], 2),
        ];

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
        let ops = vec![
            make_op(OpCode::IntRshift, &[OpRef(0), OpRef(1)], 2),
        ];

        let (_result, pass, _ctx) = run_pass_with_bounds(&ops, &initial_bounds);
        let b = pass.bounds[2].as_ref().unwrap();
        // [8, 20] >> 2 = [2, 5]
        assert_eq!(b.lower, 2);
        assert_eq!(b.upper, 5);
    }

    // ── Test: INT_IS_TRUE and INT_IS_ZERO produce bool bounds ──

    #[test]
    fn test_int_is_true_bool_bounds() {
        let ops = vec![
            make_op(OpCode::IntIsTrue, &[OpRef(0)], 1),
        ];

        let (_result, pass, _ctx) = run_pass_with_bounds(&ops, &[]);
        let b = pass.bounds[1].as_ref().unwrap();
        assert_eq!(b.lower, 0);
        assert_eq!(b.upper, 1);
    }

    #[test]
    fn test_int_is_zero_bool_bounds() {
        let ops = vec![
            make_op(OpCode::IntIsZero, &[OpRef(0)], 1),
        ];

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
        let ops = vec![
            make_op(OpCode::IntLt, &[OpRef(0), OpRef(1)], 2),
        ];

        let (result, pass, _ctx) = run_pass_with_bounds(&ops, &initial_bounds);
        assert_eq!(result.len(), 1, "INT_LT should remain when bounds are unknown");
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
        let ops = vec![
            make_op(OpCode::IntSignext, &[OpRef(0), OpRef(1)], 2),
        ];

        let (result, _pass, _ctx) = run_pass_with_bounds(&ops, &initial_bounds);
        assert!(result.is_empty(), "signext should be eliminated when value fits");
    }

    #[test]
    fn test_int_signext_kept() {
        // x in [-50000, 50000], signext to 1 byte (-128..127) -> can't eliminate
        let initial_bounds = vec![
            (OpRef(0), IntBound::bounded(-50000, 50000)),
            (OpRef(1), IntBound::from_constant(1)), // byte_size = 1
        ];
        let ops = vec![
            make_op(OpCode::IntSignext, &[OpRef(0), OpRef(1)], 2),
        ];

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
        let initial_bounds = vec![
            (OpRef(0), IntBound::bounded(0, 100)),
        ];
        let ops = vec![
            make_op(OpCode::IntIsTrue, &[OpRef(0)], 1),
            make_op(OpCode::GuardTrue, &[OpRef(1)], 2),
        ];

        let (_result, pass, _ctx) = run_pass_with_bounds(&ops, &initial_bounds);
        let b0 = pass.bounds[0].as_ref().unwrap();
        assert!(b0.lower >= 1, "After GUARD_TRUE(INT_IS_TRUE(i0)), i0.lower should be >= 1, got {}", b0.lower);
    }

    #[test]
    fn test_guard_false_on_int_is_true() {
        // i0 in [0, 100]
        // i1 = INT_IS_TRUE(i0)
        // GUARD_FALSE(i1)
        // After guard, i0 should be 0
        let initial_bounds = vec![
            (OpRef(0), IntBound::bounded(0, 100)),
        ];
        let ops = vec![
            make_op(OpCode::IntIsTrue, &[OpRef(0)], 1),
            make_op(OpCode::GuardFalse, &[OpRef(1)], 2),
        ];

        let (_result, pass, _ctx) = run_pass_with_bounds(&ops, &initial_bounds);
        let b0 = pass.bounds[0].as_ref().unwrap();
        assert!(b0.is_constant() && b0.get_constant() == 0,
            "After GUARD_FALSE(INT_IS_TRUE(i0)), i0 should be 0, got [{}, {}]", b0.lower, b0.upper);
    }

    // ── Test: x + x bounds ──

    #[test]
    fn test_int_add_x_plus_x() {
        // x in [3, 5], x + x -> should be [6, 10]
        let initial_bounds = vec![
            (OpRef(0), IntBound::bounded(3, 5)),
        ];
        let ops = vec![
            make_op(OpCode::IntAdd, &[OpRef(0), OpRef(0)], 1),
        ];

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
        let initial_bounds = vec![
            (OpRef(0), IntBound::unbounded()),
        ];
        let ops = vec![
            make_op(OpCode::IntNeg, &[OpRef(0)], 1),
        ];

        let (_result, mut pass, ctx) = run_pass_with_bounds(&ops, &initial_bounds);
        // Manually tighten the result and trigger backward prop
        pass.set_bound(OpRef(1), IntBound::bounded(-5, -1));
        pass.propagate_bounds_backward(OpRef(1), &ctx);
        let b0 = pass.bounds[0].as_ref().unwrap();
        assert!(b0.lower >= 1, "backward neg: lower should be >= 1, got {}", b0.lower);
        assert!(b0.upper <= 5, "backward neg: upper should be <= 5, got {}", b0.upper);
    }
}
