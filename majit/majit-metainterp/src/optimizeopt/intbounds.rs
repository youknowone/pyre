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

/// Integer bounds optimization pass.
///
/// Keeps track of the bounds placed on integers by guards and removes
/// redundant guards.
///
/// RPython parity: `OptIntBounds(Optimization)` has NO own bounds storage —
/// all bound state lives on `box._forwarded` and is accessed via the base
/// class `Optimization.getintbound`/`setintbound` (optimizer.py:99-125).
/// In majit the equivalent is `OptContext::forwarded[Forwarded::IntBound]`
/// accessed via `ctx.getintbound`/`ctx.setintbound`/`ctx.with_intbound_mut`.
pub struct OptIntBounds {
    /// intbounds.py: last_emitted_operation — opcode (for overflow guard handling).
    last_emitted_opcode: Option<OpCode>,
    /// intbounds.py: last_emitted_operation — args (for overflow guard handling).
    last_emitted_args: Vec<OpRef>,
    /// intbounds.py: last_emitted_operation — OpRef result.
    last_emitted_ref: OpRef,
    /// intbounds.py: pure_from_args synthesis cache.
    /// Records equivalent pure operations discovered through bounds analysis
    /// (e.g., INT_OR with non-overlapping ranges = INT_ADD).
    /// Key: (opcode, arg0, arg1), Value: result OpRef.
    pure_from_args_cache: Vec<(OpCode, OpRef, OpRef, OpRef)>,
}

impl OptIntBounds {
    pub fn new() -> Self {
        OptIntBounds {
            pure_from_args_cache: Vec::new(),
            last_emitted_opcode: None,
            last_emitted_args: Vec::new(),
            last_emitted_ref: OpRef::NONE,
        }
    }

    /// optimizer.py:99-113 getintbound — thin wrapper over `ctx.getintbound`.
    /// In RPython this lives on the base `Optimization` class; majit's
    /// `OptContext::getintbound` is the equivalent. Kept here as a method on
    /// `OptIntBounds` for call-site brevity.
    fn getintbound(&self, opref: OpRef, ctx: &mut OptContext) -> IntBound {
        ctx.getintbound(opref)
    }

    /// optimizer.py:115-125 setintbound — thin wrapper over `ctx.setintbound`.
    fn setintbound(&mut self, opref: OpRef, bound: IntBound, ctx: &mut OptContext) {
        ctx.setintbound(opref, &bound);
    }

    /// Intersect a bound into the stored bound for opref. RPython:
    /// `self.getintbound(op).intersect(bound)` (mutates the IntBound stored
    /// on `op._forwarded` in place).
    fn intersect_bound(&mut self, opref: OpRef, bound: &IntBound, ctx: &mut OptContext) {
        ctx.setintbound(opref, bound);
    }

    /// optimizer.py:434: make_constant_int(box, intvalue) — RPython just
    /// forwards to `make_constant(box, ConstInt(intvalue))`. The bounds-range
    /// safety check + `make_eq_const` shrink (optimizer.py:415-426) live in
    /// `OptContext::make_constant`.
    fn make_constant_int(&mut self, op: &Op, value: i64, ctx: &mut OptContext) {
        ctx.make_constant(op.pos, Value::Int(value));
    }

    /// OpRef-keyed variant of make_constant_int. Used by
    /// `propagate_bounds_backward` and the IntIsTrue/IsZero arms which
    /// receive an OpRef rather than an &Op.
    fn make_constant_int_ref(&mut self, opref: OpRef, value: i64, ctx: &mut OptContext) {
        ctx.make_constant(opref, Value::Int(value));
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
            OptimizationResult::PassOn
        }
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
        self.intersect_bound(op.pos, &b, ctx);
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
        self.intersect_bound(op.pos, &b, ctx);
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

    fn postprocess_int_mul(&mut self, op: &Op, ctx: &mut OptContext) {
        let b0 = self.getintbound(op.arg(0), ctx);
        let b1 = self.getintbound(op.arg(1), ctx);
        let b = b0.mul_bound(&b1);
        self.intersect_bound(op.pos, &b, ctx);
    }

    fn postprocess_int_and(&mut self, op: &Op, ctx: &mut OptContext) {
        let b0 = self.getintbound(op.arg(0), ctx);
        let b1 = self.getintbound(op.arg(1), ctx);
        let b = b0.and_bound(&b1);
        self.intersect_bound(op.pos, &b, ctx);
    }

    /// intbounds.py:60-71 postprocess_INT_OR
    fn postprocess_int_or(&mut self, op: &Op, ctx: &mut OptContext) {
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
        self.intersect_bound(op.pos, &b, ctx);
    }

    /// intbounds.py:73-84 postprocess_INT_XOR
    fn postprocess_int_xor(&mut self, op: &Op, ctx: &mut OptContext) {
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
        self.intersect_bound(op.pos, &b, ctx);
    }

    /// intbounds.py: INT_LSHIFT pure_from_args synthesis.
    /// If res = INT_LSHIFT(a, b), then a = INT_RSHIFT(res, b).
    fn postprocess_int_lshift(&mut self, op: &Op, ctx: &mut OptContext) {
        let arg0 = ctx.get_box_replacement(op.arg(0));
        let arg1 = ctx.get_box_replacement(op.arg(1));
        let b0 = self.getintbound(arg0, ctx);
        let b1 = self.getintbound(arg1, ctx);
        let b = b0.lshift_bound(&b1);
        self.intersect_bound(op.pos, &b, ctx);
        // intbounds.py:185: only synthesize reverse if lshift cannot overflow
        if b0.lshift_bound_cannot_overflow(&b1) {
            self.record_pure_from_args(OpCode::IntRshift, op.pos, arg1, arg0);
        }
    }

    fn postprocess_int_rshift(&mut self, op: &Op, ctx: &mut OptContext) {
        let b0 = self.getintbound(op.arg(0), ctx);
        let b1 = self.getintbound(op.arg(1), ctx);
        let b = b0.rshift_bound(&b1);
        self.intersect_bound(op.pos, &b, ctx);
    }

    fn postprocess_uint_rshift(&mut self, op: &Op, ctx: &mut OptContext) {
        let b0 = self.getintbound(op.arg(0), ctx);
        let b1 = self.getintbound(op.arg(1), ctx);
        let b = b0.urshift_bound(&b1);
        self.intersect_bound(op.pos, &b, ctx);
    }

    fn postprocess_int_floordiv(&mut self, op: &Op, ctx: &mut OptContext) {
        let b0 = self.getintbound(op.arg(0), ctx);
        let b1 = self.getintbound(op.arg(1), ctx);
        let b = b0.floordiv_bound(&b1);
        self.intersect_bound(op.pos, &b, ctx);
    }

    fn postprocess_int_mod(&mut self, op: &Op, ctx: &mut OptContext) {
        let b0 = self.getintbound(op.arg(0), ctx);
        let b1 = self.getintbound(op.arg(1), ctx);
        let b = b0.mod_bound(&b1);
        self.intersect_bound(op.pos, &b, ctx);
    }

    fn postprocess_int_neg(&mut self, op: &Op, ctx: &mut OptContext) {
        let b = self.getintbound(op.arg(0), ctx);
        let result = b.neg_bound();
        self.intersect_bound(op.pos, &result, ctx);
    }

    fn postprocess_int_invert(&mut self, op: &Op, ctx: &mut OptContext) {
        let b = self.getintbound(op.arg(0), ctx);
        let result = b.invert_bound();
        self.intersect_bound(op.pos, &result, ctx);
    }

    fn postprocess_int_force_ge_zero(&mut self, op: &Op, ctx: &mut OptContext) {
        let b_arg = self.getintbound(op.arg(0), ctx);
        let mut result = IntBound::nonnegative();
        if b_arg.upper >= 0 {
            let _ = result.make_le(&b_arg);
        }
        self.intersect_bound(op.pos, &result, ctx);
    }

    fn postprocess_arraylen_gc(&mut self, op: &Op, ctx: &mut OptContext) {
        // intbounds.py:503-505
        //     array = self.ensure_ptr_info_arg0(op)
        //     self.optimizer.setintbound(op, array.getlenbound(None))
        let bound = {
            let mut array = ctx.ensure_ptr_info_arg0(op);
            array.getlenbound(None)
        };
        if let Some(bound) = bound {
            ctx.setintbound(op.pos, &bound);
        }
    }

    fn postprocess_strlen(&mut self, op: &Op, ctx: &mut OptContext) {
        // intbounds.py:507-510
        //     self.make_nonnull_str(op.getarg(0), vstring.mode_string)
        //     array = getptrinfo(op.getarg(0))
        //     self.optimizer.setintbound(op, array.getlenbound(vstring.mode_string))
        //
        // Rust note: PyPy splits this into make_nonnull_str + getptrinfo
        // because make_nonnull_str installs the StrPtrInfo on box._forwarded
        // and getptrinfo reads it back. The Rust port reaches the same state
        // by calling ensure_ptr_info_arg0 (which constructs StrPtrInfo for
        // STRLEN per optimizer.py:490-491) and then invoking getlenbound on
        // the returned handle.
        ctx.make_nonnull_str(op.arg(0), 0);
        let bound = {
            let mut info = ctx.ensure_ptr_info_arg0(op);
            info.getlenbound(Some(0))
        };
        if let Some(bound) = bound {
            ctx.setintbound(op.pos, &bound);
        }
    }

    fn postprocess_unicodelen(&mut self, op: &Op, ctx: &mut OptContext) {
        // intbounds.py:512-515
        //     self.make_nonnull_str(op.getarg(0), vstring.mode_unicode)
        //     array = getptrinfo(op.getarg(0))
        //     self.optimizer.setintbound(op, array.getlenbound(vstring.mode_unicode))
        ctx.make_nonnull_str(op.arg(0), 1);
        let bound = {
            let mut info = ctx.ensure_ptr_info_arg0(op);
            info.getlenbound(Some(1))
        };
        if let Some(bound) = bound {
            ctx.setintbound(op.pos, &bound);
        }
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
        OptimizationResult::PassOn
    }

    fn postprocess_int_signext(&mut self, op: &Op, ctx: &mut OptContext) {
        let b1 = self.getintbound(op.arg(1), ctx);
        if b1.is_constant() {
            let byte_size = b1.get_constant();
            let numbits = byte_size * 8;
            let start = -(1i64 << (numbits - 1));
            let stop = 1i64 << (numbits - 1);
            let _ = ctx.with_intbound_mut(op.pos, |bm| bm.intersect_const(start, stop - 1));
        }
    }

    // ── Overflow operations ──

    /// intbounds.py:244-256 optimize_INT_ADD_OVF
    fn optimize_int_add_ovf(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let b0 = self.getintbound(op.arg(0), ctx);
        let b1 = self.getintbound(op.arg(1), ctx);
        if b0.add_bound_cannot_overflow(&b1) {
            // replace_op_with(op, INT_ADD) + send_extra_operation
            let mut new_op = Op::new(OpCode::IntAdd, &op.args);
            new_op.descr = op.descr.clone();
            new_op.pos = op.pos;
            OptimizationResult::Emit(new_op)
        } else {
            OptimizationResult::PassOn
        }
    }

    fn postprocess_int_add_ovf(&mut self, op: &Op, ctx: &mut OptContext) {
        let arg0 = ctx.get_box_replacement(op.arg(0));
        let arg1 = ctx.get_box_replacement(op.arg(1));
        let b0 = self.getintbound(arg0, ctx);
        let b = if arg0 == arg1 {
            b0.mul2_bound_no_overflow()
        } else {
            let b1 = self.getintbound(arg1, ctx);
            b0.add_bound_no_overflow(&b1)
        };
        self.intersect_bound(op.pos, &b, ctx);
    }

    /// intbounds.py:275-287 optimize_INT_SUB_OVF
    fn optimize_int_sub_ovf(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let arg0 = ctx.get_box_replacement(op.arg(0));
        let arg1 = ctx.get_box_replacement(op.arg(1));
        if arg0 == arg1 {
            // arg0.same_box(arg1) → x - x = 0
            self.make_constant_int(op, 0, ctx);
            return OptimizationResult::Remove;
        }
        let b0 = self.getintbound(arg0, ctx);
        let b1 = self.getintbound(arg1, ctx);
        if b0.sub_bound_cannot_overflow(&b1) {
            // replace_op_with(op, INT_SUB) + send_extra_operation
            let mut new_op = Op::new(OpCode::IntSub, &op.args);
            new_op.descr = op.descr.clone();
            new_op.pos = op.pos;
            OptimizationResult::Emit(new_op)
        } else {
            OptimizationResult::PassOn
        }
    }

    fn postprocess_int_sub_ovf(&mut self, op: &Op, ctx: &mut OptContext) {
        let b0 = self.getintbound(op.arg(0), ctx);
        let b1 = self.getintbound(op.arg(1), ctx);
        let b = b0.sub_bound_no_overflow(&b1);
        self.intersect_bound(op.pos, &b, ctx);
    }

    /// intbounds.py:298-305 optimize_INT_MUL_OVF
    fn optimize_int_mul_ovf(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let b0 = self.getintbound(op.arg(0), ctx);
        let b1 = self.getintbound(op.arg(1), ctx);
        if b0.mul_bound_cannot_overflow(&b1) {
            // replace_op_with(op, INT_MUL) + send_extra_operation
            let mut new_op = Op::new(OpCode::IntMul, &op.args);
            new_op.descr = op.descr.clone();
            new_op.pos = op.pos;
            OptimizationResult::Emit(new_op)
        } else {
            OptimizationResult::PassOn
        }
    }

    fn postprocess_int_mul_ovf(&mut self, op: &Op, ctx: &mut OptContext) {
        let arg0 = ctx.get_box_replacement(op.arg(0));
        let arg1 = ctx.get_box_replacement(op.arg(1));
        let b0 = self.getintbound(arg0, ctx);
        let b = if arg0 == arg1 {
            b0.square_bound_no_overflow()
        } else {
            let b1 = self.getintbound(arg1, ctx);
            b0.mul_bound_no_overflow(&b1)
        };
        self.intersect_bound(op.pos, &b, ctx);
    }

    /// intbounds.py:209-229 optimize_GUARD_NO_OVERFLOW
    fn optimize_guard_no_overflow(&mut self, op: &Op) -> OptimizationResult {
        let _ = op;
        // intbounds.py:210-219:
        //   lastop = self.last_emitted_operation
        //   if lastop is not None:
        //     opnum = lastop.getopnum()
        //     if opnum not in (INT_ADD_OVF, INT_SUB_OVF, INT_MUL_OVF):
        //       return   # guard killed
        if let Some(opcode) = self.last_emitted_opcode {
            if !matches!(
                opcode,
                OpCode::IntAddOvf | OpCode::IntSubOvf | OpCode::IntMulOvf
            ) {
                return OptimizationResult::Remove;
            }
            // intbounds.py:222-228: synthesize the non-overflowing inverse
            let result = self.last_emitted_ref;
            if self.last_emitted_args.len() >= 2 && !result.is_none() {
                let arg0 = self.last_emitted_args[0];
                let arg1 = self.last_emitted_args[1];
                match opcode {
                    OpCode::IntAddOvf => {
                        self.record_pure_from_args(OpCode::IntSub, result, arg1, arg0);
                        self.record_pure_from_args(OpCode::IntSub, result, arg0, arg1);
                    }
                    OpCode::IntSubOvf => {
                        self.record_pure_from_args(OpCode::IntAdd, result, arg1, arg0);
                        self.record_pure_from_args(OpCode::IntSub, arg0, result, arg1);
                    }
                    _ => {}
                }
            }
        }
        // intbounds.py:229: return self.emit(op)
        OptimizationResult::PassOn
    }

    /// intbounds.py:231-242 optimize_GUARD_OVERFLOW
    fn optimize_guard_overflow(&mut self, op: &Op) -> OptimizationResult {
        let _ = op;
        // intbounds.py:232-233: if lastop is None: return
        let Some(opcode) = self.last_emitted_opcode else {
            return OptimizationResult::Remove;
        };
        // intbounds.py:236-238: if opnum not in OVF_ops: raise InvalidLoop
        if !matches!(
            opcode,
            OpCode::IntAddOvf | OpCode::IntSubOvf | OpCode::IntMulOvf
        ) {
            return OptimizationResult::InvalidLoop;
        }
        // intbounds.py:240: return self.emit(op)
        OptimizationResult::PassOn
    }

    // ── Guard optimizations ──

    /// RPython intbounds.py: no optimize_GUARD_TRUE handler.
    /// Guards are NOT handled in optimize (propagate_forward).
    /// The guard is emitted via the default emit path, and
    /// propagate_bounds_backward runs in postprocess_GUARD_TRUE
    /// (intbounds.py:56).
    ///
    /// The constant/bound checks for guard removal are handled by
    /// the default dispatch in RPython. In majit, we do them here
    /// because we don't have per-opcode default dispatch.
    fn optimize_guard_true(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let cond_ref = ctx.get_box_replacement(op.arg(0));

        // Constant check: if condition is known constant nonzero, remove guard.
        if let Some(val) = ctx.get_constant_int(cond_ref) {
            if val != 0 {
                return OptimizationResult::Remove;
            }
        }

        if !matches!(ctx.opref_type(cond_ref), Some(majit_ir::Type::Int)) {
            return OptimizationResult::PassOn;
        }

        // Bound check: if bound proves always nonzero, remove guard.
        let b = self.getintbound(cond_ref, ctx);
        if b.known_gt_const(0) {
            return OptimizationResult::Remove;
        }

        OptimizationResult::PassOn
    }

    fn optimize_guard_false(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let cond_ref = ctx.get_box_replacement(op.arg(0));

        if let Some(val) = ctx.get_constant_int(cond_ref) {
            if val == 0 {
                return OptimizationResult::Remove;
            }
        }

        if !matches!(ctx.opref_type(cond_ref), Some(majit_ir::Type::Int)) {
            return OptimizationResult::PassOn;
        }

        let b = self.getintbound(cond_ref, ctx);
        if b.known_eq_const(0) {
            return OptimizationResult::Remove;
        }

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

    // ── Bound narrowing helpers ──

    /// intbounds.py:564-570 make_int_lt
    ///
    /// ```python
    /// def make_int_lt(self, box1, box2):
    ///     b1 = self.getintbound(box1)
    ///     b2 = self.getintbound(box2)
    ///     if b1.make_lt(b2):
    ///         self.propagate_bounds_backward(box1)
    ///     if b2.make_gt(b1):
    ///         self.propagate_bounds_backward(box2)
    /// ```
    ///
    /// `IntBound::make_lt` returns `Ok(true)` exactly when RPython's
    /// `make_lt` returns truthy ("the bound was actually tightened"), so
    /// `matches!(.., Ok(true))` is the line-by-line equivalent of the
    /// RPython `if`.
    fn make_int_lt(&mut self, box1: OpRef, box2: OpRef, ctx: &mut OptContext) {
        let b2 = self.getintbound(box2, ctx);
        let changed1 = ctx.with_intbound_mut(box1, |b1| matches!(b1.make_lt(&b2), Ok(true)));
        if changed1 {
            self.propagate_bounds_backward(box1, ctx);
        }
        let b1 = self.getintbound(box1, ctx);
        let changed2 = ctx.with_intbound_mut(box2, |b2| matches!(b2.make_gt(&b1), Ok(true)));
        if changed2 {
            self.propagate_bounds_backward(box2, ctx);
        }
    }

    /// intbounds.py:572-578 make_int_le
    fn make_int_le(&mut self, box1: OpRef, box2: OpRef, ctx: &mut OptContext) {
        let b2 = self.getintbound(box2, ctx);
        let changed1 = ctx.with_intbound_mut(box1, |b1| matches!(b1.make_le(&b2), Ok(true)));
        if changed1 {
            self.propagate_bounds_backward(box1, ctx);
        }
        let b1 = self.getintbound(box1, ctx);
        let changed2 = ctx.with_intbound_mut(box2, |b2| matches!(b2.make_ge(&b1), Ok(true)));
        if changed2 {
            self.propagate_bounds_backward(box2, ctx);
        }
    }

    /// intbounds.py:580-581 make_int_gt
    fn make_int_gt(&mut self, box1: OpRef, box2: OpRef, ctx: &mut OptContext) {
        self.make_int_lt(box2, box1, ctx);
    }

    /// intbounds.py:583-584 make_int_ge
    fn make_int_ge(&mut self, box1: OpRef, box2: OpRef, ctx: &mut OptContext) {
        self.make_int_le(box2, box1, ctx);
    }

    /// intbounds.py:586-592 make_unsigned_lt
    fn make_unsigned_lt(&mut self, box1: OpRef, box2: OpRef, ctx: &mut OptContext) {
        let b2 = self.getintbound(box2, ctx);
        let changed1 =
            ctx.with_intbound_mut(box1, |b1| matches!(b1.make_unsigned_lt(&b2), Ok(true)));
        if changed1 {
            self.propagate_bounds_backward(box1, ctx);
        }
        let b1 = self.getintbound(box1, ctx);
        let changed2 =
            ctx.with_intbound_mut(box2, |b2| matches!(b2.make_unsigned_gt(&b1), Ok(true)));
        if changed2 {
            self.propagate_bounds_backward(box2, ctx);
        }
    }

    /// intbounds.py:594-600 make_unsigned_le
    fn make_unsigned_le(&mut self, box1: OpRef, box2: OpRef, ctx: &mut OptContext) {
        let b2 = self.getintbound(box2, ctx);
        let changed1 =
            ctx.with_intbound_mut(box1, |b1| matches!(b1.make_unsigned_le(&b2), Ok(true)));
        if changed1 {
            self.propagate_bounds_backward(box1, ctx);
        }
        let b1 = self.getintbound(box1, ctx);
        let changed2 =
            ctx.with_intbound_mut(box2, |b2| matches!(b2.make_unsigned_ge(&b1), Ok(true)));
        if changed2 {
            self.propagate_bounds_backward(box2, ctx);
        }
    }

    fn make_unsigned_gt(&mut self, box1: OpRef, box2: OpRef, ctx: &mut OptContext) {
        self.make_unsigned_lt(box2, box1, ctx);
    }

    fn make_unsigned_ge(&mut self, box1: OpRef, box2: OpRef, ctx: &mut OptContext) {
        self.make_unsigned_le(box2, box1, ctx);
    }

    /// intbounds.py:658-664 make_eq
    ///
    /// ```python
    /// def make_eq(self, arg0, arg1):
    ///     b0 = self.getintbound(arg0)
    ///     b1 = self.getintbound(arg1)
    ///     if b0.intersect(b1):
    ///         self.propagate_bounds_backward(arg0)
    ///     if b1.intersect(b0):
    ///         self.propagate_bounds_backward(arg1)
    /// ```
    fn make_eq(&mut self, arg0: OpRef, arg1: OpRef, ctx: &mut OptContext) {
        let b1 = self.getintbound(arg1, ctx);
        let changed0 = ctx.with_intbound_mut(arg0, |b0| matches!(b0.intersect(&b1), Ok(true)));
        if changed0 {
            self.propagate_bounds_backward(arg0, ctx);
        }
        let b0 = self.getintbound(arg0, ctx);
        let changed1 = ctx.with_intbound_mut(arg1, |b1| matches!(b1.intersect(&b0), Ok(true)));
        if changed1 {
            self.propagate_bounds_backward(arg1, ctx);
        }
    }

    /// intbounds.py:666-676 make_ne
    ///
    /// ```python
    /// def make_ne(self, arg0, arg1):
    ///     b0 = self.getintbound(arg0)
    ///     b1 = self.getintbound(arg1)
    ///     if b1.is_constant():
    ///         v1 = b1.get_constant_int()
    ///         if b0.make_ne_const(v1):
    ///             self.propagate_bounds_backward(arg0)
    ///     elif b0.is_constant():
    ///         v0 = b0.get_constant_int()
    ///         if b1.make_ne_const(v0):
    ///             self.propagate_bounds_backward(arg1)
    /// ```
    fn make_ne(&mut self, arg0: OpRef, arg1: OpRef, ctx: &mut OptContext) {
        let b1 = self.getintbound(arg1, ctx);
        if b1.is_constant() {
            let v1 = b1.get_constant();
            if ctx.with_intbound_mut(arg0, |b0| b0.make_ne_const(v1)) {
                self.propagate_bounds_backward(arg0, ctx);
            }
        } else {
            let b0 = self.getintbound(arg0, ctx);
            if b0.is_constant() {
                let v0 = b0.get_constant();
                if ctx.with_intbound_mut(arg1, |b1| b1.make_ne_const(v0)) {
                    self.propagate_bounds_backward(arg1, ctx);
                }
            }
        }
    }

    // ── Backward propagation after constant discovery ──

    /// intbounds.py:40-50 propagate_bounds_backward
    ///
    /// ```python
    /// def propagate_bounds_backward(self, box):
    ///     b = self.getintbound(box)
    ///     if b.is_constant():
    ///         self.make_constant_int(box, b.get_constant_int())
    ///     box1 = self.optimizer.as_operation(box)
    ///     if box1 is not None:
    ///         dispatch_bounds_ops(self, box1)
    /// ```
    ///
    /// Both branches run unconditionally (no early return after the
    /// constant fold) — RPython falls through into the `dispatch_bounds_ops`
    /// call so the producing op of a now-constant value also gets a chance
    /// to tighten its other arguments.
    fn propagate_bounds_backward(&mut self, opref: OpRef, ctx: &mut OptContext) {
        let b = self.getintbound(opref, ctx);
        if b.is_constant() {
            self.make_constant_int_ref(opref, b.get_constant(), ctx);
        }
        if let Some(producing_op) = self.find_producing_op(opref, ctx) {
            let producing_op = producing_op.clone();
            self.propagate_bounds_backward_op(&producing_op, ctx);
        }
    }

    /// intbounds.py:678-693 _propagate_int_is_true_or_zero (helper for
    /// propagate_bounds_INT_IS_TRUE / IS_ZERO).
    fn propagate_int_is_true_or_zero(
        &mut self,
        op: &Op,
        valnonzero: i64,
        valzero: i64,
        ctx: &mut OptContext,
    ) {
        if ctx.is_raw_ptr(op.arg(0)) {
            return;
        }
        let r = self.getintbound(op.pos, ctx);
        if !r.is_constant() {
            return;
        }
        let arg0 = op.arg(0);
        let r_const = r.get_constant();
        if r_const == valnonzero {
            let b1 = self.getintbound(arg0, ctx);
            if b1.known_nonnegative() {
                let _ = ctx.with_intbound_mut(arg0, |bm| bm.make_gt_const(0));
                self.propagate_bounds_backward(arg0, ctx);
            } else if b1.known_le_const(0) {
                let _ = ctx.with_intbound_mut(arg0, |bm| bm.make_lt_const(0));
                self.propagate_bounds_backward(arg0, ctx);
            }
        } else if r_const == valzero {
            self.make_constant_int_ref(arg0, 0, ctx);
            self.propagate_bounds_backward(arg0, ctx);
        }
    }

    fn propagate_bounds_backward_op(&mut self, op: &Op, ctx: &mut OptContext) {
        match op.opcode {
            // intbounds.py:701-712 propagate_bounds_INT_ADD
            //
            // ```python
            // def propagate_bounds_INT_ADD(self, op):
            //     if self.is_raw_ptr(op.getarg(0)) or self.is_raw_ptr(op.getarg(1)):
            //         return
            //     b1 = self.getintbound(op.getarg(0))
            //     b2 = self.getintbound(op.getarg(1))
            //     r = self.getintbound(op)
            //     b = r.sub_bound(b2)
            //     if b1.intersect(b):
            //         self.propagate_bounds_backward(op.getarg(0))
            //     b = r.sub_bound(b1)
            //     if b2.intersect(b):
            //         self.propagate_bounds_backward(op.getarg(1))
            // ```
            OpCode::IntAdd | OpCode::IntAddOvf => {
                if ctx.is_raw_ptr(op.arg(0)) || ctx.is_raw_ptr(op.arg(1)) {
                    return;
                }
                let arg0 = op.arg(0);
                let arg1 = op.arg(1);
                let b1 = self.getintbound(arg0, ctx);
                let b2 = self.getintbound(arg1, ctx);
                let r = self.getintbound(op.pos, ctx);
                let b = r.sub_bound(&b2);
                let changed0 =
                    ctx.with_intbound_mut(arg0, |bm| matches!(bm.intersect(&b), Ok(true)));
                if changed0 {
                    self.propagate_bounds_backward(arg0, ctx);
                }
                let b = r.sub_bound(&b1);
                let changed1 =
                    ctx.with_intbound_mut(arg1, |bm| matches!(bm.intersect(&b), Ok(true)));
                if changed1 {
                    self.propagate_bounds_backward(arg1, ctx);
                }
            }
            // intbounds.py:714-723 propagate_bounds_INT_SUB
            OpCode::IntSub | OpCode::IntSubOvf => {
                let arg0 = op.arg(0);
                let arg1 = op.arg(1);
                let b1 = self.getintbound(arg0, ctx);
                let b2 = self.getintbound(arg1, ctx);
                let r = self.getintbound(op.pos, ctx);
                let b = r.add_bound(&b2);
                let changed0 =
                    ctx.with_intbound_mut(arg0, |bm| matches!(bm.intersect(&b), Ok(true)));
                if changed0 {
                    self.propagate_bounds_backward(arg0, ctx);
                }
                let b = r.sub_bound(&b1).neg_bound();
                let changed1 =
                    ctx.with_intbound_mut(arg1, |bm| matches!(bm.intersect(&b), Ok(true)));
                if changed1 {
                    self.propagate_bounds_backward(arg1, ctx);
                }
            }
            // intbounds.py:725-737 propagate_bounds_INT_MUL
            OpCode::IntMul | OpCode::IntMulOvf => {
                let arg0 = op.arg(0);
                let arg1 = op.arg(1);
                let b1 = self.getintbound(arg0, ctx);
                let b2 = self.getintbound(arg1, ctx);
                if op.opcode != OpCode::IntMulOvf && !b1.mul_bound_cannot_overflow(&b2) {
                    return;
                }
                let r = self.getintbound(op.pos, ctx);
                let b = r.py_div_bound(&b2);
                let changed0 =
                    ctx.with_intbound_mut(arg0, |bm| matches!(bm.intersect(&b), Ok(true)));
                if changed0 {
                    self.propagate_bounds_backward(arg0, ctx);
                }
                let b = r.py_div_bound(&b1);
                let changed1 =
                    ctx.with_intbound_mut(arg1, |bm| matches!(bm.intersect(&b), Ok(true)));
                if changed1 {
                    self.propagate_bounds_backward(arg1, ctx);
                }
            }
            // intbounds.py:739-747 propagate_bounds_INT_LSHIFT
            OpCode::IntLshift => {
                let arg0 = op.arg(0);
                let arg1 = op.arg(1);
                let b1 = self.getintbound(arg0, ctx);
                let b2 = self.getintbound(arg1, ctx);
                if !b1.lshift_bound_cannot_overflow(&b2) {
                    return;
                }
                let r = self.getintbound(op.pos, ctx);
                if let Ok(b) = r.lshift_bound_backwards(&b2) {
                    let changed =
                        ctx.with_intbound_mut(arg0, |bm| matches!(bm.intersect(&b), Ok(true)));
                    if changed {
                        self.propagate_bounds_backward(arg0, ctx);
                    }
                }
            }
            // intbounds.py:759-767 propagate_bounds_INT_RSHIFT
            OpCode::IntRshift => {
                let arg0 = op.arg(0);
                let b2 = self.getintbound(op.arg(1), ctx);
                if !b2.is_constant() {
                    return;
                }
                let r = self.getintbound(op.pos, ctx);
                let b = r.rshift_bound_backwards(&b2);
                let changed =
                    ctx.with_intbound_mut(arg0, |bm| matches!(bm.intersect(&b), Ok(true)));
                if changed {
                    self.propagate_bounds_backward(arg0, ctx);
                }
            }
            // intbounds.py:749-757 propagate_bounds_UINT_RSHIFT
            OpCode::UintRshift => {
                let arg0 = op.arg(0);
                let b2 = self.getintbound(op.arg(1), ctx);
                if !b2.is_constant() {
                    return;
                }
                let r = self.getintbound(op.pos, ctx);
                let b = r.urshift_bound_backwards(&b2);
                let changed =
                    ctx.with_intbound_mut(arg0, |bm| matches!(bm.intersect(&b), Ok(true)));
                if changed {
                    self.propagate_bounds_backward(arg0, ctx);
                }
            }
            // intbounds.py:773-782 propagate_bounds_INT_AND
            OpCode::IntAnd => {
                let arg0 = op.arg(0);
                let arg1 = op.arg(1);
                let r = self.getintbound(op.pos, ctx);
                let b0 = self.getintbound(arg0, ctx);
                let b1 = self.getintbound(arg1, ctx);
                if let Ok(b) = b0.and_bound_backwards(&r) {
                    let changed =
                        ctx.with_intbound_mut(arg1, |bm| matches!(bm.intersect(&b), Ok(true)));
                    if changed {
                        self.propagate_bounds_backward(arg1, ctx);
                    }
                }
                if let Ok(b) = b1.and_bound_backwards(&r) {
                    let changed =
                        ctx.with_intbound_mut(arg0, |bm| matches!(bm.intersect(&b), Ok(true)));
                    if changed {
                        self.propagate_bounds_backward(arg0, ctx);
                    }
                }
            }
            // intbounds.py:784-793 propagate_bounds_INT_OR
            OpCode::IntOr => {
                let arg0 = op.arg(0);
                let arg1 = op.arg(1);
                let r = self.getintbound(op.pos, ctx);
                let b0 = self.getintbound(arg0, ctx);
                let b1 = self.getintbound(arg1, ctx);
                if let Ok(b) = b0.or_bound_backwards(&r) {
                    let changed =
                        ctx.with_intbound_mut(arg1, |bm| matches!(bm.intersect(&b), Ok(true)));
                    if changed {
                        self.propagate_bounds_backward(arg1, ctx);
                    }
                }
                if let Ok(b) = b1.or_bound_backwards(&r) {
                    let changed =
                        ctx.with_intbound_mut(arg0, |bm| matches!(bm.intersect(&b), Ok(true)));
                    if changed {
                        self.propagate_bounds_backward(arg0, ctx);
                    }
                }
            }
            // intbounds.py:795-804 propagate_bounds_INT_XOR
            OpCode::IntXor => {
                let arg0 = op.arg(0);
                let arg1 = op.arg(1);
                let r = self.getintbound(op.pos, ctx);
                let b0 = self.getintbound(arg0, ctx);
                let b1 = self.getintbound(arg1, ctx);
                // xor is its own inverse
                let b = b0.xor_bound(&r);
                let changed1 =
                    ctx.with_intbound_mut(arg1, |bm| matches!(bm.intersect(&b), Ok(true)));
                if changed1 {
                    self.propagate_bounds_backward(arg1, ctx);
                }
                let b = b1.xor_bound(&r);
                let changed0 =
                    ctx.with_intbound_mut(arg0, |bm| matches!(bm.intersect(&b), Ok(true)));
                if changed0 {
                    self.propagate_bounds_backward(arg0, ctx);
                }
            }
            // intbounds.py:481-487 propagate_bounds_INT_INVERT
            OpCode::IntInvert => {
                let arg0 = op.arg(0);
                let bres = self.getintbound(op.pos, ctx);
                let bounds = bres.invert_bound();
                let changed =
                    ctx.with_intbound_mut(arg0, |bm| matches!(bm.intersect(&bounds), Ok(true)));
                if changed {
                    self.propagate_bounds_backward(arg0, ctx);
                }
            }
            // intbounds.py:489-495 propagate_bounds_INT_NEG
            OpCode::IntNeg => {
                let arg0 = op.arg(0);
                let bres = self.getintbound(op.pos, ctx);
                let bounds = bres.neg_bound();
                let changed =
                    ctx.with_intbound_mut(arg0, |bm| matches!(bm.intersect(&bounds), Ok(true)));
                if changed {
                    self.propagate_bounds_backward(arg0, ctx);
                }
            }
            // intbounds.py:608-615 propagate_bounds_INT_LT
            OpCode::IntLt => {
                let r = self.getintbound(op.pos, ctx);
                if r.is_constant() {
                    if r.lower == 1 {
                        self.make_int_lt(op.arg(0), op.arg(1), ctx);
                    } else {
                        debug_assert_eq!(r.lower, 0);
                        self.make_int_ge(op.arg(0), op.arg(1), ctx);
                    }
                }
            }
            // intbounds.py:617-624 propagate_bounds_INT_GT
            OpCode::IntGt => {
                let r = self.getintbound(op.pos, ctx);
                if r.is_constant() {
                    if r.lower == 1 {
                        self.make_int_gt(op.arg(0), op.arg(1), ctx);
                    } else {
                        debug_assert_eq!(r.lower, 0);
                        self.make_int_le(op.arg(0), op.arg(1), ctx);
                    }
                }
            }
            // intbounds.py:626-633 propagate_bounds_INT_LE
            OpCode::IntLe => {
                let r = self.getintbound(op.pos, ctx);
                if r.is_constant() {
                    if r.lower == 1 {
                        self.make_int_le(op.arg(0), op.arg(1), ctx);
                    } else {
                        debug_assert_eq!(r.lower, 0);
                        self.make_int_gt(op.arg(0), op.arg(1), ctx);
                    }
                }
            }
            // intbounds.py:635-642 propagate_bounds_INT_GE
            OpCode::IntGe => {
                let r = self.getintbound(op.pos, ctx);
                if r.is_constant() {
                    if r.lower == 1 {
                        self.make_int_ge(op.arg(0), op.arg(1), ctx);
                    } else {
                        debug_assert_eq!(r.lower, 0);
                        self.make_int_lt(op.arg(0), op.arg(1), ctx);
                    }
                }
            }
            // intbounds.py:644-651 propagate_bounds_INT_EQ
            OpCode::IntEq => {
                let r = self.getintbound(op.pos, ctx);
                if r.is_constant() && r.lower == 1 {
                    self.make_eq(op.arg(0), op.arg(1), ctx);
                } else if r.is_constant() && r.lower == 0 {
                    self.make_ne(op.arg(0), op.arg(1), ctx);
                }
            }
            // intbounds.py:653-656 propagate_bounds_INT_NE
            OpCode::IntNe => {
                let r = self.getintbound(op.pos, ctx);
                if r.is_constant() && r.lower == 0 {
                    self.make_eq(op.arg(0), op.arg(1), ctx);
                } else if r.is_constant() && r.lower == 1 {
                    self.make_ne(op.arg(0), op.arg(1), ctx);
                }
            }
            // intbounds.py:678-693 _propagate_int_is_true_or_zero +
            // propagate_bounds_INT_IS_TRUE / IS_ZERO.
            //
            // ```python
            // def _propagate_int_is_true_or_zero(self, op, valnonzero, valzero):
            //     if self.is_raw_ptr(op.getarg(0)):
            //         return
            //     r = self.getintbound(op)
            //     if r.is_constant():
            //         if r.get_constant_int() == valnonzero:
            //             b1 = self.getintbound(op.getarg(0))
            //             if b1.known_nonnegative():
            //                 b1.make_gt_const(0)
            //                 self.propagate_bounds_backward(op.getarg(0))
            //             elif b1.known_le_const(0):
            //                 b1.make_lt_const(0)
            //                 self.propagate_bounds_backward(op.getarg(0))
            //         elif r.get_constant_int() == valzero:
            //             self.make_constant_int(op.getarg(0), 0)
            //             self.propagate_bounds_backward(op.getarg(0))
            //
            // def propagate_bounds_INT_IS_TRUE(self, op):
            //     self._propagate_int_is_true_or_zero(op, 1, 0)
            //
            // def propagate_bounds_INT_IS_ZERO(self, op):
            //     self._propagate_int_is_true_or_zero(op, 0, 1)
            // ```
            OpCode::IntIsTrue => {
                self.propagate_int_is_true_or_zero(op, 1, 0, ctx);
            }
            OpCode::IntIsZero => {
                self.propagate_int_is_true_or_zero(op, 0, 1, ctx);
            }
            _ => {}
        }
    }

    /// Record what we last emitted (for overflow guard removal).
    /// Mirrors RPython's self.last_emitted_operation = op in Optimization.emit().
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

impl Optimization for OptIntBounds {
    fn propagate_forward(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
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

            // ── All other ops: default emit (RPython's Optimization.emit) ──
            // Postprocess dispatch is in propagate_postprocess, matching
            // RPython's make_dispatcher_method(OptIntBounds, 'postprocess_').
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

        result
    }

    fn setup(&mut self) {
        self.last_emitted_opcode = None;
        self.last_emitted_args.clear();
        self.last_emitted_ref = OpRef::NONE;
    }

    fn name(&self) -> &'static str {
        "intbounds"
    }

    /// RPython: have_dispatcher_method(OptIntBounds, 'postprocess_')
    ///
    /// Guard postprocess must be included in framework dispatch so
    /// intbounds can run its upstream `postprocess_GUARD_TRUE/FALSE` parity:
    /// after rewrite postprocess has fixed the guard result to 1/0,
    /// intbounds propagates that knowledge backward through the producing
    /// int expression before later ops (e.g. INT_ADD_OVF) are optimized.
    fn have_postprocess_op(&self, opcode: OpCode) -> bool {
        matches!(
            opcode,
            OpCode::GuardTrue
                | OpCode::GuardFalse
                | OpCode::GuardValue
                | OpCode::IntAdd
                | OpCode::IntSub
                | OpCode::IntMul
                | OpCode::IntAnd
                | OpCode::IntOr
                | OpCode::IntXor
                | OpCode::IntLshift
                | OpCode::IntRshift
                | OpCode::UintRshift
                | OpCode::IntFloorDiv
                | OpCode::IntMod
                | OpCode::IntNeg
                | OpCode::IntInvert
                | OpCode::IntForceGeZero
                | OpCode::IntSignext
                | OpCode::IntAddOvf
                | OpCode::IntSubOvf
                | OpCode::IntMulOvf
                | OpCode::ArraylenGc
                | OpCode::Strlen
                | OpCode::Unicodelen
                | OpCode::Strgetitem
                | OpCode::Unicodegetitem
                | OpCode::GetfieldRawI
                | OpCode::GetfieldGcI
                | OpCode::GetinteriorfieldGcI
                | OpCode::GetfieldRawR
                | OpCode::GetfieldGcR
                | OpCode::GetinteriorfieldGcR
                | OpCode::GetfieldRawF
                | OpCode::GetfieldGcF
                | OpCode::GetinteriorfieldGcF
                | OpCode::GetarrayitemRawI
                | OpCode::GetarrayitemGcI
                | OpCode::CallPureI
                | OpCode::CallI
        )
    }

    /// intbounds.py:52-58 _postprocess_guard_true_false_value parity.
    ///
    /// ```python
    /// def _postprocess_guard_true_false_value(self, op):
    ///     if op.getarg(0).type == 'i':
    ///         self.propagate_bounds_backward(op.getarg(0))
    /// ```
    ///
    /// Called AFTER the op has been emitted through ALL passes and added
    /// to new_operations. At this point, the heap pass has flushed any
    /// postponed comparison op, so find_producing_op succeeds.
    /// Matches RPython's make_dispatcher_method(OptIntBounds, 'postprocess_').
    /// Dispatches to the correct postprocess_* method based on opcode.
    fn propagate_postprocess(&mut self, op: &Op, ctx: &mut OptContext) {
        match op.opcode {
            // ── _postprocess_guard_true_false_value ──
            // intbounds.py:52-58
            OpCode::GuardTrue | OpCode::GuardFalse | OpCode::GuardValue => {
                let arg0 = ctx.get_box_replacement(op.arg(0));
                let is_int = ctx
                    .opref_type(arg0)
                    .map_or(true, |t| t == majit_ir::Type::Int);
                if !is_int {
                    return;
                }
                if op.opcode == OpCode::GuardTrue {
                    self.setintbound(arg0, IntBound::from_constant(1), ctx);
                } else if op.opcode == OpCode::GuardFalse {
                    self.setintbound(arg0, IntBound::from_constant(0), ctx);
                }
                // intbounds.py:40-50 propagate_bounds_backward
                let b = self.getintbound(arg0, ctx);
                if b.is_constant() {
                    self.make_constant_int_ref(arg0, b.get_constant(), ctx);
                }
                if let Some(producing_op) = self.find_producing_op(arg0, ctx) {
                    let producing_op = producing_op.clone();
                    self.propagate_bounds_backward_op(&producing_op, ctx);
                }
            }

            // ── Arithmetic postprocess ──
            OpCode::IntAdd => self.postprocess_int_add(op, ctx),
            OpCode::IntSub => self.postprocess_int_sub(op, ctx),
            OpCode::IntMul => self.postprocess_int_mul(op, ctx),
            OpCode::IntAnd => self.postprocess_int_and(op, ctx),
            OpCode::IntOr => self.postprocess_int_or(op, ctx),
            OpCode::IntXor => self.postprocess_int_xor(op, ctx),
            OpCode::IntLshift => self.postprocess_int_lshift(op, ctx),
            OpCode::IntRshift => self.postprocess_int_rshift(op, ctx),
            OpCode::UintRshift => self.postprocess_uint_rshift(op, ctx),
            OpCode::IntFloorDiv => self.postprocess_int_floordiv(op, ctx),
            OpCode::IntMod => self.postprocess_int_mod(op, ctx),
            OpCode::IntNeg => self.postprocess_int_neg(op, ctx),
            OpCode::IntInvert => self.postprocess_int_invert(op, ctx),
            OpCode::IntForceGeZero => self.postprocess_int_force_ge_zero(op, ctx),
            OpCode::IntSignext => self.postprocess_int_signext(op, ctx),

            // ── Overflow arithmetic postprocess ──
            OpCode::IntAddOvf => self.postprocess_int_add_ovf(op, ctx),
            OpCode::IntSubOvf => self.postprocess_int_sub_ovf(op, ctx),
            OpCode::IntMulOvf => self.postprocess_int_mul_ovf(op, ctx),

            // ── Lengths ──
            OpCode::ArraylenGc => self.postprocess_arraylen_gc(op, ctx),
            OpCode::Strlen => self.postprocess_strlen(op, ctx),
            OpCode::Unicodelen => self.postprocess_unicodelen(op, ctx),

            // ── String/Unicode items ──
            OpCode::Strgetitem => {
                self.intersect_bound(op.pos, &IntBound::bounded(0, 255), ctx);
            }
            OpCode::Unicodegetitem => {
                self.intersect_bound(op.pos, &IntBound::nonnegative(), ctx);
            }

            // ── Field accesses ──
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
                        self.intersect_bound(op.pos, &IntBound::bounded(lo, hi), ctx);
                    }
                }
            }

            // ── Array item accesses ──
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
                            self.intersect_bound(op.pos, &IntBound::bounded(lo, hi), ctx);
                        }
                    }
                }
            }

            // ── Call postprocess ──
            OpCode::CallPureI | OpCode::CallI => {
                if let Some(ref d) = op.descr {
                    if let Some(cd) = d.as_call_descr() {
                        let ei = cd.effect_info();
                        match ei.oopspec_index {
                            majit_ir::OopSpecIndex::IntPyDiv => {
                                if op.num_args() >= 3 {
                                    let divisor_bound = self.getintbound(op.arg(2), ctx);
                                    if divisor_bound.known_positive() {
                                        let result_bound = IntBound::new(
                                            0,
                                            divisor_bound.upper.saturating_sub(1),
                                            0,
                                            u64::MAX,
                                        );
                                        self.intersect_bound(op.pos, &result_bound, ctx);
                                    }
                                }
                            }
                            majit_ir::OopSpecIndex::IntPyMod => {
                                if op.num_args() >= 3 {
                                    let divisor_bound = self.getintbound(op.arg(2), ctx);
                                    if divisor_bound.known_positive() {
                                        let result_bound = IntBound::new(
                                            0,
                                            divisor_bound.upper.saturating_sub(1),
                                            0,
                                            u64::MAX,
                                        );
                                        self.intersect_bound(op.pos, &result_bound, ctx);
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }

            _ => {}
        }
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
        // The actual bounds live on box._forwarded (Forwarded::IntBound) —
        // a real implementation would iterate non-trivial bounds via
        // ctx.getintbound and generate guard ops.
    }

    fn export_arg_int_bounds(
        &self,
        args: &[OpRef],
        ctx: &OptContext,
    ) -> std::collections::HashMap<OpRef, IntBound> {
        let mut exported = std::collections::HashMap::new();
        for &arg in args {
            let resolved = ctx.get_box_replacement(arg);
            if !matches!(ctx.opref_type(resolved), Some(majit_ir::Type::Int)) {
                continue;
            }
            if let Some(bound) = ctx.peek_intbound(resolved) {
                if bound.is_unbounded() {
                    continue;
                }
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
    use majit_ir::{Descr, DescrRef};
    use std::sync::Arc;

    #[derive(Debug)]
    struct TestDescr(u32);

    impl Descr for TestDescr {
        fn index(&self) -> u32 {
            self.0
        }
    }

    fn descr(idx: u32) -> DescrRef {
        Arc::new(TestDescr(idx))
    }

    fn run_pass(ops: &[Op]) -> Vec<Op> {
        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptIntBounds::new()));
        opt.propagate_all_forward(ops)
    }

    /// Create an OptIntBounds with specific bounds pre-set and run it on ops.
    fn run_pass_with_bounds(
        ops: &[Op],
        initial_bounds: &[(OpRef, IntBound)],
    ) -> (Vec<Op>, OptContext) {
        let mut pass = OptIntBounds::new();
        // Compute num_inputs as the highest OpRef referenced anywhere (as
        // arg, result pos, or initial bound key) plus one. With bounds now
        // living on `ctx.forwarded`, reserve_pos must skip past every
        // pre-existing OpRef so freshly emitted ops (e.g. SameAsI from
        // rewrite's `x + x → x << 1`) cannot collide with an input arg.
        let max_arg = ops
            .iter()
            .flat_map(|op| op.args.iter().copied())
            .filter(|r| !r.is_constant() && !r.is_none())
            .map(|r| r.0)
            .max()
            .unwrap_or(0);
        let max_pos = ops
            .iter()
            .map(|op| op.pos)
            .filter(|r| !r.is_constant() && !r.is_none())
            .map(|r| r.0)
            .max()
            .unwrap_or(0);
        let max_initial = initial_bounds.iter().map(|(r, _)| r.0).max().unwrap_or(0);
        let num_inputs = (max_arg.max(max_pos).max(max_initial) as usize) + 1;
        let mut ctx = OptContext::with_num_inputs(ops.len(), num_inputs);

        pass.setup();
        for (opref, bound) in initial_bounds {
            ctx.setintbound(*opref, bound);
        }

        for op in ops.iter() {
            let mut resolved_op = op.clone();
            // Keep the op.pos as set by the test (not overriding with index)
            for arg in &mut resolved_op.args {
                *arg = ctx.get_box_replacement(*arg);
            }
            let emitted_op = match pass.propagate_forward(&resolved_op, &mut ctx) {
                OptimizationResult::Emit(emit_op) => {
                    ctx.emit(emit_op.clone());
                    Some(emit_op)
                }
                OptimizationResult::Replace(rep_op) => {
                    ctx.emit(rep_op.clone());
                    Some(rep_op)
                }
                OptimizationResult::Remove => None,
                OptimizationResult::PassOn => {
                    ctx.emit(resolved_op.clone());
                    Some(resolved_op)
                }
                OptimizationResult::InvalidLoop => {
                    panic!("unexpected InvalidLoop in test");
                }
            };
            if let Some(ref emitted) = emitted_op {
                // Simulate rewrite pass postprocess: make_constant on guard condition.
                // In the real optimizer, rewrite.rs postprocess_GUARD_TRUE sets
                // condition = CONST_1 before intbounds postprocess runs.
                match emitted.opcode {
                    OpCode::GuardTrue => {
                        let cond = ctx.get_box_replacement(emitted.arg(0));
                        ctx.make_constant(cond, majit_ir::Value::Int(1));
                    }
                    OpCode::GuardFalse => {
                        let cond = ctx.get_box_replacement(emitted.arg(0));
                        ctx.make_constant(cond, majit_ir::Value::Int(0));
                    }
                    _ => {}
                }
                pass.propagate_postprocess(emitted, &mut ctx);
            }
        }

        let result = ctx.new_operations.clone();
        let _ = pass;
        (result, ctx)
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

        let (result, mut ctx) = run_pass_with_bounds(&ops, &initial_bounds);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].opcode, OpCode::IntAdd);

        // The result should have bounds [5, 30]
        let b = ctx.getintbound(OpRef(2));
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

        let (result, _ctx) = run_pass_with_bounds(&ops, &initial_bounds);
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

        let (result, _ctx) = run_pass_with_bounds(&ops, &initial_bounds);
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

        let (_result, mut ctx) = run_pass_with_bounds(&ops, &initial_bounds);

        // After the guard, i0 should be < 10, meaning upper <= 9
        let b0 = ctx.getintbound(OpRef(0));
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

        let (_result, mut ctx) = run_pass_with_bounds(&ops, &initial_bounds);

        let b0 = ctx.getintbound(OpRef(0));
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

        let (result, _ctx) = run_pass_with_bounds(&ops, &initial_bounds);
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

        let (result, _ctx) = run_pass_with_bounds(&ops, &initial_bounds);
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

        let (result, _ctx) = run_pass_with_bounds(&ops, &initial_bounds);
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

        let (result, _ctx) = run_pass_with_bounds(&ops, &initial_bounds);
        assert!(
            result.iter().any(|op| op.opcode == OpCode::IntMul),
            "INT_MUL_OVF should be transformed to INT_MUL"
        );
    }

    #[test]
    fn test_second_overflow_guard_survives_after_first_guard() {
        // Use OpRef(3) for guard condition so it doesn't affect overflow operands.
        // GuardTrue(OpRef(3)) makes OpRef(3)=1 via rewrite postprocess,
        // but OpRef(0), OpRef(1), OpRef(2) remain unbounded.
        let initial_bounds = vec![
            (OpRef(0), IntBound::unbounded()),
            (OpRef(1), IntBound::unbounded()),
            (OpRef(2), IntBound::unbounded()),
            (OpRef(3), IntBound::unbounded()),
        ];
        let ops = vec![
            make_op(OpCode::GuardTrue, &[OpRef(3)], 4),
            make_op(OpCode::IntSubOvf, &[OpRef(0), OpRef(1)], 5),
            make_op(OpCode::GuardNoOverflow, &[], 6),
            make_op(OpCode::IntMulOvf, &[OpRef(2), OpRef(1)], 7),
            make_op(OpCode::GuardNoOverflow, &[], 8),
            make_op(OpCode::Jump, &[OpRef(5), OpRef(5), OpRef(7)], 9),
        ];

        let (result, _ctx) = run_pass_with_bounds(&ops, &initial_bounds);
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

        let (result, mut ctx) = run_pass_with_bounds(&ops, &initial_bounds);
        // Should be removed (replaced by constant 1)
        assert!(
            result.is_empty(),
            "INT_LT should be removed when known true"
        );
        // The constant should be set
        let b = ctx.getintbound(OpRef(2));
        assert!(b.is_constant() && b.get_constant() == 1);
    }

    #[test]
    fn test_int_lt_known_false() {
        let initial_bounds = vec![
            (OpRef(0), IntBound::bounded(10, 20)),
            (OpRef(1), IntBound::bounded(0, 5)),
        ];
        let ops = vec![make_op(OpCode::IntLt, &[OpRef(0), OpRef(1)], 2)];

        let (result, mut ctx) = run_pass_with_bounds(&ops, &initial_bounds);
        assert!(
            result.is_empty(),
            "INT_LT should be removed when known false"
        );
        let b = ctx.getintbound(OpRef(2));
        assert!(b.is_constant() && b.get_constant() == 0);
    }

    #[test]
    fn test_int_eq_same_arg() {
        let ops = vec![make_op(OpCode::IntEq, &[OpRef(0), OpRef(0)], 1)];

        let (result, mut ctx) = run_pass_with_bounds(&ops, &[]);
        assert!(
            result.is_empty(),
            "INT_EQ(x, x) should be removed (always 1)"
        );
        let b = ctx.getintbound(OpRef(1));
        assert!(b.is_constant() && b.get_constant() == 1);
    }

    #[test]
    fn test_int_ne_same_arg() {
        let ops = vec![make_op(OpCode::IntNe, &[OpRef(0), OpRef(0)], 1)];

        let (result, mut ctx) = run_pass_with_bounds(&ops, &[]);
        assert!(
            result.is_empty(),
            "INT_NE(x, x) should be removed (always 0)"
        );
        let b = ctx.getintbound(OpRef(1));
        assert!(b.is_constant() && b.get_constant() == 0);
    }

    #[test]
    fn test_int_le_known_true() {
        let initial_bounds = vec![
            (OpRef(0), IntBound::bounded(0, 5)),
            (OpRef(1), IntBound::bounded(5, 20)),
        ];
        let ops = vec![make_op(OpCode::IntLe, &[OpRef(0), OpRef(1)], 2)];

        let (result, _ctx) = run_pass_with_bounds(&ops, &initial_bounds);
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

        let (result, _ctx) = run_pass_with_bounds(&ops, &initial_bounds);
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

        let (result, mut ctx) = run_pass_with_bounds(&ops, &initial_bounds);
        assert_eq!(result.len(), 1);
        let b = ctx.getintbound(OpRef(2));
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

        let (result, mut ctx) = run_pass_with_bounds(&ops, &initial_bounds);
        assert_eq!(result.len(), 1);
        let b = ctx.getintbound(OpRef(2));
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

        let (result, mut ctx) = run_pass_with_bounds(&ops, &initial_bounds);
        assert_eq!(result.len(), 1);
        let b = ctx.getintbound(OpRef(2));
        // AND of [0, 255] and [0, 15] -> [0, 15]
        assert!(b.lower >= 0);
        assert!(b.upper <= 15);
    }

    #[test]
    fn test_int_force_ge_zero() {
        let initial_bounds = vec![(OpRef(0), IntBound::bounded(-10, 20))];
        let ops = vec![make_op(OpCode::IntForceGeZero, &[OpRef(0)], 1)];

        let (_result, mut ctx) = run_pass_with_bounds(&ops, &initial_bounds);
        let b = ctx.getintbound(OpRef(1));
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
        let mut op = make_op(OpCode::ArraylenGc, &[OpRef(0)], 1);
        op.descr = Some(descr(1));
        let ops = vec![op];

        let (_result, mut ctx) = run_pass_with_bounds(&ops, &[]);
        let b = ctx.getintbound(OpRef(1));
        assert!(b.lower >= 0, "ARRAYLEN_GC result should be non-negative");
    }

    #[test]
    fn test_strlen_nonneg() {
        let ops = vec![make_op(OpCode::Strlen, &[OpRef(0)], 1)];

        let (_result, mut ctx) = run_pass_with_bounds(&ops, &[]);
        let b = ctx.getintbound(OpRef(1));
        assert!(b.lower >= 0, "STRLEN result should be non-negative");
    }

    #[test]
    fn test_int_neg_bounds() {
        let initial_bounds = vec![(OpRef(0), IntBound::bounded(3, 10))];
        let ops = vec![make_op(OpCode::IntNeg, &[OpRef(0)], 1)];

        let (_result, mut ctx) = run_pass_with_bounds(&ops, &initial_bounds);
        let b = ctx.getintbound(OpRef(1));
        // neg([3, 10]) = [-10, -3]
        assert_eq!(b.lower, -10);
        assert_eq!(b.upper, -3);
    }

    #[test]
    fn test_int_invert_bounds() {
        let initial_bounds = vec![(OpRef(0), IntBound::bounded(3, 10))];
        let ops = vec![make_op(OpCode::IntInvert, &[OpRef(0)], 1)];

        let (_result, mut ctx) = run_pass_with_bounds(&ops, &initial_bounds);
        let b = ctx.getintbound(OpRef(1));
        // invert([3, 10]) = [!10, !3] = [-11, -4]
        assert_eq!(b.lower, -11);
        assert_eq!(b.upper, -4);
    }

    #[test]
    fn test_sub_ovf_same_arg() {
        // INT_SUB_OVF(x, x) should be replaced by constant 0
        let initial_bounds = vec![(OpRef(0), IntBound::unbounded())];
        let ops = vec![make_op(OpCode::IntSubOvf, &[OpRef(0), OpRef(0)], 1)];

        let (result, mut ctx) = run_pass_with_bounds(&ops, &initial_bounds);
        assert!(result.is_empty(), "INT_SUB_OVF(x, x) should be removed");
        let b = ctx.getintbound(OpRef(1));
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

        let (result, _ctx) = run_pass_with_bounds(&ops, &initial_bounds);
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

        let (_result, mut ctx) = run_pass_with_bounds(&ops, &initial_bounds);
        let b = ctx.getintbound(OpRef(2));
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

        let (_result, mut ctx) = run_pass_with_bounds(&ops, &initial_bounds);
        let b = ctx.getintbound(OpRef(2));
        // [8, 20] >> 2 = [2, 5]
        assert_eq!(b.lower, 2);
        assert_eq!(b.upper, 5);
    }

    // ── Test: INT_IS_TRUE and INT_IS_ZERO produce bool bounds ──

    #[test]
    fn test_int_is_true_passthrough() {
        // RPython: IntIsTrue has no postprocess — just passes through.
        let ops = vec![make_op(OpCode::IntIsTrue, &[OpRef(0)], 1)];
        let (result, _ctx) = run_pass_with_bounds(&ops, &[]);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].opcode, OpCode::IntIsTrue);
    }

    #[test]
    fn test_int_is_zero_passthrough() {
        // RPython: IntIsZero has no postprocess — just passes through.
        let ops = vec![make_op(OpCode::IntIsZero, &[OpRef(0)], 1)];
        let (result, _ctx) = run_pass_with_bounds(&ops, &[]);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].opcode, OpCode::IntIsZero);
    }

    // ── Test: Comparison with unknown bounds stays ──

    #[test]
    fn test_int_lt_unknown_not_removed() {
        let initial_bounds = vec![
            (OpRef(0), IntBound::unbounded()),
            (OpRef(1), IntBound::unbounded()),
        ];
        let ops = vec![make_op(OpCode::IntLt, &[OpRef(0), OpRef(1)], 2)];

        let (result, mut ctx) = run_pass_with_bounds(&ops, &initial_bounds);
        assert_eq!(
            result.len(),
            1,
            "INT_LT should remain when bounds are unknown"
        );
        assert_eq!(result[0].opcode, OpCode::IntLt);
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

        let (result, _ctx) = run_pass_with_bounds(&ops, &initial_bounds);
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

        let (result, mut ctx) = run_pass_with_bounds(&ops, &initial_bounds);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].opcode, OpCode::IntSignext);
        // Result should have bounds [-128, 127]
        let b = ctx.getintbound(OpRef(2));
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

        let (result, mut ctx) = run_pass_with_bounds(&ops, &initial_bounds);
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

        let (_result, mut ctx) = run_pass_with_bounds(&ops, &initial_bounds);
        let b0 = ctx.getintbound(OpRef(0));
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

        let (_result, mut ctx) = run_pass_with_bounds(&ops, &initial_bounds);
        let b0 = ctx.getintbound(OpRef(0));
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

        let (_result, mut ctx) = run_pass_with_bounds(&ops, &initial_bounds);
        let b = ctx.getintbound(OpRef(1));
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

        let (_result, mut ctx) = run_pass_with_bounds(&ops, &initial_bounds);
        // Manually tighten the result and trigger backward prop. The
        // OptIntBounds pass is stateless beyond `pure_from_args_cache`
        // (everything else lives on `ctx.forwarded`), so we can spin up a
        // fresh one to drive the backward propagation step.
        let mut pass = OptIntBounds::new();
        pass.setintbound(OpRef(1), IntBound::bounded(-5, -1), &mut ctx);
        pass.propagate_bounds_backward(OpRef(1), &mut ctx);
        let b0 = ctx.getintbound(OpRef(0));
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
        // postprocess_STRGETITEM: result should be bounded to [0, 255].
        let ops = vec![make_op(OpCode::Strgetitem, &[OpRef(0), OpRef(1)], 2)];
        let (_result, mut ctx) = run_pass_with_bounds(&ops, &[]);
        let b = ctx.getintbound(OpRef(2));
        assert!(b.lower >= 0, "STRGETITEM lower should be >= 0");
        assert!(b.upper <= 255, "STRGETITEM upper should be <= 255");
    }

    /// RPython postprocess_GUARD_TRUE parity test:
    /// GuardTrue(IntLt(i, len)) should tighten i's upper bound so that
    /// IntAddOvf(i, 1) is converted to IntAdd. This is the nbody inner
    /// loop pattern and relies on rewrite postprocess first fixing the
    /// comparison result to 1, then intbounds driving
    /// propagate_bounds_backward_op() / intbounds.py:608-651.
    #[test]
    fn test_guard_true_int_lt_enables_add_ovf_removal() {
        use crate::optimizeopt::optimizer::Optimizer;

        let mut ops = vec![
            make_op(OpCode::IntLt, &[OpRef(0), OpRef(1)], 2),
            make_op(OpCode::GuardTrue, &[OpRef(2)], 3),
            make_op(OpCode::IntAddOvf, &[OpRef(0), OpRef(200)], 4),
            make_op(OpCode::GuardNoOverflow, &[], 5),
            make_op(OpCode::Jump, &[OpRef(4)], 6),
        ];

        let mut opt = Optimizer::default_pipeline();
        // IntLt/IntAddOvf operate on Int-typed inputs — override the
        // test default (Ref) used by `optimize_with_constants_and_inputs_at`.
        opt.trace_inputarg_types = vec![majit_ir::Type::Int; 1024];
        let mut constants = std::collections::HashMap::new();
        constants.insert(200, 1i64);
        let result = opt.optimize_with_constants_and_inputs(&ops, &mut constants, 1024);

        let opcodes: Vec<_> = result.iter().map(|op| op.opcode).collect();
        assert!(
            result.iter().any(|op| op.opcode == OpCode::IntAdd),
            "INT_ADD_OVF should be transformed to INT_ADD after GuardTrue(IntLt); got {opcodes:?}"
        );
        assert!(
            !result.iter().any(|op| op.opcode == OpCode::IntAddOvf),
            "INT_ADD_OVF should not remain; got {opcodes:?}"
        );
    }
}
