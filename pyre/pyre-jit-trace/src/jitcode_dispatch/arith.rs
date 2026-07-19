//! Arithmetic / comparison / cast / ptr-and-ref record helpers.
//!
//! The `handle` dispatch arms for the `int_*` / `float_*` / `ptr_*`
//! opcode families funnel into these kind-parameterized recorders --
//! the Rust analogue of the exec-generated `opimpl_*` loops in
//! `pyjitpl.py:279-368`: read the operand register(s), record the
//! `OpCode`, write the recorder result into the destination bank.

use super::*;

/// Generic int-bank binop handler. Reads `registers_i[src1]` and
/// `registers_i[src2]`, records `record_op(opcode, [a, b])`, writes
/// the recorder's result OpRef into `registers_i[dst]`. Operand
/// layout is `ii>i` (1B src1 + 1B src2 + 1B dst).
///
/// RPython parity: `pyjitpl.py:288-292` exec-generated
/// `opimpl_int_BINOP(b1, b2): return self.execute(rop.<OPNUM>, b1,
/// b2)` + the trailing `>i` decorator that writes the result into
/// `registers_i[dst]`. Walker collapses execute+writeback into
/// `record_op + slot store`, which matches the recording-only side of
/// `execute`'s split (`pyjitpl.py:_record_helper`).
pub(crate) fn binop_int_record(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_>,
    opcode: OpCode,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    let a = read_int_reg(code, op, 0, ctx)?;
    let b = read_int_reg(code, op, 1, ctx)?;
    let result = ctx.trace_ctx.record_op(opcode, &[a, b]);
    // Box(value) parity: stamp the result from the operands' Box.value
    // carriers (BoxInt(value) — matches dispatch.rs trace_binop_i).
    // The folded value also feeds the slot-keyed `concrete_registers_i`
    // shadow via [`write_int_reg`] so handlers that read the slot
    // (Ref-bank symmetry) see the same concrete as the OpRef channel.
    let concrete = if let (Some(majit_ir::Value::Int(la)), Some(majit_ir::Value::Int(rb))) =
        (ctx.trace_ctx.box_value(a), ctx.trace_ctx.box_value(b))
    {
        let folded = majit_metainterp::eval_binop_i(opcode, la, rb);
        ctx.trace_ctx
            .set_opref_concrete(result, majit_ir::Value::Int(folded));
        ConcreteValue::Int(folded)
    } else {
        ConcreteValue::Null
    };
    let dst = code[op.pc + 3] as usize;
    write_int_reg(ctx, op.pc, dst, result, concrete)?;
    Ok((DispatchOutcome::Continue, op.next_pc))
}

/// RPython `pyjitpl.py:588-595 opimpl_int_between`:
///
/// ```python
/// b5 = self.execute(rop.INT_SUB, b3, b1)
/// if isinstance(b5, ConstInt) and b5.getint() == 1:
///     # int_between(a, b, a+1) -> b == a
///     return self.execute(rop.INT_EQ, b2, b1)
/// else:
///     b4 = self.execute(rop.INT_SUB, b2, b1)
///     return self.execute(rop.UINT_LT, b4, b5)
/// ```
///
/// Decomposes `int_between(b1, b2, b3)` — i.e. `b1 <= b2 < b3` — at
/// record time, matching upstream's choice to emit elementary
/// `INT_SUB`/`INT_EQ`/`UINT_LT` into the trace rather than relying on
/// the optimizer to lower `INT_BETWEEN`.  The blackhole semantics
/// (`blackhole.py:560-561 bhimpl_int_between(a, b, c): return a <= b
/// < c`) are preserved through the same decomposition.
///
/// Operand layout `iii>i`: 3B sources + 1B dst (=4 operand bytes after
/// the opcode).  Concrete-value propagation in [`execute_pure_binop_i`]
/// runs in two layers: all-inline-Const operand pairs fold to a
/// `const_int(...)` OpRef without recording (matching upstream
/// `_all_constants` short-circuit at `pyjitpl.py:2654-2660`); the
/// trailing concrete-tracked-pair path additionally stamps the recorded
/// op via `set_opref_concrete`.  The `ConstInt(1)` fast path at
/// `pyjitpl.py:590` keys on the inline-Const layer through
/// `inline_const_to_value`, mirroring `isinstance(b5, ConstInt)` —
/// box_value's concrete-stamp layer does not participate in that
/// branch decision.
pub(crate) fn int_between_record(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_>,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    let b1 = read_int_reg(code, op, 0, ctx)?;
    let b2 = read_int_reg(code, op, 1, ctx)?;
    let b3 = read_int_reg(code, op, 2, ctx)?;

    // b5 = execute(INT_SUB, b3, b1)
    let b5 = execute_pure_binop_i(ctx, OpCode::IntSub, b3, b1);

    // pyjitpl.py:590 `if isinstance(b5, ConstInt) and b5.getint() == 1`
    // — the `ConstInt(1)` fast path emits INT_EQ; otherwise the
    // generic INT_SUB + UINT_LT pair.  `inline_const_to_value` returns
    // `Some(_)` exactly when `b5` is an inline-Const OpRef, mirroring
    // `isinstance(b5, ConstInt)`.
    let result = if let Some(majit_ir::Value::Int(1)) = b5.inline_const_to_value() {
        // execute(INT_EQ, b2, b1)
        execute_pure_binop_i(ctx, OpCode::IntEq, b2, b1)
    } else {
        // b4 = execute(INT_SUB, b2, b1); execute(UINT_LT, b4, b5)
        let b4 = execute_pure_binop_i(ctx, OpCode::IntSub, b2, b1);
        execute_pure_binop_i(ctx, OpCode::UintLt, b4, b5)
    };

    let concrete = match result.inline_const_to_value() {
        Some(majit_ir::Value::Int(v)) => ConcreteValue::Int(v),
        _ => match ctx.trace_ctx.box_value(result) {
            Some(majit_ir::Value::Int(v)) => ConcreteValue::Int(v),
            _ => ConcreteValue::Null,
        },
    };

    let dst = code[op.pc + 4] as usize;
    write_int_reg(ctx, op.pc, dst, result, concrete)?;
    Ok((DispatchOutcome::Continue, op.next_pc))
}

/// `pyjitpl.py:2648-2662 execute_and_record` for pure integer binops.
/// When both operands are inline-Const, fold to a `ConstInt` OpRef
/// (no `record_op` call); otherwise record the op and stamp the
/// observed concrete value if both sides have one.
///
/// PyPy's `execute_and_record` short-circuits `_record_helper` via
/// `executor.wrap_constant(resvalue)` when `_all_constants(*argboxes)`
/// holds for a pure opcode.  Mirroring that here keeps the trace
/// free of all-constant subexpressions exactly where upstream keeps
/// it free — `opimpl_int_between` chains three such pure binops.
pub(crate) fn execute_pure_binop_i(
    ctx: &mut WalkContext<'_, '_>,
    opcode: OpCode,
    a: OpRef,
    b: OpRef,
) -> OpRef {
    if let (Some(majit_ir::Value::Int(va)), Some(majit_ir::Value::Int(vb))) =
        (a.inline_const_to_value(), b.inline_const_to_value())
    {
        let folded = majit_metainterp::eval_binop_i(opcode, va, vb);
        return ctx.trace_ctx.const_int(folded);
    }

    let result = ctx.trace_ctx.record_op(opcode, &[a, b]);
    if let (Some(majit_ir::Value::Int(va)), Some(majit_ir::Value::Int(vb))) =
        (ctx.trace_ctx.box_value(a), ctx.trace_ctx.box_value(b))
    {
        let folded = majit_metainterp::eval_binop_i(opcode, va, vb);
        ctx.trace_ctx
            .set_opref_concrete(result, majit_ir::Value::Int(folded));
    }
    result
}

/// Generic int-bank unary handler. Operand layout `i>i` (1B src + 1B
/// dst). RPython parity: `pyjitpl.py:356-368` exec-generated
/// `opimpl_int_<unary>` (int_neg / int_invert / int_is_zero etc.) +
/// the `>i` decorator's writeback. Walker reads `registers_i[src]`,
/// records `OpCode::<Variant>` with `[a]`, writes the recorder result
/// into `registers_i[dst]`.
pub(crate) fn unop_int_record(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_>,
    opcode: OpCode,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    let a = read_int_reg(code, op, 0, ctx)?;
    let result = ctx.trace_ctx.record_op(opcode, &[a]);
    // Box(value) parity: stamp the unary result from the operand's
    // Box.value carrier (matches dispatch.rs trace_unary_i).  The
    // folded value also feeds the slot-keyed shadow via
    // [`write_int_reg`].
    let concrete = if let Some(majit_ir::Value::Int(n)) = ctx.trace_ctx.box_value(a) {
        let folded = majit_metainterp::eval_unary_i(opcode, n);
        ctx.trace_ctx
            .set_opref_concrete(result, majit_ir::Value::Int(folded));
        ConcreteValue::Int(folded)
    } else {
        ConcreteValue::Null
    };
    let dst = code[op.pc + 2] as usize;
    write_int_reg(ctx, op.pc, dst, result, concrete)?;
    Ok((DispatchOutcome::Continue, op.next_pc))
}

/// Generic ref-bank → int-bank binop handler. Operand layout `rr>i`
/// (1B r-src1 + 1B r-src2 + 1B i-dst). RPython parity:
/// `pyjitpl.py:326-336` exec-generated `opimpl_ptr_eq` /
/// `opimpl_ptr_ne` (and instance variants) follow `self.execute(rop.<OPNUM>,
/// b1, b2)` — both `b1`/`b2` are ref boxes, result is an int box. The
/// `b1 is b2` fast path is omitted (same rationale as `binop_int_record`'s
/// comparison family — pyre's recorder shares constants by value).
pub(crate) fn binop_ref_to_int_record(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_>,
    opcode: OpCode,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    let a = read_ref_reg(code, op, 0, ctx)?;
    let b = read_ref_reg(code, op, 1, ctx)?;
    let result = ctx.trace_ctx.record_op(opcode, &[a, b]);
    // Box(value) parity: stamp the bool result from the operands' Box.value
    // carriers (matches dispatch.rs trace_binop_r_to_i).
    if let (Some(majit_ir::Value::Ref(la)), Some(majit_ir::Value::Ref(rb))) =
        (ctx.trace_ctx.box_value(a), ctx.trace_ctx.box_value(b))
    {
        let folded = match opcode {
            OpCode::PtrEq | OpCode::InstancePtrEq => (la == rb) as i64,
            OpCode::PtrNe | OpCode::InstancePtrNe => (la != rb) as i64,
            _ => panic!("binop_ref_to_int_record: unsupported opcode {opcode:?}"),
        };
        ctx.trace_ctx
            .set_opref_concrete(result, majit_ir::Value::Int(folded));
    }
    let dst = code[op.pc + 3] as usize;
    let concrete_for_shadow = concrete_from_recorded_opref(ctx, result);
    write_int_reg(ctx, op.pc, dst, result, concrete_for_shadow)?;
    Ok((DispatchOutcome::Continue, op.next_pc))
}

/// `ptr_nonzero/r>i` (`nonzero = true`) and `ptr_iszero/r>i`
/// (`nonzero = false`) handler (operand layout `r>i`: 1B r-src + 1B i-dst).
///
/// RPython parity:
/// `pyjitpl.py:378-380 opimpl_ptr_nonzero(box)`:
/// ```python
/// @arguments("box")
/// def opimpl_ptr_nonzero(self, box):
///     return self.execute(rop.PTR_NE, box, CONST_NULL)
/// ```
/// `opimpl_ptr_iszero` is the `PTR_EQ` complement.
///
/// Walker reads one `r` reg, records `OpCode::PtrNe`/`PtrEq` with
/// `[box, CONST_NULL]` (via `trace_ctx.const_null()` —
/// `history.py:361 CONST_NULL = ConstPtr(ConstPtr.value)`), and writes
/// the recorder result into `registers_i[dst]`.  RPython does the
/// same `b1 is b2` short-circuit at `pyjitpl.py:328-332` for
/// `opimpl_ptr_eq` but the nullity test against `CONST_NULL` cannot
/// short-circuit because `box` is never the literal `CONST_NULL`
/// constant (codewriter would have folded that).
pub(crate) fn ptr_nullity_record(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_>,
    nonzero: bool,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    let box_ = read_ref_reg(code, op, 0, ctx)?;
    let null_const = ctx.trace_ctx.const_null();
    let opcode = if nonzero {
        OpCode::PtrNe
    } else {
        OpCode::PtrEq
    };
    let result = ctx.trace_ctx.record_op(opcode, &[box_, null_const]);
    // Concrete stamp: prefer the box's own value carrier (constant pool /
    // standard-virtualizable shadow / `set_opref_concrete` stamp).  When the
    // OpRef has no intrinsic concrete (e.g. a residual-call result whose
    // executor was gated off, or an inline sub-walk callee argument whose
    // concrete pointer lives only in the walk's ref-register shadow), fall
    // back to the register's concrete shadow (`concrete_registers_r`).
    // A non-`Ref` shadow (`Null`) means the pointer is untracked, not
    // provably null — leave the result symbolic.
    let nonnull = match ctx.trace_ctx.box_value(box_) {
        Some(majit_ir::Value::Ref(r)) => Some(r.0 != 0),
        _ => match read_ref_reg_concrete(code, op, 0, ctx) {
            ConcreteValue::Ref(p) => Some(!p.is_null()),
            _ => None,
        },
    };
    if let Some(nonnull) = nonnull {
        ctx.trace_ctx
            .set_opref_concrete(result, majit_ir::Value::Int((nonnull == nonzero) as i64));
    }
    let dst = code[op.pc + 2] as usize;
    let concrete_for_shadow = concrete_from_recorded_opref(ctx, result);
    write_int_reg(ctx, op.pc, dst, result, concrete_for_shadow)?;
    Ok((DispatchOutcome::Continue, op.next_pc))
}

/// `ref_guard_value/r` handler (operand layout `r`: 1B r-src, no dst).
///
/// RPython parity: `pyjitpl.py:1494-1496 _opimpl_guard_value` →
/// `pyjitpl.py:1916-1927 implement_guard_value`:
///
/// ```python
/// def implement_guard_value(self, box, orgpc):
///     if isinstance(box, Const):
///         return box                     # no promotion needed
///     else:
///         promoted_box = executor.constant_from_op(box)
///         self.metainterp.generate_guard(rop.GUARD_VALUE, box,
///                                        promoted_box, resumepc=orgpc)
///         self.metainterp.replace_box(box, promoted_box)
///         return promoted_box
/// ```
///
/// Walker behaviour:
///   * Read 1B Ref operand and its concrete shadow.
///   * If the symbolic OpRef is already a Const, skip (Const arm of
///     `implement_guard_value`).
///   * If the concrete shadow is `ConcreteValue::Null`, skip — the
///     walker doesn't have a runtime value to mint the expected
///     constant from.  This is the strictest mode (sibling
///     `dispatch_switch_id` line 1207 falls into the same skip-guard
///     branch when `valuebox.is_constant()`).
///   * Otherwise mint `ConstPtr(concrete_ptr)` (executor.py:544-551
///     `constant_from_op` for a Ref-typed Box), emit `GuardValue`
///     with `[value, expected_ref]`, and call `replace_box(value,
///     expected_ref)` (pyjitpl.py:1923).  Also rewrite every
///     `registers_r` slot still pointing at `value` to `expected_ref`,
///     matching `dispatch_switch_id:1198-1202`.
///
/// TODO: guards record with empty resume data
/// (`record_guard(..., 0)`) — same caveat as `dispatch_switch_id`
/// (no MIFrame liveness / framestack in the standalone walker).
pub(crate) fn ref_guard_value_record(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_>,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    let value = read_ref_reg(code, op, 0, ctx)?;
    if value.is_constant() {
        return Ok((DispatchOutcome::Continue, op.next_pc));
    }
    let concrete = read_ref_reg_concrete(code, op, 0, ctx);
    let ConcreteValue::Ref(ptr) = concrete else {
        return Ok((DispatchOutcome::Continue, op.next_pc));
    };
    if ptr.is_null() {
        return Ok((DispatchOutcome::Continue, op.next_pc));
    }
    let expected = ctx.trace_ctx.const_ref(ptr as usize as i64);
    ctx.trace_ctx
        .record_guard(OpCode::GuardValue, &[value, expected], 0);
    walker_capture_snapshot_for_last_guard(ctx, op.pc)?;
    ctx.trace_ctx.replace_box(value, expected);
    for slot in ctx.registers_r.iter_mut() {
        if *slot == value {
            *slot = expected;
        }
    }
    Ok((DispatchOutcome::Continue, op.next_pc))
}

/// Generic float-pair-to-int handler for `float_<cmp>/ff>i` (operand
/// layout `ff>i`: 1B f-src + 1B f-src + 1B i-dst).  RPython parity:
/// `bhimpl_float_{lt,le,eq,ne,gt,ge}` (`blackhole.py:721-746`) — read
/// two `f` regs, record `OpCode::Float<Cmp>`, write the recorder
/// result into `registers_i[dst]`.
pub(crate) fn binop_float_to_int_record(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_>,
    opcode: OpCode,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    let a = read_float_reg(code, op, 0, ctx)?;
    let b = read_float_reg(code, op, 1, ctx)?;
    let result = ctx.trace_ctx.record_op(opcode, &[a, b]);
    // Box(value) parity: stamp the bool result from the operands' Box.value
    // carriers (matches dispatch.rs GOTO_IF_NOT_FLOAT_* + trace_float_compare).
    if let (Some(majit_ir::Value::Float(fa)), Some(majit_ir::Value::Float(fb))) =
        (ctx.trace_ctx.box_value(a), ctx.trace_ctx.box_value(b))
    {
        let folded =
            majit_metainterp::eval_float_cmp(opcode, fa.to_bits() as i64, fb.to_bits() as i64);
        ctx.trace_ctx
            .set_opref_concrete(result, majit_ir::Value::Int(folded));
    }
    let dst = code[op.pc + 3] as usize;
    let concrete_for_shadow = concrete_from_recorded_opref(ctx, result);
    write_int_reg(ctx, op.pc, dst, result, concrete_for_shadow)?;
    Ok((DispatchOutcome::Continue, op.next_pc))
}

/// `cast_int_to_float/i>f` handler. Operand layout `i>f` (1B i-src +
/// 1B f-dst). RPython parity: `pyjitpl.py:357 cast_int_to_float`
/// belongs to the same exec-generated unary opimpl loop —
/// `self.execute(rop.CAST_INT_TO_FLOAT, b)`. Result lands in the
/// float bank (the `>f` decorator) instead of the int bank.
pub(crate) fn cast_int_to_float_record(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_>,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    let a = read_int_reg(code, op, 0, ctx)?;
    let result = ctx.trace_ctx.record_op(OpCode::CastIntToFloat, &[a]);
    // Box.value parity — if `a`'s runtime concrete is known, stamp
    // the cast result with the corresponding float bit-pattern so
    // downstream `box_value(result)` callers see the live value.
    if let Some(majit_ir::Value::Int(n)) = ctx.trace_ctx.box_value(a) {
        ctx.trace_ctx
            .set_opref_concrete(result, majit_ir::Value::Float(n as f64));
    }
    let dst = code[op.pc + 2] as usize;
    let len = ctx.registers_f.len();
    let slot = ctx
        .registers_f
        .get_mut(dst)
        .ok_or(DispatchError::RegisterOutOfRange {
            pc: op.pc,
            reg: dst,
            len,
            bank: "f",
        })?;
    *slot = result;
    Ok((DispatchOutcome::Continue, op.next_pc))
}

/// Generic float-bank binop handler. Operand layout `ff>f` (1B src1
/// + 1B src2 + 1B dst). RPython parity: same as `binop_int_record`
/// but on the float bank — `pyjitpl.py:284-292`'s exec-generated
/// `opimpl_float_<binop>` reads two `f` regs, calls
/// `self.execute(rop.<OPNUM>, b1, b2)`, and the trailing `>f`
/// decorator writes the result into `registers_f[dst]`.
pub(crate) fn binop_float_record(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_>,
    opcode: OpCode,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    let a = read_float_reg(code, op, 0, ctx)?;
    let b = read_float_reg(code, op, 1, ctx)?;
    let result = ctx.trace_ctx.record_op(opcode, &[a, b]);
    // Box(value) parity: stamp the result from the operands' Box.value
    // carriers (matches dispatch.rs trace_binop_f).
    if let (Some(majit_ir::Value::Float(fa)), Some(majit_ir::Value::Float(fb))) =
        (ctx.trace_ctx.box_value(a), ctx.trace_ctx.box_value(b))
    {
        let bits = majit_metainterp::eval_binop_f(opcode, fa.to_bits() as i64, fb.to_bits() as i64);
        ctx.trace_ctx
            .set_opref_concrete(result, majit_ir::Value::Float(f64::from_bits(bits as u64)));
    }
    let dst = code[op.pc + 3] as usize;
    let len = ctx.registers_f.len();
    let slot = ctx
        .registers_f
        .get_mut(dst)
        .ok_or(DispatchError::RegisterOutOfRange {
            pc: op.pc,
            reg: dst,
            len,
            bank: "f",
        })?;
    *slot = result;
    Ok((DispatchOutcome::Continue, op.next_pc))
}

/// Generic float-bank unary handler. Operand layout `f>f` (1B src
/// + 1B dst). RPython equivalent: `bhimpl_float_neg(value)` →
/// `pyjitpl.py:execute(rop.FLOAT_NEG, value)`. Recording-only path
/// is the same shape as `binop_float_record` minus one read.
pub(crate) fn unop_float_record(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_>,
    opcode: OpCode,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    let a = read_float_reg(code, op, 0, ctx)?;
    let result = ctx.trace_ctx.record_op(opcode, &[a]);
    // Box(value) parity: stamp the unary float result (matches dispatch.rs
    // trace_unary_f — FloatNeg / FloatAbs).
    if let Some(majit_ir::Value::Float(fa)) = ctx.trace_ctx.box_value(a) {
        let bits = majit_metainterp::eval_unary_f(opcode, fa.to_bits() as i64);
        ctx.trace_ctx
            .set_opref_concrete(result, majit_ir::Value::Float(f64::from_bits(bits as u64)));
    }
    let dst = code[op.pc + 2] as usize;
    let len = ctx.registers_f.len();
    let slot = ctx
        .registers_f
        .get_mut(dst)
        .ok_or(DispatchError::RegisterOutOfRange {
            pc: op.pc,
            reg: dst,
            len,
            bank: "f",
        })?;
    *slot = result;
    Ok((DispatchOutcome::Continue, op.next_pc))
}

/// Opname → recorder table for the "regular" record helpers whose
/// dispatch arm is a uniform `HELPER(code, op, ctx, OpCode::VARIANT)` call.
/// `handle` consults `dispatch_regular_record` before its per-opname
/// `match`; a `Some` short-circuits it. The Rust analogue of the
/// `@arguments`-decorated exec-generated `opimpl_*` families in
/// `pyjitpl.py:279-368`, which collapse the opcode × register-bank cross
/// product into one metaprogrammed table rather than spelling out each arm.
macro_rules! regular_record_table {
    ( $( $helper:ident { $( $key:literal => $variant:ident, )+ } )* ) => {
        pub(crate) fn dispatch_regular_record(
            op: &DecodedOp,
            code: &[u8],
            ctx: &mut WalkContext<'_, '_>,
        ) -> Option<Result<(DispatchOutcome, usize), DispatchError>> {
            match op.key {
                $( $( $key => Some($helper(code, op, ctx, OpCode::$variant)), )+ )*
                _ => None,
            }
        }

        /// Every opname routed through `dispatch_regular_record`; the `tests`
        /// module asserts none of these still appear as an arm in `handle`'s
        /// `match`, so the table stays the sole dispatcher for them.
        #[cfg(test)]
        pub(crate) const REGULAR_RECORD_KEYS: &[&str] = &[ $( $( $key, )+ )* ];
    };
}

regular_record_table! {
    binop_int_record {
        "int_add/ii>i" => IntAdd,
        "int_sub/ii>i" => IntSub,
        "int_mul/ii>i" => IntMul,
        "int_and/ii>i" => IntAnd,
        "int_or/ii>i" => IntOr,
        "int_xor/ii>i" => IntXor,
        "int_lshift/ii>i" => IntLshift,
        "int_rshift/ii>i" => IntRshift,
        "int_eq/ii>i" => IntEq,
        "int_ne/ii>i" => IntNe,
        "int_lt/ii>i" => IntLt,
        "int_le/ii>i" => IntLe,
        "int_gt/ii>i" => IntGt,
        "int_ge/ii>i" => IntGe,
    }
    binop_float_record {
        "float_add/ff>f" => FloatAdd,
        "float_sub/ff>f" => FloatSub,
        "float_mul/ff>f" => FloatMul,
        "float_truediv/ff>f" => FloatTrueDiv,
    }
    unop_int_record {
        "int_neg/i>i" => IntNeg,
        "int_invert/i>i" => IntInvert,
        "int_same_as/i>i" => SameAsI,
        "int_is_true/i>i" => IntIsTrue,
    }
    unop_float_record {
        "float_neg/f>f" => FloatNeg,
    }
    binop_ref_to_int_record {
        "ptr_eq/rr>i" => PtrEq,
        "ptr_ne/rr>i" => PtrNe,
    }
}
