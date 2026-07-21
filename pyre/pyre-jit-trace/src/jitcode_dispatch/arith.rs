//! Arithmetic / comparison / cast / ptr-and-ref record helpers.
//!
//! **Parity:** trace-side recorders for the `pyjitpl.py:279-368`
//! exec-generated `opimpl_*` families; the `regular_record_table!` macro
//! below mirrors that generation rather than hand-spelling each arm.
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
/// RPython parity: `pyjitpl.py` exec-generated
/// `opimpl_int_BINOP(b1, b2): return self.execute(rop.<OPNUM>, b1,
/// b2)` + the trailing `>i` decorator that writes the result into
/// `registers_i[dst]`. Walker collapses execute+writeback into
/// `record_op + slot store`, which matches the recording-only side of
/// `execute`'s split (`pyjitpl.py:_record_helper`).
pub(crate) fn binop_int_record<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_, Sym>,
    opcode: OpCode,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    let a = read_int_reg(code, op, 0, ctx)?;
    let b = read_int_reg(code, op, 1, ctx)?;
    // This handler serves both generated loops: the plain binops (`int_add`,
    // `uint_lt`, …) and the `int_eq`..`int_ge` compares.  Only the latter
    // carry `if b1 is b2: return <const>`, so gate on membership rather than
    // on "looks like a comparison" — `uint_lt` and friends compare too and
    // must still record.
    if a == b {
        if let Some(folded) = fastpath_same_boxes(opcode) {
            let result = ctx.trace_ctx.const_int(folded);
            let dst = code[op.pc + 3] as usize;
            write_int_reg(ctx, op.pc, dst, result, ConcreteValue::Int(folded))?;
            return Ok((DispatchOutcome::Continue, op.next_pc));
        }
    }
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

/// `pyjitpl.py FASTPATHS_SAME_BOXES`, applied by the exec-generated
/// `opimpl_%s(b1, b2): if b1 is b2: return <const>` loop.
///
/// Membership is the whole point: the loop that carries this fast path spells
/// exactly `int_eq`, `int_ne`, `int_lt`, `int_le`, `int_gt`, `int_ge`,
/// `ptr_eq`, `ptr_ne`, `instance_ptr_eq`, `instance_ptr_ne`. The unsigned
/// compares (`uint_lt`/`uint_le`/`uint_gt`/`uint_ge`) sit in the *other*
/// generated loop and get no fast path, so this must not be widened to "any
/// comparison opcode".
///
/// `None` means the opcode is not a same-boxes fast-path member.
pub(crate) fn fastpath_same_boxes(opcode: OpCode) -> Option<i64> {
    match opcode {
        OpCode::IntEq | OpCode::IntLe | OpCode::IntGe => Some(1),
        OpCode::IntNe | OpCode::IntLt | OpCode::IntGt => Some(0),
        OpCode::PtrEq | OpCode::InstancePtrEq => Some(1),
        OpCode::PtrNe | OpCode::InstancePtrNe => Some(0),
        _ => None,
    }
}

pub(crate) fn record_int_cmp<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    opcode: OpCode,
    a: OpRef,
    b: OpRef,
) -> OpRef {
    if a == b {
        let folded = fastpath_same_boxes(opcode).unwrap_or_else(|| {
            unreachable!("record_int_cmp requires an integer comparison opcode")
        });
        return ctx.trace_ctx.const_int(folded);
    }
    if let (Some(Value::Int(la)), Some(Value::Int(rb))) =
        (a.inline_const_to_value(), b.inline_const_to_value())
    {
        return ctx
            .trace_ctx
            .const_int(majit_metainterp::eval_binop_i(opcode, la, rb));
    }
    let result = ctx.trace_ctx.record_op(opcode, &[a, b]);
    if let (Some(majit_ir::Value::Int(la)), Some(majit_ir::Value::Int(rb))) =
        (ctx.trace_ctx.box_value(a), ctx.trace_ctx.box_value(b))
    {
        let folded = majit_metainterp::eval_binop_i(opcode, la, rb);
        ctx.trace_ctx
            .set_opref_concrete(result, majit_ir::Value::Int(folded));
    }
    result
}

/// Record a one-operand integer op and its concrete result. The unary
/// counterpart of [`record_int_cmp`], for the fused `goto_if_not_int_is_*`
/// handlers whose condbox comes from `self.execute(rop.INT_IS_*, box)`.
pub(crate) fn record_int_unary<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    opcode: OpCode,
    a: OpRef,
) -> OpRef {
    let result = ctx.trace_ctx.record_op(opcode, &[a]);
    if let Some(majit_ir::Value::Int(la)) = ctx.trace_ctx.box_value(a) {
        let folded = majit_metainterp::eval_unary_i(opcode, la);
        ctx.trace_ctx
            .set_opref_concrete(result, majit_ir::Value::Int(folded));
    }
    result
}

/// Record an overflow-checking integer operation and its concrete result.
///
/// RPython parity: `pyjitpl.py opimpl_int_add_jump_if_ovf` records the
/// matching `INT_*_OVF`, while `pyjitpl.py handle_possible_overflow_error`
/// chooses the guard separately from the concrete overflow flag.
pub(crate) fn record_int_ovf<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    pc: usize,
    opcode: OpCode,
    b1: OpRef,
    b2: OpRef,
) -> Result<(OpRef, bool), DispatchError> {
    let v1 = match ctx.trace_ctx.concrete_of_opref(b1) {
        Some(Value::Int(value)) => value,
        _ => {
            return Err(DispatchError::IntOvfOperandNotConcrete { pc, value: b1 });
        }
    };
    let v2 = match ctx.trace_ctx.concrete_of_opref(b2) {
        Some(Value::Int(value)) => value,
        _ => {
            return Err(DispatchError::IntOvfOperandNotConcrete { pc, value: b2 });
        }
    };
    let (wrapping_result, overflow) = match opcode {
        OpCode::IntAddOvf => (v1.wrapping_add(v2), v1.checked_add(v2).is_none()),
        OpCode::IntSubOvf => (v1.wrapping_sub(v2), v1.checked_sub(v2).is_none()),
        OpCode::IntMulOvf => (v1.wrapping_mul(v2), v1.checked_mul(v2).is_none()),
        _ => unreachable!("record_int_ovf requires an IntAddOvf/IntSubOvf/IntMulOvf opcode"),
    };
    if b1.is_constant() && b2.is_constant() {
        return Ok((ctx.trace_ctx.const_int(wrapping_result), overflow));
    }
    let resbox = ctx.trace_ctx.record_op(opcode, &[b1, b2]);
    ctx.trace_ctx
        .set_opref_concrete(resbox, Value::Int(wrapping_result));
    Ok((resbox, overflow))
}

/// RPython `pyjitpl.py opimpl_int_between`:
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
/// (`blackhole.py bhimpl_int_between(a, b, c): return a <= b
/// < c`) are preserved through the same decomposition.
///
/// Operand layout `iii>i`: 3B sources + 1B dst (=4 operand bytes after
/// the opcode).  Concrete-value propagation in [`execute_pure_binop_i`]
/// runs in two layers: all-inline-Const operand pairs fold to a
/// `const_int(...)` OpRef without recording (matching upstream
/// `_all_constants` short-circuit at `pyjitpl.py`); the
/// trailing concrete-tracked-pair path additionally stamps the recorded
/// op via `set_opref_concrete`.  The `ConstInt(1)` fast path at
/// `pyjitpl.py` keys on the inline-Const layer through
/// `inline_const_to_value`, mirroring `isinstance(b5, ConstInt)` —
/// box_value's concrete-stamp layer does not participate in that
/// branch decision.
pub(crate) fn int_between_record<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_, Sym>,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    let b1 = read_int_reg(code, op, 0, ctx)?;
    let b2 = read_int_reg(code, op, 1, ctx)?;
    let b3 = read_int_reg(code, op, 2, ctx)?;

    // b5 = execute(INT_SUB, b3, b1)
    let b5 = execute_pure_binop_i(ctx, OpCode::IntSub, b3, b1);

    // pyjitpl.py `if isinstance(b5, ConstInt) and b5.getint() == 1`
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

/// `pyjitpl.py execute_and_record` for pure integer binops.
/// When both operands are inline-Const, fold to a `ConstInt` OpRef
/// (no `record_op` call); otherwise record the op and stamp the
/// observed concrete value if both sides have one.
///
/// PyPy's `execute_and_record` short-circuits `_record_helper` via
/// `executor.wrap_constant(resvalue)` when `_all_constants(*argboxes)`
/// holds for a pure opcode.  Mirroring that here keeps the trace
/// free of all-constant subexpressions exactly where upstream keeps
/// it free — `opimpl_int_between` chains three such pure binops.
pub(crate) fn execute_pure_binop_i<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
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
/// dst). RPython parity: `pyjitpl.py` exec-generated
/// `opimpl_int_<unary>` (int_neg / int_invert / int_is_zero etc.) +
/// the `>i` decorator's writeback. Walker reads `registers_i[src]`,
/// records `OpCode::<Variant>` with `[a]`, writes the recorder result
/// into `registers_i[dst]`.
pub(crate) fn unop_int_record<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_, Sym>,
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
/// `pyjitpl.py` exec-generated `opimpl_ptr_eq` /
/// `opimpl_ptr_ne` (and instance variants) follow `self.execute(rop.<OPNUM>,
/// b1, b2)` — both `b1`/`b2` are ref boxes, result is an int box. The
/// `b1 is b2` fast path is omitted (same rationale as `binop_int_record`'s
/// comparison family — pyre's recorder shares constants by value).
pub(crate) fn binop_ref_to_int_record<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_, Sym>,
    opcode: OpCode,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    let a = read_ref_reg(code, op, 0, ctx)?;
    let b = read_ref_reg(code, op, 1, ctx)?;
    // `if b1 is b2: return <const>` — all four ref compares this handler
    // serves are `FASTPATHS_SAME_BOXES` members, so an identical operand pair
    // answers without recording.
    if a == b {
        let folded = fastpath_same_boxes(opcode).unwrap_or_else(|| {
            unreachable!("binop_ref_to_int_record: unsupported opcode {opcode:?}")
        });
        let result = ctx.trace_ctx.const_int(folded);
        let dst = code[op.pc + 3] as usize;
        write_int_reg(ctx, op.pc, dst, result, ConcreteValue::Int(folded))?;
        return Ok((DispatchOutcome::Continue, op.next_pc));
    }
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

pub(crate) fn record_ptr_cmp<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    opcode: OpCode,
    a: OpRef,
    b: OpRef,
    a_concrete: ConcreteValue,
    b_concrete: ConcreteValue,
) -> OpRef {
    if a == b {
        let folded = fastpath_same_boxes(opcode)
            .unwrap_or_else(|| unreachable!("record_ptr_cmp requires PtrEq or PtrNe"));
        return ctx.trace_ctx.const_int(folded);
    }
    if let (Some(Value::Ref(la)), Some(Value::Ref(rb))) =
        (a.inline_const_to_value(), b.inline_const_to_value())
    {
        return ctx.trace_ctx.const_int(match opcode {
            OpCode::PtrEq => (la == rb) as i64,
            OpCode::PtrNe => (la != rb) as i64,
            _ => unreachable!("record_ptr_cmp requires PtrEq or PtrNe"),
        });
    }
    let result = ctx.trace_ctx.record_op(opcode, &[a, b]);
    let folded = if let (Some(majit_ir::Value::Ref(la)), Some(majit_ir::Value::Ref(rb))) =
        (ctx.trace_ctx.box_value(a), ctx.trace_ctx.box_value(b))
    {
        Some(match opcode {
            OpCode::PtrEq => (la == rb) as i64,
            OpCode::PtrNe => (la != rb) as i64,
            _ => panic!("record_ptr_cmp: unsupported opcode {opcode:?}"),
        })
    } else if let (ConcreteValue::Ref(la), ConcreteValue::Ref(rb)) = (a_concrete, b_concrete) {
        Some(match opcode {
            OpCode::PtrEq => (la == rb) as i64,
            OpCode::PtrNe => (la != rb) as i64,
            _ => panic!("record_ptr_cmp: unsupported opcode {opcode:?}"),
        })
    } else {
        None
    };
    if let Some(folded) = folded {
        ctx.trace_ctx
            .set_opref_concrete(result, majit_ir::Value::Int(folded));
    }
    result
}

/// `ptr_nonzero/r>i` (`nonzero = true`) and `ptr_iszero/r>i`
/// (`nonzero = false`) handler (operand layout `r>i`: 1B r-src + 1B i-dst).
///
/// RPython parity:
/// `pyjitpl.py opimpl_ptr_nonzero(box)`:
/// ```python
/// @arguments("box")
/// def opimpl_ptr_nonzero(self, box):
///     return self.execute(rop.PTR_NE, box, CONST_NULL)
/// ```
/// `opimpl_ptr_iszero` is the `PTR_EQ` complement.
///
/// Walker reads one `r` reg, records `OpCode::PtrNe`/`PtrEq` with
/// `[box, CONST_NULL]` (via `trace_ctx.const_null()` —
/// `history.py CONST_NULL = ConstPtr(ConstPtr.value)`), and writes
/// the recorder result into `registers_i[dst]`.  RPython does the
/// same `b1 is b2` short-circuit at `pyjitpl.py` for
/// `opimpl_ptr_eq` but the nullity test against `CONST_NULL` cannot
/// short-circuit because `box` is never the literal `CONST_NULL`
/// constant (codewriter would have folded that).
pub(crate) fn ptr_nullity_record<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_, Sym>,
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

/// `int_guard_value/i`, `ref_guard_value/r`, and `float_guard_value/f`
/// handler (operand layout is one 1B bank-specific source, no dst).
///
/// RPython parity: `pyjitpl.py _opimpl_guard_value` →
/// `pyjitpl.py implement_guard_value`:
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
/// All three banks share this body exactly as `pyjitpl.py`
/// aliases `_opimpl_guard_value` to the Int, Ref, and Float opimpl names.
///
/// Walker behaviour:
///   * Read the 1B bank-specific operand and its concrete value.
///   * If the symbolic OpRef is already a Const, skip (Const arm of
///     `implement_guard_value`).
///   * If the concrete value is null or has the wrong bank variant,
///     skip — the walker doesn't have a runtime value to mint the
///     expected constant from.  This is the strictest mode (sibling
///     `dispatch_switch_id` line 1207 falls into the same skip-guard
///     branch when `valuebox.is_constant()`).
///   * Otherwise mint the bank-specific constant (executor.py
///     `constant_from_op`), emit `GuardValue` with `[value, expected]`,
///     and call `replace_box(value, expected)` (pyjitpl.py).  Also
///     rewrite every slot in the selected register bank still pointing
///     at `value` to `expected`,
///     matching `dispatch_switch_id:1198-1202`.
///
/// The `0` in `record_guard(..., 0)` is only `record_guard`'s `num_live`
/// argument; a full production resume snapshot is attached on the next line
/// via `walker_capture_snapshot_for_last_guard`.
#[derive(Clone, Copy)]
pub(crate) enum GuardValueBank {
    Int,
    Ref,
    Float,
}

pub(crate) fn guard_value_record<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_, Sym>,
    bank: GuardValueBank,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    let value = match bank {
        GuardValueBank::Int => read_int_reg(code, op, 0, ctx)?,
        GuardValueBank::Ref => read_ref_reg(code, op, 0, ctx)?,
        GuardValueBank::Float => read_float_reg(code, op, 0, ctx)?,
    };
    if value.is_constant() {
        return Ok((DispatchOutcome::Continue, op.next_pc));
    }
    let concrete = match bank {
        GuardValueBank::Int => read_int_reg_concrete(code, op, 0, ctx),
        GuardValueBank::Ref => read_ref_reg_concrete(code, op, 0, ctx),
        GuardValueBank::Float => read_float_reg_concrete(code, op, 0, ctx),
    };
    let expected = match (bank, concrete) {
        (GuardValueBank::Int, ConcreteValue::Int(v)) => ctx.trace_ctx.const_int(v),
        (GuardValueBank::Ref, ConcreteValue::Ref(ptr)) if !ptr.is_null() => {
            ctx.trace_ctx.const_ref(ptr as usize as i64)
        }
        (GuardValueBank::Float, ConcreteValue::Float(f)) => {
            ctx.trace_ctx.const_float(f.to_bits() as i64)
        }
        _ => return Ok((DispatchOutcome::Continue, op.next_pc)),
    };
    ctx.trace_ctx
        .record_guard(OpCode::GuardValue, &[value, expected], 0);
    walker_capture_snapshot_for_last_guard(ctx, op.pc)?;
    ctx.trace_ctx.replace_box(value, expected);
    match bank {
        GuardValueBank::Int => {
            for slot in ctx.registers_i.iter_mut() {
                if *slot == value {
                    *slot = expected;
                }
            }
        }
        GuardValueBank::Ref => {
            for slot in ctx.registers_r.iter_mut() {
                if *slot == value {
                    *slot = expected;
                }
            }
        }
        GuardValueBank::Float => {
            for slot in ctx.registers_f.iter_mut() {
                if *slot == value {
                    *slot = expected;
                }
            }
        }
    }
    Ok((DispatchOutcome::Continue, op.next_pc))
}

/// Generic float-pair-to-int handler for `float_<cmp>/ff>i` (operand
/// layout `ff>i`: 1B f-src + 1B f-src + 1B i-dst).  RPython parity:
/// `bhimpl_float_{lt,le,eq,ne,gt,ge}` (`blackhole.py`) — read
/// two `f` regs, record `OpCode::Float<Cmp>`, write the recorder
/// result into `registers_i[dst]`.
pub(crate) fn binop_float_to_int_record<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_, Sym>,
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

pub(crate) fn record_float_cmp<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    opcode: OpCode,
    a: OpRef,
    b: OpRef,
) -> OpRef {
    // Mirrors `self.execute(rop.<CMP>, b1, b2)` but does not pre-fold
    // `_all_constants` / `b1 is b2` (`pyjitpl.py`);
    // the recorded compare and downstream guard are optimizer-folded/strengthened.
    let result = ctx.trace_ctx.record_op(opcode, &[a, b]);
    if let (Some(majit_ir::Value::Float(fa)), Some(majit_ir::Value::Float(fb))) =
        (ctx.trace_ctx.box_value(a), ctx.trace_ctx.box_value(b))
    {
        let folded =
            majit_metainterp::eval_float_cmp(opcode, fa.to_bits() as i64, fb.to_bits() as i64);
        ctx.trace_ctx
            .set_opref_concrete(result, majit_ir::Value::Int(folded));
    }
    result
}

/// Bank-crossing unary cast family from the `pyjitpl.py`
/// exec-generated unary loop (`cast_int_to_float` / `cast_int_to_ptr`
/// / `cast_ptr_to_int`). PyPy generates these from one template
/// because its boxes are untyped; pyre's typed register banks make
/// each cast a distinct (src-bank, dst-bank, concrete-fold) triple, so
/// the recorded `opcode` selects the shape. Operand layout `<s>><d>`
/// (1B src + 1B dst); the recorded result lands in the destination
/// bank per the trailing `>X` decorator.
pub(crate) fn unop_cast_record<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_, Sym>,
    opcode: OpCode,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    let dst = code[op.pc + 2] as usize;
    match opcode {
        // `cast_int_to_float/i>f`: Int-bank → Float-bank. Stamp the
        // result with the operand's Box.value as an f64 so downstream
        // `box_value(result)` callers see the live value.
        OpCode::CastIntToFloat => {
            let a = read_int_reg(code, op, 0, ctx)?;
            let result = ctx.trace_ctx.record_op(opcode, &[a]);
            if let Some(majit_ir::Value::Int(n)) = ctx.trace_ctx.box_value(a) {
                ctx.trace_ctx
                    .set_opref_concrete(result, majit_ir::Value::Float(n as f64));
            }
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
        }
        // `cast_int_to_ptr/i>r`: Int-bank → Ref-bank. Bit-cast the
        // operand's Box.value (`BoxInt(n)` → `BoxRef(n as ptr)`).
        OpCode::CastIntToPtr => {
            let a = read_int_reg(code, op, 0, ctx)?;
            let result = ctx.trace_ctx.record_op(opcode, &[a]);
            if let Some(majit_ir::Value::Int(n)) = ctx.trace_ctx.box_value(a) {
                ctx.trace_ctx
                    .set_opref_concrete(result, majit_ir::Value::Ref(majit_ir::GcRef(n as usize)));
            }
            write_ref_reg(ctx, op.pc, dst, result, ConcreteValue::Null)?;
        }
        // `cast_ptr_to_int/r>i`: Ref-bank → Int-bank. Bit-cast the
        // operand's Box.value (`BoxRef(p)` → `BoxInt(p as i64)`).
        OpCode::CastPtrToInt => {
            let a = read_ref_reg(code, op, 0, ctx)?;
            let result = ctx.trace_ctx.record_op(opcode, &[a]);
            if let Some(majit_ir::Value::Ref(r)) = ctx.trace_ctx.box_value(a) {
                ctx.trace_ctx
                    .set_opref_concrete(result, majit_ir::Value::Int(r.0 as i64));
            }
            let concrete_for_shadow = concrete_from_recorded_opref(ctx, result);
            write_int_reg(ctx, op.pc, dst, result, concrete_for_shadow)?;
        }
        _ => panic!("unop_cast_record: unsupported opcode {opcode:?}"),
    }
    Ok((DispatchOutcome::Continue, op.next_pc))
}

/// Generic float-bank binop handler. Operand layout `ff>f` (1B src1
/// + 1B src2 + 1B dst). RPython parity: same as `binop_int_record`
/// but on the float bank — `pyjitpl.py`'s exec-generated
/// `opimpl_float_<binop>` reads two `f` regs, calls
/// `self.execute(rop.<OPNUM>, b1, b2)`, and the trailing `>f`
/// decorator writes the result into `registers_f[dst]`.
pub(crate) fn binop_float_record<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_, Sym>,
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
pub(crate) fn unop_float_record<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_, Sym>,
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
/// `pyjitpl.py`, which collapse the opcode × register-bank cross
/// product into one metaprogrammed table rather than spelling out each arm.
macro_rules! regular_record_table {
    ( $( $helper:ident { $( $key:literal => $variant:ident, )+ } )* ) => {
        pub(crate) fn dispatch_regular_record<Sym: WalkSym>(
            op: &DecodedOp,
            code: &[u8],
            ctx: &mut WalkContext<'_, '_, Sym>,
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
    // Binary arithmetic `int_*`, from the `pyjitpl.py`
    // exec-generated `(box, box)` opimpl loop: read two `i`-coded reg
    // operands, `self.execute(rop.<OPNUM>, b1, b2)`. Operand layout
    // `ii>i` (1B src1 + 1B src2 + 1B dst). No MIFrame state — pure
    // arithmetic, EffectInfo-free / heapcache-free. `int_lshift/ii>i` is
    // the canonical shape (`bhimpl_int_lshift`); a mixed `int_lshift/ri>i`
    // stays unwired because a Ref register flowing into an Int op is a
    // kind-flow bug, not a shape to handle. The `int_eq..int_ge`
    // comparisons carry a `b1 is b2` fast path in `pyjitpl.py`;
    // the walker omits it (two distinct OpRefs record correctly and the
    // optimizer collapses tautological compares) since the recorder
    // shares constants by value rather than allocating a `ConstInt`.
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
        // The same generated loop also spells the unsigned members. They
        // share the `ii>i` shape and differ only in how the recorded
        // opcode reinterprets the operands, so they need no separate
        // helper — `eval_binop_i` already folds each with u64 semantics.
        "uint_rshift/ii>i" => UintRshift,
        "uint_mul_high/ii>i" => UintMulHigh,
        "uint_lt/ii>i" => UintLt,
        "uint_le/ii>i" => UintLe,
        "uint_gt/ii>i" => UintGt,
        "uint_ge/ii>i" => UintGe,
    }
    // Float arithmetic — same `pyjitpl.py` loop on the `f` bank.
    // Codewriter today emits only float_add/float_sub/float_truediv
    // (float_mul appears only when an explicit `*` operand reaches the
    // codewriter; the bench set has none yet).
    binop_float_record {
        "float_add/ff>f" => FloatAdd,
        "float_sub/ff>f" => FloatSub,
        "float_mul/ff>f" => FloatMul,
        "float_truediv/ff>f" => FloatTrueDiv,
    }
    // Float comparisons `bhimpl_float_{lt,le,eq,ne,gt,ge}`
    // (`blackhole.py`) — part of the same `pyjitpl.py`
    // generated loop, but the recorder result lands in the int bank
    // (`ff>i`).
    binop_float_to_int_record {
        "float_lt/ff>i" => FloatLt,
        "float_le/ff>i" => FloatLe,
        "float_eq/ff>i" => FloatEq,
        "float_ne/ff>i" => FloatNe,
        "float_gt/ff>i" => FloatGt,
        "float_ge/ff>i" => FloatGe,
    }
    // Int-bank unary ops: `pyjitpl.py` (int_neg / int_invert) +
    // 371-375 (int_same_as, which records `rop.SAME_AS_I` via
    // `_record_helper` — same shape, treated as a regular
    // record-and-writeback). `int_is_true` (`pyjitpl.py`) is
    // Int-typed on the bank though semantically bool (matches the `>i`
    // destination shape).
    unop_int_record {
        "int_neg/i>i" => IntNeg,
        "int_invert/i>i" => IntInvert,
        "int_same_as/i>i" => SameAsI,
        "int_is_true/i>i" => IntIsTrue,
    }
    unop_float_record {
        "float_neg/f>f" => FloatNeg,
        // `float_abs` sits beside `float_neg` in the generated unary loop
        // and `eval_unary_f` already folds it.
        "float_abs/f>f" => FloatAbs,
    }
    // Bank-crossing unary casts from the same `pyjitpl.py`
    // generated unary loop; `unop_cast_record` selects the src/dst bank
    // shape from the opcode (pyre's typed banks cannot share one template
    // the way PyPy's untyped boxes do).
    unop_cast_record {
        "cast_int_to_float/i>f" => CastIntToFloat,
        "cast_int_to_ptr/i>r" => CastIntToPtr,
        "cast_ptr_to_int/r>i" => CastPtrToInt,
    }
    // `ptr_eq` / `ptr_ne` (`pyjitpl.py`): Ref operands, int result. The
    // `instance_ptr_*` pair shares that generated compare loop —
    // `jtransform.py` rewrites a pointer comparison whose operands are
    // known instances into the `instance_ptr_*` spelling, which carries
    // the same operand shape and the same fold.
    binop_ref_to_int_record {
        "ptr_eq/rr>i" => PtrEq,
        "ptr_ne/rr>i" => PtrNe,
        "instance_ptr_eq/rr>i" => InstancePtrEq,
        "instance_ptr_ne/rr>i" => InstancePtrNe,
    }
}
