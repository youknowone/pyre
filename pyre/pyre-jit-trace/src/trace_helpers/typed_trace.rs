//! High-level typed trace operations (the `generated_*` functions) — the
//! analog of `pyjitpl.py`'s `opimpl_*` and `listobject.py` strategies.
//! Compose the primitives into complete guard→unbox→op→box sequences.

use super::*;
use pyre_interpreter::bytecode::{BinaryOperator, ComparisonOperator};

/// Trace a binary int operation: unbox → op → guard_ovf → box.
///
/// RPython jitcode parity: guard_class + getfield_gc_i (per operand),
/// then int_OP_ovf + guard_no_overflow (or int_OP), then
/// new_with_vtable + setfield_gc for boxing.
///
/// Returns None if the operation is not handled as an int operation
/// (unsupported op or concrete validation fails → caller should
/// fall back to residual trace_binary_value).
#[inline]
pub fn generated_binary_int_value(
    frame: &mut crate::state::MIFrame,
    ctx: &mut majit_metainterp::TraceCtx,
    a: majit_ir::OpRef,
    b: majit_ir::OpRef,
    op: BinaryOperator,
    concrete_lhs: pyre_object::PyObjectRef,
    concrete_rhs: pyre_object::PyObjectRef,
) -> Option<majit_ir::OpRef> {
    use majit_ir::OpCode;

    // Table lookup: BinaryOperator → (OpCode, has_overflow, needs_concrete_check)
    let (op_code, has_overflow, needs_concrete_check) = int_binop_lookup(op)?;

    // Concrete value extraction for range validation.
    let concrete = unsafe {
        if concrete_lhs.is_null()
            || concrete_rhs.is_null()
            || !pyre_object::is_int(concrete_lhs)
            || !pyre_object::is_int(concrete_rhs)
        {
            None
        } else {
            Some((
                pyre_object::w_int_get_value(concrete_lhs),
                pyre_object::w_int_get_value(concrete_rhs),
            ))
        }
    };

    // RPython parity: int_binop fast path only applies when both operands
    // are W_IntObject at trace time. When either is W_LongObject (or null),
    // fall back to the residual trace_binary_value path which calls the
    // Python-level __add__/__sub__/... method (no guard_class emitted).
    if concrete.is_none() {
        return None;
    }

    // boolobject.py:74-76 descr_and/or/xor: when both operands are bool the
    // result is a bool (`space.newbool`), not an int.  The bitwise op runs on
    // the shared `intval` exactly as for ints — only the boxing differs, so
    // note it here and pick the bool boxing below.  Mixed bool/int bitwise
    // yields an int and boxes as int.
    let result_is_bool = matches!(op_code, OpCode::IntAnd | OpCode::IntOr | OpCode::IntXor)
        && unsafe { pyre_object::is_bool(concrete_lhs) && pyre_object::is_bool(concrete_rhs) };

    // intobject.py range validation for FloorDiv/Mod/Shift.
    // For FloorDiv/Mod the RPython-orthodox preconditions are
    // `rhs != 0` AND `not (lhs == i64::MIN && rhs == -1)` — PyPy
    // `intobject.py:316 _floordiv` / `:341 _mod` wrap
    // `ovfcheck(x // y)` / `ovfcheck(x % y)` in `try / except
    // ZeroDivisionError`, letting `OverflowError` propagate up to
    // `_make_descr_binop:820-823` where it routes to `ovf2long`
    // (long fallback). RPython's `_handle_int_special`
    // (jtransform.py:2042) emits the `int.py_div` / `int.py_mod`
    // oopspec call as `EF_ELIDABLE_CANNOT_RAISE`; the inlined
    // `_ovf_zer` wrapper (`rint.py:429 ll_int_py_div_ovf_zer` /
    // `:520 ll_int_py_mod_ovf_zer`) contributes the explicit
    // `int_eq(rhs, 0) -> guard_false` plus the
    // `(lhs == INT_MIN) & (rhs == -1) -> guard_false` overflow
    // check ahead of the call. Pyre keeps a trace-time short-circuit
    // on both cases AND emits the matching runtime guards so a
    // re-used trace bails out before invoking the helper with bad
    // operands (the helper uses `wrapping_div` / `wrapping_rem` and
    // would silently return `INT_MIN` / a wrap value otherwise).
    if needs_concrete_check {
        match op_code {
            OpCode::IntFloorDiv | OpCode::IntMod => {
                let (lhs, rhs) = concrete?;
                if rhs == 0 || (lhs == i64::MIN && rhs == -1) {
                    return None;
                }
            }
            OpCode::IntLshift => {
                let (lhs, rhs) = concrete?;
                let shift = u32::try_from(rhs).ok()?;
                if shift >= i64::BITS {
                    return None;
                }
                // intobject.py:207 ovfcheck(a << b)
                let result = lhs.wrapping_shl(shift);
                if result.wrapping_shr(shift) != lhs {
                    return None;
                }
            }
            OpCode::IntRshift => {
                let (lhs, rhs) = concrete?;
                let shift = u32::try_from(rhs).ok()?;
                if shift >= i64::BITS {
                    // intobject.py:229-231: large shift → 0 or -1
                    let result = if lhs < 0 { -1i64 } else { 0i64 };
                    let raw = ctx.const_int(result);
                    let boxed = crate::state::wrapint(ctx, raw);
                    return Some(boxed);
                }
            }
            _ => {}
        }
    }

    // RPython jitcode: guard_class + getfield_gc_i per operand.  bool and
    // int share the `intval` field; guard each operand against its own
    // vtable (BOOL_TYPE / INT_TYPE) so a bool unboxes through its own class.
    // Skip unbox if value is already raw int (Type::Int).
    let lhs_raw = if frame.value_type(a) == majit_ir::Type::Int {
        a
    } else {
        let (type_addr, descr) = crate::state::int_or_bool_unbox_type_descr(concrete_lhs);
        crate::state::trace_unbox_int_with_resume_descr(frame, a, type_addr, descr)
    };
    let rhs_raw = if frame.value_type(b) == majit_ir::Type::Int {
        b
    } else {
        let (type_addr, descr) = crate::state::int_or_bool_unbox_type_descr(concrete_rhs);
        crate::state::trace_unbox_int_with_resume_descr(frame, b, type_addr, descr)
    };

    // The inlined `_ovf_zer` wrapper (`rint.py:429 ll_int_py_div_
    // ovf_zer` / `:520 ll_int_py_mod_ovf_zer`) contributes two
    // explicit guards to the trace ahead of the `int.py_div` /
    // `int.py_mod` residual call:
    //   1. `int_eq(rhs, 0) -> guard_false` — from the `_zer` body.
    //   2. `int_and(int_eq(lhs, INT_MIN), int_eq(rhs, -1)) ->
    //      guard_false` — from the `_ovf` body
    //      (`if (x == -sys.maxint - 1) & (y == -1): raise
    //      OverflowError`).
    // With `EF_ELIDABLE_CANNOT_RAISE` the trace has no
    // `GUARD_NO_EXCEPTION` for this call, so these guards are the
    // only barrier between a re-used trace and an unsafe
    // `ll_int_py_div` / `ll_int_py_mod` invocation: the
    // helpers are now `wrapping_div` / `wrapping_rem` precondition
    // wrappers (`blackhole.rs:5785/5802`), so a zero divisor is
    // undefined behaviour in release and panics in debug, and
    // `INT_MIN / -1` wraps to `INT_MIN` instead of routing to PyPy's
    // `ovf2long` long-fallback (`intobject.py:491`). Negative
    // operands generally are valid — PyPy `intobject.py:316/341`
    // accepts them and the helper's no-branch correction handles
    // every sign combination — so no sign-only guard is emitted
    // here beyond the overflow corner.
    if matches!(op_code, OpCode::IntFloorDiv | OpCode::IntMod) {
        let (lhs_val, rhs_val) = concrete.expect("concrete non-None enforced above");
        let zero_const = ctx.const_int(0);
        let rhs_zero = ctx.record_op(OpCode::IntEq, &[rhs_raw, zero_const]);
        ctx.set_opref_concrete(rhs_zero, majit_ir::Value::Int((rhs_val == 0) as i64));
        frame.generate_guard(ctx, OpCode::GuardFalse, &[rhs_zero]);
        let int_min_const = ctx.const_int(i64::MIN);
        let neg_one_const = ctx.const_int(-1);
        let lhs_is_min = ctx.record_op(OpCode::IntEq, &[lhs_raw, int_min_const]);
        ctx.set_opref_concrete(
            lhs_is_min,
            majit_ir::Value::Int((lhs_val == i64::MIN) as i64),
        );
        let rhs_is_neg_one = ctx.record_op(OpCode::IntEq, &[rhs_raw, neg_one_const]);
        ctx.set_opref_concrete(rhs_is_neg_one, majit_ir::Value::Int((rhs_val == -1) as i64));
        let ovf_both = ctx.record_op(OpCode::IntAnd, &[lhs_is_min, rhs_is_neg_one]);
        let ovf_concrete = ((lhs_val == i64::MIN) as i64) & ((rhs_val == -1) as i64);
        ctx.set_opref_concrete(ovf_both, majit_ir::Value::Int(ovf_concrete));
        frame.generate_guard(ctx, OpCode::GuardFalse, &[ovf_both]);
    }

    // RPython jtransform.py:576-577 `rewrite_op_int_floordiv = _do_builtin_call`
    // / `rewrite_op_int_mod = _do_builtin_call`: replace the bare primitive
    // with an `OS_INT_PY_DIV` / `OS_INT_PY_MOD`-tagged residual call so the
    // optimizer's `optimize_call_int_py_div` / `optimize_call_int_py_mod`
    // (rewrite.rs:1848 / 1788) specialize power-of-2 divisors → IntRshift,
    // const 1 → identity, const -1 → IntNeg, etc. The call is routed
    // through `call_typed_with_effect_pure` so the trace records
    // `CallI` first then patches via `record_result_of_call_pure`
    // (pyjitpl.py:1947 / 3553-3579) — populates `call_pure_results`
    // for cross-trace constant folding and reduces all-const
    // (lhs, rhs) to a `Const` directly. Other binops keep the
    // bare-primitive emission since RPython has matching `bhimpl_int_*`
    // primitives at blackhole.py.
    let (raw_result, concrete_result_value) = match op_code {
        OpCode::IntFloorDiv => {
            let (lhs_val, rhs_val) = concrete.expect("IntFloorDiv concrete check passed above");
            let func_ptr = majit_metainterp::blackhole::ll_int_py_div as *const ();
            // The runtime guards above ensure `rhs != 0` and
            // `not (lhs == INT_MIN && rhs == -1)` for the recorded
            // trace; safe to invoke the helper concretely here.
            let concrete_result = majit_metainterp::blackhole::ll_int_py_div(lhs_val, rhs_val);
            let r = ctx.call_typed_with_effect_pure(
                OpCode::CallI,
                func_ptr,
                &[lhs_raw, rhs_raw],
                &[majit_ir::Type::Int, majit_ir::Type::Int],
                majit_ir::Type::Int,
                majit_metainterp::INT_PY_DIV_EFFECT_INFO,
                &[
                    majit_ir::Value::Int(func_ptr as usize as i64),
                    majit_ir::Value::Int(lhs_val),
                    majit_ir::Value::Int(rhs_val),
                ],
                majit_ir::Value::Int(concrete_result),
            );
            (r, concrete_result)
        }
        OpCode::IntMod => {
            let (lhs_val, rhs_val) = concrete.expect("IntMod concrete check passed above");
            let func_ptr = majit_metainterp::blackhole::ll_int_py_mod as *const ();
            let concrete_result = majit_metainterp::blackhole::ll_int_py_mod(lhs_val, rhs_val);
            let r = ctx.call_typed_with_effect_pure(
                OpCode::CallI,
                func_ptr,
                &[lhs_raw, rhs_raw],
                &[majit_ir::Type::Int, majit_ir::Type::Int],
                majit_ir::Type::Int,
                majit_metainterp::INT_PY_MOD_EFFECT_INFO,
                &[
                    majit_ir::Value::Int(func_ptr as usize as i64),
                    majit_ir::Value::Int(lhs_val),
                    majit_ir::Value::Int(rhs_val),
                ],
                majit_ir::Value::Int(concrete_result),
            );
            (r, concrete_result)
        }
        _ => {
            let r = ctx.record_op(op_code, &[lhs_raw, rhs_raw]);
            let (lhs_val, rhs_val) = concrete.expect("concrete non-None enforced above");
            (r, majit_metainterp::eval_binop_i(op_code, lhs_val, rhs_val))
        }
    };
    // Box(value) parity: stamp the result OpRef with its runtime concrete
    // so downstream `box_value(opref)` consumers see the value (matches
    // BoxInt(value) carrier in execute()).
    ctx.set_opref_concrete(raw_result, majit_ir::Value::Int(concrete_result_value));
    if has_overflow {
        frame.generate_guard(ctx, OpCode::GuardNoOverflow, &[]);
    }

    // RPython jitcode: wrapint → new_with_vtable + setfield_gc.  A both-bool
    // bitwise result is boxed via `space.newbool` (boolobject.py:74-76) so it
    // keeps the bool type; the 0/1 raw is already a truth value.
    let boxed = if result_is_bool {
        crate::helpers::emit_trace_bool_value_from_truth(ctx, raw_result, false)
    } else {
        crate::state::wrapint(ctx, raw_result)
    };
    Some(boxed)
}

/// Trace a binary float operation: unbox/cast → op → box.
///
/// RPython jitcode parity: guard_class + getfield_gc_f (or
/// getfield_gc_i + cast_int_to_float for int operands), then
/// float_OP, then new_with_vtable + setfield_gc.
///
/// Returns None if the operation is not handled as a float operation
/// → caller should fall back to residual trace_binary_value.
#[inline]
pub fn generated_binary_float_value(
    frame: &mut crate::state::MIFrame,
    ctx: &mut majit_metainterp::TraceCtx,
    a: majit_ir::OpRef,
    b: majit_ir::OpRef,
    op: BinaryOperator,
    concrete_lhs: pyre_object::PyObjectRef,
    concrete_rhs: pyre_object::PyObjectRef,
) -> Option<majit_ir::OpRef> {
    use majit_ir::OpCode;

    let is_power = matches!(op, BinaryOperator::Power | BinaryOperator::InplacePower);
    let op_code = float_binop_lookup(op);

    // resoperation.py: no FLOAT_POW / FLOAT_FLOORDIV / FLOAT_MOD opcodes.
    // FloorDivide/Remainder → residual.
    // Power → call_may_force (ll_math_pow, EF_CAN_RAISE).
    if op_code.is_none() && !is_power {
        return None;
    }

    // Determine which operands are int (need int→float cast) vs float.
    // RPython: float_add etc. call space.float_w() which dispatches on
    // type — int objects go through int2float (CastIntToFloat).
    let lhs_is_int = (!concrete_lhs.is_null() && unsafe { pyre_object::is_int(concrete_lhs) })
        || frame.value_type(a) == majit_ir::Type::Int;
    let rhs_is_int = (!concrete_rhs.is_null() && unsafe { pyre_object::is_int(concrete_rhs) })
        || frame.value_type(b) == majit_ir::Type::Int;

    // floatobject.py _to_float: accepts int, float, long.
    // Long → residual fallback (long→float can lose precision).
    let lhs_is_long =
        !concrete_lhs.is_null() && unsafe { pyre_object::is_long(concrete_lhs) } && !lhs_is_int;
    let rhs_is_long =
        !concrete_rhs.is_null() && unsafe { pyre_object::is_long(concrete_rhs) } && !rhs_is_int;
    if lhs_is_long || rhs_is_long {
        return None;
    }

    let float_type_addr = &pyre_object::pyobject::FLOAT_TYPE as *const _ as i64;

    // Unbox a float object to raw f64.
    let unbox_float = |frame: &mut crate::state::MIFrame,
                       ctx: &mut majit_metainterp::TraceCtx,
                       obj: majit_ir::OpRef|
     -> majit_ir::OpRef {
        if !ctx.heap_cache().is_class_known(obj) {
            let type_const = ctx.const_int(float_type_addr);
            frame.generate_guard(ctx, OpCode::GuardClass, &[obj, type_const]);
            ctx.heap_cache_mut().class_now_known(obj, float_type_addr);
        }
        {
            let ff_descr = crate::descr::float_floatval_descr();
            let ff_idx = ff_descr.index();
            if let Some(cached) = ctx.heapcache_getfield_cached(obj, ff_idx) {
                cached
            } else {
                let r = ctx.record_op_with_descr(OpCode::GetfieldGcPureF, &[obj], ff_descr.clone());
                let live_value = if let Some(majit_ir::Value::Ref(struct_ref)) = ctx.box_value(obj)
                {
                    let struct_ptr = struct_ref.0 as i64;
                    if struct_ptr != usize::MAX as i64 && struct_ptr != 0 {
                        ctx.field_sanity_load(struct_ptr, &ff_descr, majit_ir::Type::Float)
                    } else {
                        None
                    }
                } else {
                    None
                };
                if let Some(live_value) = live_value {
                    ctx.set_opref_concrete(r, live_value);
                }
                ctx.heapcache_getfield_now_known(obj, ff_idx, r);
                r
            }
        }
    };

    // Unbox an int object to raw i64, then CastIntToFloat → f64.
    // RPython: space.float_w(w_int) → float(w_int.intval).  bool shares int's
    // `intval`, so a bool coerces through its own &BOOL_TYPE guard.
    let unbox_int_to_float = |frame: &mut crate::state::MIFrame,
                              ctx: &mut majit_metainterp::TraceCtx,
                              obj: majit_ir::OpRef,
                              concrete: pyre_object::PyObjectRef|
     -> majit_ir::OpRef {
        let raw_int = if frame.value_type(obj) == majit_ir::Type::Int {
            obj
        } else {
            let (type_addr, descr) = crate::state::int_or_bool_unbox_type_descr(concrete);
            crate::state::trace_unbox_int_with_resume_descr(frame, obj, type_addr, descr)
        };
        let r = ctx.record_op(OpCode::CastIntToFloat, &[raw_int]);
        // Box(value) parity: derive concrete float from the int's
        // stamped Box.value so downstream consumers see it.
        if let Some(majit_ir::Value::Int(n)) = ctx.box_value(raw_int) {
            ctx.set_opref_concrete(r, majit_ir::Value::Float(n as f64));
        }
        r
    };

    let lhs_raw = if frame.value_type(a) == majit_ir::Type::Float {
        a
    } else if lhs_is_int {
        unbox_int_to_float(frame, ctx, a, concrete_lhs)
    } else {
        unbox_float(frame, ctx, a)
    };
    let rhs_raw = if frame.value_type(b) == majit_ir::Type::Float {
        b
    } else if rhs_is_int {
        unbox_int_to_float(frame, ctx, b, concrete_rhs)
    } else {
        unbox_float(frame, ctx, b)
    };

    let result = if is_power {
        // floatobject.py:561 descr_pow → _pow(space, x, y) parity.
        // ll_math_pow (ll_math.py:260) is EF_CAN_RAISE, NOT force_virtual.
        // pyjitpl.py:2084-2121 execute_varargs(rop.CALL_F, ..., exc=True, pure=False)
        // → records CALL_F and then handle_possible_exception → GUARD_NO_EXCEPTION.
        let call_result = ctx.call_float_typed_with_effect(
            crate::trace_opcode::float_pow_jit as *const (),
            &[lhs_raw, rhs_raw],
            &[majit_ir::Type::Float, majit_ir::Type::Float],
            majit_metainterp::default_effect_info(),
        );
        // pyjitpl.py:3395 GUARD_NO_EXCEPTION from handle_possible_exception.
        frame.generate_guard(ctx, OpCode::GuardNoException, &[]);
        call_result
    } else {
        let r = ctx.record_op(op_code.unwrap(), &[lhs_raw, rhs_raw]);
        // Box(value) parity: stamp the float binop result with its
        // runtime concrete (BoxFloat(value) carrier).
        if let (Some(majit_ir::Value::Float(a)), Some(majit_ir::Value::Float(b))) =
            (ctx.box_value(lhs_raw), ctx.box_value(rhs_raw))
        {
            let bits = majit_metainterp::eval_binop_f(
                op_code.unwrap(),
                a.to_bits() as i64,
                b.to_bits() as i64,
            );
            ctx.set_opref_concrete(r, majit_ir::Value::Float(f64::from_bits(bits as u64)));
        }
        r
    };

    // RPython: wrapfloat → new_with_vtable + setfield_gc
    let boxed = crate::state::wrapfloat(ctx, result);
    Some(boxed)
}

/// Trace a comparison between two Python objects.
///
/// RPython jitcode parity (int path):
///   guard_class(a) → getfield_gc_i(a) → guard_class(b) → getfield_gc_i(b)
///   → int_lt(a_raw, b_raw)
/// RPython jitcode parity (float path):
///   guard_class(a) → getfield_gc_f(a) → guard_class(b) → getfield_gc_f(b)
///   → float_lt(a_raw, b_raw)
///
/// Returns None if neither int nor float path applies → caller
/// falls back to trace_compare_value (residual).
#[inline]
pub fn generated_compare_value_direct(
    frame: &mut crate::state::MIFrame,
    ctx: &mut majit_metainterp::TraceCtx,
    a: majit_ir::OpRef,
    b: majit_ir::OpRef,
    op: ComparisonOperator,
    concrete_lhs: pyre_object::PyObjectRef,
    concrete_rhs: pyre_object::PyObjectRef,
) -> Option<majit_ir::OpRef> {
    if concrete_lhs.is_null() || concrete_rhs.is_null() {
        return None;
    }

    unsafe {
        if pyre_object::is_int(concrete_lhs) && pyre_object::is_int(concrete_rhs) {
            let cmp = int_compare_lookup(op);
            // bool and int share `intval`; guard each operand against its own
            // vtable (BOOL_TYPE / INT_TYPE) so a bool comparand unboxes
            // through its own class.  The comparison result is a bool.
            let lhs_raw = if frame.value_type(a) == majit_ir::Type::Int {
                a
            } else if let Some(raw) = crate::state::try_trace_const_boxed_int(ctx, a, concrete_lhs)
            {
                raw
            } else {
                let (type_addr, descr) = crate::state::int_or_bool_unbox_type_descr(concrete_lhs);
                crate::state::trace_unbox_int_with_resume_descr(frame, a, type_addr, descr)
            };
            let rhs_raw = if frame.value_type(b) == majit_ir::Type::Int {
                b
            } else if let Some(raw) = crate::state::try_trace_const_boxed_int(ctx, b, concrete_rhs)
            {
                raw
            } else {
                let (type_addr, descr) = crate::state::int_or_bool_unbox_type_descr(concrete_rhs);
                crate::state::trace_unbox_int_with_resume_descr(frame, b, type_addr, descr)
            };
            let truth = ctx.record_op(cmp, &[lhs_raw, rhs_raw]);
            // Box(value) parity: stamp the bool result from the operands'
            // Box.value carriers (matches dispatch.rs trace_binop_i for
            // IntEq/IntNe/IntLt/IntLe/IntGt/IntGe).
            if let (Some(majit_ir::Value::Int(la)), Some(majit_ir::Value::Int(rb))) =
                (ctx.box_value(lhs_raw), ctx.box_value(rhs_raw))
            {
                let folded = majit_metainterp::eval_binop_i(cmp, la, rb);
                ctx.set_opref_concrete(truth, majit_ir::Value::Int(folded));
            }
            // pyjitpl.py:541-556 goto_if_not_int_<op> parity: when the
            // next non-trivia instruction is POP_JUMP_IF_*, the fused
            // dispatch (try_fused_compare_goto_if_not) consumes this raw
            // truth directly and emits GUARD_TRUE/GUARD_FALSE on it. For
            // any other successor, emit the bool-box so stack discipline
            // matches the generic compare_value path.
            return if frame.next_instruction_consumes_comparison_truth() {
                Some(truth)
            } else {
                Some(crate::helpers::emit_trace_bool_value_from_truth(
                    ctx, truth, false,
                ))
            };
        }
        // baseobjspace::compare step 2: is_int_or_long × is_int_or_long
        // Long comparison needs BigInt which can't be inlined → residual (None).
        if pyre_object::is_int_or_long(concrete_lhs) && pyre_object::is_int_or_long(concrete_rhs) {
            return None;
        }
        // baseobjspace::compare step 3: is_float_pair
        // Handles float×float, float×int, int×float.
        // Long in float_pair → residual (can't inline long→f64).
        let lhs_is_float = pyre_object::is_float(concrete_lhs);
        let rhs_is_float = pyre_object::is_float(concrete_rhs);
        let lhs_is_int = pyre_object::is_int(concrete_lhs);
        let rhs_is_int = pyre_object::is_int(concrete_rhs);
        let lhs_numeric = lhs_is_float || lhs_is_int;
        let rhs_numeric = rhs_is_float || rhs_is_int;
        let is_trace_float_pair = lhs_numeric && rhs_numeric && (lhs_is_float || rhs_is_float);
        if is_trace_float_pair {
            let cmp = float_compare_lookup(op);
            let float_type_addr = &pyre_object::pyobject::FLOAT_TYPE as *const _ as i64;
            // Unbox lhs: float direct, int (or bool, via its own &BOOL_TYPE
            // guard) via CastIntToFloat.
            let lhs_raw = if frame.value_type(a) == majit_ir::Type::Float {
                a
            } else if lhs_is_int || frame.value_type(a) == majit_ir::Type::Int {
                let raw_int = if frame.value_type(a) == majit_ir::Type::Int {
                    a
                } else {
                    let (type_addr, descr) =
                        crate::state::int_or_bool_unbox_type_descr(concrete_lhs);
                    crate::state::trace_unbox_int_with_resume_descr(frame, a, type_addr, descr)
                };
                let cast = ctx.record_op(majit_ir::OpCode::CastIntToFloat, &[raw_int]);
                // Box(value) parity: derive concrete float from the int's
                // Box.value (matches dispatch.rs CastIntToFloat walker).
                if let Some(majit_ir::Value::Int(n)) = ctx.box_value(raw_int) {
                    ctx.set_opref_concrete(cast, majit_ir::Value::Float(n as f64));
                }
                cast
            } else {
                crate::state::trace_unbox_float_with_resume(frame, a, float_type_addr)
            };
            // Unbox rhs: same pattern
            let rhs_raw = if frame.value_type(b) == majit_ir::Type::Float {
                b
            } else if rhs_is_int || frame.value_type(b) == majit_ir::Type::Int {
                let raw_int = if frame.value_type(b) == majit_ir::Type::Int {
                    b
                } else {
                    let (type_addr, descr) =
                        crate::state::int_or_bool_unbox_type_descr(concrete_rhs);
                    crate::state::trace_unbox_int_with_resume_descr(frame, b, type_addr, descr)
                };
                let cast = ctx.record_op(majit_ir::OpCode::CastIntToFloat, &[raw_int]);
                if let Some(majit_ir::Value::Int(n)) = ctx.box_value(raw_int) {
                    ctx.set_opref_concrete(cast, majit_ir::Value::Float(n as f64));
                }
                cast
            } else {
                crate::state::trace_unbox_float_with_resume(frame, b, float_type_addr)
            };
            let truth = ctx.record_op(cmp, &[lhs_raw, rhs_raw]);
            // Box(value) parity: stamp the float-compare bool result.
            if let (Some(majit_ir::Value::Float(fa)), Some(majit_ir::Value::Float(fb))) =
                (ctx.box_value(lhs_raw), ctx.box_value(rhs_raw))
            {
                let folded =
                    majit_metainterp::eval_float_cmp(cmp, fa.to_bits() as i64, fb.to_bits() as i64);
                ctx.set_opref_concrete(truth, majit_ir::Value::Int(folded));
            }
            return if frame.next_instruction_consumes_comparison_truth() {
                Some(truth)
            } else {
                Some(crate::helpers::emit_trace_bool_value_from_truth(
                    ctx, truth, false,
                ))
            };
        }
    }

    None
}

/// Trace a unary int operation: unbox → op → (no re-box, returns raw int).
///
/// RPython jitcode parity: guard_class + getfield_gc_i + INT_NEG/INT_INVERT.
/// For IntNeg, declines the fast path at concrete INT_MIN (descr_neg's long
/// branch) and otherwise emits guard_false(value == INT_MIN) so a later
/// INT_MIN input deopts to the long path.
///
/// Returns None if the operand is not an int, or is the INT_MIN neg special
/// case → caller should fall back to residual
/// trace_unary_negative/invert_value.
#[inline]
pub fn generated_unary_int_value(
    frame: &mut crate::state::MIFrame,
    ctx: &mut majit_metainterp::TraceCtx,
    value: majit_ir::OpRef,
    opcode: majit_ir::OpCode,
    concrete_value: pyre_object::PyObjectRef,
) -> Option<majit_ir::OpRef> {
    use majit_ir::OpCode;

    let is_int_operand =
        !concrete_value.is_null() && unsafe { pyre_object::is_int(concrete_value) };
    if !is_int_operand {
        return None;
    }

    // RPython jitcode: guard_class + getfield_gc_i.  bool shares int's
    // `intval`; unbox a bool through its own &BOOL_TYPE guard (`-True` /
    // `~True` yield an int, so the result boxing below is unchanged).
    let payload = if frame.value_type(value) == majit_ir::Type::Int {
        value
    } else {
        let (type_addr, descr) = crate::state::int_or_bool_unbox_type_descr(concrete_value);
        crate::state::trace_unbox_int_with_resume_descr(frame, value, type_addr, descr)
    };
    // intobject.py:628 descr_neg: `if a == MININT: return <long>`.  At
    // concrete MININT the interpreter negates through the long branch, so
    // decline the int fast path and let the residual long-neg be traced —
    // an int_neg here would diverge from that long result and trip the
    // guard_false below on its own recording value.
    if matches!(opcode, OpCode::IntNeg) {
        if unsafe { pyre_object::w_int_get_value(concrete_value) } == i64::MIN {
            return None;
        }
        let min_val = ctx.const_int(i64::MIN);
        let is_min = ctx.record_op(OpCode::IntEq, &[payload, min_val]);
        if let Some(majit_ir::Value::Int(n)) = ctx.box_value(payload) {
            ctx.set_opref_concrete(is_min, majit_ir::Value::Int((n == i64::MIN) as i64));
        }
        frame.generate_guard(ctx, OpCode::GuardFalse, &[is_min]);
    }
    let result = ctx.record_op(opcode, &[payload]);
    // Box(value) parity: derive the concrete result from the operand's
    // stamped Box.value (BoxInt(value) carrier) and stamp the result.
    if let Some(majit_ir::Value::Int(n)) = ctx.box_value(payload) {
        let folded = majit_metainterp::eval_unary_i(opcode, n);
        ctx.set_opref_concrete(result, majit_ir::Value::Int(folded));
    }
    Some(result)
}

/// `pyjitpl.py:832` `arraybox = opimpl_getfield_gc_r(listbox, itemsdescr)`.
/// Loads `W_List.items` / `W_Tuple.wrappeditems` (`Ptr(GcArray(
/// OBJECTPTR))`, rlist.py:116) as a Ref-typed `items_block` op.
///
/// Pair with [`crate::state::trace_items_block_getitem_value`] /
/// [`crate::state::trace_items_block_setitem_value`] which apply the
/// `pyobject_gcarray_descr` (`base_size = ITEMS_BLOCK_ITEMS_OFFSET`,
/// `item_type = Ref`) to land on `block + base_size + idx * 8`.
#[inline]
fn load_items_block(
    ctx: &mut majit_metainterp::TraceCtx,
    obj: majit_ir::OpRef,
    items_descr: majit_ir::DescrRef,
) -> majit_ir::OpRef {
    crate::state::opimpl_getfield_gc_r(ctx, obj, items_descr)
}

#[inline]
fn list_len_descr_for_strategy(strategy_id: i64) -> majit_ir::DescrRef {
    match strategy_id {
        0 => crate::descr::list_length_descr(),
        1 => crate::descr::list_int_items_len_descr(),
        2 => crate::descr::list_float_items_len_descr(),
        _ => unreachable!(),
    }
}

/// Trace list[int_key] setitem: guard_class → guard_strategy → arraylen →
/// index computation → items_ptr → raw array setitem.
///
/// Corresponds to PyPy list strategy model (pypy/objspace/std/listobject.py)
/// as compiled through the codewriter. In RPython, the jtransform expands
/// list storage access into guard_class + getfield(items) + check_neg_index
/// + setarrayitem_gc sequences; pyjitpl.py:814 opimpl_getlistitem_gc_* itself
/// is just the final getfield+getarrayitem step. This function covers the
/// full expanded sequence including strategy guard and index normalization.
///
/// strategy_id: 0 = object, 1 = int, 2 = float.
/// For int/float strategies, the value is unboxed before writing.
#[inline]
pub fn generated_list_setitem_by_strategy<F: pyre_jit_trace::walker_frame_ops::WalkerFrameOps>(
    frame: &mut F,
    obj: majit_ir::OpRef,
    key: majit_ir::OpRef,
    value: majit_ir::OpRef,
    concrete_key: i64,
    strategy_id: i64,
    unbox_long: bool,
) {
    frame.guard_class(
        obj,
        &pyre_object::pyobject::LIST_TYPE as *const _ as *const pyre_object::PyType,
    );
    frame.guard_list_strategy(obj, strategy_id);
    let len_descr = match strategy_id {
        0 => crate::descr::list_length_descr(),
        1 => crate::descr::list_int_items_len_descr(),
        2 => crate::descr::list_float_items_len_descr(),
        _ => unreachable!(),
    };
    // pyjitpl.py:841: opimpl_check_resizable_neg_index for index normalization
    let index = opimpl_check_resizable_neg_index(frame, obj, key, len_descr, concrete_key);
    match strategy_id {
        0 => {
            // pyjitpl.py:832: arraybox = opimpl_getfield_gc_r(listbox, itemsdescr)
            // followed by setarrayitem_gc(arraybox, index, value, arraydescr).
            let items_block =
                load_items_block(frame.ctx_mut(), obj, crate::descr::list_items_descr());
            crate::state::trace_items_block_setitem_value(
                frame.ctx_mut(),
                items_block,
                index,
                value,
            );
        }
        1 => {
            let block = crate::state::opimpl_getfield_gc_r(
                frame.ctx_mut(),
                obj,
                crate::descr::list_int_items_block_descr(),
            );
            let raw = unbox_int_or_long_for_int_strategy(frame, value, unbox_long);
            crate::state::trace_int_block_setitem_value(frame.ctx_mut(), block, index, raw);
        }
        2 => {
            let block = crate::state::opimpl_getfield_gc_r(
                frame.ctx_mut(),
                obj,
                crate::descr::list_float_items_block_descr(),
            );
            let raw = if frame.value_type(value) == majit_ir::Type::Float {
                value
            } else {
                let float_type_addr = &pyre_object::pyobject::FLOAT_TYPE as *const _ as i64;
                crate::state::trace_unbox_float_with_resume(frame, value, float_type_addr)
            };
            crate::state::trace_float_block_setitem_value(frame.ctx_mut(), block, index, raw);
        }
        _ => unreachable!(),
    }
}

/// Trace same-length list slice assignment for the strategy-preserving case:
/// ```text
///     list[start:stop:1] = other_list
/// ```
///
/// PyPy's `AbstractUnwrappedStrategy.setslice` mutates the underlying
/// strategy storage directly when both lists have the same strategy.  This
/// helper deliberately handles only the no-resize case; same-list replacement
/// can only enter this path for full-list replacement, so the forward copy is
/// harmless. Other cases fall back to the generic STORE_SUBSCR residual path
/// rather than risking an incorrect partial port of listobject.py's resizing
/// and overlap rules.
#[inline]
pub fn generated_list_setslice_same_len_by_strategy<
    F: pyre_jit_trace::walker_frame_ops::WalkerFrameOps,
>(
    frame: &mut F,
    obj: majit_ir::OpRef,
    value: majit_ir::OpRef,
    raw_start: i64,
    raw_stop: i64,
    start: i64,
    stop: i64,
    strategy_id: i64,
    obj_len: usize,
    value_len: usize,
) {
    frame.guard_class(
        obj,
        &pyre_object::pyobject::LIST_TYPE as *const _ as *const pyre_object::PyType,
    );
    frame.guard_class(
        value,
        &pyre_object::pyobject::LIST_TYPE as *const _ as *const pyre_object::PyType,
    );
    frame.guard_list_strategy(obj, strategy_id);
    frame.guard_list_strategy(value, strategy_id);

    let len_descr = list_len_descr_for_strategy(strategy_id);
    let obj_len_box = crate::state::opimpl_getfield_gc_i(frame.ctx_mut(), obj, len_descr.clone());
    if raw_start == start && raw_stop == stop {
        let raw_stop_box = frame.ctx_mut().const_int(raw_stop);
        let lower_bound_ok = frame
            .ctx_mut()
            .record_op(majit_ir::OpCode::IntGe, &[obj_len_box, raw_stop_box]);
        let ol_opt = frame.ctx_mut().box_value(obj_len_box);
        if let Some(majit_ir::Value::Int(ol)) = ol_opt {
            frame.ctx_mut().set_opref_concrete(
                lower_bound_ok,
                majit_ir::Value::Int((ol >= raw_stop) as i64),
            );
        }
        frame.generate_guard(majit_ir::OpCode::GuardTrue, &[lower_bound_ok]);
    } else {
        frame.implement_guard_value(obj_len_box, obj_len as i64);
    }
    let value_len_box = crate::state::opimpl_getfield_gc_i(frame.ctx_mut(), value, len_descr);
    frame.implement_guard_value(value_len_box, value_len as i64);

    match strategy_id {
        0 => {
            let dst_items =
                load_items_block(frame.ctx_mut(), obj, crate::descr::list_items_descr());
            let src_items =
                load_items_block(frame.ctx_mut(), value, crate::descr::list_items_descr());
            for i in 0..value_len {
                let src_idx = frame.ctx_mut().const_int(i as i64);
                let dst_idx = frame.ctx_mut().const_int(start + i as i64);
                let item = crate::state::trace_items_block_getitem_value(
                    frame.ctx_mut(),
                    src_items,
                    src_idx,
                );
                crate::state::trace_items_block_setitem_value(
                    frame.ctx_mut(),
                    dst_items,
                    dst_idx,
                    item,
                );
            }
        }
        1 => {
            let dst_block = crate::state::opimpl_getfield_gc_r(
                frame.ctx_mut(),
                obj,
                crate::descr::list_int_items_block_descr(),
            );
            let src_block = crate::state::opimpl_getfield_gc_r(
                frame.ctx_mut(),
                value,
                crate::descr::list_int_items_block_descr(),
            );
            for i in 0..value_len {
                let src_idx = frame.ctx_mut().const_int(i as i64);
                let dst_idx = frame.ctx_mut().const_int(start + i as i64);
                let item = crate::state::trace_int_block_getitem_value(
                    frame.ctx_mut(),
                    src_block,
                    src_idx,
                );
                crate::state::trace_int_block_setitem_value(
                    frame.ctx_mut(),
                    dst_block,
                    dst_idx,
                    item,
                );
            }
        }
        2 => {
            let dst_block = crate::state::opimpl_getfield_gc_r(
                frame.ctx_mut(),
                obj,
                crate::descr::list_float_items_block_descr(),
            );
            let src_block = crate::state::opimpl_getfield_gc_r(
                frame.ctx_mut(),
                value,
                crate::descr::list_float_items_block_descr(),
            );
            for i in 0..value_len {
                let src_idx = frame.ctx_mut().const_int(i as i64);
                let dst_idx = frame.ctx_mut().const_int(start + i as i64);
                let item = crate::state::trace_float_block_getitem_value(
                    frame.ctx_mut(),
                    src_block,
                    src_idx,
                );
                crate::state::trace_float_block_setitem_value(
                    frame.ctx_mut(),
                    dst_block,
                    dst_idx,
                    item,
                );
            }
        }
        _ => unreachable!(),
    }

    debug_assert_eq!(stop - start, value_len as i64);
}

/// Unbox a Python int into a raw i64 for the int-strategy list path.
/// `unbox_long=true` selects `trace_unbox_long_with_resume(LONG_TYPE)` to
/// accept fits_int W_LongObject (`listobject.py:1957-1958 IntegerListStrategy
/// .is_correct_type` parity); `false` selects the default W_IntObject unbox.
#[inline]
fn unbox_int_or_long_for_int_strategy<F: pyre_jit_trace::walker_frame_ops::WalkerFrameOps>(
    frame: &mut F,
    value: majit_ir::OpRef,
    unbox_long: bool,
) -> majit_ir::OpRef {
    if frame.value_type(value) == majit_ir::Type::Int {
        return value;
    }
    if unbox_long {
        let long_type_addr = &pyre_object::pyobject::LONG_TYPE as *const _ as i64;
        crate::state::trace_unbox_long_with_resume(frame, value, long_type_addr)
    } else {
        let int_type_addr = &pyre_object::pyobject::INT_TYPE as *const _ as i64;
        crate::state::trace_unbox_int_with_resume(frame, value, int_type_addr)
    }
}

/// Trace truth value (is_true) for a Python object.
///
/// RPython jitcode parity: the codewriter generates type-specialized
/// truth tests from space.is_true(w_obj):
///   bool: guard_class + getfield_gc_i(intval) → int_ne(val, 0)
///   int:  guard_class + getfield_gc_i(intval) → int_ne(val, 0)
///   float: guard_class + getfield_gc_f(floatval) → float_ne(val, 0.0)
///   None: guard_class → const_int(0)
///   str:  guard_class + getfield_raw_i(len) → int_ne(len, 0)
///   dict: guard_class + getfield_raw_i(len) → int_ne(len, 0)
///   list: guard_class + getfield_raw_i(items_len) → int_ne(len, 0)
///   tuple: guard_class + getfield_raw_i(items_len) → int_ne(len, 0)
///
/// Returns None if the concrete type is not handled → caller falls
/// back to residual trace_truth_value.
#[inline]
pub fn generated_truth_value_direct(
    frame: &mut crate::state::MIFrame,
    ctx: &mut majit_metainterp::TraceCtx,
    value: majit_ir::OpRef,
    concrete_val: pyre_object::PyObjectRef,
) -> Option<majit_ir::OpRef> {
    use majit_ir::OpCode;

    // Already-unboxed values (Type::Int or Type::Float from earlier operations).
    if frame.value_type(value) == majit_ir::Type::Int {
        let zero = ctx.const_int(0);
        let truth = ctx.record_op(OpCode::IntNe, &[value, zero]);
        // Box(value) parity: derive bool from operand's Box.value.
        if let Some(majit_ir::Value::Int(n)) = ctx.box_value(value) {
            ctx.set_opref_concrete(truth, majit_ir::Value::Int((n != 0) as i64));
        }
        return Some(truth);
    }
    if frame.value_type(value) == majit_ir::Type::Float {
        let zero = ctx.const_int(0);
        let zero_float = ctx.record_op(OpCode::CastIntToFloat, &[zero]);
        ctx.set_opref_concrete(zero_float, majit_ir::Value::Float(0.0));
        let truth = ctx.record_op(OpCode::FloatNe, &[value, zero_float]);
        if let Some(majit_ir::Value::Float(f)) = ctx.box_value(value) {
            ctx.set_opref_concrete(truth, majit_ir::Value::Int((f != 0.0) as i64));
        }
        return Some(truth);
    }

    if concrete_val.is_null() {
        return None;
    }

    unsafe {
        // boolobject.py: bool_is_true → guard_class + getfield(intval) → int_ne(0).
        // The exact-bool shortcut precedes the int arm: `is_int` is true for a
        // bool (W_BoolObject is a W_IntObject subclass), so checking int first
        // would guard the operand against &INT_TYPE — a class a bool never
        // matches — and the trace would deopt on every bool.
        if pyre_object::is_bool(concrete_val) {
            let bool_type_addr = &pyre_object::pyobject::BOOL_TYPE as *const _ as i64;
            let bool_value = if let Some(raw) =
                crate::state::try_trace_const_boxed_int(ctx, value, concrete_val)
            {
                raw
            } else {
                crate::state::trace_unbox_int_with_resume_descr(
                    frame,
                    value,
                    bool_type_addr,
                    crate::descr::bool_intval_descr(),
                )
            };
            let zero = ctx.const_int(0);
            let truth = ctx.record_op(OpCode::IntNe, &[bool_value, zero]);
            if let Some(majit_ir::Value::Int(n)) = ctx.box_value(bool_value) {
                ctx.set_opref_concrete(truth, majit_ir::Value::Int((n != 0) as i64));
            }
            return Some(truth);
        }
        // intobject.py: int_is_true → guard_class + getfield(intval) → int_ne(0)
        if pyre_object::is_int(concrete_val) {
            let int_type_addr = &pyre_object::pyobject::INT_TYPE as *const _ as i64;
            let int_value = if let Some(raw) =
                crate::state::try_trace_const_boxed_int(ctx, value, concrete_val)
            {
                raw
            } else {
                crate::state::trace_unbox_int_with_resume(frame, value, int_type_addr)
            };
            let zero = ctx.const_int(0);
            let truth = ctx.record_op(OpCode::IntNe, &[int_value, zero]);
            if let Some(majit_ir::Value::Int(n)) = ctx.box_value(int_value) {
                ctx.set_opref_concrete(truth, majit_ir::Value::Int((n != 0) as i64));
            }
            return Some(truth);
        }
        // noneobject.py: None is always false
        if pyre_object::is_none(concrete_val) {
            frame.guard_class(
                ctx,
                value,
                &pyre_object::pyobject::NONE_TYPE as *const _ as *const pyre_object::PyType,
            );
            return Some(ctx.const_int(0));
        }
        // floatobject.py: float_is_true → guard_class + getfield(floatval) → float_ne(0.0)
        if pyre_object::is_float(concrete_val) {
            let float_type_addr = &pyre_object::pyobject::FLOAT_TYPE as *const _ as i64;
            let float_value =
                crate::state::trace_unbox_float_with_resume(frame, value, float_type_addr);
            let zero = ctx.const_int(0);
            let zero_float = ctx.record_op(OpCode::CastIntToFloat, &[zero]);
            ctx.set_opref_concrete(zero_float, majit_ir::Value::Float(0.0));
            let truth = ctx.record_op(OpCode::FloatNe, &[float_value, zero_float]);
            if let Some(majit_ir::Value::Float(f)) = ctx.box_value(float_value) {
                ctx.set_opref_concrete(truth, majit_ir::Value::Int((f != 0.0) as i64));
            }
            return Some(truth);
        }
        // unicodeobject.py: str truth → guard_class + getfield_raw(len) → int_ne(0)
        if pyre_object::is_str(concrete_val) {
            frame.guard_class(
                ctx,
                value,
                &pyre_object::STR_TYPE as *const _ as *const pyre_object::PyType,
            );
            let len_descr = crate::descr::str_len_descr();
            let len = ctx.record_op_with_descr(OpCode::GetfieldRawI, &[value], len_descr.clone());
            // Box(value) parity: stamp len with the live int read so
            // the IntNe truth below sees the BoxInt payload.
            // executor.py:200 do_getfield_raw_i projects `structbox.
            // getint()` for raw field reads.  Pyre's Python objects
            // are carried as `Value::Ref` in box_value (the
            // PyObjectRef pointer) but the recorded opcode here is
            // GetfieldRawI, so the dispatch must go through the raw
            // helper family — project the Ref to `i64` and call
            // `raw_field_sanity_load` (NOT the GC variant which
            // matches executor.py:188 do_getfield_gc_i).
            if let Some(majit_ir::Value::Ref(struct_ref)) = ctx.box_value(value) {
                let struct_ptr = struct_ref.0 as i64;
                if struct_ptr != usize::MAX as i64 && struct_ptr != 0 {
                    if let Some(live) =
                        ctx.raw_field_sanity_load(struct_ptr, &len_descr, majit_ir::Type::Int)
                    {
                        ctx.set_opref_concrete(len, live);
                    }
                }
            }
            let zero = ctx.const_int(0);
            let truth = ctx.record_op(OpCode::IntNe, &[len, zero]);
            if let Some(majit_ir::Value::Int(n)) = ctx.box_value(len) {
                ctx.set_opref_concrete(truth, majit_ir::Value::Int((n != 0) as i64));
            }
            return Some(truth);
        }
        // dictmultiobject.py:107-109 W_DictMultiObject.length → strategy
        // .length(self) — pyre delegates the same way, so the JIT does
        // not have a single-instruction lowering for `bool(dict)`.
        // Fall through to the generic `bool` callee path which will
        // dispatch through `w_dict_len`.
        // listobject.py:423 W_ListObject.length() → strategy.length()
        // All list strategies determine truth by length, same as len() fast path.
        if pyre_object::is_list(concrete_val) {
            frame.guard_class(
                ctx,
                value,
                &pyre_object::pyobject::LIST_TYPE as *const _ as *const pyre_object::PyType,
            );
            let len_descr = if pyre_object::w_list_uses_object_storage(concrete_val) {
                frame.guard_list_strategy(ctx, value, 0);
                crate::descr::list_length_descr()
            } else if pyre_object::w_list_uses_int_storage(concrete_val) {
                frame.guard_list_strategy(ctx, value, 1);
                crate::descr::list_int_items_len_descr()
            } else if pyre_object::w_list_uses_float_storage(concrete_val) {
                frame.guard_list_strategy(ctx, value, 2);
                crate::descr::list_float_items_len_descr()
            } else {
                return None; // unknown strategy → residual
            };
            let len = ctx.record_op_with_descr(OpCode::GetfieldRawI, &[value], len_descr.clone());
            // executor.py:200 do_getfield_raw_i projects `structbox.
            // getint()` for raw field reads.  Pyre's Python objects
            // are carried as `Value::Ref` in box_value (the
            // PyObjectRef pointer) but the recorded opcode here is
            // GetfieldRawI, so the dispatch must go through the raw
            // helper family — project the Ref to `i64` and call
            // `raw_field_sanity_load` (NOT the GC variant which
            // matches executor.py:188 do_getfield_gc_i).
            if let Some(majit_ir::Value::Ref(struct_ref)) = ctx.box_value(value) {
                let struct_ptr = struct_ref.0 as i64;
                if struct_ptr != usize::MAX as i64 && struct_ptr != 0 {
                    if let Some(live) =
                        ctx.raw_field_sanity_load(struct_ptr, &len_descr, majit_ir::Type::Int)
                    {
                        ctx.set_opref_concrete(len, live);
                    }
                }
            }
            let zero = ctx.const_int(0);
            let truth = ctx.record_op(OpCode::IntNe, &[len, zero]);
            if let Some(majit_ir::Value::Int(n)) = ctx.box_value(len) {
                ctx.set_opref_concrete(truth, majit_ir::Value::Int((n != 0) as i64));
            }
            return Some(truth);
        }
        // tupleobject.py: tuple truth → guard_class + getfield_gc_pure_r(wrappeditems)
        //                                + arraylen_gc(items_block) → int_ne(0)
        if pyre_object::is_tuple(concrete_val) {
            frame.guard_class(
                ctx,
                value,
                &pyre_object::pyobject::TUPLE_TYPE as *const _ as *const pyre_object::PyType,
            );
            let items_block = crate::state::opimpl_getfield_gc_r(
                ctx,
                value,
                crate::descr::tuple_wrappeditems_descr(),
            );
            let len = crate::state::opimpl_arraylen_gc(
                ctx,
                items_block,
                crate::state::pyobject_gcarray_descr(),
            );
            let zero = ctx.const_int(0);
            let truth = ctx.record_op(OpCode::IntNe, &[len, zero]);
            if let Some(majit_ir::Value::Int(n)) = ctx.box_value(len) {
                ctx.set_opref_concrete(truth, majit_ir::Value::Int((n != 0) as i64));
            }
            return Some(truth);
        }
    }

    None
}

/// Trace len() for known container types.
///
/// RPython jitcode parity: guard_class → getfield(length/len) for each type.
/// Returns None if type not handled → caller falls back to residual call.
#[inline]
pub fn generated_direct_len_value(
    frame: &mut crate::state::MIFrame,
    ctx: &mut majit_metainterp::TraceCtx,
    value: majit_ir::OpRef,
    concrete_value: pyre_object::PyObjectRef,
) -> Option<majit_ir::OpRef> {
    if concrete_value.is_null() {
        return None;
    }

    unsafe {
        if pyre_object::is_str(concrete_value) {
            frame.guard_class(
                ctx,
                value,
                &pyre_object::STR_TYPE as *const _ as *const pyre_object::PyType,
            );
            let len = crate::state::trace_arraylen_gc(ctx, value, crate::descr::str_len_descr());
            return Some(len);
        }
        // dictmultiobject.py:107-109 W_DictMultiObject.length → strategy
        // .length(self) — pyre delegates the same way, so the JIT does
        // not have a single-instruction lowering for `len(dict)`.
        // Fall through to the generic `len` callee path which will
        // dispatch through `w_dict_len`.
        if pyre_object::is_list(concrete_value) {
            frame.guard_class(
                ctx,
                value,
                &pyre_object::pyobject::LIST_TYPE as *const _ as *const pyre_object::PyType,
            );
            let len_descr = if pyre_object::w_list_uses_object_storage(concrete_value) {
                frame.guard_list_strategy(ctx, value, 0);
                crate::descr::list_length_descr()
            } else if pyre_object::w_list_uses_int_storage(concrete_value) {
                frame.guard_list_strategy(ctx, value, 1);
                crate::descr::list_int_items_len_descr()
            } else if pyre_object::w_list_uses_float_storage(concrete_value) {
                frame.guard_list_strategy(ctx, value, 2);
                crate::descr::list_float_items_len_descr()
            } else {
                return None; // unknown list strategy → residual
            };
            let len = crate::state::trace_arraylen_gc(ctx, value, len_descr);
            return Some(len);
        }
        if pyre_object::is_tuple(concrete_value) {
            frame.guard_class(
                ctx,
                value,
                &pyre_object::pyobject::TUPLE_TYPE as *const _ as *const pyre_object::PyType,
            );
            // tupleobject.py:376-390 W_TupleObject — len comes from
            // arraylen_gc on the GcArray header, not a length field.
            let items_block = crate::state::opimpl_getfield_gc_r(
                ctx,
                value,
                crate::descr::tuple_wrappeditems_descr(),
            );
            let len = crate::state::opimpl_arraylen_gc(
                ctx,
                items_block,
                crate::state::pyobject_gcarray_descr(),
            );
            return Some(len);
        }
    }

    None
}
/// Trace abs() for int values: guard_class + getfield → guard(!=MIN) → branchless abs.
///
/// RPython jitcode: guard_class + getfield_gc_i(intval) → int_abs sequence.
/// Returns None if not handled → caller falls back to residual call.
#[inline]
pub fn generated_direct_abs_value(
    frame: &mut crate::state::MIFrame,
    ctx: &mut majit_metainterp::TraceCtx,
    value: majit_ir::OpRef,
    concrete_value: pyre_object::PyObjectRef,
) -> Option<majit_ir::OpRef> {
    use majit_ir::OpCode;

    if concrete_value.is_null() {
        return None;
    }

    unsafe {
        if pyre_object::is_int(concrete_value) {
            let concrete_int = pyre_object::w_int_get_value(concrete_value);
            if concrete_int == i64::MIN {
                return None; // overflow → residual
            }
            let int_value = frame.trace_guarded_int_payload(ctx, value);
            let min_value = ctx.const_int(i64::MIN);
            let is_min = ctx.record_op(OpCode::IntEq, &[int_value, min_value]);
            ctx.set_opref_concrete(is_min, majit_ir::Value::Int(0));
            frame.generate_guard(ctx, OpCode::GuardFalse, &[is_min]);
            // branchless abs: (x ^ sign) - sign
            let shift = ctx.const_int((i64::BITS - 1) as i64);
            let sign = ctx.record_op(OpCode::IntRshift, &[int_value, shift]);
            let sign_concrete = concrete_int >> (i64::BITS - 1);
            ctx.set_opref_concrete(sign, majit_ir::Value::Int(sign_concrete));
            let xor = ctx.record_op(OpCode::IntXor, &[int_value, sign]);
            let xor_concrete = concrete_int ^ sign_concrete;
            ctx.set_opref_concrete(xor, majit_ir::Value::Int(xor_concrete));
            let abs_value = ctx.record_op(OpCode::IntSub, &[xor, sign]);
            let abs_concrete = xor_concrete.wrapping_sub(sign_concrete);
            ctx.set_opref_concrete(abs_value, majit_ir::Value::Int(abs_concrete));
            debug_assert_eq!(abs_concrete, concrete_int.wrapping_abs());
            return Some(crate::state::wrapint(ctx, abs_value));
        }
    }

    None
}

/// Trace type(obj): guard_class + getfield(w_class) + GUARD_VALUE → const type.
///
/// RPython parity: objspace.py:400-402
///   def type(self, w_obj):
///       jit.promote(w_obj.__class__)
///       return w_obj.getclass(self)
///
/// With w_class on PyObject, all object types use the same pattern:
///   guard_class(ob_type) + getfield_gc_r(obj, w_class) + GUARD_VALUE(w_class)
///
/// Returns None if not handled → caller falls back to residual call.
#[inline]
pub fn generated_direct_type_value(
    frame: &mut crate::state::MIFrame,
    ctx: &mut majit_metainterp::TraceCtx,
    value: majit_ir::OpRef,
    concrete_value: pyre_object::PyObjectRef,
) -> Option<majit_ir::OpRef> {
    if concrete_value.is_null() {
        return None;
    }

    let concrete_type_obj = pyre_interpreter::typedef::r#type(concrete_value)?;

    unsafe {
        // guard_class(ob_type) — guards the dispatch-level type tag.
        let concrete_obj_type = (*concrete_value).ob_type;
        frame.guard_class(ctx, value, concrete_obj_type);

        // getfield_gc_r(obj, w_class_descr) — read the Python class.
        // RPython: getfield_gc_r(obj, typeptr_descr)
        let w_class_descr = crate::descr::w_class_descr();
        let w_class_opref =
            ctx.record_op_with_descr(majit_ir::OpCode::GetfieldGcR, &[value], w_class_descr);

        // GUARD_VALUE(w_class, concrete_type) — promote to constant.
        // RPython: jit.promote(w_obj.__class__) → ref_guard_value
        frame.implement_guard_value(ctx, w_class_opref, concrete_type_obj as i64);

        Some(ctx.const_ref(concrete_type_obj as i64))
    }
}

/// Trace isinstance(obj, cls): guard_class + getfield(w_class) + GUARD_VALUE → const bool.
///
/// RPython parity: isinstance uses space.type(w_obj) internally, which
/// calls jit.promote(w_obj.__class__). Same unified pattern as type():
///   guard_class(ob_type) + getfield_gc_r(obj, w_class) + GUARD_VALUE
/// cls is always promoted via implement_guard_value.
///
/// Returns None if not handled → caller falls back to residual call.
#[inline]
pub fn generated_direct_isinstance_value(
    frame: &mut crate::state::MIFrame,
    ctx: &mut majit_metainterp::TraceCtx,
    obj: majit_ir::OpRef,
    cls: majit_ir::OpRef,
    concrete_obj: pyre_object::PyObjectRef,
    concrete_cls: pyre_object::PyObjectRef,
) -> Option<majit_ir::OpRef> {
    if concrete_obj.is_null() || concrete_cls.is_null() {
        return None;
    }

    let concrete_result = pyre_interpreter::builtins::call_isinstance(concrete_obj, concrete_cls)?;

    unsafe {
        let concrete_type_obj = pyre_interpreter::typedef::r#type(concrete_obj);

        // guard_class(ob_type)
        let concrete_obj_type = (*concrete_obj).ob_type;
        frame.guard_class(ctx, obj, concrete_obj_type);

        // getfield_gc_r(obj, w_class) + GUARD_VALUE — promote w_class
        if let Some(type_obj) = concrete_type_obj {
            let w_class_descr = crate::descr::w_class_descr();
            let w_class_opref =
                ctx.record_op_with_descr(majit_ir::OpCode::GetfieldGcR, &[obj], w_class_descr);
            frame.implement_guard_value(ctx, w_class_opref, type_obj as i64);
        }

        // promote cls argument
        frame.implement_guard_value(ctx, cls, concrete_cls as i64);
        Some(ctx.const_ref(pyre_object::w_bool_from(concrete_result) as i64))
    }
}

/// Trace min()/max() for two ints: branchless int selection.
///
/// RPython jitcode: guard_class × 2 + getfield_gc_i × 2 + int_lt + branchless select + box.
/// Returns None if not handled → caller falls back to residual call.
#[inline]
pub fn generated_direct_minmax_value(
    frame: &mut crate::state::MIFrame,
    ctx: &mut majit_metainterp::TraceCtx,
    a: majit_ir::OpRef,
    b: majit_ir::OpRef,
    choose_max: bool,
    concrete_a: pyre_object::PyObjectRef,
    concrete_b: pyre_object::PyObjectRef,
) -> Option<majit_ir::OpRef> {
    use majit_ir::OpCode;

    if concrete_a.is_null() || concrete_b.is_null() {
        return None;
    }

    unsafe {
        if pyre_object::is_int(concrete_a) && pyre_object::is_int(concrete_b) {
            let lhs_val = pyre_object::w_int_get_value(concrete_a);
            let rhs_val = pyre_object::w_int_get_value(concrete_b);
            let concrete_result = if choose_max {
                lhs_val.max(rhs_val)
            } else {
                lhs_val.min(rhs_val)
            };
            let concrete_obj = if choose_max {
                if lhs_val >= rhs_val {
                    concrete_a
                } else {
                    concrete_b
                }
            } else if lhs_val <= rhs_val {
                concrete_a
            } else {
                concrete_b
            };
            if pyre_object::w_int_new(concrete_result) != concrete_obj {
                return None; // identity mismatch → residual
            }
            let lhs = frame.trace_guarded_int_payload(ctx, a);
            let rhs = frame.trace_guarded_int_payload(ctx, b);
            // branchless min/max: mask = 0 - (lhs < rhs)
            let cmp = ctx.record_op(OpCode::IntLt, &[lhs, rhs]);
            let cmp_concrete = (lhs_val < rhs_val) as i64;
            ctx.set_opref_concrete(cmp, majit_ir::Value::Int(cmp_concrete));
            let zero = ctx.const_int(0);
            let mask = ctx.record_op(OpCode::IntSub, &[zero, cmp]);
            let mask_concrete = 0i64.wrapping_sub(cmp_concrete);
            ctx.set_opref_concrete(mask, majit_ir::Value::Int(mask_concrete));
            let xor = ctx.record_op(OpCode::IntXor, &[lhs, rhs]);
            let xor_concrete = lhs_val ^ rhs_val;
            ctx.set_opref_concrete(xor, majit_ir::Value::Int(xor_concrete));
            let select_bits = ctx.record_op(OpCode::IntAnd, &[xor, mask]);
            let select_bits_concrete = xor_concrete & mask_concrete;
            ctx.set_opref_concrete(select_bits, majit_ir::Value::Int(select_bits_concrete));
            let (selected, selected_concrete) = if choose_max {
                (
                    ctx.record_op(OpCode::IntXor, &[lhs, select_bits]),
                    lhs_val ^ select_bits_concrete,
                )
            } else {
                (
                    ctx.record_op(OpCode::IntXor, &[rhs, select_bits]),
                    rhs_val ^ select_bits_concrete,
                )
            };
            ctx.set_opref_concrete(selected, majit_ir::Value::Int(selected_concrete));
            debug_assert_eq!(selected_concrete, concrete_result);
            return Some(crate::state::wrapint(ctx, selected));
        }
    }

    None
}
/// jtransform do_fixed_list_getitem parity:
///   guard_class + opimpl_check_neg_index + getarrayitem_gc_r_pure.
///
/// Tuples are fixed-size arrays: uses opimpl_check_neg_index for index
/// normalization (ARRAYLEN_GC for length). For arity-2 specialised
/// variants (`Cls_ii / Cls_ff / Cls_oo` per
/// `pypy/objspace/std/specialisedtupleobject.py`) the trace dispatches
/// on the runtime `ob_type` and emits a direct inline-field load —
/// `value0` / `value1` are immutable so the `GetfieldGcPureI/F/R` op
/// is constant-foldable.
#[inline]
pub fn generated_tuple_getitem(
    frame: &mut crate::state::MIFrame,
    ctx: &mut majit_metainterp::TraceCtx,
    obj: majit_ir::OpRef,
    key: majit_ir::OpRef,
    concrete_obj: pyre_object::PyObjectRef,
    concrete_key: i64,
    _concrete_len: usize,
) -> majit_ir::OpRef {
    use majit_ir::OpCode;
    let ob_type = unsafe { (*concrete_obj).ob_type };

    // pypy/objspace/std/specialisedtupleobject.py — three arity-2
    // variants. All have arity exactly 2 by construction; bounds and
    // index normalisation collapse into a single `guard_value` against
    // the trace-time `concrete_key`.
    let spec_ii = &pyre_object::specialisedtupleobject::SPECIALISED_TUPLE_II_TYPE as *const _
        as *const pyre_object::PyType;
    let spec_ff = &pyre_object::specialisedtupleobject::SPECIALISED_TUPLE_FF_TYPE as *const _
        as *const pyre_object::PyType;
    let spec_oo = &pyre_object::specialisedtupleobject::SPECIALISED_TUPLE_OO_TYPE as *const _
        as *const pyre_object::PyType;

    if std::ptr::eq(ob_type, spec_ii) {
        let normalised = if concrete_key < 0 {
            concrete_key + 2
        } else {
            concrete_key
        };
        frame.guard_class(ctx, obj, spec_ii);
        // Lock the trace to the runtime index. Caller already validated
        // 0 <= normalised < 2 via `check_index_in_bounds`.
        let key_unboxed = if frame.value_type(key) == majit_ir::Type::Int {
            key
        } else {
            let int_type_addr = &pyre_object::pyobject::INT_TYPE as *const _ as i64;
            crate::state::trace_unbox_int_with_resume(frame, key, int_type_addr)
        };
        frame.implement_guard_value(ctx, key_unboxed, concrete_key);
        let descr = if normalised == 0 {
            crate::descr::specialised_tuple_ii_value0_descr()
        } else {
            crate::descr::specialised_tuple_ii_value1_descr()
        };
        let raw = ctx.record_op_with_descr(OpCode::GetfieldGcPureI, &[obj], descr.clone());
        // Box(value) parity: stamp the pure field load with the live int
        // payload (matches getfield_gc_i_pureornot miss path).
        if let Some(majit_ir::Value::Ref(struct_ref)) = ctx.box_value(obj) {
            let struct_ptr = struct_ref.0 as i64;
            if struct_ptr != usize::MAX as i64 && struct_ptr != 0 {
                if let Some(live) = ctx.field_sanity_load(struct_ptr, &descr, majit_ir::Type::Int) {
                    ctx.set_opref_concrete(raw, live);
                }
            }
        }
        return crate::state::wrapint(ctx, raw);
    }

    if std::ptr::eq(ob_type, spec_ff) {
        let normalised = if concrete_key < 0 {
            concrete_key + 2
        } else {
            concrete_key
        };
        frame.guard_class(ctx, obj, spec_ff);
        let key_unboxed = if frame.value_type(key) == majit_ir::Type::Int {
            key
        } else {
            let int_type_addr = &pyre_object::pyobject::INT_TYPE as *const _ as i64;
            crate::state::trace_unbox_int_with_resume(frame, key, int_type_addr)
        };
        frame.implement_guard_value(ctx, key_unboxed, concrete_key);
        let descr = if normalised == 0 {
            crate::descr::specialised_tuple_ff_value0_descr()
        } else {
            crate::descr::specialised_tuple_ff_value1_descr()
        };
        let raw = ctx.record_op_with_descr(OpCode::GetfieldGcPureF, &[obj], descr.clone());
        if let Some(majit_ir::Value::Ref(struct_ref)) = ctx.box_value(obj) {
            let struct_ptr = struct_ref.0 as i64;
            if struct_ptr != usize::MAX as i64 && struct_ptr != 0 {
                if let Some(live) = ctx.field_sanity_load(struct_ptr, &descr, majit_ir::Type::Float)
                {
                    ctx.set_opref_concrete(raw, live);
                }
            }
        }
        return crate::state::wrapfloat(ctx, raw);
    }

    if std::ptr::eq(ob_type, spec_oo) {
        let normalised = if concrete_key < 0 {
            concrete_key + 2
        } else {
            concrete_key
        };
        frame.guard_class(ctx, obj, spec_oo);
        let key_unboxed = if frame.value_type(key) == majit_ir::Type::Int {
            key
        } else {
            let int_type_addr = &pyre_object::pyobject::INT_TYPE as *const _ as i64;
            crate::state::trace_unbox_int_with_resume(frame, key, int_type_addr)
        };
        frame.implement_guard_value(ctx, key_unboxed, concrete_key);
        let descr = if normalised == 0 {
            crate::descr::specialised_tuple_oo_value0_descr()
        } else {
            crate::descr::specialised_tuple_oo_value1_descr()
        };
        let raw = ctx.record_op_with_descr(OpCode::GetfieldGcPureR, &[obj], descr.clone());
        if let Some(majit_ir::Value::Ref(struct_ref)) = ctx.box_value(obj) {
            let struct_ptr = struct_ref.0 as i64;
            if struct_ptr != usize::MAX as i64 && struct_ptr != 0 {
                if let Some(live) = ctx.field_sanity_load(struct_ptr, &descr, majit_ir::Type::Ref) {
                    ctx.set_opref_concrete(raw, live);
                }
            }
        }
        return raw;
    }

    // Canonical W_TupleObject array-backed path. After dropping the
    // inline length cache (tupleobject.py:376-390 parity), the
    // arraydescr handed to opimpl_check_neg_index is the
    // pyobject_gcarray_descr — `arraylen_gc(items_block)` reads the
    // GcArray header directly per pyjitpl.py:773.
    let expected_type =
        &pyre_object::pyobject::TUPLE_TYPE as *const _ as *const pyre_object::PyType;

    frame.guard_class(ctx, obj, expected_type);

    let items_block = load_items_block(ctx, obj, crate::descr::tuple_wrappeditems_descr());
    // pyjitpl.py:767-776 opimpl_check_neg_index — arraybox is the
    // GcArray (items_block), arraydescr is pyobject_gcarray_descr.
    let index = opimpl_check_neg_index(
        frame,
        ctx,
        items_block,
        key,
        crate::state::pyobject_gcarray_descr(),
        concrete_key,
    );

    crate::state::trace_items_block_getitem_value(ctx, items_block, index)
}

/// pyjitpl.py:767-776 opimpl_check_neg_index:
///   negbox = INT_LT(indexbox, CONST_FALSE)
///   negbox = implement_guard_value(negbox, orgpc)
///   if negbox.getint():
///       lengthbox = opimpl_arraylen_gc(arraybox, arraydescr)
///       indexbox = INT_ADD(indexbox, lengthbox)
///   return indexbox
///
/// For fixed-size arrays (tuples). Bounds guards added for raw-pointer safety.
#[inline]
pub fn opimpl_check_neg_index(
    frame: &mut crate::state::MIFrame,
    ctx: &mut majit_metainterp::TraceCtx,
    arraybox: majit_ir::OpRef,
    indexbox: majit_ir::OpRef,
    arraydescr: majit_ir::DescrRef,
    concrete_key: i64,
) -> majit_ir::OpRef {
    use majit_ir::OpCode;

    let raw_index = if frame.value_type(indexbox) == majit_ir::Type::Int {
        indexbox
    } else {
        // descroperation.py getindex_w / int_w accept the whole
        // W_IntObject family uniformly; a bool index guards its own
        // &BOOL_TYPE vtable but shares the intval field. Pick the guard
        // class from the boxed key's concrete (INT_TYPE when unavailable).
        let concrete_key_obj = match ctx.box_value(indexbox) {
            Some(majit_ir::Value::Ref(gcref)) => gcref.0 as pyre_object::PyObjectRef,
            _ => pyre_object::PY_NULL,
        };
        let (type_addr, intval_descr) =
            crate::state::int_or_bool_unbox_type_descr(concrete_key_obj);
        crate::state::trace_unbox_int_with_resume_descr(frame, indexbox, type_addr, intval_descr)
    };
    // Box(value) parity: stamp the unboxed index with its concrete.
    ctx.set_opref_concrete(raw_index, majit_ir::Value::Int(concrete_key));
    let zero = ctx.const_int(0);
    // pyjitpl.py:768-770
    let negbox = ctx.record_op(OpCode::IntLt, &[raw_index, zero]);
    ctx.set_opref_concrete(negbox, majit_ir::Value::Int((concrete_key < 0) as i64));
    frame.implement_guard_value(ctx, negbox, if concrete_key < 0 { 1 } else { 0 });
    if concrete_key < 0 {
        // pyjitpl.py:773: lengthbox = self.opimpl_arraylen_gc(arraybox, arraydescr)
        let lengthbox = crate::state::opimpl_arraylen_gc(ctx, arraybox, arraydescr);
        // pyjitpl.py:774-775: indexbox = INT_ADD(indexbox, lengthbox)
        let indexbox = ctx.record_op(OpCode::IntAdd, &[raw_index, lengthbox]);
        if let Some(majit_ir::Value::Int(len)) = ctx.box_value(lengthbox) {
            ctx.set_opref_concrete(
                indexbox,
                majit_ir::Value::Int(concrete_key.wrapping_add(len)),
            );
        }
        // bounds guard (raw-pointer safety, not in RPython meta-interp)
        let in_bounds = ctx.record_op(OpCode::IntGe, &[indexbox, zero]);
        if let Some(majit_ir::Value::Int(idx)) = ctx.box_value(indexbox) {
            ctx.set_opref_concrete(in_bounds, majit_ir::Value::Int((idx >= 0) as i64));
        }
        frame.generate_guard(ctx, OpCode::GuardTrue, &[in_bounds]);
        indexbox
    } else {
        // RPython: no bounds check in check_neg_index for positive index.
        // We add one for raw-pointer safety (no GC array bounds check).
        let lengthbox = crate::state::opimpl_arraylen_gc(ctx, arraybox, arraydescr);
        let in_bounds = ctx.record_op(OpCode::IntLt, &[raw_index, lengthbox]);
        if let Some(majit_ir::Value::Int(len)) = ctx.box_value(lengthbox) {
            ctx.set_opref_concrete(in_bounds, majit_ir::Value::Int((concrete_key < len) as i64));
        }
        frame.generate_guard(ctx, OpCode::GuardTrue, &[in_bounds]);
        raw_index
    }
}

/// pyjitpl.py:841-852 opimpl_check_resizable_neg_index:
///   negbox = INT_LT(indexbox, CONST_FALSE)
///   negbox = implement_guard_value(negbox, orgpc)
///   if negbox.getint():
///       lenbox = execute_and_record(GETFIELD_GC, lengthdescr, listbox)
///       indexbox = INT_ADD(indexbox, lenbox)
///   return indexbox
///
/// For resizable lists. Uses GETFIELD_GC for length (not ARRAYLEN_GC).
/// Bounds guards added for raw-pointer safety.
#[inline]
pub fn opimpl_check_resizable_neg_index<F: pyre_jit_trace::walker_frame_ops::WalkerFrameOps>(
    frame: &mut F,
    listbox: majit_ir::OpRef,
    indexbox: majit_ir::OpRef,
    lengthdescr: majit_ir::DescrRef,
    concrete_key: i64,
) -> majit_ir::OpRef {
    use majit_ir::OpCode;

    let raw_index = if frame.value_type(indexbox) == majit_ir::Type::Int {
        indexbox
    } else {
        // descroperation.py getindex_w / int_w accept the whole
        // W_IntObject family uniformly; a bool index guards its own
        // &BOOL_TYPE vtable but shares the intval field. Pick the guard
        // class from the boxed key's concrete (INT_TYPE when unavailable).
        let concrete_key_obj = match frame.ctx().box_value(indexbox) {
            Some(majit_ir::Value::Ref(gcref)) => gcref.0 as pyre_object::PyObjectRef,
            _ => pyre_object::PY_NULL,
        };
        let (type_addr, intval_descr) =
            crate::state::int_or_bool_unbox_type_descr(concrete_key_obj);
        crate::state::trace_unbox_int_with_resume_descr(frame, indexbox, type_addr, intval_descr)
    };
    frame
        .ctx_mut()
        .set_opref_concrete(raw_index, majit_ir::Value::Int(concrete_key));
    let zero = frame.ctx_mut().const_int(0);
    // pyjitpl.py:843-845
    let negbox = frame.ctx_mut().record_op(OpCode::IntLt, &[raw_index, zero]);
    frame
        .ctx_mut()
        .set_opref_concrete(negbox, majit_ir::Value::Int((concrete_key < 0) as i64));
    frame.implement_guard_value(negbox, if concrete_key < 0 { 1 } else { 0 });
    if concrete_key < 0 {
        // pyjitpl.py:848: lenbox = execute_and_record(GETFIELD_GC, lengthdescr, listbox)
        let lenbox = crate::state::opimpl_getfield_gc_i(frame.ctx_mut(), listbox, lengthdescr);
        // pyjitpl.py:850-851: indexbox = INT_ADD(indexbox, lenbox)
        let indexbox = frame
            .ctx_mut()
            .record_op(OpCode::IntAdd, &[raw_index, lenbox]);
        let len_opt = frame.ctx_mut().box_value(lenbox);
        if let Some(majit_ir::Value::Int(len)) = len_opt {
            frame.ctx_mut().set_opref_concrete(
                indexbox,
                majit_ir::Value::Int(concrete_key.wrapping_add(len)),
            );
        }
        // bounds guard (raw-pointer safety)
        let in_bounds = frame.ctx_mut().record_op(OpCode::IntGe, &[indexbox, zero]);
        let idx_opt = frame.ctx_mut().box_value(indexbox);
        if let Some(majit_ir::Value::Int(idx)) = idx_opt {
            frame
                .ctx_mut()
                .set_opref_concrete(in_bounds, majit_ir::Value::Int((idx >= 0) as i64));
        }
        frame.generate_guard(OpCode::GuardTrue, &[in_bounds]);
        indexbox
    } else {
        // RPython: no bounds check for positive index.
        // We add one for raw-pointer safety.
        let lenbox = crate::state::opimpl_getfield_gc_i(frame.ctx_mut(), listbox, lengthdescr);
        let in_bounds = frame
            .ctx_mut()
            .record_op(OpCode::IntLt, &[raw_index, lenbox]);
        let len_opt = frame.ctx_mut().box_value(lenbox);
        if let Some(majit_ir::Value::Int(len)) = len_opt {
            frame
                .ctx_mut()
                .set_opref_concrete(in_bounds, majit_ir::Value::Int((concrete_key < len) as i64));
        }
        frame.generate_guard(OpCode::GuardTrue, &[in_bounds]);
        raw_index
    }
}

/// Backward-compat wrapper with pre-computed length parameter.
/// check_neg_index/check_resizable_neg_index read length internally;
/// this legacy path accepts a pre-read `len` for existing callers.
#[inline]
pub fn generated_dynamic_list_index(
    frame: &mut crate::state::MIFrame,
    ctx: &mut majit_metainterp::TraceCtx,
    key: majit_ir::OpRef,
    len: majit_ir::OpRef,
    concrete_key: i64,
) -> majit_ir::OpRef {
    use majit_ir::OpCode;

    let int_type_addr = &pyre_object::pyobject::INT_TYPE as *const _ as i64;
    let raw_index = if frame.value_type(key) == majit_ir::Type::Int {
        key
    } else {
        crate::state::trace_unbox_int_with_resume(frame, key, int_type_addr)
    };
    ctx.set_opref_concrete(raw_index, majit_ir::Value::Int(concrete_key));
    let zero = ctx.const_int(0);
    // pyjitpl.py:843-845 implement_guard_value parity
    let negbox = ctx.record_op(OpCode::IntLt, &[raw_index, zero]);
    ctx.set_opref_concrete(negbox, majit_ir::Value::Int((concrete_key < 0) as i64));
    frame.implement_guard_value(ctx, negbox, if concrete_key < 0 { 1 } else { 0 });
    if concrete_key < 0 {
        // pyjitpl.py:850-851: INT_ADD(indexbox, lenbox)
        let indexbox = ctx.record_op(OpCode::IntAdd, &[raw_index, len]);
        if let Some(majit_ir::Value::Int(len_val)) = ctx.box_value(len) {
            ctx.set_opref_concrete(
                indexbox,
                majit_ir::Value::Int(concrete_key.wrapping_add(len_val)),
            );
        }
        let in_bounds = ctx.record_op(OpCode::IntGe, &[indexbox, zero]);
        if let Some(majit_ir::Value::Int(idx)) = ctx.box_value(indexbox) {
            ctx.set_opref_concrete(in_bounds, majit_ir::Value::Int((idx >= 0) as i64));
        }
        frame.generate_guard(ctx, OpCode::GuardTrue, &[in_bounds]);
        indexbox
    } else {
        let in_bounds = ctx.record_op(OpCode::IntLt, &[raw_index, len]);
        if let Some(majit_ir::Value::Int(len_val)) = ctx.box_value(len) {
            ctx.set_opref_concrete(
                in_bounds,
                majit_ir::Value::Int((concrete_key < len_val) as i64),
            );
        }
        frame.generate_guard(ctx, OpCode::GuardTrue, &[in_bounds]);
        raw_index
    }
}

/// pyjitpl.py:814-827 opimpl_getlistitem_gc_{i,r,f}:
///   arraybox = getfield_gc_r(listbox, itemsdescr)
///   return getarrayitem_gc_{i,r,f}(arraybox, indexbox, arraydescr)
///
/// Combined with guard_class + guard_strategy + opimpl_check_resizable_neg_index
/// as emitted by jtransform do_resizable_list_getitem.
///
/// strategy_id: 0 = object, 1 = int, 2 = float.
#[inline]
pub fn generated_list_getitem_by_strategy(
    frame: &mut crate::state::MIFrame,
    ctx: &mut majit_metainterp::TraceCtx,
    obj: majit_ir::OpRef,
    key: majit_ir::OpRef,
    concrete_key: i64,
    strategy_id: i64,
) -> majit_ir::OpRef {
    frame.guard_class(
        ctx,
        obj,
        &pyre_object::pyobject::LIST_TYPE as *const _ as *const pyre_object::PyType,
    );
    frame.guard_list_strategy(ctx, obj, strategy_id);
    let len_descr = match strategy_id {
        0 => crate::descr::list_length_descr(),
        1 => crate::descr::list_int_items_len_descr(),
        2 => crate::descr::list_float_items_len_descr(),
        _ => unreachable!(),
    };
    let index = opimpl_check_resizable_neg_index(frame, obj, key, len_descr, concrete_key);
    match strategy_id {
        0 => {
            let items_block = load_items_block(ctx, obj, crate::descr::list_items_descr());
            crate::state::trace_items_block_getitem_value(ctx, items_block, index)
        }
        1 => {
            let block = crate::state::opimpl_getfield_gc_r(
                ctx,
                obj,
                crate::descr::list_int_items_block_descr(),
            );
            crate::state::trace_int_block_getitem_value(ctx, block, index)
        }
        2 => {
            let block = crate::state::opimpl_getfield_gc_r(
                ctx,
                obj,
                crate::descr::list_float_items_block_descr(),
            );
            crate::state::trace_float_block_getitem_value(ctx, block, index)
        }
        _ => unreachable!(),
    }
}

/// Dispatch binary subscript (getitem) to type-specialized trace paths.
///
/// jtransform do_fixed_list_getitem / do_resizable_list_getitem parity:
///   tuple → opimpl_check_neg_index + getarrayitem_gc_r_pure
///   list  → guard_class + guard_strategy + opimpl_check_resizable_neg_index
///           + opimpl_getlistitem_gc_{i,r,f}
///
/// Returns None if not a recognized subscript → caller falls back to residual.
#[inline]
pub fn generated_binary_subscr_value(
    frame: &mut crate::state::MIFrame,
    ctx: &mut majit_metainterp::TraceCtx,
    a: majit_ir::OpRef,
    b: majit_ir::OpRef,
    concrete_obj: pyre_object::PyObjectRef,
    concrete_key: pyre_object::PyObjectRef,
) -> Option<majit_ir::OpRef> {
    if concrete_obj.is_null() || concrete_key.is_null() {
        return None;
    }
    unsafe {
        if !pyre_object::pyobject::is_int(concrete_key) {
            return None;
        }
        let index = pyre_object::w_int_get_value(concrete_key);

        if pyre_object::pyobject::is_tuple(concrete_obj) {
            let concrete_len = pyre_object::w_tuple_len(concrete_obj);
            if check_index_in_bounds(index, concrete_len) {
                return Some(generated_tuple_getitem(
                    frame,
                    ctx,
                    a,
                    b,
                    concrete_obj,
                    index,
                    concrete_len,
                ));
            }
        } else if pyre_object::pyobject::is_list(concrete_obj) {
            if let Some(sid) = detect_list_getitem_strategy(concrete_obj) {
                let concrete_len = pyre_object::w_list_len(concrete_obj);
                if check_index_in_bounds(index, concrete_len) {
                    return Some(generated_list_getitem_by_strategy(
                        frame, ctx, a, b, index, sid,
                    ));
                }
            }
        }
    }
    None
}

/// Dispatch store subscript (setitem) to type-specialized trace paths.
///
/// jtransform do_resizable_list_setitem parity:
///   list + int key → guard_class + guard_strategy + opimpl_check_resizable_neg_index
///                    + opimpl_setlistitem_gc_{i,r,f}
///
/// Returns true if handled, false if caller should fall back to residual.
#[inline]
pub fn generated_store_subscr_value<F: pyre_jit_trace::walker_frame_ops::WalkerFrameOps>(
    frame: &mut F,
    obj: majit_ir::OpRef,
    key: majit_ir::OpRef,
    value: majit_ir::OpRef,
    concrete_obj: pyre_object::PyObjectRef,
    concrete_key: pyre_object::PyObjectRef,
    concrete_value: pyre_object::PyObjectRef,
) -> bool {
    if concrete_obj.is_null() || concrete_key.is_null() || concrete_value.is_null() {
        return false;
    }
    unsafe {
        if pyre_object::pyobject::is_list(concrete_obj)
            && pyre_object::pyobject::is_int(concrete_key)
        {
            if let Some((sid, unbox_long)) =
                detect_list_setitem_strategy(concrete_obj, concrete_value)
            {
                let index = pyre_object::w_int_get_value(concrete_key);
                let concrete_len = pyre_object::w_list_len(concrete_obj);
                if check_index_in_bounds(index, concrete_len) {
                    generated_list_setitem_by_strategy(
                        frame, obj, key, value, index, sid, unbox_long,
                    );
                    return true;
                }
            }
        }
    }
    false
}

/// Check if index is within bounds of a container with given length.
#[inline]
fn check_index_in_bounds(index: i64, len: usize) -> bool {
    if index >= 0 {
        (index as usize) < len
    } else {
        index
            .checked_neg()
            .and_then(|v| usize::try_from(v).ok())
            .map_or(false, |abs| abs <= len)
    }
}

/// Detect list strategy for getitem.
/// Returns strategy_id: 0 = object, 1 = int, 2 = float, or None.
#[inline]
unsafe fn detect_list_getitem_strategy(concrete_obj: pyre_object::PyObjectRef) -> Option<i64> {
    if pyre_object::w_list_uses_object_storage(concrete_obj) {
        Some(0)
    } else if pyre_object::w_list_uses_int_storage(concrete_obj) {
        Some(1)
    } else if pyre_object::w_list_uses_float_storage(concrete_obj) {
        Some(2)
    } else {
        None
    }
}

/// Detect list strategy for setitem, checking value type compatibility.
/// listobject.py: int strategy requires int value, float strategy requires float.
///
/// Returns `(strategy_id, unbox_long)` where `unbox_long=true` indicates
/// the int-strategy path must use the W_LongObject fits_int unbox helper
/// (`listobject.py:2390 is_plain_int1` accepts W_IntObject and fits_int
/// W_LongObject; the lowering branches between them).
#[inline]
unsafe fn detect_list_setitem_strategy(
    concrete_obj: pyre_object::PyObjectRef,
    concrete_value: pyre_object::PyObjectRef,
) -> Option<(i64, bool)> {
    if pyre_object::w_list_uses_object_storage(concrete_obj) {
        Some((0, false))
    } else if pyre_object::w_list_uses_int_storage(concrete_obj)
        && pyre_object::is_plain_int1(concrete_value)
    {
        let unbox_long = pyre_object::pyobject::is_long(concrete_value);
        Some((1, unbox_long))
    } else if pyre_object::w_list_uses_float_storage(concrete_obj)
        && pyre_object::pyobject::is_float(concrete_value)
    {
        Some((2, false))
    } else {
        None
    }
}

/// Trace range iterator next: read fields → step sign guard → continues guard
/// → advance current → setfield.
///
/// RPython jitcode for range.__next__:
///   getfield(current) → getfield(remaining) → getfield(step) →
///   remaining > 0 guard → int_add(current,step) →
///   setfield(current, next) → setfield(remaining, remaining - 1)
///
/// Returns (current_opref, concrete_current) or None if should fall back.
#[inline]
pub fn generated_iter_next_value(
    frame: &mut crate::state::MIFrame,
    ctx: &mut majit_metainterp::TraceCtx,
    iter: majit_ir::OpRef,
    concrete_continues: bool,
    concrete_step: i64,
    concrete_current: i64,
) -> Option<(majit_ir::OpRef, i64)> {
    use majit_ir::OpCode;

    frame.guard_range_iter(ctx, iter);

    let current =
        crate::state::opimpl_getfield_gc_i(ctx, iter, crate::descr::range_iter_current_descr());
    let remaining =
        crate::state::opimpl_getfield_gc_i(ctx, iter, crate::descr::range_iter_remaining_descr());
    let step = crate::state::opimpl_getfield_gc_i(ctx, iter, crate::descr::range_iter_step_descr());
    let zero = ctx.const_int(0);

    // pyjitpl.py:1877 opimpl_goto_if_not: boolean guards.
    let continues = ctx.record_op(OpCode::IntGt, &[remaining, zero]);
    if concrete_continues {
        frame.generate_guard(ctx, OpCode::GuardTrue, &[continues]);
    } else {
        frame.generate_guard(ctx, OpCode::GuardFalse, &[continues]);
    }

    if !concrete_continues {
        return Some((zero, 0));
    }

    // The cursor advance mirrors `w_range_iter_next`'s `current + step` (a
    // plain wrapping add, not `ovfcheck`) — RPython `W_IntRangeIterator.next`
    // uses `int_add` for the cursor write too, so emit `IntAdd` rather than
    // `int_add_ovf`/`guard_no_overflow`.
    let next_current = ctx.record_op(OpCode::IntAdd, &[current, step]);
    let ri_descr = crate::descr::range_iter_current_descr();
    let ri_descr_idx = ri_descr.index();
    ctx.record_op_with_descr(OpCode::SetfieldGc, &[iter, next_current], ri_descr);
    let one = ctx.const_int(1);
    let next_remaining = ctx.record_op(OpCode::IntSub, &[remaining, one]);
    let remaining_descr = crate::descr::range_iter_remaining_descr();
    let remaining_descr_idx = remaining_descr.index();
    ctx.record_op_with_descr(OpCode::SetfieldGc, &[iter, next_remaining], remaining_descr);
    let next_current_value = majit_ir::Value::Int(concrete_current.wrapping_add(concrete_step));
    // Stamp the `IntAdd` result so downstream `box_value` consumers
    // see the runtime concrete (matches RPython Box propagation
    // through arithmetic ops).
    ctx.set_opref_concrete(next_current, next_current_value);
    ctx.heapcache_setfield_cached(iter, ri_descr_idx, next_current);
    ctx.heapcache_setfield_cached(iter, remaining_descr_idx, next_remaining);
    Some((current, concrete_current))
}
