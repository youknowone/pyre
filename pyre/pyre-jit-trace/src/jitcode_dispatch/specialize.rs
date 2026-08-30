//! Per-opcode specialization strategies: the `try_walker_*` entry points
//! that recognize a specializable shape (int/long/float arithmetic and
//! comparisons, attribute and method loads, container builds and subscript,
//! list-append, exception construction/raise, for-iter, slice, and the
//! module/name cell folds) and either fold or record a specialized trace,
//! returning `None` to fall through to the generic path.
//!
//! **Parity:** pyre-local trace-time folding. PyPy defers most
//! specialization to `optimizeopt/` (a separate later pass); pyre folds
//! during the walk instead. The fast-path shapes still mirror the
//! `opimpl_*` fast paths and `blackhole.py`'s `bhimpl_*` folds.
//!
//! Relocated verbatim from `jitcode_dispatch/mod.rs`. The shared walker
//! primitives these build on (unbox/box, guard emission, operand reads)
//! stay in `mod.rs`; the specialization opname arms stay in `handle` and
//! call into these entry points.

use super::*;
use rustpython_wtf8::Wtf8;

/// Replace an authentically executed builtin raise with ordinary trace
/// allocations.  The exception's stored args are the authority for both
/// wording and arity; keeping their objects as rooted trace constants avoids
/// re-deriving messages while still letting escape analysis virtualize the
/// exception and its args list when a handler discards them.
fn walker_recorded_builtin_raise_is_supported(
    exc: pyre_object::PyObjectRef,
    expected_kind: pyre_object::interp_exceptions::ExcKind,
) -> bool {
    if exc.is_null()
        || unsafe { !pyre_object::is_exception(exc) }
        || unsafe { pyre_object::interp_exceptions::w_exception_get_kind(exc) } != expected_kind
    {
        return false;
    }
    let args_storage = unsafe { pyre_object::interp_exceptions::w_exception_get_args_storage(exc) };
    if args_storage.is_null() || unsafe { !pyre_object::is_list(args_storage) } {
        return false;
    }
    let args_len = unsafe { pyre_object::w_list_len(args_storage) };
    (0..args_len).all(|index| {
        let Some(arg) = (unsafe { pyre_object::w_list_getitem(args_storage, index as i64) }) else {
            return false;
        };
        unsafe { pyre_object::is_str(arg) && pyre_object::is_exact_builtin_instance(arg) }
    })
}

fn walker_emit_recorded_builtin_raise<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    ec: OpRef,
    exc: pyre_object::PyObjectRef,
    expected_kind: pyre_object::interp_exceptions::ExcKind,
) -> DispatchOutcome {
    debug_assert!(walker_recorded_builtin_raise_is_supported(
        exc,
        expected_kind
    ));
    let _roots = pyre_object::gc_roots::push_roots();
    // Read every pinned value back out of its slot instead of reusing the
    // local that was handed to `pin_root`.  `pin_root` normalizes the address
    // it publishes once a second mutator has existed (`gc_roots.rs`
    // `RootScope::pin_root`), so past that point the caller's copy can still
    // name the pre-forwarding object while the slot names the live one — and
    // these values are baked into the trace as `ConstPtr`s, which outlive the
    // walk.  Same shape as the zip/tuple concrete-shadow build below.
    let exc_slot = pyre_object::gc_roots::shadow_stack_len();
    let exc = pyre_object::gc_roots::pin_root(exc);
    let args_storage = unsafe {
        pyre_object::interp_exceptions::w_exception_get_args_storage(
            pyre_object::gc_roots::shadow_stack_get(exc_slot),
        )
    };
    let args_storage_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(args_storage);
    let args_len = unsafe {
        pyre_object::w_list_len(pyre_object::gc_roots::shadow_stack_get(args_storage_slot))
    };
    let mut concrete_args = Vec::with_capacity(args_len);
    for index in 0..args_len {
        let arg = unsafe {
            pyre_object::w_list_getitem(
                pyre_object::gc_roots::shadow_stack_get(args_storage_slot),
                index as i64,
            )
        }
        .expect("recorded exception args were validated before trace emission");
        let arg_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(arg);
        concrete_args.push(unsafe { pyre_object::gc_roots::shadow_stack_get(arg_slot) });
    }
    let args = concrete_args
        .iter()
        .map(|&arg| ctx.trace_ctx.const_ref(arg as i64))
        .collect::<Vec<_>>();
    let args_list = crate::helpers::emit_object_list_inline(ctx.trace_ctx, &args);
    let list_w_class = pyre_object::get_instantiate(&pyre_object::pyobject::LIST_TYPE);
    let list_w_class = ctx.trace_ctx.const_ref(list_w_class as i64);
    let list_w_class_descr = crate::descr::list_w_class_descr();
    let list_w_class_index = list_w_class_descr.index();
    ctx.trace_ctx.record_op_with_descr(
        OpCode::SetfieldGc,
        &[args_list, list_w_class],
        list_w_class_descr,
    );
    ctx.trace_ctx
        .heapcache_setfield_cached(args_list, list_w_class_index, list_w_class);

    let class = pyre_object::interp_exceptions::lookup_exc_class_for_kind(expected_kind);
    let class = ctx.trace_ctx.const_ref(class as i64);
    let raised =
        crate::helpers::emit_exception_new_inline(ctx.trace_ctx, expected_kind, class, args_list);
    let exc_type =
        pyre_object::interp_exceptions::exc_kind_to_pytype(expected_kind) as *const _ as i64;
    ctx.trace_ctx
        .heap_cache_mut()
        .class_now_known(raised, exc_type);
    ctx.trace_ctx
        .set_opref_concrete(raised, majit_ir::Value::Ref(majit_ir::GcRef(exc as usize)));
    fbw_built_exc_insert(raised);

    let active = ctx.trace_ctx.record_op_with_descr(
        OpCode::GetfieldGcR,
        &[ec],
        crate::descr::ec_sys_exc_value_descr(),
    );
    ctx.trace_ctx.record_op_with_descr(
        OpCode::SetfieldGc,
        &[raised, active],
        crate::descr::w_exception_context_descr(expected_kind),
    );
    fbw_context_chained_insert(raised);
    // The authentic helper has published the raised exception but has not
    // chained it yet during the authoritative walk.  Mirror the recorded
    // field write so that iteration observes the same context as replay.
    let active_concrete = pyre_interpreter::eval::get_current_exception();
    if !active_concrete.is_null() {
        unsafe {
            pyre_object::interp_exceptions::w_exception_set_context(exc, active_concrete);
        }
    }

    fbw_count_executed_residual(false, true);
    let exc_concrete = ConcreteValue::Ref(exc);
    ctx.set_last_exc_value(raised, exc_concrete);
    ctx.fbw_mode.class_of_last_exc_is_const = true;
    majit_metainterp::blackhole::BH_LAST_EXC_VALUE.with(|c| c.set(exc as i64));
    DispatchOutcome::SubRaise {
        exc: raised,
        exc_concrete,
    }
}

/// #124: walker-native truth specialization for the `truth_fn` residual
/// (oopspec [`majit_ir::RuntimeHelperKind::Truth`]).  When the sole Ref operand
/// is a concrete boxed `W_IntObject` (excluding `W_BoolObject`, which shares
/// the `intval: i64` layout but carries a distinct `BOOL_TYPE` `ob_type`, so
/// the emitted `GUARD_CLASS INT` would not match it), unbox it
/// (`GUARD_CLASS INT` + `getfield intval`) and record `int_is_true`, stamping
/// the folded concrete truth.  Returns the raw truth `OpRef` on success;
/// `None` when the operand is not a concrete int — the caller then falls
/// through to the generic may-force residual, which runs `__bool__` /
/// `__len__`.
///
/// Declining a subclass on the *recorded* operand is not enough: `is_int`
/// and the `GUARD_CLASS` below both read `ob_type`, so a trace compiled from
/// an exact int still admits a subclass that arrives later.  The `w_class`
/// pin is what rejects it.
///
/// Eliding the `CALL_MAY_FORCE` here also removes its `GUARD_NOT_FORCED` /
/// `GUARD_NO_EXCEPTION`, whose kept-stack blackhole resume reads NULL peeled
/// outer-Label slots in the short-circuit value-context shape
/// (`(i % 7) and ...`).
pub(crate) fn try_walker_specialize_truth_int<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    operand: OpRef,
) -> Result<Option<OpRef>, DispatchError> {
    let Some(obj) = walker_concrete_ref_object(ctx, operand) else {
        return Ok(None);
    };
    let val = unsafe {
        // `is_int` reads `ob_type`, which an `int` subclass shares, so it alone
        // admits one here.  Two things then go wrong at once: the walk folds the
        // truth straight off the payload instead of running the subclass's
        // `__bool__`, and `walker_numeric_builtin_class` answers with the
        // canonical `int` — a `w_class` the recorded operand does not carry, so
        // the pin below becomes a guard that fails on the very value that
        // recorded it.  Decline before unboxing, as `walker_unary_int_operand`
        // does; `walker_numeric_builtin_class` documents this gate as its
        // precondition.
        if !pyre_object::is_int(obj)
            || pyre_object::is_bool(obj)
            || !pyre_object::is_exact_builtin_instance(obj)
        {
            return Ok(None);
        }
        pyre_object::w_int_get_value(obj)
    };
    let int_type_addr = &pyre_object::pyobject::INT_TYPE as *const _ as i64;
    let raw = walker_unbox_int(ctx, op_pc, operand, int_type_addr)?;
    walker_guard_exact_w_class(ctx, op_pc, operand, walker_numeric_builtin_class(obj))?;
    let truth = ctx.trace_ctx.record_op(OpCode::IntIsTrue, &[raw]);
    ctx.trace_ctx
        .set_opref_concrete(truth, majit_ir::Value::Int((val != 0) as i64));
    Ok(Some(truth))
}

/// Truth specialization for a concrete `W_BoolObject` operand — the sibling
/// [`try_walker_specialize_truth_int`] declines it, because it emits
/// `GUARD_CLASS INT` and a bool carries `BOOL_TYPE`.  Same `intval: i64`
/// layout, so only the guarded class constant differs; `is_true` on the
/// unboxed field is `W_BoolObject.is_true`'s `self.intval != 0`.
///
/// This is the shape every `if a == b:` reaches: `COMPARE_OP` leaves a boxed
/// bool the following `TO_BOOL` / `POP_JUMP_IF_*` tests, so without this arm a
/// comparison costs two `CALL_MAY_FORCE`s and two force/exception guard pairs
/// instead of one call and a field read.  When the comparison itself already
/// specialized, its result is the `space.newbool` singleton and both the class
/// guard and the field read below fold off that constant.
pub(crate) fn try_walker_specialize_truth_bool<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    operand: OpRef,
) -> Result<Option<OpRef>, DispatchError> {
    let Some(obj) = walker_concrete_ref_object(ctx, operand) else {
        return Ok(None);
    };
    let val = unsafe {
        if !pyre_object::is_bool(obj) {
            return Ok(None);
        }
        pyre_object::w_int_get_value(obj)
    };
    let bool_type_addr = &pyre_object::pyobject::BOOL_TYPE as *const _ as i64;
    let raw = walker_unbox_int(ctx, op_pc, operand, bool_type_addr)?;
    let truth = ctx.trace_ctx.record_op(OpCode::IntIsTrue, &[raw]);
    ctx.trace_ctx
        .set_opref_concrete(truth, majit_ir::Value::Int((val != 0) as i64));
    Ok(Some(truth))
}

/// #61: walker-native identity fold for the `UNARY_POSITIVE` residual
/// (oopspec [`majit_ir::RuntimeHelperKind::UnaryPositive`]).  The object-space
/// `pos` on an exact int returns the operand unchanged, so a concrete non-bool
/// `W_IntObject` operand folds to the operand box itself behind the same guard
/// prefix the truth / binary int folds emit (a low-bit tag test for a tagged
/// immediate, `GUARD_CLASS INT` for a heap box).  The unboxed raw is discarded
/// (DCE): the result is the box, not its `intval`.
///
/// Returns `Ok(Some(()))` when the fold was emitted (caller returns
/// `Continue`); `Ok(None)` for a bool (`+True` is int `1`, not identity), a
/// numeric subclass, or a non-int operand — the caller then falls through to
/// the generic `CallMayForce` residual so a subclass / user `__pos__` still
/// runs.
pub(crate) fn try_walker_specialize_unary_positive_int<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    operand: OpRef,
    dst: usize,
    dst_bank: char,
) -> Result<Option<()>, DispatchError> {
    // `+x` is identity only for an EXACT builtin int.  A bool shares the
    // `intval` but `+True` is int `1`, not identity, and a numeric subclass
    // must reach its own `__pos__` rather than have the operand forwarded.
    let Some((_, x_class)) = walker_unary_int_operand(ctx, operand) else {
        return Ok(None);
    };
    let int_type_addr = &pyre_object::pyobject::INT_TYPE as *const _ as i64;
    // Emit the guard prefix (`GUARD_CLASS INT` / tag test) so a later non-int
    // arrival deopts; the returned raw is unused because the result is the
    // operand box itself.
    let _ = walker_unbox_int(ctx, op_pc, operand, int_type_addr)?;
    walker_guard_exact_w_class(ctx, op_pc, operand, x_class)?;
    write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, operand)?;
    Ok(Some(()))
}

/// Shared gate for the `UNARY_POSITIVE` / `UNARY_NEGATIVE` / `UNARY_INVERT` int
/// folds: the operand must be a concrete EXACT builtin non-bool `W_IntObject`.
/// A bool unboxes through its own `&BOOL_TYPE` guard (declined here for
/// simplicity — `+True` / `-True` / `~True` stay on the residual).
///
/// Returns the concrete `intval` and the canonical `int` type object the caller
/// must pin with [`walker_guard_exact_w_class`].  `is_exact_builtin_instance`
/// only settles the operand the trace RECORDED; a numeric subclass keeps the
/// builtin `ob_type`, so the `GUARD_CLASS INT` the fold emits does not stop one
/// from entering the trace later and being answered by the fold instead of its
/// own `__neg__` / `__invert__` / `__pos__`.  Pinning `w_class` is what makes
/// that arrival side-exit, and the operand that carries the null spelling of
/// "exact builtin" has no value to pin, so it declines — the same shape the
/// long folds use (`walker_exact_builtin_class` + guard).
fn walker_unary_int_operand<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    operand: OpRef,
) -> Option<(i64, pyre_object::PyObjectRef)> {
    let obj = walker_concrete_ref_object(ctx, operand)?;
    // SAFETY: `obj` is a live concrete `PyObjectRef` from the walker shadow.
    unsafe {
        if !pyre_object::is_int(obj)
            || pyre_object::is_bool(obj)
            || !pyre_object::is_exact_builtin_instance(obj)
        {
            return None;
        }
        let class = walker_exact_builtin_class(obj)?;
        Some((pyre_object::w_int_get_value(obj), class))
    }
}

/// The `W_LongObject.value` payload of a concrete long, read the way the folds
/// that pass a payload to an `rbigint` helper need it.
///
/// # Safety
/// `obj` must be a live concrete `W_LongObject` from the walker shadow.
unsafe fn long_payload_of(obj: pyre_object::PyObjectRef) -> i64 {
    unsafe { *((obj as *const u8).add(pyre_object::longobject::LONG_VALUE_OFFSET) as *const i64) }
}

/// Record the `getfield_gc_r` that reads a long operand's `value` payload.
/// A box the same trace built with [`crate::helpers::emit_box_long_inline`]
/// answers this out of the heap cache, so the read costs nothing and the box
/// keeps no reason to escape.
fn walker_read_long_payload<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    boxed: OpRef,
    concrete_payload: i64,
) -> OpRef {
    let payload = ctx.trace_ctx.record_op_with_descr(
        OpCode::GetfieldGcR,
        &[boxed],
        crate::descr::long_value_descr(),
    );
    ctx.trace_ctx.set_opref_concrete(
        payload,
        majit_ir::Value::Ref(majit_ir::GcRef(concrete_payload as usize)),
    );
    payload
}

/// `intobject.py _make_ovf2long`: the tail every int arithmetic fold shares
/// once its own guard has pinned the promoting branch — `GUARD_OVERFLOW` for
/// the `BINARY_OP` arm, `GUARD_VALUE` on the operand for unary negate. The tail
/// is the elidable raw-int bigint helper (`rbigint.py:717/788/873`) under
/// `EF_ELIDABLE_OR_MEMORYERROR`, then the inline `W_LongObject` box around the
/// payload it returns. `payload_fn` takes the two machine ints in
/// `(raw, concrete)` pairs, which is also the shape the concrete-args vector
/// wants.
///
/// The box needs no preceding fits_int guard. `newlong_from_rbigint`
/// (objspace.py:316-320) demotes through `rbigint.toint()`, whose
/// `numdigits() > MAX_DIGITS_THAT_CAN_FIT_IN_INT` test (rbigint.py) that
/// guard already answers: the helper is the *exact* int-pair sum / difference /
/// product, so a value that just overflowed a machine int cannot fit one back.
/// The same fold is what lets `try_walker_specialize_binary_op_long_int_pow`
/// skip its result-fits guard.
fn walker_emit_ovf2long_box<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    payload_fn: *const (),
    lhs: (OpRef, i64),
    rhs: (OpRef, i64),
    boxed_result_i64: i64,
) -> Result<OpRef, DispatchError> {
    let (lhs_raw, la) = lhs;
    let (rhs_raw, rb) = rhs;
    let payload_concrete =
        unsafe { long_payload_of(boxed_result_i64 as usize as pyre_object::PyObjectRef) };
    let concrete_args = [
        majit_ir::Value::Int(payload_fn as usize as i64),
        majit_ir::Value::Int(la),
        majit_ir::Value::Int(rb),
    ];
    let payload = ctx.trace_ctx.call_typed_with_effect_pure_can_raise(
        OpCode::CallR,
        payload_fn,
        &[lhs_raw, rhs_raw],
        &[majit_ir::Type::Int, majit_ir::Type::Int],
        majit_ir::Type::Ref,
        majit_metainterp::ELIDABLE_OR_MEMERROR_EFFECT_INFO,
        &concrete_args,
        majit_ir::Value::Ref(majit_ir::GcRef(payload_concrete as usize)),
    );
    ctx.trace_ctx.set_opref_concrete(
        payload,
        majit_ir::Value::Ref(majit_ir::GcRef(payload_concrete as usize)),
    );
    if payload.inline_const_to_value().is_none() {
        walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardNoException, &[])?;
    }
    let result = crate::helpers::emit_box_long_inline(
        ctx.trace_ctx,
        payload,
        crate::descr::w_long_size_descr(),
        crate::descr::long_value_descr(),
    );
    ctx.trace_ctx.set_opref_concrete(
        result,
        majit_ir::Value::Ref(majit_ir::GcRef(boxed_result_i64 as usize)),
    );
    Ok(result)
}

/// #61: walker-native int specialization for the `UNARY_NEGATIVE` residual
/// (oopspec [`majit_ir::RuntimeHelperKind::UnaryNegative`]).  `-x` on an exact
/// int is `0 - x`; the object-space `neg` promotes only `-INT_MIN` to a
/// `W_LongObject` (`intobject.py` `descr_neg` → `_make_ovf2long`).  Since
/// majit has no overflow-checked unary negate, the fold expresses `-x` as
/// `IntSubOvf(0, x)` behind a `GUARD_CLASS INT`, reusing the binary-sub
/// overflow discipline in both directions: a record value other than `INT_MIN`
/// emits `GUARD_NO_OVERFLOW` so an `INT_MIN` arrival on the reused trace deopts
/// rather than wrapping back to `INT_MIN`, and a record value of `INT_MIN`
/// pins the operand with `GUARD_VALUE` and takes the same `_make_ovf2long` tail
/// the `BINARY_OP` overflow arm takes, so the `2**63` long is built from the
/// elidable bigint helper instead of the `CallMayForce` residual.
///
/// Returns `Ok(Some(()))` when the fold was emitted (caller returns
/// `Continue`); `Ok(None)` for a bool / subclass / non-int operand, when the
/// residual result box is unavailable, or when an `INT_MIN` operand did not
/// produce the promoted `W_LongObject` the payload read expects.
pub(crate) fn try_walker_specialize_unary_negative_int<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    operand: OpRef,
    allboxes: &[OpRef],
    call_descr: &dyn majit_ir::descr::CallDescr,
    dst: usize,
    dst_bank: char,
) -> Result<Option<()>, DispatchError> {
    let Some((x, x_class)) = walker_unary_int_operand(ctx, operand) else {
        return Ok(None);
    };
    let Some(boxed_result_i64) = walker_execute_may_force_boxed(ctx, allboxes, call_descr) else {
        return Ok(None);
    };
    // `0 - INT_MIN` is the one operand `descr_neg` promotes.
    let overflows = x == i64::MIN;
    if overflows {
        let boxed_result_obj = boxed_result_i64 as usize as pyre_object::PyObjectRef;
        if boxed_result_obj == pyre_object::PY_NULL
            || !unsafe { pyre_object::is_long(boxed_result_obj) }
        {
            return Ok(None);
        }
    }
    let int_type_addr = &pyre_object::pyobject::INT_TYPE as *const _ as i64;
    let x_raw = walker_unbox_int(ctx, op_pc, operand, int_type_addr)?;
    walker_guard_exact_w_class(ctx, op_pc, operand, x_class)?;
    let zero_raw = ctx.trace_ctx.const_int(0);
    let boxed = if overflows {
        // `0 - x` overflows an i64 for exactly one operand, so "the negate
        // promoted" and "the operand is INT_MIN" name the same set: guarding
        // the value admits what `GUARD_OVERFLOW` would and nothing more. The
        // value form is the one the tail can use — with `x_raw` constant the
        // elidable bigint call folds to the `2**63` payload it returned while
        // recording instead of running once per iteration. This is the
        // `guard_value` spelling the version-tag promotes already use.
        let int_min = ctx.trace_ctx.const_int(i64::MIN);
        walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardValue, &[x_raw, int_min])?;
        walker_emit_ovf2long_box(
            ctx,
            op_pc,
            pyre_object::longobject::jit_bigint_sub_int_int as *const (),
            (zero_raw, 0),
            (x_raw, x),
            boxed_result_i64,
        )?
    } else {
        let result_value = 0i64.wrapping_sub(x);
        let raw_result = ctx
            .trace_ctx
            .record_op(OpCode::IntSubOvf, &[zero_raw, x_raw]);
        ctx.trace_ctx
            .set_opref_concrete(raw_result, majit_ir::Value::Int(result_value));
        walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardNoOverflow, &[])?;
        let boxed = walker_box_int(ctx, op_pc, raw_result, result_value)?;
        ctx.trace_ctx
            .set_opref_concrete(boxed, box_int_concrete(result_value, boxed_result_i64));
        boxed
    };
    write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, boxed)?;
    Ok(Some(()))
}

/// #57: walker-native speculative int specialization for the `BINARY_OP`
/// helper residual_call (oopspec `BinaryOp`).  Re-derives
/// the former int fast path's structure (`guard_class` + `getfield_gc_i` per
/// operand, `int_OP_ovf` + `guard_no_overflow`, `wrapint`) walker-native rather
/// than calling back into the retired trait path (which would alias the
/// reborrowed sym slices and emit `MIFrame`-style snapshots inconsistent with
/// the walker model).
///
/// The concrete boxed result is obtained from the same
/// `execute_residual_call` path the generic leg uses, so
/// `concrete_registers_r[dst]` holds the authentic runtime `W_IntObject`.
///
/// Returns `Ok(Some(outcome))` when the specialization was emitted; the
/// outcome is `Continue` for a value arm or `SubRaise` for a zero divisor.
/// `Ok(None)` means the operator is deferred
/// (FloorDiv / Mod / Shift / TrueDiv / Power / Subscr), the operands are
/// not both concrete `W_IntObject`, or an unsupported helper arm is reached — the caller
/// then falls through to the generic `CallMayForce` record so the
/// Python-level `__op__` semantics are preserved.
/// The raw machine int of an int/bool operand, for the specialized IR: guard
/// the operand's exact class, then load `intval` out of the box.
///
/// A bool `space.newbool` produced this same walk arrives as the prebuilt
/// `w_True` / `w_False` singleton behind its own truth guard, so both the class
/// guard and the `intval` load read a constant and fold away — no side table
/// has to reconnect the box to the truth it was built from.
fn walker_int_operand_raw<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    operand: OpRef,
    operand_obj: pyre_object::PyObjectRef,
    type_addr: i64,
    intval_descr: majit_ir::DescrRef,
) -> Result<OpRef, DispatchError> {
    let raw = walker_unbox_int_typed(ctx, op_pc, operand, type_addr, intval_descr)?;
    walker_guard_exact_w_class(
        ctx,
        op_pc,
        operand,
        walker_numeric_builtin_class(operand_obj),
    )?;
    Ok(raw)
}

pub(crate) fn try_walker_specialize_binary_op_int<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    op_tag: i64,
    r_args: &[OpRef],
    allboxes: &[OpRef],
    call_descr: &dyn majit_ir::descr::CallDescr,
    dst: usize,
    dst_bank: char,
) -> Result<Option<DispatchOutcome>, DispatchError> {
    if !ctx.is_authoritative_executor || r_args.len() != 2 || dst_bank != 'r' {
        return Ok(None);
    }
    let Some(bin_op) = pyre_interpreter::runtime_ops::binary_op_from_tag(op_tag) else {
        return Ok(None);
    };
    use pyre_interpreter::bytecode::BinaryOperator;
    // INT_BINOP_TABLE → (OpCode, has_overflow, needs_concrete_check).
    // Defer TrueDivide (int/int → float, separate helper) / Power /
    // Subscr to the generic leg (`_ => None`).
    let (op_code, has_overflow, needs_check) = match bin_op {
        BinaryOperator::Add | BinaryOperator::InplaceAdd => (OpCode::IntAddOvf, true, false),
        BinaryOperator::Subtract | BinaryOperator::InplaceSubtract => {
            (OpCode::IntSubOvf, true, false)
        }
        BinaryOperator::Multiply | BinaryOperator::InplaceMultiply => {
            (OpCode::IntMulOvf, true, false)
        }
        BinaryOperator::FloorDivide | BinaryOperator::InplaceFloorDivide => {
            (OpCode::IntFloorDiv, false, true)
        }
        BinaryOperator::Remainder | BinaryOperator::InplaceRemainder => {
            (OpCode::IntMod, false, true)
        }
        BinaryOperator::And | BinaryOperator::InplaceAnd => (OpCode::IntAnd, false, false),
        BinaryOperator::Or | BinaryOperator::InplaceOr => (OpCode::IntOr, false, false),
        BinaryOperator::Xor | BinaryOperator::InplaceXor => (OpCode::IntXor, false, false),
        BinaryOperator::Lshift | BinaryOperator::InplaceLshift => (OpCode::IntLshift, false, true),
        BinaryOperator::Rshift | BinaryOperator::InplaceRshift => (OpCode::IntRshift, false, true),
        _ => return Ok(None),
    };

    // boolobject.py descr_and/or/xor: when both operands are bool the
    // And/Or/Xor result is a bool (`space.newbool`), not an int.  The op runs
    // on the shared `intval` as for ints; only the boxing differs (picked
    // below).  `walker_concrete_ref_object` reads the same source as
    // `walker_int_specialization_operands`, so the flag stays consistent.
    let result_is_bool = matches!(op_code, OpCode::IntAnd | OpCode::IntOr | OpCode::IntXor)
        && match (
            walker_concrete_ref_object(ctx, r_args[0]),
            walker_concrete_ref_object(ctx, r_args[1]),
        ) {
            (Some(l), Some(r)) => unsafe { pyre_object::is_bool(l) && pyre_object::is_bool(r) },
            _ => false,
        };

    // Inspect the operands before executing the authentic helper.  A raising
    // zero-divisor arm needs the helper-produced exception as its concrete
    // shadow, but records no helper call in the trace.
    let Some((lhs, rhs, lhs_obj, rhs_obj, la, rb)) =
        walker_int_specialization_input_operands(ctx, r_args)
    else {
        return Ok(None);
    };

    if matches!(op_code, OpCode::IntFloorDiv | OpCode::IntMod) && rb == 0 {
        let Some(Err(exc_i64)) = walker_execute_may_force_boxed_outcome(ctx, allboxes, call_descr)
        else {
            return Ok(None);
        };
        // The helper publishes through both the blackhole cell (drained by
        // `execute_residual_call`) and the backend exception cells.  The
        // latter belong to compiled execution; drain the trace-time publish
        // before the walk continues into the Python handler, exactly as the
        // generic residual executor's Err arm does.
        if let Some(cb) = crate::callbacks::try_get() {
            (cb.drain_backend_jit_exc)();
        }
        let exc = exc_i64 as usize as pyre_object::PyObjectRef;
        let kind = pyre_object::interp_exceptions::ExcKind::ZeroDivisionError;
        if !walker_recorded_builtin_raise_is_supported(exc, kind) {
            return Ok(None);
        }
        let Some(ec) = walker_ensure_execution_context(ctx) else {
            return Ok(None);
        };

        // Commit to the raising arm only after every decline.  Exact-class
        // guards preserve builtin dispatch; GuardTrue(rhs == 0) is the branch
        // guard a bridge can invert when the divisor changes mid-loop.
        let (lhs_type, lhs_descr) = crate::state::int_or_bool_unbox_type_descr(lhs_obj);
        let (rhs_type, rhs_descr) = crate::state::int_or_bool_unbox_type_descr(rhs_obj);
        let _lhs_raw = walker_unbox_int_typed(ctx, op_pc, lhs, lhs_type, lhs_descr)?;
        walker_guard_exact_w_class(ctx, op_pc, lhs, walker_numeric_builtin_class(lhs_obj))?;
        let rhs_raw = walker_unbox_int_typed(ctx, op_pc, rhs, rhs_type, rhs_descr)?;
        walker_guard_exact_w_class(ctx, op_pc, rhs, walker_numeric_builtin_class(rhs_obj))?;
        let rhs_zero = walker_int_eq_const(ctx, rhs_raw, 0, 1);
        walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardTrue, &[rhs_zero])?;

        return Ok(Some(walker_emit_recorded_builtin_raise(ctx, ec, exc, kind)));
    }

    let Some(boxed_result_i64) = walker_execute_may_force_boxed(ctx, allboxes, call_descr) else {
        return Ok(None);
    };

    // intobject.py range validation (mirror the former int fast path's
    // needs_concrete_check): bail to the generic leg when the bare-IR-op
    // emission would be unsound (zero / INT_MIN-overflow divisor, oversized
    // / overflowing shift); large right-shift folds to a const.
    if needs_check {
        match op_code {
            OpCode::IntFloorDiv | OpCode::IntMod => {
                if la == i64::MIN && rb == -1 {
                    return Ok(None);
                }
            }
            OpCode::IntLshift => {
                // Don't specialize int `<<`: route to the generic (residual
                // BINARY_OP) leg, which carries the full intobject.py
                // descr_lshift semantics (promote to bignum on overflow, raise
                // ValueError on a negative count). A bare walker-native IntLshift
                // would be wrong — the trace is reused for any operands and x86
                // SHL masks the count mod 64 — and a *guarded* specialization
                // (range + round-trip guards, bail to bignum) crashes the
                // cranelift backend: when the lshift result is the loop variable
                // its box alternates small-int / bignum across the guard's
                // bridge boundary, and that trips a cranelift bridge bug (works
                // on dynasm). The generic leg handles the alternation correctly
                // on both backends.
                return Ok(None);
            }
            OpCode::IntRshift => {
                // A count >= LONG_BIT (or negative) folds to 0/-1 in
                // intobject.py, but that fold would be baked into the
                // reused trace and be wrong for an in-range count; route it to
                // the generic leg instead. An in-range recorded count is
                // specialized below behind a runtime range guard.
                let Ok(shift) = u32::try_from(rb) else {
                    return Ok(None);
                };
                if shift >= i64::BITS {
                    return Ok(None);
                }
            }
            _ => {}
        }
    }

    // pyjitpl.py handle_possible_overflow_error follows the concrete
    // Add/Sub/Mul outcome. The overflowing arm mirrors intobject.py:494
    // _make_ovf2long: guard_overflow, call the elidable raw-int bigint helper
    // (rbigint.py:717/788/873), and inline the W_LongObject box instead of
    // falling through to the generic CallMayForceR BINARY_OP leg.
    let overflows = has_overflow
        && match op_code {
            OpCode::IntAddOvf => la.checked_add(rb).is_none(),
            OpCode::IntSubOvf => la.checked_sub(rb).is_none(),
            OpCode::IntMulOvf => la.checked_mul(rb).is_none(),
            _ => false,
        };
    if overflows {
        let boxed_result_obj = boxed_result_i64 as usize as pyre_object::PyObjectRef;
        if boxed_result_obj == pyre_object::PY_NULL
            || !unsafe { pyre_object::is_long(boxed_result_obj) }
        {
            return Ok(None);
        }
    }

    // --- emit the specialized IR (walker-native) ---
    // bool and int share `intval`; guard each operand against its own vtable
    // (BOOL_TYPE / INT_TYPE) so a bool unboxes through its own class.
    let (lhs_type, lhs_descr) = crate::state::int_or_bool_unbox_type_descr(lhs_obj);
    let (rhs_type, rhs_descr) = crate::state::int_or_bool_unbox_type_descr(rhs_obj);
    let lhs_raw = walker_int_operand_raw(ctx, op_pc, lhs, lhs_obj, lhs_type, lhs_descr)?;
    let rhs_raw = walker_int_operand_raw(ctx, op_pc, rhs, rhs_obj, rhs_type, rhs_descr)?;
    if overflows {
        let concrete_value = match op_code {
            OpCode::IntAddOvf => la.wrapping_add(rb),
            OpCode::IntSubOvf => la.wrapping_sub(rb),
            OpCode::IntMulOvf => la.wrapping_mul(rb),
            _ => unreachable!("overflow arm requires Add/Sub/Mul"),
        };
        let raw_result = ctx.trace_ctx.record_op(op_code, &[lhs_raw, rhs_raw]);
        ctx.trace_ctx
            .set_opref_concrete(raw_result, majit_ir::Value::Int(concrete_value));
        walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardOverflow, &[])?;

        let payload_fn = match op_code {
            OpCode::IntAddOvf => pyre_object::longobject::jit_bigint_add_int_int as *const (),
            OpCode::IntSubOvf => pyre_object::longobject::jit_bigint_sub_int_int as *const (),
            OpCode::IntMulOvf => pyre_object::longobject::jit_bigint_mul_int_int as *const (),
            _ => unreachable!("overflow arm requires Add/Sub/Mul"),
        };
        let result = walker_emit_ovf2long_box(
            ctx,
            op_pc,
            payload_fn,
            (lhs_raw, la),
            (rhs_raw, rb),
            boxed_result_i64,
        )?;
        write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, result)?;
        return Ok(Some(DispatchOutcome::Continue));
    }
    let (raw_result, concrete_value) = match op_code {
        OpCode::IntFloorDiv | OpCode::IntMod => {
            walker_emit_int_div_domain_guards(ctx, op_pc, lhs_raw, rhs_raw, la, rb)?;
            walker_emit_int_py_div_or_mod(
                ctx,
                lhs_raw,
                rhs_raw,
                la,
                rb,
                op_code == OpCode::IntFloorDiv,
            )
        }
        OpCode::IntRshift => {
            // The machine SAR masks the count mod 64, so guard the count into
            // [0, LONG_BIT) — a reused trace bails rather than shifting by
            // `count & 63`. (The recorded count is < LONG_BIT here: a count
            // >= LONG_BIT const-folds to 0/-1 in the needs_check block above.)
            let in_range = walker_uint_lt_const(ctx, rhs_raw, i64::BITS as i64, 1);
            walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardTrue, &[in_range])?;
            let r = ctx
                .trace_ctx
                .record_op(OpCode::IntRshift, &[lhs_raw, rhs_raw]);
            (r, majit_metainterp::eval_binop_i(OpCode::IntRshift, la, rb))
        }
        _ => {
            let r = ctx.trace_ctx.record_op(op_code, &[lhs_raw, rhs_raw]);
            (r, majit_metainterp::eval_binop_i(op_code, la, rb))
        }
    };
    ctx.trace_ctx
        .set_opref_concrete(raw_result, majit_ir::Value::Int(concrete_value));
    if has_overflow {
        walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardNoOverflow, &[])?;
    }
    // A both-bool bitwise result boxes via `space.newbool` (boolobject.py:
    // 74-76) so it keeps the bool type; `boxed_result_i64` is already the
    // authentic W_Bool the forced residual produced.
    let boxed = if result_is_bool {
        match walker_newbool_guarded(ctx, op_pc, raw_result, concrete_value != 0, dst_bank)? {
            Some(boxed) => boxed,
            None => {
                let boxed = crate::helpers::emit_trace_bool_value_from_truth(
                    ctx.trace_ctx,
                    raw_result,
                    false,
                );
                ctx.trace_ctx.set_opref_concrete(
                    boxed,
                    majit_ir::Value::Ref(majit_ir::GcRef(boxed_result_i64 as usize)),
                );
                boxed
            }
        }
    } else {
        let boxed = walker_box_int(ctx, op_pc, raw_result, concrete_value)?;
        ctx.trace_ctx
            .set_opref_concrete(boxed, box_int_concrete(concrete_value, boxed_result_i64));
        boxed
    };
    write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, boxed)?;
    Ok(Some(DispatchOutcome::Continue))
}

/// rint.py `_ovf_zer` guards for a machine-int division: `int_eq(rhs,0)` →
/// `guard_false` plus `(lhs==INT_MIN)&(rhs==-1)` → `guard_false`.  Both must
/// precede the elidable `ll_int_py_div` / `ll_int_py_mod` call so a re-used
/// trace bails before the helper's `wrapping_div` / `wrapping_rem` returns a
/// wrap value.  A `divmod` site shares one guard pair across both halves.
fn walker_emit_int_div_domain_guards<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    lhs_raw: OpRef,
    rhs_raw: OpRef,
    la: i64,
    rb: i64,
) -> Result<(), DispatchError> {
    let rhs_zero = walker_int_eq_const(ctx, rhs_raw, 0, (rb == 0) as i64);
    walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardFalse, &[rhs_zero])?;
    let lhs_is_min = walker_int_eq_const(ctx, lhs_raw, i64::MIN, (la == i64::MIN) as i64);
    let rhs_is_neg_one = walker_int_eq_const(ctx, rhs_raw, -1, (rb == -1) as i64);
    let ovf_both = ctx
        .trace_ctx
        .record_op(OpCode::IntAnd, &[lhs_is_min, rhs_is_neg_one]);
    ctx.trace_ctx.set_opref_concrete(
        ovf_both,
        majit_ir::Value::Int(((la == i64::MIN) as i64) & ((rb == -1) as i64)),
    );
    walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardFalse, &[ovf_both])
}

/// jtransform.py `OS_INT_PY_DIV` / `OS_INT_PY_MOD` elidable residual call
/// (`call_typed_with_effect_pure` → `CallI` patched via
/// `record_result_of_call_pure`), returning the result op and its recorded
/// value.  The caller must have emitted
/// [`walker_emit_int_div_domain_guards`] over the same operand pair first.
fn walker_emit_int_py_div_or_mod<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    lhs_raw: OpRef,
    rhs_raw: OpRef,
    la: i64,
    rb: i64,
    is_div: bool,
) -> (OpRef, i64) {
    let (func_ptr, effect_info, concrete_result) = if is_div {
        (
            majit_metainterp::blackhole::ll_int_py_div as *const (),
            majit_metainterp::INT_PY_DIV_EFFECT_INFO,
            majit_metainterp::blackhole::ll_int_py_div(la, rb),
        )
    } else {
        (
            majit_metainterp::blackhole::ll_int_py_mod as *const (),
            majit_metainterp::INT_PY_MOD_EFFECT_INFO,
            majit_metainterp::blackhole::ll_int_py_mod(la, rb),
        )
    };
    let r = ctx.trace_ctx.call_typed_with_effect_pure(
        OpCode::CallI,
        func_ptr,
        &[lhs_raw, rhs_raw],
        &[majit_ir::Type::Int, majit_ir::Type::Int],
        majit_ir::Type::Int,
        effect_info,
        &[
            majit_ir::Value::Int(func_ptr as usize as i64),
            majit_ir::Value::Int(la),
            majit_ir::Value::Int(rb),
        ],
        majit_ir::Value::Int(concrete_result),
    );
    ctx.trace_ctx
        .set_opref_concrete(r, majit_ir::Value::Int(concrete_result));
    (r, concrete_result)
}

/// Walker-native mixed `W_LongObject` / `W_IntObject` arithmetic
/// specialization for the `BINARY_OP` helper residual_call.
///
/// This is the trace shape of
/// `pypy/objspace/std/longobject.py:_make_generic_descr_binop` and
/// `descr_sub`: add/mul/and/or/xor select `rbigint.int_*` for either operand
/// order (the operations are commutative), while sub selects `int_sub` only
/// for `long - int`.  The opposite subtraction follows upstream's
/// `descr_rsub` bigint/bigint path and is deliberately left to the generic
/// record.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_walker_specialize_binary_op_long_int<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    op_tag: i64,
    r_args: &[OpRef],
    allboxes: &[OpRef],
    call_descr: &dyn majit_ir::descr::CallDescr,
    dst: usize,
    dst_bank: char,
) -> Result<Option<()>, DispatchError> {
    if !ctx.is_authoritative_executor || r_args.len() != 2 || dst_bank != 'r' {
        return Ok(None);
    }
    use pyre_interpreter::bytecode::BinaryOperator;
    use pyre_interpreter::objspace::descroperation as desc;
    type PayloadFn = extern "C" fn(i64, i64) -> pyre_object::longobject::JitBigIntResult;
    let (helper, commutative): (PayloadFn, bool) =
        match pyre_interpreter::runtime_ops::binary_op_from_tag(op_tag) {
            Some(BinaryOperator::Add | BinaryOperator::InplaceAdd) => {
                (desc::jit_bigint_int_add, true)
            }
            Some(BinaryOperator::Subtract | BinaryOperator::InplaceSubtract) => {
                (desc::jit_bigint_int_sub, false)
            }
            Some(BinaryOperator::Multiply | BinaryOperator::InplaceMultiply) => {
                (desc::jit_bigint_int_mul, true)
            }
            Some(BinaryOperator::And | BinaryOperator::InplaceAnd) => {
                (desc::jit_bigint_int_and, true)
            }
            Some(BinaryOperator::Or | BinaryOperator::InplaceOr) => (desc::jit_bigint_int_or, true),
            Some(BinaryOperator::Xor | BinaryOperator::InplaceXor) => {
                (desc::jit_bigint_int_xor, true)
            }
            _ => return Ok(None),
        };
    let (lhs_obj, rhs_obj) = match (
        walker_concrete_ref_object(ctx, r_args[0]),
        walker_concrete_ref_object(ctx, r_args[1]),
    ) {
        (Some(lhs), Some(rhs)) => (lhs, rhs),
        _ => return Ok(None),
    };
    if pyre_object::tagged_int::CAN_BE_TAGGED
        && (pyre_object::tagged_int::is_tagged_int(lhs_obj)
            || pyre_object::tagged_int::is_tagged_int(rhs_obj))
    {
        return Ok(None);
    }
    let (long, int, long_obj, int_obj) = unsafe {
        if pyre_object::is_long(lhs_obj) && pyre_object::is_int(rhs_obj) {
            (r_args[0], r_args[1], lhs_obj, rhs_obj)
        } else if commutative && pyre_object::is_int(lhs_obj) && pyre_object::is_long(rhs_obj) {
            (r_args[1], r_args[0], rhs_obj, lhs_obj)
        } else {
            return Ok(None);
        }
    };
    let (Some(long_class), Some(int_class)) = (unsafe {
        (
            walker_exact_builtin_class(long_obj),
            walker_exact_builtin_class(int_obj),
        )
    }) else {
        return Ok(None);
    };
    let int_value = unsafe { pyre_object::w_int_get_value(int_obj) };

    let Some(boxed_result_i64) = walker_execute_may_force_boxed(ctx, allboxes, call_descr) else {
        return Ok(None);
    };
    let boxed_result_obj = boxed_result_i64 as usize as pyre_object::PyObjectRef;
    if boxed_result_obj == pyre_object::PY_NULL
        || unsafe {
            pyre_object::is_int(boxed_result_obj) || !pyre_object::is_long(boxed_result_obj)
        }
    {
        return Ok(None);
    }
    let raw_concrete = unsafe {
        *((boxed_result_obj as *const u8).add(pyre_object::longobject::LONG_VALUE_OFFSET)
            as *const i64)
    };

    let long_type_addr = &pyre_object::pyobject::LONG_TYPE as *const _ as i64;
    walker_guard_class(ctx, op_pc, long, long_type_addr)?;
    walker_guard_exact_w_class(ctx, op_pc, long, long_class)?;
    let (int_type, int_descr) = crate::state::int_or_bool_unbox_type_descr(int_obj);
    let int_raw = walker_unbox_int_typed(ctx, op_pc, int, int_type, int_descr)?;
    walker_guard_exact_w_class(ctx, op_pc, int, int_class)?;
    let off = pyre_object::longobject::LONG_VALUE_OFFSET;
    let long_payload = unsafe { *((long_obj as *const u8).add(off) as *const i64) };
    let long_pl = ctx.trace_ctx.record_op_with_descr(
        OpCode::GetfieldGcR,
        &[long],
        crate::descr::long_value_descr(),
    );
    ctx.trace_ctx.set_opref_concrete(
        long_pl,
        majit_ir::Value::Ref(majit_ir::GcRef(long_payload as usize)),
    );
    let helper_ptr = helper as *const ();
    let raw = ctx.trace_ctx.call_typed_with_effect_pure_can_raise(
        OpCode::CallR,
        helper_ptr,
        &[long_pl, int_raw],
        &[majit_ir::Type::Ref, majit_ir::Type::Int],
        majit_ir::Type::Ref,
        majit_metainterp::ELIDABLE_OR_MEMERROR_EFFECT_INFO,
        &[
            majit_ir::Value::Int(helper_ptr as usize as i64),
            majit_ir::Value::Ref(majit_ir::GcRef(long_payload as usize)),
            majit_ir::Value::Int(int_value),
        ],
        majit_ir::Value::Ref(majit_ir::GcRef(raw_concrete as usize)),
    );
    ctx.trace_ctx.set_opref_concrete(
        raw,
        majit_ir::Value::Ref(majit_ir::GcRef(raw_concrete as usize)),
    );
    if raw.inline_const_to_value().is_none() {
        walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardNoException, &[])?;
    }

    // The box needs no preceding fits_int guard. `_make_generic_descr_binop`
    // and `descr_sub` (longobject.py) wrap with
    // `W_LongObject(intop(...))`, and the interpreter arms they model
    // (`long_add`, `long_sub`, `long_mul`, `long_bitand`, `long_bitor`,
    // `long_bitxor`) wrap with `w_long_new`. Neither side demotes a
    // machine-sized result to a `W_IntObject`, so a result that fits is the
    // same object shape as one that does not, and declining on it left every
    // `x & 0xff`-shaped operation on the generic residual. The
    // `is_int(boxed_result_obj)` test above is what catches a path that does
    // demote.
    let result = crate::helpers::emit_box_long_inline(
        ctx.trace_ctx,
        raw,
        crate::descr::w_long_size_descr(),
        crate::descr::long_value_descr(),
    );
    ctx.trace_ctx.set_opref_concrete(
        result,
        majit_ir::Value::Ref(majit_ir::GcRef(boxed_result_i64 as usize)),
    );
    write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, result)?;
    Ok(Some(()))
}

/// Walker-native `W_LongObject // W_IntObject` / `%` specialization for the
/// `BINARY_OP` helper residual_call.
///
/// `pypy/objspace/std/longobject.py _make_descr_binop` selects
/// `_int_floordiv` / `_int_mod` when the right operand is a `W_IntObject`.
/// The two legs differ in their *result* representation, and that difference
/// is the whole point of specialising them apart:
///   * `_int_floordiv` (`longobject.py`) → `rbigint.int_floordiv` →
///     a bigint quotient, boxed as a `W_LongObject` — the same shape
///     [`try_walker_specialize_binary_op_long_int_shift`] emits.
///   * `_int_mod` (`longobject.py`) → `rbigint.int_mod_int_result` →
///     `space.newint`: the remainder of a long by a machine int always fits a
///     machine int, so this leg allocates **no** result bigint and boxes a
///     plain `W_IntObject`.
///
/// `long_floordiv` / `long_mod` (`descroperation.rs`) raise
/// `ZeroDivisionError` before reaching the `_nonzero` rbigint seam, so the
/// divisor test is traced as a `GUARD_TRUE(int_ne(divisor, 0))` — the guard a
/// meta-tracer records for that interpreter branch. A replay with a zero
/// divisor therefore bails to the interpreter, which raises the authentic
/// error rather than re-deriving its wording here.
///
/// `int // long` and `int % long` are `descr_rfloordiv` / `descr_rmod`, which
/// coerce the left operand to a long and take the bigint/bigint path; they are
/// deliberately left to [`try_walker_specialize_binary_op_long`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_walker_specialize_binary_op_long_int_div<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    op_tag: i64,
    r_args: &[OpRef],
    allboxes: &[OpRef],
    call_descr: &dyn majit_ir::descr::CallDescr,
    dst: usize,
    dst_bank: char,
) -> Result<Option<DispatchOutcome>, DispatchError> {
    if !ctx.is_authoritative_executor || r_args.len() != 2 || dst_bank != 'r' {
        return Ok(None);
    }
    use pyre_interpreter::bytecode::BinaryOperator;
    let is_floordiv = match pyre_interpreter::runtime_ops::binary_op_from_tag(op_tag) {
        Some(BinaryOperator::FloorDivide | BinaryOperator::InplaceFloorDivide) => true,
        Some(BinaryOperator::Remainder | BinaryOperator::InplaceRemainder) => false,
        _ => return Ok(None),
    };
    let long = r_args[0];
    let int = r_args[1];
    let (Some(long_obj), Some(int_obj)) = (
        walker_concrete_ref_object(ctx, long),
        walker_concrete_ref_object(ctx, int),
    ) else {
        return Ok(None);
    };
    if pyre_object::tagged_int::CAN_BE_TAGGED
        && (pyre_object::tagged_int::is_tagged_int(long_obj)
            || pyre_object::tagged_int::is_tagged_int(int_obj))
    {
        return Ok(None);
    }
    let (long_class, int_class, int_value) = unsafe {
        if !pyre_object::is_long(long_obj) || !pyre_object::is_int(int_obj) {
            return Ok(None);
        }
        let (Some(long_class), Some(int_class)) = (
            walker_exact_builtin_class(long_obj),
            walker_exact_builtin_class(int_obj),
        ) else {
            return Ok(None);
        };
        (long_class, int_class, pyre_object::w_int_get_value(int_obj))
    };
    if int_value == 0 {
        let Some(Err(exc_i64)) = walker_execute_may_force_boxed_outcome(ctx, allboxes, call_descr)
        else {
            return Ok(None);
        };
        if let Some(cb) = crate::callbacks::try_get() {
            (cb.drain_backend_jit_exc)();
        }
        let exc = exc_i64 as usize as pyre_object::PyObjectRef;
        let kind = pyre_object::interp_exceptions::ExcKind::ZeroDivisionError;
        if !walker_recorded_builtin_raise_is_supported(exc, kind) {
            return Ok(None);
        }
        let Some(ec) = walker_ensure_execution_context(ctx) else {
            return Ok(None);
        };

        let long_type_addr = &pyre_object::pyobject::LONG_TYPE as *const _ as i64;
        walker_guard_class(ctx, op_pc, long, long_type_addr)?;
        walker_guard_exact_w_class(ctx, op_pc, long, long_class)?;
        let (int_type, int_descr) = crate::state::int_or_bool_unbox_type_descr(int_obj);
        let int_raw = walker_unbox_int_typed(ctx, op_pc, int, int_type, int_descr)?;
        walker_guard_exact_w_class(ctx, op_pc, int, int_class)?;
        let zero = ctx.trace_ctx.const_int(0);
        let is_zero = ctx.trace_ctx.record_op(OpCode::IntEq, &[int_raw, zero]);
        ctx.trace_ctx
            .set_opref_concrete(is_zero, majit_ir::Value::Int(1));
        walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardTrue, &[is_zero])?;
        return Ok(Some(walker_emit_recorded_builtin_raise(ctx, ec, exc, kind)));
    }

    // Execute the authentic Python operation first: it supplies both the
    // observable result and the concrete payload for the pure call, without
    // running an allocating rbigint helper twice at record time.
    let Some(boxed_result_i64) = walker_execute_may_force_boxed(ctx, allboxes, call_descr) else {
        return Ok(None);
    };
    let boxed_result_obj = boxed_result_i64 as usize as pyre_object::PyObjectRef;
    if boxed_result_obj == pyre_object::PY_NULL {
        return Ok(None);
    }
    // Everything that can decline must do so before the first guard is
    // recorded — a later bail-out would leave the operand's class pinned in
    // the heap cache with no matching guard in the trace.
    let quotient_concrete = if is_floordiv {
        if unsafe {
            pyre_object::is_int(boxed_result_obj) || !pyre_object::is_long(boxed_result_obj)
        } {
            return Ok(None);
        }
        let raw_concrete = unsafe {
            *((boxed_result_obj as *const u8).add(pyre_object::longobject::LONG_VALUE_OFFSET)
                as *const i64)
        };
        Some(raw_concrete)
    } else {
        if unsafe { !pyre_object::is_int(boxed_result_obj) } {
            return Ok(None);
        }
        None
    };

    let long_type_addr = &pyre_object::pyobject::LONG_TYPE as *const _ as i64;
    walker_guard_class(ctx, op_pc, long, long_type_addr)?;
    walker_guard_exact_w_class(ctx, op_pc, long, long_class)?;
    let (int_type, int_descr) = crate::state::int_or_bool_unbox_type_descr(int_obj);
    let int_raw = walker_unbox_int_typed(ctx, op_pc, int, int_type, int_descr)?;
    walker_guard_exact_w_class(ctx, op_pc, int, int_class)?;
    let zero = ctx.trace_ctx.const_int(0);
    let nonzero = ctx.trace_ctx.record_op(OpCode::IntNe, &[int_raw, zero]);
    ctx.trace_ctx
        .set_opref_concrete(nonzero, majit_ir::Value::Int(1));
    walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardTrue, &[nonzero])?;

    let off = pyre_object::longobject::LONG_VALUE_OFFSET;
    let long_payload = unsafe { *((long_obj as *const u8).add(off) as *const i64) };
    let long_pl = ctx.trace_ctx.record_op_with_descr(
        OpCode::GetfieldGcR,
        &[long],
        crate::descr::long_value_descr(),
    );
    ctx.trace_ctx.set_opref_concrete(
        long_pl,
        majit_ir::Value::Ref(majit_ir::GcRef(long_payload as usize)),
    );

    let result = match quotient_concrete {
        Some(raw_concrete) => {
            let helper =
                pyre_interpreter::objspace::descroperation::jit_bigint_int_div_floor as *const ();
            let raw = ctx.trace_ctx.call_typed_with_effect_pure_can_raise(
                OpCode::CallR,
                helper,
                &[long_pl, int_raw],
                &[majit_ir::Type::Ref, majit_ir::Type::Int],
                majit_ir::Type::Ref,
                majit_metainterp::ELIDABLE_OR_MEMERROR_EFFECT_INFO,
                &[
                    majit_ir::Value::Int(helper as usize as i64),
                    majit_ir::Value::Ref(majit_ir::GcRef(long_payload as usize)),
                    majit_ir::Value::Int(int_value),
                ],
                majit_ir::Value::Ref(majit_ir::GcRef(raw_concrete as usize)),
            );
            ctx.trace_ctx.set_opref_concrete(
                raw,
                majit_ir::Value::Ref(majit_ir::GcRef(raw_concrete as usize)),
            );
            if raw.inline_const_to_value().is_none() {
                walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardNoException, &[])?;
            }
            // The box needs no preceding fits_int guard. `_floordiv`/
            // `_int_floordiv` (longobject.py) wrap with `newlong` and
            // `long_floordiv` with `w_long_new`, so the quotient is a
            // `W_LongObject` whatever its magnitude, and the inline box carries
            // the payload by pointer — nothing about it varies with the digit
            // count the guard was testing.
            let boxed = crate::helpers::emit_box_long_inline(
                ctx.trace_ctx,
                raw,
                crate::descr::w_long_size_descr(),
                crate::descr::long_value_descr(),
            );
            ctx.trace_ctx.set_opref_concrete(
                boxed,
                majit_ir::Value::Ref(majit_ir::GcRef(boxed_result_i64 as usize)),
            );
            boxed
        }
        None => {
            let mod_concrete = unsafe { pyre_object::w_int_get_value(boxed_result_obj) };
            let helper = pyre_interpreter::objspace::descroperation::jit_bigint_int_mod_int_result
                as *const ();
            let raw = ctx.trace_ctx.call_typed_with_effect_pure_can_raise(
                OpCode::CallI,
                helper,
                &[long_pl, int_raw],
                &[majit_ir::Type::Ref, majit_ir::Type::Int],
                majit_ir::Type::Int,
                majit_metainterp::ELIDABLE_EFFECT_INFO,
                &[
                    majit_ir::Value::Int(helper as usize as i64),
                    majit_ir::Value::Ref(majit_ir::GcRef(long_payload as usize)),
                    majit_ir::Value::Int(int_value),
                ],
                majit_ir::Value::Int(mod_concrete),
            );
            ctx.trace_ctx
                .set_opref_concrete(raw, majit_ir::Value::Int(mod_concrete));
            if raw.inline_const_to_value().is_none() {
                walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardNoException, &[])?;
            }
            let boxed = walker_box_int(ctx, op_pc, raw, mod_concrete)?;
            ctx.trace_ctx
                .set_opref_concrete(boxed, box_int_concrete(mod_concrete, boxed_result_i64));
            boxed
        }
    };
    write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, result)?;
    Ok(Some(DispatchOutcome::Continue))
}

/// Walker-native `W_LongObject ** W_IntObject` specialization for the
/// `BINARY_OP` helper residual_call.
///
/// `longobject.py descr_pow` keeps a `W_IntObject` exponent unwrapped
/// (`exp_bigint` stays `None`) and calls `rbigint.int_pow`; only a long
/// exponent reaches `rbigint.pow`. `long_pow` (`descroperation.rs`) reaches
/// that call past four short-circuits — a negative exponent goes to the float
/// path, a zero exponent returns 1, and a base of 0 / 1 / -1 returns a
/// constant. The exponent tests unbox to a machine word, so they record as a
/// single `GUARD_TRUE(int_gt(exp, 0))`.
///
/// The three base tests collapse into one guard: a base whose payload does not
/// fit a machine word has magnitude at least `2**63`, so it can be none of
/// 0 / 1 / -1. That is strictly stronger than the interpreter's three
/// comparisons — it can only bail more often, never take a different path —
/// and it costs one `jit_bigint_fits_int` call instead of three bigint
/// comparisons against baked constants. It also makes the usual trailing
/// result-fits guard redundant: `|base| >= 2**63` with `exp >= 1` cannot
/// produce a result that fits a machine word, so unlike the floordiv and shift
/// legs this one emits no guard on the result payload.
///
/// `descr_pow` wraps every branch as `W_LongObject` (no `newlong` demotion),
/// so the result boxes inline exactly like the other long legs.
/// The three-argument `pow(a, b, m)` is not a `BINARY_OP` and never reaches
/// here.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_walker_specialize_binary_op_long_int_pow<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    op_tag: i64,
    r_args: &[OpRef],
    allboxes: &[OpRef],
    call_descr: &dyn majit_ir::descr::CallDescr,
    dst: usize,
    dst_bank: char,
) -> Result<Option<()>, DispatchError> {
    if !ctx.is_authoritative_executor || r_args.len() != 2 || dst_bank != 'r' {
        return Ok(None);
    }
    use pyre_interpreter::bytecode::BinaryOperator;
    if !matches!(
        pyre_interpreter::runtime_ops::binary_op_from_tag(op_tag),
        Some(BinaryOperator::Power | BinaryOperator::InplacePower)
    ) {
        return Ok(None);
    }
    let long = r_args[0];
    let int = r_args[1];
    let (Some(long_obj), Some(int_obj)) = (
        walker_concrete_ref_object(ctx, long),
        walker_concrete_ref_object(ctx, int),
    ) else {
        return Ok(None);
    };
    if pyre_object::tagged_int::CAN_BE_TAGGED
        && (pyre_object::tagged_int::is_tagged_int(long_obj)
            || pyre_object::tagged_int::is_tagged_int(int_obj))
    {
        return Ok(None);
    }
    let (long_class, int_class, exp_value) = unsafe {
        if !pyre_object::is_long(long_obj) || !pyre_object::is_int(int_obj) {
            return Ok(None);
        }
        let (Some(long_class), Some(int_class)) = (
            walker_exact_builtin_class(long_obj),
            walker_exact_builtin_class(int_obj),
        ) else {
            return Ok(None);
        };
        (long_class, int_class, pyre_object::w_int_get_value(int_obj))
    };
    // A non-positive exponent leaves through one of the short-circuits above
    // the `rbigint.int_pow` call; record those through the generic helper.
    if exp_value <= 0 {
        return Ok(None);
    }
    let off = pyre_object::longobject::LONG_VALUE_OFFSET;
    let base_payload = unsafe { *((long_obj as *const u8).add(off) as *const i64) };
    if pyre_object::longobject::jit_bigint_fits_int(base_payload) != 0 {
        return Ok(None);
    }

    let Some(boxed_result_i64) = walker_execute_may_force_boxed(ctx, allboxes, call_descr) else {
        return Ok(None);
    };
    let boxed_result_obj = boxed_result_i64 as usize as pyre_object::PyObjectRef;
    if boxed_result_obj == pyre_object::PY_NULL
        || unsafe {
            pyre_object::is_int(boxed_result_obj) || !pyre_object::is_long(boxed_result_obj)
        }
    {
        return Ok(None);
    }
    let raw_concrete = unsafe {
        *((boxed_result_obj as *const u8).add(pyre_object::longobject::LONG_VALUE_OFFSET)
            as *const i64)
    };

    let long_type_addr = &pyre_object::pyobject::LONG_TYPE as *const _ as i64;
    walker_guard_class(ctx, op_pc, long, long_type_addr)?;
    walker_guard_exact_w_class(ctx, op_pc, long, long_class)?;
    let (int_type, int_descr) = crate::state::int_or_bool_unbox_type_descr(int_obj);
    let exp_raw = walker_unbox_int_typed(ctx, op_pc, int, int_type, int_descr)?;
    walker_guard_exact_w_class(ctx, op_pc, int, int_class)?;
    let zero = ctx.trace_ctx.const_int(0);
    let positive = ctx.trace_ctx.record_op(OpCode::IntGt, &[exp_raw, zero]);
    ctx.trace_ctx
        .set_opref_concrete(positive, majit_ir::Value::Int(1));
    walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardTrue, &[positive])?;

    let base_pl = ctx.trace_ctx.record_op_with_descr(
        OpCode::GetfieldGcR,
        &[long],
        crate::descr::long_value_descr(),
    );
    ctx.trace_ctx.set_opref_concrete(
        base_pl,
        majit_ir::Value::Ref(majit_ir::GcRef(base_payload as usize)),
    );
    let fits_fn = pyre_object::longobject::jit_bigint_fits_int as *const ();
    let base_fits = ctx.trace_ctx.call_typed_with_effect(
        OpCode::CallI,
        fits_fn,
        &[base_pl],
        &[majit_ir::Type::Ref],
        majit_ir::Type::Int,
        majit_metainterp::cannot_raise_effect_info(),
    );
    ctx.trace_ctx
        .set_opref_concrete(base_fits, majit_ir::Value::Int(0));
    walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardFalse, &[base_fits])?;

    let helper = pyre_interpreter::objspace::descroperation::jit_bigint_int_pow_nomod as *const ();
    let raw = ctx.trace_ctx.call_typed_with_effect_pure_can_raise(
        OpCode::CallR,
        helper,
        &[base_pl, exp_raw],
        &[majit_ir::Type::Ref, majit_ir::Type::Int],
        majit_ir::Type::Ref,
        majit_metainterp::ELIDABLE_OR_MEMERROR_EFFECT_INFO,
        &[
            majit_ir::Value::Int(helper as usize as i64),
            majit_ir::Value::Ref(majit_ir::GcRef(base_payload as usize)),
            majit_ir::Value::Int(exp_value),
        ],
        majit_ir::Value::Ref(majit_ir::GcRef(raw_concrete as usize)),
    );
    ctx.trace_ctx.set_opref_concrete(
        raw,
        majit_ir::Value::Ref(majit_ir::GcRef(raw_concrete as usize)),
    );
    if raw.inline_const_to_value().is_none() {
        walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardNoException, &[])?;
    }

    let result = crate::helpers::emit_box_long_inline(
        ctx.trace_ctx,
        raw,
        crate::descr::w_long_size_descr(),
        crate::descr::long_value_descr(),
    );
    ctx.trace_ctx.set_opref_concrete(
        result,
        majit_ir::Value::Ref(majit_ir::GcRef(boxed_result_i64 as usize)),
    );
    write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, result)?;
    Ok(Some(()))
}

/// Walker-native `W_LongObject << W_IntObject` / `>>` specialization for the
/// `BINARY_OP` helper residual_call.
///
/// `pypy/objspace/std/longobject.py:_make_descr_binop` selects
/// `_int_lshift` / `_int_rshift` when the right operand is a `W_IntObject`;
/// those pass its machine-word `int_w` directly to `rbigint.lshift/rshift`.
/// Preserve that distinct source shape here: guard/unbox the right operand as
/// an Int (or Bool), guard the count non-negative, and call the matching
/// `[Ref, Int] -> Ref` rbigint residual.  Converting the count to a temporary
/// bigint and routing through the two-long helper would not be the upstream
/// program and would add an allocation to every shift.
///
/// As in [`try_walker_specialize_binary_op_long`], the helper result is the
/// bare immutable bigint payload and the walker boxes it as a
/// `W_LongObject`.  A result that demotes to `W_IntObject` declines before
/// emitting IR; replay is protected by the same fits-int guard.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_walker_specialize_binary_op_long_int_shift<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    op_tag: i64,
    r_args: &[OpRef],
    allboxes: &[OpRef],
    call_descr: &dyn majit_ir::descr::CallDescr,
    dst: usize,
    dst_bank: char,
) -> Result<Option<DispatchOutcome>, DispatchError> {
    if !ctx.is_authoritative_executor || r_args.len() != 2 || dst_bank != 'r' {
        return Ok(None);
    }
    use pyre_interpreter::bytecode::BinaryOperator;
    let is_lshift = match pyre_interpreter::runtime_ops::binary_op_from_tag(op_tag) {
        Some(BinaryOperator::Lshift | BinaryOperator::InplaceLshift) => true,
        Some(BinaryOperator::Rshift | BinaryOperator::InplaceRshift) => false,
        _ => return Ok(None),
    };
    let lhs = r_args[0];
    let rhs = r_args[1];
    let (Some(lhs_obj), Some(rhs_obj)) = (
        walker_concrete_ref_object(ctx, lhs),
        walker_concrete_ref_object(ctx, rhs),
    ) else {
        return Ok(None);
    };
    let (lhs_class, rhs_class, rhs_value) = unsafe {
        if !pyre_object::is_long(lhs_obj) || !pyre_object::is_int(rhs_obj) {
            return Ok(None);
        }
        let (Some(lhs_class), Some(rhs_class)) = (
            walker_exact_builtin_class(lhs_obj),
            walker_exact_builtin_class(rhs_obj),
        ) else {
            return Ok(None);
        };
        (lhs_class, rhs_class, pyre_object::w_int_get_value(rhs_obj))
    };
    if rhs_value < 0 {
        let Some(Err(exc_i64)) = walker_execute_may_force_boxed_outcome(ctx, allboxes, call_descr)
        else {
            return Ok(None);
        };
        if let Some(cb) = crate::callbacks::try_get() {
            (cb.drain_backend_jit_exc)();
        }
        let exc = exc_i64 as usize as pyre_object::PyObjectRef;
        let kind = pyre_object::interp_exceptions::ExcKind::ValueError;
        if !walker_recorded_builtin_raise_is_supported(exc, kind) {
            return Ok(None);
        }
        let Some(ec) = walker_ensure_execution_context(ctx) else {
            return Ok(None);
        };

        let long_type_addr = &pyre_object::pyobject::LONG_TYPE as *const _ as i64;
        walker_guard_class(ctx, op_pc, lhs, long_type_addr)?;
        walker_guard_exact_w_class(ctx, op_pc, lhs, lhs_class)?;
        let (rhs_type, rhs_descr) = crate::state::int_or_bool_unbox_type_descr(rhs_obj);
        let rhs_raw = walker_unbox_int_typed(ctx, op_pc, rhs, rhs_type, rhs_descr)?;
        walker_guard_exact_w_class(ctx, op_pc, rhs, rhs_class)?;
        let zero = ctx.trace_ctx.const_int(0);
        let is_negative = ctx.trace_ctx.record_op(OpCode::IntLt, &[rhs_raw, zero]);
        ctx.trace_ctx
            .set_opref_concrete(is_negative, majit_ir::Value::Int(1));
        walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardTrue, &[is_negative])?;
        return Ok(Some(walker_emit_recorded_builtin_raise(ctx, ec, exc, kind)));
    }

    // Execute the authentic Python operation first.  It supplies both the
    // observable result and the concrete payload for the pure call without
    // running an allocating rbigint helper twice at record time.
    let Some(boxed_result_i64) = walker_execute_may_force_boxed(ctx, allboxes, call_descr) else {
        return Ok(None);
    };
    let boxed_result_obj = boxed_result_i64 as usize as pyre_object::PyObjectRef;
    if boxed_result_obj == pyre_object::PY_NULL
        || unsafe {
            pyre_object::is_int(boxed_result_obj) || !pyre_object::is_long(boxed_result_obj)
        }
    {
        return Ok(None);
    }
    let raw_concrete = unsafe {
        *((boxed_result_obj as *const u8).add(pyre_object::longobject::LONG_VALUE_OFFSET)
            as *const i64)
    };

    let long_type_addr = &pyre_object::pyobject::LONG_TYPE as *const _ as i64;
    walker_guard_class(ctx, op_pc, lhs, long_type_addr)?;
    walker_guard_exact_w_class(ctx, op_pc, lhs, lhs_class)?;
    let (rhs_type, rhs_descr) = crate::state::int_or_bool_unbox_type_descr(rhs_obj);
    let rhs_raw = walker_unbox_int_typed(ctx, op_pc, rhs, rhs_type, rhs_descr)?;
    walker_guard_exact_w_class(ctx, op_pc, rhs, rhs_class)?;
    let zero = ctx.trace_ctx.const_int(0);
    let nonnegative = ctx.trace_ctx.record_op(OpCode::IntGe, &[rhs_raw, zero]);
    ctx.trace_ctx
        .set_opref_concrete(nonnegative, majit_ir::Value::Int(1));
    walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardTrue, &[nonnegative])?;

    let off = pyre_object::longobject::LONG_VALUE_OFFSET;
    let lhs_payload = unsafe { *((lhs_obj as *const u8).add(off) as *const i64) };
    let lhs_pl = ctx.trace_ctx.record_op_with_descr(
        OpCode::GetfieldGcR,
        &[lhs],
        crate::descr::long_value_descr(),
    );
    ctx.trace_ctx.set_opref_concrete(
        lhs_pl,
        majit_ir::Value::Ref(majit_ir::GcRef(lhs_payload as usize)),
    );
    let helper = if is_lshift {
        pyre_interpreter::objspace::descroperation::jit_bigint_lshift_count as *const ()
    } else {
        pyre_interpreter::objspace::descroperation::jit_bigint_shr as *const ()
    };
    let raw = ctx.trace_ctx.call_typed_with_effect_pure_can_raise(
        OpCode::CallR,
        helper,
        &[lhs_pl, rhs_raw],
        &[majit_ir::Type::Ref, majit_ir::Type::Int],
        majit_ir::Type::Ref,
        majit_metainterp::ELIDABLE_OR_MEMERROR_EFFECT_INFO,
        &[
            majit_ir::Value::Int(helper as usize as i64),
            majit_ir::Value::Ref(majit_ir::GcRef(lhs_payload as usize)),
            majit_ir::Value::Int(rhs_value),
        ],
        majit_ir::Value::Ref(majit_ir::GcRef(raw_concrete as usize)),
    );
    ctx.trace_ctx.set_opref_concrete(
        raw,
        majit_ir::Value::Ref(majit_ir::GcRef(raw_concrete as usize)),
    );
    if raw.inline_const_to_value().is_none() {
        walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardNoException, &[])?;
    }

    // The box needs no preceding fits_int guard: `_int_lshift` wraps with
    // `W_LongObject(...)` and `_int_rshift` with `newlong` (longobject.py,
    // 402), and `long_lshift`/`long_rshift` both end in `w_long_new`. Neither
    // demotes a machine-sized result — `newlong` only reaches
    // `W_SmallLongObject`, which `withsmalllong` leaves off — so declining on
    // a fitting result put every `x >> 32`-shaped shift back on the generic
    // residual for no observable difference.
    let result = crate::helpers::emit_box_long_inline(
        ctx.trace_ctx,
        raw,
        crate::descr::w_long_size_descr(),
        crate::descr::long_value_descr(),
    );
    ctx.trace_ctx.set_opref_concrete(
        result,
        majit_ir::Value::Ref(majit_ir::GcRef(boxed_result_i64 as usize)),
    );
    write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, result)?;
    Ok(Some(DispatchOutcome::Continue))
}

/// Walker-native W_LongObject (bigint) arithmetic specialization for the
/// `BINARY_OP` helper residual_call (oopspec `BinaryOp`).  When both
/// operands are concrete `W_LongObject`, emit `GUARD_CLASS(LONG_TYPE)` per
/// operand + `GETFIELD_GC_PURE_R(value)` + a `CALL_PURE_R` to the elidable
/// `rbigint` payload helper (`long_binop_raw_helper`, `rbigint.py
/// @jit.elidable`) producing a bare Ref-typed bigint, then inline
/// `W_LongObject(...)` boxing via `new_with_vtable` + `setfield_gc('value')`.
/// Neither is the opaque
/// `CALL_MAY_FORCE` the generic leg records, so this sheds the per-iteration
/// force-token store + `GUARD_NOT_FORCED` + `GUARD_NO_EXCEPTION` from
/// bigint-heavy loops (e.g. `fib_loop`).
///
/// Specialized for add/sub/mul/and/or/xor (allocate → `EF_ELIDABLE_OR_MEMORYERROR`)
/// and floordiv/mod/lshift/rshift (`EF_ELIDABLE_CAN_RAISE`); both classes have
/// `check_can_raise()` true, so every op carries a trailing `GUARD_NO_EXCEPTION`.
/// True-divide has its own float fast path
/// ([`try_walker_specialize_truediv_op_long`]); pow and any non-`W_LongObject`
/// operand return `Ok(None)` so the caller falls through to the generic record,
/// preserving the `__op__` semantics.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_walker_specialize_binary_op_long<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    op_tag: i64,
    r_args: &[OpRef],
    allboxes: &[OpRef],
    call_descr: &dyn majit_ir::descr::CallDescr,
    dst: usize,
    dst_bank: char,
) -> Result<Option<()>, DispatchError> {
    use pyre_interpreter::bytecode::BinaryOperator;
    if !ctx.is_authoritative_executor || r_args.len() != 2 || dst_bank != 'r' {
        return Ok(None);
    }
    let Some(op) = pyre_interpreter::runtime_ops::binary_op_from_tag(op_tag) else {
        return Ok(None);
    };
    let Some(spec) = crate::trace_opcode::long_binop_raw_helper(op) else {
        return Ok(None);
    };
    let lhs = r_args[0];
    let rhs = r_args[1];
    let (Some(lhs_obj), Some(rhs_obj)) = (
        walker_concrete_ref_object(ctx, lhs),
        walker_concrete_ref_object(ctx, rhs),
    ) else {
        return Ok(None);
    };
    if !unsafe { pyre_object::is_long(lhs_obj) && pyre_object::is_long(rhs_obj) } {
        return Ok(None);
    }
    let (Some(lhs_class), Some(rhs_class)) = (unsafe {
        (
            walker_exact_builtin_class(lhs_obj),
            walker_exact_builtin_class(rhs_obj),
        )
    }) else {
        return Ok(None);
    };
    // Authentic boxed result via the same execute path the int leg uses; a
    // NULL / raised result defers to the generic record.
    let Some(boxed_result_i64) = walker_execute_may_force_boxed(ctx, allboxes, call_descr) else {
        return Ok(None);
    };
    // A NULL result means the op raised — defer to the generic record. `newlong`
    // never demotes, so an arithmetic long op always yields a W_LongObject the
    // inline-NEW box below can represent. The shift ops are the only ones that
    // can still yield a W_IntObject (`space.newint(-1)`/`(0)` on a shift count
    // that overflows a machine int); the `!is_long` decline routes that
    // huge-count case to the generic leg. Reuse the authentic boxed result's
    // payload instead of running `spec.raw_fn` a second time; the raw helpers
    // allocate/publish exception state and must not be used as a trace-time
    // probe.
    let boxed_result_obj = boxed_result_i64 as usize as pyre_object::PyObjectRef;
    if boxed_result_obj == pyre_object::PY_NULL {
        return Ok(None);
    }
    if !unsafe { pyre_object::is_long(boxed_result_obj) } {
        return Ok(None);
    }
    let raw_concrete = unsafe {
        *((boxed_result_obj as *const u8).add(pyre_object::longobject::LONG_VALUE_OFFSET)
            as *const i64)
    };
    let long_type_addr = &pyre_object::pyobject::LONG_TYPE as *const _ as i64;
    walker_guard_class(ctx, op_pc, lhs, long_type_addr)?;
    walker_guard_class(ctx, op_pc, rhs, long_type_addr)?;
    walker_guard_exact_w_class(ctx, op_pc, lhs, lhs_class)?;
    walker_guard_exact_w_class(ctx, op_pc, rhs, rhs_class)?;
    // Read each operand's immutable `value` payload, then call the
    // elidable `rbigint` op on the bare `*const BigInt` payloads. Passing
    // the payloads (not the wrappers) keeps the call pure on the immutable
    // bigints, so the optimizer forwards the field read and never reorders
    // this elidable call ahead of the boxing `setfield_gc` below — which
    // would otherwise read the freshly-allocated result wrapper's
    // uninitialized `value` (the function-loop unroll exposed exactly that
    // reorder). The forwarding is descr-keyed (`long_value_descr()` is
    // immutable), so the plain `GETFIELD_GC_R` opnum gets identical OptHeap
    // treatment — there is no pure getfield opnum. The result is a
    // GC-managed `*mut BigInt`, Ref-typed so the JIT gcmap roots it across
    // the collecting boxing NEW. Every op allocates
    // (`EF_ELIDABLE_OR_MEMORYERROR`) or divides (`EF_ELIDABLE_CAN_RAISE`),
    // so a trailing `GuardNoException` follows (`pyjitpl.py`).
    let off = pyre_object::longobject::LONG_VALUE_OFFSET;
    let lhs_payload = unsafe { *((lhs_obj as *const u8).add(off) as *const i64) };
    let rhs_payload = unsafe { *((rhs_obj as *const u8).add(off) as *const i64) };
    let lhs_pl = ctx.trace_ctx.record_op_with_descr(
        OpCode::GetfieldGcR,
        &[lhs],
        crate::descr::long_value_descr(),
    );
    ctx.trace_ctx.set_opref_concrete(
        lhs_pl,
        majit_ir::Value::Ref(majit_ir::GcRef(lhs_payload as usize)),
    );
    let rhs_pl = ctx.trace_ctx.record_op_with_descr(
        OpCode::GetfieldGcR,
        &[rhs],
        crate::descr::long_value_descr(),
    );
    ctx.trace_ctx.set_opref_concrete(
        rhs_pl,
        majit_ir::Value::Ref(majit_ir::GcRef(rhs_payload as usize)),
    );
    let add_fn = spec.payload_fn as *const ();
    let concrete_args = [
        majit_ir::Value::Int(add_fn as usize as i64),
        majit_ir::Value::Ref(majit_ir::GcRef(lhs_payload as usize)),
        majit_ir::Value::Ref(majit_ir::GcRef(rhs_payload as usize)),
    ];
    let raw = ctx.trace_ctx.call_typed_with_effect_pure_can_raise(
        OpCode::CallR,
        add_fn,
        &[lhs_pl, rhs_pl],
        &[majit_ir::Type::Ref, majit_ir::Type::Ref],
        majit_ir::Type::Ref,
        spec.effect,
        &concrete_args,
        majit_ir::Value::Ref(majit_ir::GcRef(raw_concrete as usize)),
    );
    ctx.trace_ctx.set_opref_concrete(
        raw,
        majit_ir::Value::Ref(majit_ir::GcRef(raw_concrete as usize)),
    );
    if raw.inline_const_to_value().is_none() {
        walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardNoException, &[])?;
    }
    // Shift-count demote guard: `_lshift`/`_rshift` demote to `space.newint`
    // (-1/0) when the shift count overflows a machine int (`toint()`
    // OverflowError), so guard that the count fits and let a huge-count replay
    // deopt to the generic leg. The arithmetic ops (`newlong`, no demote) emit
    // no such guard — a fitting result stays a W_LongObject in the trace.
    if matches!(
        op,
        BinaryOperator::Lshift
            | BinaryOperator::InplaceLshift
            | BinaryOperator::Rshift
            | BinaryOperator::InplaceRshift
    ) {
        let fits_fn = pyre_object::longobject::jit_bigint_fits_int as *const ();
        let count_fits = ctx.trace_ctx.call_typed_with_effect(
            OpCode::CallI,
            fits_fn,
            &[rhs_pl],
            &[majit_ir::Type::Ref],
            majit_ir::Type::Int,
            majit_metainterp::cannot_raise_effect_info(),
        );
        let count_fits_concrete =
            unsafe { pyre_object::longobject::jit_bigint_fits_int(rhs_payload) };
        ctx.trace_ctx
            .set_opref_concrete(count_fits, majit_ir::Value::Int(count_fits_concrete));
        walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardTrue, &[count_fits])?;
    }
    // Inline `W_LongObject(raw)` NEW (`new_with_vtable` + `setfield_gc('value')`).
    // NewWithVtable lowers to the collecting `CallMallocNursery` — the GC
    // safepoint that lets bigint-heavy loops reclaim dead bigints.
    let result = crate::helpers::emit_box_long_inline(
        ctx.trace_ctx,
        raw,
        crate::descr::w_long_size_descr(),
        crate::descr::long_value_descr(),
    );
    ctx.trace_ctx.set_opref_concrete(
        result,
        majit_ir::Value::Ref(majit_ir::GcRef(boxed_result_i64 as usize)),
    );
    write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, result)?;
    Ok(Some(()))
}

/// W_LongObject true-divide specialization — the float analogue of
/// [`try_walker_specialize_binary_op_long`].  Both operands are `int`-typed but
/// bigint-stored: guard each against `LONG_TYPE`, then `CallPureF` the elidable
/// `jit_w_long_truediv_raw` (correctly-rounded f64 quotient; raises
/// ZeroDivision/Overflow → `EF_ELIDABLE_CAN_RAISE` ⇒ trailing `GuardNoException`)
/// and box the f64 with `wrapfloat` (transparent `new_with_vtable` +
/// `setfield_gc_f`, the trace analogue of `_truediv`'s `space.newfloat(f)`), so a
/// downstream float op keeps the quotient unboxed.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_walker_specialize_truediv_op_long<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    op_tag: i64,
    r_args: &[OpRef],
    allboxes: &[OpRef],
    call_descr: &dyn majit_ir::descr::CallDescr,
    dst: usize,
    dst_bank: char,
) -> Result<Option<()>, DispatchError> {
    if !ctx.is_authoritative_executor || r_args.len() != 2 || dst_bank != 'r' {
        return Ok(None);
    }
    use pyre_interpreter::bytecode::BinaryOperator;
    match pyre_interpreter::runtime_ops::binary_op_from_tag(op_tag) {
        Some(BinaryOperator::TrueDivide) | Some(BinaryOperator::InplaceTrueDivide) => {}
        _ => return Ok(None),
    }
    let lhs = r_args[0];
    let rhs = r_args[1];
    let (Some(lhs_obj), Some(rhs_obj)) = (
        walker_concrete_ref_object(ctx, lhs),
        walker_concrete_ref_object(ctx, rhs),
    ) else {
        return Ok(None);
    };
    if !unsafe { pyre_object::is_long(lhs_obj) && pyre_object::is_long(rhs_obj) } {
        return Ok(None);
    }
    let (Some(lhs_class), Some(rhs_class)) = (unsafe {
        (
            walker_exact_builtin_class(lhs_obj),
            walker_exact_builtin_class(rhs_obj),
        )
    }) else {
        return Ok(None);
    };
    // Authentic boxed float via the generic execute path; a NULL / raised result
    // (zero divisor, float overflow) defers to the generic record.
    let Some(boxed_result_i64) = walker_execute_may_force_boxed(ctx, allboxes, call_descr) else {
        return Ok(None);
    };
    let long_type_addr = &pyre_object::pyobject::LONG_TYPE as *const _ as i64;
    walker_guard_class(ctx, op_pc, lhs, long_type_addr)?;
    walker_guard_class(ctx, op_pc, rhs, long_type_addr)?;
    walker_guard_exact_w_class(ctx, op_pc, lhs, lhs_class)?;
    walker_guard_exact_w_class(ctx, op_pc, rhs, rhs_class)?;
    // Pure `rbigint.truediv` → correctly-rounded f64 (CallPureF). The op already
    // ran authentically above, so the divisor is nonzero / non-overflowing here;
    // the trailing GuardNoException covers a divide-by-zero / overflow on replay.
    let truediv_fn =
        pyre_interpreter::objspace::descroperation::jit_w_long_truediv_raw as *const ();
    let f_concrete = pyre_interpreter::objspace::descroperation::jit_w_long_truediv_raw(
        lhs_obj as i64,
        rhs_obj as i64,
    );
    let concrete_args = [
        majit_ir::Value::Int(truediv_fn as usize as i64),
        majit_ir::Value::Ref(majit_ir::GcRef(lhs_obj as usize)),
        majit_ir::Value::Ref(majit_ir::GcRef(rhs_obj as usize)),
    ];
    let raw = ctx.trace_ctx.call_typed_with_effect_pure_can_raise(
        OpCode::CallF,
        truediv_fn,
        &[lhs, rhs],
        &[majit_ir::Type::Ref, majit_ir::Type::Ref],
        majit_ir::Type::Float,
        majit_metainterp::ELIDABLE_EFFECT_INFO,
        &concrete_args,
        majit_ir::Value::Float(f_concrete),
    );
    ctx.trace_ctx
        .set_opref_concrete(raw, majit_ir::Value::Float(f_concrete));
    // pyjitpl.py: no GuardNoException when the pure call folded to a Const.
    if raw.inline_const_to_value().is_none() {
        walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardNoException, &[])?;
    }
    // Box the f64 with the transparent float NEW (`new_with_vtable` +
    // `setfield_gc_f`), mirroring `space.newfloat(f)`.
    let result = crate::state::wrapfloat(ctx.trace_ctx, raw);
    ctx.trace_ctx.set_opref_concrete(
        result,
        majit_ir::Value::Ref(majit_ir::GcRef(boxed_result_i64 as usize)),
    );
    write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, result)?;
    Ok(Some(()))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SpecialisedPairKind {
    Int,
    Float,
    Object,
}

/// Identify the three classes produced by
/// `specialisedtupleobject.py makespecialisedtuple2`.
pub(crate) fn specialised_pair_kind(
    seq_type: *const pyre_object::pyobject::PyType,
) -> Option<SpecialisedPairKind> {
    use pyre_object::specialisedtupleobject::{
        SPECIALISED_TUPLE_FF_TYPE, SPECIALISED_TUPLE_II_TYPE, SPECIALISED_TUPLE_OO_TYPE,
    };
    if std::ptr::eq(seq_type, &SPECIALISED_TUPLE_II_TYPE) {
        Some(SpecialisedPairKind::Int)
    } else if std::ptr::eq(seq_type, &SPECIALISED_TUPLE_FF_TYPE) {
        Some(SpecialisedPairKind::Float)
    } else if std::ptr::eq(seq_type, &SPECIALISED_TUPLE_OO_TYPE) {
        Some(SpecialisedPairKind::Object)
    } else {
        None
    }
}

/// FBW fold of the UNPACK_SEQUENCE two-residual lowering (`unpack_sequence_fn`
/// validator + per-index `unpack_item_fn` reader emitted by the codewriter
/// UNPACK_SEQUENCE arm) for an arity-2 specialised tuple: guard the
/// specialisation's class once, then read `value0` / `value1` directly instead
/// of leaving three opaque `CALL_MAY_FORCE` residuals in the loop.
///
/// `objspace.py fixedview` reaches `tolist()` for every
/// `W_AbstractTupleObject`, and `specialisedtupleobject.py tolist`
/// unrolls over `_immutable_fields_` value slots, so upstream traces the whole
/// unpack inline and the optimizer virtualizes the pair away. Both arity-2
/// layouts are covered here because `makespecialisedtuple2`
/// (`specialisedtupleobject.py:169-179`) never falls back to a plain tuple:
///   * `ii` — `value0`/`value1` are inline machine ints, so the read is
///     `getfield_gc_pure_i` + `wrapint` and the items stay unboxed through the
///     downstream BINARY_OP int fold (the walker analogue of the retired
///     MIFrame `W_SpecialisedTupleObject_ii` reads);
///   * `ff` — the same shape with `getfield_gc_pure_f` + `wrapfloat`; this is
///     the representation `zip` produces for a pair of exact floats;
///   * `oo` — `wraps[i]` for an object slot is the identity
///     (`specialisedtupleobject.py:26-27`), so the `getfield_gc_r` result is
///     already the item. This is the layout a `divmod(long, long)` result pair
///     takes, since neither half satisfies `is_plain_int1`.
///
/// Returns `Ok(Some(()))` when folded (the caller returns `Continue`);
/// `Ok(None)` to fall through to the opaque residual record, which stays
/// correct for any other shape — so a non-foldable sequence is not declined.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_walker_specialize_unpack<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    helper: majit_ir::RuntimeHelperKind,
    i_args: &[OpRef],
    r_args: &[OpRef],
    allboxes: &[OpRef],
    call_descr: &dyn majit_ir::descr::CallDescr,
    dst: usize,
    dst_bank: char,
) -> Result<Option<()>, DispatchError> {
    if !ctx.is_authoritative_executor || dst_bank != 'r' {
        return Ok(None);
    }
    let (Some(&int_arg), Some(&seq)) = (i_args.first(), r_args.first()) else {
        return Ok(None);
    };
    let Some(majit_ir::Value::Int(int_val)) = ctx.trace_ctx.box_value(int_arg) else {
        return Ok(None);
    };
    let Some(concrete_seq) = walker_concrete_ref_object(ctx, seq) else {
        return Ok(None);
    };
    // `objspace.py:507-541 StdObjSpace.{unpackiterable,fixedview}` takes an
    // exact tuple straight to its immutable `wrappeditems` list; and
    // `pyopcode.py UNPACK_SEQUENCE` calls `fixedview_unroll`, so a
    // constant item count exposes each tuple item directly to the trace.
    // Preserve that shape for pyre's split `UnpackSequence` / `UnpackItem`
    // helpers.  In particular, `zip` produces an ordinary array-backed tuple
    // each iteration; leaving these helpers residual makes the three-item
    // comprehension trace execute one validation call plus one call per
    // projected item.
    let tuple_type = &pyre_object::pyobject::TUPLE_TYPE as *const pyre_object::pyobject::PyType;
    let canonical_tuple_class = pyre_object::pyobject::get_instantiate(unsafe { &*tuple_type });
    if unsafe {
        std::ptr::eq((*concrete_seq).ob_type, tuple_type)
            && std::ptr::eq((*concrete_seq).w_class, canonical_tuple_class)
    } {
        let concrete_len = unsafe { pyre_object::w_tuple_len(concrete_seq) };
        if int_val < 0 {
            return Ok(None);
        }
        walker_guard_class(ctx, op_pc, seq, tuple_type as i64)?;
        walker_guard_exact_w_class(ctx, op_pc, seq, canonical_tuple_class)?;
        let items = crate::state::opimpl_getfield_gc_r(
            ctx.trace_ctx,
            seq,
            crate::descr::tuple_wrappeditems_descr(),
        );
        match helper {
            majit_ir::RuntimeHelperKind::UnpackSequence => {
                if int_val as usize != concrete_len {
                    return Ok(None);
                }
                let length = crate::state::opimpl_arraylen_gc(
                    ctx.trace_ctx,
                    items,
                    crate::state::pyobject_gcarray_descr(),
                );
                let expected = ctx.trace_ctx.const_int(int_val);
                walker_emit_guard_with_snapshot(
                    ctx,
                    op_pc,
                    OpCode::GuardValue,
                    &[length, expected],
                )?;
                write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, seq)?;
                return Ok(Some(()));
            }
            majit_ir::RuntimeHelperKind::UnpackItem => {
                let index = int_val as usize;
                if index >= concrete_len {
                    return Ok(None);
                }
                // `index < concrete_len` only holds for the tuple recorded
                // here. A trace can enter between the `UnpackSequence` helper
                // and this one — a bridge resumes mid-unpack — so this arm
                // cannot rely on that helper's length guard being in the same
                // trace. Emit it; when it is present the optimizer folds this
                // one away, so a whole unpack still guards the length once.
                let length = crate::state::opimpl_arraylen_gc(
                    ctx.trace_ctx,
                    items,
                    crate::state::pyobject_gcarray_descr(),
                );
                let expected = ctx.trace_ctx.const_int(concrete_len as i64);
                walker_emit_guard_with_snapshot(
                    ctx,
                    op_pc,
                    OpCode::GuardValue,
                    &[length, expected],
                )?;
                let index_op = ctx.trace_ctx.const_int(int_val);
                let item = crate::state::trace_items_block_getitem_value_pure(
                    ctx.trace_ctx,
                    items,
                    index_op,
                );
                let concrete_item = unsafe {
                    pyre_object::w_tuple_getitem(concrete_seq, int_val)
                        .unwrap_or(pyre_object::PY_NULL)
                };
                ctx.trace_ctx.set_opref_concrete(
                    item,
                    majit_ir::Value::Ref(majit_ir::GcRef(concrete_item as usize)),
                );
                write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, item)?;
                return Ok(Some(()));
            }
            _ => {}
        }
    }
    // Both arity-2 specialisations fold; any other shape (a plain tuple, a
    // list, or a non-canonical tuple) falls through to the opaque residual
    // (correct, slower).
    let seq_type = unsafe { (*concrete_seq).ob_type };
    let Some(pair_kind) = specialised_pair_kind(seq_type) else {
        return Ok(None);
    };
    let spec_type = seq_type;
    match helper {
        majit_ir::RuntimeHelperKind::UnpackSequence => {
            // Either specialisation is always arity 2, so the class guard
            // subsumes the exact-length check `unpack_sequence_fn` performs.
            if int_val != 2 {
                return Ok(None);
            }
            walker_guard_specialised_pair_class(ctx, op_pc, seq, spec_type)?;
            // Pass `seq` through as the validated tuple; the per-index
            // `unpack_item_fn` reads below fold off it.
            write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, seq)?;
            Ok(Some(()))
        }
        majit_ir::RuntimeHelperKind::UnpackItem => {
            if !(0..2).contains(&int_val) {
                return Ok(None);
            }
            // Normally the partner `unpack_sequence_fn` fold already guarded
            // the class (its validated-tuple passthrough reg == `seq`), in
            // which case this is a no-op; guard here too so a fold that only
            // catches the item reads still proves the layout it loads from.
            walker_guard_specialised_pair_class(ctx, op_pc, seq, spec_type)?;
            let Some(item) = walker_emit_specialised_pair_item(
                ctx, op_pc, seq, pair_kind, int_val, allboxes, call_descr,
            )?
            else {
                return Ok(None);
            };
            write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, item)?;
            Ok(Some(()))
        }
        _ => Ok(None),
    }
}

/// The tuple `descr_getargs` would build from an exception's `args_w` list.
/// Built only to read the representation `newtuple` picks off it, so the caller
/// keeps the shape and drops the tuple.
unsafe fn args_tuple_shape_probe(stored: pyre_object::PyObjectRef) -> pyre_object::PyObjectRef {
    let len = unsafe { pyre_object::w_list_len(stored) };
    let items = (0..len)
        .map(|index| {
            unsafe { pyre_object::w_list_getitem(stored, index as i64) }
                .unwrap_or(pyre_object::PY_NULL)
        })
        .collect();
    pyre_object::w_tuple_new(items)
}

/// `guard_class(seq, spec)` for one of the arity-2 tuple specialisations,
/// emitted once per traced `seq` — the heap cache turns every later fold on the
/// same register into a no-op, the way upstream's optimizer keeps a single
/// class guard for a value it already proved.
fn walker_guard_specialised_pair_class<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    seq: OpRef,
    spec_type: *const pyre_object::pyobject::PyType,
) -> Result<(), DispatchError> {
    if ctx.trace_ctx.heap_cache().is_class_known(seq) {
        return Ok(());
    }
    let type_const = ctx.trace_ctx.const_int(spec_type as i64);
    ctx.trace_ctx
        .record_guard(OpCode::GuardClass, &[seq, type_const], 0);
    walker_capture_snapshot_for_last_guard(ctx, op_pc)?;
    ctx.trace_ctx
        .heap_cache_mut()
        .class_now_known(seq, spec_type as i64);
    Ok(())
}

/// Read slot `index` (0 or 1) of an arity-2 tuple specialisation whose class
/// the caller has already guarded, applying that slot's `wraps[i]`
/// (`specialisedtupleobject.py`, and `:134-142 getitem`, which unrolls
/// `iter_n` to the matching `value%s`).
///
/// `Ok(None)` declines: the `ii` / `ff` slots need the authentic box for its
/// identity, and that execution can fail.
///
/// The `ff` arm currently has no producer to serve. Upstream builds `Cls_ff`
/// from `makespecialisedtuple2` (`specialisedtupleobject.py`) and from
/// `specialized_zip_2_lists` (`:230`); pyre does not port the latter, and
/// `w_tuple_new` (`tupleobject.rs`) sends a plain-float pair to `Cls_oo`
/// instead so that `(x, x)` keeps the exact `x` object. It is kept because it
/// is the layout upstream reads, not because a trace reaches it today.
#[allow(clippy::too_many_arguments)]
fn walker_emit_specialised_pair_item<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    seq: OpRef,
    pair_kind: SpecialisedPairKind,
    index: i64,
    allboxes: &[OpRef],
    call_descr: &dyn majit_ir::descr::CallDescr,
) -> Result<Option<OpRef>, DispatchError> {
    let first = index == 0;
    if pair_kind == SpecialisedPairKind::Object {
        let descr = if first {
            crate::descr::specialised_tuple_oo_value0_descr()
        } else {
            crate::descr::specialised_tuple_oo_value1_descr()
        };
        // `wraps[i]` is the identity for an object slot, so the field read is
        // the whole item — no re-boxing, and no `may_force` execution needed
        // to recover a box identity.
        return Ok(Some(crate::state::opimpl_getfield_gc_r(
            ctx.trace_ctx,
            seq,
            descr,
        )));
    }
    // Authentic boxed element supplies the concrete shadow / identity while
    // the emitted field read and transparent wrapper replace the residual call
    // in machine code. Fall through if execution raises or cannot provide that
    // box.
    let Some(elem_ptr) = walker_execute_may_force_boxed(ctx, allboxes, call_descr) else {
        return Ok(None);
    };
    if pair_kind == SpecialisedPairKind::Float {
        let descr = if first {
            crate::descr::specialised_tuple_ff_value0_descr()
        } else {
            crate::descr::specialised_tuple_ff_value1_descr()
        };
        let raw = majit_metainterp::box_trace::getfield_gc_f_pureornot(ctx.trace_ctx, seq, descr);
        let elem = unsafe { pyre_object::w_float_get_value(elem_ptr as pyre_object::PyObjectRef) };
        ctx.trace_ctx
            .set_opref_concrete(raw, majit_ir::Value::Float(elem));
        let boxed = crate::state::wrapfloat(ctx.trace_ctx, raw);
        ctx.trace_ctx.set_opref_concrete(
            boxed,
            majit_ir::Value::Ref(majit_ir::GcRef(elem_ptr as usize)),
        );
        return Ok(Some(boxed));
    }
    // `ii`: preserve authentic small-int caching / identity.
    let descr = if first {
        crate::descr::specialised_tuple_ii_value0_descr()
    } else {
        crate::descr::specialised_tuple_ii_value1_descr()
    };
    let raw = crate::state::opimpl_getfield_gc_i(ctx.trace_ctx, seq, descr);
    let elem = unsafe { pyre_object::w_int_get_value(elem_ptr as pyre_object::PyObjectRef) };
    let boxed = walker_box_int(ctx, op_pc, raw, elem)?;
    ctx.trace_ctx
        .set_opref_concrete(boxed, box_int_concrete(elem, elem_ptr as i64));
    Ok(Some(boxed))
}

/// One hop of a `while tb is not None: names.append(tb.tb_frame.f_code.co_name);
/// tb = tb.tb_next` traceback walk.
///
/// Each of these is a `GetSetProperty` whose getter body is a slot read on a
/// receiver [`walker_specialize_traceback_walk_field`] pins by class:
/// `pytraceback.py descr_get_next` / `descr_get_tb_frame` /
/// `descr_get_tb_lineno` / `descr_get_tb_lasti` and `pyframe.py fget_code`.
/// None of them dispatches anywhere or can raise.  Left residual, every hop of
/// the walk costs a forcing call — measured at 207 ns per `tb_lineno` read
/// against 0 for the folded `tb_next` — which is what makes each traceback
/// fixture dominated by the walk rather than by the raise.
#[derive(Clone, Copy, PartialEq, Eq)]
enum TracebackWalkField {
    /// `tb.tb_next` — the chain link; a null slot is the terminator and
    /// surfaces as `None`.
    TbNext,
    /// `tb.tb_frame` — the node's frame.  Unlike its two siblings the getter
    /// ALSO runs `mark_as_escaped()`; see the escape emit.
    TbFrame,
    /// `frame.f_code` — `fget_f_code` is `self.pycode as PyObjectRef`.
    FCode,
    /// `tb.tb_lineno` — the line the node froze at.  The getter resolves the
    /// sentinel out of `w_code` and `lasti`; `record_application_traceback`
    /// stamps the real line instead, so a recorded node reads as the slot and
    /// only a hand-constructed one has to resolve.  The fold covers the stamped
    /// case and declines the other.
    ///
    /// `tb_lasti` is deliberately absent: it is the one traceback slot the
    /// walker has no reason to reach, since nothing on the walk consumes it.
    TbLineno,
    /// `code.co_name` — `code_get_field` answers it with `w_code_name_obj`,
    /// which realizes the string once and retains it on the code object, so
    /// every later read is the retained slot.
    CoName,
    /// `code.co_firstlineno` — the `co_firstlineno_raw` slot, reboxed.  The
    /// other code fields are deliberately absent: they read the host
    /// `CodeObject` behind `code_ptr` rather than a slot on the `PyCode`, so
    /// folding one means a raw load through a second indirection.
    CoFirstlineno,
}

/// Which walk hop, if any, this `(receiver, attribute)` pair is.
fn traceback_walk_field(
    concrete_obj: pyre_object::PyObjectRef,
    name: &str,
) -> Option<TracebackWalkField> {
    let ob_type = unsafe { (*concrete_obj).ob_type };
    if std::ptr::eq(ob_type, &pyre_interpreter::pytraceback::PYTRACEBACK_TYPE) {
        return match name {
            "tb_next" => Some(TracebackWalkField::TbNext),
            "tb_frame" => Some(TracebackWalkField::TbFrame),
            "tb_lineno" => Some(TracebackWalkField::TbLineno),
            _ => None,
        };
    }
    if std::ptr::eq(ob_type, &pyre_interpreter::pyframe::FRAME_TYPE) && name == "f_code" {
        return Some(TracebackWalkField::FCode);
    }
    if std::ptr::eq(ob_type, &pyre_interpreter::pycode::CODE_TYPE) {
        return match name {
            "co_name" => Some(TracebackWalkField::CoName),
            "co_firstlineno" => Some(TracebackWalkField::CoFirstlineno),
            _ => None,
        };
    }
    None
}

/// Runtime half of the optimized-frame `f_locals` getter.  The proxy owns the
/// exact frame passed to it; reading or mutating the proxy later goes through
/// that frame's existing synchronization path.
extern "C" fn jit_inline_frame_locals_proxy_new(frame: i64) -> i64 {
    pyre_interpreter::pyframe::frame_locals_proxy::new(frame as pyre_object::PyObjectRef) as i64
}

/// Prove the receiving code object still owns its host `CodeObject`, the
/// `require_code` check every code-field getter runs before reading a slot.
fn walker_guard_code_ptr_present<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    obj: OpRef,
    concrete_obj: pyre_object::PyObjectRef,
) -> Result<(), DispatchError> {
    let code_ptr = crate::state::opimpl_getfield_gc_i(
        ctx.trace_ctx,
        obj,
        crate::descr::pycode_code_ptr_descr(),
    );
    let live = unsafe { pyre_interpreter::w_code_get_ptr(concrete_obj) } as i64;
    ctx.trace_ctx
        .set_opref_concrete(code_ptr, majit_ir::Value::Int(live));
    let zero = ctx.trace_ctx.const_int(0);
    let absent = ctx.trace_ctx.record_op(OpCode::IntEq, &[code_ptr, zero]);
    walker_emit_fold_guard_with_snapshot(ctx, op_pc, OpCode::GuardFalse, &[absent])
}

/// Emit one traceback-walk hop as a guarded inline field read instead of the
/// opaque `getattr_fn` residual.
///
/// Returns `None` (fall through to the residual) BEFORE recording any guard for
/// every shape it cannot settle — an uncacheable `version_tag`, or a null slot
/// on a hop whose null is not a documented value.  A bail-out after a guard
/// would leave the caller reading the attribute as already pinned.
fn walker_specialize_traceback_walk_field<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    obj: OpRef,
    concrete_obj: pyre_object::PyObjectRef,
    field: TracebackWalkField,
    dst: usize,
    dst_bank: char,
) -> Result<Option<()>, DispatchError> {
    use pyre_interpreter::pyframe::PyFrame;

    let receiver_type = match field {
        TracebackWalkField::FCode => &pyre_interpreter::pyframe::FRAME_TYPE,
        TracebackWalkField::CoName | TracebackWalkField::CoFirstlineno => {
            &pyre_interpreter::pycode::CODE_TYPE
        }
        _ => &pyre_interpreter::pytraceback::PYTRACEBACK_TYPE,
    };
    let descr = match field {
        TracebackWalkField::TbNext => crate::descr::pytraceback_w_next_descr(),
        TracebackWalkField::TbFrame => crate::descr::pytraceback_frame_descr(),
        TracebackWalkField::FCode => crate::descr::pyframe_code_descr(),
        TracebackWalkField::TbLineno => crate::descr::pytraceback_lineno_descr(),
        TracebackWalkField::CoName => crate::descr::pycode_w_name_descr(),
        TracebackWalkField::CoFirstlineno => crate::descr::pycode_co_firstlineno_descr(),
    };
    let w_type = pyre_interpreter::typedef::gettypeobject(receiver_type);
    let version_tag = unsafe { pyre_object::typeobject::w_type_get_version_tag(w_type) };
    if version_tag == 0 {
        return Ok(None);
    }
    // The slot guard pins the receiver's `w_class` against `w_type`.  A frame
    // built before `init_typeobjects` carries a null `w_class`, which would
    // make that guard fail on its first execution, so decline instead of
    // recording a doomed trace.
    if unsafe { (*concrete_obj).w_class } != w_type {
        return Ok(None);
    }

    // Every code-field getter resolves the host `CodeObject` first
    // (`code_get_field` -> `require_code`) and raises when it is absent, so a
    // code fold owes that check.  It is a slot on the receiver, so the trace
    // proves it the same way — a read plus a non-null guard — rather than
    // trusting the record-time object.
    let code_receiver = matches!(
        field,
        TracebackWalkField::CoName | TracebackWalkField::CoFirstlineno
    );
    if code_receiver
        && unsafe { pyre_interpreter::w_code_get_ptr(concrete_obj) }
            .cast::<u8>()
            .is_null()
    {
        return Ok(None);
    }

    if field == TracebackWalkField::CoFirstlineno {
        let live =
            i64::from(unsafe { pyre_interpreter::pycode::w_code_firstlineno_raw(concrete_obj) });
        walker_guard_exception_attr_slot(ctx, op_pc, obj, concrete_obj, w_type, version_tag)?;
        walker_guard_code_ptr_present(ctx, op_pc, obj, concrete_obj)?;
        let raw_value = crate::state::opimpl_getfield_gc_i(ctx.trace_ctx, obj, descr);
        ctx.trace_ctx
            .set_opref_concrete(raw_value, majit_ir::Value::Int(live));
        // Reboxed for the same reason `TbLineno` is: the getter hands back a
        // Python int, and the boxed op is a heap `NewWithVtable`.
        let boxed = walker_box_int(ctx, op_pc, raw_value, live)?;
        let live_ptr = pyre_object::w_int_new(live) as i64;
        ctx.trace_ctx
            .set_opref_concrete(boxed, box_int_concrete(live, live_ptr));
        write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, boxed)?;
        return Ok(Some(()));
    }

    if field == TracebackWalkField::TbLineno {
        let live =
            unsafe { pyre_interpreter::pytraceback::w_pytraceback_get_lineno_raw(concrete_obj) };
        // A slot holding the sentinel is not the getter's value — the getter
        // resolves it out of `w_code` and `lasti`, which are two more slots
        // this fold would have to pin — so the slot value is the getter's value
        // only once it is pinned against the sentinel.  A node that already
        // carries the sentinel — built from a frame with no `pycode`, or handed
        // it through `TracebackType(..., -1)` — has nothing to pin, so decline
        // before recording anything.
        if live == pyre_interpreter::pytraceback::LINENO_NOT_COMPUTED {
            return Ok(None);
        }
        walker_guard_exception_attr_slot(ctx, op_pc, obj, concrete_obj, w_type, version_tag)?;
        let raw_value = crate::state::opimpl_getfield_gc_i(ctx.trace_ctx, obj, descr);
        ctx.trace_ctx
            .set_opref_concrete(raw_value, majit_ir::Value::Int(live));
        let not_computed = ctx
            .trace_ctx
            .const_int(pyre_interpreter::pytraceback::LINENO_NOT_COMPUTED);
        let is_not_computed = ctx
            .trace_ctx
            .record_op(OpCode::IntEq, &[raw_value, not_computed]);
        walker_emit_fold_guard_with_snapshot(ctx, op_pc, OpCode::GuardFalse, &[is_not_computed])?;
        // The getter returns a Python int, so the raw slot is reboxed the way
        // the unboxed mapdict read does; the boxed op is a heap `NewWithVtable`
        // so its concrete has to be a heap pointer too.
        let boxed = walker_box_int(ctx, op_pc, raw_value, live)?;
        let live_ptr = pyre_object::w_int_new(live) as i64;
        ctx.trace_ctx
            .set_opref_concrete(boxed, box_int_concrete(live, live_ptr));
        write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, boxed)?;
        return Ok(Some(()));
    }

    let stored = match field {
        TracebackWalkField::TbNext => unsafe {
            pyre_interpreter::pytraceback::w_pytraceback_get_w_next(concrete_obj)
        },
        TracebackWalkField::TbFrame => {
            (unsafe { pyre_interpreter::pytraceback::w_pytraceback_get_frame(concrete_obj) })
                as pyre_object::PyObjectRef
        }
        // `w_name` is realized on first demand, so an unread code object
        // carries a null here; that declines below and the residual realizes
        // it for the next attempt.
        TracebackWalkField::CoName => unsafe {
            (*(concrete_obj as *const pyre_interpreter::pycode::PyCode)).w_name
        },
        _ => (unsafe { (*(concrete_obj as *const PyFrame)).pycode }) as pyre_object::PyObjectRef,
    };
    // Only `tb_next` has a null with a defined meaning.  A null frame is a
    // torn-down traceback and a null `pycode` a half-built frame; both are
    // answered by a `sys.namespace` stub or `None` the residual owns.
    if stored.is_null() && field != TracebackWalkField::TbNext {
        return Ok(None);
    }
    walker_guard_exception_attr_slot(ctx, op_pc, obj, concrete_obj, w_type, version_tag)?;
    if code_receiver {
        walker_guard_code_ptr_present(ctx, op_pc, obj, concrete_obj)?;
    }
    let raw_value = crate::state::opimpl_getfield_gc_r(ctx.trace_ctx, obj, descr);
    let value = if stored.is_null() {
        // End of the chain.  A nullity test is `pyjitpl.py
        // _establish_nullity`'s GUARD_ISNULL plus a `replace_box` onto the null
        // constant `constant_from_op` gives it — not a promote.  The
        // distinction is load-bearing: `compile.py
        // make_a_counter_per_value` keys a GUARD_VALUE's jitcounter on the
        // *failing value*, and this slot holds a different PyTraceback on every
        // walk, so no one value here ever reaches `trace_eagerness` and the
        // continuation for a non-null link never gets a bridge.
        // Stamped ahead of the guard because `stamp_guard_value_concrete` only
        // does it for a GUARD_VALUE, and the snapshot the guard captures reads
        // the slot's concrete.
        ctx.trace_ctx
            .set_opref_concrete(raw_value, majit_ir::Value::Ref(majit_ir::GcRef(0)));
        walker_emit_fold_guard_with_snapshot(ctx, op_pc, OpCode::GuardIsnull, &[raw_value])?;
        let null_const = ctx.trace_ctx.const_ref(0);
        ctx.trace_ctx.replace_box(raw_value, null_const);
        ctx.trace_ctx.const_ref(pyre_object::w_none() as i64)
    } else {
        walker_emit_fold_guard_with_snapshot(ctx, op_pc, OpCode::GuardNonnull, &[raw_value])?;
        ctx.trace_ctx.set_opref_concrete(
            raw_value,
            majit_ir::Value::Ref(majit_ir::GcRef(stored as usize)),
        );
        raw_value
    };

    if field == TracebackWalkField::TbFrame {
        // `descr_get_tb_frame` also runs `frame.mark_as_escaped()`
        // (`pyframe.py mark_as_escaped`): the reference it hands out has to
        // keep the frame materialised.  `set_escaped` ORs `FLAG_ESCAPED` into
        // the `flags` byte, so the trace reads that byte, sets the bit, and
        // stores it back.
        //
        // The bit has to be set BY THE TRACE, not only stamped now: the trace
        // is reused, and each replay walks a different traceback naming a
        // different frame, so a trace-time-only mark would leave every later
        // frame unmarked.  The concrete write below is the one the
        // authoritative walk's residual executor would have performed, applied
        // here for the same reason `try_walker_lower_exc_info_residual` applies
        // its own.
        let flags_descr = crate::descr::pyframe_flags_descr();
        let live_flags =
            crate::state::opimpl_getfield_gc_i(ctx.trace_ctx, raw_value, flags_descr.clone());
        let escaped_bit = ctx.trace_ctx.const_int(i64::from(PyFrame::FLAG_ESCAPED));
        let new_flags = ctx
            .trace_ctx
            .record_op(OpCode::IntOr, &[live_flags, escaped_bit]);
        ctx.trace_ctx.record_op_with_descr(
            OpCode::SetfieldGc,
            &[raw_value, new_flags],
            flags_descr.clone(),
        );
        ctx.trace_ctx
            .heapcache_setfield_cached(raw_value, flags_descr.index(), new_flags);
        unsafe { (*(stored as *mut PyFrame)).mark_as_escaped() };
    }

    write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, value)?;
    Ok(Some(()))
}

/// Write the standard virtualizable's locals region into the array a folded
/// `f_locals` hands out.
///
/// `pyframe.py fast2locals` — the body behind `getdictscope`, and so behind
/// `f_locals` — is `@jit.unroll_safe`, and the `locals_cells_stack_w[i]` reads
/// it unrolls are `getarrayitem_vable_r` against the virtualizable BOXES.  The
/// getter therefore neither forces the virtualizable nor reads its array, which
/// is what makes folding it legitimate at all.
///
/// pyre answers `f_locals` with the 3.14 `FrameLocalsProxy`, which reads the
/// frame's array lazily instead of copying out of it at the call, so the values
/// have to be IN that array by the time the proxy is handed out.
/// `pyjitpl.py synchronize_virtualizable` (`virtualizable.py write_boxes`) is
/// the write-back that puts them there.  Upstream runs it against the
/// recording-time virtualizable after every vable store; both halves are
/// needed here, because the values have to be in the array for the walk's own
/// read of the proxy AND for the compiled run's.  So this mirrors the region
/// onto the concrete frame and emits the same store into the trace.  Without
/// it the residual getter's read barrier was the only thing writing the region
/// out, and folding the getter silently dropped every local the traced body
/// had assigned.
///
/// Only the locals/cells region is written back.  `write_boxes` covers the
/// whole array, but the operand-stack region above `nlocals` is not reachable
/// through the proxy and its shadow slots read NULL outside a merge point
/// (see [`crate::state::flush_locals_region_to_frame`]), so writing those back
/// would destroy the values the walk is holding.
///
/// A slot the shadow cannot answer declines the whole write-back, and with it
/// the fold, leaving the residual force in place.  The validation pass runs
/// before the first emission, so a decline emits nothing.
pub(crate) fn walker_write_back_standard_frame_locals<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    frame_op: OpRef,
    concrete_frame: usize,
) -> bool {
    let Some(info) = ctx.trace_ctx.virtualizable_info().cloned() else {
        return false;
    };
    let base = info.num_static_extra_boxes;
    let Some(nlocals) = crate::state::concrete_nlocals(concrete_frame) else {
        return false;
    };
    // `Value::Void` is the shadow's "no concrete half" sentinel rather than an
    // unbound local, so a slot carrying it cannot be written back.
    let mut slots = Vec::with_capacity(nlocals);
    for slot in 0..nlocals {
        match ctx.trace_ctx.virtualizable_entry_at(base + slot) {
            Some((_, majit_ir::Value::Void)) | None => return false,
            Some((value, _)) => slots.push((slot as i64, value)),
        }
    }
    // The mirror below writes the live frame's locals array, and a walk that
    // does not commit replays from its pre-walk instruction — so the pre-walk
    // values have to be recoverable.  Journal them against the walk's own
    // non-commit epilogue rather than the escape-flush capture: that capture is
    // consumed by every non-forcing residual (`try_execute_residual_call_via_
    // executor`'s tail restore), which would revert this mirror mid-walk and
    // leave a live `FrameLocalsProxy` reading pre-fold values.
    crate::jitcode_dispatch::fbw_note_locals_mirror_undo(concrete_frame, nlocals);
    if !crate::state::flush_locals_region_to_frame(ctx.trace_ctx, concrete_frame) {
        // All-or-nothing decline: nothing was written.  The journal entry is
        // harmless — restoring the values still in place is a no-op — and the
        // first-per-frame rule means dropping it could discard a real one.
        return false;
    }
    ctx.trace_ctx
        .vable_array_region_write_back(frame_op, 0, &slots)
}

/// The frame box and EXECUTING Python pc of a frame receiver the walk owns,
/// or `None` for one it does not.
///
/// `pyjitpl.py` keeps one MIFrame per inlined call and each carries its own
/// coordinate, so this resolves per level exactly as
/// [`LiveLastInstrGuard::enter`] retargets its publication: inside an inline
/// sub-walk the callee's own jitcode pc resolved through the callee's
/// metadata, at the portal the walk's `vstack_cur_pypc`.
///
/// The virtualizable boxes are NOT a source here, which a probe against the
/// residual's own answers settled rather than an argument: at three portal
/// sites the boxes' `last_instr` entry read 110/165/185 against executing
/// pcs of 115/170/197, and at an inlined-callee site it read 232 -- the
/// caller's CALL boundary -- against the callee's own 22.  They describe the
/// PORTAL frame and carry whichever of the field's two conventions their last
/// writer left.
///
/// A receiver that is not this level's own frame declines, which is what keeps
/// a suspended generator's frame, a traceback node's frame and a caller's
/// frame read from inside a callee on the residual getter that reads the heap.
/// The box and concrete address of the frame the walk is executing.
///
/// Two sources, chosen by whether a sub-walk is active, because they describe
/// different frames: the portal's virtualizable describes the PORTAL, so an
/// inlined callee has to answer from its own shadow or it reports its caller's.
fn walker_executing_frame_box<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
) -> Option<(OpRef, usize)> {
    let inline_frame = current_inline_concrete_frame();
    if inline_frame != 0 {
        let shadow = ctx.callee_shadow.as_ref()?;
        if shadow.concrete_frame != inline_frame || shadow.frame_box == OpRef::NONE {
            return None;
        }
        return Some((shadow.frame_box, inline_frame));
    }
    if ctx.fbw_mode.inline_subwalk {
        return None;
    }
    let vable_box = ctx.trace_ctx.standard_virtualizable_box()?;
    let vable_ptr = ctx.trace_ctx.standard_virtualizable_ptr()?;
    Some((vable_box, vable_ptr))
}

fn walker_frame_executing_py_pc<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
    concrete_obj: pyre_object::PyObjectRef,
    op_pc: usize,
) -> Option<(OpRef, u32)> {
    let (frame_box, frame_ptr) = walker_executing_frame_box(ctx)?;
    if frame_ptr != concrete_obj as usize {
        return None;
    }
    if current_inline_concrete_frame() != 0 {
        return Some((frame_box, residual_call::inline_callee_py_pc(ctx, op_pc)?));
    }
    // An inline sub-walk with no concrete callee frame has no level-local
    // coordinate to answer with: `vstack_cur_pypc` is the outer walk's mirror
    // and a sub-walk never advances it.
    if !ctx.vstack_valid {
        return None;
    }
    Some((frame_box, ctx.vstack_cur_pypc))
}

/// Prove the receiver IS the frame the walk is executing, and answer that
/// frame's executing pc.
///
/// Shared by the two owned-frame getter folds, which answer a coordinate the
/// walk holds rather than the one the frame's own field records and therefore
/// owe the same proof about the object in hand.
///
/// The receiver is pinned two ways.  Its class, its `w_class` and the frame
/// type's `version_tag` are guarded, so rebinding the getset on the type
/// revokes the loop instead of the fold outliving the descriptor that produced
/// it.  And when the receiver arrives in a box other than the frame's own —
/// a local the loop hoisted the frame into — a `ptr_eq` against that box is
/// guarded, so a later entry holding a different frame side-exits to the
/// residual rather than reading this trace's coordinate.
fn walker_prove_owned_frame_pc<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    obj: OpRef,
    concrete_obj: pyre_object::PyObjectRef,
) -> Result<Option<u32>, DispatchError> {
    let Some((frame_box, py_pc)) = walker_frame_executing_py_pc(ctx, concrete_obj, op_pc) else {
        return Ok(None);
    };
    let w_type = pyre_interpreter::typedef::gettypeobject(&pyre_interpreter::pyframe::FRAME_TYPE);
    let version_tag = unsafe { pyre_object::typeobject::w_type_get_version_tag(w_type) };
    if version_tag == 0 || unsafe { (*concrete_obj).w_class } != w_type {
        return Ok(None);
    }
    walker_guard_exception_attr_slot(ctx, op_pc, obj, concrete_obj, w_type, version_tag)?;
    if obj != frame_box {
        let is_own_frame = ctx.trace_ctx.record_op(OpCode::PtrEq, &[obj, frame_box]);
        ctx.trace_ctx
            .set_opref_concrete(is_own_frame, majit_ir::Value::Int(1));
        walker_emit_fold_guard_with_snapshot(ctx, op_pc, OpCode::GuardTrue, &[is_own_frame])?;
    }
    Ok(Some(py_pc))
}

/// `pyframe.py fget_f_lasti` — `return self.last_instr`, loop-free and
/// carrying no hint, so `policy.py look_inside_graph` admits it, `jtransform.py
/// rewrite_op_jit_force_virtualizable` deletes the force
/// `rvirtualizable.py hook_access_field` injects, and `pyjitpl.py
/// opimpl_getfield_vable_i` answers the field out of `virtualizable_boxes`.
/// The box there is a `ConstInt`, because the only writer is the bytecode
/// dispatch's `_opimpl_setfield_vable` of the pc it is about to run — which is
/// why upstream answers the read for less than a loop that does not perform
/// it.  Nothing forces, so the generic reader's residual boundary buys nothing
/// and this emits the same constant.
///
/// The constant owes two coordinates the residual boundary hides.
/// `last_instr` is an instruction-unit index here while the getset reports the
/// byte offset (`typedef.rs` returns `fget_f_lasti() * 2`), so the emission
/// carries the factor; without it a `dis` consumer's `f_lasti // 2` lands on
/// half the instruction index.  And the field has two writers on two
/// conventions — `flush_walk_end_state_to_frame` stores the resume coordinate
/// `pc - 1`, `LiveLastInstrGuard::enter_frame` stores the executing pc
/// unshifted — and a getter owes the executing one, which is what
/// [`walker_frame_executing_py_pc`] resolves.
///
/// `last_instr` travels as half of a pair — `capture_frame_scalars` records it
/// beside `valuestackdepth` because the interpreter derives its next opcode
/// from `last_instr + 1` and reads the operand stack at `valuestackdepth`, so
/// a consumer restoring one of them owes the other.  This emission assumes
/// nothing about `valuestackdepth` and is entitled to: it is a pure read that
/// never reaches the frame at all.  The pc it answers with comes from the
/// walk's own trace-time coordinate, not from the frame's field, so no state
/// is captured, none is restored, and the pair is never split.  Writing
/// `last_instr` from here would incur that obligation, which is the second
/// reason this path never does.
///
/// The receiver is pinned by [`walker_prove_owned_frame_pc`].
fn try_walker_specialize_frame_lasti<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    obj: OpRef,
    concrete_obj: pyre_object::PyObjectRef,
    dst: usize,
    dst_bank: char,
) -> Result<Option<()>, DispatchError> {
    let Some(py_pc) = walker_prove_owned_frame_pc(ctx, op_pc, obj, concrete_obj)? else {
        return Ok(None);
    };
    let value = py_pc as i64 * 2;
    let raw = ctx.trace_ctx.const_int(value);
    let boxed = walker_box_int(ctx, op_pc, raw, value)?;
    // `walker_box_int` emits a heap `NewWithVtable`, so the recording-time
    // shadow has to be a heap `W_IntObject` too — the same pairing
    // [`box_int_concrete`] makes for a residual whose result arrived tagged.
    ctx.trace_ctx.set_opref_concrete(
        boxed,
        majit_ir::Value::Ref(majit_ir::GcRef(
            pyre_object::intobject::w_int_new_unique(value) as usize,
        )),
    );
    write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, boxed)?;
    Ok(Some(()))
}

/// Runtime half of the owned-frame `f_lineno` getter: `pyframe.py
/// fget_f_lineno` with the executing `last_instr` supplied by its caller
/// instead of read back off the frame.
extern "C" fn jit_frame_f_lineno_at(frame: i64, last_instr: i64) -> i64 {
    let frame = frame as usize as *const pyre_interpreter::PyFrame;
    unsafe { &*frame }.f_lineno_at(last_instr as isize) as i64
}

/// `pyframe.py fget_f_lineno` — the line the frame is currently executing.
///
/// Unlike its `f_lasti` sibling this is **not** a constant, and upstream's
/// compiled shape is not one either.  `policy.py look_inside_graph` admits
/// `fget_f_lineno` and `pyjitpl.py opimpl_getfield_vable_r` answers the
/// `debugdata` test out of `virtualizable_boxes`, but the decode underneath —
/// `pytraceback.py offset2lineno` walking the line table — stays a residual
/// call.  One non-forcing call is the shape to emit.
///
/// What this removes is the FORCE, not the call.  The generic reader
/// residualizes `space.getattr` as a single `CALL_MAY_FORCE`, and a may-force
/// boundary materializes the virtualizable — which is the only reason the
/// getter could read `last_instr` off the frame at all, since that field is
/// virtualizable and a compiled loop keeps the live coordinate in its own
/// state.  Handing the leaf the coordinate the walk already holds
/// ([`walker_prove_owned_frame_pc`]) removes that reason, leaving a leaf call
/// that names the frame without forcing it.
///
/// The leaf is [`PyFrame::f_lineno_at`], i.e. the getter body whole.  Its
/// `f_trace` test, `-1` sentinel and `first_line_number` fallback are one
/// decision, and `w_f_trace` can be armed while the loop is already compiled,
/// so that decision belongs at run time rather than baked into the trace.
///
/// Measured on a 200k-iteration read against a same-shape loop that does not
/// read the frame, best of 5: the read costs 0.0274s through the residual and
/// 0.0069s through this emission, against 0.0084s on CPython 3.14.6 and
/// 0.0227s on pypy3.  What removing the boundary is worth is counted rather
/// than inferred — the optimized trace loses half its `CALL_MAY_FORCE` (16 ->
/// 8) and half its `GuardNotForced` (16 -> 8) — and the two arms report the
/// same `loops_compiled`, `loops_aborted` and `guard_failures`, so the
/// difference is the emission and not one arm compiling less.
///
/// The call states an empty `EffectInfo` descr set, which is not a claim that
/// the leaf reads nothing: `make_call_descr_sized` panics on a non-empty raw
/// descr set minted after `compute_bitstrings`, and `finish_setup_descrs` runs
/// before any trace does, so every trace-time residual states the empty set and
/// `extraeffect` carries what is claimed.
///
/// The empty set is inert because no trace op names a field the leaf reads, and
/// `force_from_effectinfo` forces only descrs already in `cached_fields`.  The
/// reads are `PyFrame.pycode` and `PyFrame.debugdata`, then
/// `FrameDebugData.w_f_trace` and the code object's `linetable` and
/// `first_line_number`; `pyframe_debugdata_descr` has no emitter, `w_f_trace`
/// has no descr at all, and neither vable slot has a writer.  A later fold that
/// gives one of them a descr and caches it owes this call an explicit op, the
/// way `ResolveExceptionContext` records its own `SetfieldGc` rather than
/// naming `w_context` in a write set it cannot carry.
///
/// `pycode` and `debugdata` are read off the frame in memory rather than
/// through `virtualizable_entry_at`, where the neighbouring `locals()` fold
/// reads `debugdata`.  Neither slot has a `setfield_vable` writer
/// (`virtualizable_spec.rs` names the ones that do), so memory holds the live
/// value while compiled code runs, while the recording-time shadow's
/// `debugdata` is a `clone_debugdata_ptr` copy of it.  Memory answers the same
/// at both times; the shadow does not.
///
/// The receiver is pinned by [`walker_prove_owned_frame_pc`].
fn try_walker_specialize_frame_lineno<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    obj: OpRef,
    concrete_obj: pyre_object::PyObjectRef,
    dst: usize,
    dst_bank: char,
) -> Result<Option<()>, DispatchError> {
    let Some(py_pc) = walker_prove_owned_frame_pc(ctx, op_pc, obj, concrete_obj)? else {
        return Ok(None);
    };
    let pc = ctx.trace_ctx.const_int(py_pc as i64);
    let value = ctx.trace_ctx.call_ref_typed_with_effect(
        jit_frame_f_lineno_at as *const (),
        &[obj, pc],
        &[majit_ir::Type::Ref, majit_ir::Type::Int],
        majit_ir::EffectInfo::new(
            majit_ir::ExtraEffect::CannotRaise,
            majit_ir::OopSpecIndex::None,
        ),
    );
    // The recording-time shadow comes from the same entry point the leaf calls,
    // so it carries the getter's own small-int caching rather than a second
    // rendering of it.
    let concrete = unsafe {
        (*(concrete_obj as *const pyre_interpreter::PyFrame)).f_lineno_at(py_pc as isize)
    };
    ctx.trace_ctx.set_opref_concrete(
        value,
        majit_ir::Value::Ref(majit_ir::GcRef(concrete as usize)),
    );
    write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, value)?;
    Ok(Some(()))
}

/// Whether `localsplus[0]` itself needs a cell dereference, and the slot the
/// `__class__` cell occupies in `code`.
///
/// This is `pyframe.py _get_self_location` plus
/// `builtins.rs super_operands_from_frame`: a positional argument is required,
/// and when its name is also a cellvar the fast-local slot holds the `Cell`
/// shared with closures rather than the receiver directly.
fn bare_super_frame_layout(code: &pyre_interpreter::CodeObject) -> Option<(bool, usize)> {
    if code.arg_count == 0 {
        return None;
    }
    let self_is_cell = code
        .varnames
        .first()
        .is_some_and(|first| code.cellvars.iter().any(|cell| cell == first));
    let class_freevar = code.freevars.iter().position(|name| name == "__class__")?;
    let class_slot =
        code.varnames.len() + pyre_interpreter::pyframe::npure_cellvars(code) + class_freevar;
    Some((self_is_cell, class_slot))
}

/// The two operands `builtins.rs super_operands_from_frame` reads off the
/// frame, resolved as SSA values: `localsplus[0]` and the `__class__` freevar
/// cell.
///
/// Which channel holds them is the frame's own: an inlined callee owns a
/// [`CalleeLocalsShadow`], and everything else reads the standard
/// virtualizable, the same split
/// `try_walker_specialize_builtin_locals_in_callee` draws.
///
/// Either way the entries are already there.  The inline seeds the shadow with
/// the argument operands and with the live closure-cell reads
/// (`function_closure_descr`, then the items block) it threaded into the new
/// callee frame; the portal's boxes are seeded from the frame it entered on.
/// Reading them records no op.
fn walker_bare_super_frame_slots<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
) -> Option<(OpRef, OpRef, bool)> {
    if let Some(shadow) = ctx.callee_shadow.as_ref() {
        // `u16::MAX` is the strict fresh-frame fold switched off and a `NONE`
        // frame box is a frame register that was never seeded; in neither case
        // is the shadow the authority for this level's slots.
        if shadow.fold_frame_reg == u16::MAX || shadow.frame_box.is_none() {
            return None;
        }
        // SAFETY: the code object outlives the walk that resolved it; read-only.
        let code = unsafe { shadow.code_ptr.as_ref()? };
        let (self_is_cell, class_slot) = bare_super_frame_layout(code)?;
        let class_slot = class_slot as i64;
        let slot_op = |slot: i64| -> Option<OpRef> {
            let op = shadow.opref.get(&slot).copied()?;
            // Only an entry recorded through THIS level's frame register
            // describes this frame -- the same per-frame isolation the
            // own-frame vable read applies.
            (shadow.concrete.get(&slot)?.frame_reg == shadow.fold_frame_reg).then_some(op)
        };
        return Some((slot_op(0)?, slot_op(class_slot)?, self_is_cell));
    }
    // A sub-walk that owns no shadow walks a frame the standard virtualizable
    // does not name, and a trace has exactly one of those.
    if ctx.fbw_mode.inline_subwalk || current_inline_concrete_frame() != 0 {
        return None;
    }
    // The frame `builtin_super`'s zero-argument tail reads is
    // `ExecutionContext::gettopframe()`.  Require it to BE the standard
    // virtualizable, so a hidden frame -- or any deeper one reached through the
    // backref chain -- declines rather than answering for someone else's slots.
    let vable_ptr = ctx.trace_ctx.standard_virtualizable_ptr()?;
    let ec = pyre_interpreter::call::getexecutioncontext();
    if ec.is_null() {
        return None;
    }
    let frame = unsafe { (*ec).gettopframe_nohidden() };
    if frame.is_null() || frame as usize != vable_ptr {
        return None;
    }
    // SAFETY: the frame is the live standard virtualizable; read-only.
    let code_ptr = unsafe { pyre_interpreter::pyframe::pyframe_get_pycode(&*frame) };
    let code = unsafe { code_ptr.as_ref()? };
    let (self_is_cell, class_slot) = bare_super_frame_layout(code)?;
    // `locals_cells_stack_w` is PyFrame's only virtualizable array
    // (`virtualizable_gen.rs arrays`), so array index 0 names it.
    let info = ctx.trace_ctx.virtualizable_info()?;
    let lengths = ctx.trace_ctx.virtualizable_array_lengths()?;
    if info.num_arrays() != 1 || lengths.first().copied().unwrap_or(0) <= class_slot {
        return None;
    }
    // The value comes from the SHADOW, never from the frame's heap array: an
    // unsynchronized virtualizable's array holds whatever the frame last wrote
    // out, which is the staleness the read barrier's `force_now` repairs before
    // the residual reads it.  The shadow already holds the repaired value.
    let self_op = ctx
        .trace_ctx
        .virtualizable_entry_at(info.get_index_in_array(0, 0, lengths))?
        .0;
    let cell_op = ctx
        .trace_ctx
        .virtualizable_entry_at(info.get_index_in_array(0, class_slot, lengths))?
        .0;
    Some((self_op, cell_op, self_is_cell))
}

/// Zero-argument `super()` folded to the proxy itself rather than re-routed to
/// a may-force residual.
///
/// [`try_walker_specialize_bare_super_call`] moves the frame force onto a
/// channel the walker can see, which is what keeps the loop from aborting; it
/// does not remove it.  What is left is a `MOST_GENERAL` call that publishes a
/// vref for the frame, wipes the trace's heap-field cache and is re-checked by
/// two guards, once per iteration.  Measured over 2,000,000 iterations:
/// `su = super(); su.m(x)` ran ~76ns each in an inlined callee and ~62 with
/// the loop in its own frame, against ~1.8 for `su = super(C, self); su.m(x)`.
///
/// The residual reads two frame slots and nothing else, and the walk holds
/// both as SSA values already ([`walker_bare_super_frame_slots`]), so the whole
/// call becomes the same `New` + `SetfieldGc` the two-argument spelling emits.
///
/// The class comes out of the `__class__` cell as a baked constant under the
/// `CellFamily.ever_mutated` quasi-immutable rather than a live read per
/// iteration: the cell a class body fills is written once, before any method
/// of that class can run, and `w_cell_set` marks the family the moment a
/// second write happens -- which retires this trace.  A cell that has already
/// been rebound declines here and keeps the residual.  A method whose own
/// `self` is a cellvar takes the same guarded live `Cell.contents` read as
/// LOAD_DEREF before `_super_check` sees the receiver.
pub(crate) fn try_walker_specialize_bare_super_virtual<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    code: &[u8],
    op: &DecodedOp,
    r_args: &[OpRef],
    dst: usize,
) -> Result<Option<()>, DispatchError> {
    // `super()` with no user arguments arrives as `[callable, null_or_self]`.
    if r_args.len() != 2 {
        return Ok(None);
    }
    if ctx.fbw_mode.inline_subwalk && !walker_inline_guard_resumes_in_callee(ctx) {
        return Ok(None);
    }
    let arg_concretes = read_ref_var_list_concrete(code, op, 1, ctx);
    let (ConcreteValue::Ref(concrete_callable), ConcreteValue::Ref(null_or_self)) =
        (arg_concretes[0], arg_concretes[1])
    else {
        return Ok(None);
    };
    if concrete_callable.is_null()
        || !null_or_self.is_null()
        || !pyre_interpreter::builtins::is_builtin_super_type(concrete_callable)
    {
        return Ok(None);
    }
    let Some((raw_self_op, class_cell_op, self_is_cell)) = walker_bare_super_frame_slots(ctx)
    else {
        return Ok(None);
    };
    let Some(concrete_raw_self) = walker_concrete_ref_object(ctx, raw_self_op) else {
        return Ok(None);
    };
    let (concrete_self, self_cell_ref) = if self_is_cell {
        if !unsafe { pyre_object::is_cell(concrete_raw_self) } {
            return Ok(None);
        }
        let family = unsafe { pyre_object::w_cell_family(concrete_raw_self) };
        if family.is_null() || unsafe { (*family).ever_mutated.get() } {
            return Ok(None);
        }
        let contents = unsafe { pyre_object::w_cell_get(concrete_raw_self) };
        if contents.is_null() {
            return Ok(None);
        }
        (contents, Some(majit_ir::GcRef(concrete_raw_self as usize)))
    } else {
        (concrete_raw_self, None)
    };
    let Some(concrete_cell) = walker_concrete_ref_object(ctx, class_cell_op) else {
        return Ok(None);
    };
    // `fast2locals` falls back to the raw slot when it does not hold a cell.
    // That shape is unreachable for an OPTIMIZED frame past its
    // `COPY_FREE_VARS` prologue, and modelling it would need a second arm with
    // its own guard.
    if !unsafe { pyre_object::is_cell(concrete_cell) } {
        return Ok(None);
    }
    let family = unsafe { pyre_object::w_cell_family(concrete_cell) };
    if family.is_null() || unsafe { (*family).ever_mutated.get() } {
        return Ok(None);
    }
    let concrete_cls = unsafe { pyre_object::w_cell_get(concrete_cell) };
    if concrete_cls.is_null() || !unsafe { pyre_object::is_type(concrete_cls) } {
        return Ok(None);
    }
    // `descriptor.py:28-30` -- `None` builds the UNBOUND proxy, whose `w_self`
    // is null and whose attribute reads take a different arm entirely.
    if unsafe { pyre_object::is_none(concrete_self) } {
        return Ok(None);
    }
    let Some(objtype) =
        pyre_interpreter::builtins::super_check_python_free(concrete_cls, concrete_self)
    else {
        return Ok(None);
    };
    let class_mode =
        unsafe { pyre_object::is_type(concrete_self) } && std::ptr::eq(objtype, concrete_self);
    // Instance mode reads the receiver class back out of `w_class`; class mode
    // is `_super_check`'s first arm and pins the class object itself below.
    if !class_mode && !std::ptr::eq(objtype, unsafe { (*concrete_self).w_class }) {
        return Ok(None);
    }

    // `_get_self_location`'s cellvar arm is the ordinary red-cell LOAD_DEREF
    // shape.  All of that helper's decline conditions were proved above (and
    // the inline-resume condition at entry), so once it emits no later
    // optional branch can abandon a partially-written fold.
    let self_op = if let Some(cell_ref) = self_cell_ref {
        let Some(value) =
            residual_call::try_walker_read_deref_cell(ctx, op.pc, raw_self_op, cell_ref)?
        else {
            return Ok(None);
        };
        value
    } else {
        raw_self_op
    };

    // Which callable `super` names is baked into the emitted body.
    let callable_op = r_args[0];
    if !callable_op.is_constant() {
        let expected = ctx.trace_ctx.const_ref(concrete_callable as i64);
        walker_emit_fold_guard_with_snapshot(
            ctx,
            op.pc,
            OpCode::GuardValue,
            &[callable_op, expected],
        )?;
        ctx.trace_ctx
            .heap_cache_mut()
            .replace_box(callable_op, expected);
    }
    let cell_type = &pyre_object::nestedscope::CELL_TYPE as *const _ as i64;
    if !ctx.trace_ctx.heap_cache().is_class_known(class_cell_op) {
        let type_const = ctx.trace_ctx.const_int(cell_type);
        walker_emit_fold_guard_with_snapshot(
            ctx,
            op.pc,
            OpCode::GuardClass,
            &[class_cell_op, type_const],
        )?;
        ctx.trace_ctx
            .heap_cache_mut()
            .class_now_known(class_cell_op, cell_type);
    }
    let owner = ctx.trace_ctx.const_ref(family as i64);
    crate::state::record_quasiimmut_field(
        ctx.trace_ctx,
        owner,
        crate::descr::cell_family_ever_mutated_descr(),
    );
    walker_flush_guard_not_invalidated(ctx, op.pc)?;
    let cls_op = crate::state::opimpl_getfield_gc_r(
        ctx.trace_ctx,
        class_cell_op,
        crate::descr::cell_contents_descr(),
    );
    if !matches!(
        ctx.trace_ctx.box_value(cls_op),
        Some(majit_ir::Value::Ref(r)) if r != majit_ir::GcRef::NO_CONCRETE
    ) {
        ctx.trace_ctx.set_opref_concrete(
            cls_op,
            majit_ir::Value::Ref(majit_ir::GcRef(concrete_cls as usize)),
        );
    }
    let proxy_op = walker_emit_super_proxy(
        ctx,
        op.pc,
        cls_op,
        self_op,
        concrete_cls,
        concrete_self,
        objtype,
        class_mode,
    )?;
    write_residual_call_result_to_dst(ctx, op.pc, dst, 'r', proxy_op)?;
    Ok(Some(()))
}

/// Zero-argument `super()` reached as a call, i.e. the `LOAD_GLOBAL super` +
/// `CALL` spelling a name binding produces rather than `LOAD_SUPER_ATTR`.
///
/// Both spellings force the virtualizable — `codewriter.rs` binds
/// `load_super_attr_fn` with `CallFlavor::MayForce`, because a descriptor
/// `__get__` may run Python — so this is not about removing a force.  What
/// differs is where the force happens.  `LOAD_SUPER_ATTR` forces inside a
/// may-force residual that carries the red frame as an operand, which the
/// walker models.  The call spelling reaches `builtin_super`'s zero-argument
/// tail, whose `ExecutionContext::gettopframe()` runs `force_frame` INSIDE an
/// opaque `bh_call_fn`; that clears `TOKEN_TRACING_RESCALL` and
/// `tracing_after_residual_call` reads it back as
/// `VableEscapedDuringResidualCall`.  The frame `gettopframe` answers with is
/// the one being traced, so that residual always escapes and the loop always
/// aborts.
///
/// So the emission is a re-route: name
/// [`crate::helpers::jit_bare_super_from_frame`] — the same `descriptor.py
/// _super_from_frame` half `bh_load_super_attr_fn` calls — as a may-force
/// residual taking the walk's own frame box.  The force still happens; it moves
/// to the channel the walker can see, which is the difference between an
/// escape and an ordinary forced residual.
///
/// Only the receivers `super_check` settles without running Python are folded.
/// [`pyre_interpreter::builtins::builtin_super_from_frame_python_free`] answers
/// `None` for the rest, and they keep the generic residual.
///
/// The walk executes its residuals concretely, and this one would otherwise run
/// `super_check`'s `__class__` lookup — arbitrary code, which is free to force
/// the very virtualizable being recorded against and whose side effects a
/// decline would then repeat under the generic residual.  Restricting the fold
/// to the settled half makes the recording-time call a read: it cannot force,
/// so it owes no `vrefs_before/after_residual_call` bracket around itself, and
/// it cannot raise, so there is no exception to carry.  The runtime leaf still
/// runs the whole entry point, which is what the trailing `GuardNotForced` and
/// `GuardNoException` are for.
pub(crate) fn try_walker_specialize_bare_super_call<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    code: &[u8],
    op: &DecodedOp,
    r_args: &[OpRef],
    dst: usize,
) -> Result<Option<()>, DispatchError> {
    // `super()` with no user arguments arrives as `[callable, null_or_self]`.
    if r_args.len() != 2 {
        return Ok(None);
    }
    let arg_concretes = read_ref_var_list_concrete(code, op, 1, ctx);
    let (ConcreteValue::Ref(concrete_callable), ConcreteValue::Ref(null_or_self)) =
        (arg_concretes[0], arg_concretes[1])
    else {
        return Ok(None);
    };
    if concrete_callable.is_null()
        || !null_or_self.is_null()
        || !pyre_interpreter::builtins::is_builtin_super_type(concrete_callable)
    {
        return Ok(None);
    }
    let Some((frame_box, frame_ptr)) = walker_executing_frame_box(ctx) else {
        return Ok(None);
    };
    // Ahead of every emission, so a decline leaves the trace untouched.
    let Some(proxy) = pyre_interpreter::builtins::builtin_super_from_frame_python_free(
        frame_ptr as *mut pyre_interpreter::PyFrame,
    ) else {
        return Ok(None);
    };
    // Pin the callable the way the constructor folds do, so rebinding the
    // global `super` side-exits instead of keeping this route.
    let callable_op = r_args[0];
    if !callable_op.is_constant() {
        let expected = ctx.trace_ctx.const_ref(concrete_callable as i64);
        ctx.trace_ctx
            .record_guard(OpCode::GuardValue, &[callable_op, expected], 0);
        walker_capture_snapshot_for_last_guard(ctx, op.pc)?;
        ctx.trace_ctx
            .heap_cache_mut()
            .replace_box(callable_op, expected);
    }
    residual_call::maybe_walker_vable_and_vrefs_before_residual_call(ctx, op.pc);
    // `MOST_GENERAL`, not a fresh `EffectInfo`: the leaf is the whole entry
    // point, whose `super_check` arm can run a `__class__` property, so no
    // read/write descr set describes it.  A constructed `EffectInfo` inherits
    // EMPTY sets, which claims the opposite and lets the optimizer keep cached
    // fields across the call.  `RandomEffects` outranks
    // `ForcesVirtualOrVirtualizable`, so this keeps the may-force reading the
    // `GuardNotForced` below depends on.
    let result = ctx.trace_ctx.call_typed_with_effect(
        OpCode::CallMayForceR,
        crate::helpers::jit_bare_super_from_frame as *const (),
        &[frame_box],
        &[majit_ir::Type::Ref],
        majit_ir::Type::Ref,
        majit_ir::EffectInfo::MOST_GENERAL,
    );
    ctx.trace_ctx.set_opref_concrete(
        result,
        majit_ir::Value::Ref(majit_ir::GcRef(proxy as usize)),
    );
    // The dst is written before both guards, the ordering
    // `_opimpl_residual_call*` keeps so the dst slot's OpRef rides their
    // snapshots.
    write_residual_call_result_to_dst(ctx, op.pc, dst, 'r', result)?;
    ctx.trace_ctx.record_guard(OpCode::GuardNotForced, &[], 0);
    walker_capture_snapshot_for_last_guard(ctx, op.pc)?;
    walker_emit_guard_with_snapshot(ctx, op.pc, OpCode::GuardNoException, &[])?;
    Ok(Some(()))
}

/// `mapdict.py LOAD_ATTR_caching` full-body-walker fast path for a
/// plain (non-method) instance attribute.  When the concrete receiver is a
/// monomorphic instance whose attribute resolves to a boxed plain storage slot
/// or an unboxed integer/float slot, emit the guarded read PyPy compiles
/// LOAD_ATTR to under the JIT —
///   * `guard_class(obj, concrete_layout)` — the receiver keeps the exact
///     layout vtable whose mapdict-carrier prefix was proved at trace time (so
///     the `map`/`storage` reads below are valid; `mapdict.py` `if map is not
///     None:` also filters non-carriers at trace time).
///   * `guard_value(getfield_gc_i(w_type, version_tag), C_version_tag)` — pins
///     the class lookup result so a later descriptor or `__getattribute__`
///     mutation deopts on trace re-entry.
///   * `guard_value(getfield_gc_i(obj, map), C_map)` — `jit.promote(self.map)`
///     (`mapdict.py`); pins the exact instance shape so `find_map_attr`
///     const-folds `storageindex` to a green constant.
///   * boxed: `getfield_gc_r(obj, storage)` +
///     `getarrayitem_gc_r(block, C_index)` for
///     `mapdict.py _mapdict_read_storage`;
///   * unboxed int/float: a non-forcing typed read plus `wrapint`/`wrapfloat`,
///     matching `_prim_direct_read` (mapdict.py).
/// — instead of the opaque `getattr_fn` `CALL_MAY_FORCE` MRO-walk residual.
///
/// Returns `Some(())` after writing the dst; `None` (fall through to the
/// residual) for every shape [`load_attr_fast_path`] declines: non-instance
/// receiver, missing map, custom `__getattribute__`, uncacheable `version_tag`,
/// a data-descriptor / `INVALID` classification, or an attribute not on this
/// instance's map.  The map `guard_value` proves the attribute is present on
/// this shape, so a successful fold provably cannot raise `AttributeError` —
/// dropping the residual's exception guard is sound even in a handler-bearing
/// body (same reasoning as the LoadGlobal fold).
///
/// `name` is the already-resolved attribute name, so the fold serves both the
/// `LOAD_ATTR` residual — whose caller reads it out of the jitcode's own
/// `co_names` — and the `getattr(obj, "name")` builtin, whose name arrives as a
/// constant string operand.  Both spell one `space.getattr`, so they must reach
/// the same read.
pub(crate) fn try_walker_specialize_load_attr<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    obj: OpRef,
    name: &str,
    dst: usize,
    dst_bank: char,
) -> Result<Option<()>, DispatchError> {
    use pyre_interpreter::pyframe::PyFrame;

    if !ctx.is_authoritative_executor || dst_bank != 'r' {
        return Ok(None);
    }
    // The receiver must be a concrete instance for the map/storageindex
    // resolution below; a non-concrete or non-instance receiver declines.
    let Some(concrete_obj) = walker_concrete_ref_object(ctx, obj) else {
        return Ok(None);
    };
    // CPython 3.14 exposes an optimized frame's locals as a fresh
    // `FrameLocalsProxy`.  Constructing that proxy does not read fast locals;
    // its operations synchronize through the frame when they are actually
    // used.  Keep the exact per-MIFrame red receiver as the proxy owner instead
    // of residualizing the getter, whose explicit read barrier would force a
    // live virtualizable while an inline MIFrame is still active.
    //
    // There are two identities the walker can prove here: the current inline
    // callee's shadow frame, whose locals region the walk flushes itself at the
    // escape, or the standard portal frame, gated by BOTH its red box and its
    // concrete pointer.  For the portal the dropped force was also what wrote
    // the locals region out of the virtualizable image, so the fold has to
    // write that region itself.
    let inline_frame = current_inline_concrete_frame();
    let is_inline_frame = inline_frame != 0
        && concrete_obj as usize == inline_frame
        && ctx
            .callee_shadow
            .as_ref()
            .is_some_and(|shadow| shadow.concrete_frame == inline_frame && shadow.frame_box == obj);
    let is_standard_frame = ctx.trace_ctx.standard_virtualizable_box() == Some(obj)
        && ctx.trace_ctx.standard_virtualizable_ptr() == Some(concrete_obj as usize);

    if name == "f_locals"
        && (is_inline_frame || is_standard_frame)
        && unsafe { (*concrete_obj).ob_type } == &pyre_interpreter::pyframe::FRAME_TYPE
        && unsafe {
            (*(concrete_obj as *const pyre_interpreter::PyFrame))
                .code()
                .flags
                .contains(pyre_interpreter::CodeFlags::OPTIMIZED)
        }
    {
        let w_type =
            pyre_interpreter::typedef::gettypeobject(&pyre_interpreter::pyframe::FRAME_TYPE);
        let version_tag = unsafe { pyre_object::typeobject::w_type_get_version_tag(w_type) };
        if version_tag == 0 || unsafe { (*concrete_obj).w_class } != w_type {
            return Ok(None);
        }
        if is_standard_frame
            && !walker_write_back_standard_frame_locals(ctx, obj, concrete_obj as usize)
        {
            return Ok(None);
        }
        let concrete_proxy = pyre_interpreter::pyframe::frame_locals_proxy::new(concrete_obj);
        walker_guard_exception_attr_slot(ctx, op_pc, obj, concrete_obj, w_type, version_tag)?;
        let proxy = ctx.trace_ctx.call_ref_typed_with_effect(
            jit_inline_frame_locals_proxy_new as *const (),
            &[obj],
            &[majit_ir::Type::Ref],
            majit_ir::EffectInfo::new(
                majit_ir::ExtraEffect::CannotRaise,
                majit_ir::OopSpecIndex::None,
            ),
        );
        ctx.trace_ctx.set_opref_concrete(
            proxy,
            majit_ir::Value::Ref(majit_ir::GcRef(concrete_proxy as usize)),
        );
        write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, proxy)?;
        return Ok(Some(()));
    }
    if name == "f_lasti"
        && unsafe { (*concrete_obj).ob_type } == &pyre_interpreter::pyframe::FRAME_TYPE
        && spec_gate(SpecFold::FrameLasti, || {
            try_walker_specialize_frame_lasti(ctx, op_pc, obj, concrete_obj, dst, dst_bank)
        })?
        .is_some()
    {
        return Ok(Some(()));
    }
    if name == "f_lineno"
        && unsafe { (*concrete_obj).ob_type } == &pyre_interpreter::pyframe::FRAME_TYPE
        && spec_gate(SpecFold::FrameLineno, || {
            try_walker_specialize_frame_lineno(ctx, op_pc, obj, concrete_obj, dst, dst_bank)
        })?
        .is_some()
    {
        return Ok(Some(()));
    }
    // `mapdict.py` resolution, returning the fold ingredients (the
    // read is left to the caller so it can be folded to a guarded inline read).
    if let Some((w_type, version_tag, map, storageindex)) =
        unsafe { pyre_interpreter::objspace::std::mapdict::load_attr_fast_path(concrete_obj, name) }
    {
        walker_guard_mapdict_instance_shape(
            ctx,
            op_pc,
            obj,
            concrete_obj,
            w_type,
            version_tag,
            map,
        )?;

        // getfield_gc_r(obj, storage) + getarrayitem_gc_r(block, C_storageindex):
        // the inline value read (`mapdict.py`).  `storageindex` is a green
        // constant (the map guard pinned it); `trace_mapdict_storage_getitem`
        // stamps the dst's concrete shadow from the live block slot.
        let block = crate::state::opimpl_getfield_gc_r(ctx.trace_ctx, obj, unsafe {
            crate::descr::mapdict_storage_descr(concrete_obj)
        });
        let idx_const = ctx.trace_ctx.const_int(storageindex as i64);
        let value = crate::state::trace_mapdict_storage_getitem(ctx.trace_ctx, block, idx_const);
        write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, value)?;
        return Ok(Some(()));
    }

    // Class attribute that object.__getattribute__ returns unchanged
    // (`Object.descr__getattribute__` / `class_attr_fast_path`).  The
    // instance-slot fold above needs a mapdict storage index; a name that
    // lives only on the type has none and would otherwise residualize
    // `space.getattr`.
    if let Some((w_type, version_tag, map, w_value)) = unsafe {
        pyre_interpreter::objspace::std::mapdict::class_attr_fast_path(concrete_obj, name)
    } {
        // The movability test is load-bearing, not conservative.  A recorded
        // `ConstPtr` is forwarded — `remove_constptrs_in` rewrites it to a
        // `LoadFromGcTable` at emit and `gcreftracer` keeps the table slot
        // current at run — but `write_residual_call_result_to_dst` first parks
        // the `OpRef` in the walker's own `registers_r`, and no registered
        // mutator extra area walks that bank.  A collection between the mint
        // and the use would leave the inline `GcRef` stale.  The ungated folds
        // are not counter-examples: `try_walker_specialize_load_type_attr`
        // bakes a type object and `emit_module_dict_cell_fold` a `malloc_typed`
        // cell, neither of which moves, whereas a class attribute is an
        // arbitrary object and can sit in the nursery.
        if !majit_gc::can_move(majit_ir::GcRef(w_value as usize)) {
            walker_guard_mapdict_instance_shape(
                ctx,
                op_pc,
                obj,
                concrete_obj,
                w_type,
                version_tag,
                map,
            )?;
            let value = ctx.trace_ctx.const_ref(w_value as i64);
            write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, value)?;
            return Ok(Some(()));
        }
    }

    if let Some(walk_field) = traceback_walk_field(concrete_obj, name) {
        if let Some(()) = walker_specialize_traceback_walk_field(
            ctx,
            op_pc,
            obj,
            concrete_obj,
            walk_field,
            dst,
            dst_bank,
        )? {
            return Ok(Some(()));
        }
    }

    // A user attribute on an exception instance. `mapdict.py:1483-1490
    // LOAD_ATTR_caching` declines this receiver upstream too — an exception is
    // not a `MapdictStorageMixin` and `_get_mapdict_map` answers None
    // (`baseobjspace.py:204-205`) — and it reaches its speed by inlining
    // `getdictvalue -> MapDictStrategy.getitem_str -> AbstractAttribute.read`
    // (`mapdict.py:55-66`, `:442-444`) instead. The attribute lives in the
    // `newdict(instance=True)` dictionary, two hops out: `w_dict` ->
    // `W_DictObject.dstorage` -> the fake carrier that holds the map.
    //
    // `w_exception_get_kind` and `w_exception_peek_dict` both cast straight to
    // `W_BaseException`, so the `is_exception` test is load-bearing: this
    // function runs for every `LOAD_ATTR` receiver that reached it.
    let exc_dict = unsafe {
        pyre_object::is_exception(concrete_obj)
            .then(|| pyre_object::interp_exceptions::w_exception_peek_dict(concrete_obj))
            .filter(|dict| !dict.is_null())
    };
    if let Some(dict) = exc_dict
        && let Some((w_type, _version_tag, carrier, map, storageindex, unboxed)) = unsafe {
            pyre_interpreter::objspace::std::mapdict::instance_dict_attr_fast_path(
                concrete_obj,
                dict,
                name,
            )
        }
        // An unboxed float slot keeps the `f64` bit pattern in the same
        // longlong block as an int, so folding it needs a bits-to-float
        // reinterpret the trace has no operation for; leave it on the residual.
        && !matches!(
            unboxed,
            Some((pyre_interpreter::objspace::std::mapdict::UnboxType::Float, _))
        )
    {
        let kind = unsafe { pyre_object::w_exception_get_kind(concrete_obj) };
        let phys_type = unsafe { (*concrete_obj).ob_type as i64 };
        if !ctx.trace_ctx.heap_cache().is_class_known(obj) {
            let type_const = ctx.trace_ctx.const_int(phys_type);
            walker_emit_fold_guard_with_snapshot(
                ctx,
                op_pc,
                OpCode::GuardClass,
                &[obj, type_const],
            )?;
            ctx.trace_ctx
                .heap_cache_mut()
                .class_now_known(obj, phys_type);
        }
        let w_class = walker_record_getfield_gc_r_uncached(ctx, obj, crate::descr::w_class_descr());
        let w_type_const = ctx.trace_ctx.const_ref(w_type as i64);
        walker_emit_fold_guard_with_snapshot(
            ctx,
            op_pc,
            OpCode::GuardValue,
            &[w_class, w_type_const],
        )?;
        ctx.trace_ctx
            .heap_cache_mut()
            .replace_box(w_class, w_type_const);
        walker_pin_type_version_tag(ctx, op_pc, w_type_const)?;

        let dict_op = walker_record_getfield_gc_r_uncached(
            ctx,
            obj,
            crate::descr::w_exception_dict_descr(kind),
        );
        walker_emit_fold_guard_with_snapshot(ctx, op_pc, OpCode::GuardNonnull, &[dict_op])?;
        ctx.trace_ctx.set_opref_concrete(
            dict_op,
            majit_ir::Value::Ref(majit_ir::GcRef(dict as usize)),
        );

        // `instance_dict_attr_fast_path` declines a dictionary that is not
        // `MapDictStrategy`-backed, and the carrier read below is out of bounds
        // on a devolved one, so pin the strategy before dereferencing
        // `dstorage`.
        let strategy = crate::state::opimpl_getfield_gc_i(
            ctx.trace_ctx,
            dict_op,
            crate::descr::dict_strategy_word_descr(),
        );
        let strategy_const = ctx.trace_ctx.const_int(
            &pyre_interpreter::objspace::std::mapdict::MAP_DICT_STRATEGY_REF as *const _ as i64,
        );
        walker_emit_fold_guard_with_snapshot(
            ctx,
            op_pc,
            OpCode::GuardValue,
            &[strategy, strategy_const],
        )?;
        ctx.trace_ctx
            .heap_cache_mut()
            .replace_box(strategy, strategy_const);

        let carrier_op =
            walker_record_getfield_gc_r_uncached(ctx, dict_op, crate::descr::dict_dstorage_descr());
        ctx.trace_ctx.set_opref_concrete(
            carrier_op,
            majit_ir::Value::Ref(majit_ir::GcRef(carrier as usize)),
        );
        let map_op = walker_record_getfield_gc_i_uncached(ctx, carrier_op, unsafe {
            crate::descr::mapdict_map_descr(carrier)
        });
        let map_const = ctx.trace_ctx.const_int(map as i64);
        walker_emit_fold_guard_with_snapshot(ctx, op_pc, OpCode::GuardValue, &[map_op, map_const])?;
        ctx.trace_ctx
            .heap_cache_mut()
            .replace_box(map_op, map_const);

        let block = crate::state::opimpl_getfield_gc_r(ctx.trace_ctx, carrier_op, unsafe {
            crate::descr::mapdict_storage_descr(carrier)
        });
        let index = ctx.trace_ctx.const_int(storageindex as i64);
        let slot = crate::state::trace_mapdict_storage_getitem(ctx.trace_ctx, block, index);
        let value = match unboxed {
            None => slot,
            // `_prim_direct_read` (mapdict.py): the storage slot holds
            // the shared longlong block, and the value is `items[listindex]`.
            // Keeping the boxing in the trace lets an immediate integer
            // consumer virtualize it away.
            Some((_, listindex)) => {
                let listindex_const = ctx.trace_ctx.const_int(listindex as i64);
                let live = unsafe {
                    pyre_interpreter::objspace::std::mapdict::read_unboxed_storage_raw(
                        carrier,
                        storageindex,
                        listindex,
                    )
                };
                let raw = crate::state::trace_int_block_getitem_value(
                    ctx.trace_ctx,
                    slot,
                    listindex_const,
                );
                ctx.trace_ctx
                    .set_opref_concrete(raw, majit_ir::Value::Int(live));
                let boxed = walker_box_int(ctx, op_pc, raw, live)?;
                let live_ptr = pyre_object::w_int_new(live) as i64;
                ctx.trace_ctx
                    .set_opref_concrete(boxed, box_int_concrete(live, live_ptr));
                boxed
            }
        };
        write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, value)?;
        return Ok(Some(()));
    }

    if let Some((slot, kind, w_type, version_tag, stored)) = unsafe {
        pyre_interpreter::baseobjspace::exception_attr_slot_fold(concrete_obj, name, false)
    } {
        if slot == pyre_interpreter::baseobjspace::ExceptionAttrSlot::Args
            && unsafe { (*(stored as *const pyre_object::listobject::W_ListObject)).strategy }
                != pyre_object::listobject::ListStrategy::Object
        {
            return Ok(None);
        }
        // `descr_getargs` copies `args_w` through `newtuple`, which picks the
        // tuple representation from the arity and the element types.  Ask
        // `newtuple` itself which shape this read produces, rather than second-
        // guessing its dispatch, and settle it here — before any guard is
        // recorded, so a decline stays clean (a bail-out after the class pin
        // would leave the caller reading this attribute as already guarded).
        //
        // Only the shape crosses into the emit below; the probe tuple is
        // dropped rather than held, since nothing roots it across the guards.
        let args_specialised_oo = if slot == pyre_interpreter::baseobjspace::ExceptionAttrSlot::Args
        {
            let ob_type = unsafe { (*args_tuple_shape_probe(stored)).ob_type };
            if std::ptr::eq(
                ob_type,
                &pyre_object::specialisedtupleobject::SPECIALISED_TUPLE_OO_TYPE,
            ) {
                true
            } else if std::ptr::eq(ob_type, &pyre_object::TUPLE_TYPE) {
                false
            } else {
                // The unboxed arity-2 specialisations (`Cls_ii` / `Cls_ff`)
                // hold machine values in their inline fields, which this copy
                // has no unboxed operand for.  Only an Object-strategy `args_w`
                // holding exactly two plain ints or two plain floats gets here.
                return Ok(None);
            }
        } else {
            false
        };
        let traceback_frame = if slot
            == pyre_interpreter::baseobjspace::ExceptionAttrSlot::Traceback
        {
            let frame = unsafe { pyre_interpreter::pytraceback::w_pytraceback_get_frame(stored) };
            // `mark_traceback_escaped` leaves a torn-down traceback alone.
            // Decline before recording the receiver guards when the
            // authoritative read sees that shape; compiled replays guard
            // the frame load below and side-exit to the same residual path.
            if frame.is_null() {
                return Ok(None);
            }
            Some(frame)
        } else {
            None
        };
        walker_guard_exception_attr_slot(ctx, op_pc, obj, concrete_obj, w_type, version_tag)?;
        let raw_value = crate::state::opimpl_getfield_gc_r(
            ctx.trace_ctx,
            obj,
            crate::descr::w_exception_attr_slot_descr(kind, slot),
        );
        walker_emit_fold_guard_with_snapshot(ctx, op_pc, OpCode::GuardNonnull, &[raw_value])?;
        ctx.trace_ctx.set_opref_concrete(
            raw_value,
            majit_ir::Value::Ref(majit_ir::GcRef(stored as usize)),
        );
        if slot == pyre_interpreter::baseobjspace::ExceptionAttrSlot::Traceback {
            // The fold replaces `descr_gettraceback`, whose read marks the
            // traceback's frame escaped so `ExecutionContext::leave` forces
            // its vref.  `descr_settraceback` admits only None or PyTraceback,
            // so the non-null slot is already type-safe without a class guard.
            // Read the node's frame, require the non-null case handled by the
            // traced path, and mirror `PyFrame.mark_as_escaped` directly.
            let frame_ref = crate::state::opimpl_getfield_gc_r(
                ctx.trace_ctx,
                raw_value,
                crate::descr::pytraceback_frame_descr(),
            );
            walker_emit_fold_guard_with_snapshot(ctx, op_pc, OpCode::GuardNonnull, &[frame_ref])?;
            let concrete_frame = traceback_frame.expect("traceback fold has no frame");
            ctx.trace_ctx.set_opref_concrete(
                frame_ref,
                majit_ir::Value::Ref(majit_ir::GcRef(concrete_frame as usize)),
            );
            let flags_descr = crate::descr::pyframe_flags_descr();
            let live_flags =
                crate::state::opimpl_getfield_gc_i(ctx.trace_ctx, frame_ref, flags_descr.clone());
            let escaped_bit = ctx.trace_ctx.const_int(i64::from(PyFrame::FLAG_ESCAPED));
            let new_flags = ctx
                .trace_ctx
                .record_op(OpCode::IntOr, &[live_flags, escaped_bit]);
            ctx.trace_ctx.record_op_with_descr(
                OpCode::SetfieldGc,
                &[frame_ref, new_flags],
                flags_descr.clone(),
            );
            ctx.trace_ctx
                .heapcache_setfield_cached(frame_ref, flags_descr.index(), new_flags);

            // The walk is the authoritative execution path, so mark its
            // concrete frame now as well as on every compiled re-execution.
            unsafe { pyre_interpreter::pytraceback::mark_traceback_escaped(stored) };
        }
        let value = if slot == pyre_interpreter::baseobjspace::ExceptionAttrSlot::Args {
            let list = unsafe { &*(stored as *const pyre_object::listobject::W_ListObject) };
            if list.strategy != pyre_object::listobject::ListStrategy::Object {
                return Ok(None);
            }
            let list_type = &pyre_object::LIST_TYPE as *const pyre_object::PyType as i64;
            if !ctx.trace_ctx.heap_cache().is_class_known(raw_value) {
                let type_const = ctx.trace_ctx.const_int(list_type);
                walker_emit_fold_guard_with_snapshot(
                    ctx,
                    op_pc,
                    OpCode::GuardClass,
                    &[raw_value, type_const],
                )?;
                ctx.trace_ctx
                    .heap_cache_mut()
                    .class_now_known(raw_value, list_type);
            }
            walker_guard_exact_w_class(
                ctx,
                op_pc,
                raw_value,
                pyre_object::get_instantiate(&pyre_object::LIST_TYPE),
            )?;
            let strategy = crate::state::opimpl_getfield_gc_i(
                ctx.trace_ctx,
                raw_value,
                crate::descr::list_strategy_descr(),
            );
            let object_strategy = ctx
                .trace_ctx
                .const_int(pyre_object::listobject::ListStrategy::Object as i64);
            walker_emit_fold_guard_with_snapshot(
                ctx,
                op_pc,
                OpCode::GuardValue,
                &[strategy, object_strategy],
            )?;
            let len = unsafe { pyre_object::w_list_len(stored) };
            let length = crate::state::opimpl_getfield_gc_i(
                ctx.trace_ctx,
                raw_value,
                crate::descr::list_length_descr(),
            );
            let len_const = ctx.trace_ctx.const_int(len as i64);
            walker_emit_fold_guard_with_snapshot(
                ctx,
                op_pc,
                OpCode::GuardValue,
                &[length, len_const],
            )?;
            let block = crate::state::opimpl_getfield_gc_r(
                ctx.trace_ctx,
                raw_value,
                crate::descr::list_items_descr(),
            );
            let mut items = Vec::with_capacity(len);
            let mut concrete_items = Vec::with_capacity(len);
            for index in 0..len {
                let index_op = ctx.trace_ctx.const_int(index as i64);
                items.push(crate::state::trace_items_block_getitem_value(
                    ctx.trace_ctx,
                    block,
                    index_op,
                ));
                concrete_items.push(
                    unsafe { pyre_object::w_list_getitem(stored, index as i64) }
                        .unwrap_or(pyre_object::PY_NULL),
                );
            }
            // Emit the representation `newtuple` picks, settled above.  Emitting
            // the array-backed shape for an arity the runtime specialises leaves
            // the trace disagreeing with its own record-time concrete, and the
            // `except <tuple>:` match fold — which dispatches on that concrete's
            // layout — then guards for a shape this trace never builds, so the
            // loop aborts instead of compiling.
            let tuple = if args_specialised_oo {
                crate::helpers::emit_specialised_tuple_oo_inline(ctx.trace_ctx, items[0], items[1])
            } else {
                crate::helpers::emit_object_tuple_inline(ctx.trace_ctx, &items)
            };
            let concrete_tuple = pyre_object::w_tuple_new(concrete_items);
            ctx.trace_ctx.set_opref_concrete(
                tuple,
                majit_ir::Value::Ref(majit_ir::GcRef(concrete_tuple as usize)),
            );
            tuple
        } else {
            raw_value
        };
        write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, value)?;
        return Ok(Some(()));
    }

    // Module attribute (`math.sqrt`): the receiver is an exact module and the
    // name is present in its dict.  Fold the module-dict read to a
    // `QUASIIMMUT_FIELD(dict, slot)` version guard + elidable cell lookup —
    // celldict.py `_getdictvalue_no_unwrapping_pure` (`@jit.elidable_promote`) —
    // so a hot `math.sqrt(x)` loop drops its per-iteration LOAD_ATTR may-force
    // residual and the `math.sqrt` callable becomes a trace constant.  A rebind
    // of the attribute bumps the module dict `version` and fails the guard.
    // All resolution below is read-only; a missing / movable / non-canonical
    // shape falls through to the residual with no IR emitted.  An exact
    // `module` `w_class` excludes a module subclass with a custom
    // `__getattribute__`; a module-level PEP 562 `__getattr__` is irrelevant
    // because the name is present (the dict lookup wins before `__getattr__`).
    // A data descriptor on the module type (e.g. `__dict__`) outranks a
    // same-named dict entry in generic getattr, so decline when the name
    // resolves to one — the descriptor result, not the dict cell, is what a
    // read returns.
    if unsafe { pyre_object::is_module(concrete_obj) }
        && std::ptr::eq(
            unsafe { (*concrete_obj).w_class },
            pyre_object::pyobject::get_instantiate(&pyre_object::pyobject::MODULE_TYPE),
        )
        && !unsafe {
            pyre_interpreter::baseobjspace::type_lookup_is_data_descr((*concrete_obj).w_class, name)
        }
    {
        let w_dict = unsafe { pyre_object::w_module_get_w_dict(concrete_obj) };
        if !w_dict.is_null() && !majit_gc::can_move(majit_ir::GcRef(w_dict as usize)) {
            if let Some(slot) = crate::state::module_dict_cell_slot_direct(w_dict, name) {
                if let Some(stored) = crate::state::module_dict_cell_value_direct(w_dict, slot) {
                    if !stored.is_null() && !majit_gc::can_move(majit_ir::GcRef(stored as usize)) {
                        // Pin the receiver to THIS module so the baked dict
                        // address is correct: a constant receiver is already
                        // pinned; a non-constant one gets a `guard_value`.
                        if !obj.is_constant() {
                            let expected = ctx.trace_ctx.const_ref(concrete_obj as i64);
                            ctx.trace_ctx
                                .record_guard(OpCode::GuardValue, &[obj, expected], 0);
                            walker_capture_snapshot_for_last_guard(ctx, op_pc)?;
                            ctx.trace_ctx.heap_cache_mut().replace_box(obj, expected);
                        }
                        // `guard_frame_globals=false`: the receiver pin above
                        // (not a frame-globals-identity guard) proves the dict.
                        if !emit_namespace_cell_fold(
                            ctx, op_pc, dst, dst_bank, w_dict, slot, stored, false,
                        )? {
                            // Nothing was written to `dst`, so the residual
                            // still owes the load.
                            return Ok(None);
                        }
                        return Ok(Some(()));
                    }
                }
            }
        }
    }

    // `module/mod.rs` gates `_cffi_backend` on the same two conditions.
    #[cfg(all(not(feature = "sandbox"), not(target_arch = "wasm32")))]
    if let Some(lib) =
        pyre_interpreter::module::_cffi_backend::lib_obj::W_LibObject::from_obj(concrete_obj)
        && spec_gate(SpecFold::LoadAttrCffiLib, || {
            let w_dict = lib.dict_w;
            if w_dict.is_null() || majit_gc::can_move(majit_ir::GcRef(w_dict as usize)) {
                return Ok(None);
            }
            let Some(slot) = crate::state::module_dict_cell_slot_direct(w_dict, name) else {
                return Ok(None);
            };
            let Some(stored) = crate::state::module_dict_cell_value_direct(w_dict, slot) else {
                return Ok(None);
            };
            if stored.is_null()
                || majit_gc::can_move(majit_ir::GcRef(stored as usize))
                // `W_LibObject.lib_getattribute` turns this support object into
                // a live C-memory read; returning the dict cell would expose
                // the support object itself, and `lib_setattr` does not mutate
                // the dict version when it writes through the support object.
                || pyre_interpreter::module::_cffi_backend::cglob::W_GlobSupport::from_obj(stored)
                    .is_some()
            {
                return Ok(None);
            }
            // Pin the receiver to THIS Lib so its baked dict address remains
            // valid.  A constant receiver is already pinned.
            if !obj.is_constant() {
                let expected = ctx.trace_ctx.const_ref(concrete_obj as i64);
                ctx.trace_ctx
                    .record_guard(OpCode::GuardValue, &[obj, expected], 0);
                walker_capture_snapshot_for_last_guard(ctx, op_pc)?;
                ctx.trace_ctx.heap_cache_mut().replace_box(obj, expected);
            }
            if !emit_namespace_cell_fold(ctx, op_pc, dst, dst_bank, w_dict, slot, stored, false)? {
                return Ok(None);
            }
            Ok(Some(()))
        })?
        .is_some()
    {
        return Ok(Some(()));
    }

    let Some((w_type, version_tag, map, storageindex, listindex, unbox_type)) = (unsafe {
        pyre_interpreter::objspace::std::mapdict::load_attr_unboxed_fast_path(concrete_obj, name)
    }) else {
        return Ok(None);
    };
    walker_guard_mapdict_instance_shape(ctx, op_pc, obj, concrete_obj, w_type, version_tag, map)?;
    let terminator = unsafe { (*map).terminator() };
    let term = unsafe { (*terminator).as_terminator() as *const _ };
    walker_pin_terminator_allow_unboxing(ctx, op_pc, term)?;

    // `_prim_direct_read` (mapdict.py): read the raw longlong from the
    // shared list through a non-forcing, non-elidable residual.  Both indices
    // are green constants pinned by the map guard; keeping boxing in the trace
    // lets an immediate integer consumer virtualize it away.
    let storageindex_const = ctx.trace_ctx.const_int(storageindex as i64);
    let listindex_const = ctx.trace_ctx.const_int(listindex as i64);
    let live = unsafe {
        pyre_interpreter::objspace::std::mapdict::read_unboxed_storage_raw(
            concrete_obj,
            storageindex,
            listindex,
        )
    };
    let boxed = match unbox_type {
        pyre_interpreter::objspace::std::mapdict::UnboxType::Int => {
            // A trace-allocated receiver reads inline so the heap cache folds
            // the chain and the instance stays virtual (a residual Ref arg
            // would force it); an escaped receiver keeps the residual.
            let raw = if ctx.trace_ctx.heap_cache().is_unescaped(obj) {
                let block = crate::state::opimpl_getfield_gc_r(ctx.trace_ctx, obj, unsafe {
                    crate::descr::mapdict_storage_descr(concrete_obj)
                });
                let slot = crate::state::trace_mapdict_storage_getitem(
                    ctx.trace_ctx,
                    block,
                    storageindex_const,
                );
                crate::state::trace_int_block_getitem_value(ctx.trace_ctx, slot, listindex_const)
            } else {
                crate::helpers::emit_trace_call_int_typed(
                    ctx.trace_ctx,
                    crate::helpers::jit_mapdict_unboxed_read_raw as *const (),
                    &[obj, storageindex_const, listindex_const],
                    &[
                        majit_ir::Type::Ref,
                        majit_ir::Type::Int,
                        majit_ir::Type::Int,
                    ],
                )
            };
            ctx.trace_ctx
                .set_opref_concrete(raw, majit_ir::Value::Int(live));
            let boxed = walker_box_int(ctx, op_pc, raw, live)?;
            // The `wrapint` op is a heap box, so its concrete must be a heap ptr too:
            // box the raw longlong through the same `w_int_new` the unboxed read uses
            // (mapdict.py `_box`); `box_int_concrete` re-homes a tagged small
            // int to a fresh heap `W_IntObject` so op(NewWithVtable) == concrete(heap).
            // Without this stamp the boxed result carries no concrete, so a downstream
            // eager void residual (e.g. the STORE_ATTR that writes `self.value`) cannot
            // resolve its value arg and the walk aborts `ResidualCallArgUnbound`.
            let live_ptr = pyre_object::w_int_new(live) as i64;
            ctx.trace_ctx
                .set_opref_concrete(boxed, box_int_concrete(live, live_ptr));
            boxed
        }
        pyre_interpreter::objspace::std::mapdict::UnboxType::Float => {
            let raw = crate::helpers::emit_trace_call_float_typed(
                ctx.trace_ctx,
                crate::helpers::jit_mapdict_unboxed_read_f as *const (),
                &[obj, storageindex_const, listindex_const],
                &[
                    majit_ir::Type::Ref,
                    majit_ir::Type::Int,
                    majit_ir::Type::Int,
                ],
            );
            let live_f = f64::from_bits(live as u64);
            ctx.trace_ctx
                .set_opref_concrete(raw, majit_ir::Value::Float(live_f));
            let boxed = crate::state::wrapfloat(ctx.trace_ctx, raw);
            ctx.trace_ctx.set_opref_concrete(
                boxed,
                majit_ir::Value::Ref(majit_ir::GcRef(pyre_object::w_float_new(live_f) as usize)),
            );
            boxed
        }
    };
    write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, boxed)?;
    Ok(Some(()))
}

/// The receiver-layout-specific half of the LOAD_METHOD fold: which field the
/// walker guards to keep "no instance attribute shadows this method" true for
/// the life of the trace.
///
/// `load_method_fast_path` proved the property once, at record time; this is
/// what re-proves it on every execution.  One variant per layout whose
/// non-allocating dictionary peek that predicate covers.
enum ShadowGuard {
    /// A `W_ObjectObject` receiver: pin the mapdict map, so adding
    /// `obj.<name>` grows the map chain and side-exits.
    InstanceMap(*const u8),
    /// A `W_BaseException` receiver: pin `w_dict` at null, so the lazy
    /// allocation `e.<name> = ...` performs side-exits.  Carries the kind
    /// because the field descrs are grouped per `ExcKind` vtable.
    ExceptionDictIsNull(pyre_object::interp_exceptions::ExcKind),
}

/// `callmethod.py LOAD_METHOD` method-cache fold for the
/// codewriter's method-form `LOAD_ATTR` residual.  The safety oracle is the
/// interpreter's `load_method_fast_path`: it declines custom
/// `__getattribute__`, uncacheable types, non-function descriptors, and
/// shadowing instance attributes.  On success the
/// walker emits the guards that keep that decision stable, then writes
/// `w_descr` as a green constant so the following `CALL` can use the existing
/// constant-callee inline path.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_walker_specialize_load_method_attr<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    obj: OpRef,
    w_code_ptr: usize,
    name_idx: usize,
    dst: usize,
    dst_bank: char,
) -> Result<Option<()>, DispatchError> {
    if !ctx.is_authoritative_executor || dst_bank != 'r' {
        return Ok(None);
    }
    let Some(concrete_obj) = walker_concrete_ref_object(ctx, obj) else {
        return Ok(None);
    };
    let Some(name) = walker_load_name_from_code(w_code_ptr, name_idx) else {
        return Ok(None);
    };
    let Some((w_type, version_tag, w_descr)) =
        (unsafe { pyre_interpreter::load_method_fast_path(concrete_obj, &name) })
    else {
        return Ok(None);
    };
    if unsafe { resolve_inlinable_callee(w_descr) }.is_none() {
        return Ok(None);
    }
    // `space.type` reaches an exception's class through the kind registry when
    // the generic stub is still installed, and the `w_class` guard below can
    // only pin a class the slot actually holds.
    if !std::ptr::eq(unsafe { (*concrete_obj).w_class }, w_type) {
        return Ok(None);
    }
    let shadow = unsafe {
        if pyre_object::is_instance(concrete_obj) {
            let map = (*(concrete_obj as *const pyre_object::W_ObjectObject)).map;
            if map == 0 {
                return Ok(None);
            }
            // A devolved instance holds its attributes in a dictionary and
            // keeps the same map across a later `e.<name> = ...`, so pinning
            // the map would not observe the shadow the assignment installs.
            // `W_ObjectObject.map` is stored as a raw word; the map layer
            // owns the node type.
            if pyre_interpreter::objspace::std::mapdict::map_is_devolved(map as *const _) {
                return Ok(None);
            }
            ShadowGuard::InstanceMap(map as *const u8)
        } else if pyre_object::is_exception(concrete_obj) {
            ShadowGuard::ExceptionDictIsNull(pyre_object::w_exception_get_kind(concrete_obj))
        } else {
            // `load_method_fast_path` admits only the layouts above; keep the
            // two in step so a new layout there cannot reach an emit that has
            // no shadowing guard for it.
            return Ok(None);
        }
    };

    // guard_class(obj, ob_type): pins the payload layout, so the `w_class` and
    // shadowing-slot reads below name the fields they were recorded against.
    let physical_type = unsafe { (*concrete_obj).ob_type } as i64;
    if !ctx.trace_ctx.heap_cache().is_class_known(obj) {
        let type_const = ctx.trace_ctx.const_int(physical_type);
        walker_emit_fold_guard_with_snapshot(ctx, op_pc, OpCode::GuardClass, &[obj, type_const])?;
        ctx.trace_ctx
            .heap_cache_mut()
            .class_now_known(obj, physical_type);
    }

    // Pin the Python-level receiver class (`w_class`) exactly.  This is the
    // per-frame method namespace anchor: a subclass with the same instance
    // payload vtable side-exits instead of reusing the caller's method.
    let w_class_op = walker_record_getfield_gc_r_uncached(ctx, obj, crate::descr::w_class_descr());
    let w_type_const = ctx.trace_ctx.const_ref(w_type as i64);
    walker_emit_fold_guard_with_snapshot(
        ctx,
        op_pc,
        OpCode::GuardValue,
        &[w_class_op, w_type_const],
    )?;
    ctx.trace_ctx
        .heap_cache_mut()
        .replace_box(w_class_op, w_type_const);

    // typeobject.py `promote(self.version_tag())`: class mutation or method
    // reassignment bumps `_version_tag`, so the old `w_descr` side-exits.
    walker_pin_type_version_tag(ctx, op_pc, w_type_const)?;

    // Re-prove the shadowing precondition: growing an instance attribute named
    // like the method must side-exit before the constant descriptor is reused.
    // mapdict.py LOAD_ATTR caching does this by pinning the map; an exception
    // has no map, and pins the still-unallocated `w_dict` slot instead.
    let (slot_op, slot_const, shadow_guard) = match shadow {
        ShadowGuard::InstanceMap(map) => (
            walker_record_getfield_gc_i_uncached(ctx, obj, unsafe {
                crate::descr::mapdict_map_descr(concrete_obj)
            }),
            ctx.trace_ctx.const_int(map as i64),
            OpCode::GuardValue,
        ),
        // Pinning `w_dict` at null is a nullity test, and `pyjitpl.py
        // _establish_nullity` proves one with GUARD_ISNULL.  As a GUARD_VALUE
        // the guard's jitcounter keys on the *failing* value
        // (`compile.py make_a_counter_per_value`), which here is
        // whatever dictionary the assignment just allocated — a fresh address
        // every time, so no one value reaches `trace_eagerness` and the
        // has-a-dictionary continuation never gets a bridge.
        ShadowGuard::ExceptionDictIsNull(kind) => (
            walker_record_getfield_gc_r_uncached(
                ctx,
                obj,
                crate::descr::w_exception_dict_descr(kind),
            ),
            ctx.trace_ctx.const_ref(0),
            OpCode::GuardIsnull,
        ),
    };
    // GUARD_ISNULL carries only the pointer; the null constant is still what
    // the box is replaced with, the way `_establish_nullity` does it.  The
    // concrete is stamped here because `stamp_guard_value_concrete` reads it
    // off a GUARD_VALUE's expected operand, which GUARD_ISNULL does not carry.
    if shadow_guard == OpCode::GuardIsnull {
        ctx.trace_ctx
            .set_opref_concrete(slot_op, majit_ir::Value::Ref(majit_ir::GcRef(0)));
        walker_emit_fold_guard_with_snapshot(ctx, op_pc, shadow_guard, &[slot_op])?;
    } else {
        walker_emit_fold_guard_with_snapshot(ctx, op_pc, shadow_guard, &[slot_op, slot_const])?;
    }
    ctx.trace_ctx
        .heap_cache_mut()
        .replace_box(slot_op, slot_const);

    let method_const = ctx.trace_ctx.const_ref(w_descr as i64);
    write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, method_const)?;
    Ok(Some(()))
}

/// `LOAD_METHOD` classmethod fold for a type receiver (`Type.cmethod(...)`).
/// The safety oracle is [`pyre_interpreter::classmethod_on_type_fast_path`]: it
/// declines a custom metaclass, a metatype-defined name, an uncacheable type,
/// and any non-`classmethod` descriptor.  On success the walker pins the exact
/// type, its version tag, and the descriptor's `w_function?` slot, then writes
/// the classmethod's `__func__` as a green constant.  Because the method-load result is the plain `__func__` (not
/// a bound `Method`), the paired [`try_walker_fold_load_method_self`] runs
/// `compute_load_method_bound`, whose `is_type` + `is_exact_classmethod` arm
/// binds the type as `cls` — the same exactness this oracle applies, so the two
/// agree on a wrapper subclass.  The following `CALL` inlines `__func__(cls, ...)` — the
/// instance-method shape with the class in the receiver slot.
///
/// Carries the inline-depth restriction
/// [`try_walker_specialize_load_bound_method_attr`] documents: under the
/// single-frame collapse a fold guard inside an inlined callee sub-walk
/// resumes at the caller's CALL, re-running side effects.  The `getattr`
/// residual resumes past the call, so declining there re-runs nothing.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_walker_specialize_load_classmethod_attr<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    obj: OpRef,
    w_code_ptr: usize,
    name_idx: usize,
    dst: usize,
    dst_bank: char,
) -> Result<Option<()>, DispatchError> {
    if !ctx.is_authoritative_executor || dst_bank != 'r' {
        return Ok(None);
    }
    if ctx.fbw_mode.inline_subwalk && !walker_inline_guard_resumes_in_callee(ctx) {
        return Ok(None);
    }
    let Some(concrete_obj) = walker_concrete_ref_object(ctx, obj) else {
        return Ok(None);
    };
    let Some(name) = walker_load_name_from_code(w_code_ptr, name_idx) else {
        return Ok(None);
    };
    if name.contains("__") {
        return Ok(None);
    }
    let Some((w_type, version_tag, w_descr, w_func)) =
        (unsafe { pyre_interpreter::classmethod_on_type_fast_path(concrete_obj, &name) })
    else {
        return Ok(None);
    };
    if unsafe { resolve_inlinable_callee(w_func) }.is_none() {
        return Ok(None);
    }

    // Pin the exact class.  The receiver IS the type, so a single GuardValue
    // anchors both the metaclass (exact `type`, via `is_type`) and the MRO the
    // classmethod lookup walks; the version tag below covers method reassignment.
    let w_type_const = ctx.trace_ctx.const_ref(w_type as i64);
    walker_emit_fold_guard_with_snapshot(ctx, op_pc, OpCode::GuardValue, &[obj, w_type_const])?;
    ctx.trace_ctx
        .heap_cache_mut()
        .replace_box(obj, w_type_const);

    // typeobject.py `promote(self.version_tag())`: class mutation or rebinding
    // the attribute to a different descriptor in the class or any base bumps
    // `_version_tag`, so the pinned `__func__` side-exits.
    walker_pin_type_version_tag(ctx, op_pc, w_type_const)?;

    // What the version tag does NOT reach: re-initialising the classmethod in
    // place leaves the class dict, the descriptor's address, and every version
    // tag alone while replacing the callable this fold is about to bake.
    // `function.py:720` declares that slot `w_function?` for exactly this, and
    // `w_classmethod_set_func` forces the invalidation.
    walker_pin_descriptor_slot(
        ctx,
        op_pc,
        w_descr,
        crate::descr::classmethod_w_function_quasi_descr(),
    )?;

    let func_const = ctx.trace_ctx.const_ref(w_func as i64);
    write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, func_const)?;
    Ok(Some(()))
}

/// `Cls.__name__` `LOAD_ATTR` fold, on the safety oracle
/// [`pyre_interpreter::baseobjspace::type_name_obj_fast_path`]: a class whose
/// metaclass is exactly `type` resolves `__name__` through the metatype data
/// descriptor `type.__name__`, whose getter returns the class's `w_name` slot.
/// The fold is that slot read, under the guards that prove the receiver is
/// such a class:
///
///   guard_class(obj, W_TypeObject layout)      — `is_type`
///   guard_value(getfield(obj, w_class), type)  — the metaclass is `type`
///   getfield(obj, w_name) + guard_nonnull      — what the getter returns
///
/// The class itself is NOT pinned, unlike the classmethod fold beside this
/// one: nothing here depends on which class arrived, so a loop reading
/// `cls.__name__` over several classes keeps one trace.  Nor is the version
/// tag pinned — it is not what a rename moves (`descr_set__name__` skips
/// `mutated()`), and the live slot read already reports one.
///
/// `w_name` is filled in on first read and never cleared, so the null the
/// guard covers is a class this trace has not served before; its side exit
/// runs the residual, which materialises the slot, and re-entry folds.
///
/// Attempted inside an inlined callee sub-walk as well, on the same terms as
/// [`try_walker_specialize_load_attr`]: the guards prove the read, so the fold
/// cannot raise where the residual would not, and `cls.__name__` in a method
/// body is precisely where the opaque residual costs the most — it is what
/// makes the enclosing `FOR_ITER` decline to inline the callee at all.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_walker_specialize_load_type_name_attr<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    obj: OpRef,
    w_code_ptr: usize,
    name_idx: usize,
    dst: usize,
    dst_bank: char,
) -> Result<Option<()>, DispatchError> {
    if !ctx.is_authoritative_executor || dst_bank != 'r' {
        return Ok(None);
    }
    let Some(concrete_obj) = walker_concrete_ref_object(ctx, obj) else {
        return Ok(None);
    };
    let Some(name) = walker_load_name_from_code(w_code_ptr, name_idx) else {
        return Ok(None);
    };
    if name != "__name__" {
        return Ok(None);
    }
    let Some((metatype, w_name)) =
        (unsafe { pyre_interpreter::baseobjspace::type_name_obj_fast_path(concrete_obj) })
    else {
        return Ok(None);
    };
    // The metaclass guard below reads the raw `w_class` slot, while the fast
    // path answers through `typedef::type`, which falls back to
    // `gettypefor(ob_type)` when that slot is null.  A receiver reached through
    // that fallback would be guarded against a value its field never holds, and
    // nothing writes the slot afterwards, so the guard would fail on every
    // execution forever — one bridge per `trace_eagerness` bucket, without ever
    // converging.  Fold only what the guard can discharge.
    if !std::ptr::eq(unsafe { (*concrete_obj).w_class }, metatype) {
        return Ok(None);
    }

    // guard_class(obj, ob_type): the `W_TypeObject` layout both field reads
    // below index into.  `is_type` is this check.
    let phys_type = unsafe { (*concrete_obj).ob_type } as i64;
    if !ctx.trace_ctx.heap_cache().is_class_known(obj) {
        let type_const = ctx.trace_ctx.const_int(phys_type);
        walker_emit_fold_guard_with_snapshot(ctx, op_pc, OpCode::GuardClass, &[obj, type_const])?;
        ctx.trace_ctx
            .heap_cache_mut()
            .class_now_known(obj, phys_type);
    }

    // The metaclass, pinned to `type`: only then is `type.__name__` the
    // descriptor `descr_getattribute` selects, and only then is it fixed —
    // `type` is immutable, a user metaclass is not.
    let w_class_op = walker_record_getfield_gc_r_uncached(ctx, obj, crate::descr::w_class_descr());
    let metatype_const = ctx.trace_ctx.const_ref(metatype as i64);
    walker_emit_fold_guard_with_snapshot(
        ctx,
        op_pc,
        OpCode::GuardValue,
        &[w_class_op, metatype_const],
    )?;
    ctx.trace_ctx
        .heap_cache_mut()
        .replace_box(w_class_op, metatype_const);

    let name_op =
        crate::state::opimpl_getfield_gc_r(ctx.trace_ctx, obj, crate::descr::type_name_obj_descr());
    walker_emit_fold_guard_with_snapshot(ctx, op_pc, OpCode::GuardNonnull, &[name_op])?;
    ctx.trace_ctx.set_opref_concrete(
        name_op,
        majit_ir::Value::Ref(majit_ir::GcRef(w_name as usize)),
    );
    write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, name_op)?;
    Ok(Some(()))
}

/// Fold `LOAD_ATTR` on a type receiver when
/// [`pyre_interpreter::type_attr_value_fast_path`] proves that
/// `typeobject.py` `getattribute` returns the class-MRO value unchanged.  The exact
/// receiver and its version tag are pinned before the value is written as a
/// green constant.  [`pyre_interpreter::mutated`] recursively invalidates
/// subclasses, so the one receiver pin covers reassignment or deletion on any
/// base class as well.
///
/// A name the metatype answers with a data descriptor is refused by the oracle,
/// so `__name__` never reaches here — [`try_walker_specialize_load_type_name_attr`]
/// is its fold, and the two cover the disjoint arms of `descr_getattribute`.
///
/// The name needs no operand guard: the codewriter baked its `co_names` index
/// into the residual.  This read-only, present-attribute fold cannot raise, so
/// unlike the classmethod method-load fold it is safe inside an inlined callee
/// sub-walk; resuming past it cannot repeat a side effect.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_walker_specialize_load_type_attr<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    obj: OpRef,
    w_code_ptr: usize,
    name_idx: usize,
    dst: usize,
    dst_bank: char,
) -> Result<Option<()>, DispatchError> {
    if !ctx.is_authoritative_executor || dst_bank != 'r' {
        return Ok(None);
    }
    let Some(concrete_obj) = walker_concrete_ref_object(ctx, obj) else {
        return Ok(None);
    };
    let Some(name) = walker_load_name_from_code(w_code_ptr, name_idx) else {
        return Ok(None);
    };
    let Some((w_type, _version_tag, w_value)) = (unsafe {
        pyre_interpreter::type_attr_value_fast_path(concrete_obj, Wtf8::new(name.as_str()))
    }) else {
        return Ok(None);
    };

    let w_type_const = ctx.trace_ctx.const_ref(w_type as i64);
    walker_emit_fold_guard_with_snapshot(ctx, op_pc, OpCode::GuardValue, &[obj, w_type_const])?;
    ctx.trace_ctx
        .heap_cache_mut()
        .replace_box(obj, w_type_const);
    walker_pin_type_version_tag(ctx, op_pc, w_type_const)?;

    let value_const = ctx.trace_ctx.const_ref(w_value as i64);
    write_residual_call_result_to_dst(ctx, op_pc, dst, 'r', value_const)?;
    Ok(Some(()))
}

/// Fold the `LOAD_ATTR`-method `getattr` residual for a receiver whose name
/// resolves to a plain builtin-code function on its type — the `lst.append`
/// shape [`try_walker_specialize_load_method_attr`] declines because upstream
/// restricts its `[w_descr, w_obj]` push to `flag_method_descriptor` types.
///
/// Both engines materialise a `Method` here (`space.getattr`), but PyPy traces
/// *through* that `getattr` — it is ordinary RPython — so the type lookup folds
/// to a constant under `guard_class` + the version pin and the `Method` itself
/// virtualizes away: its optimized LOAD_METHOD emits no ops at all in a
/// steady-state loop (`pypy/objspace/std/callmethod.py:25-80`). pyre's `getattr`
/// is an opaque `CALL_MAY_FORCE` residual, which additionally drags a
/// `GUARD_NOT_FORCED` (forcing the virtualizable frame) and a `GUARD_NO_EXCEPTION`
/// through every iteration. This fold reproduces PyPy's shape directly:
///
///   guard_class(obj, ob_type)
///   guard_value(getfield(obj, w_class), the type)
///   guard_value(getfield(the type, version_tag), the tag)
///   new_with_vtable(Method) + setfield(w_function/w_self/w_class/header)
///
/// The guards make `lookup_in_type` constant exactly as the version-tag promote
/// does upstream, and the emitted `Method` is dead once the following `CALL`
/// folds — the append fold reads `w_function` / `w_self` straight back off it.
///
/// Returns `None` (fall through to the residual, SAFE) for every shape
/// [`pyre_interpreter::baseobjspace::bound_method_attr_fast_path`] declines.
///
/// Inside an inlined callee sub-walk the fold is restricted to a depth whose
/// guards resume at their own callee coordinate
/// ([`walker_inline_guard_resumes_in_callee`]).  Under the single-frame
/// collapse the reason [`try_walker_orthodox_list_append`] documents applies: a
/// guard resumes at the caller's CALL boundary, so a failure re-runs the callee
/// from its entry and doubles any side effect it sequenced before this
/// `LOAD_ATTR`. The residual resumes past the call instead, so declining there
/// re-runs nothing extra.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_walker_specialize_load_bound_method_attr<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    obj: OpRef,
    w_code_ptr: usize,
    name_idx: usize,
    dst: usize,
    dst_bank: char,
) -> Result<Option<()>, DispatchError> {
    if !ctx.is_authoritative_executor || dst_bank != 'r' {
        return Ok(None);
    }
    if ctx.fbw_mode.inline_subwalk && !walker_inline_guard_resumes_in_callee(ctx) {
        return Ok(None);
    }
    let Some(concrete_obj) = walker_concrete_ref_object(ctx, obj) else {
        return Ok(None);
    };
    let Some(name) = walker_load_name_from_code(w_code_ptr, name_idx) else {
        return Ok(None);
    };
    let Some((w_type, version_tag, w_descr)) = (unsafe {
        pyre_interpreter::baseobjspace::bound_method_attr_fast_path(concrete_obj, &name)
    }) else {
        return Ok(None);
    };

    // guard_class(obj, ob_type): the physical layout the `w_class` read below
    // needs, and what pins the receiver's builtin kind.
    let phys_type = unsafe { (*concrete_obj).ob_type } as i64;
    if !ctx.trace_ctx.heap_cache().is_class_known(obj) {
        let type_const = ctx.trace_ctx.const_int(phys_type);
        walker_emit_fold_guard_with_snapshot(ctx, op_pc, OpCode::GuardClass, &[obj, type_const])?;
        ctx.trace_ctx
            .heap_cache_mut()
            .class_now_known(obj, phys_type);
    }

    // Pin the Python-level class exactly: a subclass reaching the same
    // physical layout can define its own `name` and must side-exit.
    let w_class_op = walker_record_getfield_gc_r_uncached(ctx, obj, crate::descr::w_class_descr());
    let w_type_const = ctx.trace_ctx.const_ref(w_type as i64);
    walker_emit_fold_guard_with_snapshot(
        ctx,
        op_pc,
        OpCode::GuardValue,
        &[w_class_op, w_type_const],
    )?;
    ctx.trace_ctx
        .heap_cache_mut()
        .replace_box(w_class_op, w_type_const);

    // typeobject.py `promote(self.version_tag())`: reassigning the method on
    // the type bumps the tag, so the constant `w_descr` side-exits.
    walker_pin_type_version_tag(ctx, op_pc, w_type_const)?;

    // `w_method_new(w_descr, obj, w_type)` + the header stamp its allocation
    // performs (`ob_type` comes from the NewWithVtable's size descr).
    let func_const = ctx.trace_ctx.const_ref(w_descr as i64);
    let header_w_class = ctx
        .trace_ctx
        .const_ref(pyre_object::get_instantiate(&pyre_object::function::METHOD_TYPE) as i64);
    let method_op = crate::helpers::emit_bound_method_inline(
        ctx.trace_ctx,
        func_const,
        obj,
        w_type_const,
        header_w_class,
    );
    let method_type_addr = &pyre_object::function::METHOD_TYPE as *const _ as i64;
    ctx.trace_ctx
        .heap_cache_mut()
        .class_now_known(method_op, method_type_addr);
    // The concrete bound method the walker's own execution must observe; a
    // fresh `Method` per evaluation is what `getattr` produces anyway, so the
    // trace allocating its own is not an identity divergence.
    let bound = pyre_object::w_method_new(w_descr, concrete_obj, w_type);
    ctx.trace_ctx.set_opref_concrete(
        method_op,
        majit_ir::Value::Ref(majit_ir::GcRef(bound as usize)),
    );
    write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, method_op)?;
    Ok(Some(()))
}

/// Fold `bh_load_method_self_fn(obj, attr, code, name_idx)` once both the
/// receiver and the attribute are concrete.  The method-attribute fold above
/// already guards class, type version, and instance map; this second residual
/// is only the pure `compute_load_method_bound` binding decision.  A plain
/// instance-method bind writes the original red receiver box, not a baked
/// `ConstRef`, matching `callmethod.py f.pushvalue(w_obj)`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_walker_fold_load_method_self<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    obj: OpRef,
    attr: OpRef,
    _attr_reg: usize,
    w_code_ptr: usize,
    name_idx: usize,
    dst: usize,
    dst_bank: char,
) -> Result<Option<()>, DispatchError> {
    if !ctx.is_authoritative_executor || dst_bank != 'r' {
        return Ok(None);
    }
    let Some(concrete_obj) = walker_concrete_ref_object(ctx, obj) else {
        return Ok(None);
    };
    let Some(concrete_attr) = walker_concrete_ref_object(ctx, attr) else {
        return Ok(None);
    };
    // `compute_load_method_bound` answers PY_NULL for an already-bound method
    // without inspecting anything else, so pinning the attribute's class is
    // the whole precondition.  Left as a residual this is a second per-iteration
    // call on top of the `getattr` one (`lst.append(x)` pays both).
    if unsafe { pyre_object::is_method(concrete_attr) } {
        let method_type_addr = &pyre_object::function::METHOD_TYPE as *const _ as i64;
        let class_pinned = attr.is_constant() || ctx.trace_ctx.heap_cache().is_class_known(attr);
        if !class_pinned {
            // Under the single-frame collapse a guard here would resume at the
            // caller's CALL, re-running whatever that callee already did;
            // leave those to the residual (which resumes past the call).
            if ctx.fbw_mode.inline_subwalk && !walker_inline_guard_resumes_in_callee(ctx) {
                return Ok(None);
            }
            let type_const = ctx.trace_ctx.const_int(method_type_addr);
            walker_emit_fold_guard_with_snapshot(
                ctx,
                op_pc,
                OpCode::GuardClass,
                &[attr, type_const],
            )?;
            ctx.trace_ctx
                .heap_cache_mut()
                .class_now_known(attr, method_type_addr);
        }
        let null_const = ctx.trace_ctx.const_ref(pyre_object::PY_NULL as i64);
        write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, null_const)?;
        return Ok(Some(()));
    }
    let Some(name) = walker_load_name_from_code(w_code_ptr, name_idx) else {
        return Ok(None);
    };
    let bound =
        pyre_interpreter::eval::compute_load_method_bound(concrete_obj, concrete_attr, &name);
    let bound_op = if std::ptr::eq(bound, concrete_obj) {
        obj
    } else if bound == pyre_object::PY_NULL {
        ctx.trace_ctx.const_ref(pyre_object::PY_NULL as i64)
    } else {
        return Ok(None);
    };
    write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, bound_op)?;
    Ok(Some(()))
}

/// Emit `super(C, self).name` as the `Method` `W_Super.getattribute` builds,
/// in place of the opaque `bh_load_super_attr_fn` residual.
///
/// The residual rebuilds the proxy, re-walks the MRO suffix and re-binds the
/// descriptor on every iteration, and being may-force it also wipes the
/// trace's heap-field cache.  Upstream needs no such fold: `super()` there is
/// `LOAD_GLOBAL` + `CALL` + `LOAD_METHOD` traced generically, `W_Super`
/// virtualizes, and `lookup_starting_at` is an unrolled MRO walk.  The
/// codewriter is bytecode-driven and cannot expand the fused 3.14
/// `LOAD_SUPER_ATTR` into those three, so the fold is where the same trace
/// shape is reached — as
/// [`try_walker_specialize_load_bound_method_attr`] does for `LOAD_ATTR`.
///
/// The stack operands are authoritative for BOTH oparg forms.  The
/// zero-argument frame path they stand in for reads `locals_w[0]` and the
/// `__class__` freevar cell (`super_operands_from_frame`), which are exactly
/// what the `LOAD_FAST 0` / `LOAD_DEREF __class__` preceding this opcode
/// pushed; `LOAD_SUPER_ATTR_ATTR` / `LOAD_SUPER_ATTR_METHOD` read the same
/// two stack entries regardless of `oparg & 2`.
///
/// This emits one runtime guard FEWER than the ordinary method load: there is
/// no instance-map / exception-dict shadow guard, because
/// `W_Super.getattribute` never consults the receiver's dict, so `o.name = x`
/// cannot shadow `super().name`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_walker_specialize_load_super_attr<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    global_super: OpRef,
    self_obj: OpRef,
    cls: OpRef,
    w_code_ptr: usize,
    name_idx: usize,
    dst: usize,
    dst_bank: char,
) -> Result<Option<()>, DispatchError> {
    if !ctx.is_authoritative_executor || dst_bank != 'r' {
        return Ok(None);
    }
    if ctx.fbw_mode.inline_subwalk && !walker_inline_guard_resumes_in_callee(ctx) {
        return Ok(None);
    }
    let Some(concrete_super) = walker_concrete_ref_object(ctx, global_super) else {
        return Ok(None);
    };
    // Only the builtin `super` resolves through `W_Super.getattribute`; a
    // rebound global names some other callable entirely.
    if !pyre_interpreter::builtins::is_builtin_super_type(concrete_super) {
        return Ok(None);
    }
    let Some(concrete_cls) = walker_concrete_ref_object(ctx, cls) else {
        return Ok(None);
    };
    let Some(concrete_self) = walker_concrete_ref_object(ctx, self_obj) else {
        return Ok(None);
    };
    let Some(name) = walker_load_name_from_code(w_code_ptr, name_idx) else {
        return Ok(None);
    };
    let Some((objtype, _version_tag, w_descr, class_mode)) = (unsafe {
        pyre_interpreter::baseobjspace::super_attr_fast_path(concrete_cls, concrete_self, &name)
    }) else {
        return Ok(None);
    };
    let Some(binding) = super_attr_binding(w_descr, concrete_self, class_mode) else {
        return Ok(None);
    };

    // Which callable `super` names and which class the walk starts after are
    // both baked into the emitted body, so both are pinned.
    if !global_super.is_constant() {
        let super_const = ctx.trace_ctx.const_ref(concrete_super as i64);
        walker_emit_fold_guard_with_snapshot(
            ctx,
            op_pc,
            OpCode::GuardValue,
            &[global_super, super_const],
        )?;
        ctx.trace_ctx
            .heap_cache_mut()
            .replace_box(global_super, super_const);
    }
    let value_op = walker_emit_super_attr_result(
        ctx,
        op_pc,
        self_obj,
        cls,
        concrete_self,
        concrete_cls,
        objtype,
        w_descr,
        class_mode,
        binding,
    )?;
    write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, value_op)?;
    Ok(Some(()))
}

/// A Python-free result of binding the descriptor found by the `super` MRO
/// suffix walk.  Descriptor identity is protected by the receiver type's
/// version tag; `slot_pin` additionally protects an in-place replacement of a
/// staticmethod/classmethod wrapper's `_immutable_fields_` callable slot.
enum SuperAttrBinding {
    Constant {
        value: pyre_object::PyObjectRef,
        slot_pin: Option<majit_ir::DescrRef>,
    },
    Method {
        w_function: pyre_object::PyObjectRef,
        header: (bool, pyre_object::PyObjectRef),
        bind_to_class: bool,
        slot_pin: Option<majit_ir::DescrRef>,
    },
}

/// Classify the exact descriptor shapes for which PyPy's
/// `space.get(w_descr, descr_obj, objtype)` runs no Python.
fn super_attr_binding(
    w_descr: pyre_object::PyObjectRef,
    concrete_self: pyre_object::PyObjectRef,
    class_mode: bool,
) -> Option<SuperAttrBinding> {
    let descr_ob_type = unsafe { (*w_descr).ob_type };
    if std::ptr::eq(descr_ob_type, &pyre_interpreter::FUNCTION_TYPE as *const _)
        || std::ptr::eq(
            descr_ob_type,
            &pyre_interpreter::METHOD_DESCRIPTOR_TYPE as *const _,
        )
    {
        if class_mode {
            return Some(SuperAttrBinding::Constant {
                value: w_descr,
                slot_pin: None,
            });
        }
        return Some(SuperAttrBinding::Method {
            w_function: w_descr,
            header: super_attr_method_header(w_descr)?,
            bind_to_class: false,
            slot_pin: None,
        });
    }
    if unsafe { pyre_object::function::is_exact_staticmethod(w_descr) } {
        let mut value = unsafe { pyre_object::function::w_staticmethod_get_func(w_descr) };
        if value.is_null() {
            value = pyre_object::w_none();
        }
        return Some(SuperAttrBinding::Constant {
            value,
            slot_pin: Some(crate::descr::staticmethod_w_function_quasi_descr()),
        });
    }
    if unsafe { pyre_object::function::is_exact_classmethod(w_descr) } {
        let w_function = unsafe { pyre_object::function::w_classmethod_get_func(w_descr) };
        if w_function.is_null() {
            return None;
        }
        let header = pyre_object::get_instantiate(&pyre_object::function::METHOD_TYPE);
        if header.is_null() {
            return None;
        }
        return Some(SuperAttrBinding::Method {
            w_function,
            header: (false, header),
            bind_to_class: true,
            slot_pin: Some(crate::descr::classmethod_w_function_quasi_descr()),
        });
    }
    // `get`'s slot-wrapper arm, which class mode never reaches: there the
    // descriptor comes back unchanged.  Its instance check is a precondition of
    // the binding rather than part of it, so it is settled here against the
    // receiver whose class the emitted guards pin; a receiver it rejects
    // declines and raises in the interpreter.
    if !class_mode
        && unsafe {
            pyre_interpreter::baseobjspace::super_attr_slot_wrapper_binds(w_descr, concrete_self)
        }
    {
        return Some(SuperAttrBinding::Method {
            w_function: w_descr,
            header: super_attr_method_header(w_descr)?,
            bind_to_class: false,
            slot_pin: None,
        });
    }
    if unsafe {
        pyre_interpreter::baseobjspace::super_attr_returns_descr_unchanged(w_descr, class_mode)
    } {
        return Some(SuperAttrBinding::Constant {
            value: w_descr,
            slot_pin: None,
        });
    }
    None
}

/// The two words that separate the `Method` `get` builds for the descriptor
/// typedefs the `super` fold binds.
///
/// A `function` binds through `w_method_new`, which leaves `w_module` null and
/// lets the allocation's own header stand.  The other two arms bind through
/// `restamped_bound_method_new`, which is that same call followed by two
/// stores: the Python-visible class becomes `builtin_function_or_method` for a
/// `method_descriptor` and `method-wrapper` for a slot wrapper, and `w_module`
/// becomes `None`.  The payload is identical in all three, so one emission
/// serves them once these two words are chosen.
///
/// `None` when the chosen type object is not registered, which is resolved
/// here — before the caller emits anything — so the decline leaves the trace
/// untouched.
fn super_attr_method_header(
    w_descr: pyre_object::PyObjectRef,
) -> Option<(bool, pyre_object::PyObjectRef)> {
    let restamped_class = if unsafe { pyre_interpreter::is_method_descriptor(w_descr) } {
        Some(&pyre_interpreter::BUILTIN_FUNCTION_TYPE)
    } else if unsafe { pyre_interpreter::is_slot_wrapper(w_descr) } {
        Some(&pyre_interpreter::METHOD_WRAPPER_TYPE)
    } else {
        None
    };
    let header = match restamped_class {
        Some(ty) => pyre_interpreter::typedef::gettypeobject(ty),
        None => pyre_object::get_instantiate(&pyre_object::function::METHOD_TYPE),
    };
    (!header.is_null()).then_some((restamped_class.is_some(), header))
}

/// The body `super(cls, self).name` compiles to, once
/// `baseobjspace.rs super_attr_fast_path` has settled which class the MRO
/// suffix walk answers with (`objtype`) and what it finds there (`w_descr`).
///
/// Emitting starts here: every operand this needs is already proved, so a
/// caller that declines does so with the trace untouched.
///
/// Shared by the two spellings that reach the same lookup — `LOAD_SUPER_ATTR`,
/// and an attribute load on a proxy an earlier op built
/// ([`try_walker_specialize_load_attr_on_super`]).  Which callable `super`
/// names is the caller's question, because the two prove it differently: the
/// opcode form pins the global it loaded, while the proxy form has a
/// `GuardClass` on the proxy itself, which no other type can pass.
#[allow(clippy::too_many_arguments)]
fn walker_emit_super_attr_result<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    self_obj: OpRef,
    cls: OpRef,
    concrete_self: pyre_object::PyObjectRef,
    concrete_cls: pyre_object::PyObjectRef,
    objtype: pyre_object::PyObjectRef,
    w_descr: pyre_object::PyObjectRef,
    class_mode: bool,
    binding: SuperAttrBinding,
) -> Result<OpRef, DispatchError> {
    let objtype_const = walker_emit_super_attr_lookup_guards(
        ctx,
        op_pc,
        self_obj,
        cls,
        concrete_self,
        concrete_cls,
        objtype,
        class_mode,
    )?;

    let slot_pin = match &binding {
        SuperAttrBinding::Constant { slot_pin, .. } | SuperAttrBinding::Method { slot_pin, .. } => {
            slot_pin.clone()
        }
    };
    if let Some(field) = slot_pin {
        walker_pin_descriptor_slot(ctx, op_pc, w_descr, field)?;
    }

    let (w_function, method_header, bind_to_class) = match binding {
        SuperAttrBinding::Constant { value, .. } => {
            return Ok(ctx.trace_ctx.const_ref(value as i64));
        }
        SuperAttrBinding::Method {
            w_function,
            header,
            bind_to_class,
            ..
        } => (w_function, header, bind_to_class),
    };

    // `get(w_descr, self, objtype)` is `w_method_new(w_descr, self, objtype)`
    // plus the header stamp its allocation performs (`ob_type` comes from the
    // NewWithVtable's size descr).  [`super_attr_method_header`] has already
    // picked the header class, and its flag says whether the two extra stores
    // `builtin_bound_method_new` performs are owed on top.
    let (restamps_header, header_w_class_obj) = method_header;
    let func_const = ctx.trace_ctx.const_ref(w_function as i64);
    let header_w_class = ctx.trace_ctx.const_ref(header_w_class_obj as i64);
    let bound_self = if bind_to_class {
        objtype_const
    } else {
        self_obj
    };
    let method_op = crate::helpers::emit_bound_method_inline(
        ctx.trace_ctx,
        func_const,
        bound_self,
        objtype_const,
        header_w_class,
    );
    if restamps_header {
        // `w_method_new` leaves `w_module` null and a virtual reads an
        // unwritten field as null, so only the restamping arms owe the slot a
        // store.
        let module_descr = crate::descr::method_w_module_descr();
        let module_index = module_descr.index();
        let none_const = ctx.trace_ctx.const_ref(pyre_object::w_none() as i64);
        ctx.trace_ctx.record_op_with_descr(
            OpCode::SetfieldGc,
            &[method_op, none_const],
            module_descr,
        );
        ctx.trace_ctx
            .heapcache_setfield_cached(method_op, module_index, none_const);
    }
    // The physical layout is `Method` either way: `restamped_bound_method_new`
    // restamps the Python-visible `w_class`, not `ob_type`.
    let method_type_addr = &pyre_object::function::METHOD_TYPE as *const _ as i64;
    ctx.trace_ctx
        .heap_cache_mut()
        .class_now_known(method_op, method_type_addr);
    // The concrete bound method the walker's own execution must observe; a
    // fresh `Method` per evaluation is what `getattribute` produces anyway, so
    // the trace allocating its own is not an identity divergence.
    let concrete_bound_self = if bind_to_class {
        objtype
    } else {
        concrete_self
    };
    let bound = if restamps_header {
        pyre_interpreter::restamped_bound_method_new(
            w_function,
            concrete_bound_self,
            objtype,
            header_w_class_obj,
        )
    } else {
        pyre_object::w_method_new(w_function, concrete_bound_self, objtype)
    };
    ctx.trace_ctx.set_opref_concrete(
        method_op,
        majit_ir::Value::Ref(majit_ir::GcRef(bound as usize)),
    );
    Ok(method_op)
}

/// Emit the guards that make a recording-time `super` MRO suffix answer valid
/// on every execution.  Kept separate from result binding so a Python property
/// getter can enter the ordinary inline-call path after the same lookup guards.
#[allow(clippy::too_many_arguments)]
pub(crate) fn walker_emit_super_attr_lookup_guards<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    self_obj: OpRef,
    cls: OpRef,
    concrete_self: pyre_object::PyObjectRef,
    concrete_cls: pyre_object::PyObjectRef,
    objtype: pyre_object::PyObjectRef,
    class_mode: bool,
) -> Result<OpRef, DispatchError> {
    // The class the walk starts after is baked into the emitted body.
    let cls_const = ctx.trace_ctx.const_ref(concrete_cls as i64);
    if !cls.is_constant() {
        walker_emit_fold_guard_with_snapshot(ctx, op_pc, OpCode::GuardValue, &[cls, cls_const])?;
        ctx.trace_ctx.heap_cache_mut().replace_box(cls, cls_const);
    }

    let objtype_const = ctx.trace_ctx.const_ref(objtype as i64);
    if class_mode {
        // `_super_check`'s first arm: the receiver is the class whose MRO is
        // walked.  Pin that class object itself; its `w_class` is a metaclass
        // and is not the namespace anchor this lookup uses.
        if !self_obj.is_constant() {
            walker_emit_fold_guard_with_snapshot(
                ctx,
                op_pc,
                OpCode::GuardValue,
                &[self_obj, objtype_const],
            )?;
            ctx.trace_ctx
                .heap_cache_mut()
                .replace_box(self_obj, objtype_const);
        }
    } else {
        // guard_class(self, ob_type): the physical layout the `w_class` read
        // below needs.
        let phys_type = unsafe { (*concrete_self).ob_type } as i64;
        if !ctx.trace_ctx.heap_cache().is_class_known(self_obj) {
            let type_const = ctx.trace_ctx.const_int(phys_type);
            walker_emit_fold_guard_with_snapshot(
                ctx,
                op_pc,
                OpCode::GuardClass,
                &[self_obj, type_const],
            )?;
            ctx.trace_ctx
                .heap_cache_mut()
                .class_now_known(self_obj, phys_type);
        }

        // Pin the Python-level class exactly: a subclass reaching the same
        // physical layout has its own MRO suffix after `cls`.
        let w_class_op =
            walker_record_getfield_gc_r_uncached(ctx, self_obj, crate::descr::w_class_descr());
        walker_emit_fold_guard_with_snapshot(
            ctx,
            op_pc,
            OpCode::GuardValue,
            &[w_class_op, objtype_const],
        )?;
        ctx.trace_ctx
            .heap_cache_mut()
            .replace_box(w_class_op, objtype_const);
    }

    // typeobject.py `promote(self.version_tag())`.  Every class the suffix
    // walk reads is an ancestor of `objtype` and `mutated()` recurses into
    // subclasses, so this one tag covers a dict store or a `__bases__`
    // reassignment anywhere in that suffix.
    walker_pin_type_version_tag(ctx, op_pc, objtype_const)?;
    Ok(objtype_const)
}

/// `descriptor.py W_Super.getattribute` for a proxy the trace already holds —
/// the `su.name` half of the `su = super(...); su.name(...)` spelling, which
/// `LOAD_SUPER_ATTR` never sees because the name binding split the two.
///
/// Left alone this is an opaque `getattr_fn` MRO walk per iteration, and being
/// may-force it also wipes the trace's heap-field cache.  What replaces it is
/// the same body [`try_walker_specialize_load_super_attr`] emits: the two
/// operands come out of the proxy instead of off the stack.
///
/// Reading them is free where it matters.  When the proxy is the virtual
/// [`try_walker_specialize_two_arg_super_call`] emitted, `opimpl_getfield_gc_r`
/// answers from that emission's own `SetfieldGc` cache and no op is recorded at
/// all -- which is also what lets the allocation die, since a virtual whose
/// every read is answered has nothing left to materialise for.
///
/// `GuardClass(su, SUPER_TYPE)` is what stands in for the `global_super` pin
/// the opcode form carries: only `w_super_new` builds one of these, so a
/// receiver that passes the guard came from `super()` whatever the global
/// named at the time.
///
/// It does not stand in for the Python class, though.  `super_descr_new`
/// allocates a subclass instance through `w_super_new` too and then retags only
/// `w_class`, so `class MySuper(super)` shares the `ob_type` this guard reads
/// and would reuse the trace.  Its `__getattribute__` override owns the answer,
/// and this body does not run it, so the operand is pinned on the `w_class`
/// axis as well -- the same split `walker_exact_builtin_class` handles for the
/// numeric folds.  A proxy this walk emitted is virtual and carries the
/// canonical class by construction, so the pin costs it nothing.
pub(crate) fn try_walker_specialize_load_attr_on_super<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    obj: OpRef,
    name: &str,
    dst: usize,
    dst_bank: char,
) -> Result<Option<()>, DispatchError> {
    if !ctx.is_authoritative_executor || dst_bank != 'r' {
        return Ok(None);
    }
    if ctx.fbw_mode.inline_subwalk && !walker_inline_guard_resumes_in_callee(ctx) {
        return Ok(None);
    }
    let Some(concrete_proxy) = walker_concrete_ref_object(ctx, obj) else {
        return Ok(None);
    };
    if !unsafe { pyre_object::descriptor::is_super(concrete_proxy) } {
        return Ok(None);
    }
    let Some(proxy_w_class) = (unsafe { walker_exact_builtin_class(concrete_proxy) }) else {
        return Ok(None);
    };
    let concrete_cls = unsafe { pyre_object::descriptor::w_super_get_type(concrete_proxy) };
    let concrete_self = unsafe { pyre_object::descriptor::w_super_get_obj(concrete_proxy) };
    // `super_attr_fast_path` refuses a null receiver (the unbound `super(C)`
    // proxy), `__class__` / `__dict__`, an uncacheable type and a name no MRO
    // suffix answers -- every shape this must not emit.
    let Some((objtype, _version_tag, w_descr, class_mode)) = (unsafe {
        pyre_interpreter::baseobjspace::super_attr_fast_path(concrete_cls, concrete_self, name)
    }) else {
        return Ok(None);
    };
    let Some(binding) = super_attr_binding(w_descr, concrete_self, class_mode) else {
        return Ok(None);
    };

    let (self_op, cls_op) = walker_guard_and_read_super_proxy(
        ctx,
        op_pc,
        obj,
        proxy_w_class,
        concrete_self,
        concrete_cls,
    )?;
    let value_op = walker_emit_super_attr_result(
        ctx,
        op_pc,
        self_op,
        cls_op,
        concrete_self,
        concrete_cls,
        objtype,
        w_descr,
        class_mode,
        binding,
    )?;
    write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, value_op)?;
    Ok(Some(()))
}

/// Guard an already-built `super` proxy as the exact builtin implementation
/// and expose its two live lookup operands to the trace.
///
/// Both the Python-free result fold and the property-getter inline use this
/// prefix.  Descriptor classification happens before entry, so no caller can
/// decline after these guards merely because the selected descriptor needs a
/// different binding path.
pub(crate) fn walker_guard_and_read_super_proxy<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    obj: OpRef,
    proxy_w_class: pyre_object::PyObjectRef,
    concrete_self: pyre_object::PyObjectRef,
    concrete_cls: pyre_object::PyObjectRef,
) -> Result<(OpRef, OpRef), DispatchError> {
    let super_type_addr = &pyre_object::descriptor::SUPER_TYPE as *const _ as i64;
    if !ctx.trace_ctx.heap_cache().is_class_known(obj) {
        let type_const = ctx.trace_ctx.const_int(super_type_addr);
        walker_emit_fold_guard_with_snapshot(ctx, op_pc, OpCode::GuardClass, &[obj, type_const])?;
        ctx.trace_ctx
            .heap_cache_mut()
            .class_now_known(obj, super_type_addr);
    }
    walker_guard_exact_w_class(ctx, op_pc, obj, proxy_w_class)?;
    let cls_op = walker_read_super_field(
        ctx,
        obj,
        crate::descr::super_start_type_descr(),
        concrete_cls,
    );
    let self_op = walker_read_super_field(ctx, obj, crate::descr::super_obj_descr(), concrete_self);
    Ok((self_op, cls_op))
}

/// One `W_Super` field read, with the recording-time value attached when the
/// read did not already carry one.
///
/// A virtual proxy answers out of its own `SetfieldGc` cache and the operand
/// comes back already concrete; a materialised one records a `GETFIELD_GC_R`
/// whose live load may be absent, and the walk cannot continue on a box with
/// no concrete half.
fn walker_read_super_field<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    proxy: OpRef,
    descr: majit_ir::DescrRef,
    concrete: pyre_object::PyObjectRef,
) -> OpRef {
    let op = crate::state::opimpl_getfield_gc_r(ctx.trace_ctx, proxy, descr);
    if !matches!(
        ctx.trace_ctx.box_value(op),
        Some(majit_ir::Value::Ref(r)) if r != majit_ir::GcRef::NO_CONCRETE
    ) {
        ctx.trace_ctx
            .set_opref_concrete(op, majit_ir::Value::Ref(majit_ir::GcRef(concrete as usize)));
    }
    op
}

/// Two-argument `super(cls, obj)` reached as a call — the spelling a name
/// binding produces, and the one `LOAD_SUPER_ATTR` does not fuse away.
///
/// `try_walker_specialize_bare_super_call` is its zero-argument sibling and
/// re-routes rather than removes, because zero-argument `super()` reads the
/// frame and the frame read has to happen on a channel the walker can see.
/// Two arguments read nothing: `descriptor.py super_init_impl` validates the
/// pair and stores three words, so the whole call is an allocation and the
/// emission is that allocation spelled out.
///
/// Removing the CALL removes more than the call.  `bh_call_fn` is may-force,
/// so the walk publishes a vref for the executing frame ahead of it
/// (`ForceToken`, a `NewWithVtable(VRef)`, a store into
/// `ExecutionContext.topframeref`) and re-checks `GuardNotForced` /
/// `GuardNoException` after -- 9 ops around one that allocates 4 words.  With
/// the proxy emitted as `New` + `SetfieldGc` and its reads answered
/// ([`try_walker_specialize_load_attr_on_super`]), the optimizer drops the
/// allocation entirely for a proxy that never escapes.
///
/// Only the pair `_super_check` settles by walking installed MROs is folded:
/// its third arm asks for `__class__`, which a property answers with arbitrary
/// Python, and the walk executes its own emissions concretely, so a fold that
/// reached that arm would run user code at recording time and then repeat it
/// under the residual on a decline.  The first arm is the class-method case:
/// it stores the class itself as both `w_objtype` and `w_self`; the proxy
/// emission below pins that class object directly rather than reading its
/// metaclass out of `w_class`.
pub(crate) fn try_walker_specialize_two_arg_super_call<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    code: &[u8],
    op: &DecodedOp,
    r_args: &[OpRef],
    dst: usize,
) -> Result<Option<()>, DispatchError> {
    // `super(cls, obj)` arrives as `[callable, null_or_self, cls, obj]` — the
    // same `bh_call_fn` operand list the zero-argument sibling reads, with the
    // two user arguments after the bound-receiver slot.
    if r_args.len() != 4 {
        return Ok(None);
    }
    if ctx.fbw_mode.inline_subwalk && !walker_inline_guard_resumes_in_callee(ctx) {
        return Ok(None);
    }
    let arg_concretes = read_ref_var_list_concrete(code, op, 1, ctx);
    let (
        ConcreteValue::Ref(concrete_callable),
        ConcreteValue::Ref(null_or_self),
        ConcreteValue::Ref(concrete_cls),
        ConcreteValue::Ref(concrete_obj),
    ) = (
        arg_concretes[0],
        arg_concretes[1],
        arg_concretes[2],
        arg_concretes[3],
    )
    else {
        return Ok(None);
    };
    if concrete_callable.is_null()
        || !null_or_self.is_null()
        || !pyre_interpreter::builtins::is_builtin_super_type(concrete_callable)
    {
        return Ok(None);
    }
    if concrete_cls.is_null() || concrete_obj.is_null() {
        return Ok(None);
    }
    // `descriptor.py:28-30` — `None` builds the UNBOUND proxy, whose `w_self`
    // is null and whose attribute reads take a different arm entirely.
    if unsafe { pyre_object::is_none(concrete_obj) } {
        return Ok(None);
    }
    if !unsafe { pyre_object::is_type(concrete_cls) } {
        return Ok(None);
    }
    let Some(objtype) =
        pyre_interpreter::builtins::super_check_python_free(concrete_cls, concrete_obj)
    else {
        return Ok(None);
    };
    let class_mode =
        unsafe { pyre_object::is_type(concrete_obj) } && std::ptr::eq(objtype, concrete_obj);
    // In instance mode the receiver's class is read back out of the object
    // below, so the two must be the same word: an exception instance carrying
    // the generic stub resolves its class through the kind registry instead.
    // In class mode `_super_check` returns the receiver class itself and its
    // `w_class` is the metaclass, so identity of `obj_op` is the guard instead.
    if !class_mode && !std::ptr::eq(objtype, unsafe { (*concrete_obj).w_class }) {
        return Ok(None);
    }

    let callable_op = r_args[0];
    let cls_op = r_args[2];
    let obj_op = r_args[3];
    // Which callable `super` names is baked into the emitted body.
    if !callable_op.is_constant() {
        let expected = ctx.trace_ctx.const_ref(concrete_callable as i64);
        walker_emit_fold_guard_with_snapshot(
            ctx,
            op.pc,
            OpCode::GuardValue,
            &[callable_op, expected],
        )?;
        ctx.trace_ctx
            .heap_cache_mut()
            .replace_box(callable_op, expected);
    }
    let proxy_op = walker_emit_super_proxy(
        ctx,
        op.pc,
        cls_op,
        obj_op,
        concrete_cls,
        concrete_obj,
        objtype,
        class_mode,
    )?;
    write_residual_call_result_to_dst(ctx, op.pc, dst, 'r', proxy_op)?;
    Ok(Some(()))
}

/// The proxy `descriptor.py super_init_impl` stores, emitted as a virtual.
///
/// Shared by the two spellings that reach it with a settled pair: the explicit
/// `super(cls, obj)` call, and the zero-argument one whose operands come out of
/// the callee's own frame slots
/// ([`try_walker_specialize_bare_super_virtual`]).  How each proves its two
/// operands is the caller's question; from here the guards and the emission are
/// the same.
///
/// Emitting starts here, so a caller that declines does so with the trace
/// untouched.
fn walker_emit_super_proxy<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    cls_op: OpRef,
    obj_op: OpRef,
    concrete_cls: pyre_object::PyObjectRef,
    concrete_obj: pyre_object::PyObjectRef,
    objtype: pyre_object::PyObjectRef,
    class_mode: bool,
) -> Result<OpRef, DispatchError> {
    let cls_const = ctx.trace_ctx.const_ref(concrete_cls as i64);
    if !cls_op.is_constant() {
        walker_emit_fold_guard_with_snapshot(ctx, op_pc, OpCode::GuardValue, &[cls_op, cls_const])?;
        ctx.trace_ctx
            .heap_cache_mut()
            .replace_box(cls_op, cls_const);
    }
    let objtype_const = ctx.trace_ctx.const_ref(objtype as i64);
    if class_mode {
        // descriptor.py `_super_check`'s first arm returns `w_obj_or_type`
        // itself.  Pin that class object: guarding its physical TYPE_TYPE
        // layout would admit every class and reading `w_class` would produce
        // the metaclass, neither of which protects the baked MRO root.
        if !obj_op.is_constant() {
            walker_emit_fold_guard_with_snapshot(
                ctx,
                op_pc,
                OpCode::GuardValue,
                &[obj_op, objtype_const],
            )?;
            ctx.trace_ctx
                .heap_cache_mut()
                .replace_box(obj_op, objtype_const);
        }
    } else {
        // guard_class(obj, ob_type): the physical layout the `w_class` read
        // below needs.
        let phys_type = unsafe { (*concrete_obj).ob_type } as i64;
        if !ctx.trace_ctx.heap_cache().is_class_known(obj_op) {
            let type_const = ctx.trace_ctx.const_int(phys_type);
            walker_emit_fold_guard_with_snapshot(
                ctx,
                op_pc,
                OpCode::GuardClass,
                &[obj_op, type_const],
            )?;
            ctx.trace_ctx
                .heap_cache_mut()
                .class_now_known(obj_op, phys_type);
        }
        // `_super_check`'s answer is baked, so pin the receiver's exact Python
        // class.  An exception instance carrying the generic stub may resolve
        // its class through the kind registry instead and was declined above.
        let w_class_op =
            walker_record_getfield_gc_r_uncached(ctx, obj_op, crate::descr::w_class_descr());
        walker_emit_fold_guard_with_snapshot(
            ctx,
            op_pc,
            OpCode::GuardValue,
            &[w_class_op, objtype_const],
        )?;
        ctx.trace_ctx
            .heap_cache_mut()
            .replace_box(w_class_op, objtype_const);
    }
    // A `__bases__` reassignment anywhere in the selected class's ancestry
    // bumps this tag and can make `issubtype_w(objtype, cls)` stop holding.
    walker_pin_type_version_tag(ctx, op_pc, objtype_const)?;

    let header_w_class = ctx
        .trace_ctx
        .const_ref(pyre_object::get_instantiate(&pyre_object::descriptor::SUPER_TYPE) as i64);
    let proxy_op = crate::helpers::emit_super_inline(
        ctx.trace_ctx,
        cls_const,
        objtype_const,
        obj_op,
        header_w_class,
    );
    let super_type_addr = &pyre_object::descriptor::SUPER_TYPE as *const _ as i64;
    ctx.trace_ctx
        .heap_cache_mut()
        .class_now_known(proxy_op, super_type_addr);
    // The concrete proxy the walker's own execution must observe.  Built last:
    // it allocates, and every address baked above is read before it runs.
    let proxy = pyre_object::descriptor::w_super_new(concrete_cls, objtype, concrete_obj);
    ctx.trace_ctx.set_opref_concrete(
        proxy_op,
        majit_ir::Value::Ref(majit_ir::GcRef(proxy as usize)),
    );
    Ok(proxy_op)
}

/// Fold `super_attr_unwrap(raw, which)` — the LOAD_SUPER_ATTR method form's
/// `[func, self_or_null]` split — once `raw` is concrete.  The interpreter
/// spells the same decision inline (`is_method` then `w_method_get_func` /
/// `w_method_get_self`, else `(raw, PY_NULL)`), so left as a residual it is a
/// second and third per-iteration call on top of the attribute one, and it
/// FORCES the `Method` [`try_walker_specialize_load_super_attr`] emits
/// instead of letting it virtualize away.
pub(crate) fn try_walker_fold_super_attr_unwrap<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    raw: OpRef,
    which: i64,
    dst: usize,
    dst_bank: char,
) -> Result<Option<()>, DispatchError> {
    if !ctx.is_authoritative_executor || dst_bank != 'r' {
        return Ok(None);
    }
    let Some(concrete_raw) = walker_concrete_ref_object(ctx, raw) else {
        return Ok(None);
    };
    // A POSITIVE class pin, which decides `is_method` in BOTH directions: a
    // `raw` that is a `Method` on one iteration and a plain function on the
    // next side-exits rather than re-running the baked arm.
    if !raw.is_constant() && !ctx.trace_ctx.heap_cache().is_class_known(raw) {
        // Under the single-frame collapse a guard here would resume at the
        // caller's CALL, re-running whatever that callee already did.
        if ctx.fbw_mode.inline_subwalk && !walker_inline_guard_resumes_in_callee(ctx) {
            return Ok(None);
        }
        let phys_type = unsafe { (*concrete_raw).ob_type } as i64;
        let type_const = ctx.trace_ctx.const_int(phys_type);
        walker_emit_fold_guard_with_snapshot(ctx, op_pc, OpCode::GuardClass, &[raw, type_const])?;
        ctx.trace_ctx
            .heap_cache_mut()
            .class_now_known(raw, phys_type);
    }
    let value = if unsafe { pyre_object::is_method(concrete_raw) } {
        let (descr, concrete) = if which == 0 {
            (crate::descr::method_w_function_descr(), unsafe {
                pyre_object::w_method_get_func(concrete_raw)
            })
        } else {
            (crate::descr::method_w_self_descr(), unsafe {
                pyre_object::w_method_get_self(concrete_raw)
            })
        };
        // The CACHED read, not the uncached one.  The producing fold primes
        // both fields through `heapcache_setfield_cached`, so this resolves to
        // the value it stored and records no op at all.  An uncached
        // `GETFIELD_GC_R` hands the following CALL a callable slot with no
        // concrete ref, which declines an inline the residual this replaces
        // did not — measured as `[inline-decline] why=callable slot carries no
        // concrete ref` and a slower loop than leaving the residual alone.
        let op = crate::state::opimpl_getfield_gc_r(ctx.trace_ctx, raw, descr);
        let resolved = matches!(
            ctx.trace_ctx.box_value(op),
            Some(majit_ir::Value::Ref(r)) if r != majit_ir::GcRef::NO_CONCRETE
        );
        if !resolved {
            // A `Method` this fold did not build reaches the cache empty; the
            // read is still the one the helper performs, so give the walker
            // the value it would have returned.
            ctx.trace_ctx
                .set_opref_concrete(op, majit_ir::Value::Ref(majit_ir::GcRef(concrete as usize)));
        }
        op
    } else if which == 0 {
        raw
    } else {
        // `PY_NULL` is the correct `self` slot for a non-`Method` attribute;
        // it flows into the following CALL's checked `null_or_self` operand.
        ctx.trace_ctx.const_ref(pyre_object::PY_NULL as i64)
    };
    write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, value)?;
    Ok(Some(()))
}

/// How the add-transition fold pins the stored value's type, resolved before
/// any guard is emitted so a decline falls cleanly to the residual.
enum StoreAttrAddValuePin {
    Boxed,
    /// Fresh unboxed int slot; `Some` pins a heap operand's canonical
    /// `w_class`, `None` means tagged (the unbox's tag guard is the pin).
    UnboxedInt(Option<pyre_object::PyObjectRef>),
}

/// `None` (unpinnable `w_class`, or a float pick) keeps the residual.
fn store_attr_add_value_pin(
    add: &pyre_interpreter::objspace::std::mapdict::StoreAttrAdd,
    concrete_value: pyre_object::PyObjectRef,
) -> Option<StoreAttrAddValuePin> {
    match add.unbox_type {
        None => Some(StoreAttrAddValuePin::Boxed),
        Some(pyre_interpreter::objspace::std::mapdict::UnboxType::Int) => {
            if pyre_object::tagged_int::CAN_BE_TAGGED
                && unsafe { pyre_object::tagged_int::is_tagged_int(concrete_value) }
            {
                return Some(StoreAttrAddValuePin::UnboxedInt(None));
            }
            unsafe { walker_exact_builtin_class(concrete_value) }
                .map(|canonical| StoreAttrAddValuePin::UnboxedInt(Some(canonical)))
        }
        // `store_attr_add_fast_path` never resolves a float pick; defensive.
        Some(pyre_interpreter::objspace::std::mapdict::UnboxType::Float) => None,
    }
}

pub(crate) fn try_walker_specialize_store_attr<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    obj: OpRef,
    value: OpRef,
    w_code_ptr: usize,
    name_idx: usize,
    original_effect: &majit_ir::EffectInfo,
) -> Result<Option<WalkerStoreAttrSpecialization>, DispatchError> {
    if !ctx.is_authoritative_executor || w_code_ptr == 0 {
        return Ok(None);
    }
    let (Some(concrete_obj), Some(concrete_value)) = (
        walker_concrete_ref_object(ctx, obj),
        walker_concrete_ref_object(ctx, value),
    ) else {
        return Ok(None);
    };
    let name = unsafe {
        let code_ptr = pyre_interpreter::w_code_get_ptr(w_code_ptr as pyre_object::PyObjectRef);
        if code_ptr.is_null() {
            return Ok(None);
        }
        let code = &*(code_ptr as *const pyre_interpreter::CodeObject);
        match pyre_interpreter::pyframe::load_name_from_code(code, name_idx) {
            Some(n) => n.to_string(),
            None => return Ok(None),
        }
    };
    if let Some((w_type, version_tag, map, storageindex, listindex, unbox_type, attr)) = unsafe {
        pyre_interpreter::objspace::std::mapdict::store_attr_unboxed_fast_path(concrete_obj, &name)
    } {
        match unbox_type {
            pyre_interpreter::objspace::std::mapdict::UnboxType::Int => {
                // Match mapdict.py `_direct_write` exactly, through the same
                // predicate the interpreter's own store uses: `is_int` reads
                // `ob_type`, which an `int` subclass shares, so unboxing on it
                // would take the raw payload and lose `w_class`.
                if !unsafe {
                    pyre_interpreter::objspace::std::mapdict::is_unboxable_int(concrete_value)
                } {
                    return Ok(None);
                }
            }
            pyre_interpreter::objspace::std::mapdict::UnboxType::Float => {
                // Match mapdict.py `_direct_write` exactly: subclasses and
                // NaNs convert the slot to boxed storage.
                if !unsafe {
                    pyre_interpreter::objspace::std::mapdict::is_unboxable_float(concrete_value)
                } {
                    return Ok(None);
                }
            }
        }

        walker_guard_mapdict_instance_shape(
            ctx,
            op_pc,
            obj,
            concrete_obj,
            w_type,
            version_tag,
            map,
        )?;
        unsafe { pyre_interpreter::objspace::std::mapdict::mark_attr_ever_mutated(attr) };
        let storageindex_const = ctx.trace_ctx.const_int(storageindex as i64);
        let listindex_const = ctx.trace_ctx.const_int(listindex as i64);
        let (helper_fn, raw, value_type) = match unbox_type {
            pyre_interpreter::objspace::std::mapdict::UnboxType::Int => {
                let int_type_addr = &pyre_object::pyobject::INT_TYPE as *const _ as i64;
                let raw = walker_unbox_int(ctx, op_pc, value, int_type_addr)?;
                // A subclass shares the builtin's `ob_type`, which is all the unbox
                // guard proves; the operand gate read `w_class`, so pin that too.
                walker_guard_exact_w_class(
                    ctx,
                    op_pc,
                    value,
                    walker_numeric_builtin_class(concrete_value),
                )?;
                (
                    crate::helpers::jit_mapdict_unboxed_write_raw as *const (),
                    raw,
                    majit_ir::Type::Int,
                )
            }
            pyre_interpreter::objspace::std::mapdict::UnboxType::Float => {
                let float_type_addr = &pyre_object::pyobject::FLOAT_TYPE as *const _ as i64;
                let raw = walker_unbox_float(ctx, op_pc, value, float_type_addr)?;
                // A subclass shares the builtin's `ob_type`, which is all the unbox
                // guard proves; the operand gate read `w_class`, so pin that too.
                walker_guard_exact_w_class(
                    ctx,
                    op_pc,
                    value,
                    walker_numeric_builtin_class(concrete_value),
                )?;
                let live_f = unsafe { pyre_object::w_float_get_value(concrete_value) };
                ctx.trace_ctx
                    .set_opref_concrete(raw, majit_ir::Value::Float(live_f));
                walker_guard_float_not_nan(ctx, op_pc, raw)?;
                (
                    crate::helpers::jit_mapdict_unboxed_write_f as *const (),
                    raw,
                    majit_ir::Type::Float,
                )
            }
        };
        let helper = ctx.trace_ctx.const_int(helper_fn as usize as i64);

        // `original_effect` is the generic setattr residual's
        // `EF_RANDOM_EFFECTS`, whose six raw sets and six bitstrings are
        // all `None` (`effectinfo.py:149-155`). Cloning it and lowering
        // only `extraeffect` would build the "non-random + raw=None"
        // shape `effectinfo.py:149-162` asserts against, and
        // `check_write_descr_field` would then unwrap a `None`
        // bitstring. Build the downgraded effect from scratch so the
        // sets match the extraeffect: the raw slot write cannot raise,
        // and it touches no field or array descr the heap cache holds
        // (the storage is reached only through this helper pair, never
        // through `getfield_gc` / `getarrayitem_gc`).
        let mut effect = majit_ir::EffectInfo::const_new(
            majit_ir::ExtraEffect::CannotRaise,
            majit_ir::OopSpecIndex::None,
        );
        effect.runtime_helper = majit_ir::RuntimeHelperKind::StoreAttr;
        let descr = majit_metainterp::make_call_descr_with_effect(
            &[
                majit_ir::Type::Ref,
                majit_ir::Type::Int,
                majit_ir::Type::Int,
                value_type,
            ],
            majit_ir::Type::Void,
            effect,
        );
        // ABI order follows the write helpers: receiver and the two guarded green
        // coordinates, then the raw symbolic value in its own bank.  No box is
        // materialized for this write.
        return Ok(Some(WalkerStoreAttrSpecialization::Residual(
            descr,
            vec![helper, obj, storageindex_const, listindex_const, raw],
        )));
    }

    if let Some((slot, kind, w_type, version_tag, _stored)) = unsafe {
        pyre_interpreter::baseobjspace::exception_attr_slot_fold(concrete_obj, &name, true)
    } {
        if slot == pyre_interpreter::baseobjspace::ExceptionAttrSlot::Args {
            let tuple_type = &pyre_object::TUPLE_TYPE as *const pyre_object::PyType;
            let canonical_tuple_class = pyre_object::get_instantiate(&pyre_object::TUPLE_TYPE);
            if !unsafe {
                std::ptr::eq((*concrete_value).ob_type, tuple_type)
                    && std::ptr::eq((*concrete_value).w_class, canonical_tuple_class)
            } {
                return Ok(None);
            }
        }
        walker_guard_exception_attr_slot(ctx, op_pc, obj, concrete_obj, w_type, version_tag)?;
        let (stored_value, concrete_stored) =
            if slot == pyre_interpreter::baseobjspace::ExceptionAttrSlot::Args {
                let tuple_type = &pyre_object::TUPLE_TYPE as *const pyre_object::PyType;
                let canonical_tuple_class = pyre_object::get_instantiate(&pyre_object::TUPLE_TYPE);
                if !unsafe {
                    std::ptr::eq((*concrete_value).ob_type, tuple_type)
                        && std::ptr::eq((*concrete_value).w_class, canonical_tuple_class)
                } {
                    return Ok(None);
                }
                let tuple_type_addr = tuple_type as i64;
                if !ctx.trace_ctx.heap_cache().is_class_known(value) {
                    let type_const = ctx.trace_ctx.const_int(tuple_type_addr);
                    walker_emit_fold_guard_with_snapshot(
                        ctx,
                        op_pc,
                        OpCode::GuardClass,
                        &[value, type_const],
                    )?;
                    ctx.trace_ctx
                        .heap_cache_mut()
                        .class_now_known(value, tuple_type_addr);
                }
                walker_guard_exact_w_class(ctx, op_pc, value, canonical_tuple_class)?;
                let block = crate::state::opimpl_getfield_gc_r(
                    ctx.trace_ctx,
                    value,
                    crate::descr::tuple_wrappeditems_descr(),
                );
                let len = unsafe { pyre_object::w_tuple_len(concrete_value) };
                let length = crate::state::opimpl_arraylen_gc(
                    ctx.trace_ctx,
                    block,
                    crate::state::pyobject_gcarray_descr(),
                );
                let len_const = ctx.trace_ctx.const_int(len as i64);
                walker_emit_fold_guard_with_snapshot(
                    ctx,
                    op_pc,
                    OpCode::GuardValue,
                    &[length, len_const],
                )?;
                let mut items = Vec::with_capacity(len);
                let mut concrete_items = Vec::with_capacity(len);
                for index in 0..len {
                    let index_op = ctx.trace_ctx.const_int(index as i64);
                    items.push(crate::state::trace_items_block_getitem_value(
                        ctx.trace_ctx,
                        block,
                        index_op,
                    ));
                    concrete_items.push(
                        unsafe { pyre_object::w_tuple_getitem(concrete_value, index as i64) }
                            .unwrap_or(pyre_object::PY_NULL),
                    );
                }
                let list = crate::helpers::emit_object_list_inline(ctx.trace_ctx, &items);
                let concrete_list = pyre_object::w_list_new_object(concrete_items);
                ctx.trace_ctx.set_opref_concrete(
                    list,
                    majit_ir::Value::Ref(majit_ir::GcRef(concrete_list as usize)),
                );
                (list, concrete_list)
            } else {
                (value, concrete_value)
            };
        let field_descr = crate::descr::w_exception_attr_slot_descr(kind, slot);
        let field_index = field_descr.index();
        ctx.trace_ctx
            .record_op_with_descr(OpCode::SetfieldGc, &[obj, stored_value], field_descr);
        ctx.trace_ctx
            .heapcache_setfield_cached(obj, field_index, stored_value);
        // The walk is the authoritative execution path.  Apply the same raw
        // slot writer now so interpreter execution after a side exit observes
        // the store; the writer supplies the host-side remembered-set barrier.
        // Compiled SetfieldGc reference stores receive CondCallGcWb from
        // majit-gc's rewrite pass, consumed by both dynasm and cranelift.
        unsafe {
            match slot {
                pyre_interpreter::baseobjspace::ExceptionAttrSlot::Args => {
                    pyre_object::interp_exceptions::w_exception_set_args(
                        concrete_obj,
                        concrete_stored,
                    )
                }
                pyre_interpreter::baseobjspace::ExceptionAttrSlot::Context
                | pyre_interpreter::baseobjspace::ExceptionAttrSlot::Cause
                | pyre_interpreter::baseobjspace::ExceptionAttrSlot::Traceback
                | pyre_interpreter::baseobjspace::ExceptionAttrSlot::Name
                | pyre_interpreter::baseobjspace::ExceptionAttrSlot::AttrObj
                | pyre_interpreter::baseobjspace::ExceptionAttrSlot::UnicodeObject
                | pyre_interpreter::baseobjspace::ExceptionAttrSlot::UnicodeStart
                | pyre_interpreter::baseobjspace::ExceptionAttrSlot::UnicodeEnd
                | pyre_interpreter::baseobjspace::ExceptionAttrSlot::UnicodeReason
                | pyre_interpreter::baseobjspace::ExceptionAttrSlot::UnicodeEncoding => {
                    // `exception_attr_slot_fold` declines these for stores, so
                    // the store fold never reaches here.
                    unreachable!("load-only exception slots fold on load only")
                }
                pyre_interpreter::baseobjspace::ExceptionAttrSlot::Code => {
                    pyre_object::interp_exceptions::w_exception_set_code(
                        concrete_obj,
                        concrete_value,
                    )
                }
                pyre_interpreter::baseobjspace::ExceptionAttrSlot::Errno => {
                    pyre_object::interp_exceptions::w_exception_set_errno(
                        concrete_obj,
                        concrete_value,
                    )
                }
                pyre_interpreter::baseobjspace::ExceptionAttrSlot::Strerror => {
                    pyre_object::interp_exceptions::w_exception_set_strerror(
                        concrete_obj,
                        concrete_value,
                    )
                }
                pyre_interpreter::baseobjspace::ExceptionAttrSlot::Filename => {
                    pyre_object::interp_exceptions::w_exception_set_filename(
                        concrete_obj,
                        concrete_value,
                    )
                }
                pyre_interpreter::baseobjspace::ExceptionAttrSlot::Filename2 => {
                    pyre_object::interp_exceptions::w_exception_set_filename2(
                        concrete_obj,
                        concrete_value,
                    )
                }
            }
        }
        return Ok(Some(WalkerStoreAttrSpecialization::Direct));
    }

    // The attribute is not in the map yet: fold the `map -> PlainAttribute`
    // transition and the grow-by-one storage rewrite into trace ops instead of
    // leaving the generic `setattr` residual, which would force the receiver.
    //
    // Only for a receiver this trace allocated and has not let escape.  The
    // emitted transition is a pair of raw field stores, so unlike the
    // interpreter's it does not hold the striped `instance_lock`, and unlike
    // the single-slot in-place write it publishes two fields: a concurrent
    // mutator of the same instance could pair one thread's `map` with
    // another's `storage`.  `is_unescaped` is what rules that out — no other
    // thread has a reference yet.  It costs almost nothing, because the fold's
    // payoff is exactly the unescaped case: an escaped receiver is one the
    // optimizer cannot remove anyway.
    if ctx.trace_ctx.heap_cache().is_unescaped(obj)
        && let Some(add) = unsafe {
            pyre_interpreter::objspace::std::mapdict::store_attr_add_fast_path(
                concrete_obj,
                &name,
                concrete_value,
            )
        }
        && let Some(value_pin) = store_attr_add_value_pin(&add, concrete_value)
    {
        walker_guard_mapdict_instance_shape(
            ctx,
            op_pc,
            obj,
            concrete_obj,
            add.w_type,
            add.version_tag,
            add.map,
        )?;
        walker_pin_holder_typ(ctx, op_pc, add.holder)?;
        // This marker is not redundant with the instance-map GuardValue. The
        // fold bakes `add.new_map`, the transition target, as a green
        // `const_int`, while the guard pins `add.map`, the instance map before
        // the transition. `holder_pick_attr` can replace `holder.attr` with a
        // fresh `PlainAttribute` without changing any instance's current map;
        // this marker protects the baked target.
        walker_pin_holder_attr(ctx, op_pc, add.holder)?;
        // General rule: plant the `allow_unboxing` marker only where the fold
        // read the flag as true. Marking a false read re-arms a permanently
        // dead field and lets later writes invalidate without bound.
        if add.picked_unbox.is_some() {
            let terminator = unsafe { (*add.map).terminator() };
            let term = unsafe { (*terminator).as_terminator() as *const _ };
            walker_pin_terminator_allow_unboxing(ctx, op_pc, term)?;
        }
        let new_map_const = ctx.trace_ctx.const_int(add.new_map as i64);
        match value_pin {
            StoreAttrAddValuePin::Boxed => {
                crate::helpers::emit_mapdict_add_attr_inline(
                    ctx.trace_ctx,
                    obj,
                    add.storageindex,
                    new_map_const,
                    value,
                );
            }
            StoreAttrAddValuePin::UnboxedInt(canonical) => {
                // Only an exactly-`int` runtime value may unbox: the tag
                // guard pins a tagged operand, the `w_class` pin a heap one
                // (a subclass shares `W_IntObject`'s `ob_type`).
                let int_type_addr = &pyre_object::pyobject::INT_TYPE as *const _ as i64;
                let raw = walker_unbox_int(ctx, op_pc, value, int_type_addr)?;
                if let Some(canonical) = canonical {
                    walker_guard_exact_w_class(ctx, op_pc, value, canonical)?;
                }
                crate::helpers::emit_mapdict_add_unboxed_attr_inline(
                    ctx.trace_ctx,
                    obj,
                    add.storageindex,
                    new_map_const,
                    raw,
                );
            }
        }
        // The walk is the authoritative execution path, so apply the resolved
        // transition now; the emitted operations reproduce it in compiled code.
        unsafe {
            pyre_interpreter::objspace::std::mapdict::store_attr_add_commit(
                concrete_obj,
                &add,
                concrete_value,
            )
        };
        return Ok(Some(WalkerStoreAttrSpecialization::Direct));
    }

    let Some((w_type, version_tag, map, storageindex, attr)) = (unsafe {
        pyre_interpreter::objspace::std::mapdict::store_attr_boxed_fast_path(concrete_obj, &name)
    }) else {
        return Ok(None);
    };
    walker_guard_mapdict_instance_shape(ctx, op_pc, obj, concrete_obj, w_type, version_tag, map)?;
    unsafe { pyre_interpreter::objspace::std::mapdict::mark_attr_ever_mutated(attr) };
    let storageindex_const = ctx.trace_ctx.const_int(storageindex as i64);
    let helper = ctx
        .trace_ctx
        .const_int(crate::helpers::jit_mapdict_boxed_write as *const () as usize as i64);

    // Unlike the unboxed arm, this write stores a GC reference, so the
    // residual's original may-force effect is kept: only the opaque `setattr_fn`
    // MRO walk is replaced by the direct slot write, while the force token, the
    // virtualizable spill, and the trailing force/exception guards stay exactly
    // as the generic setattr emitted them.
    let mut effect = original_effect.clone();
    effect.runtime_helper = majit_ir::RuntimeHelperKind::StoreAttr;
    let descr = majit_metainterp::make_call_descr_with_effect(
        &[
            majit_ir::Type::Ref,
            majit_ir::Type::Int,
            majit_ir::Type::Ref,
        ],
        majit_ir::Type::Void,
        effect,
    );
    // ABI order follows `jit_mapdict_boxed_write`: receiver, guarded green
    // storage index, and the original symbolic object reference.  The value is
    // neither unboxed nor guarded by type.
    Ok(Some(WalkerStoreAttrSpecialization::Residual(
        descr,
        vec![helper, obj, storageindex_const, value],
    )))
}

/// #171: FBW virtualization of a non-escaping BUILD_LIST.
/// `lower_tuple_build_hlop_to_insn` lowers BUILD_LIST to `new_array_clear`
/// + per-index `setarrayitem_gc` + a `newlist_from_array` residual
/// (oopspec [`majit_ir::RuntimeHelperKind::NewlistFromArray`]) whose single
/// r-arg is the already-built backing array.  Decompose that residual into
/// the virtualizable `opimpl_newlist` shape (`pyjitpl.py`) —
/// `new_with_vtable` + `new_array` + `setarrayitem_gc` + `setfield_gc` —
/// so the optimizer folds the whole list (wrapper + block) when it never
/// escapes and the array build + residual DCE.
///
/// The element boxes are recovered from the backing array (its const length
/// from `heapcache.arraylen`, then per-index element shadows via
/// `heapcache_getarrayitem`), NOT from residual args.  The storage strategy
/// is chosen from the concrete element shadows exactly as
/// `list_strategy_for` / `w_list_new` does at runtime, so the traced object
/// matches the strategy the blackhole rebuilds on deopt:
///   * `list_strategy_for` → Integer AND every element an exact
///     `W_IntObject` → Integer (`int_items` typed block, elements unboxed
///     via `walker_unbox_int`);
///   * → Float → Float (`float_items` typed block, strict `W_FloatObject`
///     elements only, so exact-type by construction);
///   * → Object → Object (boxed refs into an `ItemsBlock`).
///
/// Returns `Ok(None)` to fall through to the opaque residual (always
/// byte-correct) for any shape it cannot reproduce faithfully: empty list
/// (Empty strategy), a non-const / unrecoverable array length, an element
/// without a concrete Ref shadow, or an Integer-strategy list that carries a
/// tagged immediate (which has no `&INT_TYPE`/`&LONG_TYPE` header for the
/// unbox guard). A fits-in-word `W_LongObject` is accepted: `is_plain_int1`
/// covers it and `walker_unbox_long` supplies the `&LONG_TYPE` + `_fits_int`
/// guarded extraction.
pub(crate) fn try_walker_specialize_newlist<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    r_args: &[OpRef],
    dst: usize,
    dst_bank: char,
) -> Result<Option<()>, DispatchError> {
    if !ctx.is_authoritative_executor || dst_bank != 'r' {
        return Ok(None);
    }
    if r_args.len() != 1 {
        return Ok(None);
    }
    let arr = r_args[0];

    // Const backing-array length (`new_array_clear(Const(len))` seeded
    // `heapcache.arraylen` — a cleared array has every slot set, so read the
    // length directly rather than probing getarrayitem until a miss).  Empty
    // list → Empty strategy: decline (the residual reproduces it).
    let len = {
        let Some(len_op) = ctx.trace_ctx.heap_cache().arraylen(arr) else {
            return Ok(None);
        };
        match len_op.inline_const_to_value() {
            Some(majit_ir::Value::Int(n)) if n >= 1 => n as usize,
            _ => return Ok(None),
        }
    };

    // Recover the element boxes from the array heap-cache (the values the
    // BUILD_LIST `setarrayitem_gc` ops stored); a cache miss (clobbered array)
    // bails to the opaque residual.
    let descr_idx = crate::state::pyobject_gcarray_descr().index();
    let mut items: Vec<OpRef> = Vec::with_capacity(len);
    for i in 0..len {
        let Some(elem) =
            ctx.trace_ctx
                .heapcache_getarrayitem(arr, OpRef::ConstInt(i as i64), descr_idx)
        else {
            return Ok(None);
        };
        items.push(elem);
    }

    // Concrete element objects (needed to classify the strategy and extract
    // the payloads before any allocation).  An element without a concrete Ref
    // shadow declines to the residual.
    let mut concretes: Vec<pyre_object::PyObjectRef> = Vec::with_capacity(len);
    for &it in &items {
        let Some(obj) = walker_concrete_ref_object(ctx, it) else {
            return Ok(None);
        };
        concretes.push(obj);
    }

    // Strategy the runtime `w_list_new` would pick — the source of truth for
    // the concrete shadow, so the traced storage matches on deopt.
    let strategy = pyre_object::listobject::list_strategy_for(&concretes);
    use pyre_object::listobject::ListStrategy;

    // Pre-extract the machine payloads BEFORE `build_list_from_refs` allocates
    // (a minor collection there could move the boxed elements, so the raw
    // pointers must not be dereferenced afterwards).
    enum Emit {
        // Per element: `(unboxed i64, is_fits_long)`.  `is_fits_long` selects
        // `walker_unbox_long` (`&LONG_TYPE` + `_fits_int` guard) over the
        // plain `walker_unbox_int`.
        Int(Vec<(i64, bool)>),
        Float(Vec<f64>),
        Object,
    }
    let int_ty = &pyre_object::pyobject::INT_TYPE as *const pyre_object::pyobject::PyType;
    let emit = match strategy {
        ListStrategy::Integer => {
            // `IntegerListStrategy.is_correct_type` is `is_plain_int1`, which
            // accepts an exact `W_IntObject` or a fits-in-word `W_LongObject`;
            // both store the unboxed i64 (`plain_int_w`). A tagged immediate
            // has no header for the unbox guard, so decline it to the residual
            // (correct for any element).
            let mut vals = Vec::with_capacity(len);
            for &p in &concretes {
                if pyre_object::tagged_int::CAN_BE_TAGGED
                    && pyre_object::tagged_int::is_tagged_int(p)
                {
                    return Ok(None);
                }
                if !unsafe { pyre_object::is_plain_int1(p) } {
                    return Ok(None);
                }
                let is_fits_long = unsafe { pyre_object::pyobject::is_long(p) };
                let val = if is_fits_long {
                    pyre_object::longobject::jit_w_long_toint(p as usize as i64)
                } else {
                    unsafe { pyre_object::w_int_get_value(p) }
                };
                vals.push((val, is_fits_long));
            }
            Emit::Int(vals)
        }
        ListStrategy::Float => {
            // `list_strategy_for` admits only exact, non-NaN floats here.  Its
            // subclass term is enforced on replay by pinning each element's
            // `w_class`; `is_plain_float_strict` also admits the null spelling
            // of "exact float", which no pin can express, so decline such an
            // element rather than emit a guard it would fail itself.
            let mut vals = Vec::with_capacity(len);
            for &p in &concretes {
                if unsafe { walker_exact_builtin_class(p) }.is_none() {
                    return Ok(None);
                }
                vals.push(unsafe { pyre_object::w_float_get_value(p) });
            }
            Emit::Float(vals)
        }
        ListStrategy::Object => Emit::Object,
        // The interpreter stores this as encoded signed-longlong values.
        // The walker does not yet have an encoded numeric payload variant;
        // leave construction to the ordinary residual instead of emitting an
        // Integer array whose values would have the wrong representation.
        ListStrategy::IntOrFloat => return Ok(None),
        // The generic residual constructs the erased rpython-string array.
        // The walker has no BytesBlock payload emitter yet.
        ListStrategy::Bytes => return Ok(None),
        // The generic residual constructs AsciiListStrategy's erased UTF-8
        // storage; the walker has no raw UnicodeValueStorage emitter yet.
        ListStrategy::Ascii => return Ok(None),
        // Empty is impossible here (len >= 1); decline defensively. Range
        // storage is built only by the interpreter-internal `make_range_list`
        // seam and has no walker-native erased-tuple emitter yet.
        ListStrategy::Empty
        | ListStrategy::Size
        | ListStrategy::SimpleRange
        | ListStrategy::Range => return Ok(None),
    };

    // Concrete shadow: a fresh list built from the element shadows
    // (`w_list_new` parity — picks the same strategy). A new allocation with
    // no heap mutation, safe during the walk like `wrapint`.
    let result_concrete = pyre_interpreter::build_list_from_refs(&concretes);
    if result_concrete.is_null() {
        return Ok(None);
    }

    // --- emit the virtualizable decomposed newlist (walker-native) ---
    let list_op = match emit {
        Emit::Int(vals) => {
            let int_type_addr = int_ty as i64;
            let long_type_addr = &pyre_object::pyobject::LONG_TYPE as *const _ as i64;
            let mut raws: Vec<OpRef> = Vec::with_capacity(len);
            for (&it, &(v, is_fits_long)) in items.iter().zip(vals.iter()) {
                let raw = if is_fits_long {
                    walker_unbox_long(ctx, op_pc, it, long_type_addr)?
                } else {
                    walker_unbox_int(ctx, op_pc, it, int_type_addr)?
                };
                // The unbox proves `ob_type`, which a subclass shares; without
                // the `w_class` pin the element is rewrapped as a plain int.
                if let Some(obj) = walker_concrete_ref_object(ctx, it) {
                    walker_guard_exact_w_class(ctx, op_pc, it, walker_numeric_builtin_class(obj))?;
                }
                ctx.trace_ctx
                    .set_opref_concrete(raw, majit_ir::Value::Int(v));
                raws.push(raw);
            }
            crate::helpers::emit_typed_list_inline(
                &mut *ctx.trace_ctx,
                &raws,
                crate::state::int_gcarray_descr(),
                crate::descr::list_int_items_len_descr(),
                crate::descr::list_int_items_block_descr(),
                ListStrategy::Integer,
            )
        }
        Emit::Float(vals) => {
            let float_type_addr = &pyre_object::pyobject::FLOAT_TYPE as *const _ as i64;
            let mut raws: Vec<OpRef> = Vec::with_capacity(len);
            for (&it, &v) in items.iter().zip(vals.iter()) {
                let raw = walker_unbox_float(ctx, op_pc, it, float_type_addr)?;
                if let Some(obj) = walker_concrete_ref_object(ctx, it) {
                    walker_guard_exact_w_class(ctx, op_pc, it, walker_numeric_builtin_class(obj))?;
                }
                ctx.trace_ctx
                    .set_opref_concrete(raw, majit_ir::Value::Float(v));
                // `walker_unbox_float` guards `ob_type` only, which a float
                // SUBCLASS instance shares; pin `w_class` so it side-exits
                // instead of being unboxed into Float storage the interpreter
                // would have declined (`all_floats` is strict).
                walker_guard_exact_w_class(
                    ctx,
                    op_pc,
                    it,
                    pyre_object::pyobject::get_instantiate(&pyre_object::pyobject::FLOAT_TYPE),
                )?;
                walker_guard_float_not_nan(ctx, op_pc, raw)?;
                raws.push(raw);
            }
            crate::helpers::emit_typed_list_inline(
                &mut *ctx.trace_ctx,
                &raws,
                crate::state::float_gcarray_descr(),
                crate::descr::list_float_items_len_descr(),
                crate::descr::list_float_items_block_descr(),
                ListStrategy::Float,
            )
        }
        Emit::Object => crate::helpers::emit_object_list_inline(&mut *ctx.trace_ctx, &items),
    };

    ctx.trace_ctx.set_opref_concrete(
        list_op,
        majit_ir::Value::Ref(majit_ir::GcRef(result_concrete as usize)),
    );
    write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, list_op)?;
    Ok(Some(()))
}

/// FBW virtualization of the array-backed BUILD_TUPLE — the arities
/// `makespecialisedtuple2` does not claim.  Sibling of
/// [`try_walker_specialize_newtuple`] (arity-2 plain-int `spec_ii`) and
/// [`try_walker_specialize_newlist`], reached only after the `spec_ii` fold
/// declines, so that path stays byte-identical.
///
/// `lower_tuple_build_hlop_to_insn` lowers BUILD_TUPLE to `new_array_clear` +
/// per-index `setarrayitem_gc` + a `newtuple_from_array` residual.  Re-emit the
/// canonical `W_TupleObject` shape walker-native (`new_with_vtable` +
/// `w_class` / `wrappeditems` `setfield_gc` over a fresh items block), reading
/// the elements straight out of the array heap-cache so the array build keeps
/// no consumer and DCEs.  A tuple that never escapes then folds away entirely,
/// and one that does escape materializes from the same fields the residual
/// would have written.
///
/// Arity 2 is `makespecialisedtuple2` territory (`Cls_ii` / `Cls_ff` /
/// `Cls_oo`, `specialisedtupleobject.py`): the runtime never builds an
/// array-backed tuple there, so emitting one would diverge from what the
/// blackhole rebuilds on deopt.  Declined here — the `spec_ii` fold owns the
/// int-int case and the residual owns the rest.  The empty tuple is declined
/// too (no element to recover a length from).
///
/// Lifting that decline is not a trace-local question: the trace stays
/// self-consistent, but a side exit hands a real pair — inline `value0` /
/// `value1`, no `wrappeditems` block — to whatever consumer the trace picked
/// for the canonical layout, and
/// [`try_walker_orthodox_subscr_tuple_item`] then reads a field that is
/// not there.
///
/// Returns `Ok(Some(()))` when folded; `Ok(None)` falls through to the opaque
/// residual, which stays correct for any shape — a non-const array length or
/// an element without a concrete Ref shadow is not declined, just not folded.
pub(crate) fn try_walker_specialize_newtuple_object<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    r_args: &[OpRef],
    dst: usize,
    dst_bank: char,
) -> Result<Option<()>, DispatchError> {
    if !ctx.is_authoritative_executor || dst_bank != 'r' {
        return Ok(None);
    }
    if r_args.len() != 1 {
        return Ok(None);
    }
    let arr = r_args[0];
    // Const backing-array length (`new_array_clear(Const(len))` seeded
    // `heapcache.arraylen`; a cleared array has every slot set, so read the
    // length directly rather than probing getarrayitem until a miss).
    let len = {
        let Some(len_op) = ctx.trace_ctx.heap_cache().arraylen(arr) else {
            return Ok(None);
        };
        match len_op.inline_const_to_value() {
            Some(majit_ir::Value::Int(n)) if n >= 1 => n as usize,
            _ => return Ok(None),
        }
    };
    if len == 2 {
        return Ok(None);
    }

    // Element boxes the BUILD_TUPLE `setarrayitem_gc` ops stored; a cache miss
    // (clobbered array / non-const index) bails to the opaque residual.
    let descr_idx = crate::state::pyobject_gcarray_descr().index();
    let mut items: Vec<OpRef> = Vec::with_capacity(len);
    for i in 0..len {
        let Some(elem) =
            ctx.trace_ctx
                .heapcache_getarrayitem(arr, OpRef::ConstInt(i as i64), descr_idx)
        else {
            return Ok(None);
        };
        items.push(elem);
    }
    let mut concretes: Vec<pyre_object::PyObjectRef> = Vec::with_capacity(len);
    for &it in &items {
        let Some(obj) = walker_concrete_ref_object(ctx, it) else {
            return Ok(None);
        };
        concretes.push(obj);
    }

    // Concrete shadow: a fresh array-backed tuple from the element shadows
    // (`w_tuple_new` parity for every arity but 2). A new allocation with no
    // heap mutation, safe during the walk like `wrapint`.  Built before the
    // emit so a failure leaves no orphan ops in the trace.
    let result_concrete = pyre_object::w_tuple_new_array_backed(concretes);
    if result_concrete.is_null() {
        return Ok(None);
    }

    let tuple_op = crate::helpers::emit_object_tuple_inline(ctx.trace_ctx, &items);
    ctx.trace_ctx.set_opref_concrete(
        tuple_op,
        majit_ir::Value::Ref(majit_ir::GcRef(result_concrete as usize)),
    );
    write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, tuple_op)?;
    Ok(Some(()))
}

/// #195 / #73: FBW virtualization of an arity-2 plain-int BUILD_TUPLE.
/// `lower_tuple_build_hlop_to_insn` lowers BUILD_TUPLE to `new_array_clear`
/// + per-index `setarrayitem_gc` + a `newtuple_from_array` residual
/// (oopspec [`majit_ir::RuntimeHelperKind::NewtupleFromArray`]).  When both
/// backing-array elements are concrete plain `W_IntObject`, re-emit the
/// former trait-side spec_ii shape walker-native
/// (`new_with_vtable` + `w_class` / `value0` / `value1` `setfield_gc`),
/// reading the elements straight out of the array heap-cache so the array
/// build keeps no consumer and DCEs.  The partner
/// [`try_walker_specialize_unpack`] then folds the `value0` / `value1`
/// reads off the virtual tuple, collapsing build→unpack to a pure-int loop.
///
/// Returns `Ok(Some(()))` when folded (the caller returns `Continue`);
/// `Ok(None)` to fall through to the opaque residual, which stays correct
/// for any other shape (object tuple, arity ≠ 2, out-of-range long, tagged
/// immediate, cache miss) — so a non-foldable build is not declined.
pub(crate) fn try_walker_specialize_newtuple<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    r_args: &[OpRef],
    dst: usize,
    dst_bank: char,
) -> Result<Option<()>, DispatchError> {
    if !ctx.is_authoritative_executor || dst_bank != 'r' {
        return Ok(None);
    }
    if r_args.len() != 1 {
        return Ok(None);
    }
    let arr = r_args[0];
    // Read the two backing-array element boxes out of the heap-cache (the
    // values the BUILD_TUPLE `setarrayitem_gc` ops stored); a cache miss
    // (non-const index / clobbered array) bails to the opaque residual.
    let descr_idx = crate::state::pyobject_gcarray_descr().index();
    let (Some(e0), Some(e1)) = (
        ctx.trace_ctx
            .heapcache_getarrayitem(arr, OpRef::ConstInt(0), descr_idx),
        ctx.trace_ctx
            .heapcache_getarrayitem(arr, OpRef::ConstInt(1), descr_idx),
    ) else {
        return Ok(None);
    };
    // Arity must be exactly 2 (the only specialised int tuple).  A BUILD_TUPLE
    // array sets every index before `newtuple_from_array`, so a cached element
    // at index 2 means arity ≥ 3 → fall through to the residual (a wrongly
    // built arity-2 spec_ii would length-mismatch the arity-N unpack).
    if ctx
        .trace_ctx
        .heapcache_getarrayitem(arr, OpRef::ConstInt(2), descr_idx)
        .is_some()
    {
        return Ok(None);
    }
    let (Some(c0), Some(c1)) = (
        walker_concrete_ref_object(ctx, e0),
        walker_concrete_ref_object(ctx, e1),
    ) else {
        return Ok(None);
    };
    // The arity-2 int specialised tuple `Cls_ii` (`makespecialisedtuple2`,
    // specialisedtupleobject.py) is built when both elements pass
    // `is_plain_int1` — an exact `W_IntObject` or a fits-in-word
    // `W_LongObject`; the stored payload is `plain_int_w` of each.  A tagged
    // immediate has no real header for the unbox guard and the emit is not
    // tag-aware, so decline it to the residual (correct for any shape).
    if pyre_object::tagged_int::CAN_BE_TAGGED
        && (pyre_object::tagged_int::is_tagged_int(c0)
            || pyre_object::tagged_int::is_tagged_int(c1))
    {
        return Ok(None);
    }
    let int_ty = &pyre_object::pyobject::INT_TYPE as *const pyre_object::pyobject::PyType;
    let both_plain_int =
        unsafe { pyre_object::is_plain_int1(c0) && pyre_object::is_plain_int1(c1) };
    if !both_plain_int {
        return Ok(None);
    }
    let c0_long = unsafe { pyre_object::pyobject::is_long(c0) };
    let c1_long = unsafe { pyre_object::pyobject::is_long(c1) };
    // Concrete element int payloads (`plain_int_w`: `W_IntObject`'s `intval`
    // or a fits-int `W_LongObject`'s `toint()`).
    let v0 = if c0_long {
        pyre_object::longobject::jit_w_long_toint(c0 as usize as i64)
    } else {
        unsafe { pyre_object::w_int_get_value(c0) }
    };
    let v1 = if c1_long {
        pyre_object::longobject::jit_w_long_toint(c1 as usize as i64)
    } else {
        unsafe { pyre_object::w_int_get_value(c1) }
    };

    // --- emit the virtual spec_ii walker-native ---
    // Paired `w_class` guard per element so a runtime int subclass sharing
    // the public `int` `w_class` side-exits, then the plain-int payload unbox.
    // A fits-int `W_LongObject` also carries the public `int` `w_class`
    // (`is_plain_int1`), so the same guard covers it; the payload extraction
    // switches to `walker_unbox_long` (`&LONG_TYPE` + `_fits_int`).
    let int_typeobj = pyre_object::pyobject::get_instantiate(&pyre_object::pyobject::INT_TYPE);
    walker_guard_exact_w_class(ctx, op_pc, e0, int_typeobj)?;
    walker_guard_exact_w_class(ctx, op_pc, e1, int_typeobj)?;
    let int_type_addr = int_ty as i64;
    let long_type_addr = &pyre_object::pyobject::LONG_TYPE as *const _ as i64;
    let raw0 = if c0_long {
        walker_unbox_long(ctx, op_pc, e0, long_type_addr)?
    } else {
        walker_unbox_int_typed(
            ctx,
            op_pc,
            e0,
            int_type_addr,
            crate::descr::int_intval_descr(),
        )?
    };
    let raw1 = if c1_long {
        walker_unbox_long(ctx, op_pc, e1, long_type_addr)?
    } else {
        walker_unbox_int_typed(
            ctx,
            op_pc,
            e1,
            int_type_addr,
            crate::descr::int_intval_descr(),
        )?
    };

    let tuple = walker_emit_specialised_tuple_ii(ctx, op_pc, raw0, raw1, v0, v1)?;
    write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, tuple)?;
    Ok(Some(()))
}

/// Emit the virtual `Cls_ii` arity-2 int tuple (`makespecialisedtuple2`,
/// specialisedtupleobject.py) walker-native: `new_with_vtable` + the `w_class`
/// / `value0` / `value1` `setfield_gc`s over the two raw int payloads.
///
/// `v0` / `v1` are the recorded payloads; they build the concrete shadow the
/// partner unpack fold reads back (`walker_concrete_ref_object` +
/// `unpack_item_fn`).  The shadow is constructed LAST so the construct→root
/// window holds no intervening runtime allocation: stamping `tuple`'s concrete
/// roots the fresh spec_ii via the trace's concrete-shadow set. A concrete
/// allocation failure aborts the whole walk because the virtual allocation
/// and stores have already been recorded.
fn walker_emit_specialised_tuple_ii<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    raw0: OpRef,
    raw1: OpRef,
    v0: i64,
    v1: i64,
) -> Result<OpRef, DispatchError> {
    let tuple = ctx.trace_ctx.record_op_with_descr(
        OpCode::NewWithVtable,
        &[],
        crate::descr::specialised_tuple_ii_size_descr(),
    );
    ctx.trace_ctx.heap_cache_mut().new_object(tuple);
    crate::helpers::emit_tuple_hash_sentinel(
        ctx.trace_ctx,
        tuple,
        crate::descr::specialised_tuple_ii_hash_descr(),
    );
    // `ob_type` is the JIT vtable; Python-level `type()` reads `w_class`,
    // which all specialised tuple variants share at the public `tuple`
    // typedef.
    let tuple_w_class = pyre_object::pyobject::get_instantiate(&pyre_object::pyobject::TUPLE_TYPE);
    if !tuple_w_class.is_null() {
        let wc = ctx.trace_ctx.const_ref(tuple_w_class as i64);
        ctx.trace_ctx.record_op_with_descr(
            OpCode::SetfieldGc,
            &[tuple, wc],
            crate::descr::specialised_tuple_ii_w_class_descr(),
        );
        ctx.trace_ctx.heapcache_setfield_cached(
            tuple,
            crate::descr::specialised_tuple_ii_w_class_descr().index(),
            wc,
        );
    }
    for (raw, descr) in [
        (raw0, crate::descr::specialised_tuple_ii_value0_descr()),
        (raw1, crate::descr::specialised_tuple_ii_value1_descr()),
    ] {
        let descr_index = descr.index();
        ctx.trace_ctx
            .record_op_with_descr(OpCode::SetfieldGc, &[tuple, raw], descr);
        ctx.trace_ctx
            .heapcache_setfield_cached(tuple, descr_index, raw);
    }
    let tuple_ptr = pyre_object::specialisedtupleobject::w_specialised_tuple_ii_new(v0, v1);
    if tuple_ptr.is_null() {
        return Err(DispatchError::ConcreteShadowAllocationFailed { pc: op_pc });
    }
    ctx.trace_ctx.set_opref_concrete(
        tuple,
        majit_ir::Value::Ref(majit_ir::GcRef(tuple_ptr as usize)),
    );
    Ok(tuple)
}

/// #57 SLICE 3b: walker-native speculative int specialization for the
/// COMPARE_OP helper residual_call (oopspec `CompareOp`).  Emits
/// `guard_class` + `getfield_gc_i` per operand + `int_<cmp>` for the raw
/// truth, then boxes it to a `W_Bool`.  NON-fused: the walker sees
/// COMPARE_OP and the following `goto_if_not` as separate JitCode ops, so
/// it always materializes the boxed bool the generic `compare_fn` would
/// have produced (the retired MIFrame compare/jump fusion does not apply).
///
/// Same gate + return contract as
/// [`try_walker_specialize_binary_op_int`].
pub(crate) fn try_walker_specialize_compare_op_int<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    op_tag: i64,
    r_args: &[OpRef],
    allboxes: &[OpRef],
    call_descr: &dyn majit_ir::descr::CallDescr,
    dst: usize,
    dst_bank: char,
) -> Result<Option<()>, DispatchError> {
    if !ctx.is_authoritative_executor || dst_bank != 'r' {
        return Ok(None);
    }
    let Some(cmp_op) = pyre_interpreter::runtime_ops::compare_op_from_tag(op_tag) else {
        return Ok(None);
    };
    use pyre_interpreter::bytecode::ComparisonOperator;
    let cmp = match cmp_op {
        ComparisonOperator::Less => OpCode::IntLt,
        ComparisonOperator::LessOrEqual => OpCode::IntLe,
        ComparisonOperator::Greater => OpCode::IntGt,
        ComparisonOperator::GreaterOrEqual => OpCode::IntGe,
        ComparisonOperator::Equal => OpCode::IntEq,
        ComparisonOperator::NotEqual => OpCode::IntNe,
    };
    let Some((lhs, rhs, lhs_obj, rhs_obj, la, rb, boxed_result_i64)) =
        walker_int_specialization_operands(ctx, r_args, allboxes, call_descr)
    else {
        return Ok(None);
    };

    // --- emit the specialized IR (walker-native) ---
    // bool and int share `intval`; guard each operand against its own vtable
    // so a bool comparand unboxes through &BOOL_TYPE.  The comparison result
    // is a bool either way.
    let (lhs_type, lhs_descr) = crate::state::int_or_bool_unbox_type_descr(lhs_obj);
    let (rhs_type, rhs_descr) = crate::state::int_or_bool_unbox_type_descr(rhs_obj);
    let lhs_raw = walker_unbox_int_typed(ctx, op_pc, lhs, lhs_type, lhs_descr)?;
    // `walker_unbox_int_typed` proves only `ob_type`, which an `int` subclass
    // shares with `int`; the operand gate reads `w_class`, which it does not.
    // Without these the compiled guard admits the subclass and the comparison
    // is answered by `IntLt` instead of the overriding `__lt__`.
    walker_guard_exact_w_class(ctx, op_pc, lhs, walker_numeric_builtin_class(lhs_obj))?;
    let rhs_raw = walker_unbox_int_typed(ctx, op_pc, rhs, rhs_type, rhs_descr)?;
    walker_guard_exact_w_class(ctx, op_pc, rhs, walker_numeric_builtin_class(rhs_obj))?;
    let truth = ctx.trace_ctx.record_op(cmp, &[lhs_raw, rhs_raw]);
    let folded = majit_metainterp::eval_binop_i(cmp, la, rb);
    ctx.trace_ctx
        .set_opref_concrete(truth, majit_ir::Value::Int(folded));
    // `space.newbool` on the truth: its guard plus the prebuilt singleton
    // (`baseobjspace.py:895-900`).  The box the generic `compare_fn` residual
    // would have landed in the dst Ref register never exists, and the
    // `goto_if_not` that reads it sees a constant.  The residual box below is
    // the no-snapshot fallback only.
    let boxed = match walker_newbool_guarded(ctx, op_pc, truth, folded != 0, dst_bank)? {
        Some(boxed) => boxed,
        None => {
            let boxed =
                crate::helpers::emit_trace_bool_value_from_truth(ctx.trace_ctx, truth, false);
            ctx.trace_ctx.set_opref_concrete(
                boxed,
                majit_ir::Value::Ref(majit_ir::GcRef(boxed_result_i64 as usize)),
            );
            boxed
        }
    };
    write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, boxed)?;
    Ok(Some(()))
}

/// Walker-native fold of the CHECK_EXC_MATCH
/// residual (`bh_compare_fn(exc, match_type, op_tag=10)`,
/// `call_jit.rs`). Computes the match concretely from
/// `type(exc)` and `match_type` and emit a `const_ref` of the immortal
/// TRUE/FALSE bool singleton, eliding the opaque may-force compare (and, since
/// that singleton is a constant, the immediately-following `is_true`
/// truth-extract residual).  With the exception's constructor + raise
/// already virtualized by their own folds, folding the match to a constant
/// lets the whole exception de-escape and DCE.
///
/// Soundness — the fold result depends only on `(type(exc), match_type)`:
///   * `exc` (`r_args[0]`) is the in-trace inline-built virtual exception
///     whose kind/vtable are baked into the `NewWithVtable`, so its class
///     cannot differ at runtime — no guard needed.  (A `GuardClass` is
///     emitted defensively when the heapcache does not already know its
///     class, e.g. a non-construct-fold exc reaching here.)
///   * `match_type` (`r_args[1]`) is a runtime value (typically a
///     `LOAD_GLOBAL` of the handler class), so a `GuardValue` pins its
///     identity — a reassigned handler global side-exits and re-traces
///     instead of running the wrong handler.  (Stricter than the trait,
///     which elides this guard.)
///
/// Declines (`None` → generic residual) when either operand lacks a
/// concrete shadow, or `match_type` is not a valid exception class /
/// tuple (the residual then raises the correct `TypeError`).
/// Pin the elements of a tuple `except` clause's match target, returning
/// whether the target was a tuple layout this could read through.
///
/// `w_tuple_new` picks the layout by arity: two object elements become a
/// `W_SpecialisedTupleObject_oo` holding them in inline immutable `value0` /
/// `value1` slots, everything else an array-backed `W_TupleObject` behind
/// `wrappeditems`. Both are read here with the same guarded-load shape the
/// other tuple folds use, and each element is pinned to the class seen while
/// tracing. The `_oo` loads are pure (immutable fields), so a tuple built from
/// constants leaves nothing behind after optimization.
///
/// A match target that is neither layout — a bare class, the int/float
/// specialisations, or a tuple subclass whose `w_class` diverges — returns
/// `false` so the caller pins the object identity instead. That is correct for
/// a target that is loaded rather than built, which is the only case where the
/// identity can hold across iterations.
fn walker_guard_exc_match_tuple_items<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    match_op: OpRef,
    match_type: pyre_object::PyObjectRef,
) -> Result<bool, DispatchError> {
    let ob_type = unsafe { (*(match_type as *const pyre_object::pyobject::PyObject)).ob_type };
    let spec_oo = &pyre_object::specialisedtupleobject::SPECIALISED_TUPLE_OO_TYPE
        as *const pyre_object::pyobject::PyType;
    let tuple_type = &pyre_object::TUPLE_TYPE as *const pyre_object::pyobject::PyType;

    // Either layout may carry a subclass `w_class`, and the element reads below
    // are only the whole target when it is a plain tuple.
    let canonical_tuple_class = pyre_object::get_instantiate(&pyre_object::TUPLE_TYPE);
    if !std::ptr::eq(
        unsafe { (*(match_type as *const pyre_object::pyobject::PyObject)).w_class },
        canonical_tuple_class,
    ) {
        return Ok(false);
    }

    let mut items: Vec<(OpRef, pyre_object::PyObjectRef)> = Vec::new();
    if std::ptr::eq(ob_type, spec_oo) {
        walker_guard_exc_match_tuple_class(ctx, op_pc, match_op, spec_oo as i64)?;
        for index in 0..2usize {
            let descr = if index == 0 {
                crate::descr::specialised_tuple_oo_value0_descr()
            } else {
                crate::descr::specialised_tuple_oo_value1_descr()
            };
            let item = crate::state::opimpl_getfield_gc_r(ctx.trace_ctx, match_op, descr);
            let concrete = unsafe {
                pyre_object::specialisedtupleobject::w_specialised_tuple_oo_getvalue(
                    match_type, index,
                )
            };
            items.push((item, concrete));
        }
    } else if std::ptr::eq(ob_type, tuple_type) {
        // Read every element before recording anything: a bail-out after a
        // guard has been emitted would leave the target's class pinned, and the
        // caller reads that as "already guarded" and drops its own pin.
        let len = unsafe { pyre_object::w_tuple_len(match_type) };
        let mut concretes: Vec<pyre_object::PyObjectRef> = Vec::with_capacity(len);
        for index in 0..len {
            let Some(concrete) =
                (unsafe { pyre_object::w_tuple_getitem(match_type, index as i64) })
            else {
                return Ok(false);
            };
            concretes.push(concrete);
        }
        walker_guard_exc_match_tuple_class(ctx, op_pc, match_op, tuple_type as i64)?;
        walker_guard_exact_w_class(ctx, op_pc, match_op, canonical_tuple_class)?;
        let block = crate::state::opimpl_getfield_gc_r(
            ctx.trace_ctx,
            match_op,
            crate::descr::tuple_wrappeditems_descr(),
        );
        let length = crate::state::opimpl_arraylen_gc(
            ctx.trace_ctx,
            block,
            crate::state::pyobject_gcarray_descr(),
        );
        let len_const = ctx.trace_ctx.const_int(len as i64);
        walker_emit_fold_guard_with_snapshot(ctx, op_pc, OpCode::GuardValue, &[length, len_const])?;
        for (index, concrete) in concretes.into_iter().enumerate() {
            let index_op = ctx.trace_ctx.const_int(index as i64);
            let item =
                crate::state::trace_items_block_getitem_value(ctx.trace_ctx, block, index_op);
            items.push((item, concrete));
        }
    } else {
        return Ok(false);
    }

    for (item, concrete) in items {
        if item.is_constant() {
            continue;
        }
        let expected = ctx.trace_ctx.const_ref(concrete as i64);
        walker_emit_fold_guard_with_snapshot(ctx, op_pc, OpCode::GuardValue, &[item, expected])?;
        ctx.trace_ctx.heap_cache_mut().replace_box(item, expected);
    }
    Ok(true)
}

/// `GuardClass` on a match target's layout, skipped when the class is already
/// pinned.
fn walker_guard_exc_match_tuple_class<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    match_op: OpRef,
    type_addr: i64,
) -> Result<(), DispatchError> {
    if ctx.trace_ctx.heap_cache().is_class_known(match_op) {
        return Ok(());
    }
    let type_const = ctx.trace_ctx.const_int(type_addr);
    walker_emit_fold_guard_with_snapshot(ctx, op_pc, OpCode::GuardClass, &[match_op, type_const])?;
    ctx.trace_ctx
        .heap_cache_mut()
        .class_now_known(match_op, type_addr);
    Ok(())
}

pub(crate) fn try_walker_fold_check_exc_match<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    r_args: &[OpRef],
    dst: usize,
    dst_bank: char,
) -> Result<Option<()>, DispatchError> {
    if r_args.len() != 2 || dst_bank != 'r' {
        return Ok(None);
    }
    let exc_op = r_args[0];
    let match_op = r_args[1];
    let (Some(exc), Some(match_type)) = (
        walker_concrete_ref_object(ctx, exc_op),
        walker_concrete_ref_object(ctx, match_op),
    ) else {
        return Ok(None);
    };
    // `validate_check_exc_match_class` gates `except <non-exception>:`
    // (raising `TypeError`); on a validity error decline so the residual
    // reproduces the raise instead of baking a wrong bool into the trace.
    if pyre_interpreter::eval::validate_check_exc_match_class(match_type).is_err() {
        return Ok(None);
    }
    // The answer depends on `space.type(exc)`, and `typedef::type` reaches an
    // exception's class through the kind registry whenever the `w_class` slot
    // still holds the generic stub. The guard below can only pin a class the
    // slot itself holds, so decline the registry answer rather than emit a
    // guard that pins something else.
    let Some(exc_class) = pyre_interpreter::typedef::r#type(exc) else {
        return Ok(None);
    };
    let exc_class = exc_class.as_ptr();
    if !std::ptr::eq(unsafe { (*exc).w_class }, exc_class) {
        return Ok(None);
    }
    // `eval::check_exc_match_against` = `exception_match(type(exc), match)`
    // (eval.rs), walking the exception class MRO and accepting a tuple of
    // classes. Inlined here.
    let matched = pyre_interpreter::eval::check_exc_match_against(exc, match_type);

    // --- commit to the fold: emit IR (no further declines) ---
    // Pin `match_type` so a runtime divergence (a reassigned handler global)
    // side-exits rather than running the wrong handler.
    //
    // A tuple clause is pinned through its ELEMENTS. `except (A, B):` lowers to
    // `BUILD_TUPLE`, which allocates a fresh tuple on every visit, so an
    // identity guard on the container is unsatisfiable — it fails once per
    // visit and the loop collapses into side exits and bridge churn. The
    // elements are what the match actually reads and what a rebinding would
    // change, so guarding them is both sound and stable across the
    // re-allocation.
    //
    // A known class does not stand in for the pin: every class object shares
    // the one `type` layout, so `is_class_known` on the clause operand says
    // only that it is a class and leaves which class free to change.
    if !match_op.is_constant()
        && !walker_guard_exc_match_tuple_items(ctx, op_pc, match_op, match_type)?
    {
        let expected = ctx.trace_ctx.const_ref(match_type as i64);
        ctx.trace_ctx
            .record_guard(OpCode::GuardValue, &[match_op, expected], 0);
        walker_capture_snapshot_for_last_guard(ctx, op_pc)?;
        ctx.trace_ctx
            .heap_cache_mut()
            .replace_box(match_op, expected);
    }
    // Pin the exception's Python-level class, the value the match walked the
    // MRO of. `GuardClass` alone cannot do it: every exception of one
    // `ExcKind` carries the same `ob_type`, so `class A(Exception)` and
    // `class B(Exception)` share a layout and one's recorded answer would
    // replay for the other. The layout guard still comes first — it is what
    // makes the `w_class` read below name the field it was recorded against.
    if !exc_op.is_constant() {
        if !ctx.trace_ctx.heap_cache().is_class_known(exc_op) {
            let exc_layout =
                unsafe { (*(exc as *const pyre_object::pyobject::PyObject)).ob_type } as i64;
            let layout_const = ctx.trace_ctx.const_int(exc_layout);
            walker_emit_fold_guard_with_snapshot(
                ctx,
                op_pc,
                OpCode::GuardClass,
                &[exc_op, layout_const],
            )?;
            ctx.trace_ctx
                .heap_cache_mut()
                .class_now_known(exc_op, exc_layout);
        }
        let w_class_op =
            walker_record_getfield_gc_r_uncached(ctx, exc_op, crate::descr::w_class_descr());
        let expected = ctx.trace_ctx.const_ref(exc_class as i64);
        walker_emit_fold_guard_with_snapshot(
            ctx,
            op_pc,
            OpCode::GuardValue,
            &[w_class_op, expected],
        )?;
        ctx.trace_ctx
            .heap_cache_mut()
            .replace_box(w_class_op, expected);
    }

    // The match is a constant at trace time: emit the immortal bool singleton
    // as a `const_ref`.  The following `is_true` (the `except` clause's
    // `POP_JUMP_IF_FALSE`) reads a constant W_Bool, which
    // `try_walker_specialize_truth_bool` folds off its concrete.
    let result_obj = pyre_object::w_bool_from(matched);
    let const_bool = ctx.trace_ctx.const_ref(result_obj as i64);
    ctx.trace_ctx.set_opref_concrete(
        const_bool,
        majit_ir::Value::Ref(majit_ir::GcRef(result_obj as usize)),
    );
    write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, const_bool)?;
    Ok(Some(()))
}

/// Trace `space.newbool` as its directional truth guard and prebuilt result.
/// `baseobjspace.py:896-900` chooses `w_True` or `w_False`, the prebuilt
/// singletons from `boolobject.py:79-80`; `pyjitpl.py:511-534` records the
/// matching `GUARD_TRUE` / `GUARD_FALSE`, and `pyjitpl.py:525-526` replaces
/// the truth box with the promoted constant.
///
/// The guard is unconditional, because `newbool`'s `if b:` is: it is plain
/// RPython carrying no `@jit` hint (`baseobjspace.py:895` is only
/// `@signature`, and `boolobject.py` has none), so the tracer resolves it the
/// one way it observed and pins that with a guard no matter who consumes the
/// result.  This used to be restricted to a box that "decides one branch and
/// nothing else", on the reasoning that guarding an escaping bool pins a value
/// the trace would otherwise carry unconstrained.  Upstream grants no such
/// exemption, and the transform that does read like one —
/// `jtransform.py:196 optimize_goto_if_not` — is a different thing: it fuses a
/// compare into a block's exitswitch, and `:205-211` makes it *refuse* when the
/// boolean has any other consumer.  It never decides whether `newbool` guards.
///
/// The store that publishes the result survives the guard: it mirrors the box
/// into the operand-stack slot the guard's own resume image describes, and
/// swapping the prebuilt singleton in for a recorded call result leaves it
/// storing the same value.
fn walker_newbool_guarded<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    truth: OpRef,
    observed: bool,
    dst_bank: char,
) -> Result<Option<OpRef>, DispatchError> {
    // No resume image, no guard: emitting one without a snapshot to resume
    // into would leave the bail with nowhere to land.  That is the only thing
    // that keeps a caller on the residual box.
    if ctx.fbw_mode.snapshot_sym.is_null() || dst_bank != 'r' {
        return Ok(None);
    }
    let guard = if observed {
        OpCode::GuardTrue
    } else {
        OpCode::GuardFalse
    };
    ctx.trace_ctx.record_guard(guard, &[truth], 0);
    walker_capture_snapshot_for_last_guard(ctx, op_pc)?;
    let result_obj = pyre_object::w_bool_from(observed);
    let const_bool = ctx.trace_ctx.const_ref(result_obj as i64);
    ctx.trace_ctx.set_opref_concrete(
        const_bool,
        majit_ir::Value::Ref(majit_ir::GcRef(result_obj as usize)),
    );
    Ok(Some(const_bool))
}

/// Does `tp` name a layout whose class overrides `is_w` with a value
/// comparison?  `baseobjspace::is_w` gates one branch per overriding class,
/// each demanding both operands be that exact type: `int`
/// (`intobject.py:44`), `float` (`floatobject.py:196`), `complex`
/// (`complexobject.py:287`), `tuple` (`tupleobject.py:47`), `bytes`
/// (`bytesobject.py:25`), `str` (`unicodeobject.py:101`) and `frozenset`
/// (`setobject.py:592`).  Every other class keeps the default pointer
/// identity (`baseobjspace.py:246`).
fn is_w_compares_by_value(tp: *const pyre_object::pyobject::PyType) -> bool {
    [
        &pyre_object::pyobject::INT_TYPE as *const pyre_object::pyobject::PyType,
        &pyre_object::pyobject::FLOAT_TYPE as *const pyre_object::pyobject::PyType,
        &pyre_object::pyobject::COMPLEX_TYPE as *const pyre_object::pyobject::PyType,
        &pyre_object::pyobject::TUPLE_TYPE as *const pyre_object::pyobject::PyType,
        &pyre_object::bytesobject::BYTES_TYPE as *const pyre_object::pyobject::PyType,
        &pyre_object::pyobject::STR_TYPE as *const pyre_object::pyobject::PyType,
        &pyre_object::setobject::FROZENSET_TYPE as *const pyre_object::pyobject::PyType,
    ]
    .iter()
    .any(|special| std::ptr::eq(*special, tp))
}

/// Walker-native fold of the `IS_OP` residual — `bh_compare_fn(lhs, rhs,
/// tag)` with tag 8 (`is`) or 9 (`is_not`), the tags
/// `compare_op_tag_for_opname` assigns those two opnames.
///
/// `IS_OP` is `space.is_w(w_1, w_2)` plus a `newbool`
/// (`pyopcode.py:1078-1092`), and `is_w` (`baseobjspace.py`) dispatches
/// to `w_two.is_w(space, w_one)`, whose default is pointer identity.  Two
/// tiers, mirroring `FASTPATHS_SAME_BOXES`' `ptr_eq`/`ptr_ne` entries
/// (`pyjitpl.py:326-336`):
///
///   * Same box — `baseobjspace::is_w` answers at its opening `ptr::eq`
///     whatever the class, so the result is the constant `True`/`False`.
///     No op and no guard: this is the `b1 is b2` fast check itself.
///   * Distinct boxes whose layouts both keep the default `is_w` — no
///     value-comparison branch can fire, so `is_w` again reduces to that
///     `ptr::eq`.  A `GuardClass` per operand pins the layout, then
///     `ptr_eq`/`ptr_ne` replaces the may-force `compare_fn` and the
///     `GuardNotForced` behind it.
///
/// Declining on a value-comparing layout is what keeps the second tier
/// sound: `GuardClass` pins `ob_type`, and an `int` subclass instance
/// shares `INT_TYPE` with a plain `int` while answering the exact-type gate
/// differently, so the layout alone cannot separate them.  A tagged
/// immediate is declined outright — it carries no `ob_type` to guard.
pub(crate) fn try_walker_fold_is_op<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    op_tag: i64,
    r_args: &[OpRef],
    dst: usize,
    dst_bank: char,
) -> Result<Option<()>, DispatchError> {
    if r_args.len() != 2 || dst_bank != 'r' || !ctx.is_authoritative_executor {
        return Ok(None);
    }
    let invert = match op_tag {
        8 => false,
        9 => true,
        _ => return Ok(None),
    };
    let lhs = r_args[0];
    let rhs = r_args[1];

    if lhs.same_box(rhs) {
        // `x is x` / `x is not x` — statically determined.
        return walker_write_const_bool_result(ctx, op_pc, !invert, dst, dst_bank).map(Some);
    }

    let (Some(lhs_obj), Some(rhs_obj)) = (
        walker_concrete_ref_object(ctx, lhs),
        walker_concrete_ref_object(ctx, rhs),
    ) else {
        return Ok(None);
    };
    if lhs_obj.is_null() || rhs_obj.is_null() {
        return Ok(None);
    }
    if pyre_object::tagged_int::CAN_BE_TAGGED
        && (pyre_object::tagged_int::is_tagged_int(lhs_obj)
            || pyre_object::tagged_int::is_tagged_int(rhs_obj))
    {
        return Ok(None);
    }
    let (lhs_type, rhs_type) = unsafe {
        (
            (*(lhs_obj as *const pyre_object::pyobject::PyObject)).ob_type,
            (*(rhs_obj as *const pyre_object::pyobject::PyObject)).ob_type,
        )
    };
    if is_w_compares_by_value(lhs_type) || is_w_compares_by_value(rhs_type) {
        return Ok(None);
    }
    // The layout test above is the proof that `is_w` reduces to `ptr::eq`
    // here; cross-check it against the real thing and decline rather than
    // record a concrete the emitted `ptr_eq` disagrees with.
    let same = std::ptr::eq(lhs_obj, rhs_obj);
    if same != pyre_interpreter::baseobjspace::is_w(lhs_obj, rhs_obj) {
        return Ok(None);
    }

    // --- commit to the fold: emit IR (no further declines) ---
    for (operand, operand_type) in [(lhs, lhs_type), (rhs, rhs_type)] {
        if operand.is_constant() || ctx.trace_ctx.heap_cache().is_class_known(operand) {
            continue;
        }
        let type_addr = operand_type as usize as i64;
        let type_const = ctx.trace_ctx.const_int(type_addr);
        ctx.trace_ctx
            .record_guard(OpCode::GuardClass, &[operand, type_const], 0);
        walker_capture_snapshot_for_last_guard(ctx, op_pc)?;
        ctx.trace_ctx
            .heap_cache_mut()
            .class_now_known(operand, type_addr);
    }
    let cmp = if invert { OpCode::PtrNe } else { OpCode::PtrEq };
    let truth = ctx.trace_ctx.record_op(cmp, &[lhs, rhs]);
    let result = same != invert;
    ctx.trace_ctx
        .set_opref_concrete(truth, majit_ir::Value::Int(result as i64));
    // `space.newbool` on the truth: its guard plus the prebuilt singleton.  The
    // residual box is the no-snapshot fallback only.
    let boxed = match walker_newbool_guarded(ctx, op_pc, truth, result, dst_bank)? {
        Some(boxed) => boxed,
        None => {
            let boxed =
                crate::helpers::emit_trace_bool_value_from_truth(ctx.trace_ctx, truth, false);
            ctx.trace_ctx.set_opref_concrete(
                boxed,
                majit_ir::Value::Ref(majit_ir::GcRef(pyre_object::w_bool_from(result) as usize)),
            );
            boxed
        }
    };
    write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, boxed)?;
    Ok(Some(()))
}

/// Write an immortal `bool` singleton into a residual call's Ref dst.  An
/// immediately following `is_true` (`POP_JUMP_IF_*`) reads a constant W_Bool,
/// which [`try_walker_specialize_truth_bool`] folds off its concrete rather
/// than unboxing through a residual.
fn walker_write_const_bool_result<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    value: bool,
    dst: usize,
    dst_bank: char,
) -> Result<(), DispatchError> {
    let result_obj = pyre_object::w_bool_from(value);
    let const_bool = ctx.trace_ctx.const_ref(result_obj as i64);
    ctx.trace_ctx.set_opref_concrete(
        const_bool,
        majit_ir::Value::Ref(majit_ir::GcRef(result_obj as usize)),
    );
    write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, const_bool)
}

/// MAKE_FUNCTION inline emission: replace the
/// `jit_make_function_from_globals(globals, code)` residual with the
/// `NewWithVtable` + `SetfieldGc` set `function.py Function.__init__`
/// performs, so a `def` in a loop body virtualizes away instead of allocating a
/// `Function` per iteration.
///
/// Everything the constructor stores is loop-invariant here: `globals` and
/// `code` arrive as baked constants (`codewriter.rs` MakeFunction arm bakes the
/// frame's globals object, and the code object comes from a `LOAD_CONST`), and
/// the remaining slots are derived from them:
///
/// * `name` — `function.py:51 self.name = code.co_name`, a pointer into the
///   `Box::into_raw`'d `CodeObject`, which is never rewritten in place nor
///   freed.  This is the same pointer the residual stores
///   (`function_new_from_code` borrows it too), so a materialized function is
///   indistinguishable from an interpreted one.
/// * `w_name` / `w_qualname` — the code object's single realized `co_name` and
///   `co_qualname`, shared by every function built from it.
/// * `w_builtins` — CPython 3.14 `_PyEval_BuiltinsFromGlobals`, frozen at
///   construction from `globals['__builtins__']`.  Only the allocation-free
///   shape is reproduced: `__builtins__` naming a module, reduced to its dict.
///   An absent or non-module `__builtins__` routes
///   `pick_builtin_obj_checked` through `w_module_new_aliasing_dict` / the
///   default-module build, which mint a fresh object per call and so cannot be
///   baked — those decline to the residual.
///
/// Soundness rests on one guard beyond the constant operands: the module
/// dict's `version?` is pinned, so rebinding `globals['__builtins__']` runs
/// `mutated()` and revokes the loop, exactly as it does for a shadowing insert
/// under the LOAD_GLOBAL cell fold.  Nothing watches the code object's
/// `co_name` / `co_qualname` because neither is mutable in place —
/// `code.replace()` clones first and yields a different code object, which is a
/// different constant.
///
/// Declines (each falls through to the residual, which stays correct): a
/// non-constant operand, a non-`PyCode` or bodyless code object, globals that
/// are not a module dict, an unbakeable `__builtins__`, and any baked pointer
/// the collector may relocate.
pub(crate) fn try_walker_specialize_make_function<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    r_args: &[OpRef],
    dst: usize,
    dst_bank: char,
) -> Result<Option<()>, DispatchError> {
    if r_args.len() != 2 {
        return Ok(None);
    }
    let (globals_op, code_op) = (r_args[0], r_args[1]);
    if !globals_op.is_constant() || !code_op.is_constant() {
        return Ok(None);
    }
    let Some(w_code) = walker_concrete_ref_object(ctx, code_op) else {
        return Ok(None);
    };
    // A `BuiltinCode`-backed carrier is immortal and boxes its name through
    // `malloc_raw`; `is_code` admits only the `PyCode` shape this reproduces.
    if !unsafe { pyre_interpreter::is_code(w_code) } {
        return Ok(None);
    }
    let code_ptr =
        unsafe { pyre_interpreter::w_code_get_ptr(w_code) } as *const pyre_interpreter::CodeObject;
    if code_ptr.is_null() {
        return Ok(None);
    }

    // Realizing the name/qualname are the fold's collection points (they
    // allocate once per code object and hit the cache afterwards), so they run
    // here, while the only live pointers are the `Box::into_raw`'d code wrapper
    // and its `CodeObject` — neither of which the collector relocates.
    let w_name = unsafe { pyre_interpreter::pycode::w_code_name_obj(w_code) };
    if w_name.is_null() {
        return Ok(None);
    }
    let w_qualname = unsafe { pyre_interpreter::pycode::w_code_qualname_obj(w_code) };
    if w_qualname.is_null() {
        return Ok(None);
    }
    let Some(w_globals) = walker_concrete_ref_object(ctx, globals_op) else {
        return Ok(None);
    };
    // Restrict to a module namespace before probing it directly, so the slot
    // read below walks the same storage `pick_builtin_obj_checked`'s
    // `finditem_str` does.  This is also what `walker_pin_namespace_version`
    // needs, but that one emits IR, so it runs only once the fold commits.
    if unsafe { pyre_object::dictmultiobject::w_module_dict_strategy_or_null(w_globals) }.is_null()
    {
        return Ok(None);
    }
    // `function_new_impl`'s `w_builtins` derivation, restricted to the branch
    // that allocates nothing and therefore answers the same object each run.
    let w_builtins_module = unsafe { pyre_object::w_dict_getitem_str(w_globals, "__builtins__") }
        .unwrap_or(pyre_object::PY_NULL);
    if w_builtins_module.is_null() || !unsafe { pyre_object::is_module(w_builtins_module) } {
        return Ok(None);
    }
    let w_builtins = unsafe { pyre_object::w_module_get_w_dict(w_builtins_module) };
    if w_builtins.is_null() {
        return Ok(None);
    }
    for baked in [w_builtins, w_name, w_qualname] {
        if majit_gc::can_move(majit_ir::GcRef(baked as usize)) {
            return Ok(None);
        }
    }

    // --- commit to the fold: emit IR (no further declines) ---
    // The only mutable input: `globals['__builtins__']` may be rebound after
    // this function is built, and a later iteration must then see the new
    // mapping.  Pinning the namespace `version?` revokes the loop instead.
    walker_pin_namespace_version(ctx, op_pc, w_globals)?;
    let header_w_class = ctx
        .trace_ctx
        .const_ref(pyre_object::get_instantiate(&pyre_interpreter::FUNCTION_TYPE) as i64);
    // `function.py can_change_code = True` for a plain `def`.
    let can_change_code = ctx.trace_ctx.const_int(1);
    let name = ctx
        .trace_ctx
        .const_ref(unsafe { &(*code_ptr).obj_name } as *const String as i64);
    let w_name_const = ctx.trace_ctx.const_ref(w_name as i64);
    let w_builtins_const = ctx.trace_ctx.const_ref(w_builtins as i64);
    let w_qualname_const = ctx.trace_ctx.const_ref(w_qualname as i64);
    let func_op = crate::helpers::emit_make_function_inline(
        ctx.trace_ctx,
        header_w_class,
        code_op,
        can_change_code,
        name,
        w_name_const,
        globals_op,
        w_builtins_const,
        w_qualname_const,
    );
    ctx.trace_ctx
        .heap_cache_mut()
        .class_now_known(func_op, &pyre_interpreter::FUNCTION_TYPE as *const _ as i64);
    // Tracing is execution: build the concrete function the rest of the walk
    // observes.  A fresh `Function` per evaluation is what MAKE_FUNCTION
    // produces anyway, so the trace allocating its own is not an identity
    // divergence.
    let func = pyre_interpreter::runtime_ops::make_function_from_code_obj_with_globals_obj(
        w_code, w_globals,
    );
    ctx.trace_ctx.set_opref_concrete(
        func_op,
        majit_ir::Value::Ref(majit_ir::GcRef(func as usize)),
    );
    write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, func_op)?;
    Ok(Some(()))
}

/// SET_FUNCTION_ATTRIBUTE inline emission: replace the
/// `jit_set_function_attribute(func, attr, flag)` residual with the `SetfieldGc`
/// set that flag's setter performs, and hand the operand itself back as the
/// opcode's result.
///
/// The opcode only ever runs on the function the preceding MAKE_FUNCTION just
/// pushed, so the stores are constructor stores on an allocation this trace
/// made: nothing has read the slots, and no compiled trace can hold a folded
/// view of a field belonging to an object that did not exist when it was
/// recorded.  That is the same footing `emit_make_function_inline` already
/// writes `code`, `name`, `w_func_globals_obj` and `w_qualname` on — all
/// `Function` quasi slots — so this adds no new class of write.
/// `heap_cache.is_unescaped` is what establishes it: the operand must still be
/// an unescaped allocation of this trace, which a `Function` reaching the
/// opcode from anywhere else cannot be.
///
/// That gate is also what makes each emitted store *equivalent* to the setter
/// it stands in for, not merely safe on top of it.  The three single-slot
/// setters notify a watcher (`defs_w?`, `w_kw_defs?`, `closure`), and
/// `function_notify_quasi_immut` resolves through
/// `function_quasi_immut_field`, which answers `None` while `mutate_slots` is
/// null — which a fresh unescaped allocation's still is, because only a
/// recorded read installs the block.  So the notify the emit leaves out had
/// nothing to invalidate.  The two annotation slots carry no watcher at all,
/// so their setters have none to leave out.
///
/// Passing `func` through as the result is the half that pays.  The residual's
/// result is opaque, so the inline-call path that follows re-reads
/// `Function.code`, `Function.w_func_globals_obj` and `Function.defs_w` off it
/// and guards each one; served from the virtual instead, those reads fold to
/// what the emit stored and the guards go with them, and the whole definition
/// sequence virtualizes away when the function does not escape.
///
/// The two annotation flags name a second slot as well, and both of their
/// stores are emitted: `Annotations` writes `w_ann` and clears `w_annotate`
/// unconditionally, PEP 649's `Annotate` writes `w_annotate` and clears
/// `w_ann` for an operand that is not `None`.  That last one is the only
/// data-dependent arm, so it is taken only where the operand is provably not
/// `None` at every execution rather than at this one — a baked constant, or an
/// unescaped allocation of this trace, which is what the `__annotate__`
/// function the preceding `MAKE_FUNCTION` built is.  They matter out of
/// proportion to how often a `def` carries annotations: 3.14 emits
/// `SET_FUNCTION_ATTRIBUTE annotate` FIRST, so declining it escapes the
/// allocation and the `defaults` store behind it then declines too, leaving
/// the whole definition sequence residual.
///
/// `TypeParams` is not folded.  The codegen never sets that bit: a PEP 695
/// generic `def` stamps its type parameters through
/// `CALL_INTRINSIC_2 INTRINSIC_SET_FUNCTION_TYPE_PARAMS`, whose
/// `set_function_typeparams` writes `Function.w_typeparams`.  An arm here
/// would be unreachable, and leaving it out is what keeps this fold and the
/// residual from being able to disagree about a flag neither is ever handed.
pub(crate) fn try_walker_specialize_set_function_attribute<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    i_args: &[OpRef],
    r_args: &[OpRef],
    dst: usize,
    dst_bank: char,
) -> Result<Option<()>, DispatchError> {
    use pyre_interpreter::bytecode::MakeFunctionFlag;

    if r_args.len() != 2 || i_args.len() != 1 {
        return Ok(None);
    }
    let (func_op, attr_op) = (r_args[0], r_args[1]);
    let Some(majit_ir::Value::Int(flag)) = ctx.trace_ctx.box_value(i_args[0]) else {
        return Ok(None);
    };
    // A constant operand names a function that outlived the recording, and an
    // escaped one may already be reachable from a compiled trace's folded
    // view; both keep the residual, which notifies the watcher slot.
    if func_op.is_constant() || !ctx.trace_ctx.heap_cache().is_unescaped(func_op) {
        return Ok(None);
    }
    let Some(w_func) = walker_concrete_ref_object(ctx, func_op) else {
        return Ok(None);
    };
    if !unsafe { pyre_interpreter::is_function(w_func) } {
        return Ok(None);
    }
    let Some(w_attr) = walker_concrete_ref_object(ctx, attr_op) else {
        return Ok(None);
    };
    // What the PEP 649 setter branches on. A baked constant is the same object
    // at every execution; an unescaped allocation of this trace is a fresh
    // object, which the singleton is not.
    let attr_is_never_none = !unsafe { pyre_object::is_none(w_attr) }
        && (attr_op.is_constant() || ctx.trace_ctx.heap_cache().is_unescaped(attr_op));

    // The lowering bakes the `MakeFunctionFlag` bit-position discriminant, so
    // the enum's own `#[repr]` layout is what this reads back. Each arm is the
    // store set its `function.py` setter performs, in the setter's own order;
    // `true` carries the operand, `false` is the setter's null.
    let stores: Vec<(DescrRef, bool)> = match flag {
        f if f == MakeFunctionFlag::Defaults as i64 => {
            vec![(crate::descr::function_defs_w_descr(), true)]
        }
        f if f == MakeFunctionFlag::KwOnlyDefaults as i64 => {
            vec![(crate::descr::function_w_kw_defs_descr(), true)]
        }
        f if f == MakeFunctionFlag::Closure as i64 => {
            vec![(crate::descr::function_closure_descr(), true)]
        }
        // `function.py fset_func_annotations`: the eager dict, and the lazy
        // `__annotate__` it supersedes cleared.
        f if f == MakeFunctionFlag::Annotations as i64 => vec![
            (crate::descr::function_w_ann_descr(), true),
            (crate::descr::function_w_annotate_descr(), false),
        ],
        // PEP 649: the callable, and the eager dict it supersedes cleared --
        // only for an operand the setter's `is_none` test lets through.
        f if f == MakeFunctionFlag::Annotate as i64 => {
            if !attr_is_never_none {
                return Ok(None);
            }
            vec![
                (crate::descr::function_w_annotate_descr(), true),
                (crate::descr::function_w_ann_descr(), false),
            ]
        }
        _ => return Ok(None),
    };

    // `KwOnlyDefaults` is the one flag whose stored value is not the operand:
    // a definition's keyword-only defaults are rebuilt into the namespace
    // mapping (`function.py init_kwdefaults_dict`), whose entries a trace can
    // fold.  Record time reaches that rebuild through
    // `jit_set_function_attribute` below, so the store emitted for the compiled
    // path has to install the same flavour or the two views of
    // `Function.w_kw_defs` disagree.
    //
    // Run the rebuild here, while declining is still possible: it allocates,
    // and the function this is stamping is a fresh allocation of the traced
    // iteration, so a collection under it moves the object `w_func` names.  The
    // trace's own concrete is forwarded, so re-reading it off `func_op` is what
    // recovers the current address.
    let (w_attr, w_func) = if flag == MakeFunctionFlag::KwOnlyDefaults as i64 {
        let w_converted = unsafe { pyre_interpreter::function::init_kwdefaults_dict(w_attr) };
        let Some(w_func) = walker_concrete_ref_object(ctx, func_op) else {
            return Ok(None);
        };
        (w_converted, w_func)
    } else {
        (w_attr, w_func)
    };

    // --- commit to the fold: emit IR (no further declines) ---
    // The rebuild is a residual because it allocates — but it takes only the
    // mapping.  `func_op` is not an argument and stays unescaped, which is what
    // keeps the `defaults` store emitted behind this one foldable.  Rebuilding
    // an already-rebuilt mapping is a no-op, so the execution call below needs
    // no special case for having been handed the rebuilt one.
    let attr_op = if flag == MakeFunctionFlag::KwOnlyDefaults as i64 {
        let converted_op = ctx.trace_ctx.call_ref_typed_with_effect(
            crate::helpers::jit_init_kwdefaults_dict as *const (),
            &[attr_op],
            &[majit_ir::Type::Ref],
            majit_ir::EffectInfo::new(
                majit_ir::ExtraEffect::CannotRaise,
                majit_ir::OopSpecIndex::None,
            ),
        );
        ctx.trace_ctx.set_opref_concrete(
            converted_op,
            majit_ir::Value::Ref(majit_ir::GcRef(w_attr as usize)),
        );
        converted_op
    } else {
        attr_op
    };
    let null_op = ctx.trace_ctx.const_null();
    for (descr, carries_operand) in stores {
        let value = if carries_operand { attr_op } else { null_op };
        let index = descr.index();
        ctx.trace_ctx
            .record_op_with_descr(majit_ir::OpCode::SetfieldGc, &[func_op, value], descr);
        ctx.trace_ctx
            .heapcache_setfield_cached(func_op, index, value);
    }
    // Tracing is execution: apply the same store the residual would have.
    pyre_interpreter::runtime_ops::jit_set_function_attribute(w_func as i64, w_attr as i64, flag);
    write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, func_op)?;
    Ok(Some(()))
}

/// Mixed W_LongObject/W_IntObject COMPARE_OP specialization.
///
/// `pypy/objspace/std/longobject.py:_make_descr_cmp` selects the corresponding
/// `rbigint.int_<cmp>` method for a machine-int other operand.  For the
/// reflected order, select the inverse comparison with the bigint kept as the
/// first residual argument (`int < long` becomes `long > int`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_walker_specialize_compare_op_long_int<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    op_tag: i64,
    r_args: &[OpRef],
    allboxes: &[OpRef],
    call_descr: &dyn majit_ir::descr::CallDescr,
    dst: usize,
    dst_bank: char,
) -> Result<Option<()>, DispatchError> {
    if !ctx.is_authoritative_executor || r_args.len() != 2 || dst_bank != 'r' {
        return Ok(None);
    }
    use pyre_interpreter::bytecode::ComparisonOperator;
    use pyre_interpreter::objspace::descroperation as desc;
    type CompareFn = extern "C" fn(i64, i64) -> i64;
    let Some(cmp_op) = pyre_interpreter::runtime_ops::compare_op_from_tag(op_tag) else {
        return Ok(None);
    };
    let (lhs_obj, rhs_obj) = match (
        walker_concrete_ref_object(ctx, r_args[0]),
        walker_concrete_ref_object(ctx, r_args[1]),
    ) {
        (Some(lhs), Some(rhs)) => (lhs, rhs),
        _ => return Ok(None),
    };
    let lhs_is_long = unsafe { pyre_object::is_long(lhs_obj) };
    let rhs_is_long = unsafe { pyre_object::is_long(rhs_obj) };
    let lhs_is_int = unsafe { pyre_object::is_int(lhs_obj) };
    let rhs_is_int = unsafe { pyre_object::is_int(rhs_obj) };
    let (long, int, long_obj, int_obj, reflected) = if lhs_is_long && rhs_is_int {
        (r_args[0], r_args[1], lhs_obj, rhs_obj, false)
    } else if lhs_is_int && rhs_is_long {
        (r_args[1], r_args[0], rhs_obj, lhs_obj, true)
    } else {
        return Ok(None);
    };
    let (Some(long_class), Some(int_class)) = (unsafe {
        (
            walker_exact_builtin_class(long_obj),
            walker_exact_builtin_class(int_obj),
        )
    }) else {
        return Ok(None);
    };
    let effective_cmp = if reflected {
        match cmp_op {
            ComparisonOperator::Less => ComparisonOperator::Greater,
            ComparisonOperator::LessOrEqual => ComparisonOperator::GreaterOrEqual,
            ComparisonOperator::Greater => ComparisonOperator::Less,
            ComparisonOperator::GreaterOrEqual => ComparisonOperator::LessOrEqual,
            ComparisonOperator::Equal => ComparisonOperator::Equal,
            ComparisonOperator::NotEqual => ComparisonOperator::NotEqual,
        }
    } else {
        cmp_op
    };
    let helper: CompareFn = match effective_cmp {
        ComparisonOperator::Less => desc::jit_bigint_int_lt,
        ComparisonOperator::LessOrEqual => desc::jit_bigint_int_le,
        ComparisonOperator::Greater => desc::jit_bigint_int_gt,
        ComparisonOperator::GreaterOrEqual => desc::jit_bigint_int_ge,
        ComparisonOperator::Equal => desc::jit_bigint_int_eq,
        ComparisonOperator::NotEqual => desc::jit_bigint_int_ne,
    };
    let int_value = unsafe { pyre_object::w_int_get_value(int_obj) };

    let Some(boxed_result_i64) = walker_execute_may_force_boxed(ctx, allboxes, call_descr) else {
        return Ok(None);
    };
    let boxed_result_obj = boxed_result_i64 as usize as pyre_object::PyObjectRef;
    if boxed_result_obj == pyre_object::PY_NULL
        || !unsafe { pyre_object::is_bool(boxed_result_obj) }
    {
        return Ok(None);
    }
    let concrete_truth = unsafe { pyre_object::w_bool_get_value(boxed_result_obj) as i64 };

    let long_type_addr = &pyre_object::pyobject::LONG_TYPE as *const _ as i64;
    walker_guard_class(ctx, op_pc, long, long_type_addr)?;
    walker_guard_exact_w_class(ctx, op_pc, long, long_class)?;
    let (int_type, int_descr) = crate::state::int_or_bool_unbox_type_descr(int_obj);
    let int_raw = walker_unbox_int_typed(ctx, op_pc, int, int_type, int_descr)?;
    walker_guard_exact_w_class(ctx, op_pc, int, int_class)?;
    let off = pyre_object::longobject::LONG_VALUE_OFFSET;
    let long_payload = unsafe { *((long_obj as *const u8).add(off) as *const i64) };
    let long_pl = ctx.trace_ctx.record_op_with_descr(
        OpCode::GetfieldGcR,
        &[long],
        crate::descr::long_value_descr(),
    );
    ctx.trace_ctx.set_opref_concrete(
        long_pl,
        majit_ir::Value::Ref(majit_ir::GcRef(long_payload as usize)),
    );
    let helper_ptr = helper as *const ();
    let truth = ctx.trace_ctx.call_typed_with_effect_pure(
        OpCode::CallI,
        helper_ptr,
        &[long_pl, int_raw],
        &[majit_ir::Type::Ref, majit_ir::Type::Int],
        majit_ir::Type::Int,
        majit_metainterp::ELIDABLE_CANNOT_RAISE_EFFECT_INFO,
        &[
            majit_ir::Value::Int(helper_ptr as usize as i64),
            majit_ir::Value::Ref(majit_ir::GcRef(long_payload as usize)),
            majit_ir::Value::Int(int_value),
        ],
        majit_ir::Value::Int(concrete_truth),
    );
    ctx.trace_ctx
        .set_opref_concrete(truth, majit_ir::Value::Int(concrete_truth));
    // `space.newbool` on the truth: its guard plus the prebuilt singleton.  The
    // residual box is the no-snapshot fallback only.
    let boxed = match walker_newbool_guarded(ctx, op_pc, truth, concrete_truth != 0, dst_bank)? {
        Some(boxed) => boxed,
        None => {
            let boxed =
                crate::helpers::emit_trace_bool_value_from_truth(ctx.trace_ctx, truth, false);
            ctx.trace_ctx.set_opref_concrete(
                boxed,
                majit_ir::Value::Ref(majit_ir::GcRef(boxed_result_i64 as usize)),
            );
            boxed
        }
    };
    write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, boxed)?;
    Ok(Some(()))
}

/// W_LongObject (bigint) COMPARE_OP specialization — the long analogue of
/// [`try_walker_specialize_compare_op_int`].  Both operands are `int`-typed but
/// bigint-stored: guard each against `LONG_TYPE`, read each `value` payload,
/// then `CallPure_I` the pure
/// `jit_bigint_cmp` (sign of `a <=> b` in {-1,0,1}; a comparison neither
/// allocates nor raises, so `EF_ELIDABLE_CANNOT_RAISE` and NO trailing guard)
/// and turn the sign into the requested truth with `int_<cmp>(sign, 0)` before
/// boxing to a `W_Bool` (same #62 dead-box elision as the int path).  Same gate
/// + return contract as [`try_walker_specialize_binary_op_long`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_walker_specialize_compare_op_long<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    op_tag: i64,
    r_args: &[OpRef],
    allboxes: &[OpRef],
    call_descr: &dyn majit_ir::descr::CallDescr,
    dst: usize,
    dst_bank: char,
) -> Result<Option<()>, DispatchError> {
    if !ctx.is_authoritative_executor || r_args.len() != 2 || dst_bank != 'r' {
        return Ok(None);
    }
    let Some(cmp_op) = pyre_interpreter::runtime_ops::compare_op_from_tag(op_tag) else {
        return Ok(None);
    };
    use pyre_interpreter::bytecode::ComparisonOperator;
    // `a <cmp> b` ⟺ `sign(a <=> b) <cmp> 0`.
    let cmp = match cmp_op {
        ComparisonOperator::Less => OpCode::IntLt,
        ComparisonOperator::LessOrEqual => OpCode::IntLe,
        ComparisonOperator::Greater => OpCode::IntGt,
        ComparisonOperator::GreaterOrEqual => OpCode::IntGe,
        ComparisonOperator::Equal => OpCode::IntEq,
        ComparisonOperator::NotEqual => OpCode::IntNe,
    };
    let lhs = r_args[0];
    let rhs = r_args[1];
    let (Some(lhs_obj), Some(rhs_obj)) = (
        walker_concrete_ref_object(ctx, lhs),
        walker_concrete_ref_object(ctx, rhs),
    ) else {
        return Ok(None);
    };
    if !unsafe { pyre_object::is_long(lhs_obj) && pyre_object::is_long(rhs_obj) } {
        return Ok(None);
    }
    let (Some(lhs_class), Some(rhs_class)) = (unsafe {
        (
            walker_exact_builtin_class(lhs_obj),
            walker_exact_builtin_class(rhs_obj),
        )
    }) else {
        return Ok(None);
    };
    // Authentic boxed W_Bool via the same execute path the int leg uses; also
    // advances the concrete VM state the downstream ops read.
    let Some(boxed_result_i64) = walker_execute_may_force_boxed(ctx, allboxes, call_descr) else {
        return Ok(None);
    };
    let long_type_addr = &pyre_object::pyobject::LONG_TYPE as *const _ as i64;
    walker_guard_class(ctx, op_pc, lhs, long_type_addr)?;
    walker_guard_class(ctx, op_pc, rhs, long_type_addr)?;
    walker_guard_exact_w_class(ctx, op_pc, lhs, lhs_class)?;
    walker_guard_exact_w_class(ctx, op_pc, rhs, rhs_class)?;
    // `_make_descr_cmp` (longobject.py) compares `self.num` against
    // `w_other.num`, so the two payload reads are trace ops rather than work
    // hidden inside the callee. Spelling them out is also what keeps a
    // `W_LongObject` this same trace built from having to escape into the
    // comparison: the read hits the heap cache entry `emit_box_long_inline`
    // filed and the box stays virtual.
    let lhs_payload = unsafe { long_payload_of(lhs_obj) };
    let rhs_payload = unsafe { long_payload_of(rhs_obj) };
    let lhs_pl = walker_read_long_payload(ctx, lhs, lhs_payload);
    let rhs_pl = walker_read_long_payload(ctx, rhs, rhs_payload);
    // Pure `rbigint` comparison → sign in {-1,0,1}. Dead after the `int_<cmp>`
    // below and never spans a guard, so it needs no blackhole reconstruction.
    let cmp_fn = pyre_object::longobject::jit_bigint_cmp as *const ();
    let sign_concrete = pyre_object::longobject::jit_bigint_cmp(lhs_payload, rhs_payload);
    let concrete_args = [
        majit_ir::Value::Int(cmp_fn as usize as i64),
        majit_ir::Value::Ref(majit_ir::GcRef(lhs_payload as usize)),
        majit_ir::Value::Ref(majit_ir::GcRef(rhs_payload as usize)),
    ];
    let sign = ctx.trace_ctx.call_typed_with_effect_pure(
        OpCode::CallI,
        cmp_fn,
        &[lhs_pl, rhs_pl],
        &[majit_ir::Type::Ref, majit_ir::Type::Ref],
        majit_ir::Type::Int,
        majit_metainterp::ELIDABLE_CANNOT_RAISE_EFFECT_INFO,
        &concrete_args,
        majit_ir::Value::Int(sign_concrete),
    );
    ctx.trace_ctx
        .set_opref_concrete(sign, majit_ir::Value::Int(sign_concrete));
    let zero = ctx.trace_ctx.const_int(0);
    let truth = ctx.trace_ctx.record_op(cmp, &[sign, zero]);
    let folded = majit_metainterp::eval_binop_i(cmp, sign_concrete, 0);
    ctx.trace_ctx
        .set_opref_concrete(truth, majit_ir::Value::Int(folded));
    // `space.newbool` on the truth: its guard plus the prebuilt singleton.  The
    // residual box is the no-snapshot fallback only.
    let boxed = match walker_newbool_guarded(ctx, op_pc, truth, folded != 0, dst_bank)? {
        Some(boxed) => boxed,
        None => {
            let boxed =
                crate::helpers::emit_trace_bool_value_from_truth(ctx.trace_ctx, truth, false);
            ctx.trace_ctx.set_opref_concrete(
                boxed,
                majit_ir::Value::Ref(majit_ir::GcRef(boxed_result_i64 as usize)),
            );
            boxed
        }
    };
    write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, boxed)?;
    Ok(Some(()))
}

/// #57 SLICE 3c: walker-native speculative float specialization for the
/// `BINARY_OP` helper residual_call (oopspec `BinaryOp`), the float
/// analogue of [`try_walker_specialize_binary_op_int`].  Re-derives
/// the former float fast path's structure walker-native: per operand
/// either `guard_class FLOAT` + `getfield_gc_pure_f`, or (int operand)
/// `guard_class INT` + `getfield_gc_i` + `cast_int_to_float`; then
/// `float_OP` and `wrapfloat`.
///
/// Only the bare-primitive operators (`FloatAdd` / `FloatSub` /
/// `FloatMul` / `FloatTrueDiv`) are specialized — Power / FloorDivide /
/// Remainder have no FLOAT_* opcode and defer to the generic
/// `CALL_MAY_FORCE` leg (Power lowers to a `call_may_force` +
/// `guard_no_exception` there).  Tried as a fallback only after the int
/// specialization declines, so two-int operands keep int `__op__`
/// arithmetic.
pub(crate) fn try_walker_specialize_binary_op_float<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    op_tag: i64,
    r_args: &[OpRef],
    allboxes: &[OpRef],
    call_descr: &dyn majit_ir::descr::CallDescr,
    dst: usize,
    dst_bank: char,
) -> Result<Option<DispatchOutcome>, DispatchError> {
    if !ctx.is_authoritative_executor || r_args.len() != 2 || dst_bank != 'r' {
        return Ok(None);
    }
    let Some(bin_op) = pyre_interpreter::runtime_ops::binary_op_from_tag(op_tag) else {
        return Ok(None);
    };
    use pyre_interpreter::bytecode::BinaryOperator;
    // Power has no FLOAT_* opcode — it lowers to the raw-float
    // `float_pow_jit` call (floatobject.py descr_pow → _pow), same
    // as the trait's `is_power` arm.
    let op_code = match bin_op {
        BinaryOperator::Add | BinaryOperator::InplaceAdd => Some(OpCode::FloatAdd),
        BinaryOperator::Subtract | BinaryOperator::InplaceSubtract => Some(OpCode::FloatSub),
        BinaryOperator::Multiply | BinaryOperator::InplaceMultiply => Some(OpCode::FloatMul),
        BinaryOperator::TrueDivide | BinaryOperator::InplaceTrueDivide => {
            Some(OpCode::FloatTrueDiv)
        }
        BinaryOperator::Power | BinaryOperator::InplacePower => None,
        _ => return Ok(None),
    };

    let Some((lhs, rhs, lhs_obj, rhs_obj, lhs_is_int, rhs_is_int, lhs_f64, rhs_f64)) =
        walker_float_specialization_input_operands(ctx, r_args)
    else {
        return Ok(None);
    };

    if matches!(op_code, Some(OpCode::FloatTrueDiv)) && rhs_f64 == 0.0 {
        let Some(Err(exc_i64)) = walker_execute_may_force_boxed_outcome(ctx, allboxes, call_descr)
        else {
            return Ok(None);
        };
        if let Some(cb) = crate::callbacks::try_get() {
            (cb.drain_backend_jit_exc)();
        }
        let exc = exc_i64 as usize as pyre_object::PyObjectRef;
        let kind = pyre_object::interp_exceptions::ExcKind::ZeroDivisionError;
        if !walker_recorded_builtin_raise_is_supported(exc, kind) {
            return Ok(None);
        }
        let Some(ec) = walker_ensure_execution_context(ctx) else {
            return Ok(None);
        };

        let _lhs_raw = walker_coerce_dispatching_operand_to_float(
            ctx, op_pc, lhs, lhs_obj, lhs_is_int, lhs_f64, false,
        )?;
        let rhs_raw = walker_coerce_dispatching_operand_to_float(
            ctx, op_pc, rhs, rhs_obj, rhs_is_int, rhs_f64, false,
        )?;
        let rhs_zero = walker_float_eq_const(ctx, rhs_raw, 0.0, 1);
        walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardTrue, &[rhs_zero])?;
        return Ok(Some(walker_emit_recorded_builtin_raise(ctx, ec, exc, kind)));
    }

    let Some(Ok(boxed_result_i64)) =
        walker_execute_may_force_boxed_outcome(ctx, allboxes, call_descr)
    else {
        return Ok(None);
    };
    if boxed_result_i64 == 0 {
        return Ok(None);
    }
    if op_code.is_none() {
        // The generic helper already executed concretely (it produced
        // `boxed_result_i64`), so a non-float result here would mean
        // `float ** x` returned a non-W_FloatObject — decline rather
        // than mis-unbox the concrete stamp.
        let boxed_obj = boxed_result_i64 as pyre_object::PyObjectRef;
        if unsafe { !pyre_object::pyobject::is_float(boxed_obj) } {
            return Ok(None);
        }
    }

    // --- emit the specialized IR (walker-native) ---
    let lhs_raw = walker_coerce_dispatching_operand_to_float(
        ctx, op_pc, lhs, lhs_obj, lhs_is_int, lhs_f64, false,
    )?;
    let rhs_raw = walker_coerce_dispatching_operand_to_float(
        ctx, op_pc, rhs, rhs_obj, rhs_is_int, rhs_f64, false,
    )?;
    // rint.py `_ovf_zer` analogue for float true-division: emit a
    // `float_eq(rhs, 0.0) → guard_false` precondition ahead of the bare
    // `FloatTrueDiv` llop so a future zero divisor deopts to the checked
    // descr_truediv path (which raises ZeroDivisionError) rather than
    // computing a raw IEEE inf.  The bare llop is sound only behind this
    // non-zero guarantee.
    if matches!(op_code, Some(OpCode::FloatTrueDiv)) {
        let rhs_zero = walker_float_eq_const(ctx, rhs_raw, 0.0, (rhs_f64 == 0.0) as i64);
        walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardFalse, &[rhs_zero])?;
    }
    let raw_result = match op_code {
        Some(op_code) => {
            let r = ctx.trace_ctx.record_op(op_code, &[lhs_raw, rhs_raw]);
            let bits = majit_metainterp::eval_binop_f(
                op_code,
                lhs_f64.to_bits() as i64,
                rhs_f64.to_bits() as i64,
            );
            ctx.trace_ctx
                .set_opref_concrete(r, majit_ir::Value::Float(f64::from_bits(bits as u64)));
            r
        }
        None => {
            let result_val = unsafe { pyre_object::w_float_get_value(boxed_result_i64 as _) };
            // _pow (floatobject.py) traced inline for its fast paths:
            // every special-case `if` becomes a comparison guard and only
            // the raw libm pow stays residual.
            if let Some(r) = walker_emit_float_pow_inline(
                ctx, op_pc, lhs_raw, rhs_raw, lhs_f64, rhs_f64, result_val,
            )? {
                r
            } else {
                // Cold-path fallback (nan/inf operands, negative base):
                // the opaque `_pow` helper.  It is EF_CAN_RAISE, NOT
                // force_virtual: pyjitpl.py execute_varargs(
                // rop.CALL_F, ..., exc=True, pure=False) records CALL_F
                // and handle_possible_exception → GUARD_NO_EXCEPTION
                // (pyjitpl.py).  The raising case never reaches
                // here: `walker_float_specialization_operands` already
                // executed the helper concretely and returns `None` on a
                // raise, falling back to the generic residual leg.
                let r = ctx.trace_ctx.call_float_typed_with_effect(
                    crate::trace_opcode::float_pow_jit as *const (),
                    &[lhs_raw, rhs_raw],
                    &[majit_ir::Type::Float, majit_ir::Type::Float],
                    majit_metainterp::default_effect_info(),
                );
                walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardNoException, &[])?;
                ctx.trace_ctx
                    .set_opref_concrete(r, majit_ir::Value::Float(result_val));
                r
            }
        }
    };
    let boxed = crate::state::wrapfloat(ctx.trace_ctx, raw_result);
    ctx.trace_ctx.set_opref_concrete(
        boxed,
        majit_ir::Value::Ref(majit_ir::GcRef(boxed_result_i64 as usize)),
    );
    write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, boxed)?;
    Ok(Some(DispatchOutcome::Continue))
}

/// Two-sided bounds guard `0 <= raw_index < len` for a direct element access.
///
/// The trace is recorded from a non-negative observed index, but a later
/// NEGATIVE index would still satisfy `raw_index < len` and reach the element
/// having proved nothing about its sign.  `space.getitem` / `space.setitem`
/// remap a negative index to `index + len` (listobject.py, tupleobject.py), so
/// the direct access would address before the start of the array; the
/// lower-bound guard deopts such an index to re-execute that remap generically.
///
/// Both halves are emitted here so that an indexing arm cannot take one without
/// the other.
fn walker_emit_index_bounds_guards<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    raw_index: OpRef,
    index: i64,
    lenbox: OpRef,
    concrete_len: usize,
) -> Result<(), DispatchError> {
    let zero = ctx.trace_ctx.const_int(0);
    let nonneg = ctx.trace_ctx.record_op(OpCode::IntGe, &[raw_index, zero]);
    ctx.trace_ctx
        .set_opref_concrete(nonneg, majit_ir::Value::Int((index >= 0) as i64));
    walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardTrue, &[nonneg])?;
    let in_bounds = ctx.trace_ctx.record_op(OpCode::IntLt, &[raw_index, lenbox]);
    ctx.trace_ctx.set_opref_concrete(
        in_bounds,
        majit_ir::Value::Int(((index as usize) < concrete_len) as i64),
    );
    walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardTrue, &[in_bounds])?;
    Ok(())
}

/// #62: walker-native speculative specialization for the `BINARY_SUBSCR`
/// helper residual_call (oopspec `BinaryOp`, op_tag `Subscr`).  Ports
/// the former subscription/list-strategy path for the object-, int-, and
/// float-storage list strategies with a
/// non-negative concrete index: `guard_class LIST` + `guard_value(strategy)`
/// + unbox index + `IntLt` bounds guard, then the strategy-specific element
/// load — `getarrayitem_gc_r` against the `Ptr(GcArray(OBJECTPTR))` items
/// block for object storage (the element is a boxed Ref read directly), or a
/// raw-array getitem + `wrapint` / `wrapfloat` rebox for int/float storage.
/// The authentic boxed result is taken from the same `execute_may_force_call`
/// path the generic leg uses.
///
/// A canonical Unicode-strategy dict with an exact-str key and a concrete hit
/// records the `rordereddict.py dict.lookup` oopspec producer: exact dict/key
/// guards, a strategy guard, elidable `rstr.ll_strhash`, `dict.lookup`, a
/// non-negative guard on the returned entry index, then a guarded value read.
///
/// Tuples, dict misses, empty-strategy lists, negative indices, and
/// non-`list[int]` operands fall through to the generic `CallMayForce` record
/// (`Ok(None)`), preserving Python `__getitem__` semantics.
pub(crate) fn try_walker_specialize_subscr<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    r_args: &[OpRef],
    allboxes: &[OpRef],
    call_descr: &dyn majit_ir::descr::CallDescr,
    dst: usize,
    dst_bank: char,
) -> Result<Option<()>, DispatchError> {
    if !ctx.is_authoritative_executor || r_args.len() != 2 || dst_bank != 'r' {
        return Ok(None);
    }
    let list_op = r_args[0];
    let key_op = r_args[1];
    let (Some(list_obj), Some(key_obj)) = (
        walker_concrete_ref_object(ctx, list_op),
        walker_concrete_ref_object(ctx, key_op),
    ) else {
        return Ok(None);
    };

    if let Some(hit) = walker_probe_exact_dict_hit(list_obj, key_obj)? {
        return walker_emit_exact_dict_hit(
            ctx, op_pc, list_op, key_op, list_obj, hit, dst, dst_bank,
        );
    }

    // #171/#11 Approach C: canonical array-backed `W_TupleObject[i]`.  Two
    // gates, both required:
    //   * `ob_type == &TUPLE_TYPE` (tupleobject.py / tupleobject.rs) —
    //     NOT `is_tuple()` (which also accepts the three
    //     SPECIALISED_TUPLE_{II,FF,OO} variants).  Specialised tuples store
    //     `value0`/`value1` inline with no `wrappeditems` block, so a
    //     `getfield(wrappeditems)` on one yields garbage.
    //   * `w_class == canonical tuple` — a tuple SUBCLASS instance shares the
    //     payload `ob_type == &TUPLE_TYPE` but retags `w_class` and may
    //     override `__getitem__`; `baseobjspace::getitem` honours that
    //     override (subclass_special_override) so the pure `wrappeditems[i]`
    //     load must NOT be taken for it.
    // A failing gate falls to the generic residual.  The paired runtime
    // `guard_class(&TUPLE_TYPE)` + exact `w_class` guard (in
    // `try_walker_specialize_subscr_tuple`) deopt any later non-canonical
    // tuple or subclass instance flowing in.
    let tuple_canonical = unsafe {
        std::ptr::eq((*list_obj).ob_type, &pyre_object::pyobject::TUPLE_TYPE)
            && std::ptr::eq(
                (*list_obj).w_class,
                pyre_object::pyobject::get_instantiate(&pyre_object::pyobject::TUPLE_TYPE),
            )
    };
    // Serve the arity-2 specialisations from the real reader.  The canonical
    // layout is deliberately NOT routed here: its items are an array, so the
    // descent inlines the whole reader at every subscript, and inside a
    // recursive bridge that overruns the bridge's trace budget --
    // `selfrec_bridge_nontail_promote` loses a bridge to `abrt_bridge` and runs
    // 2.4x slower.  `try_walker_specialize_subscr_tuple` keeps that arm.
    if specialised_pair_kind(unsafe { (*list_obj).ob_type }).is_some() {
        if let Some(hit) = spec_gate(SpecFold::SubscrTupleDescent, || {
            try_walker_orthodox_subscr_tuple_item(
                ctx, op_pc, list_op, key_op, list_obj, key_obj, dst, dst_bank,
            )
        })? {
            return Ok(Some(hit));
        }
    }

    if tuple_canonical {
        return spec_gate(SpecFold::SubscrTuple, || {
            try_walker_specialize_subscr_tuple(
                ctx, op_pc, list_op, key_op, list_obj, key_obj, allboxes, call_descr, dst, dst_bank,
            )
        });
    }

    // A `str` receiver reaches `descr_getitem`'s scalar arm, which boxes one
    // code point.  It is not a storage strategy like the list arms below, so
    // it gets its own emit rather than an element load: the payload is
    // variable-width UTF-8 and a fixed-stride read would be wrong the moment
    // the string is not ASCII.
    // `is_exact_type` only checks the shared payload `ob_type`; a str subclass
    // carries that same value and distinguishes itself through `w_class`.
    // Admit exactly the shape the replay guard below will pin, so recording a
    // subclass cannot manufacture a guard which its own concrete operand
    // already fails.
    if unsafe { pyre_object::is_str(list_obj) && walker_exact_builtin_class(list_obj).is_some() } {
        return spec_gate(SpecFold::SubscrStr, || {
            try_walker_specialize_subscr_str(
                ctx, op_pc, list_op, key_op, list_obj, key_obj, allboxes, call_descr, dst, dst_bank,
            )
        });
    }

    // The `dict.lookup` gate.  Both `w_class` checks are load-bearing: a dict
    // SUBCLASS shares `ob_type == &DICT_TYPE` but retags `w_class` and reaches
    // `__missing__` on a miss, and a str SUBCLASS key may override `__hash__` /
    // `__eq__`, so neither may take the exact-str probe.  The strategy check is
    // what makes the probe non-raising: `UnicodeDictStrategy` hands the dict to
    // `ObjectDictStrategy` the moment a non-exact-str key is stored, so while
    // it holds, every stored key is an exact str and the comparisons are WTF-8
    // byte equality (`dictmultiobject.py+` `r_dict(unicode_eq,
    // unicode_hash)`).
    let canonical_dict = pyre_object::pyobject::get_instantiate(&pyre_object::pyobject::DICT_TYPE);
    let dict_unicode = !canonical_dict.is_null()
        && unsafe {
            std::ptr::eq((*list_obj).ob_type, &pyre_object::pyobject::DICT_TYPE)
                && std::ptr::eq((*list_obj).w_class, canonical_dict)
                && pyre_object::dictmultiobject::w_dict_get_strategy(list_obj).strategy_kind()
                    == pyre_object::dictmultiobject::StrategyKind::Unicode
        };
    let canonical_str = if dict_unicode {
        pyre_object::pyobject::get_instantiate(&pyre_object::pyobject::STR_TYPE)
    } else {
        std::ptr::null_mut()
    };
    let dict_unicode_hit = !canonical_str.is_null()
        && unsafe {
            std::ptr::eq((*key_obj).ob_type, &pyre_object::pyobject::STR_TYPE)
                && std::ptr::eq((*key_obj).w_class, canonical_str)
        };
    if dict_unicode_hit {
        let hash = unsafe { pyre_object::dictmultiobject::w_dict_unicode_key_hash(key_obj) };
        let index = unsafe {
            pyre_object::dictmultiobject::w_dict_unicode_lookup_index(list_obj, key_obj, hash, 0)
        };
        if index < 0 {
            return Ok(None);
        }

        let Some(boxed_result_i64) = walker_execute_may_force_boxed(ctx, allboxes, call_descr)
        else {
            return Ok(None);
        };

        walker_guard_class(
            ctx,
            op_pc,
            list_op,
            &pyre_object::pyobject::DICT_TYPE as *const _ as i64,
        )?;
        walker_guard_exact_w_class(ctx, op_pc, list_op, canonical_dict)?;
        let strategy = crate::state::opimpl_getfield_gc_i(
            ctx.trace_ctx,
            list_op,
            crate::descr::dict_strategy_word_descr(),
        );
        let unicode_strategy_const = ctx
            .trace_ctx
            .const_int(&pyre_object::dictmultiobject::UNICODE_DICT_STRATEGY_REF as *const _ as i64);
        ctx.trace_ctx
            .record_guard(OpCode::GuardValue, &[strategy, unicode_strategy_const], 0);
        walker_capture_snapshot_for_last_guard(ctx, op_pc)?;
        ctx.trace_ctx
            .heap_cache_mut()
            .replace_box(strategy, unicode_strategy_const);
        walker_guard_class(
            ctx,
            op_pc,
            key_op,
            &pyre_object::pyobject::STR_TYPE as *const _ as i64,
        )?;
        walker_guard_exact_w_class(ctx, op_pc, key_op, canonical_str)?;

        let hash_effect = majit_ir::EffectInfo::new(
            majit_ir::ExtraEffect::ElidableCannotRaise,
            majit_ir::OopSpecIndex::None,
        );
        // Both residuals bind the macro-emitted `__majit_call_target_*`
        // trampoline rather than the raw fn: the wasm backend derives a
        // residual's `call_indirect` type from the descr alone — `(i64 x n) ->
        // i64` — so a raw `*mut PyObject` argument, `i32` on wasm32, traps
        // `indirect call type mismatch`. The trampoline takes and returns the
        // uniform machine word everywhere, and is the address `jit_fnaddr`
        // registers for these paths.
        let hash_fn = {
            let f: extern "C" fn(i64) -> i64 =
                pyre_object::dictmultiobject::__majit_call_target_w_dict_unicode_key_hash;
            f as *const ()
        };
        let hash_op = ctx.trace_ctx.call_typed_with_effect_pure(
            OpCode::CallI,
            hash_fn,
            &[key_op],
            &[majit_ir::Type::Ref],
            majit_ir::Type::Int,
            hash_effect,
            &[
                majit_ir::Value::Int(hash_fn as i64),
                majit_ir::Value::Ref(majit_ir::GcRef(key_obj as usize)),
            ],
            majit_ir::Value::Int(hash),
        );

        let mut lookup_effect = majit_ir::EffectInfo::new(
            majit_ir::ExtraEffect::CannotRaise,
            majit_ir::OopSpecIndex::DictLookup,
        );
        lookup_effect.extradescrs = Some(vec![
            crate::descr::dict_lookup_namespace_descr(),
            crate::descr::dict_lookup_entries_array_descr(),
        ]);
        let lookup_flag = ctx.trace_ctx.const_int(0);
        let lookup_fn: extern "C" fn(i64, i64, i64, i64) -> i64 =
            pyre_object::dictmultiobject::__majit_call_target_w_dict_unicode_lookup_index;
        let index_op = ctx.trace_ctx.call_typed_with_effect(
            OpCode::CallI,
            lookup_fn as *const (),
            &[list_op, key_op, hash_op, lookup_flag],
            &[
                majit_ir::Type::Ref,
                majit_ir::Type::Ref,
                majit_ir::Type::Int,
                majit_ir::Type::Int,
            ],
            majit_ir::Type::Int,
            lookup_effect,
        );
        ctx.trace_ctx
            .set_opref_concrete(index_op, majit_ir::Value::Int(index));

        let zero = ctx.trace_ctx.const_int(0);
        let nonneg = ctx.trace_ctx.record_op(OpCode::IntGe, &[index_op, zero]);
        ctx.trace_ctx
            .set_opref_concrete(nonneg, majit_ir::Value::Int(1));
        walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardTrue, &[nonneg])?;

        let value = ctx.trace_ctx.call_ref_typed_with_effect(
            crate::helpers::jit_dict_value_at as *const (),
            &[list_op, index_op, key_op, hash_op],
            &[
                majit_ir::Type::Ref,
                majit_ir::Type::Int,
                majit_ir::Type::Ref,
                majit_ir::Type::Int,
            ],
            majit_ir::EffectInfo::new(
                majit_ir::ExtraEffect::CannotRaise,
                majit_ir::OopSpecIndex::None,
            ),
        );
        walker_emit_fold_guard_with_snapshot(ctx, op_pc, OpCode::GuardNonnull, &[value])?;
        ctx.trace_ctx.set_opref_concrete(
            value,
            majit_ir::Value::Ref(majit_ir::GcRef(boxed_result_i64 as usize)),
        );
        write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, value)?;
        return Ok(Some(()));
    }

    // Gate: EXACT list[int], non-negative index in bounds, int- or
    // float-storage.  A bool index (`is_int` accepts `W_BoolObject`) is fine:
    // bool shares int's `intval`, so it unboxes through its own &BOOL_TYPE
    // guard below.  A list SUBCLASS instance shares `ob_type == &LIST_TYPE`
    // but retags `w_class` and may override `__getitem__`; `is_exact_list`
    // excludes it so it falls to the generic residual (which honours the
    // override) instead of this direct-storage load.
    let (sid, index, concrete_len) = unsafe {
        if !pyre_object::is_exact_list(list_obj) || !pyre_object::is_int(key_obj) {
            return Ok(None);
        }
        let index = pyre_object::w_int_get_value(key_obj);
        if index < 0 {
            return Ok(None);
        }
        let concrete_len = pyre_object::w_list_len(list_obj);
        if index as usize >= concrete_len {
            return Ok(None);
        }
        let sid = if pyre_object::w_list_uses_int_storage(list_obj) {
            1i64
        } else if pyre_object::w_list_uses_float_storage(list_obj) {
            2i64
        } else if pyre_object::w_list_uses_object_storage(list_obj) {
            0i64
        } else {
            // Empty-strategy list: no concrete element to read.
            return Ok(None);
        };
        (sid, index, concrete_len)
    };

    // Authentic boxed result from the same may-force path the generic leg uses.
    let Some(boxed_result_i64) = walker_execute_may_force_boxed(ctx, allboxes, call_descr) else {
        return Ok(None);
    };

    // --- emit the specialized IR (walker-native) ---
    // guard_class LIST (skip when class already known / operand is constant).
    let list_type_addr = &pyre_object::pyobject::LIST_TYPE as *const _ as i64;
    if !list_op.is_constant() && !ctx.trace_ctx.heap_cache().is_class_known(list_op) {
        let type_const = ctx.trace_ctx.const_int(list_type_addr);
        ctx.trace_ctx
            .record_guard(OpCode::GuardClass, &[list_op, type_const], 0);
        walker_capture_snapshot_for_last_guard(ctx, op_pc)?;
    }
    ctx.trace_ctx
        .heap_cache_mut()
        .class_now_known(list_op, list_type_addr);

    // A list SUBCLASS instance shares `ob_type == &LIST_TYPE` (so it passes
    // the GuardClass above) but retags `w_class` and may override
    // `__getitem__`; guard the exact canonical `w_class` so such an instance
    // side-exits to the generic residual (which honours the override) rather
    // than taking this direct-storage load.
    walker_guard_exact_w_class(
        ctx,
        op_pc,
        list_op,
        pyre_object::pyobject::get_instantiate(&pyre_object::pyobject::LIST_TYPE),
    )?;

    // guard_value(strategy == sid): getfield strategy + GuardValue + replace_box.
    let strategy = crate::state::opimpl_getfield_gc_i(
        ctx.trace_ctx,
        list_op,
        crate::descr::list_strategy_descr(),
    );
    let sid_const = ctx.trace_ctx.const_int(sid);
    ctx.trace_ctx
        .record_guard(OpCode::GuardValue, &[strategy, sid_const], 0);
    walker_capture_snapshot_for_last_guard(ctx, op_pc)?;
    ctx.trace_ctx
        .heap_cache_mut()
        .replace_box(strategy, sid_const);

    // Unbox the index operand (guard_class + getfield intval).  bool shares
    // int's `intval`, so a bool index guards its own &BOOL_TYPE.
    let (idx_type, idx_descr) = crate::state::int_or_bool_unbox_type_descr(key_obj);
    let raw_index = walker_unbox_int_typed(ctx, op_pc, key_op, idx_type, idx_descr)?;
    ctx.trace_ctx
        .set_opref_concrete(raw_index, majit_ir::Value::Int(index));

    // Object storage keeps the inline `length` field (rlist.py); int/float
    // storage read the typed items-array length field.
    let len_descr = match sid {
        0 => crate::descr::list_length_descr(),
        1 => crate::descr::list_int_items_len_descr(),
        _ => crate::descr::list_float_items_len_descr(),
    };
    let lenbox = crate::state::opimpl_getfield_gc_i(ctx.trace_ctx, list_op, len_descr);
    walker_emit_index_bounds_guards(ctx, op_pc, raw_index, index, lenbox, concrete_len)?;

    // Element load.  Object storage reads the boxed Ref directly from the
    // `Ptr(GcArray(OBJECTPTR))` items block (no unbox/rebox).  Int/float
    // storage read the raw typed array and rebox; the raw element is stamped
    // with the true value from the authentic may-force result (the in-array
    // sanity load is skipped when `items_ptr` is not trace-time concrete) so
    // the `wrapint` / `wrapfloat` box's cached field matches a later unbox.
    let result_obj = boxed_result_i64 as pyre_object::PyObjectRef;
    let default_concrete = majit_ir::Value::Ref(majit_ir::GcRef(boxed_result_i64 as usize));
    let (boxed, boxed_concrete) = match sid {
        0 => {
            let items_block = crate::state::opimpl_getfield_gc_r(
                ctx.trace_ctx,
                list_op,
                crate::descr::list_items_descr(),
            );
            (
                crate::state::trace_items_block_getitem_value(
                    ctx.trace_ctx,
                    items_block,
                    raw_index,
                ),
                default_concrete,
            )
        }
        1 => {
            let block = crate::state::opimpl_getfield_gc_r(
                ctx.trace_ctx,
                list_op,
                crate::descr::list_int_items_block_descr(),
            );
            let raw = crate::state::trace_int_block_getitem_value(ctx.trace_ctx, block, raw_index);
            let elem = unsafe { pyre_object::w_int_get_value(result_obj) };
            ctx.trace_ctx
                .set_opref_concrete(raw, majit_ir::Value::Int(elem));
            (
                walker_box_int(ctx, op_pc, raw, elem)?,
                box_int_concrete(elem, boxed_result_i64),
            )
        }
        _ => {
            let block = crate::state::opimpl_getfield_gc_r(
                ctx.trace_ctx,
                list_op,
                crate::descr::list_float_items_block_descr(),
            );
            let raw =
                crate::state::trace_float_block_getitem_value(ctx.trace_ctx, block, raw_index);
            let elem = unsafe { pyre_object::w_float_get_value(result_obj) };
            ctx.trace_ctx
                .set_opref_concrete(raw, majit_ir::Value::Float(elem));
            (
                crate::state::wrapfloat(ctx.trace_ctx, raw),
                default_concrete,
            )
        }
    };
    ctx.trace_ctx.set_opref_concrete(boxed, boxed_concrete);
    write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, boxed)?;
    Ok(Some(()))
}

/// #171/#11 Approach C, SUBSCRIPT slice: walker-native PURE element load
/// for a canonical array-backed `W_TupleObject[i]` (the tuple analogue of
/// the object-storage list arm of [`try_walker_specialize_subscr`]).
///
/// Recognition (caller already verified `ob_type == &TUPLE_TYPE`): a
/// non-negative int (or bool, which shares `intval`) index in bounds.
/// Specialised tuples never reach here — the caller gates them out — so
/// reading `wrappeditems` is always sound.
///
/// IR shape: `guard_class(&TUPLE_TYPE)` → `getfield(wrappeditems)` →
/// `arraylen_gc(wrappeditems)` for the bounds length → `IntLt` +
/// `GuardTrue` (NON-pure, so an out-of-range deopt still fires) →
/// `getarrayitem_gc_pure_r(wrappeditems, idx)` (the ONLY pure op; the
/// body is immutable per `_immutable_fields_ = ['wrappeditems[*]']`).
/// Object storage → the element is a boxed Ref read directly (no
/// unbox/rebox).  The authentic boxed result is taken from the same
/// `execute_may_force` path the generic leg uses.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_walker_specialize_subscr_tuple<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    list_op: OpRef,
    key_op: OpRef,
    tuple_obj: pyre_object::PyObjectRef,
    key_obj: pyre_object::PyObjectRef,
    allboxes: &[OpRef],
    call_descr: &dyn majit_ir::descr::CallDescr,
    dst: usize,
    dst_bank: char,
) -> Result<Option<()>, DispatchError> {
    // Gate: non-negative int index in bounds.  `w_tuple_len` reads the
    // GcArray header of `wrappeditems` (no inline length field).
    let (index, concrete_len) = unsafe {
        if !pyre_object::is_int(key_obj) {
            return Ok(None);
        }
        let index = pyre_object::w_int_get_value(key_obj);
        if index < 0 {
            return Ok(None);
        }
        let concrete_len = pyre_object::w_tuple_len(tuple_obj);
        if index as usize >= concrete_len {
            return Ok(None);
        }
        (index, concrete_len)
    };

    // Authentic boxed result from the same may-force path the generic leg uses.
    let Some(boxed_result_i64) = walker_execute_may_force_boxed(ctx, allboxes, call_descr) else {
        return Ok(None);
    };

    // --- emit the specialized IR (walker-native) ---
    // guard_class TUPLE (skip when class already known / operand is constant).
    let tuple_type_addr = &pyre_object::pyobject::TUPLE_TYPE as *const _ as i64;
    if !list_op.is_constant() && !ctx.trace_ctx.heap_cache().is_class_known(list_op) {
        let type_const = ctx.trace_ctx.const_int(tuple_type_addr);
        ctx.trace_ctx
            .record_guard(OpCode::GuardClass, &[list_op, type_const], 0);
        walker_capture_snapshot_for_last_guard(ctx, op_pc)?;
    }
    ctx.trace_ctx
        .heap_cache_mut()
        .class_now_known(list_op, tuple_type_addr);

    // A tuple SUBCLASS instance shares `ob_type == &TUPLE_TYPE` (so it passes
    // the GuardClass above) but retags `w_class` and may override
    // `__getitem__`; guard the exact canonical `w_class` so such an instance
    // side-exits to the generic residual (which honours the override) rather
    // than taking this pure `wrappeditems[i]` load.
    walker_guard_exact_w_class(
        ctx,
        op_pc,
        list_op,
        pyre_object::pyobject::get_instantiate(&pyre_object::pyobject::TUPLE_TYPE),
    )?;

    // Unbox the index operand (guard_class + getfield intval).  bool shares
    // int's `intval`, so a bool index guards its own &BOOL_TYPE.
    let (idx_type, idx_descr) = crate::state::int_or_bool_unbox_type_descr(key_obj);
    let raw_index = walker_unbox_int_typed(ctx, op_pc, key_op, idx_type, idx_descr)?;
    ctx.trace_ctx
        .set_opref_concrete(raw_index, majit_ir::Value::Int(index));

    // getfield(wrappeditems): Ptr(GcArray(OBJECTPTR)) body.
    let items_block = crate::state::opimpl_getfield_gc_r(
        ctx.trace_ctx,
        list_op,
        crate::descr::tuple_wrappeditems_descr(),
    );

    // Bounds length: arraylen_gc against the wrappeditems GcArray header
    // (no inline length cache).  NON-pure: an out-of-range index must
    // still deopt.
    let lenbox = crate::state::opimpl_arraylen_gc(
        ctx.trace_ctx,
        items_block,
        crate::state::pyobject_gcarray_descr(),
    );
    walker_emit_index_bounds_guards(ctx, op_pc, raw_index, index, lenbox, concrete_len)?;

    // PURE element load.  Object storage reads the boxed Ref directly from
    // the immutable `Ptr(GcArray(OBJECTPTR))` body (no unbox/rebox).
    let boxed =
        crate::state::trace_items_block_getitem_value_pure(ctx.trace_ctx, items_block, raw_index);
    ctx.trace_ctx.set_opref_concrete(
        boxed,
        majit_ir::Value::Ref(majit_ir::GcRef(boxed_result_i64 as usize)),
    );
    write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, boxed)?;
    Ok(Some(()))
}

/// Descend `w_tuple_getitem`'s compiled body for a tuple subscript whose
/// receiver class and item index are both known at trace time, instead of
/// re-emitting that body's length test and field reads by hand.
///
/// This is the orthodox shape, and it replaced the hand-written
/// `subscr_specialised_pair` reader that stood in for it:
/// upstream's `getitem` is an ordinary graph the tracer inlines
/// (`specialisedtupleobject.py`, whose `getitem` unrolls `iter_n` to the
/// matching `value%s`), and the callee's `ob_type` chain folds against the
/// pinned class down to the one specialisation this trace saw.
///
/// The trace-time range check here is a decline gate, not the trace's safety
/// argument: it keeps an out-of-range subscript on the generic residual instead
/// of tracing a raising path.  What holds for the *next* receiver is the
/// callee's own length test, which is why this enters at `w_tuple_getitem`
/// rather than at the `_known` reader it wraps -- the reader is documented
/// "known-in-bounds" and carries no test to record.
fn try_walker_orthodox_subscr_tuple_item<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    seq_op: OpRef,
    key_op: OpRef,
    seq_obj: pyre_object::PyObjectRef,
    key_obj: pyre_object::PyObjectRef,
    dst: usize,
    dst_bank: char,
) -> Result<Option<()>, DispatchError> {
    // Arity-2 specialisations only -- see the caller for why the canonical
    // layout keeps its own arm.
    let spec_type = unsafe { (*seq_obj).ob_type };
    if specialised_pair_kind(spec_type).is_none() {
        return Ok(None);
    }
    // Exact int keys only: a slice, a bool or an int subclass reaches a
    // different objspace path, and the callee takes a machine index.
    if !unsafe { pyre_object::is_int(key_obj) } {
        return Ok(None);
    }
    let raw_key = unsafe { pyre_object::w_int_get_value(key_obj) };
    let len = unsafe { pyre_object::tupleobject::w_tuple_len(seq_obj) } as i64;
    let index = if raw_key < 0 { raw_key + len } else { raw_key };
    if !(0..len).contains(&index) {
        return Ok(None);
    }

    // Resolve every possible decline before recording a guard.
    let Some(jc_arc) = crate::jitcode_runtime::tuple_getitem_jitcode() else {
        return Ok(None);
    };
    let Some(sub_body) = sub_jitcode_body_by_index(jc_arc.index()) else {
        return Ok(None);
    };
    let sym_ptr = ctx.fbw_mode.snapshot_sym;
    if sym_ptr.is_null() {
        return Ok(None);
    }
    // SAFETY: set for the lifetime of the enclosing full-body walk.
    if unsafe { (&*sym_ptr).jitcode().is_null() } {
        return Ok(None);
    }
    let sym = unsafe { &*sym_ptr };

    let pre_fold_pos = ctx.trace_ctx.get_trace_position();
    // Only a specialisation carries its own `ob_type`, so the class guard below
    // is the whole precondition: a tuple subclass keeps `&TUPLE_TYPE` and can
    // never reach these arms, and each specialisation's length is 2 by
    // construction.
    walker_guard_specialised_pair_class(ctx, op_pc, seq_op, spec_type)?;

    // Freeze the key: the two slots are separate fields, so the callee's
    // `match idx` folds to one of them only against a constant.
    let (idx_type, idx_descr) = crate::state::int_or_bool_unbox_type_descr(key_obj);
    let key_index = walker_unbox_int_typed(ctx, op_pc, key_op, idx_type, idx_descr)?;
    ctx.trace_ctx
        .set_opref_concrete(key_index, majit_ir::Value::Int(raw_key));
    let index_arg = ctx.trace_ctx.const_int(raw_key);
    walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardValue, &[key_index, index_arg])?;
    ctx.trace_ctx
        .heap_cache_mut()
        .replace_box(key_index, index_arg);
    ctx.trace_ctx.set_opref_concrete(
        seq_op,
        majit_ir::Value::Ref(majit_ir::GcRef(seq_obj as usize)),
    );
    let walk = run_orthodox_helper_subwalk(
        ctx,
        op_pc,
        sym,
        &sub_body,
        "subscr_tuple_item_commit",
        "w_tuple_getitem_known_call_site",
        &[index_arg],
        &[ConcreteValue::Int(raw_key)],
        &[seq_op],
        &[ConcreteValue::Ref(seq_obj)],
    );
    let (walk_outcome, _walk_start) = match walk {
        Ok(pair) => pair,
        // The body reached a helper this build did not lower.  Nothing is
        // committed yet -- the read has no effect to undo -- so cut the
        // tentative IR and let the generic residual serve the subscript.
        //
        // The snapshots go with it: the class guard and the `GuardValue` on
        // the frozen key are both emitted above with snapshots attached, and
        // those name the discarded operation namespace, so leaving them in
        // the side table exposes stale boxes once a later optimizer remaps
        // every published snapshot.
        Err(DispatchError::OrthodoxSubWalkTraceUnsupported { pc, .. }) => {
            if fbw_debug_abort_enabled() {
                eprintln!("[decline-why] SUBSCR-TUPLE-SUBWALK pc={pc}");
            }
            ctx.trace_ctx.cut_trace_with_snapshots(pre_fold_pos);
            ctx.trace_ctx.heap_cache_mut().reset();
            return Ok(None);
        }
        Err(error) => return Err(error),
    };
    let result = match walk_outcome {
        DispatchOutcome::SubReturn { result } => finish_inline_callee_return(ctx, result)
            .ok_or(DispatchError::UnexpectedVoidSubReturn { pc: op_pc })?,
        _ => return Err(DispatchError::UnexpectedVoidSubReturn { pc: op_pc }),
    };
    write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, result)?;
    Ok(Some(()))
}

/// Descend `invert_inner`'s compiled body for `~x` on an exact builtin `int` or
/// `long`, instead of re-emitting its arms by hand.
///
/// This is the orthodox shape: upstream traces *through* `descr_invert`, which
/// is ordinary RPython, rather than carrying a fold per operand type.  What
/// makes it worth entering here is coverage, not the emitted shape -- the
/// hand-written `unary_invert_int` answers the exact-`int` operand only, and
/// `~` on a `long` reaches no fold at all today (measured: the fold is
/// consulted for both and fires for one).  The body's own arms cover both, so
/// descending it covers the second without a second fold.
///
/// `invert` itself cannot be entered.  Its `__invert__` override probe is
/// `dont_look_inside` and is the second operation the body executes, and its
/// bool slot raises a deprecation warning, which reaches `lookup_exc_class`.
/// `invert_inner` is that body past both.  The guards emitted here are what
/// let the caller skip them: `bool` carries its own type, and an `int` or
/// `long` subclass keeps the builtin `ob_type` but retags `w_class` and may
/// define `__invert__`, so both must side-exit to the residual, which still
/// runs the whole of `invert`.
pub(crate) fn try_walker_orthodox_unary_invert<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    operand: OpRef,
    dst: usize,
    dst_bank: char,
) -> Result<Option<()>, DispatchError> {
    let Some(operand_obj) = walker_concrete_ref_object(ctx, operand) else {
        return Ok(None);
    };
    // SAFETY: `operand_obj` is a live concrete `PyObjectRef` from the walker
    // shadow.
    let is_long = unsafe { pyre_object::is_long(operand_obj) };
    let admitted = unsafe {
        !pyre_object::is_bool(operand_obj)
            && (is_long || pyre_object::is_int(operand_obj))
            && pyre_object::is_exact_builtin_instance(operand_obj)
    };
    if !admitted {
        return Ok(None);
    }
    // SAFETY: as above.
    //
    // This cannot decline for an admitted operand.  `walker_exact_builtin_class`
    // answers `None` for an exact builtin whose `w_class` is null, and the only
    // objects born that way are the read-only singletons (`True`, `False`,
    // `None`, `Ellipsis`, `NotImplemented`) -- every other builtin is born
    // carrying `get_instantiate(ob_type)`.  None of those five survives the
    // admission above: the first two are `bool`, the rest are not `int`.
    let Some(operand_class) = (unsafe { walker_exact_builtin_class(operand_obj) }) else {
        return Ok(None);
    };

    // Resolve every possible decline before recording a guard.
    let Some(jc_arc) = crate::jitcode_runtime::invert_inner_jitcode() else {
        return Ok(None);
    };
    let Some(sub_body) = sub_jitcode_body_by_index(jc_arc.index()) else {
        return Ok(None);
    };
    let sym_ptr = ctx.fbw_mode.snapshot_sym;
    if sym_ptr.is_null() {
        return Ok(None);
    }
    // SAFETY: set for the lifetime of the enclosing full-body walk.
    if unsafe { (&*sym_ptr).jitcode().is_null() } {
        return Ok(None);
    }
    let sym = unsafe { &*sym_ptr };

    let pre_fold_pos = ctx.trace_ctx.get_trace_position();
    let type_addr = if is_long {
        &pyre_object::pyobject::LONG_TYPE as *const _ as i64
    } else {
        &pyre_object::pyobject::INT_TYPE as *const _ as i64
    };
    walker_guard_class(ctx, op_pc, operand, type_addr)?;
    walker_guard_exact_w_class(ctx, op_pc, operand, operand_class)?;
    ctx.trace_ctx.set_opref_concrete(
        operand,
        majit_ir::Value::Ref(majit_ir::GcRef(operand_obj as usize)),
    );

    let walk = run_orthodox_helper_subwalk(
        ctx,
        op_pc,
        sym,
        &sub_body,
        "unary_invert_commit",
        "invert_inner_call_site",
        &[],
        &[],
        &[operand],
        &[ConcreteValue::Ref(operand_obj)],
    );
    let (walk_outcome, _walk_start) = match walk {
        // The body reached a helper this build did not lower.  `invert_inner`
        // is a pure read on both admitted arms, so nothing is committed --
        // cut the tentative IR, with its snapshots, and let the residual serve
        // the operator.  The two guards above are emitted with snapshots
        // attached and name the discarded operation namespace, so leaving them
        // behind would expose stale boxes to a later remap.
        Err(DispatchError::OrthodoxSubWalkTraceUnsupported { pc, .. }) => {
            if fbw_debug_abort_enabled() {
                eprintln!("[decline-why] UNARY-INVERT-SUBWALK pc={pc}");
            }
            ctx.trace_ctx.cut_trace_with_snapshots(pre_fold_pos);
            ctx.trace_ctx.heap_cache_mut().reset();
            return Ok(None);
        }
        Ok(pair) => pair,
        Err(error) => return Err(error),
    };
    let result = match walk_outcome {
        DispatchOutcome::SubReturn { result } => finish_inline_callee_return(ctx, result)
            .ok_or(DispatchError::UnexpectedVoidSubReturn { pc: op_pc })?,
        _ => return Err(DispatchError::UnexpectedVoidSubReturn { pc: op_pc }),
    };
    write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, result)?;
    Ok(Some(()))
}

/// Descend `neg_inner`'s compiled body for `-x` on an exact builtin `int` or
/// `long`, instead of re-emitting its integer arm by hand.
///
/// The same orthodox shape as [`try_walker_orthodox_unary_invert`], and for the
/// same reason: upstream traces *through* `descr_neg`, which is ordinary
/// RPython.
///
/// The descent also owns the `INT_MIN` promotion.  `rbigint.neg` stays an
/// `EF_ELIDABLE_OR_MEMORYERROR` residual, while `W_LongObject.__init__` lowers
/// to `NewWithVtable` plus its ordinary `w_class` and `value` field writes.
/// This is the same allocation/body split PyPy's rtyper and metainterp expose;
/// no unary-negative manual fold remains beside it.
///
/// `neg` itself cannot be entered: its `__neg__` override probe is
/// `dont_look_inside` and is the second operation the body executes.
/// `neg_inner` is that body past it. Unlike `invert`, `neg` has no bool slot to
/// step over, so the split leaves the probe alone.
///
/// The guards emitted here are what let the caller skip the probe: an `int` or
/// `long` subclass keeps the builtin `ob_type` but retags `w_class` and may
/// define `__neg__`, so it must side-exit to the residual, which still runs the
/// whole of `neg`. `bool` is excluded for a second reason as well -- see the
/// `walker_exact_builtin_class` read below.
pub(crate) fn try_walker_orthodox_unary_negative<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    operand: OpRef,
    dst: usize,
    dst_bank: char,
) -> Result<Option<()>, DispatchError> {
    let Some(operand_obj) = walker_concrete_ref_object(ctx, operand) else {
        return Ok(None);
    };
    // SAFETY: `operand_obj` is a live concrete `PyObjectRef` from the walker
    // shadow.
    let is_long = unsafe { pyre_object::is_long(operand_obj) };
    let admitted = unsafe {
        !pyre_object::is_bool(operand_obj)
            && (is_long || pyre_object::is_int(operand_obj))
            && pyre_object::is_exact_builtin_instance(operand_obj)
    };
    if !admitted {
        return Ok(None);
    }
    // SAFETY: as above.
    //
    // This cannot decline for an admitted operand, by the argument
    // [`try_walker_orthodox_unary_invert`] gives: the only exact builtins born
    // with a null `w_class` are the five read-only singletons, and the
    // admission above already rejects every one of them.
    let Some(operand_class) = (unsafe { walker_exact_builtin_class(operand_obj) }) else {
        return Ok(None);
    };

    // Resolve every possible decline before recording a guard.
    //
    // `-INT_MIN` is the one operand `descr_neg` promotes, and it belongs to
    // [`try_walker_specialize_unary_negative_int`] rather than to this walk.
    // Walking the promotion records the two `rbigint` calls that build the
    // `2**63` long; both survive optimization as `CallPureR` short boxes and
    // the second one's result is carried across the `LABEL` as a loop
    // argument, so a later pure call reading that long -- `compare_op_long`'s
    // `jit_bigint_cmp` -- has no constant argument and is re-emitted into the
    // body as an impure `CallI` once per iteration.  The fold pins the unboxed
    // operand with `guard_value` instead, and then the whole chain is
    // constant: it exports no short box at all and the comparison folds away
    // (`unary_negative.py main_int_min`, 20 ops / 9 guards -> 17 / 8).
    if let Some((x, _)) = walker_unary_int_operand(ctx, operand)
        && x == i64::MIN
    {
        return Ok(None);
    }
    let Some(jc_arc) = crate::jitcode_runtime::neg_inner_jitcode() else {
        return Ok(None);
    };
    let Some(sub_body) = sub_jitcode_body_by_index(jc_arc.index()) else {
        return Ok(None);
    };
    let sym_ptr = ctx.fbw_mode.snapshot_sym;
    if sym_ptr.is_null() {
        return Ok(None);
    }
    // SAFETY: set for the lifetime of the enclosing full-body walk.
    if unsafe { (&*sym_ptr).jitcode().is_null() } {
        return Ok(None);
    }
    let sym = unsafe { &*sym_ptr };

    let pre_fold_pos = ctx.trace_ctx.get_trace_position();
    let type_addr = if is_long {
        &pyre_object::pyobject::LONG_TYPE as *const _ as i64
    } else {
        &pyre_object::pyobject::INT_TYPE as *const _ as i64
    };
    walker_guard_class(ctx, op_pc, operand, type_addr)?;
    walker_guard_exact_w_class(ctx, op_pc, operand, operand_class)?;
    ctx.trace_ctx.set_opref_concrete(
        operand,
        majit_ir::Value::Ref(majit_ir::GcRef(operand_obj as usize)),
    );

    let walk = run_orthodox_helper_subwalk(
        ctx,
        op_pc,
        sym,
        &sub_body,
        "unary_negative_commit",
        "neg_inner_call_site",
        &[],
        &[],
        &[operand],
        &[ConcreteValue::Ref(operand_obj)],
    );
    let (walk_outcome, _walk_start) = match walk {
        // The body reached a helper this build did not lower. Both admitted
        // arms of `neg_inner` are pure reads that allocate their result, so
        // nothing is committed -- cut the tentative IR, with its snapshots, and
        // let the residual serve the operator. The two guards above are emitted
        // with snapshots attached and name the discarded operation namespace,
        // so leaving them behind would expose stale boxes to a later remap.
        Err(DispatchError::OrthodoxSubWalkTraceUnsupported { pc, .. }) => {
            if fbw_debug_abort_enabled() {
                eprintln!("[decline-why] UNARY-NEGATIVE-SUBWALK pc={pc}");
            }
            ctx.trace_ctx.cut_trace_with_snapshots(pre_fold_pos);
            ctx.trace_ctx.heap_cache_mut().reset();
            return Ok(None);
        }
        Ok(pair) => pair,
        Err(error) => return Err(error),
    };
    let result = match walk_outcome {
        DispatchOutcome::SubReturn { result } => finish_inline_callee_return(ctx, result)
            .ok_or(DispatchError::UnexpectedVoidSubReturn { pc: op_pc })?,
        _ => return Err(DispatchError::UnexpectedVoidSubReturn { pc: op_pc }),
    };
    write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, result)?;
    Ok(Some(()))
}

/// Descend `pos_inner`'s compiled body for `+x` on an exact builtin `int` or
/// `long`, instead of re-emitting its identity arm by hand.
///
/// The same orthodox shape as [`try_walker_orthodox_unary_invert`] and
/// [`try_walker_orthodox_unary_negative`].  Upstream traces *through*
/// `descr_pos`.  The exact-int and exact-long arms are identity: they return
/// the operand, matching `_self_unaryop('pos')`.
///
/// `pos` itself cannot be entered: its `__pos__` override probe is
/// `dont_look_inside` and is the second operation the body executes.
/// `pos_inner` is that body past it. Unlike `invert`, `pos` has no bool slot
/// to step over, so the split leaves the probe alone.
///
/// The guards emitted here are what let the caller skip the probe: an `int` or
/// `long` subclass keeps the builtin `ob_type` but retags `w_class` and may
/// define `__pos__`, so it must side-exit to the residual, which still runs the
/// whole of `pos`. `bool` is excluded because `+True` is a rewrapping to a
/// plain int, not identity, and because [`walker_exact_builtin_class`] has no
/// value to pin on a singleton.
pub(crate) fn try_walker_orthodox_unary_positive<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    operand: OpRef,
    dst: usize,
    dst_bank: char,
) -> Result<Option<()>, DispatchError> {
    let Some(operand_obj) = walker_concrete_ref_object(ctx, operand) else {
        return Ok(None);
    };
    // SAFETY: `operand_obj` is a live concrete `PyObjectRef` from the walker
    // shadow.
    let is_long = unsafe { pyre_object::is_long(operand_obj) };
    let admitted = unsafe {
        !pyre_object::is_bool(operand_obj)
            && (is_long || pyre_object::is_int(operand_obj))
            && pyre_object::is_exact_builtin_instance(operand_obj)
    };
    if !admitted {
        return Ok(None);
    }
    // SAFETY: as above.
    //
    // This cannot decline for an admitted operand, by the argument
    // [`try_walker_orthodox_unary_invert`] gives: the only exact builtins born
    // with a null `w_class` are the five read-only singletons, and the
    // admission above already rejects every one of them.
    let Some(operand_class) = (unsafe { walker_exact_builtin_class(operand_obj) }) else {
        return Ok(None);
    };

    // Resolve every possible decline before recording a guard.
    let Some(jc_arc) = crate::jitcode_runtime::pos_inner_jitcode() else {
        return Ok(None);
    };
    let Some(sub_body) = sub_jitcode_body_by_index(jc_arc.index()) else {
        return Ok(None);
    };
    let sym_ptr = ctx.fbw_mode.snapshot_sym;
    if sym_ptr.is_null() {
        return Ok(None);
    }
    // SAFETY: set for the lifetime of the enclosing full-body walk.
    if unsafe { (&*sym_ptr).jitcode().is_null() } {
        return Ok(None);
    }
    let sym = unsafe { &*sym_ptr };

    let pre_fold_pos = ctx.trace_ctx.get_trace_position();
    let type_addr = if is_long {
        &pyre_object::pyobject::LONG_TYPE as *const _ as i64
    } else {
        &pyre_object::pyobject::INT_TYPE as *const _ as i64
    };
    walker_guard_class(ctx, op_pc, operand, type_addr)?;
    walker_guard_exact_w_class(ctx, op_pc, operand, operand_class)?;
    ctx.trace_ctx.set_opref_concrete(
        operand,
        majit_ir::Value::Ref(majit_ir::GcRef(operand_obj as usize)),
    );

    let walk = run_orthodox_helper_subwalk(
        ctx,
        op_pc,
        sym,
        &sub_body,
        "unary_positive_commit",
        "pos_inner_call_site",
        &[],
        &[],
        &[operand],
        &[ConcreteValue::Ref(operand_obj)],
    );
    let (walk_outcome, _walk_start) = match walk {
        // The body reached a helper this build did not lower. Both admitted
        // arms of `pos_inner` are identity reads, so nothing is committed --
        // cut the tentative IR, with its snapshots, and let the residual (or
        // the identity fold behind this descent) serve the operator. The two
        // guards above are emitted with snapshots attached and name the
        // discarded operation namespace, so leaving them behind would expose
        // stale boxes to a later remap.
        Err(DispatchError::OrthodoxSubWalkTraceUnsupported { pc, .. }) => {
            if fbw_debug_abort_enabled() {
                eprintln!("[decline-why] UNARY-POSITIVE-SUBWALK pc={pc}");
            }
            ctx.trace_ctx.cut_trace_with_snapshots(pre_fold_pos);
            ctx.trace_ctx.heap_cache_mut().reset();
            return Ok(None);
        }
        Ok(pair) => pair,
        Err(error) => return Err(error),
    };
    let result = match walk_outcome {
        DispatchOutcome::SubReturn { result } => finish_inline_callee_return(ctx, result)
            .ok_or(DispatchError::UnexpectedVoidSubReturn { pc: op_pc })?,
        _ => return Err(DispatchError::UnexpectedVoidSubReturn { pc: op_pc }),
    };
    write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, result)?;
    Ok(Some(()))
}

/// `s[i]` on an exact `str` with an exact machine-`int` index: emit the
/// guarded unbox plus one elidable [`pyre_object::jit_str_getitem`] call
/// instead of the opaque `bh_binary_op_fn` residual.
///
/// The residual it replaces is a `CallMayForce`, which forces virtualizables
/// and clears the heap cache across itself; measured against an otherwise
/// identical loop over a `list` receiver, whose storage arm below already
/// folds, the str form costs an order of magnitude more per iteration than
/// the one boxed code point it produces.
///
/// Both operands are guarded exactly. A `str` subclass may override
/// `__getitem__`, which `baseobjspace::getitem` honours, and `bool` shares
/// `int`'s `intval` while indexing as 0/1 through its own type — the same
/// pair of reasons the tuple arm states. The helper declines a negative or
/// out-of-range index with `PY_NULL`, so `IndexError` and `__index__`
/// coercion stay in the interpreter; the trailing non-null guard carries that
/// decline back. Any other shape falls through to the generic residual (SAFE).
fn try_walker_specialize_subscr_str<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    seq_op: OpRef,
    key_op: OpRef,
    seq_obj: pyre_object::PyObjectRef,
    key_obj: pyre_object::PyObjectRef,
    allboxes: &[OpRef],
    call_descr: &dyn majit_ir::descr::CallDescr,
    dst: usize,
    dst_bank: char,
) -> Result<Option<()>, DispatchError> {
    if dst_bank != 'r' {
        return Ok(None);
    }
    // A tagged immediate has no header for the `w_class` and unbox guards to
    // read, and this emit is not tag-aware.
    if pyre_object::tagged_int::CAN_BE_TAGGED && pyre_object::tagged_int::is_tagged_int(key_obj) {
        return Ok(None);
    }
    let int_typeobj = pyre_object::pyobject::get_instantiate(&pyre_object::pyobject::INT_TYPE);
    let index = unsafe {
        if !std::ptr::eq((*key_obj).ob_type, &pyre_object::pyobject::INT_TYPE)
            || !std::ptr::eq((*key_obj).w_class, int_typeobj)
        {
            return Ok(None);
        }
        pyre_object::w_int_get_value(key_obj)
    };
    if index < 0 {
        return Ok(None);
    }
    // What the helper would box, read without boxing it: the helper allocates,
    // and a nursery collection under it could move `seq_obj` and the result,
    // which this frame holds as raw pointers with no root scope.
    //
    // A receiver whose payload is not valid UTF-8 -- a lone surrogate --
    // declines here, which is also what keeps `chars().nth` (a Rust `char`
    // index) equal to the code-point index the helper uses.
    let Some(expected) = (unsafe {
        pyre_object::w_str_get_value_opt(seq_obj).and_then(|text| text.chars().nth(index as usize))
    }) else {
        return Ok(None);
    };
    let Some(boxed_result_i64) = walker_execute_may_force_boxed(ctx, allboxes, call_descr) else {
        return Ok(None);
    };
    let boxed_result = boxed_result_i64 as pyre_object::PyObjectRef;
    let boxes_the_same = unsafe {
        pyre_object::is_exact_type(boxed_result, &pyre_object::STR_TYPE)
            && pyre_object::w_str_get_value_opt(boxed_result)
                .is_some_and(|text| text.chars().eq(std::iter::once(expected)))
    };
    if !boxes_the_same {
        return Ok(None);
    }

    // --- emit the specialized IR (walker-native) ---
    let str_type_addr = &pyre_object::pyobject::STR_TYPE as *const _ as i64;
    let str_typeobj = pyre_object::pyobject::get_instantiate(&pyre_object::pyobject::STR_TYPE);
    walker_guard_class(ctx, op_pc, seq_op, str_type_addr)?;
    walker_guard_exact_w_class(ctx, op_pc, seq_op, str_typeobj)?;
    let int_type_addr = &pyre_object::pyobject::INT_TYPE as *const _ as i64;
    walker_guard_class(ctx, op_pc, key_op, int_type_addr)?;
    walker_guard_exact_w_class(ctx, op_pc, key_op, int_typeobj)?;
    let index_raw = walker_unbox_int_typed(
        ctx,
        op_pc,
        key_op,
        int_type_addr,
        crate::descr::int_intval_descr(),
    )?;
    let helper = pyre_object::unicodeobject::jit_str_getitem as *const ();
    let raw = ctx.trace_ctx.call_typed_with_effect_pure_can_raise(
        OpCode::CallR,
        helper,
        &[seq_op, index_raw],
        &[majit_ir::Type::Ref, majit_ir::Type::Int],
        majit_ir::Type::Ref,
        // The helper's own `#[majit_macros::elidable_or_memerror]`: pure but
        // allocating, so the call carries a gcmap and the trailing
        // `GuardNoException` makes the allocation's raise leg observable.
        // Recording it pure lets the optimizer share one call between two
        // `s[i]` sites on the same pair, which is unobservable for this
        // result: it is a single code point, and `is_w` unique-ifies a `str`
        // of `_len() <= 1` by WTF-8 equality, so two separate allocations
        // already answer `is` exactly as one shared box does.
        majit_metainterp::ELIDABLE_OR_MEMERROR_EFFECT_INFO,
        &[
            majit_ir::Value::Int(helper as usize as i64),
            majit_ir::Value::Ref(majit_ir::GcRef(seq_obj as usize)),
            majit_ir::Value::Int(index),
        ],
        majit_ir::Value::Ref(majit_ir::GcRef(boxed_result as usize)),
    );
    // Concrete before the guards: a guard captures a resume snapshot, and a
    // `raw` with no value yet is recorded into it without one.
    ctx.trace_ctx.set_opref_concrete(
        raw,
        majit_ir::Value::Ref(majit_ir::GcRef(boxed_result as usize)),
    );
    if raw.inline_const_to_value().is_none() {
        walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardNoException, &[])?;
    }
    walker_emit_fold_guard_with_snapshot(ctx, op_pc, OpCode::GuardNonnull, &[raw])?;
    write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, raw)?;
    Ok(Some(()))
}
fn is_builtin_dict_get_function(callable: pyre_object::PyObjectRef) -> bool {
    if callable.is_null() || !unsafe { pyre_interpreter::is_function(callable) } {
        return false;
    }
    let code = unsafe { pyre_interpreter::function_get_code(callable) } as pyre_object::PyObjectRef;
    !code.is_null()
        && unsafe { pyre_interpreter::is_builtin_code(code) }
        && unsafe { pyre_interpreter::builtin_code_get(code) as usize }
            == pyre_interpreter::type_methods::dict_method_get as *const () as usize
}

#[derive(Clone, Copy)]
enum DictFoldKeyProbe {
    Int,
    Unicode,
}

#[derive(Clone, Copy)]
struct DictFoldHit {
    concrete_value: pyre_object::PyObjectRef,
    key_probe: DictFoldKeyProbe,
}

fn walker_probe_exact_dict_hit(
    dict: pyre_object::PyObjectRef,
    key: pyre_object::PyObjectRef,
) -> Result<Option<DictFoldHit>, DispatchError> {
    let canonical_dict = pyre_object::get_instantiate(&pyre_object::pyobject::DICT_TYPE);
    if canonical_dict.is_null()
        || !unsafe {
            std::ptr::eq((*dict).ob_type, &pyre_object::pyobject::DICT_TYPE)
                && std::ptr::eq((*dict).w_class, canonical_dict)
        }
    {
        return Ok(None);
    }

    // Only the two homogeneous strategies fold: their lookups probe a native
    // table and so cannot run Python-level `__hash__` or `__eq__`.  Every other
    // strategy, a mapdict-backed instance dict included, keeps the real lookup.
    let strategy_kind =
        unsafe { pyre_object::dictmultiobject::w_dict_get_strategy(dict).strategy_kind() };
    let int_probe = strategy_kind == pyre_object::dictmultiobject::StrategyKind::Int
        && unsafe { pyre_object::listobject::is_plain_int1(key) && pyre_object::is_int(key) };
    let unicode_probe = strategy_kind == pyre_object::dictmultiobject::StrategyKind::Unicode
        && unsafe {
            pyre_object::is_exact_type(key, &pyre_object::pyobject::STR_TYPE)
                && pyre_object::w_str_get_value_opt(key).is_some()
                && pyre_object::dict_eq_hook::try_hash_str(
                    pyre_object::w_str_get_value_opt(key).unwrap().as_bytes(),
                )
                .is_some()
        };

    let found = if int_probe {
        let index =
            unsafe { pyre_object::dictmultiobject::w_dict_index_of_int_strategy(dict, key) };
        index.and_then(|index| {
            unsafe { pyre_object::dictmultiobject::w_dict_nth_value(dict, index) }
                .map(|value| (value, DictFoldKeyProbe::Int))
        })
    } else if unicode_probe {
        let index =
            unsafe { pyre_object::dictmultiobject::w_dict_index_of_unicode_strategy(dict, key) };
        index.and_then(|index| {
            unsafe { pyre_object::dictmultiobject::w_dict_nth_value(dict, index) }
                .map(|value| (value, DictFoldKeyProbe::Unicode))
        })
    } else {
        None
    };

    let Some((concrete_value, key_probe)) = found else {
        return Ok(None);
    };
    Ok(Some(DictFoldHit {
        concrete_value,
        key_probe,
    }))
}

fn walker_emit_exact_dict_hit<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    dict_op: OpRef,
    key_op: OpRef,
    dict: pyre_object::PyObjectRef,
    hit: DictFoldHit,
    dst: usize,
    dst_bank: char,
) -> Result<Option<()>, DispatchError> {
    let canonical_dict = pyre_object::get_instantiate(&pyre_object::pyobject::DICT_TYPE);
    walker_guard_class(
        ctx,
        op_pc,
        dict_op,
        &pyre_object::pyobject::DICT_TYPE as *const _ as i64,
    )?;
    walker_guard_exact_w_class(ctx, op_pc, dict_op, canonical_dict)?;

    let (key_type, canonical_key, strategy_ref, lookup_helper) = match hit.key_probe {
        DictFoldKeyProbe::Int => (
            &pyre_object::pyobject::INT_TYPE as *const _ as i64,
            pyre_object::get_instantiate(&pyre_object::pyobject::INT_TYPE),
            &pyre_object::dictmultiobject::INT_DICT_STRATEGY_REF as *const _ as i64,
            crate::helpers::jit_dict_exact_int_lookup_or_null as *const (),
        ),
        DictFoldKeyProbe::Unicode => (
            &pyre_object::pyobject::STR_TYPE as *const _ as i64,
            pyre_object::get_instantiate(&pyre_object::pyobject::STR_TYPE),
            &pyre_object::dictmultiobject::UNICODE_DICT_STRATEGY_REF as *const _ as i64,
            crate::helpers::jit_dict_exact_unicode_lookup_or_null as *const (),
        ),
    };

    let strategy = crate::state::opimpl_getfield_gc_i(
        ctx.trace_ctx,
        dict_op,
        crate::descr::dict_strategy_word_descr(),
    );
    let strategy_const = ctx.trace_ctx.const_int(strategy_ref);
    walker_emit_fold_guard_with_snapshot(
        ctx,
        op_pc,
        OpCode::GuardValue,
        &[strategy, strategy_const],
    )?;
    ctx.trace_ctx
        .heap_cache_mut()
        .replace_box(strategy, strategy_const);

    walker_guard_class(ctx, op_pc, key_op, key_type)?;
    walker_guard_exact_w_class(ctx, op_pc, key_op, canonical_key)?;

    let value = ctx.trace_ctx.call_ref_typed_with_effect(
        lookup_helper,
        &[dict_op, key_op],
        &[majit_ir::Type::Ref, majit_ir::Type::Ref],
        majit_ir::EffectInfo::new(
            majit_ir::ExtraEffect::CannotRaise,
            majit_ir::OopSpecIndex::None,
        ),
    );
    walker_emit_fold_guard_with_snapshot(ctx, op_pc, OpCode::GuardNonnull, &[value])?;
    ctx.trace_ctx.set_opref_concrete(
        value,
        majit_ir::Value::Ref(majit_ir::GcRef(hit.concrete_value as usize)),
    );
    write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, value)?;
    Ok(Some(()))
}

/// `dict.get` on an exact dictionary and an Int/Unicode strategy hit.
///
/// `dictmultiobject.py:1095-1098` probes an Int strategy with the unboxed key
/// value, and `dictmultiobject.py:1315-1318` probes a Unicode strategy with
/// exact-str bytes. The trace guards the exact dict, strategy vtable, and exact
/// key type, then performs a live strategy-specific lookup and guards that it
/// hit. Misses and object/identity strategy keys remain residual so their
/// hash/equality effects stay observable (`rdict.py:576`).
pub(crate) fn try_walker_specialize_builtin_dict_get<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    code: &[u8],
    op: &DecodedOp,
    r_args: &[OpRef],
    dst: usize,
) -> Result<Option<()>, DispatchError> {
    if !(r_args.len() == 3 || r_args.len() == 4) {
        return Ok(None);
    }
    let arg_concretes = read_ref_var_list_concrete(code, op, 1, ctx);
    let (
        ConcreteValue::Ref(callable_operand),
        ConcreteValue::Ref(null_or_self),
        ConcreteValue::Ref(key),
    ) = (arg_concretes[0], arg_concretes[1], arg_concretes[2])
    else {
        return Ok(None);
    };
    if callable_operand.is_null() || key.is_null() {
        return Ok(None);
    }
    // LOAD_ATTR's generic method path produces `[Method, PY_NULL, key]`;
    // LOAD_METHOD's split path produces `[Function, receiver, key]`.
    // `_Method._immutable_fields_` lets both converge on the same live
    // function/receiver reads.
    let bound_method =
        null_or_self.is_null() && unsafe { pyre_object::is_method(callable_operand) };
    let (callable, dict) = if bound_method {
        (
            unsafe { pyre_object::w_method_get_func(callable_operand) },
            unsafe { pyre_object::w_method_get_self(callable_operand) },
        )
    } else {
        (callable_operand, null_or_self)
    };
    if callable.is_null() || dict.is_null() || !is_builtin_dict_get_function(callable) {
        return Ok(None);
    }
    let Some(hit) = walker_probe_exact_dict_hit(dict, key)? else {
        return Ok(None);
    };

    let mut callable_op = r_args[0];
    let dict_op;
    if bound_method {
        walker_guard_class(
            ctx,
            op.pc,
            r_args[0],
            &pyre_object::function::METHOD_TYPE as *const _ as i64,
        )?;
        callable_op = crate::state::opimpl_getfield_gc_r(
            ctx.trace_ctx,
            r_args[0],
            crate::descr::method_w_function_descr(),
        );
        dict_op = crate::state::opimpl_getfield_gc_r(
            ctx.trace_ctx,
            r_args[0],
            crate::descr::method_w_self_descr(),
        );
        ctx.trace_ctx.try_set_opref_concrete(
            dict_op,
            majit_ir::Value::Ref(majit_ir::GcRef(dict as usize)),
        );
    } else {
        dict_op = r_args[1];
    }
    if !callable_op.is_constant() {
        let expected = ctx.trace_ctx.const_ref(callable as i64);
        walker_emit_fold_guard_with_snapshot(
            ctx,
            op.pc,
            OpCode::GuardValue,
            &[callable_op, expected],
        )?;
        ctx.trace_ctx
            .heap_cache_mut()
            .replace_box(callable_op, expected);
    }
    walker_emit_exact_dict_hit(ctx, op.pc, dict_op, r_args[2], dict, hit, dst, 'r')
}

/// Where the guarded receiver's length comes from — the `length()` body each
/// layout has upstream.
#[derive(Clone, Copy)]
enum BuiltinLenSource {
    /// `W_ListObject.length()` → `strategy.length` (rlist.py). Carries the
    /// storage-strategy id the read is guarded on.
    ListStrategy(i64),
    /// `EmptyListStrategy.length()` returns zero (`listobject.py`).
    /// The strategy still needs a guard because a reused list may transition
    /// to typed or object storage after tracing.
    EmptyList,
    /// `W_UnicodeObject.len` → `bh_unicodelen`; no storage strategy.
    StrField,
    /// `W_BytesObject.len` — `bytesobject.py` answers `len(self._value)` off
    /// the RPython string; pyre precomputes that count into a field, so the
    /// read is the same shape as [`BuiltinLenSource::StrField`].
    BytesField,
    /// `W_BytearrayObject.length` — `bytearrayobject.py`'s `_len` reads the
    /// length off the RPython list in `self._data`; pyre mirrors that count
    /// into a field.  Mutable, unlike [`BuiltinLenSource::BytesField`].
    BytearrayField,
    /// `W_SetObject.len` — `setobject.py W_BaseSetObject.length` answers
    /// `self.strategy.length(self)`; pyre keeps the count on the body, so the
    /// read is the same shape as [`BuiltinLenSource::BytearrayField`].
    SetField,
    /// `tupleobject.py` carries no separate length field, so the length is
    /// `arraylen_gc(wrappeditems)`.
    TupleArrayLen,
    /// `functional.py W_Range.descr_len` returns the precomputed
    /// wrapped `self.w_length` field unchanged.
    RangeField,
    /// `specialisedtupleobject.py length()` returns the constant
    /// `typelen`.
    PairArity,
}

/// `len(x)` on an exact canonical `W_ListObject` / `W_UnicodeObject` /
/// `W_BytesObject` / `W_BytearrayObject` / `W_SetObject` (as either `set` or
/// `frozenset`) / `W_TupleObject` / `W_Range`, or on an arity-2 tuple
/// specialisation:
/// lower the opaque `bh_call_fn(len_builtin, PY_NULL, x)` residual to the
/// inline length read the meta-tracer produces upstream
/// (descroperation.py `_len`): `guard_value(callable)` +
/// `guard_class` + exact `w_class` guard + the [`BuiltinLenSource`] read +
/// `wrapint`.  The exact `w_class` guard is required because a SUBCLASS shares
/// `ob_type == &LIST_TYPE`/`&STR_TYPE`/`&TUPLE_TYPE` but may override `__len__`
/// (`baseobjspace::len` dispatches `subclass_special_override`); it
/// side-exits to the generic residual.
///
/// Returns `None` (fall through to the generic residual, SAFE) for any
/// other shape: non-list/str/tuple arg, a subclass, a bound receiver, or wrong
/// arity.  `dict` is one of those: `W_DictObject` carries no length word at
/// all -- `dictmultiobject.py length` goes through the strategy to
/// `len(unerase(dstorage))`, and pyre's storage is an `IndexMap` whose count
/// is not a field the trace can read.
pub(crate) fn try_walker_specialize_builtin_len<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    code: &[u8],
    op: &DecodedOp,
    r_args: &[OpRef],
    dst: usize,
) -> Result<Option<()>, DispatchError> {
    // Plain `bh_call_fn(callable, PY_NULL, arg)` shape only.
    if r_args.len() != 3 {
        return Ok(None);
    }
    let arg_concretes = read_ref_var_list_concrete(code, op, 1, ctx);
    let (
        ConcreteValue::Ref(concrete_callable),
        ConcreteValue::Ref(null_or_self),
        ConcreteValue::Ref(list_obj),
    ) = (arg_concretes[0], arg_concretes[1], arg_concretes[2])
    else {
        return Ok(None);
    };
    // A non-null `null_or_self` is a bound receiver `bh_call_fn_impl`
    // prepends as arg0 — not a plain `len(x)` call.
    if concrete_callable.is_null() || !null_or_self.is_null() || list_obj.is_null() {
        return Ok(None);
    }
    if !pyre_interpreter::builtins::is_builtin_len_function(concrete_callable) {
        return Ok(None);
    }
    // Exact canonical list / str / tuple / range, or one of the arity-2 tuple
    // specialisations.  `arg_type_addr` pins the `guard_class` target;
    // `exact_w_class` is the subclass-`__len__` guard (see the doc comment),
    // absent for a specialisation because only `makespecialisedtuple2` builds
    // that `ob_type` and always with the canonical tuple `w_class`, so the
    // class guard alone already excludes every subclass instance.
    let (arg_type_addr, exact_w_class, len_source, concrete_len) = unsafe {
        let ob_type = (*list_obj).ob_type;
        let w_class = (*list_obj).w_class;
        if std::ptr::eq(ob_type, &pyre_object::pyobject::LIST_TYPE) {
            let exact = pyre_object::pyobject::get_instantiate(&pyre_object::pyobject::LIST_TYPE);
            if !std::ptr::eq(w_class, exact) {
                return Ok(None);
            }
            let len_source = if pyre_object::w_list_uses_int_storage(list_obj) {
                BuiltinLenSource::ListStrategy(
                    pyre_object::listobject::ListStrategy::Integer as i64,
                )
            } else if pyre_object::w_list_uses_float_storage(list_obj) {
                BuiltinLenSource::ListStrategy(pyre_object::listobject::ListStrategy::Float as i64)
            } else if pyre_object::w_list_uses_object_storage(list_obj) {
                BuiltinLenSource::ListStrategy(pyre_object::listobject::ListStrategy::Object as i64)
            } else if pyre_object::w_list_uses_empty_storage(list_obj) {
                BuiltinLenSource::EmptyList
            } else {
                return Ok(None);
            };
            (
                &pyre_object::pyobject::LIST_TYPE as *const _ as i64,
                Some(exact),
                len_source,
                pyre_object::w_list_len(list_obj),
            )
        } else if std::ptr::eq(ob_type, &pyre_object::pyobject::STR_TYPE) {
            let exact = pyre_object::pyobject::get_instantiate(&pyre_object::pyobject::STR_TYPE);
            if !std::ptr::eq(w_class, exact) {
                return Ok(None);
            }
            (
                &pyre_object::pyobject::STR_TYPE as *const _ as i64,
                Some(exact),
                BuiltinLenSource::StrField,
                pyre_object::w_str_len(list_obj),
            )
        } else if std::ptr::eq(ob_type, &pyre_object::bytesobject::BYTES_TYPE) {
            let exact =
                pyre_object::pyobject::get_instantiate(&pyre_object::bytesobject::BYTES_TYPE);
            if !std::ptr::eq(w_class, exact) {
                return Ok(None);
            }
            (
                &pyre_object::bytesobject::BYTES_TYPE as *const _ as i64,
                Some(exact),
                BuiltinLenSource::BytesField,
                pyre_object::bytesobject::w_bytes_len(list_obj),
            )
        } else if std::ptr::eq(ob_type, &pyre_object::bytearrayobject::BYTEARRAY_TYPE) {
            let exact = pyre_object::pyobject::get_instantiate(
                &pyre_object::bytearrayobject::BYTEARRAY_TYPE,
            );
            if !std::ptr::eq(w_class, exact) {
                return Ok(None);
            }
            (
                &pyre_object::bytearrayobject::BYTEARRAY_TYPE as *const _ as i64,
                Some(exact),
                BuiltinLenSource::BytearrayField,
                pyre_object::bytearrayobject::w_bytearray_len(list_obj),
            )
        } else if std::ptr::eq(ob_type, &pyre_object::setobject::SET_TYPE)
            || std::ptr::eq(ob_type, &pyre_object::setobject::FROZENSET_TYPE)
        {
            // Two types over one `W_SetObject` body, so the only thing the two
            // differ in is which of them the class guard pins — and `ob_type`
            // is already the one that matched.
            let exact = pyre_object::pyobject::get_instantiate(&*ob_type);
            if !std::ptr::eq(w_class, exact) {
                return Ok(None);
            }
            (
                ob_type as i64,
                Some(exact),
                BuiltinLenSource::SetField,
                pyre_object::setobject::w_set_len(list_obj),
            )
        } else if std::ptr::eq(ob_type, &pyre_object::pyobject::TUPLE_TYPE) {
            let exact = pyre_object::pyobject::get_instantiate(&pyre_object::pyobject::TUPLE_TYPE);
            if !std::ptr::eq(w_class, exact) {
                return Ok(None);
            }
            (
                &pyre_object::pyobject::TUPLE_TYPE as *const _ as i64,
                Some(exact),
                BuiltinLenSource::TupleArrayLen,
                pyre_object::w_tuple_len(list_obj),
            )
        } else if std::ptr::eq(ob_type, &pyre_object::functional::RANGE_TYPE) {
            let exact =
                pyre_object::pyobject::get_instantiate(&pyre_object::functional::RANGE_TYPE);
            if !std::ptr::eq(w_class, exact) {
                return Ok(None);
            }
            let Some(concrete_len) = pyre_object::functional::w_range_length_i64(list_obj) else {
                return Ok(None);
            };
            let Ok(concrete_len) = usize::try_from(concrete_len) else {
                return Ok(None);
            };
            (
                &pyre_object::functional::RANGE_TYPE as *const _ as i64,
                Some(exact),
                BuiltinLenSource::RangeField,
                concrete_len,
            )
        } else if specialised_pair_kind(ob_type).is_some() {
            (
                ob_type as i64,
                None,
                BuiltinLenSource::PairArity,
                pyre_object::w_tuple_len(list_obj),
            )
        } else {
            return Ok(None);
        }
    };

    // Authentic boxed result, produced on the plain eval loop exactly as
    // the skipped residual would (len on an exact list is side-effect-free).
    let boxed_result = {
        let _plain_guard = pyre_interpreter::call::force_plain_eval();
        pyre_interpreter::call::call_function_impl_result(concrete_callable, &[list_obj])
    };
    let Ok(boxed_result) = boxed_result else {
        return Ok(None);
    };

    // --- emit the specialized IR (walker-native) ---
    // Pin the callable identity (LOAD_GLOBAL `len` is usually already a
    // constant via the namespace cell fold).
    let callable_op = r_args[0];
    if !callable_op.is_constant() {
        let expected = ctx.trace_ctx.const_ref(concrete_callable as i64);
        ctx.trace_ctx
            .record_guard(OpCode::GuardValue, &[callable_op, expected], 0);
        walker_capture_snapshot_for_last_guard(ctx, op.pc)?;
        ctx.trace_ctx
            .heap_cache_mut()
            .replace_box(callable_op, expected);
    }
    let list_op = r_args[2];
    // guard_class (skip when class already known / operand is constant).
    if !list_op.is_constant() && !ctx.trace_ctx.heap_cache().is_class_known(list_op) {
        let type_const = ctx.trace_ctx.const_int(arg_type_addr);
        ctx.trace_ctx
            .record_guard(OpCode::GuardClass, &[list_op, type_const], 0);
        walker_capture_snapshot_for_last_guard(ctx, op.pc)?;
    }
    ctx.trace_ctx
        .heap_cache_mut()
        .class_now_known(list_op, arg_type_addr);
    if let Some(exact_w_class) = exact_w_class {
        walker_guard_exact_w_class(ctx, op.pc, list_op, exact_w_class)?;
    }
    // `functional.py W_Range.descr_len` is already a wrapped-field
    // read.  Reuse that box directly; unlike the scalar length sources below,
    // there is nothing to unwrap and box again.  A virtual range's cached
    // field makes this fold to its existing virtual wrapped-int value.
    if matches!(len_source, BuiltinLenSource::RangeField) {
        let boxed = crate::state::opimpl_getfield_gc_r(
            ctx.trace_ctx,
            list_op,
            crate::descr::range_length_descr(),
        );
        // Admission read the field and required it to fit a machine word
        // (`w_range_length_i64`), so the trace has to pin that too: the class
        // guards above prove the receiver is a range, not what its length slot
        // holds.  `descr_new` stores a `W_LongObject` there whenever
        // `compute_range_length` leaves the machine range, and without this
        // guard a later entry carrying such a range would take the recorded
        // exit and hand `len()` the long straight through.
        walker_guard_class(
            ctx,
            op.pc,
            boxed,
            &pyre_object::pyobject::INT_TYPE as *const _ as i64,
        )?;
        walker_guard_exact_w_class(
            ctx,
            op.pc,
            boxed,
            pyre_object::pyobject::get_instantiate(&pyre_object::pyobject::INT_TYPE),
        )?;
        write_residual_call_result_to_dst(ctx, op.pc, dst, 'r', boxed)?;
        return Ok(Some(()));
    }
    // Length read.  list: guard the storage strategy, then read that
    // strategy's length field (rlist.py inline field for object storage;
    // typed items-block length for int/float storage).  str: a plain
    // codepoint-length getfield (no strategy, `bh_unicodelen`).
    let raw_len = match len_source {
        BuiltinLenSource::ListStrategy(sid) => {
            let strategy = crate::state::opimpl_getfield_gc_i(
                ctx.trace_ctx,
                list_op,
                crate::descr::list_strategy_descr(),
            );
            let sid_const = ctx.trace_ctx.const_int(sid);
            ctx.trace_ctx
                .record_guard(OpCode::GuardValue, &[strategy, sid_const], 0);
            walker_capture_snapshot_for_last_guard(ctx, op.pc)?;
            ctx.trace_ctx
                .heap_cache_mut()
                .replace_box(strategy, sid_const);
            let len_descr = match sid {
                0 => crate::descr::list_length_descr(),
                1 => crate::descr::list_int_items_len_descr(),
                _ => crate::descr::list_float_items_len_descr(),
            };
            crate::state::opimpl_getfield_gc_i(ctx.trace_ctx, list_op, len_descr)
        }
        BuiltinLenSource::EmptyList => {
            let strategy = crate::state::opimpl_getfield_gc_i(
                ctx.trace_ctx,
                list_op,
                crate::descr::list_strategy_descr(),
            );
            let empty = ctx
                .trace_ctx
                .const_int(pyre_object::listobject::ListStrategy::Empty as i64);
            ctx.trace_ctx
                .record_guard(OpCode::GuardValue, &[strategy, empty], 0);
            walker_capture_snapshot_for_last_guard(ctx, op.pc)?;
            ctx.trace_ctx.heap_cache_mut().replace_box(strategy, empty);
            ctx.trace_ctx.const_int(0)
        }
        BuiltinLenSource::TupleArrayLen => {
            let wrappeditems = crate::state::opimpl_getfield_gc_r(
                ctx.trace_ctx,
                list_op,
                crate::descr::tuple_wrappeditems_descr(),
            );
            crate::state::opimpl_arraylen_gc(
                ctx.trace_ctx,
                wrappeditems,
                crate::state::pyobject_gcarray_descr(),
            )
        }
        BuiltinLenSource::StrField => crate::state::opimpl_getfield_gc_i(
            ctx.trace_ctx,
            list_op,
            crate::descr::str_len_descr(),
        ),
        BuiltinLenSource::BytesField => crate::state::opimpl_getfield_gc_i(
            ctx.trace_ctx,
            list_op,
            crate::descr::bytes_len_descr(),
        ),
        BuiltinLenSource::BytearrayField => crate::state::opimpl_getfield_gc_i(
            ctx.trace_ctx,
            list_op,
            crate::descr::bytearray_length_descr(),
        ),
        BuiltinLenSource::SetField => crate::state::opimpl_getfield_gc_i(
            ctx.trace_ctx,
            list_op,
            crate::descr::set_len_descr(),
        ),
        BuiltinLenSource::RangeField => unreachable!("range returned its wrapped length above"),
        // `specialisedtupleobject.py length()` returns the constant
        // `typelen`; there is no field to read, so the class guard above is
        // the whole proof and the box below folds to a constant.
        BuiltinLenSource::PairArity => ctx.trace_ctx.const_int(concrete_len as i64),
    };
    ctx.trace_ctx
        .set_opref_concrete(raw_len, majit_ir::Value::Int(concrete_len as i64));
    let boxed = walker_box_int(ctx, op.pc, raw_len, concrete_len as i64)?;
    ctx.trace_ctx.set_opref_concrete(
        boxed,
        box_int_concrete(concrete_len as i64, boxed_result as i64),
    );
    write_residual_call_result_to_dst(ctx, op.pc, dst, 'r', boxed)?;
    Ok(Some(()))
}

/// Fold plain `getattr(type, name)` when
/// [`pyre_interpreter::type_attr_value_fast_path`] proves that
/// `typeobject.py` `getattribute` returns the class-MRO value unchanged.  The exact
/// callable, exact receiver, exact name object, and receiver version are pinned
/// before the value is written as a green constant.  Pinning the callable makes
/// a rebound `getattr` side-exit instead of continuing to use the folded value.
/// The operand guards are tautologies when their inputs are already constants
/// and disappear during optimization.
/// [`pyre_interpreter::mutated`] recursively invalidates subclasses, so the
/// receiver's one quasi-immutable version watcher covers base-class mutation
/// and emits no per-iteration operations.
///
/// Like the `len` fold this is safe in an inlined callee sub-walk: the oracle
/// proves a read-only present attribute, so it cannot raise or introduce a
/// side effect that resume would repeat.  Every other shape declines before
/// emitting IR and falls through to the generic residual.
pub(crate) fn try_walker_specialize_builtin_type_getattr<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    code: &[u8],
    op: &DecodedOp,
    r_args: &[OpRef],
    dst: usize,
) -> Result<Option<()>, DispatchError> {
    // Plain `bh_call_fn(callable, PY_NULL, obj, name)` shape only.
    if r_args.len() != 4 {
        return Ok(None);
    }
    let arg_concretes = read_ref_var_list_concrete(code, op, 1, ctx);
    let (
        ConcreteValue::Ref(concrete_callable),
        ConcreteValue::Ref(null_or_self),
        ConcreteValue::Ref(concrete_obj),
        ConcreteValue::Ref(concrete_name),
    ) = (
        arg_concretes[0],
        arg_concretes[1],
        arg_concretes[2],
        arg_concretes[3],
    )
    else {
        return Ok(None);
    };
    // A non-null `null_or_self` is a bound receiver `bh_call_fn_impl`
    // prepends as arg0 — not a plain `getattr(type, name)` call.
    if concrete_callable.is_null()
        || !null_or_self.is_null()
        || concrete_obj.is_null()
        || concrete_name.is_null()
    {
        return Ok(None);
    }
    if !pyre_interpreter::builtins::is_builtin_getattr_function(concrete_callable) {
        return Ok(None);
    }
    if !unsafe { pyre_object::is_exact_type(concrete_name, &pyre_object::pyobject::STR_TYPE) } {
        return Ok(None);
    }
    let name = unsafe { pyre_object::w_str_get_wtf8(concrete_name) };
    let Some((w_type, _version_tag, w_value)) =
        (unsafe { pyre_interpreter::type_attr_value_fast_path(concrete_obj, name) })
    else {
        return Ok(None);
    };

    let callable_op = r_args[0];
    if !callable_op.is_constant() {
        let expected = ctx.trace_ctx.const_ref(concrete_callable as i64);
        walker_emit_fold_guard_with_snapshot(
            ctx,
            op.pc,
            OpCode::GuardValue,
            &[callable_op, expected],
        )?;
        ctx.trace_ctx
            .heap_cache_mut()
            .replace_box(callable_op, expected);
    }

    let obj_ref = r_args[2];
    let w_type_const = ctx.trace_ctx.const_ref(w_type as i64);
    if !obj_ref.is_constant() {
        walker_emit_fold_guard_with_snapshot(
            ctx,
            op.pc,
            OpCode::GuardValue,
            &[obj_ref, w_type_const],
        )?;
        ctx.trace_ctx
            .heap_cache_mut()
            .replace_box(obj_ref, w_type_const);
    }

    // The baked WTF-8 bytes remain constant only while this exact string is
    // the name operand.  Constant operands make this guard a removable
    // tautology, so it costs nothing in the steady loop.
    let name_ref = r_args[3];
    let name_const = ctx.trace_ctx.const_ref(concrete_name as i64);
    if !name_ref.is_constant() {
        walker_emit_fold_guard_with_snapshot(
            ctx,
            op.pc,
            OpCode::GuardValue,
            &[name_ref, name_const],
        )?;
    }

    // typeobject.py `promote(self.version_tag())`: this quasi-immutable watcher
    // emits no per-iteration op. `mutated` (baseobjspace.rs) recurses through
    // subclasses, so changing the attribute on any base invalidates this pin.
    walker_pin_type_version_tag(ctx, op.pc, w_type_const)?;

    let value_const = ctx.trace_ctx.const_ref(w_value as i64);
    write_residual_call_result_to_dst(ctx, op.pc, dst, 'r', value_const)?;
    Ok(Some(()))
}

/// `getattr(obj, "name")` — the builtin spelling of the `LOAD_ATTR` fold.
///
/// `space.getattr` is the one operation both `obj.name` and this builtin
/// reach, so a constant `str` name admits exactly the instance-shape read
/// [`try_walker_specialize_load_attr`] already emits, and the two forms stay on
/// one implementation rather than drifting the way a fast-path pair can
/// (`getattr(obj, 'm')` versus `obj.m` is the classic discriminator).
///
/// The three-argument `getattr(obj, name, default)` stays on the residual: the
/// fold's map guard proves the attribute is *present*, which says nothing about
/// the branch that supplies the default.
pub(crate) fn try_walker_specialize_builtin_getattr<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    code: &[u8],
    op: &DecodedOp,
    r_args: &[OpRef],
    dst: usize,
) -> Result<Option<()>, DispatchError> {
    // Plain `bh_call_fn(callable, PY_NULL, obj, name)` shape only; the
    // three-argument form arrives one operand longer and declines here.
    if r_args.len() != 4 {
        return Ok(None);
    }
    let arg_concretes = read_ref_var_list_concrete(code, op, 1, ctx);
    let (
        ConcreteValue::Ref(concrete_callable),
        ConcreteValue::Ref(null_or_self),
        ConcreteValue::Ref(concrete_obj),
        ConcreteValue::Ref(concrete_name),
    ) = (
        arg_concretes[0],
        arg_concretes[1],
        arg_concretes[2],
        arg_concretes[3],
    )
    else {
        return Ok(None);
    };
    // A non-null `null_or_self` is a bound receiver `bh_call_fn_impl` prepends
    // as arg0 — not a plain `getattr(obj, name)` call.
    if concrete_callable.is_null()
        || !null_or_self.is_null()
        || concrete_obj.is_null()
        || concrete_name.is_null()
    {
        return Ok(None);
    }
    if !pyre_interpreter::builtins::is_builtin_getattr_function(concrete_callable) {
        return Ok(None);
    }
    // The name is rejected before any lookup unless it is a string, and the
    // resolved bytes below stay valid only while this exact string is the
    // operand.  A name that is not valid UTF-8 cannot match an attribute the
    // fold's `&str` lookups can find, so it declines with the rest.
    if !unsafe { pyre_object::is_exact_type(concrete_name, &pyre_object::pyobject::STR_TYPE) } {
        return Ok(None);
    }
    let Ok(name) = (unsafe { pyre_object::w_str_get_wtf8(concrete_name) }).as_str() else {
        return Ok(None);
    };

    let pre_emit_pos = ctx.trace_ctx.get_trace_position();
    let callable_op = r_args[0];
    if !callable_op.is_constant() {
        let expected = ctx.trace_ctx.const_ref(concrete_callable as i64);
        walker_emit_fold_guard_with_snapshot(
            ctx,
            op.pc,
            OpCode::GuardValue,
            &[callable_op, expected],
        )?;
        ctx.trace_ctx
            .heap_cache_mut()
            .replace_box(callable_op, expected);
    }
    let name_ref = r_args[3];
    if !name_ref.is_constant() {
        let name_const = ctx.trace_ctx.const_ref(concrete_name as i64);
        walker_emit_fold_guard_with_snapshot(
            ctx,
            op.pc,
            OpCode::GuardValue,
            &[name_ref, name_const],
        )?;
    }

    // Every shape the read declines has to leave the trace as it found it: the
    // two guards above are the premise of a fold that is no longer there, and
    // the residual the caller falls through to recomputes the lookup from the
    // unguarded operands.
    if (try_walker_specialize_load_attr(ctx, op.pc, r_args[2], name, dst, 'r')?).is_none() {
        ctx.trace_ctx.cut_trace_with_snapshots(pre_emit_pos);
        ctx.trace_ctx.heap_cache_mut().reset();
        return Ok(None);
    }
    Ok(Some(()))
}

/// Record an overflow-checked machine-int operation and guard it.
///
/// [`record_int_ovf`] folds a both-constant operand pair to a constant without
/// recording anything, and `GuardNoOverflow` carries no operands — it reads the
/// flag of the operation immediately before it — so an unconditional guard
/// after a folded pair would attach to whatever was recorded last instead.
/// `None` means the operation cannot be represented and the caller must rewind.
fn record_int_ovf_guarded<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    opcode: OpCode,
    b1: OpRef,
    b2: OpRef,
) -> Result<Option<OpRef>, DispatchError> {
    let (result, overflow) = record_int_ovf(ctx, op_pc, opcode, b1, b2)?;
    if overflow {
        return Ok(None);
    }
    if !result.is_constant() {
        walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardNoOverflow, &[])?;
    }
    Ok(Some(result))
}

/// Fall back to the opaque `range` residual from a decline point that the
/// specializer only reaches after it has already emitted.
///
/// Every decline in `try_walker_specialize_builtin_range` past `pre_emit_pos`
/// has to rewind: the callable `GuardValue`, the per-bound class guards and
/// `intval` reads, and — for a bound converted by a user `__index__` — that
/// callee's whole inlined body sit in the trace, and the residual the caller
/// falls through to recomputes all of it.  Leaving them behind would pair the
/// residual with guards for a specialization that no longer exists.
fn walker_range_decline<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    pre_emit_pos: majit_metainterp::recorder::TracePosition,
) -> Result<Option<DispatchOutcome>, DispatchError> {
    ctx.trace_ctx.cut_trace_with_snapshots(pre_emit_pos);
    ctx.trace_ctx.heap_cache_mut().reset();
    Ok(None)
}

/// Emit the machine-int trace of `functional.py compute_range_length`
/// for a path whose converted bounds all fit signed machine words.  Each
/// source conditional becomes the guard chosen by the recording values; the
/// overflow guards side-exit to the interpreter's wrapped-int implementation.
fn walker_emit_range_length<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    start: OpRef,
    stop: OpRef,
    step: OpRef,
    concrete_start: i64,
    concrete_stop: i64,
    concrete_step: i64,
) -> Result<Option<OpRef>, DispatchError> {
    if concrete_step == 0 {
        return Ok(None);
    }
    let (normalized_start, normalized_stop, normalized_step) = if concrete_step < 0 {
        let Some(step) = concrete_step.checked_neg() else {
            return Ok(None);
        };
        (concrete_stop, concrete_start, step)
    } else {
        (concrete_start, concrete_stop, concrete_step)
    };
    let concrete_length = if normalized_start < normalized_stop {
        let Some(diff) = normalized_stop
            .checked_sub(normalized_start)
            .and_then(|diff| diff.checked_sub(1))
        else {
            return Ok(None);
        };
        let Some(length) = (diff / normalized_step).checked_add(1) else {
            return Ok(None);
        };
        length
    } else {
        0
    };

    let zero = ctx.trace_ctx.const_int(0);
    let one = ctx.trace_ctx.const_int(1);
    let step_has_recorded_sign = if concrete_step < 0 {
        ctx.trace_ctx.record_op(OpCode::IntLt, &[step, zero])
    } else {
        ctx.trace_ctx.record_op(OpCode::IntGt, &[step, zero])
    };
    ctx.trace_ctx
        .set_opref_concrete(step_has_recorded_sign, majit_ir::Value::Int(1));
    walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardTrue, &[step_has_recorded_sign])?;
    let (lo, hi, positive_step) = if concrete_step < 0 {
        let Some(negated) = record_int_ovf_guarded(ctx, op_pc, OpCode::IntSubOvf, zero, step)?
        else {
            return Ok(None);
        };
        (stop, start, negated)
    } else {
        (start, stop, step)
    };

    let nonempty = ctx.trace_ctx.record_op(OpCode::IntLt, &[lo, hi]);
    ctx.trace_ctx.set_opref_concrete(
        nonempty,
        majit_ir::Value::Int((normalized_start < normalized_stop) as i64),
    );
    if normalized_start >= normalized_stop {
        walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardFalse, &[nonempty])?;
        return Ok(Some(zero));
    }
    walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardTrue, &[nonempty])?;

    let Some(span) = record_int_ovf_guarded(ctx, op_pc, OpCode::IntSubOvf, hi, lo)? else {
        return Ok(None);
    };
    let Some(diff) = record_int_ovf_guarded(ctx, op_pc, OpCode::IntSubOvf, span, one)? else {
        return Ok(None);
    };
    let quotient = ctx
        .trace_ctx
        .record_op(OpCode::IntFloorDiv, &[diff, positive_step]);
    ctx.trace_ctx.set_opref_concrete(
        quotient,
        majit_ir::Value::Int((normalized_stop - normalized_start - 1) / normalized_step),
    );
    let Some(length) = record_int_ovf_guarded(ctx, op_pc, OpCode::IntAddOvf, quotient, one)? else {
        return Ok(None);
    };
    ctx.trace_ctx
        .set_opref_concrete(length, majit_ir::Value::Int(concrete_length));
    Ok(Some(length))
}

/// `range(stop)` / `range(start, stop)` / `range(start, stop, step)` with
/// exact canonical machine-word ints or strict inlinable user `__index__`
/// conversions: lower the opaque constructor residual
/// to a virtual `W_Range` and four virtual wrapped-int fields.  This lets the
/// existing GET_ITER specialization consume the range without forcing either
/// allocation.  All other callables and argument shapes fall through to the
/// generic residual.
pub(crate) fn try_walker_specialize_builtin_range<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    code: &[u8],
    op: &DecodedOp,
    funcptr: OpRef,
    r_args: &[OpRef],
    call_descr: &dyn majit_ir::descr::CallDescr,
    dst: usize,
) -> Result<Option<DispatchOutcome>, DispatchError> {
    if !(3..=5).contains(&r_args.len()) {
        return Ok(None);
    }
    let arg_concretes = read_ref_var_list_concrete(code, op, 1, ctx);
    let (ConcreteValue::Ref(concrete_callable), ConcreteValue::Ref(null_or_self)) =
        (arg_concretes[0], arg_concretes[1])
    else {
        return Ok(None);
    };
    let range_type = pyre_interpreter::typedef::gettypeobject(&pyre_object::functional::RANGE_TYPE);
    if concrete_callable.is_null()
        || !null_or_self.is_null()
        || !std::ptr::eq(concrete_callable, range_type)
    {
        return Ok(None);
    }

    let exact_int_class = pyre_object::pyobject::get_instantiate(&pyre_object::pyobject::INT_TYPE);
    enum BoundPlan {
        Exact {
            op: OpRef,
            concrete: pyre_object::PyObjectRef,
        },
        UserIndex(IndexInlineCandidate),
    }
    let mut plans = Vec::with_capacity(r_args.len() - 2);
    let mut has_user_index = false;
    for (&arg_op, concrete) in r_args[2..].iter().zip(&arg_concretes[2..]) {
        let ConcreteValue::Ref(arg_obj) = *concrete else {
            return Ok(None);
        };
        if walker_is_exact_machine_int_concrete(arg_obj) {
            plans.push(BoundPlan::Exact {
                op: arg_op,
                concrete: arg_obj,
            });
        } else if let Some(candidate) = prepare_walker_inline_index(ctx, arg_op, arg_obj) {
            has_user_index = true;
            plans.push(BoundPlan::UserIndex(candidate));
        } else {
            return Ok(None);
        }
    }

    // Every non-int bound has been resolved and statically preflighted before
    // this first emission.  `functional.py W_Range.descr_new` applies
    // `space.index` independently to start/stop/step; mirror that order and
    // retain each returned box as an intermediate feeding the constructor.
    let pre_emit_pos = ctx.trace_ctx.get_trace_position();
    let callable_op = r_args[0];
    if !callable_op.is_constant() {
        let expected = ctx.trace_ctx.const_ref(concrete_callable as i64);
        ctx.trace_ctx
            .record_guard(OpCode::GuardValue, &[callable_op, expected], 0);
        walker_capture_snapshot_for_last_guard(ctx, op.pc)?;
        ctx.trace_ctx
            .heap_cache_mut()
            .replace_box(callable_op, expected);
    }

    let int_type_addr = &pyre_object::pyobject::INT_TYPE as *const _ as i64;
    let mut concrete_args = Vec::with_capacity(plans.len());
    let mut concrete_values = Vec::with_capacity(plans.len());
    let mut raw_args = Vec::with_capacity(plans.len());
    for plan in plans {
        let (arg_op, arg_obj) = match plan {
            BoundPlan::Exact { op, concrete } => (op, concrete),
            BoundPlan::UserIndex(candidate) => {
                let Some((result, ConcreteValue::Ref(concrete))) = try_walker_inline_index(
                    ctx, op, code, funcptr, r_args, call_descr, dst, candidate,
                )?
                else {
                    return walker_range_decline(ctx, pre_emit_pos);
                };
                (result, concrete)
            }
        };
        // A trace-constant bound carries its class in the constant itself, so
        // record the class as known without proving it: `walker_guard_class`
        // would emit a `GuardClass` that can never fail plus the tagged-operand
        // low-bit test that guards a later entry's untagged arrival, and a
        // constant has no later arrival.  A bound returned by an inlined
        // `__index__` is live and takes the full guard.
        if arg_op.is_constant() {
            ctx.trace_ctx
                .heap_cache_mut()
                .class_now_known(arg_op, int_type_addr);
        } else {
            walker_guard_class(ctx, op.pc, arg_op, int_type_addr)?;
        }
        walker_guard_exact_w_class(ctx, op.pc, arg_op, exact_int_class)?;
        let concrete_value = unsafe { pyre_object::w_int_get_value(arg_obj) };
        let raw = crate::state::opimpl_getfield_gc_i(
            ctx.trace_ctx,
            arg_op,
            crate::descr::int_intval_descr(),
        );
        ctx.trace_ctx
            .set_opref_concrete(raw, majit_ir::Value::Int(concrete_value));
        concrete_args.push(arg_obj);
        concrete_values.push(concrete_value);
        raw_args.push(raw);
    }

    // Run only the remaining builtin range body on the converted exact ints;
    // executing the original arguments here would call user `__index__` a
    // second time during recording.
    let authentic_result = {
        let _plain_guard = pyre_interpreter::call::force_plain_eval();
        pyre_interpreter::call::call_function_impl_result(concrete_callable, &concrete_args)
    };
    if concrete_values.len() == 3 && concrete_values[2] == 0 {
        let Err(mut err) = authentic_result else {
            return walker_range_decline(ctx, pre_emit_pos);
        };
        let exc = err.to_exc_object();
        let kind = pyre_object::interp_exceptions::ExcKind::ValueError;
        if !walker_recorded_builtin_raise_is_supported(exc, kind) {
            return walker_range_decline(ctx, pre_emit_pos);
        }
        let Some(ec) = walker_ensure_execution_context(ctx) else {
            return walker_range_decline(ctx, pre_emit_pos);
        };

        let step_raw = raw_args[2];
        let zero = ctx.trace_ctx.const_int(0);
        let is_zero = ctx.trace_ctx.record_op(OpCode::IntEq, &[step_raw, zero]);
        ctx.trace_ctx
            .set_opref_concrete(is_zero, majit_ir::Value::Int(1));
        walker_emit_guard_with_snapshot(ctx, op.pc, OpCode::GuardTrue, &[is_zero])?;
        return Ok(Some(walker_emit_recorded_builtin_raise(ctx, ec, exc, kind)));
    }
    let Ok(authentic_range) = authentic_result else {
        return walker_range_decline(ctx, pre_emit_pos);
    };
    let (authentic_start, authentic_stop, authentic_step) =
        unsafe { pyre_object::functional::w_range_fields(authentic_range) };
    let authentic_length = unsafe { pyre_object::functional::w_range_length(authentic_range) };
    let authentic_fields = [
        authentic_start,
        authentic_stop,
        authentic_step,
        authentic_length,
    ];
    if authentic_fields.iter().any(|&field| unsafe {
        !std::ptr::eq((*field).ob_type, &pyre_object::pyobject::INT_TYPE)
            || !std::ptr::eq((*field).w_class, exact_int_class)
    }) {
        return walker_range_decline(ctx, pre_emit_pos);
    }
    let concrete_fields =
        authentic_fields.map(|field| unsafe { pyre_object::w_int_get_value(field) });
    let [
        concrete_start,
        concrete_stop,
        concrete_step,
        concrete_length,
    ] = concrete_fields;

    let zero = ctx.trace_ctx.const_int(0);
    let one = ctx.trace_ctx.const_int(1);
    let (start, stop, step) = match raw_args.as_slice() {
        [stop] => (zero, *stop, one),
        [start, stop] => (*start, *stop, one),
        [start, stop, step] => (*start, *stop, *step),
        _ => unreachable!("range arity gate admitted an invalid argument count"),
    };
    // Trace-constant bounds retain the existing zero-op length.  A bound
    // produced by the user `_index` call follows PyPy's traced
    // `compute_range_length` body above, so its live value feeds all four
    // virtual fields instead of being paired with a stale record-time length.
    // Other variable-bound sources keep the residual, preserving the existing
    // admission boundary for recursive and alternating-range shapes.
    let length = if start.is_constant() && stop.is_constant() && step.is_constant() {
        ctx.trace_ctx.const_int(concrete_length)
    } else if has_user_index {
        let Some(length) = walker_emit_range_length(
            ctx,
            op.pc,
            start,
            stop,
            step,
            concrete_start,
            concrete_stop,
            concrete_step,
        )?
        else {
            return walker_range_decline(ctx, pre_emit_pos);
        };
        length
    } else {
        return walker_range_decline(ctx, pre_emit_pos);
    };

    let new = ctx.trace_ctx.record_op_with_descr(
        OpCode::NewWithVtable,
        &[],
        crate::descr::w_range_size_descr(),
    );
    ctx.trace_ctx.heap_cache_mut().new_object(new);

    let field_descrs = [
        crate::descr::range_start_descr(),
        crate::descr::range_stop_descr(),
        crate::descr::range_step_descr(),
        crate::descr::range_length_descr(),
    ];
    let raw_fields = [start, stop, step, length];
    for (((descr, raw), concrete_value), authentic_field) in field_descrs
        .into_iter()
        .zip(raw_fields)
        .zip(concrete_fields)
        .zip(authentic_fields)
    {
        let boxed = crate::state::wrapint(ctx.trace_ctx, raw);
        ctx.trace_ctx.set_opref_concrete(
            boxed,
            majit_ir::Value::Ref(majit_ir::GcRef(authentic_field as usize)),
        );
        let descr_index = descr.index();
        ctx.trace_ctx
            .record_op_with_descr(OpCode::SetfieldGc, &[new, boxed], descr);
        ctx.trace_ctx
            .heapcache_setfield_cached(new, descr_index, boxed);
    }

    let range_type_addr = &pyre_object::functional::RANGE_TYPE as *const _ as i64;
    ctx.trace_ctx
        .heap_cache_mut()
        .class_now_known(new, range_type_addr);
    ctx.trace_ctx.set_opref_concrete(
        new,
        majit_ir::Value::Ref(majit_ir::GcRef(authentic_range as usize)),
    );
    write_residual_call_result_to_dst(ctx, op.pc, dst, 'r', new)?;
    Ok(Some(DispatchOutcome::Continue))
}

/// Virtualize PyPy's exact `zip(tuple0, tuple1, strict=True)` allocation chain.
pub(crate) fn try_walker_specialize_builtin_zip<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    code: &[u8],
    op: &DecodedOp,
    r_args: &[OpRef],
    dst: usize,
) -> Result<Option<()>, DispatchError> {
    // A `call_kw` residual leads with its keyword-name tuple rather than
    // trailing it: `[callable, self_or_null, kwnames, arg0..argN-1]`, the
    // order `eval.rs call_kw` pushes and `fold_call_kw_permutation` reads.
    // So `zip(p, q, strict=True)` arrives as
    // [zip, self_or_null, ("strict",), p, q, True].
    if r_args.len() != 6 {
        return Ok(None);
    }
    let arg_concretes = read_ref_var_list_concrete(code, op, 1, ctx);
    let [
        ConcreteValue::Ref(callable),
        self_or_null,
        ConcreteValue::Ref(kwnames),
        ConcreteValue::Ref(tuple0),
        ConcreteValue::Ref(tuple1),
        ConcreteValue::Ref(strict),
    ] = arg_concretes.as_slice()
    else {
        return Ok(None);
    };
    // The receiver slot has two spellings for the same "no bound receiver".
    // A keyword call lowers it to a const `PY_NULL` (`ConstPtr(GcRef(0))`),
    // whose concrete shadow is `ConcreteValue::Null` because constant pool
    // slots carry no `Ref` shadow; a positional call takes the slot off the
    // value stack, where it reads `Ref(null)`.  Only the first reaches here,
    // since `strict=` is what brings this fold in at all.
    let self_is_null = match self_or_null {
        ConcreteValue::Ref(p) => p.is_null(),
        ConcreteValue::Null => matches!(
            ctx.trace_ctx.box_value(r_args[1]),
            Some(majit_ir::Value::Ref(majit_ir::GcRef(0)))
        ),
        _ => false,
    };
    let zip_callable = pyre_interpreter::typedef::gettypeobject(&pyre_object::functional::ZIP_TYPE);
    if callable.is_null()
        || !std::ptr::eq(*callable, zip_callable)
        || !self_is_null
        || kwnames.is_null()
        || unsafe { !pyre_object::is_tuple(*kwnames) }
        || unsafe { pyre_object::w_tuple_len(*kwnames) } != 1
        || !std::ptr::eq(*strict, pyre_object::w_bool_from(true))
    {
        return Ok(None);
    }
    let Some(keyword) = (unsafe { pyre_object::w_tuple_getitem(*kwnames, 0) }) else {
        return Ok(None);
    };
    if unsafe {
        !pyre_object::is_str(keyword)
            || pyre_object::w_str_get_wtf8(keyword).as_str().ok() != Some("strict")
    } {
        return Ok(None);
    }
    let tuple_type = &pyre_object::TUPLE_TYPE as *const pyre_object::PyType;
    let tuple_class = pyre_object::get_instantiate(&pyre_object::TUPLE_TYPE);
    for tuple in [*tuple0, *tuple1] {
        if tuple.is_null()
            || unsafe {
                !std::ptr::eq((*tuple).ob_type, tuple_type)
                    || !std::ptr::eq((*tuple).w_class, tuple_class)
            }
        {
            return Ok(None);
        }
    }

    // Recognition is complete; build the authentic shadow before emitting.
    let roots = pyre_object::gc_roots::push_roots();
    let root_base = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(*tuple0);
    let _ = pyre_object::gc_roots::pin_root(*tuple1);
    let concrete_iter0 = pyre_object::w_tuple_iter_new(unsafe {
        pyre_object::gc_roots::shadow_stack_get(root_base)
    });
    let _ = pyre_object::gc_roots::pin_root(concrete_iter0);
    let iter0_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let concrete_iter1 = pyre_object::w_tuple_iter_new(unsafe {
        pyre_object::gc_roots::shadow_stack_get(root_base + 1)
    });
    let _ = pyre_object::gc_roots::pin_root(concrete_iter1);
    let iter1_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let concrete_list = pyre_object::w_list_new(vec![
        unsafe { pyre_object::gc_roots::shadow_stack_get(iter0_slot) },
        unsafe { pyre_object::gc_roots::shadow_stack_get(iter1_slot) },
    ]);
    let _ = pyre_object::gc_roots::pin_root(concrete_list);
    let list_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let concrete_zip = pyre_object::functional::w_zip_new(
        unsafe { pyre_object::gc_roots::shadow_stack_get(list_slot) },
        true,
    );
    if concrete_zip.is_null() {
        drop(roots);
        return Err(DispatchError::ConcreteShadowAllocationFailed { pc: op.pc });
    }
    let _ = pyre_object::gc_roots::pin_root(concrete_zip);
    let zip_slot = pyre_object::gc_roots::shadow_stack_len() - 1;

    walker_guard_builtin_callable_identity(ctx, op.pc, r_args[0], *callable)?;
    for (arg_op, concrete) in [(r_args[2], *kwnames), (r_args[5], *strict)] {
        if !arg_op.is_constant() {
            let expected = ctx.trace_ctx.const_ref(concrete as i64);
            walker_emit_guard_with_snapshot(ctx, op.pc, OpCode::GuardValue, &[arg_op, expected])?;
            ctx.trace_ctx.heap_cache_mut().replace_box(arg_op, expected);
        }
    }

    let zero = ctx.trace_ctx.const_int(0);
    let mut iterator_ops = Vec::with_capacity(2);
    for (tuple_op, concrete_slot) in [(r_args[3], iter0_slot), (r_args[4], iter1_slot)] {
        walker_guard_class(ctx, op.pc, tuple_op, tuple_type as i64)?;
        walker_guard_exact_w_class(ctx, op.pc, tuple_op, tuple_class)?;
        let iterator = ctx.trace_ctx.record_op_with_descr(
            OpCode::NewWithVtable,
            &[],
            crate::descr::tuple_iter_size_descr(),
        );
        ctx.trace_ctx.heap_cache_mut().new_object(iterator);
        let seq_descr = crate::descr::tuple_iter_seq_descr();
        ctx.trace_ctx.record_op_with_descr(
            OpCode::SetfieldGc,
            &[iterator, tuple_op],
            seq_descr.clone(),
        );
        ctx.trace_ctx
            .heapcache_setfield_cached(iterator, seq_descr.index(), tuple_op);
        let index_descr = crate::descr::tuple_iter_index_descr();
        ctx.trace_ctx.record_op_with_descr(
            OpCode::SetfieldGc,
            &[iterator, zero],
            index_descr.clone(),
        );
        ctx.trace_ctx
            .heapcache_setfield_cached(iterator, index_descr.index(), zero);
        ctx.trace_ctx.heap_cache_mut().class_now_known(
            iterator,
            &pyre_object::iterobject::TUPLE_ITER_TYPE as *const _ as i64,
        );
        // Re-read after allocations that may move the shadow.
        let concrete_iter = unsafe { pyre_object::gc_roots::shadow_stack_get(concrete_slot) };
        ctx.trace_ctx.set_opref_concrete(
            iterator,
            Value::Ref(majit_ir::GcRef(concrete_iter as usize)),
        );
        iterator_ops.push(iterator);
    }

    let iterator_list = crate::helpers::emit_object_list_inline(ctx.trace_ctx, &iterator_ops);
    ctx.trace_ctx.set_opref_concrete(
        iterator_list,
        Value::Ref(majit_ir::GcRef(
            unsafe { pyre_object::gc_roots::shadow_stack_get(list_slot) } as usize,
        )),
    );
    let zip = ctx.trace_ctx.record_op_with_descr(
        OpCode::NewWithVtable,
        &[],
        crate::descr::w_zip_size_descr(),
    );
    ctx.trace_ctx.heap_cache_mut().new_object(zip);
    for (value, descr) in [
        (iterator_list, crate::descr::zip_iterators_descr()),
        (ctx.trace_ctx.const_int(1), crate::descr::zip_strict_descr()),
        (zero, crate::descr::zip_iteration_progress_descr()),
    ] {
        ctx.trace_ctx
            .record_op_with_descr(OpCode::SetfieldGc, &[zip, value], descr.clone());
        ctx.trace_ctx
            .heapcache_setfield_cached(zip, descr.index(), value);
    }
    ctx.trace_ctx
        .heap_cache_mut()
        .class_now_known(zip, &pyre_object::functional::ZIP_TYPE as *const _ as i64);
    ctx.trace_ctx.set_opref_concrete(
        zip,
        Value::Ref(majit_ir::GcRef(
            unsafe { pyre_object::gc_roots::shadow_stack_get(zip_slot) } as usize,
        )),
    );
    write_residual_call_result_to_dst(ctx, op.pc, dst, 'r', zip)?;
    drop(roots);
    Ok(Some(()))
}

/// Cut the trace from INSIDE the `locals()` expansion once the recorded
/// length crosses `trace_limit`.
///
/// `pyjitpl.py` `MetaInterp._interpret` asks `blackhole_if_trace_too_long()`
/// after every jitcode step, so the `@jit.unroll_safe` `fast2locals` it looks
/// into is interrupted between its own steps and the raise leaves a partly
/// filled mapping behind.  Both arms here record the whole unroll inside ONE
/// Python opcode and `mod.rs` asks only once that opcode returns, so with no
/// cut the overshoot is the frame's own `co_nlocals` and nothing bounds it.
///
/// The cut is upstream's, not a refusal standing in for it: nothing is
/// estimated, the same `history.length() > trace_limit` decides, and what
/// happens on a yes is the abort `mod.rs` performs one opcode later --
/// `latch_abort_blackhole`, `note_root_trace_too_long`
/// (`stage_abort_reason(ABORT_TOO_LONG)`), `TraceTooLong`.  The trace is
/// discarded whole, so the half-built expansion above this point is never
/// published; resuming re-executes the opcode from `pc`, which is the position
/// every guard this expansion emits already side-exits to
/// (`walker_emit_fold_guard_with_snapshot`), and the concrete mapping the fold
/// built before emitting is a pure function of the fastlocals, so the residual
/// redoes it with the same outcome.
///
/// `trace_too_long_abort_safe`'s bar is kept: with effects already executed and
/// no blackhole image to hand them to,
/// `run_blackhole_interp_to_cancel_tracing` has nowhere to resume, so the walk
/// goes on recording exactly as it does there.
fn locals_expansion_cut_if_too_long<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
    pc: usize,
) -> Result<(), DispatchError> {
    if !ctx.trace_ctx.is_too_long() {
        return Ok(());
    }
    // The row is what makes the cut visible to `check.py`: it ends the trace
    // rather than returning a specialization, so nothing downstream of the
    // fold changes when it is removed, and only this census does.
    // `locals_expansion_trace_too_long` declares it under `spec-folds`.
    if !super::diag::spec_gate_locals_trace_limit_cut() {
        return Ok(());
    }
    let latched = residual_call::latch_abort_blackhole(ctx, pc, "locals-expansion");
    if !latched && super::fbw_state::fbw_executed_effect_count() != 0 {
        majit_metainterp::mc_diag_bump(26);
        return Ok(());
    }
    let ops = ctx.trace_ctx.num_recorded_ops();
    crate::state::note_root_trace_too_long(
        ctx.trace_ctx.current_merge_points_first_green_key_pair(),
        ctx.trace_ctx.resumekey_original_loop_token().cloned(),
    );
    Err(DispatchError::TraceTooLong { pc, ops })
}

/// Which builtin [`try_walker_specialize_builtin_locals`] is standing in for.
///
/// All three resolve their frame through the same `topframe_for_locals` and
/// read the same fastlocals, so one modelled expansion serves them; they
/// differ only in what they make of the resulting mapping.
#[derive(Clone, Copy, PartialEq, Eq)]
enum FrameLocalsBuiltin {
    /// `locals()` / `vars()` — the mapping itself is the result.
    Mapping,
    /// `dir()` — the mapping's sorted key set is the result.
    SortedNames,
}

/// Zero-argument `locals()` / `vars()` / `dir()` on the walk's own portal
/// frame: model
/// `pyframe.py fast2locals` in the trace instead of residualizing
/// `interp_inspect.py locals` → `pyframe.py getdictscope`.
///
/// `fast2locals` is `@jit.unroll_safe`, and `policy.py:60-67` cancels
/// `contains_loop` for unroll_safe graphs, so upstream LOOKS INSIDE it: each
/// `self.locals_cells_stack_w[i]` lowers to `getarrayitem_vable_r`
/// (`jtransform.py do_fixed_list_getitem`), answered from
/// `metainterp.virtualizable_boxes`, and `jtransform.py:2164-2172
/// rewrite_op_jit_force_virtualizable` returns `[]` for a read the tracer is
/// inside.  There is no residual and no virtualizable force anywhere on the
/// upstream locals-read path.
///
/// Pyre residualizes the same read as one opaque `bh_call_fn(locals, PY_NULL)`
/// `CallMayForce`, which arms `virtualizable.py:281-291
/// force_virtualizable_if_necessary` for the whole call; the read barrier
/// `force_frame_before_locals_read` then clears `TOKEN_TRACING_RESCALL` and
/// `tracing_after_residual_call` reads that clear as an escape
/// (`VableEscapedDuringResidualCall`), losing the loop.  The deviation is the
/// residual BOUNDARY, not the barrier — `rvirtualizable.py:49-53` injects the
/// same hook on reads upstream and `pyjitpl.py:3373-3390` aborts
/// unconditionally on a detected force — so this removes the boundary and
/// leaves the barrier live for every shape it declines.
///
/// Emitted shape, mirroring `pyframe.py:555-574` line by line:
/// `guard_value(callable)`; the frame's own mapping, read as
/// `getorcreatedebug()` (the `debugdata` virtualizable field, answered from
/// `virtualizable_boxes`) followed by `getfield_gc_r(w_locals)` under a
/// non-null and exact-dict guard; one `getarrayitem_vable_r(frame,
/// ConstInt(i))` per fastlocal (the same lowering `emit_load_fast_ref!`
/// already emits for LOAD_FAST); a `guard_isnull` / `guard_nonnull` pinning
/// the slot's bound-ness; and a plain non-forcing `Call` per slot —
/// `setitem_str` when bound, `delitem` when not.  None of those ops can reach
/// `force_frame`, so nothing arms the vable protocol.
///
/// The mapping is the FRAME's whenever the frame already carries one:
/// `fast2locals` rewrites only the varname keys, so a foreign key — one an
/// `f_locals` write put there (PEP 667) — survives every call, and an
/// expansion that always started from an empty dict would drop it for as long
/// as the loop stayed compiled.  A frame that carries none keeps the empty
/// `newdict` (pyframe.py:557) the residual would have materialised, under a
/// `guard_isnull` that side-exits if one appears mid-loop; nothing else
/// references that dict, so it is already the independent copy
/// `frame_locals_snapshot` hands back and its `delitem` arm is a no-op.
///
/// One further non-forcing `Call` turns that mapping into the published
/// result: `jit_locals_dict_snapshot` (the independent PEP 667 copy
/// `frame_locals_snapshot` builds) for `locals()` / `vars()`, and
/// `jit_dir_names_from_locals` (the split-out tail of `builtin_dir`'s
/// no-argument path, which reads `getdictscope` rather than the copy) for
/// `dir()`.  Both take the mapping and not the frame, so they too cannot
/// reach `force_frame`.
///
/// An inline level answers from its own frame model instead
/// ([`try_walker_specialize_builtin_locals_in_callee`]), because the frame
/// `locals()` reports on there is the callee's and a trace has exactly one
/// standard virtualizable.
///
/// Returns `None` (fall through to the generic residual, SAFE — exactly
/// today's behaviour) for every other shape: a rebound `locals` / `vars` /
/// `dir` name,
/// a bound receiver, any argument, a frame that is not the
/// standard virtualizable the boxes describe, a hidden top frame, a
/// non-OPTIMIZED (module / class / exec) frame, cellvars / freevars /
/// `CO_FAST_HIDDEN` slots, a slot the shadow cannot answer with a Ref, a
/// shadow whose mapping is not the frame's, and a frame-owned mapping that is
/// not an exact dict.
pub(crate) fn try_walker_specialize_builtin_locals<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    code: &[u8],
    op: &DecodedOp,
    r_args: &[OpRef],
    dst: usize,
) -> Result<Option<()>, DispatchError> {
    // Plain zero-argument `bh_call_fn(callable, PY_NULL)` shape only.
    if r_args.len() != 2 {
        return Ok(None);
    }
    let arg_concretes = read_ref_var_list_concrete(code, op, 1, ctx);
    let (ConcreteValue::Ref(concrete_callable), ConcreteValue::Ref(null_or_self)) =
        (arg_concretes[0], arg_concretes[1])
    else {
        return Ok(None);
    };
    // A non-null `null_or_self` is a bound receiver `bh_call_fn_impl` prepends
    // as arg0 — not a plain `locals()` call.
    if concrete_callable.is_null() || !null_or_self.is_null() {
        return Ok(None);
    }
    // `vars()` with no argument delegates straight to `builtin_locals`
    // (`app_inspect.py`), so both names share the fold; `vars(obj)` and
    // `dir(obj)` carry an extra operand and are already excluded by the arity
    // gate.
    let fold = if pyre_interpreter::builtins::is_builtin_locals_function(concrete_callable)
        || pyre_interpreter::builtins::is_builtin_vars_function(concrete_callable)
    {
        FrameLocalsBuiltin::Mapping
    } else if pyre_interpreter::builtins::is_builtin_dir_function(concrete_callable) {
        FrameLocalsBuiltin::SortedNames
    } else {
        return Ok(None);
    };
    // Inside an inline sub-walk or an inlined callee body, the frame
    // `gettopframe_nohidden()` resolves is the CALLEE's, which the expansion
    // below cannot answer for: it reads from the standard virtualizable, and a
    // trace has exactly one of those.  `MIFrame._nonstandard_virtualizable`
    // tests against `metainterp.virtualizable_boxes[-1]`, and
    // `_opimpl_recursive_call` / `perform_call` push a MIFrame without
    // rebinding those boxes, so upstream an inlined callee's frame is an
    // ordinary virtual and its `fast2locals` traces through reading that
    // virtual's fields.  Pyre's counterpart of the virtual is the level's own
    // [`CalleeLocalsShadow`], so the callee arm answers from it; every gate it
    // fails declines to the generic residual, exactly as this refusal did.
    if ctx.fbw_mode.inline_subwalk || current_inline_concrete_frame() != 0 {
        return try_walker_specialize_builtin_locals_in_callee(
            ctx,
            op,
            fold,
            r_args,
            concrete_callable,
            dst,
        );
    }
    let (Some(vable_op), Some(vable_ptr)) = (
        ctx.trace_ctx.standard_virtualizable_box(),
        ctx.trace_ctx.standard_virtualizable_ptr(),
    ) else {
        return Ok(None);
    };
    // The frame `locals()` reports on is `ec.gettopframe_nohidden()`
    // (`interp_inspect.py:7-11`).  Resolve it the same way and require it to BE
    // the standard virtualizable: a hidden portal frame, or any deeper frame
    // handed out through the backref chain, resolves elsewhere and declines.
    let ec = pyre_interpreter::call::getexecutioncontext();
    if ec.is_null() {
        return Ok(None);
    }
    let frame = unsafe { (*ec).gettopframe_nohidden() };
    if frame.is_null() || frame as usize != vable_ptr {
        return Ok(None);
    }
    let frame_ref = unsafe { &*frame };
    let code_ptr = unsafe { pyre_interpreter::pyframe::pyframe_get_pycode(frame_ref) };
    let code_obj = unsafe { &*code_ptr };
    if !pyre_interpreter::PyFrame::code_locals_are_plain_fastlocals(code_obj) {
        return Ok(None);
    }
    let numlocals = code_obj.varnames.len();
    // No width ceiling.  `pyframe.py` `PyFrame.fast2locals` carries
    // `@jit.unroll_safe` and no ceiling of its own, and the length question is
    // `trace_limit`'s.  A fixed ceiling of 32 slots asked a different one, so a
    // frame answered `locals()` from a residual because of its own local count
    // while a trace many times longer was recorded beside it.
    //
    // No preflight takes its place either.  Upstream has nothing in that
    // place: `blackhole_if_trace_too_long` runs from `MetaInterp._interpret`
    // AFTER `run_one_step`, and no refusal on an estimated cost appears
    // anywhere in that path.  One refusing when `recorded + 2 * nslots >
    // trace_limit` was written and dropped, because near the limit it turned a
    // read that would have fitted into the forcing residual.
    //
    // The length question is answered where upstream answers it -- on the
    // recorded length, inside the unroll.  Upstream's step is one jitcode, so
    // the `fast2locals` it looks into is interrupted inside its own loop; this
    // fold is one Python opcode, so `locals_expansion_cut_if_too_long` runs
    // that same check per slot and aborts the trace from there rather than
    // leaving the overshoot to `mod.rs` one opcode later.
    // `locals_cells_stack_w` is PyFrame's only virtualizable array
    // (`virtualizable_gen.rs arrays`), so array index 0 names it.
    let Some(info) = ctx.trace_ctx.virtualizable_info().cloned() else {
        return Ok(None);
    };
    let Some(lengths) = ctx
        .trace_ctx
        .virtualizable_array_lengths()
        .map(<[usize]>::to_vec)
    else {
        return Ok(None);
    };
    if info.num_arrays() != 1 || lengths.first().copied().unwrap_or(0) < numlocals {
        return Ok(None);
    }
    let (Some(fdescr), Some(adescr)) = (
        info.array_field_descrs().first().cloned(),
        info.array_descrs.first().cloned(),
    ) else {
        return Ok(None);
    };
    // `fast2locals` opens on `self.getorcreatedebug()` (pyframe.py) and
    // writes into ITS `w_locals`: the mapping is the FRAME's, carried across
    // calls, so a key written through `f_locals` outlives every `fast2locals`
    // that does not name it.  `debugdata` is a virtualizable field, so the read
    // answers from `virtualizable_boxes` and records no op — exactly like the
    // slot reads below — and the frame never becomes an operand, so nothing
    // here can reach `force_frame`.
    let Some((debugdata_op, majit_ir::Value::Ref(debugdata_ref))) = ctx
        .trace_ctx
        .virtualizable_entry_at(crate::virtualizable_spec::DEBUGDATA_VABLE_FIELD_INDEX)
    else {
        return Ok(None);
    };
    // Read the mapping through the SHADOW's payload, which is what the emitted
    // `getfield_gc_r` reads, and require it to be the one the residual would
    // have used.  The two payloads are not the same object: a root portal seed
    // bakes the vable identity against the live frame but expands the shadow
    // from the `snapshot_for_tracing` copy, whose `clone_debugdata_ptr` hands
    // out a fresh `FrameDebugData` around the same `w_locals`.  Comparing the
    // holders would decline every portal trace; comparing the mapping is the
    // invariant that actually has to hold.
    let shadow_debugdata =
        debugdata_ref.as_usize() as *const pyre_interpreter::pyframe::FrameDebugData;
    let w_locals = if shadow_debugdata.is_null() {
        pyre_object::PY_NULL
    } else {
        unsafe { (*shadow_debugdata).w_locals }
    };
    if !std::ptr::eq(w_locals, frame_ref.get_w_locals()) {
        return Ok(None);
    }
    // `f_extra_locals` is the OTHER half of what the residual reports.  A
    // proxy write whose key names no writable fast local lands there
    // (`framelocalsproxy_setitem`) and sets NEITHER `w_locals` nor a slot, and
    // `frame_locals_proxy_snapshot` copies it in ahead of the fastlocals — so
    // the mapping this fold rebuilds from slots alone would silently drop the
    // key.  Decline while the frame carries any, read through the same shadow
    // payload for the same reason `w_locals` is, and pin the null direction
    // with a guard below so a write mid-loop side-exits instead.
    let w_extra_locals = if shadow_debugdata.is_null() {
        pyre_object::PY_NULL
    } else {
        unsafe { (*shadow_debugdata).w_extra_locals }
    };
    if !w_extra_locals.is_null() || !frame_ref.get_extra_locals().is_null() {
        return Ok(None);
    }
    // Two shapes, each pinned by a guard so the compiled loop side-exits when
    // the frame moves to the other one:
    //
    // * the frame already carries its mapping — rewrite THAT, so a foreign key
    //   an `f_locals` write left in it survives, as it does across the
    //   residual's `fast2locals`;
    // * the frame carries none — `fast2locals` would materialise an empty dict
    //   (pyframe.py:556-557 `d.w_locals = space.newdict()`) and fill it from
    //   the fastlocals, and the expansion builds exactly that dict instead of
    //   modelling the store.  Nothing else references it, so it is already the
    //   independent copy `frame_locals_snapshot` would hand back, and a
    //   `delitem` on a key it never held is a no-op.
    //
    // The slot helpers are dict-keyed, so a frame-owned mapping that is not an
    // exact dict declines.
    let canonical_dict = pyre_object::get_instantiate(&pyre_object::pyobject::DICT_TYPE);
    let frame_owned = !w_locals.is_null();
    if frame_owned
        && (canonical_dict.is_null()
            || !unsafe {
                std::ptr::eq((*w_locals).ob_type, &pyre_object::pyobject::DICT_TYPE)
                    && std::ptr::eq((*w_locals).w_class, canonical_dict)
            })
    {
        return Ok(None);
    }
    // Resolve every slot's shadow entry BEFORE emitting anything, so a slot the
    // shadow cannot answer declines from a clean trace position.  The read is
    // the standard-virtualizable arm of `_opimpl_getarrayitem_vable`
    // (`virtualizable_boxes[index]`), which records no op — the emit pass below
    // re-runs it through the real entry point.
    let mut slots: Vec<pyre_object::PyObjectRef> = Vec::with_capacity(numlocals);
    for i in 0..numlocals {
        let flat = info.get_index_in_array(0, i, &lengths);
        let Some((slot_op, entry_value)) = ctx.trace_ctx.virtualizable_entry_at(flat) else {
            return Ok(None);
        };
        // The value comes from the SHADOW, never from `locals_w!(frame)`.  An
        // unsynchronized virtualizable's heap array holds whatever the frame
        // last wrote out — measured one FOR_ITER iteration behind on the loop
        // variable — which is exactly the staleness the read barrier's
        // `force_now` repairs before the residual reads it.  The shadow already
        // holds the repaired value, so sourcing from it reproduces the forced
        // residual's answer without the force, and it is what upstream's
        // traced-in `fast2locals` reads (`getarrayitem_vable_r` answered from
        // `virtualizable_boxes`).
        //
        // Prefer the OpRef's own concrete over the `virtualizable_values` copy:
        // the op table is the GC-forwarded channel, so a Ref that moved across
        // an earlier residual is current there.
        let value = match ctx
            .trace_ctx
            .concrete_of_opref(slot_op)
            .filter(|v| matches!(v, majit_ir::Value::Ref(_)))
            .unwrap_or(entry_value)
        {
            majit_ir::Value::Ref(gcref) => gcref.as_usize() as pyre_object::PyObjectRef,
            _ => return Ok(None),
        };
        slots.push(value);
    }

    // Which helper turns the mapping into the published result.  `dir()` reads
    // `getdictscope` — the mapping itself — through `builtin_dir`'s split-out
    // sorted-key-set tail.  `locals()` / `vars()` hand back
    // `frame_locals_snapshot`'s independent PEP 667 copy, which a mapping the
    // expansion just built for itself already is.
    let tail_fn: Option<extern "C" fn(i64) -> i64> = match fold {
        FrameLocalsBuiltin::Mapping if frame_owned => {
            Some(pyre_interpreter::pyframe::jit_locals_dict_snapshot)
        }
        FrameLocalsBuiltin::Mapping => None,
        FrameLocalsBuiltin::SortedNames => {
            Some(pyre_interpreter::builtins::jit_dir_names_from_locals)
        }
    };
    // Authentic mapping, built on the plain eval loop exactly as the skipped
    // residual would — through the SAME helpers the emitted calls invoke, so
    // the recording-time value and the compiled loop's value cannot diverge.
    // On the frame-owned arm this MUTATES the frame's own mapping, which is
    // exactly what the residual `fast2locals` does; the rewrite is a pure
    // function of the fastlocals, so a decline below — or a discarded walk —
    // leaves the residual free to redo it with the same outcome.
    let (concrete_locals, concrete_result) = {
        let _roots = pyre_object::gc_roots::push_roots();
        let locals_root = pyre_object::gc_roots::shadow_stack_len();
        // Re-read rather than reuse the gate's `w_locals`: the slot resolution
        // above sits between the two, so the pin takes the address the frame
        // holds NOW.
        let _ = pyre_object::gc_roots::pin_root(if frame_owned {
            frame_ref.get_w_locals()
        } else {
            unsafe { pyre_object::w_dict_new() }
        });
        let value_roots: Vec<usize> = slots
            .iter()
            .map(|&value| {
                let slot = pyre_object::gc_roots::shadow_stack_len();
                let _ = pyre_object::gc_roots::pin_root(value);
                slot
            })
            .collect();
        let mut result = pyre_object::PY_NULL;
        let mut slot_failed = false;
        for (i, &value_root) in value_roots.iter().enumerate() {
            let value = pyre_object::gc_roots::shadow_stack_get(value_root);
            let locals = pyre_object::gc_roots::shadow_stack_get(locals_root) as i64;
            // pyframe.py:566-574 — a bound slot is stored, an unbound one
            // deleted.  Both allocate, so the mapping is re-read from its
            // pinned slot on every pass.  A fresh mapping never held the key,
            // so its `delitem` arm is skipped rather than emitted.
            let updated = if !value.is_null() {
                pyre_interpreter::pyframe::jit_locals_dict_setitem_local(
                    locals,
                    code_ptr as i64,
                    i as i64,
                    value as i64,
                )
            } else if frame_owned {
                pyre_interpreter::pyframe::jit_locals_dict_delitem_local(
                    locals,
                    code_ptr as i64,
                    i as i64,
                )
            } else {
                locals
            };
            if (updated as pyre_object::PyObjectRef).is_null() {
                slot_failed = true;
                break;
            }
        }
        if !slot_failed {
            let locals = pyre_object::gc_roots::shadow_stack_get(locals_root);
            // The tail runs here too, so the recorded result is produced by the
            // very helper the emitted call names.
            result = match tail_fn {
                Some(tail) => tail(locals as i64) as pyre_object::PyObjectRef,
                None => locals,
            };
        }
        (pyre_object::gc_roots::shadow_stack_get(locals_root), result)
    };
    // A slot rewrite or the tail reports a failure as PY_NULL instead of
    // publishing it; nothing has been emitted yet, so decline and let the
    // residual raise.
    if concrete_result.is_null() {
        return Ok(None);
    }
    let concrete_locals_value = majit_ir::Value::Ref(majit_ir::GcRef(concrete_locals as usize));

    // --- emit the specialized IR (walker-native) ---
    // Pin the callable identity (LOAD_GLOBAL `locals` is usually already a
    // constant via the namespace cell fold).
    let callable_op = r_args[0];
    if !callable_op.is_constant() {
        let expected = ctx.trace_ctx.const_ref(concrete_callable as i64);
        ctx.trace_ctx
            .record_guard(OpCode::GuardValue, &[callable_op, expected], 0);
        walker_capture_snapshot_for_last_guard(ctx, op.pc)?;
        ctx.trace_ctx
            .heap_cache_mut()
            .replace_box(callable_op, expected);
    }
    // The code object is the jitdriver green this trace is keyed on
    // (`interp_jit.py:23 greens = ['next_instr', 'is_being_profiled',
    // 'pycode']`), so its address is a constant for the compiled loop and
    // carries no guard of its own.
    let code_const = ctx.trace_ctx.const_int(code_ptr as i64);
    // `d = self.getorcreatedebug()` — pyframe.py.  An absent payload has no
    // `w_locals` to read, so the guard pins that direction and the fresh-dict
    // arm below stands in for the materialisation.
    let debugdata_present = debugdata_ref.as_usize() != 0;
    if !debugdata_op.is_constant() {
        let opcode = if debugdata_present {
            OpCode::GuardNonnull
        } else {
            OpCode::GuardIsnull
        };
        walker_emit_fold_guard_with_snapshot(ctx, op.pc, opcode, &[debugdata_op])?;
    }
    // `d.w_locals` — pyframe.py:556.  Read whenever there is a payload to read
    // it from, and guarded in the direction recorded, so a frame that
    // materialises its mapping mid-loop side-exits instead of going on writing
    // into the expansion's own dict.
    let mut field_op = None;
    if debugdata_present {
        let op_ref = crate::state::opimpl_getfield_gc_r(
            ctx.trace_ctx,
            debugdata_op,
            crate::descr::frame_debug_data_w_locals_descr(),
        );
        if !op_ref.is_constant() {
            let opcode = if frame_owned {
                OpCode::GuardNonnull
            } else {
                OpCode::GuardIsnull
            };
            walker_emit_fold_guard_with_snapshot(ctx, op.pc, opcode, &[op_ref])?;
        }
        field_op = Some(op_ref);
        // `d.w_extra_locals` — the gate above required it absent, so the loop
        // side-exits the moment an `f_locals` write puts a non-fast key there
        // rather than going on publishing a mapping without it.
        let extra_op = crate::state::opimpl_getfield_gc_r(
            ctx.trace_ctx,
            debugdata_op,
            crate::descr::frame_debug_data_w_extra_locals_descr(),
        );
        if !extra_op.is_constant() {
            walker_emit_fold_guard_with_snapshot(ctx, op.pc, OpCode::GuardIsnull, &[extra_op])?;
        }
    }
    let mut dict_op = match field_op.filter(|_| frame_owned) {
        Some(op_ref) => op_ref,
        // pyframe.py `self.space.newdict(instance=True)` — the mapping
        // `fast2locals` would have materialised, built here instead of
        // modelling the store back into the debug payload.
        None => ctx.trace_ctx.call_ref_typed_with_effect(
            pyre_interpreter::pyframe::jit_locals_dict_new as *const (),
            &[],
            &[],
            majit_ir::EffectInfo::new(
                majit_ir::ExtraEffect::CannotRaise,
                majit_ir::OopSpecIndex::None,
            ),
        ),
    };
    ctx.trace_ctx
        .set_opref_concrete(dict_op, concrete_locals_value);
    if frame_owned {
        walker_guard_class(
            ctx,
            op.pc,
            dict_op,
            &pyre_object::pyobject::DICT_TYPE as *const _ as i64,
        )?;
        walker_guard_exact_w_class(
            ctx,
            op.pc,
            dict_op,
            // Re-derived rather than reusing the gate's binding: the
            // record-time rewrite above allocates, so this takes the address
            // `dict` has NOW.
            pyre_object::get_instantiate(&pyre_object::pyobject::DICT_TYPE),
        )?;
    }
    for (i, &value) in slots.iter().enumerate() {
        locals_expansion_cut_if_too_long(ctx, op.pc)?;
        // `self.locals_cells_stack_w[i]` — `jtransform.py:1877
        // do_fixed_list_getitem`, the identical lowering `emit_load_fast_ref!`
        // emits for LOAD_FAST.  On the standard virtualizable this resolves to
        // `virtualizable_boxes[index]` and records no op.
        let index_const = ctx.trace_ctx.const_int(i as i64);
        let (slot_op, _) = ctx.trace_ctx.vable_getarrayitem_ref_indexed(
            op.pc,
            vable_op,
            index_const,
            i as i64,
            fdescr.clone(),
            adescr.clone(),
        );
        // `pyframe.py:566-571` branches on the slot being bound; pin the
        // direction so a slot that changes bound-ness side-exits instead of
        // publishing a mapping with the wrong key set.  A slot the trace
        // already holds as a constant needs no guard.
        let bound = !value.is_null();
        if !slot_op.is_constant() {
            let opcode = if bound {
                OpCode::GuardNonnull
            } else {
                OpCode::GuardIsnull
            };
            walker_emit_fold_guard_with_snapshot(ctx, op.pc, opcode, &[slot_op])?;
        }
        // `pyframe.py:566-574` — a bound slot is stored, an unbound one
        // deleted.  The delete is what keeps a key from a since-unbound local
        // out of a mapping the frame carries across calls; on the fresh arm
        // the mapping never held the key, so it is skipped.
        if !bound && !frame_owned {
            continue;
        }
        let (helper, args, arg_types): (_, Vec<OpRef>, Vec<majit_ir::Type>) = if bound {
            (
                pyre_interpreter::pyframe::jit_locals_dict_setitem_local as *const (),
                vec![dict_op, code_const, index_const, slot_op],
                vec![
                    majit_ir::Type::Ref,
                    majit_ir::Type::Int,
                    majit_ir::Type::Int,
                    majit_ir::Type::Ref,
                ],
            )
        } else {
            (
                pyre_interpreter::pyframe::jit_locals_dict_delitem_local as *const (),
                vec![dict_op, code_const, index_const],
                vec![
                    majit_ir::Type::Ref,
                    majit_ir::Type::Int,
                    majit_ir::Type::Int,
                ],
            )
        };
        dict_op = ctx.trace_ctx.call_ref_typed_with_effect(
            helper,
            &args,
            &arg_types,
            majit_ir::EffectInfo::new(
                majit_ir::ExtraEffect::CannotRaise,
                majit_ir::OopSpecIndex::None,
            ),
        );
        // Every link of the chain names the SAME mapping, so the post-build
        // address is the live one for all of them.
        ctx.trace_ctx
            .set_opref_concrete(dict_op, concrete_locals_value);
        if !bound {
            // The delete reports a raising comparison as PY_NULL instead of
            // publishing it; side-exit so the residual re-runs and raises.
            walker_emit_fold_guard_with_snapshot(ctx, op.pc, OpCode::GuardNonnull, &[dict_op])?;
        }
    }
    // Asked again with the loop behind it: the check above runs BEFORE a slot
    // emits, so the last slot's own ops -- and the tail below -- would
    // otherwise reach the opcode-level check in `mod.rs` unweighed.
    locals_expansion_cut_if_too_long(ctx, op.pc)?;
    // `frame_locals_snapshot`'s PEP 667 copy for `locals()` / `vars()`, or
    // `builtin_dir`'s no-argument tail for `dir()` — each split out so the
    // trace and the eval loop run one implementation.  Both report a failure
    // as PY_NULL instead of publishing it, so the guarded side exit re-runs
    // the residual and raises from the eval loop.
    let result_op = match tail_fn {
        Some(tail) => {
            let op_ref = ctx.trace_ctx.call_ref_typed_with_effect(
                tail as *const (),
                &[dict_op],
                &[majit_ir::Type::Ref],
                majit_ir::EffectInfo::new(
                    majit_ir::ExtraEffect::CannotRaise,
                    majit_ir::OopSpecIndex::None,
                ),
            );
            ctx.trace_ctx.set_opref_concrete(
                op_ref,
                majit_ir::Value::Ref(majit_ir::GcRef(concrete_result as usize)),
            );
            walker_emit_fold_guard_with_snapshot(ctx, op.pc, OpCode::GuardNonnull, &[op_ref])?;
            op_ref
        }
        None => dict_op,
    };
    write_residual_call_result_to_dst(ctx, op.pc, dst, 'r', result_op)?;
    Ok(Some(()))
}

/// [`try_walker_specialize_builtin_locals`]'s arm for an inlined callee level.
///
/// The frame `builtin_locals` reports on here is the level's own, never the
/// standard virtualizable, so none of the portal expansion's vable reads
/// apply.  Upstream has the same asymmetry and resolves it the same way: one
/// standard virtualizable per trace, an inlined callee frame left an ordinary
/// virtual, and `pyframe.py fast2locals` — `@jit.unroll_safe`, so
/// `policy.py` looks inside it — traced through reading that virtual's
/// fields.  Pyre's counterpart of the virtual is [`CalleeLocalsShadow`]: every
/// visible fastlocal of this level is already an SSA value the walk holds, and
/// `getarrayitem_vable_via_metainterp`'s strict fresh-frame fold is what
/// answers the level's own `LOAD_FAST` from it.  Sourcing the expansion from
/// the same map is therefore the same read the callee's own bytecode makes.
///
/// Emitted shape: `guard_value(callable)` when the name is not already a trace
/// constant; `jit_locals_dict_new` (the `space.newdict(instance=True)` a fresh
/// frame's `fast2locals` materialises); one non-forcing `jit_locals_dict_setitem_local`
/// `Call` per BOUND slot, taking the slot's SSA value straight as an operand;
/// and `jit_dir_names_from_locals` for `dir()`.  No vable read, no `PyFrame`
/// operand, so nothing on it can reach `force_frame` — which is the point,
/// since the opaque residual this replaces forces the published callee frame
/// and `tracing_after_residual_call` reads that as an escape.
///
/// No per-slot boundness guard: a slot's SSA value exists precisely because a
/// param seed or a `STORE_FAST` on the traced path produced it, and the guards
/// already on that path pin which of them ran.  That is the difference from
/// the portal arm, whose slots come out of a virtualizable array the compiled
/// loop re-reads.
///
/// A level whose frame the seed block materialised is NOT excluded.  The
/// `frame_materialized` flag governs STORES — `folded_store_is_observable_local`
/// demotes a `STORE_FAST` into a recorded `SETARRAYITEM_GC` so a frame reached
/// later through a traceback, `f_locals` or `sys._getframe` sees the value —
/// and that demoted store still re-seeds `opref`, so the shadow and the heap
/// array hold the same value.  Reading is what this arm does, and it reads the
/// channel the level's own `LOAD_FAST` reads.
///
/// Returns `None` (fall through to the generic residual, SAFE — exactly the
/// behaviour this arm replaced) for every other shape: a sub-walk whose guards
/// would collapse to the caller's CALL boundary, a level with no shadow, an
/// inactive strict fold or unseeded frame register, a top frame that is not
/// this level's own, a shadow describing another code object, a non-OPTIMIZED
/// frame, cellvars / freevars / `CO_FAST_HIDDEN` slots, a frame that already
/// carries a locals mapping or an `f_extra_locals` dict, and a written slot
/// the shadow cannot resolve back to a Ref.  `PYRE_FBW_DEBUG_ABORT` names
/// which of them declined.
fn try_walker_specialize_builtin_locals_in_callee<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op: &DecodedOp,
    fold: FrameLocalsBuiltin,
    r_args: &[OpRef],
    concrete_callable: pyre_object::PyObjectRef,
    dst: usize,
) -> Result<Option<()>, DispatchError> {
    if let Some(done) = try_walker_specialize_builtin_locals_in_callee_expand(
        ctx,
        op,
        fold,
        r_args,
        concrete_callable,
        dst,
    )? {
        return Ok(Some(done));
    }
    // The expansion declined, so this level is about to run the opaque
    // residual -- and that residual's `force_frame_before_locals_read` clears
    // `TOKEN_TRACING_RESCALL` on the level's own published frame, which
    // `tracing_after_residual_call` reads as `VableEscapedDuringResidualCall`.
    // Falling through therefore does not cost one residual call; it costs the
    // enclosing loop, because the escape is a property of the callee BODY and
    // every retry rebuilds the same framestack and escapes again until
    // `MAX_TRACE_ABORT_COUNT` retires the caller.
    //
    // Refuse the callee HERE instead, before the residual runs.  The caller
    // then records the plain `bh_call_fn` it would have recorded had this body
    // never been admitted, and the escape never happens: same answer, one
    // decline instead of an abort.  What reaches this line is a shape the
    // expansion models no part of -- a non-OPTIMIZED frame, a `CO_FAST_HIDDEN`
    // slot, a frame already carrying an `f_locals` mapping or an
    // `f_extra_locals` dict, or a slot whose value this walk never saw.  Each
    // of those is named under `PYRE_FBW_DEBUG_ABORT`, because the refusal is
    // not the answer: it stands in for an expansion that does not model the
    // shape yet, and widening the expansion until this line is unreachable is
    // the convergence path.
    //
    // It is unreachable today.  Measured 2026-08-29 on release dynasm over the
    // 507 `bench/synth` fixtures, all of which exit 0: not one `[decline-why]
    // LOCALS-IN-CALLEE` line anywhere, so no shape in the corpus reaches this
    // refusal.  Before the width gate came off, `locals_in_wide_inlined_callee`
    // reported the single line `nslots-over-cap nslots=42 name=wide`; that was
    // the only one the corpus produced, and no other gate has ever been
    // observed to fire.  Each gate below now records why it does not: the
    // non-OPTIMIZED refusal is the answer rather than a gap, its
    // `CO_FAST_HIDDEN` half is a bit this compiler never sets, and the two
    // frame-payload gates sit behind writers that need a reference to the
    // level's own live frame.
    if let Some(callee) = super::fbw_state::fbw_innermost_inline_callee_key(ctx) {
        return Err(super::fbw_state::fbw_decline_inline_callee(
            ctx,
            op.pc,
            Some(callee),
        ));
    }
    Ok(None)
}

/// One slot of an inlined callee's frame that the modelled `fast2locals`
/// reproduces.
struct ModelledLocalSlot {
    /// The localsplus slot index, which is also the `varnames` index below
    /// `numlocals` and, above it, `numlocals + cell_slot_names` index.
    index: i64,
    /// What the walk holds AT the slot: the bound value for a plain
    /// fastlocal, the `Cell` for a cell slot.
    slot_op: OpRef,
    /// Whether `slot_op` is a `Cell` whose contents this slot's key takes.
    cell: bool,
    /// The recording-time value the key would be bound to, `PY_NULL` for an
    /// empty cell (which binds no key).
    value: pyre_object::PyObjectRef,
}

impl ModelledLocalSlot {
    /// The `fast2locals` binder for this slot and the index it names its key
    /// with: `code.varnames[index]` for a slot below `numlocals` — a shared
    /// cellvar slot included, since that is the name it carries — and
    /// `cell_slot_names(code)[index - numlocals]` above it.
    fn binder(&self, numlocals: usize) -> (extern "C" fn(i64, i64, i64, i64) -> i64, i64) {
        if (self.index as usize) < numlocals {
            (
                pyre_interpreter::pyframe::jit_locals_dict_setitem_local,
                self.index,
            )
        } else {
            (
                pyre_interpreter::pyframe::jit_locals_dict_setitem_cell,
                self.index - numlocals as i64,
            )
        }
    }
}

/// The expansion itself: `Ok(None)` means "this shape is not modelled", which
/// its caller turns into an inline refusal rather than a residual.
fn try_walker_specialize_builtin_locals_in_callee_expand<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op: &DecodedOp,
    fold: FrameLocalsBuiltin,
    r_args: &[OpRef],
    concrete_callable: pyre_object::PyObjectRef,
    dst: usize,
) -> Result<Option<()>, DispatchError> {
    /// Name the gate that declined.  Unlike the portal arm's decline, which
    /// falls through to the generic residual, this one costs the caller its
    /// whole inlined callee, so "the expansion models no part of this shape"
    /// has to be attributable to one gate rather than re-derived by reading.
    macro_rules! decline {
        ($why:literal) => {{
            if fbw_debug_abort_enabled() {
                eprintln!("[decline-why] LOCALS-IN-CALLEE {}", $why);
            }
            return Ok(None);
        }};
        ($why:literal, $($arg:tt)*) => {{
            if fbw_debug_abort_enabled() {
                eprintln!("[decline-why] LOCALS-IN-CALLEE {}", format!($why, $($arg)*));
            }
            return Ok(None);
        }};
    }
    // Under a single-frame collapse the resume re-executes the whole call, so
    // a guard emitted here re-runs every side effect the inline region already
    // sequenced.  Same gate the other folds that run under a sub-walk take.
    if ctx.fbw_mode.inline_subwalk && !walker_inline_guard_resumes_in_callee(ctx) {
        decline!("subwalk-no-callee-resume");
    }
    let (fold_frame_reg, shadow_code_ptr) = {
        let Some(shadow) = ctx.callee_shadow.as_ref() else {
            decline!("no-callee-shadow");
        };
        // `u16::MAX` is the strict fresh-frame fold switched off, and a
        // `NONE` frame box is a frame register that was never seeded — in
        // neither case is the shadow the authority for this level's slots.
        if shadow.fold_frame_reg == u16::MAX || shadow.frame_box.is_none() {
            decline!("unseeded-frame-register");
        }
        (shadow.fold_frame_reg, shadow.code_ptr)
    };
    // `interp_inspect.py locals` reaches its frame through
    // `gettopframe_nohidden`, and `walker_ec_enter` has published THIS level's
    // concrete frame there for the whole sub-walk.  Require the two to name
    // one frame, so a level that never entered the chain — or one whose top
    // frame is someone else's — declines instead of answering for the wrong
    // frame.  Read the identity back through the guard's root, which the
    // collector forwards.
    let inline_frame = current_inline_concrete_frame();
    let ec = pyre_interpreter::call::getexecutioncontext();
    if inline_frame == 0 || ec.is_null() {
        decline!("no-inline-frame-or-ec");
    }
    let frame = unsafe { (*ec).gettopframe_nohidden() };
    if frame.is_null() || frame as usize != inline_frame {
        decline!("top-frame-not-this-level");
    }
    let frame_ref = unsafe { &*frame };
    let code_ptr = unsafe { pyre_interpreter::pyframe::pyframe_get_pycode(frame_ref) };
    // The shadow's slots index the code object it was opened for; a
    // disagreement means the map does not describe this frame's fastlocals.
    if code_ptr.is_null() || !std::ptr::eq(code_ptr, shadow_code_ptr) {
        decline!("shadow-names-other-code");
    }
    let code_obj = unsafe { &*code_ptr };
    // Two conditions, and only the first can fire.  A non-OPTIMIZED frame
    // answers `locals()` from `PyFrame::getdictscope` — its LIVE namespace,
    // not the independent copy this arm builds — so declining is the answer
    // rather than a gap.  The `CO_FAST_HIDDEN` half is inert: `pycode.rs`
    // builds `localspluskinds` out of `CO_FAST_LOCAL` and `CO_FAST_CELL`
    // alone, so nothing this compiler produces carries the bit, and
    // `PyFrame::fast2locals` skips such a slot only on a frame this arm has
    // already refused.
    if !pyre_interpreter::PyFrame::code_locals_are_modelled_fastlocals(code_obj) {
        decline!("not-modelled-fastlocals");
    }
    let numlocals = code_obj.varnames.len();
    // The pure cellvars and the freevars occupy the slots above `varnames` in
    // the unified layout, and `fast2locals` binds each of them under the name
    // `cell_slot_names` gives it.  A cellvar that shares a varname slot is
    // named by `varnames` and is only a CELL there, which the per-slot kind
    // below picks up.
    let nslots = numlocals + pyre_interpreter::PyFrame::cell_slot_names(code_obj).count();
    // No width ceiling here, and none on the portal arm either: upstream
    // bounds this unroll with `@jit.unroll_safe` on `pyframe.py`
    // `PyFrame.fast2locals` and nothing else, and leaves the length question
    // to `trace_limit`.  What made the ceiling worth removing HERE first is
    // the price of the refusal: the portal arm's decline falls through to the
    // generic residual, while a decline on this arm denies the callee for the
    // rest of the thread's tracing, so the ceiling decided inlinability from a
    // callee's local count.
    //
    // No ceiling and no preflight, for the reason the portal arm records; the
    // per-slot `locals_expansion_cut_if_too_long` below answers the length
    // question instead.  That price is also what makes a preflight worse on
    // this arm than on that one: refusing on the walk's remaining budget would
    // let the length of the trace so far decide a callee's inlinability, the
    // way the ceiling let its local count decide it.  The cut is not that --
    // it ends the trace rather than the callee, so nothing is remembered
    // against the body.

    // Fresh mapping only.  A frame that already carries one — an `f_locals`
    // write (PEP 667), a `setdictscope` — is the portal arm's frame-owned
    // shape, whose rewrite has to reach the frame's own dict across calls;
    // this level's frame is rebuilt from scratch by the compiled trace when it
    // is built at all, so that shape has no counterpart here and declines.
    //
    // Never observed to fire, and the reason is structural: this arm has
    // already required an OPTIMIZED frame, and on one of those `w_locals` has
    // no writer the answer can follow.  `bind_unoptimized_locals_scope`
    // returns before binding it, `PyFrame::fget_getdictscope` hands an
    // optimized frame a `FrameLocalsProxy` rather than calling
    // `getdictscope`, and the line-tracing call to `fast2locals` is guarded on
    // `w_locals` being non-null already, so it cannot be the first setter.
    if !frame_ref.get_w_locals().is_null() {
        decline!("frame-has-w-locals");
    }
    // A null `w_locals` is NOT on its own a fresh mapping.  A proxy write
    // whose key names no writable fast local goes to `f_extra_locals`
    // (`framelocalsproxy_setitem`) and leaves `w_locals` null, and
    // `frame_locals_proxy_snapshot` copies that dict into every mapping it
    // hands back — so rebuilding from the shadow's slots alone would drop the
    // key.  Read the LIVE callee frame, which is the only holder: this level's
    // frame is built by the compiled trace, so its payload starts empty every
    // iteration and only a residual on the recorded path can have filled it.
    //
    // Never observed to fire either, and an earlier gate is why.  The one
    // writer is `FrameLocalsProxy::setitem_value`, so filling the dict takes a
    // reference to this level's own live frame, and that write calls
    // `force_locals` before it stores.  `locals_proxy_extra_key_hot` drives
    // exactly that shape one frame in and reports no decline at all: the
    // expansion is not reached there, because the callee is no longer an
    // un-escaped inline level by the time `locals()` is recorded.
    if !frame_ref.get_extra_locals().is_null() {
        decline!("frame-has-extra-locals");
    }
    // Collect each slot's shadow entry first, so a slot the shadow cannot
    // answer declines from a clean trace position and nothing is emitted.
    let is_cell_slot = |slot: usize| {
        slot >= numlocals
            || (slot < code_obj.localspluskinds.len()
                && code_obj.localspluskinds[slot] & pyre_interpreter::bytecode::CO_FAST_CELL != 0)
    };
    let mut slot_oprefs: Vec<Option<OpRef>> = Vec::with_capacity(nslots);
    {
        let Some(shadow) = ctx.callee_shadow.as_ref() else {
            decline!("shadow-vanished");
        };
        for slot in 0..nslots as i64 {
            match (shadow.opref.get(&slot).copied(), shadow.concrete.get(&slot)) {
                // Absent from both: this walk never wrote the slot, and the
                // frame it would otherwise have kept a value in is fresh, so
                // the slot is UNBOUND and `fast2locals` binds no key for it.
                //
                // A CELL slot is not fresh in that sense: the frame setup
                // built the cell (`MAKE_CELL`) or copied it out of the
                // closure (`COPY_FREE_VARS`) before the first opcode ran, so
                // absence here means the walk never READ it, not that the name
                // is unbound.  There is no SSA value to bind and guessing
                // "unbound" would drop a live name, so decline.
                (None, None) if is_cell_slot(slot as usize) => decline!("cell-slot-unread"),
                (None, None) => slot_oprefs.push(None),
                // Only an entry recorded through THIS level's frame register
                // describes this frame — the same per-frame isolation the
                // `getarrayitem_vable` read fallback applies.
                (Some(opref), Some(concrete)) if concrete.frame_reg == fold_frame_reg => {
                    slot_oprefs.push(Some(opref))
                }
                // Written with no reconstructable concrete half, or through
                // another frame's register: decline rather than guess.
                _ => decline!("slot-not-this-frame"),
            }
        }
    }
    // Every slot that `fast2locals` binds a key for, in slot order.
    //
    // `slot_op` is what the walk holds AT the slot: the value itself for a
    // plain fastlocal, the CELL for a cell slot.  The emit below turns the
    // latter into its contents with one `GETFIELD_GC_R`, which is why the read
    // is not done here — nothing may be emitted while a later slot can still
    // decline.
    let mut slots: Vec<ModelledLocalSlot> = Vec::with_capacity(nslots);
    for (index, entry) in slot_oprefs.iter().enumerate() {
        let Some(slot_op) = *entry else {
            continue;
        };
        // Resolve through the op table, not the shadow's raw `concrete` copy:
        // the table is the GC-forwarded channel, so a Ref that moved across an
        // earlier residual is current there.
        let Some(majit_ir::Value::Ref(gcref)) = ctx.trace_ctx.concrete_of_opref(slot_op) else {
            decline!("slot-concrete-not-ref");
        };
        if gcref == majit_ir::GcRef::NO_CONCRETE {
            decline!("slot-no-concrete");
        }
        let held = gcref.as_usize() as pyre_object::PyObjectRef;
        let cell = is_cell_slot(index);
        if cell {
            // `fast2locals` falls back to the raw slot when it does not hold a
            // cell.  That shape is unreachable for an OPTIMIZED frame past its
            // `MAKE_CELL` / `COPY_FREE_VARS` prologue, and modelling it would
            // need a second arm with its own guard, so decline instead.
            if held.is_null() || !unsafe { pyre_object::is_cell(held) } {
                decline!("cell-slot-not-a-cell");
            }
        }
        let value = if cell {
            unsafe { pyre_object::w_cell_get(held) }
        } else {
            held
        };
        if value.is_null() && !cell {
            // A slot the walk unbound (`DELETE_FAST`).  The mapping is fresh,
            // so it binds no key for it — but only a NULL the trace holds as a
            // constant is unbound on every execution of the compiled path.
            if !slot_op.is_constant() {
                decline!("unbound-slot-not-constant");
            }
            continue;
        }
        slots.push(ModelledLocalSlot {
            index: index as i64,
            slot_op,
            cell,
            value,
        });
    }

    // `dir()` takes `builtin_dir`'s split-out sorted-name tail, which reads
    // `getdictscope` — the mapping itself.  `locals()` / `vars()` need none:
    // nothing else references a mapping the expansion just built, so it is
    // already the independent copy `frame_locals_snapshot` hands back.
    let tail_fn: Option<extern "C" fn(i64) -> i64> = match fold {
        FrameLocalsBuiltin::Mapping => None,
        FrameLocalsBuiltin::SortedNames => {
            Some(pyre_interpreter::builtins::jit_dir_names_from_locals)
        }
    };
    // The slots that bind a key.  An empty cell binds none — `fast2locals`
    // deletes the name there, and this mapping is fresh, so there is nothing
    // to delete.
    let bound: Vec<&ModelledLocalSlot> = slots.iter().filter(|s| !s.value.is_null()).collect();
    // Authentic mapping, built through the SAME helpers the emitted calls
    // name, so the recording-time value and the compiled loop's value cannot
    // diverge.  Nothing here touches the frame, so a decline below — or a
    // discarded walk — leaves the residual free to redo it with the same
    // outcome.
    let (concrete_locals, concrete_result) = {
        let _roots = pyre_object::gc_roots::push_roots();
        // Values first: the `w_dict_new` below allocates, so a slot value
        // still held only as a bare pointer could be moved out from under the
        // pin that was about to take it.
        let value_roots: Vec<usize> = bound
            .iter()
            .map(|slot| {
                let root = pyre_object::gc_roots::shadow_stack_len();
                let _ = pyre_object::gc_roots::pin_root(slot.value);
                root
            })
            .collect();
        let locals_root = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(unsafe { pyre_object::w_dict_new() });
        let mut result = pyre_object::PY_NULL;
        let mut slot_failed = false;
        for (slot, &value_root) in bound.iter().zip(&value_roots) {
            // The store allocates, so both the mapping and the value are
            // re-read from their pinned slots on every pass.
            let (setitem, name_index) = slot.binder(numlocals);
            let updated = setitem(
                pyre_object::gc_roots::shadow_stack_get(locals_root) as i64,
                code_ptr as i64,
                name_index,
                pyre_object::gc_roots::shadow_stack_get(value_root) as i64,
            );
            if (updated as pyre_object::PyObjectRef).is_null() {
                slot_failed = true;
                break;
            }
        }
        if !slot_failed {
            let locals = pyre_object::gc_roots::shadow_stack_get(locals_root);
            result = match tail_fn {
                Some(tail) => tail(locals as i64) as pyre_object::PyObjectRef,
                None => locals,
            };
        }
        (pyre_object::gc_roots::shadow_stack_get(locals_root), result)
    };
    // A slot rewrite or the tail reports a failure as PY_NULL instead of
    // publishing it; nothing has been emitted yet, so decline and let the
    // caller record the plain call, which raises the same way.
    if concrete_result.is_null() {
        decline!("concrete-result-null");
    }
    let concrete_locals_value = majit_ir::Value::Ref(majit_ir::GcRef(concrete_locals as usize));

    // --- emit the specialized IR (walker-native) ---
    let callable_op = r_args[0];
    if !callable_op.is_constant() {
        let expected = ctx.trace_ctx.const_ref(concrete_callable as i64);
        ctx.trace_ctx
            .record_guard(OpCode::GuardValue, &[callable_op, expected], 0);
        walker_capture_snapshot_for_last_guard(ctx, op.pc)?;
        ctx.trace_ctx
            .heap_cache_mut()
            .replace_box(callable_op, expected);
    }
    // The callee's code object is fixed for this inline level, so its address
    // is a constant for the compiled loop and carries no guard of its own.
    let code_const = ctx.trace_ctx.const_int(code_ptr as i64);
    // `w_cell_get` per cell slot, BEFORE the mapping is allocated: each one
    // carries a guard, and a guard that fails after the `newdict` would side
    // exit to a residual that allocates a second mapping.  The read is the
    // whole of `fast2locals`' cell half — `Cell.contents` — and it touches no
    // frame, so it cannot re-arm the escape this expansion exists to remove.
    let mut value_ops: Vec<OpRef> = Vec::with_capacity(slots.len());
    for slot in &slots {
        locals_expansion_cut_if_too_long(ctx, op.pc)?;
        if !slot.cell {
            value_ops.push(slot.slot_op);
            continue;
        }
        // The slot holds a `Cell` on every execution of this path: the frame
        // prologue put it there and nothing in the body replaces it, but the
        // compiled loop re-reads the slot, so say so.
        let cell_type = &pyre_object::nestedscope::CELL_TYPE as *const _ as i64;
        if !ctx.trace_ctx.heap_cache().is_class_known(slot.slot_op) {
            let type_const = ctx.trace_ctx.const_int(cell_type);
            walker_emit_fold_guard_with_snapshot(
                ctx,
                op.pc,
                OpCode::GuardClass,
                &[slot.slot_op, type_const],
            )?;
            ctx.trace_ctx
                .heap_cache_mut()
                .class_now_known(slot.slot_op, cell_type);
        }
        let contents = walker_record_getfield_gc_r_uncached(
            ctx,
            slot.slot_op,
            crate::descr::cell_contents_descr(),
        );
        ctx.trace_ctx.set_opref_concrete(
            contents,
            majit_ir::Value::Ref(majit_ir::GcRef(slot.value as usize)),
        );
        // Boundness is what decides whether this name appears at all, and a
        // cell can be rebound or deleted between iterations, so pin the answer
        // in BOTH directions.
        let guard = if slot.value.is_null() {
            OpCode::GuardIsnull
        } else {
            OpCode::GuardNonnull
        };
        walker_emit_fold_guard_with_snapshot(ctx, op.pc, guard, &[contents])?;
        value_ops.push(contents);
    }
    // Same reason as the portal arm: the last cell read's guard lands after
    // that loop's own check.
    locals_expansion_cut_if_too_long(ctx, op.pc)?;
    // pyframe.py `self.space.newdict(instance=True)` — the mapping a fresh
    // frame's `fast2locals` materialises before filling it.
    let mut dict_op = ctx.trace_ctx.call_ref_typed_with_effect(
        pyre_interpreter::pyframe::jit_locals_dict_new as *const (),
        &[],
        &[],
        majit_ir::EffectInfo::new(
            majit_ir::ExtraEffect::CannotRaise,
            majit_ir::OopSpecIndex::None,
        ),
    );
    ctx.trace_ctx
        .set_opref_concrete(dict_op, concrete_locals_value);
    for (slot, &value_op) in slots.iter().zip(&value_ops) {
        locals_expansion_cut_if_too_long(ctx, op.pc)?;
        if slot.value.is_null() {
            continue;
        }
        // pyframe.py:566-571 — bind this slot's name to its value.  For a
        // plain fastlocal the value is the SSA operand the level's own
        // `LOAD_FAST` would have folded to; for a cell slot it is the
        // `Cell.contents` read emitted above.
        let (setitem, name_index) = slot.binder(numlocals);
        let index_const = ctx.trace_ctx.const_int(name_index);
        dict_op = ctx.trace_ctx.call_ref_typed_with_effect(
            setitem as *const (),
            &[dict_op, code_const, index_const, value_op],
            &[
                majit_ir::Type::Ref,
                majit_ir::Type::Int,
                majit_ir::Type::Int,
                majit_ir::Type::Ref,
            ],
            majit_ir::EffectInfo::new(
                majit_ir::ExtraEffect::CannotRaise,
                majit_ir::OopSpecIndex::None,
            ),
        );
        // Every link of the chain names the SAME mapping.
        ctx.trace_ctx
            .set_opref_concrete(dict_op, concrete_locals_value);
    }
    locals_expansion_cut_if_too_long(ctx, op.pc)?;
    let result_op = match tail_fn {
        Some(tail) => {
            let op_ref = ctx.trace_ctx.call_ref_typed_with_effect(
                tail as *const (),
                &[dict_op],
                &[majit_ir::Type::Ref],
                majit_ir::EffectInfo::new(
                    majit_ir::ExtraEffect::CannotRaise,
                    majit_ir::OopSpecIndex::None,
                ),
            );
            ctx.trace_ctx.set_opref_concrete(
                op_ref,
                majit_ir::Value::Ref(majit_ir::GcRef(concrete_result as usize)),
            );
            // The tail reports a failure as PY_NULL instead of publishing it,
            // so the guarded side exit re-runs the residual and raises from
            // the eval loop.
            walker_emit_fold_guard_with_snapshot(ctx, op.pc, OpCode::GuardNonnull, &[op_ref])?;
            op_ref
        }
        None => dict_op,
    };
    write_residual_call_result_to_dst(ctx, op.pc, dst, 'r', result_op)?;
    Ok(Some(()))
}

/// `sys._getframe()` / `sys._getframe(0)` at the top walk level: publish the
/// portal virtualizable itself instead of residualizing `vm.py getframe`.
///
/// `getframe` is `@jit.look_inside_iff(lambda space, depth:
/// jit.isconstant(depth))` (`pypy/module/sys/vm.py:41`), so a constant depth is
/// traced THROUGH: at the portal, `ec.gettopframe_nohidden()` is a vref read
/// that `pyjitpl.py _do_jit_force_virtual` answers with
/// `virtualizable_boxes[-1]` under a `ptr_eq` + `implement_guard_value`; in an
/// inline MIFrame its live `JitVirtualRef` is known non-standard and follows
/// the residual `jit_force_virtual` path, which `virtualize.py` removes after
/// `vrefs_after_residual_call` publishes the forced pair.  The `depth == 0`
/// test folds away, and `mark_as_escaped` is one `setfield_gc`.
/// No call survives optimization and no virtualizable is forced — pypy3
/// reports `forcings: 0` and `abort: vable escape: 0` on the fixtures where
/// pyre loses the loop.
///
/// Pyre residualizes the same walk as one opaque `bh_call_fn(_getframe,
/// PY_NULL, depth)` `CallMayForce`, and [`pyre_interpreter::module::sys::vm::getframe`]'s
/// `force_frame` on the frame it returns — the stand-in for the injection
/// `rvirtualizable.py hook_access_field` performs and pyre's rtyper
/// cannot build — clears `TOKEN_TRACING_RESCALL` inside that call whenever the
/// returned frame is the traced one, which `tracing_after_residual_call` reads
/// as an escape (`VableEscapedDuringResidualCall`).  At depth 0 the returned
/// frame is always the portal, so the residual always escapes.  Removing it
/// removes the force with it, and nothing has to replace it: `last_instr` is
/// published onto the portal frame at every may-force boundary
/// (`LiveLastInstrGuard`).  A generic reader of either getter still retains
/// that residual boundary.  Upstream does not: `pyframe.py fget_f_lasti` and
/// `fget_f_lineno` are loop-free and carry no hint, so `policy.py
/// look_inside_graph` admits them, `jtransform.py
/// rewrite_op_jit_force_virtualizable` deletes the injected force, and
/// `pyjitpl.py opimpl_getfield_vable_i` answers `last_instr` out of
/// `virtualizable_boxes`.  Measured over a 200k-iteration read against a
/// same-shape loop that does not read the frame: pypy3 answers `f_lasti`
/// faster than that control loop -- the trace constant -- and pyre was 53x it,
/// while `f_lineno` keeps one non-forcing residual on both and pyre is 2.1x.
/// The two halves are closed by two different emissions, because the shapes
/// they are closing to differ: [`try_walker_specialize_frame_lasti`] emits the
/// constant, while [`try_walker_specialize_frame_lineno`] emits the one
/// non-forcing residual upstream also keeps for the line-table decode.  Either
/// is an optimization over a correct path, not a fix, and both owe two
/// coordinates the boundary hides.  `last_instr` is an instruction-unit index
/// here and the
/// app-level getter reports it doubled (`typedef.rs`, matching `location.py
/// offset2lineno`'s `stopat // 2` on the byte offset upstream stores), so an
/// emission at the app level owes the factor.  And the field has two writers
/// on two conventions: `flush_walk_end_state_to_frame` writes
/// `resume_py_pc - 1` while `LiveLastInstrGuard::enter_frame` writes the
/// executing pc unshifted, and a getter owes the executing one.
/// An exact optimized-frame `f_locals` read is specialized
/// below instead: `pyframe.py fast2locals` is `@jit.unroll_safe`, so upstream
/// traces through it and reads the virtualizable boxes rather than forcing.
/// The force it drops was also the only writer of the frame's locals region,
/// which pyre's `FrameLocalsProxy` reads; the fold therefore performs that
/// write-back itself (`walker_write_back_standard_frame_locals`).
///
/// Emitted shape, following `getframe`'s body line by line:
/// `guard_value(callable)`; `guard_class` + exact-class + `getfield_gc_i` on
/// the depth box, whose resulting RAW int must be a trace constant — that
/// unboxed value is what `jit.isconstant(depth)` tests upstream, where
/// `@unwrap_spec(depth=int)` has already run OUTSIDE the looked-inside graph
/// (the wrapped `W_IntObject` the residual receives is built in-trace by
/// `NewWithVtable` + `SetfieldGc` and is never constant, so testing the box
/// declines 100% of the time); `getfield_gc_r(frame, execution_context)` +
/// `getfield_gc_r(ec, topframeref)` + `ptr_eq` + `guard_true`, the port of
/// `_do_jit_force_virtual`'s identity check; and one non-forcing void `Call`
/// for `mark_as_escaped`.  At depth 0 the result IS
/// `standard_virtualizable_box()`, exactly as `_do_jit_force_virtual` returns
/// `standard_box`.
///
/// For constant depth >= 1, the same top-frame proof seeds the walk, then each
/// hop emits the `ExecutionContext.getnextframe_nohidden(frame)` body shape:
/// raw `frame.f_backref` read, the residual `OS_JIT_FORCE_VIRTUAL`
/// `CallMayForceR(jit_force_vref, raw)` bracketed by FORCE_TOKEN/SETFIELD_GC and
/// `GuardNotForced`, `GuardNonnull` for the "call stack is not deep enough"
/// arm, then `frame.hide()` as `frame.pycode.hidden_applevel` with a
/// `GuardFalse`.  The residual force stays in the trace: `pyjitpl.py:2153-2172
/// _do_jit_force_virtual` returns `None` for a known non-standard
/// virtualizable, and that `None` result is precisely the caller's signal to
/// emit the residual `jit_force_virtual` call.  `optimize_jit_force_virtual`
/// only elides it later for a trace Virtual, matching upstream and giving the
/// cheap per-hop cost instead of the full `_getframe` residual.
///
/// The guard reads `topframeref` raw, without the vref force or the
/// hidden-frame walk `gettopframe_nohidden` performs, so the gate below
/// requires the record-time chain to need neither: `topframeref` must BE the
/// portal pointer (not a `JitVirtualRef` naming it) and the nohidden walk must
/// land on the same frame.  Any other chain declines, and at runtime a
/// `topframeref` that stops matching side-exits.
///
/// Returns `None` (fall through to the generic residual) for every other
/// shape: a rebound `sys._getframe`, a bound receiver, a negative / non-int /
/// inexact / non-constant depth, a walk with no frame identity, armed audit
/// hooks, a top-level `topframeref` mismatch with the portal frame, a hop whose
/// forced `f_backref` is null, or a hop whose result is hidden.
/// Declines after emission rewind to the pre-specialization trace position and
/// reset the heap cache before falling through.
fn next_op_is_f_locals_for_getframe_result<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    ctx: &WalkContext<'_, '_, Sym>,
    getframe_dst: usize,
) -> bool {
    let Some(mut next) = crate::jitcode_runtime::decode_op_at(code, op.next_pc) else {
        return false;
    };
    while next.opname == "live"
        || next.opname.starts_with("setarrayitem_vable")
        || next.opname.starts_with("setfield_vable")
    {
        let Some(after_bookkeeping) = crate::jitcode_runtime::decode_op_at(code, next.next_pc)
        else {
            return false;
        };
        next = after_bookkeeping;
    }
    let helper_kind = residual_call::residual_call_descr_index_in_body(code, &next)
        .and_then(|index| ctx.descr_refs.at(index))
        .and_then(|descr| {
            descr
                .as_call_descr()
                .map(|call| call.get_extra_info().runtime_helper)
        });
    if next.key != "residual_call_ir_r/iIRd>r"
        || helper_kind != Some(majit_ir::RuntimeHelperKind::LoadAttr)
    {
        return false;
    }

    // `iIRd>r`: funcbox, Int var-list, Ref var-list, descr, result.  The
    // LoadAttr helper's lists are `[name_idx]` and `[obj, code]`.
    let Some(&i_len_byte) = code.get(next.pc + 2) else {
        return false;
    };
    let i_len = i_len_byte as usize;
    if i_len != 1 {
        return false;
    }
    let Some(&name_reg) = code.get(next.pc + 3) else {
        return false;
    };
    let r_len_pc = next.pc + 3 + i_len;
    if code.get(r_len_pc) != Some(&2) {
        return false;
    }
    let (Some(&obj_reg), Some(&code_reg)) = (code.get(r_len_pc + 1), code.get(r_len_pc + 2)) else {
        return false;
    };
    if obj_reg as usize != getframe_dst {
        return false;
    }
    let (Some(&name_op), Some(&code_op)) = (
        ctx.registers_i.get(name_reg as usize),
        ctx.registers_r.get(code_reg as usize),
    ) else {
        return false;
    };
    let (Some(majit_ir::Value::Int(name_idx)), Some(majit_ir::Value::Ref(w_code))) = (
        ctx.trace_ctx.box_value(name_op),
        ctx.trace_ctx.box_value(code_op),
    ) else {
        return false;
    };
    if name_idx < 0 || w_code.as_usize() == 0 {
        return false;
    }
    walker_load_name_from_code(w_code.as_usize(), name_idx as usize).as_deref() == Some("f_locals")
}

pub(crate) fn try_walker_specialize_sys_getframe<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    code: &[u8],
    op: &DecodedOp,
    r_args: &[OpRef],
    dst: usize,
) -> Result<Option<()>, DispatchError> {
    // `sys._getframe()` (2) or `sys._getframe(depth)` (3).
    if !(2..=3).contains(&r_args.len()) {
        return Ok(None);
    }
    let arg_concretes = read_ref_var_list_concrete(code, op, 1, ctx);
    let (ConcreteValue::Ref(concrete_callable), ConcreteValue::Ref(null_or_self)) =
        (arg_concretes[0], arg_concretes[1])
    else {
        return Ok(None);
    };
    // A non-null `null_or_self` is a bound receiver `bh_call_fn_impl` prepends
    // as arg0, not a plain `sys._getframe(...)` call.
    if concrete_callable.is_null()
        || !null_or_self.is_null()
        || !pyre_interpreter::module::sys::vm::is_builtin_getframe_function(concrete_callable)
    {
        return Ok(None);
    }
    // The depth has to be an exact plain non-negative int before anything is
    // emitted; the guards below pin both facts for the compiled loop.
    let exact_int_class = pyre_object::pyobject::get_instantiate(&pyre_object::pyobject::INT_TYPE);
    let (depth_arg, depth_value) = if r_args.len() == 3 {
        let ConcreteValue::Ref(depth_obj) = arg_concretes[2] else {
            return Ok(None);
        };
        if depth_obj.is_null()
            || unsafe {
                !std::ptr::eq((*depth_obj).ob_type, &pyre_object::pyobject::INT_TYPE)
                    || !std::ptr::eq((*depth_obj).w_class, exact_int_class)
            }
        {
            return Ok(None);
        }
        let depth = unsafe { pyre_object::w_int_get_value(depth_obj) };
        if depth < 0 {
            return Ok(None);
        }
        (Some(r_args[2]), depth)
    } else {
        (None, 0)
    };
    // `vm.py audit(space, "sys._getframe", [f])`.  With no hook installed
    // `audit` takes its `holder.hooks_w is None` early-out (`vm.py`) and the
    // event costs nothing; the emission below pins that read so a later
    // `addaudithook` revokes this loop instead of silently missing the event.
    // With a hook already installed the event reaches `trigger_audit_events`,
    // which is `@objectmodel.dont_inline` — a residual call this arm has no
    // channel for — so it declines and the generic residual `getframe` emits.
    let audit_holder = pyre_interpreter::module::sys::vm::audit_holder_ptr();
    if audit_holder.is_null() || pyre_interpreter::module::sys::vm::audit_hooks_armed() {
        return Ok(None);
    }
    // Every MIFrame owns one red frame. At the root that is the standard
    // virtualizable; inside an inline sub-walk it is the callee frame seeded in
    // `dispatch_inline_call_dr_kind` and carried by `CalleeLocalsShadow`.
    // Starting the constant-depth walk from that per-level frame is the direct
    // counterpart of `ec.gettopframe_nohidden()` returning the live MIFrame's
    // virtual frame upstream.
    let inline_ptr = current_inline_concrete_frame();
    let inline_level = ctx.fbw_mode.inline_subwalk || inline_ptr != 0;
    let (Some(standard_vable_op), Some(standard_vable_ptr)) = (
        ctx.trace_ctx.standard_virtualizable_box(),
        ctx.trace_ctx.standard_virtualizable_ptr(),
    ) else {
        return Ok(None);
    };
    let (vable_op, vable_ptr) = if inline_level {
        let Some(shadow) = ctx.callee_shadow.as_ref() else {
            return Ok(None);
        };
        if inline_ptr == 0 || shadow.concrete_frame != inline_ptr || shadow.frame_box == OpRef::NONE
        {
            return Ok(None);
        }
        (shadow.frame_box, inline_ptr)
    } else {
        (standard_vable_op, standard_vable_ptr)
    };
    let ec =
        pyre_interpreter::call::getexecutioncontext() as *mut pyre_interpreter::PyExecutionContext;
    if ec.is_null() {
        return Ok(None);
    }
    // At the portal, prove that the raw execution-context chain still names
    // the standard frame and emit the equivalent runtime guard below. An
    // inline level already has the stronger per-MIFrame identity witness:
    // `frame_box` and `concrete_frame` were seeded together when that level was
    // pushed, and the compiled trace carries the same box directly.
    let frame = if inline_level {
        inline_ptr as *mut pyre_interpreter::PyFrame
    } else {
        if unsafe { (*ec).topframeref } as usize != vable_ptr {
            return Ok(None);
        }
        let frame = unsafe { (*ec).gettopframe_nohidden() };
        if frame.is_null() || frame as usize != vable_ptr {
            return Ok(None);
        }
        frame
    };
    // Validate the entire concrete chain before emitting or forcing anything.
    // A tracing-time `JitVirtualRef` is admissible only when it is still one of
    // `MetaInterp.virtualref_boxes`: that is the pair
    // `vrefs_after_residual_call` will publish if this walk forces it.  Reading
    // `forced` here does not force or change the token; vrefs created during
    // tracing already carry the real recording-time frame there
    // (`virtualref.py virtual_ref_during_tracing`).  This all-or-nothing gate
    // keeps a later decline from shortening the concrete frame chain before
    // the generic residual gets a chance to run.
    //
    // A hidden hop declines outright.  `executioncontext.py
    // getnextframe_nohidden` skips a hidden frame WITHOUT consuming a depth
    // level, so one raw `f_backref` per level only reproduces `getframe`'s walk
    // on a chain that has none; the emitted traversal pins that with its
    // per-hop `guard_false(hidden_applevel)`.
    let final_concrete_frame = {
        let mut scan = frame;
        for _ in 0..depth_value {
            let raw = unsafe { (*scan).f_backref };
            if raw.is_null() {
                return Ok(None);
            }
            if unsafe { majit_metainterp::virtualref::ptr_is_virtual_ref(raw as *const u8) } {
                let referent = unsafe {
                    majit_metainterp::virtualref::vref_forced(raw as *const u8)
                        as *mut pyre_interpreter::PyFrame
                };
                if referent.is_null()
                    || (ctx
                        .trace_ctx
                        .live_virtualref_pair_for_ptr(raw as usize)
                        .is_none()
                        && ctx
                            .trace_ctx
                            .virtualref_virtual_for_object_ptr(referent as usize)
                            .is_none())
                {
                    if fbw_debug_abort_enabled() {
                        let pairs = ctx.trace_ctx.snapshot_virtualref_boxes();
                        eprintln!(
                            "[getframe-decline] depth={depth_value} vref {:#x} referent={:#x} has no pair; tracked={pairs:?}",
                            raw as usize, referent as usize,
                        );
                    }
                    return Ok(None);
                }
                scan = referent;
            } else {
                scan = raw;
            }
            if unsafe { (*scan).hide() } {
                return Ok(None);
            }
        }
        scan
    };

    // Until every app-level frame getter is lowered through its own red frame,
    // admitting an arbitrary positive-depth result would expose it to a
    // generic residual whose single live-coordinate slot cannot describe a
    // nested caller chain.  The completed slice is the outer standard frame
    // immediately consumed by `f_locals`: that getter is specialized below and
    // its locals write-back names the same frame, so it crosses no such
    // residual boundary.  Preflight its whole static shape before emitting any
    // part of `_getframe`.
    if inline_level && depth_value > 0 {
        let standard_frame = final_concrete_frame as usize == standard_vable_ptr
            && unsafe { (*final_concrete_frame).ob_header.ob_type }
                == &pyre_interpreter::pyframe::FRAME_TYPE
            && unsafe {
                (*final_concrete_frame)
                    .code()
                    .flags
                    .contains(pyre_interpreter::CodeFlags::OPTIMIZED)
            };
        let w_type =
            pyre_interpreter::typedef::gettypeobject(&pyre_interpreter::pyframe::FRAME_TYPE);
        if !standard_frame
            || unsafe { (*final_concrete_frame).ob_header.w_class } != w_type
            || unsafe { pyre_object::typeobject::w_type_get_version_tag(w_type) } == 0
            || !next_op_is_f_locals_for_getframe_result(code, op, ctx, dst)
        {
            return Ok(None);
        }
    }

    // --- emit the specialized IR (walker-native) ---
    let pre_emit_pos = ctx.trace_ctx.get_trace_position();

    // `sys` is an ordinary mutable module, so nothing else keeps the name bound
    // to this builtin across iterations.
    let callable_op = r_args[0];
    if !callable_op.is_constant() {
        let expected = ctx.trace_ctx.const_ref(concrete_callable as i64);
        ctx.trace_ctx
            .record_guard(OpCode::GuardValue, &[callable_op, expected], 0);
        walker_capture_snapshot_for_last_guard(ctx, op.pc)?;
        ctx.trace_ctx
            .heap_cache_mut()
            .replace_box(callable_op, expected);
    }
    // `@unwrap_spec(depth=int)` and then `jit.isconstant(depth)`: unbox first,
    // and require the UNBOXED value to be the trace constant.  The unbox is
    // only sound behind the class guards, so the constness decline rewinds
    // rather than being hoisted above them.
    if let Some(depth_op) = depth_arg {
        let int_type_addr = &pyre_object::pyobject::INT_TYPE as *const _ as i64;
        walker_guard_class(ctx, op.pc, depth_op, int_type_addr)?;
        walker_guard_exact_w_class(ctx, op.pc, depth_op, exact_int_class)?;
        let raw = crate::state::opimpl_getfield_gc_i(
            ctx.trace_ctx,
            depth_op,
            crate::descr::int_intval_descr(),
        );
        ctx.trace_ctx
            .set_opref_concrete(raw, majit_ir::Value::Int(depth_value));
        if !raw.is_constant() {
            ctx.trace_ctx.cut_trace_with_snapshots(pre_emit_pos);
            ctx.trace_ctx.heap_cache_mut().reset();
            return Ok(None);
        }
    }
    if !inline_level {
        // `ec = space.getexecutioncontext()` is the portal's second red,
        // carried independently of the virtualizable frame.
        let Some(ec_op) = walker_ensure_execution_context(ctx) else {
            ctx.trace_ctx.cut_trace_with_snapshots(pre_emit_pos);
            ctx.trace_ctx.heap_cache_mut().reset();
            return Ok(None);
        };
        // `f = ec.gettopframe_nohidden()` followed by
        // `_do_jit_force_virtual`'s standard-box identity guard.
        let topframeref_op = ctx.trace_ctx.record_op_with_descr(
            OpCode::GetfieldGcR,
            &[ec_op],
            crate::descr::ec_topframeref_descr(),
        );
        ctx.trace_ctx.set_opref_concrete(
            topframeref_op,
            majit_ir::Value::Ref(majit_ir::GcRef(vable_ptr)),
        );
        let is_standard = ctx
            .trace_ctx
            .record_op(OpCode::PtrEq, &[topframeref_op, vable_op]);
        ctx.trace_ctx
            .set_opref_concrete(is_standard, majit_ir::Value::Int(1));
        walker_emit_fold_guard_with_snapshot(ctx, op.pc, OpCode::GuardTrue, &[is_standard])?;
    }

    let mut cur_op = vable_op;
    let mut cur_ptr = frame;
    for _ in 0..depth_value {
        let raw_ptr = unsafe { (*cur_ptr).f_backref };
        let raw_op = crate::state::opimpl_getfield_gc_r(
            ctx.trace_ctx,
            cur_op,
            crate::descr::pyframe_f_backref_descr(),
        );
        ctx.trace_ctx.set_opref_concrete(
            raw_op,
            majit_ir::Value::Ref(majit_ir::GcRef(raw_ptr as usize)),
        );

        let raw_is_vref =
            unsafe { majit_metainterp::virtualref::ptr_is_virtual_ref(raw_ptr as *const u8) };
        let (next_op, next_ptr) = if raw_is_vref {
            // `_do_jit_force_virtual` sees the vref box as a known
            // non-standard virtualizable and returns None, so
            // `do_residual_call` executes the may-force call.  Run the exact
            // vref bracket around that concrete force: the post half records
            // `VIRTUAL_REF_FINISH(vref, virtual)` before the CALL and replaces
            // the tracked vref with CONST_NULL (`pyjitpl.py`).  The optimizer
            // can then forward JIT_FORCE_VIRTUAL to the paired frame instead
            // of materialising a vref whose `forced` field is null.
            let live_pair = ctx.trace_ctx.live_virtualref_pair_for_ptr(raw_ptr as usize);
            let referent = unsafe {
                majit_metainterp::virtualref::vref_forced(raw_ptr as *const u8)
                    as *mut pyre_interpreter::PyFrame
            };
            let virtual_op = live_pair.map(|pair| pair.0).unwrap_or_else(|| {
                ctx.trace_ctx
                    .virtualref_virtual_for_object_ptr(referent as usize)
                    .expect("the pre-emission frame-chain census accepted this stopped vref")
            });
            // A field read can produce an alias box even though its concrete
            // value is the tracked vref.  Upstream's heapcache normally hands
            // `_do_jit_force_virtual` the tracked box directly.  Preserve
            // that identity for the optimizer after proving the alias at
            // runtime; `VIRTUAL_REF_FINISH` and JIT_FORCE_VIRTUAL must name
            // the same vref box for `optimize_jit_force_virtual` to forward
            // the result to `virtual_op`.
            let force_arg = if let Some((_, vref_op)) = live_pair {
                if raw_op != vref_op {
                    let is_tracked_vref =
                        ctx.trace_ctx.record_op(OpCode::PtrEq, &[raw_op, vref_op]);
                    ctx.trace_ctx
                        .set_opref_concrete(is_tracked_vref, majit_ir::Value::Int(1));
                    walker_emit_fold_guard_with_snapshot(
                        ctx,
                        op.pc,
                        OpCode::GuardTrue,
                        &[is_tracked_vref],
                    )?;
                }
                vref_op
            } else {
                raw_op
            };
            maybe_walker_vable_and_vrefs_before_residual_call(ctx, op.pc);
            ctx.trace_ctx.vrefs_before_residual_call();
            let next_ptr = pyre_interpreter::executioncontext::force_vref(raw_ptr);
            ctx.trace_ctx.vrefs_after_residual_call();
            let force_fn = crate::helpers::jit_force_vref as *const ();
            let forced_op = ctx.trace_ctx.call_typed_with_effect(
                OpCode::CallMayForceR,
                force_fn,
                &[force_arg],
                &[majit_ir::Type::Ref],
                majit_ir::Type::Ref,
                majit_ir::EffectInfo::new(
                    majit_ir::ExtraEffect::ForcesVirtualOrVirtualizable,
                    majit_ir::OopSpecIndex::JitForceVirtual,
                ),
            );
            ctx.trace_ctx.set_opref_concrete(
                forced_op,
                majit_ir::Value::Ref(majit_ir::GcRef(next_ptr as usize)),
            );
            ctx.trace_ctx.record_guard(OpCode::GuardNotForced, &[], 0);
            walker_capture_snapshot_for_last_guard(ctx, op.pc)?;
            // `VirtualRefFinish(vref, virtual)` immediately before the call is
            // the optimizer proof that `forced_op == virtual_op`.  Preserve
            // the orthodox force in IR while letting the source walker follow
            // the same forwarded box immediately.
            (virtual_op, next_ptr)
        } else if raw_ptr as usize == standard_vable_ptr {
            // `_do_jit_force_virtual`: the standard virtualizable identity
            // short-circuits before residual-call preparation.  Heapcache
            // normally gives us the same OpRef; keep the runtime proof for an
            // alias box, matching its PTR_EQ + implement_guard_value arm.
            if raw_op != standard_vable_op {
                let is_standard = ctx
                    .trace_ctx
                    .record_op(OpCode::PtrEq, &[raw_op, standard_vable_op]);
                ctx.trace_ctx
                    .set_opref_concrete(is_standard, majit_ir::Value::Int(1));
                walker_emit_fold_guard_with_snapshot(
                    ctx,
                    op.pc,
                    OpCode::GuardTrue,
                    &[is_standard],
                )?;
            }
            (standard_vable_op, raw_ptr)
        } else {
            if fbw_debug_abort_enabled() {
                eprintln!(
                    "[getframe-decline] depth={depth_value} non-vref hop {:#x} != standard {standard_vable_ptr:#x}",
                    raw_ptr as usize
                );
            }
            ctx.trace_ctx.cut_trace_with_snapshots(pre_emit_pos);
            ctx.trace_ctx.heap_cache_mut().reset();
            return Ok(None);
        };
        if next_ptr.is_null() || unsafe { (*next_ptr).hide() } {
            unreachable!("the pre-emission frame-chain census accepted this hop")
        }
        walker_emit_fold_guard_with_snapshot(ctx, op.pc, OpCode::GuardNonnull, &[next_op])?;

        let code_op = crate::state::opimpl_getfield_gc_r(
            ctx.trace_ctx,
            next_op,
            crate::descr::pyframe_code_descr(),
        );
        let code_ptr = unsafe { (*next_ptr).pycode };
        ctx.trace_ctx.set_opref_concrete(
            code_op,
            majit_ir::Value::Ref(majit_ir::GcRef(code_ptr as usize)),
        );
        // `optimizer.py` reaches for `descr.get_parent_descr()` only
        // when arg0 carries no pointer info yet; a preceding `GUARD_CLASS`
        // gives it `info.InstancePtrInfo()` and that lookup does not run here.
        // The PyCode field group also carries its parent for paths that reach
        // the read without pointer info.  Keep the guard in this path because
        // it validates the concrete pointer before the hidden flag is read —
        // the same order the code-field arm of
        // [`try_walker_specialize_traceback_walk`] uses.
        let code_type_addr = &pyre_interpreter::pycode::CODE_TYPE as *const _ as i64;
        if code_ptr.is_null()
            || unsafe {
                !std::ptr::eq(
                    (*code_ptr.cast::<pyre_object::PyObject>()).ob_type,
                    &pyre_interpreter::pycode::CODE_TYPE,
                )
            }
        {
            ctx.trace_ctx.cut_trace_with_snapshots(pre_emit_pos);
            ctx.trace_ctx.heap_cache_mut().reset();
            return Ok(None);
        }
        walker_guard_class(ctx, op.pc, code_op, code_type_addr)?;
        let hidden_op = crate::state::opimpl_getfield_gc_i(
            ctx.trace_ctx,
            code_op,
            crate::descr::pycode_hidden_applevel_descr(),
        );
        ctx.trace_ctx
            .set_opref_concrete(hidden_op, majit_ir::Value::Int(0));
        walker_emit_fold_guard_with_snapshot(ctx, op.pc, OpCode::GuardFalse, &[hidden_op])?;

        cur_op = next_op;
        cur_ptr = next_ptr;
    }

    // A depth-zero inline result exposes this callee frame. Publish its current
    // coordinate and any locals that were still held in the strict-fold shadow
    // before the frame becomes observable. This is the same per-frame state
    // the ordinary residual force path flushes, without forcing the outer
    // portal virtualizable or aborting its trace.
    if inline_level && depth_value == 0 {
        residual_call::record_and_publish_inline_callee_last_instr(ctx, op.pc);
        disarm_folded_inline_callee_after_escape(ctx, op.pc)?;
    }

    // A positive-depth inline walk can land on the standard portal frame.
    // Its symbolic `last_instr` is already current in `virtualizable_boxes`
    // (the caller CALL boundary was mirrored there before descending), and
    // residual-call preparation will emit that shadow's store before any
    // runtime frame reader.  Keep the recording-time concrete frame in step
    // with the same value so a getter executed while recording observes the
    // coordinate the compiled trace will publish, rather than baking the
    // frame's stale pre-inline heap value into the trace.
    if cur_op == standard_vable_op
        && cur_ptr as usize == standard_vable_ptr
        && let Some((_, majit_ir::Value::Int(last_instr))) = ctx
            .trace_ctx
            .virtualizable_entry_at(crate::virtualizable_spec::LAST_INSTR_VABLE_FIELD_INDEX)
    {
        // Journaled like the per-opcode publication: this store lands whether
        // or not the walk commits, and a walk that does not commit replays the
        // frame from its pre-walk coordinate.
        crate::jitcode_dispatch::fbw_note_last_instr_undo(cur_ptr as usize);
        unsafe { (*cur_ptr).last_instr = last_instr as isize };
    }

    // `f.mark_as_escaped()` — vm.py.  `escaped` is not one of the six fields
    // `interp_jit.py:25-30` declares, so the store cannot force; it is
    // load-bearing at `executioncontext.py leave`, which forces the
    // leaving frame's own vref only for a frame that escaped.  Upstream traces
    // it as the ordinary `setfield_gc` on the flag, so it is emitted as the
    // read/or/store the `tb_frame` fold above already uses — an opaque call
    // would hide the update from the optimizer and its heap cache.
    let flags_descr = crate::descr::pyframe_flags_descr();
    let live_flags = crate::state::opimpl_getfield_gc_i(ctx.trace_ctx, cur_op, flags_descr.clone());
    let escaped_bit = ctx
        .trace_ctx
        .const_int(i64::from(pyre_interpreter::PyFrame::FLAG_ESCAPED));
    let new_flags = ctx
        .trace_ctx
        .record_op(OpCode::IntOr, &[live_flags, escaped_bit]);
    ctx.trace_ctx.record_op_with_descr(
        OpCode::SetfieldGc,
        &[cur_op, new_flags],
        flags_descr.clone(),
    );
    ctx.trace_ctx
        .heapcache_setfield_cached(cur_op, flags_descr.index(), new_flags);
    // The walk IS the interpreter running, so the recorded store has to take
    // effect here too — the residual would have applied it before returning.
    unsafe { (*cur_ptr).mark_as_escaped() };

    // `audit(space, "sys._getframe", [f])` — vm.py.  The gate above resolved
    // it to the no-hook early-out, so all that is emitted is the marker for the
    // read that reached that conclusion.
    walker_pin_audit_hooks(ctx, op.pc, audit_holder)?;

    // `return f` — at depth 0 `cur_op` is still the standard virtualizable
    // `_do_jit_force_virtual` hands back as `standard_box`; each hop above
    // advanced it to the frame the walk settled on.
    write_residual_call_result_to_dst(ctx, op.pc, dst, 'r', cur_op)?;
    Ok(Some(()))
}

// ── the generated `math` float folds ──────────────────────────────────
//
// The five entry points below are one row each: they name the callable they
// answer for and hand the driver the two facts that differ between rows —
// which branches of the body the trace has to pin, and what the compiled loop
// computes.  Everything else — operand classification, the authentic
// pre-execution, the callable pin, the coercions, the cross-check and the
// boxing — is written once, here.  Upstream reaches the same surface the same
// way: `intobject.py`'s `_make_descr_binop` emits one body per row from a
// table instead of repeating it, and the codewriter looks its builtin
// lowerings up by name in `support.py`'s `_ll_<arity>_<oopspec>` table.

/// What a row's compiled body computes from the coerced operand(s).
#[derive(Clone, Copy)]
enum MathFloatEmit {
    /// A pure `CALL_F` into the row's raw helper — the shape a translated
    /// `ll_math_*` call carries once its exceptional branches are pinned.
    Call1(extern "C" fn(f64) -> f64),
    /// The two-operand form of [`MathFloatEmit::Call1`].
    Call2(extern "C" fn(f64, f64) -> f64),
    /// A bare IR op, for a row whose whole body is one machine instruction.
    /// The `f64` function beside it is that same computation, for the
    /// cross-check below.
    Unary(OpCode, fn(f64) -> f64),
}

impl MathFloatEmit {
    /// Positional argument count this emission answers for.
    fn arity(self) -> usize {
        match self {
            Self::Call1(_) | Self::Unary(..) => 1,
            Self::Call2(_) => 2,
        }
    }

    /// The value the compiled body produces for these operands.
    fn evaluate(self, values: &[f64]) -> f64 {
        match self {
            Self::Call1(raw) => raw(values[0]),
            Self::Call2(raw) => raw(values[0], values[1]),
            Self::Unary(_, compute) => compute(values[0]),
        }
    }
}

/// The branches of a row's body the trace has to pin before its
/// [`MathFloatEmit`] stands for the whole builtin.
#[derive(Clone, Copy, PartialEq)]
enum MathFloatDomain {
    /// `ll_math_sqrt`: not negative, and finite.
    NonNegativeFinite,
    /// `ll_math_log`: strictly positive, and finite.
    PositiveFinite,
    /// `ll_math_{cos,sin}`: finite.
    Finite,
    /// The body raises for no input and cannot leave the float domain, so the
    /// operand needs no pinning at all.
    Total,
    /// The helper reports every raising direction as a non-finite result, so
    /// the operand is unconstrained and the *result* is guarded instead: the
    /// guard deoptimizes into the builtin, which re-executes and raises or
    /// returns the non-finite value itself.  A row spelled this way carries no
    /// per-function domain knowledge, so every `MATH_FLOAT1_FOLDS` entry
    /// shares this one arm.
    ResultFinite,
}

impl MathFloatDomain {
    /// Whether these concrete operands are inside the arm the row lowers.  A
    /// `false` keeps the residual, which re-executes the builtin.
    fn admits(self, values: &[f64]) -> bool {
        let x = values[0];
        match self {
            Self::NonNegativeFinite => x.is_finite() && x >= 0.0,
            Self::PositiveFinite => x.is_finite() && x > 0.0,
            Self::Finite => x.is_finite(),
            Self::Total | Self::ResultFinite => true,
        }
    }

    /// Emit the guards that hold the compiled loop inside that arm.
    fn emit_operand_guards<Sym: WalkSym>(
        self,
        ctx: &mut WalkContext<'_, '_, Sym>,
        pc: usize,
        x: OpRef,
    ) -> Result<(), DispatchError> {
        if matches!(self, Self::Total | Self::ResultFinite) {
            return Ok(());
        }
        let zero = ctx.trace_ctx.const_float(0.0f64.to_bits() as i64);
        match self {
            // `x >= 0`, the `ValueError` direction of `ll_math_sqrt`.
            Self::NonNegativeFinite => {
                walker_float_cmp_guard(ctx, pc, OpCode::FloatLt, &[x, zero], false)?
            }
            // `x > 0`, the domain of `ll_math_log`.
            Self::PositiveFinite => {
                walker_float_cmp_guard(ctx, pc, OpCode::FloatLt, &[zero, x], true)?
            }
            Self::Finite => {}
            Self::Total | Self::ResultFinite => unreachable!("returned above"),
        }
        // `isfinite(x)`: `x - x == 0` holds exactly for the finite values,
        // signed zero included, so NaN and ±inf take the residual.
        let diff = ctx.trace_ctx.record_op(OpCode::FloatSub, &[x, x]);
        ctx.trace_ctx
            .set_opref_concrete(diff, majit_ir::Value::Float(0.0));
        walker_float_cmp_guard(ctx, pc, OpCode::FloatEq, &[diff, zero], true)
    }
}

/// One pure elidable `CALL_F` into a raw float helper, stamped with the value
/// it produced concretely.  `EF_ELIDABLE_CANNOT_RAISE` is what lets the
/// optimizer hoist it out of a loop and keep the result box virtual.
fn walker_call_pure_float<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    raw_fn: *const (),
    args: &[OpRef],
    values: &[f64],
    result_value: f64,
) -> OpRef {
    let types = [majit_ir::Type::Float; 2];
    let concrete = [
        majit_ir::Value::Int(raw_fn as i64),
        majit_ir::Value::Float(values[0]),
        majit_ir::Value::Float(values[values.len() - 1]),
    ];
    let raw = ctx.trace_ctx.call_typed_with_effect_pure(
        OpCode::CallF,
        raw_fn,
        args,
        &types[..args.len()],
        majit_ir::Type::Float,
        majit_metainterp::ELIDABLE_CANNOT_RAISE_NO_HEAP_EFFECT_INFO,
        &concrete[..args.len() + 1],
        majit_ir::Value::Float(result_value),
    );
    ctx.trace_ctx
        .set_opref_concrete(raw, majit_ir::Value::Float(result_value));
    raw
}

/// The body every `math` float fold shares.  `row` recognizes the callable
/// this entry point answers for and returns the row's two variable facts.
///
/// The residual this replaces costs an argument tuple, a builtin dispatch, a
/// `try_get_double` per operand and a `W_FloatObject` allocation, once per
/// loop iteration; what goes in its place is the unboxed operand, the row's
/// emission and an inline `wrapfloat` the optimizer can keep virtual.
fn walker_specialize_math_float<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    code: &[u8],
    op: &DecodedOp,
    r_args: &[OpRef],
    dst: usize,
    arity: usize,
    row: impl FnOnce(pyre_object::PyObjectRef) -> Option<(MathFloatDomain, MathFloatEmit)>,
) -> Result<Option<()>, DispatchError> {
    let Some((concrete_callable, mut operands)) =
        plain_builtin_call_concretes(ctx, code, op, r_args, arity)
    else {
        return Ok(None);
    };
    let Some((domain, emit)) = row(concrete_callable) else {
        return Ok(None);
    };
    debug_assert_eq!(
        arity,
        emit.arity(),
        "a row's emission disagrees with the call shape its entry point reads"
    );
    // Exact `int`/`bool`/`float` operands only: a numeric subclass keeps the
    // builtin `ob_type` but carries its own `w_class`, and may override
    // `__float__`.
    let mut coerced = [(false, 0.0f64); 2];
    for (slot, &obj) in coerced.iter_mut().zip(&operands[..arity]) {
        let Some(operand) = fold_float_operand(obj) else {
            return Ok(None);
        };
        *slot = operand;
    }
    let values = [coerced[0].1, coerced[1].1];
    if !domain.admits(&values[..arity]) {
        return Ok(None);
    }
    // Authentic boxed result, produced on the plain eval loop exactly as the
    // skipped residual would.  A raise, a non-float result and a non-finite
    // one all keep the residual.
    let boxed_result = {
        let _plain_guard = pyre_interpreter::call::force_plain_eval();
        pyre_interpreter::call::call_function_impl_result(concrete_callable, &operands[..arity])
    };
    let Ok(boxed_result) = boxed_result else {
        return Ok(None);
    };
    // The call above allocates, so the operand pointers read before it may have
    // been forwarded.  Re-fetch them from the walker's op cells, which the
    // collector does update, before anything reads through them again
    // (`try_walker_specialize_builtin_divmod_long_int` takes the same route).
    // The callable needs no re-fetch: a module's builtin function outlives
    // every nursery it could have been born in, while these operands are the
    // loop's own boxes.
    for (slot, &arg) in operands[..arity].iter_mut().zip(&r_args[2..2 + arity]) {
        let Some(refetched) = walker_concrete_ref_object(ctx, arg) else {
            return Ok(None);
        };
        *slot = refetched;
    }
    let Some(result_value) = fold_finite_float_result(boxed_result) else {
        return Ok(None);
    };
    // The compiled loop runs the row's emission, not the builtin it stands
    // for.  Compare the two on these operands and keep the residual when they
    // differ, so a row that disagrees is never compiled into the loop.  By
    // bits, which `==` cannot do: it cannot tell `-0.0` from `0.0`, and which
    // one is answered with is observable through `copysign`.
    if emit.evaluate(&values[..arity]).to_bits() != result_value.to_bits() {
        return Ok(None);
    }

    walker_guard_fold_callable(ctx, op.pc, r_args[0], concrete_callable)?;
    let x = walker_coerce_operand_to_float(
        ctx,
        op.pc,
        r_args[2],
        operands[0],
        coerced[0].0,
        values[0],
        false,
    )?;
    domain.emit_operand_guards(ctx, op.pc, x)?;
    let raw = match emit {
        MathFloatEmit::Unary(opcode, _) => {
            let raw = ctx.trace_ctx.record_op(opcode, &[x]);
            ctx.trace_ctx
                .set_opref_concrete(raw, majit_ir::Value::Float(result_value));
            raw
        }
        MathFloatEmit::Call1(raw_fn) => {
            walker_call_pure_float(ctx, raw_fn as *const (), &[x], &values[..1], result_value)
        }
        MathFloatEmit::Call2(raw_fn) => {
            let y = walker_coerce_operand_to_float(
                ctx,
                op.pc,
                r_args[3],
                operands[1],
                coerced[1].0,
                values[1],
                false,
            )?;
            walker_call_pure_float(ctx, raw_fn as *const (), &[x, y], &values, result_value)
        }
    };
    if domain == MathFloatDomain::ResultFinite {
        walker_guard_float_result_finite(ctx, op.pc, raw)?;
    }
    let boxed = crate::state::wrapfloat(ctx.trace_ctx, raw);
    ctx.trace_ctx.set_opref_concrete(
        boxed,
        majit_ir::Value::Ref(majit_ir::GcRef(boxed_result as usize)),
    );
    write_residual_call_result_to_dst(ctx, op.pc, dst, 'r', boxed)?;
    Ok(Some(()))
}

/// `math.sqrt(x)` on an exact int/float argument: the domain-guarded pure
/// `CALL_F(sqrt_nonneg_jit)` (ll_math.rs `ll_math_sqrt` → `sqrt_nonneg`) in
/// place of the opaque `bh_call_fn(sqrt_builtin, NULL, x)` residual.  A
/// negative argument raises in the authentic pre-execution and declines to the
/// generic residual, which records the raise.
pub(crate) fn try_walker_specialize_math_sqrt<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    code: &[u8],
    op: &DecodedOp,
    r_args: &[OpRef],
    dst: usize,
) -> Result<Option<()>, DispatchError> {
    walker_specialize_math_float(ctx, code, op, r_args, dst, 1, |callable| {
        pyre_interpreter::module::math::interp_math::is_math_sqrt_function(callable).then_some((
            MathFloatDomain::NonNegativeFinite,
            MathFloatEmit::Call1(crate::trace_opcode::sqrt_nonneg_jit),
        ))
    })
}

/// `math.log/cos/sin(x)` on an exact int/float argument — the direct
/// `ll_math_{log,cos,sin}` shape, whose exceptional branch is pinned rather
/// than guarded after the fact.
pub(crate) fn try_walker_specialize_math_log_trig<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    code: &[u8],
    op: &DecodedOp,
    r_args: &[OpRef],
    dst: usize,
) -> Result<Option<()>, DispatchError> {
    walker_specialize_math_float(ctx, code, op, r_args, dst, 1, |callable| {
        use pyre_interpreter::module::math::interp_math;
        if interp_math::is_math_log_function(callable) {
            Some((
                MathFloatDomain::PositiveFinite,
                MathFloatEmit::Call1(crate::trace_opcode::math_log_positive_jit),
            ))
        } else if interp_math::is_math_cos_function(callable) {
            Some((
                MathFloatDomain::Finite,
                MathFloatEmit::Call1(crate::trace_opcode::math_cos_finite_jit),
            ))
        } else if interp_math::is_math_sin_function(callable) {
            Some((
                MathFloatDomain::Finite,
                MathFloatEmit::Call1(crate::trace_opcode::math_sin_finite_jit),
            ))
        } else {
            None
        }
    })
}

/// `math.frexp(x)` on an exact int/float argument.  RPython lowers
/// `ll_math_frexp` to two unboxed results and `space.newtuple2`; emit the same
/// shape as two pure typed calls followed by a virtualizable object tuple.
/// This avoids the opaque builtin dispatch and the concrete tuple/element
/// allocations in numeric loops.  Rebound callables, subclasses, and other
/// coercion shapes retain the generic residual path.
pub(crate) fn try_walker_specialize_math_frexp<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    code: &[u8],
    op: &DecodedOp,
    r_args: &[OpRef],
    dst: usize,
) -> Result<Option<()>, DispatchError> {
    if r_args.len() != 3 {
        return Ok(None);
    }
    let arg_concretes = read_ref_var_list_concrete(code, op, 1, ctx);
    let (
        ConcreteValue::Ref(concrete_callable),
        ConcreteValue::Ref(null_or_self),
        ConcreteValue::Ref(arg_obj),
    ) = (arg_concretes[0], arg_concretes[1], arg_concretes[2])
    else {
        return Ok(None);
    };
    if concrete_callable.is_null() || !null_or_self.is_null() || arg_obj.is_null() {
        return Ok(None);
    }
    if !pyre_interpreter::module::math::interp_math::is_math_frexp_function(concrete_callable) {
        return Ok(None);
    }
    let (is_int, x_value) = unsafe {
        if !pyre_object::is_exact_builtin_instance(arg_obj) {
            return Ok(None);
        }
        if pyre_object::is_int(arg_obj) {
            (true, pyre_object::w_int_get_value(arg_obj) as f64)
        } else if pyre_object::is_float(arg_obj) {
            (false, pyre_object::w_float_get_value(arg_obj))
        } else {
            return Ok(None);
        }
    };

    // Execute the skipped builtin once so all concrete tuple/boxing choices
    // match the interpreter exactly.
    let boxed_result = {
        let _plain_guard = pyre_interpreter::call::force_plain_eval();
        pyre_interpreter::call::call_function_impl_result(concrete_callable, &[arg_obj])
    };
    let Ok(boxed_result) = boxed_result else {
        return Ok(None);
    };
    let (Some(mantissa_obj), Some(exponent_obj)) = (unsafe {
        (
            pyre_object::w_tuple_getitem(boxed_result, 0),
            pyre_object::w_tuple_getitem(boxed_result, 1),
        )
    }) else {
        return Ok(None);
    };
    let mantissa_value = unsafe { pyre_object::w_float_get_value(mantissa_obj) };
    let exponent_value = unsafe { pyre_object::w_int_get_value(exponent_obj) };

    let callable_op = r_args[0];
    if !callable_op.is_constant() {
        let expected = ctx.trace_ctx.const_ref(concrete_callable as i64);
        ctx.trace_ctx
            .record_guard(OpCode::GuardValue, &[callable_op, expected], 0);
        walker_capture_snapshot_for_last_guard(ctx, op.pc)?;
        ctx.trace_ctx
            .heap_cache_mut()
            .replace_box(callable_op, expected);
    }
    let x = walker_coerce_operand_to_float(ctx, op.pc, r_args[2], arg_obj, is_int, x_value, false)?;
    let mantissa = ctx.trace_ctx.call_typed_with_effect_pure(
        OpCode::CallF,
        pyre_interpreter::module::math::interp_math::jit_math_frexp_mantissa as *const (),
        &[x],
        &[majit_ir::Type::Float],
        majit_ir::Type::Float,
        majit_metainterp::ELIDABLE_CANNOT_RAISE_NO_HEAP_EFFECT_INFO,
        &[
            majit_ir::Value::Int(
                pyre_interpreter::module::math::interp_math::jit_math_frexp_mantissa as *const ()
                    as i64,
            ),
            majit_ir::Value::Float(x_value),
        ],
        majit_ir::Value::Float(mantissa_value),
    );
    ctx.trace_ctx
        .set_opref_concrete(mantissa, majit_ir::Value::Float(mantissa_value));
    let exponent = ctx.trace_ctx.call_typed_with_effect_pure(
        OpCode::CallI,
        pyre_interpreter::module::math::interp_math::jit_math_frexp_exponent as *const (),
        &[x],
        &[majit_ir::Type::Float],
        majit_ir::Type::Int,
        majit_metainterp::ELIDABLE_CANNOT_RAISE_NO_HEAP_EFFECT_INFO,
        &[
            majit_ir::Value::Int(
                pyre_interpreter::module::math::interp_math::jit_math_frexp_exponent as *const ()
                    as i64,
            ),
            majit_ir::Value::Float(x_value),
        ],
        majit_ir::Value::Int(exponent_value),
    );
    ctx.trace_ctx
        .set_opref_concrete(exponent, majit_ir::Value::Int(exponent_value));

    let mantissa_box = crate::state::wrapfloat(ctx.trace_ctx, mantissa);
    ctx.trace_ctx.set_opref_concrete(
        mantissa_box,
        majit_ir::Value::Ref(majit_ir::GcRef(mantissa_obj as usize)),
    );
    let exponent_box = walker_box_int(ctx, op.pc, exponent, exponent_value)?;
    ctx.trace_ctx.set_opref_concrete(
        exponent_box,
        box_int_concrete(exponent_value, exponent_obj as i64),
    );
    // `space.newtuple2` selects `W_SpecialisedTupleObject_oo` for the
    // float/int pair.  The traced allocation must use that same layout:
    // UNPACK_SEQUENCE specializes from the record-time concrete object's
    // class and reads the two inline `value*` fields.
    let tuple =
        crate::helpers::emit_specialised_tuple_oo_inline(ctx.trace_ctx, mantissa_box, exponent_box);
    ctx.trace_ctx.set_opref_concrete(
        tuple,
        majit_ir::Value::Ref(majit_ir::GcRef(boxed_result as usize)),
    );
    write_residual_call_result_to_dst(ctx, op.pc, dst, 'r', tuple)?;
    Ok(Some(()))
}

/// `math.ldexp(x, exp)` on exact numeric arguments.  This is the direct
/// walker equivalent of RPython's `ll_math_ldexp`: carry `x` and `exp`
/// unboxed, call the platform operation, and guard a finite result so the
/// overflow direction resumes in the builtin and raises `OverflowError`.
/// Underflow to signed zero remains on the fast path.  Non-finite concrete
/// inputs and non-int exponents retain the generic residual path.
pub(crate) fn try_walker_specialize_math_ldexp<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    code: &[u8],
    op: &DecodedOp,
    r_args: &[OpRef],
    dst: usize,
) -> Result<Option<()>, DispatchError> {
    if r_args.len() != 4 {
        return Ok(None);
    }
    let arg_concretes = read_ref_var_list_concrete(code, op, 1, ctx);
    let (
        ConcreteValue::Ref(concrete_callable),
        ConcreteValue::Ref(null_or_self),
        ConcreteValue::Ref(x_obj),
        ConcreteValue::Ref(exp_obj),
    ) = (
        arg_concretes[0],
        arg_concretes[1],
        arg_concretes[2],
        arg_concretes[3],
    )
    else {
        return Ok(None);
    };
    if concrete_callable.is_null()
        || !null_or_self.is_null()
        || x_obj.is_null()
        || exp_obj.is_null()
        || !pyre_interpreter::module::math::interp_math::is_math_ldexp_function(concrete_callable)
    {
        return Ok(None);
    }
    let (x_is_int, x_value, exp_value) = unsafe {
        if !pyre_object::is_exact_builtin_instance(x_obj)
            || !pyre_object::is_exact_builtin_instance(exp_obj)
        {
            return Ok(None);
        }
        let (x_is_int, x_value) = if pyre_object::is_int(x_obj) {
            (true, pyre_object::w_int_get_value(x_obj) as f64)
        } else if pyre_object::is_float(x_obj) {
            (false, pyre_object::w_float_get_value(x_obj))
        } else {
            return Ok(None);
        };
        if !pyre_object::is_int(exp_obj) {
            return Ok(None);
        }
        (x_is_int, x_value, pyre_object::w_int_get_value(exp_obj))
    };
    // The finite-result guard below represents RPython's errno/overflow
    // branch.  Trace non-finite inputs through the ordinary builtin because
    // ll_math_ldexp returns them unchanged rather than taking that guard.
    if !x_value.is_finite() {
        return Ok(None);
    }
    let boxed_result = {
        let _plain_guard = pyre_interpreter::call::force_plain_eval();
        pyre_interpreter::call::call_function_impl_result(concrete_callable, &[x_obj, exp_obj])
    };
    let Ok(boxed_result) = boxed_result else {
        return Ok(None);
    };
    let result_value = unsafe { pyre_object::w_float_get_value(boxed_result) };

    let callable_op = r_args[0];
    if !callable_op.is_constant() {
        let expected = ctx.trace_ctx.const_ref(concrete_callable as i64);
        ctx.trace_ctx
            .record_guard(OpCode::GuardValue, &[callable_op, expected], 0);
        walker_capture_snapshot_for_last_guard(ctx, op.pc)?;
        ctx.trace_ctx
            .heap_cache_mut()
            .replace_box(callable_op, expected);
    }
    let x = walker_coerce_operand_to_float(ctx, op.pc, r_args[2], x_obj, x_is_int, x_value, false)?;
    let (exp_type_addr, exp_descr) = crate::state::int_or_bool_unbox_type_descr(exp_obj);
    let exp = walker_unbox_int_typed(ctx, op.pc, r_args[3], exp_type_addr, exp_descr)?;
    ctx.trace_ctx
        .set_opref_concrete(exp, majit_ir::Value::Int(exp_value));
    let raw = ctx.trace_ctx.call_typed_with_effect_pure(
        OpCode::CallF,
        pyre_interpreter::module::math::interp_math::jit_math_ldexp_raw as *const (),
        &[x, exp],
        &[majit_ir::Type::Float, majit_ir::Type::Int],
        majit_ir::Type::Float,
        majit_metainterp::ELIDABLE_CANNOT_RAISE_NO_HEAP_EFFECT_INFO,
        &[
            majit_ir::Value::Int(
                pyre_interpreter::module::math::interp_math::jit_math_ldexp_raw as *const () as i64,
            ),
            majit_ir::Value::Float(x_value),
            majit_ir::Value::Int(exp_value),
        ],
        majit_ir::Value::Float(result_value),
    );
    ctx.trace_ctx
        .set_opref_concrete(raw, majit_ir::Value::Float(result_value));
    // `result - result == 0` exactly for every finite value, including
    // signed zero; infinity and NaN bail to the raising/propagating builtin.
    let diff = ctx.trace_ctx.record_op(OpCode::FloatSub, &[raw, raw]);
    ctx.trace_ctx
        .set_opref_concrete(diff, majit_ir::Value::Float(0.0));
    let zero = ctx.trace_ctx.const_float(0.0f64.to_bits() as i64);
    walker_float_cmp_guard(ctx, op.pc, OpCode::FloatEq, &[diff, zero], true)?;

    let boxed = crate::state::wrapfloat(ctx.trace_ctx, raw);
    ctx.trace_ctx.set_opref_concrete(
        boxed,
        majit_ir::Value::Ref(majit_ir::GcRef(boxed_result as usize)),
    );
    write_residual_call_result_to_dst(ctx, op.pc, dst, 'r', boxed)?;
    Ok(Some(()))
}

/// `math.isqrt(n)` on an exact nonnegative machine integer.
///
/// PyPy exposes `isqrt` from `app_math.py`, so tracing its `W_IntObject` arm
/// carries `n` unboxed through the integer algorithm and virtualizes the
/// result.  Pyre's native module wrapper otherwise materializes an `RBigInt`
/// before the walker can see that arm.  Recreate the translated shape as an
/// exact-class guard, unbox, pure non-raising integer call, and `wrapint`.
/// Longs, bools, subclasses, negative values, rebound callables, and
/// `__index__` objects retain the generic residual path.
pub(crate) fn try_walker_specialize_math_isqrt<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    code: &[u8],
    op: &DecodedOp,
    r_args: &[OpRef],
    dst: usize,
) -> Result<Option<()>, DispatchError> {
    if r_args.len() != 3 {
        return Ok(None);
    }
    let arg_concretes = read_ref_var_list_concrete(code, op, 1, ctx);
    let (
        ConcreteValue::Ref(concrete_callable),
        ConcreteValue::Ref(null_or_self),
        ConcreteValue::Ref(arg_obj),
    ) = (arg_concretes[0], arg_concretes[1], arg_concretes[2])
    else {
        return Ok(None);
    };
    if concrete_callable.is_null()
        || !null_or_self.is_null()
        || arg_obj.is_null()
        || !pyre_interpreter::module::math::interp_math::is_math_isqrt_function(concrete_callable)
    {
        return Ok(None);
    }
    let value = unsafe {
        // `is_int` also accepts a `bool`, whose singletons carry a `&BOOL_TYPE`
        // vtable and a NULL `w_class`.  The unbox and the exact-class guard
        // below both pin the canonical `int` class, so neither can hold for one;
        // decline it here as the int/long binop specializations do instead of
        // emitting a guard that fails on the operand it was recorded from.
        if !pyre_object::is_exact_builtin_instance(arg_obj)
            || !pyre_object::is_int(arg_obj)
            || pyre_object::is_bool(arg_obj)
        {
            return Ok(None);
        }
        pyre_object::w_int_get_value(arg_obj)
    };
    if value < 0 {
        return Ok(None);
    }
    let boxed_result = {
        let _plain_guard = pyre_interpreter::call::force_plain_eval();
        pyre_interpreter::call::call_function_impl_result(concrete_callable, &[arg_obj])
    };
    let Ok(boxed_result) = boxed_result else {
        return Ok(None);
    };
    if !unsafe { pyre_object::is_int(boxed_result) } {
        return Ok(None);
    }
    let result_value = unsafe { pyre_object::w_int_get_value(boxed_result) };

    let callable_op = r_args[0];
    if !callable_op.is_constant() {
        let expected = ctx.trace_ctx.const_ref(concrete_callable as i64);
        ctx.trace_ctx
            .record_guard(OpCode::GuardValue, &[callable_op, expected], 0);
        walker_capture_snapshot_for_last_guard(ctx, op.pc)?;
        ctx.trace_ctx
            .heap_cache_mut()
            .replace_box(callable_op, expected);
    }
    let arg_op = r_args[2];
    let int_type_addr = &pyre_object::pyobject::INT_TYPE as *const _ as i64;
    let raw_int = walker_unbox_int(ctx, op.pc, arg_op, int_type_addr)?;
    walker_guard_exact_w_class(
        ctx,
        op.pc,
        arg_op,
        pyre_object::pyobject::get_instantiate(&pyre_object::pyobject::INT_TYPE),
    )?;
    ctx.trace_ctx
        .set_opref_concrete(raw_int, majit_ir::Value::Int(value));
    let zero = ctx.trace_ctx.const_int(0);
    let nonnegative = ctx.trace_ctx.record_op(OpCode::IntGe, &[raw_int, zero]);
    ctx.trace_ctx
        .set_opref_concrete(nonnegative, majit_ir::Value::Int(1));
    walker_emit_guard_with_snapshot(ctx, op.pc, OpCode::GuardTrue, &[nonnegative])?;

    let raw_result = ctx.trace_ctx.call_typed_with_effect_pure(
        OpCode::CallI,
        pyre_interpreter::module::math::interp_math::jit_math_isqrt_i64 as *const (),
        &[raw_int],
        &[majit_ir::Type::Int],
        majit_ir::Type::Int,
        majit_metainterp::ELIDABLE_CANNOT_RAISE_NO_HEAP_EFFECT_INFO,
        &[
            majit_ir::Value::Int(
                pyre_interpreter::module::math::interp_math::jit_math_isqrt_i64 as *const () as i64,
            ),
            majit_ir::Value::Int(value),
        ],
        majit_ir::Value::Int(result_value),
    );
    ctx.trace_ctx
        .set_opref_concrete(raw_result, majit_ir::Value::Int(result_value));
    let boxed = walker_box_int(ctx, op.pc, raw_result, result_value)?;
    ctx.trace_ctx
        .set_opref_concrete(boxed, box_int_concrete(result_value, boxed_result as i64));
    write_residual_call_result_to_dst(ctx, op.pc, dst, 'r', boxed)?;
    Ok(Some(()))
}

/// `int(x)` for an exact float whose truncated value fits a machine Signed.
///
/// PyPy `floatobject.py:newint_from_float` first runs
/// `ovfcheck_float_to_int`; its success arm is exactly
/// `CAST_FLOAT_TO_INT + space.newint`.  Emit that arm with the corresponding
/// `-2**63 <= x < 2**63` guards, leaving NaN, infinity, out-of-range values,
/// subclasses, and rebound constructors on the ordinary residual path.  The
/// slow arm remains responsible for `newlong_from_float` and its exceptions.
pub(crate) fn try_walker_specialize_int_call<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    code: &[u8],
    op: &DecodedOp,
    r_args: &[OpRef],
    dst: usize,
) -> Result<Option<()>, DispatchError> {
    if r_args.len() != 3 {
        return Ok(None);
    }
    let arg_concretes = read_ref_var_list_concrete(code, op, 1, ctx);
    let (
        ConcreteValue::Ref(concrete_callable),
        ConcreteValue::Ref(null_or_self),
        ConcreteValue::Ref(arg_obj),
    ) = (arg_concretes[0], arg_concretes[1], arg_concretes[2])
    else {
        return Ok(None);
    };
    if concrete_callable.is_null() || !null_or_self.is_null() || arg_obj.is_null() {
        return Ok(None);
    }
    let int_type_obj = pyre_object::pyobject::get_instantiate(&pyre_object::pyobject::INT_TYPE);
    if !std::ptr::eq(concrete_callable, int_type_obj) {
        return Ok(None);
    }
    let value = unsafe {
        if !pyre_object::is_exact_builtin_instance(arg_obj) || !pyre_object::is_float(arg_obj) {
            return Ok(None);
        }
        pyre_object::w_float_get_value(arg_obj)
    };
    // `2**63` is exactly representable while `i64::MAX` is not; use a strict
    // upper bound, matching ovfcheck_float_to_int on a signed 64-bit target.
    const SIGNED_MIN_AS_FLOAT: f64 = -9223372036854775808.0;
    const SIGNED_LIMIT_AS_FLOAT: f64 = 9223372036854775808.0;
    if !(value >= SIGNED_MIN_AS_FLOAT && value < SIGNED_LIMIT_AS_FLOAT) {
        return Ok(None);
    }
    let boxed_result = {
        let _plain_guard = pyre_interpreter::call::force_plain_eval();
        pyre_interpreter::call::call_function_impl_result(concrete_callable, &[arg_obj])
    };
    let Ok(boxed_result) = boxed_result else {
        return Ok(None);
    };
    if !unsafe { pyre_object::is_int(boxed_result) } {
        return Ok(None);
    }
    let result_value = unsafe { pyre_object::w_int_get_value(boxed_result) };

    let callable_op = r_args[0];
    if !callable_op.is_constant() {
        let expected = ctx.trace_ctx.const_ref(concrete_callable as i64);
        ctx.trace_ctx
            .record_guard(OpCode::GuardValue, &[callable_op, expected], 0);
        walker_capture_snapshot_for_last_guard(ctx, op.pc)?;
        ctx.trace_ctx
            .heap_cache_mut()
            .replace_box(callable_op, expected);
    }
    let arg_op = r_args[2];
    let float_type_addr = &pyre_object::pyobject::FLOAT_TYPE as *const _ as i64;
    let raw_float = walker_unbox_float(ctx, op.pc, arg_op, float_type_addr)?;
    walker_guard_exact_w_class(
        ctx,
        op.pc,
        arg_op,
        pyre_object::pyobject::get_instantiate(&pyre_object::pyobject::FLOAT_TYPE),
    )?;
    ctx.trace_ctx
        .set_opref_concrete(raw_float, majit_ir::Value::Float(value));
    let low = ctx
        .trace_ctx
        .const_float(SIGNED_MIN_AS_FLOAT.to_bits() as i64);
    let high = ctx
        .trace_ctx
        .const_float(SIGNED_LIMIT_AS_FLOAT.to_bits() as i64);
    walker_float_cmp_guard(ctx, op.pc, OpCode::FloatGe, &[raw_float, low], true)?;
    walker_float_cmp_guard(ctx, op.pc, OpCode::FloatLt, &[raw_float, high], true)?;

    let raw_int = ctx
        .trace_ctx
        .record_op(OpCode::CastFloatToInt, &[raw_float]);
    ctx.trace_ctx
        .set_opref_concrete(raw_int, majit_ir::Value::Int(result_value));
    let boxed = walker_box_int(ctx, op.pc, raw_int, result_value)?;
    ctx.trace_ctx
        .set_opref_concrete(boxed, box_int_concrete(result_value, boxed_result as i64));
    write_residual_call_result_to_dst(ctx, op.pc, dst, 'r', boxed)?;
    Ok(Some(()))
}

/// `math.fabs(x)` on an exact int/float argument.  RPython lowers
/// `ll_math_fabs` to a sign mask, so the whole builtin is one `FloatAbs` once
/// the operand is unboxed, and `fabs` raises for no input, which is why the
/// row is [`MathFloatDomain::Total`].
///
/// `Total` only spares the row its operand guards.  The shared driver still
/// screens the authentic result through `fold_finite_float_result`, so
/// `fabs(inf)` and `fabs(nan)` decline at trace time and keep the residual
/// even though the sign mask would answer them correctly.
pub(crate) fn try_walker_specialize_math_fabs<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    code: &[u8],
    op: &DecodedOp,
    r_args: &[OpRef],
    dst: usize,
) -> Result<Option<()>, DispatchError> {
    walker_specialize_math_float(ctx, code, op, r_args, dst, 1, |callable| {
        pyre_interpreter::module::math::interp_math::is_math_fabs_function(callable).then_some((
            MathFloatDomain::Total,
            MathFloatEmit::Unary(OpCode::FloatAbs, f64::abs),
        ))
    })
}

/// Which reduction `try_walker_specialize_math_round_to_int` is folding.
#[derive(Clone, Copy)]
pub(crate) enum MathRoundMode {
    Floor,
    Ceil,
    Trunc,
}

/// `math.floor(x)` / `math.ceil(x)` / `math.trunc(x)` on an exact float.
///
/// `floor`/`ceil`/`trunc` look the dunder up on the type and call
/// it; for an exact float that resolves to `W_FloatObject`'s own reduction
/// followed by `newint_from_float`, whose `ovfcheck_float_to_int` arm is a
/// machine cast.  Recreate that shape: unbox, guard the operand into the
/// signed range, round, and cast.
///
/// Only an exact float is folded.  An `int` argument reaches
/// `int.__floor__`, which returns the argument object itself rather than a
/// fresh box, and a float subclass may override the dunder — both keep the
/// residual.
///
/// The range guard is on the operand rather than the rounded value, which is
/// sufficient for all three modes: `-2**63` is an integer so `floor` cannot
/// leave the range from below, `|trunc(x)| <= |x|`, and every float below
/// `2**63` large enough for `ceil` to move it is already integral (the ulp
/// there is 2048).
pub(crate) fn try_walker_specialize_math_round_to_int<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    code: &[u8],
    op: &DecodedOp,
    r_args: &[OpRef],
    dst: usize,
    mode: MathRoundMode,
) -> Result<Option<()>, DispatchError> {
    if r_args.len() != 3 {
        return Ok(None);
    }
    let arg_concretes = read_ref_var_list_concrete(code, op, 1, ctx);
    let (
        ConcreteValue::Ref(concrete_callable),
        ConcreteValue::Ref(null_or_self),
        ConcreteValue::Ref(arg_obj),
    ) = (arg_concretes[0], arg_concretes[1], arg_concretes[2])
    else {
        return Ok(None);
    };
    if concrete_callable.is_null() || !null_or_self.is_null() || arg_obj.is_null() {
        return Ok(None);
    }
    let is_this_builtin: fn(pyre_object::PyObjectRef) -> bool = match mode {
        MathRoundMode::Floor => pyre_interpreter::module::math::interp_math::is_math_floor_function,
        MathRoundMode::Ceil => pyre_interpreter::module::math::interp_math::is_math_ceil_function,
        MathRoundMode::Trunc => pyre_interpreter::module::math::interp_math::is_math_trunc_function,
    };
    if !is_this_builtin(concrete_callable) {
        return Ok(None);
    }
    let value = unsafe {
        if !pyre_object::is_exact_builtin_instance(arg_obj) || !pyre_object::is_float(arg_obj) {
            return Ok(None);
        }
        pyre_object::w_float_get_value(arg_obj)
    };
    // `2**63` is exactly representable while `i64::MAX` is not; use a strict
    // upper bound, matching ovfcheck_float_to_int on a signed 64-bit target.
    // NaN and both infinities fail these comparisons and keep the residual,
    // which raises for them.
    const SIGNED_MIN_AS_FLOAT: f64 = -9223372036854775808.0;
    const SIGNED_LIMIT_AS_FLOAT: f64 = 9223372036854775808.0;
    if !(value >= SIGNED_MIN_AS_FLOAT && value < SIGNED_LIMIT_AS_FLOAT) {
        return Ok(None);
    }
    let boxed_result = {
        let _plain_guard = pyre_interpreter::call::force_plain_eval();
        pyre_interpreter::call::call_function_impl_result(concrete_callable, &[arg_obj])
    };
    let Ok(boxed_result) = boxed_result else {
        return Ok(None);
    };
    if !unsafe { pyre_object::is_int(boxed_result) } {
        return Ok(None);
    }
    let result_value = unsafe { pyre_object::w_int_get_value(boxed_result) };

    let callable_op = r_args[0];
    if !callable_op.is_constant() {
        let expected = ctx.trace_ctx.const_ref(concrete_callable as i64);
        ctx.trace_ctx
            .record_guard(OpCode::GuardValue, &[callable_op, expected], 0);
        walker_capture_snapshot_for_last_guard(ctx, op.pc)?;
        ctx.trace_ctx
            .heap_cache_mut()
            .replace_box(callable_op, expected);
    }
    let arg_op = r_args[2];
    let float_type_addr = &pyre_object::pyobject::FLOAT_TYPE as *const _ as i64;
    let raw_float = walker_unbox_float(ctx, op.pc, arg_op, float_type_addr)?;
    walker_guard_exact_w_class(
        ctx,
        op.pc,
        arg_op,
        pyre_object::pyobject::get_instantiate(&pyre_object::pyobject::FLOAT_TYPE),
    )?;
    ctx.trace_ctx
        .set_opref_concrete(raw_float, majit_ir::Value::Float(value));
    let low = ctx
        .trace_ctx
        .const_float(SIGNED_MIN_AS_FLOAT.to_bits() as i64);
    let high = ctx
        .trace_ctx
        .const_float(SIGNED_LIMIT_AS_FLOAT.to_bits() as i64);
    walker_float_cmp_guard(ctx, op.pc, OpCode::FloatGe, &[raw_float, low], true)?;
    walker_float_cmp_guard(ctx, op.pc, OpCode::FloatLt, &[raw_float, high], true)?;

    // `CastFloatToInt` already truncates toward zero, so only floor and ceil
    // need a rounding step.  Both are elidable and cannot raise.
    let rounded = match mode {
        MathRoundMode::Trunc => raw_float,
        MathRoundMode::Floor | MathRoundMode::Ceil => {
            let (helper, rounded_value) = match mode {
                MathRoundMode::Floor => (
                    pyre_interpreter::module::math::interp_math::jit_math_floor_raw as *const (),
                    value.floor(),
                ),
                _ => (
                    pyre_interpreter::module::math::interp_math::jit_math_ceil_raw as *const (),
                    value.ceil(),
                ),
            };
            let rounded = ctx.trace_ctx.call_typed_with_effect_pure(
                OpCode::CallF,
                helper,
                &[raw_float],
                &[majit_ir::Type::Float],
                majit_ir::Type::Float,
                majit_metainterp::ELIDABLE_CANNOT_RAISE_NO_HEAP_EFFECT_INFO,
                &[
                    majit_ir::Value::Int(helper as i64),
                    majit_ir::Value::Float(value),
                ],
                majit_ir::Value::Float(rounded_value),
            );
            ctx.trace_ctx
                .set_opref_concrete(rounded, majit_ir::Value::Float(rounded_value));
            rounded
        }
    };
    let raw_int = ctx.trace_ctx.record_op(OpCode::CastFloatToInt, &[rounded]);
    ctx.trace_ctx
        .set_opref_concrete(raw_int, majit_ir::Value::Int(result_value));
    let boxed = walker_box_int(ctx, op.pc, raw_int, result_value)?;
    ctx.trace_ctx
        .set_opref_concrete(boxed, box_int_concrete(result_value, boxed_result as i64));
    write_residual_call_result_to_dst(ctx, op.pc, dst, 'r', boxed)?;
    Ok(Some(()))
}

/// Read a plain `bh_call_fn(callable, PY_NULL, args…)` shape's concrete
/// operands.  `None` means the call is not that shape — a bound receiver in
/// `null_or_self`, a NULL operand, or a non-`Ref` concrete.
fn plain_builtin_call_concretes<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    code: &[u8],
    op: &DecodedOp,
    r_args: &[OpRef],
    arity: usize,
) -> Option<(pyre_object::PyObjectRef, [pyre_object::PyObjectRef; 2])> {
    if r_args.len() != arity + 2 {
        return None;
    }
    let arg_concretes = read_ref_var_list_concrete(code, op, 1, ctx);
    let (ConcreteValue::Ref(concrete_callable), ConcreteValue::Ref(null_or_self)) =
        (arg_concretes[0], arg_concretes[1])
    else {
        return None;
    };
    if concrete_callable.is_null() || !null_or_self.is_null() {
        return None;
    }
    let mut operands = [pyre_object::PY_NULL; 2];
    for (slot, concrete) in operands.iter_mut().zip(&arg_concretes[2..arity + 2]) {
        let ConcreteValue::Ref(obj) = *concrete else {
            return None;
        };
        if obj.is_null() {
            return None;
        }
        *slot = obj;
    }
    Some((concrete_callable, operands))
}

/// Classify a fold operand as an exact `int`/`bool`/`float` and read its
/// value as an `f64`.  A numeric subclass keeps the builtin `ob_type` layout
/// but carries a Python-visible `w_class`, and the `guard_class` the coercion
/// emits reads `ob_type`, so it would not catch the subclass — decline here.
fn fold_float_operand(obj: pyre_object::PyObjectRef) -> Option<(bool, f64)> {
    unsafe {
        if !pyre_object::is_exact_builtin_instance(obj) {
            return None;
        }
        if pyre_object::is_int(obj) {
            Some((true, pyre_object::w_int_get_value(obj) as f64))
        } else if pyre_object::is_float(obj) {
            Some((false, pyre_object::w_float_get_value(obj)))
        } else {
            None
        }
    }
}

/// Pin a fold's callable identity.  The module-attr fold usually makes it a
/// constant already; guard only when it is not.
fn walker_guard_fold_callable<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    pc: usize,
    callable_op: OpRef,
    concrete_callable: pyre_object::PyObjectRef,
) -> Result<(), DispatchError> {
    if callable_op.is_constant() {
        return Ok(());
    }
    let expected = ctx.trace_ctx.const_ref(concrete_callable as i64);
    ctx.trace_ctx
        .record_guard(OpCode::GuardValue, &[callable_op, expected], 0);
    walker_capture_snapshot_for_last_guard(ctx, pc)?;
    ctx.trace_ctx
        .heap_cache_mut()
        .replace_box(callable_op, expected);
    Ok(())
}

/// `raw - raw == 0` holds exactly for every finite value, including signed
/// zero; an infinity or a NaN bails to the builtin.  This is the guard that
/// makes the generic float folds sound: the raw helper answers what the
/// builtin body computes and reports every raising direction as NaN, so a
/// finite result means the builtin returned this exact value.
fn walker_guard_float_result_finite<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    pc: usize,
    raw: OpRef,
) -> Result<(), DispatchError> {
    let diff = ctx.trace_ctx.record_op(OpCode::FloatSub, &[raw, raw]);
    ctx.trace_ctx
        .set_opref_concrete(diff, majit_ir::Value::Float(0.0));
    let zero = ctx.trace_ctx.const_float(0.0f64.to_bits() as i64);
    walker_float_cmp_guard(ctx, pc, OpCode::FloatEq, &[diff, zero], true)
}

/// The one-argument rows of the `math` float fold: `interp_math`'s `pm1!`
/// family, each standing for one raw helper in `MATH_FLOAT1_FOLDS`.  Every row
/// is [`MathFloatDomain::ResultFinite`], so the table is the whole of what
/// this entry point knows.
pub(crate) fn try_walker_specialize_math_float1<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    code: &[u8],
    op: &DecodedOp,
    r_args: &[OpRef],
    dst: usize,
) -> Result<Option<()>, DispatchError> {
    walker_specialize_math_float(ctx, code, op, r_args, dst, 1, |callable| {
        pyre_interpreter::module::math::interp_math::math_float1_fold_helper(callable)
            .map(|raw| (MathFloatDomain::ResultFinite, MathFloatEmit::Call1(raw)))
    })
}

/// The two-argument rows — `pow`, `fmod`, `copysign`, `remainder` and
/// `atan2`.  The finite-result guard carries `pow`'s `ValueError`
/// (`pow(0.0, -2.0)`) and `OverflowError` (`pow(1e100, 1e100)`) directions
/// back to the builtin.
pub(crate) fn try_walker_specialize_math_float2<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    code: &[u8],
    op: &DecodedOp,
    r_args: &[OpRef],
    dst: usize,
) -> Result<Option<()>, DispatchError> {
    walker_specialize_math_float(ctx, code, op, r_args, dst, 2, |callable| {
        pyre_interpreter::module::math::interp_math::math_float2_fold_helper(callable)
            .map(|raw| (MathFloatDomain::ResultFinite, MathFloatEmit::Call2(raw)))
    })
}

/// `math.isclose(a, b)` with both tolerances defaulted.
///
/// The residual costs a builtin dispatch, a keyword split and two
/// `try_get_double` conversions to produce one of two prebuilt singletons.
/// Emit instead the two unboxed operands and a pure elidable `CALL_I` into
/// `jit_math_isclose_default`, then let [`walker_newbool_guarded`] pin its
/// truth and hand back the singleton.  That helper is total, so unlike the
/// float folds this one needs no result guard of its own.
///
/// A keyword argument (which would reach `bh_call_fn_kw`, not this shape), a
/// third positional, a numeric subclass or a rebound callable all retain the
/// generic residual path (SAFE).  Where the result goes afterwards is not a
/// condition: `space.newbool` guards its `if b:` for every consumer.
pub(crate) fn try_walker_specialize_math_isclose<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    code: &[u8],
    op: &DecodedOp,
    r_args: &[OpRef],
    dst: usize,
    dst_bank: char,
) -> Result<Option<()>, DispatchError> {
    let Some((concrete_callable, operands)) =
        plain_builtin_call_concretes(ctx, code, op, r_args, 2)
    else {
        return Ok(None);
    };
    if !pyre_interpreter::module::math::interp_math::is_math_isclose_function(concrete_callable) {
        return Ok(None);
    }
    // Settle the result's shape before emitting anything: everything below
    // this point commits ops to the trace, so ask up front for the one
    // condition `walker_newbool_guarded` needs — a resume image to land the
    // truth guard's bail in, and a Ref destination for the singleton.
    if ctx.fbw_mode.snapshot_sym.is_null() || dst_bank != 'r' {
        return Ok(None);
    }
    let (Some((a_is_int, a_value)), Some((b_is_int, b_value))) = (
        fold_float_operand(operands[0]),
        fold_float_operand(operands[1]),
    ) else {
        return Ok(None);
    };
    let boxed_result = {
        let _plain_guard = pyre_interpreter::call::force_plain_eval();
        pyre_interpreter::call::call_function_impl_result(concrete_callable, &operands)
    };
    let Ok(boxed_result) = boxed_result else {
        return Ok(None);
    };
    // The helper reimplements the comparison rather than calling the module's
    // own; compare the two answers on this operand pair before committing to
    // it, so a divergence declines here instead of recording wrong code.
    let observed = std::ptr::eq(boxed_result, pyre_object::w_bool_from(true));
    if !observed && !std::ptr::eq(boxed_result, pyre_object::w_bool_from(false)) {
        return Ok(None);
    }
    let helper = pyre_interpreter::module::math::interp_math::jit_math_isclose_default;
    if (helper(a_value, b_value) != 0) != observed {
        return Ok(None);
    }

    walker_guard_fold_callable(ctx, op.pc, r_args[0], concrete_callable)?;
    let a = walker_coerce_operand_to_float(
        ctx,
        op.pc,
        r_args[2],
        operands[0],
        a_is_int,
        a_value,
        false,
    )?;
    let b = walker_coerce_operand_to_float(
        ctx,
        op.pc,
        r_args[3],
        operands[1],
        b_is_int,
        b_value,
        false,
    )?;
    let truth = ctx.trace_ctx.call_typed_with_effect_pure(
        OpCode::CallI,
        helper as *const (),
        &[a, b],
        &[majit_ir::Type::Float, majit_ir::Type::Float],
        majit_ir::Type::Int,
        majit_metainterp::ELIDABLE_CANNOT_RAISE_NO_HEAP_EFFECT_INFO,
        &[
            majit_ir::Value::Int(helper as *const () as i64),
            majit_ir::Value::Float(a_value),
            majit_ir::Value::Float(b_value),
        ],
        majit_ir::Value::Int(i64::from(observed)),
    );
    ctx.trace_ctx
        .set_opref_concrete(truth, majit_ir::Value::Int(i64::from(observed)));
    // The shape check above already established what this re-tests, so the
    // `None` arm is unreachable; keep it as the decline rather than assert.
    let Some(boxed) = walker_newbool_guarded(ctx, op.pc, truth, observed, dst_bank)? else {
        return Ok(None);
    };
    write_residual_call_result_to_dst(ctx, op.pc, dst, 'r', boxed)?;
    Ok(Some(()))
}

/// The `f64` behind an exact-`float` fold result, or `None` when the builtin
/// answered something else or something the finite-result guard would reject.
fn fold_finite_float_result(boxed_result: pyre_object::PyObjectRef) -> Option<f64> {
    unsafe {
        if boxed_result.is_null()
            || !pyre_object::is_exact_builtin_instance(boxed_result)
            || !pyre_object::is_float(boxed_result)
        {
            return None;
        }
        let value = pyre_object::w_float_get_value(boxed_result);
        value.is_finite().then_some(value)
    }
}

/// The machine int a folded builtin's authentic boxed result carries, or
/// `None` when that result is not an exact `int` at all.  `True`/`False` are
/// excluded: a raw helper reporting `1` must not be accepted for a builtin
/// that returned the bool, because the trace boxes it with `wrapint`.
fn fold_boxed_int_value(boxed_result: pyre_object::PyObjectRef) -> Option<i64> {
    unsafe {
        if boxed_result.is_null()
            || !pyre_object::is_exact_builtin_instance(boxed_result)
            || !pyre_object::is_int(boxed_result)
            || pyre_object::is_bool(boxed_result)
        {
            return None;
        }
        Some(pyre_object::w_int_get_value(boxed_result))
    }
}

/// Guard an int-channel helper's result against its decline sentinel, so every
/// operand the helper does not answer for resumes in the builtin.
fn walker_guard_int_result_not_declined<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    pc: usize,
    raw: OpRef,
) -> Result<(), DispatchError> {
    let sentinel = ctx
        .trace_ctx
        .const_int(pyre_interpreter::jit_builtin_folds::INT_FOLD_DECLINE);
    let answered = ctx.trace_ctx.record_op(OpCode::IntNe, &[raw, sentinel]);
    ctx.trace_ctx
        .set_opref_concrete(answered, majit_ir::Value::Int(1));
    walker_emit_fold_guard_with_snapshot(ctx, pc, OpCode::GuardTrue, &[answered])
}

/// The generic builtin fold, one argument.
///
/// Every builtin that is not hand-specialized reaches the interpreter as
/// `bh_call_fn(builtin, NULL, x)`, and that residual costs the same regardless
/// of what the builtin does: the frame force, the argument rooting, the
/// execution-context resolution and the gateway signature binding all run
/// before the body does.  Measured against pypy 7.3.20 the floor is an order
/// of magnitude on its own — `hash`, `ord` and `abs` all sit within a few
/// percent of each other because none of them is paying for its own work.
///
/// `jit_builtin_folds` names, per builtin, a raw helper that is the body of
/// that builtin restricted to the operands it can answer without running
/// app-level code and without allocating.  Emit a direct call into it, guard
/// the channel's decline sentinel, and box the result inline so the optimizer
/// can keep it virtual.  A declined operand — a subclass instance, a shape the
/// helper does not implement, the argument that would have raised — resumes in
/// the builtin, which re-executes the call from scratch, so the fold needs no
/// per-builtin domain knowledge and adding a table row is all it takes to
/// cover another one.  Rebound callables keep the residual (SAFE).
pub(crate) fn try_walker_specialize_builtin_fold1<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    code: &[u8],
    op: &DecodedOp,
    r_args: &[OpRef],
    dst: usize,
) -> Result<Option<()>, DispatchError> {
    use pyre_interpreter::jit_builtin_folds::{BuiltinFoldRaw, INT_FOLD_DECLINE};

    let Some((concrete_callable, operands)) =
        plain_builtin_call_concretes(ctx, code, op, r_args, 1)
    else {
        return Ok(None);
    };
    let rows: Vec<_> =
        pyre_interpreter::jit_builtin_folds::builtin_folds_for(concrete_callable, 1).collect();
    // Ask the raw helpers before the builtin runs.  A call no row answers for
    // is not this fold's shape, and the walker has to learn that without
    // executing the builtin: the residual it falls back to executes the call
    // again, so a decline taken afterwards would run a side-effecting
    // `__hash__` or `__abs__` twice in one walk.
    let answered = rows.iter().any(|fold| match fold.raw {
        BuiltinFoldRaw::Int1(raw_fn) => raw_fn(operands[0] as i64) != INT_FOLD_DECLINE,
        BuiltinFoldRaw::Float1(raw_fn) => !raw_fn(operands[0] as i64).is_nan(),
        BuiltinFoldRaw::Ref2(_) => false,
    });
    if !answered {
        return Ok(None);
    }
    // Authentic boxed result, produced on the plain eval loop exactly as the
    // skipped residual would.  Every row cross-checks its helper against this,
    // so a helper that disagrees with the builtin it stands for declines here
    // rather than compiling the disagreement into the loop.
    let boxed_result = {
        let _plain_guard = pyre_interpreter::call::force_plain_eval();
        pyre_interpreter::call::call_function_impl_result(concrete_callable, &operands[..1])
    };
    let Ok(boxed_result) = boxed_result else {
        return Ok(None);
    };

    for fold in &rows {
        match fold.raw {
            BuiltinFoldRaw::Int1(raw_fn) => {
                let value = raw_fn(operands[0] as i64);
                if value == INT_FOLD_DECLINE || fold_boxed_int_value(boxed_result) != Some(value) {
                    continue;
                }
                walker_guard_fold_callable(ctx, op.pc, r_args[0], concrete_callable)?;
                let raw = ctx.trace_ctx.call_typed_with_effect_pure(
                    OpCode::CallI,
                    raw_fn as *const (),
                    &[r_args[2]],
                    &[majit_ir::Type::Ref],
                    majit_ir::Type::Int,
                    majit_metainterp::ELIDABLE_CANNOT_RAISE_NO_HEAP_EFFECT_INFO,
                    &[
                        majit_ir::Value::Int(raw_fn as *const () as i64),
                        majit_ir::Value::Ref(majit_ir::GcRef(operands[0] as usize)),
                    ],
                    majit_ir::Value::Int(value),
                );
                ctx.trace_ctx
                    .set_opref_concrete(raw, majit_ir::Value::Int(value));
                walker_guard_int_result_not_declined(ctx, op.pc, raw)?;
                let boxed = walker_box_int(ctx, op.pc, raw, value)?;
                ctx.trace_ctx
                    .set_opref_concrete(boxed, box_int_concrete(value, boxed_result as i64));
                write_residual_call_result_to_dst(ctx, op.pc, dst, 'r', boxed)?;
                return Ok(Some(()));
            }
            BuiltinFoldRaw::Float1(raw_fn) => {
                let Some(result_value) = fold_finite_float_result(boxed_result) else {
                    continue;
                };
                let value = raw_fn(operands[0] as i64);
                // By bits: `==` cannot tell `-0.0` from `0.0`, and which one
                // the fold answers with is observable through `copysign`.
                if value.to_bits() != result_value.to_bits() {
                    continue;
                }
                walker_guard_fold_callable(ctx, op.pc, r_args[0], concrete_callable)?;
                let raw = ctx.trace_ctx.call_typed_with_effect_pure(
                    OpCode::CallF,
                    raw_fn as *const (),
                    &[r_args[2]],
                    &[majit_ir::Type::Ref],
                    majit_ir::Type::Float,
                    majit_metainterp::ELIDABLE_CANNOT_RAISE_NO_HEAP_EFFECT_INFO,
                    &[
                        majit_ir::Value::Int(raw_fn as *const () as i64),
                        majit_ir::Value::Ref(majit_ir::GcRef(operands[0] as usize)),
                    ],
                    majit_ir::Value::Float(value),
                );
                ctx.trace_ctx
                    .set_opref_concrete(raw, majit_ir::Value::Float(value));
                walker_guard_float_result_finite(ctx, op.pc, raw)?;
                let boxed = crate::state::wrapfloat(ctx.trace_ctx, raw);
                ctx.trace_ctx.set_opref_concrete(
                    boxed,
                    majit_ir::Value::Ref(majit_ir::GcRef(boxed_result as usize)),
                );
                write_residual_call_result_to_dst(ctx, op.pc, dst, 'r', boxed)?;
                return Ok(Some(()));
            }
            BuiltinFoldRaw::Ref2(_) => continue,
        }
    }
    Ok(None)
}

/// The two-argument half of the generic builtin fold — `min(a, b)` and
/// `max(a, b)`, whose helpers return one of their own arguments rather than
/// building anything.  Same shape and same soundness argument as
/// [`try_walker_specialize_builtin_fold1`]; a `PY_NULL` is the decline the
/// trailing non-null guard carries back to the builtin.
pub(crate) fn try_walker_specialize_builtin_fold2<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    code: &[u8],
    op: &DecodedOp,
    r_args: &[OpRef],
    dst: usize,
) -> Result<Option<()>, DispatchError> {
    use pyre_interpreter::jit_builtin_folds::BuiltinFoldRaw;

    let Some((concrete_callable, operands)) =
        plain_builtin_call_concretes(ctx, code, op, r_args, 2)
    else {
        return Ok(None);
    };
    let rows: Vec<_> =
        pyre_interpreter::jit_builtin_folds::builtin_folds_for(concrete_callable, 2).collect();
    // Same ordering as the one-argument half: no row may answer only after the
    // builtin has already run, or the residual re-executes it.
    let answered = rows.iter().any(|fold| match fold.raw {
        BuiltinFoldRaw::Ref2(raw_fn) => {
            !(raw_fn(operands[0] as i64, operands[1] as i64) as pyre_object::PyObjectRef).is_null()
        }
        BuiltinFoldRaw::Int1(_) | BuiltinFoldRaw::Float1(_) => false,
    });
    if !answered {
        return Ok(None);
    }
    let boxed_result = {
        let _plain_guard = pyre_interpreter::call::force_plain_eval();
        pyre_interpreter::call::call_function_impl_result(concrete_callable, &operands)
    };
    let Ok(boxed_result) = boxed_result else {
        return Ok(None);
    };

    for fold in &rows {
        let BuiltinFoldRaw::Ref2(raw_fn) = fold.raw else {
            continue;
        };
        let value = raw_fn(operands[0] as i64, operands[1] as i64) as pyre_object::PyObjectRef;
        if value.is_null() || value != boxed_result {
            continue;
        }
        // Two distinct exact machine ints need no builtin-specific branch:
        // the ordering guard determines which object `min_max_multiple_args`
        // would keep for either comparison direction.  The answer written to
        // the destination is the winning operand's own `OpRef`, so nothing is
        // allocated and no new box is made.  If a later iteration reverses
        // the ordering the guard fails and resumes in the builtin, the same
        // side exit the decline sentinel below uses.
        //
        // Recording starts only from a pair that already differs, because on
        // a tie the winner is scan order rather than an ordering this arm
        // could guard.  The guard is then recorded in whichever direction
        // held, so the `a < b` trace excludes a later tie while the `a >= b`
        // one admits it.  On such a tie `min_max_multiple_args` keeps its
        // first argument and this arm may hand back the second, but the two
        // are exact ints of equal value and `is_w` compares those by value
        // rather than by pointer, so which one the loop gets back is not
        // observable.
        let has_tagged_int = pyre_object::tagged_int::CAN_BE_TAGGED
            && (pyre_object::tagged_int::is_tagged_int(operands[0])
                || pyre_object::tagged_int::is_tagged_int(operands[1]));
        let int_typeobj = pyre_object::pyobject::get_instantiate(&pyre_object::pyobject::INT_TYPE);
        if !has_tagged_int
            && walker_is_exact_machine_int_concrete(operands[0])
            && walker_is_exact_machine_int_concrete(operands[1])
        {
            let (a_value, b_value) = unsafe {
                (
                    pyre_object::w_int_get_value(operands[0]),
                    pyre_object::w_int_get_value(operands[1]),
                )
            };
            if a_value != b_value {
                let winner = if value == operands[0] {
                    r_args[2]
                } else if value == operands[1] {
                    r_args[3]
                } else {
                    continue;
                };

                walker_guard_fold_callable(ctx, op.pc, r_args[0], concrete_callable)?;
                let (a_op, b_op) = (r_args[2], r_args[3]);
                let int_type_addr = &pyre_object::pyobject::INT_TYPE as *const _ as i64;
                walker_guard_class(ctx, op.pc, a_op, int_type_addr)?;
                walker_guard_class(ctx, op.pc, b_op, int_type_addr)?;
                walker_guard_exact_w_class(ctx, op.pc, a_op, int_typeobj)?;
                walker_guard_exact_w_class(ctx, op.pc, b_op, int_typeobj)?;
                let a_raw = walker_unbox_int_typed(
                    ctx,
                    op.pc,
                    a_op,
                    int_type_addr,
                    crate::descr::int_intval_descr(),
                )?;
                let b_raw = walker_unbox_int_typed(
                    ctx,
                    op.pc,
                    b_op,
                    int_type_addr,
                    crate::descr::int_intval_descr(),
                )?;
                let lt = ctx.trace_ctx.record_op(OpCode::IntLt, &[a_raw, b_raw]);
                ctx.trace_ctx
                    .set_opref_concrete(lt, Value::Int(i64::from(a_value < b_value)));
                let guard_opcode = if a_value < b_value {
                    OpCode::GuardTrue
                } else {
                    OpCode::GuardFalse
                };
                walker_emit_fold_guard_with_snapshot(ctx, op.pc, guard_opcode, &[lt])?;
                write_residual_call_result_to_dst(ctx, op.pc, dst, 'r', winner)?;
                return Ok(Some(()));
            }
        }
        walker_guard_fold_callable(ctx, op.pc, r_args[0], concrete_callable)?;
        let raw = ctx.trace_ctx.call_ref_typed_with_effect(
            raw_fn as *const (),
            &[r_args[2], r_args[3]],
            &[majit_ir::Type::Ref, majit_ir::Type::Ref],
            // `min` / `max` compare two exact scalars and hand back one of
            // their own arguments, so unlike the allocating ref helpers this
            // one really cannot collect -- which drops the gcmap bracket the
            // plain `CannotRaise` constructor would ask every backend for.
            //
            // The helper is a pure function of its pair, so this call could be
            // recorded elidable the way the int and float channels are.  It is
            // not, because it does not pay: what a caller does with the answer
            // is unbox it, and the returned reference's `w_class` is not an
            // immutable field, so the loop re-proves it every iteration before
            // it can read `intval`.  The derived value therefore cannot cross
            // the jump, and the optimizer re-materializes both calls at the
            // end of the body to feed it -- leaving the loop paying the calls
            // it already paid plus the re-check.  Measured on a
            // `min(a, b) + max(a, b)` loop over 8M iterations, interleaved:
            // 0.0515s as a plain call against 0.0541s as an elidable one, the
            // plain call ahead in 8 of 9 rounds.  A `w_class` that can be
            // proved away, or an int channel that answers with the winning
            // value instead of the winning object, is what would change that.
            // It is also the arm for every shape the inline comparison
            // declines: floats, equal values, tagged immediates, and any
            // operand that is not an exact machine int.
            majit_metainterp::CANNOT_RAISE_NO_HEAP_EFFECT_INFO,
        );
        // Concrete before the guard: the guard captures a resume snapshot, and
        // a `raw` with no value yet is recorded into it without one.
        ctx.trace_ctx
            .set_opref_concrete(raw, majit_ir::Value::Ref(majit_ir::GcRef(value as usize)));
        walker_emit_fold_guard_with_snapshot(ctx, op.pc, OpCode::GuardNonnull, &[raw])?;
        write_residual_call_result_to_dst(ctx, op.pc, dst, 'r', raw)?;
        return Ok(Some(()));
    }
    Ok(None)
}

/// `float(x)` on an exact int/float argument: inline the conversion
/// (`W_IntObject.descr_float` → `space.newfloat`, or the identity
/// `float(f) is f` for an exact float) instead of the opaque
/// `bh_call_fn(float_type, NULL, x)` residual, so the result virtualizes.  The
/// callable must be the exact `float` type object; a rebound name or a float
/// subclass (which reboxes rather than returning the argument) declines.  Any
/// non-matching shape falls through to the generic residual (SAFE).
pub(crate) fn try_walker_specialize_float_call<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    code: &[u8],
    op: &DecodedOp,
    r_args: &[OpRef],
    dst: usize,
) -> Result<Option<()>, DispatchError> {
    if r_args.len() != 3 {
        return Ok(None);
    }
    let arg_concretes = read_ref_var_list_concrete(code, op, 1, ctx);
    let (
        ConcreteValue::Ref(concrete_callable),
        ConcreteValue::Ref(null_or_self),
        ConcreteValue::Ref(arg_obj),
    ) = (arg_concretes[0], arg_concretes[1], arg_concretes[2])
    else {
        return Ok(None);
    };
    if concrete_callable.is_null() || !null_or_self.is_null() || arg_obj.is_null() {
        return Ok(None);
    }
    // The callable must be the canonical `float` type object.
    let float_type_obj = pyre_object::pyobject::get_instantiate(&pyre_object::pyobject::FLOAT_TYPE);
    if !std::ptr::eq(concrete_callable, float_type_obj) {
        return Ok(None);
    }
    let (is_int, val) = unsafe {
        if !pyre_object::is_exact_builtin_instance(arg_obj) {
            return Ok(None);
        }
        if pyre_object::is_int(arg_obj) {
            (true, pyre_object::w_int_get_value(arg_obj) as f64)
        } else if pyre_object::is_float(arg_obj) {
            (false, pyre_object::w_float_get_value(arg_obj))
        } else {
            return Ok(None);
        }
    };
    // Authentic boxed result (float() is side-effect-free on int/float).
    let boxed_result = {
        let _plain_guard = pyre_interpreter::call::force_plain_eval();
        pyre_interpreter::call::call_function_impl_result(concrete_callable, &[arg_obj])
    };
    let Ok(boxed_result) = boxed_result else {
        return Ok(None);
    };

    // --- emit the specialized IR (walker-native) ---
    let callable_op = r_args[0];
    if !callable_op.is_constant() {
        let expected = ctx.trace_ctx.const_ref(concrete_callable as i64);
        ctx.trace_ctx
            .record_guard(OpCode::GuardValue, &[callable_op, expected], 0);
        walker_capture_snapshot_for_last_guard(ctx, op.pc)?;
        ctx.trace_ctx
            .heap_cache_mut()
            .replace_box(callable_op, expected);
    }
    let arg_op = r_args[2];
    if is_int {
        // int/bool → CastIntToFloat + inline wrapfloat (no residual call).
        // The coercion pins `w_class`: `builtin_float` reads an int payload only
        // for an exact builtin, so an `int` subclass overriding `__float__` must
        // side-exit.  The float arm below pins its own for the same reason.
        let raw = walker_coerce_operand_to_float(ctx, op.pc, arg_op, arg_obj, true, val, false)?;
        let boxed = crate::state::wrapfloat(ctx.trace_ctx, raw);
        ctx.trace_ctx.set_opref_concrete(
            boxed,
            majit_ir::Value::Ref(majit_ir::GcRef(boxed_result as usize)),
        );
        write_residual_call_result_to_dst(ctx, op.pc, dst, 'r', boxed)?;
    } else {
        // exact float → `float(f) is f`: forward the argument unchanged.  Only
        // sound when the constructor actually returned the same object; a
        // divergence (should not happen for an exact float) declines.
        if !std::ptr::eq(boxed_result, arg_obj) {
            return Ok(None);
        }
        let float_type_addr = &pyre_object::pyobject::FLOAT_TYPE as *const _ as i64;
        if !arg_op.is_constant() && !ctx.trace_ctx.heap_cache().is_class_known(arg_op) {
            let type_const = ctx.trace_ctx.const_int(float_type_addr);
            ctx.trace_ctx
                .record_guard(OpCode::GuardClass, &[arg_op, type_const], 0);
            walker_capture_snapshot_for_last_guard(ctx, op.pc)?;
        }
        ctx.trace_ctx
            .heap_cache_mut()
            .class_now_known(arg_op, float_type_addr);
        walker_guard_exact_w_class(ctx, op.pc, arg_op, float_type_obj)?;
        write_residual_call_result_to_dst(ctx, op.pc, dst, 'r', arg_op)?;
    }
    Ok(Some(()))
}

/// `str(i)` on an exact `int`: emit the guarded unbox plus one elidable
/// `jit_int_str` call instead of the opaque `bh_call_fn(str_type, NULL, i)`
/// residual.  `rint.py rtype_str` / `rstr.py ll_int2dec` lower an unboxed
/// `str(int)` to a `direct_call` of the decimal-render helper, which the
/// optimized trace carries as one `call_r(ll_str__IntegerR_SignedConst_Signed,
/// i, EF=3)` + `guard_no_exception`.  `jtransform` already lowers the
/// graph-level `UnaryOp { op: "str" }` over an Int operand to that same
/// `jit_int_str`; this is the Python-level call site taking the same channel.
///
/// The residual it replaces is a `CallMayForce`, so it also clears the heap
/// cache and forces virtualizables across itself — the reason a `str(i)` loop
/// costs far more than the one allocation it performs.
///
/// The callable must be the canonical `str` type object or the `repr` builtin
/// itself: a rebound name or a `str` subclass reboxes through `__new__`
/// instead. `repr(i)` shares the arm because it renders the same decimal text
/// for an exact `int` — the cross-check below is what holds that, so the two
/// callables need no separate reasoning. The argument must be an exact `int`,
/// because `bool` renders `True`/`False`, an `int` subclass may override
/// `__str__` / `__repr__`, and a `W_LongObject`'s payload is a pointer where
/// `intval` would be. Any other shape falls through to the generic residual
/// (SAFE).
pub(crate) fn try_walker_specialize_str_call<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    code: &[u8],
    op: &DecodedOp,
    r_args: &[OpRef],
    dst: usize,
) -> Result<Option<()>, DispatchError> {
    if r_args.len() != 3 {
        return Ok(None);
    }
    let arg_concretes = read_ref_var_list_concrete(code, op, 1, ctx);
    let (
        ConcreteValue::Ref(concrete_callable),
        ConcreteValue::Ref(null_or_self),
        ConcreteValue::Ref(arg_obj),
    ) = (arg_concretes[0], arg_concretes[1], arg_concretes[2])
    else {
        return Ok(None);
    };
    // A non-null `null_or_self` is a bound receiver `bh_call_fn_impl` prepends
    // as arg0 — not a plain `str(i)` call.
    if concrete_callable.is_null() || !null_or_self.is_null() || arg_obj.is_null() {
        return Ok(None);
    }
    let str_type_obj = pyre_object::pyobject::get_instantiate(&pyre_object::pyobject::STR_TYPE);
    let renders_an_int = std::ptr::eq(concrete_callable, str_type_obj)
        || pyre_interpreter::jit_builtin_folds::is_repr_builtin(concrete_callable);
    if !renders_an_int {
        return Ok(None);
    }
    // A tagged immediate has no header for the `w_class` and unbox guards to
    // read, and this emit is not tag-aware.
    if pyre_object::tagged_int::CAN_BE_TAGGED && pyre_object::tagged_int::is_tagged_int(arg_obj) {
        return Ok(None);
    }
    let int_typeobj = pyre_object::pyobject::get_instantiate(&pyre_object::pyobject::INT_TYPE);
    let int_value = unsafe {
        if !std::ptr::eq((*arg_obj).ob_type, &pyre_object::pyobject::INT_TYPE)
            || !std::ptr::eq((*arg_obj).w_class, int_typeobj)
        {
            return Ok(None);
        }
        pyre_object::w_int_get_value(arg_obj)
    };
    // Authentic boxed result, produced on the plain eval loop exactly as the
    // skipped residual would, then cross-checked against what `jit_int_str`
    // renders.  A disagreement declines rather than compiling itself in.
    //
    // The check reads `int_str_text` rather than calling the helper, because
    // the helper allocates: a nursery collection under it could move
    // `arg_obj` and `boxed_result`, which this frame still holds as raw
    // pointers, and there is no root scope around them.
    let boxed_result = {
        let _plain_guard = pyre_interpreter::call::force_plain_eval();
        pyre_interpreter::call::call_function_impl_result(concrete_callable, &[arg_obj])
    };
    let Ok(boxed_result) = boxed_result else {
        return Ok(None);
    };
    let renders_the_same = unsafe {
        pyre_object::is_exact_type(boxed_result, &pyre_object::STR_TYPE)
            && pyre_object::w_str_get_value_opt(boxed_result)
                == Some(pyre_object::unicodeobject::int_str_text(int_value).as_str())
    };
    if !renders_the_same {
        return Ok(None);
    }

    // --- emit the specialized IR (walker-native) ---
    walker_guard_fold_callable(ctx, op.pc, r_args[0], concrete_callable)?;
    let arg_op = r_args[2];
    let int_type_addr = &pyre_object::pyobject::INT_TYPE as *const _ as i64;
    walker_guard_class(ctx, op.pc, arg_op, int_type_addr)?;
    walker_guard_exact_w_class(ctx, op.pc, arg_op, int_typeobj)?;
    let int_raw = walker_unbox_int_typed(
        ctx,
        op.pc,
        arg_op,
        int_type_addr,
        crate::descr::int_intval_descr(),
    )?;
    let helper = pyre_object::unicodeobject::jit_int_str as *const ();
    let raw = ctx.trace_ctx.call_typed_with_effect(
        OpCode::CallR,
        helper,
        &[int_raw],
        &[majit_ir::Type::Int],
        majit_ir::Type::Ref,
        // `EF_CAN_RAISE`, matching the helper's `#[dont_look_inside]` and NOT
        // an elidable effect.  `descr_repr` (intobject.py) splits the render
        // from the wrapper — `str(self.intval)` is the `@jit.elidable`
        // `ll_int2dec` and `space.newutf8(res, len(res))` is a plain
        // allocation — while this helper performs both in one call.  Recording
        // the pair pure let the pure pass share one call between two `str(i)`
        // sites on the same operand, and `is_w` gives a `str` of `_len() > 1`
        // storage identity, so a compiled loop answered `str(i) is str(i)`
        // True where the interpreter, pypy3 and CPython all answer False.
        // Recovering the elidable half needs the render and the wrapper split
        // into two ops, the shape `emit_box_long_inline` already gives the
        // bigint arms.
        //
        // The read/write sets stay empty: the call allocates and touches no
        // field the trace has cached.
        majit_ir::EffectInfo::const_new(
            majit_ir::ExtraEffect::CanRaise,
            majit_ir::OopSpecIndex::None,
        ),
    );
    // Concrete before the guard: the guard captures a resume snapshot, and a
    // `raw` with no value yet is recorded into it without one.
    ctx.trace_ctx.set_opref_concrete(
        raw,
        majit_ir::Value::Ref(majit_ir::GcRef(boxed_result as usize)),
    );
    walker_emit_guard_with_snapshot(ctx, op.pc, OpCode::GuardNoException, &[])?;
    write_residual_call_result_to_dst(ctx, op.pc, dst, 'r', raw)?;
    Ok(Some(()))
}

/// `divmod(a, b)` on two exact `W_IntObject` operands: emit the inline pair
/// shape the meta-tracer produces upstream (intobject.py `_divmod` →
/// `space.newtuple2(space.newint(z), space.newint(m))`) instead of the opaque
/// `bh_call_fn(divmod_builtin, NULL, a, b)` residual.
///
/// The divmod row rejects a zero divisor before dispatching, so the trace
/// carries the same domain guards the `//` / `%` specialization emits, then
/// runs the two `OS_INT_PY_DIV` / `OS_INT_PY_MOD` elidable calls over one
/// guarded operand pair.  The result is the virtual `Cls_ii` specialised
/// tuple, so a `q, r = divmod(...)` site pairs with
/// [`try_walker_specialize_unpack`] and the tuple never materializes.
///
/// The exact `w_class` guard is required because an `int` SUBCLASS shares
/// `ob_type == &INT_TYPE` but may override `__divmod__`; it side-exits to the
/// generic residual.
///
/// Returns `None` (fall through to the generic residual, SAFE) for any other
/// shape: wrong arity, a bound receiver, a non-`divmod` callable, an operand
/// that is not an exact `int` (long / float / bool / subclass), a tagged
/// immediate, a zero divisor, or the `INT_MIN // -1` pair that escapes to a
/// bigint result.
pub(crate) fn try_walker_specialize_builtin_divmod<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    code: &[u8],
    op: &DecodedOp,
    r_args: &[OpRef],
    dst: usize,
) -> Result<Option<()>, DispatchError> {
    // Plain `bh_call_fn(callable, PY_NULL, a, b)` shape only.
    if r_args.len() != 4 {
        return Ok(None);
    }
    let arg_concretes = read_ref_var_list_concrete(code, op, 1, ctx);
    let (
        ConcreteValue::Ref(concrete_callable),
        ConcreteValue::Ref(null_or_self),
        ConcreteValue::Ref(lhs_obj),
        ConcreteValue::Ref(rhs_obj),
    ) = (
        arg_concretes[0],
        arg_concretes[1],
        arg_concretes[2],
        arg_concretes[3],
    )
    else {
        return Ok(None);
    };
    // A non-null `null_or_self` is a bound receiver `bh_call_fn_impl`
    // prepends as arg0 — not a plain `divmod(a, b)` call.
    if concrete_callable.is_null()
        || !null_or_self.is_null()
        || lhs_obj.is_null()
        || rhs_obj.is_null()
    {
        return Ok(None);
    }
    if !pyre_interpreter::builtins::is_builtin_divmod_function(concrete_callable) {
        return Ok(None);
    }
    // A tagged immediate has no real header for the `w_class` / unbox guards
    // and this emit is not tag-aware, so decline it to the residual.
    if pyre_object::tagged_int::CAN_BE_TAGGED
        && (pyre_object::tagged_int::is_tagged_int(lhs_obj)
            || pyre_object::tagged_int::is_tagged_int(rhs_obj))
    {
        return Ok(None);
    }
    let int_typeobj = pyre_object::pyobject::get_instantiate(&pyre_object::pyobject::INT_TYPE);
    let is_exact_int = |o: pyre_object::PyObjectRef| unsafe {
        std::ptr::eq((*o).ob_type, &pyre_object::pyobject::INT_TYPE)
            && std::ptr::eq((*o).w_class, int_typeobj)
    };
    if !is_exact_int(lhs_obj) || !is_exact_int(rhs_obj) {
        // `_make_descr_binop(_divmod, _int_divmod)` (longobject.py) keeps a
        // dedicated long/int arm; every other operand shape stays generic.
        return spec_gate(SpecFold::BuiltinDivmodLongInt, || {
            try_walker_specialize_builtin_divmod_long_int(
                ctx,
                op,
                r_args,
                dst,
                concrete_callable,
                lhs_obj,
                rhs_obj,
            )
        });
    }
    let (la, rb) = unsafe {
        (
            pyre_object::w_int_get_value(lhs_obj),
            pyre_object::w_int_get_value(rhs_obj),
        )
    };
    // A zero divisor raises ZeroDivisionError and `INT_MIN // -1` escapes to
    // the bigint pair; both are outside the guarded domain the emit below
    // covers, so decline rather than record a guard the recorded operands
    // already fail.
    if rb == 0 || (la == i64::MIN && rb == -1) {
        return Ok(None);
    }

    // --- emit the specialized IR (walker-native) ---
    walker_guard_builtin_callable_identity(ctx, op.pc, r_args[0], concrete_callable)?;
    let (lhs_op, rhs_op) = (r_args[2], r_args[3]);
    let int_type_addr = &pyre_object::pyobject::INT_TYPE as *const _ as i64;
    // `GuardClass` before the `w_class` read: it is the guard that proves the
    // operand is a real heap header rather than a tagged immediate, so it has
    // to precede any `getfield` off that header.
    walker_guard_class(ctx, op.pc, lhs_op, int_type_addr)?;
    walker_guard_class(ctx, op.pc, rhs_op, int_type_addr)?;
    walker_guard_exact_w_class(ctx, op.pc, lhs_op, int_typeobj)?;
    walker_guard_exact_w_class(ctx, op.pc, rhs_op, int_typeobj)?;
    let lhs_raw = walker_unbox_int_typed(
        ctx,
        op.pc,
        lhs_op,
        int_type_addr,
        crate::descr::int_intval_descr(),
    )?;
    let rhs_raw = walker_unbox_int_typed(
        ctx,
        op.pc,
        rhs_op,
        int_type_addr,
        crate::descr::int_intval_descr(),
    )?;
    walker_emit_int_div_domain_guards(ctx, op.pc, lhs_raw, rhs_raw, la, rb)?;
    let (div_raw, div_value) = walker_emit_int_py_div_or_mod(ctx, lhs_raw, rhs_raw, la, rb, true);
    let (mod_raw, mod_value) = walker_emit_int_py_div_or_mod(ctx, lhs_raw, rhs_raw, la, rb, false);
    let tuple =
        walker_emit_specialised_tuple_ii(ctx, op.pc, div_raw, mod_raw, div_value, mod_value)?;
    write_residual_call_result_to_dst(ctx, op.pc, dst, 'r', tuple)?;
    Ok(Some(()))
}

/// One element of a concrete `W_SpecialisedTupleObject_oo`, or `None` when the
/// value is any other tuple layout. `newtuple` picks the representation, so a
/// fold that emits the object-pair shape has to confirm the record-time value
/// took the same one before reading it through those offsets.
fn walker_specialised_tuple_oo_item(
    tuple: pyre_object::PyObjectRef,
    index: usize,
) -> Option<pyre_object::PyObjectRef> {
    if tuple.is_null() {
        return None;
    }
    let ob_type = unsafe { (*(tuple as *const pyre_object::pyobject::PyObject)).ob_type };
    if !std::ptr::eq(
        ob_type,
        &pyre_object::specialisedtupleobject::SPECIALISED_TUPLE_OO_TYPE
            as *const pyre_object::pyobject::PyType,
    ) {
        return None;
    }
    let item = unsafe {
        pyre_object::specialisedtupleobject::w_specialised_tuple_oo_getvalue(tuple, index)
    };
    (!item.is_null()).then_some(item)
}

/// Pin a builtin's identity before folding its call away. `LOAD_GLOBAL divmod`
/// is usually already a constant via the namespace cell fold, in which case the
/// guard is unnecessary; a rebound global takes the side exit.
fn walker_guard_builtin_callable_identity<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    callable_op: OpRef,
    concrete_callable: pyre_object::PyObjectRef,
) -> Result<(), DispatchError> {
    if callable_op.is_constant() {
        return Ok(());
    }
    let expected = ctx.trace_ctx.const_ref(concrete_callable as i64);
    ctx.trace_ctx
        .record_guard(OpCode::GuardValue, &[callable_op, expected], 0);
    walker_capture_snapshot_for_last_guard(ctx, op_pc)?;
    ctx.trace_ctx
        .heap_cache_mut()
        .replace_box(callable_op, expected);
    Ok(())
}

/// `divmod(W_LongObject, W_IntObject)` — `longobject.py _int_divmod`.
///
/// One `rbigint.int_divmod` (rbigint.py `@jit.elidable`) produces both
/// halves, so the trace is the upstream shape: a single `CallR` returning the
/// RPython `tuple2`, two `GetfieldGcR` off it, then `newlong` ×2 and the
/// arity-2 tuple — all three allocations trace-visible, so the shipped oo
/// unpack fold can virtualize the tuple away at an unpacking use.
///
/// Emitting `int_div_floor` and `int_mod_int_result` instead would run the
/// division twice; `_int_divmod` exists precisely to avoid that.
///
/// `int % long` and `divmod(int, long)` are `descr_rdivmod`, which coerces the
/// left operand and takes the bigint/bigint path — not this arm.
fn try_walker_specialize_builtin_divmod_long_int<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op: &DecodedOp,
    r_args: &[OpRef],
    dst: usize,
    concrete_callable: pyre_object::PyObjectRef,
    long_obj: pyre_object::PyObjectRef,
    int_obj: pyre_object::PyObjectRef,
) -> Result<Option<()>, DispatchError> {
    let (long_class, int_class, int_value) = unsafe {
        if !pyre_object::is_long(long_obj) || !pyre_object::is_int(int_obj) {
            return Ok(None);
        }
        let (Some(long_class), Some(int_class)) = (
            walker_exact_builtin_class(long_obj),
            walker_exact_builtin_class(int_obj),
        ) else {
            return Ok(None);
        };
        (long_class, int_class, pyre_object::w_int_get_value(int_obj))
    };
    // A zero divisor raises before reaching rbigint; the interpreter owns the
    // authentic message.
    if int_value == 0 {
        return Ok(None);
    }

    // Take the concretes from the authentic builtin, not from the residual:
    // the residual's payload allocator collects, and it is only safe to do so
    // under a gcmap-carrying `CallR`, which the host-side walker is not. This
    // is the `math.isqrt` fold's route, and it is also what makes the recorded
    // tuple the one `newtuple` actually picked for these two operands.
    let tuple_concrete = {
        let _plain_guard = pyre_interpreter::call::force_plain_eval();
        pyre_interpreter::call::call_function_impl_result(concrete_callable, &[long_obj, int_obj])
    };
    let Ok(tuple_concrete) = tuple_concrete else {
        return Ok(None);
    };
    // `_int_divmod` boxes both halves with `newlong`, which does not demote, so
    // a pair of longs is the only shape this arm may emit — and two longs are
    // what routes `newtuple` to the object-pair variant.
    let (Some(w_div), Some(w_mod)) = (
        walker_specialised_tuple_oo_item(tuple_concrete, 0),
        walker_specialised_tuple_oo_item(tuple_concrete, 1),
    ) else {
        return Ok(None);
    };
    if !unsafe { pyre_object::is_long(w_div) && pyre_object::is_long(w_mod) } {
        return Ok(None);
    }
    // The call above allocates, so the operand pointers this function was
    // handed may have been forwarded. Re-fetch them from the walker's op cells,
    // which the collector does update, before reading anything through them.
    let (long_op, int_op) = (r_args[2], r_args[3]);
    let (Some(long_obj), Some(int_obj)) = (
        walker_concrete_ref_object(ctx, long_op),
        walker_concrete_ref_object(ctx, int_op),
    ) else {
        return Ok(None);
    };
    let read_payload = |o: pyre_object::PyObjectRef| unsafe {
        *((o as *const u8).add(pyre_object::longobject::LONG_VALUE_OFFSET) as *const i64)
            as *mut majit_rlib::rbigint::RBigInt
    };
    let long_payload = read_payload(long_obj) as i64;
    let (div_payload, mod_payload) = (read_payload(w_div), read_payload(w_mod));
    // Both halves are already reachable from `tuple_concrete`, and this
    // allocation cannot collect, so neither can move under it.
    let pair = pyre_object::longobject::alloc_bigint_pair_no_collect(div_payload, mod_payload);
    if pair.is_null() {
        return Err(DispatchError::ConcreteShadowAllocationFailed { pc: op.pc });
    }

    // --- emit ---
    walker_guard_builtin_callable_identity(ctx, op.pc, r_args[0], concrete_callable)?;
    let long_type_addr = &pyre_object::pyobject::LONG_TYPE as *const _ as i64;
    walker_guard_class(ctx, op.pc, long_op, long_type_addr)?;
    walker_guard_exact_w_class(ctx, op.pc, long_op, long_class)?;
    let (int_type, int_descr) = crate::state::int_or_bool_unbox_type_descr(int_obj);
    let int_raw = walker_unbox_int_typed(ctx, op.pc, int_op, int_type, int_descr)?;
    walker_guard_exact_w_class(ctx, op.pc, int_op, int_class)?;
    let zero = ctx.trace_ctx.const_int(0);
    let nonzero = ctx.trace_ctx.record_op(OpCode::IntNe, &[int_raw, zero]);
    ctx.trace_ctx
        .set_opref_concrete(nonzero, majit_ir::Value::Int(1));
    walker_emit_guard_with_snapshot(ctx, op.pc, OpCode::GuardTrue, &[nonzero])?;

    let long_pl = ctx.trace_ctx.record_op_with_descr(
        OpCode::GetfieldGcR,
        &[long_op],
        crate::descr::long_value_descr(),
    );
    ctx.trace_ctx.set_opref_concrete(
        long_pl,
        majit_ir::Value::Ref(majit_ir::GcRef(long_payload as usize)),
    );

    let helper = pyre_interpreter::objspace::descroperation::jit_bigint_int_divmod as *const ();
    let pair_op = ctx.trace_ctx.call_typed_with_effect_pure_can_raise(
        OpCode::CallR,
        helper,
        &[long_pl, int_raw],
        &[majit_ir::Type::Ref, majit_ir::Type::Int],
        majit_ir::Type::Ref,
        majit_metainterp::ELIDABLE_OR_MEMERROR_EFFECT_INFO,
        &[
            majit_ir::Value::Int(helper as usize as i64),
            majit_ir::Value::Ref(majit_ir::GcRef(long_payload as usize)),
            majit_ir::Value::Int(int_value),
        ],
        majit_ir::Value::Ref(majit_ir::GcRef(pair as usize)),
    );
    ctx.trace_ctx.set_opref_concrete(
        pair_op,
        majit_ir::Value::Ref(majit_ir::GcRef(pair as usize)),
    );
    if pair_op.inline_const_to_value().is_none() {
        walker_emit_guard_with_snapshot(ctx, op.pc, OpCode::GuardNoException, &[])?;
    }

    let mut boxed = Vec::with_capacity(2);
    for (descr, payload, wrapper) in [
        (crate::descr::rbigint_pair_item0_descr(), div_payload, w_div),
        (crate::descr::rbigint_pair_item1_descr(), mod_payload, w_mod),
    ] {
        let half = ctx
            .trace_ctx
            .record_op_with_descr(OpCode::GetfieldGcR, &[pair_op], descr);
        ctx.trace_ctx.set_opref_concrete(
            half,
            majit_ir::Value::Ref(majit_ir::GcRef(payload as usize)),
        );
        let w = crate::helpers::emit_box_long_inline(
            ctx.trace_ctx,
            half,
            crate::descr::w_long_size_descr(),
            crate::descr::long_value_descr(),
        );
        ctx.trace_ctx
            .set_opref_concrete(w, majit_ir::Value::Ref(majit_ir::GcRef(wrapper as usize)));
        boxed.push(w);
    }

    let tuple = crate::helpers::emit_specialised_tuple_oo_inline(ctx.trace_ctx, boxed[0], boxed[1]);
    ctx.trace_ctx.set_opref_concrete(
        tuple,
        majit_ir::Value::Ref(majit_ir::GcRef(tuple_concrete as usize)),
    );
    write_residual_call_result_to_dst(ctx, op.pc, dst, 'r', tuple)?;
    Ok(Some(()))
}

/// #171 ORTHODOX descent of the real `w_list_append` charon body (WIP).
///
/// Instead of hand-rolling the int-storage append IR (the fold below), walk
/// the compiled `w_list_append` jitcode (`list_append_jitcode()`): its
/// strategy `switch` folds to `guard_value(strategy==Integer)` over the
/// concrete receiver, the `is_plain_int1` / `plain_int_w` leaves recurse via
/// `inline_call`, the `ll_list_int_*` leaves are oopspec-lowered to
/// getfield/setfield/setarrayitem, and the capacity `goto_if_not` guards the
/// spare-capacity fast path.
///
/// The sub-walk's guards must resume at the `lst.append` CALL site (re-execute
/// the append generically on deopt — any of strategy / plain-int / capacity
/// failing).  The inline-subwalk capture (`walker_capture_snapshot_for_last_
/// guard_impl` single-frame fallthrough) reads `ctx.{outer_active_boxes,
/// outer_jitcode_index,entry_py_pc}` + the vable shadow directly, so this
/// pre-publishes that ONE call-site coordinate (mapped from `op.pc`) before
/// the sub-walk with inline-subwalk mode enabled.
///
/// Like the fold, the walker only RECORDS the array-op IR; this applies the
/// append to the concrete list + journals the rewind. Recognition declines
/// before emitting IR; an unsupported body sub-walk rolls its tentative IR
/// back before falling through to the residual call.
///
/// STATUS: the descr-pool
/// wiring, the host-static const relocation, and the list header field
/// descr-group bridge (`make_descr_from_bh` strategy/length/items →
/// `W_LIST_DESCR_GROUP`) are all in place — the strategy `switch` and the
/// inlined `is_int`/`is_bool` type predicates fold over the concrete receiver,
/// the `W_ListObject.strategy` read resolves a parent_descr, and the walk
/// descends the full append into the Integer fast-path.  The unit-`()` return
/// aggregate (`SyntheticTransparentCtor "Tuple"`) is elided to `ConstRefNull`
/// at build time (`jtransform.rs`), so the descent completes and commits a
/// working trace.  Safety net: if a stale build-time jitcode kept that ctor as
/// a symbolic (tagged) fnaddr, `try_execute_residual_call_via_executor`
/// declines it (`OrthodoxSubWalkTraceUnsupported`) and the method-call form
/// records the append as a residual call instead of baking the hash as a code
/// address and branching to garbage.
pub(crate) fn try_walker_orthodox_list_append<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    code: &[u8],
    op: &DecodedOp,
    r_args: &[OpRef],
    dst: usize,
) -> Result<Option<()>, DispatchError> {
    if r_args.len() != 3 {
        return Ok(None);
    }
    let arg_concretes = read_ref_var_list_concrete(code, op, 1, ctx);
    let (ConcreteValue::Ref(callable), ConcreteValue::Ref(null_or_self), ConcreteValue::Ref(value)) =
        (arg_concretes[0], arg_concretes[1], arg_concretes[2])
    else {
        return Ok(None);
    };
    if callable.is_null() || !null_or_self.is_null() || value.is_null() {
        return Ok(None);
    }

    // Recognition: the callable must be the bound builtin `list.append`; the
    // receiver + value then pass the shared storage/spare-capacity gate.
    let (inner_func, inner_self, len_before) = unsafe {
        if !pyre_object::function::is_method(callable) {
            return Ok(None);
        }
        let inner_func = pyre_object::function::w_method_get_func(callable);
        let inner_self = pyre_object::function::w_method_get_self(callable);
        if inner_func.is_null() || inner_self.is_null() {
            return Ok(None);
        }
        let list_type = pyre_interpreter::typedef::gettypeobject(&pyre_object::pyobject::LIST_TYPE);
        if pyre_interpreter::lookup_in_type(list_type, "append") != Some(inner_func) {
            return Ok(None);
        }
        let Some(len_before) = orthodox_list_append_recognize(inner_self, value) else {
            return Ok(None);
        };
        (inner_func, inner_self, len_before)
    };

    // Resolve the compiled `w_list_append` body + the full-body sym (the
    // resume-coordinate source) BEFORE emitting any guard — a decline must
    // leave the trace untouched.
    let Some((sub_body, sym_ptr)) = orthodox_list_append_body_and_sym(ctx) else {
        return Ok(None);
    };
    // SAFETY: `sym_ptr` is non-null with a set `jitcode` (checked in the
    // resolver) and stays live for the enclosing full-body walk.
    let sym = unsafe { &*sym_ptr };

    let pre_fold_pos = ctx.trace_ctx.get_trace_position();
    // Mirror the commit's own promotion predicate exactly: a rollback must
    // undo a promotion only when one was performed, or it would pop another
    // list's journal entry.
    let promoted_empty = unsafe { pyre_object::w_list_uses_empty_storage(inner_self) };

    // ── tentative commit ──
    let callable_op = r_args[0];
    let value_op = r_args[2];

    // Pin the callable to `list.append`: guard_class METHOD + guard_value on
    // the stable function slot (these guards resume via the full-body path at
    // `op.pc`, ignoring the call-site fields set below).
    let method_type_addr = &pyre_object::function::METHOD_TYPE as *const _ as i64;
    if !callable_op.is_constant() && !ctx.trace_ctx.heap_cache().is_class_known(callable_op) {
        let type_const = ctx.trace_ctx.const_int(method_type_addr);
        ctx.trace_ctx
            .record_guard(OpCode::GuardClass, &[callable_op, type_const], 0);
        walker_capture_snapshot_for_last_guard(ctx, op.pc)?;
    }
    ctx.trace_ctx
        .heap_cache_mut()
        .class_now_known(callable_op, method_type_addr);
    let func_ref = crate::state::opimpl_getfield_gc_r(
        ctx.trace_ctx,
        callable_op,
        crate::descr::method_w_function_descr(),
    );
    let func_const = ctx.trace_ctx.const_ref(inner_func as i64);
    ctx.trace_ctx
        .record_guard(OpCode::GuardValue, &[func_ref, func_const], 0);
    walker_capture_snapshot_for_last_guard(ctx, op.pc)?;
    ctx.trace_ctx
        .heap_cache_mut()
        .replace_box(func_ref, func_const);

    // Recover the receiver list OpRef; the sub-walk reads it as ref-arg 0.
    let self_ref = crate::state::opimpl_getfield_gc_r(
        ctx.trace_ctx,
        callable_op,
        crate::descr::method_w_self_descr(),
    );

    let commit_result = orthodox_list_append_commit(
        ctx, op, sym, &sub_body, self_ref, value_op, inner_self, value, len_before,
    );
    match commit_result {
        Ok(()) => {}
        Err(DispatchError::OrthodoxSubWalkTraceUnsupported { pc, .. }) => {
            if fbw_debug_abort_enabled() {
                eprintln!("[decline-why] LIST-APPEND-SUBWALK pc={pc}");
            }
            ctx.trace_ctx.cut_trace(pre_fold_pos);
            ctx.trace_ctx.heap_cache_mut().reset();
            if promoted_empty {
                fbw_append_promote_journal_rollback_last(inner_self);
            }
            return Ok(None);
        }
        Err(error) => return Err(error),
    }

    // The `list.append(x)` call's `None` return (the residual's Ref dst).
    let none_ref = ctx.trace_ctx.const_ref(pyre_object::w_none() as i64);
    write_residual_call_result_to_dst(ctx, op.pc, dst, 'r', none_ref)?;
    Ok(Some(()))
}

/// The substituted residual [`try_walker_specialize_set_add_method`] hands
/// back: the funcbox, the minted `(Ref, Ref) -> Ref` MayForce call descr, and
/// the `[funcbox, receiver, value]` arglist.  The generic residual path
/// records and executes it exactly as it would the call it replaces, so the
/// force/exception guards, the heapcache invalidation and the result
/// writeback all stay where they are.
pub(crate) struct SetAddDirectResidual {
    pub(crate) funcptr: OpRef,
    pub(crate) descr: DescrRef,
    pub(crate) allboxes: Vec<OpRef>,
}

/// `s.add(x)`: record the direct `set_add` residual the SET_ADD accumulator
/// opcode records, in place of the generic `bh_call_fn` dispatch the
/// bound-method spelling otherwise leaves behind.
///
/// `pyopcode.py SET_ADD` is `space.call_method(w_set, 'add', w_value)`, so the
/// two spellings name one operation.  The codewriter lowers the opcode to a
/// `set_add` residual (`bh_set_add_fn`), while the method call reaches the
/// same store through `bh_call_fn`, which re-reads the bound method's
/// function, rejects keywords and rebuilds the argument vector on every
/// iteration.  This arm pins the callable to the `set.add` builtin and the
/// receiver to an exact `set`, then hands `(receiver, value)` to
/// [`pyre_interpreter::runtime_ops::jit_set_add_method`] — the same
/// `set_add_value` store, entered directly.  Measured on a 3M-iteration
/// `s.add(i & 3)` loop, that is the whole difference between the method-call
/// form and the comprehension: 0.220s -> 0.130s against `list.append`'s
/// 0.040s.
///
/// The insert stays a MayForce residual, and deliberately so: `set_add_value`
/// hashes the element, which can run a user `__hash__`.  That is the other
/// half of the gap against `list.append` (`GuardNotForced`, which even the
/// dispatch-free comprehension carries), and this arm does not claim it.
///
/// Recognition declines before emitting IR, and what it admits is deliberately
/// the builtin's own predicate: `require_set_receiver` is `is_set`, an
/// `ob_type == &SET_TYPE` layout test, so a `set` subclass that inherits `add`
/// passes both it and the class guard below and is substituted — the builtin
/// would have run this identical body.  A subclass that OVERRIDES `add` is
/// excluded instead by the `GuardValue` pinning the bound function, and a
/// frozenset receiver by the layout guard, which matters because
/// [`pyre_interpreter::opcode_ops::set_add_value`] itself accepts
/// `is_set_or_frozenset` and would mutate one.  Anything else falls through to
/// the generic residual, which still runs the builtin's receiver and arity
/// checks.
pub(crate) fn try_walker_specialize_set_add_method<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    code: &[u8],
    op: &DecodedOp,
    r_args: &[OpRef],
) -> Result<Option<SetAddDirectResidual>, DispatchError> {
    if r_args.len() != 3 {
        return Ok(None);
    }
    let arg_concretes = read_ref_var_list_concrete(code, op, 1, ctx);
    let (ConcreteValue::Ref(callable), ConcreteValue::Ref(null_or_self), ConcreteValue::Ref(value)) =
        (arg_concretes[0], arg_concretes[1], arg_concretes[2])
    else {
        return Ok(None);
    };
    if callable.is_null() || !null_or_self.is_null() || value.is_null() {
        return Ok(None);
    }

    // Recognition: the callable must be the bound builtin `set.add`, over the
    // receiver `require_set_receiver` accepts.  `is_set` is that predicate
    // verbatim, and the class guard below is emitted in the same spelling, so
    // recognition and guard admit the same set of receivers.  It excludes a
    // frozenset, which `set_add_value` would otherwise mutate.
    let inner_func = unsafe {
        if !pyre_object::function::is_method(callable) {
            return Ok(None);
        }
        let inner_func = pyre_object::function::w_method_get_func(callable);
        let inner_self = pyre_object::function::w_method_get_self(callable);
        if inner_func.is_null() || !pyre_object::setobject::is_set(inner_self) {
            return Ok(None);
        }
        let set_type = pyre_interpreter::typedef::gettypeobject(&pyre_object::setobject::SET_TYPE);
        if pyre_interpreter::lookup_in_type(set_type, "add") != Some(inner_func) {
            return Ok(None);
        }
        inner_func
    };

    // ── tentative commit ──
    // Pin the callable to `set.add`: guard_class METHOD + guard_value on the
    // stable function slot, both resuming at the call site so a deopt
    // re-executes the call generically.
    let callable_op = r_args[0];
    let method_type_addr = &pyre_object::function::METHOD_TYPE as *const _ as i64;
    if !callable_op.is_constant() && !ctx.trace_ctx.heap_cache().is_class_known(callable_op) {
        let type_const = ctx.trace_ctx.const_int(method_type_addr);
        ctx.trace_ctx
            .record_guard(OpCode::GuardClass, &[callable_op, type_const], 0);
        walker_capture_snapshot_for_last_guard(ctx, op.pc)?;
    }
    ctx.trace_ctx
        .heap_cache_mut()
        .class_now_known(callable_op, method_type_addr);
    let func_ref = crate::state::opimpl_getfield_gc_r(
        ctx.trace_ctx,
        callable_op,
        crate::descr::method_w_function_descr(),
    );
    let func_const = ctx.trace_ctx.const_ref(inner_func as i64);
    ctx.trace_ctx
        .record_guard(OpCode::GuardValue, &[func_ref, func_const], 0);
    walker_capture_snapshot_for_last_guard(ctx, op.pc)?;
    ctx.trace_ctx
        .heap_cache_mut()
        .replace_box(func_ref, func_const);

    // The receiver the substituted call takes is the bound method's `w_self`.
    let self_ref = crate::state::opimpl_getfield_gc_r(
        ctx.trace_ctx,
        callable_op,
        crate::descr::method_w_self_descr(),
    );
    let set_type_addr = &pyre_object::setobject::SET_TYPE as *const _ as i64;
    if !self_ref.is_constant() && !ctx.trace_ctx.heap_cache().is_class_known(self_ref) {
        let type_const = ctx.trace_ctx.const_int(set_type_addr);
        ctx.trace_ctx
            .record_guard(OpCode::GuardClass, &[self_ref, type_const], 0);
        walker_capture_snapshot_for_last_guard(ctx, op.pc)?;
    }
    ctx.trace_ctx
        .heap_cache_mut()
        .class_now_known(self_ref, set_type_addr);

    let funcptr = ctx
        .trace_ctx
        .const_int(pyre_interpreter::runtime_ops::jit_set_add_method as *const () as i64);
    Ok(Some(SetAddDirectResidual {
        funcptr,
        descr: set_add_method_descr(),
        allboxes: vec![funcptr, self_ref, r_args[2]],
    }))
}

/// The descr the `s.add(x)` substitution installs: `(Ref, Ref) -> Ref`,
/// `MOST_GENERAL`, tagged [`majit_ir::RuntimeHelperKind::SetAddMethod`].
///
/// The EI `bind(..., CallFlavor::MayForce)` gives the SET_ADD residual this one
/// stands in for: `EffectInfo::MOST_GENERAL`, not the analyzer-empty forcing
/// shape.  Hashing the element runs arbitrary Python, so no write set was ever
/// computed for it and an empty one would assert something false
/// (`effect_info_for_call_flavor`).
///
/// `SetAddMethod` on top of it: the call inserts into the live set and returns
/// `None`, so the `Void`-result write proxy in `writes_live_heap` misses it and
/// the helper tag is all that discriminator has left to read.  The generic
/// `bh_call_fn` this stands in for was counted through its own `CallFn` tag;
/// dropping to an untagged descr would take a completed insert out of the
/// executed-effect odometer, which is what a nested abort consults before
/// rewinding.  Built here rather than inline so that invariant is testable.
pub(crate) fn set_add_method_descr() -> DescrRef {
    majit_metainterp::make_call_descr_with_effect(
        &[Type::Ref, Type::Ref],
        Type::Ref,
        majit_ir::EffectInfo {
            runtime_helper: majit_ir::RuntimeHelperKind::SetAddMethod,
            ..default_effect_info()
        },
    )
}

/// Shared recognition for the #171 orthodox list-append fold: the receiver
/// must be a list with spare capacity whose storage strategy matches the
/// value's strict type predicate (Integer / Object / Float).  Returns the
/// list length before the append (the journal rewind point) on a match, or
/// `None` (decline) otherwise.  No IR is emitted.
///
/// # Safety
/// `inner_self` / `value` must be live `PyObjectRef`s.
unsafe fn orthodox_list_append_recognize(
    inner_self: pyre_object::PyObjectRef,
    value: pyre_object::PyObjectRef,
) -> Option<usize> {
    // `is_plain_int1` accepts an exact `W_IntObject` or a fits-int
    // `W_LongObject`; both route to Integer storage. The commit path pins
    // `guard_class(value, LONG_TYPE)` for a long value (vs `INT_TYPE` for an
    // int) so the descended `w_list_append` body observes the right `ob_type`.
    // The body's `is_plain_int1(value)` / `plain_int_w(value)` then unbox the
    // long through the compiled `_fits_int` / `toint` path; when that path
    // reaches a helper the sub-walk cannot lower it declines
    // (`OrthodoxSubWalkTraceUnsupported`) and rolls back to the generic
    // residual (correctness-safe for any element).
    if !pyre_object::pyobject::is_list(inner_self) {
        return None;
    }
    // Empty-strategy first-append promotion. `w_list_can_append_without_realloc`
    // is false for Empty (no backing block yet), so classify by the value's
    // type using switch_to_correct_strategy's int -> float -> object order
    // (listobject.py) and let the commit path install the typed storage.
    if pyre_object::w_list_uses_empty_storage(inner_self) {
        let int_ok = pyre_object::is_plain_int1(value)
            && !(pyre_object::tagged_int::CAN_BE_TAGGED
                && pyre_object::tagged_int::is_tagged_int(value));
        // NaNs select Object storage to preserve identity.
        let float_ok = pyre_object::is_float_strategy_item(value);
        // switch_to_correct_strategy routes `is_plain_int1` (exact int or
        // fits-in-word long) -> Integer with no tagged exclusion. Exclude any
        // plain-int / float from the object fallback so a tagged-int DECLINES
        // (generic residual) instead of mis-routing to Object and diverging the
        // traced strategy from the concrete one the commit installs.
        let obj_ok = !value.is_null()
            && !pyre_object::is_plain_int1(value)
            && !pyre_object::is_float_strategy_item(value)
            && !pyre_object::is_bytes_strategy_item(value)
            && !pyre_object::is_ascii_strategy_item(value);
        if !int_ok && !float_ok && !obj_ok {
            return None;
        }
        // Empty length is 0 (the journal rewind point).
        return Some(0);
    }
    if !pyre_object::w_list_can_append_without_realloc(inner_self) {
        return None;
    }
    // Int-storage specialization: `is_plain_int1` value (exact `W_IntObject`
    // or fits-int `W_LongObject`) stored unboxed. A tagged-immediate value
    // would need a tag-aware unboxed store and no `w_class` pin; decline to
    // the generic residual append instead.
    let int_ok = pyre_object::w_list_uses_int_storage(inner_self)
        && pyre_object::is_plain_int1(value)
        && !(pyre_object::tagged_int::CAN_BE_TAGGED
            && pyre_object::tagged_int::is_tagged_int(value));
    // Object-storage extension: any non-null `Ref` value stored into the
    // object items block — no unboxing, so the value carries no type
    // precondition.
    let obj_ok = pyre_object::w_list_uses_object_storage(inner_self) && !value.is_null();
    // Match `FloatListStrategy.is_correct_type`; NaNs take the residual path
    // that converts the receiver to Object storage.
    let float_ok = pyre_object::w_list_uses_float_storage(inner_self)
        && pyre_object::is_float_strategy_item(value);
    if !int_ok && !obj_ok && !float_ok {
        return None;
    }
    Some(pyre_object::w_list_len(inner_self))
}

/// Resolve the compiled `w_list_append` body + the full-body snapshot sym
/// (the resume-coordinate source) shared by both list-append fold forms.
/// Returns `None` (decline — no IR emitted yet) when the body jitcode is not
/// compiled or the snapshot sym is absent.  The returned `sym_ptr` is
/// non-null with a set `jitcode` field.  Word size does not enter here: the
/// `d` operands resolve through a descr pool built from the target's own
/// Charon layouts (`jitcode_runtime::build_time_field_offset`), whose array
/// descrs place the items at the first element-aligned offset past the length
/// word — the offset a 4-byte word and an 8-byte item disagree on.
pub(crate) fn orthodox_list_append_body_and_sym<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
) -> Option<(SubJitCodeBody, *const Sym)> {
    let jc_arc = crate::jitcode_runtime::list_append_jitcode()?;
    let sub_body = sub_jitcode_body_by_index(jc_arc.index())?;
    let sym_ptr = ctx.fbw_mode.snapshot_sym;
    if sym_ptr.is_null() {
        return None;
    }
    // SAFETY: set for the lifetime of the enclosing full-body walk.
    if unsafe { (&*sym_ptr).jitcode().is_null() } {
        return None;
    }
    Some((sub_body, sym_ptr))
}

/// Enter a canonical helper body as a sub-jitcode walk from a walker fold.
///
/// Publishes the call-site resume coordinate the enclosing full-body walk needs
/// to rebuild this frame, mirrors the virtualizable's `last_instr` /
/// `valuestackdepth` when the enclosing frame owns the shadow, swaps in the
/// callee's GLOBAL descr pool for the duration, runs the walk, then restores
/// every field it moved.  A build-time canonical body carries no per-fn descr
/// pool, so its `d` / `j` operands resolve through `all_descr_refs()` /
/// `RawDescrPool::Global` -- not the parent loop's per-fn pool, which
/// mis-resolves the first `residual_call` descr.
///
/// Returns the walk outcome together with the trace position taken immediately
/// before the walk, for a caller that has to reason about which ops the callee
/// contributed.  The position is captured after the resume mirroring above, so
/// it names the callee's first op and not the mirror's.
///
/// `fallback_label` names this site in the empty-twin coordinate note;
/// `call_site_label` names it in the active-box collection.
#[allow(clippy::too_many_arguments)]
fn run_orthodox_helper_subwalk<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    sym: &Sym,
    sub_body: &SubJitCodeBody,
    fallback_label: &'static str,
    call_site_label: &'static str,
    int_args: &[OpRef],
    int_arg_concretes: &[ConcreteValue],
    ref_args: &[OpRef],
    ref_arg_concretes: &[ConcreteValue],
) -> Result<(DispatchOutcome, majit_metainterp::recorder::TracePosition), DispatchError> {
    let (call_site_py_pc, vsd_value, outer_jitcode_index, call_site_marker) = unsafe {
        let jc = &*sym.jitcode();
        let jc_index = jc.index as u32;
        let marker = jc.payload.resume_marker_for_jitcode_pc(op_pc);
        // Forward py twin first (#73 phase-3): equals the containing
        // coordinate plus trivia normalization by construction; the containing
        // lookup survives for the empty-twin class, and the trivia skip below
        // is an identity on the twin path.
        let mut py = jc
            .payload
            .forward_py_pc_for_jitcode_pc(op_pc)
            .unwrap_or_else(|| {
                crate::py_coord::note_empty_twin_fallback(fallback_label, jc.index, op_pc as i32);
                crate::py_coord::containing_py_pc_for_jitcode_pc(&jc.payload.metadata, op_pc)
            });
        if jc.payload.code_ptr.is_null() {
            (py, sym.valuestackdepth() as i64, jc_index, marker)
        } else {
            let codeobj = &*jc.payload.code_ptr;
            py = skip_python_trivia_forward(codeobj, py as usize) as u32;
            // Read the depth off the jitcode-pc-keyed trivia twin, which equals
            // `depth_at_py_pc()[skip_python_trivia_forward(containing_py_pc_for_jitcode_pc(op_pc))]`
            // by construction; fall back to the py_pc-keyed static-liveness read
            // where the twin is unpopulated (skeleton / fixture install).
            let depth = if jc.payload.depth_trivia_populated() {
                jc.payload.depth_trivia_for_jitcode_pc(op_pc)
            } else {
                crate::liveness::liveness_for(jc.payload.code_ptr)
                    .depth_at_py_pc()
                    .get(py as usize)
                    .copied()
            };
            let vsd = match depth {
                Some(d) => (sym.nlocals() + d as usize) as i64,
                None => sym.valuestackdepth() as i64,
            };
            (py, vsd, jc_index, marker)
        }
    };
    if sym.owns_virtualizable_shadow() {
        let li = call_site_py_pc as i64 - 1;
        let li_op = ctx.trace_ctx.const_int(li);
        crate::trace_opcode::mirror_vable_static_to_boxes(
            ctx.trace_ctx,
            "last_instr",
            li_op,
            Value::Int(li),
        );
        let vsd_op = ctx.trace_ctx.const_int(vsd_value);
        crate::trace_opcode::mirror_vable_static_to_boxes(
            ctx.trace_ctx,
            "valuestackdepth",
            vsd_op,
            Value::Int(vsd_value),
        );
    }
    let call_site_word = call_site_marker
        .map(|marker| marker as i32)
        .unwrap_or(majit_ir::resumedata::NO_JITCODE_PC);
    let active = collect_outer_active_boxes(
        sym,
        ctx.trace_ctx,
        ctx.registers_i,
        ctx.registers_r,
        ctx.registers_f,
        outer_jitcode_index,
        false,
        call_site_word,
        op_pc as i32,
        OuterActiveBoxesEntryTwin::Plain,
        call_site_label,
        None,
        &[],
        // Not a branch-guard reconstruction: this is the pre-call site
        // snapshot, so there is no kept operand-stack slot to report as
        // unsourced.
        None,
    );

    let saved_entry = ctx.entry_py_pc;
    let saved_marker = ctx.outer_resume_marker_jit_pc;
    let saved_oji = ctx.outer_jitcode_index;
    let saved_active = std::mem::take(&mut ctx.outer_active_boxes);
    let saved_descr_refs = ctx.descr_refs;
    let saved_raw_descrs = ctx.raw_descrs;
    let saved_lookup = ctx.sub_jitcode_lookup;
    ctx.entry_py_pc = EntryPyPc::Jit(op_pc);
    ctx.outer_resume_marker_jit_pc = call_site_marker;
    ctx.outer_jitcode_index = outer_jitcode_index;
    ctx.outer_active_boxes = active;
    ctx.descr_refs = crate::jitcode_runtime::descr_ref_table();
    ctx.raw_descrs = RawDescrPool::Global;
    ctx.sub_jitcode_lookup = &GLOBAL_SUB_JITCODE_LOOKUP_FN;

    let walk_start = ctx.trace_ctx.get_trace_position();
    let saved_fbw_mode = ctx.fbw_mode;
    ctx.fbw_mode.inline_subwalk = true;
    let walk_result = run_sub_jitcode_walk(
        ctx,
        op_pc,
        sub_body,
        int_args,
        int_arg_concretes,
        ref_args,
        ref_arg_concretes,
        &[],
    );
    ctx.fbw_mode = saved_fbw_mode;
    ctx.entry_py_pc = saved_entry;
    ctx.outer_resume_marker_jit_pc = saved_marker;
    ctx.outer_jitcode_index = saved_oji;
    ctx.outer_active_boxes = saved_active;
    ctx.descr_refs = saved_descr_refs;
    ctx.raw_descrs = saved_raw_descrs;
    ctx.sub_jitcode_lookup = saved_lookup;

    Ok((walk_result?, walk_start))
}

/// Commit core of the #171 orthodox list-append fold, shared by the
/// method-call (`try_walker_orthodox_list_append`) and LIST_APPEND-opcode
/// (`try_walker_orthodox_list_append_opcode`) forms.  Stamps the receiver
/// concrete, pins the value's class (Integer/Float storage), publishes the
/// single append-site resume coordinate, descends the real `w_list_append`
/// body as a sub-jitcode walk recording its native array store, then journals
/// + applies the concrete append.  `self_ref` is the receiver list OpRef the
/// caller supplies (the bound method's `w_self` field, or the opcode's list
/// operand); `sym` / `sub_body` are the pre-resolved resume source + callee
/// body.  The caller writes any residual result (the method form's `None`; the
/// opcode form is void).  Records IR unconditionally — a body sub-walk abort
/// propagates as `DispatchError` (graceful interpreter fallback), never a wrong
/// trace.
#[allow(clippy::too_many_arguments)]
pub(crate) fn orthodox_list_append_commit<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op: &DecodedOp,
    sym: &Sym,
    sub_body: &SubJitCodeBody,
    self_ref: OpRef,
    value_op: OpRef,
    mut inner_self: pyre_object::PyObjectRef,
    value: pyre_object::PyObjectRef,
    len_before: usize,
) -> Result<(), DispatchError> {
    let allocated_before = unsafe { pyre_object::listobject::w_list_allocated(inner_self) };
    // Keep the original Ref box across the helper-frame boundary.  For a
    // virtual W_IntObject/W_FloatObject, its cached payload field is the live
    // SSA box recorded by `trace_box_int`/`trace_box_float`; the descended
    // `plain_int_w`/float unbox therefore forwards that field exactly like
    // `OptVirtualize.optimize_GETFIELD_GC_I/F` in
    // `rpython/jit/metainterp/optimizeopt/virtualize.py`.  Making the Ref's
    // identity observable here would force an otherwise non-escaping virtual.
    // Stamp the receiver concrete (the sub-walk reads it as ref-arg 0; its
    // strategy switch needs the concrete receiver).
    ctx.trace_ctx.set_opref_concrete(
        self_ref,
        majit_ir::Value::Ref(majit_ir::GcRef(inner_self as usize)),
    );

    // Empty-strategy first-append promotion (gated): install typed storage on
    // the receiver BEFORE the value-class pin / storage read below, so those
    // observe the post-promotion strategy. Classify the target strategy from
    // the value with recognize's int -> float -> object guards
    // (switch_to_correct_strategy, listobject.py), then emit the
    // transition IR mutating the existing wrapper, promote the concrete list,
    // and journal the rewind to Empty.
    use pyre_object::listobject::ListStrategy;
    let promote_empty = unsafe { pyre_object::w_list_uses_empty_storage(inner_self) };
    if promote_empty {
        let target = unsafe {
            let int_ok = pyre_object::is_plain_int1(value)
                && !(pyre_object::tagged_int::CAN_BE_TAGGED
                    && pyre_object::tagged_int::is_tagged_int(value));
            if int_ok {
                ListStrategy::Integer
            } else if pyre_object::is_float_strategy_item(value) {
                ListStrategy::Float
            } else {
                ListStrategy::Object
            }
        };
        // Guard the current (Empty) strategy so a deopt re-enters the empty
        // path (mirror of `MIFrame::guard_list_strategy`: getfield strategy +
        // GuardValue + replace_box).
        let strategy_ref = crate::state::opimpl_getfield_gc_i(
            ctx.trace_ctx,
            self_ref,
            crate::descr::list_strategy_descr(),
        );
        let expected = ctx.trace_ctx.const_int(ListStrategy::Empty as i64);
        ctx.trace_ctx
            .record_guard(OpCode::GuardValue, &[strategy_ref, expected], 0);
        walker_capture_snapshot_for_last_guard(ctx, op.pc)?;
        ctx.trace_ctx
            .heap_cache_mut()
            .replace_box(strategy_ref, expected);
        // Emit the transition IR mutating the existing wrapper (helpers.rs).
        // It stages the same first 0 -> 4 RPython grow as the concrete helper,
        // leaving the append body to record the length/item stores.
        crate::helpers::emit_promote_empty_list_inline(ctx.trace_ctx, self_ref, target);
        // Concrete promotion of the real list, then journal so a non-commit
        // walk rolls back to Empty.
        inner_self = unsafe { pyre_object::w_list_switch_to_strategy_for(inner_self, value) };
        ctx.trace_ctx.set_opref_concrete(
            self_ref,
            majit_ir::Value::Ref(majit_ir::GcRef(inner_self as usize)),
        );
        fbw_append_promote_journal_push(inner_self);
    }

    // Pin the appended value's class so the inlined `is_plain_int1` type
    // predicate folds during the sub-walk: guard_class(value, <TYPE>) +
    // class_now_known, so its `is_int`/`is_long`/`is_bool` typeptr reads fold
    // to the pinned const (the typeptr fold in `getfield_gc_via_heapcache`).
    // The recognition gate already proved `is_plain_int1(value)`; this guard
    // enforces the observed ob_type at runtime.  The value's integer payload
    // stays symbolic — only its class is pinned.
    //
    // Object-storage append stores the value as a
    // plain GC ref with no unboxing, so it carries no type precondition —
    // skip the class pin (the sub-walk's object-storage store path does
    // not read the value's class).
    let is_obj_storage = unsafe { pyre_object::w_list_uses_object_storage(inner_self) };
    if !is_obj_storage {
        // Integer and Float storage both pin the value's class so the body's
        // strict type test folds during the sub-walk; the ob_type const is
        // FLOAT_TYPE for float storage, and INT_TYPE / LONG_TYPE for int
        // storage depending on whether the value is an exact int or a fits-int
        // `W_LongObject` (both pass `is_plain_int1` -> Integer storage, but
        // carry distinct `ob_type`s the sub-walk's `is_plain_int1` folds on).
        let is_float_storage = unsafe { pyre_object::w_list_uses_float_storage(inner_self) };
        let value_is_long = unsafe { pyre_object::pyobject::is_long(value) };
        let value_type_addr = if is_float_storage {
            &pyre_object::pyobject::FLOAT_TYPE as *const _ as i64
        } else if value_is_long {
            &pyre_object::pyobject::LONG_TYPE as *const _ as i64
        } else {
            &pyre_object::pyobject::INT_TYPE as *const _ as i64
        };
        if !value_op.is_constant() && !ctx.trace_ctx.heap_cache().is_class_known(value_op) {
            let type_const = ctx.trace_ctx.const_int(value_type_addr);
            ctx.trace_ctx
                .record_guard(OpCode::GuardClass, &[value_op, type_const], 0);
            walker_capture_snapshot_for_last_guard(ctx, op.pc)?;
        }
        ctx.trace_ctx
            .heap_cache_mut()
            .class_now_known(value_op, value_type_addr);
        // The strict predicate (`is_plain_int1` / `is_plain_float_strict`)
        // rejects subclasses by reading `value.w_class` and requiring it null
        // or == `get_instantiate(<type>)`. The ob_type pin above only folds the
        // `is_int`/`is_float` typeptr reads; the w_class compare stays symbolic,
        // so the inlined predicate is non-concrete and the strategy arm's
        // `if <pred>(value)` branch cannot fold — the sub-walk then descends the
        // dead else-leg `switch_to_object_strategy`, whose `ListStrategy::Object`
        // unit-variant ctor is a symbolic fnaddr the descent declines
        // (`OrthodoxSubWalkTraceUnsupported`). Pin w_class to the concrete
        // value's field so the subclass test folds too (the recognition gate
        // already proved the strict predicate).
        let concrete_w_class = unsafe { (*value).w_class } as i64;
        let w_class_ref = crate::state::opimpl_getfield_gc_r(
            ctx.trace_ctx,
            value_op,
            crate::descr::w_class_descr(),
        );
        let w_class_const = ctx.trace_ctx.const_ref(concrete_w_class);
        ctx.trace_ctx
            .record_guard(OpCode::GuardValue, &[w_class_ref, w_class_const], 0);
        walker_capture_snapshot_for_last_guard(ctx, op.pc)?;
        ctx.trace_ctx
            .heap_cache_mut()
            .replace_box(w_class_ref, w_class_const);
    }

    // Pre-publish the ONE append-site resume coordinate the sub-walk's guards
    // collapse to (mirror the full-body path's last_instr / valuestackdepth
    // publication, keyed to the append op's py_pc — the CALL for the method
    // form, the LIST_APPEND for the opcode form).
    let (walk_outcome, _walk_start) = run_orthodox_helper_subwalk(
        ctx,
        op.pc,
        sym,
        sub_body,
        "list_append_commit",
        "w_list_append_call_site",
        &[],
        &[],
        &[self_ref, value_op],
        &[ConcreteValue::Ref(inner_self), ConcreteValue::Ref(value)],
    )?;

    match walk_outcome {
        DispatchOutcome::SubReturn { result } => {
            if finish_inline_callee_return(ctx, result).is_some() {
                return Err(DispatchError::UnexpectedNonVoidSubReturn { pc: op.pc });
            }
        }
        _ => return Err(DispatchError::UnexpectedNonVoidSubReturn { pc: op.pc }),
    }

    // Reaching here means the body sub-walk completed without hitting an
    // un-lowered helper: the strategy switch folded over the concrete
    // receiver, the strict type-predicate leaves recursed (`is_plain_int1`
    // for Integer / `is_plain_float_strict` for Float; Object stores with no
    // type test), the `ll_list_{int,float,obj}_*` leaves lowered to
    // getfield/setfield/setarrayitem, and the unit-`()` return aggregate
    // (`SyntheticTransparentCtor "Tuple"`) was elided to `ConstRefNull` at
    // build time.  Any residual that does NOT lower —
    // e.g. a stale build-time jitcode whose tuple ctor kept a symbolic
    // symbolic-tagged funcbox — is declined by `try_execute_residual_call_via_executor`
    // (`OrthodoxSubWalkTraceUnsupported`) and the sub-walk helper propagates that
    // abort before this point (graceful interpreter fallback, never a wrong
    // trace).  The descr-pool wiring above (strategy/header field descrs) is
    // exercised on the way in.

    // Tracing is execution: apply the append + journal the rewind.  The
    // journal entry is unconditional — it rewinds the receiver to
    // `len_before` on an aborted walk, whichever side actually grew it.
    //
    // The sub-walk normally records the store as IR without touching the
    // concrete list, so the append below is what applies it.  It is not
    // guaranteed to: the per-strategy store the descended arm reaches
    // (`W_ListObject::object_push`, `IntArray::push`, `FloatArray::push`) is a
    // `residual_call`, and a residual whose funcptr resolves to a real address
    // is EXECUTED by `try_execute_residual_call_via_executor` rather than only
    // recorded.  Those three carry runtime bindings, so on a target where the
    // arm keeps them as residuals the sub-walk has already appended, and
    // appending again puts the value in twice — one extra element per compiled
    // append, which is how it surfaces (`len(keep)` 20048 for 20000
    // iterations, a traceback name list with its last frame doubled).
    // Re-read the length instead of assuming which side ran: it is the
    // receiver's own state, so it answers for both.
    fbw_list_journal_push_append(inner_self, len_before, allocated_before);
    if unsafe { pyre_object::w_list_len(inner_self) } == len_before {
        unsafe { pyre_object::w_list_append(inner_self, value) };
    }
    Ok(())
}

/// Descend the Integer-strategy `w_list_pop_end_inner` body for a bound
/// `list.pop()` call, recording its length/item array operations instead of an
/// opaque residual call.
///
/// "Guard-free" elsewhere about this body (`listobject.rs` `w_list_pop_end`,
/// `jitcode_runtime.rs` `list_pop_end_jitcode`) names the *lock* guard: a
/// `w_list_lock` pair inside the body would decline the sub-walk. It is not a
/// claim about trace guards, and the two must not be conflated here, because
/// what the fold's soundness rests on is a trace-guard ordering property:
///
/// The sub-walk gets no callee frame — `outer_*` and `snapshot_sym` below stay
/// the caller's — so a guard recorded inside it resumes at this CALL boundary
/// and re-executes the whole `pop()`. That is only sound while every guard
/// lands *before* the body's first committed store. It does today: the Integer
/// arm's `ll_list_int_set_len` is a native `setfield_gc_i`, and the sole op
/// after it is the `w_int_new` call, which `dispatch_inline_call_dir_kind`
/// short-circuits into `walker_box_int` (`NewWithVtable` + `SetfieldGc`,
/// recording no guard) and returns before `run_sub_jitcode_walk`. Take that
/// short-circuit away and the walk records the boxing body's own null and
/// exception guards after the length is already shrunk, and a failure there
/// pops twice. The pre-fold `GuardClass` / `GuardValue` and the strategy
/// switch's guard are ahead of the store by construction; the body's own ops
/// are checked instead of assumed —
/// [`subwalk_guard_follows_store`] reads the window the sub-walk recorded and
/// declines the fold if a guard landed past the first store.
pub(crate) fn try_walker_orthodox_list_pop<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    code: &[u8],
    op: &DecodedOp,
    r_args: &[OpRef],
    dst: usize,
) -> Result<Option<()>, DispatchError> {
    if r_args.len() != 2 {
        return Ok(None);
    }
    let arg_concretes = read_ref_var_list_concrete(code, op, 1, ctx);
    let (ConcreteValue::Ref(callable), ConcreteValue::Ref(null_or_self)) =
        (arg_concretes[0], arg_concretes[1])
    else {
        return Ok(None);
    };
    if callable.is_null() || !null_or_self.is_null() {
        return Ok(None);
    }

    let (inner_func, inner_self, len_before, raw_item) = unsafe {
        if !pyre_object::function::is_method(callable) {
            return Ok(None);
        }
        let inner_func = pyre_object::function::w_method_get_func(callable);
        let inner_self = pyre_object::function::w_method_get_self(callable);
        if inner_func.is_null() || inner_self.is_null() {
            return Ok(None);
        }
        let list_type = pyre_interpreter::typedef::gettypeobject(&pyre_object::pyobject::LIST_TYPE);
        if pyre_interpreter::lookup_in_type(list_type, "pop") != Some(inner_func) {
            return Ok(None);
        }
        let Some((len_before, raw_item)) = orthodox_list_pop_recognize(inner_self) else {
            return Ok(None);
        };
        (inner_func, inner_self, len_before, raw_item)
    };

    // Resolve every possible decline before recording a guard.
    let Some((sub_body, sym_ptr)) = orthodox_list_pop_body_and_sym(ctx) else {
        return Ok(None);
    };
    let sym = unsafe { &*sym_ptr };
    let pre_fold_pos = ctx.trace_ctx.get_trace_position();
    let callable_op = r_args[0];

    let method_type_addr = &pyre_object::function::METHOD_TYPE as *const _ as i64;
    if !callable_op.is_constant() && !ctx.trace_ctx.heap_cache().is_class_known(callable_op) {
        let type_const = ctx.trace_ctx.const_int(method_type_addr);
        ctx.trace_ctx
            .record_guard(OpCode::GuardClass, &[callable_op, type_const], 0);
        walker_capture_snapshot_for_last_guard(ctx, op.pc)?;
    }
    ctx.trace_ctx
        .heap_cache_mut()
        .class_now_known(callable_op, method_type_addr);
    let func_ref = crate::state::opimpl_getfield_gc_r(
        ctx.trace_ctx,
        callable_op,
        crate::descr::method_w_function_descr(),
    );
    let func_const = ctx.trace_ctx.const_ref(inner_func as i64);
    ctx.trace_ctx
        .record_guard(OpCode::GuardValue, &[func_ref, func_const], 0);
    walker_capture_snapshot_for_last_guard(ctx, op.pc)?;
    ctx.trace_ctx
        .heap_cache_mut()
        .replace_box(func_ref, func_const);
    let self_ref = crate::state::opimpl_getfield_gc_r(
        ctx.trace_ctx,
        callable_op,
        crate::descr::method_w_self_descr(),
    );

    match orthodox_list_pop_commit(
        ctx, op, sym, &sub_body, self_ref, inner_self, len_before, raw_item, dst,
    ) {
        Ok(()) => Ok(Some(())),
        Err(DispatchError::OrthodoxSubWalkTraceUnsupported { pc, .. }) => {
            if fbw_debug_abort_enabled() {
                eprintln!("[decline-why] LIST-POP-SUBWALK pc={pc}");
            }
            ctx.trace_ctx.cut_trace(pre_fold_pos);
            ctx.trace_ctx.heap_cache_mut().reset();
            Ok(None)
        }
        Err(error) => Err(error),
    }
}

/// Recognize a non-empty Integer-strategy list and sample its final unboxed
/// item before the descended executor can mutate the live list.
unsafe fn orthodox_list_pop_recognize(
    inner_self: pyre_object::PyObjectRef,
) -> Option<(usize, i64)> {
    if !pyre_object::pyobject::is_list(inner_self)
        || !pyre_object::w_list_uses_int_storage(inner_self)
    {
        return None;
    }
    let len_before = pyre_object::w_list_len(inner_self);
    if len_before == 0 {
        return None;
    }
    let list = &*(inner_self as *const pyre_object::listobject::W_ListObject);
    let raw_item = pyre_object::listobject::ll_list_int_getitem_fast(list, len_before - 1);
    Some((len_before, raw_item))
}

/// Resolve the pop body and enclosing full-body snapshot before IR emission.
pub(crate) fn orthodox_list_pop_body_and_sym<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
) -> Option<(SubJitCodeBody, *const Sym)> {
    let jc_arc = crate::jitcode_runtime::list_pop_end_jitcode()?;
    let sub_body = sub_jitcode_body_by_index(jc_arc.index())?;
    let sym_ptr = ctx.fbw_mode.snapshot_sym;
    if sym_ptr.is_null() || unsafe { (&*sym_ptr).jitcode().is_null() } {
        return None;
    }
    Some((sub_body, sym_ptr))
}

/// Whether the ops recorded since `start` put a guard after a store.
///
/// A sub-walk that gets no callee frame resumes its guards at the caller's CALL
/// boundary, so a failure re-executes the whole call. That is sound only while
/// every guard precedes the body's first store: past one, the resumed call
/// re-applies an effect the body already recorded.
///
/// This reads the recorded IR only. A `setfield` here says the walk *recorded*
/// a store, not that one reached the heap — `setfield_gc_via_heapcache` writes
/// through only for boxes the walk itself allocated — so a caller that wants to
/// decline on the answer still owes its own proof that nothing observable has
/// been applied yet.
///
/// A `start` past the end means the trace was cut below the capture point, so
/// the window this answers about is gone; report the guard rather than read an
/// empty one as "sound".
pub(crate) fn subwalk_guard_follows_store(
    trace_ctx: &TraceCtx,
    start: majit_metainterp::recorder::TracePosition,
) -> bool {
    let ops = trace_ctx.ops();
    let Some(recorded) = ops.get(start._pos..) else {
        return true;
    };
    let mut stored = false;
    for op in recorded {
        if op.opcode.is_guard() && stored {
            return true;
        }
        stored |= op.opcode.is_setfield()
            || op.opcode.is_setarrayitem()
            || op.opcode.is_setinteriorfield();
    }
    false
}

/// Publish the pop call-site resume coordinate, descend the real helper body,
/// write its Ref result, and journal/apply the concrete shrink exactly once.
#[allow(clippy::too_many_arguments)]
pub(crate) fn orthodox_list_pop_commit<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op: &DecodedOp,
    sym: &Sym,
    sub_body: &SubJitCodeBody,
    self_ref: OpRef,
    inner_self: pyre_object::PyObjectRef,
    len_before: usize,
    raw_item: i64,
    dst: usize,
) -> Result<(), DispatchError> {
    ctx.trace_ctx
        .set_opref_concrete(self_ref, Value::Ref(majit_ir::GcRef(inner_self as usize)));

    let (walk_outcome, walk_start) = run_orthodox_helper_subwalk(
        ctx,
        op.pc,
        sym,
        sub_body,
        "list_pop_commit",
        "w_list_pop_end_call_site",
        &[],
        &[],
        &[self_ref],
        &[ConcreteValue::Ref(inner_self)],
    )?;

    let result = match walk_outcome {
        DispatchOutcome::SubReturn { result } => finish_inline_callee_return(ctx, result)
            .ok_or(DispatchError::UnexpectedVoidSubReturn { pc: op.pc })?,
        _ => return Err(DispatchError::UnexpectedVoidSubReturn { pc: op.pc }),
    };
    // The Integer arm commits `ll_list_int_set_len` before it boxes, so a guard
    // recorded after that store would resume at this CALL boundary and pop a
    // second time. Today none is: the boxing call is short-circuited into
    // `NewWithVtable` + `SetfieldGc`. Decline rather than inherit that as an
    // assumption — the caller cuts back to the generic residual, which pops
    // exactly once.
    //
    // Declining is only safe while the receiver is untouched, so it takes the
    // same length re-read the commit below does: on a target whose
    // `ll_list_int_set_len` keeps a runtime binding, the sub-walk executed it
    // for real rather than recording it, and cutting back to a residual that
    // pops again is the very double-pop this fold already had to fix on the
    // append side. In that case the store is a `call`, not a `setfield`, so
    // the ordering read has nothing to say about it either.
    if unsafe { pyre_object::w_list_len(inner_self) } == len_before
        && subwalk_guard_follows_store(ctx.trace_ctx, walk_start)
    {
        // This decline is an ordering verdict on the receiver, not a descent
        // that reached an unlowered helper, so there is no symbolic address to
        // carry.  Zero is unambiguous: a real symbolic hash always carries the
        // `SYMBOLIC_FNADDR_BASE` tag.
        return Err(DispatchError::OrthodoxSubWalkTraceUnsupported {
            pc: op.pc,
            symbolic: 0,
        });
    }
    write_residual_call_result_to_dst(ctx, op.pc, dst, 'r', result)?;

    let w_item = pyre_object::w_int_new(raw_item);
    fbw_list_journal_push_pop_end(inner_self, len_before, w_item);
    if unsafe { pyre_object::w_list_len(inner_self) } == len_before {
        unsafe { pyre_object::w_list_pop_end(inner_self) };
    }
    Ok(())
}

/// LIST_APPEND-opcode form of the #171 orthodox list-append fold (comprehension
/// append, e.g. `[f(x) for x in xs]` inlines LIST_APPEND into the enclosing
/// function).  The codewriter lowers LIST_APPEND to a void
/// `jit_list_append(list, value)` residual tagged `ListAppendValue`; here
/// `r_args = [list, value]` (the peeked receiver + the popped value — no
/// bound-method callable).  Recognises the receiver/value against the shared
/// gate and descends the same `w_list_append` body as the method-call form
/// ([`try_walker_orthodox_list_append`]).  Returns `None` (fall through to the
/// generic residual, SAFE — identical to the retired MIFrame tracer's `jit_list_append`)
/// for any non-matching shape, and likewise after rolling the tentative IR back
/// when the body sub-walk hits an un-lowered helper; the residual is void so no
/// result is written.
pub(crate) fn try_walker_orthodox_list_append_opcode<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    code: &[u8],
    op: &DecodedOp,
    r_args: &[OpRef],
    dst: usize,
) -> Result<Option<()>, DispatchError> {
    let _ = dst; // LIST_APPEND residual is void — no result to write.
    if r_args.len() != 2 {
        return Ok(None);
    }
    let arg_concretes = read_ref_var_list_concrete(code, op, 1, ctx);
    let (ConcreteValue::Ref(list), ConcreteValue::Ref(value)) =
        (arg_concretes[0], arg_concretes[1])
    else {
        return Ok(None);
    };
    if list.is_null() || value.is_null() {
        return Ok(None);
    }

    // Recognition: no bound-method callable to pin — the list and value are the
    // residual's two Ref operands directly.
    let Some(len_before) = (unsafe { orthodox_list_append_recognize(list, value) }) else {
        return Ok(None);
    };

    // Resolve the compiled body BEFORE emitting any IR — the opcode form emits
    // no guard before the commit, so this is the only decline point.
    let Some((sub_body, sym_ptr)) = orthodox_list_append_body_and_sym(ctx) else {
        return Ok(None);
    };
    // SAFETY: `sym_ptr` is non-null with a set `jitcode` (checked in the
    // resolver) and stays live for the enclosing full-body walk.
    let sym = unsafe { &*sym_ptr };

    let pre_fold_pos = ctx.trace_ctx.get_trace_position();
    // Mirror the commit's own promotion predicate exactly: a rollback must
    // undo a promotion only when one was performed, or it would pop another
    // list's journal entry.
    let promoted_empty = unsafe { pyre_object::w_list_uses_empty_storage(list) };

    // ── tentative commit ──
    // The receiver list OpRef + value OpRef are the residual's Ref operands.
    let commit_result = orthodox_list_append_commit(
        ctx, op, sym, &sub_body, r_args[0], r_args[1], list, value, len_before,
    );
    match commit_result {
        Ok(()) => {}
        Err(DispatchError::OrthodoxSubWalkTraceUnsupported { pc, .. }) => {
            if fbw_debug_abort_enabled() {
                eprintln!("[decline-why] LIST-APPEND-SUBWALK pc={pc}");
            }
            ctx.trace_ctx.cut_trace(pre_fold_pos);
            ctx.trace_ctx.heap_cache_mut().reset();
            if promoted_empty {
                fbw_append_promote_journal_rollback_last(list);
            }
            return Ok(None);
        }
        Err(error) => return Err(error),
    }
    Ok(Some(()))
}

/// Walker-native exception-construction fold.  A
/// `Type(args)` `CallFn` residual for a canonical builtin exception class or
/// a heap subclass with the same `__new__` / `__init__` descriptors becomes a
/// traced `NewWithVtable` + `SetfieldGc` (kind / w_class / args_w) the
/// optimizer can virtualize when the exception never escapes, instead of
/// the opaque `bh_call_fn` constructor residual + its
/// `GUARD_NOT_FORCED` / `GUARD_NO_EXCEPTION`.
///
/// The `CallFn` arglist is `r_args = [callable, PY_NULL, args...]`
/// (the `bh_call_fn_N` shape — see `try_walker_specialize_list_append`);
/// the positional args are `r_args[2..]` (the `PY_NULL` self slot is
/// skipped).  Records the fresh `NewWithVtable` OpRef in
/// [`FBW_BUILT_EXC`] so a following `RaiseVarargs` takes the instance
/// fast path; writes the trace-time concrete exception into the dst
/// shadow so the `raise/r` GUARD_CLASS reads it.
///
/// PyPy's `W_TypeObject.descr_call` promotes the class, then resolves
/// `__new__` and `__init__` through its versioned MRO
/// (`typeobject.py`).  When both resolve to
/// `W_BaseException.descr_new` / `descr_init`
/// (`interp_exceptions.py`), a trivial subclass has the same traced
/// allocation and `args_w` store as its builtin base; only `w_class` differs.
///
/// Returns `None` (fall through to the generic residual) for any non-matching
/// shape: an overriding or uncacheable subclass, an unsupported
/// non-trivial-args kind, or a null concrete arg.  OSError's parsed fields and
/// SystemExit's code field are emitted alongside the base exception fields;
/// the remaining non-trivial constructors stay on the runtime path.
pub(crate) fn try_walker_trace_exception_new<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    code: &[u8],
    op: &DecodedOp,
    r_args: &[OpRef],
    dst: usize,
) -> Result<Option<()>, DispatchError> {
    // Plain `bh_call_fn(callable, PY_NULL, args...)` shape only.
    if r_args.len() < 2 {
        return Ok(None);
    }
    let arg_concretes = read_ref_var_list_concrete(code, op, 1, ctx);
    let (ConcreteValue::Ref(concrete_callable), ConcreteValue::Ref(null_or_self)) =
        (arg_concretes[0], arg_concretes[1])
    else {
        return Ok(None);
    };
    // A non-null `null_or_self` is a bound receiver `bh_call_fn_impl`
    // prepends as arg0 — not a plain `Type(args)` call.
    if concrete_callable.is_null() || !null_or_self.is_null() {
        return Ok(None);
    }

    // Concrete positional args (skip callable + PY_NULL self).  The
    // residual `args_w` list must match the runtime `descr_init` list
    // exactly, so reject any null.
    let args = &r_args[2..];
    let concrete_args: Vec<pyre_object::PyObjectRef> = arg_concretes[2..]
        .iter()
        .map(|c| match c {
            ConcreteValue::Ref(p) => *p,
            _ => std::ptr::null_mut(),
        })
        .collect();

    let is_exc_class = unsafe {
        pyre_interpreter::baseobjspace::exception_is_valid_obj_as_class_w(concrete_callable)
    };
    if !is_exc_class || concrete_args.iter().any(|a| a.is_null()) {
        return Ok(None);
    }

    // OSError can rebind `args_w` after parsing a filename, so its final
    // slice is selected below, once the concrete constructor has exposed the
    // value-dependent branch result.

    let is_canonical = pyre_object::interp_exceptions::is_canonical_exc_class(concrete_callable);
    let mut subclass_lookups = None;
    let subclass_version_tag = if is_canonical {
        None
    } else {
        // A heap subclass is safe to construct concretely only after both MRO
        // lookups have been proved identical to a canonical exception class.
        // Consequently force_plain_eval below can execute only the builtin
        // Rust `descr_new` / `descr_init`, never user Python code.  This is the
        // promoted-class lookup contract of typeobject.py.
        if !unsafe { pyre_object::typeobject::w_type_is_heaptype(concrete_callable) } {
            return Ok(None);
        }
        let version_tag =
            unsafe { pyre_object::typeobject::w_type_get_version_tag(concrete_callable) };
        if version_tag == 0 {
            return Ok(None);
        }
        let Some(class_new) = (unsafe {
            pyre_interpreter::baseobjspace::lookup_in_type(concrete_callable, "__new__")
        }) else {
            return Ok(None);
        };
        let Some(class_init) = (unsafe {
            pyre_interpreter::baseobjspace::lookup_in_type(concrete_callable, "__init__")
        }) else {
            return Ok(None);
        };
        let matches_canonical = (0..pyre_object::interp_exceptions::EXC_KIND_COUNT).any(|disc| {
            // ExcKind is repr(u8) with contiguous discriminants through
            // EXC_KIND_COUNT, as required by the kind-indexed registry.
            let candidate_kind: pyre_object::interp_exceptions::ExcKind =
                unsafe { std::mem::transmute(disc as u8) };
            let candidate =
                pyre_object::interp_exceptions::lookup_exc_class_for_kind(candidate_kind);
            if candidate.is_null() {
                return false;
            }
            unsafe {
                pyre_interpreter::baseobjspace::lookup_in_type(candidate, "__new__")
                    == Some(class_new)
                    && pyre_interpreter::baseobjspace::lookup_in_type(candidate, "__init__")
                        == Some(class_init)
            }
        });
        if !matches_canonical {
            return Ok(None);
        }
        subclass_lookups = Some((class_new, class_init));
        Some(version_tag)
    };
    // Build the exception concretely on the plain eval loop (no tracer
    // re-entry) to read its kind and confirm a flat builtin instance.
    // Trace-time only; discarded after the read.
    let exc = {
        let _plain_guard = pyre_interpreter::call::force_plain_eval();
        pyre_interpreter::call::call_function_impl_result(concrete_callable, &concrete_args)
    };
    let Ok(exc) = exc else { return Ok(None) };
    let kind = unsafe {
        if !pyre_object::is_exception(exc) {
            return Ok(None);
        }
        pyre_object::interp_exceptions::w_exception_get_kind(exc)
    };
    let canonical_class = pyre_object::interp_exceptions::lookup_exc_class_for_kind(kind);
    if is_canonical {
        // Preserve the canonical arm's registry identity check.
        if canonical_class != concrete_callable {
            return Ok(None);
        }
    } else {
        // The pre-construction descriptor check excludes Python execution;
        // repeat it for the concrete result's eventual kind so aliases whose
        // builtin wrapper produces a different physical kind still decline.
        let Some((class_new, class_init)) = subclass_lookups else {
            return Ok(None);
        };
        if canonical_class.is_null()
            || unsafe {
                pyre_interpreter::baseobjspace::lookup_in_type(canonical_class, "__new__")
                    != Some(class_new)
                    || pyre_interpreter::baseobjspace::lookup_in_type(canonical_class, "__init__")
                        != Some(class_init)
            }
        {
            return Ok(None);
        }
    }
    // `exc_new_wrapper` retags only `w_class`; the physical layout remains
    // the eventual kind's builtin pytype for canonical classes and subclasses.
    let exc_type_ptr = unsafe {
        (*(exc as *const pyre_object::interp_exceptions::W_BaseException))
            .ob_header
            .ob_type
    };
    if !std::ptr::eq(
        exc_type_ptr,
        pyre_object::interp_exceptions::exc_kind_to_pytype(kind),
    ) {
        return Ok(None);
    }
    let is_os_error_family = matches!(
        kind,
        pyre_object::interp_exceptions::ExcKind::OSError
            | pyre_object::interp_exceptions::ExcKind::FileNotFoundError
    );
    let is_system_exit = kind == pyre_object::interp_exceptions::ExcKind::SystemExit;
    // `W_OSError._parse_init_args` / `_init_error`
    // (`interp_exceptions.py`) fill the flattened slots only for 2..=5
    // arguments.  Outside that range the ordinary args-only emit is exact.
    // Unicode constructors still require their dedicated parsing and remain
    // residual.
    let fills_os_error_slots = is_os_error_family && (2..=5).contains(&args.len());

    // Admit the kind exactly when the concretely built instance left its extra
    // slots defaulted — the slot-content test [`try_walker_trace_raise_bare_class`]
    // already runs, in place of a per-kind tag that rejected a whole kind on
    // faith and so kept `AttributeError(msg)` / `NameError(msg)` /
    // `StopIteration()` on the opaque constructor residual.  A `NULL` slot needs
    // no store; a `None` one takes an explicit `SetfieldGc` below.
    //
    // The bare-class sibling censuses an instance built with NO arguments, so
    // every slot it sees is a trace-time constant.  Here the instance is built
    // from the runtime operands `args`, which nothing pins — only the callable
    // is guarded.  A slot that reads `None` because an ARGUMENT was `None`
    // would therefore be emitted as a constant `None` store while `args_w`
    // keeps the live operand: `StopIteration(x)` traced with `x is None` would
    // answer `e.value is None` for every later `x`.  Each of these
    // constructors fills a slot with either a constant default or one of the
    // passed values, so requiring every argument to be non-`None` makes a
    // `None` slot provably a default.  The check is read only once a defaulted
    // slot is actually found, leaving the all-`NULL` kinds this fold already
    // admitted on their existing path.
    //
    // OSError / SystemExit fill their slots from the arguments, and the emit
    // tail writes them from the argument OpRefs; they skip the census.
    let w_none = pyre_object::w_none();
    let mut w_none_slot_descrs = Vec::new();
    if !is_os_error_family && !is_system_exit {
        let any_none_arg = concrete_args.iter().any(|a| std::ptr::eq(*a, w_none));
        for (offset, value) in
            unsafe { pyre_object::interp_exceptions::w_exception_traced_construction_slots(exc) }
        {
            if value.is_null() {
                continue;
            }
            if !std::ptr::eq(value, w_none) || any_none_arg {
                return Ok(None);
            }
            let Some(descr) = crate::descr::w_exception_slot_descr(kind, offset) else {
                return Ok(None);
            };
            w_none_slot_descrs.push(descr);
        }
    }

    // `interp_exceptions.py W_SystemExit.descr_init` stores one
    // argument verbatim and several as the tuple selected by `newtuple`.
    // Settle the multi-argument representation before emitting any guards so
    // an unsupported unboxed pair can still decline without leaving trace
    // state behind.
    let system_exit_code = if !is_system_exit || args.is_empty() {
        None
    } else if args.len() == 1 {
        Some((Some(args[0]), None))
    } else {
        let concrete_code = unsafe { pyre_object::interp_exceptions::w_exception_get_code(exc) };
        let code_type = unsafe { (*concrete_code).ob_type };
        if std::ptr::eq(code_type, &pyre_object::TUPLE_TYPE) {
            Some((None, Some((false, concrete_code))))
        } else if std::ptr::eq(
            code_type,
            &pyre_object::specialisedtupleobject::SPECIALISED_TUPLE_OO_TYPE,
        ) {
            Some((None, Some((true, concrete_code))))
        } else {
            return Ok(None);
        }
    };

    let exact_os_error = pyre_interpreter::builtins::lookup_exc_class("OSError")
        .is_some_and(|w_os_error| std::ptr::eq(concrete_callable, w_os_error));
    if fills_os_error_slots && exact_os_error {
        // PyPy traces the errno-to-subclass lookup with a loop-variant errno.
        // The flat NewWithVtable emit needs a constant w_class, so pinning the
        // unboxed errno deliberately creates per-errno traces/bridges.
        let errno = concrete_args[0];
        let exact_int = pyre_object::tagged_int::CAN_BE_TAGGED
            && pyre_object::tagged_int::is_tagged_int(errno)
            || unsafe {
                pyre_object::is_plain_int1(errno)
                    && std::ptr::eq(
                        (*errno).ob_type,
                        &pyre_object::pyobject::INT_TYPE as *const _,
                    )
            };
        if !exact_int {
            return Ok(None);
        }
    }

    let concrete_w_class = unsafe { (*exc).w_class };
    let is_blocking_io_error = pyre_interpreter::builtins::lookup_exc_class("BlockingIOError")
        .is_some_and(|blocking| std::ptr::eq(concrete_w_class, blocking));
    // `W_OSError._init_error` gives an exact BlockingIOError's numeric third
    // argument the characters_written meaning.  Keep every three-or-more-arg
    // instance of that concrete class on the complete runtime path.
    if fills_os_error_slots && args.len() >= 3 && is_blocking_io_error {
        return Ok(None);
    }

    // Where the platform reads the fourth argument, `_parse_init_args` derives
    // the errno and the retagged class from it and stores it in its own slot.
    // Both depend on a value this emit neither guards nor writes, so those
    // instances stay on the runtime path.
    if cfg!(windows) && fills_os_error_slots && args.len() >= 4 {
        return Ok(None);
    }

    let has_filename = fills_os_error_slots
        && args.len() >= 3
        && !unsafe { pyre_object::is_none(concrete_args[2]) };
    let final_args_len = if has_filename { 2 } else { args.len() };
    let final_args = &args[..final_args_len];

    // GuardClass pins each None-sensitive `_init_error` branch.  A tagged
    // immediate cannot be consumed by GuardClass; retain the residual path for
    // that uncommon filename shape.
    if fills_os_error_slots {
        for index in [2usize, 4] {
            if index >= args.len() || (index == 4 && args.len() != 5) {
                continue;
            }
            if pyre_object::tagged_int::CAN_BE_TAGGED
                && pyre_object::tagged_int::is_tagged_int(concrete_args[index])
            {
                return Ok(None);
            }
        }
    }
    // --- commit to the specialization: emit IR (no further declines) ---
    // Pin the callable identity so the trace-time kind / vtable stay
    // valid across iterations (`implement_guard_value`).
    let callable_op = r_args[0];
    if !callable_op.is_constant() {
        let expected = ctx.trace_ctx.const_ref(concrete_callable as i64);
        ctx.trace_ctx
            .record_guard(OpCode::GuardValue, &[callable_op, expected], 0);
        walker_capture_snapshot_for_last_guard(ctx, op.pc)?;
        ctx.trace_ctx
            .heap_cache_mut()
            .replace_box(callable_op, expected);
    }
    if subclass_version_tag.is_some() {
        // Pin the promoted class version that made both MRO descriptor
        // identities constant.  `W_TypeObject.mutated` recursively changes
        // subclass tags (`typeobject.py`), so mutating this class or a
        // base revokes the loop before the folded constructor is reused.
        let class_const = ctx.trace_ctx.const_ref(concrete_callable as i64);
        walker_pin_type_version_tag(ctx, op.pc, class_const)?;
    }

    if fills_os_error_slots && exact_os_error {
        let int_type_addr = &pyre_object::pyobject::INT_TYPE as *const _ as i64;
        let raw_errno = walker_unbox_int(ctx, op.pc, args[0], int_type_addr)?;
        let errno_value = unsafe { pyre_object::w_int_get_value(concrete_args[0]) };
        let errno_const = ctx.trace_ctx.const_int(errno_value);
        walker_emit_fold_guard_with_snapshot(
            ctx,
            op.pc,
            OpCode::GuardValue,
            &[raw_errno, errno_const],
        )?;
        ctx.trace_ctx
            .heap_cache_mut()
            .replace_box(raw_errno, errno_const);
    }
    if fills_os_error_slots {
        for index in [2usize, 4] {
            if index >= args.len() || (index == 4 && args.len() != 5) {
                continue;
            }
            let arg = args[index];
            if !ctx.trace_ctx.heap_cache().is_class_known(arg) {
                let physical_type = unsafe { (*concrete_args[index]).ob_type } as i64;
                let type_const = ctx.trace_ctx.const_int(physical_type);
                walker_emit_fold_guard_with_snapshot(
                    ctx,
                    op.pc,
                    OpCode::GuardClass,
                    &[arg, type_const],
                )?;
                ctx.trace_ctx
                    .heap_cache_mut()
                    .class_now_known(arg, physical_type);
            }
        }
    }

    // Build `args_w` inline so its wrapper and backing block virtualize
    // alongside the exception.  `w_exception_args_new` pins the object
    // representation at every arity, so this reproduces that one shape
    // instead of picking a layout from the element types.
    let args_list = crate::helpers::emit_object_list_inline(ctx.trace_ctx, final_args);
    // A raised exception can keep args_w live through the execution-context
    // slot, forcing the otherwise-virtual list.  Stamp the canonical list
    // class just as w_list_new does so that materialization preserves the
    // `space.type(args_w) is list` branch used by descr_getargs.
    let list_w_class = pyre_object::get_instantiate(&pyre_object::pyobject::LIST_TYPE);
    let list_w_class = ctx.trace_ctx.const_ref(list_w_class as i64);
    let list_w_class_descr = crate::descr::list_w_class_descr();
    let list_w_class_index = list_w_class_descr.index();
    ctx.trace_ctx.record_op_with_descr(
        OpCode::SetfieldGc,
        &[args_list, list_w_class],
        list_w_class_descr,
    );
    ctx.trace_ctx
        .heapcache_setfield_cached(args_list, list_w_class_index, list_w_class);

    // `W_OSError.descr_new` can retag exact OSError by errno while retaining
    // the OSError physical kind.  The guarded errno makes the concrete final
    // class a valid constant; dedicated classes and subclasses keep the called
    // class operand as in the ordinary constructor emit.
    let emitted_w_class = if fills_os_error_slots && exact_os_error {
        ctx.trace_ctx.const_ref(concrete_w_class as i64)
    } else {
        callable_op
    };
    let new_op =
        crate::helpers::emit_exception_new_inline(ctx.trace_ctx, kind, emitted_w_class, args_list);

    // The slots the constructor defaulted to `None`.  `NewWithVtable` leaves
    // them null, which reads as "unset" rather than `None`, so each one the
    // census collected needs its own store.
    let w_none_const = ctx.trace_ctx.const_ref(w_none as i64);
    for descr in w_none_slot_descrs {
        let descr_index = descr.index();
        ctx.trace_ctx
            .record_op_with_descr(OpCode::SetfieldGc, &[new_op, w_none_const], descr);
        ctx.trace_ctx
            .heapcache_setfield_cached(new_op, descr_index, w_none_const);
    }

    if let Some((direct_code, tuple_shape)) = system_exit_code {
        let code = if let Some((specialised_oo, concrete_code)) = tuple_shape {
            let code = if specialised_oo {
                crate::helpers::emit_specialised_tuple_oo_inline(ctx.trace_ctx, args[0], args[1])
            } else {
                crate::helpers::emit_object_tuple_inline(ctx.trace_ctx, args)
            };
            ctx.trace_ctx.set_opref_concrete(
                code,
                majit_ir::Value::Ref(majit_ir::GcRef(concrete_code as usize)),
            );
            code
        } else {
            direct_code.expect("SystemExit code has neither direct nor tuple value")
        };
        let descr = crate::descr::w_exception_attr_slot_descr(
            kind,
            pyre_interpreter::baseobjspace::ExceptionAttrSlot::Code,
        );
        let descr_index = descr.index();
        ctx.trace_ctx
            .record_op_with_descr(OpCode::SetfieldGc, &[new_op, code], descr);
        ctx.trace_ctx
            .heapcache_setfield_cached(new_op, descr_index, code);
    }

    if fills_os_error_slots {
        use pyre_interpreter::baseobjspace::ExceptionAttrSlot;
        let mut stores = vec![
            (ExceptionAttrSlot::Errno, args[0]),
            (ExceptionAttrSlot::Strerror, args[1]),
        ];
        if has_filename {
            stores.push((ExceptionAttrSlot::Filename, args[2]));
            // The fourth positional argument is winerror and is ignored on
            // non-Windows builds, matching W_OSError._parse_init_args.
            if args.len() == 5 && !unsafe { pyre_object::is_none(concrete_args[4]) } {
                stores.push((ExceptionAttrSlot::Filename2, args[4]));
            }
        }
        for (slot, value) in stores {
            let descr = crate::descr::w_exception_attr_slot_descr(kind, slot);
            let descr_index = descr.index();
            ctx.trace_ctx
                .record_op_with_descr(OpCode::SetfieldGc, &[new_op, value], descr);
            ctx.trace_ctx
                .heapcache_setfield_cached(new_op, descr_index, value);
        }
    }

    // Mark the class known so the following `raise/r` skips its
    // redundant GUARD_CLASS (mirrors the retired raise path's
    // `heapcache.class_now_known`).  The vtable on the NewWithVtable
    // already pins the class for the optimizer; this keeps the heapcache
    // model in agreement.
    ctx.trace_ctx
        .heap_cache_mut()
        .class_now_known(new_op, exc_type_ptr as usize as i64);

    // Record the fresh instance so a following `RaiseVarargs` recovers
    // the concrete and takes the instance fast path; stamp the dst shadow
    // so the `raise/r` GUARD_CLASS reads it.
    ctx.trace_ctx
        .set_opref_concrete(new_op, majit_ir::Value::Ref(majit_ir::GcRef(exc as usize)));
    fbw_built_exc_insert(new_op);
    write_residual_call_result_to_dst(ctx, op.pc, dst, 'r', new_op)?;
    Ok(Some(()))
}

/// Walker-native RAISE_VARARGS inline-built-exception fast path. The
/// `RaiseVarargs` residual is `normalize_raise_varargs_jit(frame, exc,
/// cause)` — `r_args = [frame, exc, cause]`.  When `exc` was built inline by
/// [`try_walker_trace_exception_new`] (∈ [`FBW_BUILT_EXC`]) and there is
/// no explicit `from` cause (concrete `cause` is `PY_NULL`), skip the
/// residual publish + its `GUARD_NOT_FORCED` / `GUARD_NO_EXCEPTION` and
/// emit `__context__` as a `SetfieldGc` on the (still virtual) exception:
///
///   active = GETFIELD_GC_R(ec, sys_exc_value)
///   SETFIELD_GC(exc, active, w_exception.w_context)
///
/// For a fresh exception `w_context` is null and the self-cycle is
/// impossible, so `attach_raise_cause`'s conditional `w_context = active`
/// reduces to the unconditional store (a null store when no exception is
/// active is a no-op that DCEs).  The normalized result is the same
/// instance for a flat builtin, so the inline-built `exc` OpRef is
/// written straight to the dst that fed the following `raise/r`.
///
/// Returns `None` (fall through to the residual) when `exc` was not
/// inline-built or a `from` cause is present.
pub(crate) fn try_walker_trace_raise_builtin<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    code: &[u8],
    op: &DecodedOp,
    r_args: &[OpRef],
    dst: usize,
) -> Result<Option<()>, DispatchError> {
    if r_args.len() != 3 {
        return Ok(None);
    }
    let exc_op = r_args[1];
    // Take (remove) the inline-built marker: a second raise of the same
    // object (whose `w_context` is now stamped) must take the residual
    // path so its runtime `attach_raise_cause` keeps the existing
    // `__context__` and avoids the self-cycle.
    if !fbw_built_exc_take(exc_op) {
        return Ok(None);
    }
    // Explicit `raise X from Y` (concrete non-null cause) keeps the
    // residual: `attach_raise_cause` sets both `__cause__` and
    // `__suppress_context__`, which the inline `__context__` store alone
    // does not reproduce.  Re-insert the marker so the raise still routes
    // through the residual (the marker was consumed above).
    //
    // `raise X` without a cause lowers the cause operand to a const
    // `PY_NULL` (`ConstPtr(GcRef(0))`), whose concrete shadow is
    // `ConcreteValue::Null` (constant pool slots carry no `Ref` shadow);
    // `raise X from Y` passes a live non-null Ref.  Treat the const-null
    // operand AND a `ConcreteValue::Null`/`Ref(null)` shadow all as "no
    // cause"; any concrete non-null Ref is an explicit cause.
    let cause_op = r_args[2];
    let cause_concrete = read_ref_var_list_concrete(code, op, 1, ctx);
    let cause_is_null = match cause_concrete.get(2) {
        Some(ConcreteValue::Ref(p)) => p.is_null(),
        Some(ConcreteValue::Null) | None => {
            // No live concrete: the operand is "no cause" only if it is a
            // const PY_NULL.  A non-const opref with an unknown concrete
            // is conservatively treated as a possible cause (decline).
            matches!(
                ctx.trace_ctx.box_value(cause_op),
                Some(majit_ir::Value::Ref(majit_ir::GcRef(0)))
            )
        }
        _ => false,
    };
    if !cause_is_null {
        fbw_built_exc_insert(exc_op);
        return Ok(None);
    }

    // Recover the concrete exception + kind for the per-kind w_context
    // descr.  Always present (the construct fold stamped the dst shadow).
    let Some(exc) = walker_concrete_ref_object(ctx, exc_op) else {
        // No concrete recovered — re-insert and decline so the residual
        // runs (defensive; should not happen for a construct-fold exc).
        fbw_built_exc_insert(exc_op);
        return Ok(None);
    };
    let kind = unsafe {
        if !pyre_object::is_exception(exc) {
            fbw_built_exc_insert(exc_op);
            return Ok(None);
        }
        pyre_object::interp_exceptions::w_exception_get_kind(exc)
    };

    // --- commit: emit the `__context__` chaining, skip the publish ---
    // active = GETFIELD_GC_R(ec, sys_exc_value).
    //
    // Route the EC through `walker_ensure_execution_context` so the
    // `__context__` read shares the ONE seeded EC OpRef the PUSH_EXC_INFO /
    // POP_EXCEPT exc-info lowering already consumes (`try_walker_lower_exc_
    // info_residual`).  A fresh `GETFIELD_GC_R(frame, execution_context)` here
    // would mint a DISTINCT OpRef from the seeded `input_arg` EC, so the POP
    // `sys_exc_value` store would `possible_aliasing`-mismatch the buffered
    // PUSH store and force it to materialize — keeping the virtual exception
    // escaped and defeating the balanced save/restore dead-store elimination
    // that lets the locally-caught exception DCE.
    let Some(ec) = walker_ensure_execution_context(ctx) else {
        fbw_built_exc_insert(exc_op);
        return Ok(None);
    };
    let active = ctx.trace_ctx.record_op_with_descr(
        OpCode::GetfieldGcR,
        &[ec],
        crate::descr::ec_sys_exc_value_descr(),
    );
    ctx.trace_ctx.record_op_with_descr(
        OpCode::SetfieldGc,
        &[exc_op, active],
        crate::descr::w_exception_context_descr(kind),
    );
    fbw_context_chained_insert(exc_op);
    // The full-body walk is also the authoritative execution of the
    // tracing iteration.  Apply the same context write to its concrete,
    // freshly-built exception that the recorded SETFIELD performs on later
    // compiled iterations; otherwise Python code reached later in this walk
    // observes a missing __context__ exactly once, while the trace itself is
    // correct.  This object is private to the inline construction, so no
    // rollback journal is needed.
    let active_concrete = pyre_interpreter::eval::get_current_exception();
    if !active_concrete.is_null() {
        unsafe {
            pyre_object::interp_exceptions::w_exception_set_context(exc, active_concrete);
        }
    }

    // The normalized publish result is the same flat builtin instance;
    // forward the inline-built exc OpRef (carrying its concrete shadow)
    // to the dst that feeds the following `raise/r`.
    fbw_built_exc_insert(exc_op);
    write_residual_call_result_to_dst(ctx, op.pc, dst, 'r', exc_op)?;
    Ok(Some(()))
}

/// Walker-native fold for a bare-class `raise Type`
/// (no call parentheses).  Unlike `raise Type()`, a bare class has no
/// preceding `CallFn` construct residual — `normalize_raise_varargs_jit`
/// instantiates the class itself — so no virtualizable `NewWithVtable`
/// exists and `try_walker_trace_raise_builtin` declines it to the residual
/// (a per-iteration heap alloc + may-force).
///
/// `do_raise` instantiates a raised class with no arguments, so a bare
/// `raise ValueError` is `raise ValueError()`.  When the operand is a
/// canonical builtin exception class whose concrete zero-argument instance
/// can be reproduced exactly and with no explicit `from` cause, build it
/// inline using the `try_walker_trace_exception_new` Empty-args shape and chain
/// `__context__` (the `try_walker_trace_raise_builtin` tail), so the whole
/// exception virtualizes and DCEs when it never escapes.  A subclass or an
/// instance carrying any other pointer-slot value declines to the residual.
///
/// Returns `None` (fall through to the generic residual) for any
/// non-matching shape.
pub(crate) fn try_walker_trace_raise_bare_class<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    code: &[u8],
    op: &DecodedOp,
    r_args: &[OpRef],
    dst: usize,
) -> Result<Option<()>, DispatchError> {
    if r_args.len() != 3 {
        return Ok(None);
    }
    let class_op = r_args[1];
    // The residual arg concretes are `[frame, exc, cause]`.  Recover the live
    // exception operand (index 1) from the residual list rather than the opref
    // shadow: a bare class comes straight from `LOAD_GLOBAL`, not the inline
    // construct fold, so its shadow is not stamped.
    let arg_concretes = read_ref_var_list_concrete(code, op, 1, ctx);
    let Some(ConcreteValue::Ref(concrete_class)) = arg_concretes.get(1).copied() else {
        return Ok(None);
    };
    // The operand must be a canonical builtin exception CLASS.  An already
    // built instance (`raise ValueError()`) is not in the class registry and
    // is handled by `try_walker_trace_raise_builtin`; a non-exception operand
    // (`raise obj`) also declines here.
    if concrete_class.is_null()
        || !pyre_object::interp_exceptions::is_canonical_exc_class(concrete_class)
    {
        return Ok(None);
    }

    // Explicit `raise X from Y` keeps the residual: `attach_raise_cause` sets
    // `__cause__` and `__suppress_context__`, which the inline `__context__`
    // store alone does not reproduce.  The cause operand is a const `PY_NULL`
    // (or a `Null` / `Ref(null)` shadow) when there is no cause; any concrete
    // non-null Ref is an explicit cause.
    let cause_op = r_args[2];
    let cause_is_null = match arg_concretes.get(2) {
        Some(ConcreteValue::Ref(p)) => p.is_null(),
        Some(ConcreteValue::Null) | None => matches!(
            ctx.trace_ctx.box_value(cause_op),
            Some(majit_ir::Value::Ref(majit_ir::GcRef(0)))
        ),
        _ => false,
    };
    if !cause_is_null {
        return Ok(None);
    }

    // Build the exception concretely on the plain eval loop (no tracer
    // re-entry) to read its kind and confirm a flat builtin instance.  A
    // canonical class has the builtin `descr_new` / `descr_init`, so a
    // zero-argument construction runs no user code.  Trace-time only.
    let exc = {
        let _plain_guard = pyre_interpreter::call::force_plain_eval();
        pyre_interpreter::call::call_function_impl_result(concrete_class, &[])
    };
    let Ok(exc) = exc else { return Ok(None) };
    let kind = unsafe {
        if !pyre_object::is_exception(exc) {
            return Ok(None);
        }
        pyre_object::interp_exceptions::w_exception_get_kind(exc)
    };
    if pyre_object::interp_exceptions::lookup_exc_class_for_kind(kind) != concrete_class {
        return Ok(None);
    }
    let exc_type_ptr = unsafe {
        (*(exc as *const pyre_object::interp_exceptions::W_BaseException))
            .ob_header
            .ob_type
    };
    if !std::ptr::eq(
        exc_type_ptr,
        pyre_object::interp_exceptions::exc_kind_to_pytype(kind),
    ) {
        return Ok(None);
    }

    let w_none = pyre_object::w_none();
    let mut w_none_slot_descrs = Vec::new();
    for (offset, value) in
        unsafe { pyre_object::interp_exceptions::w_exception_traced_construction_slots(exc) }
    {
        if value.is_null() {
            continue;
        }
        if !std::ptr::eq(value, w_none) {
            return Ok(None);
        }
        let Some(descr) = crate::descr::w_exception_slot_descr(kind, offset) else {
            return Ok(None);
        };
        w_none_slot_descrs.push(descr);
    }

    // Resolve the EC while declining is still free.  `walker_ensure_execution_
    // context` returns `None` on a null snapshot sym or a frameless walk, and
    // its recovery records a `GETFIELD_GC_R` that must not land after a guard
    // referencing it — `ensure_execution_context` recovers eagerly at walk
    // entry for that reason.  A decline past the commit below would also leave
    // the construction ops orphaned and the heap-cache shadows describing an
    // object the caller's generic-residual fall-through never built.
    let Some(ec) = walker_ensure_execution_context(ctx) else {
        return Ok(None);
    };

    // --- commit: pin the class identity, emit the construction + raise ---
    // Guard the class operand so the trace-time kind / vtable stay valid
    // across iterations (`implement_guard_value`).
    if !class_op.is_constant() {
        let expected = ctx.trace_ctx.const_ref(concrete_class as i64);
        ctx.trace_ctx
            .record_guard(OpCode::GuardValue, &[class_op, expected], 0);
        walker_capture_snapshot_for_last_guard(ctx, op.pc)?;
        ctx.trace_ctx
            .heap_cache_mut()
            .replace_box(class_op, expected);
    }

    // Empty `args_w` list (zero-argument construction), stamped with the
    // canonical list class exactly as `w_list_new` does.
    let args_list = crate::helpers::emit_object_list_inline(ctx.trace_ctx, &[]);
    let list_w_class = pyre_object::get_instantiate(&pyre_object::pyobject::LIST_TYPE);
    let list_w_class = ctx.trace_ctx.const_ref(list_w_class as i64);
    let list_w_class_descr = crate::descr::list_w_class_descr();
    let list_w_class_index = list_w_class_descr.index();
    ctx.trace_ctx.record_op_with_descr(
        OpCode::SetfieldGc,
        &[args_list, list_w_class],
        list_w_class_descr,
    );
    ctx.trace_ctx
        .heapcache_setfield_cached(args_list, list_w_class_index, list_w_class);

    let new_op =
        crate::helpers::emit_exception_new_inline(ctx.trace_ctx, kind, class_op, args_list);
    let w_none_const = ctx.trace_ctx.const_ref(w_none as i64);
    for descr in w_none_slot_descrs {
        let descr_index = descr.index();
        ctx.trace_ctx
            .record_op_with_descr(OpCode::SetfieldGc, &[new_op, w_none_const], descr);
        ctx.trace_ctx
            .heapcache_setfield_cached(new_op, descr_index, w_none_const);
    }
    ctx.trace_ctx
        .heap_cache_mut()
        .class_now_known(new_op, exc_type_ptr as usize as i64);
    ctx.trace_ctx
        .set_opref_concrete(new_op, majit_ir::Value::Ref(majit_ir::GcRef(exc as usize)));

    // `__context__` chaining on the still-virtual exception, mirroring the
    // `try_walker_trace_raise_builtin` tail: `active = GETFIELD_GC_R(ec,
    // sys_exc_value)` then `SETFIELD_GC(exc, active, w_context)`.  `ec` came
    // from `walker_ensure_execution_context` above, so the read shares the one
    // seeded EC OpRef the PUSH_EXC_INFO / POP_EXCEPT lowering consumes.
    let active = ctx.trace_ctx.record_op_with_descr(
        OpCode::GetfieldGcR,
        &[ec],
        crate::descr::ec_sys_exc_value_descr(),
    );
    ctx.trace_ctx.record_op_with_descr(
        OpCode::SetfieldGc,
        &[new_op, active],
        crate::descr::w_exception_context_descr(kind),
    );
    fbw_context_chained_insert(new_op);
    // Apply the same context write to the concrete, freshly-built exception so
    // Python code reached later in this authoritative walk observes the
    // `__context__` the recorded SETFIELD performs on compiled iterations.
    let active_concrete = pyre_interpreter::eval::get_current_exception();
    if !active_concrete.is_null() {
        unsafe {
            pyre_object::interp_exceptions::w_exception_set_context(exc, active_concrete);
        }
    }

    // Mark the inline-built instance FBW-built so the following `raise/r`
    // records its frame node via the virtual `record_fresh_application_
    // traceback` (an inline PyTraceback `NewWithVtable` + SETFIELDs on the
    // exception) rather than `record_top_level_application_traceback`, whose
    // runtime hook passes the exception to a `CallN` and forces it to
    // materialize — defeating the save/restore DCE that virtualizes a
    // locally-caught raise.  Mirrors `try_walker_trace_raise_builtin`.
    fbw_built_exc_insert(new_op);
    write_residual_call_result_to_dst(ctx, op.pc, dst, 'r', new_op)?;
    Ok(Some(()))
}

/// Walker-native fold for the deterministic immutable-type
/// STORE_ATTR / DELETE_ATTR raise (`int.x = v` / `del str.x` →
/// TypeError from the `object_setattr` / `object_delattr` non-heaptype
/// guard, `typeobject.py:416/437`).
///
/// The generic path records the raise as an opaque
/// `CallMayForceN(bh_store_attr_fn)` + `GuardNotForced` +
/// `GuardException`, whose result box (the exception materialised
/// *inside* the residual by `PyError::to_exc_object`) can never
/// virtualize — every compiled iteration re-allocates the TypeError,
/// its message string, and its args list through the runtime GC hooks.
/// PyPy traces `space.setattr` itself, so the same raise shows up as
/// `new_with_vtable` + `setfield_gc` ops its optimizer removes when the
/// exception never escapes.
///
/// This fold restores that shape for the one attribute-store raise
/// whose outcome is provably iteration-invariant: when
/// `type_immutable_attr_raise_is_stable` holds (constant non-heaptype
/// receiver, canonical `type` metaclass, no metaclass descriptor for
/// `name` — every consulted dict frozen), the raise and its message
/// depend only on trace-time constants.  Pin the receiver with
/// `GuardValue` (when not already constant), run the authentic
/// `setattr_str` / `delattr_str` concretely for the authoritative
/// walk's exception, and emit the [`try_walker_trace_exception_new`]
/// construction (`NewWithVtable` + args-list `SetfieldGc`s, message as
/// a rooted trace constant) in place of the residual + guards.  The
/// raise then routes through the ordinary `SubRaise` path with a
/// virtualizable exception OpRef, and a locally-caught `except` DCEs
/// the whole allocation exactly as the explicit-`raise` fold does.
///
/// Returns `None` (fall through to the generic residual) for any
/// non-matching or unprovable shape.
pub(crate) fn try_walker_trace_immutable_type_attr_raise<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op: &DecodedOp,
    obj_op: OpRef,
    store_value: Option<OpRef>,
    w_code_ptr: usize,
    name_idx: usize,
) -> Result<Option<(DispatchOutcome, usize)>, DispatchError> {
    if !ctx.is_authoritative_executor || w_code_ptr == 0 {
        return Ok(None);
    }
    let Some(concrete_obj) = walker_concrete_ref_object(ctx, obj_op) else {
        return Ok(None);
    };
    // STORE_ATTR runs the authentic concrete `setattr_str` below only for
    // its raising exception.  The value plays no role in that raise — the
    // stability predicate proves no data descriptor for `name`, so the
    // terminal raises before consulting the value — so a non-constant store
    // value (the common `int.x = i` loop case) still folds: a `None`
    // concrete value substitutes a placeholder for the authentic run, and
    // the value operand never enters the emitted trace.
    let concrete_value = match store_value {
        Some(value_op) => {
            Some(walker_concrete_ref_object(ctx, value_op).unwrap_or_else(pyre_object::w_none))
        }
        None => None,
    };
    let name = unsafe {
        let code_ptr = pyre_interpreter::w_code_get_ptr(w_code_ptr as pyre_object::PyObjectRef);
        if code_ptr.is_null() {
            return Ok(None);
        }
        let code = &*(code_ptr as *const pyre_interpreter::CodeObject);
        match pyre_interpreter::pyframe::load_name_from_code(code, name_idx) {
            Some(n) => n.to_string(),
            None => return Ok(None),
        }
    };
    if !pyre_interpreter::baseobjspace::type_immutable_attr_raise_is_stable(
        concrete_obj,
        &name,
        store_value.is_none(),
    ) {
        return Ok(None);
    }

    // The raise decision also reads the metaclass-MRO descriptor state — the
    // branch-F `lookup_in_type_where(type, name)` walk and the forwarding
    // `type.__setattr__` / `type.__delattr__` — which the receiver
    // `GuardValue` below does not cover.  A `version_tag` guard on the
    // metaclass pins that state (the guard the sibling method/attr folds
    // carry, `typeobject.py promote(self.version_tag())`): mutating `type`'s
    // dict bumps its tag directly, and mutating `object`'s dict bumps it too
    // because `mutated()` propagates down to the `type` subclass — so one
    // guard covers the whole `(type, object)` MRO the walk reads.  Branch C
    // proved the metaclass is the canonical `type`.  A tagless metaclass
    // (`version_tag == 0`) is uncacheable, so decline before emitting guards.
    let metaclass = pyre_object::get_instantiate(&pyre_object::pyobject::TYPE_TYPE);
    let metaclass_version_tag =
        unsafe { pyre_object::typeobject::w_type_get_version_tag(metaclass) };
    if metaclass_version_tag == 0 {
        return Ok(None);
    }

    // Resolve the EC while declining is still free, for the `__context__` tail
    // below.  `walker_ensure_execution_context` returns `None` on a null
    // snapshot sym or a frameless walk, and its recovery records a
    // `GETFIELD_GC_R` that must not land after a guard referencing it
    // (`try_walker_trace_raise_bare_class` resolves it at the same boundary).
    let Some(ec) = walker_ensure_execution_context(ctx) else {
        return Ok(None);
    };

    // --- commit: pin the receiver, run the authentic raise, emit inline ---
    // The stability predicate makes the raise a pure function of `(obj,
    // name)`; `GuardValue` pins the one live input (`name` is a co_names
    // constant).
    if !obj_op.is_constant() {
        let expected = ctx.trace_ctx.const_ref(concrete_obj as i64);
        ctx.trace_ctx
            .record_guard(OpCode::GuardValue, &[obj_op, expected], 0);
        walker_capture_snapshot_for_last_guard(ctx, op.pc)?;
        ctx.trace_ctx.heap_cache_mut().replace_box(obj_op, expected);
    }
    // Pin the metaclass `version_tag` (see above): a `GETFIELD_GC_I` +
    // `GuardValue` that side-exits on any `type`/`object` dict mutation.
    let metaclass_const = ctx.trace_ctx.const_ref(metaclass as i64);
    let vt_op = walker_record_getfield_gc_i_uncached(
        ctx,
        metaclass_const,
        crate::descr::type_version_tag_descr(),
    );
    let vt_const = ctx.trace_ctx.const_int(metaclass_version_tag as i64);
    ctx.trace_ctx
        .record_guard(OpCode::GuardValue, &[vt_op, vt_const], 0);
    walker_capture_snapshot_for_last_guard(ctx, op.pc)?;
    ctx.trace_ctx.heap_cache_mut().replace_box(vt_op, vt_const);

    // The authoritative walk's concrete execution — the same call the
    // residual executor would have made, raising before any heap
    // mutation.  Plain eval: the predicate excludes every user-code path.
    let result = {
        let _plain_guard = pyre_interpreter::call::force_plain_eval();
        match concrete_value {
            Some(value) => pyre_interpreter::baseobjspace::setattr_str(concrete_obj, &name, value),
            None => pyre_interpreter::baseobjspace::delattr_str(concrete_obj, &name),
        }
    };
    let Err(mut err) = result else {
        // Unreachable under the predicate (a non-heaptype dict rejects
        // every mutation).  Fail loud rather than falling through: the
        // generic residual would re-run the (somehow) committed effect.
        return Err(DispatchError::UnsupportedOpname {
            pc: op.pc,
            key: "immutable-type attr raise fold: stable raise unexpectedly succeeded",
        });
    };
    let exc = err.to_exc_object();
    let kind = unsafe {
        if !pyre_object::is_exception(exc) {
            return Ok(None);
        }
        pyre_object::interp_exceptions::w_exception_get_kind(exc)
    };
    // The folded raise is exactly the immutable-type TypeError; any other
    // kind means the runtime path diverged from the predicate's model.
    if kind != pyre_object::interp_exceptions::ExcKind::TypeError {
        return Ok(None);
    }
    let exc_type_ptr = unsafe {
        (*(exc as *const pyre_object::interp_exceptions::W_BaseException))
            .ob_header
            .ob_type
    };
    if !std::ptr::eq(
        exc_type_ptr,
        pyre_object::interp_exceptions::exc_kind_to_pytype(kind),
    ) {
        return Ok(None);
    }

    // Message as a trace constant: deterministic per `(obj, name)` under
    // the predicate, so one shared immutable string is exact (the same
    // sharing a `raise TypeError("...")` gets from co_consts).  Pin the
    // fresh exception across the string allocation; the recorded ConstPtr
    // slot is forwarded across minor collections by the op-graph walker
    // and rooted by the compiled loop's gcref table thereafter.
    let _roots = pyre_object::gc_roots::push_roots();
    let exc_root = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(exc);
    let msg = pyre_object::w_str_from_wtf8(err.message.clone());
    // The root keeps the exception alive across that allocation but does not
    // fix its address: a minor collection moves the object and rewrites the
    // slot, which leaves this local naming a forwarded corpse.  Read the
    // address back out of the slot the pin claimed.
    let exc = pyre_object::gc_roots::shadow_stack_get(exc_root);
    let msg_const = ctx.trace_ctx.const_ref(msg as i64);
    let args_list = crate::helpers::emit_object_list_inline(ctx.trace_ctx, &[msg_const]);
    // Stamp the canonical list class exactly as `w_list_new` does (the
    // `try_walker_trace_exception_new` args tail), so a materialised
    // `args_w` still satisfies `space.type(args_w) is list`.
    let list_w_class = pyre_object::get_instantiate(&pyre_object::pyobject::LIST_TYPE);
    let list_w_class = ctx.trace_ctx.const_ref(list_w_class as i64);
    let list_w_class_descr = crate::descr::list_w_class_descr();
    let list_w_class_index = list_w_class_descr.index();
    ctx.trace_ctx.record_op_with_descr(
        OpCode::SetfieldGc,
        &[args_list, list_w_class],
        list_w_class_descr,
    );
    ctx.trace_ctx
        .heapcache_setfield_cached(args_list, list_w_class_index, list_w_class);

    let class_const = ctx
        .trace_ctx
        .const_ref(pyre_object::interp_exceptions::lookup_exc_class_for_kind(kind) as i64);
    let new_op =
        crate::helpers::emit_exception_new_inline(ctx.trace_ctx, kind, class_const, args_list);
    ctx.trace_ctx
        .heap_cache_mut()
        .class_now_known(new_op, exc_type_ptr as usize as i64);
    ctx.trace_ctx
        .set_opref_concrete(new_op, majit_ir::Value::Ref(majit_ir::GcRef(exc as usize)));

    // `__context__` chaining on the still-virtual exception, the tail
    // `try_walker_trace_raise_bare_class` carries: `active = GETFIELD_GC_R(ec,
    // sys_exc_value)` then `SETFIELD_GC(exc, active, w_context)`.  Without it
    // the catch-side `record_inline_exception_context` compensation finds the
    // context unchained and passes this exception to the resolver call, which
    // forces the very allocation this fold exists to keep virtual.
    let active = ctx.trace_ctx.record_op_with_descr(
        OpCode::GetfieldGcR,
        &[ec],
        crate::descr::ec_sys_exc_value_descr(),
    );
    ctx.trace_ctx.record_op_with_descr(
        OpCode::SetfieldGc,
        &[new_op, active],
        crate::descr::w_exception_context_descr(kind),
    );
    fbw_context_chained_insert(new_op);
    // Apply the same context write to the concrete exception, which the
    // registration above stops the compensation from performing, so Python
    // code reached later in this authoritative walk observes the
    // `__context__` the recorded SETFIELD performs on compiled iterations.
    let active_concrete = pyre_interpreter::eval::get_current_exception();
    if !active_concrete.is_null() {
        unsafe {
            pyre_object::interp_exceptions::w_exception_set_context(exc, active_concrete);
        }
    }

    // Inline-built marker: the downstream raise routing records the frame
    // node via the virtual `record_fresh_application_traceback` instead of
    // the forcing runtime hook (mirrors `try_walker_trace_raise_bare_class`).
    fbw_built_exc_insert(new_op);

    // The residual-executor Err-arm state, minus the call itself: seed the
    // standing exception for the `SubRaise` routing (`execute_raised`
    // analogue) and restore the blackhole cell so an aborting walk still
    // delivers the pending raise to the live frame.  The class IS proven
    // constant here — the `NewWithVtable` vtable pins it.
    fbw_count_executed_residual(true, true);
    ctx.set_last_exc_value(new_op, ConcreteValue::Ref(exc));
    ctx.fbw_mode.class_of_last_exc_is_const = true;
    majit_metainterp::blackhole::BH_LAST_EXC_VALUE.with(|c| c.set(exc as i64));

    Ok(Some((
        DispatchOutcome::SubRaise {
            exc: new_op,
            exc_concrete: ConcreteValue::Ref(exc),
        },
        op.next_pc,
    )))
}

/// Walker-native fold for the read-only-data-descriptor STORE_ATTR raise.
///
/// `objspace.py:723-739` and `descroperation.py:114-126` raise
/// AttributeError after resolving a descriptor with no `__set__` and a
/// reachable `__delete__`.  The interpreter predicate excludes every shortcut
/// and user-code branch; class-version guards pin the two MRO lookups, while
/// the descriptor type's `w_name` guard pins the rendered message across a
/// `type.__name__` assignment that does not change its version tag.
pub(crate) fn try_walker_trace_readonly_descr_attr_raise<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op: &DecodedOp,
    obj_op: OpRef,
    value_op: OpRef,
    w_code_ptr: usize,
    name_idx: usize,
) -> Result<Option<(DispatchOutcome, usize)>, DispatchError> {
    if !ctx.is_authoritative_executor || w_code_ptr == 0 {
        return Ok(None);
    }
    let Some(concrete_obj) = walker_concrete_ref_object(ctx, obj_op) else {
        return Ok(None);
    };
    let concrete_value =
        walker_concrete_ref_object(ctx, value_op).unwrap_or_else(pyre_object::w_none);
    let name = unsafe {
        let code_ptr = pyre_interpreter::w_code_get_ptr(w_code_ptr as pyre_object::PyObjectRef);
        if code_ptr.is_null() {
            return Ok(None);
        }
        let code = &*(code_ptr as *const pyre_interpreter::CodeObject);
        match pyre_interpreter::pyframe::load_name_from_code(code, name_idx) {
            Some(n) => n.to_string(),
            None => return Ok(None),
        }
    };
    let Some(descr) =
        pyre_interpreter::baseobjspace::readonly_descr_attr_raise_is_stable(concrete_obj, &name)
    else {
        return Ok(None);
    };

    let w_type = unsafe { pyre_object::w_instance_get_type(concrete_obj) };
    let Some(descr_type) = (unsafe { pyre_interpreter::typedef::r#type(descr) }) else {
        return Ok(None);
    };
    let descr_type = descr_type.as_ptr();
    let w_type_version_tag = unsafe { pyre_object::w_type_get_version_tag(w_type) };
    let descr_type_version_tag = unsafe { pyre_object::w_type_get_version_tag(descr_type) };
    if w_type_version_tag == 0 || descr_type_version_tag == 0 {
        return Ok(None);
    }
    let descr_type_w_name = unsafe { pyre_object::typeobject::w_type_peek_name_obj(descr_type) };

    // Resolve the execution context while declining is still effect-free.  Its
    // recovery may record an op, which must precede the fold's guard sequence.
    let Some(ec) = walker_ensure_execution_context(ctx) else {
        return Ok(None);
    };

    // --- commit: pin both MRO decisions, run the authentic raise, emit inline ---
    // GuardClass pins the receiver payload without pinning its identity.
    let physical_type = unsafe { (*concrete_obj).ob_type } as i64;
    let physical_type_const = ctx.trace_ctx.const_int(physical_type);
    walker_emit_fold_guard_with_snapshot(
        ctx,
        op.pc,
        OpCode::GuardClass,
        &[obj_op, physical_type_const],
    )?;
    ctx.trace_ctx
        .heap_cache_mut()
        .class_now_known(obj_op, physical_type);

    // `typeobject.py` promotes the version tag before an MRO lookup.
    // Pinning the receiver type covers both the named descriptor resolution
    // and the default-`__setattr__` answer.
    let w_type_const = ctx.trace_ctx.const_ref(w_type as i64);
    let w_type_vt_op = walker_record_getfield_gc_i_uncached(
        ctx,
        w_type_const,
        crate::descr::type_version_tag_descr(),
    );
    let w_type_vt_const = ctx.trace_ctx.const_int(w_type_version_tag as i64);
    ctx.trace_ctx
        .record_guard(OpCode::GuardValue, &[w_type_vt_op, w_type_vt_const], 0);
    walker_capture_snapshot_for_last_guard(ctx, op.pc)?;
    ctx.trace_ctx
        .heap_cache_mut()
        .replace_box(w_type_vt_op, w_type_vt_const);

    // The descriptor type's tag pins its general `__set__` / `__delete__` MRO
    // answers (`descroperation.py:117-125`).
    let descr_type_const = ctx.trace_ctx.const_ref(descr_type as i64);
    let descr_type_vt_op = walker_record_getfield_gc_i_uncached(
        ctx,
        descr_type_const,
        crate::descr::type_version_tag_descr(),
    );
    let descr_type_vt_const = ctx.trace_ctx.const_int(descr_type_version_tag as i64);
    ctx.trace_ctx.record_guard(
        OpCode::GuardValue,
        &[descr_type_vt_op, descr_type_vt_const],
        0,
    );
    walker_capture_snapshot_for_last_guard(ctx, op.pc)?;
    ctx.trace_ctx
        .heap_cache_mut()
        .replace_box(descr_type_vt_op, descr_type_vt_const);

    // `typeobject.py:1046-1058` rewrites `w_name` without mutating the class
    // dictionary or its version tag.  Pin the raw slot, including its initial
    // null state, because it shadows the type name rendered in the message.
    let descr_type_name_op = ctx.trace_ctx.record_op_with_descr(
        OpCode::GetfieldGcR,
        &[descr_type_const],
        crate::descr::type_name_obj_descr(),
    );
    let descr_type_name_const = ctx.trace_ctx.const_ref(descr_type_w_name as i64);
    ctx.trace_ctx.record_guard(
        OpCode::GuardValue,
        &[descr_type_name_op, descr_type_name_const],
        0,
    );
    walker_capture_snapshot_for_last_guard(ctx, op.pc)?;
    ctx.trace_ctx
        .heap_cache_mut()
        .replace_box(descr_type_name_op, descr_type_name_const);

    // Pyre stores the Python-visible class separately from the physical class
    // GuardClass reads.  Pin that class after the mandated MRO/name guard
    // sequence; unlike GuardValue on `obj_op`, this still accepts every
    // receiver of the same class and ties `w_type_const` to the receiver.
    walker_guard_exact_w_class(ctx, op.pc, obj_op, w_type)?;

    let result = {
        let _plain_guard = pyre_interpreter::call::force_plain_eval();
        pyre_interpreter::baseobjspace::setattr_str(concrete_obj, &name, concrete_value)
    };
    let Err(mut err) = result else {
        // The concrete store has already run, so falling through would execute
        // it twice.  The predicate promises the descriptor terminal instead.
        return Err(DispatchError::UnsupportedOpname {
            pc: op.pc,
            key: "read-only descriptor attr raise fold: stable raise unexpectedly succeeded",
        });
    };
    let exc = err.to_exc_object();
    let kind = unsafe {
        if !pyre_object::is_exception(exc) {
            return Ok(None);
        }
        pyre_object::interp_exceptions::w_exception_get_kind(exc)
    };
    if kind != pyre_object::interp_exceptions::ExcKind::AttributeError {
        return Ok(None);
    }
    let exc_type_ptr = unsafe {
        (*(exc as *const pyre_object::interp_exceptions::W_BaseException))
            .ob_header
            .ob_type
    };
    if !std::ptr::eq(
        exc_type_ptr,
        pyre_object::interp_exceptions::exc_kind_to_pytype(kind),
    ) {
        return Ok(None);
    }

    let _roots = pyre_object::gc_roots::push_roots();
    let exc_root = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(exc);
    let msg = pyre_object::w_str_from_wtf8(err.message.clone());
    // The allocation may move the exception and leave the local pointer naming
    // its forwarded corpse; the shadow slot contains the live address.
    let exc = pyre_object::gc_roots::shadow_stack_get(exc_root);
    let msg_const = ctx.trace_ctx.const_ref(msg as i64);
    let args_list = crate::helpers::emit_object_list_inline(ctx.trace_ctx, &[msg_const]);
    let list_w_class = pyre_object::get_instantiate(&pyre_object::pyobject::LIST_TYPE);
    let list_w_class = ctx.trace_ctx.const_ref(list_w_class as i64);
    let list_w_class_descr = crate::descr::list_w_class_descr();
    let list_w_class_index = list_w_class_descr.index();
    ctx.trace_ctx.record_op_with_descr(
        OpCode::SetfieldGc,
        &[args_list, list_w_class],
        list_w_class_descr,
    );
    ctx.trace_ctx
        .heapcache_setfield_cached(args_list, list_w_class_index, list_w_class);

    let class_const = ctx
        .trace_ctx
        .const_ref(pyre_object::interp_exceptions::lookup_exc_class_for_kind(kind) as i64);
    let new_op =
        crate::helpers::emit_exception_new_inline(ctx.trace_ctx, kind, class_const, args_list);
    ctx.trace_ctx
        .heap_cache_mut()
        .class_now_known(new_op, exc_type_ptr as usize as i64);
    ctx.trace_ctx
        .set_opref_concrete(new_op, majit_ir::Value::Ref(majit_ir::GcRef(exc as usize)));

    let active = ctx.trace_ctx.record_op_with_descr(
        OpCode::GetfieldGcR,
        &[ec],
        crate::descr::ec_sys_exc_value_descr(),
    );
    ctx.trace_ctx.record_op_with_descr(
        OpCode::SetfieldGc,
        &[new_op, active],
        crate::descr::w_exception_context_descr(kind),
    );
    fbw_context_chained_insert(new_op);
    let active_concrete = pyre_interpreter::eval::get_current_exception();
    if !active_concrete.is_null() {
        unsafe {
            pyre_object::interp_exceptions::w_exception_set_context(exc, active_concrete);
        }
    }

    fbw_built_exc_insert(new_op);
    fbw_count_executed_residual(true, true);
    ctx.set_last_exc_value(new_op, ConcreteValue::Ref(exc));
    ctx.fbw_mode.class_of_last_exc_is_const = true;
    majit_metainterp::blackhole::BH_LAST_EXC_VALUE.with(|c| c.set(exc as i64));

    Ok(Some((
        DispatchOutcome::SubRaise {
            exc: new_op,
            exc_concrete: ConcreteValue::Ref(exc),
        },
        op.next_pc,
    )))
}

/// Lower the PUSH_EXC_INFO / POP_EXCEPT
/// exc-info-stack residuals to GETFIELD_GC_R / SETFIELD_GC on the EC's
/// `sys_exc_value` slot (`ec_sys_exc_value_descr`), and consume pyre's
/// propagation-root clear without recording a runtime call.
/// Recognised by the codewriter-stamped `runtime_helper` tag, NOT a funcptr
/// address (the residual calls the cross-crate `cpu.{get,set}_current_
/// exception_fn` wrappers in `pyre-jit`, which `pyre-jit-trace` cannot name).
///
///   * `GetCurrentException` — `get_current_exception()` (`[]→Ref`,
///     dst_bank `'r'`): the PUSH_EXC_INFO `prev` save, and also the read a
///     catch-covered bare `raise` uses to obtain the exception it re-raises.
///     Only the first owns a matching store and POP_EXCEPT restore, so only
///     the first pushes onto the saved-prev stack.  Emit
///     `GETFIELD_GC_R(ec, sys_exc_value)`, stamp the live `prev` concrete
///     (the residual executor would have returned it) so a downstream read
///     of the dst sees the right value.
///   * `SetCurrentException` — `set_current_exception(exc)` (`[Ref]→void`,
///     dst_bank `'v'`): the PUSH_EXC_INFO store and the POP_EXCEPT restore.
///     Emit `SETFIELD_GC(ec, exc, sys_exc_value)` and apply the concrete
///     write the authoritative walk's residual executor would have done.
///   * `ClearInFlightException` — `set_in_flight_exception(PY_NULL)`
///     (`[]→void`, dst_bank `'v'`): apply the clear to the authoritative
///     recording walk, but emit no IR.  PyPy keeps the propagating exception
///     in the local `OperationError` and PUSH_EXC_INFO transfers it directly
///     to `ExecutionContext.sys_exc_operror` (`pyopcode.py:123-185, 836-863`),
///     so there is no equivalent residual clear in its compiled trace.  Pyre's
///     extra TLS carrier only exposes the Rust `PyError`'s GC children while
///     the interpreter unwinds.  The walk's inline traceback construction
///     never publishes that carrier at compiled runtime; leaving its clear as
///     a CallN would therefore execute an unmatched TLS write on every caught
///     exception.
///
/// A balanced save (`GETFIELD`) + store + restore (`SETFIELD`) on the same
/// descr-identity field with no intervening read is dead-store-eliminated,
/// so a non-escaping exception virtualizes and DCEs (no per-raise
/// `CallMallocNursery`).  Declines (`None` → generic residual) when the EC
/// cannot be recovered or the operand shape does not match (SAFE).
pub(crate) fn try_walker_lower_exc_info_residual<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    code: &[u8],
    op: &DecodedOp,
    runtime_helper: majit_ir::RuntimeHelperKind,
    r_args: &[OpRef],
    dst_bank: char,
    dst: usize,
) -> Result<Option<()>, DispatchError> {
    if runtime_helper == majit_ir::RuntimeHelperKind::ClearInFlightException {
        // The authoritative walk executed record_application_traceback and
        // published its concrete exception in the interpreter-only carrier.
        // Complete that concrete ownership transfer now.  Compiled traceback
        // recording is emitted as GC IR and never publishes the carrier, so
        // there is deliberately no corresponding runtime operation to record.
        if !r_args.is_empty() || dst_bank != 'v' {
            return Ok(None);
        }
        pyre_interpreter::eval::set_in_flight_exception(pyre_object::PY_NULL);
        return Ok(Some(()));
    }

    if runtime_helper == majit_ir::RuntimeHelperKind::GetCurrentException {
        // PUSH_EXC_INFO `prev = ec.sys_exc_value` — `[]→Ref`.
        if !r_args.is_empty() || dst_bank != 'r' {
            return Ok(None);
        }
        // Two Python instructions lower to this helper, and they want
        // different things from a bridge seed.  A bare `raise` wants
        // the exception the bridge is resuming with — the compiled loop is free
        // to elide its `sys_exc_value` store (a balanced save/store/restore
        // DCEs), so the live slot is not a source there and only the seed
        // names the exception to re-raise.  `PUSH_EXC_INFO`'s `prev` save wants
        // the field itself, and at a bridge that resumes AT the handler the
        // seed is the exception this opcode is two ops away from publishing:
        // saving it as `prev` makes the matching `POP_EXCEPT` reinstate the
        // exception the handler just finished with.  Read the live slot for
        // that one — `_prepare_pendingfields` (state.rs, its `execute` block)
        // runs every decoded pending write through `bh_setfield_gc_r` at bridge
        // entry, so the slot is current.  A seed this walk stored itself is a
        // view of the field either way, and reusing its OpRef keeps the
        // save/store/restore triple balanced.
        // The predicate is true for `RAISE_VARARGS 0`, `RERAISE` and `FOR_ITER`,
        // but only the first can reach here: `RERAISE` reads its exception off
        // the vable stack and `FOR_ITER` re-raises the value its own
        // `catch_exception` caught, so neither emits this helper.  The name
        // records the one shape that does.
        let is_covered_bare_raise_read =
            super::recording_raise_keeps_existing_traceback(ctx, op.pc);
        let seed_answers_this_read =
            ctx.fbw_mode.current_exception_seed_from_walk_store || is_covered_bare_raise_read;
        let (prev, prev_obj) = if let Some(seed) = ctx
            .fbw_mode
            .current_exception_seed
            .filter(|_| seed_answers_this_read)
        {
            (seed, ctx.fbw_mode.current_exception_seed_concrete)
        } else {
            let Some(ec) = walker_ensure_execution_context(ctx) else {
                return Ok(None);
            };
            let prev = ctx.trace_ctx.record_op_with_descr(
                OpCode::GetfieldGcR,
                &[ec],
                crate::descr::ec_sys_exc_value_descr(),
            );
            (prev, pyre_interpreter::eval::get_current_exception())
        };
        // Stamp the concrete `prev` so a downstream read sees the value the
        // residual executor would have returned at this resume point.
        ctx.trace_ctx.set_opref_concrete(
            prev,
            majit_ir::Value::Ref(majit_ir::GcRef(prev_obj as usize)),
        );
        // Only PUSH_EXC_INFO owns a matching set + POP_EXCEPT pair.  A covered
        // bare raise uses the same read helper to obtain the exception it
        // re-raises, but has no following PUSH store.  Treating that read as a
        // save arms the next POP as a PUSH and leaves the bare raise's value on
        // this stack, so a second enclosing POP restores the inner exception.
        // For PUSH_EXC_INFO, save (OpRef, concrete) for the matching restore and
        // mark the immediately-following set as this PUSH's slot store.  The
        // codewriter pushes `prev` then `exc` onto the operand stack and
        // POP_EXCEPT pops them, but the walker resolves the popped `prev`
        // operand to the caught exception, not the saved prev; the LIFO stack
        // carries the authoritative value instead.
        if !is_covered_bare_raise_read {
            FBW_EXC_PREV.with(|s| s.borrow_mut().push((prev, prev_obj)));
            FBW_EXC_PENDING_PUSH_SET.with(|c| c.set(true));
        }
        write_residual_call_result_to_dst(ctx, op.pc, dst, 'r', prev)?;
        return Ok(Some(()));
    }

    // `SetCurrentException`: PUSH_EXC_INFO store (stores the caught EXC) or
    // POP_EXCEPT restore (restores the saved prev).  The two are identical at
    // the residual level; `FBW_EXC_PENDING_PUSH_SET` (set by the immediately-
    // preceding PUSH_EXC_INFO prev save) tells them apart.
    if r_args.len() != 1 || dst_bank != 'v' {
        return Ok(None);
    }
    let Some(ec) = walker_ensure_execution_context(ctx) else {
        return Ok(None);
    };
    let is_push_set = FBW_EXC_PENDING_PUSH_SET.with(|c| c.replace(false));
    // POP_EXCEPT restore consumes the prev its matching PUSH_EXC_INFO saved.
    // If unbalanced (no saved prev — e.g. a POP whose PUSH was not lowered),
    // or this is the PUSH's own store, fall back to the operand value.
    let restore = if is_push_set {
        None
    } else {
        FBW_EXC_PREV.with(|s| s.borrow_mut().pop())
    };
    let (mut store_op, mut store_concrete) = match restore {
        // POP_EXCEPT: restore the saved prev, NOT the operand-stack value
        // (which the walker resolves to the just-caught exception).  Restoring
        // the saved prev makes the PUSH store + this restore a balanced no-op,
        // so a locally-caught exception de-escapes and DCEs, and keeps the slot
        // (`sys.exc_info()`) correct after the handler unwinds.
        Some((prev_op, prev_concrete)) => (prev_op, prev_concrete),
        None => {
            let exc_concrete = match read_ref_var_list_concrete(code, op, 1, ctx).first() {
                Some(ConcreteValue::Ref(p)) => *p,
                Some(ConcreteValue::Null) | None => std::ptr::null_mut(),
                _ => return Ok(None),
            };
            (r_args[0], exc_concrete)
        }
    };
    // A PUSH_EXC_INFO store publishes the exception being handled, which IS the
    // tracked active exception (`ctx.last_exc_value()`, the walker's mirror of
    // RPython `metainterp.last_exc_box`).  The graph-side codewriter binds the
    // popped `exc_value`'s producer to a `last_exc_value` re-read for exactly
    // this reason (`codewriter.rs` PushExcInfo arm), but that producer is
    // graph-only — the walker reads the operand-stack slot directly on the
    // assumption that runtime register threading already holds the caught
    // exception there.  At a bridge resume into a handler the slot's per-PC
    // resume reconstruction can alias a non-exception constant (e.g. the vable
    // `f_code` scalar when the catch-landing exception slot shares its color),
    // so the published current exception would become a code object.  The
    // reconstruction can also leave the slot NULL (a bare handler entry whose
    // caught-exception slot was filled with a null sentinel), which would
    // publish `set_current_exception(NULL)` and lose the active exception for a
    // following bare `raise` / `sys.exc_info()`.  When the PUSH store's operand
    // resolves to NULL or a non-exception, recover the authoritative exception
    // from the tracked channel, matching the graph-side producer.
    if is_push_set
        && (store_concrete.is_null() || !unsafe { pyre_object::is_exception(store_concrete) })
    {
        if let (Some(tracked_op), ConcreteValue::Ref(tracked_obj)) =
            (ctx.last_exc_value(), ctx.last_exc_value_concrete())
        {
            if !tracked_obj.is_null() && unsafe { pyre_object::is_exception(tracked_obj) } {
                store_op = tracked_op;
                store_concrete = tracked_obj;
            }
        }
    }
    ctx.trace_ctx.record_op_with_descr(
        OpCode::SetfieldGc,
        &[ec, store_op],
        crate::descr::ec_sys_exc_value_descr(),
    );
    // The walk is authoritative: apply the concrete store the residual
    // executor would have performed, so the live EC tracks the symbolic
    // SETFIELD in lock-step (a following `get_current_exception` /
    // POP_EXCEPT restore reads the right value).  Journal the displaced
    // prior value first: this store mutates the LIVE per-thread EC, so a
    // non-commit walk exit must restore it (the store journal's discipline).
    // Without the undo an exception propagating OUT of an except-handler
    // aborts the walk before its POP_EXCEPT restore, leaking the caught
    // exception into the next frame's `sys_exc_value`.
    fbw_sys_exc_journal_push(pyre_interpreter::eval::get_current_exception());
    pyre_interpreter::eval::set_current_exception(store_concrete);
    ctx.fbw_mode.current_exception_seed = Some(store_op);
    ctx.fbw_mode.current_exception_seed_concrete = store_concrete;
    ctx.fbw_mode.current_exception_seed_from_walk_store = true;
    Ok(Some(()))
}

/// #62: walker-native speculative specialization for the `STORE_SUBSCR`
/// helper residual_call (oopspec `StoreSubscr`, void result).  Records the
/// strategy-dispatched list store inline
/// for the object-, int-, and float-storage list strategies with a non-negative
/// concrete index (and a type-matching value for the unboxed strategies): `guard_class LIST` +
/// `guard_value(strategy)` + unbox index + `IntLt` bounds guard + unbox
/// value + the strategy's `setarrayitem_gc`.
///
/// No residual execution: the recorded `setarrayitem_gc` performs the
/// mutation at runtime (the void residual was likewise not walk-executed —
/// `try_execute_residual_call_via_executor` skips Void results), so the walk's
/// concrete state is unchanged relative to the generic leg. Long values,
/// strategy mismatches, negative indices, and
/// non-`list[int]` operands fall through to the generic `CALL_MAY_FORCE`
/// record (`Ok(None)`), preserving Python `__setitem__` semantics.
pub(crate) fn try_walker_specialize_store_subscr<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    r_args: &[OpRef],
) -> Result<Option<()>, DispatchError> {
    if !ctx.is_authoritative_executor || r_args.len() != 3 {
        return Ok(None);
    }
    let list_op = r_args[0];
    let key_op = r_args[1];
    let value_op = r_args[2];
    let (Some(list_obj), Some(key_obj), Some(value_obj)) = (
        walker_concrete_ref_object(ctx, list_op),
        walker_concrete_ref_object(ctx, key_op),
        walker_concrete_ref_object(ctx, value_op),
    ) else {
        return Ok(None);
    };

    // Gate: list[int] = value, non-negative index in bounds. Object storage
    // accepts every reference; the unboxed strategies additionally require a
    // matching value type (int storage ← W_IntObject, float storage ←
    // W_FloatObject). This is jtransform.py `do_resizable_list_setitem`'s
    // kind=`r` arm, not a separate object-list shortcut.
    let (sid, index, concrete_len) = unsafe {
        // A bool index is fine: bool shares int's `intval`, unboxed below via
        // its own &BOOL_TYPE guard.  A bool *value* into int storage must still
        // route through the generic path — PyPy's IntegerListStrategy rejects a
        // W_BoolObject (`is_correct_type` is exact-type), switching the list to
        // object storage, so the int-storage fast path would drop the bool type.
        // Float subclasses and NaNs switch the list to Object storage.
        // EXACT list only: a list SUBCLASS instance shares `ob_type ==
        // &LIST_TYPE` but retags `w_class` and may override `__setitem__`;
        // `is_exact_list` excludes it so it falls to the generic residual
        // (which honours the override) instead of this direct-storage store.
        if !pyre_object::is_exact_list(list_obj) || !pyre_object::is_int(key_obj) {
            return Ok(None);
        }
        let index = pyre_object::w_int_get_value(key_obj);
        if index < 0 {
            return Ok(None);
        }
        let concrete_len = pyre_object::w_list_len(list_obj);
        if index as usize >= concrete_len {
            return Ok(None);
        }
        // Object storage keeps the value boxed, so a subclass survives it; the
        // unboxed strategies write the raw payload and would drop the subclass
        // identity the read-back must return.  `is_int`/`is_float` read
        // `ob_type`, which a subclass shares, so they alone do not establish
        // that -- and `walker_numeric_builtin_class` below answers with the
        // canonical class, which such a value does not carry.
        let sid = if pyre_object::w_list_uses_object_storage(list_obj) {
            0i64
        } else if !pyre_object::is_exact_builtin_instance(value_obj) {
            return Ok(None);
        } else if pyre_object::w_list_uses_int_storage(list_obj)
            && pyre_object::is_int(value_obj)
            && !pyre_object::is_bool(value_obj)
        {
            1i64
        } else if pyre_object::w_list_uses_float_storage(list_obj)
            && pyre_object::is_float_strategy_item(value_obj)
        {
            // The subclass term of that predicate is enforced on replay by
            // pinning `w_class` below.  `is_plain_float_strict` also admits the
            // null spelling of "exact float", which no pin can express, so
            // decline such an operand here rather than emit a guard it would
            // fail itself (see `walker_guard_exact_w_class`).
            if walker_exact_builtin_class(value_obj).is_none() {
                return Ok(None);
            }
            2i64
        } else {
            return Ok(None);
        };
        (sid, index, concrete_len)
    };

    // --- emit the specialized IR (walker-native) ---
    // guard_class LIST (skip when class already known / operand is constant).
    let list_type_addr = &pyre_object::pyobject::LIST_TYPE as *const _ as i64;
    if !list_op.is_constant() && !ctx.trace_ctx.heap_cache().is_class_known(list_op) {
        let type_const = ctx.trace_ctx.const_int(list_type_addr);
        ctx.trace_ctx
            .record_guard(OpCode::GuardClass, &[list_op, type_const], 0);
        walker_capture_snapshot_for_last_guard(ctx, op_pc)?;
    }
    ctx.trace_ctx
        .heap_cache_mut()
        .class_now_known(list_op, list_type_addr);

    // A list SUBCLASS instance shares `ob_type == &LIST_TYPE` (so it passes
    // the GuardClass above) but retags `w_class` and may override
    // `__setitem__`; guard the exact canonical `w_class` so such an instance
    // side-exits to the generic residual (which honours the override) rather
    // than taking this direct-storage store.
    walker_guard_exact_w_class(
        ctx,
        op_pc,
        list_op,
        pyre_object::pyobject::get_instantiate(&pyre_object::pyobject::LIST_TYPE),
    )?;

    // guard_value(strategy == sid): getfield strategy + GuardValue + replace_box.
    let strategy = crate::state::opimpl_getfield_gc_i(
        ctx.trace_ctx,
        list_op,
        crate::descr::list_strategy_descr(),
    );
    let sid_const = ctx.trace_ctx.const_int(sid);
    ctx.trace_ctx
        .record_guard(OpCode::GuardValue, &[strategy, sid_const], 0);
    walker_capture_snapshot_for_last_guard(ctx, op_pc)?;
    ctx.trace_ctx
        .heap_cache_mut()
        .replace_box(strategy, sid_const);

    // Unbox the index operand.  bool shares int's `intval`, so a bool index
    // guards its own &BOOL_TYPE.
    let (idx_type, idx_descr) = crate::state::int_or_bool_unbox_type_descr(key_obj);
    let raw_index = walker_unbox_int_typed(ctx, op_pc, key_op, idx_type, idx_descr)?;
    ctx.trace_ctx
        .set_opref_concrete(raw_index, majit_ir::Value::Int(index));

    // Object storage keeps the inline `length` field (rlist.py); int/float
    // storage read the typed items-array length field.
    let len_descr = match sid {
        0 => crate::descr::list_length_descr(),
        1 => crate::descr::list_int_items_len_descr(),
        2 => crate::descr::list_float_items_len_descr(),
        _ => unreachable!(),
    };
    let lenbox = crate::state::opimpl_getfield_gc_i(ctx.trace_ctx, list_op, len_descr);
    walker_emit_index_bounds_guards(ctx, op_pc, raw_index, index, lenbox, concrete_len)?;

    // Store the reference directly for ObjectListStrategy; only the typed
    // strategies unwrap their payload.  The object arm is what keeps Python
    // 3.14's `lst[i] is value` guarantee while removing the opaque
    // STORE_SUBSCR helper from hot loops.
    if sid == 0 {
        let block = crate::state::opimpl_getfield_gc_r(
            ctx.trace_ctx,
            list_op,
            crate::descr::list_items_descr(),
        );
        crate::state::trace_items_block_setitem_value(ctx.trace_ctx, block, raw_index, value_op);
    } else if sid == 1 {
        let block = crate::state::opimpl_getfield_gc_r(
            ctx.trace_ctx,
            list_op,
            crate::descr::list_int_items_block_descr(),
        );
        // The value is a true W_IntObject (the gate excludes bool from int
        // storage), so it unboxes through the plain INT_TYPE guard.
        let int_type_addr = &pyre_object::pyobject::INT_TYPE as *const _ as i64;
        let raw = walker_unbox_int(ctx, op_pc, value_op, int_type_addr)?;
        // The list gets an exact-`w_class` guard above; the VALUE needs its own,
        // because the unbox proves only `ob_type` and a subclass shares it —
        // storing its payload would drop the element's Python class.
        walker_guard_exact_w_class(
            ctx,
            op_pc,
            value_op,
            walker_numeric_builtin_class(value_obj),
        )?;
        let elem = unsafe { pyre_object::w_int_get_value(value_obj) };
        ctx.trace_ctx
            .set_opref_concrete(raw, majit_ir::Value::Int(elem));
        crate::state::trace_int_block_setitem_value(ctx.trace_ctx, block, raw_index, raw);
    } else {
        let block = crate::state::opimpl_getfield_gc_r(
            ctx.trace_ctx,
            list_op,
            crate::descr::list_float_items_block_descr(),
        );
        let float_type_addr = &pyre_object::pyobject::FLOAT_TYPE as *const _ as i64;
        let raw = walker_unbox_float(ctx, op_pc, value_op, float_type_addr)?;
        walker_guard_exact_w_class(
            ctx,
            op_pc,
            value_op,
            walker_numeric_builtin_class(value_obj),
        )?;
        let elem = unsafe { pyre_object::w_float_get_value(value_obj) };
        ctx.trace_ctx
            .set_opref_concrete(raw, majit_ir::Value::Float(elem));
        // A float SUBCLASS instance shares `ob_type == &FLOAT_TYPE` (so it
        // passes the unbox guard) but retags `w_class`;
        // `FloatListStrategy.is_correct_type` rejects it, so the interpreter
        // switches the list to Object storage instead of writing raw f64.
        // Pin the canonical class the same way the list operand is pinned
        // above, so such an instance side-exits to the generic residual.
        walker_guard_exact_w_class(
            ctx,
            op_pc,
            value_op,
            pyre_object::pyobject::get_instantiate(&pyre_object::pyobject::FLOAT_TYPE),
        )?;
        walker_guard_float_not_nan(ctx, op_pc, raw)?;
        crate::state::trace_float_block_setitem_value(ctx.trace_ctx, block, raw_index, raw);
    }

    // Tracing is execution (pyjitpl.py execute_and_record): apply the
    // store to the concrete list now, so the walk's own region — and a
    // walk-end commit that hands the END state to the interpreter with no
    // replay — sees the mutation exactly once.  The displaced element goes
    // into the undo log first: a walk that does NOT commit returns to the
    // legacy replay, which re-executes the region and must find the
    // pre-walk heap (see `FBW_STORE_JOURNAL`).
    let Some(displaced) = (unsafe { pyre_object::w_list_getitem(list_obj, index) }) else {
        unreachable!(
            "store_subscr specialization: in-bounds index {index} has no element \
             (strategy/bounds gates above admitted it)"
        );
    };
    // For typed storage `w_list_getitem` boxes the displaced int/float; that
    // allocation can run a minor collection and move the operands. Object
    // storage returns its exact existing reference without allocation. Re-read the
    // forwarded refs from the shadow before touching the heap.  (The
    // freshly boxed `displaced` itself cannot move before the journal
    // push roots it — nothing below allocates.)
    let (Some(list_obj), Some(key_obj), Some(value_obj)) = (
        walker_concrete_ref_object(ctx, list_op),
        walker_concrete_ref_object(ctx, key_op),
        walker_concrete_ref_object(ctx, value_op),
    ) else {
        unreachable!(
            "store_subscr specialization: operand concrete vanished from the shadow \
             across the displaced-element boxing"
        );
    };
    fbw_store_journal_push(list_obj, key_obj, displaced);
    let stored = unsafe { pyre_object::w_list_setitem(list_obj, index, value_obj) };
    debug_assert!(
        stored,
        "store_subscr specialization: in-bounds store failed"
    );
    Ok(Some(()))
}

/// Walker-native `GetIter` for an exact machine-word `range`.
///
/// Emits the virtual `W_IntRangeIterator` allocation shape directly — the
/// iterator PyPy's inlined `descr_iter` would trace — so a locally consumed
/// iterator stays a removable virtual `New`.
pub(crate) fn try_walker_specialize_get_iter<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    r_args: &[OpRef],
    _dst: usize,
    dst_bank: char,
) -> Result<Option<OpRef>, DispatchError> {
    if !ctx.is_authoritative_executor
        || dst_bank != 'r'
        || r_args.len() != 1
        || ctx.fbw_mode.inline_subwalk
    {
        return Ok(None);
    }

    let range_op = r_args[0];
    let Some(range_obj) = walker_concrete_ref_object(ctx, range_op) else {
        return Ok(None);
    };

    // `W_Zip.iter_w` is identity; exact-class guards preserve overrides.
    let zip_type = &pyre_object::functional::ZIP_TYPE as *const pyre_object::PyType;
    let zip_class = pyre_object::get_instantiate(&pyre_object::functional::ZIP_TYPE);
    if unsafe {
        !range_obj.is_null()
            && std::ptr::eq((*range_obj).ob_type, zip_type)
            && std::ptr::eq((*range_obj).w_class, zip_class)
    } {
        walker_guard_class(ctx, op_pc, range_op, zip_type as i64)?;
        walker_guard_exact_w_class(ctx, op_pc, range_op, zip_class)?;
        ctx.vstack_last_ref = range_op;
        return Ok(Some(range_op));
    }

    let (concrete_start, concrete_step, concrete_length, concrete_mul, concrete_one_past) = unsafe {
        if !pyre_object::functional::is_w_range(range_obj)
            || !pyre_object::functional::is_exact_w_range(range_obj)
        {
            return Ok(None);
        }
        let (start_obj, _stop_obj, step_obj) = pyre_object::functional::w_range_fields(range_obj);
        let length_obj = pyre_object::functional::w_range_length(range_obj);
        if !pyre_object::is_int(start_obj)
            || pyre_object::is_bool(start_obj)
            || !pyre_object::is_int(step_obj)
            || pyre_object::is_bool(step_obj)
            || !pyre_object::is_int(length_obj)
            || pyre_object::is_bool(length_obj)
        {
            return Ok(None);
        }
        let Some((start, _stop, step)) = pyre_object::functional::w_range_fields_i64(range_obj)
        else {
            return Ok(None);
        };
        let Some(length) = pyre_object::functional::w_range_length_i64(range_obj) else {
            return Ok(None);
        };
        let one_past_i128 = start as i128 + length as i128 * step as i128;
        let Ok(one_past) = i64::try_from(one_past_i128) else {
            return Ok(None);
        };
        let Some(mul) = length.checked_mul(step) else {
            return Ok(None);
        };
        let Some(one_past_checked) = start.checked_add(mul) else {
            return Ok(None);
        };
        debug_assert_eq!(one_past_checked, one_past);
        (start, step, length, mul, one_past)
    };

    let range_type_addr = &pyre_object::functional::RANGE_TYPE as *const _ as i64;
    if !range_op.is_constant() && !ctx.trace_ctx.heap_cache().is_class_known(range_op) {
        let range_type_const = ctx.trace_ctx.const_int(range_type_addr);
        ctx.trace_ctx
            .record_guard(OpCode::GuardClass, &[range_op, range_type_const], 0);
        walker_capture_snapshot_for_last_guard(ctx, op_pc)?;
    }
    ctx.trace_ctx
        .heap_cache_mut()
        .class_now_known(range_op, range_type_addr);

    let int_type_addr = &pyre_object::pyobject::INT_TYPE as *const _ as i64;
    let int_type_const = ctx.trace_ctx.const_int(int_type_addr);

    let start_r = crate::state::opimpl_getfield_gc_r(
        ctx.trace_ctx,
        range_op,
        crate::descr::range_start_descr(),
    );
    if !ctx.trace_ctx.heap_cache().is_class_known(start_r) {
        ctx.trace_ctx
            .record_guard(OpCode::GuardClass, &[start_r, int_type_const], 0);
        walker_capture_snapshot_for_last_guard(ctx, op_pc)?;
        ctx.trace_ctx
            .heap_cache_mut()
            .class_now_known(start_r, int_type_addr);
    }
    let start_i = crate::state::opimpl_getfield_gc_i(
        ctx.trace_ctx,
        start_r,
        crate::descr::int_intval_descr(),
    );
    ctx.trace_ctx
        .set_opref_concrete(start_i, majit_ir::Value::Int(concrete_start));

    let step_r = crate::state::opimpl_getfield_gc_r(
        ctx.trace_ctx,
        range_op,
        crate::descr::range_step_descr(),
    );
    if !ctx.trace_ctx.heap_cache().is_class_known(step_r) {
        ctx.trace_ctx
            .record_guard(OpCode::GuardClass, &[step_r, int_type_const], 0);
        walker_capture_snapshot_for_last_guard(ctx, op_pc)?;
        ctx.trace_ctx
            .heap_cache_mut()
            .class_now_known(step_r, int_type_addr);
    }
    let step_i =
        crate::state::opimpl_getfield_gc_i(ctx.trace_ctx, step_r, crate::descr::int_intval_descr());
    ctx.trace_ctx
        .set_opref_concrete(step_i, majit_ir::Value::Int(concrete_step));

    let length_r = crate::state::opimpl_getfield_gc_r(
        ctx.trace_ctx,
        range_op,
        crate::descr::range_length_descr(),
    );
    if !ctx.trace_ctx.heap_cache().is_class_known(length_r) {
        ctx.trace_ctx
            .record_guard(OpCode::GuardClass, &[length_r, int_type_const], 0);
        walker_capture_snapshot_for_last_guard(ctx, op_pc)?;
        ctx.trace_ctx
            .heap_cache_mut()
            .class_now_known(length_r, int_type_addr);
    }
    let length_i = crate::state::opimpl_getfield_gc_i(
        ctx.trace_ctx,
        length_r,
        crate::descr::int_intval_descr(),
    );
    ctx.trace_ctx
        .set_opref_concrete(length_i, majit_ir::Value::Int(concrete_length));

    let mul = ctx
        .trace_ctx
        .record_op(OpCode::IntMulOvf, &[length_i, step_i]);
    ctx.trace_ctx
        .set_opref_concrete(mul, majit_ir::Value::Int(concrete_mul));
    walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardNoOverflow, &[])?;

    let one_past = ctx.trace_ctx.record_op(OpCode::IntAddOvf, &[start_i, mul]);
    ctx.trace_ctx
        .set_opref_concrete(one_past, majit_ir::Value::Int(concrete_one_past));
    walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardNoOverflow, &[])?;

    let new = ctx.trace_ctx.record_op_with_descr(
        OpCode::NewWithVtable,
        &[],
        crate::descr::w_range_iter_size_descr(),
    );
    ctx.trace_ctx.heap_cache_mut().new_object(new);

    let current_descr = crate::descr::range_iter_current_descr();
    let current_index = current_descr.index();
    ctx.trace_ctx
        .record_op_with_descr(OpCode::SetfieldGc, &[new, start_i], current_descr);
    ctx.trace_ctx
        .heapcache_setfield_cached(new, current_index, start_i);

    let remaining_descr = crate::descr::range_iter_remaining_descr();
    let remaining_index = remaining_descr.index();
    ctx.trace_ctx
        .record_op_with_descr(OpCode::SetfieldGc, &[new, length_i], remaining_descr);
    ctx.trace_ctx
        .heapcache_setfield_cached(new, remaining_index, length_i);

    let step_descr = crate::descr::range_iter_step_descr();
    let step_index = step_descr.index();
    ctx.trace_ctx
        .record_op_with_descr(OpCode::SetfieldGc, &[new, step_i], step_descr);
    ctx.trace_ctx
        .heapcache_setfield_cached(new, step_index, step_i);

    let range_iter_type_addr = &pyre_object::functional::RANGE_ITER_TYPE as *const _ as i64;
    ctx.trace_ctx
        .heap_cache_mut()
        .class_now_known(new, range_iter_type_addr);

    let real_iter = unsafe { pyre_object::functional::w_range_iter(range_obj) };
    ctx.trace_ctx.set_opref_concrete(
        new,
        majit_ir::Value::Ref(majit_ir::GcRef(real_iter as usize)),
    );
    ctx.vstack_last_ref = new;

    Ok(Some(new))
}

/// Walker-native `ForIterNext` for an arity-two `zip` over two
/// `W_TupleIterObject` cursors.
///
/// The generic residual advances both shared iterators before an abort can
/// occur, and forward-delivery preserves the consumed pair.  This inline path
/// keeps that deliberately irreversible advance: it never journals or rolls
/// either cursor back.  It emits both `W_TupleIterObject` index updates and a
/// continuation guard whose false side resumes at the same FOR_ITER coordinate
/// as the codewriter's ordinary exhaustion edge; `strict=True` routes its
/// uneven-length arm through the generic path so the interpreter owns the
/// authentic `ValueError`.
///
/// The continuation item is the object pair `W_Zip.next_w` builds with
/// `newtuple2` at arity two; allocation removal elides it until an escaping
/// consumer or a deopt needs a real box.
fn try_walker_specialize_zip_two_tuple_iters<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    zip_op: OpRef,
    zip_obj: pyre_object::PyObjectRef,
) -> Result<Option<OpRef>, DispatchError> {
    if ctx.fbw_mode.inline_subwalk {
        return Ok(None);
    }

    let zip_type = &pyre_object::functional::ZIP_TYPE as *const pyre_object::PyType;
    let zip_class = pyre_object::get_instantiate(&pyre_object::functional::ZIP_TYPE);
    let (iterators_obj, inner_objs, steps, strict) = unsafe {
        if zip_obj.is_null()
            || !std::ptr::eq((*zip_obj).ob_type, zip_type)
            || !std::ptr::eq((*zip_obj).w_class, zip_class)
        {
            return Ok(None);
        }
        let iterators_obj = pyre_object::functional::w_zip_get_iterators(zip_obj);
        if iterators_obj.is_null()
            || !pyre_object::is_list(iterators_obj)
            || !pyre_object::is_exact_builtin_instance(iterators_obj)
        {
            return Ok(None);
        }
        let list = &*(iterators_obj as *const pyre_object::listobject::W_ListObject);
        if list.strategy != pyre_object::listobject::ListStrategy::Object
            || pyre_object::w_list_len(iterators_obj) != 2
        {
            return Ok(None);
        }
        let Some(inner0) = pyre_object::w_list_getitem(iterators_obj, 0) else {
            return Ok(None);
        };
        let Some(inner1) = pyre_object::w_list_getitem(iterators_obj, 1) else {
            return Ok(None);
        };

        let mut steps = Vec::with_capacity(2);
        for inner in [inner0, inner1] {
            if !pyre_object::is_tuple_iter(inner)
                || !std::ptr::eq((*inner).ob_type, &pyre_object::iterobject::TUPLE_ITER_TYPE)
            {
                return Ok(None);
            }
            let seq = pyre_object::w_tuple_iter_seq(inner);
            let index = pyre_object::w_tuple_iter_index(inner);
            if seq.is_null()
                || !pyre_object::is_tuple(seq)
                || !pyre_object::is_exact_builtin_instance(seq)
                || index < 0
            {
                return Ok(None);
            }
            let len = pyre_object::w_tuple_len(seq) as i64;
            let item = pyre_object::w_tuple_getitem(seq, index);
            steps.push((seq, index, len, item));
        }
        (
            iterators_obj,
            [inner0, inner1],
            steps,
            pyre_object::functional::w_zip_get_strict(zip_obj),
        )
    };
    let concrete_continues = steps.iter().all(|step| step.3.is_some());
    let concrete_exhausted = steps.iter().all(|step| step.1 >= step.2);
    // Mixed bounds must fall through to the interpreter's strict error path.
    if !concrete_continues && !(strict && concrete_exhausted) {
        return Ok(None);
    }

    // Reproduce PyPy's unrolled arity-two `W_Zip.next_w` shape.
    walker_guard_class(ctx, op_pc, zip_op, zip_type as i64)?;
    walker_guard_exact_w_class(ctx, op_pc, zip_op, zip_class)?;
    let iterators_op = crate::state::opimpl_getfield_gc_r(
        ctx.trace_ctx,
        zip_op,
        crate::descr::zip_iterators_descr(),
    );
    ctx.trace_ctx.set_opref_concrete(
        iterators_op,
        Value::Ref(majit_ir::GcRef(iterators_obj as usize)),
    );

    let list_type = &pyre_object::LIST_TYPE as *const pyre_object::PyType as i64;
    walker_guard_class(ctx, op_pc, iterators_op, list_type)?;
    walker_guard_exact_w_class(
        ctx,
        op_pc,
        iterators_op,
        pyre_object::get_instantiate(&pyre_object::LIST_TYPE),
    )?;
    let strategy = crate::state::opimpl_getfield_gc_i(
        ctx.trace_ctx,
        iterators_op,
        crate::descr::list_strategy_descr(),
    );
    ctx.trace_ctx.set_opref_concrete(
        strategy,
        Value::Int(pyre_object::listobject::ListStrategy::Object as i64),
    );
    let object_strategy = ctx
        .trace_ctx
        .const_int(pyre_object::listobject::ListStrategy::Object as i64);
    walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardValue, &[strategy, object_strategy])?;
    let list_len = crate::state::opimpl_getfield_gc_i(
        ctx.trace_ctx,
        iterators_op,
        crate::descr::list_length_descr(),
    );
    ctx.trace_ctx.set_opref_concrete(list_len, Value::Int(2));
    let two = ctx.trace_ctx.const_int(2);
    walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardValue, &[list_len, two])?;
    let iterator_block = crate::state::opimpl_getfield_gc_r(
        ctx.trace_ctx,
        iterators_op,
        crate::descr::list_items_descr(),
    );

    let mut inner_ops = Vec::with_capacity(2);
    for (index, inner_obj) in inner_objs.into_iter().enumerate() {
        let index_op = ctx.trace_ctx.const_int(index as i64);
        let inner_op =
            crate::state::trace_items_block_getitem_value(ctx.trace_ctx, iterator_block, index_op);
        ctx.trace_ctx
            .set_opref_concrete(inner_op, Value::Ref(majit_ir::GcRef(inner_obj as usize)));
        inner_ops.push(inner_op);
    }

    // Guard both cursors before either is advanced.
    let tuple_iter_type =
        &pyre_object::iterobject::TUPLE_ITER_TYPE as *const pyre_object::PyType as i64;
    let tuple_type = &pyre_object::TUPLE_TYPE as *const pyre_object::PyType as i64;
    let tuple_class = pyre_object::get_instantiate(&pyre_object::TUPLE_TYPE);
    let mut emitted_steps = Vec::with_capacity(2);
    let mut both_match = None;
    for (inner_op, (seq_obj, index, len, item_obj)) in
        inner_ops.iter().copied().zip(steps.iter().copied())
    {
        walker_guard_class(ctx, op_pc, inner_op, tuple_iter_type)?;
        let seq_op = crate::state::opimpl_getfield_gc_r(
            ctx.trace_ctx,
            inner_op,
            crate::descr::tuple_iter_seq_descr(),
        );
        ctx.trace_ctx
            .set_opref_concrete(seq_op, Value::Ref(majit_ir::GcRef(seq_obj as usize)));
        walker_guard_class(ctx, op_pc, seq_op, tuple_type)?;
        walker_guard_exact_w_class(ctx, op_pc, seq_op, tuple_class)?;
        let raw_index = crate::state::opimpl_getfield_gc_i(
            ctx.trace_ctx,
            inner_op,
            crate::descr::tuple_iter_index_descr(),
        );
        ctx.trace_ctx
            .set_opref_concrete(raw_index, Value::Int(index));
        let items = crate::state::opimpl_getfield_gc_r(
            ctx.trace_ctx,
            seq_op,
            crate::descr::tuple_wrappeditems_descr(),
        );
        let raw_len = crate::state::opimpl_arraylen_gc(
            ctx.trace_ctx,
            items,
            crate::state::pyobject_gcarray_descr(),
        );
        ctx.trace_ctx.set_opref_concrete(raw_len, Value::Int(len));
        let matches_arm = if concrete_continues {
            let zero = ctx.trace_ctx.const_int(0);
            let nonnegative = ctx.trace_ctx.record_op(OpCode::IntGe, &[raw_index, zero]);
            ctx.trace_ctx
                .set_opref_concrete(nonnegative, Value::Int((index >= 0) as i64));
            let in_bounds = ctx
                .trace_ctx
                .record_op(OpCode::IntLt, &[raw_index, raw_len]);
            ctx.trace_ctx
                .set_opref_concrete(in_bounds, Value::Int((index < len) as i64));
            let matches = ctx
                .trace_ctx
                .record_op(OpCode::IntAnd, &[nonnegative, in_bounds]);
            ctx.trace_ctx
                .set_opref_concrete(matches, Value::Int((index >= 0 && index < len) as i64));
            matches
        } else {
            let matches = ctx
                .trace_ctx
                .record_op(OpCode::IntGe, &[raw_index, raw_len]);
            ctx.trace_ctx
                .set_opref_concrete(matches, Value::Int((index >= len) as i64));
            matches
        };
        both_match = Some(match both_match {
            None => matches_arm,
            Some(prior) => {
                let both = ctx
                    .trace_ctx
                    .record_op(OpCode::IntAnd, &[prior, matches_arm]);
                ctx.trace_ctx.set_opref_concrete(both, Value::Int(1));
                both
            }
        });
        emitted_steps.push((inner_op, raw_index, items, item_obj));
    }
    walker_emit_guard_with_snapshot(
        ctx,
        op_pc,
        OpCode::GuardTrue,
        &[both_match.expect("zip arity-two recognition emitted two cursors")],
    )?;

    let body = fbw_foriter_body_from_op_pc(ctx, op_pc)
        .unwrap_or_else(|| InflightForiterBody::Py(ctx.entry_py_pc() as usize + 1));
    fbw_foriter_inflight_mark_attempt(body);

    if concrete_exhausted {
        // PyPy clears both exhausted tuple iterators before StopIteration.
        let strict_op = crate::state::opimpl_getfield_gc_i(
            ctx.trace_ctx,
            zip_op,
            crate::descr::zip_strict_descr(),
        );
        ctx.trace_ctx.set_opref_concrete(strict_op, Value::Int(1));
        walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardTrue, &[strict_op])?;
        let null_ref = ctx.trace_ctx.const_ref(0);
        for (inner_op, step) in inner_ops.into_iter().zip(steps.iter()) {
            let seq_descr = crate::descr::tuple_iter_seq_descr();
            ctx.trace_ctx.record_op_with_descr(
                OpCode::SetfieldGc,
                &[inner_op, null_ref],
                seq_descr.clone(),
            );
            ctx.trace_ctx
                .heapcache_setfield_cached(inner_op, seq_descr.index(), null_ref);
            let inner_obj = walker_concrete_ref_object(ctx, inner_op)
                .expect("zip tuple iterator concrete survived exhaustion emission");
            if ctx.trace_ctx.is_bridge_trace {
                let pre_seq = unsafe { pyre_object::w_tuple_iter_seq(inner_obj) };
                fbw_bridge_tuple_iter_journal_push(inner_obj, pre_seq, step.1);
            }
            unsafe { pyre_object::w_tuple_iter_set_seq(inner_obj, pyre_object::PY_NULL) };
        }
        let zero = ctx.trace_ctx.const_int(0);
        let null_item = ctx.trace_ctx.record_op(OpCode::CastIntToPtr, &[zero]);
        ctx.trace_ctx
            .set_opref_concrete(null_item, Value::Ref(majit_ir::GcRef(0)));
        return Ok(Some(null_item));
    }

    let one = ctx.trace_ctx.const_int(1);
    let mut item_ops = Vec::with_capacity(2);
    for (inner_op, raw_index, items, item_obj) in emitted_steps {
        let item_op =
            crate::state::trace_items_block_getitem_value_pure(ctx.trace_ctx, items, raw_index);
        let next_index = ctx.trace_ctx.record_op(OpCode::IntAdd, &[raw_index, one]);
        let concrete_index = steps[item_ops.len()].1;
        ctx.trace_ctx
            .set_opref_concrete(next_index, Value::Int(concrete_index + 1));
        let index_descr = crate::descr::tuple_iter_index_descr();
        ctx.trace_ctx.record_op_with_descr(
            OpCode::SetfieldGc,
            &[inner_op, next_index],
            index_descr.clone(),
        );
        ctx.trace_ctx
            .heapcache_setfield_cached(inner_op, index_descr.index(), next_index);
        let item_obj = item_obj.expect("continue-arm zip step has an item");
        ctx.trace_ctx
            .set_opref_concrete(item_op, Value::Ref(majit_ir::GcRef(item_obj as usize)));
        item_ops.push(item_op);
    }

    let tuple_op =
        crate::helpers::emit_specialised_tuple_oo_inline(ctx.trace_ctx, item_ops[0], item_ops[1]);

    // Advance the authentic shadows and retain the yielded pair for abort.
    let concrete_tuple = pyre_object::w_specialised_tuple_oo_new(
        steps[0].3.expect("continue-arm zip step has item 0"),
        steps[1].3.expect("continue-arm zip step has item 1"),
    );
    if concrete_tuple.is_null() {
        return Err(DispatchError::ConcreteShadowAllocationFailed { pc: op_pc });
    }
    for (inner_op, step) in inner_ops.into_iter().zip(steps.iter()) {
        let inner_obj = walker_concrete_ref_object(ctx, inner_op)
            .expect("zip tuple iterator concrete survived tuple allocation");
        if ctx.trace_ctx.is_bridge_trace {
            let pre_seq = unsafe { pyre_object::w_tuple_iter_seq(inner_obj) };
            fbw_bridge_tuple_iter_journal_push(inner_obj, pre_seq, step.1);
        }
        unsafe { pyre_object::w_tuple_iter_set_index(inner_obj, step.1 + 1) };
    }
    let concrete_item0 = unsafe {
        pyre_object::specialisedtupleobject::w_specialised_tuple_oo_getvalue(concrete_tuple, 0)
    };
    let concrete_item1 = unsafe {
        pyre_object::specialisedtupleobject::w_specialised_tuple_oo_getvalue(concrete_tuple, 1)
    };
    for (item_op, concrete_item) in item_ops.into_iter().zip([concrete_item0, concrete_item1]) {
        ctx.trace_ctx
            .set_opref_concrete(item_op, Value::Ref(majit_ir::GcRef(concrete_item as usize)));
    }
    ctx.trace_ctx.set_opref_concrete(
        tuple_op,
        Value::Ref(majit_ir::GcRef(concrete_tuple as usize)),
    );
    fbw_foriter_inflight_capture(concrete_tuple, body);
    ctx.vstack_last_ref = tuple_op;
    Ok(Some(tuple_op))
}

pub(crate) fn try_walker_specialize_for_iter_next<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    r_args: &[OpRef],
    _dst: usize,
    dst_bank: char,
) -> Result<Option<OpRef>, DispatchError> {
    if !ctx.is_authoritative_executor || dst_bank != 'r' || r_args.len() != 1 {
        return Ok(None);
    }

    // A range class-guard failure at this FOR_ITER green key is a definitive
    // polymorphism witness.  Once the failure path has demoted it, retain the
    // generic residual rather than recreating the range guard on retrace.
    let range_green_key = walker_foriter_green_key(ctx, op_pc);
    if range_green_key.is_some_and(crate::trace::range_foriter_demoted) {
        return Ok(None);
    }

    let iter_op = r_args[0];
    let Some(iter_obj) = walker_concrete_ref_object(ctx, iter_op) else {
        return Ok(None);
    };
    if unsafe { pyre_object::functional::is_zip(iter_obj) } {
        return spec_gate(SpecFold::ZipTwoTupleIters, || {
            try_walker_specialize_zip_two_tuple_iters(ctx, op_pc, iter_op, iter_obj)
        });
    }
    let (concrete_current, concrete_remaining, concrete_step) = unsafe {
        if !pyre_object::functional::is_range_iter(iter_obj) {
            return Ok(None);
        }
        pyre_object::functional::w_range_iter_fields(iter_obj)
    };
    let concrete_continues = concrete_remaining != 0;

    // A new consume attempt completes the prior in-flight iteration before
    // this irreversible concrete advance, matching the residual executor.
    let body = fbw_foriter_body_from_op_pc(ctx, op_pc)
        .unwrap_or_else(|| InflightForiterBody::Py(ctx.entry_py_pc() as usize + 1));
    fbw_foriter_inflight_mark_attempt(body);

    // guard_class W_IntRangeIterator, unless the operand is already known.
    let range_iter_type_addr = &pyre_object::functional::RANGE_ITER_TYPE as *const _ as i64;
    if !iter_op.is_constant() && !ctx.trace_ctx.heap_cache().is_class_known(iter_op) {
        let type_const = ctx.trace_ctx.const_int(range_iter_type_addr);
        // Pre-mint the guard's FailDescr tagged with this FOR_ITER green key
        // so its runtime failure — a definitive polymorphism witness —
        // demotes the specialization by descr identity, independent of the
        // guard's per-trace fail index.  `store_final_boxes_in_guard`
        // preserves an existing ResumeGuardDescr (only refreshing
        // fail_arg_types), so the tag survives optimizer guard-folding and
        // unroll; a copied guard chases `prev` to this donor.  With no green
        // key available the guard is untagged and the site is simply never
        // demoted.
        match range_green_key {
            Some(green_key) => {
                let descr = majit_metainterp::make_resume_guard_descr_range_foriter(green_key);
                ctx.trace_ctx.record_guard_with_descr(
                    OpCode::GuardClass,
                    &[iter_op, type_const],
                    descr,
                );
            }
            None => {
                ctx.trace_ctx
                    .record_guard(OpCode::GuardClass, &[iter_op, type_const], 0);
            }
        }
        walker_capture_snapshot_for_last_guard(ctx, op_pc)?;
    }
    ctx.trace_ctx
        .heap_cache_mut()
        .class_now_known(iter_op, range_iter_type_addr);

    if !concrete_continues {
        // Exhausted arrival: the walker concretely reached remaining==0 (a nested
        // inner loop run to completion inside the outer body).  Record the
        // routing guard for the false continue predicate, then present the
        // exhaustion edge exactly as the residual does: a NULL Ref that the
        // codewriter's trailing GuardNonnull consumes as the loop exit.  The
        // iterator is already exhausted, so no cursor advance and no in-flight
        // capture.
        let zero = ctx.trace_ctx.const_int(0);
        let remaining = crate::state::opimpl_getfield_gc_i(
            ctx.trace_ctx,
            iter_op,
            crate::descr::range_iter_remaining_descr(),
        );
        let continues = ctx.trace_ctx.record_op(OpCode::IntGt, &[remaining, zero]);
        ctx.trace_ctx.set_opref_concrete(continues, Value::Int(0));
        walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardFalse, &[continues])?;
        let null_item = ctx.trace_ctx.record_op(OpCode::CastIntToPtr, &[zero]);
        ctx.trace_ctx
            .set_opref_concrete(null_item, Value::Ref(majit_ir::GcRef(0)));
        return Ok(Some(null_item));
    }

    // Guard the continue arm before constructing the item.  The false arm
    // resumes at this FOR_ITER, where the interpreter takes the existing
    // exhaustion edge (iterator retained, no item pushed).  This avoids the
    // pointer-mask representation which forced the item to be materialized.
    let current = crate::state::opimpl_getfield_gc_i(
        ctx.trace_ctx,
        iter_op,
        crate::descr::range_iter_current_descr(),
    );
    let remaining = crate::state::opimpl_getfield_gc_i(
        ctx.trace_ctx,
        iter_op,
        crate::descr::range_iter_remaining_descr(),
    );
    let step = crate::state::opimpl_getfield_gc_i(
        ctx.trace_ctx,
        iter_op,
        crate::descr::range_iter_step_descr(),
    );
    let zero = ctx.trace_ctx.const_int(0);
    let continues = ctx.trace_ctx.record_op(OpCode::IntGt, &[remaining, zero]);
    ctx.trace_ctx
        .set_opref_concrete(continues, Value::Int(concrete_continues as i64));
    walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardTrue, &[continues])?;

    // The continue guard establishes `continues == 1` on the trace path. Keep
    // the wrapping IntAdd and live-iterator SetfieldGc updates intact.
    let delta = ctx.trace_ctx.record_op(OpCode::IntMul, &[step, continues]);
    ctx.trace_ctx.set_opref_concrete(
        delta,
        Value::Int(concrete_step.wrapping_mul(concrete_continues as i64)),
    );
    let next_current = ctx.trace_ctx.record_op(OpCode::IntAdd, &[current, delta]);
    let next_current_concrete =
        concrete_current.wrapping_add(concrete_step.wrapping_mul(concrete_continues as i64));
    ctx.trace_ctx
        .set_opref_concrete(next_current, Value::Int(next_current_concrete));
    let current_descr = crate::descr::range_iter_current_descr();
    ctx.trace_ctx.record_op_with_descr(
        OpCode::SetfieldGc,
        &[iter_op, next_current],
        current_descr.clone(),
    );
    ctx.trace_ctx
        .heapcache_setfield_cached(iter_op, current_descr.index(), next_current);

    let next_remaining = ctx
        .trace_ctx
        .record_op(OpCode::IntSub, &[remaining, continues]);
    let next_remaining_concrete = concrete_remaining.wrapping_sub(concrete_continues as i64);
    ctx.trace_ctx
        .set_opref_concrete(next_remaining, Value::Int(next_remaining_concrete));
    let remaining_descr = crate::descr::range_iter_remaining_descr();
    ctx.trace_ctx.record_op_with_descr(
        OpCode::SetfieldGc,
        &[iter_op, next_remaining],
        remaining_descr.clone(),
    );
    ctx.trace_ctx
        .heapcache_setfield_cached(iter_op, remaining_descr.index(), next_remaining);

    // `wrapint` is the transparent `NewWithVtable(W_IntObject)` +
    // `SetfieldGc(intval=current)` shape allocation removal virtualizes.  Do
    // not feed it through pointer arithmetic: locally consumed items stay
    // virtual, while normal forcing materializes escaping items.
    let item = crate::state::wrapint(ctx.trace_ctx, current);

    // Tracing executes the real range cursor advance.  The direct helper is
    // the same `W_IntRangeIterator.next` implementation used by the residual;
    // do not journal it, because abort recovery forwards this exact item.
    let concrete_item = unsafe { pyre_object::functional::w_range_iter_next(iter_obj) };
    debug_assert_eq!(concrete_item.is_some(), concrete_continues);
    let concrete_item_ptr = concrete_item.expect("GuardTrue(continues) implies a range item");
    ctx.trace_ctx.set_opref_concrete(
        item,
        Value::Ref(majit_ir::GcRef(concrete_item_ptr as usize)),
    );

    // Keep the virtual payload's concrete shadow paired with the concrete New.
    // A later body guard can then encode the virtual `i` in its snapshot and
    // blackhole will rematerialize the right item on deopt.
    ctx.trace_ctx
        .set_opref_concrete(current, Value::Int(concrete_current));

    if ctx.trace_ctx.is_bridge_trace {
        // A bridge/retrace recording walk has no in-flight forward-delivery on
        // abort, so journal the pre-advance cursor for restore if the walk does
        // not commit (keeps the aborted recording side-effect neutral).
        fbw_bridge_iter_journal_push(iter_obj, concrete_current, concrete_remaining);
    }
    fbw_foriter_inflight_capture(concrete_item_ptr, body);
    // Range iteration stays at the C level, so the operand-stack mirror
    // remains valid and must receive the item produced by FOR_ITER.  Its
    // virtual state is captured by subsequent body-guard snapshots.
    ctx.vstack_last_ref = item;

    Ok(Some(item))
}

/// Specialize `STORE_SUBSCR target[const_slice] = source` for a same-length,
/// step-1 slice between two Integer-strategy exact lists, eliding the
/// `CALL_MAY_FORCE` `store_subscr` residual that would force the virtualizable
/// source list (the freshly built BUILD_LIST temp from
/// [`try_walker_specialize_newlist`]) every iteration.  The same-length gate
/// makes the assignment `slice_len` independent in-bounds setitems —
/// `target[start + j] = source[j]` — with no resize and no strategy change, so
/// it rides the existing `FBW_STORE_JOURNAL` per-element undo log.
///
/// Reads the source elements through `getfield_gc(int_items)` +
/// `getarrayitem_gc` ops keyed on the source `OpRef`, so when the source is the
/// freshly built virtual list the optimizer folds the reads against its
/// recorded `SetarrayitemGc` stores and removes the whole temporary.
///
/// The slice key must be a trace constant (a `slice(...)` from `co_consts`);
/// `start` / `stop` are read off the slice object and baked into the emitted
/// index constants.  Falls through to the generic residual (returns `Ok(None)`)
/// for anything outside the gate: a non-constant / `None` / negative bound, a
/// non-unit step, a resizing (length-changing) slice, an empty slice, a
/// non-Integer-storage target or source, or a list subclass (which may override
/// `__setitem__` / `__iter__`).
pub(crate) fn try_walker_specialize_setslice<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    r_args: &[OpRef],
) -> Result<Option<()>, DispatchError> {
    if !ctx.is_authoritative_executor || r_args.len() != 3 {
        return Ok(None);
    }
    let list_op = r_args[0];
    let key_op = r_args[1];
    let value_op = r_args[2];
    // The slice key must be a trace constant (a `slice(...)` from `co_consts`):
    // its `start` / `stop` are baked into the emitted index constants, so a
    // non-constant slice (whose bounds could differ at runtime) cannot be
    // specialized this way.
    if !key_op.is_constant() {
        return Ok(None);
    }
    let (Some(list_obj), Some(key_obj), Some(value_obj)) = (
        walker_concrete_ref_object(ctx, list_op),
        walker_concrete_ref_object(ctx, key_op),
        walker_concrete_ref_object(ctx, value_op),
    ) else {
        return Ok(None);
    };

    // Gate, all read from the concrete shadows: `target[start:stop:1] =
    // source`, both exact-list Integer storage, `stop - start == len(source)`
    // (no resize), `1 <= slice_len`, `0 <= start <= stop <= len(target)`.
    let (start, slice_len) = unsafe {
        // EXACT list for BOTH target and source: a list subclass shares
        // `ob_type == &LIST_TYPE` but retags `w_class` and may override
        // `__setitem__` (target) or `__iter__` (source); both must route
        // through the generic residual.
        if !pyre_object::pyobject::is_exact_list(list_obj)
            || !pyre_object::is_slice(key_obj)
            || !pyre_object::pyobject::is_exact_list(value_obj)
        {
            return Ok(None);
        }
        // step == 1 (None defaults to 1; an explicit non-1 step needs the
        // strided path).
        let step_o = pyre_object::w_slice_get_step(key_obj);
        let step_is_one = pyre_object::is_none(step_o)
            || (pyre_object::is_int(step_o)
                && !pyre_object::is_bool(step_o)
                && pyre_object::w_int_get_value(step_o) == 1);
        if !step_is_one {
            return Ok(None);
        }
        // start / stop must be explicit non-negative plain ints (None bounds and
        // negative indices route through the generic residual, which normalises
        // them).
        let start_o = pyre_object::w_slice_get_start(key_obj);
        let stop_o = pyre_object::w_slice_get_stop(key_obj);
        if !(pyre_object::is_int(start_o)
            && !pyre_object::is_bool(start_o)
            && pyre_object::is_int(stop_o)
            && !pyre_object::is_bool(stop_o))
        {
            return Ok(None);
        }
        let start = pyre_object::w_int_get_value(start_o);
        let stop = pyre_object::w_int_get_value(stop_o);
        let target_len = pyre_object::w_list_len(list_obj) as i64;
        if start < 0 || stop < start || stop > target_len {
            return Ok(None);
        }
        let slice_len = stop - start;
        let src_len = pyre_object::w_list_len(value_obj) as i64;
        // Same-length only — a resizing slice changes the target length and can
        // switch strategy.
        if slice_len != src_len || slice_len < 1 {
            return Ok(None);
        }
        if !(pyre_object::w_list_uses_int_storage(list_obj)
            && pyre_object::w_list_uses_int_storage(value_obj))
        {
            return Ok(None);
        }
        (start, slice_len)
    };

    // --- emit the specialized IR (walker-native) ---
    // For BOTH target (`list_op`) and source (`value_op`): guard_class LIST +
    // exact `w_class` (a list subclass sharing `ob_type == &LIST_TYPE` but with
    // an overridden `__setitem__` / `__iter__` side-exits to the generic
    // residual) + guard strategy == Integer.  Folds away when the operand is the
    // just-built virtual list.
    let list_type_addr = &pyre_object::pyobject::LIST_TYPE as *const _ as i64;
    let list_instantiate =
        pyre_object::pyobject::get_instantiate(&pyre_object::pyobject::LIST_TYPE);
    let sid_const_val = pyre_object::listobject::ListStrategy::Integer as i64;
    for &lst_op in &[list_op, value_op] {
        if !lst_op.is_constant() && !ctx.trace_ctx.heap_cache().is_class_known(lst_op) {
            let type_const = ctx.trace_ctx.const_int(list_type_addr);
            ctx.trace_ctx
                .record_guard(OpCode::GuardClass, &[lst_op, type_const], 0);
            walker_capture_snapshot_for_last_guard(ctx, op_pc)?;
        }
        ctx.trace_ctx
            .heap_cache_mut()
            .class_now_known(lst_op, list_type_addr);
        walker_guard_exact_w_class(ctx, op_pc, lst_op, list_instantiate)?;

        let strategy = crate::state::opimpl_getfield_gc_i(
            ctx.trace_ctx,
            lst_op,
            crate::descr::list_strategy_descr(),
        );
        let sid_const = ctx.trace_ctx.const_int(sid_const_val);
        ctx.trace_ctx
            .record_guard(OpCode::GuardValue, &[strategy, sid_const], 0);
        walker_capture_snapshot_for_last_guard(ctx, op_pc)?;
        ctx.trace_ctx
            .heap_cache_mut()
            .replace_box(strategy, sid_const);
    }

    // Bounds guard on the target: the highest written index `start + slice_len -
    // 1` must be in range.  For an Integer-strategy list the `W_ListObject`
    // `length` field is 0 — the authoritative length is `int_items.len`, so read
    // it via `list_int_items_len_descr` (exactly as store_subscr's bounds
    // guard).  IntLt(start+slice_len-1, target.int_items.len).
    let tgt_len_box = crate::state::opimpl_getfield_gc_i(
        ctx.trace_ctx,
        list_op,
        crate::descr::list_int_items_len_descr(),
    );
    let last_idx_const = ctx.trace_ctx.const_int(start + slice_len - 1);
    let in_bounds = ctx
        .trace_ctx
        .record_op(OpCode::IntLt, &[last_idx_const, tgt_len_box]);
    let concrete_target_len = unsafe { pyre_object::w_list_len(list_obj) as i64 };
    ctx.trace_ctx.set_opref_concrete(
        in_bounds,
        majit_ir::Value::Int(((start + slice_len - 1) < concrete_target_len) as i64),
    );
    walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardTrue, &[in_bounds])?;

    // Length guard on the source: source.int_items.len == slice_len (folds for
    // the virtual temp; protects a non-virtual source).
    let src_len_box = crate::state::opimpl_getfield_gc_i(
        ctx.trace_ctx,
        value_op,
        crate::descr::list_int_items_len_descr(),
    );
    let src_len_const = ctx.trace_ctx.const_int(slice_len);
    ctx.trace_ctx
        .record_guard(OpCode::GuardValue, &[src_len_box, src_len_const], 0);
    walker_capture_snapshot_for_last_guard(ctx, op_pc)?;
    ctx.trace_ctx
        .heap_cache_mut()
        .replace_box(src_len_box, src_len_const);

    // items[start + j] = source.items[j] for j in 0..slice_len, through the
    // int_items blocks (`list_int_items_block_descr`, matching
    // `emit_typed_list_inline`'s `SetfieldGc`), so a virtual source temp's
    // `SetarrayitemGc` stores fold against these reads.
    let src_block = crate::state::opimpl_getfield_gc_r(
        ctx.trace_ctx,
        value_op,
        crate::descr::list_int_items_block_descr(),
    );
    let tgt_block = crate::state::opimpl_getfield_gc_r(
        ctx.trace_ctx,
        list_op,
        crate::descr::list_int_items_block_descr(),
    );
    for j in 0..slice_len {
        let src_idx = ctx.trace_ctx.const_int(j);
        let src_raw =
            crate::state::trace_int_block_getitem_value(ctx.trace_ctx, src_block, src_idx);
        let tgt_idx = ctx.trace_ctx.const_int(start + j);
        crate::state::trace_int_block_setitem_value(ctx.trace_ctx, tgt_block, tgt_idx, src_raw);
    }

    // Tracing is execution (pyjitpl.py execute_and_record): apply the
    // assignment to the concrete lists now as `slice_len` in-bounds setitems,
    // journaling each displaced element first so a non-committing walk's legacy
    // replay re-executes against the pre-walk heap (FBW_STORE_JOURNAL).  Each
    // `w_list_getitem` / `w_int_new` boxes, and a minor collection there can
    // move any live GC object.  Following the push_roots/pop_roots reload
    // discipline, every live ref is reloaded after each boxing allocation,
    // before its next use: walker operands (`list_obj`/`value_obj`) from the
    // forwarded shadow via `walker_concrete_ref_object`, and the pinned fresh
    // boxes (`src_item`/`displaced`) from their shadow-stack slot via
    // `shadow_stack_get` (the slot index captured just before the pin).
    {
        let _roots = pyre_object::gc_roots::push_roots();
        for j in 0..slice_len {
            let tgt_index = start + j;
            let Some(value_obj) = walker_concrete_ref_object(ctx, value_op) else {
                unreachable!("setslice specialization: operand concrete vanished from the shadow");
            };
            let Some(src_item) = (unsafe { pyre_object::w_list_getitem(value_obj, j) }) else {
                unreachable!("setslice specialization: source index {j} has no element");
            };
            let src_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(src_item);
            let Some(list_obj) = walker_concrete_ref_object(ctx, list_op) else {
                unreachable!("setslice specialization: operand concrete vanished from the shadow");
            };
            let Some(displaced) = (unsafe { pyre_object::w_list_getitem(list_obj, tgt_index) })
            else {
                unreachable!(
                    "setslice specialization: target index {tgt_index} has no element \
                     (bounds gate admitted it)"
                );
            };
            let disp_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(displaced);
            let key_box = pyre_object::w_int_new(tgt_index);
            let key_box = pyre_object::gc_roots::pin_root(key_box);
            let Some(list_obj) = walker_concrete_ref_object(ctx, list_op) else {
                unreachable!("setslice specialization: list concrete vanished mid-apply");
            };
            let src_item = pyre_object::gc_roots::shadow_stack_get(src_slot);
            let displaced = pyre_object::gc_roots::shadow_stack_get(disp_slot);
            fbw_store_journal_push(list_obj, key_box, displaced);
            let stored = unsafe { pyre_object::w_list_setitem(list_obj, tgt_index, src_item) };
            debug_assert!(stored, "setslice specialization: in-bounds store failed");
        }
    }
    Ok(Some(()))
}

/// #57 SLICE 3c (compare): walker-native speculative float specialization
/// for the `COMPARE_OP` helper residual_call (oopspec `CompareOp`), the
/// float analogue of [`try_walker_specialize_compare_op_int`] and the
/// former float-compare arm.  Per operand
/// either `guard_class FLOAT` + `getfield_gc_pure_f`, or (int operand)
/// `guard_class INT` + `getfield_gc_i` + `cast_int_to_float`; then
/// `float_<cmp>` for the raw truth, then NON-fused box to a `W_Bool`.
///
/// Tried as a fallback only after the int compare specialization declines,
/// so two-int operands keep int comparison.  All six `ComparisonOperator`
/// variants are handled (float compare has no deferred operators).
pub(crate) fn try_walker_specialize_compare_op_float<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    op_tag: i64,
    r_args: &[OpRef],
    allboxes: &[OpRef],
    call_descr: &dyn majit_ir::descr::CallDescr,
    dst: usize,
    dst_bank: char,
) -> Result<Option<()>, DispatchError> {
    if !ctx.is_authoritative_executor || dst_bank != 'r' {
        return Ok(None);
    }
    let Some(cmp_op) = pyre_interpreter::runtime_ops::compare_op_from_tag(op_tag) else {
        return Ok(None);
    };
    use pyre_interpreter::bytecode::ComparisonOperator;
    let cmp = match cmp_op {
        ComparisonOperator::Less => OpCode::FloatLt,
        ComparisonOperator::LessOrEqual => OpCode::FloatLe,
        ComparisonOperator::Greater => OpCode::FloatGt,
        ComparisonOperator::GreaterOrEqual => OpCode::FloatGe,
        ComparisonOperator::Equal => OpCode::FloatEq,
        ComparisonOperator::NotEqual => OpCode::FloatNe,
    };
    let Some((
        lhs,
        rhs,
        lhs_obj,
        rhs_obj,
        lhs_is_int,
        rhs_is_int,
        lhs_f64,
        rhs_f64,
        boxed_result_i64,
    )) = walker_float_specialization_operands(ctx, r_args, allboxes, call_descr)
    else {
        return Ok(None);
    };

    // floatobject.py — an int wider than a double represents exactly
    // is compared through its bigint, which this fold cannot express.  Decline
    // so the residual call decides it; the in-range case emits the same
    // precondition as a guard (`exact_int` below).
    let out_of_range = |is_int: bool, obj| {
        is_int && !int_is_exact_as_float(unsafe { pyre_object::w_int_get_value(obj) })
    };
    if out_of_range(lhs_is_int, lhs_obj) || out_of_range(rhs_is_int, rhs_obj) {
        return Ok(None);
    }

    // --- emit the specialized IR (walker-native) ---
    // `_compare` reaches `__lt__` and friends, which a `float` subclass can
    // override, so both arms are pinned: the dispatching coercion adds the float
    // one on top of the int pin every coercion carries.
    let lhs_raw = walker_coerce_dispatching_operand_to_float(
        ctx, op_pc, lhs, lhs_obj, lhs_is_int, lhs_f64, true,
    )?;
    let rhs_raw = walker_coerce_dispatching_operand_to_float(
        ctx, op_pc, rhs, rhs_obj, rhs_is_int, rhs_f64, true,
    )?;
    let truth = ctx.trace_ctx.record_op(cmp, &[lhs_raw, rhs_raw]);
    let folded =
        majit_metainterp::eval_float_cmp(cmp, lhs_f64.to_bits() as i64, rhs_f64.to_bits() as i64);
    ctx.trace_ctx
        .set_opref_concrete(truth, majit_ir::Value::Int(folded));
    // `space.newbool` on the truth: its guard plus the prebuilt singleton.  The
    // residual box is the no-snapshot fallback only.
    let boxed = match walker_newbool_guarded(ctx, op_pc, truth, folded != 0, dst_bank)? {
        Some(boxed) => boxed,
        None => {
            let boxed =
                crate::helpers::emit_trace_bool_value_from_truth(ctx.trace_ctx, truth, false);
            ctx.trace_ctx.set_opref_concrete(
                boxed,
                majit_ir::Value::Ref(majit_ir::GcRef(boxed_result_i64 as usize)),
            );
            boxed
        }
    };
    write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, boxed)?;
    Ok(Some(()))
}

/// #62 LoadGlobal cell-cache fold — walker mirror of the retired trait
/// LOAD_GLOBAL fast path.
///
/// When `ns` is a `W_ModuleDictObject` still in `ModuleDictStrategy` mode
/// whose slot for `name` holds a raw value or an `ObjectMutableCell`, emit
/// `QUASIIMMUT_FIELD(ns, slot)` + `RECORD_KNOWN_RESULT` + an elidable cell
/// lookup that the optimizer folds to the constant cell pointer.  The
/// strategy's `version?` watcher invalidates the loop (GUARD_NOT_INVALIDATED)
/// on any rebind, so the fold is sound while `load_global_fn` itself stays
/// `CallFlavor::Plain`.  Returns `Ok(true)` when the fold was emitted;
/// `Ok(false)` when the receiver is not a foldable cell (the caller then
/// falls through to the generic residual, which stays correct).
///
/// Callers fall back to the residual call when this fold declines. When the
/// loaded global is a function that is then CALLed, folding it to a
/// loop-invariant constant callee routes the call through the FBW call-inlining
/// path (#68).
///
/// Builtins fallback: when `name` is ABSENT from the
/// module dict but resolves through `frame.get_builtin()` (e.g.
/// `raise ValueError` / `except ValueError`), the same cell fold is emitted
/// against the BUILTINS dict, guarded additionally by a `QUASIIMMUT_FIELD` on
/// the module dict so adding `name` to globals (shadowing the builtin) bumps
/// the module-dict `version` and fails the loop's GUARD_NOT_INVALIDATED.  This
/// mirrors `bh_load_global_fn`'s `finditem_str(globals)` →
/// `get_builtin().getdictvalue` fallback chain.
pub(crate) fn try_walker_load_global_cell_fold<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    dst: usize,
    dst_bank: char,
    ns_ptr: usize,
    w_code_ptr: usize,
    frame_ptr: usize,
    namei: i64,
) -> Result<bool, DispatchError> {
    if w_code_ptr == 0 {
        return Ok(false);
    }
    let w_globals = ns_ptr as pyre_object::PyObjectRef;
    // The namespace operand is the fold's authority: both legs end at
    // `guard_current_frame_globals_identity`, which bakes it as the expected
    // `ConstPtr` and declines outright on a null one.  An inlined callee whose
    // namespace register is unseeded presents it as a null `Ref`, so decline
    // here instead of walking the builtins leg, which reads `__builtins__`
    // straight out of it.  The residual re-resolves the globals from the frame
    // it runs on, so declining stays correct.
    if w_globals.is_null() {
        return Ok(false);
    }
    // Raw dict access in the inlined-callee builtins leg below is valid only
    // for an exact plain dict or module dict.  Dict subclasses are legal exec
    // namespaces but use a different object layout, so leave them to the live
    // residual lookup.
    if !unsafe { pyre_object::is_dict(w_globals) } {
        return Ok(false);
    }
    // `namei` is the raw `LOAD_GLOBAL` oparg; bit 0 is the push-NULL flag,
    // so the `co_names` index is `namei >> 1` (mirror `bh_load_global_fn`).
    let name_idx = (namei as usize) >> 1;
    let name = unsafe {
        // The wrapper being non-null does not make its `code_ptr` non-null:
        // `w_code_new_with_hidden_applevel` (pycode.rs) leaves the field
        // null for a gateway builtin or a test fixture, and every sibling
        // name lookup screens it the same way.
        let code_ptr = pyre_interpreter::w_code_get_ptr(w_code_ptr as pyre_object::PyObjectRef);
        if code_ptr.is_null() {
            return Ok(false);
        }
        let code = &*(code_ptr as *const pyre_interpreter::CodeObject);
        match pyre_interpreter::pyframe::load_name_from_code(code, name_idx) {
            Some(n) => n.to_string(),
            None => return Ok(false),
        }
    };
    if emit_module_dict_cell_fold(ctx, op_pc, dst, dst_bank, w_globals, &name)? {
        return Ok(true);
    }

    // Builtins fallback: the name is absent from the
    // `ns_ptr` module dict.  Mirror `bh_load_global_fn`'s second leg —
    // `frame.get_builtin().getdictvalue(name)` — and fold the builtins cell
    // when the name resolves there.  Requires the live frame operand.
    // The builtins fallback needs the module `pick_builtin(w_globals)` picks
    // (`frame.get_builtin()`).  A live frame supplies it directly and also lets
    // us double-check the operand against the frame's AUTHORITATIVE globals
    // — `bh_load_global_fn` re-resolves the globals it consults from the LIVE
    // frame (`frame.get_w_globals()` when the frame owns `w_code`, else the
    // code's bound globals) and IGNORES the `namespace_ptr` operand.  The
    // `ns_ptr` hint usually equals that live dict; when it does not, nothing
    // here can prove what the residual would read, so decline.
    // An INLINED callee has no materialised frame (`frame_ptr == 0`, its
    // `portal_frame_reg` unseeded); derive the builtin module from the concrete
    // globals' `__builtins__` cell instead — the same object `pick_builtin`
    // resolves (`pick_builtin_obj` in baseobjspace.rs) and the one the
    // interpreter fallback would
    // rebuild for the resumed callee frame.  #670 keeps `__builtins__` in every
    // module dict, `ns_ptr` is the callee's own namespace field (so it is the
    // authoritative globals), and guard (a) below watches the globals `version`,
    // so a later `__builtins__` rebind fails the loop exactly as a
    // shadowing-name insert would.
    let w_builtin = if frame_ptr != 0 {
        let frame = unsafe { &*(frame_ptr as *const pyre_interpreter::PyFrame) };
        let live_globals = if frame.pycode as usize == w_code_ptr {
            frame.get_w_globals()
        } else {
            unsafe {
                pyre_interpreter::w_code_get_w_globals(w_code_ptr as pyre_object::PyObjectRef)
            }
        };
        // Only the SAME dict makes the absence provable.  `module_dict_cell_slot_direct`
        // answers `None` both for a name that is absent and for a dict it cannot
        // read at all — a plain dict, or a module dict that ran
        // `switch_to_object_strategy` — so on a different dict its `None` says
        // nothing.  Guard (a) below pins `w_globals`' version, which watches the
        // wrong dict in that case, and the residual it replaces resolves
        // `live_globals`; a name present there would read the builtin instead of
        // the global.
        if live_globals.is_null() || live_globals as usize != w_globals as usize {
            return Ok(false);
        }
        frame.get_builtin()
    } else {
        unsafe { pyre_object::w_dict_getitem_str(w_globals, "__builtins__") }
            .unwrap_or(pyre_object::PY_NULL)
    };
    emit_builtins_cell_fold(ctx, op_pc, dst, dst_bank, w_globals, w_builtin, &name)
}

/// Builtins-fallback half of the LOAD_GLOBAL and module-scope LOAD_NAME cell
/// folds: the name resolves through the frame's builtin module rather than the
/// module dict.  Mirrors `_load_global`'s second leg,
/// `get_builtin().getdictvalue(varname)`, and is reached only once
/// [`emit_module_dict_cell_fold`] has declined.
///
/// Two guards carry it.  (a) The name must stay ABSENT from the module dict,
/// which pinning that dict's `version?` is what proves: the insert that would
/// shadow the builtin runs `mutated()` and fails GUARD_NOT_INVALIDATED.
/// (b) The builtins value itself folds through [`emit_namespace_cell_fold`],
/// whose `QUASIIMMUT_FIELD` on the builtins dict fails the loop on a rebind or
/// delete there.
///
/// Returns `Ok(false)` — the caller then keeps the live residual — for a name
/// still present in the module dict, a missing or non-module builtin, an
/// unfoldable builtins slot (absent / null / `IntMutableCell` / movable), or a
/// movable builtins dict.
fn emit_builtins_cell_fold<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    dst: usize,
    dst_bank: char,
    w_globals: pyre_object::PyObjectRef,
    w_builtin: pyre_object::PyObjectRef,
    name: &str,
) -> Result<bool, DispatchError> {
    // `emit_module_dict_cell_fold` returns `false` for BOTH an absent name and
    // a present-but-unfoldable one (`IntMutableCell` / strategy switched).
    // Only an ABSENT name may fall through to the builtins fold — a
    // present global shadows the builtin, so keep the residual (which reads the
    // live globals slot) when the slot still exists.
    if crate::state::module_dict_cell_slot_direct(w_globals, name).is_some() {
        return Ok(false);
    }
    if w_builtin.is_null() || !unsafe { pyre_object::is_module(w_builtin) } {
        return Ok(false);
    }
    let w_builtin_dict = unsafe { pyre_object::w_module_get_w_dict(w_builtin) };
    if w_builtin_dict.is_null() {
        return Ok(false);
    }
    let Some(b_slot) = crate::state::module_dict_cell_slot_direct(w_builtin_dict, name) else {
        return Ok(false);
    };
    let Some(b_stored) = crate::state::module_dict_cell_value_direct(w_builtin_dict, b_slot) else {
        return Ok(false);
    };
    if b_stored.is_null() || unsafe { pyre_object::celldict::is_int_mutable_cell(b_stored) } {
        return Ok(false);
    }
    if majit_gc::can_move(majit_ir::GcRef(b_stored as usize)) {
        return Ok(false);
    }
    // Guard (a): the name must stay ABSENT from the module dict so the lookup
    // keeps falling through to builtins.  Pinning the module dict's `version?`
    // is what proves that: the new-key insert that would shadow the builtin
    // runs `mutated()`, which fails GUARD_NOT_INVALIDATED.  It is the same
    // field a present-name fold on this namespace pins, so the two share one
    // marker.
    if !guard_current_frame_globals_identity(ctx, op_pc, w_globals)? {
        return Ok(false);
    }
    if !walker_pin_namespace_version(ctx, op_pc, w_globals)? {
        return Ok(false);
    }
    // Guard (b): the builtins value for `name` must be unchanged.  The
    // `emit_namespace_cell_fold` below records a `QUASIIMMUT_FIELD` on the
    // builtins dict + the elidable cell lookup, so a rebind/del of the
    // builtin bumps the builtins-dict `version` and fails the loop.
    if majit_gc::can_move(majit_ir::GcRef(w_builtin_dict as usize)) {
        return Ok(false);
    }
    if !emit_namespace_cell_fold(
        ctx,
        op_pc,
        dst,
        dst_bank,
        w_builtin_dict,
        b_slot,
        b_stored,
        false,
    )? {
        return Ok(false);
    }
    Ok(true)
}

/// LoadName cell fold — module-scope LOAD_NAME mirror of
/// [`try_walker_load_global_cell_fold`].  At module scope the frame's
/// `w_locals` is null and `w_locals` aliases `w_globals`
/// (`createframe` sets `debugdata.w_locals = w_globals_storage`,
/// pyframe.rs), so `load_name_value`'s probe + LOAD_GLOBAL fallthrough
/// both resolve in `w_globals` — the same dict the global cell fold reads.
/// A non-module frame (class body / `exec(code, g, l)` with separate locals)
/// has a non-null `w_locals`, so the gate routes it to the live
/// residual `bh_load_name_fn`.
///
/// Builtins fallback: when `name` is absent from the module dict, module-scope
/// `LOAD_NAME` falls through via `load_global_value` to
/// `frame.get_builtin().getdictvalue(name)`.  The builtins cell fold pins the
/// module dict `version?` so a later global insertion that shadows the builtin
/// fails GUARD_NOT_INVALIDATED, then folds the builtins dict cell like the
/// LOAD_GLOBAL fallback.
pub(crate) fn try_walker_load_name_cell_fold<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    dst: usize,
    dst_bank: char,
    frame_ptr: usize,
    w_name_ptr: usize,
) -> Result<bool, DispatchError> {
    if frame_ptr == 0 {
        return Ok(false);
    }
    let frame = unsafe { &*(frame_ptr as *const pyre_interpreter::pyframe::PyFrame) };
    let w_globals = frame.get_w_globals();
    if w_globals.is_null() {
        return Ok(false);
    }
    // Only module scope (w_locals IS w_globals) is foldable. Module frames bind
    // `w_locals = w_globals` (pyframe.py); a `w_locals`
    // that is a DIFFERENT object means the LOAD_NAME probe targets a separate
    // locals namespace the module-dict cell fold (keyed on `w_globals`) would
    // skip. (Class bodies / `exec(code, g, l)` set a separate one; they also do
    // not portal-trace, so the only LOAD_NAME the walker reaches in practice is
    // module-scope.)
    let w_locals = frame.get_w_locals();
    if !w_locals.is_null() && !std::ptr::eq(w_locals, w_globals) {
        return Ok(false);
    }
    let name = unsafe {
        pyre_object::unicodeobject::w_str_get_value(w_name_ptr as pyre_object::PyObjectRef)
    };
    if emit_module_dict_cell_fold(ctx, op_pc, dst, dst_bank, w_globals, name)? {
        return Ok(true);
    }
    emit_builtins_cell_fold(
        ctx,
        op_pc,
        dst,
        dst_bank,
        w_globals,
        frame.get_builtin(),
        name,
    )
}

/// StoreName/StoreGlobal cell fold — module-scope store dual of
/// [`try_walker_load_name_cell_fold`].  Folds `i = <int>` on a hot module
/// global whose slot has stabilised to an `IntMutableCell` (the in-place
/// shape `write_cell` reaches after the 2nd int store) to a single
/// `setfield_gc_i(cell, intvalue)`, eliding the value boxing + residual dict
/// setitem.  Declines (→ residual `bh_store_name_fn`, which runs the full
/// `write_cell`) when the frame is non-module, the slot is not an immovable
/// `IntMutableCell`, or the value is not a provably-plain-int box (bool /
/// int-subclass / long / object all fall through — `write_cell` REPLACES the
/// cell + bumps the version for those, which the setfield fast path must not).
pub(crate) fn try_walker_store_name_cell_fold<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    helper: majit_ir::RuntimeHelperKind,
    frame_ptr: usize,
    w_name_ptr: usize,
    value_opref: OpRef,
) -> Result<bool, DispatchError> {
    if frame_ptr == 0 {
        return Ok(false);
    }
    let frame = unsafe { &*(frame_ptr as *const pyre_interpreter::pyframe::PyFrame) };
    let w_globals = frame.get_w_globals();
    if w_globals.is_null() {
        return Ok(false);
    }
    // STORE_NAME writes `get_or_create_w_locals`, so only a module frame —
    // where `w_locals` aliases `w_globals` — targets the dict this folds.  An
    // ABSENT `w_locals` does NOT stand in for globals on the write path (unlike
    // the LOAD fold): the store would land in a fresh locals mapping while the
    // fold set the module cell.  STORE_GLOBAL names globals outright, and its
    // frame is a function frame whose `w_locals` is legitimately null, so the
    // gate must not apply to it.
    if helper == majit_ir::RuntimeHelperKind::StoreName {
        let w_locals = frame.get_w_locals();
        if !std::ptr::eq(w_locals, w_globals) {
            return Ok(false);
        }
    }
    let name = unsafe {
        pyre_object::unicodeobject::w_str_get_value(w_name_ptr as pyre_object::PyObjectRef)
    };
    // Slot must hold an immovable `IntMutableCell`.  `can_move` gates the same
    // baked-address relocation hazard as the LOAD fold; mutable cells are
    // `malloc_typed` (never nursery) so a stabilised int global folds.
    let Some(slot) = crate::state::module_dict_cell_slot_direct(w_globals, name) else {
        return Ok(false);
    };
    let Some(stored) = crate::state::module_dict_cell_value_direct(w_globals, slot) else {
        return Ok(false);
    };
    if stored.is_null() || !unsafe { pyre_object::celldict::is_int_mutable_cell(stored) } {
        return Ok(false);
    }
    if majit_gc::can_move(majit_ir::GcRef(stored as usize)) {
        return Ok(false);
    }
    // The stored value must be a provably-plain-int box. `is_plain_int1` accepts
    // a fits-int `W_LongObject`, whose `write_cell` REPLACES the cell rather than
    // mutating `intvalue`, so exclude `long` explicitly; the remaining int box's
    // raw `intvalue` (populated only by JIT int boxes, `emit_box_int_inline`) is
    // recovered by the heapcache lookup, so the setfield needs no runtime class
    // guard — exactly as pypy's optimized trace folds the `is_plain_int1` check
    // away for an `int_add` result. (bool / int-subclass are already excluded by
    // `is_plain_int1`.)
    let is_plain_int = matches!(
        ctx.trace_ctx.box_value(value_opref),
        Some(majit_ir::Value::Ref(majit_ir::GcRef(p)))
            if p != 0
                && unsafe { pyre_object::listobject::is_plain_int1(p as pyre_object::PyObjectRef) }
                && !unsafe { pyre_object::is_long(p as pyre_object::PyObjectRef) }
    );
    if !is_plain_int {
        return Ok(false);
    }
    let Some(raw_int) = ctx
        .trace_ctx
        .heapcache_getfield_cached(value_opref, crate::descr::int_intval_descr().index())
    else {
        return Ok(false);
    };
    // The eager concrete write needs the raw int the store applies; a
    // raw-int box with no concrete shadow declines to the residual.
    let Some(majit_ir::Value::Int(new_int)) = ctx.trace_ctx.box_value(raw_int) else {
        return Ok(false);
    };
    emit_namespace_cell_store_fold(ctx, op_pc, w_globals, slot, stored, raw_int, new_int)
}
