//! JIT helper functions — `extern "C"` wrappers called from compiled traces.
//!
//! The JIT backend (Cranelift) emits C-ABI calls to these functions.
//! Each wraps a pyre-object or pyre-interpreter operation with the
//! correct calling convention and integer-based parameter passing.

use majit_ir::{EffectInfo, ExtraEffect, GcRef, OopSpecIndex, OpCode, OpRef, Type, Value};
use majit_metainterp::{TraceCtx, default_effect_info};

use pyre_interpreter::{
    PyBigInt, PyError, binary_op_tag, compare_op_tag, jit_next, jit_range_iter_next_or_null,
    jit_sequence_getitem,
};
use pyre_interpreter::{
    jit_binary_value_from_tag, jit_bool_value_from_truth, jit_compare_value_from_tag, jit_getattr,
    jit_setattr, jit_setitem, jit_truth_value, jit_unary_invert_value, jit_unary_negative_value,
};
use pyre_object::*;

pub use pyre_interpreter::{
    FlatBuildKind, callable_call_helper, flat_build_helper, jit_build_list_0, jit_build_list_1,
    jit_build_list_2, jit_build_list_3, jit_build_list_4, jit_build_list_5, jit_build_list_6,
    jit_build_list_7, jit_build_list_8, jit_build_map_0, jit_build_map_1, jit_build_map_2,
    jit_build_map_3, jit_build_map_4, jit_build_tuple_0, jit_build_tuple_1, jit_build_tuple_2,
    jit_build_tuple_3, jit_build_tuple_4, jit_build_tuple_5, jit_build_tuple_6, jit_build_tuple_7,
    jit_build_tuple_8, jit_call_callable_0, jit_call_callable_1, jit_call_callable_2,
    jit_call_callable_3, jit_call_callable_4, jit_call_callable_5, jit_call_callable_6,
    jit_call_callable_7, jit_call_callable_8, jit_load_name_from_namespace,
    jit_make_function_from_globals, jit_store_name_to_namespace, known_builtin_call_helper,
    known_function_call_helper, register_jit_function_caller,
};

fn trace_name_args(ctx: &mut TraceCtx, name: &str) -> [OpRef; 2] {
    [
        ctx.const_int(name.as_ptr() as usize as i64),
        ctx.const_int(name.len() as i64),
    ]
}

pub fn emit_trace_call_int(ctx: &mut TraceCtx, helper: *const (), args: &[OpRef]) -> OpRef {
    ctx.call_int(helper, args)
}

pub fn emit_trace_call_int_typed(
    ctx: &mut TraceCtx,
    helper: *const (),
    args: &[OpRef],
    arg_types: &[Type],
) -> OpRef {
    // pyjitpl.py:1995-2068 do_residual_call parity: thread the
    // codewriter-analyzed `EffectInfo` through `record_nospec`. The
    // codewriter's `CallControl::getcalldescr`
    // (`majit-translate/src/codewriter/call.rs`) ports call.py:210-335
    // including the raise / random-effects / write / collect /
    // virtualizable / quasi-immut analyzers; the gap is the trace-side
    // plumbing — pyre-jit-trace helpers live outside the codewriter
    // pipeline so the analyzer's per-callee EI never reaches this
    // emit site. Until per-helper EI registration lands,
    // fall back to the conservative `default_effect_info()`
    // (≡ `effectinfo.MOST_GENERAL` for unanalyzed callees: CanRaise +
    // all-writes-set bitmasks).
    ctx.call_int_typed_with_effect(helper, args, arg_types, default_effect_info())
}

/// Float-bank counterpart of [`emit_trace_call_int_typed`], recording a `CallF`
/// for a helper whose C ABI returns its `f64` in a floating-point register.
/// Carries the same conservative `default_effect_info()`, so the two differ only
/// in the result bank.
pub fn emit_trace_call_float_typed(
    ctx: &mut TraceCtx,
    helper: *const (),
    args: &[OpRef],
    arg_types: &[Type],
) -> OpRef {
    ctx.call_float_typed_with_effect(helper, args, arg_types, default_effect_info())
}

/// `pyjitpl.py:1941-1958 MIFrame.execute_varargs(opnum, argboxes, descr,
/// exc=False, pure=True)` parity for direct trace emit paths.
///
/// `emit_trace_call_int_typed` calls into the tracer with
/// `default_effect_info()` (`effectinfo.MOST_GENERAL`, CanRaise +
/// all-writes-set), so even an `#[elidable_cannot_raise]` callee is
/// recorded as a plain `CallI`.  This wrapper threads an explicit
/// `ElidableCannotRaise` `EffectInfo` through
/// `record_result_of_call_pure` (`pyjitpl.py:3553-3579`) and patches the
/// trace to `CallPureI`.
///
/// `concrete_arg_values` follows `_build_allboxes`
/// (`pyjitpl.py:1960-1993`): the funcbox's concrete value sits in slot 0
/// and the per-argument concrete values follow.  Direct trace-emit paths
/// have no jitcode-dispatch frame-operand-fetch, so the caller supplies
/// the values directly.
pub fn emit_trace_call_int_typed_elidable_cannot_raise(
    ctx: &mut TraceCtx,
    helper: *const (),
    args: &[OpRef],
    arg_types: &[Type],
    concrete_arg_values: &[Value],
    concrete_result: Value,
) -> OpRef {
    let effect = EffectInfo::new(ExtraEffect::ElidableCannotRaise, OopSpecIndex::None);
    ctx.call_typed_with_effect_pure(
        OpCode::CallI,
        helper,
        args,
        arg_types,
        Type::Int,
        effect,
        concrete_arg_values,
        concrete_result,
    )
}

pub fn emit_trace_call_ref(ctx: &mut TraceCtx, helper: *const (), args: &[OpRef]) -> OpRef {
    ctx.call_ref(helper, args)
}

pub fn emit_trace_call_ref_typed(
    ctx: &mut TraceCtx,
    helper: *const (),
    args: &[OpRef],
    arg_types: &[Type],
) -> OpRef {
    // See emit_trace_call_int_typed for the plumbing-gap rationale.
    ctx.call_ref_typed_with_effect(helper, args, arg_types, default_effect_info())
}

pub fn emit_trace_call_ref_typed_elidable_cannot_raise(
    ctx: &mut TraceCtx,
    helper: *const (),
    args: &[OpRef],
    arg_types: &[Type],
    concrete_arg_values: &[Value],
    concrete_result: Value,
) -> OpRef {
    let effect = EffectInfo::new(ExtraEffect::ElidableCannotRaise, OopSpecIndex::None);
    ctx.call_typed_with_effect_pure(
        OpCode::CallR,
        helper,
        args,
        arg_types,
        Type::Ref,
        effect,
        concrete_arg_values,
        concrete_result,
    )
}

/// `celldict.py:42-54 getdictvalue_no_unwrapping` residual: returns the
/// raw stored value-or-cell at `slot` of the module dict `namespace_ptr`
/// (the object, not its storage), _not_ unwrapped.  The elidable form of
/// this — keyed on `version?` via `QuasiimmutField` — folds to a constant
/// cell pointer; the live `cell.w_value` read is emitted separately as a
/// `GetfieldGcR`.  Null on a non-module dict, missing slot, or after
/// `switch_to_object_strategy`.
pub(crate) extern "C" fn jit_namespace_cell_lookup(namespace_ptr: i64, slot: i64) -> i64 {
    let w_globals = namespace_ptr as pyre_object::PyObjectRef;
    if w_globals.is_null() || slot < 0 {
        return PY_NULL as i64;
    }
    let cell =
        unsafe { pyre_object::dictmultiobject::module_dict_cell_at(w_globals, slot as usize) };
    cell.unwrap_or(PY_NULL) as i64
}

pub(crate) fn namespace_slot_lookup_values(
    func_ptr: *const (),
    w_globals: pyre_object::PyObjectRef,
    slot: usize,
) -> [Value; 3] {
    [
        Value::Int(func_ptr as usize as i64),
        Value::Ref(GcRef(w_globals as usize)),
        Value::Int(slot as i64),
    ]
}

pub(crate) fn namespace_slot_lookup_result(result: PyObjectRef) -> Value {
    Value::Ref(GcRef(result as usize))
}

/// `typeobject.py:516 _pure_lookup_where_with_method_cache` residual: the
/// `@elidable` method-cache lookup keyed on `(w_type, w_name, version_tag)`,
/// recorded as a foldable `CALL_PURE_R` so repeated same-key lookups in a
/// hot loop collapse to a constant `w_descr` pointer (null = `None`).  The
/// promoted `version_tag` and interned immortal `w_name`
/// (`box_str_constant`) are the green tokens the trace folds on.
///
/// A plain `extern "C"` i64-ABI wrapper, like `jit_namespace_cell_lookup`:
/// the `#[elidable]` macro cannot emit a trampoline for the
/// `PyObjectRef`-aliased signature (it recognises only literal pointer /
/// primitive types), so the recordable surface is this wrapper, recorded
/// with an explicit `ElidableCannotRaise` effect — the raw-pointer return
/// (null = `None`) genuinely cannot raise.  Null on a null `w_type` /
/// `w_name`; callers (the front door) already guarded `is_type` /
/// `version_tag == 0`.
pub extern "C" fn jit_lookup_where_with_method_cache(
    w_type: i64,
    w_name: i64,
    version_tag: i64,
) -> i64 {
    let w_type = w_type as PyObjectRef;
    let w_name = w_name as PyObjectRef;
    if w_type.is_null() || w_name.is_null() {
        return PY_NULL as i64;
    }
    let w_descr = unsafe {
        pyre_interpreter::_pure_lookup_where_with_method_cache(w_type, w_name, version_tag as u64)
    };
    w_descr as i64
}

/// `mapdict.py:846-847 getdictvalue` residual: the instance-dict shadowing
/// read the `LOAD_METHOD` fast path performs after the type lookup
/// (`callmethod.py:66 w_value = w_obj.getdictvalue(space, name)`), to make
/// sure no instance attribute shadows the class method.  Returns the
/// shadowing value or `PY_NULL` when the attribute is absent from the
/// instance dict.
///
/// A plain `extern "C"` i64-ABI wrapper (like [`jit_namespace_cell_lookup`]),
/// recorded as a normal residual call — NOT pure: the instance dict mutates,
/// so the result is guarded `null` per iteration rather than folded.  `w_name`
/// is the interned immortal name pointer; the body reads it back via
/// `w_str_get_wtf8`, mirroring [`jit_lookup_where_with_method_cache`].  Null
/// on a null receiver / name or a non-instance receiver (the fast path
/// already pinned the receiver type with `guard_class`).
pub extern "C" fn jit_instance_getdictvalue(w_obj: i64, w_name: i64) -> i64 {
    let w_obj = w_obj as PyObjectRef;
    let w_name = w_name as PyObjectRef;
    if w_obj.is_null() || w_name.is_null() || !unsafe { is_instance(w_obj) } {
        return PY_NULL as i64;
    }
    let name = unsafe { w_str_get_wtf8(w_name) };
    let w_value = unsafe {
        pyre_interpreter::objspace::std::mapdict::instance_node_getdictvalue(w_obj, name)
    };
    w_value.unwrap_or(PY_NULL) as i64
}

/// mapdict.py:914-916 `_mapdict_read_storage(storageindex)` — the LOAD_ATTR
/// fast-path storage read.  A plain `extern "C"` i64-ABI wrapper recorded as a
/// residual call, like [`jit_instance_getdictvalue`]: the value changes per
/// instance so it is not folded, but `storageindex` is a green constant (the
/// fast path resolved it off the promoted map) and the surrounding class / map
/// / version_tag guards already established the shape, so this replaces
/// `getattr_str`'s MRO walk + name hash + descriptor dispatch with a single
/// `storage[index]` fetch.  Null receiver / non-instance returns `PY_NULL`
/// (the fast path pinned the receiver type with `guard_class`, so this only
/// guards against a torn recording).
pub extern "C" fn jit_mapdict_read(w_obj: i64, storageindex: i64) -> i64 {
    let w_obj = w_obj as PyObjectRef;
    if w_obj.is_null() || !unsafe { is_instance(w_obj) } {
        return PY_NULL as i64;
    }
    unsafe {
        pyre_interpreter::objspace::std::mapdict::read_boxed_storage(w_obj, storageindex as usize)
            as i64
    }
}

/// Non-forcing boxed write for an existing mapdict attribute.  The guarded
/// instance class and exact map pin the storage index, and a boxed slot accepts
/// the incoming object reference directly (mapdict.py:446-447).  A torn
/// recording with a null/non-instance receiver is a defensive no-op.
pub extern "C" fn jit_mapdict_boxed_write(w_obj: i64, storageindex: i64, value: i64) {
    let w_obj = w_obj as PyObjectRef;
    if w_obj.is_null() || !unsafe { is_instance(w_obj) } {
        return;
    }
    unsafe {
        pyre_interpreter::objspace::std::mapdict::write_boxed_storage(
            w_obj,
            storageindex as usize,
            value as PyObjectRef,
        );
    }
}

/// Raw unboxed counterpart of [`jit_mapdict_read`].  The guarded map pins the
/// shared longlong-list coordinates, so this non-forcing helper performs only
/// `_prim_direct_read`'s storage read (mapdict.py:600-601); boxing stays in the
/// trace so an immediate consumer can virtualize it away.  Null receiver /
/// non-instance returns zero only for a torn recording.
pub extern "C" fn jit_mapdict_unboxed_read_raw(
    w_obj: i64,
    storageindex: i64,
    listindex: i64,
) -> i64 {
    let w_obj = w_obj as PyObjectRef;
    if w_obj.is_null() || !unsafe { is_instance(w_obj) } {
        return 0;
    }
    unsafe {
        pyre_interpreter::objspace::std::mapdict::read_unboxed_storage_raw(
            w_obj,
            storageindex as usize,
            listindex as usize,
        )
    }
}

/// Float-bank counterpart of [`jit_mapdict_unboxed_read_raw`].  Unboxed float
/// storage already contains the value's IEEE-754 bit pattern, so this helper
/// performs the raw read and reconstructs the float (mapdict.py:577-584).
/// Null receiver / non-instance returns zero only for a torn recording.
pub extern "C" fn jit_mapdict_unboxed_read_f(w_obj: i64, storageindex: i64, listindex: i64) -> f64 {
    let w_obj = w_obj as PyObjectRef;
    if w_obj.is_null() || !unsafe { is_instance(w_obj) } {
        return 0.0;
    }
    unsafe {
        f64::from_bits(
            pyre_interpreter::objspace::std::mapdict::read_unboxed_storage_raw(
                w_obj,
                storageindex as usize,
                listindex as usize,
            ) as u64,
        )
    }
}

/// Non-forcing raw write for a mapdict unboxed attribute.  The full-body
/// walker has already guarded the receiver's instance class and exact map,
/// and proved that the incoming value is an integer, so this is only the
/// same-type longlong-list update (mapdict.py:615-619).  A torn recording can
/// reach the wrapper with a null/non-instance receiver; keep that defensive
/// path a no-op.
pub extern "C" fn jit_mapdict_unboxed_write_raw(
    w_obj: i64,
    storageindex: i64,
    listindex: i64,
    raw: i64,
) {
    let w_obj = w_obj as usize as PyObjectRef;
    if w_obj.is_null() || !unsafe { is_instance(w_obj) } {
        return;
    }
    unsafe {
        pyre_interpreter::objspace::std::mapdict::write_unboxed_storage_raw(
            w_obj,
            storageindex as usize,
            listindex as usize,
            raw,
        );
    }
}

/// Float-bank counterpart of [`jit_mapdict_unboxed_write_raw`].  A same-type
/// float update writes its IEEE-754 bit pattern to the existing longlong-list
/// slot (mapdict.py:615-619).  A torn recording with a null/non-instance
/// receiver is a defensive no-op.
pub extern "C" fn jit_mapdict_unboxed_write_f(
    w_obj: i64,
    storageindex: i64,
    listindex: i64,
    value: f64,
) {
    let w_obj = w_obj as usize as PyObjectRef;
    if w_obj.is_null() || !unsafe { is_instance(w_obj) } {
        return;
    }
    unsafe {
        pyre_interpreter::objspace::std::mapdict::write_unboxed_storage_raw(
            w_obj,
            storageindex as usize,
            listindex as usize,
            value.to_bits() as i64,
        );
    }
}

pub fn emit_trace_call_void(ctx: &mut TraceCtx, helper: *const (), args: &[OpRef]) {
    ctx.call_void(helper, args);
}

/// Record a void residual whose hand-written `extern "C"` helper returns a
/// dummy machine word (`-> i64`, value ignored) — the convention of this
/// module's i64-ABI wrappers ([`jit_store_name_to_namespace`],
/// [`jit_list_append`]). The word-ABI descr lets a signature-exact backend
/// lowering (wasm direct `call_indirect`) call the helper in-module. A
/// helper that genuinely returns `()` must use `ctx.call_void_typed`
/// instead.
///
/// `effect_info`: these helpers write the heap (namespace cells, list
/// storage), so the caller must supply the effect — normally
/// `EffectInfo::MOST_GENERAL` (`graphanalyze.py:60
/// analyze_external_call` top for an unanalyzed external writer).  The
/// opcode-default empty write set would let optheap CSE a getfield
/// across the call: `acc = acc + a; acc = acc + b` at module level then
/// reuses the pre-store cell value and drops the first term.
pub fn emit_trace_call_void_word_abi(
    ctx: &mut TraceCtx,
    helper: *const (),
    args: &[OpRef],
    arg_types: &[Type],
    effect_info: majit_ir::EffectInfo,
) {
    ctx.call_void_typed_word_abi(helper, args, arg_types, effect_info);
}

pub fn emit_trace_call_may_force_ref_typed(
    ctx: &mut TraceCtx,
    helper: *const (),
    args: &[OpRef],
    arg_types: &[Type],
) -> OpRef {
    ctx.call_may_force_ref_typed(helper, args, arg_types)
}

pub fn emit_trace_call_may_force_void_typed(
    ctx: &mut TraceCtx,
    helper: *const (),
    args: &[OpRef],
    arg_types: &[Type],
) {
    ctx.call_may_force_void_typed(helper, args, arg_types);
}

pub fn emit_trace_call_may_force_int_typed(
    ctx: &mut TraceCtx,
    helper: *const (),
    args: &[OpRef],
    arg_types: &[Type],
) -> OpRef {
    ctx.call_may_force_int_typed(helper, args, arg_types)
}

pub fn emit_trace_build_flat(
    ctx: &mut TraceCtx,
    kind: FlatBuildKind,
    items: &[OpRef],
) -> Result<OpRef, PyError> {
    let helper_count = match kind {
        FlatBuildKind::Map => items.len() / 2,
        FlatBuildKind::List | FlatBuildKind::Tuple => items.len(),
    };
    let Some(helper) = flat_build_helper(kind, helper_count) else {
        let opname = match kind {
            FlatBuildKind::List => "list",
            FlatBuildKind::Tuple => "tuple",
            FlatBuildKind::Map => "map",
        };
        return Err(PyError::type_error(format!(
            "{opname} build arity not supported by JIT"
        )));
    };
    let arg_types = vec![Type::Ref; items.len()];
    Ok(ctx.call_ref_typed_with_effect(helper, items, &arg_types, default_effect_info()))
}

pub fn emit_trace_call_callable(
    ctx: &mut TraceCtx,
    frame: OpRef,
    callable: OpRef,
    args: &[OpRef],
) -> Result<OpRef, PyError> {
    let helper = callable_call_helper(args.len())
        .ok_or_else(|| PyError::type_error("call arity not supported by JIT"))?;
    let mut call_args = vec![frame, callable];
    call_args.extend_from_slice(args);
    let mut arg_types = vec![Type::Ref, Type::Ref];
    arg_types.extend(std::iter::repeat_n(Type::Ref, args.len()));
    Ok(ctx.call_may_force_ref_typed(helper, &call_args, &arg_types))
}

pub fn emit_trace_call_known_builtin(
    ctx: &mut TraceCtx,
    callable: OpRef,
    args: &[OpRef],
) -> Result<OpRef, PyError> {
    let helper = known_builtin_call_helper(args.len())
        .ok_or_else(|| PyError::type_error("builtin call arity not supported by JIT"))?;
    let mut call_args = vec![callable];
    call_args.extend_from_slice(args);
    let mut arg_types = vec![Type::Ref];
    arg_types.extend(std::iter::repeat_n(Type::Ref, args.len()));
    Ok(ctx.call_ref_typed_with_effect(helper, &call_args, &arg_types, default_effect_info()))
}

pub fn emit_trace_call_known_function(
    ctx: &mut TraceCtx,
    frame: OpRef,
    callable: OpRef,
    args: &[OpRef],
) -> Result<OpRef, PyError> {
    let helper = known_function_call_helper(args.len())
        .ok_or_else(|| PyError::type_error("function call arity not supported by JIT"))?;
    let mut call_args = vec![frame, callable];
    call_args.extend_from_slice(args);
    let mut arg_types = vec![Type::Ref, Type::Ref];
    arg_types.extend(std::iter::repeat_n(Type::Ref, args.len()));
    Ok(ctx.call_may_force_ref_typed(helper, &call_args, &arg_types))
}

pub fn emit_trace_unpack_sequence(
    ctx: &mut TraceCtx,
    seq: OpRef,
    count: usize,
) -> Result<Vec<OpRef>, PyError> {
    let mut items = Vec::with_capacity(count);
    for idx in 0..count {
        let idx_const = ctx.const_int(idx as i64);
        items.push(emit_trace_call_ref_typed(
            ctx,
            jit_sequence_getitem as *const (),
            &[seq, idx_const],
            &[Type::Ref, Type::Int],
        ));
    }
    Ok(items)
}

pub fn emit_trace_load_name_from_namespace(
    ctx: &mut TraceCtx,
    frame: OpRef,
    namespace: OpRef,
    name: &str,
) -> OpRef {
    let [name_ptr, name_len] = trace_name_args(ctx, name);
    emit_trace_call_ref_typed(
        ctx,
        jit_load_name_from_namespace as *const (),
        &[frame, namespace, name_ptr, name_len],
        &[Type::Ref, Type::Ref, Type::Int, Type::Int],
    )
}

pub fn emit_trace_store_name_to_namespace(
    ctx: &mut TraceCtx,
    namespace: OpRef,
    name: &str,
    value: OpRef,
) {
    let [name_ptr, name_len] = trace_name_args(ctx, name);
    // The helper runs `w_dict_setitem_str` → ModuleDictStrategy
    // `write_cell` — an in-place `ObjectMutableCell.w_value` write with
    // no version bump, exactly what `load_name_value`'s cell fast path
    // reads back as `GetfieldGcR(cell)`.  MOST_GENERAL makes the
    // optimizer drop that field cache so the next LOAD re-reads the
    // cell instead of reusing the pre-store value.
    emit_trace_call_void_word_abi(
        ctx,
        jit_store_name_to_namespace as *const (),
        &[namespace, name_ptr, name_len, value],
        &[Type::Ref, Type::Int, Type::Int, Type::Ref],
        majit_ir::EffectInfo::MOST_GENERAL,
    );
}

pub fn emit_trace_truth_value(ctx: &mut TraceCtx, value: OpRef) -> OpRef {
    emit_trace_call_may_force_int_typed(ctx, jit_truth_value as *const (), &[value], &[Type::Ref])
}

pub fn emit_trace_bool_value_from_truth(ctx: &mut TraceCtx, truth: OpRef, negate: bool) -> OpRef {
    let truth = if negate {
        let one = ctx.const_int(1);
        let neg = ctx.record_op(OpCode::IntSub, &[one, truth]);
        // Box(value) parity: derive negated bool from truth's Box.value.
        if let Some(majit_ir::Value::Int(n)) = ctx.box_value(truth) {
            ctx.set_opref_concrete(neg, majit_ir::Value::Int(1 - n));
        }
        neg
    } else {
        truth
    };
    // `space.newbool` selects the `w_True` / `w_False` singleton: it cannot
    // raise, so the residual is EF_CANNOT_RAISE (no trailing GuardNoException).
    // It is NOT elidable here — the boxed bool feeds consumers (COMPARE_OP
    // boxing, the bool-bitwise lowering) that need a recorded OpRef, so a pure
    // call folding to an inline Const would break their `OpRef` reads.
    ctx.call_ref_typed_with_effect(
        jit_bool_value_from_truth as *const (),
        &[truth],
        &[Type::Int],
        EffectInfo::new(ExtraEffect::CannotRaise, OopSpecIndex::None),
    )
}

pub fn emit_trace_binary_value(
    ctx: &mut TraceCtx,
    a: OpRef,
    b: OpRef,
    op: pyre_interpreter::bytecode::BinaryOperator,
) -> Result<OpRef, PyError> {
    let Some(tag) = binary_op_tag(op) else {
        return Err(PyError::type_error(format!(
            "binary operation {op:?} not yet traceable"
        )));
    };
    let tag = ctx.const_int(tag);
    Ok(emit_trace_call_may_force_ref_typed(
        ctx,
        jit_binary_value_from_tag as *const (),
        &[a, b, tag],
        &[Type::Ref, Type::Ref, Type::Int],
    ))
}

pub fn emit_trace_compare_value(
    ctx: &mut TraceCtx,
    a: OpRef,
    b: OpRef,
    op: pyre_interpreter::bytecode::ComparisonOperator,
) -> OpRef {
    let tag = ctx.const_int(compare_op_tag(op));
    emit_trace_call_may_force_ref_typed(
        ctx,
        jit_compare_value_from_tag as *const (),
        &[a, b, tag],
        &[Type::Ref, Type::Ref, Type::Int],
    )
}

pub fn emit_trace_range_iter_next_or_null(ctx: &mut TraceCtx, iter: OpRef) -> OpRef {
    // The residual advances the iterator cursor (`W_SeqIterObject.index` /
    // long-range fields), and the long-range leg allocates a fresh bigint, so
    // it both writes the heap and can collect. `emit_trace_call_ref_typed`'s
    // `default_effect_info()` declares an empty write set (cached heap state
    // survives), which would let optheap CSE an iterator/seq getfield across
    // the call. Record `MOST_GENERAL` (RandomEffects + can_invalidate +
    // can_collect) like the other unanalyzed external-writer residuals
    // (`emit_trace_call_void_word_abi`, `emit_trace_build_flat`).
    ctx.call_ref_typed_with_effect(
        jit_range_iter_next_or_null as *const (),
        &[iter],
        &[Type::Ref],
        majit_ir::EffectInfo::MOST_GENERAL,
    )
}

/// Residual FOR_ITER `space.next` for non-range iterators. May invoke a
/// user `__next__` / generator resume, so it can force the virtualizable
/// and can raise (StopIteration / a real exception).
pub fn emit_trace_next(ctx: &mut TraceCtx, iter: OpRef) -> OpRef {
    emit_trace_call_may_force_ref_typed(ctx, jit_next as *const (), &[iter], &[Type::Ref])
}

/// RPython ConstPtr parity: boxed-object constants are Ref-typed.
/// The optimizer can constant-fold immutable field reads from Ref
/// constants (heap.py:640 constant_fold).
pub fn emit_trace_int_constant(ctx: &mut TraceCtx, value: i64) -> OpRef {
    ctx.const_ref(w_int_new(value) as i64)
}

pub fn emit_trace_float_constant(ctx: &mut TraceCtx, value: f64) -> OpRef {
    ctx.const_ref(box_float_constant(value) as i64)
}

pub fn emit_trace_complex_constant(ctx: &mut TraceCtx, re: f64, im: f64) -> OpRef {
    ctx.const_ref(pyre_object::complexobject::w_complex_new(re, im) as i64)
}

pub fn emit_trace_unary_negative_value(ctx: &mut TraceCtx, value: OpRef) -> OpRef {
    emit_trace_call_may_force_ref_typed(
        ctx,
        jit_unary_negative_value as *const (),
        &[value],
        &[Type::Ref],
    )
}

pub fn emit_trace_unary_invert_value(ctx: &mut TraceCtx, value: OpRef) -> OpRef {
    emit_trace_call_may_force_ref_typed(
        ctx,
        jit_unary_invert_value as *const (),
        &[value],
        &[Type::Ref],
    )
}

pub trait TraceHelperAccess {
    fn with_trace_ctx<R>(&mut self, f: impl FnOnce(&mut TraceCtx) -> R) -> R;
    fn trace_frame(&self) -> OpRef;
    fn trace_globals_ptr(&mut self) -> OpRef;
    fn trace_record_not_forced_guard(&mut self);
    /// Per-caller emit hook for `GuardNoException`, matching PyPy
    /// `do_residual_call` (pyjitpl.py:2082) which emits this guard
    /// inline after every can-raise CALL_*.
    fn trace_record_no_exception_guard(&mut self);

    fn trace_make_function(&mut self, code_obj: OpRef) -> Result<OpRef, PyError> {
        let globals = self.trace_globals_ptr();
        let result = self.with_trace_ctx(|ctx| {
            emit_trace_call_ref_typed(
                ctx,
                jit_make_function_from_globals as *const (),
                &[globals, code_obj],
                &[Type::Ref, Type::Ref],
            )
        });
        self.trace_record_no_exception_guard();
        Ok(result)
    }

    fn trace_call_callable(&mut self, callable: OpRef, args: &[OpRef]) -> Result<OpRef, PyError> {
        let frame = self.trace_frame();
        let result =
            self.with_trace_ctx(|ctx| emit_trace_call_callable(ctx, frame, callable, args))?;
        // `call_may_force_ref_typed` (exc=True/may_force): emit
        // GUARD_NOT_FORCED then GUARD_NO_EXCEPTION per `execute_varargs`
        // (pyjitpl.py:1990-2000).
        self.trace_record_not_forced_guard();
        self.trace_record_no_exception_guard();
        Ok(result)
    }

    fn trace_build_list(&mut self, items: &[OpRef]) -> Result<OpRef, PyError> {
        let result = self.with_trace_ctx(|ctx| {
            // STRUCTURAL ADAPTATION: PyPy lowers `newlist(*items)` as
            // trace-visible allocation + item stores, so virtual boxed items
            // remain live through the list construction naturally.  Pyre still
            // routes flat list construction through an opaque helper call;
            // dynasm can otherwise drop a virtual item that is only consumed by
            // that helper (`list_setslice`: `[i, i+1, i+2]` lost item2).  Keep
            // item OpRefs pinned until list construction is ported to the
            // PyPy/RPython `newlist` allocation shape.
            for &item in items {
                ctx.record_op(OpCode::Keepalive, &[item]);
            }
            emit_trace_build_flat(ctx, FlatBuildKind::List, items)
        })?;
        self.trace_record_no_exception_guard();
        Ok(result)
    }

    fn trace_build_tuple(&mut self, items: &[OpRef]) -> Result<OpRef, PyError> {
        let result = self.with_trace_ctx(|ctx| {
            // Same helper-call adaptation as trace_build_list: tuple
            // construction still goes through an opaque helper, so virtual
            // items that are only consumed by that helper must stay live until
            // tuple allocation is ported to trace-visible newtuple stores.
            for &item in items {
                ctx.record_op(OpCode::Keepalive, &[item]);
            }
            emit_trace_build_flat(ctx, FlatBuildKind::Tuple, items)
        })?;
        self.trace_record_no_exception_guard();
        Ok(result)
    }

    fn trace_build_map(&mut self, items: &[OpRef]) -> Result<OpRef, PyError> {
        let result =
            self.with_trace_ctx(|ctx| emit_trace_build_flat(ctx, FlatBuildKind::Map, items))?;
        self.trace_record_no_exception_guard();
        Ok(result)
    }

    fn trace_store_subscr(&mut self, obj: OpRef, key: OpRef, value: OpRef) -> Result<(), PyError> {
        self.with_trace_ctx(|ctx| {
            // STORE_SUBSCR drops `space.setitem`'s result, so the opimpl's
            // residual call has `op.result.concretetype == Void`
            // (`jtransform.py handle_residual_call` keys `result_kind` off
            // it) → void may-force `CALL_N`.  `space.setitem` itself returns the
            // `__setitem__` result (descroperation.py:389); the void shim
            // `jit_setitem` drops it to match the opcode.  Its raise is
            // surfaced by the `guard_no_exception` recorded below, mirroring
            // the rtyper presenting only the success type as `FUNC.RESULT`
            // (`call.py:222`) with out-of-band `OperationError` propagation.
            emit_trace_call_may_force_void_typed(
                ctx,
                jit_setitem as *const (),
                &[obj, key, value],
                &[Type::Ref, Type::Ref, Type::Ref],
            );
        });
        // pyjitpl.py:2079-2082: `space.setitem` may invoke `__setitem__`, which
        // can force the virtualizable, so the residual is a may-force call
        // guarded by GUARD_NOT_FORCED before handle_possible_exception.
        self.trace_record_not_forced_guard();
        self.trace_record_no_exception_guard();
        Ok(())
    }

    fn trace_load_attr(&mut self, obj: OpRef, name: &str) -> Result<OpRef, PyError> {
        let result = self.with_trace_ctx(|ctx| {
            let [name_ptr, name_len] = trace_name_args(ctx, name);
            emit_trace_call_may_force_ref_typed(
                ctx,
                jit_getattr as *const (),
                &[obj, name_ptr, name_len],
                &[Type::Ref, Type::Int, Type::Int],
            )
        });
        // pyjitpl.py:2079-2082: `space.getattr` may invoke `__getattribute__` /
        // `__getattr__`, forcing the virtualizable → may-force + GUARD_NOT_FORCED.
        self.trace_record_not_forced_guard();
        self.trace_record_no_exception_guard();
        Ok(result)
    }

    fn trace_store_attr(&mut self, obj: OpRef, name: &str, value: OpRef) -> Result<(), PyError> {
        self.with_trace_ctx(|ctx| {
            let [name_ptr, name_len] = trace_name_args(ctx, name);
            let _ = emit_trace_call_may_force_int_typed(
                ctx,
                jit_setattr as *const (),
                &[obj, name_ptr, name_len, value],
                &[Type::Ref, Type::Int, Type::Int, Type::Ref],
            );
        });
        // pyjitpl.py:2079-2082: `space.setattr` may invoke `__setattr__`,
        // forcing the virtualizable → may-force + GUARD_NOT_FORCED.
        self.trace_record_not_forced_guard();
        self.trace_record_no_exception_guard();
        Ok(())
    }

    fn trace_list_append(&mut self, list: OpRef, value: OpRef) -> Result<(), PyError> {
        self.with_trace_ctx(|ctx| {
            // Writes the list's strategy storage (may also realloc it);
            // see `emit_trace_call_void_word_abi` — an unanalyzed
            // external writer records MOST_GENERAL.
            emit_trace_call_void_word_abi(
                ctx,
                jit_list_append as *const (),
                &[list, value],
                &[Type::Ref, Type::Ref],
                majit_ir::EffectInfo::MOST_GENERAL,
            );
        });
        self.trace_record_no_exception_guard();
        Ok(())
    }

    fn trace_unpack_sequence(&mut self, seq: OpRef, count: usize) -> Result<Vec<OpRef>, PyError> {
        let items = self.with_trace_ctx(|ctx| emit_trace_unpack_sequence(ctx, seq, count))?;
        // A single guard after the loop satisfies the dedup gate; strict
        // PyPy parity (one guard per call) is a follow-up that requires
        // plumbing the MIFrame guard hook into `emit_trace_unpack_sequence`.
        if count > 0 {
            self.trace_record_no_exception_guard();
        }
        Ok(items)
    }

    fn trace_iter_next_value(&mut self, iter: OpRef) -> Result<OpRef, PyError> {
        // `jit_range_iter_next_or_null` returns value-or-null (null =
        // exhaustion, caught by the trailing for-iter GuardNonnull) and
        // `panic!`s rather than raising on any other iterator kind — which the
        // caller's `guard_class` now makes unreachable. It never sets the
        // thread-local exception, so no trailing GuardNoException (mirrors the
        // inline range leg and `trace_bool_value_from_truth`'s cannot-raise
        // rationale).
        let result = self.with_trace_ctx(|ctx| emit_trace_range_iter_next_or_null(ctx, iter));
        Ok(result)
    }

    fn trace_next(&mut self, iter: OpRef) -> Result<OpRef, PyError> {
        let result = self.with_trace_ctx(|ctx| emit_trace_next(ctx, iter));
        // `space.next` may resume a generator / run a user `__next__`,
        // forcing the virtualizable; StopIteration / a real raise lands on
        // the trailing GuardNoException, side-exiting to the interpreter.
        self.trace_record_not_forced_guard();
        self.trace_record_no_exception_guard();
        Ok(result)
    }

    fn trace_truth_value(&mut self, value: OpRef) -> Result<OpRef, PyError> {
        let result = self.with_trace_ctx(|ctx| emit_trace_truth_value(ctx, value));
        // pyjitpl.py:2079: `__bool__` may force the virtualizable.
        self.trace_record_not_forced_guard();
        self.trace_record_no_exception_guard();
        Ok(result)
    }

    fn trace_bool_value_from_truth(
        &mut self,
        truth: OpRef,
        negate: bool,
    ) -> Result<OpRef, PyError> {
        // `space.newbool` selects a singleton (EF_CANNOT_RAISE), so no
        // trailing GuardNoException — matching the walker / codegen legs that
        // box the bool without one.
        self.with_trace_ctx(|ctx| Ok(emit_trace_bool_value_from_truth(ctx, truth, negate)))
    }

    fn trace_binary_value(
        &mut self,
        a: OpRef,
        b: OpRef,
        op: pyre_interpreter::bytecode::BinaryOperator,
    ) -> Result<OpRef, PyError> {
        // `?` short-circuits on unsupported binary ops before any call is
        // recorded, so the guard only fires when a call was emitted.
        let result = self.with_trace_ctx(|ctx| emit_trace_binary_value(ctx, a, b, op))?;
        // pyjitpl.py:2079: the generic binary op may invoke `__add__` etc.,
        // forcing the virtualizable.
        self.trace_record_not_forced_guard();
        self.trace_record_no_exception_guard();
        Ok(result)
    }

    fn trace_compare_value(
        &mut self,
        a: OpRef,
        b: OpRef,
        op: pyre_interpreter::bytecode::ComparisonOperator,
    ) -> Result<OpRef, PyError> {
        let result = self.with_trace_ctx(|ctx| emit_trace_compare_value(ctx, a, b, op));
        // pyjitpl.py:2079: the generic compare may invoke `__lt__` etc.,
        // forcing the virtualizable.
        self.trace_record_not_forced_guard();
        self.trace_record_no_exception_guard();
        Ok(result)
    }

    fn trace_unary_negative_value(&mut self, value: OpRef) -> Result<OpRef, PyError> {
        let result = self.with_trace_ctx(|ctx| emit_trace_unary_negative_value(ctx, value));
        // pyjitpl.py:2079: `__neg__` may force the virtualizable.
        self.trace_record_not_forced_guard();
        self.trace_record_no_exception_guard();
        Ok(result)
    }

    fn trace_unary_invert_value(&mut self, value: OpRef) -> Result<OpRef, PyError> {
        let result = self.with_trace_ctx(|ctx| emit_trace_unary_invert_value(ctx, value));
        // pyjitpl.py:2079: `__invert__` may force the virtualizable.
        self.trace_record_not_forced_guard();
        self.trace_record_no_exception_guard();
        Ok(result)
    }

    fn trace_int_constant(&mut self, value: i64) -> Result<OpRef, PyError> {
        self.with_trace_ctx(|ctx| Ok(emit_trace_int_constant(ctx, value)))
    }

    fn trace_bigint_constant(&mut self, value: &PyBigInt) -> Result<OpRef, PyError> {
        self.with_trace_ctx(|ctx| Ok(ctx.const_ref(box_bigint_constant(value) as i64)))
    }

    fn trace_float_constant(&mut self, value: f64) -> Result<OpRef, PyError> {
        self.with_trace_ctx(|ctx| Ok(emit_trace_float_constant(ctx, value)))
    }

    fn trace_complex_constant(&mut self, re: f64, im: f64) -> Result<OpRef, PyError> {
        self.with_trace_ctx(|ctx| Ok(emit_trace_complex_constant(ctx, re, im)))
    }

    fn trace_bool_constant(&mut self, value: bool) -> Result<OpRef, PyError> {
        self.with_trace_ctx(|ctx| Ok(ctx.const_ref(w_bool_from(value) as i64)))
    }

    fn trace_str_constant(&mut self, value: &rustpython_wtf8::Wtf8) -> Result<OpRef, PyError> {
        self.with_trace_ctx(|ctx| Ok(ctx.const_ref(box_str_constant(value) as i64)))
    }

    fn trace_bytes_constant(&mut self, value: &[u8]) -> Result<OpRef, PyError> {
        let r = pyre_object::bytesobject::w_bytes_from_bytes(value);
        self.with_trace_ctx(|ctx| Ok(ctx.const_ref(r as i64)))
    }

    fn trace_code_constant(
        &mut self,
        code: &pyre_interpreter::CodeObject,
    ) -> Result<OpRef, PyError> {
        self.with_trace_ctx(|ctx| {
            Ok(ctx.const_ref(pyre_interpreter::box_code_constant(code) as i64))
        })
    }

    fn trace_none_constant(&mut self) -> Result<OpRef, PyError> {
        self.with_trace_ctx(|ctx| Ok(ctx.const_ref(w_none() as i64)))
    }

    fn trace_ellipsis_constant(&mut self) -> Result<OpRef, PyError> {
        self.with_trace_ctx(|ctx| Ok(ctx.const_ref(pyre_object::special::w_ellipsis() as i64)))
    }
}

/// `ll_unboxed_getclass` low-bit test (rtagged.py:155): IntAnd(CastPtrToInt(obj),1).
/// Caller emits the GuardTrue (tagged leg) / GuardFalse (boxed leg) via its
/// path-native guard mechanism. `observed_tagged` stamps the folded bit.
pub(crate) fn emit_tag_lowbit_test(ctx: &mut TraceCtx, obj: OpRef, observed_tagged: bool) -> OpRef {
    let as_int = ctx.record_op(OpCode::CastPtrToInt, &[obj]);
    let one = ctx.const_int(1);
    let lowbit = ctx.record_op(OpCode::IntAnd, &[as_int, one]);
    ctx.set_opref_concrete(lowbit, majit_ir::Value::Int(observed_tagged as i64));
    lowbit
}

/// `ll_unboxed_to_int` (rtagged.py:147): arithmetic IntRshift(CastPtrToInt(obj),1).
pub(crate) fn emit_untag_int(ctx: &mut TraceCtx, obj: OpRef, value: i64) -> OpRef {
    let as_int = ctx.record_op(OpCode::CastPtrToInt, &[obj]);
    let one = ctx.const_int(1);
    let raw = ctx.record_op(OpCode::IntRshift, &[as_int, one]);
    ctx.set_opref_concrete(raw, majit_ir::Value::Int(value));
    raw
}

/// Emit inline W_Int creation (NewWithVtable + SetfieldGc).
///
/// jtransform.py:908-911 rewrite_op_setfield: setfield on typeptr is dropped
/// — `new_with_vtable` writes the typeptr in the backend (llmodel.py:778-782).
pub fn emit_box_int_inline(
    ctx: &mut TraceCtx,
    raw_int: OpRef,
    size_descr: majit_ir::DescrRef,
    intval_descr: majit_ir::DescrRef,
) -> OpRef {
    // jtransform.py:908-911: rewrite_op_setfield skips typeptr setfield
    // entirely ("ignore the operation completely -- instead, it's done by
    // 'new'"). rewrite.py:479-484 handle_malloc_operation emits the vtable
    // setfield via fielddescr_vtable during GC rewrite of NEW_WITH_VTABLE.
    let new_op = ctx.record_op_with_descr(OpCode::NewWithVtable, &[], size_descr);
    ctx.heap_cache_mut().new_object(new_op);
    // Emit: SetfieldGc(v, intval, raw_int)
    let intval_idx = intval_descr.index();
    ctx.record_op_with_descr(OpCode::SetfieldGc, &[new_op, raw_int], intval_descr);
    // `upd.setfield(valuebox)` parity — the cache stores the Box
    // identity (`raw_int` OpRef); cache-hit readers fetch the
    // intrinsic value via `box_value(cached)` at hit time.
    ctx.heapcache_setfield_cached(new_op, intval_idx, raw_int);
    new_op
}

/// Emit inline W_LongObject creation (NewWithVtable + SetfieldGc) for the
/// boxing of a bigint result — the PyPy `W_LongObject(rbigint)` shape
/// (`new_with_vtable` + `setfield_gc('num', z)`). `bigint_ref` is the
/// (Ref-typed) `jit_w_long_*_raw` result; the collecting `NewWithVtable`
/// gcmap-roots it, and the SetfieldGc into the registered `value`
/// gc-pointer field carries the write barrier.
///
/// Like [`emit_box_int_inline`], `w_class` is left zero-filled — the JIT int
/// box does the same; `type(x)`/`isinstance` resolve through `ob_type` (the
/// vtable the NewWithVtable writes).
pub fn emit_box_long_inline(
    ctx: &mut TraceCtx,
    bigint_ref: OpRef,
    size_descr: majit_ir::DescrRef,
    value_descr: majit_ir::DescrRef,
) -> OpRef {
    let new_op = ctx.record_op_with_descr(OpCode::NewWithVtable, &[], size_descr);
    ctx.heap_cache_mut().new_object(new_op);
    let value_idx = value_descr.index();
    ctx.record_op_with_descr(OpCode::SetfieldGc, &[new_op, bigint_ref], value_descr);
    ctx.heapcache_setfield_cached(new_op, value_idx, bigint_ref);
    new_op
}

/// Emit inline `W_BaseException` creation (NewWithVtable + SetfieldGc
/// for `kind` / `w_class` / `args_w`), so a builtin exception built by a
/// Python `raise Type(args)` becomes traced New+SetField ops the
/// optimizer can virtualize when the exception never escapes — instead
/// of the opaque residual `jit_call_callable_N` constructor call.
///
/// Mirrors the runtime construction:
/// `w_exception_new_empty(kind)` (zeroed slots) + `exc_new_wrapper`
/// (`w_class = the called type`) + `descr_init` (`args_w = args list`).
/// `w_cause`/`w_context`/… stay PY_NULL from the NewWithVtable memzero.
pub fn emit_exception_new_inline(
    ctx: &mut TraceCtx,
    kind: pyre_object::interp_exceptions::ExcKind,
    w_class: OpRef,
    args_w: OpRef,
) -> OpRef {
    let (size_descr, kind_descr, w_class_descr, args_w_descr) =
        crate::descr::w_exception_descrs(kind);
    let new_op = ctx.record_op_with_descr(OpCode::NewWithVtable, &[], size_descr);
    ctx.heap_cache_mut().new_object(new_op);
    let kind_const = ctx.const_int(kind as u8 as i64);
    let kind_idx = kind_descr.index();
    ctx.record_op_with_descr(OpCode::SetfieldGc, &[new_op, kind_const], kind_descr);
    ctx.heapcache_setfield_cached(new_op, kind_idx, kind_const);
    let w_class_idx = w_class_descr.index();
    ctx.record_op_with_descr(OpCode::SetfieldGc, &[new_op, w_class], w_class_descr);
    ctx.heapcache_setfield_cached(new_op, w_class_idx, w_class);
    let args_w_idx = args_w_descr.index();
    ctx.record_op_with_descr(OpCode::SetfieldGc, &[new_op, args_w], args_w_descr);
    ctx.heapcache_setfield_cached(new_op, args_w_idx, args_w);
    new_op
}

/// Emit inline Object-strategy `W_ListObject` creation as traced
/// `NewArrayClear` + `SetarrayitemGc` + `NewWithVtable` + `SetfieldGc`
/// ops the optimizer can virtualize when the list never escapes — instead
/// of the opaque residual `jit_build_list` CallR.  Shared by the
/// `BUILD_LIST` decomposition (`try_walker_specialize_newlist`, Object
/// strategy) and the exception `args_w` list a `raise Type(a, b, ...)`
/// constructs.
///
/// Mirrors `listobject.rs::w_list_new` for the Object strategy:
///   - `items` points at an `ItemsBlock` GcArray (capacity == `len`);
///     `pyobject_gcarray_descr` is byte-compatible with the runtime
///     `ItemsBlock` (`base_size = ITEMS_BLOCK_ITEMS_OFFSET = 8`).
///   - `length` = `items.len()`.
///   - `strategy` = `Object` (0). `NewWithVtable` already zero-fills the
///     payload (`int_items` / `float_items` stay empty, never read under
///     the Object strategy); the explicit store keeps the heap cache and
///     optimizer field model in agreement.
///
/// Caller must restrict to Object-strategy-eligible args (non-empty AND
/// not all-int AND not all-float); the typed Integer / Float strategies
/// use `int_items` / `float_items` with `items` null and are NOT emitted
/// here.
pub fn emit_object_list_inline(ctx: &mut TraceCtx, items: &[OpRef]) -> OpRef {
    use crate::descr::{
        list_items_descr, list_length_descr, list_strategy_descr, w_list_size_descr,
    };
    use crate::state::pyobject_gcarray_descr;

    let len = items.len();
    // Step 1 — allocate the ItemsBlock GcArray (capacity == len). Clear
    // so the GcArray walker sees valid refs in every slot.
    let len_ref = ctx.const_int(len as i64);
    let array_descr = pyobject_gcarray_descr();
    let items_block =
        ctx.record_op_with_descr(OpCode::NewArrayClear, &[len_ref], array_descr.clone());
    ctx.heap_cache_mut().new_object(items_block);

    // Step 2 — items_block[i] = items[i].
    for (i, &item) in items.iter().enumerate() {
        let idx = ctx.const_int(i as i64);
        ctx.record_op_with_descr(
            OpCode::SetarrayitemGc,
            &[items_block, idx, item],
            array_descr.clone(),
        );
    }

    // Step 3 — allocate the W_ListObject wrapper.
    let list = ctx.record_op_with_descr(OpCode::NewWithVtable, &[], w_list_size_descr());
    ctx.heap_cache_mut().new_object(list);

    // Step 4 — length / items / strategy SetfieldGc, mirroring the
    // Object-strategy arm of `w_list_new`.
    let length_descr = list_length_descr();
    let length_idx = length_descr.index();
    ctx.record_op_with_descr(OpCode::SetfieldGc, &[list, len_ref], length_descr);
    ctx.heapcache_setfield_cached(list, length_idx, len_ref);

    let items_descr = list_items_descr();
    let items_idx = items_descr.index();
    ctx.record_op_with_descr(OpCode::SetfieldGc, &[list, items_block], items_descr);
    ctx.heapcache_setfield_cached(list, items_idx, items_block);

    let strategy_const = ctx.const_int(pyre_object::listobject::ListStrategy::Object as i64);
    let strategy_descr = list_strategy_descr();
    let strategy_idx = strategy_descr.index();
    ctx.record_op_with_descr(OpCode::SetfieldGc, &[list, strategy_const], strategy_descr);
    ctx.heapcache_setfield_cached(list, strategy_idx, strategy_const);

    list
}

/// Trace-visible canonical `W_TupleObject` construction from boxed items.
/// This is the allocation half of `W_BaseException.descr_getargs`: the raw
/// `args_w` list is copied into a fresh tuple on every public attribute read.
pub fn emit_object_tuple_inline(ctx: &mut TraceCtx, items: &[OpRef]) -> OpRef {
    use crate::state::pyobject_gcarray_descr;

    let len = ctx.const_int(items.len() as i64);
    let array_descr = pyobject_gcarray_descr();
    let items_block = ctx.record_op_with_descr(OpCode::NewArrayClear, &[len], array_descr.clone());
    ctx.heap_cache_mut().new_array(items_block, len, true);
    for (index, &item) in items.iter().enumerate() {
        let index = ctx.const_int(index as i64);
        crate::state::trace_items_block_setitem_value(ctx, items_block, index, item);
    }

    let tuple = ctx.record_op_with_descr(
        OpCode::NewWithVtable,
        &[],
        crate::descr::w_tuple_size_descr(),
    );
    ctx.heap_cache_mut().new_object(tuple);
    let w_class = pyre_object::get_instantiate(&pyre_object::TUPLE_TYPE);
    let w_class = ctx.const_ref(w_class as i64);
    let class_descr = crate::descr::tuple_w_class_descr();
    ctx.record_op_with_descr(OpCode::SetfieldGc, &[tuple, w_class], class_descr.clone());
    ctx.heapcache_setfield_cached(tuple, class_descr.index(), w_class);
    let items_descr = crate::descr::tuple_wrappeditems_descr();
    ctx.record_op_with_descr(
        OpCode::SetfieldGc,
        &[tuple, items_block],
        items_descr.clone(),
    );
    ctx.heapcache_setfield_cached(tuple, items_descr.index(), items_block);
    tuple
}

/// Emit inline Integer / Float-strategy `W_ListObject` creation: a typed
/// length-prefixed backing block (`NewArray` + per-element `SetarrayitemGc`)
/// holding the already-unboxed machine values in `raws`, then the
/// `W_ListObject` wrapper (`NewWithVtable` + `strategy` / `int_items.len` /
/// `int_items.block` — or the `float_items` pair — `SetfieldGc`).  Mirrors the
/// Integer / Float arm of `listobject.rs::w_list_new` (`IntArray::from_vec` /
/// `FloatArray::from_vec`), so OptVirtualize can fold the whole list (wrapper +
/// block) when it never escapes — the orthodox `newlist` shape
/// (`rlist.py:324 ll_newlist`, two mallocs).
///
/// The typed strategy keeps `length = 0` and `items = null`
/// (`w_list_new_with_strategy` non-Object arm) — both stay zero-filled by
/// `NewWithVtable`, matching the runtime.  `items_len_descr` / `items_block_descr`
/// select the `int_items` / `float_items` sub-struct fields and `array_descr`
/// the matching `int_gcarray_descr` / `float_gcarray_descr`.  Caller must have
/// guarded + unboxed each element into `raws` already.
pub fn emit_typed_list_inline(
    ctx: &mut TraceCtx,
    raws: &[OpRef],
    array_descr: majit_ir::DescrRef,
    items_len_descr: majit_ir::DescrRef,
    items_block_descr: majit_ir::DescrRef,
    strategy: pyre_object::listobject::ListStrategy,
) -> OpRef {
    use crate::descr::{list_strategy_descr, w_list_size_descr};

    let len = raws.len();
    let len_ref = ctx.const_int(len as i64);

    // Step 1 — allocate the typed backing block (length-prefixed
    // `[capacity][i64|f64 ...]`, capacity == len).  The elements are machine
    // ints / floats (not refs), so `NewArray` (no GC-safe zeroing) matches
    // `IntArray::from_vec` / `FloatArray::from_vec`; every slot is filled
    // immediately below.
    let block = ctx.record_op_with_descr(OpCode::NewArray, &[len_ref], array_descr.clone());
    ctx.heap_cache_mut().new_array(block, len_ref, true);

    // Step 2 — block[i] = raws[i].
    let block_descr_idx = array_descr.index();
    for (i, &raw) in raws.iter().enumerate() {
        let idx = ctx.const_int(i as i64);
        ctx.record_op_with_descr(
            OpCode::SetarrayitemGc,
            &[block, idx, raw],
            array_descr.clone(),
        );
        ctx.heapcache_setarrayitem(block, idx, block_descr_idx, raw);
    }

    // Step 3 — allocate the W_ListObject wrapper.
    let list = ctx.record_op_with_descr(OpCode::NewWithVtable, &[], w_list_size_descr());
    ctx.heap_cache_mut().new_object(list);

    // Step 4 — strategy / typed-items `len` + `block` SetfieldGc.  `length`
    // and `items` stay zero-filled (0 / null) by NewWithVtable, as
    // `w_list_new_with_strategy` leaves them for the typed strategies.
    let strategy_const = ctx.const_int(strategy as i64);
    let strategy_descr = list_strategy_descr();
    let strategy_idx = strategy_descr.index();
    ctx.record_op_with_descr(OpCode::SetfieldGc, &[list, strategy_const], strategy_descr);
    ctx.heapcache_setfield_cached(list, strategy_idx, strategy_const);

    let items_len_idx = items_len_descr.index();
    ctx.record_op_with_descr(OpCode::SetfieldGc, &[list, len_ref], items_len_descr);
    ctx.heapcache_setfield_cached(list, items_len_idx, len_ref);

    let items_block_idx = items_block_descr.index();
    ctx.record_op_with_descr(OpCode::SetfieldGc, &[list, block], items_block_descr);
    ctx.heapcache_setfield_cached(list, items_block_idx, block);

    list
}

/// Empty->typed in-place promotion of an existing `W_ListObject` wrapper (the
/// comprehension accumulator). Mirrors `switch_to_correct_strategy`'s concrete
/// effect as field mutations on `list_op`: allocate the capacity-1 typed
/// backing block, then set strategy and the matching empty storage fields.
/// Length stays 0; the subsequent append body sub-walk fills slot 0 through
/// the spare-capacity leg.
pub fn emit_promote_empty_list_inline(
    ctx: &mut TraceCtx,
    list_op: OpRef,
    strategy: pyre_object::listobject::ListStrategy,
) {
    use crate::descr::{
        list_float_items_block_descr, list_float_items_len_descr, list_int_items_block_descr,
        list_int_items_len_descr, list_items_descr, list_length_descr, list_strategy_descr,
    };
    use crate::state::{float_gcarray_descr, int_gcarray_descr, pyobject_gcarray_descr};

    let cap_ref = ctx.const_int(1);
    let zero_ref = ctx.const_int(0);

    match strategy {
        pyre_object::listobject::ListStrategy::Integer => {
            let array_descr = int_gcarray_descr();
            let block = ctx.record_op_with_descr(OpCode::NewArray, &[cap_ref], array_descr);
            ctx.heap_cache_mut().new_array(block, cap_ref, true);

            let strategy_const = ctx.const_int(strategy as i64);
            let strategy_descr = list_strategy_descr();
            let strategy_idx = strategy_descr.index();
            ctx.record_op_with_descr(
                OpCode::SetfieldGc,
                &[list_op, strategy_const],
                strategy_descr,
            );
            ctx.heapcache_setfield_cached(list_op, strategy_idx, strategy_const);

            let items_len_descr = list_int_items_len_descr();
            let items_len_idx = items_len_descr.index();
            ctx.record_op_with_descr(OpCode::SetfieldGc, &[list_op, zero_ref], items_len_descr);
            ctx.heapcache_setfield_cached(list_op, items_len_idx, zero_ref);

            let items_block_descr = list_int_items_block_descr();
            let items_block_idx = items_block_descr.index();
            ctx.record_op_with_descr(OpCode::SetfieldGc, &[list_op, block], items_block_descr);
            ctx.heapcache_setfield_cached(list_op, items_block_idx, block);
            // Seed the block's capacity getfield cache with the const (1). The
            // block is a fresh const-size allocation whose capacity is known,
            // matching the heapcache length tracking a `new_array` gets for a
            // const-length array (heapcache.py:508 `new_array` →
            // `arraylen_now_known`). The append body sub-walk reads
            // `ItemsBlock.capacity` via a getfield (not arraylen), so seed that
            // field-index channel explicitly; otherwise the read stays symbolic
            // and the spare-capacity `0 < capacity` branch cannot fold.
            let cap_idx = crate::descr::items_block_capacity_descr().index();
            ctx.heapcache_setfield_cached(block, cap_idx, cap_ref);
        }
        pyre_object::listobject::ListStrategy::Float => {
            let array_descr = float_gcarray_descr();
            let block = ctx.record_op_with_descr(OpCode::NewArray, &[cap_ref], array_descr);
            ctx.heap_cache_mut().new_array(block, cap_ref, true);

            let strategy_const = ctx.const_int(strategy as i64);
            let strategy_descr = list_strategy_descr();
            let strategy_idx = strategy_descr.index();
            ctx.record_op_with_descr(
                OpCode::SetfieldGc,
                &[list_op, strategy_const],
                strategy_descr,
            );
            ctx.heapcache_setfield_cached(list_op, strategy_idx, strategy_const);

            let items_len_descr = list_float_items_len_descr();
            let items_len_idx = items_len_descr.index();
            ctx.record_op_with_descr(OpCode::SetfieldGc, &[list_op, zero_ref], items_len_descr);
            ctx.heapcache_setfield_cached(list_op, items_len_idx, zero_ref);

            let items_block_descr = list_float_items_block_descr();
            let items_block_idx = items_block_descr.index();
            ctx.record_op_with_descr(OpCode::SetfieldGc, &[list_op, block], items_block_descr);
            ctx.heapcache_setfield_cached(list_op, items_block_idx, block);
            // Seed the block's capacity getfield cache with the const (1); see
            // the Integer arm above for the rationale (const-size block, getfield
            // capacity channel distinct from the `new_array` arraylen seed).
            let cap_idx = crate::descr::items_block_capacity_descr().index();
            ctx.heapcache_setfield_cached(block, cap_idx, cap_ref);
        }
        pyre_object::listobject::ListStrategy::Object => {
            let array_descr = pyobject_gcarray_descr();
            let block = ctx.record_op_with_descr(OpCode::NewArrayClear, &[cap_ref], array_descr);
            ctx.heap_cache_mut().new_array(block, cap_ref, true);

            let strategy_const = ctx.const_int(strategy as i64);
            let strategy_descr = list_strategy_descr();
            let strategy_idx = strategy_descr.index();
            ctx.record_op_with_descr(
                OpCode::SetfieldGc,
                &[list_op, strategy_const],
                strategy_descr,
            );
            ctx.heapcache_setfield_cached(list_op, strategy_idx, strategy_const);

            let length_descr = list_length_descr();
            let length_idx = length_descr.index();
            ctx.record_op_with_descr(OpCode::SetfieldGc, &[list_op, zero_ref], length_descr);
            ctx.heapcache_setfield_cached(list_op, length_idx, zero_ref);

            let items_descr = list_items_descr();
            let items_idx = items_descr.index();
            ctx.record_op_with_descr(OpCode::SetfieldGc, &[list_op, block], items_descr);
            ctx.heapcache_setfield_cached(list_op, items_idx, block);
            // Object storage needs no capacity seed: the append body reads
            // capacity through `list.items` (list_items_descr), a path that
            // already resolves to the concrete block.
        }
        pyre_object::listobject::ListStrategy::Empty => {
            debug_assert_ne!(strategy, pyre_object::listobject::ListStrategy::Empty);
        }
    }
}

/// Emit inline `space.newslice(w_start, w_end, w_step)` creation
/// (NewWithVtable + 3 SetfieldGc).
///
/// `pypy/objspace/std/objspace.py:385` `space.newslice` returns
/// `W_SliceObject(w_start, w_end, w_step)` — a fresh allocation per
/// invocation (matching `pypy/interpreter/pyopcode.py:1463 BUILD_SLICE`).
/// `_immutable_fields_ = ['w_start', 'w_stop', 'w_step']`
/// (`sliceobject.py:13`) marks all three slots immutable, so the
/// `optimizeopt/virtualize.py optimize_NEW_WITH_VTABLE` pass can
/// virtualize the allocation when the slice never escapes — the IR
/// shape (NewWithVtable + 3 SetfieldGc) preserves the operand
/// dependencies the optimizer needs to reason about that.
///
/// `jtransform.py:908-911 rewrite_op_setfield` skips the typeptr
/// setfield (the backend writes typeptr inside `new_with_vtable` per
/// `llmodel.py:778-782`); `rewrite.py:479-484 handle_malloc_operation`
/// emits the vtable setfield via `fielddescr_vtable` during the GC
/// rewrite pass.
pub fn emit_box_slice_inline(
    ctx: &mut TraceCtx,
    w_start: OpRef,
    w_stop: OpRef,
    w_step: OpRef,
    size_descr: majit_ir::DescrRef,
    w_start_descr: majit_ir::DescrRef,
    w_stop_descr: majit_ir::DescrRef,
    w_step_descr: majit_ir::DescrRef,
) -> OpRef {
    let new_op = ctx.record_op_with_descr(OpCode::NewWithVtable, &[], size_descr);
    ctx.heap_cache_mut().new_object(new_op);
    let w_start_idx = w_start_descr.index();
    ctx.record_op_with_descr(OpCode::SetfieldGc, &[new_op, w_start], w_start_descr);
    ctx.heapcache_setfield_cached(new_op, w_start_idx, w_start);
    let w_stop_idx = w_stop_descr.index();
    ctx.record_op_with_descr(OpCode::SetfieldGc, &[new_op, w_stop], w_stop_descr);
    ctx.heapcache_setfield_cached(new_op, w_stop_idx, w_stop);
    let w_step_idx = w_step_descr.index();
    ctx.record_op_with_descr(OpCode::SetfieldGc, &[new_op, w_step], w_step_descr);
    ctx.heapcache_setfield_cached(new_op, w_step_idx, w_step);
    new_op
}

/// Emit inline W_Float creation (NewWithVtable + SetfieldGc).
pub fn emit_box_float_inline(
    ctx: &mut TraceCtx,
    raw_float: OpRef,
    size_descr: majit_ir::DescrRef,
    floatval_descr: majit_ir::DescrRef,
) -> OpRef {
    // jtransform.py:908-911 parity: typeptr setfield filtered in trace.
    let new_op = ctx.record_op_with_descr(OpCode::NewWithVtable, &[], size_descr);
    ctx.heap_cache_mut().new_object(new_op);
    let floatval_idx = floatval_descr.index();
    ctx.record_op_with_descr(OpCode::SetfieldGc, &[new_op, raw_float], floatval_descr);
    ctx.heapcache_setfield_cached(new_op, floatval_idx, raw_float);
    new_op
}

/// Emit a fresh callee `PyFrame` directly into the trace IR for the
/// self-recursive single-int-argument fast path.
///
/// Replaces the opaque `jit_create_self_recursive_callee_frame_1_raw_int`
/// CallR that today (`call_jit.rs:2814`) wraps `arena.take()` + reuse
/// check + locals zero-fill + raw_int boxing in an opaque helper. The
/// helper is `#[dont_look_inside]` so the optimizer cannot virtualize
/// the new frame nor fold the boxing — every fib(35) iteration pays
/// the full helper trampoline (~336k calls/run, observed in
/// `phase2_3_self_recursive_call_perf_plan_2026_04_28.md`). PyPy emits
/// `direct_assembler_call` with `NewWithVtable(jitframe) + SetfieldGc(...)`
/// in trace IR (`backend/aarch64/opassembler.py:1080-1200`); this helper
/// is the closest pyre analogue given that pyre's `PyFrame` plays the
/// role both of CPython's `PyFrame` and of PyPy's separate `jitframe` /
/// virtualizable.
///
/// Restrictions held by the caller:
///   - self-recursive (callee `pycode` ≡ caller `pycode`), so the
///     caller passes `pycode` / `w_globals` / `execution_context` in
///     directly: `pycode` and `w_globals` arrive as trace-time
///     constants (the bound `PyCode` and its `function.w_globals`,
///     both immutable for the trace's lifetime), and `execution_context`
///     arrives as the loop's already-materialised `sym.execution_context`
///     OpRef (per-thread; not safe to const-fold across thread entries).
///     This mirrors PyPy aarch64 `direct_assembler_call` (`backend/aarch64/
///     opassembler.py:1080-1200`) which writes the callee jitframe's vable
///     scalars from constants known at trace-compile time.
///   - 1 raw-int argument (no boxed-arg path; caller is responsible
///     for the `trace_guarded_int_payload` unbox).
///   - no cellvars/freevars on the callee — `init_cells` is skipped.
///     The caller verifies this against the concrete `PyCode`
///     before invoking the helper.
///
/// The IR sequence mirrors `pyframe.rs::PyFrame::new_for_call_with_closure`:
///
/// 1. `emit_box_int_inline(raw_int_arg)` → `boxed` W_IntObject (one
///    nursery alloc, optimizer can fold into a virtual when the boxed
///    value never escapes through `GuardNotForced`).
/// 2. `NewArrayClear(array_size)` with `pyobject_gcarray_descr()` —
///    the locals_cells_stack_w `FixedObjectArray<PyObjectRef>`
///    backing storage. Clear so unset slots read as `PY_NULL`.
/// 3. `SetarrayitemGc(locals_array, 0, boxed)` — bind the lone
///    positional argument. Other slots stay `PY_NULL`.
/// 4. `NewWithVtable(pyframe_size_descr())` — `vtable=0` because
///    `PyFrame` is not an `rclass.OBJECT` instance (registered via
///    `TypeInfo::with_gc_ptrs`, see `pyre-jit/src/eval.rs::initialize_gc`),
///    so `handle_new` skips the vtable setfield (rewrite.py:925-933
///    `gen_new_with_vtable` early-out for `vtable == 0`).
/// 5. `SetfieldGc` ops for the constructor-visible fields. The non-zero
///    fields (`execution_context`, `pycode`, `w_globals`,
///    `locals_cells_stack_w`, `valuestackdepth`, `last_instr=-1`) mirror
///    `new_for_call_with_closure`; the nullable GC fields
///    (`f_generator_nowref`, `w_yielding_from`, `f_backref`) are written
///    explicitly to match the same constructor shape instead of relying on
///    an implicit backend zero-fill side effect.
/// Build a VIRTUAL callee `PyFrame` for a multi-frame inline (#68) from
/// already-boxed positional argument refs.  Same field-complete frame shape as
/// [`emit_new_pyframe_inline_self_recursive`] but seeds `locals[0..nparams]`
/// from `param_boxes` (Ref boxes at the Python call boundary) instead of
/// boxing a single raw int.  The frame is the callee MIFrame's `frame` red —
/// `_opimpl_inline_call*` / `perform_call`+`setup_call` create a fresh frame
/// per inlined call (`pyjitpl.py:2445-2476,1862-1874`); the box stays virtual
/// on the hot path (the optimizer folds `NewWithVtable`+`SetfieldGc`) and is
/// materialized lazily only on guard failure.  Field-complete so a forced
/// materialization (`materialize_virtual_from_rd`) never dereferences an unset
/// field.
pub fn emit_new_pyframe_inline_with_params(
    ctx: &mut TraceCtx,
    param_boxes: &[OpRef],
    array_size: usize,
    valuestackdepth: usize,
    pycode: OpRef,
    w_globals: OpRef,
    ec: OpRef,
) -> OpRef {
    use crate::descr::{
        pyframe_code_descr, pyframe_execution_context_descr, pyframe_locals_cells_stack_descr,
        pyframe_next_instr_descr, pyframe_size_descr, pyframe_stack_depth_descr,
        pyframe_w_globals_obj_descr,
    };
    use crate::state::pyobject_gcarray_descr;

    // locals_cells_stack_w array, zero-filled so an unbound local reads PY_NULL.
    let len_ref = ctx.const_int(array_size as i64);
    let array_descr = pyobject_gcarray_descr();
    let locals_array =
        ctx.record_op_with_descr(OpCode::NewArrayClear, &[len_ref], array_descr.clone());
    ctx.heap_cache_mut().new_object(locals_array);

    // locals[i] = param_boxes[i] — the positional arguments (already boxed).
    // Register each store in the heapcache so a later nonstandard
    // `getarrayitem_vable` read of this VIRTUAL inline-callee frame forwards the
    // stored param box (with its concrete shadow) instead of recording a fresh
    // `GetarrayitemGc` whose result has no concrete — the gap that made an
    // in-callee pure comparison branch surface `GotoIfNotValueNotConcrete`.
    // Heapcache key for the locals-array elements must match the descr the
    // codewriter's `getarrayitem_vable`/`setarrayitem_vable` ops carry — the
    // virtualizable info's array descr (`info.array_descrs[0]`), NOT the
    // struct-layout `pyobject_gcarray_descr` the recorded `SetarrayitemGc` uses
    // for materialization.  The heapcache `heap_array_cache` is keyed by descr
    // FIRST, so a struct-vs-vinfo descr mismatch silently misses every forward.
    let heapcache_item_descr_index = ctx
        .virtualizable_info()
        .map(|info| info.array_item_descr(0).index())
        .unwrap_or_else(|| array_descr.index());
    for (i, &p) in param_boxes.iter().enumerate() {
        // A NONE slot (an unbound local in a reconstructed bridge-carrier
        // callee frame) keeps the NewArrayClear zero-fill (PY_NULL); only
        // bound slots are stored.  Forward-inline callees pass dense param
        // boxes, so this skips nothing on that path.
        if p.is_none() {
            continue;
        }
        let idx = ctx.const_int(i as i64);
        ctx.record_op_with_descr(
            OpCode::SetarrayitemGc,
            &[locals_array, idx, p],
            array_descr.clone(),
        );
        ctx.heapcache_setarrayitem(locals_array, idx, heapcache_item_descr_index, p);
    }

    let new_frame = ctx.record_op_with_descr(OpCode::NewWithVtable, &[], pyframe_size_descr());
    ctx.heap_cache_mut().new_object(new_frame);

    let ec_descr = pyframe_execution_context_descr();
    let ec_idx = ec_descr.index();
    ctx.record_op_with_descr(OpCode::SetfieldGc, &[new_frame, ec], ec_descr);
    ctx.heapcache_setfield_cached(new_frame, ec_idx, ec);

    let code_descr = pyframe_code_descr();
    let code_idx = code_descr.index();
    ctx.record_op_with_descr(OpCode::SetfieldGc, &[new_frame, pycode], code_descr);
    ctx.heapcache_setfield_cached(new_frame, code_idx, pycode);

    let globals_obj_descr = pyframe_w_globals_obj_descr();
    let globals_obj_idx = globals_obj_descr.index();
    ctx.record_op_with_descr(
        OpCode::SetfieldGc,
        &[new_frame, w_globals],
        globals_obj_descr,
    );
    ctx.heapcache_setfield_cached(new_frame, globals_obj_idx, w_globals);

    let locals_descr = pyframe_locals_cells_stack_descr();
    let locals_idx = locals_descr.index();
    ctx.record_op_with_descr(OpCode::SetfieldGc, &[new_frame, locals_array], locals_descr);
    ctx.heapcache_setfield_cached(new_frame, locals_idx, locals_array);

    let vsd = ctx.const_int(valuestackdepth as i64);
    let vsd_descr = pyframe_stack_depth_descr();
    let vsd_idx = vsd_descr.index();
    ctx.record_op_with_descr(OpCode::SetfieldGc, &[new_frame, vsd], vsd_descr);
    ctx.heapcache_setfield_cached(new_frame, vsd_idx, vsd);

    let neg_one = ctx.const_int(-1);
    let last_instr_descr = pyframe_next_instr_descr();
    let last_instr_idx = last_instr_descr.index();
    ctx.record_op_with_descr(OpCode::SetfieldGc, &[new_frame, neg_one], last_instr_descr);
    ctx.heapcache_setfield_cached(new_frame, last_instr_idx, neg_one);

    // pyframe.py:76-79 `f_generator_nowref`/`w_yielding_from`/`f_backref`
    // are class-level defaults (None/None/vref_None), never assigned in the
    // frame constructor. The trace of frame construction therefore emits no
    // setfield for them; the freshly allocated payload is already zeroed
    // (zero_gc_pointers_inside, incminimark.py:960), so the fields read back
    // as PY_NULL. No explicit store here.

    new_frame
}

pub fn emit_new_pyframe_inline_self_recursive(
    ctx: &mut TraceCtx,
    arg_box: OpRef,
    array_size: usize,
    valuestackdepth: usize,
    pycode: OpRef,
    w_globals: OpRef,
    ec: OpRef,
) -> OpRef {
    use crate::descr::{
        pyframe_code_descr, pyframe_execution_context_descr, pyframe_locals_cells_stack_descr,
        pyframe_next_instr_descr, pyframe_size_descr, pyframe_stack_depth_descr,
        pyframe_w_globals_obj_descr,
    };
    use crate::state::pyobject_gcarray_descr;

    // Step 1 — `locals[0]` receives the caller's already-boxed positional
    // argument box.  The caller supplies the shape-correct box so the callee
    // reads back the same representation it was traced against: under
    // `CAN_BE_TAGGED` a small `int` stays a tagged immediate (`ll_int_box`),
    // otherwise a heap `W_IntObject` (`w_int_new` fallback).  Re-boxing a raw
    // payload heap-side here would force a heap box even when the value fits
    // the tagged range, and the callee's speculative low-bit guard on the
    // local would then deopt on every recursion.
    let boxed = arg_box;

    // Step 2 — allocate the locals_cells_stack_w array. `NewArrayClear`
    // zeros every slot so any future LOAD_FAST on an unbound local
    // observes `PY_NULL` (UnboundLocalError parity).
    let len_ref = ctx.const_int(array_size as i64);
    let array_descr = pyobject_gcarray_descr();
    let locals_array =
        ctx.record_op_with_descr(OpCode::NewArrayClear, &[len_ref], array_descr.clone());
    ctx.heap_cache_mut().new_object(locals_array);

    // Step 3 — locals[0] = boxed. The single positional argument of
    // the self-recursive call.
    let zero = ctx.const_int(0);
    ctx.record_op_with_descr(
        OpCode::SetarrayitemGc,
        &[locals_array, zero, boxed],
        array_descr,
    );

    // Step 4 — allocate the new PyFrame. NewWithVtable zero-fills the
    // payload; the GC tags it with `PYFRAME_GC_TYPE_ID` because the
    // size descr's parent type id is registered in `pyre-jit/src/eval.rs`.
    let new_frame = ctx.record_op_with_descr(OpCode::NewWithVtable, &[], pyframe_size_descr());
    ctx.heap_cache_mut().new_object(new_frame);

    // Step 5 — SetfieldGc for the constructor-visible fields, mirroring
    // the explicit assignments inside `new_for_call_with_closure`.
    // Order matches the field declaration so the optimizer's lazy-set
    // replace logic groups them together.
    let ec_descr = pyframe_execution_context_descr();
    let ec_idx = ec_descr.index();
    ctx.record_op_with_descr(OpCode::SetfieldGc, &[new_frame, ec], ec_descr);
    ctx.heapcache_setfield_cached(new_frame, ec_idx, ec);

    // `pycode` arrives as a trace-time Ref Const (the bound `PyCode`).
    let code_descr = pyframe_code_descr();
    let code_idx = code_descr.index();
    ctx.record_op_with_descr(OpCode::SetfieldGc, &[new_frame, pycode], code_descr);
    ctx.heapcache_setfield_cached(new_frame, code_idx, pycode);

    // pyframe.py:49 `self.w_globals = w_globals` — store the canonical dict.
    let globals_obj_descr = pyframe_w_globals_obj_descr();
    let globals_obj_idx = globals_obj_descr.index();
    ctx.record_op_with_descr(
        OpCode::SetfieldGc,
        &[new_frame, w_globals],
        globals_obj_descr,
    );
    ctx.heapcache_setfield_cached(new_frame, globals_obj_idx, w_globals);

    // `locals_array` is a fresh `NewArrayClear` op result.  PyPy's
    // executor-while-trace model would have `Box.value` carry the
    // actual allocated array ref; pyre's `record_op` does not execute
    // the alloc, so the runtime ref does not exist until codegen +
    // execution.  Honest carrier: `Value::Void` (= "no Box.value
    // known"), so the downstream cache-hit sanity check skips
    // explicitly.  Fabricating a `Ref(0)` here would lie about the
    // payload.
    let locals_descr = pyframe_locals_cells_stack_descr();
    let locals_idx = locals_descr.index();
    ctx.record_op_with_descr(OpCode::SetfieldGc, &[new_frame, locals_array], locals_descr);
    ctx.heapcache_setfield_cached(new_frame, locals_idx, locals_array);

    let vsd = ctx.const_int(valuestackdepth as i64);
    let vsd_descr = pyframe_stack_depth_descr();
    let vsd_idx = vsd_descr.index();
    ctx.record_op_with_descr(OpCode::SetfieldGc, &[new_frame, vsd], vsd_descr);
    ctx.heapcache_setfield_cached(new_frame, vsd_idx, vsd);

    let neg_one = ctx.const_int(-1);
    let last_instr_descr = pyframe_next_instr_descr();
    let last_instr_idx = last_instr_descr.index();
    ctx.record_op_with_descr(OpCode::SetfieldGc, &[new_frame, neg_one], last_instr_descr);
    ctx.heapcache_setfield_cached(new_frame, last_instr_idx, neg_one);

    // pyframe.py:76-79 `f_generator_nowref`/`w_yielding_from`/`f_backref`
    // are class-level defaults (None/None/vref_None), never assigned in the
    // frame constructor. The trace of frame construction therefore emits no
    // setfield for them; the freshly allocated payload is already zeroed
    // (zero_gc_pointers_inside, incminimark.py:960), so the fields read back
    // as PY_NULL. No explicit store here.

    new_frame
}

// ── Elidable canary helper ──────────────────────────────────────────
//
// rlib/jit.py:13 `@jit.elidable` parity.  PyPy `intobject.py:891-895
// wrapint` in-range check parity: returns true iff `value` falls inside
// the prebuilt-int small cache range AND the cache is enabled.
//
// Deterministic for any `value`, no side effects, no raise →
// `EF_ELIDABLE_CANNOT_RAISE` (`call.py:299`).
//
// Pyre's first production-crate `#[elidable_cannot_raise]` callee.  The
// trace-side effect (`record_result_of_call_pure` patching `CallI` to
// `CallPureI`) is exercised by
// `pyre/pyre-jit-trace/tests/elidable_helper_canary_test.rs`, which
// invokes the helper through `TraceCtx::call_typed_with_effect_pure`
// with an explicit `ElidableCannotRaise` `EffectInfo`.  Production
// `emit_trace_call_int_typed` callsites still pass
// `default_effect_info()` until per-helper EI registration
// lands; this canary closes the macro-side gap so the EI side can
// proceed in a separate slice.
#[majit_macros::elidable_cannot_raise]
pub fn jit_int_in_small_cache_range(value: i64) -> bool {
    w_int_small_cached(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyre_interpreter::PyExecutionContext;
    use pyre_interpreter::{ConstantData, compile_exec};

    /// Verifies that `#[elidable_cannot_raise]` emits the
    /// `INT_ELIDABLE_CANNOT_RAISE = 19` policy byte (`call.py:299`
    /// parity / `call_policy_byte.rs:96`) and produces non-null
    /// trace_target / concrete_target trampolines.  Lives in-crate
    /// because external integration tests cannot reach the macro's
    /// `pub(crate)` `__majit_call_policy_*` symbol from outside.
    #[test]
    fn elidable_helper_macro_emits_int_elidable_cannot_raise_byte() {
        let (policy, _, trace_target, concrete_target, _, _) =
            __majit_call_policy_jit_int_in_small_cache_range();
        assert_eq!(policy, 19u8);
        assert!(!trace_target.is_null());
        assert!(!concrete_target.is_null());
    }

    /// Confirms the helper still runs correctly under the elidable
    /// attribute and preserves the elidable invariant (same input →
    /// same output).  With the `WITHPREBUILTINT=false` default
    /// (`intobject.rs:44`) every input returns false.
    #[test]
    fn elidable_helper_is_deterministic_and_matches_pyre_object() {
        for &value in &[0i64, 7, -3, 42, i64::MIN, i64::MAX] {
            let r1 = jit_int_in_small_cache_range(value);
            let r2 = jit_int_in_small_cache_range(value);
            assert_eq!(r1, r2, "elidable helper must be deterministic");
            assert_eq!(
                r1,
                w_int_small_cached(value),
                "elidable wrapper must mirror pyre-object's underlying check",
            );
        }
    }

    #[test]
    fn test_callable_call_helper_dispatches_builtin_without_trace_side_branching() {
        // `pypy/interpreter/pyopcode.py:921 LOAD_GLOBAL_cached` resolves
        // builtin names through `frame.get_builtin().getdictvalue(name)`.
        // Pyre's `get_builtin()` returns the builtin Module whose
        // `w_dict` is a `W_ModuleDictObject` (per `dictmultiobject.py:60-69
        // allocate_and_init_instance(module=True)`); reach `abs` via
        // that path instead of the legacy raw storage pointer so the
        // test exercises the same dispatch the JIT trace helpers will
        // see once the remaining caller cutover lands.
        let ctx = PyExecutionContext::default();
        let w_builtin = ctx.get_builtin();
        let w_builtin_dict = unsafe { pyre_object::w_module_get_w_dict(w_builtin) };
        let abs = unsafe { pyre_object::w_dict_getitem_str(w_builtin_dict, "abs") }
            .expect("abs builtin must exist");
        let result = jit_call_callable_1(0, abs as i64, w_int_new(-11) as i64);
        unsafe {
            assert_eq!(w_int_get_value(result as PyObjectRef), 11);
        }
    }

    #[test]
    fn test_container_helpers_dispatch_expected_runtime_shapes() {
        let result = jit_build_tuple_2(w_int_new(3) as i64, w_int_new(5) as i64);
        let tuple = result as PyObjectRef;
        unsafe {
            assert!(is_tuple(tuple));
            assert_eq!(w_int_get_value(w_tuple_getitem(tuple, 0).unwrap()), 3);
            assert_eq!(w_int_get_value(w_tuple_getitem(tuple, 1).unwrap()), 5);
        }

        let list = w_list_new(vec![w_int_new(2), w_int_new(4)]);
        let tuple = w_tuple_new(vec![w_int_new(7), w_int_new(9)]);
        unsafe {
            assert_eq!(
                w_int_get_value(jit_sequence_getitem(list as i64, 1) as PyObjectRef),
                4
            );
            assert_eq!(
                w_int_get_value(jit_sequence_getitem(tuple as i64, 0) as PyObjectRef),
                7
            );
        }

        let result = jit_build_map_2(
            w_int_new(1) as i64,
            w_int_new(10) as i64,
            w_int_new(2) as i64,
            w_int_new(20) as i64,
        );
        let dict = result as PyObjectRef;
        unsafe {
            assert!(is_dict(dict));
            assert_eq!(w_int_get_value(w_dict_getitem(dict, 1).unwrap()), 10);
            assert_eq!(w_int_get_value(w_dict_getitem(dict, 2).unwrap()), 20);
        }
    }

    #[test]
    fn test_numeric_helpers_reuse_objspace_semantics() {
        let result = jit_binary_value_from_tag(w_int_new(9) as i64, w_int_new(4) as i64, 1);
        unsafe {
            assert_eq!(w_int_get_value(result as PyObjectRef), 5);
        }

        let result = jit_compare_value_from_tag(w_int_new(2) as i64, w_int_new(7) as i64, 0);
        unsafe {
            assert!(w_bool_get_value(result as PyObjectRef));
        }

        let result = jit_unary_invert_value(w_int_new(5) as i64);
        unsafe {
            assert_eq!(w_int_get_value(result as PyObjectRef), !5);
        }
    }

    #[test]
    fn test_make_function_helper_wraps_code_object() {
        let module = compile_exec("def f(x):\n    return x").expect("compile failed");
        let code = module
            .constants
            .iter()
            .find_map(|constant| match constant {
                ConstantData::Code { code } => Some(code.as_ref().clone()),
                _ => None,
            })
            .expect("expected nested function code");
        let code_ptr = Box::into_raw(Box::new(code)) as *const ();
        let code_obj = pyre_interpreter::w_code_new(code_ptr);
        let func = jit_make_function_from_globals(0, code_obj as i64) as PyObjectRef;

        unsafe {
            assert!(pyre_interpreter::is_function(func));
            // Function.code now stores the PyCode, not the raw CodeObject.
            assert_eq!(
                pyre_interpreter::function_get_code(func),
                code_obj as *const ()
            );
        }
    }

    #[test]
    fn test_range_iter_next_helper_uses_runtime_iterator_step() {
        let iter = w_range_iter_new(0, 2, 1);
        let first = jit_range_iter_next_or_null(iter as i64) as PyObjectRef;
        let second = jit_range_iter_next_or_null(iter as i64) as PyObjectRef;
        let done = jit_range_iter_next_or_null(iter as i64) as PyObjectRef;
        unsafe {
            assert_eq!(w_int_get_value(first), 0);
            assert_eq!(w_int_get_value(second), 1);
            assert!(done.is_null());
        }
    }
}
