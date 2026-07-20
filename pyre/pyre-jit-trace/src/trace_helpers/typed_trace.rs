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

/// Trace a binary float operation: unbox/cast → op → box.
///
/// RPython jitcode parity: guard_class + getfield_gc_f (or
/// getfield_gc_i + cast_int_to_float for int operands), then
/// float_OP, then new_with_vtable + setfield_gc.
///
/// Returns None if the operation is not handled as a float operation
/// → caller should fall back to residual trace_binary_value.

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
    frame.guard_exact_w_class(
        obj,
        pyre_object::pyobject::get_instantiate(&pyre_object::pyobject::LIST_TYPE),
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

/// Trace len() for known container types.
///
/// RPython jitcode parity: guard_class → getfield(length/len) for each type.
/// Returns None if type not handled → caller falls back to residual call.
/// Trace abs() for int values: guard_class + getfield → guard(!=MIN) → branchless abs.
///
/// RPython jitcode: guard_class + getfield_gc_i(intval) → int_abs sequence.
/// Returns None if not handled → caller falls back to residual call.

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

/// Trace isinstance(obj, cls): guard_class + getfield(w_class) + GUARD_VALUE → const bool.
///
/// RPython parity: isinstance uses space.type(w_obj) internally, which
/// calls jit.promote(w_obj.__class__). Same unified pattern as type():
///   guard_class(ob_type) + getfield_gc_r(obj, w_class) + GUARD_VALUE
/// cls is always promoted via implement_guard_value.
///
/// Returns None if not handled → caller falls back to residual call.

/// Trace min()/max() for two ints: branchless int selection.
///
/// RPython jitcode: guard_class × 2 + getfield_gc_i × 2 + int_lt + branchless select + box.
/// Returns None if not handled → caller falls back to residual call.
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

/// pyjitpl.py:767-776 opimpl_check_neg_index:
///   negbox = INT_LT(indexbox, CONST_FALSE)
///   negbox = implement_guard_value(negbox, orgpc)
///   if negbox.getint():
///       lengthbox = opimpl_arraylen_gc(arraybox, arraydescr)
///       indexbox = INT_ADD(indexbox, lengthbox)
///   return indexbox
///
/// For fixed-size arrays (tuples). Bounds guards added for raw-pointer safety.

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

/// pyjitpl.py:814-827 opimpl_getlistitem_gc_{i,r,f}:
///   arraybox = getfield_gc_r(listbox, itemsdescr)
///   return getarrayitem_gc_{i,r,f}(arraybox, indexbox, arraydescr)
///
/// Combined with guard_class + guard_strategy + opimpl_check_resizable_neg_index
/// as emitted by jtransform do_resizable_list_getitem.
///
/// The runtime exact-`w_class` subclass guard for the LIVE getitem path is
/// enforced by walker-native `try_walker_specialize_subscr`
/// (jitcode_dispatch.rs, guard at the `walker_guard_exact_w_class` call);
/// this typed-trace helper is reached only by the retired executor-trait
/// path (trace_opcode.rs), so it carries no separate exact-w_class guard.
///
/// strategy_id: 0 = object, 1 = int, 2 = float.

/// Dispatch binary subscript (getitem) to type-specialized trace paths.
///
/// jtransform do_fixed_list_getitem / do_resizable_list_getitem parity:
///   tuple → opimpl_check_neg_index + getarrayitem_gc_r_pure
///   list  → guard_class + guard_strategy + opimpl_check_resizable_neg_index
///           + opimpl_getlistitem_gc_{i,r,f}
///
/// Returns None if not a recognized subscript → caller falls back to residual.

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
        // EXACT list only: a list SUBCLASS instance shares `ob_type ==
        // &LIST_TYPE` but retags `w_class` and may override `__setitem__`;
        // exclude it so the store falls to the generic residual (which honours
        // the override) instead of a direct backing-storage write.
        if pyre_object::is_exact_list(concrete_obj) && pyre_object::pyobject::is_int(concrete_key) {
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

// The removed MIFrame range-iterator fast path used to mirror:
// getfield(current) -> getfield(remaining) -> getfield(step) ->
// remaining > 0 guard -> int_add(current, step) ->
// setfield(current, next) -> setfield(remaining, remaining - 1).
