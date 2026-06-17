use crate::PyError;
use crate::bytecode::{BinaryOperator, ComparisonOperator};
use pyre_object::{PyObjectRef, w_bool_from};

use crate::{
    CompareOp, add, and_, compare, floordiv, getitem, invert, is_true, lshift, mod_, mul, neg, or_,
    pow, rshift, sub, truediv, xor,
};

/// Maps an in-place `BinaryOperator` to its special-method name
/// (`__iadd__` etc.), or `None` for non-in-place operators.
fn inplace_dunder_name(op: BinaryOperator) -> Option<&'static str> {
    Some(match op {
        BinaryOperator::InplaceAdd => "__iadd__",
        BinaryOperator::InplaceSubtract => "__isub__",
        BinaryOperator::InplaceMultiply => "__imul__",
        BinaryOperator::InplaceFloorDivide => "__ifloordiv__",
        BinaryOperator::InplaceRemainder => "__imod__",
        BinaryOperator::InplaceTrueDivide => "__itruediv__",
        BinaryOperator::InplacePower => "__ipow__",
        BinaryOperator::InplaceLshift => "__ilshift__",
        BinaryOperator::InplaceRshift => "__irshift__",
        BinaryOperator::InplaceAnd => "__iand__",
        BinaryOperator::InplaceOr => "__ior__",
        BinaryOperator::InplaceXor => "__ixor__",
        _ => return None,
    })
}

pub fn binary_value(
    a: PyObjectRef,
    b: PyObjectRef,
    op: BinaryOperator,
) -> Result<PyObjectRef, PyError> {
    let a = crate::baseobjspace::unwrap_cell(a);
    let b = crate::baseobjspace::unwrap_cell(b);
    // descroperation.py:825 `inplace_impl` — consult the in-place
    // special first; fall through to the binary op below when absent or
    // `NotImplemented`.
    if let Some(idunder) = inplace_dunder_name(op) {
        // `seq_bug_compat` applies only to `+=` / `*=`; pass the reflected
        // name so the builtin-sequence rhs-first branch can fire.
        let (rdunder, seq_bug_compat) = match op {
            BinaryOperator::InplaceAdd => (Some("__radd__"), true),
            BinaryOperator::InplaceMultiply => (Some("__rmul__"), true),
            _ => (None, false),
        };
        if let Some(result) = crate::objspace::descroperation::try_inplace_special(
            a,
            b,
            idunder,
            rdunder,
            seq_bug_compat,
        )? {
            return Ok(result);
        }
    }
    match op {
        BinaryOperator::Add | BinaryOperator::InplaceAdd => add(a, b),
        BinaryOperator::Subtract | BinaryOperator::InplaceSubtract => sub(a, b),
        BinaryOperator::Multiply | BinaryOperator::InplaceMultiply => mul(a, b),
        BinaryOperator::FloorDivide | BinaryOperator::InplaceFloorDivide => floordiv(a, b),
        BinaryOperator::Remainder | BinaryOperator::InplaceRemainder => mod_(a, b),
        BinaryOperator::TrueDivide | BinaryOperator::InplaceTrueDivide => truediv(a, b),
        BinaryOperator::Power | BinaryOperator::InplacePower => pow(a, b),
        BinaryOperator::Lshift | BinaryOperator::InplaceLshift => lshift(a, b),
        BinaryOperator::Rshift | BinaryOperator::InplaceRshift => rshift(a, b),
        BinaryOperator::And | BinaryOperator::InplaceAnd => and_(a, b),
        // mappingproxy `__ior__` (read-only) raises TypeError, handled
        // above by `try_inplace_special`; both fall through to `or_`.
        BinaryOperator::Or | BinaryOperator::InplaceOr => or_(a, b),
        BinaryOperator::Xor | BinaryOperator::InplaceXor => xor(a, b),
        BinaryOperator::Subscr => getitem(a, b),
        _ => Err(PyError::type_error(format!(
            "binary operation {op:?} not yet implemented"
        ))),
    }
}

pub fn binary_value_from_tag(
    a: PyObjectRef,
    b: PyObjectRef,
    op_tag: i64,
) -> Result<PyObjectRef, PyError> {
    // In-place tags (13-24) must consult the in-place special (`__iadd__`
    // etc.) first; route them through `binary_value`.  Tags 0-12 use the
    // plain dispatch below.
    if op_tag > 12 {
        let Some(op) = crate::runtime_ops::binary_op_from_tag(op_tag) else {
            return Err(PyError::type_error(format!(
                "unsupported binary op tag: {op_tag}"
            )));
        };
        return binary_value(a, b, op);
    }
    match op_tag {
        0 => add(a, b),
        1 => sub(a, b),
        2 => mul(a, b),
        3 => floordiv(a, b),
        4 => mod_(a, b),
        5 => truediv(a, b),
        6 => getitem(a, b),
        7 => pow(a, b),
        8 => lshift(a, b),
        9 => rshift(a, b),
        10 => and_(a, b),
        11 => or_(a, b),
        12 => xor(a, b),
        _ => Err(PyError::type_error(format!(
            "unsupported binary op tag: {op_tag}"
        ))),
    }
}

pub fn compare_value(
    a: PyObjectRef,
    b: PyObjectRef,
    op: ComparisonOperator,
) -> Result<PyObjectRef, PyError> {
    let a = crate::baseobjspace::unwrap_cell(a);
    let b = crate::baseobjspace::unwrap_cell(b);
    let cmp_op = match op {
        ComparisonOperator::Less => CompareOp::Lt,
        ComparisonOperator::LessOrEqual => CompareOp::Le,
        ComparisonOperator::Greater => CompareOp::Gt,
        ComparisonOperator::GreaterOrEqual => CompareOp::Ge,
        ComparisonOperator::Equal => CompareOp::Eq,
        ComparisonOperator::NotEqual => CompareOp::Ne,
    };
    compare(a, b, cmp_op)
}

pub fn compare_value_from_tag(
    a: PyObjectRef,
    b: PyObjectRef,
    op_tag: i64,
) -> Result<PyObjectRef, PyError> {
    // CONTAINS_OP routes through the compare-residual machinery: tag 6 =
    // `in`, tag 7 = `not in`. `a` is the needle, `b` the container (flatten
    // lowers the args as `[item, container]`).
    if op_tag == 6 || op_tag == 7 {
        let found = crate::baseobjspace::contains(b, a)?;
        let result = if op_tag == 7 { !found } else { found };
        return Ok(w_bool_from(result));
    }
    let op = match op_tag {
        0 => CompareOp::Lt,
        1 => CompareOp::Le,
        2 => CompareOp::Gt,
        3 => CompareOp::Ge,
        4 => CompareOp::Eq,
        5 => CompareOp::Ne,
        _ => {
            return Err(PyError::type_error(format!(
                "unsupported compare op tag: {op_tag}"
            )));
        }
    };
    compare(a, b, op)
}

pub fn unary_negative_value(value: PyObjectRef) -> Result<PyObjectRef, PyError> {
    let value = crate::baseobjspace::unwrap_cell(value);
    neg(value)
}

pub fn unary_invert_value(value: PyObjectRef) -> Result<PyObjectRef, PyError> {
    let value = crate::baseobjspace::unwrap_cell(value);
    invert(value)
}

pub fn truth_value(value: PyObjectRef) -> bool {
    let value = crate::baseobjspace::unwrap_cell(value);
    is_true(value)
}

pub fn bool_value_from_truth(value: bool) -> PyObjectRef {
    w_bool_from(value)
}

/// LIST_EXTEND — extend `list` in place with the items of `iterable`.
/// Shared by the interpreter's `list_extend` handler and the JIT residual
/// `bh_list_extend_fn`.  Mirrors `list.extend`: fast paths for list/tuple
/// sources, generic iterator-protocol fallback otherwise (which surfaces
/// "Value after * must be an iterable, not <T>" when not iterable).
pub fn list_extend_value(list: PyObjectRef, iterable: PyObjectRef) -> Result<(), PyError> {
    unsafe {
        if pyre_object::is_list(iterable) {
            let src_len = pyre_object::w_list_len(iterable);
            for j in 0..src_len {
                if let Some(item) = pyre_object::w_list_getitem(iterable, j as i64) {
                    pyre_object::w_list_append(list, item);
                }
            }
            return Ok(());
        }
        if pyre_object::is_tuple(iterable) {
            let src_len = pyre_object::w_tuple_len(iterable);
            for j in 0..src_len {
                if let Some(item) = pyre_object::w_tuple_getitem(iterable, j as i64) {
                    pyre_object::w_list_append(list, item);
                }
            }
            return Ok(());
        }
        // Generic iter-protocol fallback for dict/set/range/generator/etc.
        let iter = crate::baseobjspace::iter(iterable).map_err(|_| {
            let type_name = (*(*iterable).ob_type).name;
            PyError::type_error(format!(
                "Value after * must be an iterable, not {}",
                type_name
            ))
        })?;
        loop {
            match crate::baseobjspace::next(iter) {
                Ok(item) => {
                    pyre_object::w_list_append(list, item);
                }
                Err(e) if e.kind == crate::PyErrorKind::StopIteration => break,
                Err(e) => return Err(e),
            }
        }
    }
    Ok(())
}

#[majit_macros::jit_may_force]
pub extern "C" fn jit_truth_value(value: i64) -> i64 {
    truth_value(value as PyObjectRef) as i64
}

#[majit_macros::jit_may_force]
pub extern "C" fn jit_bool_value_from_truth(value: i64) -> i64 {
    bool_value_from_truth(value != 0) as i64
}

#[majit_macros::jit_may_force]
pub extern "C" fn jit_binary_value_from_tag(a: i64, b: i64, op_tag: i64) -> i64 {
    match binary_value_from_tag(a as PyObjectRef, b as PyObjectRef, op_tag) {
        Ok(value) => value as i64,
        Err(err) => {
            // llmodel.py:194-199 _store_exception: publish into the backend
            // exception cells so the trailing GuardNoException deopts and
            // re-raises through the blackhole.  Return null — the guard fires
            // before the result is used.
            crate::runtime_ops::jit_publish_exception(err.to_exc_object());
            0
        }
    }
}

#[majit_macros::jit_may_force]
pub extern "C" fn jit_compare_value_from_tag(a: i64, b: i64, op_tag: i64) -> i64 {
    match compare_value_from_tag(a as PyObjectRef, b as PyObjectRef, op_tag) {
        Ok(value) => value as i64,
        Err(err) => {
            // Publish + null so the trailing GuardNoException deopts and
            // re-raises (llmodel.py:194-199 _store_exception).
            crate::runtime_ops::jit_publish_exception(err.to_exc_object());
            0
        }
    }
}

#[majit_macros::jit_may_force]
pub extern "C" fn jit_unary_negative_value(value: i64) -> i64 {
    match unary_negative_value(value as PyObjectRef) {
        Ok(result) => result as i64,
        Err(err) => {
            // Publish + null so the trailing GuardNoException deopts and
            // re-raises (llmodel.py:194-199 _store_exception).
            crate::runtime_ops::jit_publish_exception(err.to_exc_object());
            0
        }
    }
}

#[majit_macros::jit_may_force]
pub extern "C" fn jit_unary_invert_value(value: i64) -> i64 {
    match unary_invert_value(value as PyObjectRef) {
        Ok(result) => result as i64,
        Err(err) => {
            // Publish + null so the trailing GuardNoException deopts and
            // re-raises (llmodel.py:194-199 _store_exception).
            crate::runtime_ops::jit_publish_exception(err.to_exc_object());
            0
        }
    }
}

#[majit_macros::jit_may_force]
pub extern "C" fn jit_getitem(obj: i64, index: i64) -> i64 {
    match getitem(obj as PyObjectRef, index as PyObjectRef) {
        Ok(value) => value as i64,
        Err(err) => {
            // llmodel.py:194-199 _store_exception: publish the exception into
            // the backend pos_exception cells so the GuardNoException recorded
            // after BINARY_SUBSCR (instruction_may_raise) deopts and re-raises
            // through the blackhole resume instead of crashing.  Return null —
            // the guard fires before the result ref is used.
            crate::runtime_ops::jit_publish_exception(err.to_exc_object());
            0
        }
    }
}

#[majit_macros::jit_may_force]
pub extern "C" fn jit_setitem(obj: i64, index: i64, value: i64) {
    match crate::setitem(
        obj as PyObjectRef,
        index as PyObjectRef,
        value as PyObjectRef,
    ) {
        Ok(()) => {}
        Err(err) => {
            // llmodel.py:194-199 _store_exception: publish the exception into
            // the backend pos_exception cells so the GuardNoException recorded
            // after STORE_SUBSCR (instruction_may_raise) deopts and re-raises
            // through the blackhole resume instead of crashing.
            crate::runtime_ops::jit_publish_exception(err.to_exc_object());
        }
    }
}

#[majit_macros::jit_may_force]
pub extern "C" fn jit_getattr(obj: i64, name_ptr: i64, name_len: i64) -> i64 {
    let bytes = unsafe { std::slice::from_raw_parts(name_ptr as *const u8, name_len as usize) };
    let name = std::str::from_utf8(bytes).expect("invalid attr name in JIT");
    match crate::getattr_str(obj as PyObjectRef, name) {
        Ok(value) => value as i64,
        Err(err) => {
            // llmodel.py:194-199 _store_exception: publish the exception into
            // the backend pos_exception cells so the GuardNoException recorded
            // after LOAD_ATTR (instruction_may_raise) deopts and re-raises
            // through the blackhole resume instead of crashing.  Return null —
            // the guard fires before the result ref is used.
            crate::runtime_ops::jit_publish_exception(err.to_exc_object());
            0
        }
    }
}

#[majit_macros::jit_may_force]
pub extern "C" fn jit_setattr(obj: i64, name_ptr: i64, name_len: i64, value: i64) -> i64 {
    let bytes = unsafe { std::slice::from_raw_parts(name_ptr as *const u8, name_len as usize) };
    let name = std::str::from_utf8(bytes).expect("invalid attr name in JIT");
    match crate::setattr_str(obj as PyObjectRef, name, value as PyObjectRef) {
        Ok(_) => 0,
        Err(err) => {
            // llmodel.py:194-199 _store_exception: publish the exception into
            // the backend pos_exception cells so the GuardNoException recorded
            // after STORE_ATTR (instruction_may_raise) deopts and re-raises
            // through the blackhole resume instead of crashing.  Return garbage
            // — the guard fires before the result is used.
            crate::runtime_ops::jit_publish_exception(err.to_exc_object());
            0
        }
    }
}

/// C-ABI bridge for the `execute_store_subscr` arm helper consumed by the
/// production walker.  Mirrors RPython's `bh_call_*` calling convention:
/// a single `*mut PyFrame` arg widened to `i64`, success encoded as a
/// non-zero `i64`, errors propagated via
/// `majit_metainterp::blackhole::BH_LAST_EXC_VALUE`.  Required because
/// `crate::execute_store_subscr` itself returns `Result<StepResult<_>,
/// PyError>` whose fat-enum payload does not fit the residual_call's
/// single-register Ref-result slot.
#[allow(improper_ctypes_definitions)]
pub extern "C" fn bh_execute_store_subscr(executor_ptr: i64) -> i64 {
    let executor = unsafe { &mut *(executor_ptr as *mut crate::pyframe::PyFrame) };
    match crate::pyopcode::execute_store_subscr(executor) {
        Ok(_step_result) => 1,
        Err(err) => {
            let exc_obj = err.to_exc_object();
            majit_metainterp::blackhole::BH_LAST_EXC_VALUE.with(|c| c.set(exc_obj as i64));
            0
        }
    }
}

/// C-ABI 3-arg `(obj, key, value) → i64` store_subscr helper bound by
/// `pyre-jit::cpu.store_subscr_fn` (`pyre-jit/src/jit/cpu.rs:151`).
/// The codewriter emits a `residual_call_r_v(store_subscr_fn, obj,
/// key, value)` (`codewriter.rs:7042
/// build_store_subscr_fn_residual_call_r_v_insn`); the runtime
/// dispatcher calls this thin wrapper to mutate the heap via
/// `baseobjspace::setitem` — `baseobjspace.py` parity for
/// `ObjSpace.setitem(w_obj, w_key, w_value) → space.descr_setitem(...)`.
///
/// Lives in `pyre-interpreter` so `pyre-jit-trace` can reach the address
/// through `pyre_interpreter::jit_trace_fnaddrs()` without adding a
/// `pyre-jit-trace -> pyre-jit` dependency edge.  `pyre-jit-trace`
/// already depends on pyre-interpreter for the normal recording-time
/// helpers (`jit_setitem`, `jit_getitem`, ...).
///
/// `BH_LAST_EXC_VALUE`); the 1/0 polarity parallels
/// `bh_execute_store_subscr` above and matches the executor's
/// raise-vs-success ABI for void residual_calls.  The trait-side void
/// `jit_setitem` instead publishes the raise into the backend
/// pos_exception cells (the recorded GuardNoException re-raises through
/// the blackhole resume) — `bh_store_subscr_fn` is the residual-call
/// leg, which signals the raise to the dispatcher via the 0 return.
#[allow(improper_ctypes_definitions)]
pub extern "C" fn bh_store_subscr_fn(obj: i64, key: i64, value: i64) -> i64 {
    let obj = obj as pyre_object::PyObjectRef;
    let key = key as pyre_object::PyObjectRef;
    let value = value as pyre_object::PyObjectRef;
    if let Err(err) = crate::baseobjspace::setitem(obj, key, value) {
        let exc_obj = err.to_exc_object();
        majit_metainterp::blackhole::BH_LAST_EXC_VALUE.with(|c| c.set(exc_obj as i64));
        return 0;
    }
    1
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyre_object::{w_bool_get_value, w_int_get_value, w_int_new};

    #[test]
    fn test_binary_value_reuses_objspace_dispatch() {
        let result = binary_value(w_int_new(8), w_int_new(3), BinaryOperator::Subtract)
            .expect("binary dispatch should succeed");
        unsafe {
            assert_eq!(w_int_get_value(result), 5);
        }
    }

    #[test]
    fn test_compare_value_reuses_objspace_dispatch() {
        let result = compare_value(w_int_new(2), w_int_new(7), ComparisonOperator::Less)
            .expect("compare dispatch should succeed");
        unsafe {
            assert!(w_bool_get_value(result));
        }
    }

    #[test]
    fn test_truth_and_unary_helpers_share_objspace_semantics() {
        assert!(!truth_value(w_int_new(0)));
        let neg = unary_negative_value(w_int_new(4)).expect("unary negate should succeed");
        let inv = unary_invert_value(w_int_new(5)).expect("unary invert should succeed");
        unsafe {
            assert_eq!(w_int_get_value(neg), -4);
            assert_eq!(w_int_get_value(inv), !5);
        }
    }

    #[test]
    fn test_jit_abi_helpers_share_same_objspace_semantics() {
        assert_eq!(jit_truth_value(w_int_new(0) as i64), 0);
        let neg = jit_unary_negative_value(w_int_new(4) as i64) as PyObjectRef;
        let cmp =
            jit_compare_value_from_tag(w_int_new(2) as i64, w_int_new(7) as i64, 0) as PyObjectRef;
        unsafe {
            assert_eq!(w_int_get_value(neg), -4);
            assert!(w_bool_get_value(cmp));
        }
    }

    #[test]
    fn test_jit_getitem_and_setitem_share_objspace_semantics() {
        let list = pyre_object::w_list_new(vec![w_int_new(2), w_int_new(4)]);
        let item = jit_getitem(list as i64, w_int_new(1) as i64) as PyObjectRef;
        unsafe {
            assert_eq!(w_int_get_value(item), 4);
        }
        jit_setitem(list as i64, w_int_new(0) as i64, w_int_new(9) as i64);
        let updated = jit_getitem(list as i64, w_int_new(0) as i64) as PyObjectRef;
        unsafe {
            assert_eq!(w_int_get_value(updated), 9);
        }
    }
}
