//! ObjSpace — Python object operation dispatch.
//!
//! The ObjSpace mediates all operations on Python objects. This is the layer
//! where type-specific fast paths live, and where the JIT inserts `GuardClass`
//! to specialize operations.

// Suppress unsafe-in-unsafe-fn warnings; our unsafe fns are inherently
// working with raw pointers throughout and wrapping every call in an
// additional unsafe block adds noise without safety benefit.
#![allow(unsafe_op_in_unsafe_fn)]

use malachite_bigint::BigInt;
use num_integer::Integer;
use num_traits::ToPrimitive;

use std::cell::RefCell;
use std::collections::HashMap;

pub use crate::{PyError, PyErrorKind, PyResult};
use pyre_object::strobject::is_str;
use pyre_object::*;

// ── Cell unwrap ──────────────────────────────────────────────────────
// CPython 3.13 unified locals+cells means LoadFast can return cell
// objects. All operations must transparently unwrap cells.
// PyPy: each opcode implementation calls space.unwrap_cell() implicitly.

/// Unwrap a cell object to its contents. Non-cells pass through.
#[inline(always)]
pub fn unwrap_cell(obj: PyObjectRef) -> PyObjectRef {
    if obj.is_null() {
        return obj;
    }
    if unsafe { is_cell(obj) } {
        let inner = unsafe { w_cell_get(obj) };
        if !inner.is_null() {
            return inner;
        }
        // Cell with null content — return cell itself (caller will handle)
        return obj;
    }
    obj
}

// ── BigInt helpers ──────────────────────────────────────────────────

/// Extract a BigInt from an int or long object.

unsafe fn as_bigint(obj: PyObjectRef) -> BigInt {
    if is_int(obj) {
        BigInt::from(w_int_get_value(obj))
    } else {
        w_long_get_value(obj).clone()
    }
}

/// Box a BigInt result, demoting to W_IntObject if it fits in i64.

fn bigint_result(value: BigInt) -> PyObjectRef {
    match value.to_i64() {
        Some(v) => w_int_new(v),
        None => w_long_new(value),
    }
}

// ── Arithmetic operations ─────────────────────────────────────────────

/// Integer addition fast path.
///
/// The JIT will specialize this via:
///   GuardClass(a, &INT_TYPE)
///   GuardClass(b, &INT_TYPE)
///   GetfieldGcI(a, intval_offset) → va
///   GetfieldGcI(b, intval_offset) → vb
///   IntAdd(va, vb) → result
///   New(W_IntObject) + SetfieldGcI(result)

unsafe fn int_add(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let va = w_int_get_value(a);
    let vb = w_int_get_value(b);
    match va.checked_add(vb) {
        Some(r) => Ok(w_int_new(r)),
        None => Ok(w_long_new(BigInt::from(va) + BigInt::from(vb))),
    }
}

unsafe fn int_sub(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let va = w_int_get_value(a);
    let vb = w_int_get_value(b);
    match va.checked_sub(vb) {
        Some(r) => Ok(w_int_new(r)),
        None => Ok(w_long_new(BigInt::from(va) - BigInt::from(vb))),
    }
}

unsafe fn int_mul(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let va = w_int_get_value(a);
    let vb = w_int_get_value(b);
    match va.checked_mul(vb) {
        Some(r) => Ok(w_int_new(r)),
        None => Ok(w_long_new(BigInt::from(va) * BigInt::from(vb))),
    }
}

unsafe fn int_floordiv(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let va = w_int_get_value(a);
    let vb = w_int_get_value(b);
    if vb == 0 {
        return Err(PyError::zero_division("integer division or modulo by zero"));
    }
    // i64::MIN / -1 overflows
    match va.checked_div_euclid(vb) {
        Some(r) => Ok(w_int_new(r)),
        None => Ok(bigint_result(BigInt::from(va).div_floor(&BigInt::from(vb)))),
    }
}

unsafe fn int_mod(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let va = w_int_get_value(a);
    let vb = w_int_get_value(b);
    if vb == 0 {
        return Err(PyError::zero_division("integer division or modulo by zero"));
    }
    Ok(w_int_new(va.rem_euclid(vb)))
}

// ── Long (BigInt) arithmetic operations ─────────────────────────────

unsafe fn long_add(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    Ok(bigint_result(as_bigint(a) + as_bigint(b)))
}

unsafe fn long_sub(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    Ok(bigint_result(as_bigint(a) - as_bigint(b)))
}

unsafe fn long_mul(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    Ok(bigint_result(as_bigint(a) * as_bigint(b)))
}

unsafe fn long_floordiv(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let vb = as_bigint(b);
    if vb == BigInt::from(0) {
        return Err(PyError::zero_division("integer division or modulo by zero"));
    }
    Ok(bigint_result(as_bigint(a).div_floor(&vb)))
}

unsafe fn long_mod(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let vb = as_bigint(b);
    if vb == BigInt::from(0) {
        return Err(PyError::zero_division("integer division or modulo by zero"));
    }
    Ok(bigint_result(as_bigint(a).mod_floor(&vb)))
}

// ── Float arithmetic operations ──────────────────────────────────────

/// Coerce an operand to f64. Works for int, long, and float objects.
unsafe fn as_float(obj: PyObjectRef) -> f64 {
    if is_float(obj) {
        w_float_get_value(obj)
    } else if is_int(obj) {
        w_int_get_value(obj) as f64
    } else {
        // long → f64 (may lose precision for very large values)
        w_long_get_value(obj).to_f64().unwrap_or(f64::INFINITY)
    }
}

/// True if both operands are numeric and at least one is float.

unsafe fn is_float_pair(a: PyObjectRef, b: PyObjectRef) -> bool {
    let a_num = is_int(a) || is_float(a) || is_long(a);
    let b_num = is_int(b) || is_float(b) || is_long(b);
    a_num && b_num && (is_float(a) || is_float(b))
}

unsafe fn float_add(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    Ok(w_float_new(as_float(a) + as_float(b)))
}

unsafe fn float_sub(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    Ok(w_float_new(as_float(a) - as_float(b)))
}

unsafe fn float_mul(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    Ok(w_float_new(as_float(a) * as_float(b)))
}

unsafe fn float_truediv(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let vb = as_float(b);
    if vb == 0.0 {
        return Err(PyError::zero_division("float division by zero"));
    }
    Ok(w_float_new(as_float(a) / vb))
}

unsafe fn float_floordiv(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let vb = as_float(b);
    if vb == 0.0 {
        return Err(PyError::zero_division("float floor division by zero"));
    }
    Ok(w_float_new((as_float(a) / vb).floor()))
}

unsafe fn float_mod(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let vb = as_float(b);
    if vb == 0.0 {
        return Err(PyError::zero_division("float modulo"));
    }
    let va = as_float(a);
    // Python modulo: result has the sign of the divisor
    let r = va % vb;
    let result = if r != 0.0 && ((r > 0.0) != (vb > 0.0)) {
        r + vb
    } else {
        r
    };
    Ok(w_float_new(result))
}

// ── Power ────────────────────────────────────────────────────────────

unsafe fn int_pow(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let va = w_int_get_value(a);
    let vb = w_int_get_value(b);
    if vb < 0 {
        // Negative exponent → float result
        return Ok(w_float_new((va as f64).powf(vb as f64)));
    }
    let vb = vb as u32;
    match va.checked_pow(vb) {
        Some(r) => Ok(w_int_new(r)),
        None => Ok(w_long_new(BigInt::from(va).pow(vb))),
    }
}

unsafe fn long_pow(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let vb = as_bigint(b);
    if vb < BigInt::from(0) {
        let fa = as_float(a);
        let fb = as_float(b);
        return Ok(w_float_new(fa.powf(fb)));
    }
    let exp = vb.to_u32().unwrap_or(u32::MAX);
    Ok(bigint_result(as_bigint(a).pow(exp)))
}

// ── Shift operations ─────────────────────────────────────────────────

unsafe fn int_lshift(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let va = w_int_get_value(a);
    let vb = w_int_get_value(b);
    if vb < 0 {
        return Err(PyError::type_error("negative shift count"));
    }
    let vb = vb as u32;
    if vb < 63 {
        if let Some(r) = va.checked_shl(vb) {
            return Ok(w_int_new(r));
        }
    }
    Ok(w_long_new(BigInt::from(va) << vb))
}

unsafe fn int_rshift(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let va = w_int_get_value(a);
    let vb = w_int_get_value(b);
    if vb < 0 {
        return Err(PyError::type_error("negative shift count"));
    }
    let vb = vb as u32;
    Ok(w_int_new(va >> vb.min(63)))
}

unsafe fn long_lshift(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let vb = as_bigint(b);
    if vb < BigInt::from(0) {
        return Err(PyError::type_error("negative shift count"));
    }
    let shift = vb.to_u32().unwrap_or(u32::MAX);
    Ok(bigint_result(as_bigint(a) << shift))
}

unsafe fn long_rshift(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let vb = as_bigint(b);
    if vb < BigInt::from(0) {
        return Err(PyError::type_error("negative shift count"));
    }
    let shift = vb.to_u32().unwrap_or(u32::MAX);
    Ok(bigint_result(as_bigint(a) >> shift))
}

// ── Bitwise operations ───────────────────────────────────────────────

unsafe fn int_bitand(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    Ok(w_int_new(w_int_get_value(a) & w_int_get_value(b)))
}

unsafe fn int_bitor(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    Ok(w_int_new(w_int_get_value(a) | w_int_get_value(b)))
}

unsafe fn int_bitxor(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    Ok(w_int_new(w_int_get_value(a) ^ w_int_get_value(b)))
}

unsafe fn long_bitand(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    Ok(bigint_result(as_bigint(a) & as_bigint(b)))
}

unsafe fn long_bitor(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    Ok(bigint_result(as_bigint(a) | as_bigint(b)))
}

unsafe fn long_bitxor(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    Ok(bigint_result(as_bigint(a) ^ as_bigint(b)))
}

// ── String operations ────────────────────────────────────────────────

/// Concatenate two str objects.

unsafe fn str_concat(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let sa = w_str_get_value(a);
    let sb = w_str_get_value(b);
    let mut result = String::with_capacity(sa.len() + sb.len());
    result.push_str(sa);
    result.push_str(sb);
    Ok(w_str_new(&result))
}

/// Repeat a str object `n` times.

unsafe fn str_repeat(s: PyObjectRef, n: PyObjectRef) -> PyResult {
    let sv = w_str_get_value(s);
    let nv = w_int_get_value(n);
    let count = if nv < 0 { 0 } else { nv as usize };
    Ok(w_str_new(&sv.repeat(count)))
}

unsafe fn list_concat(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let len_a = w_list_len(a);
    let len_b = w_list_len(b);
    let mut items = Vec::with_capacity(len_a + len_b);
    for i in 0..len_a {
        if let Some(item) = w_list_getitem(a, i as i64) {
            items.push(item);
        }
    }
    for i in 0..len_b {
        if let Some(item) = w_list_getitem(b, i as i64) {
            items.push(item);
        }
    }
    Ok(w_list_new(items))
}

unsafe fn tuple_concat(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let len_a = w_tuple_len(a);
    let len_b = w_tuple_len(b);
    let mut items = Vec::with_capacity(len_a + len_b);
    for i in 0..len_a {
        if let Some(item) = w_tuple_getitem(a, i as i64) {
            items.push(item);
        }
    }
    for i in 0..len_b {
        if let Some(item) = w_tuple_getitem(b, i as i64) {
            items.push(item);
        }
    }
    Ok(w_tuple_new(items))
}

unsafe fn list_repeat(list: PyObjectRef, n: PyObjectRef) -> PyResult {
    let nv = w_int_get_value(n);
    let count = if nv < 0 { 0 } else { nv as usize };
    let len = w_list_len(list);
    let mut items = Vec::with_capacity(len * count);
    for _ in 0..count {
        for i in 0..len {
            if let Some(item) = w_list_getitem(list, i as i64) {
                items.push(item);
            }
        }
    }
    Ok(w_list_new(items))
}

// ── Comparison operations ─────────────────────────────────────────────

unsafe fn int_lt(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    Ok(w_bool_from(w_int_get_value(a) < w_int_get_value(b)))
}

unsafe fn int_le(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    Ok(w_bool_from(w_int_get_value(a) <= w_int_get_value(b)))
}

unsafe fn int_gt(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    Ok(w_bool_from(w_int_get_value(a) > w_int_get_value(b)))
}

unsafe fn int_ge(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    Ok(w_bool_from(w_int_get_value(a) >= w_int_get_value(b)))
}

unsafe fn int_eq(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    Ok(w_bool_from(w_int_get_value(a) == w_int_get_value(b)))
}

unsafe fn int_ne(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    Ok(w_bool_from(w_int_get_value(a) != w_int_get_value(b)))
}

unsafe fn float_lt(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    Ok(w_bool_from(as_float(a) < as_float(b)))
}

unsafe fn float_le(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    Ok(w_bool_from(as_float(a) <= as_float(b)))
}

unsafe fn float_gt(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    Ok(w_bool_from(as_float(a) > as_float(b)))
}

unsafe fn float_ge(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    Ok(w_bool_from(as_float(a) >= as_float(b)))
}

unsafe fn float_eq(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    Ok(w_bool_from(as_float(a) == as_float(b)))
}

unsafe fn float_ne(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    Ok(w_bool_from(as_float(a) != as_float(b)))
}

// ── Public dispatch API ───────────────────────────────────────────────

/// Try to call a dunder method on an instance for binary ops.
///
/// PyPy: descroperation.py `_binop_impl` →
///   1. Try `a.__op__(b)` (forward)
///   2. If not found or returns NotImplemented, try `b.__rop__(a)` (reverse)
unsafe fn try_instance_binop(a: PyObjectRef, b: PyObjectRef, dunder: &str) -> Option<PyResult> {
    let a_is_inst = is_instance(a);
    let b_is_inst = is_instance(b);

    // PyPy: descroperation.py _binop_impl
    // If b's type is a proper subtype of a's type, try reverse first.
    // This matches Python's "subclass reflected op takes priority" rule.
    let try_reverse_first = if a_is_inst && b_is_inst {
        if let Some(rdunder) = reverse_dunder(dunder) {
            let a_type = w_instance_get_type(a);
            let b_type = w_instance_get_type(b);
            !std::ptr::eq(a_type, b_type)
                && is_subtype_cached(b_type, a_type)
                && lookup_in_type_mro(b_type, rdunder).is_some()
        } else {
            false
        }
    } else {
        false
    };

    if try_reverse_first {
        let rdunder = reverse_dunder(dunder).unwrap();
        let w_type = w_instance_get_type(b);
        if let Some(method) = lookup_in_type_mro(w_type, rdunder) {
            let result = crate::space_call_function(method, &[b, a]);
            if !is_not_implemented(result) {
                return Some(Ok(result));
            }
        }
    }

    // Forward: a.__op__(b)
    if a_is_inst {
        let w_type = w_instance_get_type(a);
        if let Some(method) = lookup_in_type_mro(w_type, dunder) {
            let result = crate::space_call_function(method, &[a, b]);
            if !is_not_implemented(result) {
                return Some(Ok(result));
            }
        }
    }

    // Reverse: b.__rop__(a) — only if not already tried above
    if !try_reverse_first && b_is_inst {
        if let Some(rdunder) = reverse_dunder(dunder) {
            let w_type = w_instance_get_type(b);
            if let Some(method) = lookup_in_type_mro(w_type, rdunder) {
                let result = crate::space_call_function(method, &[b, a]);
                if !is_not_implemented(result) {
                    return Some(Ok(result));
                }
            }
        }
    }

    None
}

/// Check if w_type is a subtype of cls using cached MRO.
unsafe fn is_subtype_cached(w_type: PyObjectRef, cls: PyObjectRef) -> bool {
    let mro_ptr = w_type_get_mro(w_type);
    if !mro_ptr.is_null() {
        return (*mro_ptr).iter().any(|&t| std::ptr::eq(t, cls));
    }
    false
}

/// Map forward dunder to reverse dunder.
/// PyPy: descroperation.py `_make_binop_impl` generates both directions.
fn reverse_dunder(dunder: &str) -> Option<&'static str> {
    Some(match dunder {
        // Arithmetic — PyPy: descroperation.py _make_binop_impl
        "__add__" => "__radd__",
        "__sub__" => "__rsub__",
        "__mul__" => "__rmul__",
        "__truediv__" => "__rtruediv__",
        "__floordiv__" => "__rfloordiv__",
        "__mod__" => "__rmod__",
        "__pow__" => "__rpow__",
        "__lshift__" => "__rlshift__",
        "__rshift__" => "__rrshift__",
        "__and__" => "__rand__",
        "__or__" => "__ror__",
        "__xor__" => "__rxor__",
        // Comparison reflected — PyPy: descroperation.py _cmp_dispatch
        "__lt__" => "__gt__",
        "__le__" => "__ge__",
        "__gt__" => "__lt__",
        "__ge__" => "__le__",
        "__eq__" => "__eq__",
        "__ne__" => "__ne__",
        _ => return None,
    })
}

/// Try to call a unary dunder on an instance.
///
/// PyPy: `space.call_function(space.lookup(w_obj, dunder), w_obj)`
unsafe fn try_instance_unaryop(a: PyObjectRef, dunder: &str) -> Option<PyResult> {
    if is_instance(a) {
        let w_type = w_instance_get_type(a);
        if let Some(method) = lookup_in_type_mro(w_type, dunder) {
            return Some(Ok(crate::space_call_function(method, &[a])));
        }
    }
    None
}

/// Binary operation dispatch.
///
/// Checks types and dispatches to the appropriate fast path.
/// The JIT traces through this function, recording `GuardClass` on operand types.

pub fn py_add(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let a = unwrap_cell(a);
    let b = unwrap_cell(b);
    unsafe {
        if is_int(a) && is_int(b) {
            return int_add(a, b);
        }
        if is_int_or_long(a) && is_int_or_long(b) {
            return long_add(a, b);
        }
        if is_float_pair(a, b) {
            return float_add(a, b);
        }
        if is_str(a) && is_str(b) {
            return str_concat(a, b);
        }
        if is_list(a) && is_list(b) {
            return list_concat(a, b);
        }
        if is_tuple(a) && is_tuple(b) {
            return tuple_concat(a, b);
        }
        // Instance dunder dispatch: __add__
        if let Some(result) = try_instance_binop(a, b, "__add__") {
            return result;
        }
        Err(PyError::type_error(format!(
            "unsupported operand type(s) for +: '{}' and '{}'",
            (*(*a).ob_type).tp_name,
            (*(*b).ob_type).tp_name,
        )))
    }
}

pub fn py_sub(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let a = unwrap_cell(a);
    let b = unwrap_cell(b);
    unsafe {
        if is_int(a) && is_int(b) {
            return int_sub(a, b);
        }
        if is_int_or_long(a) && is_int_or_long(b) {
            return long_sub(a, b);
        }
        if is_float_pair(a, b) {
            return float_sub(a, b);
        }
        if let Some(result) = try_instance_binop(a, b, "__sub__") {
            return result;
        }
        Err(PyError::type_error(format!(
            "unsupported operand type(s) for -: '{}' and '{}'",
            (*(*a).ob_type).tp_name,
            (*(*b).ob_type).tp_name,
        )))
    }
}

pub fn py_mul(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let a = unwrap_cell(a);
    let b = unwrap_cell(b);
    unsafe {
        if is_int(a) && is_int(b) {
            return int_mul(a, b);
        }
        if is_int_or_long(a) && is_int_or_long(b) {
            return long_mul(a, b);
        }
        if is_float_pair(a, b) {
            return float_mul(a, b);
        }
        if is_str(a) && is_int(b) {
            return str_repeat(a, b);
        }
        if is_int(a) && is_str(b) {
            return str_repeat(b, a);
        }
        // list * int
        if is_list(a) && is_int(b) {
            return list_repeat(a, b);
        }
        if is_int(a) && is_list(b) {
            return list_repeat(b, a);
        }
        if let Some(result) = try_instance_binop(a, b, "__mul__") {
            return result;
        }
        Err(PyError::type_error(format!(
            "unsupported operand type(s) for *: '{}' and '{}'",
            (*(*a).ob_type).tp_name,
            (*(*b).ob_type).tp_name,
        )))
    }
}

pub fn py_floordiv(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let a = unwrap_cell(a);
    let b = unwrap_cell(b);
    unsafe {
        if is_int(a) && is_int(b) {
            return int_floordiv(a, b);
        }
        if is_int_or_long(a) && is_int_or_long(b) {
            return long_floordiv(a, b);
        }
        if is_float_pair(a, b) {
            return float_floordiv(a, b);
        }
        if let Some(result) = try_instance_binop(a, b, "__floordiv__") {
            return result;
        }
        Err(PyError::type_error(format!(
            "unsupported operand type(s) for //: '{}' and '{}'",
            (*(*a).ob_type).tp_name,
            (*(*b).ob_type).tp_name,
        )))
    }
}

pub fn py_mod(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let a = unwrap_cell(a);
    let b = unwrap_cell(b);
    unsafe {
        if is_int(a) && is_int(b) {
            return int_mod(a, b);
        }
        if is_int_or_long(a) && is_int_or_long(b) {
            return long_mod(a, b);
        }
        if is_float_pair(a, b) {
            return float_mod(a, b);
        }
        // str % args — PyPy: unicodeobject.py mod__String_ANY
        if is_str(a) {
            return str_format_percent(a, b);
        }
        if let Some(result) = try_instance_binop(a, b, "__mod__") {
            return result;
        }
        Err(PyError::type_error(format!(
            "unsupported operand type(s) for %: '{}' and '{}'",
            (*(*a).ob_type).tp_name,
            (*(*b).ob_type).tp_name,
        )))
    }
}

/// `str % args` — printf-style string formatting.
/// PyPy: unicodeobject.py mod__String_ANY → formatting.py
unsafe fn str_format_percent(fmt: PyObjectRef, args: PyObjectRef) -> PyResult {
    let fmt_str = w_str_get_value(fmt);
    // Collect args into a Vec
    let arg_list: Vec<PyObjectRef> = if is_tuple(args) {
        let n = w_tuple_len(args);
        (0..n)
            .filter_map(|i| w_tuple_getitem(args, i as i64))
            .collect()
    } else {
        vec![args]
    };

    let mut result = String::new();
    let mut arg_idx = 0;
    let bytes = fmt_str.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 1 < bytes.len() {
            i += 1;
            // Parse optional width/flags
            let mut width = String::new();
            while i < bytes.len()
                && (bytes[i] == b'-'
                    || bytes[i] == b'+'
                    || bytes[i] == b'0'
                    || bytes[i] == b' '
                    || bytes[i].is_ascii_digit()
                    || bytes[i] == b'.')
            {
                width.push(bytes[i] as char);
                i += 1;
            }
            if i >= bytes.len() {
                break;
            }
            let spec = bytes[i] as char;
            i += 1;
            if spec == '%' {
                result.push('%');
                continue;
            }
            let arg = if arg_idx < arg_list.len() {
                arg_list[arg_idx]
            } else {
                pyre_object::w_none()
            };
            arg_idx += 1;
            match spec {
                's' => {
                    result.push_str(&crate::py_str(arg));
                }
                'r' => {
                    result.push_str(&crate::py_repr(arg));
                }
                'd' | 'i' => {
                    if is_int(arg) {
                        let val = w_int_get_value(arg);
                        if width.is_empty() {
                            result.push_str(&format!("{val}"));
                        } else {
                            let zero_pad = width.starts_with('0');
                            let w: usize = width.trim_start_matches('0').parse().unwrap_or(0);
                            let w = if w == 0 && zero_pad { width.len() } else { w };
                            if zero_pad {
                                result.push_str(&format!("{val:0>w$}"));
                            } else {
                                result.push_str(&format!("{val:>w$}"));
                            }
                        }
                    } else {
                        result.push_str(&crate::py_str(arg));
                    }
                }
                'f' => {
                    let val = if is_float(arg) {
                        pyre_object::floatobject::w_float_get_value(arg)
                    } else if is_int(arg) {
                        w_int_get_value(arg) as f64
                    } else {
                        0.0
                    };
                    if width.contains('.') {
                        let prec: usize = width
                            .split('.')
                            .nth(1)
                            .and_then(|s| s.parse().ok())
                            .unwrap_or(6);
                        result.push_str(&format!("{val:.prec$}"));
                    } else {
                        result.push_str(&format!("{val:.6}"));
                    }
                }
                'x' => {
                    if is_int(arg) {
                        result.push_str(&format!("{:x}", w_int_get_value(arg)));
                    }
                }
                'o' => {
                    if is_int(arg) {
                        result.push_str(&format!("{:o}", w_int_get_value(arg)));
                    }
                }
                _ => {
                    result.push('%');
                    result.push(spec);
                }
            }
        } else {
            result.push(bytes[i] as char);
            i += 1;
        }
    }
    Ok(w_str_new(&result))
}

/// True division (`/` operator) — always produces a float result.

pub fn py_truediv(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let a = unwrap_cell(a);
    let b = unwrap_cell(b);
    unsafe {
        let a_num = is_int(a) || is_float(a) || is_long(a);
        let b_num = is_int(b) || is_float(b) || is_long(b);
        if a_num && b_num {
            return float_truediv(a, b);
        }
        if let Some(result) = try_instance_binop(a, b, "__truediv__") {
            return result;
        }
        Err(PyError::type_error(format!(
            "unsupported operand type(s) for /: '{}' and '{}'",
            (*(*a).ob_type).tp_name,
            (*(*b).ob_type).tp_name,
        )))
    }
}

/// Power operation dispatch (`**` operator).

pub fn py_pow(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let a = unwrap_cell(a);
    let b = unwrap_cell(b);
    unsafe {
        if is_int(a) && is_int(b) {
            return int_pow(a, b);
        }
        if is_int_or_long(a) && is_int_or_long(b) {
            return long_pow(a, b);
        }
        if is_float_pair(a, b) {
            return Ok(w_float_new(as_float(a).powf(as_float(b))));
        }
        if let Some(result) = try_instance_binop(a, b, "__pow__") {
            return result;
        }
        Err(PyError::type_error(format!(
            "unsupported operand type(s) for **: '{}' and '{}'",
            (*(*a).ob_type).tp_name,
            (*(*b).ob_type).tp_name,
        )))
    }
}

/// Left shift dispatch (`<<` operator).

pub fn py_lshift(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let a = unwrap_cell(a);
    let b = unwrap_cell(b);
    unsafe {
        if is_int(a) && is_int(b) {
            return int_lshift(a, b);
        }
        if is_int_or_long(a) && is_int_or_long(b) {
            return long_lshift(a, b);
        }
        if let Some(result) = try_instance_binop(a, b, "__lshift__") {
            return result;
        }
        Err(PyError::type_error(format!(
            "unsupported operand type(s) for <<: '{}' and '{}'",
            (*(*a).ob_type).tp_name,
            (*(*b).ob_type).tp_name,
        )))
    }
}

/// Right shift dispatch (`>>` operator).

pub fn py_rshift(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let a = unwrap_cell(a);
    let b = unwrap_cell(b);
    unsafe {
        if is_int(a) && is_int(b) {
            return int_rshift(a, b);
        }
        if is_int_or_long(a) && is_int_or_long(b) {
            return long_rshift(a, b);
        }
        if let Some(result) = try_instance_binop(a, b, "__rshift__") {
            return result;
        }
        Err(PyError::type_error(format!(
            "unsupported operand type(s) for >>: '{}' and '{}'",
            (*(*a).ob_type).tp_name,
            (*(*b).ob_type).tp_name,
        )))
    }
}

/// Bitwise AND dispatch (`&` operator).

pub fn py_bitand(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let a = unwrap_cell(a);
    let b = unwrap_cell(b);
    unsafe {
        if is_int(a) && is_int(b) {
            return int_bitand(a, b);
        }
        if is_int_or_long(a) && is_int_or_long(b) {
            return long_bitand(a, b);
        }
        if let Some(result) = try_instance_binop(a, b, "__and__") {
            return result;
        }
        Err(PyError::type_error(format!(
            "unsupported operand type(s) for &: '{}' and '{}'",
            (*(*a).ob_type).tp_name,
            (*(*b).ob_type).tp_name,
        )))
    }
}

/// Bitwise OR dispatch (`|` operator).

pub fn py_bitor(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let a = unwrap_cell(a);
    let b = unwrap_cell(b);
    unsafe {
        if is_int(a) && is_int(b) {
            return int_bitor(a, b);
        }
        if is_int_or_long(a) && is_int_or_long(b) {
            return long_bitor(a, b);
        }
        if let Some(result) = try_instance_binop(a, b, "__or__") {
            return result;
        }
        Err(PyError::type_error(format!(
            "unsupported operand type(s) for |: '{}' and '{}'",
            (*(*a).ob_type).tp_name,
            (*(*b).ob_type).tp_name,
        )))
    }
}

/// Bitwise XOR dispatch (`^` operator).

pub fn py_bitxor(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let a = unwrap_cell(a);
    let b = unwrap_cell(b);
    unsafe {
        if is_int(a) && is_int(b) {
            return int_bitxor(a, b);
        }
        if is_int_or_long(a) && is_int_or_long(b) {
            return long_bitxor(a, b);
        }
        if let Some(result) = try_instance_binop(a, b, "__xor__") {
            return result;
        }
        Err(PyError::type_error(format!(
            "unsupported operand type(s) for ^: '{}' and '{}'",
            (*(*a).ob_type).tp_name,
            (*(*b).ob_type).tp_name,
        )))
    }
}

/// Comparison operation dispatch.

pub fn py_compare(a: PyObjectRef, b: PyObjectRef, op: CompareOp) -> PyResult {
    let a = unwrap_cell(a);
    let b = unwrap_cell(b);
    unsafe {
        if is_int(a) && is_int(b) {
            return match op {
                CompareOp::Lt => int_lt(a, b),
                CompareOp::Le => int_le(a, b),
                CompareOp::Gt => int_gt(a, b),
                CompareOp::Ge => int_ge(a, b),
                CompareOp::Eq => int_eq(a, b),
                CompareOp::Ne => int_ne(a, b),
            };
        }
        if is_int_or_long(a) && is_int_or_long(b) {
            let va = as_bigint(a);
            let vb = as_bigint(b);
            return Ok(w_bool_from(match op {
                CompareOp::Lt => va < vb,
                CompareOp::Le => va <= vb,
                CompareOp::Gt => va > vb,
                CompareOp::Ge => va >= vb,
                CompareOp::Eq => va == vb,
                CompareOp::Ne => va != vb,
            }));
        }
        if is_float_pair(a, b) {
            return match op {
                CompareOp::Lt => float_lt(a, b),
                CompareOp::Le => float_le(a, b),
                CompareOp::Gt => float_gt(a, b),
                CompareOp::Ge => float_ge(a, b),
                CompareOp::Eq => float_eq(a, b),
                CompareOp::Ne => float_ne(a, b),
            };
        }
        if is_str(a) && is_str(b) {
            let sa = w_str_get_value(a);
            let sb = w_str_get_value(b);
            return Ok(w_bool_from(match op {
                CompareOp::Lt => sa < sb,
                CompareOp::Le => sa <= sb,
                CompareOp::Gt => sa > sb,
                CompareOp::Ge => sa >= sb,
                CompareOp::Eq => sa == sb,
                CompareOp::Ne => sa != sb,
            }));
        }
        // Instance dunder dispatch for comparison
        let dunder = match op {
            CompareOp::Lt => "__lt__",
            CompareOp::Le => "__le__",
            CompareOp::Gt => "__gt__",
            CompareOp::Ge => "__ge__",
            CompareOp::Eq => "__eq__",
            CompareOp::Ne => "__ne__",
        };
        if let Some(result) = try_instance_binop(a, b, dunder) {
            return result;
        }
        // Identity comparison fallback for == and !=
        if matches!(op, CompareOp::Eq) {
            return Ok(w_bool_from(std::ptr::eq(a, b)));
        }
        if matches!(op, CompareOp::Ne) {
            return Ok(w_bool_from(!std::ptr::eq(a, b)));
        }
        Err(PyError::type_error(format!(
            "'{op:?}' not supported between instances of '{}' and '{}'",
            (*(*a).ob_type).tp_name,
            (*(*b).ob_type).tp_name,
        )))
    }
}

/// Comparison operator enum (mirrors RustPython's ComparisonOperator).
#[derive(Debug, Clone, Copy)]
pub enum CompareOp {
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
}

/// Test if an object is truthy (for branch conditions).
///
/// Python truthiness rules:
/// - None → false
/// - bool → its value
/// - int → nonzero

pub fn py_is_true(obj: PyObjectRef) -> bool {
    let obj = unwrap_cell(obj);
    unsafe {
        if is_bool(obj) {
            return w_bool_get_value(obj);
        }
        if is_int(obj) {
            return w_int_get_value(obj) != 0;
        }
        if is_long(obj) {
            return *w_long_get_value(obj) != BigInt::from(0);
        }
        if is_float(obj) {
            return w_float_get_value(obj) != 0.0;
        }
        if is_str(obj) {
            return w_str_len(obj) != 0;
        }
        if is_list(obj) {
            return w_list_len(obj) > 0;
        }
        if is_tuple(obj) {
            return w_tuple_len(obj) > 0;
        }
        if is_dict(obj) {
            return w_dict_len(obj) > 0;
        }
        if is_none(obj) {
            return false;
        }
        // Instance __bool__ / __len__ — PyPy: descroperation.py is_true
        if is_instance(obj) {
            let w_type = w_instance_get_type(obj);
            // Try __bool__ first
            if let Some(method) = lookup_in_type_mro(w_type, "__bool__") {
                let result = crate::space_call_function(method, &[obj]);
                if !result.is_null() && is_bool(result) {
                    return w_bool_get_value(result);
                }
            }
            // Then __len__ (nonzero length = truthy)
            if let Some(method) = lookup_in_type_mro(w_type, "__len__") {
                let result = crate::space_call_function(method, &[obj]);
                if !result.is_null() && is_int(result) {
                    return w_int_get_value(result) != 0;
                }
            }
        }
        true // default: objects are truthy
    }
}

/// Unary negation.

pub fn py_negative(a: PyObjectRef) -> PyResult {
    let a = unwrap_cell(a);
    unsafe {
        if is_int(a) {
            let v = w_int_get_value(a);
            return match v.checked_neg() {
                Some(r) => Ok(w_int_new(r)),
                None => Ok(w_long_new(-BigInt::from(v))),
            };
        }
        if is_long(a) {
            return Ok(bigint_result(-w_long_get_value(a).clone()));
        }
        if is_float(a) {
            return Ok(w_float_new(-w_float_get_value(a)));
        }
        // Instance __neg__
        if let Some(result) = try_instance_unaryop(a, "__neg__") {
            return result;
        }
        if a.is_null() {
            return Err(PyError::type_error(
                "bad operand type for unary -: 'NoneType'",
            ));
        }
        Err(PyError::type_error(format!(
            "bad operand type for unary -: '{}'",
            (*(*a).ob_type).tp_name,
        )))
    }
}

/// Unary bitwise inversion.

pub fn py_invert(a: PyObjectRef) -> PyResult {
    let a = unwrap_cell(a);
    unsafe {
        if is_int(a) {
            return Ok(w_int_new(!w_int_get_value(a)));
        }
        if is_long(a) {
            return Ok(bigint_result(!w_long_get_value(a).clone()));
        }
        if let Some(result) = try_instance_unaryop(a, "__invert__") {
            return result;
        }
        Err(PyError::type_error(format!(
            "bad operand type for unary ~: '{}'",
            (*(*a).ob_type).tp_name,
        )))
    }
}

// ── Subscript operations ─────────────────────────────────────────────

/// Get item by index: `obj[index]`.
///
/// Dispatches based on the type of `obj`.

pub fn py_getitem(obj: PyObjectRef, index: PyObjectRef) -> PyResult {
    let obj = unwrap_cell(obj);
    let index = unwrap_cell(index);
    unsafe {
        if is_list(obj) {
            if is_slice(index) {
                let len = w_list_len(obj) as i64;
                let start = w_slice_get_start(index);
                let stop = w_slice_get_stop(index);
                let s = if is_none(start) {
                    0
                } else {
                    w_int_get_value(start)
                };
                let e = if is_none(stop) {
                    len
                } else {
                    w_int_get_value(stop)
                };
                let s = if s < 0 { (len + s).max(0) } else { s.min(len) } as usize;
                let e = if e < 0 { (len + e).max(0) } else { e.min(len) } as usize;
                let mut items = Vec::new();
                for i in s..e {
                    if let Some(v) = w_list_getitem(obj, i as i64) {
                        items.push(v);
                    }
                }
                return Ok(w_list_new(items));
            }
            if !is_int(index) {
                return Err(PyError::type_error(
                    "list indices must be integers or slices",
                ));
            }
            let idx = w_int_get_value(index);
            match w_list_getitem(obj, idx) {
                Some(val) => Ok(val),
                None => Err(PyError::new(
                    PyErrorKind::IndexError,
                    "list index out of range",
                )),
            }
        } else if is_tuple(obj) {
            if !is_int(index) {
                return Err(PyError::type_error("tuple indices must be integers"));
            }
            let idx = w_int_get_value(index);
            match w_tuple_getitem(obj, idx) {
                Some(val) => Ok(val),
                None => Err(PyError::new(
                    PyErrorKind::IndexError,
                    "tuple index out of range",
                )),
            }
        } else if is_dict(obj) {
            match w_dict_lookup(obj, index) {
                Some(val) => Ok(val),
                None => Err(PyError::new(PyErrorKind::KeyError, "key not found")),
            }
        } else if is_str(obj) {
            let s = w_str_get_value(obj);
            if is_slice(index) {
                let len = s.len() as i64;
                let start = w_slice_get_start(index);
                let stop = w_slice_get_stop(index);
                let step = w_slice_get_step(index);
                let s_val = if is_none(start) {
                    0
                } else {
                    w_int_get_value(start)
                };
                let e_val = if is_none(stop) {
                    len
                } else {
                    w_int_get_value(stop)
                };
                let step_val = if is_none(step) {
                    1
                } else {
                    w_int_get_value(step)
                };
                let s_idx = if s_val < 0 {
                    (len + s_val).max(0)
                } else {
                    s_val.min(len)
                } as usize;
                let e_idx = if e_val < 0 {
                    (len + e_val).max(0)
                } else {
                    e_val.min(len)
                } as usize;
                if step_val == 1 {
                    let chars: Vec<char> = s.chars().collect();
                    let sliced: String = chars[s_idx..e_idx].iter().collect();
                    Ok(w_str_new(&sliced))
                } else {
                    let chars: Vec<char> = s.chars().collect();
                    let mut result = String::new();
                    let mut i = s_idx as i64;
                    while (step_val > 0 && i < e_idx as i64) || (step_val < 0 && i > e_idx as i64) {
                        if (i as usize) < chars.len() {
                            result.push(chars[i as usize]);
                        }
                        i += step_val;
                    }
                    Ok(w_str_new(&result))
                }
            } else if is_int(index) {
                let idx = w_int_get_value(index);
                let chars: Vec<char> = s.chars().collect();
                let actual_idx = if idx < 0 {
                    chars.len() as i64 + idx
                } else {
                    idx
                } as usize;
                if actual_idx < chars.len() {
                    Ok(w_str_new(&chars[actual_idx].to_string()))
                } else {
                    Err(PyError::new(
                        PyErrorKind::IndexError,
                        "string index out of range",
                    ))
                }
            } else {
                Err(PyError::type_error("string indices must be integers"))
            }
        } else if is_type(obj) {
            // Python 3.9+ generic subscript: type[X] → __class_getitem__(X)
            // PyPy: typeobject.py type.__class_getitem__
            if let Some(method) = lookup_in_type_mro(obj, "__class_getitem__") {
                return Ok(crate::space_call_function(method, &[obj, index]));
            }
            // Default: return the type itself (stub for GenericAlias)
            Ok(obj)
        } else if is_instance(obj) {
            // PyPy: descroperation.py __getitem__
            if let Some(method) = lookup_in_type_mro(w_instance_get_type(obj), "__getitem__") {
                return Ok(crate::space_call_function(method, &[obj, index]));
            }
            Err(PyError::type_error(format!(
                "'{}' object is not subscriptable",
                w_type_get_name(w_instance_get_type(obj)),
            )))
        } else {
            Err(PyError::type_error(format!(
                "'{}' object is not subscriptable",
                (*(*obj).ob_type).tp_name,
            )))
        }
    }
}

/// Set item by index: `obj[index] = value`.

pub fn py_setitem(obj: PyObjectRef, index: PyObjectRef, value: PyObjectRef) -> PyResult {
    let obj = unwrap_cell(obj);
    let index = unwrap_cell(index);
    let value = unwrap_cell(value);
    unsafe {
        if is_list(obj) {
            if !is_int(index) {
                return Err(PyError::type_error("list indices must be integers"));
            }
            let idx = w_int_get_value(index);
            if w_list_setitem(obj, idx, value) {
                Ok(w_none())
            } else {
                Err(PyError::new(
                    PyErrorKind::IndexError,
                    "list assignment index out of range",
                ))
            }
        } else if is_dict(obj) {
            w_dict_store(obj, index, value);
            Ok(w_none())
        } else if is_instance(obj) {
            // PyPy: descroperation.py __setitem__
            if let Some(method) = lookup_in_type_mro(w_instance_get_type(obj), "__setitem__") {
                crate::space_call_function(method, &[obj, index, value]);
                return Ok(w_none());
            }
            Err(PyError::type_error(format!(
                "'{}' object does not support item assignment",
                w_type_get_name(w_instance_get_type(obj)),
            )))
        } else {
            Err(PyError::type_error(format!(
                "'{}' object does not support item assignment",
                (*(*obj).ob_type).tp_name,
            )))
        }
    }
}

/// Get the length of a container: `len(obj)`.
pub fn py_len(obj: PyObjectRef) -> PyResult {
    unsafe {
        if is_list(obj) {
            Ok(w_int_new(w_list_len(obj) as i64))
        } else if is_tuple(obj) {
            Ok(w_int_new(w_tuple_len(obj) as i64))
        } else if is_dict(obj) {
            Ok(w_int_new(w_dict_len(obj) as i64))
        } else if is_str(obj) {
            Ok(w_int_new(w_str_len(obj) as i64))
        } else if is_instance(obj) {
            // Instance __len__ — PyPy: descroperation.py len
            if let Some(method) = lookup_in_type_mro(w_instance_get_type(obj), "__len__") {
                return Ok(crate::space_call_function(method, &[obj]));
            }
            Err(PyError::type_error(format!(
                "object of type '{}' has no len()",
                w_type_get_name(w_instance_get_type(obj)),
            )))
        } else {
            Err(PyError::type_error(format!(
                "object of type '{}' has no len()",
                (*(*obj).ob_type).tp_name,
            )))
        }
    }
}

// ── Attribute operations ──────────────────────────────────────────────

thread_local! {
    /// Side table mapping object addresses to their instance __dict__.
    ///
    /// Every object can have attributes stored here. This avoids modifying
    /// the repr(C) layout of existing object types.
    pub static ATTR_TABLE: RefCell<HashMap<usize, HashMap<String, PyObjectRef>>> =
        RefCell::new(HashMap::new());
}

/// Get an attribute from an object: `obj.name`.
///
/// For module objects, looks up the name in the module's namespace dict
/// (PyPy: Module.getdict → w_dict lookup).
/// For other objects, looks up the attribute in the per-object side table.

pub fn py_getattr(obj: PyObjectRef, name: &str) -> PyResult {
    let obj = unwrap_cell(obj);

    // super proxy — PyPy: superobject.py super_getattro
    // Looks up `name` in cls's MRO starting AFTER super_type.
    unsafe {
        if pyre_object::superobject::is_super(obj) {
            let super_type = pyre_object::superobject::w_super_get_type(obj);
            let bound_obj = pyre_object::superobject::w_super_get_obj(obj);

            // Walk obj's type MRO, skip until we pass super_type
            let obj_type = if is_instance(bound_obj) {
                w_instance_get_type(bound_obj)
            } else if is_type(bound_obj) {
                bound_obj
            } else {
                return Err(PyError::type_error("super: bad obj type"));
            };
            let mro_ptr = w_type_get_mro(obj_type);
            if !mro_ptr.is_null() {
                let mro = &*mro_ptr;
                let mut past_super = false;
                for &t in mro {
                    if std::ptr::eq(t, super_type) {
                        past_super = true;
                        continue;
                    }
                    if !past_super {
                        continue;
                    }
                    if is_type(t) {
                        if let Some(method) = lookup_in_type_mro(t, name) {
                            return Ok(method);
                        }
                    }
                }
            }
            return Err(PyError::new(
                PyErrorKind::AttributeError,
                format!("'super' object has no attribute '{name}'"),
            ));
        }
    }

    // Generator/coroutine methods — PyPy: generator.py GeneratorIterator
    unsafe {
        if pyre_object::generatorobject::is_generator(obj) {
            match name {
                "close" | "send" | "throw" | "__next__" | "__iter__" => {
                    return Ok(crate::w_builtin_func_new("gen_method", gen_stub_method));
                }
                _ => {}
            }
        }
    }

    // Module objects: look up in module namespace
    // PyPy: space.getattr(w_module, w_name) → Module.getdictvalue(space, name)
    unsafe {
        if is_module(obj) {
            let ns_ptr = w_module_get_dict_ptr(obj) as *mut crate::PyNamespace;
            if !ns_ptr.is_null() {
                if let Some(&value) = (*ns_ptr).get(name) {
                    if !value.is_null() {
                        return Ok(value);
                    }
                }
            }
        }
    }

    // Instance objects — PyPy: descroperation.py descr__getattribute__
    //
    // Full descriptor protocol (PEP 252):
    //   1. Look up name in type MRO → w_descr
    //   2. If w_descr is a data descriptor (__get__ + __set__/__delete__):
    //      → call w_descr.__get__(obj, type)
    //   3. Check instance dict
    //   4. If w_descr is a non-data descriptor (__get__ only):
    //      → call w_descr.__get__(obj, type)
    //   5. Return w_descr as-is
    unsafe {
        if is_instance(obj) {
            let w_type = w_instance_get_type(obj);

            // Step 1: look up in type MRO
            let w_descr = lookup_in_type_mro(w_type, name);

            // Step 2: data descriptor takes priority over instance dict
            if let Some(descr) = w_descr {
                if is_data_descriptor(descr) {
                    if let Some(result) = call_descriptor_get(descr, obj, w_type) {
                        return Ok(result);
                    }
                }
            }

            // Step 3: instance dict (ATTR_TABLE)
            let found = ATTR_TABLE.with(|table| {
                let table = table.borrow();
                table
                    .get(&(obj as usize))
                    .and_then(|d| d.get(name).copied())
            });
            if let Some(value) = found {
                return Ok(value);
            }

            // Step 4: non-data descriptor
            if let Some(descr) = w_descr {
                if let Some(result) = call_descriptor_get(descr, obj, w_type) {
                    return Ok(result);
                }
                // Step 5: return descriptor as-is
                return Ok(descr);
            }

            // Special attributes — PyPy: descroperation.py
            if name == "__class__" {
                return Ok(w_type);
            }

            return Err(PyError::new(
                PyErrorKind::AttributeError,
                format!(
                    "'{}' object has no attribute '{name}'",
                    w_type_get_name(w_type),
                ),
            ));
        }
    }

    // Type objects: look up in type's own dict → base dicts
    // PyPy: typeobject.py lookup_where → MRO search + descriptor unwrap
    unsafe {
        if is_type(obj) {
            // Special type attributes — PyPy: typeobject.py
            if name == "__name__" {
                return Ok(w_str_new(w_type_get_name(obj)));
            }
            if name == "__qualname__" {
                // Check if __qualname__ was explicitly set in class body
                if let Some(qn) = lookup_in_type_mro(obj, "__qualname__") {
                    return Ok(qn);
                }
                return Ok(w_str_new(w_type_get_name(obj)));
            }
            if name == "__mro__" {
                let mro_ptr = w_type_get_mro(obj);
                if !mro_ptr.is_null() {
                    return Ok(w_tuple_new((*mro_ptr).clone()));
                }
            }
            if name == "__dict__" {
                // Return the type's namespace as a dict
                let dict_ptr = w_type_get_dict_ptr(obj) as *const crate::PyNamespace;
                let d = pyre_object::w_dict_new();
                if !dict_ptr.is_null() {
                    for (k, &v) in (*dict_ptr).entries() {
                        if !v.is_null() {
                            pyre_object::w_dict_store(d, w_str_new(k), v);
                        }
                    }
                }
                return Ok(d);
            }
            if name == "__bases__" {
                return Ok(w_type_get_bases(obj));
            }
            if name == "__doc__"
                || name == "__module__"
                || name == "__abstractmethods__"
                || name == "__flags__"
            {
                // Check class dict first, then return None
                if let Some(v) = lookup_in_type_mro(obj, name) {
                    return Ok(v);
                }
                return Ok(w_none());
            }

            if let Some(value) = lookup_in_type_mro(obj, name) {
                // Unwrap staticmethod/classmethod/property descriptors
                // PyPy: type.__getattribute__ calls space.get on descriptors
                if is_staticmethod(value) {
                    return Ok(w_staticmethod_get_func(value));
                }
                if is_classmethod(value) {
                    return Ok(w_classmethod_get_func(value));
                }
                if is_property(value) {
                    // property accessed on class → return property itself
                    return Ok(value);
                }
                return Ok(value);
            }
            // Metaclass attribute lookup — PyPy: type.__getattribute__
            // Check if this type was created by a custom metaclass.
            // The metaclass is stored as __metaclass__ in ATTR_TABLE.
            let metaclass_obj = ATTR_TABLE.with(|table| {
                let table = table.borrow();
                table
                    .get(&(obj as usize))
                    .and_then(|d| d.get("__metaclass__").copied())
            });
            // PyPy: type.__getattribute__ → metatype descriptor protocol.
            // Search metaclass MRO. Binding is handled by load_method.
            let metatypes: [Option<PyObjectRef>; 2] = [
                metaclass_obj,
                crate::typedef::get_type_object((*obj).ob_type),
            ];
            for mt in metatypes.iter().flatten() {
                let mc = *mt;
                if is_type(mc) {
                    if let Some(value) = lookup_in_type_mro(mc, name) {
                        return Ok(value);
                    }
                }
            }
            return Err(PyError::new(
                PyErrorKind::AttributeError,
                format!(
                    "type object '{}' has no attribute '{name}'",
                    w_type_get_name(obj),
                ),
            ));
        }
    }

    // Builtin type method lookup via TypeDef registry.
    //
    // PyPy: space.type(w_obj) → W_TypeObject → MRO lookup in type dict.
    // Each builtin type (list, str, dict, etc.) has a W_TypeObject with
    // methods pre-installed, matching PyPy's TypeDef interpleveldefs.
    if let Some(type_obj) = crate::typedef::type_of(obj) {
        if let Some(method) = unsafe { lookup_in_type_mro(type_obj, name) } {
            return Ok(method);
        }
    }

    // Function object attributes — PyPy: funcobject.py W_Function
    // Check ATTR_TABLE first (for dynamically set attrs like __name__, __doc__)
    if unsafe { crate::is_func(obj) || crate::is_builtin_func(obj) } {
        let found = ATTR_TABLE.with(|table| {
            table
                .borrow()
                .get(&(obj as usize))
                .and_then(|d| d.get(name).copied())
        });
        if let Some(v) = found {
            return Ok(v);
        }
    }
    unsafe {
        if crate::is_func(obj) {
            match name {
                "__code__" => {
                    // Return the code pointer as an opaque object (stub)
                    let code_ptr = crate::w_func_get_code_ptr(obj);
                    let code = &*(code_ptr as *const pyre_bytecode::CodeObject);
                    let co = pyre_object::w_instance_new(crate::typedef::get_object_type());
                    let _ = py_setattr(
                        co,
                        "co_varnames",
                        w_tuple_new(
                            code.varnames
                                .iter()
                                .map(|s| w_str_new(s) as PyObjectRef)
                                .collect(),
                        ),
                    );
                    let _ = py_setattr(co, "co_argcount", w_int_new(code.arg_count as i64));
                    let _ = py_setattr(
                        co,
                        "co_kwonlyargcount",
                        w_int_new(code.kwonlyarg_count as i64),
                    );
                    let _ = py_setattr(co, "co_name", w_str_new(&*code.obj_name));
                    let _ = py_setattr(co, "co_filename", w_str_new(&*code.source_path));
                    let _ = py_setattr(co, "co_flags", w_int_new(code.flags.bits() as i64));
                    return Ok(co);
                }
                "__name__" => {
                    return Ok(w_str_new(crate::w_func_get_name(obj)));
                }
                "__qualname__" | "__defaults__" | "__kwdefaults__" | "__globals__"
                | "__closure__" => {
                    let found = ATTR_TABLE.with(|table| {
                        table
                            .borrow()
                            .get(&(obj as usize))
                            .and_then(|d| d.get(name).copied())
                    });
                    return Ok(found.unwrap_or(w_none()));
                }
                _ => {}
            }
        }
    }

    // Common special attributes — return defaults for any object type
    if name == "__doc__"
        || name == "__module__"
        || name == "__wrapped__"
        || name == "__annotations__"
    {
        // Check ATTR_TABLE first, then return None as default
        let found = ATTR_TABLE.with(|table| {
            let table = table.borrow();
            table
                .get(&(obj as usize))
                .and_then(|d| d.get(name).copied())
        });
        return Ok(found.unwrap_or(w_none()));
    }
    if name == "__dict__" {
        let d = pyre_object::w_dict_new();
        ATTR_TABLE.with(|table| {
            let table = table.borrow();
            if let Some(attrs) = table.get(&(obj as usize)) {
                for (k, &v) in attrs {
                    unsafe { pyre_object::w_dict_store(d, pyre_object::w_str_new(k), v) };
                }
            }
        });
        return Ok(d);
    }
    if name == "__class__" {
        if let Some(tp) = crate::typedef::type_of(obj) {
            return Ok(tp);
        }
    }

    // All other objects: use side table
    ATTR_TABLE.with(|table| {
        let table = table.borrow();
        let key = obj as usize;
        if let Some(dict) = table.get(&key) {
            if let Some(&value) = dict.get(name) {
                return Ok(value);
            }
        }
        unsafe {
            Err(PyError::new(
                PyErrorKind::AttributeError,
                format!(
                    "'{}' object has no attribute '{name}'",
                    (*(*obj).ob_type).tp_name,
                ),
            ))
        }
    })
}

// Builtin type method implementations moved to type_methods.rs
// (PyPy: listobject.py, unicodeobject.py, dictobject.py, tupleobject.py)

/// Look up a name by walking the C3 MRO.
///
/// Public wrapper for external callers (eval.rs load_method).
pub unsafe fn lookup_in_type_mro_pub(w_type: PyObjectRef, name: &str) -> Option<PyObjectRef> {
    lookup_in_type_mro(w_type, name)
}

/// PyPy equivalent: typeobject.py `_lookup_where(self, key)` →
/// linear search through `self.mro_w`.
unsafe fn lookup_in_type_mro(w_type: PyObjectRef, name: &str) -> Option<PyObjectRef> {
    if w_type.is_null() || !is_type(w_type) {
        return None;
    }
    // Use cached MRO if available (PyPy: W_TypeObject.mro_w)
    let cached = w_type_get_mro(w_type);
    let mro_owned;
    let mro: &[PyObjectRef] = if !cached.is_null() {
        &*cached
    } else {
        mro_owned = compute_mro(w_type);
        &mro_owned
    };
    for cls in mro {
        let ns_ptr = w_type_get_dict_ptr(*cls) as *mut crate::PyNamespace;
        if !ns_ptr.is_null() {
            let ns = &*ns_ptr;
            if let Some(&value) = ns.get(name) {
                if !value.is_null() {
                    return Some(value);
                }
            }
        }
    }
    None
}

/// C3 linearization — PyPy: typeobject.py `compute_default_mro`.
///
/// Computes the Method Resolution Order for a type following the C3
/// algorithm (Python 2.3+). Handles diamond inheritance correctly.
///
/// Public wrapper for use by isinstance and other external callers.
pub unsafe fn compute_mro_pub(w_type: PyObjectRef) -> Vec<PyObjectRef> {
    compute_mro(w_type)
}

unsafe fn compute_mro(w_type: PyObjectRef) -> Vec<PyObjectRef> {
    let mut result = vec![w_type];
    let bases = w_type_get_bases(w_type);
    if bases.is_null() || !is_tuple(bases) {
        return result;
    }
    let n = w_tuple_len(bases);
    if n == 0 {
        return result;
    }

    // Build candidate lists: [base.mro() for base in bases] + [list(bases)]
    let mut lists: Vec<Vec<PyObjectRef>> = Vec::with_capacity(n + 1);
    for i in 0..n {
        if let Some(base) = w_tuple_getitem(bases, i as i64) {
            if is_type(base) {
                lists.push(compute_mro(base));
            }
        }
    }
    let mut bases_list = Vec::with_capacity(n);
    for i in 0..n {
        if let Some(base) = w_tuple_getitem(bases, i as i64) {
            bases_list.push(base);
        }
    }
    lists.push(bases_list);

    // C3 merge
    loop {
        // Remove empty lists
        lists.retain(|l| !l.is_empty());
        if lists.is_empty() {
            break;
        }
        // Find a candidate: head of some list that doesn't appear in
        // the tail of any other list.
        let mut found = None;
        for list in &lists {
            let candidate = list[0];
            let in_tail = lists.iter().any(|other| {
                other.len() > 1 && other[1..].iter().any(|&x| std::ptr::eq(x, candidate))
            });
            if !in_tail {
                found = Some(candidate);
                break;
            }
        }
        let Some(next) = found else {
            // C3 inconsistency — fall back to first available
            break;
        };
        result.push(next);
        // Remove next from the head of all lists
        for list in &mut lists {
            if !list.is_empty() && std::ptr::eq(list[0], next) {
                list.remove(0);
            }
        }
    }
    result
}

// ── Descriptor protocol ──────────────────────────────────────────────
// PyPy equivalent: descroperation.py is_data_descr / space.get

/// Check if a descriptor is a data descriptor (has __set__ or __delete__).
///
/// PyPy: descroperation.py `space.is_data_descr(w_descr)`
///
/// In Python, a data descriptor is any object whose type defines __set__
/// or __delete__. For pyre's current object model, we check the ATTR_TABLE
/// and type dict for these names.
unsafe fn is_data_descriptor(descr: PyObjectRef) -> bool {
    if descr.is_null() {
        return false;
    }
    // property objects are always data descriptors
    // PyPy: W_Property has __get__, __set__, __delete__
    if is_property(descr) {
        return true;
    }
    // Check if the descriptor's class has __set__ or __delete__
    if is_instance(descr) {
        let w_type = w_instance_get_type(descr);
        if !w_type.is_null() && is_type(w_type) {
            return lookup_in_type_mro(w_type, "__set__").is_some()
                || lookup_in_type_mro(w_type, "__delete__").is_some();
        }
    }
    false
}

/// Call a descriptor's __get__ method.
///
/// PyPy: descroperation.py `space.get(w_descr, w_obj)` →
/// `w_descr.__get__(w_obj, w_type)`
///
/// Returns Some(result) if __get__ was found and called, None otherwise.
/// Call a descriptor's __get__ method.
///
/// PyPy: descroperation.py `space.get(w_descr, w_obj)` →
/// dispatch on descriptor type, then fallback to __get__ MRO lookup.
unsafe fn call_descriptor_get(
    descr: PyObjectRef,
    obj: PyObjectRef,
    w_type: PyObjectRef,
) -> Option<PyObjectRef> {
    if descr.is_null() {
        return None;
    }

    // property: PyPy W_Property.get → call fget(obj)
    if is_property(descr) {
        let fget = w_property_get_fget(descr);
        if fget.is_null() || is_none(fget) {
            return None;
        }
        return call_func_1(fget, obj);
    }

    // staticmethod: PyPy StaticMethod.descr_staticmethod_get → return w_function
    if is_staticmethod(descr) {
        return Some(w_staticmethod_get_func(descr));
    }

    // classmethod: PyPy ClassMethod.descr_classmethod_get → func bound to class
    // Simplified: return a closure-like call that prepends class.
    // For now, return the inner function (class arg not yet prepended).
    if is_classmethod(descr) {
        return Some(w_classmethod_get_func(descr));
    }

    // General __get__: look up __get__ on the descriptor's own type MRO
    // PyPy: descroperation.py → space.get_and_call_function(w_get, descr, obj, type)
    if is_instance(descr) {
        let descr_type = w_instance_get_type(descr);
        if let Some(get_fn) = lookup_in_type_mro(descr_type, "__get__") {
            if !get_fn.is_null() {
                // Call __get__(descr, obj, type) via space.call_function
                return Some(crate::space_call_function(get_fn, &[descr, obj, w_type]));
            }
        }
    }
    None
}

/// Call a Python callable with one arg via space_call_function.
///
/// PyPy: `space.call_function(w_func, w_arg)`
unsafe fn call_func_1(func: PyObjectRef, arg: PyObjectRef) -> Option<PyObjectRef> {
    Some(crate::space_call_function(func, &[arg]))
}

/// Call a descriptor's __set__ method.
///
/// PyPy: descroperation.py `descr__setattr__` →
/// `space.get_and_call_function(w_set, w_descr, w_obj, w_value)`
unsafe fn call_descriptor_set(descr: PyObjectRef, obj: PyObjectRef, value: PyObjectRef) -> bool {
    if descr.is_null() {
        return false;
    }

    // property: PyPy W_Property.set → call_function(fset, obj, value)
    if is_property(descr) {
        let fset = w_property_get_fset(descr);
        if fset.is_null() || is_none(fset) {
            return false;
        }
        crate::space_call_function(fset, &[obj, value]);
        return true;
    }

    // General __set__: look up on descriptor's type MRO
    if is_instance(descr) {
        let descr_type = w_instance_get_type(descr);
        if let Some(set_fn) = lookup_in_type_mro(descr_type, "__set__") {
            if !set_fn.is_null() {
                crate::space_call_function(set_fn, &[obj, value]);
                return true;
            }
        }
    }
    false
}

/// Set an attribute on an object: `obj.name = value`.
///
/// Stores the attribute in the per-object side table.
/// PyPy: descroperation.py descr__setattr__

pub fn py_setattr(obj: PyObjectRef, name: &str, value: PyObjectRef) -> PyResult {
    let obj = unwrap_cell(obj);
    let value = unwrap_cell(value);
    // Data descriptor __set__ takes priority (PyPy: descr__setattr__ step 1)
    unsafe {
        if is_instance(obj) {
            let w_type = w_instance_get_type(obj);
            if let Some(descr) = lookup_in_type_mro(w_type, name) {
                if call_descriptor_set(descr, obj, value) {
                    return Ok(w_none());
                }
            }
        }
    }
    // Type objects: store in the type's own namespace (class dict).
    // PyPy: typeobject.py type.__setattr__ → w_type.dict_w[name] = w_value
    unsafe {
        if is_type(obj) {
            let dict_ptr = w_type_get_dict_ptr(obj) as *mut crate::PyNamespace;
            if !dict_ptr.is_null() {
                crate::namespace_store(&mut *dict_ptr, name, value);
                return Ok(w_none());
            }
        }
    }
    // Store in instance dict (ATTR_TABLE)
    ATTR_TABLE.with(|table| {
        let mut table = table.borrow_mut();
        let key = obj as usize;
        table
            .entry(key)
            .or_default()
            .insert(name.to_string(), value);
    });
    Ok(w_none())
}

/// Delete an attribute: `del obj.name`.
///
/// PyPy: descroperation.py descr__delattr__
pub fn py_delattr(obj: PyObjectRef, name: &str) -> PyResult {
    let obj = unwrap_cell(obj);
    // Type objects: set to PY_NULL in class dict
    // (PyNamespace doesn't support removal, null slot acts as deleted)
    unsafe {
        if is_type(obj) {
            let dict_ptr = w_type_get_dict_ptr(obj) as *mut crate::PyNamespace;
            if !dict_ptr.is_null() {
                crate::namespace_store(&mut *dict_ptr, name, PY_NULL);
                return Ok(w_none());
            }
        }
    }
    // Instance/general: remove from ATTR_TABLE
    let removed = ATTR_TABLE.with(|table| {
        let mut table = table.borrow_mut();
        table
            .get_mut(&(obj as usize))
            .and_then(|d| d.remove(name))
            .is_some()
    });
    if removed {
        Ok(w_none())
    } else {
        let tp_name = unsafe { (*(*obj).ob_type).tp_name };
        Err(PyError::new(
            PyErrorKind::AttributeError,
            format!("'{tp_name}' object has no attribute '{name}'"),
        ))
    }
}

/// `iter(obj)` — PyPy: space.iter(w_obj)
/// Calls __iter__ on the object if available.
pub fn py_iter(obj: PyObjectRef) -> PyResult {
    let obj = unwrap_cell(obj);
    unsafe {
        // Builtin iterables
        if is_list(obj) {
            return Ok(pyre_object::w_seq_iter_new(obj, w_list_len(obj)));
        }
        if is_tuple(obj) {
            return Ok(pyre_object::w_seq_iter_new(obj, w_tuple_len(obj)));
        }
        if is_str(obj) {
            let len = w_str_get_value(obj).len();
            return Ok(pyre_object::w_seq_iter_new(obj, len));
        }
        // Already an iterator
        if is_range_iter(obj) || is_seq_iter(obj) || pyre_object::generatorobject::is_generator(obj)
        {
            return Ok(obj);
        }
        // Instance __iter__ — check type MRO and ATTR_TABLE
        if is_instance(obj) {
            if let Ok(method) = py_getattr(obj, "__iter__") {
                return Ok(crate::space_call_function(method, &[obj]));
            }
        }
    }
    Err(PyError::type_error(format!(
        "'{}' object is not iterable",
        unsafe { (*(*obj).ob_type).tp_name }
    )))
}

/// `next(iterator)` — PyPy: space.next(w_iter)
pub fn py_next(obj: PyObjectRef) -> PyResult {
    let obj = unwrap_cell(obj);
    unsafe {
        // Seq iterator
        if is_seq_iter(obj) {
            let iter = &mut *(obj as *mut pyre_object::W_SeqIterator);
            let seq = iter.seq;
            let idx = iter.index;
            let item = if is_list(seq) {
                pyre_object::w_list_getitem(seq, idx)
            } else if is_tuple(seq) {
                pyre_object::w_tuple_getitem(seq, idx)
            } else if is_str(seq) {
                let s = w_str_get_value(seq);
                s.chars().nth(idx as usize).map(|c| {
                    let mut buf = [0u8; 4];
                    w_str_new(c.encode_utf8(&mut buf))
                })
            } else {
                None
            };
            if let Some(v) = item {
                iter.index += 1;
                return Ok(v);
            }
            return Err(PyError {
                kind: PyErrorKind::StopIteration,
                message: "".to_string(),
                exc_object: std::ptr::null_mut(),
            });
        }
        // Range iterator
        if is_range_iter(obj) {
            let iter = &mut *(obj as *mut pyre_object::rangeobject::W_RangeIterator);
            if iter.current < iter.stop {
                let val = w_int_new(iter.current);
                iter.current += iter.step;
                return Ok(val);
            }
            return Err(PyError {
                kind: PyErrorKind::StopIteration,
                message: "".to_string(),
                exc_object: std::ptr::null_mut(),
            });
        }
        // Generator __next__ — PyPy: generator.py GeneratorIterator.next
        if pyre_object::generatorobject::is_generator(obj) {
            return generator_next(obj);
        }
        // Instance __next__
        if is_instance(obj) {
            let w_type = w_instance_get_type(obj);
            if let Some(method) = lookup_in_type_mro(w_type, "__next__") {
                return Ok(crate::space_call_function(method, &[obj]));
            }
        }
    }
    Err(PyError::type_error("not an iterator"))
}

/// Stub method for generator.close/send/throw — no-op.
fn gen_stub_method(args: &[PyObjectRef]) -> PyResult {
    // close(): mark exhausted
    if !args.is_empty() {
        unsafe {
            if pyre_object::generatorobject::is_generator(args[0]) {
                pyre_object::generatorobject::w_generator_set_exhausted(args[0]);
            }
        }
    }
    Ok(w_none())
}

/// Resume a generator frame until YIELD_VALUE or RETURN_VALUE.
///
/// PyPy: generator.py GeneratorIterator.next() → execute_frame()
fn generator_next(gen_obj: PyObjectRef) -> PyResult {
    use pyre_object::generatorobject::*;
    unsafe {
        if w_generator_is_exhausted(gen_obj) {
            return Err(PyError {
                kind: PyErrorKind::StopIteration,
                message: "".to_string(),
                exc_object: std::ptr::null_mut(),
            });
        }
        let frame_ptr = w_generator_get_frame(gen_obj) as *mut crate::frame::PyFrame;
        if frame_ptr.is_null() {
            w_generator_set_exhausted(gen_obj);
            return Err(PyError {
                kind: PyErrorKind::StopIteration,
                message: "".to_string(),
                exc_object: std::ptr::null_mut(),
            });
        }
        let frame = &mut *frame_ptr;

        // On resume (not first call), push the sent value (None for __next__).
        // CPython: YIELD_VALUE pops value, RESUME+POP_TOP expects sent value on stack.
        if w_generator_is_started(gen_obj) {
            frame.push(w_none());
        }
        w_generator_set_started(gen_obj);

        // Resume the frame from where it left off (frame.next_instr is preserved)
        match crate::eval::eval_loop_for_force(frame) {
            Ok(value) => {
                // Check if this is a yield or a return.
                // StepResult::Yield returns the yielded value via Ok.
                // StepResult::Return also returns via Ok but means generator is done.
                // We distinguish by checking if frame reached the end.
                let code = &*frame.code;
                if frame.next_instr >= code.instructions.len() {
                    // Generator returned (finished) — raise StopIteration
                    w_generator_set_exhausted(gen_obj);
                    return Err(PyError {
                        kind: PyErrorKind::StopIteration,
                        message: "".to_string(),
                        exc_object: std::ptr::null_mut(),
                    });
                }
                // Frame is suspended at YIELD_VALUE — return the yielded value
                Ok(value)
            }
            Err(e) => {
                w_generator_set_exhausted(gen_obj);
                Err(e)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_int_add() {
        let a = w_int_new(3);
        let b = w_int_new(4);
        let result = py_add(a, b).unwrap();
        unsafe { assert_eq!(w_int_get_value(result), 7) };
    }

    #[test]
    fn test_int_compare() {
        let a = w_int_new(5);
        let b = w_int_new(10);
        let result = py_compare(a, b, CompareOp::Lt).unwrap();
        unsafe { assert!(w_bool_get_value(result)) };
    }

    #[test]
    fn test_zero_division() {
        let a = w_int_new(5);
        let b = w_int_new(0);
        assert!(py_floordiv(a, b).is_err());
    }

    #[test]
    fn test_truthiness() {
        assert!(py_is_true(w_int_new(1)));
        assert!(!py_is_true(w_int_new(0)));
        assert!(!py_is_true(w_none()));
        assert!(py_is_true(w_bool_from(true)));
        assert!(!py_is_true(w_bool_from(false)));
    }

    #[test]
    fn test_int_add_overflow() {
        let a = w_int_new(i64::MAX);
        let b = w_int_new(1);
        let result = py_add(a, b).unwrap();
        unsafe {
            assert!(is_long(result));
            assert_eq!(
                *w_long_get_value(result),
                BigInt::from(i64::MAX) + BigInt::from(1)
            );
        }
    }

    #[test]
    fn test_int_sub_overflow() {
        let a = w_int_new(i64::MIN);
        let b = w_int_new(1);
        let result = py_sub(a, b).unwrap();
        unsafe {
            assert!(is_long(result));
            assert_eq!(
                *w_long_get_value(result),
                BigInt::from(i64::MIN) - BigInt::from(1)
            );
        }
    }

    #[test]
    fn test_int_mul_overflow() {
        let a = w_int_new(i64::MAX);
        let b = w_int_new(2);
        let result = py_mul(a, b).unwrap();
        unsafe {
            assert!(is_long(result));
            assert_eq!(
                *w_long_get_value(result),
                BigInt::from(i64::MAX) * BigInt::from(2)
            );
        }
    }

    #[test]
    fn test_long_add() {
        let a = w_long_new(BigInt::from(i64::MAX) + BigInt::from(1));
        let b = w_int_new(100);
        let result = py_add(a, b).unwrap();
        unsafe {
            assert!(is_long(result));
            assert_eq!(
                *w_long_get_value(result),
                BigInt::from(i64::MAX) + BigInt::from(101)
            );
        }
    }

    #[test]
    fn test_long_demote_to_int() {
        // long + long that fits back in i64 → W_IntObject
        let a = w_long_new(BigInt::from(i64::MAX) + BigInt::from(1));
        let b = w_int_new(-1);
        let result = py_add(a, b).unwrap();
        unsafe {
            assert!(is_int(result));
            assert_eq!(w_int_get_value(result), i64::MAX);
        }
    }

    #[test]
    fn test_negate_min_int() {
        let a = w_int_new(i64::MIN);
        let result = py_negative(a).unwrap();
        unsafe {
            assert!(is_long(result));
            assert_eq!(*w_long_get_value(result), -BigInt::from(i64::MIN));
        }
    }

    #[test]
    fn test_invert_int() {
        let result = py_invert(w_int_new(6)).unwrap();
        unsafe {
            assert!(is_int(result));
            assert_eq!(w_int_get_value(result), !6);
        }
    }

    #[test]
    fn test_long_compare() {
        let a = w_long_new(BigInt::from(i64::MAX) + BigInt::from(1));
        let b = w_int_new(i64::MAX);
        let result = py_compare(a, b, CompareOp::Gt).unwrap();
        unsafe { assert!(w_bool_get_value(result)) };
    }

    #[test]
    fn test_long_truthiness() {
        assert!(py_is_true(w_long_new(
            BigInt::from(i64::MAX) + BigInt::from(1)
        )));
        assert!(!py_is_true(w_long_new(BigInt::from(0))));
    }

    #[test]
    fn test_int_pow() {
        let result = py_pow(w_int_new(2), w_int_new(10)).unwrap();
        unsafe { assert_eq!(w_int_get_value(result), 1024) };
    }

    #[test]
    fn test_int_pow_overflow() {
        let result = py_pow(w_int_new(2), w_int_new(63)).unwrap();
        unsafe {
            // 2^63 overflows i64, should be long
            assert!(is_long(result));
            assert_eq!(*w_long_get_value(result), BigInt::from(2).pow(63));
        }
    }

    #[test]
    fn test_int_pow_negative_exponent() {
        let result = py_pow(w_int_new(2), w_int_new(-1)).unwrap();
        unsafe {
            assert!(is_float(result));
            assert_eq!(w_float_get_value(result), 0.5);
        }
    }

    #[test]
    fn test_int_lshift() {
        let result = py_lshift(w_int_new(1), w_int_new(10)).unwrap();
        unsafe { assert_eq!(w_int_get_value(result), 1024) };
    }

    #[test]
    fn test_int_lshift_overflow() {
        let result = py_lshift(w_int_new(1), w_int_new(64)).unwrap();
        unsafe {
            assert!(is_long(result));
            assert_eq!(*w_long_get_value(result), BigInt::from(1) << 64);
        }
    }

    #[test]
    fn test_int_rshift() {
        let result = py_rshift(w_int_new(1024), w_int_new(3)).unwrap();
        unsafe { assert_eq!(w_int_get_value(result), 128) };
    }

    #[test]
    fn test_negative_shift_count() {
        assert!(py_lshift(w_int_new(1), w_int_new(-1)).is_err());
        assert!(py_rshift(w_int_new(1), w_int_new(-1)).is_err());
    }

    #[test]
    fn test_int_bitand() {
        let result = py_bitand(w_int_new(0xFF), w_int_new(0x0F)).unwrap();
        unsafe { assert_eq!(w_int_get_value(result), 0x0F) };
    }

    #[test]
    fn test_int_bitor() {
        let result = py_bitor(w_int_new(0xF0), w_int_new(0x0F)).unwrap();
        unsafe { assert_eq!(w_int_get_value(result), 0xFF) };
    }

    #[test]
    fn test_int_bitxor() {
        let result = py_bitxor(w_int_new(0xFF), w_int_new(0x0F)).unwrap();
        unsafe { assert_eq!(w_int_get_value(result), 0xF0) };
    }

    #[test]
    fn test_long_bitand() {
        let a = w_long_new(BigInt::from(i64::MAX) + BigInt::from(1));
        let b = w_int_new(0xFF);
        let result = py_bitand(a, b).unwrap();
        unsafe { assert_eq!(w_int_get_value(result), 0) };
    }

    #[test]
    fn test_invert_long() {
        let a = w_long_new(BigInt::from(i64::MAX) + BigInt::from(1));
        let result = py_invert(a).unwrap();
        unsafe {
            assert!(is_long(result));
            assert_eq!(
                *w_long_get_value(result),
                !(BigInt::from(i64::MAX) + BigInt::from(1))
            );
        }
    }

    #[test]
    fn test_setattr_getattr() {
        let obj = w_int_new(42);
        py_setattr(obj, "name", w_int_new(100)).unwrap();
        let result = py_getattr(obj, "name").unwrap();
        unsafe { assert_eq!(w_int_get_value(result), 100) };
    }

    #[test]
    fn test_getattr_missing() {
        let obj = w_int_new(1);
        let err = py_getattr(obj, "missing").unwrap_err();
        assert!(matches!(err.kind, PyErrorKind::AttributeError));
    }

    #[test]
    fn test_setattr_overwrite() {
        let obj = w_int_new(42);
        py_setattr(obj, "x", w_int_new(1)).unwrap();
        py_setattr(obj, "x", w_int_new(2)).unwrap();
        let result = py_getattr(obj, "x").unwrap();
        unsafe { assert_eq!(w_int_get_value(result), 2) };
    }

    #[test]
    fn test_py_contains_manual_list() {
        let list = w_list_new(vec![w_int_new(1), w_int_new(2), w_int_new(3)]);
        let needle = w_int_new(1);
        unsafe {
            assert!(
                is_list(list),
                "should be list, got type: {}",
                (*(*list).ob_type).tp_name
            );
        }
        let result = super::py_contains(list, needle).expect("py_contains failed");
        assert!(result, "1 should be in [1, 2, 3]");
    }
}

/// `in` operator: check if `needle` is in `haystack`.
/// PyPy: space.contains_w(haystack, needle)
pub fn py_contains(haystack: PyObjectRef, needle: PyObjectRef) -> Result<bool, PyError> {
    use pyre_object::*;
    unsafe {
        if is_list(haystack) {
            let len = w_list_len(haystack);
            for i in 0..len {
                if let Some(item) = w_list_getitem(haystack, i as i64) {
                    if py_eq_bool(item, needle) {
                        return Ok(true);
                    }
                }
            }
            return Ok(false);
        }
        if is_tuple(haystack) {
            let len = w_tuple_len(haystack);
            for i in 0..len {
                if let Some(item) = w_tuple_getitem(haystack, i as i64) {
                    if py_eq_bool(item, needle) {
                        return Ok(true);
                    }
                }
            }
            return Ok(false);
        }
        if is_str(haystack) && is_str(needle) {
            let h = w_str_get_value(haystack);
            let n = w_str_get_value(needle);
            return Ok(h.contains(n));
        }
    }
    // Instance __contains__ — PyPy: descroperation.py contains_w
    unsafe {
        if is_instance(haystack) {
            let w_type = w_instance_get_type(haystack);
            if let Some(method) = lookup_in_type_mro(w_type, "__contains__") {
                let result = crate::space_call_function(method, &[haystack, needle]);
                return Ok(py_is_true(result));
            }
        }
    }
    // Fallback: try iterating with py_getitem(obj, i) for i=0,1,...
    unsafe {
        let mut i = 0i64;
        loop {
            match py_getitem(haystack, pyre_object::w_int_new(i)) {
                Ok(item) => {
                    if py_eq_bool(item, needle) {
                        return Ok(true);
                    }
                    i += 1;
                }
                Err(_) => return Ok(false), // IndexError → not found
            }
        }
    }
}

/// Compare two objects for equality (returns bool, not PyObjectRef).
fn py_eq_bool(a: PyObjectRef, b: PyObjectRef) -> bool {
    if a == b {
        return true;
    }
    unsafe {
        use pyre_object::*;
        if is_int(a) && is_int(b) {
            return w_int_get_value(a) == w_int_get_value(b);
        }
        if is_str(a) && is_str(b) {
            return w_str_get_value(a) == w_str_get_value(b);
        }
    }
    py_compare(a, b, CompareOp::Eq)
        .map(|r| py_is_true(r))
        .unwrap_or(false)
}

/// Delete item: `del obj[index]`
pub fn py_delitem(obj: PyObjectRef, index: PyObjectRef) -> Result<(), PyError> {
    use pyre_object::*;
    unsafe {
        if is_list(obj) && is_int(index) {
            let i = w_int_get_value(index);
            let len = w_list_len(obj) as i64;
            let idx = if i < 0 { len + i } else { i };
            if idx >= 0 && idx < len {
                // For Phase 1: set to PY_NULL (proper removal needs list mutation API)
                w_list_setitem(obj, idx, PY_NULL);
                return Ok(());
            }
            return Err(PyError::type_error("list index out of range"));
        }
    }
    // Instance __delitem__ — PyPy: descroperation.py delitem
    unsafe {
        if pyre_object::is_instance(obj) {
            if let Some(method) =
                lookup_in_type_mro(pyre_object::w_instance_get_type(obj), "__delitem__")
            {
                crate::space_call_function(method, &[obj, index]);
                return Ok(());
            }
        }
    }
    Err(PyError::type_error("object does not support item deletion"))
}

// py_str and py_repr are defined in display.rs (with __str__/__repr__ dispatch).
// Re-exported via crate::display::*.
