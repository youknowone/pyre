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

use crate::function::is_function;
pub use crate::{PyError, PyErrorKind, PyResult};
use pyre_object::strobject::is_str;
use pyre_object::*;

/// Compatibility alias for PyPy's base-object type.
/// PyPy frequently models interpreter values as subclasses of `W_Root`.
pub type W_Root = PyObjectRef;

/// Compatibility marker for a type mismatch in descriptor lookup.
#[derive(Debug, Clone)]
pub struct DescrMismatch;

/// Compatibility marker for lock-sensitive APIs that are disabled under
/// this no-GIL runtime.
#[derive(Debug, Clone)]
pub struct CannotHaveLock;

/// Minimal compatibility placeholder for PyPy-style cache objects.
#[derive(Debug, Default)]
pub struct SpaceCache {
    space: PyObjectRef,
    _entries: RefCell<HashMap<usize, PyObjectRef>>,
}

impl SpaceCache {
    pub fn new(space: PyObjectRef) -> Self {
        Self {
            space,
            _entries: RefCell::new(HashMap::new()),
        }
    }

    #[inline]
    pub fn getorbuild(&self, _key: PyObjectRef) -> PyObjectRef {
        std::ptr::null_mut()
    }

    #[inline]
    pub fn ready(&self, _result: PyObjectRef) {}
}

/// Compatibility cache variant with `callable(self)` construction path.
#[derive(Debug, Default)]
pub struct InternalSpaceCache {
    base: SpaceCache,
}

impl InternalSpaceCache {
    pub fn new(space: PyObjectRef) -> Self {
        Self {
            base: SpaceCache::new(space),
        }
    }

    #[inline]
    pub fn getorbuild<F>(&self, f: F) -> PyObjectRef
    where
        F: FnOnce(PyObjectRef) -> PyObjectRef,
    {
        let _ = self.base.space;
        f(std::ptr::null_mut())
    }
}

/// Compatibility helper used by `ObjSpace` bootstrap in PyPy.
#[derive(Debug, Default)]
pub struct AppExecCache {
    base: SpaceCache,
}

impl AppExecCache {
    pub fn new(space: PyObjectRef) -> Self {
        Self {
            base: SpaceCache::new(space),
        }
    }

    pub fn build(&self, _source: PyObjectRef) -> PyObjectRef {
        let _ = self.base.space;
        std::ptr::null_mut()
    }
}

/// Very small compatibility object for PyPy's `ObjSpace` interface.
/// The full object-space API is implemented as free functions in this module.
#[derive(Debug, Default)]
pub struct ObjSpace {
    fromcache: Option<PyObjectRef>,
}

impl ObjSpace {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn fromcache<T, F>(&self, mut build: F, cache: &SpaceCache) -> T
    where
        T: Default,
        F: FnMut(&SpaceCache) -> T,
    {
        let _ = cache.getorbuild(std::ptr::null_mut());
        build(cache)
    }
}

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

/// floatobject.py:508-512: descr_floordiv → _divmod_w()[0].
unsafe fn float_floordiv(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let (floordiv, _mod) = float_divmod_w(as_float(a), as_float(b))?;
    Ok(w_float_new(floordiv))
}

/// floatobject.py:520-540: descr_mod with math_fmod + sign correction.
unsafe fn float_mod(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let x = as_float(a);
    let y = as_float(b);
    if y == 0.0 {
        // floatobject.py:526
        return Err(PyError::zero_division("float modulo"));
    }
    let mut m = x % y; // fmod
    if m != 0.0 {
        // floatobject.py:529-531: ensure remainder has same sign as denominator
        if (y < 0.0) != (m < 0.0) {
            m += y;
        }
    } else {
        // floatobject.py:536-538: signed zero — copysign(0.0, y)
        m = f64::copysign(0.0, y);
    }
    Ok(w_float_new(m))
}

/// floatobject.py:758-793: _divmod_w.
fn float_divmod_w(x: f64, y: f64) -> Result<(f64, f64), PyError> {
    if y == 0.0 {
        // floatobject.py:761
        return Err(PyError::zero_division("float modulo"));
    }
    let mut m = x % y; // fmod
    // floatobject.py:767: div = (x - mod) / y
    let mut div = (x - m) / y;
    if m != 0.0 {
        // floatobject.py:769-771: sign correction
        if (y < 0.0) != (m < 0.0) {
            m += y;
            div -= 1.0;
        }
    } else {
        // floatobject.py:776-778: signed zero
        // "mod *= mod" hides "+0" from optimizer, then negate if y < 0
        m = m * m; // hide from optimizer
        if y < 0.0 {
            m = -m;
        }
    }
    // floatobject.py:784-790: snap quotient to nearest integral value
    let floordiv = if div != 0.0 {
        let f = div.floor();
        if div - f > 0.5 { f + 1.0 } else { f }
    } else {
        // floatobject.py:789-790: zero with sign of true quotient
        let d = div * div; // hide from optimizer
        d * x / y
    };
    Ok((floordiv, m))
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
                && issubtype_cached(b_type, a_type)
                && lookup_in_type_where(b_type, rdunder).is_some()
        } else {
            false
        }
    } else {
        false
    };

    if try_reverse_first {
        let rdunder = reverse_dunder(dunder).unwrap();
        let w_type = w_instance_get_type(b);
        if let Some(method) = lookup_in_type_where(w_type, rdunder) {
            let result = crate::call_function(method, &[b, a]);
            if !is_not_implemented(result) {
                return Some(Ok(result));
            }
        }
    }

    // Forward: a.__op__(b)
    if a_is_inst {
        let w_type = w_instance_get_type(a);
        if let Some(method) = lookup_in_type_where(w_type, dunder) {
            let result = crate::call_function(method, &[a, b]);
            if !is_not_implemented(result) {
                return Some(Ok(result));
            }
        }
        // Also check per-instance attributes (ATTR_TABLE)
        if let Ok(method) = getattr(a, dunder) {
            let result = crate::call_function(method, &[a, b]);
            if !is_not_implemented(result) {
                return Some(Ok(result));
            }
        }
    }

    // Reverse: b.__rop__(a) — only if not already tried above
    if !try_reverse_first && b_is_inst {
        if let Some(rdunder) = reverse_dunder(dunder) {
            let w_type = w_instance_get_type(b);
            if let Some(method) = lookup_in_type_where(w_type, rdunder) {
                let result = crate::call_function(method, &[b, a]);
                if !is_not_implemented(result) {
                    return Some(Ok(result));
                }
            }
        }
    }

    None
}

/// Check if w_type is a subtype of cls using cached MRO.
unsafe fn issubtype_cached(w_type: PyObjectRef, cls: PyObjectRef) -> bool {
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
/// PyPy: `ObjSpace.call_function(space.lookup(w_obj, dunder), w_obj)`
unsafe fn try_instance_unaryop(a: PyObjectRef, dunder: &str) -> Option<PyResult> {
    if is_instance(a) {
        if let Some(method) = lookup(a, dunder) {
            return Some(Ok(crate::call_function(method, &[a])));
        }
    }
    None
}

/// Binary operation dispatch.
///
/// Checks types and dispatches to the appropriate fast path.
/// The JIT traces through this function, recording `GuardClass` on operand types.

pub fn add(a: PyObjectRef, b: PyObjectRef) -> PyResult {
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
        if pyre_object::bytearrayobject::is_bytearray(a)
            && pyre_object::bytearrayobject::is_bytearray(b)
        {
            let a_data = pyre_object::bytearrayobject::w_bytearray_data(a);
            let b_data = pyre_object::bytearrayobject::w_bytearray_data(b);
            let mut result = a_data.to_vec();
            result.extend_from_slice(b_data);
            return Ok(pyre_object::bytearrayobject::w_bytearray_from_bytes(
                &result,
            ));
        }
        // Instance dunder dispatch: __add__
        if let Some(result) = try_instance_binop(a, b, "__add__") {
            return result;
        }
        Err(PyError::type_error(format!(
            "unsupported operand type(s) for +: '{}' and '{}'",
            (*(*a).ob_type).name,
            (*(*b).ob_type).name,
        )))
    }
}

pub fn sub(a: PyObjectRef, b: PyObjectRef) -> PyResult {
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
            (*(*a).ob_type).name,
            (*(*b).ob_type).name,
        )))
    }
}

pub fn mul(a: PyObjectRef, b: PyObjectRef) -> PyResult {
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
            (*(*a).ob_type).name,
            (*(*b).ob_type).name,
        )))
    }
}

pub fn floordiv(a: PyObjectRef, b: PyObjectRef) -> PyResult {
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
            (*(*a).ob_type).name,
            (*(*b).ob_type).name,
        )))
    }
}

pub fn mod_(a: PyObjectRef, b: PyObjectRef) -> PyResult {
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
            (*(*a).ob_type).name,
            (*(*b).ob_type).name,
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
            // Named format: %(name)s — look up key in dict
            let named_arg = if i < bytes.len() && bytes[i] == b'(' {
                i += 1; // skip '('
                let mut key = String::new();
                while i < bytes.len() && bytes[i] != b')' {
                    key.push(bytes[i] as char);
                    i += 1;
                }
                if i < bytes.len() {
                    i += 1; // skip ')'
                }
                // args must be a dict
                if is_dict(args) {
                    w_dict_getitem_str(args, &key)
                } else {
                    None
                }
            } else {
                None
            };
            // Parse optional width/flags
            let mut width = String::new();
            while i < bytes.len()
                && (bytes[i] == b'-'
                    || bytes[i] == b'+'
                    || bytes[i] == b'0'
                    || bytes[i] == b' '
                    || bytes[i].is_ascii_digit()
                    || bytes[i] == b'*'
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
            let arg = if let Some(na) = named_arg {
                na
            } else if arg_idx < arg_list.len() {
                let a = arg_list[arg_idx];
                arg_idx += 1;
                a
            } else {
                pyre_object::w_none()
            };
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

pub fn truediv(a: PyObjectRef, b: PyObjectRef) -> PyResult {
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
            (*(*a).ob_type).name,
            (*(*b).ob_type).name,
        )))
    }
}

/// Power operation dispatch (`**` operator).

pub fn pow(a: PyObjectRef, b: PyObjectRef) -> PyResult {
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
            return float_pow_impl(as_float(a), as_float(b));
        }
        if let Some(result) = try_instance_binop(a, b, "__pow__") {
            return result;
        }
        Err(PyError::type_error(format!(
            "unsupported operand type(s) for **: '{}' and '{}'",
            (*(*a).ob_type).name,
            (*(*b).ob_type).name,
        )))
    }
}

/// floatobject.py:799-881: `_pow(space, x, y)` parity — raw float power
/// with Python-correct semantics. Handles NaN/Inf edge cases and raises
/// ZeroDivisionError / ValueError / OverflowError for domain errors.
///
/// Returns raw `f64` (matching RPython which returns `r_float`) so the
/// JIT fast path can call this directly with raw-float arguments without
/// any intermediate W_FloatObject allocation. The interpreter wrapper
/// `float_pow_impl` boxes the result into a W_FloatObject.
pub fn float_pow_raw(x: f64, y: f64) -> Result<f64, PyError> {
    // floatobject.py:800-801
    if y == 2.0 {
        return Ok(x * x);
    }
    // floatobject.py:803-804
    if y == 0.0 {
        return Ok(1.0);
    }
    // floatobject.py:806-807
    if x.is_nan() {
        return Ok(x);
    }
    // floatobject.py:809-814
    if y.is_nan() {
        return Ok(if x == 1.0 { 1.0 } else { y });
    }
    // floatobject.py:815-827
    if y.is_infinite() {
        let ax = x.abs();
        if ax == 1.0 {
            return Ok(1.0);
        }
        return Ok(if (y > 0.0) == (ax > 1.0) {
            f64::INFINITY
        } else {
            0.0
        });
    }
    // floatobject.py:828-842
    if x.is_infinite() {
        let y_is_odd = y.abs() % 2.0 == 1.0;
        return Ok(if y > 0.0 {
            if y_is_odd { x } else { x.abs() }
        } else if y_is_odd {
            f64::copysign(0.0, x)
        } else {
            0.0
        });
    }
    // floatobject.py:844-847
    if x == 0.0 && y < 0.0 {
        return Err(PyError::zero_division(
            "0.0 cannot be raised to a negative power",
        ));
    }
    // floatobject.py:849-862
    let mut negate_result = false;
    let mut bx = x;
    if bx < 0.0 {
        if y.floor() != y {
            return Err(PyError::value_error(
                "negative number cannot be raised to a fractional power",
            ));
        }
        bx = -bx;
        negate_result = y.abs() % 2.0 == 1.0;
    }
    // floatobject.py:864-869
    if bx == 1.0 {
        return Ok(if negate_result { -1.0 } else { 1.0 });
    }
    // floatobject.py:871-877
    let z = bx.powf(y);
    if z.is_infinite() && !bx.is_infinite() {
        return Err(PyError::overflow_error("float power"));
    }
    // floatobject.py:879-881
    Ok(if negate_result { -z } else { z })
}

/// floatobject.py:562 `W_FloatObject.descr_pow` boxing wrapper over `_pow`.
/// Calls `float_pow_raw` and boxes the raw result into W_FloatObject.
fn float_pow_impl(x: f64, y: f64) -> PyResult {
    use pyre_object::w_float_new;
    Ok(w_float_new(float_pow_raw(x, y)?))
}

/// Left shift dispatch (`<<` operator).

pub fn lshift(a: PyObjectRef, b: PyObjectRef) -> PyResult {
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
            (*(*a).ob_type).name,
            (*(*b).ob_type).name,
        )))
    }
}

/// Right shift dispatch (`>>` operator).

pub fn rshift(a: PyObjectRef, b: PyObjectRef) -> PyResult {
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
            (*(*a).ob_type).name,
            (*(*b).ob_type).name,
        )))
    }
}

/// Bitwise AND dispatch (`&` operator).

pub fn and_(a: PyObjectRef, b: PyObjectRef) -> PyResult {
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
            (*(*a).ob_type).name,
            (*(*b).ob_type).name,
        )))
    }
}

/// Check if an object can participate in `X | Y` union syntax.
///
/// PyPy equivalent: _unionable() in _pypy_generic_alias.py
#[inline]
fn unionable(obj: PyObjectRef) -> bool {
    unsafe { is_none(obj) || is_type(obj) || pyre_object::is_union(obj) }
}

/// Bitwise OR dispatch (`|` operator).

pub fn or_(a: PyObjectRef, b: PyObjectRef) -> PyResult {
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
        // type | type — PEP 604 union types (Python 3.10+)
        // PyPy: typeobject.py descr_or → _pypy_generic_alias._create_union
        if unionable(a) && unionable(b) {
            return Ok(pyre_object::w_union_new(a, b));
        }
        // set | set bitwise OR
        if let Some(result) = try_instance_binop(a, b, "__ror__") {
            return result;
        }
        Err(PyError::type_error(format!(
            "unsupported operand type(s) for |: '{}' and '{}'",
            (*(*a).ob_type).name,
            (*(*b).ob_type).name,
        )))
    }
}

/// Bitwise XOR dispatch (`^` operator).

pub fn xor(a: PyObjectRef, b: PyObjectRef) -> PyResult {
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
            (*(*a).ob_type).name,
            (*(*b).ob_type).name,
        )))
    }
}

/// Comparison operation dispatch.

pub fn compare(a: PyObjectRef, b: PyObjectRef, op: CompareOp) -> PyResult {
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
        // Tuple lexicographic comparison — PyPy: tupleobject.py descr_lt / _eq / etc.
        if is_tuple(a) && is_tuple(b) {
            let la = w_tuple_len(a);
            let lb = w_tuple_len(b);
            let min_len = la.min(lb);
            for i in 0..min_len {
                let ea = w_tuple_getitem(a, i as i64).unwrap_or(PY_NULL);
                let eb = w_tuple_getitem(b, i as i64).unwrap_or(PY_NULL);
                let eq = match compare(ea, eb, CompareOp::Eq) {
                    Ok(r) => is_true(r),
                    Err(_) => false,
                };
                if !eq {
                    return compare(ea, eb, op);
                }
            }
            return Ok(w_bool_from(match op {
                CompareOp::Lt => la < lb,
                CompareOp::Le => la <= lb,
                CompareOp::Gt => la > lb,
                CompareOp::Ge => la >= lb,
                CompareOp::Eq => la == lb,
                CompareOp::Ne => la != lb,
            }));
        }
        // List lexicographic comparison — same logic as tuple.
        if is_list(a) && is_list(b) {
            let la = pyre_object::w_list_len(a);
            let lb = pyre_object::w_list_len(b);
            let min_len = la.min(lb);
            for i in 0..min_len {
                let ea = pyre_object::w_list_getitem(a, i as i64).unwrap_or(PY_NULL);
                let eb = pyre_object::w_list_getitem(b, i as i64).unwrap_or(PY_NULL);
                let eq = match compare(ea, eb, CompareOp::Eq) {
                    Ok(r) => is_true(r),
                    Err(_) => false,
                };
                if !eq {
                    return compare(ea, eb, op);
                }
            }
            return Ok(w_bool_from(match op {
                CompareOp::Lt => la < lb,
                CompareOp::Le => la <= lb,
                CompareOp::Gt => la > lb,
                CompareOp::Ge => la >= lb,
                CompareOp::Eq => la == lb,
                CompareOp::Ne => la != lb,
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
            (*(*a).ob_type).name,
            (*(*b).ob_type).name,
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

pub fn is_true(obj: PyObjectRef) -> bool {
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
            // Try __bool__ first (type MRO)
            if let Some(method) = lookup_in_type_where(w_type, "__bool__") {
                let result = crate::call_function(method, &[obj]);
                if !result.is_null() && is_bool(result) {
                    return w_bool_get_value(result);
                }
            }
            // Then __len__ (type MRO) — nonzero length = truthy
            if let Some(method) = lookup_in_type_where(w_type, "__len__") {
                let result = crate::call_function(method, &[obj]);
                if !result.is_null() && is_int(result) {
                    return w_int_get_value(result) != 0;
                }
            }
            // Also check per-instance __len__ (ATTR_TABLE)
            if let Ok(method) = getattr(obj, "__len__") {
                let result = crate::call_function(method, &[obj]);
                if !result.is_null() && is_int(result) {
                    return w_int_get_value(result) != 0;
                }
            }
        }
        true // default: objects are truthy
    }
}

/// Unary negation.

pub fn neg(a: PyObjectRef) -> PyResult {
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
            (*(*a).ob_type).name,
        )))
    }
}

/// Unary bitwise inversion.

pub fn invert(a: PyObjectRef) -> PyResult {
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
            (*(*a).ob_type).name,
        )))
    }
}

// ── Subscript operations ─────────────────────────────────────────────

/// Get item by index: `obj[index]`.
///
/// Dispatches based on the type of `obj`.

pub fn getitem(obj: PyObjectRef, index: PyObjectRef) -> PyResult {
    let obj = unwrap_cell(obj);
    let index = unwrap_cell(index);
    unsafe {
        if is_list(obj) {
            if is_slice(index) {
                let len = w_list_len(obj) as i64;
                let start = w_slice_get_start(index);
                let stop = w_slice_get_stop(index);
                let step = w_slice_get_step(index);
                let step_val = if is_none(step) {
                    1
                } else {
                    w_int_get_value(step)
                };
                let s = if is_none(start) {
                    if step_val < 0 { len - 1 } else { 0 }
                } else {
                    let v = w_int_get_value(start);
                    if v < 0 { (len + v).max(0) } else { v.min(len) }
                };
                let e = if is_none(stop) {
                    if step_val < 0 { -1 } else { len }
                } else {
                    let v = w_int_get_value(stop);
                    if v < 0 { (len + v).max(-1) } else { v.min(len) }
                };
                let mut items = Vec::new();
                if step_val == 1 {
                    for i in s..e {
                        if let Some(v) = w_list_getitem(obj, i) {
                            items.push(v);
                        }
                    }
                } else if step_val > 0 {
                    let mut i = s;
                    while i < e {
                        if let Some(v) = w_list_getitem(obj, i) {
                            items.push(v);
                        }
                        i += step_val;
                    }
                } else if step_val < 0 {
                    let mut i = s;
                    while i > e {
                        if let Some(v) = w_list_getitem(obj, i) {
                            items.push(v);
                        }
                        i += step_val;
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
            if is_slice(index) {
                // PyPy: tupleobject.py descr_getslice
                let len = w_tuple_len(obj) as i64;
                let start = w_slice_get_start(index);
                let stop = w_slice_get_stop(index);
                let step = w_slice_get_step(index);
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
                let step_val = if is_none(step) {
                    1
                } else {
                    w_int_get_value(step)
                };
                let s = if s < 0 { (len + s).max(0) } else { s.min(len) } as usize;
                let e = if e < 0 { (len + e).max(0) } else { e.min(len) } as usize;
                let mut items = Vec::new();
                if step_val == 1 {
                    for i in s..e {
                        if let Some(v) = w_tuple_getitem(obj, i as i64) {
                            items.push(v);
                        }
                    }
                } else if step_val > 0 {
                    let mut i = s as i64;
                    while (i as usize) < e {
                        if let Some(v) = w_tuple_getitem(obj, i) {
                            items.push(v);
                        }
                        i += step_val;
                    }
                }
                return Ok(w_tuple_new(items));
            }
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
                None => {
                    let key_repr = if is_str(index) {
                        format!("'{}'", w_str_get_value(index))
                    } else if is_int(index) {
                        format!("{}", w_int_get_value(index))
                    } else {
                        "key".to_string()
                    };
                    Err(PyError::new(
                        PyErrorKind::KeyError,
                        format!("KeyError: {key_repr}"),
                    ))
                }
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
        } else if pyre_object::bytearrayobject::is_bytearray(obj) {
            if is_int(index) {
                let idx = w_int_get_value(index);
                let len = pyre_object::bytearrayobject::w_bytearray_len(obj) as i64;
                let actual = if idx < 0 { len + idx } else { idx };
                if actual >= 0 && actual < len {
                    return Ok(w_int_new(
                        pyre_object::bytearrayobject::w_bytearray_getitem(obj, actual as usize)
                            as i64,
                    ));
                }
                return Err(PyError::new(
                    PyErrorKind::IndexError,
                    "bytearray index out of range",
                ));
            }
            if is_slice(index) {
                let len = pyre_object::bytearrayobject::w_bytearray_len(obj) as i64;
                let start = w_slice_get_start(index);
                let stop = w_slice_get_stop(index);
                let step = w_slice_get_step(index);
                let step_val = if is_none(step) {
                    1
                } else {
                    w_int_get_value(step)
                };
                let s_val = if is_none(start) {
                    if step_val < 0 { len - 1 } else { 0 }
                } else {
                    let v = w_int_get_value(start);
                    if v < 0 { (len + v).max(0) } else { v.min(len) }
                };
                let e_val = if is_none(stop) {
                    if step_val < 0 { -1 } else { len }
                } else {
                    let v = w_int_get_value(stop);
                    if v < 0 { (len + v).max(-1) } else { v.min(len) }
                };
                let mut result = Vec::new();
                let mut i = s_val;
                if step_val > 0 {
                    while i < e_val {
                        if i >= 0 && i < len {
                            result.push(pyre_object::bytearrayobject::w_bytearray_getitem(
                                obj, i as usize,
                            ));
                        }
                        i += step_val;
                    }
                } else if step_val < 0 {
                    while i > e_val {
                        if i >= 0 && i < len {
                            result.push(pyre_object::bytearrayobject::w_bytearray_getitem(
                                obj, i as usize,
                            ));
                        }
                        i += step_val;
                    }
                }
                return Ok(pyre_object::bytearrayobject::w_bytearray_from_bytes(
                    &result,
                ));
            }
            return Err(PyError::type_error("bytearray indices must be integers"));
        } else if is_type(obj) {
            // Python 3.9+ generic subscript: type[X] → __class_getitem__(X)
            // PyPy: typeobject.py type.__class_getitem__
            if let Some(method) = lookup_in_type_where(obj, "__class_getitem__") {
                return Ok(crate::call_function(method, &[obj, index]));
            }
            // Default: return the type itself (stub for GenericAlias)
            Ok(obj)
        } else if is_instance(obj) {
            // PyPy: descroperation.py __getitem__
            if let Some(method) = lookup_in_type_where(w_instance_get_type(obj), "__getitem__") {
                return Ok(crate::call_function(method, &[obj, index]));
            }
            Err(PyError::type_error(format!(
                "'{}' object is not subscriptable",
                w_type_get_name(w_instance_get_type(obj)),
            )))
        } else if is_range_iter(obj) {
            let r = &*(obj as *const pyre_object::rangeobject::W_RangeIterator);
            let len = if r.step > 0 {
                (r.stop - r.current + r.step - 1) / r.step
            } else if r.step < 0 {
                (r.current - r.stop - r.step - 1) / (-r.step)
            } else {
                0
            };
            if is_int(index) {
                // range[i]
                let i = w_int_get_value(index);
                let idx = if i < 0 { len + i } else { i };
                if idx < 0 || idx >= len {
                    Err(PyError::new(
                        PyErrorKind::IndexError,
                        "range object index out of range",
                    ))
                } else {
                    Ok(w_int_new(r.current + idx * r.step))
                }
            } else if is_slice(index) {
                // range[start:stop:step] → returns a list
                let s_raw = w_slice_get_start(index);
                let e_raw = w_slice_get_stop(index);
                let step_raw = w_slice_get_step(index);
                let s = if is_none(s_raw) {
                    0
                } else {
                    w_int_get_value(s_raw)
                };
                let e = if is_none(e_raw) {
                    len
                } else {
                    w_int_get_value(e_raw)
                };
                let sl_step = if is_none(step_raw) {
                    1
                } else {
                    w_int_get_value(step_raw)
                };
                let s = if s < 0 { (len + s).max(0) } else { s.min(len) };
                let e = if e < 0 { (len + e).max(0) } else { e.min(len) };
                let mut items = Vec::new();
                let mut i = s;
                while (sl_step > 0 && i < e) || (sl_step < 0 && i > e) {
                    items.push(w_int_new(r.current + i * r.step));
                    i += sl_step;
                }
                Ok(w_list_new(items))
            } else {
                Err(PyError::type_error(
                    "range indices must be integers or slices",
                ))
            }
        } else {
            Err(PyError::type_error(format!(
                "'{}' object is not subscriptable",
                (*(*obj).ob_type).name,
            )))
        }
    }
}

/// PyPy-compatible lookup that returns `None` instead of raising `KeyError`.
pub fn finditem(obj: PyObjectRef, index: PyObjectRef) -> Option<PyObjectRef> {
    match getitem(obj, index) {
        Ok(value) => Some(value),
        Err(err) => {
            if err.kind == crate::PyErrorKind::KeyError {
                None
            } else {
                panic!("space.finditem: unexpected {err:?}");
            }
        }
    }
}

/// Set item by index: `obj[index] = value`.

pub fn setitem(obj: PyObjectRef, index: PyObjectRef, value: PyObjectRef) -> PyResult {
    let obj = unwrap_cell(obj);
    let index = unwrap_cell(index);
    let value = unwrap_cell(value);
    unsafe {
        if is_list(obj) {
            if is_slice(index) {
                // list[start:stop] = value — slice assignment
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
                // Replace items[s..e] with the iterable value
                let new_items = crate::builtins::collect_iterable(value)?;
                let list_obj = &mut *(obj as *mut pyre_object::listobject::W_ListObject);
                let mut items = pyre_object::listobject::items_to_vec(list_obj);
                items.splice(s..e, new_items);
                pyre_object::listobject::rebuild_object_items(list_obj, items);
                return Ok(w_none());
            }
            if !is_int(index) {
                let tp = if index.is_null() {
                    "NULL"
                } else {
                    (*(*index).ob_type).name
                };
                return Err(PyError::type_error(format!(
                    "list indices must be integers, not {tp}"
                )));
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
        } else if pyre_object::bytearrayobject::is_bytearray(obj) {
            if is_int(index) {
                let idx = w_int_get_value(index);
                let len = pyre_object::bytearrayobject::w_bytearray_len(obj) as i64;
                let actual = if idx < 0 { len + idx } else { idx };
                if actual >= 0 && actual < len {
                    let val = w_int_get_value(value) as u8;
                    pyre_object::bytearrayobject::w_bytearray_setitem(obj, actual as usize, val);
                    return Ok(w_none());
                }
                return Err(PyError::new(
                    PyErrorKind::IndexError,
                    "bytearray index out of range",
                ));
            }
            Err(PyError::type_error("bytearray indices must be integers"))
        } else if is_instance(obj) {
            // PyPy: descroperation.py __setitem__
            if let Some(method) = lookup_in_type_where(w_instance_get_type(obj), "__setitem__") {
                crate::call_function(method, &[obj, index, value]);
                return Ok(w_none());
            }
            Err(PyError::type_error(format!(
                "'{}' object does not support item assignment",
                w_type_get_name(w_instance_get_type(obj)),
            )))
        } else {
            Err(PyError::type_error(format!(
                "'{}' object does not support item assignment",
                (*(*obj).ob_type).name,
            )))
        }
    }
}

/// PyPy-compatible string-keyed item lookup that returns `None` on miss.
pub fn finditem_str(obj: PyObjectRef, key: &str) -> Option<PyObjectRef> {
    finditem(obj, w_str_new(key))
}

/// PyPy-compatible identity check returning a raw boolean value.
pub fn is_w(w_one: PyObjectRef, w_two: PyObjectRef) -> bool {
    std::ptr::eq(w_one, w_two)
}

/// PyPy-compatible identity check returning a Python bool object.
pub fn is_(w_one: PyObjectRef, w_two: PyObjectRef) -> PyObjectRef {
    w_bool_from(is_w(w_one, w_two))
}

/// Python-level `not` operation.
pub fn not_(obj: PyObjectRef) -> PyObjectRef {
    w_bool_from(!is_true(obj))
}

/// PyPy-compatible attribute lookup returning `None` when not found.
pub fn findattr(obj: PyObjectRef, name: &str) -> Option<PyObjectRef> {
    if unsafe { is_none(obj) } {
        return None;
    }
    match getattr(obj, name) {
        Ok(value) => Some(value),
        Err(err) => {
            if err.kind == crate::PyErrorKind::AttributeError
                || err.kind == crate::PyErrorKind::NameError
            {
                None
            } else {
                panic!("space.findattr: unexpected {err:?}");
            }
        }
    }
}

/// Check whether `exc_type` matches `check_class`, including tuple/list class inputs.
pub fn exception_match(exc_type: PyObjectRef, check_class: PyObjectRef) -> bool {
    let (exc_type, check_class) = (exc_type, check_class);
    if unsafe { is_none(check_class) || is_none(exc_type) } {
        return false;
    }

    let is_tuple_check = unsafe { is_tuple(check_class) };
    if is_tuple_check {
        let len = unsafe { w_tuple_len(check_class) };
        for i in 0..len {
            let candidate = unsafe { w_tuple_getitem(check_class, i as i64) };
            if let Some(candidate) = candidate {
                if exception_match(exc_type, candidate) {
                    return true;
                }
            }
        }
        return false;
    }

    if unsafe { is_list(check_class) } {
        let len = unsafe { w_list_len(check_class) };
        for i in 0..len {
            let candidate = unsafe { w_list_getitem(check_class, i as i64) };
            if let Some(candidate) = candidate {
                if exception_match(exc_type, candidate) {
                    return true;
                }
            }
        }
        return false;
    }

    if !unsafe { is_type(check_class) } {
        return false;
    }

    if is_w(exc_type, check_class) {
        return true;
    }

    let mro_ptr = unsafe { w_type_get_mro(exc_type) };
    if mro_ptr.is_null() {
        return false;
    }

    let mro = unsafe { &*mro_ptr };
    mro.iter().any(|&klass| is_w(klass, check_class))
}

/// Get the length of a container: `len(obj)`.
pub fn len(obj: PyObjectRef) -> PyResult {
    unsafe {
        if is_list(obj) {
            Ok(w_int_new(w_list_len(obj) as i64))
        } else if is_tuple(obj) {
            Ok(w_int_new(w_tuple_len(obj) as i64))
        } else if is_dict(obj) {
            Ok(w_int_new(w_dict_len(obj) as i64))
        } else if is_str(obj) {
            Ok(w_int_new(w_str_len(obj) as i64))
        } else if pyre_object::bytearrayobject::is_bytearray(obj) {
            Ok(w_int_new(
                pyre_object::bytearrayobject::w_bytearray_len(obj) as i64,
            ))
        } else if is_instance(obj) {
            // Instance __len__ — PyPy: descroperation.py len
            if let Some(method) = lookup_in_type_where(w_instance_get_type(obj), "__len__") {
                return Ok(crate::call_function(method, &[obj]));
            }
            // Also check per-instance attributes (ATTR_TABLE)
            if let Ok(method) = getattr(obj, "__len__") {
                return Ok(crate::call_function(method, &[obj]));
            }
            Err(PyError::type_error(format!(
                "object of type '{}' has no len()",
                w_type_get_name(w_instance_get_type(obj)),
            )))
        } else {
            Err(PyError::type_error(format!(
                "object of type '{}' has no len()",
                (*(*obj).ob_type).name,
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

fn namespace_to_dict(ns_ptr: *const crate::PyNamespace) -> PyObjectRef {
    if ns_ptr.is_null() {
        return pyre_object::w_dict_new();
    }
    // Create a dict backed by the namespace so that dict.update() etc.
    // can sync changes back to the original namespace.
    let dict = pyre_object::dictobject::w_dict_new_with_namespace(ns_ptr as *mut u8);
    unsafe {
        for (key, &value) in (*ns_ptr).entries() {
            if !value.is_null() {
                pyre_object::w_dict_store(dict, w_str_new(key), value);
            }
        }
    }
    dict
}

/// Get an attribute from an object: `obj.name`.
///
/// For module objects, looks up the name in the module's namespace dict
/// (PyPy: Module.getdict → w_dict lookup).
/// For other objects, looks up the attribute in the per-object side table.

pub fn getattr(obj: PyObjectRef, name: &str) -> PyResult {
    let obj = unwrap_cell(obj);

    // super proxy — PyPy: superobject.py super_getattro
    // Looks up `name` in cls's MRO starting AFTER super_type.
    unsafe {
        if pyre_object::superobject::is_super(obj) {
            let super_type = pyre_object::superobject::w_super_get_type(obj);
            let bound_obj = pyre_object::superobject::w_super_get_obj(obj);

            // Walk obj's type MRO, skip until we pass super_type
            let w_obj_type = if is_instance(bound_obj) {
                w_instance_get_type(bound_obj)
            } else if is_type(bound_obj) {
                bound_obj
            } else {
                return Err(PyError::type_error("super: bad obj type"));
            };
            let mro_ptr = w_type_get_mro(w_obj_type);
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
                        if let Some(method) = lookup_in_type_where(t, name) {
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
                    return Ok(crate::make_builtin_function("gen_method", gen_stub_method));
                }
                _ => {}
            }
        }
    }

    // Property descriptor methods — PyPy: descriptor.py W_Property.setter / getter / deleter
    // Returns a bound method (W_Method) that captures the property via w_self,
    // so the static handler can extract the property from args[0].
    unsafe {
        if is_property(obj) {
            let static_name: Option<(
                &'static str,
                fn(&[PyObjectRef]) -> Result<PyObjectRef, crate::PyError>,
            )> = match name {
                "setter" => Some(("setter", property_setter_impl)),
                "getter" => Some(("getter", property_getter_impl)),
                "deleter" => Some(("deleter", property_deleter_impl)),
                _ => None,
            };
            if let Some((sname, func)) = static_name {
                let builtin = crate::make_builtin_function(sname, func);
                return Ok(pyre_object::methodobject::w_method_new(
                    builtin,
                    obj,
                    pyre_object::PY_NULL,
                ));
            }
            match name {
                "fget" => return Ok(w_property_get_fget(obj)),
                "fset" => return Ok(w_property_get_fset(obj)),
                "fdel" => return Ok(w_property_get_fdel(obj)),
                _ => {}
            }
        }
    }

    // Module objects: look up in module namespace
    // PyPy: space.getattr(w_module, w_name) → Module.getdictvalue(space, name)
    unsafe {
        if is_module(obj) {
            if name == "__dict__" {
                let ns_ptr = w_module_get_dict_ptr(obj) as *const crate::PyNamespace;
                return Ok(namespace_to_dict(ns_ptr));
            }
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
            let w_descr = lookup_in_type_where(w_type, name);

            // Step 2: data descriptor takes priority over instance dict
            if let Some(descr) = w_descr {
                if is_data_descr(descr) {
                    if let Some(result) = get(descr, obj, w_type) {
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
            // PyPy: descroperation.py — invoke __get__ to bind methods
            if let Some(descr) = w_descr {
                if let Some(result) = get(descr, obj, w_type) {
                    return Ok(result);
                }
                // Step 5: builtin methods found in base type MRO need binding
                // CPython: PyFunction_GET_CODE slot → bound method
                if crate::is_function(descr)
                    && !crate::is_builtin_code(
                        crate::function_get_code(descr) as pyre_object::PyObjectRef
                    )
                {
                    return Ok(pyre_object::w_method_new(descr, obj, w_type));
                }
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
                if let Some(qn) = lookup_in_type_where(obj, "__qualname__") {
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
                return Ok(namespace_to_dict(dict_ptr));
            }
            if name == "__bases__" {
                return Ok(w_type_get_bases(obj));
            }
            if name == "__doc__"
                || name == "__module__"
                || name == "__abstractmethods__"
                || name == "__flags__"
                || name == "__code__"
                || name == "__func__"
                || name == "__self__"
                || name == "__wrapped__"
                || name == "__annotations__"
                || name == "__globals__"
                || name == "__closure__"
                || name == "__defaults__"
                || name == "__kwdefaults__"
            {
                // Check class dict first, then return None
                if let Some(v) = lookup_in_type_where(obj, name) {
                    return Ok(v);
                }
                return Ok(w_none());
            }

            if let Some(value) = lookup_in_type_where(obj, name) {
                if let Some(result) = get(value, PY_NULL, obj) {
                    return Ok(result);
                }
                return Ok(value);
            }
            // Metaclass attribute lookup — PyPy: type.__getattribute__
            // baseobjspace.py:76 — the metaclass is type(C), read from w_class.
            let w_metaclass = {
                let w_class = (*obj).w_class;
                let w_type_type = crate::typedef::w_type();
                if !w_class.is_null() && !std::ptr::eq(w_class, w_type_type) {
                    Some(w_class)
                } else {
                    None
                }
            };
            // PyPy: type.__getattribute__ → metatype descriptor protocol.
            // Search metaclass MRO. Binding is handled by load_method.
            let w_metaclasses: [Option<PyObjectRef>; 2] =
                [w_metaclass, crate::typedef::gettypefor((*obj).ob_type)];
            for w_metaclass in w_metaclasses.iter().flatten() {
                let w_metaclass = *w_metaclass;
                if is_type(w_metaclass) {
                    if let Some(value) = lookup_in_type_where(w_metaclass, name) {
                        if let Some(result) = get(value, obj, w_metaclass) {
                            return Ok(result);
                        }
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
    if let Some(w_type) = crate::typedef::r#type(obj) {
        if let Some(method) = unsafe { lookup_in_type_where(w_type, name) } {
            if unsafe { crate::is_function(method) } {
                return Ok(pyre_object::w_method_new(method, obj, w_type));
            }
            if let Some(result) = unsafe { get(method, obj, w_type) } {
                return Ok(result);
            }
            return Ok(method);
        }
    }

    // Function object attributes — PyPy: funcobject.py W_Function
    // Check ATTR_TABLE first (for dynamically set attrs like __name__, __doc__)
    if unsafe { crate::is_function(obj) } {
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
        if crate::is_function(obj) {
            match name {
                "__code__" => {
                    // function_get_code returns Code-level pointer (W_CodeObject or BuiltinCode)
                    let code = crate::function_get_code(obj) as PyObjectRef;
                    if code.is_null() {
                        return Ok(w_none());
                    }
                    return Ok(code);
                }
                "__name__" => {
                    return Ok(w_str_new(crate::function_get_name(obj)));
                }
                "__closure__" => {
                    let closure = crate::function_get_closure(obj);
                    return Ok(if closure.is_null() { w_none() } else { closure });
                }
                "__globals__" => {
                    return Ok(namespace_to_dict(crate::function_get_globals(obj)));
                }
                "__defaults__" => {
                    let defaults = crate::function_get_defaults(obj);
                    return Ok(if defaults.is_null() {
                        w_none()
                    } else {
                        defaults
                    });
                }
                "__kwdefaults__" => {
                    let kwdefaults = crate::function_get_kwdefaults(obj);
                    return Ok(if kwdefaults.is_null() {
                        w_none()
                    } else {
                        kwdefaults
                    });
                }
                "__qualname__" => {
                    let found = ATTR_TABLE.with(|table| {
                        table
                            .borrow()
                            .get(&(obj as usize))
                            .and_then(|d| d.get(name).copied())
                    });
                    if let Some(value) = found {
                        return Ok(value);
                    }
                    let code = crate::function_get_code(obj) as PyObjectRef;
                    if !code.is_null() && crate::pycode::is_code(code) {
                        let raw_code_ptr =
                            crate::pycode::w_code_get_ptr(code) as *const crate::CodeObject;
                        if !raw_code_ptr.is_null() {
                            return Ok(w_str_new((*raw_code_ptr).qualname.as_ref()));
                        }
                    }
                    return Ok(w_str_new(crate::function_get_name(obj)));
                }
                _ => {}
            }
        }
        // staticmethod/classmethod descriptor attributes
        // PyPy: function.py StaticMethod.__func__, ClassMethod.__func__
        if pyre_object::is_staticmethod(obj) {
            if name == "__func__" || name == "__wrapped__" {
                return Ok(pyre_object::w_staticmethod_get_func(obj));
            }
        }
        if pyre_object::is_classmethod(obj) {
            if name == "__func__" || name == "__wrapped__" {
                return Ok(pyre_object::w_classmethod_get_func(obj));
            }
        }
        if crate::pycode::is_code(obj) {
            let code_ptr = crate::pycode::w_code_get_ptr(obj) as *const crate::CodeObject;
            if code_ptr.is_null() {
                return Ok(w_none());
            }
            let code = &*code_ptr;
            match name {
                "co_varnames" => {
                    let items = code
                        .varnames
                        .iter()
                        .map(|item| w_str_new(item.as_ref()))
                        .collect();
                    return Ok(w_tuple_new(items));
                }
                "co_argcount" => return Ok(w_int_new(code.arg_count as i64)),
                "co_kwonlyargcount" => return Ok(w_int_new(code.kwonlyarg_count as i64)),
                "co_name" => return Ok(w_str_new(code.obj_name.as_ref())),
                "co_filename" => return Ok(w_str_new(code.source_path.as_ref())),
                "co_flags" => return Ok(w_int_new(code.flags.bits() as i64)),
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
    // Exception attributes — PyPy: W_BaseException attributes
    if unsafe { pyre_object::is_exception(obj) } {
        match name {
            "__traceback__" => {
                // Stub traceback object with tb_frame, tb_lineno, tb_next
                let tb = pyre_object::w_instance_new(crate::typedef::w_object());
                let frame_obj = pyre_object::w_instance_new(crate::typedef::w_object());
                ATTR_TABLE.with(|t| {
                    let mut t = t.borrow_mut();
                    let fd = t.entry(frame_obj as usize).or_default();
                    fd.insert("f_locals".into(), w_dict_new());
                    fd.insert("f_globals".into(), w_dict_new());
                    fd.insert("f_code".into(), w_none());
                    fd.insert("f_lineno".into(), w_int_new(0));
                    let td = t.entry(tb as usize).or_default();
                    td.insert("tb_frame".into(), frame_obj);
                    td.insert("tb_lineno".into(), w_int_new(0));
                    td.insert("tb_next".into(), w_none());
                });
                return Ok(tb);
            }
            "__cause__" | "__context__" | "__suppress_context__" => {
                return Ok(w_none());
            }
            "args" => {
                return Ok(w_tuple_new(vec![]));
            }
            _ => {}
        }
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
    // __class__: read directly from w_class field (the single source of truth).
    // objectobject.py:133-134 descr_get___class__ → space.type(w_obj)
    if name == "__class__" {
        if let Some(tp) = crate::typedef::r#type(obj) {
            return Ok(tp);
        }
    }

    // Instance attributes from side table (excludes __class__ which lives
    // in the w_class header field, not ATTR_TABLE).
    let found = ATTR_TABLE.with(|table| {
        let table = table.borrow();
        let key = obj as usize;
        table.get(&key).and_then(|dict| dict.get(name).copied())
    });
    if let Some(value) = found {
        return Ok(value);
    }

    // MRO lookup on the object's Python class (w_class) for method resolution.
    let w_class = unsafe { (*obj).w_class };
    if !w_class.is_null() && unsafe { is_type(w_class) } {
        if let Some(method) = unsafe { lookup_in_type_where(w_class, name) } {
            if unsafe {
                crate::is_function(method)
                    && !crate::is_builtin_code(
                        crate::function_get_code(method) as pyre_object::PyObjectRef
                    )
            } {
                return Ok(pyre_object::w_method_new(method, obj, w_class));
            }
            if let Some(result) = unsafe { get(method, obj, w_class) } {
                return Ok(result);
            }
            return Ok(method);
        }
    }

    unsafe {
        let tp_name = if obj.is_null() {
            "NULL"
        } else {
            (*(*obj).ob_type).name
        };
        Err(PyError::new(
            PyErrorKind::AttributeError,
            format!("'{tp_name}' object has no attribute '{name}'"),
        ))
    }
}

// Builtin type method implementations moved to type_methods.rs
// (PyPy: listobject.py, unicodeobject.py, dictobject.py, tupleobject.py)

/// Look up a descriptor on an object's type.
///
/// PyPy equivalent: `space.lookup(w_obj, name)`.
pub unsafe fn lookup(obj: PyObjectRef, name: &str) -> Option<PyObjectRef> {
    let w_type = crate::typedef::r#type(obj)?;
    lookup_in_type(w_type, name)
}

/// Look up a name on a type by walking the C3 MRO.
///
/// PyPy equivalent: `space.lookup_in_type(w_type, name)`.
pub unsafe fn lookup_in_type(w_type: PyObjectRef, name: &str) -> Option<PyObjectRef> {
    lookup_in_type_where(w_type, name)
}

/// typeobject.py `_lookup_where(self, key)` — linear search through `self.mro_w`.
/// NOTE: PyPy's elidable wrapper (_pure_lookup_where_with_method_cache) takes
/// a version_tag argument to invalidate on type mutation. Until pyre has
/// version tags, this raw lookup must NOT be marked elidable.
unsafe fn lookup_in_type_where(w_type: PyObjectRef, name: &str) -> Option<PyObjectRef> {
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
        if (*cls).is_null() || !is_type(*cls) {
            continue;
        }
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

/// Determine what `self` value to bind for a super-resolved attribute.
///
/// Walks the MRO of `self_obj` starting after `super_type`, finds the
/// raw descriptor for `name`, and returns:
///   - PY_NULL       if it is a staticmethod (no binding)
///   - the class obj if it is a classmethod  (bind class)
///   - `self_obj`    otherwise                (bind instance)
pub unsafe fn super_lookup_binding(
    super_type: PyObjectRef,
    self_obj: PyObjectRef,
    name: &str,
) -> PyObjectRef {
    use pyre_object::*;
    let w_obj_type = if is_instance(self_obj) {
        w_instance_get_type(self_obj)
    } else if is_type(self_obj) {
        self_obj
    } else {
        return self_obj;
    };
    let mro_ptr = w_type_get_mro(w_obj_type);
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
                if let Some(raw) = lookup_in_type_where(t, name) {
                    if is_staticmethod(raw) {
                        return PY_NULL;
                    }
                    if is_classmethod(raw) {
                        return w_obj_type;
                    }
                    // `__new__` is implicitly static (type.__new__ is a
                    // builtin_function_or_method, not a Python function)
                    if name == "__new__" {
                        return PY_NULL;
                    }
                    return self_obj;
                }
            }
        }
    }
    self_obj
}

/// C3 linearization — PyPy: typeobject.py `compute_default_mro`.
///
/// Computes the Method Resolution Order for a type following the C3
/// algorithm (Python 2.3+). Handles diamond inheritance correctly.
///
/// Public wrapper for use by isinstance and other external callers.
pub unsafe fn compute_default_mro(w_type: PyObjectRef) -> Vec<PyObjectRef> {
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
unsafe fn is_data_descr(descr: PyObjectRef) -> bool {
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
            return lookup_in_type_where(w_type, "__set__").is_some()
                || lookup_in_type_where(w_type, "__delete__").is_some();
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
unsafe fn get(descr: PyObjectRef, obj: PyObjectRef, w_type: PyObjectRef) -> Option<PyObjectRef> {
    if descr.is_null() {
        return None;
    }

    // PyPy removes __get__ from BuiltinFunction.typedef. Builtin functions
    // stored on user classes therefore stay plain callables instead of
    // binding `self` like Python functions do.
    if crate::is_function(descr)
        && crate::is_builtin_code(crate::function_get_code(descr) as pyre_object::PyObjectRef)
    {
        return None;
    }

    // property: PyPy W_Property.get → call fget(obj)
    if is_property(descr) {
        if obj.is_null() {
            return Some(descr);
        }
        let fget = w_property_get_fget(descr);
        if fget.is_null() || is_none(fget) {
            return None;
        }
        return Some(crate::call_function(fget, &[obj]));
    }

    // staticmethod: PyPy StaticMethod.descr_staticmethod_get → return w_function
    if is_staticmethod(descr) {
        return Some(w_staticmethod_get_func(descr));
    }

    // classmethod: PyPy ClassMethod.descr_classmethod_get → func bound to class
    if is_classmethod(descr) {
        let func = w_classmethod_get_func(descr);
        let receiver = if !w_type.is_null() { w_type } else { obj };
        if receiver.is_null() || is_none(receiver) {
            return Some(func);
        }
        let owner = crate::typedef::r#type(receiver).unwrap_or(PY_NULL);
        return Some(pyre_object::w_method_new(func, receiver, owner));
    }

    // General __get__: look up __get__ on the descriptor's own type MRO
    // PyPy: descroperation.py → space.get_and_call_function(w_get, descr, obj, type)
    if let Some(descr_type) = crate::typedef::r#type(descr) {
        if let Some(get_fn) = lookup_in_type_where(descr_type, "__get__") {
            if !get_fn.is_null() {
                // Call __get__(descr, obj, type) via ObjSpace.call_function
                return Some(crate::call_function(get_fn, &[descr, obj, w_type]));
            }
        }
    }
    None
}

/// Call a descriptor's __set__ method.
///
/// PyPy: descroperation.py `descr__setattr__` →
/// `space.get_and_call_function(w_set, w_descr, w_obj, w_value)`
unsafe fn set(descr: PyObjectRef, obj: PyObjectRef, value: PyObjectRef) -> bool {
    if descr.is_null() {
        return false;
    }

    // property: PyPy W_Property.set → call_function(fset, obj, value)
    if is_property(descr) {
        let fset = w_property_get_fset(descr);
        if fset.is_null() || is_none(fset) {
            return false;
        }
        crate::call_function(fset, &[obj, value]);
        return true;
    }

    // General __set__: look up on descriptor's type MRO
    if is_instance(descr) {
        let descr_type = w_instance_get_type(descr);
        if let Some(set_fn) = lookup_in_type_where(descr_type, "__set__") {
            if !set_fn.is_null() {
                crate::call_function(set_fn, &[obj, value]);
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

/// objectobject.py:137-154 `descr_set___class__(space, w_obj, w_newcls)`.
///
/// Validates and performs `obj.__class__ = newcls`.
fn descr_set___class__(w_obj: PyObjectRef, w_newcls: PyObjectRef) -> PyResult {
    unsafe {
        // objectobject.py:139-142 — w_newcls must be a W_TypeObject
        if !is_type(w_newcls) {
            return Err(crate::PyError::type_error(format!(
                "__class__ must be set to new-style class, not '{}' object",
                (*(*w_newcls).ob_type).name,
            )));
        }
        // objectobject.py:143-145 — w_newcls must be a heap type.
        // In pyre, heap types are user-defined classes created by
        // class statement / type(). Built-in types (int, str, etc.)
        // are NOT heap types.
        if !is_heaptype(w_newcls) {
            return Err(crate::PyError::type_error(
                "__class__ assignment: only for heap types".to_string(),
            ));
        }
        // objectobject.py:146-147 — get the old class
        let w_oldcls = match crate::typedef::r#type(w_obj) {
            Some(c) => c,
            None => {
                return Err(crate::PyError::type_error(
                    "__class__ assignment: cannot determine current class".to_string(),
                ));
            }
        };
        // objectobject.py:148-154 — full instance layout must match.
        // In pyre, layout compatibility means both old and new class
        // must be user-defined (INSTANCE_TYPE) instances. Built-in
        // types have incompatible C layouts (W_IntObject vs W_ListObject).
        let obj_ob_type = (*w_obj).ob_type;
        if !std::ptr::eq(obj_ob_type, &INSTANCE_TYPE) {
            return Err(crate::PyError::type_error(format!(
                "__class__ assignment: '{}' object layout differs from '{}'",
                (*obj_ob_type).name,
                pyre_object::w_type_get_name(w_newcls),
            )));
        }
        // objectobject.py:150 — w_obj.setclass(space, w_newcls)
        (*w_obj).w_class = w_newcls;
    }
    Ok(w_none())
}

/// typeobject.py — check if a type is a heap type (user-defined class).
///
/// Heap types are dynamically created by `class` statement or `type()`.
/// Built-in types (int, str, list, etc.) are NOT heap types.
/// In pyre, heap types have `ob_type == TYPE_TYPE` and were created
/// by `w_type_new()` (not by init_typeobjects static definitions).
#[inline]
fn is_heaptype(w_type: PyObjectRef) -> bool {
    // A type is a heap type if it was created dynamically.
    // Built-in types created by init_typeobjects have their w_class
    // set to the 'type' type object but are conceptually not heap types.
    // For now, all user-defined classes are heap types, and built-in
    // types are not. We distinguish by checking if the type appears
    // in the TYPEOBJECT_CACHE (built-in → not heap).
    let addr = w_type as usize;
    let is_builtin = crate::typedef::TYPEOBJECT_CACHE
        .get()
        .map_or(false, |cache| cache.values().any(|&v| v == addr));
    !is_builtin
}

pub fn setattr(obj: PyObjectRef, name: &str, value: PyObjectRef) -> PyResult {
    let obj = unwrap_cell(obj);
    let value = unwrap_cell(value);
    // Module objects: store directly in the module namespace so `sys.ps1 = ...`
    // and similar interactive mutations are visible through module getattr/import.
    unsafe {
        if is_module(obj) {
            let ns_ptr = w_module_get_dict_ptr(obj) as *mut crate::PyNamespace;
            if !ns_ptr.is_null() {
                crate::namespace_store(&mut *ns_ptr, name, value);
                return Ok(w_none());
            }
        }
    }
    // Data descriptor __set__ takes priority (PyPy: descr__setattr__ step 1)
    unsafe {
        if is_instance(obj) {
            let w_type = w_instance_get_type(obj);
            if let Some(descr) = lookup_in_type_where(w_type, name) {
                if set(descr, obj, value) {
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
    // objectobject.py:137-154 descr_set___class__
    if name == "__class__" {
        return descr_set___class__(obj, value);
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
pub fn delattr(obj: PyObjectRef, name: &str) -> PyResult {
    let obj = unwrap_cell(obj);
    unsafe {
        if is_module(obj) {
            let ns_ptr = w_module_get_dict_ptr(obj) as *mut crate::PyNamespace;
            if !ns_ptr.is_null() {
                crate::namespace_store(&mut *ns_ptr, name, PY_NULL);
                return Ok(w_none());
            }
        }
    }
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
        let tp_name = unsafe { (*(*obj).ob_type).name };
        Err(PyError::new(
            PyErrorKind::AttributeError,
            format!("'{tp_name}' object has no attribute '{name}'"),
        ))
    }
}

/// PyPy: baseobjspace.py `call`.
///
/// Call a Python callable with packed positional arguments and optional kwargs.
pub fn call(
    callable: PyObjectRef,
    w_args: PyObjectRef,
    w_kwds: Option<PyObjectRef>,
) -> PyObjectRef {
    if let Some(w_kwargs) = w_kwds {
        if !w_kwargs.is_null() && !unsafe { is_none(w_kwargs) } {
            panic!("call with kwargs is not yet implemented in pyre");
        }
    }

    let mut args = Vec::new();
    unsafe {
        if is_tuple(w_args) {
            let len = w_tuple_len(w_args);
            args.reserve(len);
            for i in 0..len {
                if let Some(arg) = w_tuple_getitem(w_args, i as i64) {
                    args.push(arg);
                }
            }
        } else if is_list(w_args) {
            let len = w_list_len(w_args);
            args.reserve(len);
            for i in 0..len {
                if let Some(arg) = w_list_getitem(w_args, i as i64) {
                    args.push(arg);
                }
            }
        } else if !w_args.is_null() {
            panic!("call() expects tuple or list positional arguments");
        }
    }
    call_function(callable, &args)
}

/// PyPy: baseobjspace.py `call_obj_args` — add a leading object before args.
pub fn call_obj_args(callable: PyObjectRef, obj: PyObjectRef, args: &[PyObjectRef]) -> PyObjectRef {
    if obj.is_null() {
        return call_function(callable, args);
    }
    let mut call_args = Vec::with_capacity(1 + args.len());
    call_args.push(obj);
    call_args.extend_from_slice(args);
    call_function(callable, &call_args)
}

/// PyPy: baseobjspace.py `call_valuestack`.
pub fn call_valuestack(
    callable: PyObjectRef,
    nargs: usize,
    frame: &mut crate::pyframe::PyFrame,
    dropvalues: usize,
    methodcall: bool,
) -> PyObjectRef {
    let mut args = Vec::with_capacity(nargs);
    for _ in 0..nargs {
        args.push(frame.pop());
    }
    args.reverse();

    let mut remaining_to_drop = dropvalues.saturating_sub(nargs);

    let null_or_self = if methodcall {
        let value = if remaining_to_drop > 0 {
            remaining_to_drop -= 1;
            Some(frame.pop())
        } else {
            None
        };
        if remaining_to_drop > 0 {
            frame.pop();
            remaining_to_drop -= 1;
        }
        value
    } else {
        if remaining_to_drop > 0 {
            frame.pop();
            remaining_to_drop -= 1;
        }
        None
    };

    for _ in 0..remaining_to_drop {
        frame.pop();
    }

    if let Some(null_or_self) = null_or_self {
        if !null_or_self.is_null() && !unsafe { is_none(null_or_self) } {
            args.insert(0, null_or_self);
        }
    }
    call_function(callable, &args)
}

/// PyPy: baseobjspace.py `call_args_and_c_profile`.
pub fn call_args_and_c_profile(
    _frame: &mut crate::pyframe::PyFrame,
    callable: PyObjectRef,
    args: &[PyObjectRef],
) -> PyObjectRef {
    call_function(callable, args)
}

/// PyPy: baseobjspace.py `call_method`.
pub fn call_method(obj: PyObjectRef, methname: &str, args: &[PyObjectRef]) -> PyObjectRef {
    let method =
        getattr(obj, methname).unwrap_or_else(|e| panic!("call_method({methname}) failed: {e}"));
    call_function(method, args)
}

/// PyPy: baseobjspace.py `call_function`.
///
/// Dispatches to builtins, user functions, and type objects.
pub fn call_function(callable: PyObjectRef, args: &[PyObjectRef]) -> PyObjectRef {
    crate::call::call_function_impl(callable, args)
}

/// PyPy: baseobjspace.py `callable_w`.
pub fn callable_w(obj: PyObjectRef) -> bool {
    unsafe {
        is_function(obj)
            || is_type(obj)
            || (is_instance(obj) && lookup_in_type(w_instance_get_type(obj), "__call__").is_some())
    }
}

/// PyPy: baseobjspace.py `callable`.
pub fn callable(obj: PyObjectRef) -> PyObjectRef {
    if callable_w(obj) {
        w_bool_from(true)
    } else {
        w_bool_from(false)
    }
}

/// PyPy `ObjSpace.call_function_or_identity`.
pub fn call_function_or_identity(obj: PyObjectRef, dunder: &str) -> PyObjectRef {
    unsafe {
        if is_instance(obj) {
            if let Some(method) = lookup(obj, dunder) {
                return call_function(method, &[obj]);
            }
        }
    }
    obj
}

/// PyPy baseobjspace.py equivalent.
pub fn get_printable_location(greenkey: PyObjectRef) -> String {
    format!("unpackiterable [{:?}]", greenkey)
}

/// PyPy baseobjspace.py equivalent.
pub fn wrappable_class_name(class: PyObjectRef) -> String {
    if class.is_null() {
        return "internal subclass".to_string();
    }
    unsafe {
        let type_name = (*(*class).ob_type).name;
        if is_type(class) {
            type_name.to_string()
        } else {
            format!("internal subclass of {type_name}")
        }
    }
}

/// `iter(obj)` — PyPy: space.iter(w_obj)
/// Calls __iter__ on the object if available.
pub fn iter(obj: PyObjectRef) -> PyResult {
    let obj = unwrap_cell(obj);
    if obj.is_null() {
        return Err(PyError::type_error("'NoneType' object is not iterable"));
    }
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
        if pyre_object::bytearrayobject::is_bytearray(obj) {
            let len = pyre_object::bytearrayobject::w_bytearray_len(obj);
            // Convert to list of ints for iteration
            let mut items = Vec::with_capacity(len);
            for i in 0..len {
                items.push(w_int_new(
                    pyre_object::bytearrayobject::w_bytearray_getitem(obj, i) as i64,
                ));
            }
            let list = pyre_object::w_list_new(items);
            return Ok(pyre_object::w_seq_iter_new(list, len));
        }
        // dict → iterate over keys (dictobject.py __iter__)
        if is_dict(obj) {
            let d = &*(obj as *const pyre_object::dictobject::W_DictObject);
            let entries = &*d.entries;
            let keys: Vec<PyObjectRef> = entries.iter().map(|&(k, _)| k).collect();
            let key_list = pyre_object::w_list_new(keys);
            let len = entries.len();
            return Ok(pyre_object::w_seq_iter_new(key_list, len));
        }
        // Already an iterator
        if is_range_iter(obj) || is_seq_iter(obj) || pyre_object::generatorobject::is_generator(obj)
        {
            return Ok(obj);
        }
        // Instance __iter__ — check type MRO and ATTR_TABLE
        if is_instance(obj) {
            // Check type MRO first (PyPy: type(obj).__iter__)
            let w_type = w_instance_get_type(obj);
            if let Some(method) = lookup_in_type_where(w_type, "__iter__") {
                return Ok(crate::call_function(method, &[obj]));
            }
            // Per-instance __iter__ in ATTR_TABLE
            let inst_iter = ATTR_TABLE.with(|t| {
                t.borrow()
                    .get(&(obj as usize))
                    .and_then(|d| d.get("__iter__").copied())
            });
            if let Some(method) = inst_iter {
                return Ok(crate::call_function(method, &[obj]));
            }
            // Fallback: __getitem__ protocol — if object has __getitem__,
            // iterate by calling __getitem__(0), __getitem__(1), ...
            // until IndexError is raised. Use __len__ to determine bound.
            let w_type = w_instance_get_type(obj);
            if lookup_in_type_where(w_type, "__getitem__").is_some()
                || getattr(obj, "__getitem__").is_ok()
            {
                // Try to use __len__ to bound the iteration.
                let mut items = Vec::new();
                if let Ok(len_result) = len(obj) {
                    if is_int(len_result) {
                        let n = w_int_get_value(len_result);
                        for i in 0..n {
                            match getitem(obj, w_int_new(i)) {
                                Ok(item) => items.push(item),
                                Err(_) => break,
                            }
                        }
                        let count = items.len();
                        let list = w_list_new(items);
                        return Ok(pyre_object::w_seq_iter_new(list, count));
                    }
                }
                // No __len__: iterate up to a reasonable bound, breaking on
                // any error (PyPy: descroperation iter_via_getitem with sentinel).
                for i in 0..1_000_000i64 {
                    match getitem(obj, w_int_new(i)) {
                        Ok(item) => items.push(item),
                        Err(_) => break,
                    }
                }
                let count = items.len();
                let list = w_list_new(items);
                return Ok(pyre_object::w_seq_iter_new(list, count));
            }
        }
        // Type object: check metaclass __iter__ (NOT the type's own MRO)
        // PyPy/CPython: iter(X) calls type(X).__iter__(X), not X.__iter__
        // For type objects, type(X) is the metaclass.
        if is_type(obj) {
            // baseobjspace.py:76 — metaclass from w_class
            let w_metaclass = unsafe {
                let w_class = (*obj).w_class;
                let w_type_type = crate::typedef::w_type();
                if !w_class.is_null() && !std::ptr::eq(w_class, w_type_type) {
                    Some(w_class)
                } else {
                    None
                }
            };
            if let Some(w_metaclass) = w_metaclass {
                if let Some(method) = lookup_in_type_where(w_metaclass, "__iter__") {
                    return Ok(crate::call_function(method, &[obj]));
                }
            }
            // Fallback: check type type's MRO
            if let Some(w_type_type) = crate::typedef::gettypefor(&pyre_object::pyobject::TYPE_TYPE)
            {
                if let Some(method) = lookup_in_type_where(w_type_type, "__iter__") {
                    return Ok(crate::call_function(method, &[obj]));
                }
            }
        }
    }
    Err(PyError::type_error(format!(
        "'{}' object is not iterable",
        unsafe { (*(*obj).ob_type).name }
    )))
}

/// `next(iterator)` — PyPy: space.next(w_iter)
pub fn next(obj: PyObjectRef) -> PyResult {
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
            let has_next = if iter.step > 0 {
                iter.current < iter.stop
            } else if iter.step < 0 {
                iter.current > iter.stop
            } else {
                false
            };
            if has_next {
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
            if let Some(method) = lookup_in_type_where(w_type, "__next__") {
                return Ok(crate::call_function(method, &[obj]));
            }
        }
    }
    Err(PyError::type_error("not an iterator"))
}

/// Stub method for generator.close/send/throw — no-op.
/// Property setter/getter/deleter helpers — PyPy: W_Property.setter/getter/deleter.
/// args[0] is the owning property (bound via W_Method), args[1] is the new fn.
fn property_setter_impl(args: &[PyObjectRef]) -> PyResult {
    let prop = args[0];
    let new_fn = args.get(1).copied().unwrap_or(pyre_object::PY_NULL);
    unsafe {
        let fget = w_property_get_fget(prop);
        let fdel = w_property_get_fdel(prop);
        Ok(pyre_object::w_property_new(fget, new_fn, fdel))
    }
}

fn property_getter_impl(args: &[PyObjectRef]) -> PyResult {
    let prop = args[0];
    let new_fn = args.get(1).copied().unwrap_or(pyre_object::PY_NULL);
    unsafe {
        let fset = w_property_get_fset(prop);
        let fdel = w_property_get_fdel(prop);
        Ok(pyre_object::w_property_new(new_fn, fset, fdel))
    }
}

fn property_deleter_impl(args: &[PyObjectRef]) -> PyResult {
    let prop = args[0];
    let new_fn = args.get(1).copied().unwrap_or(pyre_object::PY_NULL);
    unsafe {
        let fget = w_property_get_fget(prop);
        let fset = w_property_get_fset(prop);
        Ok(pyre_object::w_property_new(fget, fset, new_fn))
    }
}

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
        let frame_ptr = w_generator_get_frame(gen_obj) as *mut crate::pyframe::PyFrame;
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
                // Distinguish yield vs return: if the frame is at a YIELD_VALUE
                // instruction (pc-1), it's a yield. Otherwise it's a return.
                let code = &*crate::pyframe_get_pycode(&*frame);
                let pc = frame.next_instr;
                let is_yield = if pc > 0 && pc <= code.instructions.len() {
                    matches!(
                        code.instructions[pc - 1].op,
                        crate::Instruction::YieldValue { .. }
                    )
                } else {
                    false
                };
                if is_yield {
                    Ok(value)
                } else {
                    w_generator_set_exhausted(gen_obj);
                    Err(PyError {
                        kind: PyErrorKind::StopIteration,
                        message: "".to_string(),
                        exc_object: std::ptr::null_mut(),
                    })
                }
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
        let result = add(a, b).unwrap();
        unsafe { assert_eq!(w_int_get_value(result), 7) };
    }

    #[test]
    fn test_int_compare() {
        let a = w_int_new(5);
        let b = w_int_new(10);
        let result = compare(a, b, CompareOp::Lt).unwrap();
        unsafe { assert!(w_bool_get_value(result)) };
    }

    #[test]
    fn test_zero_division() {
        let a = w_int_new(5);
        let b = w_int_new(0);
        assert!(floordiv(a, b).is_err());
    }

    #[test]
    fn test_truthiness() {
        assert!(is_true(w_int_new(1)));
        assert!(!is_true(w_int_new(0)));
        assert!(!is_true(w_none()));
        assert!(is_true(w_bool_from(true)));
        assert!(!is_true(w_bool_from(false)));
    }

    #[test]
    fn test_int_add_overflow() {
        let a = w_int_new(i64::MAX);
        let b = w_int_new(1);
        let result = add(a, b).unwrap();
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
        let result = sub(a, b).unwrap();
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
        let result = mul(a, b).unwrap();
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
        let result = add(a, b).unwrap();
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
        let result = add(a, b).unwrap();
        unsafe {
            assert!(is_int(result));
            assert_eq!(w_int_get_value(result), i64::MAX);
        }
    }

    #[test]
    fn test_negate_min_int() {
        let a = w_int_new(i64::MIN);
        let result = neg(a).unwrap();
        unsafe {
            assert!(is_long(result));
            assert_eq!(*w_long_get_value(result), -BigInt::from(i64::MIN));
        }
    }

    #[test]
    fn test_invert_int() {
        let result = invert(w_int_new(6)).unwrap();
        unsafe {
            assert!(is_int(result));
            assert_eq!(w_int_get_value(result), !6);
        }
    }

    #[test]
    fn test_long_compare() {
        let a = w_long_new(BigInt::from(i64::MAX) + BigInt::from(1));
        let b = w_int_new(i64::MAX);
        let result = compare(a, b, CompareOp::Gt).unwrap();
        unsafe { assert!(w_bool_get_value(result)) };
    }

    #[test]
    fn test_long_truthiness() {
        assert!(is_true(w_long_new(
            BigInt::from(i64::MAX) + BigInt::from(1)
        )));
        assert!(!is_true(w_long_new(BigInt::from(0))));
    }

    #[test]
    fn test_int_pow() {
        let result = pow(w_int_new(2), w_int_new(10)).unwrap();
        unsafe { assert_eq!(w_int_get_value(result), 1024) };
    }

    #[test]
    fn test_int_pow_overflow() {
        let result = pow(w_int_new(2), w_int_new(63)).unwrap();
        unsafe {
            // 2^63 overflows i64, should be long
            assert!(is_long(result));
            assert_eq!(*w_long_get_value(result), BigInt::from(2).pow(63));
        }
    }

    #[test]
    fn test_int_pow_negative_exponent() {
        let result = pow(w_int_new(2), w_int_new(-1)).unwrap();
        unsafe {
            assert!(is_float(result));
            assert_eq!(w_float_get_value(result), 0.5);
        }
    }

    #[test]
    fn test_int_lshift() {
        let result = lshift(w_int_new(1), w_int_new(10)).unwrap();
        unsafe { assert_eq!(w_int_get_value(result), 1024) };
    }

    #[test]
    fn test_int_lshift_overflow() {
        let result = lshift(w_int_new(1), w_int_new(64)).unwrap();
        unsafe {
            assert!(is_long(result));
            assert_eq!(*w_long_get_value(result), BigInt::from(1) << 64);
        }
    }

    #[test]
    fn test_int_rshift() {
        let result = rshift(w_int_new(1024), w_int_new(3)).unwrap();
        unsafe { assert_eq!(w_int_get_value(result), 128) };
    }

    #[test]
    fn test_negative_shift_count() {
        assert!(lshift(w_int_new(1), w_int_new(-1)).is_err());
        assert!(rshift(w_int_new(1), w_int_new(-1)).is_err());
    }

    #[test]
    fn test_int_bitand() {
        let result = and_(w_int_new(0xFF), w_int_new(0x0F)).unwrap();
        unsafe { assert_eq!(w_int_get_value(result), 0x0F) };
    }

    #[test]
    fn test_int_bitor() {
        let result = or_(w_int_new(0xF0), w_int_new(0x0F)).unwrap();
        unsafe { assert_eq!(w_int_get_value(result), 0xFF) };
    }

    #[test]
    fn test_int_bitxor() {
        let result = xor(w_int_new(0xFF), w_int_new(0x0F)).unwrap();
        unsafe { assert_eq!(w_int_get_value(result), 0xF0) };
    }

    #[test]
    fn test_long_bitand() {
        let a = w_long_new(BigInt::from(i64::MAX) + BigInt::from(1));
        let b = w_int_new(0xFF);
        let result = and_(a, b).unwrap();
        unsafe { assert_eq!(w_int_get_value(result), 0) };
    }

    #[test]
    fn test_invert_long() {
        let a = w_long_new(BigInt::from(i64::MAX) + BigInt::from(1));
        let result = invert(a).unwrap();
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
        setattr(obj, "name", w_int_new(100)).unwrap();
        let result = getattr(obj, "name").unwrap();
        unsafe { assert_eq!(w_int_get_value(result), 100) };
    }

    #[test]
    fn test_getattr_missing() {
        let obj = w_int_new(1);
        let err = getattr(obj, "missing").unwrap_err();
        assert!(matches!(err.kind, PyErrorKind::AttributeError));
    }

    #[test]
    fn test_setattr_overwrite() {
        let obj = w_int_new(42);
        setattr(obj, "x", w_int_new(1)).unwrap();
        setattr(obj, "x", w_int_new(2)).unwrap();
        let result = getattr(obj, "x").unwrap();
        unsafe { assert_eq!(w_int_get_value(result), 2) };
    }

    #[test]
    fn test_module_setattr_getattr() {
        let mut namespace = Box::new(crate::PyNamespace::default());
        namespace.fix_ptr();
        let module = pyre_object::moduleobject::w_module_new(
            "test_module",
            Box::into_raw(namespace) as *mut u8,
        );

        setattr(module, "ps1", w_str_new("py> ")).unwrap();
        let result = getattr(module, "ps1").unwrap();
        unsafe { assert_eq!(w_str_get_value(result), "py> ") };
    }

    #[test]
    fn test_module_delattr() {
        let mut namespace = Box::new(crate::PyNamespace::default());
        namespace.fix_ptr();
        let module = pyre_object::moduleobject::w_module_new(
            "test_module",
            Box::into_raw(namespace) as *mut u8,
        );

        setattr(module, "ps1", w_str_new("py> ")).unwrap();
        delattr(module, "ps1").unwrap();
        let err = getattr(module, "ps1").unwrap_err();
        assert!(matches!(err.kind, PyErrorKind::AttributeError));
    }

    #[test]
    fn test_py_contains_manual_list() {
        let list = w_list_new(vec![w_int_new(1), w_int_new(2), w_int_new(3)]);
        let needle = w_int_new(1);
        unsafe {
            assert!(
                is_list(list),
                "should be list, got type: {}",
                (*(*list).ob_type).name
            );
        }
        let result = super::contains(list, needle).expect("contains failed");
        assert!(result, "1 should be in [1, 2, 3]");
    }
}

/// `in` operator: check if `needle` is in `haystack`.
/// PyPy: space.contains_w(haystack, needle)
pub fn contains(haystack: PyObjectRef, needle: PyObjectRef) -> Result<bool, PyError> {
    use pyre_object::*;
    unsafe {
        if is_list(haystack) {
            let len = w_list_len(haystack);
            for i in 0..len {
                if let Some(item) = w_list_getitem(haystack, i as i64) {
                    if eq_w(item, needle) {
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
                    if eq_w(item, needle) {
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
        // dict: key containment (dictobject.py __contains__)
        if is_dict(haystack) {
            return Ok(w_dict_lookup(haystack, needle).is_some());
        }
    }
    // Instance __contains__ — PyPy: descroperation.py contains_w
    unsafe {
        if is_instance(haystack) {
            let w_type = w_instance_get_type(haystack);
            if let Some(method) = lookup_in_type_where(w_type, "__contains__") {
                let result = crate::call_function(method, &[haystack, needle]);
                return Ok(is_true(result));
            }
            // Also check per-instance attributes (ATTR_TABLE)
            if let Ok(method) = getattr(haystack, "__contains__") {
                let result = crate::call_function(method, &[haystack, needle]);
                return Ok(is_true(result));
            }
        }
    }
    // Fallback: try iterating with getitem(obj, i) for i=0,1,...
    unsafe {
        let mut i = 0i64;
        loop {
            match getitem(haystack, pyre_object::w_int_new(i)) {
                Ok(item) => {
                    if eq_w(item, needle) {
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
fn eq_w(a: PyObjectRef, b: PyObjectRef) -> bool {
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
    compare(a, b, CompareOp::Eq)
        .map(|r| is_true(r))
        .unwrap_or(false)
}

/// Delete item: `del obj[index]`
///
/// PyPy: descroperation.py delitem → dispatches to type-specific __delitem__.
pub fn delitem(obj: PyObjectRef, index: PyObjectRef) -> Result<(), PyError> {
    use pyre_object::*;
    unsafe {
        if is_list(obj) {
            if is_int(index) {
                let i = w_int_get_value(index);
                let len = w_list_len(obj) as i64;
                let idx = if i < 0 { len + i } else { i };
                if idx >= 0 && idx < len {
                    w_list_pop(obj, idx);
                    return Ok(());
                }
                return Err(PyError::type_error("list index out of range"));
            }
            if is_slice(index) {
                let len = w_list_len(obj) as i64;
                let start = w_slice_get_start(index);
                let stop = w_slice_get_stop(index);
                let s = if is_none(start) {
                    0
                } else {
                    let v = w_int_get_value(start);
                    if v < 0 { (len + v).max(0) } else { v.min(len) }
                } as usize;
                let e = if is_none(stop) {
                    len
                } else {
                    let v = w_int_get_value(stop);
                    if v < 0 { (len + v).max(0) } else { v.min(len) }
                } as usize;
                w_list_delslice(obj, s, e);
                return Ok(());
            }
        }
        if is_dict(obj) {
            return dict_delitem(obj, index);
        }
    }
    // Instance __delitem__ — PyPy: descroperation.py delitem
    unsafe {
        if pyre_object::is_instance(obj) {
            if let Some(method) =
                lookup_in_type_where(pyre_object::w_instance_get_type(obj), "__delitem__")
            {
                crate::call_function(method, &[obj, index]);
                return Ok(());
            }
        }
    }
    Err(PyError::type_error("object does not support item deletion"))
}

/// Delete item from dict by key.
fn dict_delitem(obj: PyObjectRef, key: PyObjectRef) -> Result<(), PyError> {
    use pyre_object::*;
    unsafe {
        let dict = &mut *(obj as *mut dictobject::W_DictObject);
        let entries = &mut *dict.entries;
        for i in 0..entries.len() {
            let eq = if std::ptr::eq(entries[i].0, key) {
                true
            } else if is_int(entries[i].0) && is_int(key) {
                w_int_get_value(entries[i].0) == w_int_get_value(key)
            } else if is_str(entries[i].0) && is_str(key) {
                w_str_get_value(entries[i].0) == w_str_get_value(key)
            } else {
                false
            };
            if eq {
                entries.remove(i);
                dict.len -= 1;
                return Ok(());
            }
        }
    }
    Err(PyError::type_error("KeyError"))
}

// py_str and py_repr are defined in display.rs (with __str__/__repr__ dispatch).
// Re-exported via crate::display::*.
