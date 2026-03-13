//! ObjSpace — Python object operation dispatch.
//!
//! The ObjSpace mediates all operations on Python objects. This is the layer
//! where type-specific fast paths live, and where the JIT inserts `GuardClass`
//! to specialize operations.

// Suppress unsafe-in-unsafe-fn warnings; our unsafe fns are inherently
// working with raw pointers throughout and wrapping every call in an
// additional unsafe block adds noise without safety benefit.
#![allow(unsafe_op_in_unsafe_fn)]

use pyre_object::*;

/// Result type for Python operations.
///
/// `Ok(PyObjectRef)` for normal return, `Err(PyError)` for exceptions.
pub type PyResult = Result<PyObjectRef, PyError>;

/// Python exception (simplified for Phase 1).
#[derive(Debug, Clone)]
pub struct PyError {
    pub kind: PyErrorKind,
    pub message: String,
}

#[derive(Debug, Clone)]
pub enum PyErrorKind {
    TypeError,
    ZeroDivisionError,
    NameError,
}

impl PyError {
    pub fn type_error(msg: impl Into<String>) -> Self {
        PyError {
            kind: PyErrorKind::TypeError,
            message: msg.into(),
        }
    }

    pub fn zero_division(msg: impl Into<String>) -> Self {
        PyError {
            kind: PyErrorKind::ZeroDivisionError,
            message: msg.into(),
        }
    }
}

impl std::fmt::Display for PyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}: {}", self.kind, self.message)
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
    Ok(w_int_new(va + vb))
}

unsafe fn int_sub(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let va = w_int_get_value(a);
    let vb = w_int_get_value(b);
    Ok(w_int_new(va - vb))
}

unsafe fn int_mul(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let va = w_int_get_value(a);
    let vb = w_int_get_value(b);
    Ok(w_int_new(va * vb))
}

unsafe fn int_floordiv(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let va = w_int_get_value(a);
    let vb = w_int_get_value(b);
    if vb == 0 {
        return Err(PyError::zero_division("integer division or modulo by zero"));
    }
    Ok(w_int_new(va.div_euclid(vb)))
}

unsafe fn int_mod(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let va = w_int_get_value(a);
    let vb = w_int_get_value(b);
    if vb == 0 {
        return Err(PyError::zero_division("integer division or modulo by zero"));
    }
    Ok(w_int_new(va.rem_euclid(vb)))
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

// ── Public dispatch API ───────────────────────────────────────────────

/// Binary operation dispatch.
///
/// Checks types and dispatches to the appropriate fast path.
/// The JIT traces through this function, recording `GuardClass` on operand types.
pub fn py_add(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    unsafe {
        if is_int(a) && is_int(b) {
            return int_add(a, b);
        }
        Err(PyError::type_error(format!(
            "unsupported operand type(s) for +: '{}' and '{}'",
            (*(*a).ob_type).tp_name,
            (*(*b).ob_type).tp_name,
        )))
    }
}

pub fn py_sub(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    unsafe {
        if is_int(a) && is_int(b) {
            return int_sub(a, b);
        }
        Err(PyError::type_error(format!(
            "unsupported operand type(s) for -: '{}' and '{}'",
            (*(*a).ob_type).tp_name,
            (*(*b).ob_type).tp_name,
        )))
    }
}

pub fn py_mul(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    unsafe {
        if is_int(a) && is_int(b) {
            return int_mul(a, b);
        }
        Err(PyError::type_error(format!(
            "unsupported operand type(s) for *: '{}' and '{}'",
            (*(*a).ob_type).tp_name,
            (*(*b).ob_type).tp_name,
        )))
    }
}

pub fn py_floordiv(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    unsafe {
        if is_int(a) && is_int(b) {
            return int_floordiv(a, b);
        }
        Err(PyError::type_error(format!(
            "unsupported operand type(s) for //: '{}' and '{}'",
            (*(*a).ob_type).tp_name,
            (*(*b).ob_type).tp_name,
        )))
    }
}

pub fn py_mod(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    unsafe {
        if is_int(a) && is_int(b) {
            return int_mod(a, b);
        }
        Err(PyError::type_error(format!(
            "unsupported operand type(s) for %: '{}' and '{}'",
            (*(*a).ob_type).tp_name,
            (*(*b).ob_type).tp_name,
        )))
    }
}

/// Comparison operation dispatch.
pub fn py_compare(a: PyObjectRef, b: PyObjectRef, op: CompareOp) -> PyResult {
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
    unsafe {
        if is_bool(obj) {
            return w_bool_get_value(obj);
        }
        if is_int(obj) {
            return w_int_get_value(obj) != 0;
        }
        if is_none(obj) {
            return false;
        }
        true // default: objects are truthy
    }
}

/// Unary negation.
pub fn py_negative(a: PyObjectRef) -> PyResult {
    unsafe {
        if is_int(a) {
            return Ok(w_int_new(-w_int_get_value(a)));
        }
        Err(PyError::type_error(format!(
            "bad operand type for unary -: '{}'",
            (*(*a).ob_type).tp_name,
        )))
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
}
