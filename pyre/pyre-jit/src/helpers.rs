//! JIT helper functions — `extern "C"` wrappers called from compiled traces.
//!
//! The JIT backend (Cranelift) emits C-ABI calls to these functions.
//! Each wraps a pyre-object or pyre-objspace operation with the
//! correct calling convention and integer-based parameter passing.

use pyre_object::*;
use pyre_objspace::py_is_true;

/// Allocate a new `W_IntObject` wrapping the given i64.
pub extern "C" fn jit_w_int_new(value: i64) -> i64 {
    w_int_new(value) as i64
}

/// Allocate a `W_BoolObject` from an integer truth value (0 = false, nonzero = true).
pub extern "C" fn jit_w_bool_from(value: i64) -> i64 {
    w_bool_from(value != 0) as i64
}

/// Test truthiness of a Python object. Returns 1 for truthy, 0 for falsy.
pub extern "C" fn jit_py_is_true(obj: i64) -> i64 {
    let obj = obj as PyObjectRef;
    py_is_true(obj) as i64
}
