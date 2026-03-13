//! JIT helper functions — `extern "C"` wrappers called from compiled traces.
//!
//! The JIT backend (Cranelift) emits C-ABI calls to these functions.
//! Each wraps a pyre-object or pyre-objspace operation with the
//! correct calling convention and integer-based parameter passing.

use pyre_object::*;
use pyre_objspace::py_is_true;
use pyre_runtime::{w_builtin_func_get, w_func_get_code_ptr};

/// Allocate a new `W_IntObject` wrapping the given i64.
pub extern "C" fn jit_w_int_new(value: i64) -> i64 {
    w_int_new(value) as i64
}

/// Allocate a `W_BoolObject` from an integer truth value (0 = false, nonzero = true).
pub extern "C" fn jit_w_bool_from(value: i64) -> i64 {
    w_bool_from(value != 0) as i64
}

/// Allocate a new `W_FloatObject` wrapping the given f64 (passed as i64 bit pattern).
pub extern "C" fn jit_w_float_new(value_bits: i64) -> i64 {
    let value = f64::from_bits(value_bits as u64);
    w_float_new(value) as i64
}

/// Test truthiness of a Python object. Returns 1 for truthy, 0 for falsy.
pub extern "C" fn jit_py_is_true(obj: i64) -> i64 {
    let obj = obj as PyObjectRef;
    py_is_true(obj) as i64
}

/// Concatenate two str objects. Returns a new str object as i64.
pub extern "C" fn jit_str_concat(a: i64, b: i64) -> i64 {
    let a = a as PyObjectRef;
    let b = b as PyObjectRef;
    unsafe {
        let sa = pyre_object::w_str_get_value(a);
        let sb = pyre_object::w_str_get_value(b);
        let mut result = String::with_capacity(sa.len() + sb.len());
        result.push_str(sa);
        result.push_str(sb);
        pyre_object::w_str_new(&result) as i64
    }
}

/// Repeat a str object `n` times. Returns a new str object as i64.
pub extern "C" fn jit_str_repeat(s: i64, n: i64) -> i64 {
    let s = s as PyObjectRef;
    unsafe {
        let sv = pyre_object::w_str_get_value(s);
        let count = if n < 0 { 0 } else { n as usize };
        pyre_object::w_str_new(&sv.repeat(count)) as i64
    }
}

/// Compare two str objects lexicographically. Returns -1, 0, or 1.
pub extern "C" fn jit_str_compare(a: i64, b: i64) -> i64 {
    let a = a as PyObjectRef;
    let b = b as PyObjectRef;
    unsafe {
        let sa = pyre_object::w_str_get_value(a);
        let sb = pyre_object::w_str_get_value(b);
        match sa.cmp(sb) {
            std::cmp::Ordering::Less => -1,
            std::cmp::Ordering::Equal => 0,
            std::cmp::Ordering::Greater => 1,
        }
    }
}

/// Test truthiness of a str object. Returns 1 for non-empty, 0 for empty.
pub extern "C" fn jit_str_is_true(s: i64) -> i64 {
    let s = s as PyObjectRef;
    unsafe { !pyre_object::w_str_get_value(s).is_empty() as i64 }
}

// ── Container helpers ────────────────────────────────────────────────

/// Build a list from a variable number of arguments.
///
/// Called with `count` followed by `count` PyObjectRef items encoded as i64.
/// Since the JIT call infrastructure passes a fixed number of args,
/// we use a two-step approach: build empty, then append items via separate calls.
///
/// This simplified version builds a list from a known number of arguments
/// (up to 8 for now — the trace emits individual append calls for larger lists).
pub extern "C" fn jit_build_list_0() -> i64 {
    w_list_new(vec![]) as i64
}

/// Append an item to a list. Void return (returns 0 for JIT compatibility).
pub extern "C" fn jit_list_append(list: i64, item: i64) -> i64 {
    let list = list as PyObjectRef;
    let item = item as PyObjectRef;
    unsafe { w_list_append(list, item) };
    0
}

/// Get item from a list by int index. Returns the item as i64 (PyObjectRef).
pub extern "C" fn jit_list_getitem(list: i64, index: i64) -> i64 {
    let list = list as PyObjectRef;
    unsafe {
        match w_list_getitem(list, index) {
            Some(val) => val as i64,
            None => panic!("list index out of range in JIT"),
        }
    }
}

/// Set item in a list by int index. Returns 0 for JIT compatibility.
pub extern "C" fn jit_list_setitem(list: i64, index: i64, value: i64) -> i64 {
    let list = list as PyObjectRef;
    let value = value as PyObjectRef;
    unsafe {
        if !w_list_setitem(list, index, value) {
            panic!("list assignment index out of range in JIT");
        }
    }
    0
}

/// Build an empty tuple (for the JIT, items are set via separate getitem calls on the source).
pub extern "C" fn jit_build_tuple_0() -> i64 {
    w_tuple_new(vec![]) as i64
}

/// Get item from a tuple by int index. Returns the item as i64 (PyObjectRef).
pub extern "C" fn jit_tuple_getitem(tuple: i64, index: i64) -> i64 {
    let tuple = tuple as PyObjectRef;
    unsafe {
        match w_tuple_getitem(tuple, index) {
            Some(val) => val as i64,
            None => panic!("tuple index out of range in JIT"),
        }
    }
}

/// Subscript dispatch for JIT: obj[index]. Returns the result item.
pub extern "C" fn jit_getitem(obj: i64, index: i64) -> i64 {
    let obj = obj as PyObjectRef;
    let index = index as PyObjectRef;
    match pyre_objspace::py_getitem(obj, index) {
        Ok(val) => val as i64,
        Err(e) => panic!("getitem failed in JIT: {e}"),
    }
}

/// Store subscript dispatch for JIT: obj[index] = value. Returns 0.
pub extern "C" fn jit_setitem(obj: i64, index: i64, value: i64) -> i64 {
    let obj = obj as PyObjectRef;
    let index = index as PyObjectRef;
    let value = value as PyObjectRef;
    match pyre_objspace::py_setitem(obj, index, value) {
        Ok(_) => 0,
        Err(e) => panic!("setitem failed in JIT: {e}"),
    }
}

// ── Function call helpers ──────────────────────────────────────────

/// Call a builtin function with a variable number of arguments.
///
/// The first argument is the callable, followed by the positional arguments.
/// The callable is a W_BuiltinFunction; its function pointer is extracted
/// and invoked with the remaining args.
pub extern "C" fn jit_call_builtin(callable: i64, args: *const i64, nargs: i64) -> i64 {
    let callable = callable as PyObjectRef;
    let nargs = nargs as usize;
    unsafe {
        let func = w_builtin_func_get(callable);
        let arg_slice = std::slice::from_raw_parts(args as *const PyObjectRef, nargs);
        func(arg_slice) as i64
    }
}

/// Call a user-defined function.
///
/// Takes callable, argc, then argc argument pointers.
/// Creates a new frame, binds arguments, and calls eval_frame recursively.
pub extern "C" fn jit_call_function(callable: i64, argc: i64, args: *const i64) -> i64 {
    let callable = callable as PyObjectRef;
    let argc = argc as usize;
    unsafe {
        let code_ptr = w_func_get_code_ptr(callable);
        let code = &*(code_ptr as *const pyre_bytecode::CodeObject);
        let arg_slice = std::slice::from_raw_parts(args as *const PyObjectRef, argc);
        // Build a mini-frame and evaluate — placeholder for Phase 1
        // In a full implementation, this would create a PyFrame and call eval_frame.
        // For now, this path is only reached if the trace includes function calls,
        // which is not expected in Phase 1 tracing.
        let _ = (code, arg_slice);
        panic!("jit_call_function: user function calls not yet supported in JIT helpers");
    }
}

// ── Range iterator helpers ──────────────────────────────────────────

/// Allocate a new range iterator from (start, stop, step).
pub extern "C" fn jit_range_iter_new(start: i64, stop: i64, step: i64) -> i64 {
    w_range_iter_new(start, stop, step) as i64
}
