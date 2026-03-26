use malachite_bigint::BigInt;
use num_traits::ToPrimitive;

use crate::executioncontext::PyNamespace;
use crate::{PyDisplay, w_builtin_func_new};
use pyre_object::*;

/// Install the default builtins into a namespace.
pub fn install_default_builtins(namespace: &mut PyNamespace) {
    namespace.get_or_insert_with("print", || w_builtin_func_new("print", builtin_print));
    namespace.get_or_insert_with("range", || w_builtin_func_new("range", builtin_range));
    namespace.get_or_insert_with("len", || w_builtin_func_new("len", builtin_len));
    namespace.get_or_insert_with("abs", || w_builtin_func_new("abs", builtin_abs));
    namespace.get_or_insert_with("min", || w_builtin_func_new("min", builtin_min));
    namespace.get_or_insert_with("max", || w_builtin_func_new("max", builtin_max));
    namespace.get_or_insert_with("type", || w_builtin_func_new("type", builtin_type));
    namespace.get_or_insert_with("isinstance", || {
        w_builtin_func_new("isinstance", builtin_isinstance)
    });

    // Exception type constructors — callable for `raise ValueError("msg")`
    // and identifiable by name for CHECK_EXC_MATCH.
    namespace.get_or_insert_with("BaseException", || {
        w_builtin_func_new("BaseException", exc_base_exception)
    });
    namespace.get_or_insert_with("Exception", || {
        w_builtin_func_new("Exception", exc_exception)
    });
    namespace.get_or_insert_with("ArithmeticError", || {
        w_builtin_func_new("ArithmeticError", exc_arithmetic_error)
    });
    namespace.get_or_insert_with("ZeroDivisionError", || {
        w_builtin_func_new("ZeroDivisionError", exc_zero_division)
    });
    namespace.get_or_insert_with("TypeError", || {
        w_builtin_func_new("TypeError", exc_type_error)
    });
    namespace.get_or_insert_with("ValueError", || {
        w_builtin_func_new("ValueError", exc_value_error)
    });
    namespace.get_or_insert_with("KeyError", || w_builtin_func_new("KeyError", exc_key_error));
    namespace.get_or_insert_with("IndexError", || {
        w_builtin_func_new("IndexError", exc_index_error)
    });
    namespace.get_or_insert_with("AttributeError", || {
        w_builtin_func_new("AttributeError", exc_attribute_error)
    });
    namespace.get_or_insert_with("NameError", || {
        w_builtin_func_new("NameError", exc_name_error)
    });
    namespace.get_or_insert_with("RuntimeError", || {
        w_builtin_func_new("RuntimeError", exc_runtime_error)
    });
    namespace.get_or_insert_with("StopIteration", || {
        w_builtin_func_new("StopIteration", exc_stop_iteration)
    });
    namespace.get_or_insert_with("OverflowError", || {
        w_builtin_func_new("OverflowError", exc_overflow_error)
    });
    namespace.get_or_insert_with("ImportError", || {
        w_builtin_func_new("ImportError", exc_import_error)
    });
    namespace.get_or_insert_with("NotImplementedError", || {
        w_builtin_func_new("NotImplementedError", exc_not_implemented_error)
    });
    namespace.get_or_insert_with("AssertionError", || {
        w_builtin_func_new("AssertionError", exc_assertion_error)
    });
}

/// Create a fresh namespace seeded with the default builtins.
pub fn new_builtin_namespace() -> PyNamespace {
    let mut namespace = PyNamespace::new();
    install_default_builtins(&mut namespace);
    namespace
}

/// `print(*args)` — write space-separated str representations to stdout.
fn builtin_print(args: &[PyObjectRef]) -> PyObjectRef {
    let parts: Vec<String> = args
        .iter()
        .map(|&obj| format!("{}", PyDisplay(obj)))
        .collect();
    println!("{}", parts.join(" "));
    w_none()
}

/// Extract an i64 from an int or long object. Panics if the long value
/// exceeds i64 range (range() cannot iterate over such large spans).
unsafe fn range_arg_to_i64(obj: PyObjectRef) -> i64 {
    unsafe {
        if is_int(obj) {
            return w_int_get_value(obj);
        }
        if is_long(obj) {
            let val = w_long_get_value(obj);
            return val
                .to_i64()
                .unwrap_or_else(|| panic!("range() argument too large for iteration"));
        }
        panic!(
            "range() integer argument expected, got {}",
            (*(*obj).ob_type).tp_name
        )
    }
}

/// `range(stop)` or `range(start, stop)` or `range(start, stop, step)`.
///
/// Returns a range iterator directly (simplified: no separate range object).
fn builtin_range(args: &[PyObjectRef]) -> PyObjectRef {
    match args.len() {
        1 => {
            let stop = unsafe { range_arg_to_i64(args[0]) };
            w_range_iter_new(0, stop, 1)
        }
        2 => {
            let start = unsafe { range_arg_to_i64(args[0]) };
            let stop = unsafe { range_arg_to_i64(args[1]) };
            w_range_iter_new(start, stop, 1)
        }
        3 => {
            let start = unsafe { range_arg_to_i64(args[0]) };
            let stop = unsafe { range_arg_to_i64(args[1]) };
            let step = unsafe { range_arg_to_i64(args[2]) };
            w_range_iter_new(start, stop, step)
        }
        _ => panic!("range() takes 1 to 3 arguments"),
    }
}

/// `len(obj)` — return the length of an object.
fn builtin_len(args: &[PyObjectRef]) -> PyObjectRef {
    assert!(args.len() == 1, "len() takes exactly one argument");
    let obj = args[0];
    unsafe {
        if is_list(obj) {
            return w_int_new(w_list_len(obj) as i64);
        }
        if is_tuple(obj) {
            return w_int_new(w_tuple_len(obj) as i64);
        }
        if is_dict(obj) {
            return w_int_new(w_dict_len(obj) as i64);
        }
        if is_str(obj) {
            return w_int_new(w_str_len(obj) as i64);
        }
    }
    panic!("object has no len()")
}

/// `abs(x)` — return the absolute value of a number.
fn builtin_abs(args: &[PyObjectRef]) -> PyObjectRef {
    assert!(args.len() == 1, "abs() takes exactly one argument");
    let obj = args[0];
    unsafe {
        if is_int(obj) {
            let v = w_int_get_value(obj);
            // i64::MIN.abs() overflows; promote to long
            return match v.checked_abs() {
                Some(r) => w_int_new(r),
                None => w_long_new(-BigInt::from(v)),
            };
        }
        if is_long(obj) {
            let val = w_long_get_value(obj).clone();
            return w_long_new(if val < BigInt::from(0) { -val } else { val });
        }
        if is_float(obj) {
            return w_float_new(w_float_get_value(obj).abs());
        }
    }
    panic!("bad operand type for abs()")
}

/// Convert an int or long object to BigInt for comparison.
unsafe fn obj_to_bigint(obj: PyObjectRef) -> BigInt {
    unsafe {
        if is_int(obj) {
            BigInt::from(w_int_get_value(obj))
        } else {
            w_long_get_value(obj).clone()
        }
    }
}

/// `min(a, b)` — return the smaller of two values.
fn builtin_min(args: &[PyObjectRef]) -> PyObjectRef {
    assert!(args.len() == 2, "min() takes exactly two arguments");
    let a = args[0];
    let b = args[1];
    unsafe {
        if is_int(a) && is_int(b) {
            let va = w_int_get_value(a);
            let vb = w_int_get_value(b);
            return if va <= vb { a } else { b };
        }
        if is_int_or_long(a) && is_int_or_long(b) {
            let va = obj_to_bigint(a);
            let vb = obj_to_bigint(b);
            return if va <= vb { a } else { b };
        }
    }
    panic!("min() not supported for these types")
}

/// `max(a, b)` — return the larger of two values.
fn builtin_max(args: &[PyObjectRef]) -> PyObjectRef {
    assert!(args.len() == 2, "max() takes exactly two arguments");
    let a = args[0];
    let b = args[1];
    unsafe {
        if is_int(a) && is_int(b) {
            let va = w_int_get_value(a);
            let vb = w_int_get_value(b);
            return if va >= vb { a } else { b };
        }
        if is_int_or_long(a) && is_int_or_long(b) {
            let va = obj_to_bigint(a);
            let vb = obj_to_bigint(b);
            return if va >= vb { a } else { b };
        }
    }
    panic!("max() not supported for these types")
}

/// `type(obj)` — return the type name as a string (simplified).
fn builtin_type(args: &[PyObjectRef]) -> PyObjectRef {
    assert!(args.len() == 1, "type() takes exactly one argument");
    let obj = args[0];
    let name = unsafe { (*(*obj).ob_type).tp_name };
    box_str_constant(name)
}

/// `isinstance(obj, type_name)` — simplified type check using type name string.
fn builtin_isinstance(args: &[PyObjectRef]) -> PyObjectRef {
    assert!(args.len() == 2, "isinstance() takes exactly two arguments");
    let obj = args[0];
    let type_name_obj = args[1];
    let obj_type = unsafe { (*(*obj).ob_type).tp_name };
    let check_type = unsafe { w_str_get_value(type_name_obj) };
    w_bool_from(obj_type == check_type)
}

/// Exception type constructor — called as e.g. `ValueError("msg")`.
/// Extracts the message from the first argument and creates a W_ExceptionObject.
macro_rules! exc_constructor {
    ($fn_name:ident, $kind:expr) => {
        fn $fn_name(args: &[PyObjectRef]) -> PyObjectRef {
            let msg = if args.is_empty() {
                ""
            } else if unsafe { is_str(args[0]) } {
                unsafe { w_str_get_value(args[0]) }
            } else {
                ""
            };
            pyre_object::excobject::w_exception_new($kind, msg)
        }
    };
}

exc_constructor!(
    exc_base_exception,
    pyre_object::excobject::ExcKind::BaseException
);
exc_constructor!(exc_exception, pyre_object::excobject::ExcKind::Exception);
exc_constructor!(
    exc_arithmetic_error,
    pyre_object::excobject::ExcKind::ArithmeticError
);
exc_constructor!(
    exc_zero_division,
    pyre_object::excobject::ExcKind::ZeroDivisionError
);
exc_constructor!(exc_type_error, pyre_object::excobject::ExcKind::TypeError);
exc_constructor!(exc_value_error, pyre_object::excobject::ExcKind::ValueError);
exc_constructor!(exc_key_error, pyre_object::excobject::ExcKind::KeyError);
exc_constructor!(exc_index_error, pyre_object::excobject::ExcKind::IndexError);
exc_constructor!(
    exc_attribute_error,
    pyre_object::excobject::ExcKind::AttributeError
);
exc_constructor!(exc_name_error, pyre_object::excobject::ExcKind::NameError);
exc_constructor!(
    exc_runtime_error,
    pyre_object::excobject::ExcKind::RuntimeError
);
exc_constructor!(
    exc_stop_iteration,
    pyre_object::excobject::ExcKind::StopIteration
);
exc_constructor!(
    exc_overflow_error,
    pyre_object::excobject::ExcKind::OverflowError
);
exc_constructor!(
    exc_import_error,
    pyre_object::excobject::ExcKind::ImportError
);
exc_constructor!(
    exc_not_implemented_error,
    pyre_object::excobject::ExcKind::NotImplementedError
);
exc_constructor!(
    exc_assertion_error,
    pyre_object::excobject::ExcKind::AssertionError
);

/// `__build_class__(body, name, *bases)` — stub for class creation.
/// Phase 1: returns None (class bodies not yet executed).
fn builtin_build_class(args: &[PyObjectRef]) -> PyObjectRef {
    let _ = args;
    w_none()
}

/// Get a reference to the `__build_class__` builtin function.
pub fn get_build_class_func() -> PyObjectRef {
    w_builtin_func_new("__build_class__", builtin_build_class)
}
