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
    namespace.get_or_insert_with("str", || w_builtin_func_new("str", builtin_str));
    namespace.get_or_insert_with("repr", || w_builtin_func_new("repr", builtin_repr));
    namespace.get_or_insert_with("int", || w_builtin_func_new("int", builtin_int));
    namespace.get_or_insert_with("float", || w_builtin_func_new("float", builtin_float));
    namespace.get_or_insert_with("bool", || w_builtin_func_new("bool", builtin_bool));
    namespace.get_or_insert_with("True", || w_bool_from(true));
    namespace.get_or_insert_with("False", || w_bool_from(false));
    namespace.get_or_insert_with("None", || w_none());
    namespace.get_or_insert_with("hasattr", || w_builtin_func_new("hasattr", builtin_hasattr));
    namespace.get_or_insert_with("getattr", || w_builtin_func_new("getattr", builtin_getattr));
    namespace.get_or_insert_with("setattr", || w_builtin_func_new("setattr", builtin_setattr));

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

    // Descriptor types
    namespace.get_or_insert_with("property", || {
        w_builtin_func_new("property", builtin_property)
    });
    namespace.get_or_insert_with("staticmethod", || {
        w_builtin_func_new("staticmethod", builtin_staticmethod)
    });
    namespace.get_or_insert_with("classmethod", || {
        w_builtin_func_new("classmethod", builtin_classmethod)
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

/// `isinstance(obj, cls)` — type check supporting user-defined classes.
///
/// PyPy: baseobjspace.py `isinstance_w` → check MRO chain.
fn builtin_isinstance(args: &[PyObjectRef]) -> PyObjectRef {
    assert!(args.len() == 2, "isinstance() takes exactly two arguments");
    let obj = args[0];
    let cls = args[1];
    unsafe {
        // If cls is a W_TypeObject, check if obj is an instance of it
        // by walking the instance's type chain (bases).
        if is_type(cls) && is_instance(obj) {
            let w_type = w_instance_get_type(obj);
            return w_bool_from(is_subtype(w_type, cls));
        }
        // Fallback: type name comparison for builtin types
        if is_str(cls) {
            let obj_type = (*(*obj).ob_type).tp_name;
            let check_type = w_str_get_value(cls);
            return w_bool_from(obj_type == check_type);
        }
    }
    w_bool_from(false)
}

/// Check if w_type is cls or a subtype of cls by walking the C3 MRO.
///
/// PyPy: baseobjspace.py `issubtype_w` → checks `cls in w_type.mro_w`.
/// Uses the single C3 MRO implementation in space.rs (no duplication).
unsafe fn is_subtype(w_type: PyObjectRef, cls: PyObjectRef) -> bool {
    if w_type.is_null() || !is_type(w_type) {
        return false;
    }
    for t in crate::space::compute_mro_pub(w_type) {
        if std::ptr::eq(t, cls) {
            return true;
        }
    }
    false
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

/// Callback type for the real __build_class__ implementation.
///
/// PyPy equivalent: pyopcode.py BUILD_CLASS calls space.call_function(metaclass, ...)
/// pyre-runtime cannot depend on pyre-interp, so we use a registered callback.
type BuildClassFn = fn(&[PyObjectRef]) -> PyObjectRef;

static BUILD_CLASS_IMPL: std::sync::OnceLock<BuildClassFn> = std::sync::OnceLock::new();

/// Register the real __build_class__ implementation from pyre-interp.
pub fn register_build_class_impl(f: BuildClassFn) {
    let _ = BUILD_CLASS_IMPL.set(f);
}

/// `__build_class__(body, name, *bases)` — class creation.
///
/// PyPy equivalent: pyopcode.py BUILD_CLASS
/// Delegates to the registered implementation (in pyre-interp).
fn builtin_build_class(args: &[PyObjectRef]) -> PyObjectRef {
    if let Some(impl_fn) = BUILD_CLASS_IMPL.get() {
        return impl_fn(args);
    }
    panic!("__build_class__ called but no implementation registered");
}

/// Get a reference to the `__build_class__` builtin function.
pub fn get_build_class_func() -> PyObjectRef {
    w_builtin_func_new("__build_class__", builtin_build_class)
}

/// `property(fget=None, fset=None, fdel=None, doc=None)` → W_PropertyObject
///
/// PyPy: descriptor.py W_Property
fn builtin_property(args: &[PyObjectRef]) -> PyObjectRef {
    let fget = args.first().copied().unwrap_or(pyre_object::PY_NULL);
    let fset = args.get(1).copied().unwrap_or(pyre_object::PY_NULL);
    let fdel = args.get(2).copied().unwrap_or(pyre_object::PY_NULL);
    pyre_object::w_property_new(fget, fset, fdel)
}

/// `staticmethod(func)` → W_StaticMethodObject
///
/// PyPy: function.py StaticMethod — __get__ returns wrapped func as-is.
fn builtin_staticmethod(args: &[PyObjectRef]) -> PyObjectRef {
    assert!(
        !args.is_empty(),
        "staticmethod requires a callable argument"
    );
    pyre_object::w_staticmethod_new(args[0])
}

/// `classmethod(func)` → W_ClassMethodObject
///
/// PyPy: function.py ClassMethod — __get__ binds the class as first arg.
fn builtin_classmethod(args: &[PyObjectRef]) -> PyObjectRef {
    assert!(!args.is_empty(), "classmethod requires a callable argument");
    pyre_object::w_classmethod_new(args[0])
}

/// `str(obj)` → convert to string
fn builtin_str(args: &[PyObjectRef]) -> PyObjectRef {
    if args.is_empty() {
        return w_str_new("");
    }
    let obj = args[0];
    unsafe {
        if is_str(obj) {
            return obj;
        }
    }
    let s = unsafe { crate::py_str(obj) };
    w_str_new(&s)
}

/// `repr(obj)` → string representation
fn builtin_repr(args: &[PyObjectRef]) -> PyObjectRef {
    assert!(args.len() == 1, "repr() takes exactly one argument");
    let s = unsafe { crate::py_repr(args[0]) };
    w_str_new(&s)
}

/// `int(obj)` → convert to int
fn builtin_int(args: &[PyObjectRef]) -> PyObjectRef {
    if args.is_empty() {
        return w_int_new(0);
    }
    let obj = args[0];
    unsafe {
        if is_int(obj) {
            return obj;
        }
        if is_float(obj) {
            return w_int_new(floatobject::w_float_get_value(obj) as i64);
        }
        if is_bool(obj) {
            return w_int_new(if w_bool_get_value(obj) { 1 } else { 0 });
        }
        if is_str(obj) {
            let s = w_str_get_value(obj);
            if let Ok(v) = s.trim().parse::<i64>() {
                return w_int_new(v);
            }
        }
    }
    w_int_new(0)
}

/// `float(obj)` → convert to float
fn builtin_float(args: &[PyObjectRef]) -> PyObjectRef {
    if args.is_empty() {
        return floatobject::w_float_new(0.0);
    }
    let obj = args[0];
    unsafe {
        if is_float(obj) {
            return obj;
        }
        if is_int(obj) {
            return floatobject::w_float_new(w_int_get_value(obj) as f64);
        }
        if is_str(obj) {
            let s = w_str_get_value(obj);
            if let Ok(v) = s.trim().parse::<f64>() {
                return floatobject::w_float_new(v);
            }
        }
    }
    floatobject::w_float_new(0.0)
}

/// `bool(obj)` → convert to bool (simplified truthiness)
fn builtin_bool(args: &[PyObjectRef]) -> PyObjectRef {
    if args.is_empty() {
        return w_bool_from(false);
    }
    let obj = args[0];
    unsafe {
        if is_bool(obj) {
            return obj;
        }
        if is_int(obj) {
            return w_bool_from(w_int_get_value(obj) != 0);
        }
        if is_none(obj) {
            return w_bool_from(false);
        }
    }
    w_bool_from(true)
}

/// `hasattr(obj, name)` → bool — direct call (no callback needed after merge)
fn builtin_hasattr(args: &[PyObjectRef]) -> PyObjectRef {
    assert!(args.len() == 2, "hasattr() takes exactly two arguments");
    let obj = args[0];
    let name = unsafe { w_str_get_value(args[1]) };
    w_bool_from(crate::space::py_getattr(obj, name).is_ok())
}

/// `getattr(obj, name[, default])` → value — direct call
fn builtin_getattr(args: &[PyObjectRef]) -> PyObjectRef {
    assert!(args.len() >= 2, "getattr() takes at least two arguments");
    let obj = args[0];
    let name = unsafe { w_str_get_value(args[1]) };
    match crate::space::py_getattr(obj, name) {
        Ok(val) => val,
        Err(_) => args
            .get(2)
            .copied()
            .unwrap_or_else(|| panic!("getattr: attribute '{name}' not found")),
    }
}

/// `setattr(obj, name, value)` — direct call
fn builtin_setattr(args: &[PyObjectRef]) -> PyObjectRef {
    assert!(args.len() == 3, "setattr() takes exactly three arguments");
    let obj = args[0];
    let name = unsafe { w_str_get_value(args[1]) };
    let _ = crate::space::py_setattr(obj, name, args[2]);
    w_none()
}
