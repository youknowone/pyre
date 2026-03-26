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
    namespace.get_or_insert_with("str", || crate::typedef::get_builtin_type(&STR_TYPE));
    namespace.get_or_insert_with("repr", || w_builtin_func_new("repr", builtin_repr));
    namespace.get_or_insert_with("int", || crate::typedef::get_builtin_type(&INT_TYPE));
    namespace.get_or_insert_with("float", || crate::typedef::get_builtin_type(&FLOAT_TYPE));
    namespace.get_or_insert_with("bool", || crate::typedef::get_builtin_type(&BOOL_TYPE));
    namespace.get_or_insert_with("True", || w_bool_from(true));
    namespace.get_or_insert_with("False", || w_bool_from(false));
    namespace.get_or_insert_with("None", || w_none());
    namespace.get_or_insert_with("hasattr", || w_builtin_func_new("hasattr", builtin_hasattr));
    namespace.get_or_insert_with("getattr", || w_builtin_func_new("getattr", builtin_getattr));
    namespace.get_or_insert_with("setattr", || w_builtin_func_new("setattr", builtin_setattr));
    namespace.get_or_insert_with("tuple", || crate::typedef::get_builtin_type(&TUPLE_TYPE));
    namespace.get_or_insert_with("list", || crate::typedef::get_builtin_type(&LIST_TYPE));
    namespace.get_or_insert_with("dict", || crate::typedef::get_builtin_type(&DICT_TYPE));
    namespace.get_or_insert_with("object", || {
        // `object` is a W_TypeObject, not a builtin function.
        // PyPy: baseobjspace.py w_object = W_TypeObject("object", ...)
        crate::typedef::get_object_type()
    });
    namespace.get_or_insert_with("super", || w_builtin_func_new("super", builtin_super));
    namespace.get_or_insert_with("id", || w_builtin_func_new("id", builtin_id));
    namespace.get_or_insert_with("hash", || w_builtin_func_new("hash", builtin_hash));
    namespace.get_or_insert_with("ord", || w_builtin_func_new("ord", builtin_ord));
    namespace.get_or_insert_with("chr", || w_builtin_func_new("chr", builtin_chr));
    namespace.get_or_insert_with("map", || w_builtin_func_new("map", builtin_map));
    namespace.get_or_insert_with("zip", || w_builtin_func_new("zip", builtin_zip));
    namespace.get_or_insert_with("enumerate", || {
        w_builtin_func_new("enumerate", builtin_enumerate)
    });
    namespace.get_or_insert_with("reversed", || {
        w_builtin_func_new("reversed", builtin_reversed)
    });
    namespace.get_or_insert_with("sorted", || w_builtin_func_new("sorted", builtin_sorted));
    namespace.get_or_insert_with("any", || w_builtin_func_new("any", builtin_any));
    namespace.get_or_insert_with("all", || w_builtin_func_new("all", builtin_all));
    namespace.get_or_insert_with("sum", || w_builtin_func_new("sum", builtin_sum));
    namespace.get_or_insert_with("round", || w_builtin_func_new("round", builtin_round));
    namespace.get_or_insert_with("divmod", || w_builtin_func_new("divmod", builtin_divmod));
    namespace.get_or_insert_with("pow", || w_builtin_func_new("pow", builtin_pow));
    namespace.get_or_insert_with("hex", || w_builtin_func_new("hex", builtin_hex));
    namespace.get_or_insert_with("oct", || w_builtin_func_new("oct", builtin_oct));
    namespace.get_or_insert_with("bin", || w_builtin_func_new("bin", builtin_bin));
    namespace.get_or_insert_with("format", || w_builtin_func_new("format", builtin_format));
    namespace.get_or_insert_with("__import__", || {
        w_builtin_func_new("__import__", builtin_import_stub)
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
fn builtin_print(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let parts: Vec<String> = args
        .iter()
        .map(|&obj| format!("{}", PyDisplay(obj)))
        .collect();
    println!("{}", parts.join(" "));
    Ok(w_none())
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
fn builtin_range(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    match args.len() {
        1 => {
            let stop = unsafe { range_arg_to_i64(args[0]) };
            Ok(w_range_iter_new(0, stop, 1))
        }
        2 => {
            let start = unsafe { range_arg_to_i64(args[0]) };
            let stop = unsafe { range_arg_to_i64(args[1]) };
            Ok(w_range_iter_new(start, stop, 1))
        }
        3 => {
            let start = unsafe { range_arg_to_i64(args[0]) };
            let stop = unsafe { range_arg_to_i64(args[1]) };
            let step = unsafe { range_arg_to_i64(args[2]) };
            Ok(w_range_iter_new(start, stop, step))
        }
        _ => panic!("range() takes 1 to 3 arguments"),
    }
}

/// `len(obj)` — return the length of an object.
/// `len(obj)` — PyPy: operation.py len → space.len_w
fn builtin_len(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() == 1, "len() takes exactly one argument");
    crate::space::py_len(args[0])
}

/// `abs(x)` — return the absolute value of a number.
fn builtin_abs(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() == 1, "abs() takes exactly one argument");
    let obj = args[0];
    unsafe {
        if is_int(obj) {
            let v = w_int_get_value(obj);
            // i64::MIN.abs() overflows; promote to long
            return Ok(match v.checked_abs() {
                Some(r) => w_int_new(r),
                None => w_long_new(-BigInt::from(v)),
            });
        }
        if is_long(obj) {
            let val = w_long_get_value(obj).clone();
            return Ok(w_long_new(if val < BigInt::from(0) { -val } else { val }));
        }
        if is_float(obj) {
            return Ok(w_float_new(w_float_get_value(obj).abs()));
        }
    }
    // Instance __abs__ — PyPy: baseobjspace.py abs
    unsafe {
        if pyre_object::is_instance(obj) {
            let w_type = pyre_object::w_instance_get_type(obj);
            if let Some(method) = crate::space::lookup_in_type_mro_pub(w_type, "__abs__") {
                return Ok(crate::space_call_function(method, &[obj]));
            }
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
fn builtin_min(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() == 2, "min() takes exactly two arguments");
    let a = args[0];
    let b = args[1];
    unsafe {
        if is_int(a) && is_int(b) {
            let va = w_int_get_value(a);
            let vb = w_int_get_value(b);
            return Ok(if va <= vb { a } else { b });
        }
        if is_int_or_long(a) && is_int_or_long(b) {
            let va = obj_to_bigint(a);
            let vb = obj_to_bigint(b);
            return Ok(if va <= vb { a } else { b });
        }
    }
    panic!("min() not supported for these types")
}

/// `max(a, b)` — return the larger of two values.
fn builtin_max(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() == 2, "max() takes exactly two arguments");
    let a = args[0];
    let b = args[1];
    unsafe {
        if is_int(a) && is_int(b) {
            let va = w_int_get_value(a);
            let vb = w_int_get_value(b);
            return Ok(if va >= vb { a } else { b });
        }
        if is_int_or_long(a) && is_int_or_long(b) {
            let va = obj_to_bigint(a);
            let vb = obj_to_bigint(b);
            return Ok(if va >= vb { a } else { b });
        }
    }
    panic!("max() not supported for these types")
}

/// `type(obj)` — return the type name as a string (simplified).
fn builtin_type(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() == 1, "type() takes exactly one argument");
    let obj = args[0];
    let name = unsafe { (*(*obj).ob_type).tp_name };
    Ok(box_str_constant(name))
}

/// `isinstance(obj, cls)` — type check supporting user-defined classes.
///
/// PyPy: baseobjspace.py `isinstance_w` → check MRO chain.
fn builtin_isinstance(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() == 2, "isinstance() takes exactly two arguments");
    let obj = args[0];
    let cls = args[1];

    // isinstance(obj, (cls1, cls2, ...)) — any match
    if unsafe { is_tuple(cls) } {
        let len = unsafe { w_tuple_len(cls) };
        for i in 0..len {
            if let Some(c) = unsafe { w_tuple_getitem(cls, i as i64) } {
                if isinstance_check(obj, c) {
                    return Ok(w_bool_from(true));
                }
            }
        }
        return Ok(w_bool_from(false));
    }

    Ok(w_bool_from(isinstance_check(obj, cls)))
}

/// Single-type isinstance check.
///
/// PyPy: baseobjspace.py `isinstance_w` → `abstract_issubclass_w`
fn isinstance_check(obj: PyObjectRef, cls: PyObjectRef) -> bool {
    unsafe {
        if !is_type(cls) {
            return false;
        }
        // Get the W_TypeObject for obj's type.
        // For instances: w_instance_get_type.
        // For builtins: type registry lookup via ob_type.
        let obj_type = if is_instance(obj) {
            w_instance_get_type(obj)
        } else {
            // Builtin object → resolve via type registry (PyPy: space.type(w_obj))
            match crate::typedef::type_of(obj) {
                Some(t) => t,
                None => return false,
            }
        };
        is_subtype(obj_type, cls)
    }
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
        fn $fn_name(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
            let msg = if args.is_empty() {
                ""
            } else if unsafe { is_str(args[0]) } {
                unsafe { w_str_get_value(args[0]) }
            } else {
                ""
            };
            Ok(pyre_object::excobject::w_exception_new($kind, msg))
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

/// `__build_class__(body, name, *bases)` — class creation.
///
/// PyPy equivalent: pyopcode.py BUILD_CLASS
/// Direct call to call::real_build_class (no callback needed —
/// interpreter and runtime are in the same crate).
fn builtin_build_class(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    crate::call::real_build_class(args)
}

/// Get a reference to the `__build_class__` builtin function.
pub fn get_build_class_func() -> PyObjectRef {
    w_builtin_func_new("__build_class__", builtin_build_class)
}

/// `property(fget=None, fset=None, fdel=None, doc=None)` → W_PropertyObject
///
/// PyPy: descriptor.py W_Property
fn builtin_property(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let fget = args.first().copied().unwrap_or(pyre_object::PY_NULL);
    let fset = args.get(1).copied().unwrap_or(pyre_object::PY_NULL);
    let fdel = args.get(2).copied().unwrap_or(pyre_object::PY_NULL);
    Ok(pyre_object::w_property_new(fget, fset, fdel))
}

/// `staticmethod(func)` → W_StaticMethodObject
///
/// PyPy: function.py StaticMethod — __get__ returns wrapped func as-is.
fn builtin_staticmethod(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(
        !args.is_empty(),
        "staticmethod requires a callable argument"
    );
    Ok(pyre_object::w_staticmethod_new(args[0]))
}

/// `classmethod(func)` → W_ClassMethodObject
///
/// PyPy: function.py ClassMethod — __get__ binds the class as first arg.
fn builtin_classmethod(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(!args.is_empty(), "classmethod requires a callable argument");
    Ok(pyre_object::w_classmethod_new(args[0]))
}

/// `str(obj)` → convert to string
pub(crate) fn builtin_str_pub(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    builtin_str(args)
}
fn builtin_str(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.is_empty() {
        return Ok(w_str_new(""));
    }
    let obj = args[0];
    unsafe {
        if is_str(obj) {
            return Ok(obj);
        }
    }
    let s = unsafe { crate::py_str(obj) };
    Ok(w_str_new(&s))
}

/// `repr(obj)` → string representation
fn builtin_repr(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() == 1, "repr() takes exactly one argument");
    let s = unsafe { crate::py_repr(args[0]) };
    Ok(w_str_new(&s))
}

pub(crate) fn builtin_int_pub(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    builtin_int(args)
}
/// `int(obj)` → convert to int
fn builtin_int(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.is_empty() {
        return Ok(w_int_new(0));
    }
    let obj = args[0];
    unsafe {
        if is_int(obj) {
            return Ok(obj);
        }
        if is_float(obj) {
            return Ok(w_int_new(floatobject::w_float_get_value(obj) as i64));
        }
        if is_bool(obj) {
            return Ok(w_int_new(if w_bool_get_value(obj) { 1 } else { 0 }));
        }
        if is_str(obj) {
            let s = w_str_get_value(obj);
            if let Ok(v) = s.trim().parse::<i64>() {
                return Ok(w_int_new(v));
            }
        }
    }
    Ok(w_int_new(0))
}

pub(crate) fn builtin_float_pub(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    builtin_float(args)
}
/// `float(obj)` → convert to float
fn builtin_float(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.is_empty() {
        return Ok(floatobject::w_float_new(0.0));
    }
    let obj = args[0];
    unsafe {
        if is_float(obj) {
            return Ok(obj);
        }
        if is_int(obj) {
            return Ok(floatobject::w_float_new(w_int_get_value(obj) as f64));
        }
        if is_str(obj) {
            let s = w_str_get_value(obj);
            if let Ok(v) = s.trim().parse::<f64>() {
                return Ok(floatobject::w_float_new(v));
            }
        }
    }
    Ok(floatobject::w_float_new(0.0))
}

pub(crate) fn builtin_bool_pub(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    builtin_bool(args)
}
/// `bool(obj)` — PyPy: operation.py bool → space.is_true
fn builtin_bool(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.is_empty() {
        return Ok(w_bool_from(false));
    }
    Ok(w_bool_from(crate::space::py_is_true(args[0])))
}

/// `hasattr(obj, name)` → bool — direct call (no callback needed after merge)
fn builtin_hasattr(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() == 2, "hasattr() takes exactly two arguments");
    let obj = args[0];
    let name = unsafe { w_str_get_value(args[1]) };
    Ok(w_bool_from(crate::space::py_getattr(obj, name).is_ok()))
}

/// `getattr(obj, name[, default])` → value — direct call
fn builtin_getattr(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() >= 2, "getattr() takes at least two arguments");
    let obj = args[0];
    let name = unsafe { w_str_get_value(args[1]) };
    match crate::space::py_getattr(obj, name) {
        Ok(val) => Ok(val),
        Err(_) => Ok(args
            .get(2)
            .copied()
            .unwrap_or_else(|| panic!("getattr: attribute '{name}' not found"))),
    }
}

/// `setattr(obj, name, value)` — direct call
fn builtin_setattr(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() == 3, "setattr() takes exactly three arguments");
    let obj = args[0];
    let name = unsafe { w_str_get_value(args[1]) };
    let _ = crate::space::py_setattr(obj, name, args[2]);
    Ok(w_none())
}

pub(crate) fn builtin_tuple_pub(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    builtin_tuple(args)
}
fn builtin_tuple(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.is_empty() {
        return Ok(w_tuple_new(vec![]));
    }
    let obj = args[0];
    unsafe {
        if is_tuple(obj) {
            return Ok(obj);
        }
        if is_list(obj) {
            let n = w_list_len(obj);
            let items: Vec<_> = (0..n)
                .filter_map(|i| w_list_getitem(obj, i as i64))
                .collect();
            return Ok(w_tuple_new(items));
        }
    }
    Ok(w_tuple_new(vec![]))
}

pub(crate) fn builtin_list_ctor_pub(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    builtin_list_ctor(args)
}
fn builtin_list_ctor(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.is_empty() {
        return Ok(w_list_new(vec![]));
    }
    let obj = args[0];
    unsafe {
        if is_list(obj) {
            return Ok(obj);
        }
        if is_tuple(obj) {
            let n = w_tuple_len(obj);
            let items: Vec<_> = (0..n)
                .filter_map(|i| w_tuple_getitem(obj, i as i64))
                .collect();
            return Ok(w_list_new(items));
        }
    }
    Ok(w_list_new(vec![]))
}

pub(crate) fn builtin_dict_ctor_pub(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    builtin_dict_ctor(args)
}
/// `dict()` — PyPy: dictobject.py W_DictMultiObject.descr_init
fn builtin_dict_ctor(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.is_empty() {
        return Ok(w_dict_new());
    }
    unsafe {
        if is_dict(args[0]) {
            return Ok(args[0]);
        }
    }
    // Try to construct from iterable of (key, value) pairs
    // PyPy: dictobject.py W_DictMultiObject.descr_init
    let src = args[0];
    unsafe {
        if is_list(src) {
            let dict = w_dict_new();
            let n = w_list_len(src);
            for i in 0..n {
                if let Some(pair) = w_list_getitem(src, i as i64) {
                    if is_tuple(pair) && w_tuple_len(pair) == 2 {
                        let k = w_tuple_getitem(pair, 0).unwrap();
                        let v = w_tuple_getitem(pair, 1).unwrap();
                        w_dict_store(dict, k, v);
                    }
                }
            }
            return Ok(dict);
        }
    }
    panic!("dict() from this type not yet implemented");
}

/// `object()` — PyPy: objectobject.py descr__new__
fn builtin_object(_args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    // PyPy: objectobject.py descr__new__ → allocate bare object
    // In Python, object() returns a featureless object instance.
    // Use PY_NULL as type since there's no builtin 'object' W_TypeObject yet.
    Ok(pyre_object::w_instance_new(pyre_object::PY_NULL))
    // Full implementation requires a base object type in TypeDef.
}

/// `super()` — PyPy: descriptor.py W_Super
fn builtin_super(_args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    panic!("super() not yet implemented (requires __class__ cell + frame introspection)");
}

/// `id(obj)` — PyPy: baseobjspace.py id → object identity as int
fn builtin_id(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(!args.is_empty(), "id() takes exactly one argument");
    Ok(w_int_new(args[0] as i64))
}

/// `hash(obj)` — PyPy: baseobjspace.py hash → identity for now
fn builtin_hash(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(!args.is_empty(), "hash() takes exactly one argument");
    unsafe {
        if is_int(args[0]) {
            return Ok(args[0]); // int hashes to itself
        }
        if is_str(args[0]) {
            // Simplified string hash — deterministic within one run
            let s = w_str_get_value(args[0]);
            let mut h: i64 = 0;
            for b in s.bytes() {
                h = h.wrapping_mul(1000003).wrapping_add(b as i64);
            }
            return Ok(w_int_new(h));
        }
    }
    // Instance __hash__ — PyPy: baseobjspace.py hash_w
    unsafe {
        if pyre_object::is_instance(args[0]) {
            let w_type = pyre_object::w_instance_get_type(args[0]);
            if let Some(method) = crate::space::lookup_in_type_mro_pub(w_type, "__hash__") {
                return Ok(crate::space_call_function(method, &[args[0]]));
            }
        }
    }
    Ok(w_int_new(args[0] as i64)) // identity hash fallback
}

/// `ord(c)` — PyPy: operation.py ord
fn builtin_ord(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() == 1, "ord() takes exactly one argument");
    let s = unsafe { w_str_get_value(args[0]) };
    assert!(s.len() == 1, "ord() expected a character");
    Ok(w_int_new(s.chars().next().unwrap() as i64))
}

/// `chr(i)` — PyPy: operation.py chr
fn builtin_chr(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() == 1, "chr() takes exactly one argument");
    let code = unsafe { w_int_get_value(args[0]) } as u32;
    let c = char::from_u32(code).expect("chr() arg not in range");
    Ok(w_str_new(&c.to_string()))
}

/// `map()` — PyPy: functional.py W_Map (returns iterator)
fn builtin_map(_args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    panic!("map() not yet implemented (requires iterator protocol)");
}

/// `zip()` — PyPy: functional.py W_Zip (returns iterator)
fn builtin_zip(_args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    panic!("zip() not yet implemented (requires iterator protocol)");
}

/// `enumerate()` — PyPy: functional.py W_Enumerate (returns iterator)
fn builtin_enumerate(_args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    panic!("enumerate() not yet implemented (requires iterator protocol)");
}

/// `reversed()` — PyPy: functional.py W_ReversedIterator
fn builtin_reversed(_args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    panic!("reversed() not yet implemented (requires iterator protocol)");
}

/// `sorted(iterable)` — PyPy: listobject.py listsort
///
/// Returns a new sorted list. Simplified: sorts by int value.
fn builtin_sorted(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(!args.is_empty(), "sorted() takes at least one argument");
    let iterable = args[0];
    let mut items = Vec::new();
    unsafe {
        if is_list(iterable) {
            let n = w_list_len(iterable);
            for i in 0..n {
                if let Some(item) = w_list_getitem(iterable, i as i64) {
                    items.push(item);
                }
            }
        } else if is_tuple(iterable) {
            let n = w_tuple_len(iterable);
            for i in 0..n {
                if let Some(item) = w_tuple_getitem(iterable, i as i64) {
                    items.push(item);
                }
            }
        } else {
            panic!("sorted() argument must be iterable");
        }
        items.sort_by(|a, b| {
            if is_int(*a) && is_int(*b) {
                w_int_get_value(*a).cmp(&w_int_get_value(*b))
            } else if is_str(*a) && is_str(*b) {
                w_str_get_value(*a).cmp(w_str_get_value(*b))
            } else {
                std::cmp::Ordering::Equal
            }
        });
    }
    Ok(w_list_new(items))
}

/// `any(iterable)` — PyPy: operation.py any
fn builtin_any(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(!args.is_empty(), "any() takes exactly one argument");
    let iterable = args[0];
    unsafe {
        if is_list(iterable) {
            let n = w_list_len(iterable);
            for i in 0..n {
                if let Some(item) = w_list_getitem(iterable, i as i64) {
                    if crate::space::py_is_true(item) {
                        return Ok(w_bool_from(true));
                    }
                }
            }
            return Ok(w_bool_from(false));
        }
        if is_tuple(iterable) {
            let n = w_tuple_len(iterable);
            for i in 0..n {
                if let Some(item) = w_tuple_getitem(iterable, i as i64) {
                    if crate::space::py_is_true(item) {
                        return Ok(w_bool_from(true));
                    }
                }
            }
            return Ok(w_bool_from(false));
        }
    }
    panic!("any() argument must be list or tuple");
}

/// `all(iterable)` — PyPy: operation.py all
fn builtin_all(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(!args.is_empty(), "all() takes exactly one argument");
    let iterable = args[0];
    unsafe {
        if is_list(iterable) {
            let n = w_list_len(iterable);
            for i in 0..n {
                if let Some(item) = w_list_getitem(iterable, i as i64) {
                    if !crate::space::py_is_true(item) {
                        return Ok(w_bool_from(false));
                    }
                }
            }
            return Ok(w_bool_from(true));
        }
        if is_tuple(iterable) {
            let n = w_tuple_len(iterable);
            for i in 0..n {
                if let Some(item) = w_tuple_getitem(iterable, i as i64) {
                    if !crate::space::py_is_true(item) {
                        return Ok(w_bool_from(false));
                    }
                }
            }
            return Ok(w_bool_from(true));
        }
    }
    panic!("all() argument must be list or tuple");
}

/// `sum(iterable, start=0)` — PyPy: operation.py sum
fn builtin_sum(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(!args.is_empty(), "sum() takes at least one argument");
    let iterable = args[0];
    let start = args.get(1).copied().unwrap_or_else(|| w_int_new(0));
    let mut acc = start;
    unsafe {
        if is_list(iterable) {
            let n = w_list_len(iterable);
            for i in 0..n {
                if let Some(item) = w_list_getitem(iterable, i as i64) {
                    acc = crate::space::py_add(acc, item).expect("sum: unsupported type");
                }
            }
            return Ok(acc);
        }
        if is_tuple(iterable) {
            let n = w_tuple_len(iterable);
            for i in 0..n {
                if let Some(item) = w_tuple_getitem(iterable, i as i64) {
                    acc = crate::space::py_add(acc, item).expect("sum: unsupported type");
                }
            }
            return Ok(acc);
        }
    }
    panic!("sum() argument must be list or tuple");
}

/// `round(number, ndigits=None)` — PyPy: operation.py round
fn builtin_round(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(!args.is_empty(), "round() takes at least one argument");
    let obj = args[0];
    let ndigits = args.get(1);
    unsafe {
        if is_float(obj) {
            let v = floatobject::w_float_get_value(obj);
            return Ok(match ndigits {
                Some(nd) if is_int(*nd) => {
                    let n = w_int_get_value(*nd);
                    let factor = 10f64.powi(n as i32);
                    floatobject::w_float_new((v * factor).round() / factor)
                }
                _ => w_int_new(v.round() as i64),
            });
        }
        if is_int(obj) {
            return Ok(obj);
        }
    }
    panic!("round() not supported for this type");
}

/// `divmod(a, b)` — PyPy: baseobjspace.py divmod
fn builtin_divmod(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() == 2, "divmod() takes exactly two arguments");
    unsafe {
        if is_int(args[0]) && is_int(args[1]) {
            let a = w_int_get_value(args[0]);
            let b = w_int_get_value(args[1]);
            assert!(b != 0, "integer division or modulo by zero");
            return Ok(w_tuple_new(vec![
                w_int_new(a.div_euclid(b)),
                w_int_new(a.rem_euclid(b)),
            ]));
        }
    }
    panic!("divmod() not supported for these types");
}

/// `pow(base, exp)` — PyPy: baseobjspace.py pow
fn builtin_pow(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() >= 2, "pow() takes at least two arguments");
    crate::space::py_pow(args[0], args[1])
}

/// `hex(x)` — PyPy: operation.py hex
fn builtin_hex(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() == 1, "hex() takes exactly one argument");
    let v = unsafe { w_int_get_value(args[0]) };
    let s = if v < 0 {
        format!("-0x{:x}", -v)
    } else {
        format!("0x{v:x}")
    };
    Ok(w_str_new(&s))
}

/// `oct(x)` — PyPy: operation.py oct
fn builtin_oct(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() == 1, "oct() takes exactly one argument");
    let v = unsafe { w_int_get_value(args[0]) };
    let s = if v < 0 {
        format!("-0o{:o}", -v)
    } else {
        format!("0o{v:o}")
    };
    Ok(w_str_new(&s))
}

/// `bin(x)` — PyPy: operation.py bin
fn builtin_bin(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() == 1, "bin() takes exactly one argument");
    let v = unsafe { w_int_get_value(args[0]) };
    let s = if v < 0 {
        format!("-0b{:b}", -v)
    } else {
        format!("0b{v:b}")
    };
    Ok(w_str_new(&s))
}

/// `format(value, format_spec='')` — PyPy: operation.py format
fn builtin_format(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(!args.is_empty(), "format() takes at least one argument");
    // Simplified: format without format_spec returns str(value)
    let s = unsafe { crate::py_str(args[0]) };
    Ok(w_str_new(&s))
}

/// `__import__()` — PyPy: pyopcode.py IMPORT_NAME invokes this
fn builtin_import_stub(_args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    panic!("__import__() not callable directly — use import statement");
}
