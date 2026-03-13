use std::fmt;

use pyre_object::pyobject::{
    BOOL_TYPE, FLOAT_TYPE, INT_TYPE, LONG_TYPE, NONE_TYPE, PyObjectRef, PyType, STR_TYPE,
};

use crate::{BUILTIN_FUNC_TYPE, FUNCTION_TYPE, w_builtin_func_name, w_func_get_name};

/// Format a PyObjectRef for debug display.
///
/// # Safety
/// `obj` must be a valid pointer to a known Python object type.
pub unsafe fn py_repr(obj: PyObjectRef) -> String {
    if obj.is_null() {
        return "NULL".to_string();
    }
    unsafe {
        let tp = (*obj).ob_type;
        if std::ptr::eq(tp, &INT_TYPE as *const PyType) {
            let int_obj = obj as *const pyre_object::intobject::W_IntObject;
            format!("{}", (*int_obj).intval)
        } else if std::ptr::eq(tp, &FLOAT_TYPE as *const PyType) {
            let float_obj = obj as *const pyre_object::floatobject::W_FloatObject;
            let val = (*float_obj).floatval;
            if val.fract() == 0.0 && val.is_finite() {
                format!("{val:.1}")
            } else {
                format!("{val}")
            }
        } else if std::ptr::eq(tp, &LONG_TYPE as *const PyType) {
            let long_obj = obj as *const pyre_object::longobject::W_LongObject;
            format!("{}", &*(*long_obj).value)
        } else if std::ptr::eq(tp, &BOOL_TYPE as *const PyType) {
            let bool_obj = obj as *const pyre_object::boolobject::W_BoolObject;
            if (*bool_obj).boolval {
                "True".to_string()
            } else {
                "False".to_string()
            }
        } else if std::ptr::eq(tp, &STR_TYPE as *const PyType) {
            let str_obj = obj as *const pyre_object::strobject::W_StrObject;
            format!("'{}'", &*(*str_obj).value)
        } else if std::ptr::eq(tp, &NONE_TYPE as *const PyType) {
            "None".to_string()
        } else if std::ptr::eq(tp, &BUILTIN_FUNC_TYPE as *const PyType) {
            let name = w_builtin_func_name(obj);
            format!("<built-in function {name}>")
        } else if std::ptr::eq(tp, &FUNCTION_TYPE as *const PyType) {
            let name = w_func_get_name(obj);
            format!("<function {name}>")
        } else {
            format!("<{} object at {:?}>", (*tp).tp_name, obj)
        }
    }
}

/// Display wrapper for PyObjectRef.
pub struct PyDisplay(pub PyObjectRef);

impl fmt::Display for PyDisplay {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0.is_null() {
            write!(f, "NULL")
        } else {
            write!(f, "{}", unsafe { py_repr(self.0) })
        }
    }
}
