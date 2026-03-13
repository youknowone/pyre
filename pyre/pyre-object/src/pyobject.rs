//! Core Python object model with `#[repr(C)]` layout for JIT compatibility.
//!
//! Every Python object starts with a `PyObject` header containing a type pointer.
//! Concrete types (W_IntObject, W_BoolObject, etc.) embed this header as their
//! first field, enabling safe pointer casts between `*mut PyObject` and typed pointers.

use std::fmt;

/// Type descriptor for Python objects.
///
/// Each built-in type has a single static `PyType` instance.
/// The JIT uses `GuardClass` on the `ob_type` pointer to specialize code paths.
#[repr(C)]
pub struct PyType {
    pub tp_name: &'static str,
}

/// Common header for all Python objects.
///
/// The `ob_type` pointer identifies the object's type at runtime.
/// JIT code reads this field via `GetfieldGcR` and guards on it via `GuardClass`.
#[repr(C)]
pub struct PyObject {
    pub ob_type: *const PyType,
}

/// The universal Python object reference — a raw pointer to `PyObject`.
///
/// Maps directly to majit's `Type::Ref` / `GcRef`.
/// Phase 1 uses leaked Box allocations; a proper GC will replace this later.
// Safety: PyType instances are read-only static data, safe to share across threads.
unsafe impl Sync for PyType {}
unsafe impl Send for PyType {}

// Safety: PyObject's ob_type points to immutable static PyType instances.
unsafe impl Sync for PyObject {}
unsafe impl Send for PyObject {}

pub type PyObjectRef = *mut PyObject;

/// Null object reference, used as a sentinel for "no value".
pub const PY_NULL: PyObjectRef = std::ptr::null_mut();

// ── Type identity ─────────────────────────────────────────────────────

pub static INT_TYPE: PyType = PyType { tp_name: "int" };
pub static BOOL_TYPE: PyType = PyType { tp_name: "bool" };
pub static NONE_TYPE: PyType = PyType {
    tp_name: "NoneType",
};

/// Field offset of `ob_type` within PyObject, for JIT field access.
pub const OB_TYPE_OFFSET: usize = std::mem::offset_of!(PyObject, ob_type);

// ── Type checks ───────────────────────────────────────────────────────

/// Check if an object is of a given type (pointer identity comparison).
///
/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn py_type_check(obj: PyObjectRef, tp: &PyType) -> bool {
    unsafe { std::ptr::eq((*obj).ob_type, tp as *const PyType) }
}

#[inline]
pub unsafe fn is_int(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &INT_TYPE) }
}

#[inline]
pub unsafe fn is_bool(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &BOOL_TYPE) }
}

#[inline]
pub unsafe fn is_none(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &NONE_TYPE) }
}

// ── Debug formatting ──────────────────────────────────────────────────

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
            let int_obj = obj as *const super::intobject::W_IntObject;
            format!("{}", (*int_obj).intval)
        } else if std::ptr::eq(tp, &BOOL_TYPE as *const PyType) {
            let bool_obj = obj as *const super::boolobject::W_BoolObject;
            if (*bool_obj).boolval {
                "True".to_string()
            } else {
                "False".to_string()
            }
        } else if std::ptr::eq(tp, &NONE_TYPE as *const PyType) {
            "None".to_string()
        } else if std::ptr::eq(tp, &super::builtinfunc::BUILTIN_FUNC_TYPE as *const PyType) {
            let name = super::builtinfunc::w_builtin_func_name(obj);
            format!("<built-in function {name}>")
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
            // Safety: caller ensures the pointer is valid
            write!(f, "{}", unsafe { py_repr(self.0) })
        }
    }
}
