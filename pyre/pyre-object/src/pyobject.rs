//! Core Python object model with `#[repr(C)]` layout for JIT compatibility.
//!
//! Every Python object starts with a `PyObject` header containing a type pointer.
//! Concrete types (W_IntObject, W_BoolObject, etc.) embed this header as their
//! first field, enabling safe pointer casts between `*mut PyObject` and typed pointers.

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
/// JIT code reads this field as a raw machine word and guards on it via `GuardClass`.
#[repr(C)]
pub struct PyObject {
    pub ob_type: *const PyType,
}

/// The universal Python object reference — a raw pointer to `PyObject`.
///
/// `pyre` currently passes this through the JIT as an integer-sized raw pointer.
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
pub static FLOAT_TYPE: PyType = PyType { tp_name: "float" };
pub static STR_TYPE: PyType = PyType { tp_name: "str" };
pub static LIST_TYPE: PyType = PyType { tp_name: "list" };
pub static TUPLE_TYPE: PyType = PyType { tp_name: "tuple" };
pub static DICT_TYPE: PyType = PyType { tp_name: "dict" };
pub static LONG_TYPE: PyType = PyType { tp_name: "int" };
pub static NONE_TYPE: PyType = PyType {
    tp_name: "NoneType",
};
pub static MODULE_TYPE: PyType = PyType { tp_name: "module" };
pub static TYPE_TYPE: PyType = PyType { tp_name: "type" };
pub static INSTANCE_TYPE: PyType = PyType { tp_name: "object" };

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
pub unsafe fn is_float(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &FLOAT_TYPE) }
}

#[inline]
pub unsafe fn is_long(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &LONG_TYPE) }
}

#[inline]
pub unsafe fn is_int_or_long(obj: PyObjectRef) -> bool {
    unsafe { is_int(obj) || is_long(obj) }
}

#[inline]
pub unsafe fn is_list(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &LIST_TYPE) }
}

#[inline]
pub unsafe fn is_tuple(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &TUPLE_TYPE) }
}

#[inline]
pub unsafe fn is_dict(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &DICT_TYPE) }
}

#[inline]
pub unsafe fn is_none(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &NONE_TYPE) }
}
