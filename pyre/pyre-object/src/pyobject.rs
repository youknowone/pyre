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
/// RPython rclass.py: OBJECT = GcStruct('object', ('typeptr', CLASSTYPE))
///
/// - `ob_type`: static dispatch tag (like RPython's typeptr for guard_class)
/// - `w_class`: Python class pointer (like RPython's gettypefor(typeptr) result)
///
/// `w_class` is set at allocation time when the type registry is available,
/// or populated lazily by `init_typeobjects()` for static singletons.
#[repr(C)]
pub struct PyObject {
    pub ob_type: *const PyType,
    pub w_class: *mut PyObject,
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
pub static NOTIMPLEMENTED_TYPE: PyType = PyType {
    tp_name: "NotImplementedType",
};
pub static MODULE_TYPE: PyType = PyType { tp_name: "module" };
pub static TYPE_TYPE: PyType = PyType { tp_name: "type" };
pub static INSTANCE_TYPE: PyType = PyType { tp_name: "object" };

/// Field offset of `ob_type` within PyObject, for JIT field access.
pub const OB_TYPE_OFFSET: usize = std::mem::offset_of!(PyObject, ob_type);

/// Field offset of `w_class` within PyObject, for JIT field access.
/// RPython: this corresponds to reading typeptr + gettypefor (fused into one field).
pub const W_CLASS_OFFSET: usize = std::mem::offset_of!(PyObject, w_class);

/// Every built-in `PyType` static that represents a full `PyObject`
/// subtype (i.e. instances carry `ob_type` at offset 0, matching
/// `rclass.OBJECT` layout), paired with its parent class.
///
/// Modelled on RPython's `assign_inheritance_ids`
/// (normalizecalls.py:373-389) which walks `classdef.getmro()` to build
/// the reversed-MRO witness for each class. The JIT registers each
/// `(type, parent)` pair with the GC via `register_vtable_for_type`,
/// using the parent typeid as `TypeInfo::object_subclass`'s `parent`
/// argument so the resulting `subclassrange_{min,max}` faithfully
/// represents the `rclass.OBJECT` hierarchy. `GUARD_SUBCLASS` then
/// resolves to `int_between(cls.min, subcls.min, cls.max)` per
/// rclass.py:1133-1137 `ll_issubclass`.
///
/// `INSTANCE_TYPE` (the `tp_name = "object"` root) is intentionally
/// absent: it is registered separately as the `rclass.OBJECT` root
/// with no parent. `INT_TYPE` and `FLOAT_TYPE` are also absent: they
/// get their own ids (`W_INT_GC_TYPE_ID` / `W_FLOAT_GC_TYPE_ID`)
/// because the JIT backend allocates W_IntObject / W_FloatObject
/// through NewWithVtable and needs the correct payload size.
pub fn all_foreign_pytypes() -> &'static [(&'static PyType, &'static PyType)] {
    static PYTYPES: &[(&PyType, &PyType)] = &[
        // bool inherits from int (objectobject.py W_BoolObject.typedef).
        (&BOOL_TYPE, &INT_TYPE),
        (&STR_TYPE, &INSTANCE_TYPE),
        (&LIST_TYPE, &INSTANCE_TYPE),
        (&TUPLE_TYPE, &INSTANCE_TYPE),
        (&DICT_TYPE, &INSTANCE_TYPE),
        // longobject.py W_LongObject — Python 3 unifies long under int,
        // but pyre carries a separate static for the BigInt-backed flavour.
        (&LONG_TYPE, &INSTANCE_TYPE),
        (&NONE_TYPE, &INSTANCE_TYPE),
        (&NOTIMPLEMENTED_TYPE, &INSTANCE_TYPE),
        (&MODULE_TYPE, &INSTANCE_TYPE),
        (&TYPE_TYPE, &INSTANCE_TYPE),
        (&crate::superobject::SUPER_TYPE, &INSTANCE_TYPE),
        (&crate::bytearrayobject::BYTEARRAY_TYPE, &INSTANCE_TYPE),
        (&crate::generatorobject::GENERATOR_TYPE, &INSTANCE_TYPE),
        (&crate::unionobject::UNION_TYPE, &INSTANCE_TYPE),
        (&crate::rangeobject::RANGE_ITER_TYPE, &INSTANCE_TYPE),
        (&crate::rangeobject::SEQ_ITER_TYPE, &INSTANCE_TYPE),
        (&crate::cellobject::CELL_TYPE, &INSTANCE_TYPE),
        (&crate::methodobject::METHOD_TYPE, &INSTANCE_TYPE),
        (&crate::propertyobject::PROPERTY_TYPE, &INSTANCE_TYPE),
        (&crate::propertyobject::STATICMETHOD_TYPE, &INSTANCE_TYPE),
        (&crate::propertyobject::CLASSMETHOD_TYPE, &INSTANCE_TYPE),
        (&crate::excobject::EXCEPTION_TYPE, &INSTANCE_TYPE),
        (&crate::sliceobject::SLICE_TYPE, &INSTANCE_TYPE),
    ];
    PYTYPES
}

// ── Type checks ───────────────────────────────────────────────────────

/// Check if an object is of a given type (pointer identity comparison).
///
/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn py_type_check(obj: PyObjectRef, tp: &PyType) -> bool {
    !obj.is_null() && unsafe { std::ptr::eq((*obj).ob_type, tp as *const PyType) }
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

#[inline]
pub unsafe fn is_not_implemented(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &NOTIMPLEMENTED_TYPE) }
}
