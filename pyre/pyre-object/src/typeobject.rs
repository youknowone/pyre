//! W_TypeObject — Python `type` object for user-defined classes.
//!
//! PyPy equivalent: pypy/objspace/std/typeobject.py → W_TypeObject
//!
//! A type object holds the class name, tuple of base types, and a namespace
//! dict containing class-level attributes and methods.

#![allow(unsafe_op_in_unsafe_fn)]

use crate::pyobject::*;

/// Python type object (user-defined class).
///
/// PyPy: pypy/objspace/std/typeobject.py W_TypeObject
///
/// - `name`: heap-allocated class name string
/// - `bases`: tuple of base type objects (PyObjectRef to tuple)
/// - `dict`: raw pointer to PyNamespace (class methods/attrs)
/// - `flag_heaptype`: typeobject.py:544 — True for dynamically created types
///   (class statement / type()), False for builtin types
#[repr(C)]
pub struct W_TypeObject {
    pub ob_header: PyObject,
    /// Class name (heap-allocated, leaked).
    pub name: *mut String,
    /// Tuple of base type objects (PyObjectRef → W_TupleObject or PY_NULL).
    pub bases: PyObjectRef,
    /// Raw pointer to the class namespace (PyNamespace from pyre-interpreter).
    pub dict: *mut u8,
    /// Cached C3 MRO — W_TypeObject.mro_w.
    /// Computed once at type creation and cached.
    pub mro_w: *mut Vec<PyObjectRef>,
    /// typeobject.py:144,184 `flag_heaptype` — immutable after creation.
    /// True for user-defined classes (class statement / type()),
    /// False for builtin types created by init_typeobjects.
    pub flag_heaptype: bool,
    /// typeobject.py:153,336 `layout` — immutable after creation.
    ///
    /// Points to the PyType that describes the RPython-level instance
    /// layout. For user-defined classes this is `&INSTANCE_TYPE`;
    /// for builtin subclasses (e.g. int subclass) it's `&INT_TYPE`.
    /// `get_full_instance_layout()` compares this pointer to decide
    /// whether `__class__` reassignment is safe.
    ///
    /// Corresponds to `Layout.typedef` in PyPy's typeobject.py:113.
    pub layout: *const PyType,
    /// typeobject.py:114 `Layout.nslots` — number of `__slots__` entries.
    /// 0 for classes without `__slots__`. Used in layout comparison:
    /// classes with different nslots have incompatible layouts.
    pub nslots: u32,
    /// typeobject.py:115 `Layout.newslotnames` — sorted list of slot names
    /// introduced by THIS class (not inherited from bases).
    /// Used in Layout.expand() tuple for __class__ assignment compatibility.
    /// Null means empty (no __slots__ or all slots inherited).
    pub newslotnames: *const Vec<String>,
    /// typeobject.py:116 `Layout.base_layout` — pointer to the best base
    /// type's W_TypeObject. Identity comparison in Layout.expand().
    /// PY_NULL for root types (object, int, etc.).
    pub base_layout: PyObjectRef,
    /// typeobject.py:179 `hasdict` — True when instances have __dict__.
    /// True for classes without __slots__, or with "__dict__" in __slots__.
    pub hasdict: bool,
    /// typeobject.py:181 `weakrefable` — True when instances support weakrefs.
    /// True for classes without __slots__, or with "__weakref__" in __slots__.
    pub weakrefable: bool,
}

/// Allocate a new W_TypeObject.
///
/// PyPy equivalent: W_TypeObject.__init__(space, name, bases_w, dict_w)
/// Allocate a new W_TypeObject with `flag_heaptype = true`.
///
/// typeobject.py:174 `__init__(..., is_heaptype=True)` — dynamically
/// created types (class statement / type()) are heap types.
/// Layout is `INSTANCE_TYPE` (all user-defined instances share
/// the same RPython-level struct layout).
pub fn w_type_new(name: &str, bases: PyObjectRef, dict_ptr: *mut u8) -> PyObjectRef {
    let obj = Box::new(W_TypeObject {
        ob_header: PyObject {
            ob_type: &TYPE_TYPE as *const PyType,
            w_class: std::ptr::null_mut(),
        },
        mro_w: std::ptr::null_mut(), // set after construction via set_mro
        name: Box::into_raw(Box::new(name.to_string())),
        bases,
        dict: dict_ptr,
        flag_heaptype: true,
        layout: &INSTANCE_TYPE as *const PyType,
        nslots: 0,
        newslotnames: std::ptr::null(),
        base_layout: std::ptr::null_mut(),
        hasdict: true,
        weakrefable: true,
    });
    Box::into_raw(obj) as PyObjectRef
}

/// Set the number of `__slots__` entries on a type object.
///
/// # Safety
/// `obj` must be a valid W_TypeObject pointer.
pub unsafe fn w_type_set_nslots(obj: PyObjectRef, n: u32) {
    (*(obj as *mut W_TypeObject)).nslots = n;
}

/// Get the number of `__slots__` entries on a type object.
///
/// # Safety
/// `obj` must be a valid W_TypeObject pointer.
#[inline]
pub unsafe fn w_type_get_nslots(obj: PyObjectRef) -> u32 {
    (*(obj as *const W_TypeObject)).nslots
}

/// Allocate a new W_TypeObject with `flag_heaptype = false`.
///
/// typeobject.py:174 `__init__(..., is_heaptype=False)` — builtin types
/// created by init_typeobjects are not heap types.
/// `layout_pytype` specifies which PyType describes the instance layout.
pub fn w_type_new_builtin(
    name: &str,
    bases: PyObjectRef,
    dict_ptr: *mut u8,
    layout_pytype: *const PyType,
) -> PyObjectRef {
    let obj = Box::new(W_TypeObject {
        ob_header: PyObject {
            ob_type: &TYPE_TYPE as *const PyType,
            w_class: std::ptr::null_mut(),
        },
        mro_w: std::ptr::null_mut(),
        name: Box::into_raw(Box::new(name.to_string())),
        bases,
        dict: dict_ptr,
        flag_heaptype: false,
        layout: layout_pytype,
        nslots: 0,
        newslotnames: std::ptr::null(),
        base_layout: std::ptr::null_mut(),
        hasdict: false,
        weakrefable: false,
    });
    Box::into_raw(obj) as PyObjectRef
}

/// Get the class name.
pub unsafe fn w_type_get_name(obj: PyObjectRef) -> &'static str {
    &*(*(obj as *const W_TypeObject)).name
}

/// Get the bases tuple.
pub unsafe fn w_type_get_bases(obj: PyObjectRef) -> PyObjectRef {
    (*(obj as *const W_TypeObject)).bases
}

/// Get the class namespace pointer (as *mut u8).
pub unsafe fn w_type_get_dict_ptr(obj: PyObjectRef) -> *mut u8 {
    (*(obj as *const W_TypeObject)).dict
}

/// Get the cached MRO, or null if not yet set.
pub unsafe fn w_type_get_mro(obj: PyObjectRef) -> *mut Vec<PyObjectRef> {
    (*(obj as *const W_TypeObject)).mro_w
}

/// Set the cached MRO.
pub unsafe fn w_type_set_mro(obj: PyObjectRef, mro: Vec<PyObjectRef>) {
    (*(obj as *mut W_TypeObject)).mro_w = Box::into_raw(Box::new(mro));
}

/// Check if an object is a type (user-defined class).
#[inline]
pub unsafe fn is_type(obj: PyObjectRef) -> bool {
    py_type_check(obj, &TYPE_TYPE)
}

/// typeobject.py:115 `Layout.newslotnames` setter.
/// `names` must be sorted (PyPy's create_all_slots sorts them).
///
/// # Safety
/// `obj` must be a valid W_TypeObject pointer.
pub unsafe fn w_type_set_newslotnames(obj: PyObjectRef, names: Vec<String>) {
    (*(obj as *mut W_TypeObject)).newslotnames = Box::into_raw(Box::new(names));
}

/// typeobject.py:115 `Layout.newslotnames` getter.
///
/// # Safety
/// `obj` must be a valid W_TypeObject pointer.
pub unsafe fn w_type_get_newslotnames(obj: PyObjectRef) -> &'static [String] {
    let ptr = (*(obj as *const W_TypeObject)).newslotnames;
    if ptr.is_null() { &[] } else { &*ptr }
}

/// typeobject.py:116 `Layout.base_layout` setter.
pub unsafe fn w_type_set_base_layout(obj: PyObjectRef, base: PyObjectRef) {
    (*(obj as *mut W_TypeObject)).base_layout = base;
}

/// typeobject.py:116 `Layout.base_layout` getter.
pub unsafe fn w_type_get_base_layout(obj: PyObjectRef) -> PyObjectRef {
    (*(obj as *const W_TypeObject)).base_layout
}

/// typeobject.py:179 `hasdict` setter.
pub unsafe fn w_type_set_hasdict(obj: PyObjectRef, v: bool) {
    (*(obj as *mut W_TypeObject)).hasdict = v;
}

/// typeobject.py:179 `hasdict` getter.
pub unsafe fn w_type_get_hasdict(obj: PyObjectRef) -> bool {
    (*(obj as *const W_TypeObject)).hasdict
}

/// typeobject.py:181 `weakrefable` setter.
pub unsafe fn w_type_set_weakrefable(obj: PyObjectRef, v: bool) {
    (*(obj as *mut W_TypeObject)).weakrefable = v;
}

/// typeobject.py:181 `weakrefable` getter.
pub unsafe fn w_type_get_weakrefable(obj: PyObjectRef) -> bool {
    (*(obj as *const W_TypeObject)).weakrefable
}

/// typeobject.py:543-544 `is_heaptype(self)`.
///
/// # Safety
/// `obj` must be a valid W_TypeObject pointer.
#[inline]
pub unsafe fn w_type_is_heaptype(obj: PyObjectRef) -> bool {
    (*(obj as *const W_TypeObject)).flag_heaptype
}

/// typeobject.py:336-337 `get_full_instance_layout(self)`.
///
/// Returns the layout key (PyType pointer) that describes the instance's
/// RPython-level struct layout. Two types have compatible layouts iff
/// their layout keys are identical.
///
/// # Safety
/// `obj` must be a valid W_TypeObject pointer.
#[inline]
pub unsafe fn w_type_get_layout(obj: PyObjectRef) -> *const PyType {
    (*(obj as *const W_TypeObject)).layout
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_type_create_and_check() {
        let obj = w_type_new("Foo", PY_NULL, std::ptr::null_mut());
        unsafe {
            assert!(is_type(obj));
            assert!(!is_int(obj));
            assert_eq!(w_type_get_name(obj), "Foo");
            assert!(w_type_get_dict_ptr(obj).is_null());
        }
    }
}
