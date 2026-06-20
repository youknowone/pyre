//! W_PropertyObject — Python `property` descriptor.
//!
//! PyPy equivalent: pypy/module/__builtin__/descriptor.py → W_Property
//!
//! A property holds fget, fset, fdel function references.
//! Used by the descriptor protocol in getattr/setattr.

#![allow(unsafe_op_in_unsafe_fn)]

use crate::pyobject::*;
use pyre_macros::pyre_class;

/// Python property descriptor object.
///
/// Layout: `[ob_type | fget | fset | fdel | w_doc | getter_doc]`
#[pyre_class("property", type_id = 19, static_name = "PROPERTY")]
pub struct W_PropertyObject {
    pub fget: PyObjectRef,
    pub fset: PyObjectRef,
    pub fdel: PyObjectRef,
    /// `descriptor.py:181 self.w_doc = space.w_None` — the instance
    /// `__doc__` exposed through `GetSetProperty(get_doc, set_doc)`
    /// (descriptor.py:316-318).  NULL plays None.
    pub w_doc: PyObjectRef,
    /// `descriptor.py:183 self.w_name = None` — set by `__set_name__`
    /// (descriptor.py:274-276) when the property is assigned as a class
    /// attribute.  Surfaced through `__name__` and woven into the
    /// `_properror` accessor messages.  NULL plays unset.
    pub w_name: PyObjectRef,
    /// `descriptor.py:182 self.getter_doc = False` — True when the doc
    /// was copied from `fget.__doc__` (descriptor.py:196-204); `_copy`
    /// uses it to drop the inherited doc when the getter is replaced.
    pub getter_doc: bool,
}

/// Allocate a new property object.
///
/// PyPy: W_Property.__init__(space, w_fget, w_fset, w_fdel, w_doc)
pub fn w_property_new(fget: PyObjectRef, fset: PyObjectRef, fdel: PyObjectRef) -> PyObjectRef {
    // `gct_fv_gc_malloc` bracket pattern (`framework.py:853-856`): pin the
    // three accessors across the GC malloc and read back relocated
    // addresses. A property whose `fget`/`fset`/`fdel` is reachable only
    // through it must be GC-traced; a `malloc_typed` property is invisible
    // to mark-sweep. The `w_doc`/`w_name` setters already carry the write
    // barrier (`set_doc`/`set_name`).
    let _roots = crate::gc_roots::push_roots();
    let save_point = crate::gc_roots::shadow_stack_len();
    crate::gc_roots::pin_root(fget);
    crate::gc_roots::pin_root(fset);
    crate::gc_roots::pin_root(fdel);

    let header = PyObject {
        ob_type: &PROPERTY_TYPE as *const PyType,
        w_class: get_instantiate(&PROPERTY_TYPE),
    };
    let raw = crate::gc_hook::try_gc_alloc_stable(W_PROPERTY_GC_TYPE_ID, W_PROPERTY_OBJECT_SIZE)
        .filter(|p| !p.is_null());
    let fget = crate::gc_roots::shadow_stack_get(save_point);
    let fset = crate::gc_roots::shadow_stack_get(save_point + 1);
    let fdel = crate::gc_roots::shadow_stack_get(save_point + 2);
    if let Some(raw) = raw {
        unsafe {
            std::ptr::write(
                raw as *mut W_PropertyObject,
                W_PropertyObject {
                    ob: header,
                    fget,
                    fset,
                    fdel,
                    w_doc: PY_NULL,
                    w_name: PY_NULL,
                    getter_doc: false,
                },
            );
        }
        crate::gc_hook::try_gc_write_barrier(raw);
        return raw as PyObjectRef;
    }
    W_PropertyObject::allocate(W_PropertyObject {
        ob: header,
        fget,
        fset,
        fdel,
        w_doc: PY_NULL,
        w_name: PY_NULL,
        getter_doc: false,
    })
}

pub unsafe fn w_property_get_fget(obj: PyObjectRef) -> PyObjectRef {
    (*(obj as *const W_PropertyObject)).fget
}

pub unsafe fn w_property_get_fset(obj: PyObjectRef) -> PyObjectRef {
    (*(obj as *const W_PropertyObject)).fset
}

pub unsafe fn w_property_get_fdel(obj: PyObjectRef) -> PyObjectRef {
    (*(obj as *const W_PropertyObject)).fdel
}

/// `descriptor.py:249-250 W_Property.get_doc` — returns the raw slot
/// (NULL plays None; the caller wraps).
pub unsafe fn w_property_get_doc(obj: PyObjectRef) -> PyObjectRef {
    (*(obj as *const W_PropertyObject)).w_doc
}

/// `descriptor.py:252-254 W_Property.set_doc` — explicit doc writes
/// also clear `getter_doc`.
pub unsafe fn w_property_set_doc(obj: PyObjectRef, w_doc: PyObjectRef) {
    let prop = obj as *mut W_PropertyObject;
    (*prop).w_doc = w_doc;
    (*prop).getter_doc = false;
    // Record the old→young edge: `w_doc` is a traced slot and the
    // property may already have been promoted out of the nursery.
    crate::gc_hook::try_gc_write_barrier(obj as *mut u8);
}

/// `descriptor.py:199-204` — stamp a doc inherited from `fget.__doc__`
/// at construction time, marking `getter_doc`.
pub unsafe fn w_property_set_getter_doc(obj: PyObjectRef, w_doc: PyObjectRef) {
    let prop = obj as *mut W_PropertyObject;
    (*prop).w_doc = w_doc;
    (*prop).getter_doc = true;
    crate::gc_hook::try_gc_write_barrier(obj as *mut u8);
}

/// `self.w_name` — NULL plays unset.
pub unsafe fn w_property_get_name(obj: PyObjectRef) -> PyObjectRef {
    (*(obj as *const W_PropertyObject)).w_name
}

/// `descriptor.py:274-276 W_Property.set_name` — record the name the
/// property was assigned under.
pub unsafe fn w_property_set_name(obj: PyObjectRef, w_name: PyObjectRef) {
    let prop = obj as *mut W_PropertyObject;
    (*prop).w_name = w_name;
    crate::gc_hook::try_gc_write_barrier(obj as *mut u8);
}

#[inline]
pub unsafe fn is_property(obj: PyObjectRef) -> bool {
    py_type_check(obj, &PROPERTY_TYPE)
}

// ── StaticMethod ─────────────────────────────────────────────────────
// PyPy: pypy/interpreter/function.py StaticMethod
//
// __get__ returns the wrapped function unchanged (no self binding).

/// Python staticmethod descriptor.
#[pyre_class("staticmethod", type_id = 20, static_name = "STATICMETHOD")]
pub struct W_StaticMethodObject {
    pub w_function: PyObjectRef,
}

pub fn w_staticmethod_new(func: PyObjectRef) -> PyObjectRef {
    // `gct_fv_gc_malloc` bracket pattern (`framework.py:853-856`): pin the
    // wrapped function across the GC malloc and read its relocated address.
    let _roots = crate::gc_roots::push_roots();
    let save_point = crate::gc_roots::shadow_stack_len();
    crate::gc_roots::pin_root(func);

    let header = PyObject {
        ob_type: &STATICMETHOD_TYPE as *const PyType,
        w_class: get_instantiate(&STATICMETHOD_TYPE),
    };
    let raw =
        crate::gc_hook::try_gc_alloc_stable(W_STATICMETHOD_GC_TYPE_ID, W_STATICMETHOD_OBJECT_SIZE)
            .filter(|p| !p.is_null());
    let func = crate::gc_roots::shadow_stack_get(save_point);
    if let Some(raw) = raw {
        unsafe {
            std::ptr::write(
                raw as *mut W_StaticMethodObject,
                W_StaticMethodObject {
                    ob: header,
                    w_function: func,
                },
            );
        }
        crate::gc_hook::try_gc_write_barrier(raw);
        return raw as PyObjectRef;
    }
    W_StaticMethodObject::allocate(W_StaticMethodObject {
        ob: header,
        w_function: func,
    })
}

pub unsafe fn w_staticmethod_get_func(obj: PyObjectRef) -> PyObjectRef {
    (*(obj as *const W_StaticMethodObject)).w_function
}

#[inline]
pub unsafe fn is_staticmethod(obj: PyObjectRef) -> bool {
    py_type_check(obj, &STATICMETHOD_TYPE)
}

// ── ClassMethod ──────────────────────────────────────────────────────
// PyPy: pypy/interpreter/function.py ClassMethod
//
// __get__ returns a bound method with the class as first arg.

/// Python classmethod descriptor.
#[pyre_class("classmethod", type_id = 21, static_name = "CLASSMETHOD")]
pub struct W_ClassMethodObject {
    pub w_function: PyObjectRef,
}

pub fn w_classmethod_new(func: PyObjectRef) -> PyObjectRef {
    // `gct_fv_gc_malloc` bracket pattern (`framework.py:853-856`): pin the
    // wrapped function across the GC malloc and read its relocated address.
    let _roots = crate::gc_roots::push_roots();
    let save_point = crate::gc_roots::shadow_stack_len();
    crate::gc_roots::pin_root(func);

    let header = PyObject {
        ob_type: &CLASSMETHOD_TYPE as *const PyType,
        w_class: get_instantiate(&CLASSMETHOD_TYPE),
    };
    let raw =
        crate::gc_hook::try_gc_alloc_stable(W_CLASSMETHOD_GC_TYPE_ID, W_CLASSMETHOD_OBJECT_SIZE)
            .filter(|p| !p.is_null());
    let func = crate::gc_roots::shadow_stack_get(save_point);
    if let Some(raw) = raw {
        unsafe {
            std::ptr::write(
                raw as *mut W_ClassMethodObject,
                W_ClassMethodObject {
                    ob: header,
                    w_function: func,
                },
            );
        }
        crate::gc_hook::try_gc_write_barrier(raw);
        return raw as PyObjectRef;
    }
    W_ClassMethodObject::allocate(W_ClassMethodObject {
        ob: header,
        w_function: func,
    })
}

pub unsafe fn w_classmethod_get_func(obj: PyObjectRef) -> PyObjectRef {
    (*(obj as *const W_ClassMethodObject)).w_function
}

#[inline]
pub unsafe fn is_classmethod(obj: PyObjectRef) -> bool {
    py_type_check(obj, &CLASSMETHOD_TYPE)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_property_create() {
        let obj = w_property_new(PY_NULL, PY_NULL, PY_NULL);
        unsafe {
            assert!(is_property(obj));
            assert!(!is_int(obj));
        }
    }

    #[test]
    fn w_property_gc_type_id_matches_descr() {
        assert_eq!(W_PROPERTY_GC_TYPE_ID, 19);
        assert_eq!(
            <W_PropertyObject as crate::lltype::GcType>::type_id(),
            W_PROPERTY_GC_TYPE_ID
        );
        assert_eq!(
            <W_PropertyObject as crate::lltype::GcType>::SIZE,
            W_PROPERTY_OBJECT_SIZE
        );
    }

    #[test]
    fn w_staticmethod_gc_type_id_matches_descr() {
        assert_eq!(W_STATICMETHOD_GC_TYPE_ID, 20);
        assert_eq!(
            <W_StaticMethodObject as crate::lltype::GcType>::type_id(),
            W_STATICMETHOD_GC_TYPE_ID
        );
        assert_eq!(
            <W_StaticMethodObject as crate::lltype::GcType>::SIZE,
            W_STATICMETHOD_OBJECT_SIZE
        );
    }

    #[test]
    fn w_classmethod_gc_type_id_matches_descr() {
        assert_eq!(W_CLASSMETHOD_GC_TYPE_ID, 21);
        assert_eq!(
            <W_ClassMethodObject as crate::lltype::GcType>::type_id(),
            W_CLASSMETHOD_GC_TYPE_ID
        );
        assert_eq!(
            <W_ClassMethodObject as crate::lltype::GcType>::SIZE,
            W_CLASSMETHOD_OBJECT_SIZE
        );
    }
}
