//! `pypy/module/__builtin__/descriptor.py` descriptor object ports.

#![allow(unsafe_op_in_unsafe_fn)]

use crate::pyobject::*;
use pyre_macros::pyre_class;

// W_Super — Python `super` proxy object.
//
// PyPy equivalent: pypy/module/__builtin__/descriptor.py W_Super
//
// Stores (super_type, obj) and resolves attribute lookups
// starting from the next class after super_type in obj's MRO.

/// super proxy: [ob_type | super_type (cls) | obj (self)]
#[pyre_class("super", type_id = 18, static_name = "SUPER")]
pub struct W_Super {
    /// The class passed to super() — lookup starts after this in MRO.
    pub super_type: PyObjectRef,
    /// The instance (self) or class for classmethod.
    pub obj: PyObjectRef,
}

/// Create a new super proxy.
pub fn w_super_new(super_type: PyObjectRef, obj: PyObjectRef) -> PyObjectRef {
    // `gct_fv_gc_malloc` bracket pattern (`framework.py:853-856`): pin the
    // `super_type`/`obj` pair across the GC malloc and re-read their
    // relocated addresses afterwards (a minor collection inside the malloc
    // may move them). A super proxy whose members are reachable only
    // through it must be GC-traced; a `malloc_typed` proxy is invisible to
    // mark-sweep, whereas `register_pyre_class` registers this layout's
    // `ptr_offsets`, so mark-sweep follows the members. The write barrier
    // below keeps the old-gen proxy in the remembered set so young members
    // survive a later minor collection.
    let _roots = crate::gc_roots::push_roots();
    let save_point = crate::gc_roots::shadow_stack_len();
    crate::gc_roots::pin_root(super_type);
    crate::gc_roots::pin_root(obj);

    let header = PyObject {
        ob_type: &SUPER_TYPE as *const PyType,
        w_class: get_instantiate(&SUPER_TYPE),
    };
    let raw = crate::gc_hook::try_gc_alloc_stable(W_SUPER_GC_TYPE_ID, W_SUPER_OBJECT_SIZE)
        .filter(|p| !p.is_null());
    let super_type = crate::gc_roots::shadow_stack_get(save_point);
    let obj = crate::gc_roots::shadow_stack_get(save_point + 1);
    if let Some(raw) = raw {
        unsafe {
            std::ptr::write(
                raw as *mut W_Super,
                W_Super {
                    ob: header,
                    super_type,
                    obj,
                },
            );
        }
        crate::gc_hook::try_gc_write_barrier(raw);
        return raw as PyObjectRef;
    }
    W_Super::allocate(W_Super {
        ob: header,
        super_type,
        obj,
    })
}

#[inline]
pub unsafe fn is_super(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &SUPER_TYPE) }
}

/// Get the super_type (cls) from a super proxy.
#[inline]
pub unsafe fn w_super_get_type(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_Super)).super_type }
}

/// Get the bound object (self) from a super proxy.
#[inline]
pub unsafe fn w_super_get_obj(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_Super)).obj }
}

#[cfg(test)]
mod super_tests {
    use super::*;

    #[test]
    fn w_super_gc_type_id_matches_descr() {
        assert_eq!(W_SUPER_GC_TYPE_ID, 18);
        assert_eq!(
            <W_Super as crate::lltype::GcType>::type_id(),
            W_SUPER_GC_TYPE_ID
        );
        assert_eq!(
            <W_Super as crate::lltype::GcType>::SIZE,
            W_SUPER_OBJECT_SIZE
        );
    }
}

// ── W_Property ─────────────────────────────────────────────────────

/// Python property descriptor object.
///
/// Layout: `[ob_type | fget | fset | fdel | w_doc | getter_doc]`
#[pyre_class("property", type_id = 19, static_name = "PROPERTY")]
pub struct W_Property {
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
                raw as *mut W_Property,
                W_Property {
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
    W_Property::allocate(W_Property {
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
    (*(obj as *const W_Property)).fget
}

pub unsafe fn w_property_get_fset(obj: PyObjectRef) -> PyObjectRef {
    (*(obj as *const W_Property)).fset
}

pub unsafe fn w_property_get_fdel(obj: PyObjectRef) -> PyObjectRef {
    (*(obj as *const W_Property)).fdel
}

/// `descriptor.py:249-250 W_Property.get_doc` — returns the raw slot
/// (NULL plays None; the caller wraps).
pub unsafe fn w_property_get_doc(obj: PyObjectRef) -> PyObjectRef {
    (*(obj as *const W_Property)).w_doc
}

/// `descriptor.py:252-254 W_Property.set_doc` — explicit doc writes
/// also clear `getter_doc`.
pub unsafe fn w_property_set_doc(obj: PyObjectRef, w_doc: PyObjectRef) {
    let prop = obj as *mut W_Property;
    (*prop).w_doc = w_doc;
    (*prop).getter_doc = false;
    // Record the old→young edge: `w_doc` is a traced slot and the
    // property may already have been promoted out of the nursery.
    crate::gc_hook::try_gc_write_barrier(obj as *mut u8);
}

/// `descriptor.py:199-204` — stamp a doc inherited from `fget.__doc__`
/// at construction time, marking `getter_doc`.
pub unsafe fn w_property_set_getter_doc(obj: PyObjectRef, w_doc: PyObjectRef) {
    let prop = obj as *mut W_Property;
    (*prop).w_doc = w_doc;
    (*prop).getter_doc = true;
    crate::gc_hook::try_gc_write_barrier(obj as *mut u8);
}

/// `self.w_name` — NULL plays unset.
pub unsafe fn w_property_get_name(obj: PyObjectRef) -> PyObjectRef {
    (*(obj as *const W_Property)).w_name
}

/// `descriptor.py:274-276 W_Property.set_name` — record the name the
/// property was assigned under.
pub unsafe fn w_property_set_name(obj: PyObjectRef, w_name: PyObjectRef) {
    let prop = obj as *mut W_Property;
    (*prop).w_name = w_name;
    crate::gc_hook::try_gc_write_barrier(obj as *mut u8);
}

#[inline]
pub unsafe fn is_property(obj: PyObjectRef) -> bool {
    py_type_check(obj, &PROPERTY_TYPE)
}

#[cfg(test)]
mod property_tests {
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
            <W_Property as crate::lltype::GcType>::type_id(),
            W_PROPERTY_GC_TYPE_ID
        );
        assert_eq!(
            <W_Property as crate::lltype::GcType>::SIZE,
            W_PROPERTY_OBJECT_SIZE
        );
    }
}
