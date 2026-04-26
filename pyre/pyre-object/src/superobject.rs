//! W_Super — Python `super` proxy object.
//!
//! PyPy equivalent: pypy/objspace/descroperation.py + superobject.py
//!
//! Stores (super_type, obj) and resolves attribute lookups
//! starting from the next class after super_type in obj's MRO.

use crate::pyobject::*;

pub static SUPER_TYPE: PyType = crate::pyobject::new_pytype("super");

/// super proxy: [ob_type | super_type (cls) | obj (self)]
#[repr(C)]
pub struct W_SuperObject {
    pub ob: PyObject,
    /// The class passed to super() — lookup starts after this in MRO.
    pub super_type: PyObjectRef,
    /// The instance (self) or class for classmethod.
    pub obj: PyObjectRef,
}

/// Field offsets of inline `PyObjectRef` slots.
pub const SUPER_SUPER_TYPE_OFFSET: usize = std::mem::offset_of!(W_SuperObject, super_type);
pub const SUPER_OBJ_OFFSET: usize = std::mem::offset_of!(W_SuperObject, obj);

/// GC type id assigned to `W_SuperObject` at JitDriver init time.
pub const W_SUPER_GC_TYPE_ID: u32 = 18;

/// Fixed payload size (`framework.py:811`).
pub const W_SUPER_OBJECT_SIZE: usize = std::mem::size_of::<W_SuperObject>();

/// Byte offsets of the inline `PyObjectRef` fields the GC must trace.
pub const W_SUPER_GC_PTR_OFFSETS: [usize; 2] = [SUPER_SUPER_TYPE_OFFSET, SUPER_OBJ_OFFSET];

impl crate::lltype::GcType for W_SuperObject {
    const TYPE_ID: u32 = W_SUPER_GC_TYPE_ID;
    const SIZE: usize = W_SUPER_OBJECT_SIZE;
}

/// Create a new super proxy.
pub fn w_super_new(super_type: PyObjectRef, obj: PyObjectRef) -> PyObjectRef {
    // `gct_fv_gc_malloc` bracket pattern (`framework.py:853-856`).
    let _roots = crate::gc_roots::push_roots();
    crate::gc_roots::pin_root(super_type);
    crate::gc_roots::pin_root(obj);

    crate::lltype::malloc_typed(W_SuperObject {
        ob: PyObject {
            ob_type: &SUPER_TYPE as *const PyType,
            w_class: get_instantiate(&SUPER_TYPE),
        },
        super_type,
        obj,
    }) as PyObjectRef
}

#[inline]
pub unsafe fn is_super(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &SUPER_TYPE) }
}

/// Get the super_type (cls) from a super proxy.
#[inline]
pub unsafe fn w_super_get_type(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_SuperObject)).super_type }
}

/// Get the bound object (self) from a super proxy.
#[inline]
pub unsafe fn w_super_get_obj(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_SuperObject)).obj }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn w_super_gc_type_id_matches_descr() {
        assert_eq!(W_SUPER_GC_TYPE_ID, 18);
        assert_eq!(
            <W_SuperObject as crate::lltype::GcType>::TYPE_ID,
            W_SUPER_GC_TYPE_ID
        );
        assert_eq!(
            <W_SuperObject as crate::lltype::GcType>::SIZE,
            W_SUPER_OBJECT_SIZE
        );
    }
}
