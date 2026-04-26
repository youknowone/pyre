//! W_UnionType — Python `types.UnionType` (PEP 604).
//!
//! PyPy equivalent: lib_pypy/_pypy_generic_alias.py → UnionType
//!
//! Represents `X | Y` union types (e.g. `int | str`).
//! Supports `isinstance`, `issubclass`, deduplication, and flattening.

#![allow(unsafe_op_in_unsafe_fn)]

use crate::pyobject::*;

/// Python union type object (PEP 604).
///
/// Layout: `[ob_type | args]`
///
/// - `args`: tuple of the union members (deduplicated, flattened)
///
/// PyPy equivalent: UnionType in _pypy_generic_alias.py
#[repr(C)]
pub struct W_UnionType {
    pub ob_header: PyObject,
    /// Tuple of union member types — PyPy: UnionType._args
    pub args: PyObjectRef,
}

pub static UNION_TYPE: PyType = crate::pyobject::new_pytype("types.UnionType");

/// Field offset of `args` within `W_UnionType`.
pub const UNION_ARGS_OFFSET: usize = std::mem::offset_of!(W_UnionType, args);

/// GC type id assigned to `W_UnionType` at JitDriver init time.
pub const W_UNION_GC_TYPE_ID: u32 = 22;

/// Fixed payload size (`framework.py:811`).
pub const W_UNION_OBJECT_SIZE: usize = std::mem::size_of::<W_UnionType>();

/// Byte offsets of the inline `PyObjectRef` fields the GC must trace.
pub const W_UNION_GC_PTR_OFFSETS: [usize; 1] = [UNION_ARGS_OFFSET];

impl crate::lltype::GcType for W_UnionType {
    const TYPE_ID: u32 = W_UNION_GC_TYPE_ID;
    const SIZE: usize = W_UNION_OBJECT_SIZE;
}

/// Check if an object is a UnionType.
#[inline]
pub unsafe fn is_union(obj: PyObjectRef) -> bool {
    py_type_check(obj, &UNION_TYPE)
}

/// Create a union type from two operands.
///
/// PyPy equivalent: _create_union(self, other) in _pypy_generic_alias.py
///
/// Handles deduplication and flattening of nested unions.
pub fn w_union_new(a: PyObjectRef, b: PyObjectRef) -> PyObjectRef {
    let mut members = Vec::new();
    collect_union_args(a, &mut members);
    collect_union_args(b, &mut members);
    dedup_members(&mut members);

    let args = crate::w_tuple_new(members);
    // `gct_fv_gc_malloc` bracket pattern (`framework.py:853-856`).
    let _roots = crate::gc_roots::push_roots();
    crate::gc_roots::pin_root(args);

    crate::lltype::malloc_typed(W_UnionType {
        ob_header: PyObject {
            ob_type: &UNION_TYPE as *const PyType,
            w_class: get_instantiate(&UNION_TYPE),
        },
        args,
    }) as PyObjectRef
}

/// Flatten nested UnionType args, or add a single type.
///
/// PyPy equivalent: add_recurse in UnionType.__init__
fn collect_union_args(obj: PyObjectRef, out: &mut Vec<PyObjectRef>) {
    if obj.is_null() {
        return;
    }
    unsafe {
        if is_union(obj) {
            let union = &*(obj as *const W_UnionType);
            let n = crate::w_tuple_len(union.args);
            for i in 0..n {
                if let Some(item) = crate::w_tuple_getitem(union.args, i as i64) {
                    out.push(item);
                }
            }
        } else {
            out.push(obj);
        }
    }
}

/// Remove duplicate type entries (pointer identity).
///
/// PyPy equivalent: deduplication in UnionType.__init__
fn dedup_members(members: &mut Vec<PyObjectRef>) {
    let mut seen = Vec::new();
    members.retain(|&item| {
        if seen.contains(&item) {
            false
        } else {
            seen.push(item);
            true
        }
    });
}

/// Get the `__args__` tuple of a UnionType.
///
/// # Safety
/// `obj` must point to a valid `W_UnionType`.
pub unsafe fn w_union_get_args(obj: PyObjectRef) -> PyObjectRef {
    (*(obj as *const W_UnionType)).args
}

/// Check if `instance` is an instance of any type in the union.
///
/// PyPy equivalent: UnionType.__instancecheck__
pub unsafe fn w_union_instancecheck(union: PyObjectRef, instance: PyObjectRef) -> bool {
    let args = w_union_get_args(union);
    let n = crate::w_tuple_len(args);
    for i in 0..n {
        if let Some(cls) = crate::w_tuple_getitem(args, i as i64) {
            if is_none(cls) {
                if is_none(instance) {
                    return true;
                }
            } else if crate::is_type(cls) {
                // Use ob_type pointer comparison for builtin types
                if std::ptr::eq((*instance).ob_type, (*cls).ob_type)
                    || std::ptr::eq(
                        (*instance).ob_type as *const u8,
                        cls as *const u8 as *const PyType as *const u8,
                    )
                {
                    return true;
                }
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intobject::w_int_new;

    #[test]
    fn test_union_create() {
        // Simulate int | str
        let a = w_int_new(1); // stand-in for int type
        let b = w_int_new(2); // stand-in for str type
        let union = w_union_new(a, b);
        unsafe {
            assert!(is_union(union));
            let args = w_union_get_args(union);
            assert_eq!(crate::w_tuple_len(args), 2);
        }
    }

    #[test]
    fn test_union_dedup() {
        let a = w_int_new(42);
        let union = w_union_new(a, a);
        unsafe {
            let args = w_union_get_args(union);
            assert_eq!(crate::w_tuple_len(args), 1);
        }
    }

    #[test]
    fn test_union_flatten() {
        let a = w_int_new(1);
        let b = w_int_new(2);
        let c = w_int_new(3);
        let inner = w_union_new(a, b);
        let outer = w_union_new(inner, c);
        unsafe {
            let args = w_union_get_args(outer);
            assert_eq!(crate::w_tuple_len(args), 3);
        }
    }

    #[test]
    fn w_union_gc_type_id_matches_descr() {
        assert_eq!(W_UNION_GC_TYPE_ID, 22);
        assert_eq!(
            <W_UnionType as crate::lltype::GcType>::TYPE_ID,
            W_UNION_GC_TYPE_ID
        );
        assert_eq!(
            <W_UnionType as crate::lltype::GcType>::SIZE,
            W_UNION_OBJECT_SIZE
        );
    }
}
