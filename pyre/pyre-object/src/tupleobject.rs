//! W_TupleObject — Python `tuple` type backed by Vec<PyObjectRef>.
//!
//! Tuples are immutable after creation. Phase 1 uses a heap-allocated Vec
//! behind a raw pointer. The JIT treats tuple operations as opaque residual calls.

#![allow(unsafe_op_in_unsafe_fn)]

use crate::pyobject::*;

/// Python tuple object.
///
/// Layout: `[ob_type: *const PyType | items: *mut Vec<PyObjectRef>]`
/// Items are immutable after creation.
#[repr(C)]
pub struct W_TupleObject {
    pub ob_header: PyObject,
    pub items: *mut Vec<PyObjectRef>,
}

/// Allocate a new W_TupleObject from a Vec of items.
pub fn w_tuple_new(items: Vec<PyObjectRef>) -> PyObjectRef {
    let obj = Box::new(W_TupleObject {
        ob_header: PyObject {
            ob_type: &TUPLE_TYPE as *const PyType,
        },
        items: Box::into_raw(Box::new(items)),
    });
    Box::into_raw(obj) as PyObjectRef
}

/// Get the item at the given index from a tuple.
///
/// Supports negative indexing. Returns None if out of bounds.
///
/// # Safety
/// `obj` must point to a valid `W_TupleObject`.
pub unsafe fn w_tuple_getitem(obj: PyObjectRef, index: i64) -> Option<PyObjectRef> {
    let tuple = &*(obj as *const W_TupleObject);
    let items = &*tuple.items;
    let len = items.len() as i64;
    let idx = if index < 0 { index + len } else { index };
    if idx < 0 || idx >= len {
        return None;
    }
    Some(items[idx as usize])
}

/// Get the length of a tuple.
///
/// # Safety
/// `obj` must point to a valid `W_TupleObject`.
pub unsafe fn w_tuple_len(obj: PyObjectRef) -> usize {
    let tuple = &*(obj as *const W_TupleObject);
    let items = &*tuple.items;
    items.len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intobject::w_int_new;

    #[test]
    fn test_tuple_create_and_access() {
        let items = vec![w_int_new(1), w_int_new(2), w_int_new(3)];
        let tup = w_tuple_new(items);
        unsafe {
            assert!(is_tuple(tup));
            assert_eq!(w_tuple_len(tup), 3);
            let item = w_tuple_getitem(tup, 0).unwrap();
            assert_eq!(crate::intobject::w_int_get_value(item), 1);
            let item = w_tuple_getitem(tup, 2).unwrap();
            assert_eq!(crate::intobject::w_int_get_value(item), 3);
        }
    }

    #[test]
    fn test_tuple_negative_index() {
        let items = vec![w_int_new(10), w_int_new(20)];
        let tup = w_tuple_new(items);
        unsafe {
            let item = w_tuple_getitem(tup, -1).unwrap();
            assert_eq!(crate::intobject::w_int_get_value(item), 20);
        }
    }

    #[test]
    fn test_tuple_out_of_bounds() {
        let tup = w_tuple_new(vec![w_int_new(1)]);
        unsafe {
            assert!(w_tuple_getitem(tup, 5).is_none());
            assert!(w_tuple_getitem(tup, -5).is_none());
        }
    }
}
