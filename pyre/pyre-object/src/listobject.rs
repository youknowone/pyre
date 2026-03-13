//! W_ListObject — Python `list` type backed by Vec<PyObjectRef>.
//!
//! Phase 1 uses a heap-allocated Vec behind a raw pointer.
//! The JIT treats list operations as opaque residual calls.

#![allow(unsafe_op_in_unsafe_fn)]

use crate::pyobject::*;

/// Python list object.
///
/// Layout: `[ob_type: *const PyType | items: *mut Vec<PyObjectRef>]`
/// Items are stored on the heap via `Box::into_raw`.
#[repr(C)]
pub struct W_ListObject {
    pub ob_header: PyObject,
    pub items: *mut Vec<PyObjectRef>,
}

/// Allocate a new W_ListObject from a Vec of items.
pub fn w_list_new(items: Vec<PyObjectRef>) -> PyObjectRef {
    let obj = Box::new(W_ListObject {
        ob_header: PyObject {
            ob_type: &LIST_TYPE as *const PyType,
        },
        items: Box::into_raw(Box::new(items)),
    });
    Box::into_raw(obj) as PyObjectRef
}

/// Get the item at the given index from a list.
///
/// Supports negative indexing. Returns None if out of bounds.
///
/// # Safety
/// `obj` must point to a valid `W_ListObject`.
pub unsafe fn w_list_getitem(obj: PyObjectRef, index: i64) -> Option<PyObjectRef> {
    let list = &*(obj as *const W_ListObject);
    let items = &*list.items;
    let len = items.len() as i64;
    let idx = if index < 0 { index + len } else { index };
    if idx < 0 || idx >= len {
        return None;
    }
    Some(items[idx as usize])
}

/// Set the item at the given index in a list.
///
/// Supports negative indexing. Returns false if out of bounds.
///
/// # Safety
/// `obj` must point to a valid `W_ListObject`.
pub unsafe fn w_list_setitem(obj: PyObjectRef, index: i64, value: PyObjectRef) -> bool {
    let list = &*(obj as *const W_ListObject);
    let items = &mut *list.items;
    let len = items.len() as i64;
    let idx = if index < 0 { index + len } else { index };
    if idx < 0 || idx >= len {
        return false;
    }
    items[idx as usize] = value;
    true
}

/// Append an item to a list.
///
/// # Safety
/// `obj` must point to a valid `W_ListObject`.
pub unsafe fn w_list_append(obj: PyObjectRef, value: PyObjectRef) {
    let list = &*(obj as *const W_ListObject);
    let items = &mut *list.items;
    items.push(value);
}

/// Get the length of a list.
///
/// # Safety
/// `obj` must point to a valid `W_ListObject`.
pub unsafe fn w_list_len(obj: PyObjectRef) -> usize {
    let list = &*(obj as *const W_ListObject);
    let items = &*list.items;
    items.len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intobject::w_int_new;

    #[test]
    fn test_list_create_and_access() {
        let items = vec![w_int_new(10), w_int_new(20), w_int_new(30)];
        let list = w_list_new(items);
        unsafe {
            assert!(is_list(list));
            assert_eq!(w_list_len(list), 3);
            let item = w_list_getitem(list, 0).unwrap();
            assert_eq!(crate::intobject::w_int_get_value(item), 10);
            let item = w_list_getitem(list, 2).unwrap();
            assert_eq!(crate::intobject::w_int_get_value(item), 30);
        }
    }

    #[test]
    fn test_list_negative_index() {
        let items = vec![w_int_new(1), w_int_new(2), w_int_new(3)];
        let list = w_list_new(items);
        unsafe {
            let item = w_list_getitem(list, -1).unwrap();
            assert_eq!(crate::intobject::w_int_get_value(item), 3);
        }
    }

    #[test]
    fn test_list_setitem() {
        let items = vec![w_int_new(1), w_int_new(2)];
        let list = w_list_new(items);
        unsafe {
            assert!(w_list_setitem(list, 0, w_int_new(99)));
            let item = w_list_getitem(list, 0).unwrap();
            assert_eq!(crate::intobject::w_int_get_value(item), 99);
        }
    }

    #[test]
    fn test_list_append() {
        let list = w_list_new(vec![]);
        unsafe {
            w_list_append(list, w_int_new(42));
            assert_eq!(w_list_len(list), 1);
            let item = w_list_getitem(list, 0).unwrap();
            assert_eq!(crate::intobject::w_int_get_value(item), 42);
        }
    }

    #[test]
    fn test_list_out_of_bounds() {
        let list = w_list_new(vec![w_int_new(1)]);
        unsafe {
            assert!(w_list_getitem(list, 5).is_none());
            assert!(w_list_getitem(list, -5).is_none());
            assert!(!w_list_setitem(list, 5, w_int_new(0)));
        }
    }
}
