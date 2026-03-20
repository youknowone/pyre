//! W_ListObject — Python `list` with a minimal PyPy-style strategy split.
//!
//! Homogeneous integer and float lists keep unboxed storage, matching PyPy's
//! `IntegerListStrategy` / `FloatListStrategy` direction. Mixed lists fall back
//! to object storage.
//! The JIT's current raw-array fast path only handles object storage.

#![allow(unsafe_op_in_unsafe_fn)]

use crate::{
    FloatArray, IntArray, PyObjectArray, floatobject::w_float_get_value, floatobject::w_float_new,
    intobject::w_int_get_value, intobject::w_int_new,
};
use crate::pyobject::*;

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ListStrategy {
    Object = 0,
    Integer = 1,
    Float = 2,
}

/// Python list object.
///
/// Layout: `[ob_type: *const PyType | items: *mut Vec<PyObjectRef>]`
/// Items are stored on the heap via `Box::into_raw`.
#[repr(C)]
pub struct W_ListObject {
    pub ob_header: PyObject,
    pub items: PyObjectArray,
    pub strategy: ListStrategy,
    pub int_items: IntArray,
    pub float_items: FloatArray,
}

fn all_ints(items: &[PyObjectRef]) -> bool {
    items.iter().all(|&item| !item.is_null() && unsafe { is_int(item) })
}

fn all_floats(items: &[PyObjectRef]) -> bool {
    items.iter().all(|&item| !item.is_null() && unsafe { is_float(item) })
}

fn object_array_from_ints(values: &[i64]) -> PyObjectArray {
    let boxed: Vec<PyObjectRef> = values.iter().map(|&value| w_int_new(value)).collect();
    PyObjectArray::from_vec(boxed)
}

fn object_array_from_floats(values: &[f64]) -> PyObjectArray {
    let boxed: Vec<PyObjectRef> = values.iter().map(|&value| w_float_new(value)).collect();
    PyObjectArray::from_vec(boxed)
}

unsafe fn switch_to_object_strategy(list: &mut W_ListObject) {
    if list.strategy == ListStrategy::Object {
        return;
    }
    list.items = match list.strategy {
        ListStrategy::Integer => object_array_from_ints(list.int_items.as_slice()),
        ListStrategy::Float => object_array_from_floats(list.float_items.as_slice()),
        ListStrategy::Object => PyObjectArray::from_vec(Vec::new()),
    };
    list.items.fix_ptr();
    list.int_items = IntArray::from_vec(Vec::new());
    list.int_items.fix_ptr();
    list.float_items = FloatArray::from_vec(Vec::new());
    list.float_items.fix_ptr();
    list.strategy = ListStrategy::Object;
}

/// Allocate a new W_ListObject from a Vec of items.
pub fn w_list_new(items: Vec<PyObjectRef>) -> PyObjectRef {
    let strategy = if all_ints(&items) {
        ListStrategy::Integer
    } else if all_floats(&items) {
        ListStrategy::Float
    } else {
        ListStrategy::Object
    };
    let (items, int_items, float_items) = match strategy {
        ListStrategy::Object => (
            PyObjectArray::from_vec(items),
            IntArray::from_vec(Vec::new()),
            FloatArray::from_vec(Vec::new()),
        ),
        ListStrategy::Integer => {
            let values = items
                .into_iter()
                .map(|item| unsafe { w_int_get_value(item) })
                .collect();
            (
                PyObjectArray::from_vec(Vec::new()),
                IntArray::from_vec(values),
                FloatArray::from_vec(Vec::new()),
            )
        }
        ListStrategy::Float => {
            let values = items
                .into_iter()
                .map(|item| unsafe { w_float_get_value(item) })
                .collect();
            (
                PyObjectArray::from_vec(Vec::new()),
                IntArray::from_vec(Vec::new()),
                FloatArray::from_vec(values),
            )
        }
    };
    let mut obj = Box::new(W_ListObject {
        ob_header: PyObject {
            ob_type: &LIST_TYPE as *const PyType,
        },
        items,
        strategy,
        int_items,
        float_items,
    });
    obj.items.fix_ptr();
    obj.int_items.fix_ptr();
    obj.float_items.fix_ptr();
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
    match list.strategy {
        ListStrategy::Object => {
            let items = list.items.as_slice();
            let len = items.len() as i64;
            let idx = if index < 0 { index + len } else { index };
            if idx < 0 || idx >= len {
                return None;
            }
            Some(items[idx as usize])
        }
        ListStrategy::Integer => {
            let items = list.int_items.as_slice();
            let len = items.len() as i64;
            let idx = if index < 0 { index + len } else { index };
            if idx < 0 || idx >= len {
                return None;
            }
            Some(w_int_new(items[idx as usize]))
        }
        ListStrategy::Float => {
            let items = list.float_items.as_slice();
            let len = items.len() as i64;
            let idx = if index < 0 { index + len } else { index };
            if idx < 0 || idx >= len {
                return None;
            }
            Some(w_float_new(items[idx as usize]))
        }
    }
}

/// Set the item at the given index in a list.
///
/// Supports negative indexing. Returns false if out of bounds.
///
/// # Safety
/// `obj` must point to a valid `W_ListObject`.
pub unsafe fn w_list_setitem(obj: PyObjectRef, index: i64, value: PyObjectRef) -> bool {
    let list = &mut *(obj as *mut W_ListObject);
    match list.strategy {
        ListStrategy::Object => {
            let items = list.items.as_mut_slice();
            let len = items.len() as i64;
            let idx = if index < 0 { index + len } else { index };
            if idx < 0 || idx >= len {
                return false;
            }
            items[idx as usize] = value;
            true
        }
        ListStrategy::Integer => {
            let len = list.int_items.len() as i64;
            let idx = if index < 0 { index + len } else { index };
            if idx < 0 || idx >= len {
                return false;
            }
            if !value.is_null() && is_int(value) {
                list.int_items[idx as usize] = w_int_get_value(value);
                true
            } else {
                switch_to_object_strategy(list);
                w_list_setitem(obj, index, value)
            }
        }
        ListStrategy::Float => {
            let len = list.float_items.len() as i64;
            let idx = if index < 0 { index + len } else { index };
            if idx < 0 || idx >= len {
                return false;
            }
            if !value.is_null() && is_float(value) {
                list.float_items[idx as usize] = w_float_get_value(value);
                true
            } else {
                switch_to_object_strategy(list);
                w_list_setitem(obj, index, value)
            }
        }
    }
}

/// Append an item to a list.
///
/// # Safety
/// `obj` must point to a valid `W_ListObject`.
pub unsafe fn w_list_append(obj: PyObjectRef, value: PyObjectRef) {
    let list = &mut *(obj as *mut W_ListObject);
    match list.strategy {
        ListStrategy::Object => list.items.push(value),
        ListStrategy::Integer => {
            if !value.is_null() && is_int(value) {
                list.int_items.push(w_int_get_value(value));
            } else {
                switch_to_object_strategy(list);
                list.items.push(value);
            }
        }
        ListStrategy::Float => {
            if !value.is_null() && is_float(value) {
                list.float_items.push(w_float_get_value(value));
            } else {
                switch_to_object_strategy(list);
                list.items.push(value);
            }
        }
    }
}

/// Get the length of a list.
///
/// # Safety
/// `obj` must point to a valid `W_ListObject`.
pub unsafe fn w_list_len(obj: PyObjectRef) -> usize {
    let list = &*(obj as *const W_ListObject);
    match list.strategy {
        ListStrategy::Object => list.items.len(),
        ListStrategy::Integer => list.int_items.len(),
        ListStrategy::Float => list.float_items.len(),
    }
}

/// Check whether appending one element can complete without reallocating.
///
/// # Safety
/// `obj` must point to a valid `W_ListObject`.
pub unsafe fn w_list_can_append_without_realloc(obj: PyObjectRef) -> bool {
    let list = &*(obj as *const W_ListObject);
    match list.strategy {
        ListStrategy::Object => list.items.spare_capacity() > 0,
        ListStrategy::Integer => list.int_items.spare_capacity() > 0,
        ListStrategy::Float => list.float_items.spare_capacity() > 0,
    }
}

/// Check whether the list is currently using inline array storage.
///
/// # Safety
/// `obj` must point to a valid `W_ListObject`.
pub unsafe fn w_list_is_inline_storage(obj: PyObjectRef) -> bool {
    let list = &*(obj as *const W_ListObject);
    match list.strategy {
        ListStrategy::Object => list.items.is_inline(),
        ListStrategy::Integer => list.int_items.is_inline(),
        ListStrategy::Float => list.float_items.is_inline(),
    }
}

pub unsafe fn w_list_uses_object_storage(obj: PyObjectRef) -> bool {
    let list = &*(obj as *const W_ListObject);
    list.strategy == ListStrategy::Object
}

pub unsafe fn w_list_uses_int_storage(obj: PyObjectRef) -> bool {
    let list = &*(obj as *const W_ListObject);
    list.strategy == ListStrategy::Integer
}

pub unsafe fn w_list_uses_float_storage(obj: PyObjectRef) -> bool {
    let list = &*(obj as *const W_ListObject);
    list.strategy == ListStrategy::Float
}

pub extern "C" fn jit_list_append(list: i64, item: i64) -> i64 {
    unsafe { w_list_append(list as PyObjectRef, item as PyObjectRef) };
    0
}

pub extern "C" fn jit_list_getitem(list: i64, index: i64) -> i64 {
    unsafe {
        match w_list_getitem(list as PyObjectRef, index) {
            Some(value) => value as i64,
            None => panic!("list index out of range in JIT"),
        }
    }
}

pub extern "C" fn jit_list_setitem(list: i64, index: i64, value: i64) -> i64 {
    unsafe {
        if !w_list_setitem(list as PyObjectRef, index, value as PyObjectRef) {
            panic!("list assignment index out of range in JIT");
        }
    }
    0
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

    #[test]
    fn test_jit_list_helpers_share_list_semantics() {
        let list = w_list_new(vec![w_int_new(1), w_int_new(2)]);
        unsafe {
            assert_eq!(
                crate::intobject::w_int_get_value(jit_list_getitem(list as i64, 1) as PyObjectRef),
                2
            );
        }
        assert_eq!(jit_list_setitem(list as i64, 0, w_int_new(9) as i64), 0);
        assert_eq!(jit_list_append(list as i64, w_int_new(7) as i64), 0);
        unsafe {
            assert_eq!(w_list_len(list), 3);
            assert_eq!(
                crate::intobject::w_int_get_value(w_list_getitem(list, 0).unwrap()),
                9
            );
            assert_eq!(
                crate::intobject::w_int_get_value(w_list_getitem(list, 2).unwrap()),
                7
            );
        }
    }

    #[test]
    fn test_list_uses_integer_strategy_for_homogeneous_ints() {
        let list = w_list_new(vec![w_int_new(1), w_int_new(2), w_int_new(3)]);
        unsafe {
            assert!(w_list_uses_int_storage(list));
            assert!(!w_list_uses_object_storage(list));
            assert_eq!(w_list_len(list), 3);
        }
    }

    #[test]
    fn test_list_setitem_mixed_value_switches_to_object_strategy() {
        let list = w_list_new(vec![w_int_new(1), w_int_new(2)]);
        let float = crate::floatobject::w_float_new(3.5);
        unsafe {
            assert!(w_list_uses_int_storage(list));
            assert!(w_list_setitem(list, 0, float));
            assert!(w_list_uses_object_storage(list));
            let value = w_list_getitem(list, 0).unwrap();
            assert!(crate::pyobject::is_float(value));
        }
    }

    #[test]
    fn test_list_append_mixed_value_switches_to_object_strategy() {
        let list = w_list_new(vec![w_int_new(1), w_int_new(2)]);
        let float = crate::floatobject::w_float_new(3.5);
        unsafe {
            assert!(w_list_uses_int_storage(list));
            w_list_append(list, float);
            assert!(w_list_uses_object_storage(list));
            assert_eq!(w_list_len(list), 3);
            let value = w_list_getitem(list, 2).unwrap();
            assert!(crate::pyobject::is_float(value));
        }
    }

    #[test]
    fn test_list_uses_float_strategy_for_homogeneous_floats() {
        let list = w_list_new(vec![
            crate::floatobject::w_float_new(1.25),
            crate::floatobject::w_float_new(2.5),
            crate::floatobject::w_float_new(3.75),
        ]);
        unsafe {
            assert!(w_list_uses_float_storage(list));
            assert!(!w_list_uses_object_storage(list));
            assert_eq!(w_list_len(list), 3);
            let value = w_list_getitem(list, 1).unwrap();
            assert!(crate::pyobject::is_float(value));
            assert_eq!(crate::floatobject::w_float_get_value(value), 2.5);
        }
    }

    #[test]
    fn test_list_setitem_mixed_on_float_strategy_switches_to_object_strategy() {
        let list = w_list_new(vec![
            crate::floatobject::w_float_new(1.0),
            crate::floatobject::w_float_new(2.0),
        ]);
        unsafe {
            assert!(w_list_uses_float_storage(list));
            assert!(w_list_setitem(list, 0, w_int_new(7)));
            assert!(w_list_uses_object_storage(list));
            let value = w_list_getitem(list, 0).unwrap();
            assert!(crate::pyobject::is_int(value));
        }
    }

    #[test]
    fn test_list_append_mixed_on_float_strategy_switches_to_object_strategy() {
        let list = w_list_new(vec![
            crate::floatobject::w_float_new(1.0),
            crate::floatobject::w_float_new(2.0),
        ]);
        unsafe {
            assert!(w_list_uses_float_storage(list));
            w_list_append(list, w_int_new(7));
            assert!(w_list_uses_object_storage(list));
            assert_eq!(w_list_len(list), 3);
            let value = w_list_getitem(list, 2).unwrap();
            assert!(crate::pyobject::is_int(value));
        }
    }
}
