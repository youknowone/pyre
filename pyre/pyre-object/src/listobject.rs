//! W_ListObject — Python `list` with a minimal PyPy-style strategy split.
//!
//! Homogeneous integer and float lists keep unboxed storage, matching PyPy's
//! `IntegerListStrategy` / `FloatListStrategy` direction. Mixed lists fall back
//! to object storage.
//! The JIT's current raw-array fast path only handles object storage.

#![allow(unsafe_op_in_unsafe_fn)]

use crate::pyobject::*;
use crate::{
    FloatArray, IntArray, PyObjectArray, floatobject::w_float_get_value, floatobject::w_float_new,
    intobject::w_int_get_value, intobject::w_int_new, intobject::w_int_small_cached,
};

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

/// Pyre equivalent of RPython `is_plain_int1(w_obj)` (listobject.py:2397):
///   `type(w_obj) is W_IntObject`
/// Unique ints (created by w_int_new_unique for int subclass instances) are excluded
/// because they may carry per-object attributes that require pointer identity.
#[inline]
unsafe fn is_plain_int1(item: PyObjectRef) -> bool {
    if item.is_null() || !is_int(item) {
        return false;
    }
    let v = w_int_get_value(item);
    // For values in the small-int cache range, verify pointer identity.
    // A non-matching pointer means w_int_new_unique made a subclass instance.
    if w_int_small_cached(v) {
        std::ptr::eq(item, w_int_new(v))
    } else {
        true
    }
}

fn all_ints(items: &[PyObjectRef]) -> bool {
    items.iter().all(|&item| unsafe { is_plain_int1(item) })
}

fn all_floats(items: &[PyObjectRef]) -> bool {
    items
        .iter()
        .all(|&item| !item.is_null() && unsafe { is_float(item) })
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
            w_class: get_instantiate(&LIST_TYPE),
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
            // AbstractUnwrappedStrategy.setitem (listobject.py:1737):
            //   if is_correct_type(w_item): l[index] = unwrap(w_item)
            if is_plain_int1(value) {
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
        // AbstractUnwrappedStrategy.append (listobject.py:1695):
        //   if self.is_correct_type(w_item): l.append(self.unwrap(w_item)); return
        //   self.switch_to_next_strategy(w_list, w_item); w_list.append(w_item)
        ListStrategy::Object => list.items.push(value),
        ListStrategy::Integer => {
            if is_plain_int1(value) {
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

/// Rebuild the list's object storage from a Vec.
/// Used by mutation operations that need insert/remove.
pub unsafe fn rebuild_object_items(list: &mut W_ListObject, items: Vec<PyObjectRef>) {
    list.items = PyObjectArray::from_vec(items);
    list.items.fix_ptr();
}

/// Set a slice of the list: `list[start:end] = new_items` (step=1).
/// Mirrors RPython `AbstractUnwrappedStrategy.setslice` (listobject.py:1750) for step==1.
/// Switches to ObjectStrategy when new_items don't match the current strategy type.
pub unsafe fn w_list_setslice(
    obj: PyObjectRef,
    start: usize,
    end: usize,
    new_items: Vec<PyObjectRef>,
) {
    let list = &mut *(obj as *mut W_ListObject);
    // Determine if all new_items match current strategy
    match list.strategy {
        ListStrategy::Integer => {
            // AbstractUnwrappedStrategy.setslice (listobject.py:1750):
            //   list_is_correct_type(w_other): typed splice
            let all_int = new_items.iter().all(|&v| is_plain_int1(v));
            if all_int {
                // typed splice: remove [start..end] then insert new int values
                let end_clamped = end.min(list.int_items.len());
                let start_clamped = start.min(list.int_items.len());
                let remove_count = end_clamped.saturating_sub(start_clamped);
                // drain remove_count elements starting at start_clamped
                for _ in 0..remove_count {
                    list.int_items.remove(start_clamped);
                }
                // insert new values starting at start_clamped
                for (i, &v) in new_items.iter().enumerate() {
                    list.int_items.insert(start_clamped + i, w_int_get_value(v));
                }
                return;
            }
            // fall through to object strategy
            switch_to_object_strategy(list);
        }
        ListStrategy::Float => {
            let all_float = new_items
                .iter()
                .all(|&v| !v.is_null() && is_float(v));
            if all_float {
                let end_clamped = end.min(list.float_items.len());
                let start_clamped = start.min(list.float_items.len());
                let remove_count = end_clamped.saturating_sub(start_clamped);
                for _ in 0..remove_count {
                    list.float_items.remove(start_clamped);
                }
                for (i, &v) in new_items.iter().enumerate() {
                    list.float_items.insert(start_clamped + i, w_float_get_value(v));
                }
                return;
            }
            switch_to_object_strategy(list);
        }
        ListStrategy::Object => {}
    }
    // Object strategy splice
    let len = list.items.len();
    let s = start.min(len);
    let e = end.min(len);
    let remove_count = e.saturating_sub(s);
    for _ in 0..remove_count {
        list.items.remove(s);
    }
    for (i, v) in new_items.into_iter().enumerate() {
        list.items.insert(s + i, v);
    }
}

/// Insert item at index.
/// Mirrors RPython `AbstractUnwrappedStrategy.insert` (listobject.py:1714):
///   if `is_correct_type(w_item)`: `l.insert(index, unwrap(w_item))`
///   else: `switch_to_next_strategy` then delegate.
pub unsafe fn w_list_insert(obj: PyObjectRef, index: i64, value: PyObjectRef) {
    let list = &mut *(obj as *mut W_ListObject);
    let len = w_list_len(obj) as i64;
    let idx = if index < 0 {
        (index + len).max(0) as usize
    } else {
        (index as usize).min(len as usize)
    };
    match list.strategy {
        ListStrategy::Object => list.items.insert(idx, value),
        ListStrategy::Integer => {
            if is_plain_int1(value) {
                list.int_items.insert(idx, w_int_get_value(value));
            } else {
                switch_to_object_strategy(list);
                list.items.insert(idx, value);
            }
        }
        ListStrategy::Float => {
            if !value.is_null() && is_float(value) {
                list.float_items.insert(idx, w_float_get_value(value));
            } else {
                switch_to_object_strategy(list);
                list.items.insert(idx, value);
            }
        }
    }
}

/// Remove and return item at index.
/// Mirrors RPython `AbstractUnwrappedStrategy.pop` (listobject.py:1855):
///   `item = l.pop(index); return self.wrap(item)`
pub unsafe fn w_list_pop(obj: PyObjectRef, index: i64) -> Option<PyObjectRef> {
    let list = &mut *(obj as *mut W_ListObject);
    let len = w_list_len(obj) as i64;
    if len == 0 {
        return None;
    }
    let idx = if index < 0 { index + len } else { index };
    if idx < 0 || idx >= len {
        return None;
    }
    let idx = idx as usize;
    match list.strategy {
        ListStrategy::Object => Some(list.items.remove(idx)),
        ListStrategy::Integer => Some(w_int_new(list.int_items.remove(idx))),
        ListStrategy::Float => Some(w_float_new(list.float_items.remove(idx))),
    }
}

/// Remove and return last item.
/// Mirrors RPython `AbstractUnwrappedStrategy.pop_end` (listobject.py:1848):
///   `return self.wrap(l.pop())`
pub unsafe fn w_list_pop_end(obj: PyObjectRef) -> Option<PyObjectRef> {
    let list = &mut *(obj as *mut W_ListObject);
    match list.strategy {
        ListStrategy::Object => {
            if list.items.len() == 0 {
                return None;
            }
            Some(list.items.pop())
        }
        ListStrategy::Integer => {
            if list.int_items.len() == 0 {
                return None;
            }
            Some(w_int_new(list.int_items.pop()))
        }
        ListStrategy::Float => {
            if list.float_items.len() == 0 {
                return None;
            }
            Some(w_float_new(list.float_items.pop()))
        }
    }
}

/// Clear all items.
/// Mirrors RPython `W_ListObject.clear` (listobject.py:359) which switches to
/// EmptyListStrategy. We keep Object strategy with empty storage.
pub unsafe fn w_list_clear(obj: PyObjectRef) {
    let list = &mut *(obj as *mut W_ListObject);
    match list.strategy {
        ListStrategy::Object => {
            list.items = PyObjectArray::from_vec(Vec::new());
            list.items.fix_ptr();
        }
        ListStrategy::Integer => {
            list.int_items = IntArray::from_vec(Vec::new());
            list.int_items.fix_ptr();
        }
        ListStrategy::Float => {
            list.float_items = FloatArray::from_vec(Vec::new());
            list.float_items.fix_ptr();
        }
    }
}

/// Reverse in place.
/// Mirrors RPython `AbstractUnwrappedStrategy.reverse` (listobject.py:1880):
///   `self.unerase(w_list.lstorage).reverse()`
pub unsafe fn w_list_reverse(obj: PyObjectRef) {
    let list = &mut *(obj as *mut W_ListObject);
    match list.strategy {
        ListStrategy::Object => list.items.reverse(),
        ListStrategy::Integer => list.int_items.reverse(),
        ListStrategy::Float => list.float_items.reverse(),
    }
}

/// Delete a range of items [start..end) by index.
/// Mirrors RPython `AbstractUnwrappedStrategy.deleteslice` (listobject.py:1815)
/// for the step==1 case used by `list_delslice`.
pub unsafe fn w_list_delslice(obj: PyObjectRef, start: usize, end: usize) {
    let list = &mut *(obj as *mut W_ListObject);
    match list.strategy {
        ListStrategy::Object => {
            let len = list.items.len();
            let s = start.min(len);
            let e = end.min(len);
            if s < e {
                // shift elements left: items[s..len-count] = items[e..len]
                let count = e - s;
                let slice = list.items.as_mut_slice();
                slice.copy_within(e..len, s);
                // truncate: reduce len by count
                for _ in 0..count {
                    list.items.pop();
                }
            }
        }
        ListStrategy::Integer => {
            let len = list.int_items.len();
            let s = start.min(len);
            let e = end.min(len);
            if s < e {
                let count = e - s;
                let slice = list.int_items.as_mut_slice();
                slice.copy_within(e..len, s);
                for _ in 0..count {
                    list.int_items.pop();
                }
            }
        }
        ListStrategy::Float => {
            let len = list.float_items.len();
            let s = start.min(len);
            let e = end.min(len);
            if s < e {
                let count = e - s;
                let slice = list.float_items.as_mut_slice();
                slice.copy_within(e..len, s);
                for _ in 0..count {
                    list.float_items.pop();
                }
            }
        }
    }
}

/// Remove first occurrence of value.
/// Mirrors RPython `W_ListObject.descr_remove` (listobject.py:790):
///   find via `find_or_count`, then `self.pop(i)`.
pub unsafe fn w_list_remove(obj: PyObjectRef, value: PyObjectRef) -> bool {
    let list = &mut *(obj as *mut W_ListObject);
    match list.strategy {
        ListStrategy::Object => {
            let items = list.items.as_slice();
            for i in 0..items.len() {
                let item = items[i];
                let eq = if std::ptr::eq(item, value) {
                    true
                } else if is_int(item) && is_int(value) {
                    w_int_get_value(item) == w_int_get_value(value)
                } else {
                    false
                };
                if eq {
                    list.items.remove(i);
                    return true;
                }
            }
            false
        }
        ListStrategy::Integer => {
            if !value.is_null() && is_int(value) {
                let target = w_int_get_value(value);
                let items = list.int_items.as_slice();
                for i in 0..items.len() {
                    if items[i] == target {
                        list.int_items.remove(i);
                        return true;
                    }
                }
            }
            false
        }
        ListStrategy::Float => {
            if !value.is_null() && is_float(value) {
                let target = w_float_get_value(value);
                let items = list.float_items.as_slice();
                for i in 0..items.len() {
                    if items[i] == target {
                        list.float_items.remove(i);
                        return true;
                    }
                }
            }
            false
        }
    }
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

    // ── per-strategy operation tests ─────────────────────────────────────────
    // These verify that pop/pop_end/insert/reverse/clear/delslice do NOT
    // switch to ObjectStrategy when the list is homogeneous (int or float).
    // Mirrors RPython AbstractUnwrappedStrategy parity (listobject.py:1714-1891).

    #[test]
    fn test_int_list_pop_stays_integer_strategy() {
        // AbstractUnwrappedStrategy.pop (listobject.py:1855)
        let list = w_list_new(vec![w_int_new(1), w_int_new(2), w_int_new(3)]);
        unsafe {
            assert!(w_list_uses_int_storage(list));
            let popped = w_list_pop(list, 1).unwrap();
            assert_eq!(crate::intobject::w_int_get_value(popped), 2);
            assert!(w_list_uses_int_storage(list), "pop must not switch strategy");
            assert_eq!(w_list_len(list), 2);
            assert_eq!(crate::intobject::w_int_get_value(w_list_getitem(list, 0).unwrap()), 1);
            assert_eq!(crate::intobject::w_int_get_value(w_list_getitem(list, 1).unwrap()), 3);
        }
    }

    #[test]
    fn test_int_list_pop_end_stays_integer_strategy() {
        // AbstractUnwrappedStrategy.pop_end (listobject.py:1848)
        let list = w_list_new(vec![w_int_new(10), w_int_new(20)]);
        unsafe {
            assert!(w_list_uses_int_storage(list));
            let popped = w_list_pop_end(list).unwrap();
            assert_eq!(crate::intobject::w_int_get_value(popped), 20);
            assert!(w_list_uses_int_storage(list), "pop_end must not switch strategy");
            assert_eq!(w_list_len(list), 1);
        }
    }

    #[test]
    fn test_int_list_insert_stays_integer_strategy() {
        // AbstractUnwrappedStrategy.insert (listobject.py:1714)
        let list = w_list_new(vec![w_int_new(1), w_int_new(3)]);
        unsafe {
            assert!(w_list_uses_int_storage(list));
            w_list_insert(list, 1, w_int_new(2));
            assert!(w_list_uses_int_storage(list), "insert int must not switch strategy");
            assert_eq!(w_list_len(list), 3);
            assert_eq!(crate::intobject::w_int_get_value(w_list_getitem(list, 1).unwrap()), 2);
        }
    }

    #[test]
    fn test_int_list_insert_float_switches_to_object() {
        // AbstractUnwrappedStrategy.switch_to_next_strategy (listobject.py:1720)
        let list = w_list_new(vec![w_int_new(1), w_int_new(2)]);
        let fv = crate::floatobject::w_float_new(9.0);
        unsafe {
            assert!(w_list_uses_int_storage(list));
            w_list_insert(list, 1, fv);
            assert!(w_list_uses_object_storage(list));
            assert_eq!(w_list_len(list), 3);
        }
    }

    #[test]
    fn test_int_list_reverse_stays_integer_strategy() {
        // AbstractUnwrappedStrategy.reverse (listobject.py:1880)
        let list = w_list_new(vec![w_int_new(1), w_int_new(2), w_int_new(3)]);
        unsafe {
            assert!(w_list_uses_int_storage(list));
            w_list_reverse(list);
            assert!(w_list_uses_int_storage(list), "reverse must not switch strategy");
            assert_eq!(crate::intobject::w_int_get_value(w_list_getitem(list, 0).unwrap()), 3);
            assert_eq!(crate::intobject::w_int_get_value(w_list_getitem(list, 2).unwrap()), 1);
        }
    }

    #[test]
    fn test_int_list_clear_stays_integer_strategy() {
        // W_ListObject.clear: switches to EmptyListStrategy; here we keep int strategy empty
        let list = w_list_new(vec![w_int_new(1), w_int_new(2)]);
        unsafe {
            assert!(w_list_uses_int_storage(list));
            w_list_clear(list);
            assert!(w_list_uses_int_storage(list), "clear must not switch to Object");
            assert_eq!(w_list_len(list), 0);
        }
    }

    #[test]
    fn test_int_list_delslice_stays_integer_strategy() {
        // AbstractUnwrappedStrategy.deleteslice (listobject.py:1815)
        let list = w_list_new(vec![w_int_new(1), w_int_new(2), w_int_new(3), w_int_new(4)]);
        unsafe {
            assert!(w_list_uses_int_storage(list));
            w_list_delslice(list, 1, 3);
            assert!(w_list_uses_int_storage(list), "delslice must not switch strategy");
            assert_eq!(w_list_len(list), 2);
            assert_eq!(crate::intobject::w_int_get_value(w_list_getitem(list, 0).unwrap()), 1);
            assert_eq!(crate::intobject::w_int_get_value(w_list_getitem(list, 1).unwrap()), 4);
        }
    }

    #[test]
    fn test_float_list_pop_stays_float_strategy() {
        // AbstractUnwrappedStrategy.pop (listobject.py:1855)
        let list = w_list_new(vec![
            crate::floatobject::w_float_new(1.0),
            crate::floatobject::w_float_new(2.0),
            crate::floatobject::w_float_new(3.0),
        ]);
        unsafe {
            assert!(w_list_uses_float_storage(list));
            let popped = w_list_pop(list, 0).unwrap();
            assert_eq!(crate::floatobject::w_float_get_value(popped), 1.0);
            assert!(w_list_uses_float_storage(list), "pop must not switch strategy");
            assert_eq!(w_list_len(list), 2);
        }
    }

    #[test]
    fn test_float_list_reverse_stays_float_strategy() {
        // AbstractUnwrappedStrategy.reverse (listobject.py:1880)
        let list = w_list_new(vec![
            crate::floatobject::w_float_new(1.0),
            crate::floatobject::w_float_new(2.0),
        ]);
        unsafe {
            assert!(w_list_uses_float_storage(list));
            w_list_reverse(list);
            assert!(w_list_uses_float_storage(list), "reverse must not switch strategy");
            assert_eq!(crate::floatobject::w_float_get_value(w_list_getitem(list, 0).unwrap()), 2.0);
        }
    }

    #[test]
    fn test_int_list_remove_stays_integer_strategy() {
        // W_ListObject.descr_remove (listobject.py:790): find_or_count then pop
        let list = w_list_new(vec![w_int_new(1), w_int_new(2), w_int_new(3)]);
        unsafe {
            assert!(w_list_uses_int_storage(list));
            assert!(w_list_remove(list, w_int_new(2)));
            assert!(w_list_uses_int_storage(list), "remove must not switch strategy");
            assert_eq!(w_list_len(list), 2);
            assert_eq!(crate::intobject::w_int_get_value(w_list_getitem(list, 0).unwrap()), 1);
            assert_eq!(crate::intobject::w_int_get_value(w_list_getitem(list, 1).unwrap()), 3);
        }
    }
}
