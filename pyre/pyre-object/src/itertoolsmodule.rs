//! itertools module — iterator objects.
//!
//! PyPy: pypy/module/itertools/interp_itertools.py
//!
//! Line-by-line port of the W_Count / W_Repeat / W_Cycle / W_Chain etc.
//! classes. Each class becomes a `#[repr(C)]` struct with a static PyType.

use crate::pyobject::*;

// ── W_Count — pypy/module/itertools/interp_itertools.py:class W_Count ──
//
// ```python
// class W_Count(W_Root):
//     def __init__(self, space, w_firstval, w_step):
//         self.space = space
//         self.w_c = w_firstval
//         self.w_step = w_step
//
//     def iter_w(self):
//         return self
//
//     def next_w(self):
//         w_c = self.w_c
//         self.w_c = self.space.add(w_c, self.w_step)
//         return w_c
// ```
//
// The receiver stores `w_c` (current value) and `w_step` which are both
// PyObjectRef so that count(1.5, 0.5) works for float too.

pub static COUNT_TYPE: PyType = new_pytype("itertools.count");

#[repr(C)]
pub struct W_Count {
    pub ob: PyObject,
    pub w_c: PyObjectRef,
    pub w_step: PyObjectRef,
}

pub fn w_count_new(w_firstval: PyObjectRef, w_step: PyObjectRef) -> PyObjectRef {
    let obj = Box::new(W_Count {
        ob: PyObject {
            ob_type: &COUNT_TYPE as *const PyType,
            w_class: get_instantiate(&COUNT_TYPE),
        },
        w_c: w_firstval,
        w_step,
    });
    Box::into_raw(obj) as PyObjectRef
}

/// Check if an object is a `W_Count`.
///
/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn is_count(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &COUNT_TYPE) }
}

/// Read the current `w_c` field.
///
/// # Safety
/// `obj` must point to a valid `W_Count`.
pub unsafe fn w_count_get_c(obj: PyObjectRef) -> PyObjectRef {
    (*(obj as *const W_Count)).w_c
}

/// Write the current `w_c` field.
///
/// # Safety
/// `obj` must point to a valid `W_Count`.
pub unsafe fn w_count_set_c(obj: PyObjectRef, v: PyObjectRef) {
    (*(obj as *mut W_Count)).w_c = v;
}

/// Read the `w_step` field.
///
/// # Safety
/// `obj` must point to a valid `W_Count`.
pub unsafe fn w_count_get_step(obj: PyObjectRef) -> PyObjectRef {
    (*(obj as *const W_Count)).w_step
}

// ── W_Repeat — pypy/module/itertools/interp_itertools.py:class W_Repeat ──
//
// ```python
// class W_Repeat(W_Root):
//     def __init__(self, space, w_obj, w_times):
//         self.space = space
//         self.w_obj = w_obj
//         if w_times is None:
//             self.counting = False
//             self.count = 0
//         else:
//             self.counting = True
//             self.count = max(self.space.int_w(w_times), 0)
//
//     def next_w(self):
//         if self.counting:
//             if self.count <= 0:
//                 raise OperationError(self.space.w_StopIteration, self.space.w_None)
//             self.count -= 1
//         return self.w_obj
// ```

pub static REPEAT_TYPE: PyType = new_pytype("itertools.repeat");

#[repr(C)]
pub struct W_Repeat {
    pub ob: PyObject,
    pub w_obj: PyObjectRef,
    pub counting: bool,
    pub count: i64,
}

pub fn w_repeat_new(w_obj: PyObjectRef, w_times: Option<i64>) -> PyObjectRef {
    let (counting, count) = match w_times {
        None => (false, 0),
        Some(n) => (true, n.max(0)),
    };
    let obj = Box::new(W_Repeat {
        ob: PyObject {
            ob_type: &REPEAT_TYPE as *const PyType,
            w_class: get_instantiate(&REPEAT_TYPE),
        },
        w_obj,
        counting,
        count,
    });
    Box::into_raw(obj) as PyObjectRef
}

/// Check if an object is a `W_Repeat`.
///
/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn is_repeat(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &REPEAT_TYPE) }
}

/// Read the `w_obj` field.
///
/// # Safety
/// `obj` must point to a valid `W_Repeat`.
pub unsafe fn w_repeat_get_obj(obj: PyObjectRef) -> PyObjectRef {
    (*(obj as *const W_Repeat)).w_obj
}

/// Read the `counting` field.
///
/// # Safety
/// `obj` must point to a valid `W_Repeat`.
pub unsafe fn w_repeat_get_counting(obj: PyObjectRef) -> bool {
    (*(obj as *const W_Repeat)).counting
}

/// Read the `count` field.
///
/// # Safety
/// `obj` must point to a valid `W_Repeat`.
pub unsafe fn w_repeat_get_count(obj: PyObjectRef) -> i64 {
    (*(obj as *const W_Repeat)).count
}

/// Decrement the `count` field by 1.
///
/// # Safety
/// `obj` must point to a valid `W_Repeat`.
pub unsafe fn w_repeat_dec_count(obj: PyObjectRef) {
    (*(obj as *mut W_Repeat)).count -= 1;
}
