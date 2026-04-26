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

/// Field offsets of inline `PyObjectRef` slots within `W_Count`.
pub const COUNT_W_C_OFFSET: usize = std::mem::offset_of!(W_Count, w_c);
pub const COUNT_W_STEP_OFFSET: usize = std::mem::offset_of!(W_Count, w_step);

/// GC type id assigned to `W_Count` at JitDriver init time.
pub const W_COUNT_GC_TYPE_ID: u32 = 24;

/// Fixed payload size (`framework.py:811`).
pub const W_COUNT_OBJECT_SIZE: usize = std::mem::size_of::<W_Count>();

/// Byte offsets of the inline `PyObjectRef` fields the GC must trace.
pub const W_COUNT_GC_PTR_OFFSETS: [usize; 2] = [COUNT_W_C_OFFSET, COUNT_W_STEP_OFFSET];

impl crate::lltype::GcType for W_Count {
    const TYPE_ID: u32 = W_COUNT_GC_TYPE_ID;
    const SIZE: usize = W_COUNT_OBJECT_SIZE;
}

pub fn w_count_new(w_firstval: PyObjectRef, w_step: PyObjectRef) -> PyObjectRef {
    // `gct_fv_gc_malloc` bracket pattern (`framework.py:853-856`).
    let _roots = crate::gc_roots::push_roots();
    crate::gc_roots::pin_root(w_firstval);
    crate::gc_roots::pin_root(w_step);

    crate::lltype::malloc_typed(W_Count {
        ob: PyObject {
            ob_type: &COUNT_TYPE as *const PyType,
            w_class: get_instantiate(&COUNT_TYPE),
        },
        w_c: w_firstval,
        w_step,
    }) as PyObjectRef
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
    unsafe { (*(obj as *const W_Count)).w_c }
}

/// Write the current `w_c` field.
///
/// # Safety
/// `obj` must point to a valid `W_Count`.
pub unsafe fn w_count_set_c(obj: PyObjectRef, v: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_Count)).w_c = v;
    }
}

/// Read the `w_step` field.
///
/// # Safety
/// `obj` must point to a valid `W_Count`.
pub unsafe fn w_count_get_step(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_Count)).w_step }
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

/// Field offset of `w_obj` within `W_Repeat`.
pub const REPEAT_W_OBJ_OFFSET: usize = std::mem::offset_of!(W_Repeat, w_obj);

/// GC type id assigned to `W_Repeat` at JitDriver init time.
pub const W_REPEAT_GC_TYPE_ID: u32 = 25;

/// Fixed payload size (`framework.py:811`).
pub const W_REPEAT_OBJECT_SIZE: usize = std::mem::size_of::<W_Repeat>();

/// Byte offsets of the inline `PyObjectRef` fields the GC must trace.
pub const W_REPEAT_GC_PTR_OFFSETS: [usize; 1] = [REPEAT_W_OBJ_OFFSET];

impl crate::lltype::GcType for W_Repeat {
    const TYPE_ID: u32 = W_REPEAT_GC_TYPE_ID;
    const SIZE: usize = W_REPEAT_OBJECT_SIZE;
}

pub fn w_repeat_new(w_obj: PyObjectRef, w_times: Option<i64>) -> PyObjectRef {
    let (counting, count) = match w_times {
        None => (false, 0),
        Some(n) => (true, n.max(0)),
    };
    // `gct_fv_gc_malloc` bracket pattern (`framework.py:853-856`).
    let _roots = crate::gc_roots::push_roots();
    crate::gc_roots::pin_root(w_obj);

    crate::lltype::malloc_typed(W_Repeat {
        ob: PyObject {
            ob_type: &REPEAT_TYPE as *const PyType,
            w_class: get_instantiate(&REPEAT_TYPE),
        },
        w_obj,
        counting,
        count,
    }) as PyObjectRef
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
    unsafe { (*(obj as *const W_Repeat)).w_obj }
}

/// Read the `counting` field.
///
/// # Safety
/// `obj` must point to a valid `W_Repeat`.
pub unsafe fn w_repeat_get_counting(obj: PyObjectRef) -> bool {
    unsafe { (*(obj as *const W_Repeat)).counting }
}

/// Read the `count` field.
///
/// # Safety
/// `obj` must point to a valid `W_Repeat`.
pub unsafe fn w_repeat_get_count(obj: PyObjectRef) -> i64 {
    unsafe { (*(obj as *const W_Repeat)).count }
}

/// Decrement the `count` field by 1.
///
/// # Safety
/// `obj` must point to a valid `W_Repeat`.
pub unsafe fn w_repeat_dec_count(obj: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_Repeat)).count -= 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn w_count_gc_type_id_matches_descr() {
        assert_eq!(W_COUNT_GC_TYPE_ID, 24);
        assert_eq!(
            <W_Count as crate::lltype::GcType>::TYPE_ID,
            W_COUNT_GC_TYPE_ID
        );
        assert_eq!(
            <W_Count as crate::lltype::GcType>::SIZE,
            W_COUNT_OBJECT_SIZE
        );
    }

    #[test]
    fn w_repeat_gc_type_id_matches_descr() {
        assert_eq!(W_REPEAT_GC_TYPE_ID, 25);
        assert_eq!(
            <W_Repeat as crate::lltype::GcType>::TYPE_ID,
            W_REPEAT_GC_TYPE_ID
        );
        assert_eq!(
            <W_Repeat as crate::lltype::GcType>::SIZE,
            W_REPEAT_OBJECT_SIZE
        );
    }
}
