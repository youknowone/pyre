//! W_GeneratorObject — Python generator iterator.
//!
//! PyPy equivalent: pypy/interpreter/generator.py GeneratorIterator
//!
//! Wraps a suspended frame. __next__() resumes the frame until
//! YIELD_VALUE (produces a value) or RETURN_VALUE (raises StopIteration).

use crate::pyobject::*;

pub static GENERATOR_TYPE: PyType = PyType {
    tp_name: "generator",
};

/// Generator object: holds a boxed frame that can be resumed.
///
/// The frame is stored as a raw pointer to avoid generic type parameters
/// in the object layout (keeps it JIT-compatible).
#[repr(C)]
pub struct W_GeneratorObject {
    pub ob: PyObject,
    /// Opaque pointer to the suspended PyFrame (Box<PyFrame>).
    /// NULL when the generator is exhausted.
    pub frame_ptr: *mut u8,
    /// Whether the generator has been started (first __next__ called).
    pub started: bool,
    /// Whether the generator is exhausted.
    pub exhausted: bool,
}

pub fn w_generator_new(frame_ptr: *mut u8) -> PyObjectRef {
    let obj = Box::new(W_GeneratorObject {
        ob: PyObject {
            ob_type: &GENERATOR_TYPE as *const PyType,
            w_class: std::ptr::null_mut(),
        },
        frame_ptr,
        started: false,
        exhausted: false,
    });
    Box::into_raw(obj) as PyObjectRef
}

#[inline]
pub unsafe fn is_generator(obj: PyObjectRef) -> bool {
    py_type_check(obj, &GENERATOR_TYPE)
}

pub unsafe fn w_generator_get_frame(obj: PyObjectRef) -> *mut u8 {
    (*(obj as *const W_GeneratorObject)).frame_ptr
}

pub unsafe fn w_generator_is_exhausted(obj: PyObjectRef) -> bool {
    (*(obj as *const W_GeneratorObject)).exhausted
}

pub unsafe fn w_generator_set_exhausted(obj: PyObjectRef) {
    (*(obj as *mut W_GeneratorObject)).exhausted = true;
}

pub unsafe fn w_generator_is_started(obj: PyObjectRef) -> bool {
    (*(obj as *const W_GeneratorObject)).started
}

pub unsafe fn w_generator_set_started(obj: PyObjectRef) {
    (*(obj as *mut W_GeneratorObject)).started = true;
}
