//! W_GeneratorObject — Python generator iterator.
//!
//! PyPy equivalent: pypy/interpreter/generator.py GeneratorIterator
//!
//! Wraps a suspended frame. __next__() resumes the frame until
//! YIELD_VALUE (produces a value) or RETURN_VALUE (raises StopIteration).

use crate::pyobject::*;

pub static GENERATOR_TYPE: PyType = crate::pyobject::new_pytype("generator");

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
    /// Whether the generator is currently executing (prevents reentrant calls).
    /// PyPy: GeneratorIterator.running
    pub running: bool,
}

/// GC type id assigned to `W_GeneratorObject` at JitDriver init time.
pub const W_GENERATOR_GC_TYPE_ID: u32 = 32;

/// Fixed payload size (`framework.py:811`).
pub const W_GENERATOR_OBJECT_SIZE: usize = std::mem::size_of::<W_GeneratorObject>();

impl crate::lltype::GcType for W_GeneratorObject {
    const TYPE_ID: u32 = W_GENERATOR_GC_TYPE_ID;
    const SIZE: usize = W_GENERATOR_OBJECT_SIZE;
}

pub fn w_generator_new(frame_ptr: *mut u8) -> PyObjectRef {
    crate::lltype::malloc_typed(W_GeneratorObject {
        ob: PyObject {
            ob_type: &GENERATOR_TYPE as *const PyType,
            w_class: get_instantiate(&GENERATOR_TYPE),
        },
        frame_ptr,
        started: false,
        exhausted: false,
        running: false,
    }) as PyObjectRef
}

#[inline]
pub unsafe fn is_generator(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &GENERATOR_TYPE) }
}

pub unsafe fn w_generator_get_frame(obj: PyObjectRef) -> *mut u8 {
    unsafe { (*(obj as *const W_GeneratorObject)).frame_ptr }
}

pub unsafe fn w_generator_is_exhausted(obj: PyObjectRef) -> bool {
    unsafe { (*(obj as *const W_GeneratorObject)).exhausted }
}

pub unsafe fn w_generator_set_exhausted(obj: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_GeneratorObject)).exhausted = true;
    }
}

pub unsafe fn w_generator_is_started(obj: PyObjectRef) -> bool {
    unsafe { (*(obj as *const W_GeneratorObject)).started }
}

pub unsafe fn w_generator_set_started(obj: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_GeneratorObject)).started = true;
    }
}

pub unsafe fn w_generator_is_running(obj: PyObjectRef) -> bool {
    unsafe { (*(obj as *const W_GeneratorObject)).running }
}

pub unsafe fn w_generator_set_running(obj: PyObjectRef, val: bool) {
    unsafe {
        (*(obj as *mut W_GeneratorObject)).running = val;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn w_generator_gc_type_id_matches_descr() {
        assert_eq!(W_GENERATOR_GC_TYPE_ID, 32);
        assert_eq!(
            <W_GeneratorObject as crate::lltype::GcType>::TYPE_ID,
            W_GENERATOR_GC_TYPE_ID
        );
        assert_eq!(
            <W_GeneratorObject as crate::lltype::GcType>::SIZE,
            W_GENERATOR_OBJECT_SIZE
        );
    }
}
