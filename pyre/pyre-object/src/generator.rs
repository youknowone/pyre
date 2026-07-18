//! `pypy/interpreter/generator.py` — Python generator iterator.
//!
//! Wraps a suspended frame. __next__() resumes the frame until
//! YIELD_VALUE (produces a value) or RETURN_VALUE (raises StopIteration).

use crate::pyobject::*;
use pyre_macros::pyre_class;

pub static GENERATOR_TYPE: PyType = crate::pyobject::new_pytype("generator");
pub static COROUTINE_TYPE: PyType = crate::pyobject::new_pytype("coroutine");

/// Generator object: holds a boxed frame that can be resumed.
///
/// The frame is stored as a raw pointer to avoid generic type parameters
/// in the object layout (keeps it JIT-compatible).
#[repr(C)]
pub struct GeneratorIterator {
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
    /// Per-generator writable `__name__` override. NULL means read the
    /// suspended frame code's original name.
    pub name: PyObjectRef,
    /// Per-generator writable `__qualname__` override. NULL means read the
    /// suspended frame code's original qualified name.
    pub qualname: PyObjectRef,
    /// PyPy: `Coroutine.w_cr_origin`. Generators keep this at `PY_NULL`;
    /// coroutines initialise it to `None` until origin tracking captures a
    /// tuple of frame summaries.
    pub cr_origin: PyObjectRef,
    /// PyPy: `Coroutine._warned_unawaited`.
    pub warned_unawaited: bool,
}

/// PyPy `generator.py CoroutineWrapper`: the iterator returned by
/// `Coroutine.__await__`, holding the coroutine as its sole GC edge.
#[pyre_class("coroutine_wrapper", static_name = "COROUTINE_WRAPPER")]
pub struct CoroutineWrapper {
    pub coroutine: PyObjectRef,
}

/// GC type id assigned to `GeneratorIterator` at JitDriver init time.
pub const W_GENERATOR_GC_TYPE_ID: u32 = 32;

/// Fixed payload size (`framework.py:811`).
pub const W_GENERATOR_OBJECT_SIZE: usize = std::mem::size_of::<GeneratorIterator>();

impl crate::lltype::GcType for GeneratorIterator {
    fn type_id() -> u32 {
        W_GENERATOR_GC_TYPE_ID
    }
    const SIZE: usize = W_GENERATOR_OBJECT_SIZE;
}

fn w_generator_or_coroutine_new(frame_ptr: *mut u8, coroutine: bool) -> PyObjectRef {
    let value = GeneratorIterator {
        ob: PyObject {
            ob_type: if coroutine {
                &COROUTINE_TYPE as *const PyType
            } else {
                &GENERATOR_TYPE as *const PyType
            },
            w_class: if coroutine {
                get_instantiate(&COROUTINE_TYPE)
            } else {
                get_instantiate(&GENERATOR_TYPE)
            },
        },
        frame_ptr,
        started: false,
        exhausted: false,
        running: false,
        name: PY_NULL,
        qualname: PY_NULL,
        cr_origin: if coroutine { crate::w_none() } else { PY_NULL },
        warned_unawaited: false,
    };
    // A generator must be GC-managed, not immortal `malloc_typed`: the
    // collector never reaches an immortal object, so the registered
    // `generator_object_custom_trace` (which walks the SUSPENDED frame's
    // locals/cells/valuestack via `walk_suspended_generator_frame`) would
    // never run, and a value live only across a `yield` would be reclaimed by
    // a major collection — resuming the generator then dereferences freed
    // memory. Allocate stable (non-moving old-gen) so the many raw
    // `*GeneratorIterator` / `frame_ptr` readers keep a fixed address, and
    // fall back to the immortal alloc only when the GC is not installed.
    let raw =
        crate::gc_hook::try_gc_alloc_stable_raw(W_GENERATOR_GC_TYPE_ID, W_GENERATOR_OBJECT_SIZE);
    if !raw.is_null() {
        crate::gc_interp::note_alloc();
        unsafe {
            std::ptr::write(raw as *mut GeneratorIterator, value);
        }
        // The old-gen generator may reference young frame contents (walked via
        // the custom trace), so remember it for the next minor's tracer.
        crate::gc_hook::try_gc_write_barrier(raw);
        return raw as PyObjectRef;
    }
    crate::lltype::malloc_typed(value) as PyObjectRef
}

pub fn w_generator_new(frame_ptr: *mut u8) -> PyObjectRef {
    w_generator_or_coroutine_new(frame_ptr, false)
}

pub fn w_coroutine_new(frame_ptr: *mut u8) -> PyObjectRef {
    w_generator_or_coroutine_new(frame_ptr, true)
}

pub fn w_coroutine_wrapper_new(coroutine: PyObjectRef) -> PyObjectRef {
    let _roots = crate::gc_roots::push_roots();
    crate::gc_roots::pin_root(coroutine);
    CoroutineWrapper::allocate(CoroutineWrapper {
        ob: PyObject {
            ob_type: std::ptr::null(),
            w_class: std::ptr::null_mut(),
        },
        coroutine,
    })
}

#[inline]
pub unsafe fn is_generator(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &GENERATOR_TYPE) }
}

#[inline]
pub unsafe fn is_coroutine(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &COROUTINE_TYPE) }
}

#[inline]
pub unsafe fn is_generator_or_coroutine(obj: PyObjectRef) -> bool {
    unsafe { is_generator(obj) || is_coroutine(obj) }
}

#[inline]
pub unsafe fn is_coroutine_wrapper(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &COROUTINE_WRAPPER_TYPE) }
}

#[inline]
pub unsafe fn w_coroutine_wrapper_get_coroutine(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const CoroutineWrapper)).coroutine }
}

pub unsafe fn w_generator_get_frame(obj: PyObjectRef) -> *mut u8 {
    unsafe { (*(obj as *const GeneratorIterator)).frame_ptr }
}

pub unsafe fn w_generator_is_exhausted(obj: PyObjectRef) -> bool {
    unsafe { (*(obj as *const GeneratorIterator)).exhausted }
}

pub unsafe fn w_generator_set_exhausted(obj: PyObjectRef) {
    unsafe {
        (*(obj as *mut GeneratorIterator)).exhausted = true;
    }
}

pub unsafe fn w_generator_is_started(obj: PyObjectRef) -> bool {
    unsafe { (*(obj as *const GeneratorIterator)).started }
}

pub unsafe fn w_generator_set_started(obj: PyObjectRef) {
    unsafe {
        (*(obj as *mut GeneratorIterator)).started = true;
    }
}

pub unsafe fn w_generator_is_running(obj: PyObjectRef) -> bool {
    unsafe { (*(obj as *const GeneratorIterator)).running }
}

pub unsafe fn w_generator_set_running(obj: PyObjectRef, val: bool) {
    unsafe {
        (*(obj as *mut GeneratorIterator)).running = val;
    }
}

#[inline]
pub unsafe fn w_generator_get_name(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const GeneratorIterator)).name }
}

#[inline]
pub unsafe fn w_generator_set_name(obj: PyObjectRef, value: PyObjectRef) {
    unsafe { (*(obj as *mut GeneratorIterator)).name = value };
    crate::gc_hook::try_gc_write_barrier(obj as *mut u8);
}

#[inline]
pub unsafe fn w_generator_get_qualname(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const GeneratorIterator)).qualname }
}

#[inline]
pub unsafe fn w_generator_set_qualname(obj: PyObjectRef, value: PyObjectRef) {
    unsafe { (*(obj as *mut GeneratorIterator)).qualname = value };
    crate::gc_hook::try_gc_write_barrier(obj as *mut u8);
}

#[inline]
pub unsafe fn w_coroutine_get_origin(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const GeneratorIterator)).cr_origin }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn w_generator_gc_type_id_matches_descr() {
        assert_eq!(W_GENERATOR_GC_TYPE_ID, 32);
        assert_eq!(
            <GeneratorIterator as crate::lltype::GcType>::type_id(),
            W_GENERATOR_GC_TYPE_ID
        );
        assert_eq!(
            <GeneratorIterator as crate::lltype::GcType>::SIZE,
            W_GENERATOR_OBJECT_SIZE
        );
    }
}
