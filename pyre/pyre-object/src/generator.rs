//! `pypy/interpreter/generator.py` — Python generator iterator.
//!
//! Wraps a suspended frame. __next__() resumes the frame until
//! YIELD_VALUE (produces a value) or RETURN_VALUE (raises StopIteration).

use crate::pyobject::*;
use pyre_macros::pyre_class;

pub static GENERATOR_TYPE: PyType = crate::pyobject::new_pytype("generator");
pub static COROUTINE_TYPE: PyType = crate::pyobject::new_pytype("coroutine");
pub static ASYNC_GENERATOR_TYPE: PyType = crate::pyobject::new_pytype("async_generator");

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
    /// `generator.py:21` `self.pycode = frame.pycode`.  This remains owned by
    /// the generator after `frame` is cleared on exhaustion.
    pub pycode: PyObjectRef,
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
    /// `generator.py:29` `GeneratorOrCoroutine.saved_operr`.  Pyre stores
    /// the materialized exception value carried by `ExecutionContext` rather
    /// than an `OperationError`, but preserves the same per-generator owner.
    pub saved_exc_value: PyObjectRef,
    /// `generator.py:30` `previous_gen_or_coroutine`: the execution-context
    /// linked-list edge while this generator is running.
    pub previous_gen_or_coroutine: PyObjectRef,
    /// `AsyncGenerator.hooks_inited` / `ag_running` / `w_finalizer`.
    /// Generator and coroutine instances keep the neutral values.
    pub hooks_inited: bool,
    pub ag_running: bool,
    pub w_finalizer: PyObjectRef,
}

/// PyPy `generator.py CoroutineWrapper`: the iterator returned by
/// `Coroutine.__await__`, holding the coroutine as its sole GC edge.
#[pyre_class("coroutine_wrapper", static_name = "COROUTINE_WRAPPER")]
pub struct CoroutineWrapper {
    pub coroutine: PyObjectRef,
}

/// PyPy `generator.py AsyncGenValueWrapper`: distinguishes a value yielded
/// by an async generator from a value yielded by an await inside its body.
#[pyre_class(
    "async_generator_wrapped_value",
    static_name = "ASYNC_GEN_VALUE_WRAPPER"
)]
pub struct AsyncGenValueWrapper {
    pub w_value: PyObjectRef,
}

/// Shared state values from PyPy `AsyncGenABase`.
pub const ASYNC_GEN_STATE_INIT: u8 = 0;
pub const ASYNC_GEN_STATE_ITER: u8 = 1;
pub const ASYNC_GEN_STATE_CLOSED: u8 = 2;

#[pyre_class("async_generator_asend", static_name = "ASYNC_GEN_ASEND")]
pub struct AsyncGenASend {
    pub async_gen: PyObjectRef,
    pub w_value_to_send: PyObjectRef,
    pub state: u8,
}

#[pyre_class("async_generator_athrow", static_name = "ASYNC_GEN_ATHROW")]
pub struct AsyncGenAThrow {
    pub async_gen: PyObjectRef,
    /// `PY_NULL` identifies `aclose()`; otherwise these are the arguments to
    /// `athrow()` and optional values use Python `None`.
    pub w_exc_type: PyObjectRef,
    pub w_exc_value: PyObjectRef,
    pub w_exc_tb: PyObjectRef,
    pub state: u8,
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

fn w_generator_or_coroutine_new(
    frame_ptr: *mut u8,
    pycode: PyObjectRef,
    kind: GeneratorKind,
) -> PyObjectRef {
    let _roots = crate::gc_roots::push_roots();
    crate::gc_roots::pin_root(pycode);
    let value = GeneratorIterator {
        ob: PyObject {
            ob_type: match kind {
                GeneratorKind::Generator => &GENERATOR_TYPE as *const PyType,
                GeneratorKind::Coroutine => &COROUTINE_TYPE as *const PyType,
                GeneratorKind::AsyncGenerator => &ASYNC_GENERATOR_TYPE as *const PyType,
            },
            w_class: match kind {
                GeneratorKind::Generator => get_instantiate(&GENERATOR_TYPE),
                GeneratorKind::Coroutine => get_instantiate(&COROUTINE_TYPE),
                GeneratorKind::AsyncGenerator => get_instantiate(&ASYNC_GENERATOR_TYPE),
            },
        },
        frame_ptr,
        pycode,
        started: false,
        exhausted: false,
        running: false,
        name: PY_NULL,
        qualname: PY_NULL,
        cr_origin: if kind == GeneratorKind::Coroutine {
            crate::w_none()
        } else {
            PY_NULL
        },
        warned_unawaited: false,
        saved_exc_value: PY_NULL,
        previous_gen_or_coroutine: PY_NULL,
        hooks_inited: false,
        ag_running: false,
        w_finalizer: PY_NULL,
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

#[derive(Clone, Copy, PartialEq, Eq)]
enum GeneratorKind {
    Generator,
    Coroutine,
    AsyncGenerator,
}

pub fn w_generator_new(frame_ptr: *mut u8, pycode: PyObjectRef) -> PyObjectRef {
    w_generator_or_coroutine_new(frame_ptr, pycode, GeneratorKind::Generator)
}

pub fn w_coroutine_new(frame_ptr: *mut u8, pycode: PyObjectRef) -> PyObjectRef {
    w_generator_or_coroutine_new(frame_ptr, pycode, GeneratorKind::Coroutine)
}

pub fn w_async_generator_new(frame_ptr: *mut u8, pycode: PyObjectRef) -> PyObjectRef {
    w_generator_or_coroutine_new(frame_ptr, pycode, GeneratorKind::AsyncGenerator)
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
pub unsafe fn is_async_generator(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &ASYNC_GENERATOR_TYPE) }
}

#[inline]
pub unsafe fn is_generator_or_coroutine(obj: PyObjectRef) -> bool {
    unsafe { is_generator(obj) || is_coroutine(obj) || is_async_generator(obj) }
}

pub fn w_async_gen_value_wrapper_new(w_value: PyObjectRef) -> PyObjectRef {
    let _roots = crate::gc_roots::push_roots();
    crate::gc_roots::pin_root(w_value);
    AsyncGenValueWrapper::allocate(AsyncGenValueWrapper {
        ob: PyObject {
            ob_type: std::ptr::null(),
            w_class: std::ptr::null_mut(),
        },
        w_value,
    })
}

pub fn w_async_gen_asend_new(async_gen: PyObjectRef, w_value_to_send: PyObjectRef) -> PyObjectRef {
    let _roots = crate::gc_roots::push_roots();
    crate::gc_roots::pin_root(async_gen);
    crate::gc_roots::pin_root(w_value_to_send);
    AsyncGenASend::allocate(AsyncGenASend {
        ob: PyObject {
            ob_type: std::ptr::null(),
            w_class: std::ptr::null_mut(),
        },
        async_gen,
        w_value_to_send,
        state: ASYNC_GEN_STATE_INIT,
    })
}

pub fn w_async_gen_athrow_new(
    async_gen: PyObjectRef,
    w_exc_type: PyObjectRef,
    w_exc_value: PyObjectRef,
    w_exc_tb: PyObjectRef,
) -> PyObjectRef {
    let _roots = crate::gc_roots::push_roots();
    for value in [async_gen, w_exc_type, w_exc_value, w_exc_tb] {
        if !value.is_null() {
            crate::gc_roots::pin_root(value);
        }
    }
    AsyncGenAThrow::allocate(AsyncGenAThrow {
        ob: PyObject {
            ob_type: std::ptr::null(),
            w_class: std::ptr::null_mut(),
        },
        async_gen,
        w_exc_type,
        w_exc_value,
        w_exc_tb,
        state: ASYNC_GEN_STATE_INIT,
    })
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

#[inline]
pub unsafe fn w_generator_get_pycode(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const GeneratorIterator)).pycode }
}

#[inline]
pub unsafe fn w_generator_set_frame(obj: PyObjectRef, frame_ptr: *mut u8) {
    unsafe { (*(obj as *mut GeneratorIterator)).frame_ptr = frame_ptr };
    if !frame_ptr.is_null() {
        crate::gc_hook::try_gc_write_barrier(obj as *mut u8);
    }
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
pub unsafe fn w_generator_get_saved_exc_value(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const GeneratorIterator)).saved_exc_value }
}

#[inline]
pub unsafe fn w_generator_set_saved_exc_value(obj: PyObjectRef, value: PyObjectRef) {
    unsafe { (*(obj as *mut GeneratorIterator)).saved_exc_value = value };
    crate::gc_hook::try_gc_write_barrier(obj as *mut u8);
}

#[inline]
pub unsafe fn w_generator_get_previous(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const GeneratorIterator)).previous_gen_or_coroutine }
}

#[inline]
pub unsafe fn w_generator_set_previous(obj: PyObjectRef, value: PyObjectRef) {
    unsafe { (*(obj as *mut GeneratorIterator)).previous_gen_or_coroutine = value };
    crate::gc_hook::try_gc_write_barrier(obj as *mut u8);
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

#[inline]
pub unsafe fn w_async_generator_hooks_inited(obj: PyObjectRef) -> bool {
    unsafe { (*(obj as *const GeneratorIterator)).hooks_inited }
}

#[inline]
pub unsafe fn w_async_generator_set_hooks_inited(obj: PyObjectRef) {
    unsafe { (*(obj as *mut GeneratorIterator)).hooks_inited = true };
}

#[inline]
pub unsafe fn w_async_generator_is_running(obj: PyObjectRef) -> bool {
    unsafe { (*(obj as *const GeneratorIterator)).ag_running }
}

#[inline]
pub unsafe fn w_async_generator_set_running(obj: PyObjectRef, value: bool) {
    unsafe { (*(obj as *mut GeneratorIterator)).ag_running = value };
}

#[inline]
pub unsafe fn w_async_generator_get_finalizer(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const GeneratorIterator)).w_finalizer }
}

#[inline]
pub unsafe fn w_async_generator_set_finalizer(obj: PyObjectRef, value: PyObjectRef) {
    unsafe { (*(obj as *mut GeneratorIterator)).w_finalizer = value };
    crate::gc_hook::try_gc_write_barrier(obj as *mut u8);
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
