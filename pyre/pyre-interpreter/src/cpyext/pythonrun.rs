//! Starting and stopping the interpreter -- PyPy `cpyext/pythonrun.py`.
//!
//! An extension does not start pyre; it is loaded by an interpreter that is
//! already running. What is left of this file upstream is therefore the pair
//! of questions an extension asks about that interpreter's state, and both
//! answer from what the running process already knows.

use std::ffi::c_int;

/// `pythonrun.py Py_IsInitialized` — an extension can only be here because
/// the interpreter that loaded it is running, so the answer is constant.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn Py_IsInitialized() -> c_int {
    1
}

/// `Py_IsFinalizing` — whether interpreter teardown has begun.
///
/// The flag is the one `sys.is_finalizing` reports, set once atexit callbacks
/// are done and module teardown is about to start. An extension asks so that a
/// finalizer of its own can decline to call back in, which is exactly the
/// window this covers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn Py_IsFinalizing() -> c_int {
    crate::module::thread::is_finalizing() as c_int
}

pub(super) fn ensure_linked() {
    std::hint::black_box(Py_IsInitialized as *const ());
    std::hint::black_box(Py_IsFinalizing as *const ());
}
