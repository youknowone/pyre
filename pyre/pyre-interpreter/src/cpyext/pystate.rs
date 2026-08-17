//! Thread state and the GIL -- PyPy `cpyext/pystate.py`.
//!
//! A thread holds the GIL for as long as it runs pyre code, so an extension
//! that blocks has to give it up (`PyEval_SaveThread`, which
//! `Py_BEGIN_ALLOW_THREADS` expands to) and one that calls in from a thread of
//! its own has to take it (`PyGILState_Ensure`).  Both are the runtime's own
//! boundary guards; [`crate::module::thread`] owns them, because a build
//! without this layer exports `PyGILState_Ensure` and `PyGILState_Release` too
//! -- cffi's embedding header calls them.
//!
//! `PyThreadState` is opaque.  Upstream keeps one per execution context and
//! records whether it is the current one (`pystate.py:179 PyThreadState_Get`,
//! `:228 PyThreadState_Swap` through `ec.cpyext_threadstate_is_current`); the
//! same pair here is per thread, since that is what an extension can name.

use std::cell::{Cell, UnsafeCell};
use std::ffi::c_int;

/// The per-thread state an extension is handed a pointer to.
///
/// Opaque: what upstream's `PyThreadState` carries is interpreter bookkeeping
/// an extension reaches through entry points, not through the struct.
#[repr(C)]
pub struct CPyThreadState {
    /// Never read through the pointer; a distinct byte so two threads' states
    /// are distinct addresses.
    _private: u8,
}

thread_local! {
    /// Stable for as long as the thread lives, which is as long as a
    /// `PyThreadState *` naming it is valid.
    static THREAD_STATE: UnsafeCell<CPyThreadState> =
        const { UnsafeCell::new(CPyThreadState { _private: 0 }) };
    /// `ec.cpyext_threadstate_is_current`.  Swapping NULL in clears it, which
    /// is how an extension says the thread state it was handed is no longer the
    /// one in force.
    static STATE_IS_CURRENT: Cell<bool> = const { Cell::new(true) };
}

/// This thread's state, whether or not it is the current one.
fn thread_state() -> *mut CPyThreadState {
    THREAD_STATE.with(|state| state.get())
}

/// `pystate.py:29 PyEval_SaveThread` — drop the GIL, and answer the state the
/// matching [`PyEval_RestoreThread`] is to be handed back.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyEval_SaveThread() -> *mut CPyThreadState {
    let state = thread_state();
    STATE_IS_CURRENT.with(|current| current.set(false));
    crate::module::thread::save_thread();
    state
}

/// `pystate.py:42 PyEval_RestoreThread` — retake the GIL.
///
/// The argument is the state [`PyEval_SaveThread`] answered with, which for any
/// one thread is the only state it can be: the release is bound to the thread
/// that took it, so there is nothing to switch to.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyEval_RestoreThread(_state: *mut CPyThreadState) {
    crate::module::thread::restore_thread();
    STATE_IS_CURRENT.with(|current| current.set(true));
}

/// `PyEval_AcquireThread(tstate)` — take the GIL and make `tstate` current.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyEval_AcquireThread(state: *mut CPyThreadState) {
    unsafe { PyEval_RestoreThread(state) };
}

/// `PyEval_ReleaseThread(tstate)` — the drop half of
/// [`PyEval_AcquireThread`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyEval_ReleaseThread(_state: *mut CPyThreadState) {
    unsafe { PyEval_SaveThread() };
}

/// `pystate.py:179 PyThreadState_Get`.
///
/// Upstream ends the process with a fatal error when there is no current thread
/// state.  A thread that reaches here is running pyre code and so has one.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyThreadState_Get() -> *mut CPyThreadState {
    thread_state()
}

/// `pystate.py:188 _PyThreadState_UncheckedGet` — [`PyThreadState_Get`]
/// without the fatal error, so NULL when this thread's state is not current.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn _PyThreadState_UncheckedGet() -> *mut CPyThreadState {
    if STATE_IS_CURRENT.with(|current| current.get()) {
        thread_state()
    } else {
        std::ptr::null_mut()
    }
}

/// `pystate.py:228 PyThreadState_Swap` — make `state` the current one and
/// answer the one it replaces, or NULL if there was none.
///
/// Upstream refuses to switch to a *different* thread state, which is the only
/// use this can be put to that it cannot serve; a thread can name no state but
/// its own, so NULL ("no state is current") is the only other argument.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyThreadState_Swap(state: *mut CPyThreadState) -> *mut CPyThreadState {
    let previous = if STATE_IS_CURRENT.with(|current| current.get()) {
        thread_state()
    } else {
        std::ptr::null_mut()
    };
    STATE_IS_CURRENT.with(|current| current.set(!state.is_null()));
    previous
}

/// `pystate.py:418 PyGILState_Check` — whether this thread holds the GIL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyGILState_Check() -> c_int {
    crate::module::thread::gilstate_check() as c_int
}

/// `PyGILState_GetThisThreadState()` — this thread's state, or NULL when it has
/// never held the GIL and so has no state of pyre's to name.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyGILState_GetThisThreadState() -> *mut CPyThreadState {
    if crate::module::thread::gilstate_check() {
        thread_state()
    } else {
        std::ptr::null_mut()
    }
}

/// `pystate.py:51 PyEval_InitThreads` — threads are always available, so
/// upstream's `setup_threads` has nothing left to do here.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyEval_InitThreads() {}

/// `pystate.py:57 PyEval_ThreadsInitialized`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyEval_ThreadsInitialized() -> c_int {
    1
}

pub(super) fn ensure_linked() {
    std::hint::black_box(PyEval_SaveThread as *const ());
    std::hint::black_box(PyEval_RestoreThread as *const ());
    std::hint::black_box(PyEval_AcquireThread as *const ());
    std::hint::black_box(PyEval_ReleaseThread as *const ());
    std::hint::black_box(PyThreadState_Get as *const ());
    std::hint::black_box(_PyThreadState_UncheckedGet as *const ());
    std::hint::black_box(PyThreadState_Swap as *const ());
    std::hint::black_box(PyGILState_Check as *const ());
    std::hint::black_box(PyGILState_GetThisThreadState as *const ());
    std::hint::black_box(PyEval_InitThreads as *const ());
    std::hint::black_box(PyEval_ThreadsInitialized as *const ());
}
