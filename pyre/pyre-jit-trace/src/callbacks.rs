//! Function pointer table for pyre-jit → pyre-jit-trace callback bridge.
//!
//! pyre-jit-trace cannot depend on pyre-jit (that would be circular).
//! Instead, pyre-jit registers function pointers at init time, and
//! pyre-jit-trace calls them through this table.

use pyre_interpreter::CodeObject;
use pyre_object::PyObjectRef;
use std::cell::Cell;

/// Callback table populated by pyre-jit at initialization.
///
/// All function pointers are `extern "C"` JIT helpers from `call_jit.rs`
/// and eval.rs driver access.
pub struct CallJitCallbacks {
    // call_jit.rs helpers
    pub callee_frame_helper: fn(usize) -> Option<*const ()>,
    pub callable_prefers_function_entry: fn(PyObjectRef) -> bool,
    pub recursive_force_cache_safe: fn(PyObjectRef) -> bool,
    pub jit_drop_callee_frame: *const (),
    pub jit_force_callee_frame: *const (),
    pub jit_force_recursive_call_1: *const (),
    pub jit_force_recursive_call_argraw_boxed_1: *const (),
    pub jit_force_self_recursive_call_argraw_boxed_1: *const (),
    pub jit_create_callee_frame_1: *const (),
    pub jit_create_callee_frame_1_raw_int: *const (),
    pub jit_create_self_recursive_callee_frame_1: *const (),
    pub jit_create_self_recursive_callee_frame_1_raw_int: *const (),
    // eval.rs driver access (opaque pointer to JitDriverPair)
    pub driver_pair: fn() -> *mut u8,
    /// codewriter.py:make_jitcodes parity: build the majit JitCode for
    /// `code` (with full liveness) so that get_list_of_active_boxes uses
    /// the same liveness data as resume.py:1022 enumerate_vars at decode
    /// time. Idempotent: no-op when already built.
    pub ensure_majit_jitcode: fn(*const CodeObject, *const ()),
}

// Safety: function pointers are 'static and never mutated after init
unsafe impl Send for CallJitCallbacks {}
unsafe impl Sync for CallJitCallbacks {}

thread_local! {
    static CALLBACKS: Cell<Option<&'static CallJitCallbacks>> = const { Cell::new(None) };
}

/// Register the callback table. Called once from pyre-jit's eval init.
pub fn init(cb: &'static CallJitCallbacks) {
    CALLBACKS.with(|c| c.set(Some(cb)));
}

/// Get the callback table. Panics if not initialized.
#[inline]
pub fn get() -> &'static CallJitCallbacks {
    CALLBACKS.with(|c| c.get().expect("CallJitCallbacks not initialized"))
}
