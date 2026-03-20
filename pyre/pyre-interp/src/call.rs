//! Function call dispatch — pure interpreter, no JIT dependencies.
//!
//! JIT-specific call infrastructure (force/bridge callbacks, callee frame
//! creation helpers, frame pool) lives in pyre-jit/src/call_jit.rs.

use std::cell::Cell;
use std::sync::OnceLock;

use pyre_object::PyObjectRef;
use pyre_runtime::{
    PyResult, dispatch_callable, w_builtin_func_get, w_func_get_closure, w_func_get_code_ptr,
    w_func_get_globals,
};

use crate::eval::eval_frame_plain;
use crate::frame::PyFrame;

/// Store an inline-handled concrete result on the owning caller frame.
pub fn set_pending_inline_result(frame: &mut PyFrame, result: PyObjectRef) {
    frame.pending_inline_result = Some(result);
}

fn take_pending_inline_result(frame: &mut PyFrame) -> Option<PyObjectRef> {
    frame.pending_inline_result.take()
}

// ── Eval function injection ──────────────────────────────────────
type EvalFn = fn(&mut PyFrame) -> PyResult;
static EVAL_OVERRIDE: OnceLock<EvalFn> = OnceLock::new();

type DepthBumpFn = fn() -> Option<Box<dyn std::any::Any>>;
static DEPTH_BUMP_OVERRIDE: OnceLock<DepthBumpFn> = OnceLock::new();

type InlineCallOverrideFn = fn(&PyFrame, PyObjectRef, &[PyObjectRef]) -> Option<PyResult>;
static INLINE_CALL_OVERRIDE: OnceLock<InlineCallOverrideFn> = OnceLock::new();

thread_local! {
    static INLINE_CALL_OVERRIDE_DEPTH: Cell<u32> = const { Cell::new(0) };
}

/// Register the JIT-aware eval function. Called by pyre-jit at startup.
pub fn register_eval_override(f: EvalFn) {
    let _ = EVAL_OVERRIDE.set(f);
}

/// Register the JIT call-depth bump function. Called by pyre-jit at startup.
pub fn register_depth_bump(f: DepthBumpFn) {
    let _ = DEPTH_BUMP_OVERRIDE.set(f);
}

/// Register an inline-only concrete call override. Called by pyre-jit at startup.
pub fn register_inline_call_override(f: InlineCallOverrideFn) {
    let _ = INLINE_CALL_OVERRIDE.set(f);
}

pub struct InlineCallOverrideGuard;

impl Drop for InlineCallOverrideGuard {
    fn drop(&mut self) {
        INLINE_CALL_OVERRIDE_DEPTH.with(|d| d.set(d.get() - 1));
    }
}

/// Enable the inline concrete-call override for the current thread.
pub fn inline_call_override_guard() -> InlineCallOverrideGuard {
    INLINE_CALL_OVERRIDE_DEPTH.with(|d| d.set(d.get() + 1));
    InlineCallOverrideGuard
}

pub struct SuspendInlineCallOverrideGuard {
    previous_depth: u32,
}

impl Drop for SuspendInlineCallOverrideGuard {
    fn drop(&mut self) {
        INLINE_CALL_OVERRIDE_DEPTH.with(|d| d.set(self.previous_depth));
    }
}

/// Temporarily disable the inline concrete-call override for the current
/// thread. Helper-boundary fallback execution must not accidentally reuse
/// the outer inline trace's concrete-call interception seam.
pub fn suspend_inline_call_override() -> SuspendInlineCallOverrideGuard {
    let previous_depth = INLINE_CALL_OVERRIDE_DEPTH.with(|d| {
        let previous = d.get();
        d.set(0);
        previous
    });
    SuspendInlineCallOverrideGuard { previous_depth }
}

pub struct SuspendInlineHandledResultGuard;

impl Drop for SuspendInlineHandledResultGuard {
    fn drop(&mut self) {}
}

/// Temporarily isolate inline-result replay from helper-boundary execution.
///
/// Results are now owned by the concrete caller frame, so helper-boundary
/// interpreters no longer share a thread-global result channel. The guard
/// remains as an explicit protocol marker for residual-call slow paths.
pub fn suspend_inline_handled_result() -> SuspendInlineHandledResultGuard {
    SuspendInlineHandledResultGuard
}

fn try_inline_call_override(
    frame: &PyFrame,
    callable: PyObjectRef,
    args: &[PyObjectRef],
) -> Option<PyResult> {
    let enabled = INLINE_CALL_OVERRIDE_DEPTH.with(|d| d.get() > 0);
    if !enabled {
        return None;
    }
    INLINE_CALL_OVERRIDE
        .get()
        .and_then(|override_fn| override_fn(frame, callable, args))
}

fn call_user_function_with_eval(
    frame: &PyFrame,
    callable: PyObjectRef,
    args: &[PyObjectRef],
    eval_fn: EvalFn,
) -> PyResult {
    let code_ptr = unsafe { w_func_get_code_ptr(callable) };
    let globals = unsafe { w_func_get_globals(callable) };
    let closure = unsafe { w_func_get_closure(callable) };
    let func_code = code_ptr as *const pyre_bytecode::CodeObject;
    let mut func_frame = PyFrame::new_for_call_with_closure(
        func_code,
        args,
        globals,
        frame.execution_context,
        closure,
    );
    func_frame.fix_array_ptrs();
    eval_fn(&mut func_frame)
}

pub fn call_callable(frame: &mut PyFrame, callable: PyObjectRef, args: &[PyObjectRef]) -> PyResult {
    if let Some(result) = take_pending_inline_result(frame) {
        return Ok(result);
    }
    dispatch_callable(
        callable,
        |callable| {
            let func = unsafe { w_builtin_func_get(callable) };
            Ok(func(args))
        },
        |callable| call_user_function(frame, callable, args),
    )
}

pub fn call_user_function(
    frame: &PyFrame,
    callable: PyObjectRef,
    args: &[PyObjectRef],
) -> PyResult {
    if let Some(result) = try_inline_call_override(frame, callable, args) {
        return result;
    }

    let _guard = DEPTH_BUMP_OVERRIDE.get().and_then(|f| f());
    let eval_fn = EVAL_OVERRIDE.get().copied().unwrap_or(eval_frame_plain);
    call_user_function_with_eval(frame, callable, args, eval_fn)
}

/// Plain interpreter-only user-function call.
///
/// JIT residual helpers should use this instead of the injected eval override.
/// PyPy residual calls are opaque slow paths; they should not accidentally
/// re-enter the caller's portal/tracing state.
pub fn call_user_function_plain(
    frame: &PyFrame,
    callable: PyObjectRef,
    args: &[PyObjectRef],
) -> PyResult {
    call_user_function_with_eval(frame, callable, args, eval_frame_plain)
}

/// Explicit residual-call protocol used by JIT inline framestack concrete
/// execution.
///
/// PyPy treats residual calls reached from inline execution as opaque slow
/// paths. They must not accidentally reuse the generic JIT-aware
/// `call_user_function()` entry, because that can re-enter portal state that
/// belongs to the outer trace instead of the active inline framestack.
pub fn call_callable_inline_residual(
    frame: &PyFrame,
    callable: PyObjectRef,
    args: &[PyObjectRef],
) -> PyResult {
    let _suspend_inline_result = suspend_inline_handled_result();
    dispatch_callable(
        callable,
        |callable| {
            let func = unsafe { w_builtin_func_get(callable) };
            Ok(func(args))
        },
        |callable| call_user_function_plain(frame, callable, args),
    )
}
