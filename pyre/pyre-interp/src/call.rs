//! Function call dispatch — pure interpreter, no JIT dependencies.
//!
//! JIT-specific call infrastructure (force/bridge callbacks, callee frame
//! creation helpers, frame pool) lives in pyre-jit/src/call_jit.rs.

use std::cell::Cell;
use std::sync::OnceLock;

use pyre_object::PyObjectRef;
use pyre_runtime::{
    PyResult, dispatch_callable, w_builtin_func_get, w_func_get_code_ptr, w_func_get_globals,
};

use crate::eval::eval_frame_plain;
use crate::frame::PyFrame;

thread_local! {
    /// When trace inlining executes the callee synchronously inside the
    /// trace handler, the concrete result is stored here so that the
    /// concrete CALL_FUNCTION handler can pick it up without re-executing.
    static INLINE_HANDLED_RESULT: Cell<Option<PyObjectRef>> = const { Cell::new(None) };
}

/// Store an inline-handled result for the concrete handler to pick up.
pub fn set_inline_handled_result(result: PyObjectRef) {
    INLINE_HANDLED_RESULT.with(|c| c.set(Some(result)));
}

// ── Eval function injection ──────────────────────────────────────
type EvalFn = fn(&mut PyFrame) -> PyResult;
static EVAL_OVERRIDE: OnceLock<EvalFn> = OnceLock::new();

type DepthBumpFn = fn() -> Option<Box<dyn std::any::Any>>;
static DEPTH_BUMP_OVERRIDE: OnceLock<DepthBumpFn> = OnceLock::new();

/// Register the JIT-aware eval function. Called by pyre-jit at startup.
pub fn register_eval_override(f: EvalFn) {
    let _ = EVAL_OVERRIDE.set(f);
}

/// Register the JIT call-depth bump function. Called by pyre-jit at startup.
pub fn register_depth_bump(f: DepthBumpFn) {
    let _ = DEPTH_BUMP_OVERRIDE.set(f);
}

pub fn call_callable(frame: &mut PyFrame, callable: PyObjectRef, args: &[PyObjectRef]) -> PyResult {
    if let Ok(Some(result)) = INLINE_HANDLED_RESULT.try_with(|c| c.take()) {
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
    let code_ptr = unsafe { w_func_get_code_ptr(callable) };
    let globals = unsafe { w_func_get_globals(callable) };
    let func_code = code_ptr as *const pyre_bytecode::CodeObject;
    let mut func_frame = PyFrame::new_for_call(func_code, args, globals, frame.execution_context);
    func_frame.fix_array_ptrs();

    let _guard = DEPTH_BUMP_OVERRIDE.get().and_then(|f| f());

    let eval_fn = EVAL_OVERRIDE.get().copied().unwrap_or(eval_frame_plain);
    eval_fn(&mut func_frame)
}
