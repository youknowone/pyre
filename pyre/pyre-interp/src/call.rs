use std::cell::Cell;
use std::sync::Once;

use pyre_object::PyObjectRef;
use pyre_runtime::{
    PyResult, dispatch_callable, register_jit_function_caller, w_builtin_func_get,
    w_func_get_code_ptr, w_func_get_globals,
};

use crate::eval::{eval_frame, jit_call_depth_bump};
use crate::frame::PyFrame;

thread_local! {
    /// When trace inlining executes the callee synchronously inside the
    /// trace handler, the concrete result is stored here so that the
    /// concrete CALL_FUNCTION handler can pick it up without re-executing.
    static INLINE_HANDLED_RESULT: Cell<Option<PyObjectRef>> = const { Cell::new(None) };
}

/// Store an inline-handled result for the concrete handler to pick up.
pub(crate) fn set_inline_handled_result(result: PyObjectRef) {
    INLINE_HANDLED_RESULT.with(|c| c.set(Some(result)));
}

// Fast 2-entry cache for force callback results.
// Covers the common case (e.g. fib base cases: n=0 and n=1).
thread_local! {
    static FORCE_CACHE: Cell<[(usize, usize, i64); 2]> =
        const { Cell::new([(0, 0, 0); 2]) };
}

// ── Callee frame pool ────────────────────────────────────────────
//
// Thread-local LIFO pool of pre-allocated PyFrame boxes.
// Eliminates heap allocation overhead for the common case where
// frames are created and destroyed in rapid succession (e.g.
// recursive call_assembler calls).

const FRAME_POOL_CAP: usize = 4;

thread_local! {
    static FRAME_POOL: Cell<([*mut PyFrame; FRAME_POOL_CAP], usize)> =
        const { Cell::new(([std::ptr::null_mut(); FRAME_POOL_CAP], 0)) };
}

#[inline]
fn pool_take() -> *mut PyFrame {
    FRAME_POOL
        .try_with(|c| {
            let (mut arr, len) = c.get();
            if len > 0 {
                let new_len = len - 1;
                let ptr = arr[new_len];
                arr[new_len] = std::ptr::null_mut();
                c.set((arr, new_len));
                ptr
            } else {
                std::ptr::null_mut()
            }
        })
        .unwrap_or(std::ptr::null_mut())
}

#[inline]
fn pool_put(ptr: *mut PyFrame) -> bool {
    FRAME_POOL
        .try_with(|c| {
            let (mut arr, len) = c.get();
            if len < FRAME_POOL_CAP {
                arr[len] = ptr;
                c.set((arr, len + 1));
                true
            } else {
                false
            }
        })
        .unwrap_or(false)
}

pub fn call_callable(frame: &mut PyFrame, callable: PyObjectRef, args: &[PyObjectRef]) -> PyResult {
    // If the trace handler already executed this call inline,
    // return the pre-computed result without re-executing.
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

    // When inside a JIT context (tracing or compiled code), bump the
    // call depth so that the callee's eval_loop skips merge_points
    // and try_function_entry_jit. This prevents inner instructions
    // from polluting the outer trace, and prevents infinite recursion
    // through compiled → residual → compiled call chains.
    // Matches PyPy's residual call suspension.
    let _guard = jit_call_depth_bump();
    eval_frame(&mut func_frame)
}

extern "C" fn jit_call_user_function_from_frame(
    frame_ptr: i64,
    callable: i64,
    args_ptr: *const i64,
    nargs: i64,
) -> i64 {
    let frame = unsafe { &*(frame_ptr as *const PyFrame) };
    let args =
        unsafe { std::slice::from_raw_parts(args_ptr as *const PyObjectRef, nargs as usize) };
    match call_user_function(frame, callable as PyObjectRef, args) {
        Ok(result) => result as i64,
        Err(err) => panic!("jit user-function call failed: {err}"),
    }
}

/// Force a callee frame through the interpreter when call_assembler
/// hits a guard failure.
///
/// Uses a small result cache keyed by (code_ptr, first_local) so that
/// repeated base-case calls with the same argument (e.g. fib(0),
/// fib(1)) return instantly without re-entering the interpreter.
extern "C" fn jit_force_callee_frame(frame_ptr: i64) -> i64 {
    let frame = unsafe { &mut *(frame_ptr as *mut PyFrame) };

    let code_key = frame.code as usize;
    let arg_key = if frame.locals_w.len() > 0 {
        frame.locals_w[0] as usize
    } else {
        0
    };

    // Fast path: check 2-entry cache
    if let Ok(entries) = FORCE_CACHE.try_with(|c| c.get()) {
        if entries[0].0 == code_key && entries[0].1 == arg_key {
            return entries[0].2;
        }
        if entries[1].0 == code_key && entries[1].1 == arg_key {
            return entries[1].2;
        }
    }

    // Slow path: run interpreter, then cache the result
    match crate::eval::eval_loop_for_force(frame) {
        Ok(result) => {
            let _ = FORCE_CACHE.try_with(|c| {
                let mut entries = c.get();
                // Shift entry 0 → 1, insert new at 0
                entries[1] = entries[0];
                entries[0] = (code_key, arg_key, result as i64);
                c.set(entries);
            });
            result as i64
        }
        Err(err) => panic!("jit force callee frame failed: {err}"),
    }
}

pub fn install_jit_call_bridge() {
    static INSTALL: Once = Once::new();
    INSTALL.call_once(|| {
        register_jit_function_caller(jit_call_user_function_from_frame);
        majit_codegen_cranelift::register_call_assembler_force(jit_force_callee_frame);
    });
}

// ── Callee frame creation for call_assembler ─────────────────────

fn create_callee_frame_impl(caller_frame: i64, callable: i64, args: &[PyObjectRef]) -> i64 {
    let callable = callable as PyObjectRef;
    let code_ptr = unsafe { w_func_get_code_ptr(callable) };

    // Check force cache before allocating a frame.
    // On hit, return a tagged pointer (bit 0 = 1, bits 1.. = result)
    // that the call_assembler shim recognizes as a pre-computed result.
    let code_key = code_ptr as usize;
    let arg_key = if !args.is_empty() {
        args[0] as usize
    } else {
        0
    };
    if let Ok(entries) = FORCE_CACHE.try_with(|c| c.get()) {
        if entries[0].0 == code_key && entries[0].1 == arg_key {
            return (entries[0].2 << 1) | 1;
        }
        if entries[1].0 == code_key && entries[1].1 == arg_key {
            return (entries[1].2 << 1) | 1;
        }
    }

    // Cache miss: create frame normally
    let frame = unsafe { &*(caller_frame as *const PyFrame) };
    let globals = unsafe { w_func_get_globals(callable) };
    let func_code = code_ptr as *const pyre_bytecode::CodeObject;

    let raw = pool_take();
    let frame_ptr = if raw.is_null() {
        Box::into_raw(Box::new(PyFrame::new_for_call(
            func_code,
            args,
            globals,
            frame.execution_context,
        )))
    } else {
        unsafe {
            std::ptr::drop_in_place(raw);
            std::ptr::write(
                raw,
                PyFrame::new_for_call(func_code, args, globals, frame.execution_context),
            );
        }
        raw
    };
    unsafe { &mut *frame_ptr }.fix_array_ptrs();
    frame_ptr as i64
}

pub extern "C" fn jit_create_callee_frame_0(caller_frame: i64, callable: i64) -> i64 {
    create_callee_frame_impl(caller_frame, callable, &[])
}

pub extern "C" fn jit_create_callee_frame_1(
    caller_frame: i64,
    callable: i64,
    arg0: i64,
) -> i64 {
    create_callee_frame_impl(caller_frame, callable, &[arg0 as PyObjectRef])
}

pub extern "C" fn jit_create_callee_frame_2(
    caller_frame: i64,
    callable: i64,
    arg0: i64,
    arg1: i64,
) -> i64 {
    create_callee_frame_impl(
        caller_frame,
        callable,
        &[arg0 as PyObjectRef, arg1 as PyObjectRef],
    )
}

pub extern "C" fn jit_create_callee_frame_3(
    caller_frame: i64,
    callable: i64,
    arg0: i64,
    arg1: i64,
    arg2: i64,
) -> i64 {
    create_callee_frame_impl(
        caller_frame,
        callable,
        &[
            arg0 as PyObjectRef,
            arg1 as PyObjectRef,
            arg2 as PyObjectRef,
        ],
    )
}

pub extern "C" fn jit_create_callee_frame_4(
    caller_frame: i64,
    callable: i64,
    arg0: i64,
    arg1: i64,
    arg2: i64,
    arg3: i64,
) -> i64 {
    create_callee_frame_impl(
        caller_frame,
        callable,
        &[
            arg0 as PyObjectRef,
            arg1 as PyObjectRef,
            arg2 as PyObjectRef,
            arg3 as PyObjectRef,
        ],
    )
}

/// Return the arity-specific callee-frame creation helper.
pub fn callee_frame_helper(nargs: usize) -> Option<*const ()> {
    match nargs {
        0 => Some(jit_create_callee_frame_0 as *const ()),
        1 => Some(jit_create_callee_frame_1 as *const ()),
        2 => Some(jit_create_callee_frame_2 as *const ()),
        3 => Some(jit_create_callee_frame_3 as *const ()),
        4 => Some(jit_create_callee_frame_4 as *const ()),
        _ => None,
    }
}

/// Return a callee frame to the pool, or drop it if the pool is full.
/// Tagged pointers (odd values from cached results) are ignored.
pub extern "C" fn jit_drop_callee_frame(frame_ptr: i64) {
    if frame_ptr & 1 != 0 {
        return; // Tagged cached result — no frame was allocated
    }
    let ptr = frame_ptr as *mut PyFrame;
    if !pool_put(ptr) {
        unsafe { drop(Box::from_raw(ptr)) };
    }
}
