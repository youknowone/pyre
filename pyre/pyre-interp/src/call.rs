use std::cell::Cell;
use std::sync::{Once, OnceLock};

use pyre_object::PyObjectRef;
use pyre_object::intobject::w_int_get_value;
use pyre_object::pyobject::is_int;
use pyre_runtime::{
    PyResult, dispatch_callable, register_jit_function_caller, w_builtin_func_get,
    w_func_get_code_ptr, w_func_get_globals,
};

use crate::eval::{eval_frame, eval_frame_plain, jit_call_depth_bump};
use crate::frame::PyFrame;

// ── Eval function injection ──────────────────────────────────────
// PyPy's __extend__(PyFrame) replaces dispatch() at load time.
// Rust equivalent: OnceLock<fn> set by pyre-mjit's generated code.
// After initialization, get() is a single atomic load (branch-predicted).

type EvalFn = fn(&mut PyFrame) -> PyResult;
static EVAL_OVERRIDE: OnceLock<EvalFn> = OnceLock::new();

/// Register the JIT-aware eval function. Called by pyre-mjit at startup.
/// After this, all recursive function calls go through the JIT eval loop.
pub fn register_eval_override(f: EvalFn) {
    let _ = EVAL_OVERRIDE.set(f);
}

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
// Uses raw statics instead of thread_local! to avoid TLS lookup overhead
// on the hot path (~29M calls for fib(35)).
static mut FORCE_CACHE_0: (usize, usize, i64) = (0, 0, 0);
static mut FORCE_CACHE_1: (usize, usize, i64) = (0, 0, 0);

/// Compute a stable cache key for a Python argument.
///
/// For integers, uses the unboxed value so that different allocations
/// of the same int value produce the same key. For other types, falls
/// back to the object pointer address.
#[inline]
fn force_cache_arg_key(arg: PyObjectRef) -> usize {
    if arg as usize == 0 {
        return 0;
    }
    if unsafe { is_int(arg) } {
        // Use the unboxed int value. Shift left by 1 and set bit 0
        // to distinguish from pointer-based keys (pointers are aligned,
        // so their bit 0 is always 0).
        let v = unsafe { w_int_get_value(arg) };
        ((v as usize) << 1) | 1
    } else {
        arg as usize
    }
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
    let _guard = jit_call_depth_bump();

    // Use JIT-aware eval if registered (by pyre-mjit), otherwise plain.
    // OnceLock::get() is a single atomic load after initialization.
    let eval_fn = EVAL_OVERRIDE.get().copied().unwrap_or(eval_frame_plain);
    eval_fn(&mut func_frame)
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
        force_cache_arg_key(frame.locals_w[0])
    } else {
        0
    };

    // Fast path: check 2-entry static cache (no TLS overhead)
    unsafe {
        if FORCE_CACHE_0.0 == code_key && FORCE_CACHE_0.1 == arg_key {
            return FORCE_CACHE_0.2;
        }
        if FORCE_CACHE_1.0 == code_key && FORCE_CACHE_1.1 == arg_key {
            return FORCE_CACHE_1.2;
        }
    }

    // Slow path: run interpreter, then cache the result
    match crate::eval::eval_loop_for_force(frame) {
        Ok(result) => {
            unsafe {
                FORCE_CACHE_1 = FORCE_CACHE_0;
                FORCE_CACHE_0 = (code_key, arg_key, result as i64);
            }
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
        majit_codegen_cranelift::register_call_assembler_bridge(jit_bridge_compile_callee);
    });
}

/// Bridge compilation callback for call_assembler guard failures.
///
/// When a guard in the callee's compiled trace fails enough times
/// (>= DEFAULT_BRIDGE_THRESHOLD), this callback is invoked to compile
/// a bridge. The bridge calls `jit_force_callee_frame` in compiled code,
/// eliminating the Rust overhead of call_assembler_fast_path on
/// subsequent guard failures.
extern "C" fn jit_bridge_compile_callee(
    frame_ptr: i64,
    fail_index: u32,
    trace_id: u64,
    green_key: u64,
) -> i64 {
    use majit_ir::{InputArg, Op, OpCode, OpRef, Type};
    use std::collections::HashMap;

    // 1. Run the interpreter to get the concrete result (same as force_fn)
    let frame = unsafe { &mut *(frame_ptr as *mut PyFrame) };
    let result = match crate::eval::eval_loop_for_force(frame) {
        Ok(r) => r as i64,
        Err(e) => panic!("bridge force failed: {e}"),
    };

    // 2. Populate FORCE_CACHE with this result
    let code_key = frame.code as usize;
    let arg_key = if frame.locals_w.len() > 0 {
        force_cache_arg_key(frame.locals_w[0])
    } else {
        0
    };
    unsafe {
        FORCE_CACHE_1 = FORCE_CACHE_0;
        FORCE_CACHE_0 = (code_key, arg_key, result);
    }

    // 3. Build bridge IR: inputarg(frame_ptr) → call force_fn → FINISH
    //    Function pointers are encoded as constants (first arg to CALL_I).
    let bridge_inputargs = vec![InputArg::from_type(Type::Int, 0)];
    let frame_opref = OpRef(0); // inputarg 0

    // Constant for force_fn pointer
    let force_fn_ptr = jit_force_callee_frame as *const () as i64;
    let func_const_ref = OpRef(10_000); // constant OpRef
    let mut constants: HashMap<u32, i64> = HashMap::new();
    constants.insert(func_const_ref.0, force_fn_ptr);

    // CALL_I(force_fn, frame_ptr) → result
    let call_descr = majit_meta::make_call_descr(&[Type::Int], Type::Int);
    let call_result = OpRef(1);
    let mut call_op = Op::with_descr(OpCode::CallI, &[func_const_ref, frame_opref], call_descr);
    call_op.pos = call_result;

    // FINISH(result)
    let finish_descr = majit_meta::make_fail_descr_typed(vec![Type::Int]);
    let mut finish_op = Op::with_descr(OpCode::Finish, &[call_result], finish_descr);
    finish_op.pos = OpRef(2);

    let bridge_ops = vec![call_op, finish_op];

    // 4. Get the MetaInterp and compile the bridge
    let driver = crate::eval::driver_pair();
    let meta = driver.0.meta_interp_mut();

    if let Some(fail_descr) = meta.get_fail_descr_for_bridge(green_key, trace_id, fail_index) {
        meta.compile_bridge(
            green_key,
            fail_index,
            fail_descr.as_ref(),
            &bridge_ops,
            &bridge_inputargs,
            constants,
        );
    }

    result
}

// ── Callee frame creation for call_assembler ─────────────────────

fn create_callee_frame_impl(caller_frame: i64, callable: i64, args: &[PyObjectRef]) -> i64 {
    let callable = callable as PyObjectRef;
    let code_ptr = unsafe { w_func_get_code_ptr(callable) };

    // Force cache tagged pointer disabled here — downstream compiled
    // GETFIELD ops dereference the return value, so it must be a valid
    // frame pointer. Result caching is handled in call_assembler_shim's
    // fast_path after target code execution.

    // Create frame
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

pub extern "C" fn jit_create_callee_frame_1(caller_frame: i64, callable: i64, arg0: i64) -> i64 {
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
