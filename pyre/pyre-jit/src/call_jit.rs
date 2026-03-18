//! JIT-specific call infrastructure — force/bridge callbacks, callee
//! frame creation helpers, frame pool.
//!
//! Separated from pyre-interp/src/call.rs so pyre-interp stays JIT-free.

use std::cell::{Cell, UnsafeCell};
use std::mem::MaybeUninit;
use std::sync::Once;

use pyre_object::PyObjectRef;
use pyre_object::intobject::w_int_get_value;
use pyre_object::pyobject::is_int;
use pyre_runtime::{register_jit_function_caller, w_func_get_code_ptr, w_func_get_globals};

use pyre_interp::frame::PyFrame;

// ── Force cache (memoization for recursive force results) ────────
//
// PyPy doesn't need this (compiled code dispatches directly), but
// our force_fn path benefits from caching intermediate fib results.
const FORCE_CACHE_SIZE: usize = 64;
static mut FORCE_CACHE: [(usize, usize, i64); FORCE_CACHE_SIZE] =
    [(0, 0, 0); FORCE_CACHE_SIZE];

#[inline]
fn force_cache_arg_key(arg: PyObjectRef) -> usize {
    if arg as usize == 0 {
        return 0;
    }
    if unsafe { is_int(arg) } {
        let v = unsafe { w_int_get_value(arg) };
        ((v as usize) << 1) | 1
    } else {
        arg as usize
    }
}

// ── Callee frame arena (RPython nursery bump equivalent) ─────────
//
// LIFO stack of pre-allocated PyFrame slots. Recursive call/return
// order is naturally LIFO, so arena_take/arena_put are O(1).
// Eliminates heap allocation for recursion depths up to ARENA_CAP.

const ARENA_CAP: usize = 64;

struct FrameArena {
    buf: Box<[MaybeUninit<PyFrame>; ARENA_CAP]>,
    /// Number of frames currently in use (LIFO stack pointer).
    top: usize,
    /// Frames below this index have been initialized at least once.
    /// Reuse only needs reinit of changed fields, not full new_for_call.
    initialized: usize,
}

impl FrameArena {
    fn new() -> Self {
        Self {
            buf: Box::new([const { MaybeUninit::uninit() }; ARENA_CAP]),
            top: 0,
            initialized: 0,
        }
    }

    /// Take the next frame slot. Returns (ptr, was_previously_initialized).
    #[inline]
    fn take(&mut self) -> Option<(*mut PyFrame, bool)> {
        if self.top < ARENA_CAP {
            let idx = self.top;
            self.top += 1;
            let ptr = self.buf[idx].as_mut_ptr();
            let was_init = idx < self.initialized;
            Some((ptr, was_init))
        } else {
            None
        }
    }

    /// Return a frame to the arena. Must be the most recently taken frame (LIFO).
    #[inline]
    fn put(&mut self, ptr: *mut PyFrame) -> bool {
        if self.top > 0 && ptr == self.buf[self.top - 1].as_mut_ptr() {
            self.top -= 1;
            return true;
        }
        // Check if within arena range — don't free, but mark as non-LIFO.
        let base = self.buf[0].as_mut_ptr() as usize;
        let end = unsafe { self.buf.as_ptr().add(ARENA_CAP) as usize };
        let addr = ptr as usize;
        addr >= base && addr < end
    }

    /// Mark that frames up to `top` have been fully initialized.
    #[inline]
    fn mark_initialized(&mut self) {
        if self.top > self.initialized {
            self.initialized = self.top;
        }
    }
}

thread_local! {
    static FRAME_ARENA: UnsafeCell<FrameArena> = UnsafeCell::new(FrameArena::new());
}

#[inline]
fn arena_ref() -> &'static mut FrameArena {
    FRAME_ARENA.with(|cell| unsafe { &mut *cell.get() })
}

// ── JIT call callbacks ───────────────────────────────────────────

extern "C" fn jit_call_user_function_from_frame(
    frame_ptr: i64,
    callable: i64,
    args_ptr: *const i64,
    nargs: i64,
) -> i64 {
    let frame = unsafe { &*(frame_ptr as *const PyFrame) };
    let args =
        unsafe { std::slice::from_raw_parts(args_ptr as *const PyObjectRef, nargs as usize) };
    match pyre_interp::call::call_user_function(frame, callable as PyObjectRef, args) {
        Ok(result) => result as i64,
        Err(err) => panic!("jit user-function call failed: {err}"),
    }
}

pub extern "C" fn jit_force_callee_frame(frame_ptr: i64) -> i64 {
    let frame = unsafe { &mut *(frame_ptr as *mut PyFrame) };

    let code_key = frame.code as usize;
    let arg_key = if frame.locals_cells_stack_w.len() > 0 {
        force_cache_arg_key(frame.locals_cells_stack_w[0])
    } else {
        0
    };

    // Hash-based force cache lookup (64 entries)
    let hash_idx = (code_key.wrapping_mul(2654435761) ^ arg_key) % FORCE_CACHE_SIZE;
    unsafe {
        let entry = &FORCE_CACHE[hash_idx];
        if entry.0 == code_key && entry.1 == arg_key {
            return entry.2;
        }
    }

    // PyPy assembler_call_helper (warmspot.py:1021): try compiled code
    // dispatch first, fall back to interpreter.
    let green_key = frame.code as u64;
    let (driver, _) = crate::eval::driver_pair();
    if let Some(token) = driver.get_loop_token(green_key) {
        let token_num = token.number;
        let nlocals = unsafe { (&*frame.code).varnames.len() };
        let mut inputs = vec![
            frame_ptr,
            frame.next_instr as i64,
            frame.valuestackdepth as i64,
        ];
        for i in 0..nlocals {
            inputs.push(frame.locals_cells_stack_w[i] as i64);
        }
        if let Some(raw) = majit_codegen_cranelift::execute_call_assembler_direct(
            token_num,
            &inputs,
            jit_force_callee_frame_interp,
        ) {
            unsafe { FORCE_CACHE[hash_idx] = (code_key, arg_key, raw); }
            return raw;
        }
    }

    // Fallback: full eval_with_jit (interpreter + JIT)
    match crate::eval::eval_with_jit(frame) {
        Ok(result) => {
            let raw = if !result.is_null() && unsafe { is_int(result) } {
                unsafe { w_int_get_value(result) }
            } else {
                result as i64
            };
            unsafe { FORCE_CACHE[hash_idx] = (code_key, arg_key, raw); }
            raw
        }
        Err(err) => panic!("jit force callee frame failed: {err}"),
    }
}

/// Interpreter-only force: used by execute_call_assembler_direct
/// to handle guard failures without recursive compiled dispatch.
extern "C" fn jit_force_callee_frame_interp(frame_ptr: i64) -> i64 {
    let frame = unsafe { &mut *(frame_ptr as *mut PyFrame) };

    let code_key = frame.code as usize;
    let arg_key = if frame.locals_cells_stack_w.len() > 0 {
        force_cache_arg_key(frame.locals_cells_stack_w[0])
    } else {
        0
    };

    let hash_idx = (code_key.wrapping_mul(2654435761) ^ arg_key) % FORCE_CACHE_SIZE;
    unsafe {
        let entry = &FORCE_CACHE[hash_idx];
        if entry.0 == code_key && entry.1 == arg_key {
            return entry.2;
        }
    }

    match pyre_interp::eval::eval_loop_for_force(frame) {
        Ok(result) => {
            let raw = if !result.is_null() && unsafe { is_int(result) } {
                unsafe { w_int_get_value(result) }
            } else {
                result as i64
            };
            unsafe { FORCE_CACHE[hash_idx] = (code_key, arg_key, raw); }
            raw
        }
        Err(err) => panic!("jit force callee frame (interp) failed: {err}"),
    }
}

/// Fused recursive call: create frame + compiled dispatch + drop, all in one.
///
/// Eliminates 2 of 4 Call ops per recursive call in the trace:
///   Before: CallI(box) + CallI(create_frame) + CallMayForce(force) + CallN(drop)
///   After:  CallI(box) + CallMayForce(force_recursive, caller_frame, callable, boxed_arg)
///
/// Takes a boxed arg (already traced via trace_box_int), creates callee frame
/// internally, dispatches to compiled code, drops frame, returns raw int result.
pub extern "C" fn jit_force_recursive_call_1(
    caller_frame: i64,
    callable: i64,
    boxed_arg: i64,
) -> i64 {
    let callable_ref = callable as PyObjectRef;
    let code_ptr = unsafe { w_func_get_code_ptr(callable_ref) };
    let code_key = code_ptr as usize;
    let arg_key = force_cache_arg_key(boxed_arg as PyObjectRef);

    // FORCE_CACHE check
    let hash_idx = (code_key.wrapping_mul(2654435761) ^ arg_key) % FORCE_CACHE_SIZE;
    unsafe {
        let entry = &FORCE_CACHE[hash_idx];
        if entry.0 == code_key && entry.1 == arg_key {
            return entry.2;
        }
    }

    // Create callee frame (arena), dispatch, drop — all internal
    let frame_ptr = create_callee_frame_impl(caller_frame, callable, &[boxed_arg as PyObjectRef]);
    let result = jit_force_callee_frame(frame_ptr);
    jit_drop_callee_frame(frame_ptr);

    // Cache result
    unsafe { FORCE_CACHE[hash_idx] = (code_key, arg_key, result); }
    result
}

pub fn install_jit_call_bridge() {
    static INSTALL: Once = Once::new();
    INSTALL.call_once(|| {
        register_jit_function_caller(jit_call_user_function_from_frame);
        majit_codegen_cranelift::register_call_assembler_force(jit_force_callee_frame);
        majit_codegen_cranelift::register_call_assembler_bridge(jit_bridge_compile_callee);
    });
}

extern "C" fn jit_bridge_compile_callee(
    frame_ptr: i64,
    fail_index: u32,
    trace_id: u64,
    green_key: u64,
) -> i64 {
    use majit_ir::{InputArg, Op, OpCode, OpRef, Type};
    use std::collections::HashMap;

    let frame = unsafe { &mut *(frame_ptr as *mut PyFrame) };
    let result = match pyre_interp::eval::eval_loop_for_force(frame) {
        Ok(r) => {
            // Raw-int protocol: unbox int result to match Finish(raw_int)
            if !r.is_null() && unsafe { is_int(r) } {
                unsafe { w_int_get_value(r) }
            } else {
                r as i64
            }
        }
        Err(e) => panic!("bridge force failed: {e}"),
    };

    let code_key = frame.code as usize;
    let arg_key = if frame.locals_cells_stack_w.len() > 0 {
        force_cache_arg_key(frame.locals_cells_stack_w[0])
    } else {
        0
    };
    let hash_idx = (code_key.wrapping_mul(2654435761) ^ arg_key) % FORCE_CACHE_SIZE;
    unsafe { FORCE_CACHE[hash_idx] = (code_key, arg_key, result); }

    let bridge_inputargs = vec![InputArg::from_type(Type::Int, 0)];
    let frame_opref = OpRef(0);

    let force_fn_ptr = jit_force_callee_frame as *const () as i64;
    let func_const_ref = OpRef(10_000);
    let mut constants: HashMap<u32, i64> = HashMap::new();
    constants.insert(func_const_ref.0, force_fn_ptr);

    let call_descr = majit_meta::make_call_descr(&[Type::Int], Type::Int);
    let call_result = OpRef(1);
    let mut call_op = Op::with_descr(OpCode::CallI, &[func_const_ref, frame_opref], call_descr);
    call_op.pos = call_result;

    let finish_descr = majit_meta::make_fail_descr_typed(vec![Type::Int]);
    let mut finish_op = Op::with_descr(OpCode::Finish, &[call_result], finish_descr);
    finish_op.pos = OpRef(2);

    let bridge_ops = vec![call_op, finish_op];

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
    let caller = unsafe { &*(caller_frame as *const PyFrame) };
    let globals = unsafe { w_func_get_globals(callable) };
    let func_code = code_ptr as *const pyre_bytecode::CodeObject;

    let arena = arena_ref();
    if let Some((ptr, was_init)) = arena.take() {
        if was_init {
            // Fast reinit: only update fields that change between calls.
            // code, execution_context, namespace, locals_cells_stack_w.ptr
            // are stable for self-recursion (same function, same module).
            let f = unsafe { &mut *ptr };
            let same_code = f.code == func_code;
            if same_code {
                // Self-recursion hot path: ~4 writes instead of ~40
                let nargs = args.len().min(f.locals_cells_stack_w.len());
                for i in 0..nargs {
                    f.locals_cells_stack_w[i] = args[i];
                }
                f.valuestackdepth = f.nlocals();
                f.next_instr = 0;
                f.vable_token = 0;
            } else {
                // Different function: full reinit (rare for fib)
                unsafe {
                    std::ptr::write(
                        ptr,
                        PyFrame::new_for_call(func_code, args, globals, caller.execution_context),
                    );
                    (&mut *ptr).fix_array_ptrs();
                }
            }
        } else {
            // First-time init for this arena slot
            unsafe {
                std::ptr::write(
                    ptr,
                    PyFrame::new_for_call(func_code, args, globals, caller.execution_context),
                );
                (&mut *ptr).fix_array_ptrs();
            }
            arena.mark_initialized();
        }
        return ptr as i64;
    }

    // Arena full: heap fallback (should not happen for recursion < 64)
    let frame_ptr = Box::into_raw(Box::new(PyFrame::new_for_call(
        func_code,
        args,
        globals,
        caller.execution_context,
    )));
    unsafe { &mut *frame_ptr }.fix_array_ptrs();
    frame_ptr as i64
}

pub extern "C" fn jit_create_callee_frame_0(caller_frame: i64, callable: i64) -> i64 {
    create_callee_frame_impl(caller_frame, callable, &[])
}

pub extern "C" fn jit_create_callee_frame_1(caller_frame: i64, callable: i64, arg0: i64) -> i64 {
    create_callee_frame_impl(caller_frame, callable, &[arg0 as PyObjectRef])
}

/// Raw-int variant: accepts a raw int and boxes it internally.
/// Eliminates trace_box_int CallI from the trace (boxing folded into frame creation).
pub extern "C" fn jit_create_callee_frame_1_raw_int(
    caller_frame: i64,
    callable: i64,
    raw_int_arg: i64,
) -> i64 {
    let boxed = pyre_object::intobject::w_int_new(raw_int_arg);
    create_callee_frame_impl(caller_frame, callable, &[boxed])
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

/// Force callee and return BOXED result (for inline_function_call).
/// Unlike jit_force_callee_frame which returns raw int for the CA protocol,
/// this returns a boxed PyObjectRef for use in traces that expect boxed values.
pub extern "C" fn jit_force_callee_frame_boxed(frame_ptr: i64) -> i64 {
    let raw = jit_force_callee_frame(frame_ptr);
    // Re-box: raw int → PyObjectRef
    pyre_object::intobject::w_int_new(raw) as i64
}

pub extern "C" fn jit_drop_callee_frame(frame_ptr: i64) {
    if frame_ptr & 1 != 0 {
        return;
    }
    let ptr = frame_ptr as *mut PyFrame;
    let arena = arena_ref();
    if !arena.put(ptr) {
        // Not an arena frame (heap fallback) — free it
        unsafe { drop(Box::from_raw(ptr)) };
    }
}
