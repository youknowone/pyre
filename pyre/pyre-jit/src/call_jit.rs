//! JIT-specific call infrastructure — force/bridge callbacks, callee
//! frame creation helpers, frame pool.
//!
//! Separated from pyre-interpreter/src/call.rs so pyre-interpreter stays JIT-free.

use std::cell::UnsafeCell;
use std::mem::MaybeUninit;
use std::sync::Once;

use pyre_bytecode::bytecode::{Instruction, OpArgState};
use pyre_interpreter::{
    PyResult, function_get_closure, function_get_code, function_get_globals, function_get_name,
    is_function, register_jit_function_caller,
};
use pyre_object::intobject::w_int_get_value;
use pyre_object::intobject::w_int_new;
use pyre_object::pyobject::is_int;
use pyre_object::{PY_NULL, PyObjectRef};

use pyre_interpreter::pyframe::PyFrame;

// Force cache removed: CallAssemblerI + bridge handles recursion
// natively without memoization.

thread_local! {
    static RECURSIVE_DISPATCH_CACHE: UnsafeCell<Option<(u64, FinishProtocol, Option<u64>, bool)>> =
        const { UnsafeCell::new(None) };
    static SELF_RECURSIVE_DISPATCH_CACHE: UnsafeCell<Option<(u64, FinishProtocol, Option<u64>)>> =
        const { UnsafeCell::new(None) };
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FinishProtocol {
    RawInt,
    Boxed,
}

// force_cache_index removed

#[inline]
fn finish_protocol(green_key: u64) -> FinishProtocol {
    let (driver, _) = crate::eval::driver_pair();
    if driver.has_raw_int_finish(green_key) {
        FinishProtocol::RawInt
    } else {
        FinishProtocol::Boxed
    }
}

#[inline]
fn normalize_direct_finish_result(protocol: FinishProtocol, raw: i64) -> i64 {
    match protocol {
        FinishProtocol::RawInt => {
            let maybe_boxed = raw as PyObjectRef;
            let looks_like_heap_ptr = raw > 4096 && (raw as usize & 0x7) == 0;
            if looks_like_heap_ptr && !maybe_boxed.is_null() && unsafe { is_int(maybe_boxed) } {
                unsafe { w_int_get_value(maybe_boxed) }
            } else {
                raw
            }
        }
        FinishProtocol::Boxed => raw,
    }
}

fn debug_instruction_window(frame: &PyFrame) -> String {
    let code = unsafe { &*frame.code };
    let mut out = String::new();
    let mut arg_state = OpArgState::default();
    let current = frame.next_instr;
    let start = current.saturating_sub(6);
    let end = (current + 6).min(code.instructions.len());
    for idx in 0..code.instructions.len() {
        let code_unit = code.instructions[idx];
        let (instruction, op_arg) = arg_state.get(code_unit);
        if idx >= start && idx < end {
            let marker = if idx == current { ">>" } else { "  " };
            let _ = std::fmt::Write::write_fmt(
                &mut out,
                format_args!("{marker} {idx:>3}: {:?} arg={:?}\n", instruction, op_arg),
            );
        }
    }
    out
}

#[inline]
pub(crate) extern "C" fn jit_loop_arg_box_int(raw: i64) -> i64 {
    w_int_new(raw) as i64
}

#[inline]
pub(crate) fn recursive_force_cache_safe(callable: PyObjectRef) -> bool {
    unsafe {
        if !function_get_closure(callable).is_null() {
            return false;
        }
        let code = &*(function_get_code(callable) as *const pyre_bytecode::CodeObject);
        let func_name = function_get_name(callable);
        let mut arg_state = OpArgState::default();
        let mut saw_self_reference = false;

        for code_unit in code.instructions.iter().copied() {
            let (instruction, op_arg) = arg_state.get(code_unit);
            match instruction {
                Instruction::LoadName { namei } => {
                    let idx = namei.get(op_arg) as usize;
                    if code.names[idx].as_str() != func_name {
                        return false;
                    }
                    saw_self_reference = true;
                }
                Instruction::LoadGlobal { namei } => {
                    let raw = namei.get(op_arg) as usize;
                    let name_idx = raw >> 1;
                    if code.names[name_idx].as_str() != func_name {
                        return false;
                    }
                    saw_self_reference = true;
                }
                Instruction::StoreName { .. }
                | Instruction::StoreGlobal { .. }
                | Instruction::LoadDeref { .. }
                | Instruction::StoreDeref { .. }
                | Instruction::DeleteDeref { .. }
                | Instruction::ImportName { .. }
                | Instruction::ImportFrom { .. }
                | Instruction::DeleteName { .. }
                | Instruction::DeleteGlobal { .. }
                | Instruction::MakeCell { .. }
                | Instruction::CopyFreeVars { .. } => return false,
                _ => {}
            }
        }

        if !saw_self_reference {
            return false;
        }
    }

    true
}

/// Whether this concrete frame is a good candidate for immediate
/// function-entry tracing as a self-recursive pure function.
///
/// We reuse the same closure-free / globals-only recursive check as the
/// raw recursive helper path, but discover the function object from the
/// frame's shared globals namespace. This lets eval-time function-entry
/// tracing converge earlier, closer to PyPy's recursive functrace path.
pub(crate) fn self_recursive_function_entry_candidate(frame: &PyFrame) -> bool {
    if frame.next_instr != 0 {
        return false;
    }
    let namespace_ptr = frame.namespace;
    let Some(namespace) = (!namespace_ptr.is_null()).then_some(unsafe { &*namespace_ptr }) else {
        return false;
    };
    let code_ptr = frame.code as *const ();

    for idx in 0..namespace.len() {
        let Some(value) = namespace.get_slot(idx) else {
            continue;
        };
        if value.is_null() {
            continue;
        }
        let is_candidate = unsafe {
            is_function(value)
                && function_get_code(value) == code_ptr
                && function_get_globals(value) == namespace_ptr
                && function_get_closure(value).is_null()
        };
        if is_candidate && recursive_force_cache_safe(value) {
            return true;
        }
    }

    false
}

/// Whether calling this function should prefer its own function-entry portal
/// over caller-side trace-through.
///
/// This mirrors `self_recursive_function_entry_candidate(frame)` but starts
/// from the callee object directly so caller-side inline decisions can leave
/// recursive pure functions on the dedicated function-entry path.
pub(crate) fn callable_prefers_function_entry(callable: PyObjectRef) -> bool {
    unsafe {
        if !is_function(callable) || !function_get_closure(callable).is_null() {
            return false;
        }
        let globals = function_get_globals(callable);
        let Some(namespace) = (!globals.is_null()).then_some(&*globals) else {
            return false;
        };
        let code_ptr = function_get_code(callable);

        for idx in 0..namespace.len() {
            let Some(value) = namespace.get_slot(idx) else {
                continue;
            };
            if value.is_null() {
                continue;
            }
            let is_candidate = is_function(value)
                && function_get_code(value) == code_ptr
                && function_get_globals(value) == globals
                && function_get_closure(value).is_null();
            if is_candidate && recursive_force_cache_safe(value) {
                return true;
            }
        }
    }

    false
}

/// Inline concrete-call override used only while `inline_trace_and_execute`
/// is running. This avoids recursing back through `call_user_function()` for
/// closure-free recursive calls and keeps concrete execution closer to
/// PyPy's frame-switch behavior.
pub fn maybe_handle_inline_concrete_call(
    frame: &PyFrame,
    callable: PyObjectRef,
    args: &[PyObjectRef],
) -> Option<PyResult> {
    if args.len() != 1 || !unsafe { is_function(callable) } {
        return None;
    }
    let arg0 = args[0];
    if arg0.is_null() || !unsafe { is_int(arg0) } {
        return None;
    }
    if !recursive_force_cache_safe(callable) {
        return None;
    }
    let callable_globals = unsafe { function_get_globals(callable) };
    if callable_globals != frame.namespace || !unsafe { function_get_closure(callable) }.is_null() {
        return None;
    }

    let raw_arg = unsafe { w_int_get_value(arg0) };
    let code_ptr = unsafe { function_get_code(callable) };
    let green_key = crate::eval::make_green_key(code_ptr as *const _, 0);
    // Inline concrete execution sometimes falls back to a helper-boundary
    // recursive call instead of a real frame switch. That helper must run as
    // plain concrete execution: if it reuses the outer inline override or the
    // outer trace's merge-point path, it will keep tracing against the wrong
    // symbolic frame and corrupt concrete stack reads.
    let _suspend_inline_override = pyre_interpreter::call::suspend_inline_call_override();
    let _suspend_inline_result = pyre_interpreter::call::suspend_inline_handled_result();
    // Depth tracked by pyre_interpreter::call::CALL_DEPTH (call_user_function path).
    // RPython blackhole.py:1095: bhimpl_recursive_call → portal_runner
    // CAN enter JIT. JIT_TRACING_DEPTH prevents re-entrant tracing,
    // so nested calls safely enter compiled code without force_plain_eval.
    let forced = if code_ptr == frame.code as *const () {
        jit_force_self_recursive_call_raw_1(frame as *const PyFrame as i64, raw_arg)
    } else {
        jit_force_recursive_call_raw_1(frame as *const PyFrame as i64, callable as i64, raw_arg)
    };
    let result = match finish_protocol(green_key) {
        FinishProtocol::RawInt => w_int_new(forced),
        FinishProtocol::Boxed => forced as PyObjectRef,
    };
    Some(Ok(result))
}

fn recursive_dispatch(
    callable: PyObjectRef,
    green_key: u64,
) -> (FinishProtocol, Option<u64>, bool) {
    RECURSIVE_DISPATCH_CACHE.with(|cell| unsafe {
        let slot = &mut *cell.get();
        if let Some((cached_key, protocol, token_num, memo_safe)) = *slot {
            if cached_key == green_key && token_num.is_some() {
                return (protocol, token_num, memo_safe);
            }
        }

        let (driver, _) = crate::eval::driver_pair();
        let protocol = finish_protocol(green_key);
        let token_num = driver.get_loop_token(green_key).map(|token| token.number);
        let memo_safe = recursive_force_cache_safe(callable);
        if token_num.is_some() {
            *slot = Some((green_key, protocol, token_num, memo_safe));
        }
        (protocol, token_num, memo_safe)
    })
}

fn self_recursive_dispatch(green_key: u64) -> (FinishProtocol, Option<u64>) {
    SELF_RECURSIVE_DISPATCH_CACHE.with(|cell| unsafe {
        let slot = &mut *cell.get();
        if let Some((cached_key, protocol, token_num)) = *slot {
            if cached_key == green_key && token_num.is_some() {
                return (protocol, token_num);
            }
        }

        let (driver, _) = crate::eval::driver_pair();
        let protocol = finish_protocol(green_key);
        let token_num = driver.get_loop_token(green_key).map(|token| token.number);
        if token_num.is_some() {
            *slot = Some((green_key, protocol, token_num));
        }
        (protocol, token_num)
    })
}

// Force cache implementation removed — CallAssemblerI + bridge
// handles recursive dispatch natively.

// ── Callee frame arena (RPython nursery bump equivalent) ─────────
// ── Global arena pointers for Cranelift inline access ──────────────
//
// Single-threaded JIT invariant: only one thread executes compiled code
// at a time, so these globals need no synchronization.
static mut ARENA_BUF_BASE: *mut u8 = std::ptr::null_mut();
static mut ARENA_TOP: usize = 0;
static mut ARENA_INITIALIZED: usize = 0;

/// Returns addresses of the global arena state variables and frame
/// layout constants needed by Cranelift to inline arena take/put.
pub fn arena_global_info() -> majit_codegen_cranelift::InlineFrameArenaInfo {
    use majit_metainterp::jitframe::*;
    majit_codegen_cranelift::InlineFrameArenaInfo {
        buf_base_addr: unsafe { std::ptr::addr_of!(ARENA_BUF_BASE) as usize },
        top_addr: unsafe { std::ptr::addr_of!(ARENA_TOP) as usize },
        initialized_addr: unsafe { std::ptr::addr_of!(ARENA_INITIALIZED) as usize },
        frame_size: std::mem::size_of::<pyre_interpreter::pyframe::PyFrame>(),
        frame_code_offset: pyre_interpreter::pyframe::PYFRAME_CODE_OFFSET,
        frame_next_instr_offset: pyre_interpreter::pyframe::PYFRAME_NEXT_INSTR_OFFSET,
        frame_vable_token_offset: pyre_interpreter::pyframe::PYFRAME_VABLE_TOKEN_OFFSET,
        create_fn_addr: jit_create_self_recursive_callee_frame_1_raw_int as usize,
        drop_fn_addr: jit_drop_callee_frame as usize,
        arena_cap: ARENA_CAP,
        jitframe_descrs: Some(majit_gc::rewrite::JitFrameDescrs {
            create_fn_addr: jit_create_self_recursive_callee_frame_1_raw_int as usize,
            drop_fn_addr: jit_drop_callee_frame as usize,
            jitframe_tid: crate::jit::descr::JITFRAME_GC_TYPE_ID,
            jitframe_fixed_size: JITFRAME_FIXED_SIZE,
            jf_frame_info_ofs: JF_FRAME_INFO_OFS,
            jf_descr_ofs: JF_DESCR_OFS,
            jf_force_descr_ofs: JF_FORCE_DESCR_OFS,
            jf_savedata_ofs: JF_SAVEDATA_OFS,
            jf_guard_exc_ofs: JF_GUARD_EXC_OFS,
            jf_forward_ofs: JF_FORWARD_OFS,
            jf_frame_ofs: JF_FRAME_OFS,
            jf_frame_baseitemofs: BASEITEMOFS,
            jf_frame_lengthofs: LENGTHOFS,
            sign_size: SIGN_SIZE,
            pyframe_alloc_size: std::mem::size_of::<pyre_interpreter::pyframe::PyFrame>(),
            pyframe_code_ofs: pyre_interpreter::pyframe::PYFRAME_CODE_OFFSET,
            pyframe_namespace_ofs: std::mem::offset_of!(
                pyre_interpreter::pyframe::PyFrame,
                namespace
            ),
            pyframe_next_instr_ofs: pyre_interpreter::pyframe::PYFRAME_NEXT_INSTR_OFFSET,
            pyframe_vable_token_ofs: pyre_interpreter::pyframe::PYFRAME_VABLE_TOKEN_OFFSET,
        }),
    }
}

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
        let mut arena = Self {
            buf: Box::new([const { MaybeUninit::uninit() }; ARENA_CAP]),
            top: 0,
            initialized: 0,
        };
        // Publish stable pointers so Cranelift-generated code can
        // inline arena take/put without going through TLS.
        unsafe {
            ARENA_BUF_BASE = arena.buf.as_mut_ptr() as *mut u8;
            ARENA_TOP = 0;
            ARENA_INITIALIZED = 0;
        }
        arena
    }

    /// Take the next frame slot. Returns (ptr, was_previously_initialized).
    #[inline]
    fn take(&mut self) -> Option<(*mut PyFrame, bool)> {
        if self.top < ARENA_CAP {
            let idx = self.top;
            self.top += 1;
            unsafe {
                ARENA_TOP = self.top;
            }
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
            unsafe {
                ARENA_TOP = self.top;
            }
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
            unsafe {
                ARENA_INITIALIZED = self.top;
            }
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
    let _suspend_inline_override = pyre_interpreter::call::suspend_inline_call_override();
    // Depth tracked by pyre_interpreter::call::CALL_DEPTH (call_user_function path).
    match pyre_interpreter::call::call_user_function(frame, callable as PyObjectRef, args) {
        Ok(result) => result as i64,
        Err(err) => panic!("jit user-function call failed: {err}"),
    }
}

pub extern "C" fn jit_force_callee_frame(frame_ptr: i64) -> i64 {
    let _suspend_inline_result = pyre_interpreter::call::suspend_inline_handled_result();
    let _ = majit_codegen_cranelift::take_pending_frame_restore();
    let pending = majit_codegen_cranelift::take_pending_force_local0();

    // Lazy frame (RPython parity): when CallR(create_frame) is elided,
    // frame_ptr is the CALLER frame. pending_force_local0 contains the
    // raw int arg. Create callee frame lazily and execute it.
    if let Some(raw_local0) = pending {
        return jit_force_self_recursive_call_raw_1(frame_ptr, raw_local0);
    }

    // Nursery-safe force: read code/namespace/exec_ctx via raw offsets
    // (valid for both arena PyFrame AND nursery-allocated raw blocks).
    // Then create a proper PyFrame for the interpreter.
    //
    // warmspot.py:1021 assembler_call_helper parity: the callee frame
    // (deadframe) may be a nursery-allocated JitFrame-like block. We
    // reconstruct a proper interpreter frame from its raw fields.
    let (code, namespace, exec_ctx) = unsafe {
        use pyre_interpreter::pyframe::*;
        let p = frame_ptr as *const u8;
        let code = *(p.add(PYFRAME_CODE_OFFSET) as *const *const pyre_bytecode::CodeObject);
        let ns = *(p.add(std::mem::offset_of!(PyFrame, namespace))
            as *const *mut pyre_interpreter::PyNamespace);
        let ec = *(p.add(std::mem::offset_of!(PyFrame, execution_context))
            as *const *const pyre_interpreter::PyExecutionContext);
        (code, ns, ec)
    };

    let green_key = crate::eval::make_green_key(code as *const _, 0);
    let protocol = finish_protocol(green_key);

    let mut func_frame = PyFrame::new_for_call(code, &[], namespace, exec_ctx);
    func_frame.fix_array_ptrs();

    pyre_interpreter::call::register_eval_override(pyre_interpreter::eval::eval_frame_plain);
    let result = match pyre_interpreter::eval::eval_frame_plain(&mut func_frame) {
        Ok(r) => r,
        Err(_) => pyre_object::PY_NULL,
    };
    pyre_interpreter::call::register_eval_override(crate::eval::eval_with_jit);

    match protocol {
        FinishProtocol::RawInt if !result.is_null() && unsafe { is_int(result) } => unsafe {
            w_int_get_value(result)
        },
        FinishProtocol::RawInt => result as i64,
        FinishProtocol::Boxed => result as i64,
    }
}

/// warmspot.py:1021-1028 — assembler_call_helper.
///
/// Called when CALL_ASSEMBLER guard-fails (not a finish exit).
/// Receives a JitFrame pointer, reconstructs a PyFrame from the
/// jf_frame slots, and resumes execution in the interpreter.
///
/// This is the JitFrame-aware counterpart to `jit_force_callee_frame`
/// (which operates on PyFrame directly). When the GC rewriter wires
/// nursery JitFrame allocation, this function replaces the force path.
#[allow(dead_code)]
pub extern "C" fn assembler_call_helper(jitframe_ptr: i64, _virtualizable_ref: i64) -> i64 {
    use majit_metainterp::jitframe::JitFrame;

    let jf = jitframe_ptr as *mut JitFrame;

    // warmspot.py:1022 — fail_descr = cpu.get_latest_descr(deadframe)
    // compile.py:701 handle_fail: dispatches on fail_descr to either
    // _trace_and_compile_from_bridge or resume_in_blackhole.
    // Bridge compilation is driven by must_compile() in jitdriver.
    // This force path always resumes in the interpreter (blackhole).
    let _descr = unsafe { JitFrame::get_latest_descr(jf) };

    // For now, reconstruct a PyFrame and run it in the interpreter.
    // This is the "blackhole" path — RPython resume.py parity.
    //
    // Step 1: read the raw int arg from jf_frame[0]
    let raw_arg = unsafe { JitFrame::get_int_value(jf, 0) };

    // Step 2: get caller frame from the force context
    let pending = majit_codegen_cranelift::take_pending_force_local0();
    let raw_local0 = pending.unwrap_or(raw_arg as i64);

    // Step 3: create a PyFrame and run it
    // The caller_frame is in inputs[0] which was the JitFrame's first
    // virtualizable input. For now, fall back to the existing force path.
    jit_force_self_recursive_call_raw_1(jitframe_ptr, raw_local0)
}

fn jit_force_callee_frame_raw(frame_ptr: i64) -> i64 {
    let _suspend_inline_result = pyre_interpreter::call::suspend_inline_handled_result();
    let frame = unsafe { &mut *(frame_ptr as *mut PyFrame) };

    let green_key = crate::eval::make_green_key(frame.code, frame.next_instr);
    let protocol = finish_protocol(green_key);

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
            let value = match protocol {
                FinishProtocol::RawInt => normalize_direct_finish_result(protocol, raw),
                FinishProtocol::Boxed => w_int_new(raw) as i64,
            };
            // force cache removed
            return value;
        }
    }

    match crate::eval::eval_with_jit(frame) {
        Ok(result) => {
            let value = match protocol {
                FinishProtocol::RawInt if !result.is_null() && unsafe { is_int(result) } => unsafe {
                    w_int_get_value(result)
                },
                FinishProtocol::RawInt => result as i64,
                FinishProtocol::Boxed => result as i64,
            };
            // force cache removed
            value
        }
        Err(err) => panic!("jit force callee frame raw failed: {err}"),
    }
}

/// Interpreter-only force: used by execute_call_assembler_direct
/// to handle guard failures without recursive compiled dispatch.
extern "C" fn jit_force_callee_frame_interp(frame_ptr: i64) -> i64 {
    let _suspend_inline_result = pyre_interpreter::call::suspend_inline_handled_result();
    // RPython: blackhole interp — no blackhole_entry_bump needed.
    let frame = unsafe { &mut *(frame_ptr as *mut PyFrame) };

    let green_key = crate::eval::make_green_key(frame.code, frame.next_instr);
    let protocol = finish_protocol(green_key);

    let result = blackhole_from_jit_frame(frame);

    match protocol {
        FinishProtocol::RawInt if !result.is_null() && unsafe { is_int(result) } => unsafe {
            w_int_get_value(result)
        },
        FinishProtocol::RawInt => result as i64,
        FinishProtocol::Boxed => result as i64,
    }
}

/// RPython: blackhole.py resume_in_blackhole()
///
/// Public wrapper for guard failure recovery.
pub fn resume_in_blackhole_pub(frame: &mut PyFrame) -> pyre_object::PyObjectRef {
    blackhole_from_jit_frame(frame)
}

/// RPython: blackhole.py resume_in_blackhole()
///
/// Compiles the frame's CodeObject to JitCode (via CodeWriter), creates
/// a BlackholeInterpreter, loads frame state, and runs it.
/// The blackhole has NO JIT entry points — structural isolation.
fn blackhole_from_jit_frame(frame: &mut PyFrame) -> PyObjectRef {
    // RPython parity: blackhole has no jit_merge_point, but
    // bhimpl_recursive_call calls portal_runner which CAN enter JIT.
    // eval_frame_plain handles THIS frame without JIT hooks;
    // nested calls go through eval_with_jit normally.
    let code = unsafe { &*frame.code };
    let py_pc = frame.next_instr;

    // RPython: blackhole_from_resumedata() → setposition + consume_one_section
    // For pyre, we compile Python bytecodes to JitCode and load frame state.
    let writer = crate::jit::codewriter::CodeWriter::new(
        bh_call_fn,
        bh_load_global_fn,
        bh_compare_fn,
        bh_binary_op_fn,
        bh_box_int_fn,
        bh_truth_fn,
        bh_load_const_fn,
        bh_store_subscr_fn,
        crate::call_jit::bh_build_list_fn,
    );
    let pyjitcode = crate::jit::codewriter::get_jitcode(code, &writer);

    // Map Python PC → JitCode PC
    let jitcode_pc = if py_pc < pyjitcode.pc_map.len() {
        pyjitcode.pc_map[py_pc]
    } else {
        0
    };

    // RPython: blackholeinterp = builder.acquire_interp()
    // Use thread-local builder pool to avoid per-call allocation.
    thread_local! {
        static BH_BUILDER: std::cell::UnsafeCell<majit_metainterp::blackhole::BlackholeInterpBuilder> =
            std::cell::UnsafeCell::new(majit_metainterp::blackhole::BlackholeInterpBuilder::new());
    }
    let builder = BH_BUILDER.with(|cell| unsafe { &mut *cell.get() });
    let mut bh = builder.acquire_interp();
    // RPython: blackholeinterp.setposition(jitcode, pc)
    bh.setposition(pyjitcode.jitcode.clone(), jitcode_pc);

    // RPython resume.py:1381-1430 consume_one_section: load register values.
    // pyre fast locals → int registers (blackhole dispatch reads as i64).
    let nlocals = code.varnames.len();
    for i in 0..nlocals {
        if i < frame.locals_cells_stack_w.len() {
            bh.setarg_i(i, frame.locals_cells_stack_w[i] as i64);
        }
    }

    // Load value stack into blackhole runtime stack
    let stack_base = nlocals;
    let vsd = frame.valuestackdepth;
    for i in stack_base..vsd {
        if i < frame.locals_cells_stack_w.len() {
            bh.runtime_stack_push(0, frame.locals_cells_stack_w[i] as i64);
        }
    }

    // frame_reg = 3 (fixed in codewriter, independent of nlocals)
    const FRAME_REG: usize = 3;
    if FRAME_REG < bh.registers_i.len() {
        bh.setarg_i(FRAME_REG, frame as *mut PyFrame as i64);
    }

    if majit_metainterp::majit_log_enabled() {
        eprintln!(
            "[jit][blackhole] setup pc={} nlocals={} regs_i_len={} local0={} frame_reg={}",
            py_pc,
            nlocals,
            bh.registers_i.len(),
            bh.registers_i.get(0).copied().unwrap_or(-999),
            bh.registers_i.get(nlocals + 3).copied().unwrap_or(-999),
        );
    }

    // RPython: _run_forever(blackholeinterp, current_exc)
    bh.run();

    // Return value is in registers_i[0] (set by RETURN_VALUE → move_i(0, tmp0))
    let result = bh.registers_i[0] as PyObjectRef;

    // RPython: builder.release_interp(blackholeinterp) — return to pool
    builder.release_interp(bh);

    result
}

/// RPython blackhole.py _run_forever + jitexc.py parity.
pub enum BlackholeResult {
    /// RPython jitexc.py:53 ContinueRunningNormally: blackhole reached
    /// the merge point → restart portal to re-enter compiled code.
    ContinueRunningNormally,
    /// RPython jitexc.py:68 DoneWithThisFrame: blackhole ran the
    /// function to completion (RETURN_VALUE).
    DoneWithThisFrame(PyResult),
    /// Blackhole couldn't run (bad resume data, BC_ABORT, etc).
    Failed,
}

/// RPython resume.py:1312 blackhole_from_resumedata +
/// blackhole.py:1752 _run_forever parity.
///
/// Multi-frame fail_args: [callee_section..., caller_section...]
/// Each section: [Ref(frame), Int(ni), Int(vsd), locals..., stack...]
///
/// Builds a blackhole chain (innermost first), then runs _run_forever:
/// callee blackhole → RETURN_VALUE → caller blackhole → merge point.
/// blackhole.py:1782 resume_in_blackhole parity.
pub fn resume_in_blackhole(
    _caller_frame: &mut PyFrame,
    typed_values: &[majit_ir::Value],
    merge_py_pc: usize,
) -> BlackholeResult {
    use majit_ir::Value;

    let writer = crate::jit::codewriter::CodeWriter::new(
        bh_call_fn,
        bh_load_global_fn,
        bh_compare_fn,
        bh_binary_op_fn,
        bh_box_int_fn,
        bh_truth_fn,
        bh_load_const_fn,
        bh_store_subscr_fn,
        crate::call_jit::bh_build_list_fn,
    );

    // RPython resume.py:1333-1343 blackhole_from_resumedata:
    // Parse sections and build blackhole chain.
    let sections = parse_fail_arg_sections(typed_values);
    if sections.is_empty() {
        return BlackholeResult::Failed;
    }

    thread_local! {
        static BH_BUILDER3: std::cell::UnsafeCell<majit_metainterp::blackhole::BlackholeInterpBuilder> =
            std::cell::UnsafeCell::new(majit_metainterp::blackhole::BlackholeInterpBuilder::new());
    }
    let builder = BH_BUILDER3.with(|cell| unsafe { &mut *cell.get() });

    // Build chain bottom-up: first section = callee (innermost),
    // last section = caller (outermost). Chain: callee.next = caller.
    let mut prev_bh: Option<majit_metainterp::blackhole::BlackholeInterpreter> = None;

    // Process sections in REVERSE (caller first, then callee on top).
    // RPython builds the chain so that the LAST acquired interp is the
    // innermost (callee), with nextblackholeinterp pointing to the caller.
    for (sec_idx, section) in sections.iter().enumerate().rev() {
        let frame_ptr = match section.get(0) {
            Some(Value::Ref(r)) => r.as_usize() as *mut PyFrame,
            Some(Value::Int(v)) => *v as *mut PyFrame,
            _ => {
                builder.release_chain(prev_bh);
                return BlackholeResult::Failed;
            }
        };
        if frame_ptr.is_null() {
            builder.release_chain(prev_bh);
            return BlackholeResult::Failed;
        }
        if majit_metainterp::majit_log_enabled() {
            eprintln!(
                "[jit][blackhole-section] idx={} frame_ptr={:#x} py_pc={:?} vsd={:?}",
                sec_idx,
                frame_ptr as usize,
                section.get(1),
                section.get(2),
            );
        }
        let frame = unsafe { &*frame_ptr };
        let code = unsafe { &*frame.code };
        let nlocals = code.varnames.len();

        let mut py_pc = match section.get(1) {
            Some(Value::Int(v)) => *v as usize,
            _ => {
                builder.release_chain(prev_bh);
                return BlackholeResult::Failed;
            }
        };
        // pyre-specific: the virtualizable's next_instr may land on a
        // Cache code unit (CPython 3.13 inserts Cache after opcodes).
        // RPython has no Cache concept. Scan backward to the actual opcode
        // so the blackhole starts at the correct JitCode position.
        while py_pc > 0 {
            match pyre_interpreter::decode_instruction_at(code, py_pc) {
                Some((pyre_bytecode::bytecode::Instruction::Cache, _))
                | Some((pyre_bytecode::bytecode::Instruction::ExtendedArg, _))
                | Some((pyre_bytecode::bytecode::Instruction::NotTaken, _)) => {
                    py_pc -= 1;
                }
                _ => break,
            }
        }
        // RPython: resume data always encodes valid PCs within the
        // frame's own code object. If py_pc is out of range, the
        // resume data is invalid — fail cleanly.
        if py_pc >= code.instructions.len() {
            builder.release_chain(prev_bh);
            return BlackholeResult::Failed;
        }
        let vsd = match section.get(2) {
            Some(Value::Int(v)) => *v as usize,
            _ => {
                builder.release_chain(prev_bh);
                return BlackholeResult::Failed;
            }
        };
        let stack_only = vsd.saturating_sub(nlocals);

        let pyjitcode = crate::jit::codewriter::get_jitcode(code, &writer);
        // Skip blackhole if jitcode has actual BC_ABORT opcodes (not just
        // data bytes that happen to equal 13). Walk bytecodes properly
        // to distinguish opcodes from operands.
        if pyjitcode.has_abort_opcode() {
            builder.release_chain(prev_bh);
            return BlackholeResult::Failed;
        }
        let jitcode_pc = if py_pc < pyjitcode.pc_map.len() {
            pyjitcode.pc_map[py_pc]
        } else {
            builder.release_chain(prev_bh);
            return BlackholeResult::Failed;
        };

        // RPython: curbh.setposition(jitcode, pc)
        let mut bh = builder.acquire_interp();
        bh.setposition(pyjitcode.jitcode.clone(), jitcode_pc);

        // Set merge_point on the OUTERMOST (last section = caller) blackhole.
        if sec_idx == sections.len() - 1 {
            if let Some(&merge_jitcode_pc) = pyjitcode.pc_map.get(merge_py_pc) {
                bh.merge_point_jitcode_pc = Some(merge_jitcode_pc);
            }
        }

        // RPython resume.py consume_one_section parity: load values
        // into TYPED register files based on the Value variant.
        // RPython: decode_int→setarg_i, decode_ref→setarg_r, decode_float→setarg_f.
        // pyre codewriter puts all Python locals in ref registers, so
        // RPython resume.py consume_one_section parity: Python locals
        // are ref-typed at the jitcode level. Unboxed Int/Float values
        // from the optimizer are materialized back to PyObjectRef via
        // materialize_virtual (RPython getvirtual_ptr equivalent).
        for i in 0..nlocals {
            let slot = 3 + i;
            if let Some(val) = section.get(slot) {
                bh.setarg_r(i, materialize_virtual(val));
            } else {
                // RPython resume.py:1381 parity: rd_numb always encodes
                // all locals. Missing slot = incomplete resume data.
                // Abort blackhole rather than inject stale frame values.
                builder.release_interp(bh);
                return BlackholeResult::Failed;
            }
        }
        for i in 0..stack_only {
            let slot = 3 + nlocals + i;
            if let Some(val) = section.get(slot) {
                bh.runtime_stack_push(0, materialize_virtual(val));
            }
        }
        if 3 < bh.registers_i.len() {
            bh.setarg_i(3, frame_ptr as i64);
        }

        // RPython: nextbh.nextblackholeinterp = curbh
        bh.nextblackholeinterp = prev_bh.map(Box::new);
        prev_bh = Some(bh);
    }

    let Some(mut bh) = prev_bh else {
        return BlackholeResult::Failed;
    };

    if majit_metainterp::majit_log_enabled() {
        eprintln!(
            "[jit][blackhole-resume] chain_len={} merge_pc={}",
            sections.len(),
            merge_py_pc,
        );
    }

    // RPython blackhole.py:1752 _run_forever parity:
    // Run the innermost blackhole. On RETURN_VALUE (LeaveFrame),
    // pop to caller blackhole and continue.
    loop {
        bh.run();

        if bh.reached_merge_point {
            // RPython bhimpl_jit_merge_point:
            // if nextblackholeinterp is None → ContinueRunningNormally
            // Write back to the frame that owns this merge point.
            let frame_ptr = bh.registers_i.get(3).copied().unwrap_or(0) as *mut PyFrame;
            if !frame_ptr.is_null() {
                let frame = unsafe { &mut *frame_ptr };
                let code = unsafe { &*frame.code };
                let nlocals = code.varnames.len();
                for i in 0..nlocals {
                    if i < bh.registers_r.len() && i < frame.locals_cells_stack_w.len() {
                        frame.locals_cells_stack_w[i] =
                            bh.registers_r[i] as pyre_object::PyObjectRef;
                    }
                }
                let stack = bh.runtime_stack_drain(0);
                frame.valuestackdepth = nlocals + stack.len();
                for (i, val) in stack.iter().enumerate() {
                    let idx = nlocals + i;
                    if idx < frame.locals_cells_stack_w.len() {
                        frame.locals_cells_stack_w[idx] = *val as pyre_object::PyObjectRef;
                    }
                }
                frame.next_instr = merge_py_pc;
            }
            builder.release_interp(bh);
            return BlackholeResult::ContinueRunningNormally;
        }

        // BC_ABORT: unsupported bytecode hit during execution.
        // Return Failed so caller falls back to restore_guard_failure_values.
        if bh.aborted {
            builder.release_interp(bh);
            return BlackholeResult::Failed;
        }

        // blackhole.py:1752 _run_forever exception propagation:
        // Exception not handled in this frame (no handler found by
        // dispatch_one's handle_exception_in_frame). Propagate to caller.
        if bh.got_exception {
            let exc_value = bh.exception_last_value;
            let next = bh.nextblackholeinterp.take();
            builder.release_interp(bh);

            let Some(mut caller_bh) = next.map(|b| *b) else {
                return BlackholeResult::Failed;
            };

            // blackhole.py:396 handle_exception_in_frame in caller.
            // Ensure last_opcode_position reflects the caller's call-site PC.
            // The caller may not have run any opcodes in this blackhole session
            // (it was suspended), so last_opcode_position must match position.
            caller_bh.last_opcode_position = caller_bh.position;
            if caller_bh.handle_exception_in_frame(exc_value) {
                bh = caller_bh;
                continue;
            }

            // No handler in caller: propagate further up.
            caller_bh.exception_last_value = exc_value;
            caller_bh.got_exception = true;
            bh = caller_bh;
            continue;
        }

        // DoneWithThisFrame: callee finished (RETURN_VALUE).
        let return_value = bh.registers_r.get(0).copied().unwrap_or(0);
        let next = bh.nextblackholeinterp.take();
        builder.release_interp(bh);

        let Some(mut caller_bh) = next.map(|b| *b) else {
            // Outermost frame finished — function return.
            return BlackholeResult::DoneWithThisFrame(
                Ok(return_value as pyre_object::PyObjectRef),
            );
        };

        // RPython _run_forever: pass return value to caller blackhole.
        // The caller's CALL instruction result goes onto the runtime stack.
        caller_bh.runtime_stack_push(0, return_value);

        bh = caller_bh;
    }
}

/// Parse multi-frame fail_args into per-frame sections.
/// Single frame: [[Ref, Int, Int, locals..., stack...]]
/// Multi frame: [[callee: Ref, Int, Int, ...], [caller: Ref, Int, Int, ...]]
/// Re-box optimized raw values for the blackhole's ref register file.
///
/// RPython resume.py decode_ref(TAGVIRTUAL) → getvirtual_ptr() parity.
///
/// Python locals are always ref-typed at the jitcode level (both in
/// RPython/PyPy and pyre). The optimizer unboxes values in the IR
/// (guard_class + getfield_gc_pure_i). On guard failure, the resume
/// reader materializes (re-boxes) unboxed values back to refs.
/// RPython: getvirtual_ptr() allocates W_IntObject from virtual fields.
/// pyre: w_int_new / w_float_new from the raw Value payload.
fn materialize_virtual(val: &majit_ir::Value) -> i64 {
    use majit_ir::Value;
    match val {
        Value::Ref(r) => r.as_usize() as i64,
        Value::Int(v) => pyre_object::intobject::w_int_new(*v) as i64,
        Value::Float(v) => pyre_object::floatobject::w_float_new(*v) as i64,
        Value::Void => 0i64,
    }
}

fn parse_fail_arg_sections(typed_values: &[majit_ir::Value]) -> Vec<&[majit_ir::Value]> {
    use majit_ir::Value;
    let mut sections = Vec::new();
    let mut cursor = 0usize;
    while cursor + 3 <= typed_values.len() {
        let section = &typed_values[cursor..];
        let has_header = matches!(section[0], Value::Ref(_) | Value::Int(_))
            && matches!(section[1], Value::Int(_))
            && matches!(section[2], Value::Int(_));
        if !has_header {
            break;
        }
        let vsd = match section[2] {
            Value::Int(v) if v >= 0 => v as usize,
            _ => break,
        };
        let section_len = 3 + vsd;
        if cursor + section_len > typed_values.len() {
            break;
        }
        sections.push(&typed_values[cursor..cursor + section_len]);
        cursor += section_len;
    }
    sections
}

/// bhimpl_jit_merge_point parity: run the blackhole from `guard_py_pc`
/// until it reaches `merge_py_pc` (the loop header). On success, writes
/// back the blackhole register state into the frame and returns true.
///
/// This implements the RPython flow:
///   guard fail → resume_in_blackhole → jit_merge_point → ContinueRunningNormally
pub fn resume_in_blackhole_to_merge_point(frame: &mut PyFrame, merge_py_pc: usize) -> bool {
    let code = unsafe { &*frame.code };
    let py_pc = frame.next_instr;

    let writer = crate::jit::codewriter::CodeWriter::new(
        bh_call_fn,
        bh_load_global_fn,
        bh_compare_fn,
        bh_binary_op_fn,
        bh_box_int_fn,
        bh_truth_fn,
        bh_load_const_fn,
        bh_store_subscr_fn,
        crate::call_jit::bh_build_list_fn,
    );
    let pyjitcode = crate::jit::codewriter::get_jitcode(code, &writer);

    let jitcode_pc = if py_pc < pyjitcode.pc_map.len() {
        pyjitcode.pc_map[py_pc]
    } else {
        return false;
    };
    let merge_jitcode_pc = if merge_py_pc < pyjitcode.pc_map.len() {
        pyjitcode.pc_map[merge_py_pc]
    } else {
        return false;
    };

    thread_local! {
        static BH_BUILDER2: std::cell::UnsafeCell<majit_metainterp::blackhole::BlackholeInterpBuilder> =
            std::cell::UnsafeCell::new(majit_metainterp::blackhole::BlackholeInterpBuilder::new());
    }
    let builder = BH_BUILDER2.with(|cell| unsafe { &mut *cell.get() });
    let mut bh = builder.acquire_interp();
    bh.setposition(pyjitcode.jitcode.clone(), jitcode_pc);
    bh.merge_point_jitcode_pc = Some(merge_jitcode_pc);

    let nlocals = code.varnames.len();
    for i in 0..nlocals {
        if i < frame.locals_cells_stack_w.len() {
            bh.setarg_i(i, frame.locals_cells_stack_w[i] as i64);
        }
    }
    let stack_base = nlocals;
    let vsd = frame.valuestackdepth;
    for i in stack_base..vsd {
        if i < frame.locals_cells_stack_w.len() {
            bh.runtime_stack_push(0, frame.locals_cells_stack_w[i] as i64);
        }
    }
    if nlocals + 3 < bh.registers_i.len() {
        bh.setarg_i(nlocals + 3, frame as *mut PyFrame as i64);
    }

    if majit_metainterp::majit_log_enabled() {
        eprintln!(
            "[jit][blackhole-merge] guard_pc={} merge_pc={} jit_pc={} merge_jit_pc={}",
            py_pc, merge_py_pc, jitcode_pc, merge_jitcode_pc,
        );
    }

    bh.run();

    if !bh.reached_merge_point {
        // Blackhole finished without reaching merge point (RETURN_VALUE
        // or exception). Frame state is stale — caller should not use it.
        builder.release_interp(bh);
        return false;
    }

    // Write back blackhole register state → frame locals.
    for i in 0..nlocals {
        if i < bh.registers_i.len() && i < frame.locals_cells_stack_w.len() {
            frame.locals_cells_stack_w[i] = bh.registers_i[i] as pyre_object::PyObjectRef;
        }
    }
    // Write back runtime stack → frame value stack.
    let stack = bh.runtime_stack_drain(0);
    frame.valuestackdepth = nlocals + stack.len();
    for (i, val) in stack.iter().enumerate() {
        let idx = nlocals + i;
        if idx < frame.locals_cells_stack_w.len() {
            frame.locals_cells_stack_w[idx] = *val as pyre_object::PyObjectRef;
        }
    }
    frame.next_instr = merge_py_pc;

    if majit_metainterp::majit_log_enabled() {
        eprintln!(
            "[jit][blackhole-merge] reached merge point, frame.next_instr={}",
            frame.next_instr,
        );
    }

    builder.release_interp(bh);
    true
}

/// Interpreter-only force without memoization.
///
/// Used by the fused raw-int recursive helper, which already maintains
/// its own outer force cache keyed by (callee code, raw arg).
extern "C" fn jit_force_callee_frame_interp_nocache(frame_ptr: i64) -> i64 {
    let _suspend_inline_result = pyre_interpreter::call::suspend_inline_handled_result();
    let frame = unsafe { &mut *(frame_ptr as *mut PyFrame) };
    let green_key = crate::eval::make_green_key(frame.code, frame.next_instr);
    let protocol = finish_protocol(green_key);

    match crate::eval::eval_with_jit(frame) {
        Ok(result) => match protocol {
            FinishProtocol::RawInt if !result.is_null() && unsafe { is_int(result) } => unsafe {
                w_int_get_value(result)
            },
            FinishProtocol::RawInt => result as i64,
            FinishProtocol::Boxed => result as i64,
        },
        Err(_) => 0i64,
    }
}

/// Fused recursive call with boxed arg.
pub extern "C" fn jit_force_recursive_call_1(
    caller_frame: i64,
    callable: i64,
    boxed_arg: i64,
) -> i64 {
    let callable_ref = callable as PyObjectRef;
    let boxed_arg_ref = boxed_arg as PyObjectRef;
    let code_ptr = unsafe { function_get_code(callable_ref) };
    let green_key = crate::eval::make_green_key(code_ptr as *const _, 0);
    if matches!(finish_protocol(green_key), FinishProtocol::RawInt)
        && !boxed_arg_ref.is_null()
        && unsafe { is_int(boxed_arg_ref) }
    {
        let raw_arg = unsafe { w_int_get_value(boxed_arg_ref) };
        let forced = jit_force_recursive_call_raw_1(caller_frame, callable, raw_arg);
        return w_int_new(forced) as i64;
    }

    if majit_metainterp::majit_log_enabled() {
        let caller = unsafe { &*(caller_frame as *const PyFrame) };
        let caller_arg0 = if caller.locals_cells_stack_w.len() > 0
            && !caller.locals_cells_stack_w[0].is_null()
            && unsafe { is_int(caller.locals_cells_stack_w[0]) }
        {
            Some(unsafe { w_int_get_value(caller.locals_cells_stack_w[0]) })
        } else {
            None
        };
        let boxed = boxed_arg as PyObjectRef;
        let callee_arg0 = if !boxed.is_null() && unsafe { is_int(boxed) } {
            Some(unsafe { w_int_get_value(boxed) })
        } else {
            None
        };
        eprintln!(
            "[jit][force-recursive-boxed] enter caller_arg0={:?} callee_arg0={:?}",
            caller_arg0, callee_arg0
        );
    }
    let frame_ptr = create_callee_frame_impl(caller_frame, callable, &[boxed_arg_ref]);
    let result = jit_force_callee_frame(frame_ptr);
    jit_drop_callee_frame(frame_ptr);
    if majit_metainterp::majit_log_enabled() {
        let caller = unsafe { &*(caller_frame as *const PyFrame) };
        let caller_arg0 = if caller.locals_cells_stack_w.len() > 0
            && !caller.locals_cells_stack_w[0].is_null()
            && unsafe { is_int(caller.locals_cells_stack_w[0]) }
        {
            Some(unsafe { w_int_get_value(caller.locals_cells_stack_w[0]) })
        } else {
            None
        };
        eprintln!(
            "[jit][force-recursive-boxed] exit caller_arg0={:?}",
            caller_arg0
        );
    }
    result
}

/// Fused recursive call with RAW INT arg, boxed result.
///
/// This keeps the trace-side argument in raw-int form even before the callee
/// has stabilized on a raw-int finish protocol. It is a closer match to
/// RPython's recursive portal argument flow than boxing the argument in the
/// trace before every helper-boundary call.
pub extern "C" fn jit_force_recursive_call_argraw_boxed_1(
    caller_frame: i64,
    callable: i64,
    raw_int_arg: i64,
) -> i64 {
    let _suspend_inline_result = pyre_interpreter::call::suspend_inline_handled_result();
    let callable_ref = callable as PyObjectRef;
    let code_ptr = unsafe { function_get_code(callable_ref) };
    let green_key = crate::eval::make_green_key(code_ptr as *const _, 0);
    if matches!(finish_protocol(green_key), FinishProtocol::RawInt) {
        let forced = jit_force_recursive_call_raw_1(caller_frame, callable, raw_int_arg);
        return w_int_new(forced) as i64;
    }

    let boxed = pyre_object::intobject::w_int_new(raw_int_arg);
    jit_force_recursive_call_1(caller_frame, callable, boxed as i64)
}

/// Self-recursive single-arg boxed helper.
///
/// Keeps the boxed helper path off the generic callable redispatch and
/// blackhole fallback route. This mirrors the specialized raw helper:
/// the callee frame is created directly from the caller's code/globals.
pub extern "C" fn jit_force_self_recursive_call_1(caller_frame: i64, boxed_arg: i64) -> i64 {
    let _suspend_inline_result = pyre_interpreter::call::suspend_inline_handled_result();
    let boxed_arg_ref = boxed_arg as PyObjectRef;
    let caller = unsafe { &*(caller_frame as *const PyFrame) };
    let green_key = crate::eval::make_green_key(caller.code, 0);
    if matches!(finish_protocol(green_key), FinishProtocol::RawInt)
        && !boxed_arg_ref.is_null()
        && unsafe { is_int(boxed_arg_ref) }
    {
        let raw_arg = unsafe { w_int_get_value(boxed_arg_ref) };
        let forced = jit_force_self_recursive_call_raw_1(caller_frame, raw_arg);
        return w_int_new(forced) as i64;
    }

    let frame_ptr = create_self_recursive_callee_frame_impl_1_boxed(caller_frame, boxed_arg_ref);
    // RPython warmspot.py:941 portal_runner parity
    let result = {
        let frame = unsafe { &mut *(frame_ptr as *mut PyFrame) };
        match crate::eval::eval_with_jit(frame) {
            Ok(r) => r as i64,
            Err(_) => 0i64,
        }
    };
    jit_drop_callee_frame(frame_ptr);
    result
}

/// Self-recursive single-arg helper with raw-int arg and boxed result.
///
/// Mirrors `jit_force_self_recursive_call_1`, but keeps the trace-side
/// argument unboxed so recursive helper-boundary calls do not allocate a
/// temporary `W_Int` in the trace.
pub extern "C" fn jit_force_self_recursive_call_argraw_boxed_1(
    caller_frame: i64,
    raw_int_arg: i64,
) -> i64 {
    let _suspend_inline_result = pyre_interpreter::call::suspend_inline_handled_result();
    let caller = unsafe { &*(caller_frame as *const PyFrame) };
    let green_key = crate::eval::make_green_key(caller.code, 0);
    if matches!(finish_protocol(green_key), FinishProtocol::RawInt) {
        let forced = jit_force_self_recursive_call_raw_1(caller_frame, raw_int_arg);
        return w_int_new(forced) as i64;
    }

    let boxed = pyre_object::intobject::w_int_new(raw_int_arg);
    jit_force_self_recursive_call_1(caller_frame, boxed as i64)
}

/// Fully fused recursive call with RAW INT arg — no boxing in trace at all.
///
/// Eliminates ALL per-recursive-call overhead from trace:
///   Before: CallI(box) + CallMayForce(force_1, frame, callable, boxed)
///   After:  CallMayForce(force_raw_1, frame, callable, raw_int)
///
/// Boxing happens inside this function, not in the trace.
pub extern "C" fn jit_force_recursive_call_raw_1(
    caller_frame: i64,
    callable: i64,
    raw_int_arg: i64,
) -> i64 {
    let _suspend_inline_result = pyre_interpreter::call::suspend_inline_handled_result();
    let callable_ref = callable as PyObjectRef;
    let code_ptr = unsafe { function_get_code(callable_ref) };
    let green_key = crate::eval::make_green_key(code_ptr as *const _, 0);
    let (protocol, _token_num, _memo_safe) = recursive_dispatch(callable_ref, green_key);

    let boxed = pyre_object::intobject::w_int_new(raw_int_arg);
    let frame_ptr = create_callee_frame_impl_1_boxed(caller_frame, callable_ref, boxed);
    // RPython parity: nested calls CAN enter JIT (blackhole.py:1095)
    let result = {
        let frame = unsafe { &mut *(frame_ptr as *mut PyFrame) };
        let bh_result = blackhole_from_jit_frame(frame);
        match protocol {
            FinishProtocol::RawInt if !bh_result.is_null() && unsafe { is_int(bh_result) } => unsafe {
                w_int_get_value(bh_result)
            },
            FinishProtocol::RawInt => bh_result as i64,
            FinishProtocol::Boxed => bh_result as i64,
        }
    };
    jit_drop_callee_frame(frame_ptr);
    result
}

/// Specialized raw-int recursive helper for closure-free self-recursion.
///
/// Unlike `jit_force_recursive_call_raw_1`, this does not need to rediscover
/// the callee's code/globals from a function object on every call. The caller
/// frame already carries the exact recursive target:
/// - `caller.code` is the callee code object
/// - `caller.namespace` is the module globals
/// - `caller.execution_context` is the shared execution context
///
/// Trace-time recursive CALL_ASSEMBLER handles the optimized path. The
/// concrete helper should mirror RPython's force_fn behavior: execute the
/// callee's own frame without JIT on that frame, but let nested portal
/// calls re-enter compiled code through the normal portal runner path.
pub extern "C" fn jit_force_self_recursive_call_raw_1(caller_frame: i64, raw_int_arg: i64) -> i64 {
    let _suspend_inline_result = pyre_interpreter::call::suspend_inline_handled_result();
    if majit_metainterp::majit_log_enabled() && raw_int_arg <= 4 {
        eprintln!("[jit][force-self-recursive] enter arg={}", raw_int_arg);
    }
    let caller = unsafe { &*(caller_frame as *const PyFrame) };
    let code_ptr = caller.code;
    let green_key = crate::eval::make_green_key(code_ptr as *const _, 0);
    let (protocol, _token_num) = self_recursive_dispatch(green_key);

    let boxed = pyre_object::intobject::w_int_new(raw_int_arg);
    let frame_ptr = create_self_recursive_callee_frame_impl_1_boxed(caller_frame, boxed);
    // RPython warmspot.py:941 portal_runner parity
    let result = {
        let frame = unsafe { &mut *(frame_ptr as *mut PyFrame) };
        let pr_result = match crate::eval::eval_with_jit(frame) {
            Ok(r) => r,
            Err(_) => pyre_object::PY_NULL,
        };
        match protocol {
            FinishProtocol::RawInt if !pr_result.is_null() && unsafe { is_int(pr_result) } => unsafe {
                w_int_get_value(pr_result)
            },
            FinishProtocol::RawInt => pr_result as i64,
            FinishProtocol::Boxed => pr_result as i64,
        }
    };
    jit_drop_callee_frame(frame_ptr);
    if majit_metainterp::majit_log_enabled() && raw_int_arg <= 4 {
        eprintln!(
            "[jit][force-self-recursive] exit arg={} result={} protocol={:?}",
            raw_int_arg, result, protocol
        );
    }
    result
}

pub fn install_jit_call_bridge() {
    static INSTALL: Once = Once::new();
    INSTALL.call_once(|| {
        register_jit_function_caller(jit_call_user_function_from_frame);
        majit_codegen_cranelift::register_call_assembler_force(jit_force_callee_frame);
        majit_codegen_cranelift::register_call_assembler_bridge(jit_bridge_compile_callee);
        majit_codegen_cranelift::register_call_assembler_blackhole(jit_blackhole_resume_from_guard);
        // Bridge compilation is triggered by MetaInterp.must_compile()
        // (compile.py:783-784), not by backend fail_count threshold.
        majit_codegen_cranelift::register_inline_frame_arena(arena_global_info());
        // Bridge compilation for guards hit via try_function_entry_jit is
        // handled at the eval.rs level: run_compiled_detailed_with_bridge_keyed
        // returns BridgeCompilationRequest, and try_function_entry_jit calls
        // jit_bridge_compile_for_guard after releasing the driver borrow.
    });
}

/// RPython resume_in_blackhole parity: resume execution from the guard
/// failure point using the IR-based blackhole interpreter.
///
/// RPython warmspot.py:1021 assembler_call_helper → handle_fail →
/// resume_in_blackhole → BlackholeInterpreter.dispatch_loop from guard PC.
///
/// In majit, this calls MetaInterp::blackhole_guard_failure which executes
/// the remaining IR ops from guard+1 to Finish, returning the raw result.
fn jit_blackhole_resume_from_guard(
    green_key: u64,
    trace_id: u64,
    fail_index: u32,
    fail_values_ptr: *const i64,
    num_fail_values: usize,
) -> Option<i64> {
    if fail_values_ptr.is_null() || num_fail_values == 0 {
        return None;
    }
    let fail_values = unsafe { std::slice::from_raw_parts(fail_values_ptr, num_fail_values) };
    // The green_key from the target may be 0 for function-entry traces.
    // Recover the real green_key from the callee frame's code pointer.
    let actual_green_key = if green_key == 0 && num_fail_values >= 1 {
        let frame_ptr = fail_values[0] as *const pyre_interpreter::pyframe::PyFrame;
        if !frame_ptr.is_null() {
            let code = unsafe { (*frame_ptr).code };
            crate::eval::make_green_key(code, 0)
        } else {
            green_key
        }
    } else {
        green_key
    };
    let (driver, _) = crate::eval::driver_pair();
    let exception = majit_metainterp::blackhole::ExceptionState::default();
    if majit_metainterp::majit_log_enabled() {
        eprintln!(
            "[blackhole-resume] gk={} trace={} fail_idx={} nvals={}",
            actual_green_key,
            trace_id,
            fail_index,
            fail_values.len(),
        );
    }
    // RPython _run_forever parity: blackhole may return Jump (loop back)
    // or GuardFailed (nested guard failure). Keep running until Finish.
    // Zero-copy: pass fail_values slice directly (from caller's jf_frame)
    // on first iteration, avoiding heap Vec allocation.
    let mut current_fail_index = fail_index;
    let mut _owned_values: Option<Vec<i64>> = None;
    let mut current_slice: &[i64] = fail_values;
    loop {
        let bh_opt = driver.blackhole_guard_failure(
            actual_green_key,
            trace_id,
            current_fail_index,
            current_slice,
            exception.clone(),
        );
        let (bh_result, _bh_exc) = bh_opt?;
        match bh_result {
            majit_metainterp::blackhole::BlackholeResult::Finish { values, .. } => {
                return values.first().copied();
            }
            majit_metainterp::blackhole::BlackholeResult::Jump { values, .. } => {
                // Loop back: re-enter from the loop header (fail_index=0)
                current_fail_index = 0;
                _owned_values = Some(values);
                current_slice = _owned_values.as_ref().unwrap();
                // Jump means re-enter the compiled code from the loop header.
                // For now, fall back to force_fn since we don't have loop
                // re-entry support in the blackhole yet.
                return None;
            }
            majit_metainterp::blackhole::BlackholeResult::GuardFailed { fail_values, .. } => {
                // Nested guard failure inside blackhole. Fall back.
                return None;
            }
            _ => return None,
        }
    }
}

/// compile.py:714 (_trace_and_compile_from_bridge):
/// Called when a guard failure reaches the trace_eagerness threshold.
/// Traces the alternative path from the guard failure point and compiles
/// a bridge.
///
/// pyjitpl.py:2884 handle_guard_failure:
///   initialize_state_from_guard_failure(resumedescr, deadframe)
///   prepare_resume_from_failure(deadframe, inputargs, resumedescr, excdata)
///   self.interpret()
///
/// The tracing loop mirrors pyjitpl.py interpret(): execute bytecodes
/// from the guard failure PC until a Finish (return) or CloseLoop
/// (back-edge to loop header) is reached.
/// compile.py:714 _trace_and_compile_from_bridge parity.
///
/// Returns true if the bridge was successfully compiled and attached.
/// On failure (trace abort, start failure), returns false so the caller
/// falls through to resume_in_blackhole (RPython pyjitpl.py:2906-2907
/// SwitchToBlackhole → run_blackhole_interp_to_cancel_tracing).
pub fn jit_bridge_compile_for_guard(
    green_key: u64,
    trace_id: u64,
    fail_index: u32,
    frame: &mut PyFrame,
    resume_pc_hint: usize,
) -> bool {
    use crate::eval::build_jit_state;
    use crate::jit::state::PyreEnv;
    use crate::jit::trace::trace_bytecode;

    let (driver, info) = crate::eval::driver_pair();

    // compile.py:702-709: try/finally start_compiling/done_compiling.
    // Use Drop guard to ensure ST_BUSY_FLAG is always cleared.
    driver
        .meta_interp_mut()
        .set_guard_compiling(green_key, trace_id, fail_index, true);
    struct DoneCompilingGuard {
        green_key: u64,
        trace_id: u64,
        fail_index: u32,
    }
    impl Drop for DoneCompilingGuard {
        fn drop(&mut self) {
            let (driver, _) = crate::eval::driver_pair();
            driver.meta_interp_mut().set_guard_compiling(
                self.green_key,
                self.trace_id,
                self.fail_index,
                false,
            );
        }
    }
    let _done_compiling = DoneCompilingGuard {
        green_key,
        trace_id,
        fail_index,
    };

    // RPython resume_in_blackhole parity: use resume_pc from guard's
    // resume data (via LAST_GUARD_RESUME_PC or recovery_layout), not
    // frame.next_instr which may have been reset by force_fn.
    let resume_pc = if resume_pc_hint > 0 {
        resume_pc_hint
    } else {
        frame.next_instr
    };
    if resume_pc == 0 {
        return false;
    }
    frame.next_instr = resume_pc;
    let code = unsafe { &*frame.code };
    let env = PyreEnv;
    let mut jit_state = build_jit_state(frame, info);
    // resume.py:1042: for bridge guards, adjust jit_state to match
    // fail_arg_types shape. Don't modify the real frame — only the
    // jit_state that controls trace_meta construction.
    let bridge_adjusted_vsd;
    {
        let (driver, _) = crate::eval::driver_pair();
        let n_fail_args = driver
            .meta_interp_mut()
            .fail_arg_count_for(green_key, trace_id, fail_index);
        if n_fail_args >= 3 {
            let n_slots = n_fail_args - 3;
            let nlocals = frame.nlocals();
            let new_vsd = nlocals + n_slots.saturating_sub(nlocals);
            if new_vsd < jit_state.valuestackdepth {
                jit_state.valuestackdepth = new_vsd;
            }
            bridge_adjusted_vsd = Some(new_vsd);
        } else {
            bridge_adjusted_vsd = None;
        }
    }
    let loop_header_pc = 0; // not used by start_bridge_tracing

    if majit_metainterp::majit_log_enabled() {
        eprintln!(
            "[jit][bridge-trace] start key={} trace={} fail={} resume_pc={}",
            green_key, trace_id, fail_index, resume_pc
        );
    }

    // compile.py:714: start_retrace_from_guard + set bridge_info.
    if !driver.start_bridge_tracing(
        green_key,
        trace_id,
        fail_index,
        &mut jit_state,
        &env,
        resume_pc,
        loop_header_pc,
    ) {
        if majit_metainterp::majit_log_enabled() {
            eprintln!(
                "[jit][bridge-trace] start_bridge_tracing failed key={} trace={} fail={}",
                green_key, trace_id, fail_index
            );
        }
        return false;
    }

    // RPython pyjitpl.py:3101 _prepare_exception_resumption +
    // pyjitpl.py:3132 prepare_resume_from_failure parity:
    // For exception guard bridges (GUARD_EXCEPTION / GUARD_NO_EXCEPTION),
    // emit SAVE_EXC_CLASS + SAVE_EXCEPTION at trace start, then
    // RESTORE_EXCEPTION before the guard. The exception class/value
    // are read from the TLS exception state set by Cranelift codegen.
    if driver.last_bridge_is_exception_guard {
        let exc_class = majit_codegen_cranelift::jit_exc_class_raw();
        let exc_value = majit_codegen_cranelift::jit_exc_value_raw();
        if exc_class != 0 {
            // RPython pyjitpl.py:3125-3126 + 3138:
            // SAVE_EXC_CLASS, SAVE_EXCEPTION, RESTORE_EXCEPTION
            driver
                .meta_interp_mut()
                .emit_exception_bridge_prologue(exc_class, exc_value);
            if majit_metainterp::majit_log_enabled() {
                eprintln!(
                    "[jit][bridge-exc] exception guard bridge: class={:#x} value={:#x}",
                    exc_class, exc_value
                );
            }
        }
        driver.last_bridge_is_exception_guard = false;
    }

    // pyjitpl.py:2841 interpret(): trace bytecodes from guard failure PC
    // until the bridge path terminates (Finish or CloseLoop).
    let mut trace_frame = Box::new(frame.snapshot_for_tracing());
    // resume.py:1042: adjust snapshot's valuestackdepth to match fail_arg_types.
    if let Some(vsd) = bridge_adjusted_vsd {
        if vsd < trace_frame.valuestackdepth {
            trace_frame.valuestackdepth = vsd;
        }
    }
    let max_bridge_ops = 200;

    for step in 0..max_bridge_ops {
        let pc = trace_frame.next_instr;
        if pc >= code.instructions.len() {
            break;
        }

        // Feed one bytecode to the active trace via merge_point.
        let outcome = driver.jit_merge_point_keyed(
            green_key,
            pc,
            &mut jit_state,
            &env,
            || {},
            |ctx, sym| {
                // Bridge tracing: create a per-step snapshot.
                let snapshot = Box::new(trace_frame.snapshot_for_tracing());
                let (action, _executed) = trace_bytecode(ctx, sym, code, pc, snapshot);
                action
            },
        );

        // merge_point handles Finish/CloseLoop via bridge_info.
        // If it returns an outcome, the bridge was compiled.
        if outcome.is_some() {
            if majit_metainterp::majit_log_enabled() {
                eprintln!(
                    "[jit][bridge-trace] compiled at step={} pc={} key={}",
                    step, pc, green_key
                );
            }
            return true;
        }

        // If the driver is no longer tracing, the bridge was compiled
        // (or aborted) inside merge_point. Check whether a bridge was
        // actually attached to distinguish success from abort.
        if !driver.is_tracing() {
            let compiled = driver
                .meta_interp()
                .bridge_was_compiled(green_key, trace_id, fail_index);
            if majit_metainterp::majit_log_enabled() {
                eprintln!(
                    "[jit][bridge-trace] trace ended at step={} pc={} key={} compiled={}",
                    step, pc, green_key, compiled
                );
            }
            return compiled;
        }

        // Advance the trace frame's PC for the next instruction.
        // execute_opcode_step on MIFrame updates symbolic state
        // but not the concrete frame's next_instr. Advance manually.
        trace_frame.next_instr = pc + 1;
    }

    // Trace didn't converge — abort.
    if driver.is_tracing() {
        if majit_metainterp::majit_log_enabled() {
            eprintln!(
                "[jit][bridge-trace] abort: too many ops key={} trace={} fail={}",
                green_key, trace_id, fail_index
            );
        }
        driver.meta_interp_mut().abort_trace(false);
    }
    false
}

extern "C" fn jit_bridge_compile_callee(
    frame_ptr: i64,
    fail_index: u32,
    trace_id: u64,
    green_key: u64,
) -> i64 {
    let _suspend_inline_result = pyre_interpreter::call::suspend_inline_handled_result();
    use majit_ir::{InputArg, Op, OpCode, OpRef, Type};
    use std::collections::HashMap;

    let frame = unsafe { &mut *(frame_ptr as *mut PyFrame) };
    // Reset frame state — compiled code may have modified next_instr/vsd.
    let nlocals = unsafe { &*frame.code }.varnames.len();
    frame.next_instr = 0;
    frame.valuestackdepth = nlocals;
    // If green_key=0, compute from the callee frame's code pointer.
    let green_key = if green_key == 0 {
        crate::eval::make_green_key(frame.code, 0)
    } else {
        green_key
    };
    let protocol = finish_protocol(green_key);
    let result = match crate::eval::eval_with_jit(frame) {
        Ok(r) => match protocol {
            FinishProtocol::RawInt if !r.is_null() && unsafe { is_int(r) } => unsafe {
                w_int_get_value(r)
            },
            FinishProtocol::RawInt => r as i64,
            FinishProtocol::Boxed => r as i64,
        },
        Err(_) => 0i64,
    };

    let bridge_inputargs = vec![InputArg::from_type(Type::Int, 0)];
    let frame_opref = OpRef(0);

    let force_fn_ptr = jit_force_callee_frame as *const () as i64;
    let func_const_ref = OpRef(10_000);
    let mut constants: HashMap<u32, i64> = HashMap::new();
    constants.insert(func_const_ref.0, force_fn_ptr);

    // pyjitpl.py:3198: compile_done_with_this_frame selects descr based
    // on result_type. result_type=Int → done_with_this_frame_descr_int.
    let call_descr = majit_metainterp::make_call_descr(&[Type::Int], Type::Int);
    let call_result = OpRef(1);
    let mut call_op = Op::with_descr(OpCode::CallI, &[func_const_ref, frame_opref], call_descr);
    call_op.pos = call_result;

    let finish_descr = majit_metainterp::make_fail_descr_typed(vec![Type::Int]);
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
            HashMap::new(),
            HashMap::new(),
        );
    }

    // done_compiling happens automatically via DoneCompilingGuard Drop.
    result
}

// ── Callee frame creation for call_assembler ─────────────────────

/// Public wrapper for trace-through inlining.
pub fn create_callee_frame_impl_pub(caller_frame: i64, callable: i64, args: &[PyObjectRef]) -> i64 {
    create_callee_frame_impl(caller_frame, callable, args)
}

#[inline]
fn reset_reused_call_frame(frame: &mut PyFrame, args: &[PyObjectRef]) {
    frame.locals_cells_stack_w.as_mut_slice().fill(PY_NULL);
    let nargs = args.len().min(frame.nlocals());
    for (idx, value) in args.iter().take(nargs).enumerate() {
        frame.locals_cells_stack_w[idx] = *value;
    }
    frame.valuestackdepth = frame.stack_base();
    frame.next_instr = 0;
    frame.vable_token = 0;
    frame.block_stack.clear();
    frame.pending_inline_results.clear();
    frame.pending_inline_resume_pc = None;
}

fn create_callee_frame_impl_1_boxed(
    caller_frame: i64,
    callable: PyObjectRef,
    boxed_arg: PyObjectRef,
) -> i64 {
    let code_ptr = unsafe { function_get_code(callable) };
    let caller = unsafe { &*(caller_frame as *const PyFrame) };
    let globals = unsafe { function_get_globals(callable) };
    let func_code = code_ptr as *const pyre_bytecode::CodeObject;

    let arena = arena_ref();
    if let Some((ptr, was_init)) = arena.take() {
        if was_init {
            let f = unsafe { &mut *ptr };
            if f.code == func_code
                && f.namespace == globals
                && f.execution_context == caller.execution_context
            {
                reset_reused_call_frame(f, &[boxed_arg]);
            } else {
                unsafe {
                    std::ptr::write(
                        ptr,
                        PyFrame::new_for_call(
                            func_code,
                            &[boxed_arg],
                            globals,
                            caller.execution_context,
                        ),
                    );
                    (&mut *ptr).fix_array_ptrs();
                }
            }
        } else {
            unsafe {
                std::ptr::write(
                    ptr,
                    PyFrame::new_for_call(
                        func_code,
                        &[boxed_arg],
                        globals,
                        caller.execution_context,
                    ),
                );
                (&mut *ptr).fix_array_ptrs();
            }
            arena.mark_initialized();
        }
        return ptr as i64;
    }

    let frame_ptr = Box::into_raw(Box::new(PyFrame::new_for_call(
        func_code,
        &[boxed_arg],
        globals,
        caller.execution_context,
    )));
    unsafe { &mut *frame_ptr }.fix_array_ptrs();
    frame_ptr as i64
}

fn create_self_recursive_callee_frame_impl_1_boxed(
    caller_frame: i64,
    boxed_arg: PyObjectRef,
) -> i64 {
    let caller = unsafe { &*(caller_frame as *const PyFrame) };
    let func_code = caller.code;
    let globals = caller.namespace;
    let execution_context = caller.execution_context;

    let arena = arena_ref();
    if let Some((ptr, was_init)) = arena.take() {
        if was_init {
            let f = unsafe { &mut *ptr };
            if f.code == func_code
                && f.namespace == globals
                && f.execution_context == execution_context
            {
                reset_reused_call_frame(f, &[boxed_arg]);
            } else {
                unsafe {
                    std::ptr::write(
                        ptr,
                        PyFrame::new_for_call(func_code, &[boxed_arg], globals, execution_context),
                    );
                    (&mut *ptr).fix_array_ptrs();
                }
            }
        } else {
            unsafe {
                std::ptr::write(
                    ptr,
                    PyFrame::new_for_call(func_code, &[boxed_arg], globals, execution_context),
                );
                (&mut *ptr).fix_array_ptrs();
            }
            arena.mark_initialized();
        }
        return ptr as i64;
    }

    let frame_ptr = Box::into_raw(Box::new(PyFrame::new_for_call(
        func_code,
        &[boxed_arg],
        globals,
        execution_context,
    )));
    unsafe { &mut *frame_ptr }.fix_array_ptrs();
    frame_ptr as i64
}

fn create_callee_frame_impl(caller_frame: i64, callable: i64, args: &[PyObjectRef]) -> i64 {
    let callable = callable as PyObjectRef;
    let code_ptr = unsafe { function_get_code(callable) };
    let caller = unsafe { &*(caller_frame as *const PyFrame) };
    let globals = unsafe { function_get_globals(callable) };
    let func_code = code_ptr as *const pyre_bytecode::CodeObject;

    let arena = arena_ref();
    if let Some((ptr, was_init)) = arena.take() {
        if was_init {
            // Fast reinit: only update fields that change between calls.
            // code, execution_context, namespace, locals_cells_stack_w.ptr
            // are stable for self-recursion (same function, same module).
            let f = unsafe { &mut *ptr };
            if f.code == func_code
                && f.namespace == globals
                && f.execution_context == caller.execution_context
            {
                reset_reused_call_frame(f, args);
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
    create_callee_frame_impl_1_boxed(caller_frame, callable as PyObjectRef, arg0 as PyObjectRef)
}

/// Self-recursive single-arg variant.
///
/// This skips rediscovering code/globals from a function object and reuses the
/// caller frame's code/namespace/execution_context directly, which matches the
/// existing self-recursive raw helper path more closely.
pub extern "C" fn jit_create_self_recursive_callee_frame_1(caller_frame: i64, arg0: i64) -> i64 {
    create_self_recursive_callee_frame_impl_1_boxed(caller_frame, arg0 as PyObjectRef)
}

/// Self-recursive raw-int variant: creates the frame WITHOUT boxing
/// the argument. The raw int is passed directly to compiled code via
/// CallAssemblerI inputargs. Boxing only happens on guard failure
/// (in force_fn / jit_force_self_recursive_call_raw_1).
///
/// RPython parity: compiled code uses jitframe slots, not PyFrame
/// locals. Frame locals are only needed for interpreter fallback.
pub extern "C" fn jit_create_self_recursive_callee_frame_1_raw_int(
    caller_frame: i64,
    raw_int_arg: i64,
) -> i64 {
    let caller = unsafe { &*(caller_frame as *const PyFrame) };
    let func_code = caller.code;
    let globals = caller.namespace;
    let execution_context = caller.execution_context;

    let arena = arena_ref();
    if let Some((ptr, was_init)) = arena.take() {
        let f = unsafe { &mut *ptr };
        if was_init && f.code == func_code {
            // Fast path: reuse frame, only reset next_instr.
            // Skip boxing + locals fill — compiled code uses raw inputargs.
            // Guard failure path (force_fn) will box raw_int_arg into
            // frame.locals when interpreter resume is needed.
            f.next_instr = 0;
            f.vable_token = 0;
        } else {
            // Cold: different code or first use — full init needed.
            let boxed = pyre_object::intobject::w_int_new(raw_int_arg);
            unsafe {
                std::ptr::write(
                    ptr,
                    PyFrame::new_for_call(func_code, &[boxed], globals, execution_context),
                );
                (&mut *ptr).fix_array_ptrs();
            }
            if !was_init {
                arena.mark_initialized();
            }
        }
        return ptr as i64;
    }

    let boxed = pyre_object::intobject::w_int_new(raw_int_arg);
    let frame_ptr = Box::into_raw(Box::new(PyFrame::new_for_call(
        func_code,
        &[boxed],
        globals,
        execution_context,
    )));
    unsafe { &mut *frame_ptr }.fix_array_ptrs();
    frame_ptr as i64
}

/// Raw-int variant: accepts a raw int and boxes it internally.
/// Eliminates trace_box_int CallI from the trace (boxing folded into frame creation).
pub extern "C" fn jit_create_callee_frame_1_raw_int(
    caller_frame: i64,
    callable: i64,
    raw_int_arg: i64,
) -> i64 {
    let boxed = pyre_object::intobject::w_int_new(raw_int_arg);
    create_callee_frame_impl_1_boxed(caller_frame, callable as PyObjectRef, boxed)
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
    let frame = unsafe { &*(frame_ptr as *const PyFrame) };
    let green_key = crate::eval::make_green_key(frame.code, frame.next_instr);
    let protocol = finish_protocol(green_key);
    let result = jit_force_callee_frame(frame_ptr);
    match protocol {
        FinishProtocol::RawInt => w_int_new(result) as i64,
        FinishProtocol::Boxed => result,
    }
}

pub extern "C" fn jit_drop_callee_frame(frame_ptr: i64) {
    if frame_ptr & 1 != 0 {
        return;
    }
    let ptr = frame_ptr as *mut PyFrame;
    let arena = arena_ref();
    let reused = arena.put(ptr);
    if !reused {
        // Not an arena frame (heap fallback) — free it
        unsafe { drop(Box::from_raw(ptr)) };
    }
}

// ===========================================================================
// Blackhole helper functions
//
// RPython blackhole.py: bhimpl_recursive_call_i, bhimpl_residual_call_*
//
// These are called by the BlackholeInterpreter through JitCode.fn_ptrs.
// They execute Python operations WITHOUT JIT re-entry, matching RPython's
// structural isolation: the blackhole never calls maybe_compile_and_run.
// ===========================================================================

/// RPython: bhimpl_recursive_call_i — call a Python function in blackhole mode.
///
/// The blackhole pops callable and args into registers before calling this.
/// blackhole.py bhimpl_residual_call parity: variable-arity call helper.
///
/// Convention: call_int_function dispatches with args=[callable, arg0, ..., argN, frame_ptr].
/// frame_ptr is always the LAST argument. The number of Python args = total_args - 2
/// (subtract callable and frame_ptr).
///
/// For nargs=0: fn(callable, frame_ptr) → 2 args
/// For nargs=1: fn(callable, arg0, frame_ptr) → 3 args
/// For nargs=2: fn(callable, arg0, arg1, frame_ptr) → 4 args
/// For nargs=3: fn(callable, arg0, arg1, arg2, frame_ptr) → 5 args
/// etc.
///
/// call_int_function in machine.rs transmutes to the correct arity.
pub extern "C" fn bh_call_fn(callable: i64, arg0: i64, frame_ptr: i64) -> i64 {
    // This is the 3-arg entry point (nargs=1). For other arities,
    // call_int_function transmutes to bh_call_fn_N variants below.
    bh_call_fn_impl(
        callable as PyObjectRef,
        &[arg0 as PyObjectRef],
        frame_ptr as *const PyFrame,
    )
}

pub extern "C" fn bh_call_fn_0(callable: i64, frame_ptr: i64) -> i64 {
    bh_call_fn_impl(callable as PyObjectRef, &[], frame_ptr as *const PyFrame)
}

pub extern "C" fn bh_call_fn_2(callable: i64, arg0: i64, arg1: i64, frame_ptr: i64) -> i64 {
    bh_call_fn_impl(
        callable as PyObjectRef,
        &[arg0 as PyObjectRef, arg1 as PyObjectRef],
        frame_ptr as *const PyFrame,
    )
}

pub extern "C" fn bh_call_fn_3(
    callable: i64,
    arg0: i64,
    arg1: i64,
    arg2: i64,
    frame_ptr: i64,
) -> i64 {
    bh_call_fn_impl(
        callable as PyObjectRef,
        &[
            arg0 as PyObjectRef,
            arg1 as PyObjectRef,
            arg2 as PyObjectRef,
        ],
        frame_ptr as *const PyFrame,
    )
}

pub extern "C" fn bh_call_fn_4(
    callable: i64,
    arg0: i64,
    arg1: i64,
    arg2: i64,
    arg3: i64,
    frame_ptr: i64,
) -> i64 {
    bh_call_fn_impl(
        callable as PyObjectRef,
        &[
            arg0 as PyObjectRef,
            arg1 as PyObjectRef,
            arg2 as PyObjectRef,
            arg3 as PyObjectRef,
        ],
        frame_ptr as *const PyFrame,
    )
}

pub extern "C" fn bh_call_fn_5(
    callable: i64,
    a0: i64,
    a1: i64,
    a2: i64,
    a3: i64,
    a4: i64,
    frame_ptr: i64,
) -> i64 {
    bh_call_fn_impl(
        callable as PyObjectRef,
        &[
            a0 as PyObjectRef,
            a1 as PyObjectRef,
            a2 as PyObjectRef,
            a3 as PyObjectRef,
            a4 as PyObjectRef,
        ],
        frame_ptr as *const PyFrame,
    )
}

pub extern "C" fn bh_call_fn_6(
    callable: i64,
    a0: i64,
    a1: i64,
    a2: i64,
    a3: i64,
    a4: i64,
    a5: i64,
    frame_ptr: i64,
) -> i64 {
    bh_call_fn_impl(
        callable as PyObjectRef,
        &[
            a0 as PyObjectRef,
            a1 as PyObjectRef,
            a2 as PyObjectRef,
            a3 as PyObjectRef,
            a4 as PyObjectRef,
            a5 as PyObjectRef,
        ],
        frame_ptr as *const PyFrame,
    )
}

pub extern "C" fn bh_call_fn_7(
    callable: i64,
    a0: i64,
    a1: i64,
    a2: i64,
    a3: i64,
    a4: i64,
    a5: i64,
    a6: i64,
    frame_ptr: i64,
) -> i64 {
    bh_call_fn_impl(
        callable as PyObjectRef,
        &[
            a0 as PyObjectRef,
            a1 as PyObjectRef,
            a2 as PyObjectRef,
            a3 as PyObjectRef,
            a4 as PyObjectRef,
            a5 as PyObjectRef,
            a6 as PyObjectRef,
        ],
        frame_ptr as *const PyFrame,
    )
}

pub extern "C" fn bh_call_fn_8(
    callable: i64,
    a0: i64,
    a1: i64,
    a2: i64,
    a3: i64,
    a4: i64,
    a5: i64,
    a6: i64,
    a7: i64,
    frame_ptr: i64,
) -> i64 {
    bh_call_fn_impl(
        callable as PyObjectRef,
        &[
            a0 as PyObjectRef,
            a1 as PyObjectRef,
            a2 as PyObjectRef,
            a3 as PyObjectRef,
            a4 as PyObjectRef,
            a5 as PyObjectRef,
            a6 as PyObjectRef,
            a7 as PyObjectRef,
        ],
        frame_ptr as *const PyFrame,
    )
}

fn bh_call_fn_impl(
    callable: PyObjectRef,
    args: &[PyObjectRef],
    parent_frame: *const PyFrame,
) -> i64 {
    if callable.is_null() {
        let err = pyre_interpreter::PyError::new(
            pyre_interpreter::PyErrorKind::TypeError,
            "call on null callable".to_string(),
        );
        majit_metainterp::blackhole::BH_LAST_EXC_VALUE.with(|c| c.set(err.to_exc_object() as i64));
        return 0;
    }
    // RPython bhimpl_residual_call: dispatch builtin vs user function.
    if !unsafe { is_function(callable) } {
        let func = unsafe { pyre_interpreter::builtin_code_get(callable) };
        match func(args) {
            Ok(result) if !result.is_null() => return result as i64,
            Ok(_) => return 0,
            Err(err) => {
                let exc_obj = err.to_exc_object();
                majit_metainterp::blackhole::BH_LAST_EXC_VALUE.with(|c| c.set(exc_obj as i64));
                return 0;
            }
        }
    }
    let parent_frame = unsafe { &*parent_frame };
    match pyre_interpreter::call::call_user_function_plain(parent_frame, callable, args) {
        Ok(result) => result as i64,
        Err(err) => {
            let exc_obj = err.to_exc_object();
            majit_metainterp::blackhole::BH_LAST_EXC_VALUE.with(|c| c.set(exc_obj as i64));
            0
        }
    }
}

/// RPython: bhimpl_residual_call — LOAD_GLOBAL helper.
///
/// Loads a global name from the frame's namespace (globals + builtins).
/// namei is the raw oparg from LOAD_GLOBAL: name_idx = namei >> 1.
pub extern "C" fn bh_load_global_fn(frame_ptr: i64, namei: i64) -> i64 {
    let frame = unsafe { &*(frame_ptr as *const PyFrame) };
    let code = unsafe { &*frame.code };
    let raw = namei as usize;
    let idx = raw >> 1;

    if idx >= code.names.len() {
        return 0;
    }

    let name = code.names[idx].as_ref();
    // PyFrame.namespace = globals; look up name there.
    // opcode_load_name dispatches through the NamespaceOpcodeHandler trait;
    // for the blackhole we call the namespace directly.
    let ns = unsafe { &*frame.namespace };
    match ns.get(name) {
        Some(&value) => value as i64,
        None => {
            // NameError: set exception object in TLS.
            let err = pyre_interpreter::PyError::new(
                pyre_interpreter::PyErrorKind::NameError,
                format!("name '{}' is not defined", name),
            );
            let exc_obj = err.to_exc_object();
            majit_metainterp::blackhole::BH_LAST_EXC_VALUE.with(|c| c.set(exc_obj as i64));
            0
        }
    }
}

/// Load a constant from the frame's code object.
/// RPython assembler.py parity: constants are resolved at blackhole runtime.
pub extern "C" fn bh_load_const_fn(frame_ptr: i64, consti: i64) -> i64 {
    let frame = unsafe { &*(frame_ptr as *const PyFrame) };
    frame.load_const_pyobj(consti as usize) as i64
}

/// Box a raw integer into a PyObject (w_int_new wrapper).
pub extern "C" fn bh_box_int_fn(value: i64) -> i64 {
    w_int_new(value) as i64
}

/// Truthiness check: PyObjectRef → raw 0 or 1.
pub extern "C" fn bh_truth_fn(value: i64) -> i64 {
    let obj = value as PyObjectRef;
    if obj.is_null() {
        return 0;
    }
    pyre_interpreter::opcode_ops::truth_value(obj) as i64
}

/// RPython: bhimpl_int_lt, bhimpl_int_eq, etc. — comparison helper.
///
/// Performs a Python-level comparison and returns a boolean PyObject.
/// op_code encodes the CompareOp tag from CPython 3.13 COMPARE_OP.
pub extern "C" fn bh_compare_fn(lhs: i64, rhs: i64, op_code: i64) -> i64 {
    let lhs = lhs as PyObjectRef;
    let rhs = rhs as PyObjectRef;
    if lhs.is_null() || rhs.is_null() {
        let err = pyre_interpreter::PyError::new(
            pyre_interpreter::PyErrorKind::TypeError,
            "comparison on null operand".to_string(),
        );
        majit_metainterp::blackhole::BH_LAST_EXC_VALUE.with(|c| c.set(err.to_exc_object() as i64));
        return 0;
    }

    // op_code is the raw CPython COMPARE_OP oparg (= ComparisonOperator discriminant).
    // Transmute back and call compare_value.
    use pyre_bytecode::bytecode::ComparisonOperator;
    let op: ComparisonOperator = unsafe { std::mem::transmute(op_code as u8) };
    match pyre_interpreter::opcode_ops::compare_value(lhs, rhs, op) {
        Ok(result) => result as i64,
        Err(err) => {
            let exc_obj = err.to_exc_object();
            majit_metainterp::blackhole::BH_LAST_EXC_VALUE.with(|c| c.set(exc_obj as i64));
            0
        }
    }
}

/// RPython: bhimpl_int_add, bhimpl_int_sub, etc. — binary op helper.
///
/// Performs a Python-level binary operation.
/// op_code is the BinaryOperator tag from CPython 3.13 BINARY_OP.
pub extern "C" fn bh_binary_op_fn(lhs: i64, rhs: i64, op_code: i64) -> i64 {
    let lhs = lhs as PyObjectRef;
    let rhs = rhs as PyObjectRef;
    if lhs.is_null() || rhs.is_null() {
        let err = pyre_interpreter::PyError::new(
            pyre_interpreter::PyErrorKind::TypeError,
            "binary op on null operand".to_string(),
        );
        majit_metainterp::blackhole::BH_LAST_EXC_VALUE.with(|c| c.set(err.to_exc_object() as i64));
        return 0;
    }

    // op_code is the raw CPython BINARY_OP oparg (= BinaryOperator discriminant).
    // Transmute back to BinaryOperator enum and call binary_value.
    use pyre_bytecode::bytecode::BinaryOperator;
    let op: BinaryOperator = unsafe { std::mem::transmute(op_code as u8) };
    match pyre_interpreter::opcode_ops::binary_value(lhs, rhs, op) {
        Ok(result) => result as i64,
        Err(err) => {
            let exc_obj = err.to_exc_object();
            majit_metainterp::blackhole::BH_LAST_EXC_VALUE.with(|c| c.set(exc_obj as i64));
            0
        }
    }
}

/// STORE_SUBSCR: obj[key] = value.
/// RPython bhimpl_newlist: create a list from N items.
/// argc is a raw integer count; items follow as PyObjectRef args.
/// Blackhole's call_may_force_ref passes args from registers.
pub extern "C" fn bh_build_list_fn(argc: i64, item0: i64, item1: i64) -> i64 {
    let n = argc as usize;
    let items: Vec<pyre_object::PyObjectRef> = match n {
        0 => vec![],
        1 => vec![item0 as pyre_object::PyObjectRef],
        2 => vec![
            item0 as pyre_object::PyObjectRef,
            item1 as pyre_object::PyObjectRef,
        ],
        _ => vec![], // argc > 2 not supported via this helper
    };
    pyre_interpreter::runtime_ops::build_list_from_refs(&items) as i64
}

pub extern "C" fn bh_store_subscr_fn(obj: i64, key: i64, value: i64) -> i64 {
    let obj = obj as pyre_object::PyObjectRef;
    let key = key as pyre_object::PyObjectRef;
    let value = value as pyre_object::PyObjectRef;
    if obj.is_null() || key.is_null() {
        let err = pyre_interpreter::PyError::new(
            pyre_interpreter::PyErrorKind::TypeError,
            "store subscript on null operand".to_string(),
        );
        majit_metainterp::blackhole::BH_LAST_EXC_VALUE.with(|c| c.set(err.to_exc_object() as i64));
        return 0;
    }
    if let Err(err) = pyre_interpreter::baseobjspace::setitem(obj, key, value) {
        let exc_obj = err.to_exc_object();
        majit_metainterp::blackhole::BH_LAST_EXC_VALUE.with(|c| c.set(exc_obj as i64));
        return 0;
    }
    1 // success (non-zero)
}
