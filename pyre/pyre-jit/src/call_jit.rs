//! JIT-specific call infrastructure — force/bridge callbacks, callee
//! frame creation helpers, frame pool.
//!
//! Separated from pyre-interpreter/src/call.rs so pyre-interpreter stays JIT-free.

use std::cell::UnsafeCell;
use std::mem::MaybeUninit;
use std::sync::Once;

use pyre_interpreter::bytecode::{Instruction, OpArgState};
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
    /// Stash Python exceptions from blackhole/force paths that cross
    /// FFI boundaries (compiled code → callback → exception).
    static LAST_CA_EXCEPTION: std::cell::RefCell<Option<pyre_interpreter::error::PyError>> =
        const { std::cell::RefCell::new(None) };
}

/// Take stashed exception from blackhole/force FFI paths.
pub fn take_ca_exception() -> Option<pyre_interpreter::error::PyError> {
    LAST_CA_EXCEPTION.with(|c| c.borrow_mut().take())
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
        let code = &*(function_get_code(callable) as *const pyre_interpreter::CodeObject);
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
#[cfg(not(target_arch = "wasm32"))]
pub fn arena_global_info() -> majit_backend_cranelift::InlineFrameArenaInfo {
    use majit_metainterp::jitframe::*;
    majit_backend_cranelift::InlineFrameArenaInfo {
        buf_base_addr: unsafe { std::ptr::addr_of!(ARENA_BUF_BASE) as usize },
        top_addr: unsafe { std::ptr::addr_of!(ARENA_TOP) as usize },
        initialized_addr: unsafe { std::ptr::addr_of!(ARENA_INITIALIZED) as usize },
        frame_size: std::mem::size_of::<pyre_interpreter::pyframe::PyFrame>(),
        frame_code_offset: pyre_interpreter::pyframe::PYFRAME_CODE_OFFSET,
        frame_next_instr_offset: pyre_interpreter::pyframe::PYFRAME_NEXT_INSTR_OFFSET,
        frame_vable_token_offset: pyre_interpreter::pyframe::PYFRAME_VABLE_TOKEN_OFFSET,
        create_fn_addr: jit_create_self_recursive_callee_frame_1_raw_int as *const () as usize,
        drop_fn_addr: jit_drop_callee_frame as *const () as usize,
        arena_cap: ARENA_CAP,
        jitframe_descrs: Some(majit_gc::rewrite::JitFrameDescrs {
            create_fn_addr: jit_create_self_recursive_callee_frame_1_raw_int as *const () as usize,
            drop_fn_addr: jit_drop_callee_frame as *const () as usize,
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
    #[cfg(not(target_arch = "wasm32"))]
    let _ = majit_backend_cranelift::take_pending_frame_restore();
    #[cfg(not(target_arch = "wasm32"))]
    let pending = majit_backend_cranelift::take_pending_force_local0();
    #[cfg(target_arch = "wasm32")]
    let pending: Option<i64> = None;

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
        let code = *(p.add(PYFRAME_CODE_OFFSET) as *const *const pyre_interpreter::CodeObject);
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

    // warmspot.py:1021 assembler_call_helper: execute callee in
    // plain interpreter. Do NOT change EVAL_OVERRIDE — it's a global
    // OnceLock that must stay as eval_with_jit for all other calls.
    let result = match pyre_interpreter::eval::eval_frame_plain(&mut func_frame) {
        Ok(r) => r,
        Err(_) => pyre_object::PY_NULL,
    };

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
    #[cfg(not(target_arch = "wasm32"))]
    let pending = majit_backend_cranelift::take_pending_force_local0();
    #[cfg(target_arch = "wasm32")]
    let pending: Option<i64> = None;
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
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(raw) = majit_backend_cranelift::execute_call_assembler_direct(
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

    // Direct PyFrame loading (not resume data — no enumerate_vars dispatch).
    // frame.locals_cells_stack_w is always PyObjectRef → write_a_ref only.
    // RPython's resume path uses enumerate_vars typed callbacks, but this
    // pyre-specific path loads from the concrete frame directly.
    let nlocals = code.varnames.len();
    for i in 0..nlocals {
        if i < frame.locals_cells_stack_w.len() {
            bh.setarg_r(i, frame.locals_cells_stack_w[i] as i64);
        }
    }

    // Load value stack into blackhole runtime stack.
    // locals_cells_stack_w layout: [locals..., cells..., stack...]
    // stack_base = nlocals + ncells (pyframe.py stack_base parity).
    let ncells = pyre_interpreter::pyframe::ncells(code);
    let stack_base = nlocals + ncells;
    let vsd = frame.valuestackdepth;
    for i in stack_base..vsd {
        if i < frame.locals_cells_stack_w.len() {
            bh.runtime_stack_push(0, frame.locals_cells_stack_w[i] as i64);
        }
    }

    // blackhole.py bhimpl_getfield_vable_*: set virtualizable pointer.
    // RPython: the virtualizable ptr is the frame itself.
    bh.virtualizable_ptr = frame as *mut PyFrame as i64;
    bh.virtualizable_info = crate::eval::get_virtualizable_info();

    if majit_metainterp::majit_log_enabled() {
        eprintln!(
            "[jit][blackhole] setup pc={} nlocals={} regs_r_len={} vable_ptr={:#x}",
            py_pc,
            nlocals,
            bh.registers_r.len(),
            bh.virtualizable_ptr as u64,
        );
    }

    // RPython: _run_forever(blackholeinterp, current_exc)
    bh.run();

    // blackhole.py:1664-1677 _done_with_this_frame:
    // RPython dispatches by _return_type (i/r/f/v).
    use majit_metainterp::blackhole::BhReturnType;
    let result = match bh.return_type {
        BhReturnType::Int => {
            // blackhole.py:1671 DoneWithThisFrameInt(get_tmpreg_i)
            pyre_object::intobject::w_int_new(bh.get_tmpreg_i()) as PyObjectRef
        }
        BhReturnType::Ref => {
            // blackhole.py:1673 DoneWithThisFrameRef(get_tmpreg_r)
            bh.get_tmpreg_r() as PyObjectRef
        }
        BhReturnType::Float => {
            // blackhole.py:1675 DoneWithThisFrameFloat(get_tmpreg_f)
            let bits = bh.get_tmpreg_f() as u64;
            pyre_object::floatobject::w_float_new(f64::from_bits(bits)) as PyObjectRef
        }
        BhReturnType::Void => std::ptr::null_mut(),
    };

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

/// resume.py:1042 rebuild_from_numbering / read_jitcode_pos_pc output.
/// Each decoded frame section from rd_numb.
pub struct ResumedFrame {
    /// resume.py:1050 jitcode_pos → jitcodes[jitcode_pos].
    /// pyre: CodeObject pointer (resolved from jitcode_index).
    pub code: *const pyre_interpreter::CodeObject,
    /// resume.py:1050 pc (Python bytecode PC for blackhole setposition).
    pub py_pc: usize,
    /// Raw frame.pc from rd_numb (= orgpc from snapshot).
    /// Some(pc): snapshot guard — orgpc known, liveness-based filling.
    ///   pc=0 is valid (function start / loop header at bytecode 0).
    /// None: no-snapshot guard (rd_numb pc=-1), positional fallback.
    pub rd_numb_pc: Option<usize>,
    /// Frame pointer for this stack frame.
    pub frame_ptr: *mut PyFrame,
    /// valuestackdepth extracted from vable_values (snapshot).
    pub vsd: usize,
    /// resume.py:928-931 consume_one_section: resolved values.
    /// Structure: [live_registers...] — no [frame, ni, vsd] header.
    /// RPython parity: vable values come from snapshot, not fail_args.
    pub values: Vec<majit_ir::Value>,
}

/// RPython resume.py:1312 blackhole_from_resumedata +
/// blackhole.py:1752 _run_forever parity.
///
/// Takes rd_numb-decoded per-frame data (ResumedFrame) instead of a
/// flat Value array. Frame boundaries come from rd_numb, not from
/// heuristic header detection.
///
/// Builds a blackhole chain (innermost first), then runs _run_forever:
/// callee blackhole → RETURN_VALUE → caller blackhole → merge point.
pub fn resume_in_blackhole(
    _caller_frame: &mut PyFrame,
    frames: &[ResumedFrame],
    merge_py_pc: usize,
) -> BlackholeResult {
    use majit_ir::Value;

    if frames.is_empty() {
        if majit_metainterp::majit_log_enabled() {
            eprintln!("[jit][bh-fail] resume_in_blackhole: empty frames");
        }
        return BlackholeResult::Failed;
    }

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

    thread_local! {
        static BH_BUILDER3: std::cell::UnsafeCell<majit_metainterp::blackhole::BlackholeInterpBuilder> =
            std::cell::UnsafeCell::new(majit_metainterp::blackhole::BlackholeInterpBuilder::new());
    }
    let builder = BH_BUILDER3.with(|cell| unsafe { &mut *cell.get() });

    // resume.py:1333-1343 blackhole_from_resumedata:
    // Build chain bottom-up. Process in reverse so the LAST acquired
    // interp is the innermost (callee), with nextblackholeinterp
    // pointing to the caller.
    let mut prev_bh: Option<majit_metainterp::blackhole::BlackholeInterpreter> = None;

    for (sec_idx, section) in frames.iter().enumerate().rev() {
        if section.frame_ptr.is_null() || section.code.is_null() {
            if majit_metainterp::majit_log_enabled() {
                eprintln!(
                    "[jit][bh-fail] resume_in_blackhole: null ptr at sec={} frame={:?} code={:?} py_pc={}",
                    sec_idx, section.frame_ptr, section.code, section.py_pc,
                );
            }
            builder.release_chain(prev_bh);
            return BlackholeResult::Failed;
        }
        let code = unsafe { &*section.code };
        let nlocals = code.varnames.len();
        let frame_ptr = section.frame_ptr;

        // resume.py:1340 curbh.setposition(jitcode, pc)
        let mut py_pc = section.py_pc;
        // Skip Cache/ExtendedArg/NotTaken (CPython 3.13 pseudo-instructions).
        while py_pc > 0 {
            match pyre_interpreter::decode_instruction_at(code, py_pc) {
                Some((pyre_interpreter::bytecode::Instruction::Cache, _))
                | Some((pyre_interpreter::bytecode::Instruction::ExtendedArg, _))
                | Some((pyre_interpreter::bytecode::Instruction::NotTaken, _)) => {
                    py_pc -= 1;
                }
                _ => break,
            }
        }
        if py_pc >= code.instructions.len() {
            builder.release_chain(prev_bh);
            return BlackholeResult::Failed;
        }

        // RPython parity: vsd from vable_values (snapshot), stored in
        // ResumedFrame.vsd by build_resumed_frames.
        let vsd = section.vsd;
        let stack_only = vsd.saturating_sub(nlocals);

        let pyjitcode = crate::jit::codewriter::get_jitcode(code, &writer);
        // RPython blackhole_from_resumedata does NOT pre-check for abort
        // opcodes. If the blackhole hits BC_ABORT during execution, it
        // sets bh.aborted=true and the _run_forever loop handles it.
        let jitcode_pc = if py_pc < pyjitcode.pc_map.len() {
            pyjitcode.pc_map[py_pc]
        } else {
            builder.release_chain(prev_bh);
            return BlackholeResult::Failed;
        };

        let mut bh = builder.acquire_interp();
        bh.setposition(pyjitcode.jitcode.clone(), jitcode_pc);

        // Set merge_point on the OUTERMOST (last = caller) blackhole.
        // When the blackhole reaches this PC, it exits with
        // ContinueRunningNormally so the JIT dispatch can re-enter
        // compiled code.
        if sec_idx == frames.len() - 1 {
            if let Some(merge_jitcode_pc) =
                crate::jit::codewriter::jitcode_pc_for_loop_header(&pyjitcode.pc_map, merge_py_pc)
            {
                bh.merge_point_jitcode_pc = Some(merge_jitcode_pc);
            }
        }

        if majit_metainterp::majit_log_enabled() {
            eprintln!(
                "[jit][blackhole-section] idx={} frame={:#x} py_pc={} vsd={} nvals={} vals={:?}",
                sec_idx,
                frame_ptr as usize,
                py_pc,
                vsd,
                section.values.len(),
                section.values.iter().take(5).collect::<Vec<_>>(),
            );
        }

        // resume.py:1381 consume_one_section → 1017 _prepare_next_section
        // → jitcode.py:147 enumerate_vars(info, liveness_info, cb_r)
        // RPython: both encoder (get_list_of_active_boxes) and decoder
        // use the SAME all_liveness data. Use JitCode.liveness (same
        // source as get_list_of_active_boxes) via majit_jitcode pointer.
        let live_pc = section.rd_numb_pc.unwrap_or(section.py_pc);
        // Look up LivenessInfo from JitCode (same data as capture side).
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
        let code_ref = unsafe { &*section.code };
        let pyjitcode = crate::jit::codewriter::get_jitcode(code_ref, &writer);
        let jc = &pyjitcode.jitcode;
        let liveness_info = jc
            .py_to_jit_pc
            .get(live_pc)
            .and_then(|&jit_pc| jc.liveness.iter().find(|info| info.pc as usize == jit_pc));
        let n_live = liveness_info
            .map(|info| info.live_r_regs.len())
            .unwrap_or_else(|| {
                // Fallback to LiveVars when JitCode liveness unavailable.
                let live = pyre_jit_trace::state::liveness_for(section.code);
                let max_stack = vsd.saturating_sub(nlocals);
                let n_live_locals = (0..nlocals)
                    .filter(|&i| live.is_local_live(live_pc, i))
                    .count();
                let n_live_stack = (0..max_stack)
                    .filter(|&i| live.is_stack_live(live_pc, i))
                    .count();
                n_live_locals + n_live_stack
            });
        // RPython parity: values = active boxes only (no header).
        let n_vals = section.values.len();
        let use_liveness = n_live == n_vals;
        if majit_metainterp::majit_log_enabled() {
            eprintln!(
                "[jit][liveness-fill] py_pc={} live_pc={} nlocals={} live={} vals={} mode={}",
                section.py_pc,
                live_pc,
                nlocals,
                n_live,
                n_vals,
                if use_liveness {
                    "liveness"
                } else {
                    "positional"
                },
            );
        }
        // resume.py:1017-1038 _prepare_next_section:
        // _callback_r → write_a_ref. pyre: all Python locals are
        // PyObjectRef → ref bank only (codewriter uses move_r).
        if use_liveness {
            if let Some(info) = liveness_info {
                for (val_idx, &reg_idx) in info.live_r_regs.iter().enumerate() {
                    let idx = reg_idx as usize;
                    if let Some(val) = section.values.get(val_idx) {
                        if idx < nlocals {
                            bh.setarg_r(idx, materialize_virtual(val));
                        } else {
                            bh.runtime_stack_push(0, materialize_virtual(val));
                        }
                    }
                }
            } else {
                // Fallback: LiveVars path — all slots ref.
                let live = pyre_jit_trace::state::liveness_for(section.code);
                let max_stack = vsd.saturating_sub(nlocals);
                let mut val_idx = 0;
                for i in 0..nlocals {
                    if live.is_local_live(live_pc, i) {
                        if let Some(val) = section.values.get(val_idx) {
                            bh.setarg_r(i, materialize_virtual(val));
                        }
                        val_idx += 1;
                    }
                }
                for i in 0..max_stack {
                    if live.is_stack_live(live_pc, i) {
                        if let Some(val) = section.values.get(val_idx) {
                            bh.runtime_stack_push(0, materialize_virtual(val));
                        }
                        val_idx += 1;
                    }
                }
            }
        } else {
            for i in 0..nlocals {
                if let Some(val) = section.values.get(i) {
                    bh.setarg_r(i, materialize_virtual(val));
                }
            }
            let stack_only = vsd.saturating_sub(nlocals);
            for i in 0..stack_only {
                if let Some(val) = section.values.get(nlocals + i) {
                    bh.runtime_stack_push(0, materialize_virtual(val));
                }
            }
        }
        // blackhole.py bhimpl_getfield_vable_*: set virtualizable pointer.
        bh.virtualizable_ptr = frame_ptr as i64;
        bh.virtualizable_info = crate::eval::get_virtualizable_info();

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
            frames.len(),
            merge_py_pc,
        );
    }

    // RPython blackhole.py:1752 _run_forever parity:
    // Run the innermost blackhole. On RETURN_VALUE (LeaveFrame),
    // pop to caller blackhole and continue.
    loop {
        bh.run();

        if bh.reached_merge_point {
            // blackhole.py:1068 ContinueRunningNormally parity:
            // RPython raises ContinueRunningNormally(*args) at
            // jit_merge_point, carrying live values to warmspot.py
            // which calls portal_runner. pyre writes live values
            // back to the frame and returns ContinueRunningNormally.
            // RPython blackhole.py:1068 — virtualizable_ptr IS the frame.
            let frame_ptr = bh.virtualizable_ptr as *mut PyFrame;
            if !frame_ptr.is_null() {
                let frame = unsafe { &mut *frame_ptr };
                let code = unsafe { &*frame.code };
                let nlocals = code.varnames.len();
                // blackhole.py:1068 parity: only live values are carried
                // by ContinueRunningNormally. Use liveness at merge_py_pc
                // to determine which locals to write back. Dead locals
                // keep their existing frame values (they will be
                // reassigned before next use by definition of liveness).
                let live = pyre_jit_trace::state::liveness_for(code);
                for i in 0..nlocals {
                    if i < bh.registers_r.len()
                        && i < frame.locals_cells_stack_w.len()
                        && live.is_local_live(merge_py_pc, i)
                    {
                        frame.locals_cells_stack_w[i] =
                            bh.registers_r[i] as pyre_object::PyObjectRef;
                    }
                }
                let ncells = pyre_interpreter::pyframe::ncells(code);
                let stack_base = nlocals + ncells;
                let stack = bh.runtime_stack_drain(0);
                frame.valuestackdepth = stack_base + stack.len();
                for (i, val) in stack.iter().enumerate() {
                    let idx = stack_base + i;
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
        if bh.aborted {
            if majit_metainterp::majit_log_enabled() {
                eprintln!(
                    "[jit][blackhole] ABORT at jitcode_pc={} last_opcode_pos={}",
                    bh.position, bh.last_opcode_position
                );
            }
            builder.release_interp(bh);
            return BlackholeResult::Failed;
        }

        if bh.got_exception {
            if majit_metainterp::majit_log_enabled() {
                let jitcode_bytes = &bh.jitcode.code;
                let opcode_at = if bh.last_opcode_position < jitcode_bytes.len() {
                    jitcode_bytes[bh.last_opcode_position]
                } else {
                    255
                };
                // Try to get fn_ptr info for call opcodes
                let fn_info = if bh.last_opcode_position + 2 < jitcode_bytes.len() {
                    let fn_idx = u16::from_le_bytes([
                        jitcode_bytes[bh.last_opcode_position + 1],
                        jitcode_bytes[bh.last_opcode_position + 2],
                    ]) as usize;
                    if fn_idx < bh.jitcode.fn_ptrs.len() {
                        format!(
                            "fn_ptr={:#x}",
                            bh.jitcode.fn_ptrs[fn_idx].concrete_ptr as usize
                        )
                    } else {
                        format!("fn_idx={} (out of range)", fn_idx)
                    }
                } else {
                    String::new()
                };
                eprintln!(
                    "[jit][blackhole] EXCEPTION at jitcode_pc={} last_opcode_pos={} opcode={} {}",
                    bh.position, bh.last_opcode_position, opcode_at, fn_info
                );
            }
        }

        // blackhole.py:1752 _run_forever exception propagation:
        // Exception not handled in this frame (no handler found by
        // dispatch_one's handle_exception_in_frame). Propagate to caller.
        if bh.got_exception {
            let exc_value = bh.exception_last_value;
            let next = bh.nextblackholeinterp.take();
            builder.release_interp(bh);

            let Some(mut caller_bh) = next.map(|b| *b) else {
                // blackhole.py:1752 _run_forever parity: exception in topmost
                // frame with no caller → propagate to JIT driver.
                // TODO: currently returns Failed because the blackhole may be
                // executing on corrupt state (resume data bug). Once blackhole
                // resume is sound, this should return DoneWithThisFrame(Err).
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

        // blackhole.py:1636-1640: get_tmpreg_r() for ref return.
        let return_value = bh.get_tmpreg_r();
        let next = bh.nextblackholeinterp.take();
        builder.release_interp(bh);

        let Some(mut caller_bh) = next.map(|b| *b) else {
            // blackhole.py:1664-1672 _done_with_this_frame
            return BlackholeResult::DoneWithThisFrame(
                Ok(return_value as pyre_object::PyObjectRef),
            );
        };

        // blackhole.py:1657-1659 _setup_return_value_r
        caller_bh.setup_return_value_r(return_value);

        bh = caller_bh;
    }
}

/// resume.py:945-956 decode_ref / getvirtual_ptr parity.
///
/// Re-box optimizer-unboxed values back to PyObjectRef for the
/// blackhole's ref register file. RPython's decode_ref dispatches
/// on TAGVIRTUAL/TAGCONST/TAGBOX/TAGSMALLINT; pyre's deadframe
/// already contains typed Values, so we just box Int/Float to
/// W_IntObject/W_FloatObject.
fn materialize_virtual(val: &majit_ir::Value) -> i64 {
    use majit_ir::Value;
    match val {
        Value::Ref(r) => r.as_usize() as i64,
        Value::Int(v) => pyre_object::intobject::w_int_new(*v) as i64,
        Value::Float(v) => pyre_object::floatobject::w_float_new(*v) as i64,
        Value::Void => 0i64,
    }
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
    // jitcode_pc_for_loop_header handles Cache-skip offset mismatch
    // between the interpreter's merge_py_pc and the codewriter's label.
    let Some(merge_jitcode_pc) =
        crate::jit::codewriter::jitcode_pc_for_loop_header(&pyjitcode.pc_map, merge_py_pc)
    else {
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

    // Direct PyFrame loading (not resume data — no enumerate_vars dispatch).
    // frame.locals_cells_stack_w is always PyObjectRef → write_a_ref only.
    let nlocals = code.varnames.len();
    for i in 0..nlocals {
        if i < frame.locals_cells_stack_w.len() {
            bh.setarg_r(i, frame.locals_cells_stack_w[i] as i64);
        }
    }
    // locals_cells_stack_w: [locals..., cells..., stack...]
    let ncells = pyre_interpreter::pyframe::ncells(code);
    let stack_base = nlocals + ncells;
    let vsd = frame.valuestackdepth;
    for i in stack_base..vsd {
        if i < frame.locals_cells_stack_w.len() {
            bh.runtime_stack_push(0, frame.locals_cells_stack_w[i] as i64);
        }
    }
    // blackhole.py bhimpl_getfield_vable_*: set virtualizable pointer.
    bh.virtualizable_ptr = frame as *mut PyFrame as i64;
    bh.virtualizable_info = crate::eval::get_virtualizable_info();

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
    // Locals are in ref register bank (codewriter uses move_r).
    for i in 0..nlocals {
        if i < bh.registers_r.len() && i < frame.locals_cells_stack_w.len() {
            frame.locals_cells_stack_w[i] = bh.registers_r[i] as pyre_object::PyObjectRef;
        }
    }
    // Write back runtime stack → frame value stack.
    // stack starts at nlocals + ncells in locals_cells_stack_w.
    let ncells = pyre_interpreter::pyframe::ncells(code);
    let stack_base = nlocals + ncells;
    let stack = bh.runtime_stack_drain(0);
    frame.valuestackdepth = stack_base + stack.len();
    for (i, val) in stack.iter().enumerate() {
        let idx = stack_base + i;
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
        #[cfg(not(target_arch = "wasm32"))]
        {
            majit_backend_cranelift::register_call_assembler_force(jit_force_callee_frame);
            majit_backend_cranelift::register_call_assembler_bridge(jit_ca_handle_guard_failure);
            majit_backend_cranelift::register_call_assembler_blackhole(
                jit_blackhole_resume_from_guard,
            );
            majit_backend_cranelift::register_inline_frame_arena(arena_global_info());
        }
    });
}

/// compile.py:701-716 handle_fail → resume_in_blackhole parity.
///
/// RPython: guard failure always resumes via jitcode-level blackhole
/// (blackhole_from_resumedata → _run_forever). There is no IR-level
/// blackhole in RPython.
///
/// When rd_numb is available, uses ResumeDataDirectReader for exact
/// frame decoding (resume.py:1312 parity).
fn jit_blackhole_resume_from_guard(
    green_key: u64,
    trace_id: u64,
    fail_index: u32,
    fail_values_ptr: *const i64,
    num_fail_values: usize,
    raw_deadframe_ptr: *const i64,
    num_raw_deadframe: usize,
) -> Option<i64> {
    use majit_ir::{Type, Value};

    if fail_values_ptr.is_null() || num_fail_values == 0 {
        return None;
    }
    let fail_values = unsafe { std::slice::from_raw_parts(fail_values_ptr, num_fail_values) };
    let raw_deadframe = if !raw_deadframe_ptr.is_null() && num_raw_deadframe > 0 {
        unsafe { std::slice::from_raw_parts(raw_deadframe_ptr, num_raw_deadframe) }
    } else {
        fail_values
    };
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

    if majit_metainterp::majit_log_enabled() {
        eprintln!(
            "[blackhole-resume] gk={} trace={} fail_idx={} nvals={}",
            actual_green_key, trace_id, fail_index, num_fail_values,
        );
    }

    // --- Path 1: rd_numb-based resume (resume.py:1312 exact parity) ---
    // When rd_numb is present, use ResumeDataDirectReader to decode
    // frame sections precisely, matching RPython blackhole_from_resumedata.
    if let Some((rd_numb, rd_consts_typed)) =
        driver.get_rd_numb(actual_green_key, trace_id, fail_index)
    {
        let rd_consts: Vec<i64> = rd_consts_typed.iter().map(|(v, _)| *v).collect();
        if majit_metainterp::majit_log_enabled() {
            eprintln!(
                "[blackhole-resume] rd_numb len={} rd_consts len={} raw_deadframe len={}",
                rd_numb.len(),
                rd_consts.len(),
                raw_deadframe.len(),
            );
        }
        // resume.py:924-926 _prepare: get rd_virtuals for TAGVIRTUAL materialization.
        let rd_virtuals = driver.get_rd_virtuals(actual_green_key, trace_id, fail_index);
        let rd_virtuals_ref = rd_virtuals.as_deref();
        // resume.py parity: deadframe_types tells decode_ref() whether a
        // TAGBOX slot holds a raw int (needs boxing) or a GcRef (use as-is).
        // Without this, unboxed ints are treated as pointers → SIGSEGV.
        let deadframe_types =
            driver.get_recovery_slot_types(actual_green_key, trace_id, fail_index);
        let result = blackhole_resume_via_rd_numb(
            &rd_numb,
            &rd_consts,
            raw_deadframe,
            None,
            rd_virtuals_ref,
            deadframe_types.as_deref(),
        );
        return handle_blackhole_result(result, fail_values);
    }

    // RPython: every guard has rd_numb from capture_resumedata +
    // store_final_boxes_in_guard. No heuristic fallback path.
    None
}

/// resume.py:1312 blackhole_from_resumedata parity:
/// Decode rd_numb via ResumeDataDirectReader, build blackhole chain,
/// run _run_forever.
pub fn blackhole_resume_via_rd_numb(
    rd_numb: &[u8],
    rd_consts: &[i64],
    deadframe: &[i64],
    rd_guard_pendingfields: Option<&[majit_ir::GuardPendingFieldEntry]>,
    rd_virtuals: Option<&[majit_ir::RdVirtualInfo]>,
    deadframe_types: Option<&[majit_ir::Type]>,
) -> BlackholeResult {
    use majit_metainterp::resume;

    // Thread-local BH pool (RPython BlackholeInterpBuilder).
    thread_local! {
        static BH_BUILDER_RD: std::cell::UnsafeCell<majit_metainterp::blackhole::BlackholeInterpBuilder> =
            std::cell::UnsafeCell::new(majit_metainterp::blackhole::BlackholeInterpBuilder::new());
    }
    let builder = BH_BUILDER_RD.with(|cell| unsafe { &mut *cell.get() });

    // resume.py:1339 jitcodes[jitcode_pos]: resolve jitcode_index + pc
    // to a pyre JitCode via CodeWriter.
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
    // resume.py:1339 jitcodes[jitcode_pos]:
    // jitcode_index is a sequential index into MetaInterpStaticData.jitcodes.
    // Resolve to CodeObject via code_for_jitcode_index, then compile jitcode.
    let resolve_jitcode =
        |jitcode_index: i32, pc: i32| -> Option<(majit_metainterp::jitcode::JitCode, usize)> {
            if pc < 0 {
                return None;
            }
            let code_ptr = pyre_jit_trace::state::code_for_jitcode_index(jitcode_index)?;
            if code_ptr.is_null() {
                return None;
            }
            let code = unsafe { &*code_ptr };
            let pyjitcode = crate::jit::codewriter::get_jitcode(code, &writer);
            if pyjitcode.has_abort_opcode() {
                return None;
            }
            // Convert Python PC → JitCode byte offset via pc_map.
            let jitcode_pc = pyjitcode.pc_map.get(pc as usize).copied().unwrap_or(0);
            Some((pyjitcode.jitcode.clone(), jitcode_pc))
        };

    // resume.py:983-991 _prepare_virtuals: convert RdVirtualInfo → VirtualInfo
    // for lazy materialization in getvirtual_ptr/getvirtual_int.
    let count = deadframe.len() as i32;
    let rd_virtuals_converted: Option<Vec<resume::VirtualInfo>> = rd_virtuals.map(|rd_virts| {
        rd_virts
            .iter()
            .map(|rd| resume::rd_virtual_to_virtual_info(rd, rd_consts, count))
            .collect()
    });
    let rd_virtuals_slice = rd_virtuals_converted.as_deref();

    // resume.py:1312-1343 blackhole_from_resumedata:
    // ResumeDataDirectReader decodes rd_numb, builds BH chain.
    // compile.py:990: vinfo = self.jitdriver_sd.virtualizable_info
    let vinfo = pyre_jit_trace::virtualizable_gen::build_virtualizable_info();
    // resume.py:1314: vrefinfo = metainterp_sd.virtualref_info
    // resume.py:1316: ginfo = jitdriver_sd.greenfield_info
    let allocator = crate::eval::PyreBlackholeAllocator;
    let bh = resume::blackhole_from_resumedata(
        builder,
        &resolve_jitcode,
        rd_numb,
        rd_consts,
        deadframe,
        deadframe_types,        // deadframe_types: decode_ref boxes TAGBOX ints
        rd_virtuals_slice,      // rd_virtuals
        None,                   // rd_pendingfields
        rd_guard_pendingfields, // rd_guard_pendingfields
        None,                   // vrefinfo — pyre has no virtualref mechanism
        Some(&vinfo as &dyn resume::VirtualizableInfo),
        None, // ginfo — pyre has no greenfield mechanism
        &allocator,
    );

    let Some((mut bh, virtualizable_ptr)) = bh else {
        return BlackholeResult::Failed;
    };

    // resume.py:1404: virtualizable_ptr was read by consume_vable_info
    // from the vable section. Set on the blackhole for vable bytecodes.
    if virtualizable_ptr != 0 {
        bh.virtualizable_ptr = virtualizable_ptr;
    } else if !deadframe.is_empty() {
        // Fallback for guards without vable section.
        bh.virtualizable_ptr = deadframe[0];
    }
    bh.virtualizable_info = crate::eval::get_virtualizable_info();

    if majit_metainterp::majit_log_enabled() {
        eprintln!("[blackhole-resume] rd_numb path, chain built, running _run_forever",);
    }

    // blackhole.py:1752 _run_forever parity.
    loop {
        bh.run();

        if bh.reached_merge_point {
            builder.release_interp(bh);
            return BlackholeResult::ContinueRunningNormally;
        }
        if bh.aborted {
            builder.release_interp(bh);
            return BlackholeResult::Failed;
        }
        if bh.got_exception {
            let exc_value = bh.exception_last_value;
            let next = bh.nextblackholeinterp.take();
            builder.release_interp(bh);
            let Some(mut caller_bh) = next.map(|b| *b) else {
                return BlackholeResult::DoneWithThisFrame(Err(pyre_interpreter::PyError::new(
                    pyre_interpreter::PyErrorKind::RuntimeError,
                    "blackhole exception",
                )));
            };
            caller_bh.last_opcode_position = caller_bh.position;
            if caller_bh.handle_exception_in_frame(exc_value) {
                bh = caller_bh;
                continue;
            }
            caller_bh.exception_last_value = exc_value;
            caller_bh.got_exception = true;
            bh = caller_bh;
            continue;
        }

        // blackhole.py:1632-1644: pass return value to caller by _return_type.
        use majit_metainterp::blackhole::BhReturnType;
        let rt = bh.return_type;
        let next = bh.nextblackholeinterp.take();
        let caller = next.map(|b| *b);
        if caller.is_none() {
            // blackhole.py:1664-1677 _done_with_this_frame
            let result = match rt {
                BhReturnType::Int => pyre_object::intobject::w_int_new(bh.get_tmpreg_i()),
                BhReturnType::Ref => bh.get_tmpreg_r() as pyre_object::PyObjectRef,
                BhReturnType::Float => {
                    // blackhole.py:1674-1675: DoneWithThisFrameFloat(get_tmpreg_f())
                    // pyre: box float bits into W_FloatObject
                    let bits = bh.get_tmpreg_f() as u64;
                    pyre_object::floatobject::w_float_new(f64::from_bits(bits))
                }
                BhReturnType::Void => pyre_object::PY_NULL,
            };
            builder.release_interp(bh);
            return BlackholeResult::DoneWithThisFrame(Ok(result));
        }
        let mut caller_bh = caller.unwrap();
        // blackhole.py:1637-1644: dispatch by _return_type
        match rt {
            BhReturnType::Int => caller_bh.setup_return_value_i(bh.get_tmpreg_i()),
            BhReturnType::Ref => caller_bh.setup_return_value_r(bh.get_tmpreg_r()),
            BhReturnType::Float => caller_bh.setup_return_value_f(bh.get_tmpreg_f()),
            BhReturnType::Void => {}
        }
        builder.release_interp(bh);
        bh = caller_bh;
    }
}

/// Common result handling for both rd_numb and heuristic paths.
fn handle_blackhole_result(bh_result: BlackholeResult, fail_values: &[i64]) -> Option<i64> {
    match bh_result {
        BlackholeResult::DoneWithThisFrame(Ok(result)) => {
            // blackhole.py:1673 / warmspot.py:986: DoneWithThisFrameRef
            // always carries a GCREF. No heuristic detection needed.
            let raw = if !result.is_null() && unsafe { is_int(result) } {
                unsafe { w_int_get_value(result) }
            } else {
                result as i64
            };
            if majit_metainterp::majit_log_enabled() {
                eprintln!("[blackhole-resume] DoneWithThisFrame result={}", raw);
            }
            Some(raw)
        }
        BlackholeResult::DoneWithThisFrame(Err(_)) => {
            if majit_metainterp::majit_log_enabled() {
                eprintln!("[blackhole-resume] DoneWithThisFrame exception");
            }
            None
        }
        BlackholeResult::ContinueRunningNormally => {
            // warmspot.py:970-983 handle_jitexception:
            // ContinueRunningNormally → call portal_ptr(*args) directly.
            // RPython calls the interpreter (not JIT) to avoid re-entering
            // compiled code that would fail at the same guard again.
            // force_plain_eval ensures nested function calls also use the
            // interpreter, not the JIT (prevents recursive guard failures).
            let callee_frame_ptr = fail_values[0] as *mut PyFrame;
            if callee_frame_ptr.is_null() {
                return None;
            }
            let callee_frame = unsafe { &mut *callee_frame_ptr };
            if majit_metainterp::majit_log_enabled() {
                eprintln!(
                    "[blackhole-resume] ContinueRunningNormally, interpreter from pc={}",
                    callee_frame.next_instr,
                );
            }
            let _plain_guard = pyre_interpreter::call::force_plain_eval();
            match pyre_interpreter::eval::eval_frame_plain(callee_frame) {
                Ok(result) => {
                    let raw = if !result.is_null() && unsafe { is_int(result) } {
                        unsafe { w_int_get_value(result) }
                    } else {
                        result as i64
                    };
                    Some(raw)
                }
                Err(_) => None,
            }
        }
        BlackholeResult::Failed => {
            if majit_metainterp::majit_log_enabled() {
                eprintln!("[blackhole-resume] Failed");
            }
            None
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
pub fn trace_and_compile_from_bridge(
    green_key: u64,
    trace_id: u64,
    fail_index: u32,
    frame: &mut PyFrame,
    raw_values: &[i64],
    exit_layout: &majit_metainterp::CompiledExitLayout,
) -> bool {
    use crate::eval::build_jit_state;
    use crate::jit::state::PyreEnv;
    use crate::jit::trace::trace_bytecode;

    let (driver, info) = crate::eval::driver_pair();

    // pyjitpl.py:2890-2911 handle_guard_failure parity:
    // RPython creates a fresh MetaInterp and calls
    // initialize_state_from_guard_failure(resumedescr, deadframe)
    // which internally calls rebuild_from_resumedata (resume.py:1042).
    // This restores the complete frame stack INSIDE the bridge function.
    let meta = driver.meta_interp().get_compiled_meta(green_key).cloned();
    let mut jit_state_local = build_jit_state(frame, info);
    let resume_pc = if let Some(ref meta) = meta {
        if let Some((_, pc)) = crate::eval::decode_and_restore_guard_failure(
            &mut jit_state_local,
            meta,
            raw_values,
            exit_layout,
        ) {
            pc
        } else {
            0
        }
    } else {
        0
    };
    if resume_pc == 0 {
        return false;
    }
    frame.next_instr = resume_pc;
    let code = unsafe { &*frame.code };
    let env = PyreEnv;
    let mut jit_state = build_jit_state(frame, info);

    // NOTE: guard resume_pc pointing to LOAD_CONST + RETURN_VALUE does NOT
    // mean the guard is a loop-exit guard. It means the blackhole resume
    // path leads to function return. RPython handles this correctly via
    // blackhole resume → interpreter runs remaining code → natural return.
    // Direct FINISH bridges are WRONG here — they skip the remaining loop
    // body that the blackhole should execute.
    // RPython rebuild_from_resumedata (pyjitpl.py:2901,3400)
    // restores the complete frame stack before bridge tracing.
    // Bridge tracing sees the full frame layout — no truncation.
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
        #[cfg(not(target_arch = "wasm32"))]
        let exc_class = majit_backend_cranelift::jit_exc_class_raw();
        #[cfg(target_arch = "wasm32")]
        let exc_class: i64 = 0;
        #[cfg(not(target_arch = "wasm32"))]
        let exc_value = majit_backend_cranelift::jit_exc_value_raw();
        #[cfg(target_arch = "wasm32")]
        let exc_value: i64 = 0;
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

/// compile.py:701-717 handle_fail for call_assembler guard failures.
/// Checks must_compile (jitcounter.tick), and if threshold reached,
/// traces the alternate path via trace_and_compile_from_bridge.
fn jit_ca_handle_guard_failure(
    green_key: u64,
    trace_id: u64,
    fail_index: u32,
    raw_values_ptr: *const i64,
    num_values: usize,
    status: u64,
    descr_addr: usize,
) -> bool {
    if raw_values_ptr.is_null() || num_values == 0 {
        return false;
    }
    let raw_values = unsafe { std::slice::from_raw_parts(raw_values_ptr, num_values) };

    // compile.py:738-784 must_compile: jitcounter.tick(guard_hash, increment)
    let (should_bridge, owning_key) = {
        let (driver, _) = crate::eval::driver_pair();
        driver.meta_interp_mut().must_compile_with_values(
            green_key, trace_id, fail_index, raw_values, status, descr_addr,
        )
    };
    if !should_bridge {
        return false;
    }

    if majit_metainterp::majit_log_enabled() {
        eprintln!(
            "[jit][ca-bridge] must_compile fired: key={} trace={} fail={}",
            green_key, trace_id, fail_index,
        );
    }

    // compile.py:719-726: get exit_layout from the compiled trace.
    let exit_layout = {
        let (driver, _) = crate::eval::driver_pair();
        driver
            .meta_interp()
            .get_compiled_exit_layout_in_trace(green_key, trace_id, fail_index)
    };
    let Some(exit_layout) = exit_layout else {
        return false;
    };

    // Obtain callee frame from deadframe vable header.
    // pyre vable_boxes = [frame, ni, vsd, locals..., stack...],
    // so raw_values[0] is the callee's PyFrame pointer.
    let frame_ptr = raw_values[0] as *mut PyFrame;
    if frame_ptr.is_null() {
        return false;
    }
    let frame = unsafe { &mut *frame_ptr };

    // compile.py:786-788 start_compiling: set ST_BUSY_FLAG
    {
        let (driver, _) = crate::eval::driver_pair();
        driver
            .meta_interp_mut()
            .start_guard_compiling(owning_key, trace_id, fail_index);
    }

    // compile.py:706-708 _trace_and_compile_from_bridge
    let compiled = trace_and_compile_from_bridge(
        owning_key,
        trace_id,
        fail_index,
        frame,
        raw_values,
        &exit_layout,
    );

    // compile.py:790-795 done_compiling: clear ST_BUSY_FLAG
    {
        let (driver, _) = crate::eval::driver_pair();
        driver
            .meta_interp_mut()
            .done_guard_compiling(owning_key, trace_id, fail_index);
    }

    if majit_metainterp::majit_log_enabled() {
        eprintln!(
            "[jit][ca-bridge] compiled={} key={} trace={} fail={}",
            compiled, green_key, trace_id, fail_index,
        );
    }

    compiled
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
    let func_code = code_ptr as *const pyre_interpreter::CodeObject;

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
    let func_code = code_ptr as *const pyre_interpreter::CodeObject;

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
/// bhimpl_residual_call: parent frame via BH_VABLE_PTR.
/// call_int_function in machine.rs transmutes to the correct arity.
pub extern "C" fn bh_call_fn(callable: i64, arg0: i64) -> i64 {
    bh_call_fn_impl(callable as PyObjectRef, &[arg0 as PyObjectRef])
}

pub extern "C" fn bh_call_fn_0(callable: i64) -> i64 {
    bh_call_fn_impl(callable as PyObjectRef, &[])
}

pub extern "C" fn bh_call_fn_2(callable: i64, arg0: i64, arg1: i64) -> i64 {
    bh_call_fn_impl(
        callable as PyObjectRef,
        &[arg0 as PyObjectRef, arg1 as PyObjectRef],
    )
}

pub extern "C" fn bh_call_fn_3(callable: i64, a0: i64, a1: i64, a2: i64) -> i64 {
    bh_call_fn_impl(
        callable as PyObjectRef,
        &[a0 as PyObjectRef, a1 as PyObjectRef, a2 as PyObjectRef],
    )
}

pub extern "C" fn bh_call_fn_4(callable: i64, a0: i64, a1: i64, a2: i64, a3: i64) -> i64 {
    bh_call_fn_impl(
        callable as PyObjectRef,
        &[
            a0 as PyObjectRef,
            a1 as PyObjectRef,
            a2 as PyObjectRef,
            a3 as PyObjectRef,
        ],
    )
}

pub extern "C" fn bh_call_fn_5(callable: i64, a0: i64, a1: i64, a2: i64, a3: i64, a4: i64) -> i64 {
    bh_call_fn_impl(
        callable as PyObjectRef,
        &[
            a0 as PyObjectRef,
            a1 as PyObjectRef,
            a2 as PyObjectRef,
            a3 as PyObjectRef,
            a4 as PyObjectRef,
        ],
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
    )
}

/// blackhole.py bhimpl_residual_call: parent frame from BH_VABLE_PTR.
/// RPython: bhimpl_residual_call dispatches via cpu.bh_call_*.
/// pyre: dispatches builtin vs user function. Parent frame provides
/// full call protocol (defaults, kw-only, varargs, generator).
fn bh_call_fn_impl(callable: PyObjectRef, args: &[PyObjectRef]) -> i64 {
    // RPython bhimpl_residual_call parity: the direct call below uses
    // call_user_function_plain, but the CALLED function's own recursive
    // calls go through call_callable → call_user_function which checks
    // EVAL_OVERRIDE. Without force_plain_eval, those nested calls would
    // re-enter JIT compiled code, causing nested blackhole resume.
    let _plain_guard = pyre_interpreter::call::force_plain_eval();
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
    // blackhole.py bhimpl_residual_call: parent frame from BH_VABLE_PTR.
    // call_user_function_plain handles full call protocol (defaults,
    // kw-only, pack_varargs, generator/coroutine).
    let parent_frame_ptr =
        majit_metainterp::blackhole::BH_VABLE_PTR.with(|c| c.get()) as *const PyFrame;
    let parent_frame = unsafe { &*parent_frame_ptr };
    match pyre_interpreter::call::call_user_function_plain(parent_frame, callable, args) {
        Ok(result) => result as i64,
        Err(err) => {
            let exc_obj = err.to_exc_object();
            majit_metainterp::blackhole::BH_LAST_EXC_VALUE.with(|c| c.set(exc_obj as i64));
            0
        }
    }
}

/// jtransform.py parity: namespace and code come from getfield_vable_r.
/// namespace = getfield_vable_r(frame, w_globals), code = getfield_vable_r(frame, pycode).
/// namei is the raw oparg from LOAD_GLOBAL: name_idx = namei >> 1.
pub extern "C" fn bh_load_global_fn(namespace_ptr: i64, code_ptr: i64, namei: i64) -> i64 {
    let code = unsafe { &*(code_ptr as *const pyre_interpreter::CodeObject) };
    let raw = namei as usize;
    let idx = raw >> 1;

    if idx >= code.names.len() {
        return 0;
    }

    let name = code.names[idx].as_ref();
    let ns = unsafe { &*(namespace_ptr as *const pyre_interpreter::PyNamespace) };
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

/// Load a constant from the code object.
/// jtransform.py parity: code comes from getfield_vable_r(frame, pycode).
pub extern "C" fn bh_load_const_fn(code_ptr: i64, consti: i64) -> i64 {
    let code = unsafe { &*(code_ptr as *const pyre_interpreter::CodeObject) };
    pyre_interpreter::pyframe::load_const_from_code(code, consti as usize) as i64
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

    // op_code 10 = CHECK_EXC_MATCH isinstance check (from codewriter CheckExcMatch).
    // lhs = exception value, rhs = exception type to match.
    if op_code == 10 {
        let matched = unsafe {
            if !pyre_object::is_exception(lhs) {
                true
            } else {
                let kind = pyre_object::w_exception_get_kind(lhs);
                if pyre_object::is_str(rhs) {
                    let type_name = pyre_object::w_str_get_value(rhs);
                    pyre_object::exc_kind_matches(kind, type_name)
                } else if pyre_interpreter::is_builtin_code(rhs) {
                    let type_name = pyre_interpreter::builtin_code_name(rhs);
                    pyre_object::exc_kind_matches(kind, type_name)
                } else {
                    true
                }
            }
        };
        return pyre_object::w_bool_from(matched) as i64;
    }

    // op_code is the compact tag from compare_op_tag (0-5), NOT the raw
    // ComparisonOperator discriminant. Reverse the mapping to get the enum.
    let Some(op) = pyre_interpreter::runtime_ops::compare_op_from_tag(op_code) else {
        let err = pyre_interpreter::PyError::new(
            pyre_interpreter::PyErrorKind::TypeError,
            format!("unknown compare op tag {op_code}"),
        );
        majit_metainterp::blackhole::BH_LAST_EXC_VALUE.with(|c| c.set(err.to_exc_object() as i64));
        return 0;
    };
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

    // op_code is the compact tag from binary_op_tag (0-12), NOT the raw
    // BinaryOperator discriminant. Reverse the mapping to get the enum.
    let Some(op) = pyre_interpreter::runtime_ops::binary_op_from_tag(op_code) else {
        let err = pyre_interpreter::PyError::new(
            pyre_interpreter::PyErrorKind::TypeError,
            format!("unknown binary op tag {op_code}"),
        );
        majit_metainterp::blackhole::BH_LAST_EXC_VALUE.with(|c| c.set(err.to_exc_object() as i64));
        return 0;
    };
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
