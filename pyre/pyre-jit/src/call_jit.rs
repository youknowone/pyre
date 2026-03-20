//! JIT-specific call infrastructure — force/bridge callbacks, callee
//! frame creation helpers, frame pool.
//!
//! Separated from pyre-interp/src/call.rs so pyre-interp stays JIT-free.

use std::cell::UnsafeCell;
use std::mem::MaybeUninit;
use std::sync::Once;

use pyre_bytecode::bytecode::{Instruction, OpArgState};
use pyre_object::intobject::w_int_get_value;
use pyre_object::intobject::w_int_new;
use pyre_object::pyobject::is_int;
use pyre_object::{PY_NULL, PyObjectRef};
use pyre_runtime::{
    PyResult, is_func, register_jit_function_caller, w_func_get_closure, w_func_get_code_ptr,
    w_func_get_globals, w_func_get_name,
};

use pyre_interp::frame::PyFrame;

// RPython/PyPy's call_assembler path does not memoize recursive helper
// results in a side cache. Keep the plumbing sites in place, but make the
// cache a no-op so boxed results do not escape the GC's root graph through
// ad-hoc global storage.
const FORCE_CACHE_SIZE: usize = 1;

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

#[inline]
fn force_cache_index(code_key: usize, arg_key: usize) -> usize {
    (code_key.wrapping_mul(2654435761) ^ arg_key) % FORCE_CACHE_SIZE
}

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
pub(crate) fn recursive_force_cache_safe(callable: PyObjectRef) -> bool {
    unsafe {
        if !w_func_get_closure(callable).is_null() {
            return false;
        }
        let code = &*(w_func_get_code_ptr(callable) as *const pyre_bytecode::CodeObject);
        let func_name = w_func_get_name(callable);
        let mut arg_state = OpArgState::default();

        for code_unit in code.instructions.iter().copied() {
            let (instruction, op_arg) = arg_state.get(code_unit);
            match instruction {
                Instruction::LoadName { namei } => {
                    let idx = namei.get(op_arg) as usize;
                    if code.names[idx].as_str() != func_name {
                        return false;
                    }
                }
                Instruction::LoadGlobal { namei } => {
                    let raw = namei.get(op_arg) as usize;
                    let name_idx = raw >> 1;
                    if code.names[name_idx].as_str() != func_name {
                        return false;
                    }
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
            is_func(value)
                && w_func_get_code_ptr(value) == code_ptr
                && w_func_get_globals(value) == namespace_ptr
                && w_func_get_closure(value).is_null()
        };
        if is_candidate && recursive_force_cache_safe(value) {
            return true;
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
    if args.len() != 1 || !unsafe { is_func(callable) } {
        return None;
    }
    let arg0 = args[0];
    if arg0.is_null() || !unsafe { is_int(arg0) } {
        return None;
    }
    if !recursive_force_cache_safe(callable) {
        return None;
    }
    let callable_globals = unsafe { w_func_get_globals(callable) };
    if callable_globals != frame.namespace || !unsafe { w_func_get_closure(callable) }.is_null() {
        return None;
    }

    let raw_arg = unsafe { w_int_get_value(arg0) };
    let code_ptr = unsafe { w_func_get_code_ptr(callable) };
    let green_key = crate::eval::make_green_key(code_ptr as *const _, 0);
    // Inline concrete execution sometimes falls back to a helper-boundary
    // recursive call instead of a real frame switch. That helper must run as
    // plain concrete execution: if it reuses the outer inline override or the
    // outer trace's merge-point path, it will keep tracing against the wrong
    // symbolic frame and corrupt concrete stack reads.
    let _suspend_inline_override = pyre_interp::call::suspend_inline_call_override();
    let _suspend_inline_result = pyre_interp::call::suspend_inline_handled_result();
    let _jit_depth = crate::eval::jit_call_depth_bump();
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
        let protocol = if driver.has_raw_int_finish(green_key) {
            FinishProtocol::RawInt
        } else {
            FinishProtocol::Boxed
        };
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
        let protocol = if driver.has_raw_int_finish(green_key) {
            FinishProtocol::RawInt
        } else {
            FinishProtocol::Boxed
        };
        let token_num = driver.get_loop_token(green_key).map(|token| token.number);
        if token_num.is_some() {
            *slot = Some((green_key, protocol, token_num));
        }
        (protocol, token_num)
    })
}

#[inline]
fn force_cache_lookup(
    _protocol: FinishProtocol,
    _hash_idx: usize,
    _code_key: usize,
    _arg_key: usize,
) -> Option<i64> {
    None
}

#[inline]
fn force_cache_store(
    _protocol: FinishProtocol,
    _hash_idx: usize,
    _code_key: usize,
    _arg_key: usize,
    _value: i64,
) {
}

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
    let _suspend_inline_override = pyre_interp::call::suspend_inline_call_override();
    let _jit_depth = crate::eval::jit_call_depth_bump();
    match pyre_interp::call::call_user_function(frame, callable as PyObjectRef, args) {
        Ok(result) => result as i64,
        Err(err) => panic!("jit user-function call failed: {err}"),
    }
}

pub extern "C" fn jit_force_callee_frame(frame_ptr: i64) -> i64 {
    let _suspend_inline_result = pyre_interp::call::suspend_inline_handled_result();
    let frame = unsafe { &mut *(frame_ptr as *mut PyFrame) };

    let code_key = frame.code as usize;
    let green_key = crate::eval::make_green_key(frame.code, frame.next_instr);
    let protocol = finish_protocol(green_key);
    let arg_key = if frame.locals_cells_stack_w.len() > 0 {
        force_cache_arg_key(frame.locals_cells_stack_w[0])
    } else {
        0
    };

    // Hash-based force cache lookup (64 entries)
    let hash_idx = force_cache_index(code_key, arg_key);
    if let Some(cached) = force_cache_lookup(protocol, hash_idx, code_key, arg_key) {
        return cached;
    }

    if majit_meta::majit_log_enabled() {
        let arg0 = if frame.locals_cells_stack_w.len() > 0
            && !frame.locals_cells_stack_w[0].is_null()
            && unsafe { is_int(frame.locals_cells_stack_w[0]) }
        {
            Some(unsafe { w_int_get_value(frame.locals_cells_stack_w[0]) })
        } else {
            None
        };
        eprintln!(
            "[jit][force-boxed] enter key={} ni={} vsd={} arg0={:?} raw_finish={}",
            green_key,
            frame.next_instr,
            frame.valuestackdepth,
            arg0,
            matches!(protocol, FinishProtocol::RawInt)
        );
    }

    // RPython parity: assembler_call_helper (warmspot.py:1021) calls
    // fail_descr.handle_fail() → resume_in_blackhole(). It does NOT
    // try to re-execute compiled code. Always use the blackhole.
    //
    // Bump recursive_force_entry to prevent JIT re-entry from any
    // interpreter code called during blackhole execution.
    let _force_guard = crate::eval::recursive_force_entry_bump();
    let result = resume_in_blackhole(frame);
    let value = match protocol {
        FinishProtocol::RawInt if !result.is_null() && unsafe { is_int(result) } => {
            unsafe { w_int_get_value(result) }
        }
        FinishProtocol::RawInt => result as i64,
        FinishProtocol::Boxed => result as i64,
    };
    force_cache_store(protocol, hash_idx, code_key, arg_key, value);
    match protocol {
        FinishProtocol::RawInt => w_int_new(value) as i64,
        FinishProtocol::Boxed => value,
    }
}

fn jit_force_callee_frame_raw(frame_ptr: i64) -> i64 {
    let _suspend_inline_result = pyre_interp::call::suspend_inline_handled_result();
    let frame = unsafe { &mut *(frame_ptr as *mut PyFrame) };

    let code_key = frame.code as usize;
    let green_key = crate::eval::make_green_key(frame.code, frame.next_instr);
    let protocol = finish_protocol(green_key);
    let arg_key = if frame.locals_cells_stack_w.len() > 0 {
        force_cache_arg_key(frame.locals_cells_stack_w[0])
    } else {
        0
    };

    let hash_idx = force_cache_index(code_key, arg_key);
    if let Some(cached) = force_cache_lookup(protocol, hash_idx, code_key, arg_key) {
        return cached;
    }

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
            force_cache_store(protocol, hash_idx, code_key, arg_key, value);
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
            force_cache_store(protocol, hash_idx, code_key, arg_key, value);
            value
        }
        Err(err) => panic!("jit force callee frame raw failed: {err}"),
    }
}

/// Interpreter-only force: used by execute_call_assembler_direct
/// to handle guard failures without recursive compiled dispatch.
extern "C" fn jit_force_callee_frame_interp(frame_ptr: i64) -> i64 {
    let _suspend_inline_result = pyre_interp::call::suspend_inline_handled_result();
    let _force_guard = crate::eval::recursive_force_entry_bump();
    let frame = unsafe { &mut *(frame_ptr as *mut PyFrame) };

    let code_key = frame.code as usize;
    let green_key = crate::eval::make_green_key(frame.code, frame.next_instr);
    let protocol = finish_protocol(green_key);
    let arg_key = if frame.locals_cells_stack_w.len() > 0 {
        force_cache_arg_key(frame.locals_cells_stack_w[0])
    } else {
        0
    };

    let hash_idx = force_cache_index(code_key, arg_key);
    if let Some(cached) = force_cache_lookup(protocol, hash_idx, code_key, arg_key) {
        return cached;
    }

    // RPython: ResumeGuardForcedDescr.handle_fail() calls
    // resume_in_blackhole() which runs the blackhole interpreter
    // on jitcode bytecodes. This ensures no JIT re-entry.
    let result = resume_in_blackhole(frame);

    let value = match protocol {
        FinishProtocol::RawInt if !result.is_null() && unsafe { is_int(result) } => unsafe {
            w_int_get_value(result)
        },
        FinishProtocol::RawInt => result as i64,
        FinishProtocol::Boxed => result as i64,
    };
    force_cache_store(protocol, hash_idx, code_key, arg_key, value);
    value
}

/// RPython: blackhole.py resume_in_blackhole()
///
/// Public wrapper for guard failure recovery.
pub fn resume_in_blackhole_pub(frame: &mut PyFrame) -> pyre_object::PyObjectRef {
    resume_in_blackhole(frame)
}

/// RPython: blackhole.py resume_in_blackhole()
///
/// Compiles the frame's CodeObject to JitCode (via CodeWriter), creates
/// a BlackholeInterpreter, loads frame state, and runs it.
/// The blackhole has NO JIT entry points — structural isolation.
fn resume_in_blackhole(frame: &mut PyFrame) -> PyObjectRef {
    // Ensure JIT re-entry is blocked for all code called during
    // blackhole execution, including nested calls through bh_call_fn.
    let _force_guard = crate::eval::recursive_force_entry_bump();
    let code = unsafe { &*frame.code };

    // RPython: blackhole_from_resumedata() → setposition + consume_one_section
    // For pyre, we compile Python bytecodes to JitCode and load frame state.
    let writer = crate::jit::codewriter::CodeWriter::new(
        bh_call_fn,
        bh_load_global_fn,
        bh_compare_fn,
        bh_binary_op_fn,
        bh_box_int_fn,
        bh_truth_fn,
    );
    let pyjitcode = crate::jit::codewriter::get_or_compile_jitcode(code, &writer);

    // Map Python PC → JitCode PC
    let py_pc = frame.next_instr;
    let jitcode_pc = if py_pc < pyjitcode.pc_map.len() {
        pyjitcode.pc_map[py_pc]
    } else {
        0
    };

    // RPython: blackholeinterp = builder.acquire_interp()
    let mut bh = majit_meta::blackhole::BlackholeInterpreter::new();
    // RPython: blackholeinterp.setposition(jitcode, pc)
    bh.setposition(pyjitcode.jitcode.clone(), jitcode_pc);

    // RPython: resumereader.consume_one_section(curbh) — load register values
    // For pyre: fast locals → int registers, value stack → runtime stack
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

    // Set frame pointer in frame_reg (nlocals+3) for LOAD_GLOBAL and CALL
    if nlocals + 3 < bh.registers_i.len() {
        bh.setarg_i(nlocals + 3, frame as *mut PyFrame as i64);
    }

    if majit_meta::majit_log_enabled() {
        eprintln!(
            "[jit][blackhole] setup pc={} nlocals={} regs_i_len={} local0={} frame_reg={}",
            py_pc, nlocals, bh.registers_i.len(),
            bh.registers_i.get(0).copied().unwrap_or(-999),
            bh.registers_i.get(nlocals + 3).copied().unwrap_or(-999),
        );
    }

    // RPython: _run_forever(blackholeinterp, current_exc)
    bh.run();

    // Return value is in registers_i[0] (set by RETURN_VALUE → move_i(0, tmp0))
    let result = bh.registers_i[0] as PyObjectRef;
    if majit_meta::majit_log_enabled() {
        let int_val = if !result.is_null() && unsafe { is_int(result) } {
            Some(unsafe { w_int_get_value(result) })
        } else {
            None
        };
        eprintln!(
            "[jit][blackhole] resume result raw={} int_val={:?} pc={}",
            bh.registers_i[0], int_val, py_pc
        );
    }
    result
}

/// Interpreter-only force without memoization.
///
/// Used by the fused raw-int recursive helper, which already maintains
/// its own outer force cache keyed by (callee code, raw arg).
extern "C" fn jit_force_callee_frame_interp_nocache(frame_ptr: i64) -> i64 {
    let _suspend_inline_result = pyre_interp::call::suspend_inline_handled_result();
    let frame = unsafe { &mut *(frame_ptr as *mut PyFrame) };
    let green_key = crate::eval::make_green_key(frame.code, frame.next_instr);
    let protocol = finish_protocol(green_key);

    match pyre_interp::eval::eval_loop_for_force(frame) {
        Ok(result) => match protocol {
            FinishProtocol::RawInt if !result.is_null() && unsafe { is_int(result) } => unsafe {
                w_int_get_value(result)
            },
            FinishProtocol::RawInt => result as i64,
            FinishProtocol::Boxed => result as i64,
        },
        Err(err) => panic!("jit force callee frame (interp, nocache) failed: {err}"),
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
    let code_ptr = unsafe { w_func_get_code_ptr(callable_ref) };
    let green_key = crate::eval::make_green_key(code_ptr as *const _, 0);
    if matches!(finish_protocol(green_key), FinishProtocol::RawInt)
        && !boxed_arg_ref.is_null()
        && unsafe { is_int(boxed_arg_ref) }
    {
        let raw_arg = unsafe { w_int_get_value(boxed_arg_ref) };
        let forced = jit_force_recursive_call_raw_1(caller_frame, callable, raw_arg);
        return w_int_new(forced) as i64;
    }

    if majit_meta::majit_log_enabled() {
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
        let callee_arg0 =
            if !boxed.is_null() && unsafe { is_int(boxed) } { Some(unsafe { w_int_get_value(boxed) }) } else { None };
        eprintln!(
            "[jit][force-recursive-boxed] enter caller_arg0={:?} callee_arg0={:?}",
            caller_arg0, callee_arg0
        );
    }
    let frame_ptr = create_callee_frame_impl(caller_frame, callable, &[boxed_arg_ref]);
    let result = jit_force_callee_frame(frame_ptr);
    jit_drop_callee_frame(frame_ptr);
    if majit_meta::majit_log_enabled() {
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

/// Self-recursive single-arg boxed helper.
///
/// Keeps the boxed helper path off the generic callable redispatch and
/// blackhole fallback route. This mirrors the specialized raw helper:
/// the callee frame is created directly from the caller's code/globals.
pub extern "C" fn jit_force_self_recursive_call_1(caller_frame: i64, boxed_arg: i64) -> i64 {
    let _suspend_inline_result = pyre_interp::call::suspend_inline_handled_result();
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
    let result = {
        let frame = unsafe { &mut *(frame_ptr as *mut PyFrame) };
        let _recursive_entry = crate::eval::recursive_force_entry_bump();
        match crate::eval::eval_with_jit(frame) {
            Ok(result) => result as i64,
            Err(err) => panic!("jit force self-recursive call boxed failed: {err}"),
        }
    };
    jit_drop_callee_frame(frame_ptr);
    result
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
    let _suspend_inline_result = pyre_interp::call::suspend_inline_handled_result();
    let _force_guard = crate::eval::recursive_force_entry_bump();
    let callable_ref = callable as PyObjectRef;
    let code_ptr = unsafe { w_func_get_code_ptr(callable_ref) };
    let green_key = crate::eval::make_green_key(code_ptr as *const _, 0);
    let code_key = code_ptr as usize;
    let (protocol, token_num, memo_safe) = recursive_dispatch(callable_ref, green_key);
    let arg_key = ((raw_int_arg as usize) << 1) | 1;

    let hash_idx = force_cache_index(code_key, arg_key);
    if memo_safe {
        if let Some(cached) = force_cache_lookup(protocol, hash_idx, code_key, arg_key) {
            return cached;
        }
    }

    let boxed = pyre_object::intobject::w_int_new(raw_int_arg);
    let frame_ptr = create_callee_frame_impl_1_boxed(caller_frame, callable_ref, boxed);
    let result = {
        let frame = unsafe { &mut *(frame_ptr as *mut PyFrame) };
        // RPython parity: force callbacks always use blackhole, never
        // re-enter compiled code (no execute_call_assembler_direct).
        let bh_result = resume_in_blackhole(frame);
        match protocol {
            FinishProtocol::RawInt if !bh_result.is_null() && unsafe { is_int(bh_result) } => {
                unsafe { w_int_get_value(bh_result) }
            }
            FinishProtocol::RawInt => bh_result as i64,
            FinishProtocol::Boxed => bh_result as i64,
        }
    };
    jit_drop_callee_frame(frame_ptr);

    if memo_safe {
        force_cache_store(protocol, hash_idx, code_key, arg_key, result);
    }
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
/// This is closer to PyPy's recursive portal behavior, where the hot recursive
/// path re-enters the same frame shape instead of dispatching through an
/// additional callable metadata layer.
pub extern "C" fn jit_force_self_recursive_call_raw_1(caller_frame: i64, raw_int_arg: i64) -> i64 {
    let _suspend_inline_result = pyre_interp::call::suspend_inline_handled_result();
    if majit_meta::majit_log_enabled() && raw_int_arg <= 4 {
        eprintln!("[jit][force-self-recursive] enter arg={}", raw_int_arg);
    }
    let caller = unsafe { &*(caller_frame as *const PyFrame) };
    let code_ptr = caller.code;
    let green_key = crate::eval::make_green_key(code_ptr as *const _, 0);
    let code_key = code_ptr as usize;
    let (protocol, _token_num) = self_recursive_dispatch(green_key);
    let arg_key = ((raw_int_arg as usize) << 1) | 1;

    let hash_idx = force_cache_index(code_key, arg_key);
    if let Some(cached) = force_cache_lookup(protocol, hash_idx, code_key, arg_key) {
        return cached;
    }

    let boxed = pyre_object::intobject::w_int_new(raw_int_arg);
    let frame_ptr = create_self_recursive_callee_frame_impl_1_boxed(caller_frame, boxed);
    let result = {
        let frame = unsafe { &mut *(frame_ptr as *mut PyFrame) };
        let _recursive_entry = crate::eval::recursive_force_entry_bump();
        match crate::eval::eval_with_jit(frame) {
            Ok(result) => match protocol {
                FinishProtocol::RawInt if !result.is_null() && unsafe { is_int(result) } => unsafe {
                    w_int_get_value(result)
                },
                FinishProtocol::RawInt => result as i64,
                FinishProtocol::Boxed => result as i64,
            },
            Err(err) => panic!("jit force self-recursive call raw failed: {err}"),
        }
    };
    jit_drop_callee_frame(frame_ptr);
    force_cache_store(protocol, hash_idx, code_key, arg_key, result);
    if majit_meta::majit_log_enabled() && raw_int_arg <= 4 {
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
    });
}

extern "C" fn jit_bridge_compile_callee(
    frame_ptr: i64,
    fail_index: u32,
    trace_id: u64,
    green_key: u64,
) -> i64 {
    let _suspend_inline_result = pyre_interp::call::suspend_inline_handled_result();
    use majit_ir::{InputArg, Op, OpCode, OpRef, Type};
    use std::collections::HashMap;

    let frame = unsafe { &mut *(frame_ptr as *mut PyFrame) };
    let protocol = finish_protocol(green_key);
    let result = match pyre_interp::eval::eval_loop_for_force(frame) {
        Ok(r) => match protocol {
            FinishProtocol::RawInt if !r.is_null() && unsafe { is_int(r) } => unsafe {
                w_int_get_value(r)
            },
            FinishProtocol::RawInt => r as i64,
            FinishProtocol::Boxed => r as i64,
        },
        Err(e) => panic!("bridge force failed: {e}"),
    };

    let code_key = frame.code as usize;
    let arg_key = if frame.locals_cells_stack_w.len() > 0 {
        force_cache_arg_key(frame.locals_cells_stack_w[0])
    } else {
        0
    };
    let hash_idx = force_cache_index(code_key, arg_key);
    force_cache_store(protocol, hash_idx, code_key, arg_key, result);

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
    frame.pending_inline_result = None;
}

fn create_callee_frame_impl_1_boxed(
    caller_frame: i64,
    callable: PyObjectRef,
    boxed_arg: PyObjectRef,
) -> i64 {
    let code_ptr = unsafe { w_func_get_code_ptr(callable) };
    let caller = unsafe { &*(caller_frame as *const PyFrame) };
    let globals = unsafe { w_func_get_globals(callable) };
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
    // jit_force_callee_frame now returns boxed, so no re-boxing needed.
    jit_force_callee_frame(frame_ptr)
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
/// For CALL 1: args = [callable, arg0, frame_ptr]
/// For CALL 0: args = [callable, frame_ptr]
///
/// This function creates a callee frame and recursively runs the blackhole
/// on the callee's JitCode, ensuring no JIT re-entry.
pub extern "C" fn bh_call_fn(callable: i64, arg0: i64, frame_ptr: i64) -> i64 {
    let callable = callable as PyObjectRef;
    if callable.is_null() {
        return 0;
    }

    // RPython: bhimpl_recursive_call_i calls portal_runner directly.
    // pyre: we call call_user_function_plain which uses eval_frame_plain
    // (no JIT hooks). For full RPython parity, this should recursively
    // use the blackhole, but call_user_function_plain is sufficient
    // when combined with the IN_BLACKHOLE guard in try_function_entry_jit.
    let parent_frame = unsafe { &*(frame_ptr as *const PyFrame) };

    if !unsafe { is_func(callable) } {
        // Builtin function: call directly
        let func = unsafe { pyre_runtime::w_builtin_func_get(callable) };
        let args = [arg0 as PyObjectRef];
        return func(&args) as i64;
    }

    let code_ptr = unsafe { w_func_get_code_ptr(callable) };
    let globals = unsafe { w_func_get_globals(callable) };
    let closure = unsafe { w_func_get_closure(callable) };
    let func_code = code_ptr as *const pyre_bytecode::CodeObject;

    let args = [arg0 as PyObjectRef];
    let mut callee_frame = PyFrame::new_for_call_with_closure(
        func_code,
        &args,
        globals,
        parent_frame.execution_context,
        closure,
    );
    callee_frame.fix_array_ptrs();

    // RPython: bhimpl_recursive_call_i calls portal_runner directly,
    // which runs the blackhole interpreter recursively.
    // We recursively call resume_in_blackhole on the callee frame.
    resume_in_blackhole(&mut callee_frame) as i64
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
    // exec_load_name dispatches through the NamespaceOpcodeHandler trait;
    // for the blackhole we call the namespace directly.
    let ns = unsafe { &*frame.namespace };
    match ns.get(name) {
        Some(&value) => value as i64,
        None => 0,
    }
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
    pyre_objspace::opcode_ops::truth_value(obj) as i64
}

/// RPython: bhimpl_int_lt, bhimpl_int_eq, etc. — comparison helper.
///
/// Performs a Python-level comparison and returns a boolean PyObject.
/// op_code encodes the CompareOp tag from CPython 3.13 COMPARE_OP.
pub extern "C" fn bh_compare_fn(lhs: i64, rhs: i64, op_code: i64) -> i64 {
    let lhs = lhs as PyObjectRef;
    let rhs = rhs as PyObjectRef;
    if lhs.is_null() || rhs.is_null() {
        return 0;
    }

    // op_code is the raw CPython COMPARE_OP oparg (= ComparisonOperator discriminant).
    // Transmute back and call compare_value.
    use pyre_bytecode::bytecode::ComparisonOperator;
    let op: ComparisonOperator = unsafe { std::mem::transmute(op_code as u8) };
    match pyre_objspace::opcode_ops::compare_value(lhs, rhs, op) {
        Ok(result) => result as i64,
        Err(_) => 0,
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
        return 0;
    }

    // op_code is the raw CPython BINARY_OP oparg (= BinaryOperator discriminant).
    // Transmute back to BinaryOperator enum and call binary_value.
    use pyre_bytecode::bytecode::BinaryOperator;
    let op: BinaryOperator = unsafe { std::mem::transmute(op_code as u8) };
    match pyre_objspace::opcode_ops::binary_value(lhs, rhs, op) {
        Ok(result) => result as i64,
        Err(_) => 0,
    }
}
