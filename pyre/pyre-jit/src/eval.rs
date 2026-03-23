//! JIT-enabled evaluation — the sole entry point for JIT execution.
//!
//! This module owns the JitDriver, tracing hooks, and compiled-code
//! execution. pyre-interp provides the pure interpreter (eval_frame_plain)
//! and the opcode trait implementations on PyFrame.
//!
//! Equivalent to PyPy's `pypyjit/interp_jit.py` — the JIT is injected
//! from outside the interpreter.

use crate::jit::state::{PyreEnv, PyreJitState};
use crate::jit::trace::trace_bytecode;
use pyre_interp::frame::PyFrame;
use pyre_object::w_none;
use pyre_runtime::{PyResult, StepResult, execute_opcode_step};
use std::cell::{Cell, UnsafeCell};
use std::collections::HashSet;

use majit_gc::trace::TypeInfo;
use majit_ir::Value;
use majit_meta::blackhole::ExceptionState;
use majit_meta::{CompiledExitLayout, DetailedDriverRunOutcome, JitState};

/// RPython jitexc.py:53 ContinueRunningNormally parity.
enum LoopResult {
    Done(PyResult),
    ContinueRunningNormally,
}

/// Action from handle_jit_outcome for eval_loop_jit dispatch.
enum JitAction {
    Return(PyResult),
    Continue,
    /// RPython jitexc.py:53: guard-restored → restart portal.
    ContinueRunningNormally,
}

use crate::jit::descr::{W_FLOAT_GC_TYPE_ID, W_INT_GC_TYPE_ID};
use crate::jit::frame_layout::build_pyframe_virtualizable_info;
use majit_gc::collector::MiniMarkGC;
use majit_meta::JitDriver;
use pyre_object::floatobject::W_FloatObject;
use pyre_object::intobject::W_IntObject;

const JIT_THRESHOLD: u32 = 200;
type JitDriverPair = (
    JitDriver<PyreJitState>,
    majit_meta::virtualizable::VirtualizableInfo,
);

thread_local! {
    static JIT_DRIVER: UnsafeCell<JitDriverPair> = UnsafeCell::new({
        let info = build_pyframe_virtualizable_info();
        let mut d = JitDriver::new(JIT_THRESHOLD);
        d.set_virtualizable_info(info.clone());
        let mut gc = MiniMarkGC::new();
        let w_int_tid = gc.register_type(TypeInfo::simple(std::mem::size_of::<W_IntObject>()));
        debug_assert_eq!(w_int_tid, W_INT_GC_TYPE_ID);
        let w_float_tid =
            gc.register_type(TypeInfo::simple(std::mem::size_of::<W_FloatObject>()));
        debug_assert_eq!(w_float_tid, W_FLOAT_GC_TYPE_ID);
        d.set_gc_allocator(Box::new(gc));
        d.register_raw_int_box_helper(pyre_object::intobject::jit_w_int_new as *const ());
        d.register_raw_int_force_helper(crate::call_jit::jit_force_recursive_call_raw_1 as *const ());
        d.register_raw_int_force_helper(crate::call_jit::jit_force_self_recursive_call_raw_1 as *const ());
        d.register_create_frame_raw(
            crate::call_jit::jit_create_callee_frame_1 as *const (),
            crate::call_jit::jit_create_callee_frame_1_raw_int as *const (),
        );
        // PyPy interp_jit.py:75 — JitDriver(is_recursive=True)
        d.set_is_recursive(true);
        // warmspot.py:449 — portal function returns a Python object (int).
        // pyre's portal always returns PyObjectRef, but the JIT unboxes
        // int results to raw i64 via unbox_finish_result, so the static
        // result_type is Int.
        d.set_result_type(majit_ir::Type::Int);
        (d, info)
    });
}

#[inline]
pub fn driver_pair() -> &'static mut JitDriverPair {
    JIT_DRIVER.with(|cell| unsafe { &mut *cell.get() })
}

thread_local! {
    static JIT_CALL_DEPTH: Cell<u32> = Cell::new(0);
    /// Call depth at which the current trace started.
    static JIT_TRACING_DEPTH: Cell<u32> = Cell::new(0);
    /// Resume PC from the most recent guard failure restoration.
    static LAST_GUARD_RESUME_PC: Cell<usize> = Cell::new(0);
}

/// RPython green_key = (pycode, next_instr).
/// Each (code, pc) pair has independent warmup counter and compiled loop.
#[inline(always)]
pub fn make_green_key(code_ptr: *const pyre_bytecode::CodeObject, pc: usize) -> u64 {
    (code_ptr as u64).wrapping_mul(1000003) ^ (pc as u64)
}

/// RAII guard that decrements `JIT_CALL_DEPTH` on drop.
pub struct JitCallDepthGuard;

impl Drop for JitCallDepthGuard {
    fn drop(&mut self) {
        JIT_CALL_DEPTH.with(|d| d.set(d.get() - 1));
    }
}

/// Bump the JIT call depth. Returns a guard that restores the
/// depth when dropped.
///
/// RPython parity: track call depth unconditionally. jit_merge_point
/// (which only runs at depth 0) is gated by is_tracing_key, so
/// nested calls don't interfere with tracing. The depth prevents
/// re-entrant jit_merge_point during concrete execution.
#[inline]
pub fn jit_call_depth_bump() -> Option<JitCallDepthGuard> {
    JIT_CALL_DEPTH.with(|d| d.set(d.get() + 1));
    Some(JitCallDepthGuard)
}

/// RPython rstack.stack_almost_full() parity.
#[inline]
fn stack_almost_full() -> bool {
    JIT_CALL_DEPTH.with(|d| d.get()) > 20
}

/// Evaluate a Python frame with JIT compilation.
///
/// This is the main entry point for pyre-jit.
fn depth_bump_callback() -> Option<Box<dyn std::any::Any>> {
    jit_call_depth_bump().map(|g| Box::new(g) as Box<dyn std::any::Any>)
}

pub fn eval_with_jit(frame: &mut PyFrame) -> PyResult {
    pyre_interp::call::register_eval_override(eval_with_jit);
    pyre_interp::call::register_depth_bump(depth_bump_callback);
    pyre_interp::call::register_inline_call_override(
        crate::call_jit::maybe_handle_inline_concrete_call,
    );
    crate::call_jit::install_jit_call_bridge();
    frame.fix_array_ptrs();

    // RPython warmspot.py ll_portal_runner:
    //   maybe_compile_and_run(increment_threshold, *args)
    //   return portal_ptr(*args)
    //
    // maybe_compile_and_run = try_function_entry_jit: checks for compiled
    // code (dispatch) or threshold (start tracing). Internally guards on
    // JC_TRACING (driver.is_tracing()) to avoid re-entry during tracing.
    //
    // portal_ptr = eval_loop_jit at depth 0 (has jit_merge_point +
    // can_enter_jit back-edge), plain interpreter at depth > 0.
    if let Some(result) = try_function_entry_jit(frame) {
        return result;
    }
    handle_jitexception(frame)
}

/// RPython warmspot.py:941 portal_runner for force callbacks.
///
/// Unlike eval_with_jit, this does NOT register callbacks (already done)
/// and is safe to call from force helpers during compiled code execution.
/// RPython: bhimpl_recursive_call → portal_runner → CAN enter JIT.
pub(crate) fn portal_runner_for_force(frame: &mut PyFrame) -> PyResult {
    frame.fix_array_ptrs();
    if let Some(result) = try_function_entry_jit(frame) {
        return result;
    }
    handle_jitexception(frame)
}

/// RPython warmspot.py:961-1007 handle_jitexception parity.
#[inline(always)]
fn handle_jitexception(frame: &mut PyFrame) -> PyResult {
    loop {
        match eval_loop_jit(frame) {
            LoopResult::Done(result) => return result,
            LoopResult::ContinueRunningNormally => {
                frame.fix_array_ptrs();
                continue;
            }
        }
    }
}

fn debug_first_arg_int(frame: &PyFrame) -> Option<i64> {
    if frame.locals_cells_stack_w.len() == 0 {
        return None;
    }
    let value = frame.locals_cells_stack_w[0];
    if value.is_null() || !unsafe { pyre_object::pyobject::is_int(value) } {
        return None;
    }
    Some(unsafe { pyre_object::intobject::w_int_get_value(value) })
}

/// JIT-enabled evaluation loop (PyPy interp_jit.py dispatch()).
///
/// Calls merge_point on EVERY iteration (PyPy line 85-87), not just
/// when tracing. This matches PyPy's jit_merge_point placement.
/// RPython interp_jit.py dispatch() parity.
///
/// The hot loop mirrors RPython's structure exactly:
///   while True:
///       jit_merge_point(...)      # thin inline check
///       next_instr = handle_bytecode(...)
///
/// JIT hooks are thin inline checks; all heavy logic is in #[cold] helpers.
fn eval_loop_jit(frame: &mut PyFrame) -> LoopResult {
    let code = unsafe { &*frame.code };
    let env = PyreEnv;
    let (driver, info) = driver_pair();
    let is_portal: bool = &*code.obj_name != "<module>";

    loop {
        if frame.next_instr >= code.instructions.len() {
            return LoopResult::Done(Ok(w_none()));
        }

        let pc = frame.next_instr;
        let Some((instruction, op_arg)) = pyre_runtime::decode_instruction_at(code, pc) else {
            return LoopResult::Done(Ok(w_none()));
        };

        // ── jit_merge_point (RPython interp_jit.py:85-87) ──
        // Fast path: check TLS tracing_depth first (cheap, L1 cache).
        // driver.is_tracing() is checked only when depth==0 and trace just started.
        if is_portal {
            let tracing_depth = JIT_TRACING_DEPTH.with(|t| t.get());
            if tracing_depth != 0 {
                let current_depth = JIT_CALL_DEPTH.with(|d| d.get());
                if current_depth == tracing_depth {
                    if let Some(loop_result) =
                        jit_merge_point_hook(frame, code, pc, driver, info, &env)
                    {
                        return loop_result;
                    }
                }
            } else if driver.is_tracing() {
                // First merge_point after trace start — depth not yet set.
                if let Some(loop_result) = jit_merge_point_hook(frame, code, pc, driver, info, &env)
                {
                    return loop_result;
                }
            }
        }

        // ── inline replay (tracing bookkeeping) ──
        if frame.pending_inline_resume_pc == Some(pc) {
            if matches!(
                instruction,
                pyre_bytecode::bytecode::Instruction::Call { .. }
            ) {
                frame.pending_inline_resume_pc = None;
                continue;
            }
        }
        if let pyre_bytecode::bytecode::Instruction::Call { argc } = instruction {
            if !frame.pending_inline_results.is_empty() {
                frame.next_instr = pc + 1;
                if pyre_interp::call::replay_pending_inline_call(frame, argc.get(op_arg) as usize) {
                    continue;
                }
                frame.next_instr = pc;
            }
        }

        // ── handle_bytecode (RPython interp_jit.py:90) ──
        frame.next_instr += 1;
        let next_instr = frame.next_instr;
        if let pyre_bytecode::bytecode::Instruction::Call { argc } = instruction {
            if pyre_interp::call::replay_pending_inline_call(frame, argc.get(op_arg) as usize) {
                continue;
            }
        }
        match execute_opcode_step(frame, code, instruction, op_arg, next_instr) {
            Ok(StepResult::Continue) => {}
            Ok(StepResult::CloseLoop { loop_header_pc, .. }) if is_portal => {
                // ── can_enter_jit (RPython interp_jit.py:114) ──
                // Thin inline: counter tick only. Heavy logic in cold helper.
                let green_key = make_green_key(frame.code, loop_header_pc);
                if driver
                    .meta_interp_mut()
                    .warm_state_mut()
                    .counter
                    .tick(green_key)
                    && !driver.is_tracing()
                {
                    if let Some(loop_result) =
                        can_enter_jit_hook(frame, green_key, loop_header_pc, driver, info, &env)
                    {
                        return loop_result;
                    }
                    driver
                        .meta_interp_mut()
                        .warm_state_mut()
                        .counter
                        .reset(green_key);
                }
            }
            Ok(StepResult::CloseLoop { .. }) => {}
            Ok(StepResult::Return(result)) => return LoopResult::Done(Ok(result)),
            Ok(StepResult::Yield(result)) => return LoopResult::Done(Ok(result)),
            Err(err) => {
                if pyre_interp::eval::handle_exception(frame, &err) {
                    continue;
                }
                return LoopResult::Done(Err(err));
            }
        }
    }
}

/// RPython jit_merge_point slow path — only called when tracing is active.
#[cold]
#[inline(never)]
fn jit_merge_point_hook(
    frame: &mut PyFrame,
    code: &pyre_bytecode::CodeObject,
    pc: usize,
    driver: &mut JitDriver<PyreJitState>,
    info: &majit_meta::virtualizable::VirtualizableInfo,
    env: &PyreEnv,
) -> Option<LoopResult> {
    let concrete_frame = frame as *mut PyFrame as usize;
    let green_key = make_green_key(frame.code, pc);
    let mut jit_state = build_jit_state(frame, info);
    let current_depth = JIT_CALL_DEPTH.with(|d| d.get());
    if let Some(outcome) = driver.jit_merge_point_keyed(
        green_key,
        pc,
        &mut jit_state,
        env,
        || {},
        |ctx, sym| {
            JIT_TRACING_DEPTH.with(|t| t.set(current_depth));
            let mut trace_frame = frame.snapshot_for_tracing();
            let trace_frame_ptr = (&mut trace_frame) as *mut PyFrame as usize;
            let _ = concrete_frame;
            trace_bytecode(ctx, sym, code, pc, trace_frame_ptr)
        },
    ) {
        match handle_jit_outcome(outcome, &jit_state, frame, info, green_key) {
            JitAction::Return(result) => return Some(LoopResult::Done(result)),
            JitAction::ContinueRunningNormally => return Some(LoopResult::ContinueRunningNormally),
            JitAction::Continue => {}
        }
    }
    // Trace completed or aborted — clear tracing depth if it was us.
    if !driver.is_tracing() {
        JIT_TRACING_DEPTH.with(|t| t.set(0));
    }
    None
}

/// RPython can_enter_jit / maybe_compile_and_run slow path.
/// Called only when counter threshold fires.
#[cold]
#[inline(never)]
fn can_enter_jit_hook(
    frame: &mut PyFrame,
    green_key: u64,
    loop_header_pc: usize,
    driver: &mut JitDriver<PyreJitState>,
    info: &majit_meta::virtualizable::VirtualizableInfo,
    env: &PyreEnv,
) -> Option<LoopResult> {
    // Early exit for blacklisted keys: avoid build_jit_state + build_meta overhead.
    let has_compiled = driver.has_compiled_loop(green_key);
    if !has_compiled
        && !driver
            .meta_interp()
            .warm_state_ref()
            .counter_would_fire(green_key)
    {
        return None;
    }
    let mut jit_state = build_jit_state(frame, info);
    if majit_meta::majit_log_enabled() {
        eprintln!(
            "[jit][root-backedge] enter key={} pc={} arg0={:?} has_compiled={}",
            green_key,
            loop_header_pc,
            debug_first_arg_int(frame),
            has_compiled
        );
    }
    // RPython can_enter_jit only runs compiled code — tracing starts
    // from jit_merge_point via the eval_loop_jit dispatch path.
    let outcome = if driver.has_compiled_loop(green_key) {
        Some(driver.run_compiled_detailed_with_bridge_keyed(
            green_key,
            loop_header_pc,
            &mut jit_state,
            env,
            || {},
            restore_guard_failure_for_loop,
        ))
    } else if !driver.is_tracing() {
        if stack_almost_full() {
            return None;
        }
        driver.back_edge_or_run_compiled_keyed(
            green_key,
            loop_header_pc,
            &mut jit_state,
            env,
            || {},
        )
    } else {
        None
    };
    if let Some(outcome) = outcome {
        if majit_meta::majit_log_enabled() {
            let kind = match &outcome {
                DetailedDriverRunOutcome::Finished { .. } => "finished",
                DetailedDriverRunOutcome::Jump { .. } => "jump",
                DetailedDriverRunOutcome::Abort { .. } => "abort",
                DetailedDriverRunOutcome::GuardFailure { restored: true, .. } => "guard-restored",
                DetailedDriverRunOutcome::GuardFailure { .. } => "guard-unrestored",
            };
            eprintln!(
                "[jit][root-backedge] outcome key={} pc={} kind={}",
                green_key, loop_header_pc, kind
            );
        }
        match handle_jit_outcome(outcome, &jit_state, frame, info, green_key) {
            JitAction::Return(result) => return Some(LoopResult::Done(result)),
            JitAction::ContinueRunningNormally => return Some(LoopResult::ContinueRunningNormally),
            JitAction::Continue => {}
        }
    }
    JIT_TRACING_DEPTH.with(|t| t.set(0));
    None
}

/// RPython warmstate.py maybe_compile_and_run parity.
///
/// Called at every portal entry (function call). Must be fast for the
/// common case (no compiled code, not tracing, threshold not reached).
pub fn try_function_entry_jit(frame: &mut PyFrame) -> Option<PyResult> {
    let green_key = make_green_key(frame.code, frame.next_instr);
    let (driver, info) = driver_pair();

    // RPython warmstate.py maybe_compile_and_run fast path:
    // if no compiled loop and not tracing, just tick the counter.
    if !driver.has_compiled_loop(green_key) && !driver.is_tracing() {
        let should_trace = driver
            .meta_interp_mut()
            .warm_state_mut()
            .should_trace_function_entry(green_key);
        if !should_trace {
            return None;
        }
    }

    if driver.has_compiled_loop(green_key) {
        if majit_meta::majit_log_enabled() {
            eprintln!(
                "[jit][func-entry] run compiled key={} arg0={:?} depth={} raw_finish_known={}",
                green_key,
                debug_first_arg_int(frame),
                JIT_CALL_DEPTH.with(|d| d.get()),
                driver.has_raw_int_finish(green_key)
            );
        }
        let env = PyreEnv;
        let mut jit_state = build_jit_state(frame, info);
        let outcome = driver.run_compiled_detailed_with_bridge_keyed(
            green_key,
            frame.next_instr,
            &mut jit_state,
            &env,
            || {},
            restore_guard_failure_for_loop,
        );
        // RPython compile.py handle_fail parity: extract fail info
        // before consuming outcome, for bridge compilation.
        let guard_fail_info = match &outcome {
            DetailedDriverRunOutcome::GuardFailure {
                fail_index: Some(fi),
                trace_id: Some(tid),
                ..
            } => Some((*fi, *tid)),
            _ => None,
        };

        {
            if majit_meta::majit_log_enabled() {
                let kind = match &outcome {
                    DetailedDriverRunOutcome::Finished { .. } => "finished",
                    DetailedDriverRunOutcome::Jump { .. } => "jump",
                    DetailedDriverRunOutcome::Abort { .. } => "abort",
                    DetailedDriverRunOutcome::GuardFailure { restored: true, .. } => {
                        "guard-restored"
                    }
                    DetailedDriverRunOutcome::GuardFailure {
                        restored: false, ..
                    } => "guard-unrestored",
                };
                eprintln!(
                    "[jit][func-entry] compiled outcome key={} arg0={:?} kind={}",
                    green_key,
                    debug_first_arg_int(frame),
                    kind
                );
            }
            match handle_jit_outcome(outcome, &jit_state, frame, info, green_key) {
                JitAction::Return(result) => {
                    // Drain pending bridge requests before returning.
                    let pending = majit_codegen_cranelift::take_pending_bridge_compile();
                    for (gk, tid, fi, resume_pc) in pending {
                        let effective_gk = if gk == 0 { green_key } else { gk };
                        let effective_rpc = if resume_pc == 0 {
                            LAST_GUARD_RESUME_PC.with(|c| c.get())
                        } else {
                            resume_pc
                        };
                        crate::call_jit::jit_bridge_compile_for_guard(
                            effective_gk,
                            tid,
                            fi,
                            frame,
                            effective_rpc,
                        );
                    }
                    return Some(result);
                }
                JitAction::ContinueRunningNormally | JitAction::Continue => {}
            }
        }

        // After compiled code guard-restored fallback, re-establish the
        // frame's array pointer.
        frame.fix_array_ptrs();
        return None;
    }

    if majit_meta::majit_log_enabled() {
        eprintln!(
            "[jit][func-entry] probe key={} arg0={:?} tracing={}",
            green_key,
            debug_first_arg_int(frame),
            driver.is_tracing(),
        );
    }

    if driver.is_tracing() {
        return None;
    }

    // RPython warmstate.py:446 maybe_compile_and_run: standard counter
    // threshold for ALL functions. No self-recursive boosting.
    let should_trace = driver
        .meta_interp_mut()
        .warm_state_mut()
        .should_trace_function_entry(green_key);
    let count = driver
        .meta_interp()
        .warm_state_ref()
        .function_entry_count(green_key);
    let function_threshold = driver.meta_interp().warm_state_ref().function_threshold();
    let boosted = driver.is_function_boosted(green_key);
    if majit_meta::majit_log_enabled() {
        eprintln!(
            "[jit][func-entry] count key={} arg0={:?} count={} boosted={} threshold={}",
            green_key,
            debug_first_arg_int(frame),
            count,
            boosted,
            function_threshold
        );
    }
    if !should_trace {
        return None;
    }
    let env = PyreEnv;
    let mut jit_state = build_jit_state(frame, info);
    driver
        .meta_interp_mut()
        .warm_state_mut()
        .reset_function_entry_count(green_key);
    if majit_meta::majit_log_enabled() {
        eprintln!(
            "[jit][func-entry] start tracing key={} arg0={:?} count={} boosted={}",
            green_key,
            debug_first_arg_int(frame),
            count,
            boosted
        );
    }
    driver.force_start_tracing(green_key, frame.next_instr, &mut jit_state, &env);
    None
}

fn handle_jit_outcome(
    outcome: DetailedDriverRunOutcome,
    _jit_state: &PyreJitState,
    frame: &mut PyFrame,
    _info: &majit_meta::virtualizable::VirtualizableInfo,
    green_key: u64,
) -> JitAction {
    match outcome {
        DetailedDriverRunOutcome::Finished {
            typed_values,
            raw_int_result,
            ..
        } => {
            let (driver, _) = driver_pair();
            let raw_int_result = raw_int_result || driver.has_raw_int_finish(green_key);
            if majit_meta::majit_log_enabled() {
                eprintln!(
                    "[jit][handle-outcome] finished key={} raw_flag={} typed_values={:?}",
                    green_key, raw_int_result, typed_values
                );
            }
            let [value] = typed_values.as_slice() else {
                return JitAction::Return(Err(pyre_runtime::PyError::type_error(
                    "compiled finish did not produce a single object return value",
                )));
            };
            let value = match value {
                majit_ir::Value::Int(raw) => {
                    // RPython parity: top-level finish values are already typed
                    // (MIFrame.make_result_of_lastop / finishframe).  An INT
                    // result register represents a Python int payload, not an
                    // object pointer.  Re-box at the interpreter boundary
                    // regardless of whether majit marked the trace as a
                    // raw-int-finish optimization.
                    let _ = raw_int_result;
                    pyre_object::intobject::w_int_new(*raw)
                }
                majit_ir::Value::Ref(value) => value.as_usize() as pyre_object::PyObjectRef,
                majit_ir::Value::Float(f) => {
                    // Re-box the raw float into a W_FloatObject.
                    pyre_object::floatobject::w_float_new(*f)
                }
                majit_ir::Value::Void => {
                    return JitAction::Return(Err(pyre_runtime::PyError::type_error(
                        "compiled finish produced a void return value",
                    )));
                }
            };
            JitAction::Return(Ok(value))
        }
        DetailedDriverRunOutcome::Jump { .. } => {
            let _ = frame;
            JitAction::Continue
        }
        DetailedDriverRunOutcome::GuardFailure { restored: true, .. } => {
            // RPython: guard failure → resume_in_blackhole → interprets
            // from guard PC to loop header → ContinueRunningNormally.
            // pyre skips the blackhole step: frame is already restored,
            // eval_loop_jit continues from the restored PC with JIT hooks.
            // This is equivalent because the next back-edge will trigger
            // can_enter_jit normally.
            let _ = frame;
            JitAction::Continue
        }
        DetailedDriverRunOutcome::GuardFailure {
            restored: false, ..
        }
        | DetailedDriverRunOutcome::Abort { .. } => JitAction::Continue,
    }
}

fn decode_exit_layout_values(raw_values: &[i64], layout: &CompiledExitLayout) -> Vec<Value> {
    layout
        .exit_types
        .iter()
        .enumerate()
        .map(|(index, tp)| {
            let raw = raw_values.get(index).copied().unwrap_or(0);
            match tp {
                majit_ir::Type::Int => Value::Int(raw),
                majit_ir::Type::Ref => Value::Ref(majit_ir::GcRef(raw as usize)),
                majit_ir::Type::Float => Value::Float(f64::from_bits(raw as u64)),
                majit_ir::Type::Void => Value::Void,
            }
        })
        .collect()
}

fn restore_guard_failure_for_loop(
    jit_state: &mut PyreJitState,
    meta: &crate::jit::state::PyreMeta,
    raw_values: &[i64],
    exit_layout: &CompiledExitLayout,
) -> Option<usize> {
    if majit_meta::majit_log_enabled() {
        let nraw = raw_values.len().min(8);
        let slots: Vec<String> = (0..nraw)
            .map(|i| format!("{:#x}", raw_values[i] as usize))
            .collect();
        eprintln!(
            "[jit] guard-fail: fail_idx={} types={:?} raw=[{}]",
            exit_layout.fail_index,
            exit_layout.exit_types,
            slots.join(", ")
        );
    }
    let mut typed = decode_exit_layout_values(raw_values, exit_layout);
    // RPython resume.py: materialize virtual objects from recovery_layout.
    // Virtual slots in fail_args are NONE (null Ref); their field values
    // are stored as extra fail_args. Reconstruct the concrete PyObject
    // from the field values and replace the null slot.
    materialize_recovery_virtuals(&mut typed, exit_layout);
    // Safety: if any null Ref slots remain after materialization,
    // the frame state is unsafe for the interpreter. Return None
    // (= not restored) so the interpreter ignores the guard result
    // and continues from scratch rather than dereferencing null.
    let has_null_ref = typed
        .iter()
        .skip(3)
        .any(|v| matches!(v, Value::Ref(majit_ir::GcRef(0))));
    if has_null_ref {
        if majit_meta::majit_log_enabled() {
            eprintln!("[jit] guard-fail: null Ref in restored values, invalidating compiled code");
        }
        // Blacklist all compiled keys AND the originating trace key.
        // Clear loop_tokens so counter_would_fire returns false.
        let (driver, _) = driver_pair();
        let keys: Vec<u64> = driver.meta_interp().all_compiled_keys();
        for key in keys {
            driver
                .meta_interp_mut()
                .warm_state_mut()
                .abort_tracing(key, true);
            driver
                .meta_interp_mut()
                .warm_state_mut()
                .clear_loop_token(key);
        }
        driver.invalidate_all_compiled();
        return None;
    }
    let restored = jit_state.restore_guard_failure_values(meta, &typed, &ExceptionState::default());
    // Store resume PC for bridge compilation. Guard failures inside
    // CallAssemblerI force callbacks lose the frame state, so we save
    // the restored next_instr here for later use.
    LAST_GUARD_RESUME_PC.with(|c| c.set(jit_state.next_instr));
    if majit_meta::majit_log_enabled() {
        eprintln!(
            "[jit] guard-fail restored: ni={} vsd={}",
            jit_state.next_instr, jit_state.valuestackdepth
        );
    }
    restored.then_some(jit_state.next_instr)
}

/// Guard failure recovery: reconstruct virtual objects from their
/// field values stored as extra fail_args after null (NONE) slots.
///
/// When the optimizer places a virtual in fail_args, it sets the
/// virtual's slot to NONE and appends field values (ob_type, intval).
/// On guard failure, we detect contiguous null Ref slots at the end
/// of the locals/stack region and pair them with trailing Int fields.
///
/// Pattern: [..., NONE, NONE] [ob_type1, intval1, ob_type2, intval2]
/// Only apply when trailing fields are available (2 Int per null slot).
/// Otherwise, replace remaining null Refs with w_none() for safety.
fn materialize_recovery_virtuals(typed: &mut Vec<Value>, _exit_layout: &CompiledExitLayout) {
    // Find all null Ref slots (virtual markers) after the header (frame/ni/vsd)
    let null_slots: Vec<usize> = (3..typed.len())
        .filter(|&i| matches!(typed.get(i), Some(Value::Ref(majit_ir::GcRef(0)))))
        .collect();
    if null_slots.is_empty() {
        return;
    }

    // Check if trailing values after the LAST null slot contain enough
    // Int pairs to reconstruct all null slots as virtuals.
    let last_null = *null_slots.last().unwrap();
    let trailing_start = last_null + 1;
    let trailing_count = typed.len() - trailing_start;
    let needed_fields = null_slots.len() * 2;

    if trailing_count >= needed_fields {
        // Validate: first field of each pair should be a W_IntObject type id.
        // If not, the trailing data is not virtual field data.
        let w_int_type_id = pyre_object::intobject::w_int_type_id();
        let mut valid = true;
        let mut check_cursor = trailing_start;
        for _ in &null_slots {
            if check_cursor + 1 >= typed.len() {
                valid = false;
                break;
            }
            if let Value::Int(ob_type) = typed[check_cursor] {
                if ob_type as usize != w_int_type_id {
                    valid = false;
                    break;
                }
            } else {
                valid = false;
                break;
            }
            check_cursor += 2;
        }
        if !valid {
            // Not virtual field data — leave null Refs as-is.
            // The caller's has_null_ref check will detect this and
            // skip the restore entirely.
            return;
        }
        // Enough trailing fields: reconstruct virtuals
        let mut field_cursor = trailing_start;
        for &slot_idx in &null_slots {
            let intval_pos = field_cursor + 1;
            if intval_pos >= typed.len() {
                break;
            }
            if let Value::Int(v) = typed[intval_pos] {
                let obj = pyre_object::intobject::w_int_new(v);
                typed[slot_idx] = Value::Ref(majit_ir::GcRef(obj as usize));
                if majit_meta::majit_log_enabled() {
                    eprintln!(
                        "[jit] materialized virtual W_IntObject(intval={}) at slot {}",
                        v, slot_idx
                    );
                }
            }
            field_cursor += 2;
        }
        // Truncate consumed field values
        typed.truncate(trailing_start);
    } else {
        // Not enough trailing fields — null slots are dead/unused.
        // Replace with w_none() to prevent SIGSEGV.
        for &slot_idx in &null_slots {
            typed[slot_idx] = Value::Ref(majit_ir::GcRef(w_none() as usize));
        }
    }
}

pub(crate) fn build_jit_state(
    frame: &PyFrame,
    virtualizable_info: &majit_meta::virtualizable::VirtualizableInfo,
) -> PyreJitState {
    let mut jit_state = PyreJitState {
        frame: frame as *const PyFrame as usize,
        next_instr: frame.next_instr,
        valuestackdepth: frame.valuestackdepth,
    };
    if !jit_state.sync_from_virtualizable(virtualizable_info) {
        jit_state.next_instr = frame.next_instr;
        jit_state.valuestackdepth = frame.valuestackdepth;
    }
    jit_state
}

fn sync_jit_state_to_frame(
    jit_state: &PyreJitState,
    frame: &mut PyFrame,
    virtualizable_info: &majit_meta::virtualizable::VirtualizableInfo,
) {
    if !jit_state.sync_to_virtualizable(virtualizable_info) {
        frame.next_instr = jit_state.next_instr;
        frame.valuestackdepth = jit_state.valuestackdepth;
    }
    frame.next_instr = jit_state.next_instr;
    frame.valuestackdepth = jit_state.valuestackdepth;
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyre_runtime::{is_func, w_func_get_code_ptr};

    #[test]
    fn test_eval_simple_addition() {
        let source = "x = 1 + 2";
        let code = pyre_bytecode::compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_with_jit(&mut frame);
        unsafe {
            let x = *(*frame.namespace).get("x").unwrap();
            assert_eq!(pyre_object::intobject::w_int_get_value(x), 3);
        }
    }

    #[test]
    fn test_eval_while_loop() {
        let source = "\
i = 0
s = 0
while i < 100:
    s = s + i
    i = i + 1";
        let code = pyre_bytecode::compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_with_jit(&mut frame);
        unsafe {
            let s = *(*frame.namespace).get("s").unwrap();
            assert_eq!(pyre_object::intobject::w_int_get_value(s), 4950);
        }
    }

    #[test]
    fn test_eval_with_jit_redecodes_opargs_after_extended_arg_jumps() {
        let source = "\
def fannkuch(n):
    p = [0] * n
    q = [0] * n
    s = [0] * n
    i = 0
    while i < n:
        p[i] = i
        q[i] = i
        s[i] = i
        i = i + 1
    maxflips = 0
    checksum = 0
    sign = 1
    while True:
        q0 = p[0]
        if q0 != 0:
            i = 1
            while i < n:
                q[i] = p[i]
                i = i + 1
            flips = 1
            while True:
                qq = q[q0]
                if qq == 0:
                    break
                q[q0] = q0
                if q0 >= 3:
                    i = 1
                    j = q0 - 1
                    while i < j:
                        t = q[i]
                        q[i] = q[j]
                        q[j] = t
                        i = i + 1
                        j = j - 1
                q0 = qq
                flips = flips + 1
            if flips > maxflips:
                maxflips = flips
            checksum = checksum + sign * flips
        if sign == 1:
            t = p[0]
            p[0] = p[1]
            p[1] = t
            sign = -1
        else:
            t = p[1]
            p[1] = p[2]
            p[2] = t
            sign = 1
            i = 2
            while i < n:
                sx = s[i]
                if sx != 0:
                    s[i] = sx - 1
                    break
                if i == n - 1:
                    return 999
                s[i] = i
                t = p[0]
                j = 0
                while j < i + 1:
                    p[j] = p[j + 1]
                    j = j + 1
                p[i + 1] = t
                i = i + 1

r = fannkuch(6)";
        let code = pyre_bytecode::compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_with_jit(&mut frame);
        unsafe {
            let r = *(*frame.namespace).get("r").unwrap();
            assert_eq!(pyre_object::intobject::w_int_get_value(r), 999);
        }
    }

    #[test]
    fn test_eval_recursive_fib_compiles_function_entry_trace() {
        let source = "\
def fib(n):
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)

result = fib(12)";
        let code = pyre_bytecode::compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_with_jit(&mut frame);
        unsafe {
            let fib = *(*frame.namespace).get("fib").unwrap();
            assert!(is_func(fib));
            let fib_key = make_green_key(w_func_get_code_ptr(fib) as *const _, 0);
            let result = *(*frame.namespace).get("result").unwrap();
            assert_eq!(pyre_object::intobject::w_int_get_value(result), 144);
            let (driver, _) = driver_pair();
            assert!(
                driver.has_compiled_loop(fib_key),
                "recursive fib should compile a function-entry trace"
            );
            assert!(
                driver.has_raw_int_finish(fib_key),
                "recursive fib compiled finish should use the raw-int protocol"
            );
        }
    }

    #[test]
    fn test_recursive_fib_callable_prefers_function_entry() {
        let source = "\
def fib(n):
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)
";
        let code = pyre_bytecode::compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_with_jit(&mut frame);
        unsafe {
            let fib = *(*frame.namespace).get("fib").unwrap();
            assert!(crate::call_jit::callable_prefers_function_entry(fib));
        }
    }

    #[test]
    fn test_nonrecursive_helper_does_not_prefer_function_entry() {
        let source = "\
def add(a, b):
    return a + b
";
        let code = pyre_bytecode::compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_with_jit(&mut frame);
        unsafe {
            let add = *(*frame.namespace).get("add").unwrap();
            assert!(!crate::call_jit::callable_prefers_function_entry(add));
        }
    }

    #[test]
    fn test_recursive_global_reads_do_not_reuse_force_cache_across_global_mutation() {
        let source = "\
factor = 1
def g(n):
    if n < 2:
        return n * factor
    return g(n - 1) + g(n - 2) + factor

first = g(12)
factor = 2
second = g(12)";
        let code = pyre_bytecode::compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_with_jit(&mut frame);
        unsafe {
            let first = *(*frame.namespace).get("first").unwrap();
            let second = *(*frame.namespace).get("second").unwrap();
            assert_eq!(pyre_object::intobject::w_int_get_value(first), 376);
            assert_eq!(pyre_object::intobject::w_int_get_value(second), 752);
        }
    }

    #[test]
    fn test_inline_residual_user_call_with_many_args_stays_correct() {
        let source = "\
def helper(a, b, c, d, e):
    return a + b + c + d + e

def outer(x):
    return helper(x, x, x, x, x)

s = 0
i = 0
while i < 300:
    s = s + outer(i)
    i = i + 1";
        let code = pyre_bytecode::compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_with_jit(&mut frame);
        unsafe {
            let s = *(*frame.namespace).get("s").unwrap();
            assert_eq!(pyre_object::intobject::w_int_get_value(s), 224_250);
        }
    }

    #[test]
    fn test_nested_direct_helper_calls_stay_correct() {
        let source = "\
def add(a, b):
    return a + b

def mul(a, b):
    return a * b

def square(x):
    return mul(x, x)

def compute(i):
    return add(square(i), i)

s = 0
i = 0
while i < 300:
    s = add(s, compute(i))
    i = add(i, 1)";
        let code = pyre_bytecode::compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_with_jit(&mut frame);
        unsafe {
            let s = *(*frame.namespace).get("s").unwrap();
            assert_eq!(pyre_object::intobject::w_int_get_value(s), 8_999_900);
        }
    }

    #[test]
    fn test_dynamic_int_list_indexing_stays_correct() {
        let source = "\
q = [0, 1, 2, 3, 4]
i = 0
s = 0
while i < 200:
    q0 = i % 5
    s = s + q[q0]
    q[q0] = q[q0] + 1
    i = i + 1";
        let code = pyre_bytecode::compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_with_jit(&mut frame);
        unsafe {
            let s = *(*frame.namespace).get("s").unwrap();
            let q = *(*frame.namespace).get("q").unwrap();
            assert_eq!(pyre_object::intobject::w_int_get_value(s), 4_300);
            assert_eq!(
                pyre_object::intobject::w_int_get_value(
                    pyre_object::listobject::w_list_getitem(q, 0).unwrap()
                ),
                40
            );
            assert_eq!(
                pyre_object::intobject::w_int_get_value(
                    pyre_object::listobject::w_list_getitem(q, 4).unwrap()
                ),
                44
            );
        }
    }
}
