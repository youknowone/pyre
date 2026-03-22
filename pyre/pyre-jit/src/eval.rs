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
        (d, info)
    });
}

#[inline]
pub fn driver_pair() -> &'static mut JitDriverPair {
    JIT_DRIVER.with(|cell| unsafe { &mut *cell.get() })
}

thread_local! {
    static PYRE_RAW_INT_FINISH_HINTS: UnsafeCell<HashSet<u64>> =
        UnsafeCell::new(HashSet::new());

    static JIT_CALL_DEPTH: Cell<u32> = Cell::new(0);
    static JIT_TRACING: Cell<bool> = Cell::new(false);
    static RECURSIVE_FORCE_ENTRY_DEPTH: Cell<u32> = Cell::new(0);
    static BLACKHOLE_ENTRY_DEPTH: Cell<u32> = Cell::new(0);
}

fn raw_int_finish_hints() -> &'static mut HashSet<u64> {
    PYRE_RAW_INT_FINISH_HINTS.with(|cell| unsafe { &mut *cell.get() })
}

#[inline]
pub(crate) fn note_finish_protocol_hint(green_key: u64, raw_int: bool) {
    let hints = raw_int_finish_hints();
    if raw_int {
        hints.insert(green_key);
    } else {
        hints.remove(&green_key);
    }
}

#[inline]
pub(crate) fn has_finish_protocol_hint(green_key: u64) -> bool {
    raw_int_finish_hints().contains(&green_key)
}

/// RPython green_key = (pycode, next_instr).
/// Each (code, pc) pair has independent warmup counter and compiled loop.
#[inline]
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

pub(crate) struct RecursiveForceEntryGuard;

impl Drop for RecursiveForceEntryGuard {
    fn drop(&mut self) {
        RECURSIVE_FORCE_ENTRY_DEPTH.with(|d| d.set(d.get() - 1));
    }
}

pub(crate) struct BlackholeEntryGuard;

impl Drop for BlackholeEntryGuard {
    fn drop(&mut self) {
        BLACKHOLE_ENTRY_DEPTH.with(|d| d.set(d.get() - 1));
    }
}

/// Bump the JIT call depth. Returns a guard that restores the
/// depth when dropped.
///
/// PyPy interp_jit.py parity: only bump depth when actively tracing.
/// When NOT tracing, inner function calls run eval_loop_jit with
/// depth==0, allowing their own loops to independently warm up and
/// compile with separate green keys — matching PyPy's is_recursive
/// JitDriver behavior where each (pycode, next_instr) gets its own
/// compiled loop.
#[inline]
pub fn jit_call_depth_bump() -> Option<JitCallDepthGuard> {
    if JIT_TRACING.with(|t| t.get()) {
        JIT_CALL_DEPTH.with(|d| d.set(d.get() + 1));
        Some(JitCallDepthGuard)
    } else {
        None
    }
}

#[inline]
pub(crate) fn recursive_force_entry_bump() -> RecursiveForceEntryGuard {
    RECURSIVE_FORCE_ENTRY_DEPTH.with(|d| d.set(d.get() + 1));
    RecursiveForceEntryGuard
}

#[inline]
pub(crate) fn in_recursive_force_entry() -> bool {
    RECURSIVE_FORCE_ENTRY_DEPTH.with(|d| d.get() > 0)
}

#[inline]
pub(crate) fn blackhole_entry_bump() -> BlackholeEntryGuard {
    BLACKHOLE_ENTRY_DEPTH.with(|d| d.set(d.get() + 1));
    BlackholeEntryGuard
}

#[inline]
pub(crate) fn in_blackhole_entry() -> bool {
    BLACKHOLE_ENTRY_DEPTH.with(|d| d.get() > 0)
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

    if let Some(result) = try_function_entry_jit(frame) {
        return result;
    }

    eval_loop_jit(frame)
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
pub fn eval_loop_jit(frame: &mut PyFrame) -> PyResult {
    let code = unsafe { &*frame.code };

    let env = PyreEnv;
    let (driver, info) = driver_pair();

    loop {
        if frame.next_instr >= code.instructions.len() {
            return Ok(w_none());
        }

        let pc = frame.next_instr;
        let Some((instruction, op_arg)) = pyre_runtime::decode_instruction_at(code, pc) else {
            return Ok(w_none());
        };

        // PyPy interp_jit.py:85-87 — jit_merge_point on EVERY iteration.
        //
        // RPython interp_jit.py dispatch() calls jit_merge_point for ALL
        // Python code, including <module>-level loops. However, module-level
        // code uses LOAD_NAME/STORE_NAME (dict access) instead of
        // LOAD_FAST/STORE_FAST, so the virtualizable mechanism cannot carry
        // locals as loop-carried values. Skip <module> until dict caching
        // is implemented in the optimizer (RPython OptHeap dict caching).
        let is_portal: bool = {
            let name: &str = &code.obj_name;
            name != "<module>"
        };
        if is_portal && JIT_CALL_DEPTH.with(|d| d.get()) == 0 {
            let concrete_frame = frame as *mut PyFrame as usize;
            let green_key = make_green_key(frame.code, pc);
            let mut jit_state = build_jit_state(frame, info);
            if let Some(outcome) = driver.jit_merge_point_keyed(
                green_key,
                pc,
                &mut jit_state,
                &env,
                || {},
                |ctx, sym| {
                    JIT_TRACING.with(|t| t.set(true));
                    let mut trace_frame = frame.snapshot_for_tracing();
                    let trace_frame_ptr = (&mut trace_frame) as *mut PyFrame as usize;
                    let _ = concrete_frame;
                    trace_bytecode(ctx, sym, code, pc, trace_frame_ptr)
                },
            ) {
                if let Some(result) = handle_jit_outcome(outcome, &jit_state, frame, info, green_key) {
                    return result;
                }
            }
            if !driver.is_tracing() {
                JIT_TRACING.with(|t| t.set(false));
            }
        }

        // RPython perform_call()/finishframe() resumes the parent frame
        // immediately after an inlined child returns. If tracing already
        // concretely executed the CALL and advanced next_instr, do not replay
        // or execute the original opcode again here.
        if matches!(instruction, pyre_bytecode::bytecode::Instruction::Call { .. }) {
            if frame.pending_inline_resume_pc == Some(pc) {
                frame.pending_inline_resume_pc = None;
                continue;
            }
        }

        // RPython interp_jit.py places jit_merge_point at the dispatch-loop
        // head, before any opcode-specific handoff. Keep replay-only CALL
        // result delivery after jit_merge_point so loop headers at CALL sites
        // still warm up under the same ownership as PyPy.
        if let pyre_bytecode::bytecode::Instruction::Call { argc } = instruction {
            if !frame.pending_inline_results.is_empty() {
                frame.next_instr = pc + 1;
                if pyre_interp::call::replay_pending_inline_call(
                    frame,
                    argc.get(op_arg) as usize,
                ) {
                    continue;
                }
                frame.next_instr = pc;
            }
        }
        frame.next_instr += 1;
        let next_instr = frame.next_instr;
        if let pyre_bytecode::bytecode::Instruction::Call { argc } = instruction {
            if pyre_interp::call::replay_pending_inline_call(frame, argc.get(op_arg) as usize) {
                continue;
            }
        }
        match execute_opcode_step(frame, code, instruction, op_arg, next_instr)? {
            StepResult::Continue => {}
            StepResult::CloseLoop { loop_header_pc, .. } if is_portal => {
                let mut jit_state = build_jit_state(frame, info);
                let green_key = make_green_key(frame.code, loop_header_pc);
                if majit_meta::majit_log_enabled() {
                    eprintln!(
                        "[jit][root-backedge] enter key={} pc={} arg0={:?} has_compiled={}",
                        green_key,
                        loop_header_pc,
                        debug_first_arg_int(frame),
                        driver.has_compiled_loop(green_key)
                    );
                }
                let outcome = if driver.has_compiled_loop(green_key) {
                    if majit_meta::majit_log_enabled() {
                        eprintln!("[jit] ENTERING run_compiled_with_bridge key={}", green_key);
                    }
                    Some(driver.run_compiled_detailed_with_bridge_keyed(
                        green_key,
                        loop_header_pc,
                        &mut jit_state,
                        &env,
                        || {},
                        restore_guard_failure_for_loop,
                    ))
                } else {
                    driver.back_edge_or_run_compiled_keyed(
                        green_key,
                        loop_header_pc,
                        &mut jit_state,
                        &env,
                        || {},
                    )
                };
                if let Some(outcome) = outcome {
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
                            "[jit][root-backedge] outcome key={} pc={} kind={}",
                            green_key, loop_header_pc, kind
                        );
                    }
                    if let Some(result) =
                        handle_jit_outcome(outcome, &jit_state, frame, info, green_key)
                    {
                        return result;
                    }
                }
                if !driver.is_tracing() {
                    JIT_TRACING.with(|t| t.set(false));
                }
            }
            StepResult::CloseLoop { .. } => {
                // Non-portal (module-level) backedge — no JIT dispatch
            }
            StepResult::Return(result) => return Ok(result),
            StepResult::Yield(result) => return Ok(result),
        }
    }
}

/// Try running compiled code or entering the recursive function-entry path.
pub fn try_function_entry_jit(frame: &mut PyFrame) -> Option<PyResult> {
    // RPython parity: blackhole's bhimpl_recursive_call calls portal_runner
    // which CAN enter JIT for callee frames. Only the force callback's OWN
    // frame runs as interpreter (eval_loop_for_force); nested calls freely
    // enter compiled code. No in_blackhole_entry() check needed here.
    let green_key = make_green_key(frame.code, frame.next_instr);

    // Process deferred bridge compile requests from call_assembler
    // guard failures. Must be done BEFORE driver_pair() to avoid
    // double mutable borrow (jit_bridge_compile_for_guard also borrows).
    let (driver, info) = driver_pair();

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
            if let Some(result) =
                handle_jit_outcome(outcome, &jit_state, frame, info, green_key)
            {
                if majit_meta::majit_log_enabled() {
                    let rendered = result.as_ref().ok().and_then(|value| {
                        if value.is_null() || !unsafe { pyre_object::pyobject::is_int(*value) } {
                            return None;
                        }
                        Some(unsafe { pyre_object::intobject::w_int_get_value(*value) })
                    });
                    eprintln!(
                        "[jit][func-entry] compiled return key={} arg0={:?} result={:?}",
                        green_key,
                        debug_first_arg_int(frame),
                        rendered
                    );
                }
                return Some(result);
            }
        }
        // After compiled code guard-restored fallback, re-establish the
        // frame's array pointer. Compiled code may have read the ptr field
        // but virtualizable sync does not write it; however fix_array_ptrs
        // ensures inline-mode consistency for the interpreter.
        frame.fix_array_ptrs();
        return None;
    }

    if majit_meta::majit_log_enabled() {
        eprintln!(
            "[jit][func-entry] probe key={} arg0={:?} tracing={} recursive_entry={} self_recursive_candidate={}",
            green_key,
            debug_first_arg_int(frame),
            driver.is_tracing(),
            in_recursive_force_entry(),
            crate::call_jit::self_recursive_function_entry_candidate(frame),
        );
    }

    if driver.is_tracing() {
        return None;
    }

    let self_recursive_candidate = crate::call_jit::self_recursive_function_entry_candidate(frame);
    if !driver.is_function_boosted(green_key) && self_recursive_candidate {
        // WarmState boosts are "trace the next entry quickly", not "trace the
        // current entry immediately". Mirror that RPython behavior here so
        // the first outermost recursive call only seeds the counter and the
        // next recursive portal entry starts the separate function trace.
        driver.boost_function_entry(green_key);
        return None;
    }
    let should_trace = driver
        .meta_interp_mut()
        .warm_state_mut()
        .should_trace_function_entry(green_key);
    let count = driver.meta_interp().warm_state_ref().function_entry_count(green_key);
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
) -> Option<PyResult> {
    match outcome {
        DetailedDriverRunOutcome::Finished {
            typed_values,
            raw_int_result,
            ..
        } => {
            let (driver, _) = driver_pair();
            let raw_int_result = raw_int_result
                || driver.has_raw_int_finish(green_key)
                || has_finish_protocol_hint(green_key);
            if majit_meta::majit_log_enabled() {
                eprintln!(
                    "[jit][handle-outcome] finished key={} raw_flag={} typed_values={:?}",
                    green_key, raw_int_result, typed_values
                );
            }
            let [value] = typed_values.as_slice() else {
                return Some(Err(pyre_runtime::PyError::type_error(
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
                    return Some(Err(pyre_runtime::PyError::type_error(
                        "compiled finish produced a void return value",
                    )));
                }
            };
            Some(Ok(value))
        }
        DetailedDriverRunOutcome::Jump { .. } => {
            // RPython run_compiled path already restored the loop-header state
            // into the virtualizable before returning a Jump outcome.
            // Do not perform an extra interpreter-side writeback here.
            let _ = frame;
            None
        }
        DetailedDriverRunOutcome::GuardFailure { restored: true, .. } => {
            // Guard-recovery paths already rebuild the restored frame state
            // before returning to the interpreter. Avoid a second writeback.
            let _ = frame;
            None
        }
        DetailedDriverRunOutcome::GuardFailure {
            restored: false, ..
        }
        | DetailedDriverRunOutcome::Abort { .. } => None,
    }
}

fn decode_exit_layout_values(
    raw_values: &[i64],
    layout: &CompiledExitLayout,
) -> Vec<Value> {
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
        let nraw = raw_values.len().min(16);
        let slots: Vec<String> = (0..nraw).map(|i| format!("{:#x}", raw_values[i] as usize)).collect();
        eprintln!(
            "[jit] guard-fail: fail_idx={} types={:?} raw=[{}]",
            exit_layout.fail_index, exit_layout.exit_types, slots.join(", ")
        );
    }
    let mut typed = decode_exit_layout_values(raw_values, exit_layout);
    // RPython resume.py: materialize virtual objects from recovery_layout.
    // Virtual slots in fail_args are NONE (null Ref); their field values
    // are stored as extra fail_args. Reconstruct the concrete PyObject
    // from the field values and replace the null slot.
    materialize_recovery_virtuals(&mut typed, exit_layout);
    let restored = jit_state.restore_guard_failure_values(meta, &typed, &ExceptionState::default());
    if majit_meta::majit_log_enabled() {
        eprintln!("[jit] guard-fail restored: ni={} vsd={}", jit_state.next_instr, jit_state.valuestackdepth);
    }
    restored.then_some(jit_state.next_instr)
}

/// RPython resume.py parity: reconstruct virtual objects from their
/// field values stored as extra fail_args after null (NONE) slots.
///
/// When the optimizer encodes a virtual in fail_args (via rd_virtuals),
/// it sets the virtual's slot to NONE and appends the field values.
/// On guard failure, we detect null Ref slots and pair them with
/// their field values to reconstruct concrete PyObjects.
///
/// Pattern: [header...] [non-virtual slots] [NONE, NONE, ...] [ob_type1, intval1, ob_type2, intval2, ...]
/// Each null Ref slot consumes 2 field values (ob_type + intval) from the tail.
fn materialize_recovery_virtuals(
    typed: &mut Vec<Value>,
    _exit_layout: &CompiledExitLayout,
) {
    // Find all null Ref slots (virtual markers)
    let null_slots: Vec<usize> = (3..typed.len())
        .filter(|&i| matches!(typed.get(i), Some(Value::Ref(majit_ir::GcRef(0)))))
        .collect();
    if null_slots.is_empty() {
        return;
    }
    // The extra field values start after the last non-virtual slot.
    // Each virtual has 2 fields: ob_type (Int) and intval (Int).
    let first_extra = null_slots.last().map(|&s| s + 1).unwrap_or(typed.len());
    let mut field_cursor = first_extra;
    for &slot_idx in &null_slots {
        // Consume 2 fields: ob_type and intval
        let _ob_type_pos = field_cursor;
        let intval_pos = field_cursor + 1;
        if intval_pos >= typed.len() {
            break;
        }
        if let Value::Int(v) = typed[intval_pos] {
            let obj = pyre_object::intobject::w_int_new(v);
            typed[slot_idx] = Value::Ref(majit_ir::GcRef(obj as usize));
            if majit_meta::majit_log_enabled() {
                eprintln!("[jit] materialized virtual W_IntObject(intval={}) at slot {}", v, slot_idx);
            }
        }
        field_cursor += 2; // skip ob_type + intval
    }
    // Truncate extra field values so restore_guard_failure_values
    // doesn't count them toward valuestackdepth.
    typed.truncate(first_extra);
}

fn build_jit_state(
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
                driver.has_raw_int_finish(fib_key) || has_finish_protocol_hint(fib_key),
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
