//! JIT-enabled evaluation — the sole entry point for JIT execution.
//!
//! This module owns the JitDriver, tracing hooks, and compiled-code
//! execution. pyre-interp provides the pure interpreter (eval_frame_plain)
//! and the opcode trait implementations on PyFrame.
//!
//! Equivalent to PyPy's `pypyjit/interp_jit.py` — the JIT is injected
//! from outside the interpreter.

use std::cell::{Cell, UnsafeCell};
use std::collections::HashMap;
use crate::jit::state::{PyreEnv, PyreJitState};
use crate::jit::trace::trace_bytecode;
use pyre_bytecode::bytecode::OpArgState;
use pyre_interp::frame::PyFrame;
use pyre_object::w_none;
use pyre_runtime::{PyResult, StepResult, execute_opcode_step};

use majit_gc::trace::TypeInfo;
use majit_meta::DetailedDriverRunOutcome;

use crate::jit::frame_layout::build_pyframe_virtualizable_info;
use crate::jit::descr::{W_FLOAT_GC_TYPE_ID, W_INT_GC_TYPE_ID};
use majit_gc::collector::MiniMarkGC;
use majit_meta::JitDriver;
use pyre_object::floatobject::W_FloatObject;
use pyre_object::intobject::W_IntObject;

const JIT_THRESHOLD: u32 = 1039;
/// Function-entry tracing threshold.
/// PyPy uses WarmEnterState.function_threshold (default 1619).
/// We use a lower value for faster JIT warmup in pyre's use case.
/// TODO: read from driver.warm_state().function_threshold() instead.
const FUNC_ENTRY_THRESHOLD: u32 = 7;

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
    static FUNC_ENTRY_COUNTS: UnsafeCell<HashMap<u64, u32>> =
        UnsafeCell::new(HashMap::new());

    static JIT_CALL_DEPTH: Cell<u32> = Cell::new(0);
    static JIT_TRACING: Cell<bool> = Cell::new(false);
    static RECURSIVE_FORCE_ENTRY_DEPTH: Cell<u32> = Cell::new(0);
    static BLACKHOLE_ENTRY_DEPTH: Cell<u32> = Cell::new(0);
}

#[inline]
fn func_entry_counts() -> &'static mut HashMap<u64, u32> {
    FUNC_ENTRY_COUNTS.with(|cell| unsafe { &mut *cell.get() })
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
#[inline]
pub fn jit_call_depth_bump() -> Option<JitCallDepthGuard> {
    let depth = JIT_CALL_DEPTH.with(|d| d.get());
    if depth > 0 || JIT_TRACING.with(|t| t.get()) {
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

    // Always use JIT-enabled loop — it handles back-edges, merge
    // points, and compiled-code execution. eval_frame_plain (no JIT)
    // is only used for residual calls (jit_call_depth > 0).
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
    let env = PyreEnv;
    let mut arg_state = OpArgState::default();
    let code = unsafe { &*frame.code };
    let (driver, info) = driver_pair();

    loop {
        if frame.next_instr >= code.instructions.len() {
            return Ok(w_none());
        }

        // PyPy interp_jit.py:85-87 — jit_merge_point on EVERY iteration.
        // pypyjitdriver.jit_merge_point(ec=ec, frame=self, next_instr=next_instr,
        //     pycode=pycode, is_being_profiled=is_being_profiled)
        if JIT_CALL_DEPTH.with(|d| d.get()) == 0 && driver.is_tracing() {
            JIT_TRACING.with(|t| t.set(true));
            let pc = frame.next_instr;
            let concrete_frame = frame as *mut PyFrame as usize;
            driver.merge_point(|ctx, sym| trace_bytecode(ctx, sym, code, pc, concrete_frame));
            if !driver.is_tracing() {
                JIT_TRACING.with(|t| t.set(false));
            }
        }

        let code_unit = code.instructions[frame.next_instr];
        let (instruction, op_arg) = arg_state.get(code_unit);
        frame.next_instr += 1;
        let next_instr = frame.next_instr;
        match execute_opcode_step(frame, code, instruction, op_arg, next_instr)? {
            StepResult::Continue => {}
            StepResult::CloseLoop(_) => {
                let mut jit_state = build_jit_state(frame, info);
                // RPython green_key = (pycode, next_instr).
                // Each (code, pc) pair has independent warmup counter
                // and compiled loop, so inner/outer loops are separate.
                let green_key =
                    make_green_key(frame.code, frame.next_instr);
                if let Some(outcome) = driver.back_edge_or_run_compiled_keyed(
                    green_key,
                    frame.next_instr,
                    &mut jit_state,
                    &env,
                    || {},
                ) {
                    if let Some(result) = handle_jit_outcome(outcome, &jit_state, frame, info) {
                        return result;
                    }
                }
            }
            StepResult::Return(result) => return Ok(result),
            StepResult::Yield(result) => return Ok(result),
        }
    }
}

/// Try running compiled code or entering the recursive function-entry path.
pub fn try_function_entry_jit(frame: &mut PyFrame) -> Option<PyResult> {
    // RPython parity: blackhole interpreter has no JIT entry points.
    // When executing inside a force callback (guard failure recovery),
    // never re-enter compiled code. RPython achieves this structurally
    // (blackhole dispatch loop has no can_enter_jit / jit_merge_point).
    // pyre uses a thread-local flag until full structural isolation is
    // implemented.
    if in_blackhole_entry() {
        return None;
    }

    let green_key = make_green_key(frame.code, frame.next_instr);
    let (driver, info) = driver_pair();

    if driver.has_compiled_loop(green_key) {
        if majit_meta::majit_log_enabled() {
            eprintln!(
                "[jit][func-entry] run compiled key={} arg0={:?} depth={}",
                green_key,
                debug_first_arg_int(frame),
                JIT_CALL_DEPTH.with(|d| d.get())
            );
        }
        let env = PyreEnv;
        let mut jit_state = build_jit_state(frame, info);
        if let Some(outcome) = driver.back_edge_or_run_compiled_keyed(
            green_key,
            frame.next_instr,
            &mut jit_state,
            &env,
            || {},
        ) {
            if majit_meta::majit_log_enabled() {
                let kind = match &outcome {
                    DetailedDriverRunOutcome::Finished { .. } => "finished",
                    DetailedDriverRunOutcome::Jump { .. } => "jump",
                    DetailedDriverRunOutcome::Abort { .. } => "abort",
                    DetailedDriverRunOutcome::GuardFailure { restored: true, .. } => {
                        "guard-restored"
                    }
                    DetailedDriverRunOutcome::GuardFailure { restored: false, .. } => {
                        "guard-unrestored"
                    }
                };
                eprintln!(
                    "[jit][func-entry] compiled outcome key={} arg0={:?} kind={}",
                    green_key,
                    debug_first_arg_int(frame),
                    kind
                );
            }
            if let Some(result) = handle_jit_outcome(outcome, &jit_state, frame, info) {
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
        return None;
    }

    if driver.is_tracing() {
        return None;
    }

    let counts = func_entry_counts();
    let count = counts.entry(green_key).or_insert(0);
    *count += 1;
    let recursive_entry = in_recursive_force_entry();
    let self_recursive_candidate = crate::call_jit::self_recursive_function_entry_candidate(frame);
    let boosted = if driver.is_function_boosted(green_key) {
        true
    } else if *count == 1
        && recursive_entry
        && self_recursive_candidate
    {
        // PyPy converges recursive hot paths by boosting the callee toward a
        // separate function-entry trace once recursion becomes obvious.  We
        // only fast-track entries reached through our recursive force helper,
        // not the outermost user call, so warmup stays close to PyPy's
        // "recursion became hot" behavior.
        driver.boost_function_entry(green_key);
        true
    } else {
        false
    };

    if *count < FUNC_ENTRY_THRESHOLD && !boosted {
        return None;
    }

    let env = PyreEnv;
    let mut jit_state = build_jit_state(frame, info);
    if majit_meta::majit_log_enabled() {
        eprintln!(
            "[jit][func-entry] start tracing key={} arg0={:?} count={} boosted={}",
            green_key,
            debug_first_arg_int(frame),
            *count,
            boosted
        );
    }
    driver.force_start_tracing(green_key, frame.next_instr, &mut jit_state, &env);
    None
}

fn handle_jit_outcome(
    outcome: DetailedDriverRunOutcome,
    jit_state: &PyreJitState,
    frame: &mut PyFrame,
    info: &majit_meta::virtualizable::VirtualizableInfo,
) -> Option<PyResult> {
    match outcome {
        DetailedDriverRunOutcome::Finished {
            typed_values,
            raw_int_result,
            ..
        } => {
            let [value] = typed_values.as_slice() else {
                return Some(Err(pyre_runtime::PyError::type_error(
                    "compiled finish did not produce a single object return value",
                )));
            };
            let value = match value {
                majit_ir::Value::Int(raw) => {
                    if raw_int_result {
                        // Re-box: the Finish was unboxed for the raw-int
                        // CallAssembler protocol. Top-level exit must re-box.
                        pyre_object::intobject::w_int_new(*raw)
                    } else {
                        *raw as pyre_object::PyObjectRef
                    }
                }
                majit_ir::Value::Ref(value) => value.as_usize() as pyre_object::PyObjectRef,
                _ => {
                    return Some(Err(pyre_runtime::PyError::type_error(
                        "compiled finish produced a non-object return value",
                    )));
                }
            };
            Some(Ok(value))
        }
        DetailedDriverRunOutcome::Jump { .. } => {
            // Jump = guard failure that restored frame state.
            // Return None to let the caller's interpreter loop continue.
            // in_blackhole_entry() prevents JIT re-entry from nested calls.
            sync_jit_state_to_frame(jit_state, frame, info);
            None
        }
        DetailedDriverRunOutcome::GuardFailure { restored: true, .. } => {
            // RPython parity: guard failure → resume_in_blackhole().
            // The remaining execution runs in the blackhole with NO
            // JIT re-entry.
            sync_jit_state_to_frame(jit_state, frame, info);
            let result = crate::call_jit::resume_in_blackhole_pub(frame);
            Some(Ok(result))
        }
        DetailedDriverRunOutcome::GuardFailure {
            restored: false, ..
        }
        | DetailedDriverRunOutcome::Abort { .. } => None,
    }
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
}
