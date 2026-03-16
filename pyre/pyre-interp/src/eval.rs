//! Bytecode evaluation loop with JIT integration.
//!
//! The main dispatch loop processes RustPython instructions.
//! `JitDriver` merge points and back-edge hooks are integrated from day 1.
//!
//! The JitDriver lives in an `UnsafeCell` to allow re-entrant access from
//! recursive function calls. When compiled code makes a residual call,
//! the callee's `eval_frame` can access the same driver for function-entry
//! JIT — matching PyPy's single-MetaInterp-per-thread model.

use pyre_bytecode::bytecode::{BinaryOperator, ComparisonOperator, Instruction, OpArgState};
use pyre_object::*;
use pyre_objspace::*;
use pyre_runtime::{
    ArithmeticOpcodeHandler, BranchOpcodeHandler, ConstantOpcodeHandler, ControlFlowOpcodeHandler,
    IterOpcodeHandler, LocalOpcodeHandler, NamespaceOpcodeHandler, OpcodeStepExecutor, PyError,
    PyErrorKind, PyResult, SharedOpcodeHandler, StackOpcodeHandler, StepResult, TruthOpcodeHandler,
    build_list_from_refs, build_map_from_refs, build_tuple_from_refs, ensure_range_iter,
    execute_opcode_step, make_function_from_code_obj, namespace_load, namespace_store,
    range_iter_continues, range_iter_next_or_null, stack_underflow_error, unpack_sequence_exact,
    w_code_new,
};

use std::cell::{Cell, UnsafeCell};
use std::collections::HashMap;

use crate::jit::state::{PyreEnv, PyreJitState};
use crate::jit::trace::trace_bytecode;
use majit_meta::{DetailedDriverRunOutcome, JitDriver};

use crate::call::{call_callable, install_jit_call_bridge};
use crate::frame::{PyFrame, build_pyframe_virtualizable_info};

/// JIT hot-count threshold for back-edge (loop) detection.
const JIT_THRESHOLD: u32 = 1039;

/// Function-entry tracing threshold. Kept low so that tracing starts
/// early in a deep recursion (when n is still large), capturing the
/// recursive path rather than the base case.
const FUNC_ENTRY_THRESHOLD: u32 = 7;

type JitDriverPair = (
    JitDriver<PyreJitState>,
    majit_meta::virtualizable::VirtualizableInfo,
);

thread_local! {
    /// Shared JitDriver accessed via `UnsafeCell` to allow re-entrant
    /// access from recursive function calls.
    ///
    /// SAFETY: thread_local guarantees single-thread access. Re-entrant
    /// calls through compiled-code residual calls create overlapping
    /// `&mut` references on the stack, but the outer reference is never
    /// accessed while the inner one is active — the JitDriver's state
    /// is consistent at each re-entry point.
    static JIT_DRIVER: UnsafeCell<JitDriverPair> = UnsafeCell::new({
        let info = build_pyframe_virtualizable_info();
        let mut d = JitDriver::new(JIT_THRESHOLD);
        d.set_virtualizable_info(info.clone());
        (d, info)
    });
}

thread_local! {
    /// Per-code-object function-entry counter. Tracks how many times
    /// each function has been entered, using the code object pointer
    /// as key. Only when the count reaches JIT_THRESHOLD do we invoke
    /// the expensive driver path (build_meta, extract_live, etc.).
    static FUNC_ENTRY_COUNTS: UnsafeCell<HashMap<u64, u32>> =
        UnsafeCell::new(HashMap::new());

    /// JIT call depth counter. Tracks how deep we are in nested
    /// eval_frame calls due to JIT activity (tracing or compiled code).
    ///
    /// When > 0, this eval_loop is executing a residual call from
    /// either the traced function or compiled code. Effects:
    ///   - merge_points are skipped (inner instructions must not
    ///     pollute the outer trace)
    ///   - try_function_entry_jit is skipped (prevents infinite
    ///     recursion: compiled fib → residual call → run compiled
    ///     fib → residual call → ...)
    ///
    /// Matches PyPy's model where residual calls run in a
    /// "blackhole" interpreter without JIT hooks.
    /// JIT call depth counter. When > 0, this eval_loop is executing
    /// a residual call — merge_points and function-entry JIT are
    /// suppressed.
    static JIT_CALL_DEPTH: Cell<u32> = Cell::new(0);

    /// Lightweight tracing-active flag. Mirrors `driver.is_tracing()`
    /// to avoid the expensive UnsafeCell TLS lookup per call.
    static JIT_TRACING: Cell<bool> = Cell::new(false);
}

/// Get a mutable reference to the thread-local JitDriver pair.
///
/// SAFETY: Only call from `eval_frame` and its JIT helper functions.
/// The caller must ensure that no other live `&mut` reference to the
/// driver is being actively used (suspended references on the call
/// stack from re-entrant compiled-code execution are acceptable).
#[inline]
pub(crate) fn driver_pair() -> &'static mut JitDriverPair {
    JIT_DRIVER.with(|cell| unsafe { &mut *cell.get() })
}

#[inline]
#[allow(dead_code)]
fn func_entry_counts() -> &'static mut HashMap<u64, u32> {
    FUNC_ENTRY_COUNTS.with(|cell| unsafe { &mut *cell.get() })
}

/// RAII guard that decrements `JIT_CALL_DEPTH` on drop.
pub struct JitCallDepthGuard;

impl Drop for JitCallDepthGuard {
    fn drop(&mut self) {
        JIT_CALL_DEPTH.with(|d| d.set(d.get() - 1));
    }
}

/// Bump the JIT call depth. Returns a guard that restores the
/// depth when dropped. Only bumps if we're inside a JIT context
/// (depth > 0 or tracing active).
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

/// Execute a Python code object and return its result.
///
/// The returned value is the result of `ReturnValue`,
/// or `None` if the code falls through without returning.
pub fn eval_frame(frame: &mut PyFrame) -> PyResult {
    install_jit_call_bridge();
    frame.fix_array_ptrs();
    if let Some(result) = try_function_entry_jit(frame) {
        return result;
    }
    // If function-entry triggered tracing, use the JIT-enabled loop
    // so merge_point records trace ops. Without this, eval_loop would
    // skip merge points and the trace would never complete.
    let (driver, _) = driver_pair();
    if driver.is_tracing() {
        return eval_loop_jit(frame);
    }
    eval_loop(frame)
}

/// Run the interpreter on a frame whose state was partially updated by
/// compiled code (call_assembler guard failure forcing).
///
/// Skips the JIT function-entry check and runs the plain eval_loop
/// from the frame's current next_instr.
pub(crate) fn eval_loop_for_force(frame: &mut PyFrame) -> PyResult {
    eval_loop(frame)
}

/// Evaluation loop with deferred JIT initialization.
///
/// The JitDriver TLS lookup is deferred until the first back-edge
/// (CloseLoop). Functions without loops (like fib) never pay the
/// driver_pair() TLS cost. Once a back-edge is seen, we switch to
/// the full JIT-enabled loop for the remainder of the function.
fn eval_loop(frame: &mut PyFrame) -> PyResult {
    let mut arg_state = OpArgState::default();
    let code = unsafe { &*frame.code };

    loop {
        if frame.next_instr >= code.instructions.len() {
            return Ok(w_none());
        }

        let code_unit = code.instructions[frame.next_instr];
        let (instruction, op_arg) = arg_state.get(code_unit);
        frame.next_instr += 1;
        let next_instr = frame.next_instr;
        match execute_opcode_step(frame, &code, instruction, op_arg, next_instr)? {
            StepResult::Continue => {}
            StepResult::CloseLoop(_) => {
                // First back-edge: switch to the JIT-enabled loop
                // that fetches the driver and handles merge points.
                return eval_loop_jit(frame);
            }
            StepResult::Return(result) => return Ok(result),
        }
    }
}

/// JIT-enabled evaluation loop, entered after the first back-edge.
///
/// The JitDriver reference is obtained once via TLS lookup and held
/// as a local variable. Recursive calls through `call_callable`
/// re-enter `eval_frame` → `eval_loop` (the deferred variant),
/// creating a new `&mut` alias only if they also hit a back-edge.
fn eval_loop_jit(frame: &mut PyFrame) -> PyResult {
    let env = PyreEnv;
    let mut arg_state = OpArgState::default();
    let code = unsafe { &*frame.code };
    let (driver, info) = driver_pair();

    loop {
        if frame.next_instr >= code.instructions.len() {
            return Ok(w_none());
        }

        // ── JIT merge point ──────────────────────────────────────
        // Only record trace ops at depth 0 (the function being traced).
        // Deeper levels are residual calls whose instructions must not
        // pollute the outer trace.
        if driver.is_tracing() {
            JIT_TRACING.with(|t| t.set(true));
            if JIT_CALL_DEPTH.with(|d| d.get()) == 0 {
                let pc = frame.next_instr;
                let concrete_frame = frame as *mut PyFrame as usize;
                driver.merge_point(|ctx, sym| trace_bytecode(ctx, sym, code, pc, concrete_frame));
            }
            // Tracing may have ended inside merge_point (abort/compile).
            if !driver.is_tracing() {
                JIT_TRACING.with(|t| t.set(false));
            }
        }

        let code_unit = code.instructions[frame.next_instr];
        let (instruction, op_arg) = arg_state.get(code_unit);
        frame.next_instr += 1;
        let next_instr = frame.next_instr;
        match execute_opcode_step(frame, &code, instruction, op_arg, next_instr)? {
            StepResult::Continue => {}
            StepResult::CloseLoop(_) => {
                let mut jit_state = build_jit_state(frame, info);
                if let Some(outcome) =
                    driver.back_edge_or_run_compiled(frame.next_instr, &mut jit_state, &env, || {})
                {
                    if let Some(result) = handle_jit_outcome(outcome, &jit_state, frame, info) {
                        return result;
                    }
                    // Guard failure with restored state — continue interpreting
                }
            }
            StepResult::Return(result) => return Ok(result),
        }
    }
}

/// Try running compiled code or count function entry.
///
/// Uses the code object pointer as a green key, so all calls to the
/// same function share one counter and one compiled trace.
///
/// Fast path: increment a lightweight local counter. Only call into
/// the expensive driver path when the counter reaches the threshold
/// or when compiled code already exists for this key.
///
fn try_function_entry_jit(frame: &mut PyFrame) -> Option<PyResult> {
    // Residual calls (from tracing or compiled code) run in pure
    // interpreter — no JIT hooks. Matches PyPy's blackhole model.
    if JIT_CALL_DEPTH.with(|d| d.get()) > 0 {
        return None;
    }

    let green_key = frame.code as u64;
    let (driver, info) = driver_pair();

    // Run compiled code if available.
    if driver.has_compiled_loop(green_key) {
        let env = PyreEnv;
        let mut jit_state = build_jit_state(frame, info);
        if let Some(outcome) = driver.back_edge_or_run_compiled_keyed(
            green_key,
            frame.next_instr,
            &mut jit_state,
            &env,
            || {},
        ) {
            if let Some(result) = handle_jit_outcome(outcome, &jit_state, frame, info) {
                return Some(result);
            }
        }
        // Guard failure → fall through to interpreter
        return None;
    }

    // Fast path: if already tracing, skip counting.
    if driver.is_tracing() {
        return None;
    }

    // Lightweight local counting with a low threshold so that tracing
    // starts early in a deep recursion (capturing the recursive path).
    let counts = func_entry_counts();
    let count = counts.entry(green_key).or_insert(0);
    *count += 1;
    if *count != FUNC_ENTRY_THRESHOLD {
        return None;
    }

    // Threshold reached — force-start tracing (bypasses driver's
    // internal hot counter since we already did our own counting).
    let env = PyreEnv;
    let mut jit_state = build_jit_state(frame, info);
    driver.force_start_tracing(green_key, frame.next_instr, &mut jit_state, &env);
    None
}

/// Handle a `DetailedDriverRunOutcome` from compiled code execution.
fn handle_jit_outcome(
    outcome: DetailedDriverRunOutcome,
    jit_state: &PyreJitState,
    frame: &mut PyFrame,
    info: &majit_meta::virtualizable::VirtualizableInfo,
) -> Option<PyResult> {
    match outcome {
        DetailedDriverRunOutcome::Finished { typed_values, .. } => {
            let [value] = typed_values.as_slice() else {
                return Some(Err(PyError::type_error(
                    "compiled finish did not produce a single object return value",
                )));
            };
            let value = match value {
                majit_ir::Value::Int(raw) => *raw as PyObjectRef,
                majit_ir::Value::Ref(value) => value.as_usize() as PyObjectRef,
                _ => {
                    return Some(Err(PyError::type_error(
                        "compiled finish produced a non-object return value",
                    )));
                }
            };
            Some(Ok(value))
        }
        DetailedDriverRunOutcome::Jump { .. }
        | DetailedDriverRunOutcome::GuardFailure { restored: true, .. } => {
            sync_jit_state_to_frame(jit_state, frame, info);
            None // Continue interpretation
        }
        DetailedDriverRunOutcome::GuardFailure {
            restored: false, ..
        }
        | DetailedDriverRunOutcome::Abort { .. } => None,
    }
}

impl SharedOpcodeHandler for PyFrame {
    type Value = PyObjectRef;

    fn push_value(&mut self, value: Self::Value) -> Result<(), PyError> {
        PyFrame::push(self, value);
        Ok(())
    }

    fn pop_value(&mut self) -> Result<Self::Value, PyError> {
        if self.stack_depth == 0 {
            return Err(stack_underflow_error("interpreter opcode"));
        }
        Ok(PyFrame::pop(self))
    }

    fn peek_at(&mut self, depth: usize) -> Result<Self::Value, PyError> {
        if self.stack_depth <= depth {
            return Err(stack_underflow_error("interpreter peek"));
        }
        Ok(PyFrame::peek_at(self, depth))
    }

    fn make_function(&mut self, code_obj: Self::Value) -> Result<Self::Value, PyError> {
        Ok(make_function_from_code_obj(code_obj, self.namespace))
    }

    fn call_callable(
        &mut self,
        callable: Self::Value,
        args: &[Self::Value],
    ) -> Result<Self::Value, PyError> {
        call_callable(self, callable, args)
    }

    fn build_list(&mut self, items: &[Self::Value]) -> Result<Self::Value, PyError> {
        Ok(build_list_from_refs(items))
    }

    fn build_tuple(&mut self, items: &[Self::Value]) -> Result<Self::Value, PyError> {
        Ok(build_tuple_from_refs(items))
    }

    fn build_map(&mut self, items: &[Self::Value]) -> Result<Self::Value, PyError> {
        Ok(build_map_from_refs(items))
    }

    fn store_subscr(
        &mut self,
        obj: Self::Value,
        key: Self::Value,
        value: Self::Value,
    ) -> Result<(), PyError> {
        py_setitem(obj, key, value).map(|_| ())
    }

    fn list_append(&mut self, list: Self::Value, value: Self::Value) -> Result<(), PyError> {
        unsafe { w_list_append(list, value) };
        Ok(())
    }

    fn unpack_sequence(
        &mut self,
        seq: Self::Value,
        count: usize,
    ) -> Result<Vec<Self::Value>, PyError> {
        unpack_sequence_exact(seq, count)
    }
}

impl LocalOpcodeHandler for PyFrame {
    fn load_local_value(&mut self, idx: usize) -> Result<Self::Value, PyError> {
        Ok(self.locals_w[idx])
    }

    fn load_local_checked_value(&mut self, idx: usize, name: &str) -> Result<Self::Value, PyError> {
        let value = self.locals_w[idx];
        if value.is_null() {
            return Err(PyError {
                kind: PyErrorKind::NameError,
                message: format!("local variable '{name}' referenced before assignment"),
            });
        }
        Ok(value)
    }

    fn store_local_value(&mut self, idx: usize, value: Self::Value) -> Result<(), PyError> {
        self.locals_w[idx] = value;
        Ok(())
    }
}

impl NamespaceOpcodeHandler for PyFrame {
    fn load_name_value(&mut self, name: &str) -> Result<Self::Value, PyError> {
        let ns = unsafe { &*self.namespace };
        namespace_load(ns, name)
    }

    fn store_name_value(&mut self, name: &str, value: Self::Value) -> Result<(), PyError> {
        let ns = unsafe { &mut *self.namespace };
        namespace_store(ns, name, value);
        Ok(())
    }

    fn null_value(&mut self) -> Result<Self::Value, PyError> {
        Ok(PY_NULL)
    }
}

impl StackOpcodeHandler for PyFrame {
    fn swap_values(&mut self, depth: usize) -> Result<(), PyError> {
        let top_idx = self.stack_depth - 1;
        let other_idx = self.stack_depth - depth;
        self.value_stack_w.swap(top_idx, other_idx);
        Ok(())
    }
}

impl IterOpcodeHandler for PyFrame {
    fn ensure_iter_value(&mut self, iter: Self::Value) -> Result<(), PyError> {
        ensure_range_iter(iter)
    }

    fn concrete_iter_continues(&mut self, iter: Self::Value) -> Result<bool, PyError> {
        range_iter_continues(iter)
    }

    fn iter_next_value(&mut self, iter: Self::Value) -> Result<Self::Value, PyError> {
        range_iter_next_or_null(iter)
    }

    fn on_iter_exhausted(&mut self, target: usize) -> Result<(), PyError> {
        self.next_instr = target;
        Ok(())
    }
}

impl TruthOpcodeHandler for PyFrame {
    type Truth = bool;

    fn truth_value(&mut self, value: Self::Value) -> Result<Self::Truth, PyError> {
        Ok(truth_value(value))
    }

    fn bool_value_from_truth(
        &mut self,
        truth: Self::Truth,
        negate: bool,
    ) -> Result<Self::Value, PyError> {
        Ok(bool_value_from_truth(if negate { !truth } else { truth }))
    }
}

impl ControlFlowOpcodeHandler for PyFrame {
    fn fallthrough_target(&mut self) -> usize {
        self.next_instr
    }

    fn set_next_instr(&mut self, target: usize) -> Result<(), PyError> {
        self.next_instr = target;
        Ok(())
    }

    fn close_loop(&mut self, _target: usize) -> Result<StepResult<Self::Value>, PyError> {
        // Signal a back-edge to the main eval_loop, which handles
        // JIT counting and compiled code execution via try_back_edge_jit.
        Ok(StepResult::CloseLoop(vec![]))
    }
}

impl BranchOpcodeHandler for PyFrame {
    fn concrete_truth_as_bool(&mut self, truth: Self::Truth) -> Result<bool, PyError> {
        Ok(truth)
    }
}

impl ArithmeticOpcodeHandler for PyFrame {
    fn binary_value(
        &mut self,
        a: Self::Value,
        b: Self::Value,
        op: BinaryOperator,
    ) -> Result<Self::Value, PyError> {
        binary_value(a, b, op)
    }

    fn compare_value(
        &mut self,
        a: Self::Value,
        b: Self::Value,
        op: ComparisonOperator,
    ) -> Result<Self::Value, PyError> {
        compare_value(a, b, op)
    }

    fn unary_negative_value(&mut self, value: Self::Value) -> Result<Self::Value, PyError> {
        unary_negative_value(value)
    }

    fn unary_invert_value(&mut self, value: Self::Value) -> Result<Self::Value, PyError> {
        unary_invert_value(value)
    }
}

impl ConstantOpcodeHandler for PyFrame {
    fn int_constant(&mut self, value: i64) -> Result<Self::Value, PyError> {
        Ok(w_int_new(value))
    }

    fn bigint_constant(&mut self, value: &pyre_runtime::PyBigInt) -> Result<Self::Value, PyError> {
        Ok(w_long_new(value.clone()))
    }

    fn float_constant(&mut self, value: f64) -> Result<Self::Value, PyError> {
        Ok(w_float_new(value))
    }

    fn bool_constant(&mut self, value: bool) -> Result<Self::Value, PyError> {
        Ok(w_bool_from(value))
    }

    fn str_constant(&mut self, value: &str) -> Result<Self::Value, PyError> {
        Ok(box_str_constant(value))
    }

    fn code_constant(
        &mut self,
        code: &pyre_bytecode::bytecode::CodeObject,
    ) -> Result<Self::Value, PyError> {
        let code_ptr = Box::into_raw(Box::new(code.clone())) as *const ();
        Ok(w_code_new(code_ptr))
    }

    fn none_constant(&mut self) -> Result<Self::Value, PyError> {
        Ok(w_none())
    }
}

impl OpcodeStepExecutor for PyFrame {
    type Error = PyError;

    fn unsupported(
        &mut self,
        instruction: &Instruction,
    ) -> Result<StepResult<PyObjectRef>, Self::Error> {
        Err(PyError::type_error(format!(
            "unimplemented instruction: {instruction:?}"
        )))
    }
}

// ── JitState ↔ PyFrame conversion ────────────────────────────────────

/// Build a `PyreJitState` from the current frame state.
///
/// Namespace entries are sorted by key for deterministic ordering.
fn build_jit_state(
    frame: &PyFrame,
    virtualizable_info: &majit_meta::virtualizable::VirtualizableInfo,
) -> PyreJitState {
    let mut jit_state = PyreJitState {
        frame: frame as *const PyFrame as usize,
        next_instr: frame.next_instr,
        stack_depth: frame.stack_depth,
    };

    if !jit_state.sync_from_virtualizable(virtualizable_info) {
        jit_state.next_instr = frame.next_instr;
        jit_state.stack_depth = frame.stack_depth;
    }

    jit_state
}

/// Restore frame state from a `PyreJitState` after JIT code execution.
fn sync_jit_state_to_frame(
    jit_state: &PyreJitState,
    frame: &mut PyFrame,
    virtualizable_info: &majit_meta::virtualizable::VirtualizableInfo,
) {
    if !jit_state.sync_to_virtualizable(virtualizable_info) {
        frame.next_instr = jit_state.next_instr;
        frame.stack_depth = jit_state.stack_depth;
    }

    frame.next_instr = jit_state.next_instr;
    frame.stack_depth = jit_state.stack_depth;
}

#[cfg(test)]
mod tests {
    use super::*;
    use majit_ir::Value;
    use majit_meta::JitState;
    use majit_meta::blackhole::ExceptionState;
    use pyre_bytecode::*;
    use pyre_runtime::{PyExecutionContext, w_func_new};
    use std::rc::Rc;

    fn run_eval(source: &str) -> PyResult {
        let code = compile_eval(source).expect("compile failed");
        let mut frame = PyFrame::new_with_context(code, Rc::new(PyExecutionContext::default()));
        eval_frame(&mut frame)
    }

    fn run_exec_frame(source: &str) -> (PyResult, PyFrame) {
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new_with_context(code, Rc::new(PyExecutionContext::default()));
        let result = eval_frame(&mut frame);
        (result, frame)
    }

    fn nested_function_code(source: &str) -> CodeObject {
        let module = compile_exec(source).expect("compile failed");
        module
            .constants
            .iter()
            .find_map(|constant| match constant {
                ConstantData::Code { code } => Some(code.as_ref().clone()),
                _ => None,
            })
            .expect("expected nested function code object")
    }

    #[test]
    fn test_jit_function_helper_reuses_interpreter_call_semantics() {
        install_jit_call_bridge();

        let caller_code = compile_exec("z = 2").expect("compile failed");
        let caller = PyFrame::new_with_context(caller_code, Rc::new(PyExecutionContext::default()));
        unsafe { (*caller.namespace).insert("z".to_string(), w_int_new(2)) };

        let code = nested_function_code("z = 2\ndef f(x):\n    return x + z");
        let code_ptr = Box::into_raw(Box::new(code)) as *const ();
        let callable = w_func_new(code_ptr, "f".to_string(), caller.namespace);

        let result = crate::jit::helpers::jit_call_callable_1(
            (&caller as *const PyFrame) as i64,
            callable as i64,
            w_int_new(40) as i64,
        );

        unsafe {
            assert_eq!(w_int_get_value(result as PyObjectRef), 42);
        }
    }

    #[test]
    fn test_jit_namespace_helpers_read_and_write_frame_namespace() {
        let code = compile_exec("i = 0").expect("compile failed");
        let frame = PyFrame::new_with_context(code, Rc::new(PyExecutionContext::default()));
        unsafe { (*frame.namespace).insert("i".to_string(), w_int_new(7)) };

        let name = frame.code().names[0].to_string();
        let namespace_ptr = frame.namespace as i64;
        let name_ptr = name.as_ptr() as i64;
        let name_len = name.len() as i64;

        let loaded =
            crate::jit::helpers::jit_load_name_from_namespace(namespace_ptr, name_ptr, name_len);
        unsafe {
            assert_eq!(w_int_get_value(loaded as PyObjectRef), 7);
        }

        crate::jit::helpers::jit_store_name_to_namespace(
            namespace_ptr,
            name_ptr,
            name_len,
            w_int_new(11) as i64,
        );
        unsafe {
            assert_eq!(w_int_get_value(*(*frame.namespace).get("i").unwrap()), 11);
        }
    }

    #[test]
    fn test_dump_for_bytecode() {
        let source = "s = 0\nfor i in range(10):\n    s = s + i";
        let code = compile_exec(source).expect("compile failed");
        eprintln!("Instructions for 'for i in range(10)':");
        for (i, cu) in code.instructions.iter().enumerate() {
            let mut s = pyre_bytecode::bytecode::OpArgState::default();
            let (ins, oparg) = s.get(*cu);
            eprintln!("{i:3}: {ins:?}  (arg={})", u32::from(oparg));
        }
        eprintln!("\nNames: {:?}", code.names);
        eprintln!("Varnames: {:?}", code.varnames);
    }

    #[test]
    fn test_build_jit_state_imports_virtualizable_frame_state() {
        let code = nested_function_code("def f(x):\n    y = x\n    return y");
        let mut frame = PyFrame::new_with_context(code, Rc::new(PyExecutionContext::default()));
        frame.fix_array_ptrs();
        frame.next_instr = 7;
        frame.stack_depth = 1;
        frame.locals_w[0] = w_int_new(11);
        frame.locals_w[1] = w_int_new(13);
        frame.value_stack_w[0] = w_int_new(29);
        unsafe { (*frame.namespace).insert("x".to_string(), w_int_new(41)) };

        let info = build_pyframe_virtualizable_info();
        let jit_state = build_jit_state(&frame, &info);

        assert_eq!(jit_state.frame, &frame as *const PyFrame as usize);
        assert_eq!(jit_state.next_instr, 7);
        assert_eq!(jit_state.stack_depth, 1);
        assert_eq!(jit_state.local_count(), frame.locals_w.len());
        assert_eq!(jit_state.stack_capacity(), frame.value_stack_w.len());
        unsafe {
            assert_eq!(w_int_get_value(jit_state.local_at(0).unwrap()), 11);
            assert_eq!(w_int_get_value(jit_state.local_at(1).unwrap()), 13);
            assert_eq!(w_int_get_value(jit_state.stack_at(0).unwrap()), 29);
        }
    }

    #[test]
    fn test_sync_jit_state_to_frame_exports_virtualizable_state() {
        let code = nested_function_code("def f(x):\n    y = x\n    return y");
        let mut frame = PyFrame::new_with_context(code, Rc::new(PyExecutionContext::default()));
        frame.fix_array_ptrs();
        unsafe { (*frame.namespace).insert("x".to_string(), w_int_new(5)) };

        let info = build_pyframe_virtualizable_info();
        let mut jit_state = build_jit_state(&frame, &info);
        jit_state.next_instr = 13;
        jit_state.stack_depth = 1;
        assert!(jit_state.set_local_at(0, w_int_new(17)));
        assert!(jit_state.set_local_at(1, w_int_new(19)));
        assert!(jit_state.set_stack_at(0, w_int_new(23)));

        sync_jit_state_to_frame(&jit_state, &mut frame, &info);

        assert_eq!(frame.next_instr, 13);
        assert_eq!(frame.stack_depth, 1);
        unsafe {
            assert_eq!(w_int_get_value(frame.locals_w[0]), 17);
            assert_eq!(w_int_get_value(frame.locals_w[1]), 19);
            assert_eq!(w_int_get_value(frame.value_stack_w[0]), 23);
            assert_eq!(w_int_get_value(*(*frame.namespace).get("x").unwrap()), 5);
        }
    }

    #[test]
    fn test_jit_state_extract_live_uses_frame_only_red_state() {
        let code = nested_function_code("def f(x):\n    y = x\n    return y");
        let mut frame = PyFrame::new_with_context(code, Rc::new(PyExecutionContext::default()));
        frame.fix_array_ptrs();
        frame.next_instr = 7;
        frame.stack_depth = 1;
        frame.locals_w[0] = w_int_new(11);
        frame.locals_w[1] = w_int_new(13);
        frame.value_stack_w[0] = w_int_new(29);

        let info = build_pyframe_virtualizable_info();
        let jit_state = build_jit_state(&frame, &info);
        let meta = <PyreJitState as JitState>::build_meta(&jit_state, frame.next_instr, &PyreEnv);
        let live = <PyreJitState as JitState>::extract_live(&jit_state, &meta);

        let frame_ptr = (&frame as *const PyFrame as usize) as i64;
        let l0 = frame.locals_w[0] as i64;
        let l1 = frame.locals_w[1] as i64;
        assert_eq!(live, vec![frame_ptr, 7, 1, l0, l1]);
    }

    #[test]
    fn test_jit_state_restore_values_refreshes_frame_backed_scalars_when_only_frame_is_live() {
        let code = nested_function_code("def f(x):\n    y = x\n    return y");
        let mut frame = PyFrame::new_with_context(code, Rc::new(PyExecutionContext::default()));
        frame.fix_array_ptrs();

        let info = build_pyframe_virtualizable_info();
        let mut jit_state = build_jit_state(&frame, &info);
        let meta = <PyreJitState as JitState>::build_meta(&jit_state, frame.next_instr, &PyreEnv);

        frame.next_instr = 19;
        frame.stack_depth = 1;
        frame.locals_w[0] = w_int_new(17);
        frame.locals_w[1] = w_int_new(23);
        frame.value_stack_w[0] = w_int_new(29);

        <PyreJitState as JitState>::restore_values(
            &mut jit_state,
            &meta,
            &[Value::Int((&frame as *const PyFrame as usize) as i64)],
        );

        assert_eq!(jit_state.frame, &frame as *const PyFrame as usize);
        assert_eq!(jit_state.next_instr, 19);
        assert_eq!(jit_state.stack_depth, 1);
        unsafe {
            assert_eq!(w_int_get_value(jit_state.local_at(0).unwrap()), 17);
            assert_eq!(w_int_get_value(jit_state.local_at(1).unwrap()), 23);
            assert_eq!(w_int_get_value(jit_state.stack_at(0).unwrap()), 29);
        }
    }

    #[test]
    fn test_jit_state_reconstructed_frame_types_use_frame_only_slot() {
        let code = nested_function_code("def f(x):\n    y = x\n    return y");
        let frame = PyFrame::new_with_context(code, Rc::new(PyExecutionContext::default()));
        let info = build_pyframe_virtualizable_info();
        let jit_state = build_jit_state(&frame, &info);
        let meta = <PyreJitState as JitState>::build_meta(&jit_state, frame.next_instr, &PyreEnv);

        let slot_types =
            <PyreJitState as JitState>::reconstructed_frame_value_types(&jit_state, &meta, 0, 1, 0)
                .expect("root frame should expose reconstructed slot types");

        assert_eq!(slot_types, vec![majit_ir::Type::Int]);
    }

    #[test]
    fn test_jit_state_virtualizable_lengths_follow_frame_shape() {
        let code = nested_function_code("def f(x):\n    y = x\n    return y");
        let mut frame = PyFrame::new_with_context(code, Rc::new(PyExecutionContext::default()));
        frame.fix_array_ptrs();

        let info = build_pyframe_virtualizable_info();
        let jit_state = build_jit_state(&frame, &info);
        let meta = <PyreJitState as JitState>::build_meta(&jit_state, frame.next_instr, &PyreEnv);

        let lengths = <PyreJitState as JitState>::virtualizable_array_lengths(
            &jit_state, &meta, "frame", &info,
        )
        .expect("frame-backed virtualizable lengths");

        assert_eq!(
            lengths,
            vec![frame.locals_w.len(), frame.value_stack_w.len()]
        );
    }

    #[test]
    fn test_jit_state_meta_reads_namespace_shape_from_frame() {
        let code = nested_function_code("def f(x):\n    return x");
        let mut frame = PyFrame::new_with_context(code, Rc::new(PyExecutionContext::default()));
        frame.fix_array_ptrs();
        unsafe {
            (*frame.namespace).insert("z".to_string(), w_int_new(7));
            (*frame.namespace).insert("a".to_string(), w_int_new(11));
        }

        let info = build_pyframe_virtualizable_info();
        let jit_state = build_jit_state(&frame, &info);
        let meta = <PyreJitState as JitState>::build_meta(&jit_state, frame.next_instr, &PyreEnv);

        assert!(meta.ns_keys.windows(2).all(|pair| pair[0] <= pair[1]));
        assert!(meta.ns_keys.iter().any(|key| key == "a"));
        assert!(meta.ns_keys.iter().any(|key| key == "z"));
        assert!(meta.ns_keys.iter().any(|key| key == "abs"));
    }

    #[test]
    fn test_jit_state_restore_reconstructed_frame_values_refreshes_from_frame_only_slot() {
        let code = nested_function_code("def f(x):\n    y = x\n    return y");
        let mut frame = PyFrame::new_with_context(code, Rc::new(PyExecutionContext::default()));
        frame.fix_array_ptrs();

        let info = build_pyframe_virtualizable_info();
        let mut jit_state = build_jit_state(&frame, &info);
        let meta = <PyreJitState as JitState>::build_meta(&jit_state, frame.next_instr, &PyreEnv);

        frame.next_instr = 23;
        frame.stack_depth = 1;
        frame.locals_w[0] = w_int_new(31);
        frame.locals_w[1] = w_int_new(37);
        frame.value_stack_w[0] = w_int_new(41);

        let restored = <PyreJitState as JitState>::restore_reconstructed_frame_values(
            &mut jit_state,
            &meta,
            0,
            1,
            29,
            &[Value::Int((&frame as *const PyFrame as usize) as i64)],
            &ExceptionState::default(),
        );

        assert!(restored);
        assert_eq!(jit_state.frame, &frame as *const PyFrame as usize);
        assert_eq!(jit_state.next_instr, 29);
        assert_eq!(jit_state.stack_depth, 1);
        unsafe {
            assert_eq!(w_int_get_value(jit_state.local_at(0).unwrap()), 31);
            assert_eq!(w_int_get_value(jit_state.local_at(1).unwrap()), 37);
            assert_eq!(w_int_get_value(jit_state.stack_at(0).unwrap()), 41);
        }
    }

    #[test]
    fn test_literal() {
        let result = run_eval("42").unwrap();
        unsafe { assert_eq!(w_int_get_value(result), 42) };
    }

    #[test]
    fn test_addition() {
        let result = run_eval("1 + 2").unwrap();
        unsafe { assert_eq!(w_int_get_value(result), 3) };
    }

    #[test]
    fn test_subtraction() {
        let result = run_eval("10 - 3").unwrap();
        unsafe { assert_eq!(w_int_get_value(result), 7) };
    }

    #[test]
    fn test_multiplication() {
        let result = run_eval("6 * 7").unwrap();
        unsafe { assert_eq!(w_int_get_value(result), 42) };
    }

    #[test]
    fn test_complex_expr() {
        let result = run_eval("(2 + 3) * 4 - 1").unwrap();
        unsafe { assert_eq!(w_int_get_value(result), 19) };
    }

    #[test]
    fn test_comparison() {
        let result = run_eval("3 < 5").unwrap();
        unsafe { assert!(w_bool_get_value(result)) };
    }

    #[test]
    fn test_comparison_false() {
        let result = run_eval("5 < 3").unwrap();
        unsafe { assert!(!w_bool_get_value(result)) };
    }

    #[test]
    fn test_store_load_namespace() {
        let source = "x = 5\ny = x * x";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_frame(&mut frame);
        unsafe {
            let x = *(*frame.namespace).get("x").unwrap();
            let y = *(*frame.namespace).get("y").unwrap();
            assert_eq!(w_int_get_value(x), 5);
            assert_eq!(w_int_get_value(y), 25);
        }
    }

    #[test]
    fn test_while_loop() {
        let source = "i = 0\nwhile i < 10:\n    i = i + 1";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_frame(&mut frame);
        unsafe {
            let i = *(*frame.namespace).get("i").unwrap();
            assert_eq!(w_int_get_value(i), 10);
        }
    }

    #[test]
    fn test_none_result() {
        let result = run_eval("None").unwrap();
        unsafe { assert!(is_none(result)) };
    }

    #[test]
    fn test_bool_result() {
        let result = run_eval("True").unwrap();
        unsafe {
            assert!(is_bool(result));
            assert!(w_bool_get_value(result));
        }
    }

    #[test]
    fn test_float_literal() {
        let result = run_eval("1.5").unwrap();
        unsafe {
            assert!(is_float(result));
            assert_eq!(w_float_get_value(result), 1.5);
        }
    }

    #[test]
    fn test_float_addition() {
        let result = run_eval("1.5 + 2.5").unwrap();
        unsafe {
            assert!(is_float(result));
            assert_eq!(w_float_get_value(result), 4.0);
        }
    }

    #[test]
    fn test_float_truediv() {
        let result = run_eval("10 / 4").unwrap();
        unsafe {
            assert!(is_float(result));
            assert_eq!(w_float_get_value(result), 2.5);
        }
    }

    #[test]
    fn test_float_comparison() {
        let result = run_eval("1.5 < 2.5").unwrap();
        unsafe {
            assert!(is_bool(result));
            assert!(w_bool_get_value(result));
        }
    }

    #[test]
    fn test_float_int_mixed() {
        let result = run_eval("1.5 + 2").unwrap();
        unsafe {
            assert!(is_float(result));
            assert_eq!(w_float_get_value(result), 3.5);
        }
    }

    #[test]
    fn test_float_negation() {
        let result = run_eval("-3.14").unwrap();
        unsafe {
            assert!(is_float(result));
            assert_eq!(w_float_get_value(result), -3.14);
        }
    }

    #[test]
    fn test_float_truthiness() {
        // Test via py_is_true directly since `not` uses ToBool instruction
        assert!(!py_is_true(w_float_new(0.0)));
        assert!(py_is_true(w_float_new(1.5)));
        assert!(py_is_true(w_float_new(-0.1)));
    }

    // ── str tests ────────────────────────────────────────────────────

    #[test]
    fn test_str_literal() {
        let result = run_eval("'hello'").unwrap();
        unsafe {
            assert!(is_str(result));
            assert_eq!(w_str_get_value(result), "hello");
        }
    }

    #[test]
    fn test_str_concat() {
        let result = run_eval("'hello' + ' world'").unwrap();
        unsafe {
            assert!(is_str(result));
            assert_eq!(w_str_get_value(result), "hello world");
        }
    }

    #[test]
    fn test_str_repeat() {
        let result = run_eval("'ab' * 3").unwrap();
        unsafe {
            assert!(is_str(result));
            assert_eq!(w_str_get_value(result), "ababab");
        }
    }

    #[test]
    fn test_str_comparison() {
        let result = run_eval("'abc' < 'abd'").unwrap();
        unsafe {
            assert!(is_bool(result));
            assert!(w_bool_get_value(result));
        }
    }

    // ── for loop / range tests ──────────────────────────────────────

    #[test]
    fn test_for_range() {
        let source = "s = 0\nfor i in range(10):\n    s = s + i";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_frame(&mut frame);
        unsafe {
            let s = *(*frame.namespace).get("s").unwrap();
            assert_eq!(w_int_get_value(s), 45);
        }
    }

    #[test]
    fn test_hot_range_loop_survives_compiled_trace() {
        let source = "s = 0\nfor i in range(3000):\n    s = s + i";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_frame(&mut frame);
        unsafe {
            let s = *(*frame.namespace).get("s").unwrap();
            assert_eq!(w_int_get_value(s), 4_498_500);
        }
    }

    #[test]
    fn test_hot_module_branch_loop_survives_compiled_trace() {
        let source = "\
i = 0
acc = 0
while i < 3000:
    if i < 1500:
        acc = acc + 1
    else:
        acc = acc + 2
    i = i + 1";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_frame(&mut frame);
        unsafe {
            let i = *(*frame.namespace).get("i").unwrap();
            let acc = *(*frame.namespace).get("acc").unwrap();
            assert_eq!(w_int_get_value(i), 3000);
            assert_eq!(w_int_get_value(acc), 4500);
        }
    }

    #[test]
    fn test_hot_tuple_unpack_loop_survives_compiled_trace() {
        let source = "\
i = 0
acc = 0
while i < 3000:
    a, b = (i, 1)
    acc = acc + a + b
    i = i + 1";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_frame(&mut frame);
        unsafe {
            let i = *(*frame.namespace).get("i").unwrap();
            let acc = *(*frame.namespace).get("acc").unwrap();
            assert_eq!(w_int_get_value(i), 3000);
            assert_eq!(w_int_get_value(acc), 4_501_500);
        }
    }

    #[test]
    fn test_hot_list_index_store_loop_survives_compiled_trace() {
        let source = "\
lst = [0]
i = 0
acc = 0
while i < 3000:
    lst[0] = i
    acc = acc + lst[0]
    i = i + 1";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_frame(&mut frame);
        unsafe {
            let i = *(*frame.namespace).get("i").unwrap();
            let acc = *(*frame.namespace).get("acc").unwrap();
            let lst = *(*frame.namespace).get("lst").unwrap();
            assert_eq!(w_int_get_value(i), 3000);
            assert_eq!(w_int_get_value(acc), 4_498_500);
            assert_eq!(w_int_get_value(w_list_getitem(lst, 0).unwrap()), 2999);
        }
    }

    #[test]
    fn test_hot_bitwise_or_loop_survives_compiled_trace() {
        let source = "\
i = 0
acc = 0
while i < 3000:
    acc = acc | i
    i = i + 1";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_frame(&mut frame);
        unsafe {
            let i = *(*frame.namespace).get("i").unwrap();
            let acc = *(*frame.namespace).get("acc").unwrap();
            assert_eq!(w_int_get_value(i), 3000);
            assert_eq!(w_int_get_value(acc), 4095);
        }
    }

    #[test]
    fn test_hot_unary_invert_loop_survives_compiled_trace() {
        let source = "\
i = 0
acc = 0
while i < 3000:
    acc = acc + (~i)
    i = i + 1";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_frame(&mut frame);
        unsafe {
            let i = *(*frame.namespace).get("i").unwrap();
            let acc = *(*frame.namespace).get("acc").unwrap();
            assert_eq!(w_int_get_value(i), 3000);
            assert_eq!(w_int_get_value(acc), -4_501_500);
        }
    }

    #[test]
    fn test_hot_positive_floordiv_loop_survives_compiled_trace() {
        let source = "\
i = 0
acc = 0
while i < 3000:
    acc = acc + (i // 3)
    i = i + 1";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_frame(&mut frame);
        unsafe {
            let i = *(*frame.namespace).get("i").unwrap();
            let acc = *(*frame.namespace).get("acc").unwrap();
            assert_eq!(w_int_get_value(i), 3000);
            assert_eq!(w_int_get_value(acc), 1_498_500);
        }
    }

    #[test]
    fn test_hot_positive_mod_loop_survives_compiled_trace() {
        let source = "\
i = 0
acc = 0
while i < 3000:
    acc = acc + (i % 7)
    i = i + 1";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_frame(&mut frame);
        unsafe {
            let i = *(*frame.namespace).get("i").unwrap();
            let acc = *(*frame.namespace).get("acc").unwrap();
            assert_eq!(w_int_get_value(i), 3000);
            assert_eq!(w_int_get_value(acc), 8_994);
        }
    }

    #[test]
    fn test_hot_builtin_abs_loop_survives_compiled_trace() {
        let source = "\
i = 0
acc = 0
while i < 3000:
    acc = acc + abs(i - 1500)
    i = i + 1";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_frame(&mut frame);
        unsafe {
            let i = *(*frame.namespace).get("i").unwrap();
            let acc = *(*frame.namespace).get("acc").unwrap();
            assert_eq!(w_int_get_value(i), 3000);
            assert_eq!(w_int_get_value(acc), 2_250_000);
        }
    }

    #[test]
    fn test_hot_list_truth_loop_survives_compiled_trace() {
        let source = "\
lst = [1]
i = 0
acc = 0
while i < 3000:
    if lst:
        acc = acc + 1
    i = i + 1";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_frame(&mut frame);
        unsafe {
            let i = *(*frame.namespace).get("i").unwrap();
            let acc = *(*frame.namespace).get("acc").unwrap();
            assert_eq!(w_int_get_value(i), 3000);
            assert_eq!(w_int_get_value(acc), 3000);
        }
    }

    #[test]
    fn test_hot_empty_tuple_truth_loop_survives_compiled_trace() {
        let source = "\
tpl = ()
i = 0
acc = 0
while i < 3000:
    if tpl:
        acc = acc + 100
    else:
        acc = acc + 1
    i = i + 1";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_frame(&mut frame);
        unsafe {
            let i = *(*frame.namespace).get("i").unwrap();
            let acc = *(*frame.namespace).get("acc").unwrap();
            assert_eq!(w_int_get_value(i), 3000);
            assert_eq!(w_int_get_value(acc), 3000);
        }
    }

    #[test]
    fn test_hot_none_truth_loop_survives_compiled_trace() {
        let source = "\
value = None
i = 0
acc = 0
while i < 3000:
    if value:
        acc = acc + 100
    else:
        acc = acc + 1
    i = i + 1";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_frame(&mut frame);
        unsafe {
            let i = *(*frame.namespace).get("i").unwrap();
            let acc = *(*frame.namespace).get("acc").unwrap();
            assert_eq!(w_int_get_value(i), 3000);
            assert_eq!(w_int_get_value(acc), 3000);
        }
    }

    #[test]
    fn test_hot_float_truth_loop_survives_compiled_trace() {
        let source = "\
value = 0.5
i = 0
acc = 0
while i < 3000:
    if value:
        acc = acc + 1
    i = i + 1";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_frame(&mut frame);
        unsafe {
            let i = *(*frame.namespace).get("i").unwrap();
            let acc = *(*frame.namespace).get("acc").unwrap();
            assert_eq!(w_int_get_value(i), 3000);
            assert_eq!(w_int_get_value(acc), 3000);
        }
    }

    #[test]
    fn test_hot_string_truth_loop_survives_compiled_trace() {
        let source = "\
value = \"pyre\"
i = 0
acc = 0
while i < 3000:
    if value:
        acc = acc + 1
    i = i + 1";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_frame(&mut frame);
        unsafe {
            let i = *(*frame.namespace).get("i").unwrap();
            let acc = *(*frame.namespace).get("acc").unwrap();
            assert_eq!(w_int_get_value(i), 3000);
            assert_eq!(w_int_get_value(acc), 3000);
        }
    }

    #[test]
    fn test_hot_empty_string_truth_loop_survives_compiled_trace() {
        let source = "\
value = \"\"
i = 0
acc = 0
while i < 3000:
    if value:
        acc = acc + 100
    else:
        acc = acc + 1
    i = i + 1";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_frame(&mut frame);
        unsafe {
            let i = *(*frame.namespace).get("i").unwrap();
            let acc = *(*frame.namespace).get("acc").unwrap();
            assert_eq!(w_int_get_value(i), 3000);
            assert_eq!(w_int_get_value(acc), 3000);
        }
    }

    #[test]
    fn test_hot_dict_truth_loop_survives_compiled_trace() {
        let source = "\
value = {1: 2}
i = 0
acc = 0
while i < 3000:
    if value:
        acc = acc + 1
    i = i + 1";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_frame(&mut frame);
        unsafe {
            let i = *(*frame.namespace).get("i").unwrap();
            let acc = *(*frame.namespace).get("acc").unwrap();
            assert_eq!(w_int_get_value(i), 3000);
            assert_eq!(w_int_get_value(acc), 3000);
        }
    }

    #[test]
    fn test_hot_builtin_len_string_loop_survives_compiled_trace() {
        let source = "\
value = \"pyre\"
i = 0
acc = 0
while i < 3000:
    acc = acc + len(value)
    i = i + 1";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_frame(&mut frame);
        unsafe {
            let i = *(*frame.namespace).get("i").unwrap();
            let acc = *(*frame.namespace).get("acc").unwrap();
            assert_eq!(w_int_get_value(i), 3000);
            assert_eq!(w_int_get_value(acc), 12_000);
        }
    }

    #[test]
    fn test_hot_builtin_len_dict_loop_survives_compiled_trace() {
        let source = "\
value = {1: 2, 3: 4}
i = 0
acc = 0
while i < 3000:
    acc = acc + len(value)
    i = i + 1";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_frame(&mut frame);
        unsafe {
            let i = *(*frame.namespace).get("i").unwrap();
            let acc = *(*frame.namespace).get("acc").unwrap();
            assert_eq!(w_int_get_value(i), 3000);
            assert_eq!(w_int_get_value(acc), 6_000);
        }
    }

    #[test]
    fn test_hot_builtin_isinstance_true_loop_survives_compiled_trace() {
        let source = "\
x = 42
i = 0
acc = 0
while i < 3000:
    if isinstance(x, \"int\"):
        acc = acc + 1
    i = i + 1";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_frame(&mut frame);
        unsafe {
            let i = *(*frame.namespace).get("i").unwrap();
            let acc = *(*frame.namespace).get("acc").unwrap();
            assert_eq!(w_int_get_value(i), 3000);
            assert_eq!(w_int_get_value(acc), 3000);
        }
    }

    #[test]
    fn test_hot_builtin_isinstance_false_loop_survives_compiled_trace() {
        let source = "\
x = []
i = 0
acc = 0
while i < 3000:
    if isinstance(x, \"int\"):
        acc = acc + 1
    else:
        acc = acc + 2
    i = i + 1";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_frame(&mut frame);
        unsafe {
            let i = *(*frame.namespace).get("i").unwrap();
            let acc = *(*frame.namespace).get("acc").unwrap();
            assert_eq!(w_int_get_value(i), 3000);
            assert_eq!(w_int_get_value(acc), 6000);
        }
    }

    #[test]
    fn test_hot_builtin_type_loop_survives_compiled_trace() {
        let source = "\
x = []
i = 0
acc = 0
while i < 3000:
    if type(x) == \"list\":
        acc = acc + 1
    i = i + 1";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_frame(&mut frame);
        unsafe {
            let i = *(*frame.namespace).get("i").unwrap();
            let acc = *(*frame.namespace).get("acc").unwrap();
            assert_eq!(w_int_get_value(i), 3000);
            assert_eq!(w_int_get_value(acc), 3000);
        }
    }

    #[test]
    fn test_hot_builtin_min_small_int_loop_survives_compiled_trace() {
        let source = "\
i = 0
acc = 0
while i < 3000:
    acc = acc + min(i % 7, 3)
    i = i + 1";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_frame(&mut frame);
        unsafe {
            let i = *(*frame.namespace).get("i").unwrap();
            let acc = *(*frame.namespace).get("acc").unwrap();
            assert_eq!(w_int_get_value(i), 3000);
            assert_eq!(w_int_get_value(acc), 6426);
        }
    }

    #[test]
    fn test_hot_builtin_max_small_int_loop_survives_compiled_trace() {
        let source = "\
i = 0
acc = 0
while i < 3000:
    acc = acc + max(i % 7, 3)
    i = i + 1";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_frame(&mut frame);
        unsafe {
            let i = *(*frame.namespace).get("i").unwrap();
            let acc = *(*frame.namespace).get("acc").unwrap();
            assert_eq!(w_int_get_value(i), 3000);
            assert_eq!(w_int_get_value(acc), 11568);
        }
    }

    #[test]
    fn test_hot_empty_dict_truth_loop_survives_compiled_trace() {
        let source = "\
value = {}
i = 0
acc = 0
while i < 3000:
    if value:
        acc = acc + 100
    else:
        acc = acc + 1
    i = i + 1";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_frame(&mut frame);
        unsafe {
            let i = *(*frame.namespace).get("i").unwrap();
            let acc = *(*frame.namespace).get("acc").unwrap();
            assert_eq!(w_int_get_value(i), 3000);
            assert_eq!(w_int_get_value(acc), 3000);
        }
    }

    #[test]
    fn test_hot_list_negative_index_store_loop_survives_compiled_trace() {
        let source = "\
lst = [0, 1]
i = 0
acc = 0
while i < 3000:
    lst[-1] = i
    acc = acc + lst[-1]
    i = i + 1";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_frame(&mut frame);
        unsafe {
            let i = *(*frame.namespace).get("i").unwrap();
            let acc = *(*frame.namespace).get("acc").unwrap();
            let lst = *(*frame.namespace).get("lst").unwrap();
            assert_eq!(w_int_get_value(i), 3000);
            assert_eq!(w_int_get_value(acc), 4_498_500);
            assert_eq!(w_int_get_value(w_list_getitem(lst, -1).unwrap()), 2999);
        }
    }

    #[test]
    fn test_hot_tuple_negative_index_load_loop_survives_compiled_trace() {
        let source = "\
tpl = (3, 5)
i = 0
acc = 0
while i < 3000:
    acc = acc + tpl[-1]
    i = i + 1";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_frame(&mut frame);
        unsafe {
            let i = *(*frame.namespace).get("i").unwrap();
            let acc = *(*frame.namespace).get("acc").unwrap();
            assert_eq!(w_int_get_value(i), 3000);
            assert_eq!(w_int_get_value(acc), 15_000);
        }
    }

    #[test]
    fn test_hot_user_function_loop_survives_compiled_trace() {
        let source = "\
def inc(x):
    return x + 1
i = 0
acc = 0
while i < 3000:
    acc = acc + inc(i)
    i = i + 1";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_frame(&mut frame);
        unsafe {
            let i = *(*frame.namespace).get("i").unwrap();
            let acc = *(*frame.namespace).get("acc").unwrap();
            assert_eq!(w_int_get_value(i), 3000);
            assert_eq!(w_int_get_value(acc), 4_501_500);
        }
    }

    #[test]
    fn test_for_range_start_stop() {
        let source = "s = 0\nfor i in range(5, 10):\n    s = s + i";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_frame(&mut frame);
        unsafe {
            let s = *(*frame.namespace).get("s").unwrap();
            assert_eq!(w_int_get_value(s), 35);
        }
    }

    #[test]
    fn test_for_range_step() {
        let source = "s = 0\nfor i in range(0, 10, 2):\n    s = s + i";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_frame(&mut frame);
        unsafe {
            let s = *(*frame.namespace).get("s").unwrap();
            // 0 + 2 + 4 + 6 + 8 = 20
            assert_eq!(w_int_get_value(s), 20);
        }
    }

    #[test]
    fn test_for_range_empty() {
        let source = "s = 42\nfor i in range(0):\n    s = 0";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_frame(&mut frame);
        unsafe {
            let s = *(*frame.namespace).get("s").unwrap();
            assert_eq!(w_int_get_value(s), 42);
        }
    }

    #[test]
    fn test_builtin_range_print() {
        let source = "s = 0\nfor i in range(5):\n    s = s + i";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_frame(&mut frame);
        unsafe {
            let s = *(*frame.namespace).get("s").unwrap();
            // 0 + 1 + 2 + 3 + 4 = 10
            assert_eq!(w_int_get_value(s), 10);
        }
    }

    // ── builtin tests ───────────────────────────────────────────────

    #[test]
    fn test_builtin_len() {
        let source = "x = len([1, 2, 3])";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_frame(&mut frame);
        unsafe {
            let x = *(*frame.namespace).get("x").unwrap();
            assert_eq!(w_int_get_value(x), 3);
        }
    }

    #[test]
    fn test_builtin_abs() {
        let source = "x = abs(-5)";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_frame(&mut frame);
        unsafe {
            let x = *(*frame.namespace).get("x").unwrap();
            assert_eq!(w_int_get_value(x), 5);
        }
    }

    #[test]
    fn test_builtin_min_max() {
        let source = "a = min(3, 7)\nb = max(3, 7)";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_frame(&mut frame);
        unsafe {
            let a = *(*frame.namespace).get("a").unwrap();
            let b = *(*frame.namespace).get("b").unwrap();
            assert_eq!(w_int_get_value(a), 3);
            assert_eq!(w_int_get_value(b), 7);
        }
    }

    // ── container tests ────────────────────────────────────────────

    #[test]
    fn test_list_literal() {
        let source = "x = [1, 2, 3]";
        let (_, frame) = run_exec_frame(source);
        unsafe {
            let x = *(*frame.namespace).get("x").unwrap();
            assert!(is_list(x));
            assert_eq!(w_list_len(x), 3);
            assert_eq!(w_int_get_value(w_list_getitem(x, 0).unwrap()), 1);
            assert_eq!(w_int_get_value(w_list_getitem(x, 1).unwrap()), 2);
            assert_eq!(w_int_get_value(w_list_getitem(x, 2).unwrap()), 3);
        }
    }

    #[test]
    fn test_tuple_unpack() {
        let source = "a, b = 1, 2";
        let (_, frame) = run_exec_frame(source);
        unsafe {
            let a = *(*frame.namespace).get("a").unwrap();
            let b = *(*frame.namespace).get("b").unwrap();
            assert_eq!(w_int_get_value(a), 1);
            assert_eq!(w_int_get_value(b), 2);
        }
    }

    #[test]
    fn test_list_subscr() {
        let source = "lst = [10, 20, 30]\nx = lst[1]";
        let (_, frame) = run_exec_frame(source);
        unsafe {
            let x = *(*frame.namespace).get("x").unwrap();
            assert_eq!(w_int_get_value(x), 20);
        }
    }

    #[test]
    fn test_list_store_subscr() {
        let source = "lst = [1, 2, 3]\nlst[0] = 99\nx = lst[0]";
        let (_, frame) = run_exec_frame(source);
        unsafe {
            let x = *(*frame.namespace).get("x").unwrap();
            assert_eq!(w_int_get_value(x), 99);
        }
    }

    #[test]
    fn test_dict_literal_and_subscr() {
        let source = "d = {1: 10, 2: 20}\nx = d[1]";
        let (_, frame) = run_exec_frame(source);
        unsafe {
            let x = *(*frame.namespace).get("x").unwrap();
            assert_eq!(w_int_get_value(x), 10);
        }
    }

    // ── function definition and call tests ──────────────────────────

    #[test]
    fn test_simple_function() {
        let source = "def double(x):\n    return x * 2\nresult = double(21)";
        let (_, frame) = run_exec_frame(source);
        unsafe {
            let result = *(*frame.namespace).get("result").unwrap();
            assert_eq!(w_int_get_value(result), 42);
        }
    }

    #[test]
    fn test_function_with_locals() {
        let source = "\
def add_squares(a, b):
    aa = a * a
    bb = b * b
    return aa + bb
result = add_squares(3, 4)";
        let (_, frame) = run_exec_frame(source);
        unsafe {
            let result = *(*frame.namespace).get("result").unwrap();
            assert_eq!(w_int_get_value(result), 25);
        }
    }

    #[test]
    fn test_recursive_function() {
        let source = "\
def factorial(n):
    if n < 2:
        return 1
    return n * factorial(n - 1)
result = factorial(5)";
        let (_, frame) = run_exec_frame(source);
        unsafe {
            let result = *(*frame.namespace).get("result").unwrap();
            assert_eq!(w_int_get_value(result), 120);
        }
    }
}
