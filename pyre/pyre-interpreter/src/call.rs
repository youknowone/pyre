//! Function call dispatch — pure interpreter, no JIT dependencies.
//!
//! JIT-specific call infrastructure (force/bridge callbacks, callee frame
//! creation helpers, frame pool) lives in pyre-jit/src/call_jit.rs.

use std::cell::Cell;
use std::sync::OnceLock;

use crate::{
    PyError, PyErrorKind, PyNamespace, PyResult, dispatch_callable, w_builtin_func_get,
    w_func_get_closure, w_func_get_code_ptr, w_func_get_globals,
};
use pyre_object::{PY_NULL, PyObjectRef};

use crate::eval::eval_frame_plain;
use crate::frame::{PendingInlineResult, PyFrame};

/// Store an inline-handled concrete result on the owning caller frame.
pub fn set_pending_inline_result(frame: &mut PyFrame, result: PendingInlineResult) {
    frame.pending_inline_results.push_back(result);
}

fn take_pending_inline_result(frame: &mut PyFrame) -> Option<PyObjectRef> {
    match frame.pending_inline_results.pop_front()? {
        PendingInlineResult::Ref(result) => Some(result),
        PendingInlineResult::Int(value) => Some(pyre_object::w_int_new(value)),
        PendingInlineResult::Float(value) => Some(pyre_object::floatobject::w_float_new(value)),
    }
}

/// Replay the concrete result of a just-traced inline call.
///
/// PyPy's `finishframe()` writes the return value into the parent frame and
/// resumes there; it does not route the value through the generic call
/// dispatcher.  Keep pyre's concrete replay explicit as well: when the
/// interpreter reaches the original CALL opcode, it consumes the pending
/// result, pops callable/args, and pushes the materialized return value.
pub fn replay_pending_inline_call(frame: &mut PyFrame, nargs: usize) -> bool {
    let Some(result) = take_pending_inline_result(frame) else {
        return false;
    };
    for _ in 0..nargs {
        let _ = frame.pop();
    }
    let _ = frame.pop(); // null_or_self
    let _ = frame.pop(); // callable
    frame.push(result);
    true
}

// ── Eval function injection ──────────────────────────────────────
type EvalFn = fn(&mut PyFrame) -> PyResult;
static EVAL_OVERRIDE: OnceLock<EvalFn> = OnceLock::new();

type DepthBumpFn = fn() -> Option<Box<dyn std::any::Any>>;
static DEPTH_BUMP_OVERRIDE: OnceLock<DepthBumpFn> = OnceLock::new();

type InlineCallOverrideFn = fn(&PyFrame, PyObjectRef, &[PyObjectRef]) -> Option<PyResult>;
static INLINE_CALL_OVERRIDE: OnceLock<InlineCallOverrideFn> = OnceLock::new();

thread_local! {
    static INLINE_CALL_OVERRIDE_DEPTH: Cell<u32> = const { Cell::new(0) };
    /// Call depth counter — incremented on every user function call,
    /// decremented on return. Replaces the Box<dyn Any> depth bump
    /// callback with a zero-allocation TLS increment.
    static CALL_DEPTH: Cell<u32> = const { Cell::new(0) };
}

/// Get current call depth. Used by pyre-jit for JIT_CALL_DEPTH parity.
#[inline(always)]
pub fn call_depth() -> u32 {
    CALL_DEPTH.with(|d| d.get())
}

/// RAII guard that decrements CALL_DEPTH on drop.
struct CallDepthGuard;
impl Drop for CallDepthGuard {
    #[inline(always)]
    fn drop(&mut self) {
        CALL_DEPTH.with(|d| d.set(d.get().saturating_sub(1)));
    }
}

/// Register the JIT-aware eval function. Called by pyre-jit at startup.
pub fn register_eval_override(f: EvalFn) {
    let _ = EVAL_OVERRIDE.set(f);
}

thread_local! {
    static FORCE_PLAIN_EVAL: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

/// Guard that temporarily forces all nested calls to use the plain
/// interpreter, bypassing eval_with_jit. Used by force_fn to avoid
/// re-entering compiled code from blackhole execution.
pub struct ForcePlainEvalGuard;

impl Drop for ForcePlainEvalGuard {
    fn drop(&mut self) {
        let _ = FORCE_PLAIN_EVAL.try_with(|c| c.set(c.get().saturating_sub(1)));
    }
}

/// Enter "force plain eval" mode. While active, `call_user_function` uses
/// `eval_frame_plain` instead of the JIT-aware eval override.
pub fn force_plain_eval() -> ForcePlainEvalGuard {
    FORCE_PLAIN_EVAL.with(|c| c.set(c.get() + 1));
    ForcePlainEvalGuard
}

/// Register the JIT call-depth bump function. Called by pyre-jit at startup.
pub fn register_depth_bump(f: DepthBumpFn) {
    let _ = DEPTH_BUMP_OVERRIDE.set(f);
}

/// Register an inline-only concrete call override. Called by pyre-jit at startup.
pub fn register_inline_call_override(f: InlineCallOverrideFn) {
    let _ = INLINE_CALL_OVERRIDE.set(f);
}

pub struct InlineCallOverrideGuard;

impl Drop for InlineCallOverrideGuard {
    fn drop(&mut self) {
        INLINE_CALL_OVERRIDE_DEPTH.with(|d| d.set(d.get() - 1));
    }
}

/// Enable the inline concrete-call override for the current thread.
pub fn inline_call_override_guard() -> InlineCallOverrideGuard {
    INLINE_CALL_OVERRIDE_DEPTH.with(|d| d.set(d.get() + 1));
    InlineCallOverrideGuard
}

pub struct SuspendInlineCallOverrideGuard {
    previous_depth: u32,
}

impl Drop for SuspendInlineCallOverrideGuard {
    fn drop(&mut self) {
        INLINE_CALL_OVERRIDE_DEPTH.with(|d| d.set(self.previous_depth));
    }
}

/// Temporarily disable the inline concrete-call override for the current
/// thread. Helper-boundary fallback execution must not accidentally reuse
/// the outer inline trace's concrete-call interception seam.
pub fn suspend_inline_call_override() -> SuspendInlineCallOverrideGuard {
    let previous_depth = INLINE_CALL_OVERRIDE_DEPTH.with(|d| {
        let previous = d.get();
        d.set(0);
        previous
    });
    SuspendInlineCallOverrideGuard { previous_depth }
}

pub struct SuspendInlineHandledResultGuard;

impl Drop for SuspendInlineHandledResultGuard {
    fn drop(&mut self) {}
}

/// Temporarily isolate inline-result replay from helper-boundary execution.
///
/// Results are now owned by the concrete caller frame, so helper-boundary
/// interpreters no longer share a thread-global result channel. The guard
/// remains as an explicit protocol marker for residual-call slow paths.
pub fn suspend_inline_handled_result() -> SuspendInlineHandledResultGuard {
    SuspendInlineHandledResultGuard
}

fn try_inline_call_override(
    frame: &PyFrame,
    callable: PyObjectRef,
    args: &[PyObjectRef],
) -> Option<PyResult> {
    let enabled = INLINE_CALL_OVERRIDE_DEPTH.with(|d| d.get() > 0);
    if !enabled {
        return None;
    }
    INLINE_CALL_OVERRIDE
        .get()
        .and_then(|override_fn| override_fn(frame, callable, args))
}

fn call_user_function_with_eval(
    frame: &PyFrame,
    callable: PyObjectRef,
    args: &[PyObjectRef],
    eval_fn: EvalFn,
) -> PyResult {
    let code_ptr = unsafe { w_func_get_code_ptr(callable) };
    let globals = unsafe { w_func_get_globals(callable) };
    let closure = unsafe { w_func_get_closure(callable) };
    let func_code = code_ptr as *const pyre_bytecode::CodeObject;
    let mut func_frame = PyFrame::new_for_call_with_closure(
        func_code,
        args,
        globals,
        frame.execution_context,
        closure,
    );
    func_frame.fix_array_ptrs();
    eval_fn(&mut func_frame)
}

pub fn call_callable(frame: &mut PyFrame, callable: PyObjectRef, args: &[PyObjectRef]) -> PyResult {
    let callable = crate::space::unwrap_cell(callable);
    if unsafe { pyre_object::is_type(callable) } {
        return call_type_object(frame, callable, args);
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
    if let Some(result) = try_inline_call_override(frame, callable, args) {
        return result;
    }

    // Direct TLS increment — no Box allocation. Replaces DEPTH_BUMP_OVERRIDE callback.
    CALL_DEPTH.with(|d| d.set(d.get() + 1));
    let _depth_guard = CallDepthGuard;
    let plain_mode = FORCE_PLAIN_EVAL.with(|c| c.get() > 0);
    let eval_fn = if plain_mode {
        eval_frame_plain
    } else {
        EVAL_OVERRIDE.get().copied().unwrap_or(eval_frame_plain)
    };
    call_user_function_with_eval(frame, callable, args, eval_fn)
}

/// Plain interpreter-only user-function call.
///
/// JIT residual helpers should use this instead of the injected eval override.
/// PyPy residual calls are opaque slow paths; they should not accidentally
/// re-enter the caller's portal/tracing state.
pub fn call_user_function_plain(
    frame: &PyFrame,
    callable: PyObjectRef,
    args: &[PyObjectRef],
) -> PyResult {
    call_user_function_with_eval(frame, callable, args, eval_frame_plain)
}

/// Call a user function with an explicit execution context pointer.
/// Used by MIFrame Box tracking when concrete_frame is unavailable.
pub fn call_user_function_plain_with_ctx(
    execution_context: *const crate::PyExecutionContext,
    callable: PyObjectRef,
    args: &[PyObjectRef],
) -> PyResult {
    let code_ptr = unsafe { w_func_get_code_ptr(callable) };
    let globals = unsafe { w_func_get_globals(callable) };
    let closure = unsafe { w_func_get_closure(callable) };
    let func_code = code_ptr as *const pyre_bytecode::CodeObject;
    let mut func_frame =
        PyFrame::new_for_call_with_closure(func_code, args, globals, execution_context, closure);
    func_frame.fix_array_ptrs();
    eval_frame_plain(&mut func_frame)
}

/// Explicit residual-call protocol used by JIT inline framestack concrete
/// execution.
///
/// PyPy treats residual calls reached from inline execution as opaque slow
/// paths. They must not accidentally reuse the generic JIT-aware
/// `call_user_function()` entry, because that can re-enter portal state that
/// belongs to the outer trace instead of the active inline framestack.
pub fn call_callable_inline_residual(
    frame: &PyFrame,
    callable: PyObjectRef,
    args: &[PyObjectRef],
) -> PyResult {
    let _suspend_inline_result = suspend_inline_handled_result();
    dispatch_callable(
        callable,
        |callable| {
            let func = unsafe { w_builtin_func_get(callable) };
            Ok(func(args))
        },
        |callable| call_user_function_plain(frame, callable, args),
    )
}

// ── __build_class__ implementation ───────────────────────────────────
// PyPy equivalent: pyopcode.py BUILD_CLASS
//   1. Execute class body function with fresh namespace (class_locals)
//   2. Create W_TypeObject from the harvested namespace

/// Initialize interpreter callbacks and type registry.
///
/// PyPy: setup_builtin_modules / make_builtins — called once at startup.
pub fn register_build_class() {
    crate::typedef::install_builtin_typedefs();
}

/// `space.call_function(callable, *args)` — direct implementation.
///
/// PyPy: baseobjspace.py `call_function`. Now a direct function call
/// (no callback — interpreter and runtime are in the same crate).
pub(crate) fn space_call_function_impl(callable: PyObjectRef, args: &[PyObjectRef]) -> PyObjectRef {
    unsafe {
        // Builtin function: direct Rust call
        if crate::is_builtin_func(callable) {
            let func = crate::w_builtin_func_get(callable);
            return func(args);
        }
        // User function: create frame + eval
        if crate::is_func(callable) {
            return call_user_func_with_args(callable, args);
        }
        // Type object: instance creation
        if pyre_object::is_type(callable) {
            // Minimal: create instance + call __init__
            let instance = pyre_object::w_instance_new(callable);
            if let Ok(init_fn) = crate::space::py_getattr(callable, "__init__") {
                let mut init_args = Vec::with_capacity(1 + args.len());
                init_args.push(instance);
                init_args.extend_from_slice(args);
                let _ = space_call_function_impl(init_fn, &init_args);
            }
            return instance;
        }
    }
    panic!("space_call_function: '{}' object is not callable", unsafe {
        (*(*callable).ob_type).tp_name
    });
}

/// Helper: call a user function with arbitrary args from descriptor context.
fn call_user_func_with_args(func: PyObjectRef, args: &[PyObjectRef]) -> PyObjectRef {
    let code_ptr = unsafe { w_func_get_code_ptr(func) };
    let globals = unsafe { w_func_get_globals(func) };
    let closure = unsafe { w_func_get_closure(func) };
    let func_code = code_ptr as *const pyre_bytecode::CodeObject;
    let exec_ctx = BUILD_CLASS_EXEC_CTX.with(|c| c.get());
    let mut frame = PyFrame::new_for_call_with_closure(func_code, args, globals, exec_ctx, closure);
    frame.fix_array_ptrs();
    eval_frame_plain(&mut frame).unwrap_or(PY_NULL)
}

/// The real __build_class__(body_fn, name, *bases) implementation.
///
/// PyPy equivalent: pyopcode.py BUILD_CLASS →
///   w_methodsdict = call(body_fn)
///   w_newclass = call(metaclass, name, bases, methodsdict)
pub(crate) fn real_build_class(args: &[PyObjectRef]) -> PyObjectRef {
    assert!(
        args.len() >= 2,
        "__build_class__ requires at least 2 arguments"
    );
    let body_fn = args[0];
    let name_obj = args[1];
    let bases = &args[2..];

    let name = unsafe { pyre_object::w_str_get_value(name_obj) };
    let bases_tuple = pyre_object::w_tuple_new(bases.to_vec());

    match build_class_inner(body_fn, name, bases_tuple) {
        Ok(cls) => cls,
        Err(e) => {
            // Propagate as exception object — the caller's exception handler
            // will catch it (e.g., try/except ImportError in datetime.py).
            e.to_exc_object()
        }
    }
}

fn build_class_inner(body_fn: PyObjectRef, name: &str, bases: PyObjectRef) -> PyResult {
    let code_ptr = unsafe { w_func_get_code_ptr(body_fn) };
    let globals = unsafe { w_func_get_globals(body_fn) };
    let closure = unsafe { w_func_get_closure(body_fn) };
    let func_code = code_ptr as *const pyre_bytecode::CodeObject;

    // Create a fresh namespace for the class body (PyPy: w_locals for class scope)
    let mut class_ns = Box::new(PyNamespace::new());
    class_ns.fix_ptr();
    let class_ns_ptr = Box::into_raw(class_ns);

    let stored = BUILD_CLASS_EXEC_CTX.with(|c| c.get());
    let exec_ctx = if stored.is_null() {
        std::ptr::null::<crate::PyExecutionContext>()
    } else {
        stored
    };

    // Create frame with class_locals set AND closure from enclosing scope.
    // PyPy: executes class body with w_locals = fresh dict, w_globals = module globals,
    // and the closure tuple is passed through for LOAD_DEREF access.
    // Debug: dump code object for __class__ cell investigation
    let code_ref = unsafe { &*func_code };
    if std::env::var("PYRE_DEBUG_CLASS").is_ok() {
        eprintln!("[build_class] name={name}");
        eprintln!("  varnames: {:?}", code_ref.varnames);
        eprintln!("  cellvars: {:?}", code_ref.cellvars);
        eprintln!("  freevars: {:?}", code_ref.freevars);
        eprintln!(
            "  nlocals={} ncells={} nfree={}",
            code_ref.varnames.len(),
            code_ref.cellvars.len(),
            code_ref.freevars.len()
        );
        for (i, instr) in code_ref.instructions.iter().enumerate().take(20) {
            eprintln!("  {i}: {:?}", instr);
        }
    }

    let mut frame = PyFrame::new_for_call_with_closure(func_code, &[], globals, exec_ctx, closure);
    frame.class_locals = class_ns_ptr;

    eval_frame_plain(&mut frame)?;

    // Create W_TypeObject from the class namespace
    // PyPy: type.__new__(type, name, bases, dict_w) + compute_mro + ready()
    let w_type = pyre_object::w_type_new(name, bases, class_ns_ptr as *mut u8);

    // CPython: if __classcell__ is in the namespace, set the cell's content
    // to the newly created class. This enables `__class__` references in methods.
    let class_ns = unsafe { &*class_ns_ptr };
    if let Some(&classcell) = class_ns.get("__classcell__") {
        if !classcell.is_null() && unsafe { pyre_object::is_cell(classcell) } {
            unsafe { pyre_object::w_cell_set(classcell, w_type) };
        }
    }
    // Cache C3 MRO (PyPy: W_TypeObject.mro_w set during ready())
    let mro = unsafe { crate::space::compute_mro_pub(w_type) };
    unsafe { pyre_object::w_type_set_mro(w_type, mro) };
    Ok(w_type)
}

thread_local! {
    /// Execution context for __build_class__ calls.
    /// Set before eval_loop starts so build_class can access it.
    static BUILD_CLASS_EXEC_CTX: Cell<*const crate::PyExecutionContext> =
        const { Cell::new(std::ptr::null()) };
}

/// Set the execution context for __build_class__ to use.
pub fn set_build_class_exec_ctx(ctx: *const crate::PyExecutionContext) {
    BUILD_CLASS_EXEC_CTX.with(|c| c.set(ctx));
}

// ── Type calling (instance creation) ─────────────────────────────────
// PyPy equivalent: typeobject.py descr_call → __new__ + __init__

fn call_type_object(frame: &mut PyFrame, w_type: PyObjectRef, args: &[PyObjectRef]) -> PyResult {
    // Step 1: Look up __new__ via type MRO → allocate instance
    // PyPy: descr_call → w_newfunc = space.lookup(w_type, '__new__')
    //       w_newobject = space.call_obj_args(w_newfunc, w_type, __args__)
    let instance =
        if let Some(new_fn) = unsafe { crate::space::lookup_in_type_mro_pub(w_type, "__new__") } {
            // Call __new__(cls, *args)
            let mut new_args = Vec::with_capacity(1 + args.len());
            new_args.push(w_type);
            new_args.extend_from_slice(args);
            crate::space_call_function(new_fn, &new_args)
        } else {
            // Default: allocate bare instance
            pyre_object::w_instance_new(w_type)
        };

    // Step 2: Look up __init__ via type MRO → initialize instance
    // PyPy: descr_call → space.lookup(w_newobject, '__init__')
    if let Some(init_fn) = unsafe { crate::space::lookup_in_type_mro_pub(w_type, "__init__") } {
        let mut init_args = Vec::with_capacity(1 + args.len());
        init_args.push(instance);
        init_args.extend_from_slice(args);
        let _ = call_callable(frame, init_fn, &init_args)?;
    }

    Ok(instance)
}
