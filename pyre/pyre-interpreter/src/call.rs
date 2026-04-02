//! Function call dispatch — pure interpreter, no JIT dependencies.
//!
//! JIT-specific call infrastructure (force/bridge callbacks, callee frame
//! creation helpers, frame pool) lives in pyre-jit/src/call_jit.rs.

use std::cell::Cell;
use std::sync::OnceLock;

use crate::{
    PyError, PyErrorKind, PyNamespace, PyResult, builtin_code_get, dispatch_callable,
    function_get_closure, function_get_code, function_get_globals,
};
use pyre_object::{PY_NULL, PyObjectRef};

use crate::eval::eval_frame_plain;
use crate::pyframe::{PendingInlineResult, PyFrame};

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
    /// Last known valid execution context — for call_user_function_with_args.
    static LAST_EXEC_CTX: std::cell::Cell<*const crate::PyExecutionContext> =
        const { std::cell::Cell::new(std::ptr::null()) };
}

/// Set the last known execution context (called at eval loop entry).
pub fn set_last_exec_ctx(ctx: *const crate::PyExecutionContext) {
    LAST_EXEC_CTX.with(|c| c.set(ctx));
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
    let code_ptr = unsafe { function_get_code(callable) };
    let globals = unsafe { function_get_globals(callable) };
    let closure = unsafe { function_get_closure(callable) };
    let defaults = unsafe { crate::function_get_defaults(callable) };
    let func_code = code_ptr as *const crate::CodeObject;

    // PyPy: pyframe.py handle_operation_error / init_cells
    // Fill missing positional args from defaults tuple.
    let code_ref = unsafe { &*func_code };
    let nparams = code_ref.arg_count as usize;
    let nargs = args.len();
    let filled_args = if nargs < nparams && !defaults.is_null() {
        let defaults = crate::baseobjspace::unwrap_cell(defaults);
        let mut full = Vec::with_capacity(nparams);
        full.extend_from_slice(args);
        // Defaults cover the LAST (nparams - first_default) parameters.
        // number of defaults = tuple length
        let ndefaults = if unsafe { pyre_object::is_tuple(defaults) } {
            unsafe { pyre_object::w_tuple_len(defaults) }
        } else {
            0
        };
        let first_default = nparams - ndefaults;
        for i in nargs..nparams {
            if i >= first_default {
                let default_idx = i - first_default;
                if let Some(val) =
                    unsafe { pyre_object::w_tuple_getitem(defaults, default_idx as i64) }
                {
                    full.push(val);
                } else {
                    full.push(pyre_object::PY_NULL);
                }
            } else {
                full.push(pyre_object::PY_NULL);
            }
        }
        full
    } else {
        args.to_vec()
    };

    // Fill keyword-only defaults from kwdefaults dict
    let nkwonly = code_ref.kwonlyarg_count as usize;
    let mut filled_args = filled_args;
    if nkwonly > 0 {
        let kwdefaults = unsafe { crate::function_get_kwdefaults(callable) };
        // Ensure filled_args covers all positional + kwonly slots
        while filled_args.len() < nparams + nkwonly {
            filled_args.push(pyre_object::PY_NULL);
        }
        if !kwdefaults.is_null() && unsafe { pyre_object::is_dict(kwdefaults) } {
            for ki in 0..nkwonly {
                let slot = nparams + ki;
                if filled_args[slot].is_null() {
                    let param_name = &code_ref.varnames[slot];
                    let key = pyre_object::w_str_new(param_name);
                    if let Some(val) = unsafe { pyre_object::w_dict_lookup(kwdefaults, key) } {
                        filled_args[slot] = val;
                    }
                }
            }
        }
    }

    let final_args = pack_varargs(code_ref, filled_args);

    // Generator function: create generator object instead of executing.
    // PyPy: generator.py GeneratorIterator.__init__ wraps PyFrame.
    // RustPython compiler uses CodeFlags::GENERATOR instead of RETURN_GENERATOR opcode.
    if code_ref
        .flags
        .intersects(crate::CodeFlags::GENERATOR | crate::CodeFlags::COROUTINE)
    {
        let mut gen_frame = PyFrame::new_for_call_with_closure(
            func_code,
            &final_args,
            globals,
            frame.execution_context,
            closure,
        );
        gen_frame.fix_array_ptrs();
        let frame_ptr = Box::into_raw(Box::new(gen_frame)) as *mut u8;
        return Ok(pyre_object::generatorobject::w_generator_new(frame_ptr));
    }

    let mut func_frame = PyFrame::new_for_call_with_closure(
        func_code,
        &final_args,
        globals,
        frame.execution_context,
        closure,
    );
    func_frame.fix_array_ptrs();
    eval_fn(&mut func_frame)
}

pub fn call_callable(frame: &mut PyFrame, callable: PyObjectRef, args: &[PyObjectRef]) -> PyResult {
    let callable = crate::baseobjspace::unwrap_cell(callable);
    if unsafe { pyre_object::is_method(callable) } {
        let func = unsafe { pyre_object::w_method_get_func(callable) };
        let receiver = unsafe {
            let w_self = pyre_object::w_method_get_self(callable);
            if !w_self.is_null() && !pyre_object::is_none(w_self) {
                w_self
            } else {
                pyre_object::w_method_get_class(callable)
            }
        };
        let mut call_args = Vec::with_capacity(1 + args.len());
        if !receiver.is_null() && unsafe { !pyre_object::is_none(receiver) } {
            call_args.push(receiver);
        }
        call_args.extend_from_slice(args);
        return call_callable(frame, func, &call_args);
    }
    if unsafe { pyre_object::is_type(callable) } {
        return type_descr_call(frame, callable, args);
    }

    // staticmethod → unwrap
    // PyPy: function.py StaticMethod.descr_staticmethod__call__
    if unsafe { pyre_object::is_staticmethod(callable) } {
        let func = unsafe { pyre_object::w_staticmethod_get_func(callable) };
        return call_callable(frame, func, args);
    }
    // classmethod → unwrap
    if unsafe { pyre_object::is_classmethod(callable) } {
        let func = unsafe { pyre_object::w_classmethod_get_func(callable) };
        return call_callable(frame, func, args);
    }

    // Instance with __call__ — PyPy: descroperation.py descr_call
    if unsafe { pyre_object::is_instance(callable) } {
        let w_type = unsafe { pyre_object::w_instance_get_type(callable) };
        if let Some(call_fn) = unsafe { crate::baseobjspace::lookup_in_type(w_type, "__call__") } {
            let mut call_args = Vec::with_capacity(1 + args.len());
            call_args.push(callable);
            call_args.extend_from_slice(args);
            return call_callable(frame, call_fn, &call_args);
        }
    }

    dispatch_callable(
        callable,
        |callable| {
            let func = unsafe { builtin_code_get(callable) };
            func(args)
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
    let code_ptr = unsafe { function_get_code(callable) };
    let globals = unsafe { function_get_globals(callable) };
    let closure = unsafe { function_get_closure(callable) };
    let func_code = code_ptr as *const crate::CodeObject;
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
    if unsafe { pyre_object::is_method(callable) } {
        let func = unsafe { pyre_object::w_method_get_func(callable) };
        let receiver = unsafe {
            let w_self = pyre_object::w_method_get_self(callable);
            if !w_self.is_null() && !pyre_object::is_none(w_self) {
                w_self
            } else {
                pyre_object::w_method_get_class(callable)
            }
        };
        let mut call_args = Vec::with_capacity(1 + args.len());
        if !receiver.is_null() && unsafe { !pyre_object::is_none(receiver) } {
            call_args.push(receiver);
        }
        call_args.extend_from_slice(args);
        return call_callable_inline_residual(frame, func, &call_args);
    }
    dispatch_callable(
        callable,
        |callable| {
            let func = unsafe { builtin_code_get(callable) };
            func(args)
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
/// Resolve keyword arguments into positional order.
///
/// PyPy: argument.py `_match_signature` + `_match_keywords`
///
/// Given:
///   - callable: function with code.varnames defining parameter names
///   - args: [positional_args..., kwarg_values...] (mixed)
///   - kwarg_names: tuple of str names for the last N args
///
/// Returns args rearranged so that keyword values are in the correct
/// parameter positions. This runs BEFORE frame creation so the JIT
/// eval loop sees correctly-positioned locals.
pub(crate) fn resolve_kwargs(
    callable: PyObjectRef,
    args: &[PyObjectRef],
    kwarg_names: PyObjectRef,
) -> Vec<PyObjectRef> {
    if kwarg_names.is_null() {
        return args.to_vec();
    }
    let nkw = if unsafe { pyre_object::is_tuple(kwarg_names) } {
        unsafe { pyre_object::w_tuple_len(kwarg_names) }
    } else {
        return args.to_vec();
    };
    if nkw == 0 {
        return args.to_vec();
    }

    // Resolve the target function's code object.
    // For user functions: direct code_ptr.
    // For type objects: look up __new__ in MRO (PyPy: Arguments used by descr_call).
    //
    // When callable is a type, type_descr_call will prepend `cls` as the first
    // arg to __new__, so the stack args correspond to __new__'s params[1:]
    // (skip_cls=1). For plain function calls skip_cls=0.
    let (target_func, skip_cls) = if unsafe { crate::is_function(callable) } {
        (callable, 0usize)
    } else if unsafe { pyre_object::is_type(callable) } {
        // For type objects, resolve kwargs against the winning metaclass's __new__
        let bases_arg = if args.len() >= nkw + 2 {
            args[1]
        } else {
            pyre_object::PY_NULL
        };
        let w_winner = calculate_metaclass(callable, bases_arg).unwrap_or(callable);
        if let Some(new_fn) = unsafe { crate::baseobjspace::lookup_in_type(w_winner, "__new__") } {
            if unsafe { crate::is_function(new_fn) } {
                (new_fn, 1usize) // __new__(cls, ...) → skip cls
            } else {
                return args.to_vec();
            }
        } else {
            return args.to_vec();
        }
    } else {
        return args.to_vec();
    };

    let code_ptr = unsafe { function_get_code(target_func) };
    let code = unsafe { &*(code_ptr as *const crate::CodeObject) };
    // Total named params = positional + keyword-only
    let total_params = (code.arg_count + code.kwonlyarg_count) as usize;
    // Effective params = params visible to the caller (excludes implicit cls for types)
    let nparams = total_params - skip_cls;
    let n_pos = args.len() - nkw; // number of positional args

    // Start with PY_NULL for all effective params
    let mut result = vec![pyre_object::PY_NULL; nparams];

    // Fill positional args (PyPy: _match_signature step 1)
    for i in 0..n_pos.min(nparams) {
        result[i] = args[i];
    }

    // Match keywords to parameter names (PyPy: _match_keywords)
    // varnames[skip_cls..total_params] are the effective param names
    for ki in 0..nkw {
        let kw_name = unsafe { pyre_object::w_tuple_getitem(kwarg_names, ki as i64) };
        let Some(kw_name_obj) = kw_name else { continue };
        let kw_str = unsafe { pyre_object::w_str_get_value(kw_name_obj) };
        let kw_value = args[n_pos + ki];

        let mut matched = false;
        for pi in 0..nparams {
            if &*code.varnames[skip_cls + pi] == kw_str {
                result[pi] = kw_value;
                matched = true;
                break;
            }
        }
        if !matched {
            // Extra kwarg — not yet handled (**kwargs).
        }
    }

    // Fill positional defaults (PyPy: _match_signature defs_w)
    // Defaults cover the LAST N of the positional params (arg_count).
    let n_pos_params = code.arg_count as usize - skip_cls;
    let defaults = unsafe { crate::function_get_defaults(target_func) };
    if !defaults.is_null() {
        let defaults = crate::baseobjspace::unwrap_cell(defaults);
        if unsafe { pyre_object::is_tuple(defaults) } {
            let ndefaults = unsafe { pyre_object::w_tuple_len(defaults) };
            let first_default = n_pos_params.saturating_sub(ndefaults);
            for pi in first_default..n_pos_params {
                if result[pi].is_null() {
                    let di = pi - first_default;
                    if let Some(v) = unsafe { pyre_object::w_tuple_getitem(defaults, di as i64) } {
                        result[pi] = v;
                    }
                }
            }
        }
    }

    // Fill keyword-only defaults from kwdefaults dict
    // PyPy: _match_signature fills from w_kw_defs
    let kwdefaults = unsafe { crate::function_get_kwdefaults(target_func) };
    if !kwdefaults.is_null() && unsafe { pyre_object::is_dict(kwdefaults) } {
        let nkwonly = code.kwonlyarg_count as usize;
        for ki in 0..nkwonly {
            let pi = n_pos_params + ki; // position in result
            if result[pi].is_null() {
                let param_name = &code.varnames[skip_cls + pi];
                let key = pyre_object::w_str_new(param_name);
                if let Some(val) = unsafe { pyre_object::w_dict_lookup(kwdefaults, key) } {
                    result[pi] = val;
                }
            }
        }
    }

    result
}

pub fn register_build_class() {
    crate::typedef::init_typeobjects();
}

/// `ObjSpace.call_function(callable, *args)` — direct implementation.
///
/// PyPy: baseobjspace.py `call_function`. Now a direct function call
/// (no callback — interpreter and runtime are in the same crate).
/// PyPy: baseobjspace.py `call_function`
///
/// Dispatches to builtins, user functions, and type objects.
/// Type call uses the same __new__ + __init__ protocol as type_descr_call.
pub(crate) fn call_function_impl(callable: PyObjectRef, args: &[PyObjectRef]) -> PyObjectRef {
    unsafe {
        if pyre_object::is_method(callable) {
            let func = pyre_object::w_method_get_func(callable);
            let w_self = pyre_object::w_method_get_self(callable);
            let receiver = if !w_self.is_null() && !pyre_object::is_none(w_self) {
                w_self
            } else {
                pyre_object::w_method_get_class(callable)
            };
            let mut call_args = Vec::with_capacity(1 + args.len());
            if !receiver.is_null() && !pyre_object::is_none(receiver) {
                call_args.push(receiver);
            }
            call_args.extend_from_slice(args);
            return call_function_impl(func, &call_args);
        }
        // Builtin function: direct Rust call
        if crate::is_builtin_code(callable) {
            let func = crate::builtin_code_get(callable);
            return match func(args) {
                Ok(result) => result,
                Err(e) => {
                    if std::env::var("PYRE_DEBUG_CALL").is_ok() {
                        eprintln!("[call_function_impl] builtin error: {}", e.message);
                    }
                    pyre_object::w_none()
                }
            };
        }
        // User function: create frame + eval
        if crate::is_function(callable) {
            return call_user_function_with_args(callable, args);
        }
        // Type object → descr_call: __new__ + __init__
        // PyPy: typeobject.py descr_call → lookup __new__, call, then __init__
        if pyre_object::is_type(callable) {
            return type_descr_call_impl(callable, args);
        }
        // staticmethod → unwrap and call the wrapped function
        // PyPy: function.py StaticMethod.descr_staticmethod__call__
        if pyre_object::is_staticmethod(callable) {
            let func = pyre_object::w_staticmethod_get_func(callable);
            return call_function_impl(func, args);
        }
        // classmethod → unwrap and call the wrapped function
        // PyPy: function.py ClassMethod.descr_classmethod__call__
        if pyre_object::is_classmethod(callable) {
            let func = pyre_object::w_classmethod_get_func(callable);
            return call_function_impl(func, args);
        }
        // Instance with __call__ — PyPy: descroperation.py
        if pyre_object::is_instance(callable) {
            let w_type = pyre_object::w_instance_get_type(callable);
            if let Some(call_fn) = crate::baseobjspace::lookup_in_type(w_type, "__call__") {
                let mut call_args = Vec::with_capacity(1 + args.len());
                call_args.push(callable);
                call_args.extend_from_slice(args);
                return call_function_impl(call_fn, &call_args);
            }
        }
    }
    panic!("call_function: '{}' object is not callable", unsafe {
        (*(*callable).ob_type).tp_name
    });
}

/// CPython: typeobject.c calculate_metaclass
pub(crate) fn calculate_metaclass(
    mut w_winner: PyObjectRef,
    bases: PyObjectRef,
) -> Result<PyObjectRef, PyError> {
    if w_winner.is_null() {
        w_winner = crate::typedef::w_type();
    }
    if bases.is_null() || unsafe { !pyre_object::is_tuple(bases) } {
        return Ok(w_winner);
    }
    let n = unsafe { pyre_object::w_tuple_len(bases) };
    for i in 0..n {
        let Some(base) = (unsafe { pyre_object::w_tuple_getitem(bases, i as i64) }) else {
            continue;
        };
        let Some(w_base_type) = crate::typedef::r#type(base) else {
            continue;
        };
        if std::ptr::eq(w_winner, w_base_type) || issubtype_ptr(w_winner, w_base_type) {
            continue;
        }
        if issubtype_ptr(w_base_type, w_winner) {
            w_winner = w_base_type;
            continue;
        }
        return Err(PyError::type_error("metaclass conflict"));
    }
    Ok(w_winner)
}

/// Type call without a PyFrame.
/// PyPy: typeobject.py descr_call
fn type_descr_call_impl(w_type: PyObjectRef, args: &[PyObjectRef]) -> PyObjectRef {
    // Step 1: __new__
    let instance =
        if let Some(new_fn) = unsafe { crate::baseobjspace::lookup_in_type(w_type, "__new__") } {
            let mut new_args = Vec::with_capacity(1 + args.len());
            new_args.push(w_type);
            new_args.extend_from_slice(args);
            call_function_impl(new_fn, &new_args)
        } else {
            pyre_object::w_instance_new(w_type)
        };

    // Step 2: __init__ — only if __new__ returned an instance of w_type
    // PyPy: descr_call skips __init__ when __new__ returns a different type
    if !instance.is_null() && unsafe { pyre_object::is_instance(instance) } {
        let w_insttype = unsafe { pyre_object::w_instance_get_type(instance) };
        if std::ptr::eq(w_insttype, w_type) || issubtype_ptr(w_insttype, w_type) {
            if let Some(init_fn) =
                unsafe { crate::baseobjspace::lookup_in_type(w_type, "__init__") }
            {
                let mut init_args = Vec::with_capacity(1 + args.len());
                init_args.push(instance);
                init_args.extend_from_slice(args);
                let _ = call_function_impl(init_fn, &init_args);
            }
        }
    }

    instance
}

/// Pointer-based subtype check for descr_call __init__ guard.
fn issubtype_ptr(w_type: PyObjectRef, cls: PyObjectRef) -> bool {
    let mro_ptr = unsafe { pyre_object::w_type_get_mro(w_type) };
    if mro_ptr.is_null() {
        return false;
    }
    unsafe { (*mro_ptr).iter().any(|&t| std::ptr::eq(t, cls)) }
}

/// Helper: call a user function with arbitrary args from descriptor context.
fn call_user_function_with_args(func: PyObjectRef, args: &[PyObjectRef]) -> PyObjectRef {
    let code_ptr = unsafe { function_get_code(func) };
    let globals = unsafe { function_get_globals(func) };
    let closure = unsafe { function_get_closure(func) };
    let defaults = unsafe { crate::function_get_defaults(func) };
    let func_code = code_ptr as *const crate::CodeObject;
    let exec_ctx = BUILD_CLASS_EXEC_CTX.with(|c| c.get());
    let exec_ctx = if exec_ctx.is_null() {
        LAST_EXEC_CTX.with(|c| c.get())
    } else {
        exec_ctx
    };

    // Fill defaults for missing args
    let code_ref = unsafe { &*func_code };
    let nparams = code_ref.arg_count as usize;
    let nargs = args.len();
    let filled_args = if nargs < nparams && !defaults.is_null() {
        let defaults = crate::baseobjspace::unwrap_cell(defaults);
        let mut full = Vec::with_capacity(nparams);
        full.extend_from_slice(args);
        let ndefaults = if unsafe { pyre_object::is_tuple(defaults) } {
            unsafe { pyre_object::w_tuple_len(defaults) }
        } else {
            0
        };
        let first_default = nparams - ndefaults;
        for i in nargs..nparams {
            if i >= first_default {
                let di = i - first_default;
                full.push(
                    unsafe { pyre_object::w_tuple_getitem(defaults, di as i64) }.unwrap_or(PY_NULL),
                );
            } else {
                full.push(PY_NULL);
            }
        }
        full
    } else {
        args.to_vec()
    };

    // Fill keyword-only defaults (same logic as call_user_function_with_eval)
    let nkwonly = code_ref.kwonlyarg_count as usize;
    let mut filled_args = filled_args;
    if nkwonly > 0 {
        let kwdefaults = unsafe { crate::function_get_kwdefaults(func) };
        while filled_args.len() < nparams + nkwonly {
            filled_args.push(PY_NULL);
        }
        if !kwdefaults.is_null() && unsafe { pyre_object::is_dict(kwdefaults) } {
            for ki in 0..nkwonly {
                let slot = nparams + ki;
                if filled_args[slot].is_null() {
                    let param_name = &code_ref.varnames[slot];
                    let key = pyre_object::w_str_new(param_name);
                    if let Some(val) = unsafe { pyre_object::w_dict_lookup(kwdefaults, key) } {
                        filled_args[slot] = val;
                    }
                }
            }
        }
    }

    let final_args = pack_varargs(code_ref, filled_args);

    // Generator function: wrap frame in generator object
    if code_ref
        .flags
        .intersects(crate::CodeFlags::GENERATOR | crate::CodeFlags::COROUTINE)
    {
        let mut gen_frame =
            PyFrame::new_for_call_with_closure(func_code, &final_args, globals, exec_ctx, closure);
        gen_frame.fix_array_ptrs();
        let frame_ptr = Box::into_raw(Box::new(gen_frame)) as *mut u8;
        return pyre_object::generatorobject::w_generator_new(frame_ptr);
    }

    let mut frame =
        PyFrame::new_for_call_with_closure(func_code, &final_args, globals, exec_ctx, closure);
    frame.fix_array_ptrs();
    match eval_frame_plain(&mut frame) {
        Ok(v) => v,
        Err(e) => {
            if std::env::var("PYRE_DEBUG_CALL").is_ok() {
                eprintln!("[call_user_function_with_args] error: {e}");
            }
            PY_NULL
        }
    }
}

/// Call a metaclass with extra keyword arguments.
///
/// PyPy: metaclass(name, bases, namespace, **kwds).
/// Resolves kwargs to the metaclass __new__'s kwonly / **kwds parameters.
fn call_metaclass_with_kwargs(
    w_metaclass: PyObjectRef,
    name: PyObjectRef,
    bases: PyObjectRef,
    w_namespace_dict: PyObjectRef,
    kwargs: PyObjectRef,
) -> PyObjectRef {
    // Find the metaclass __new__ method
    let new_fn = if unsafe { pyre_object::is_type(w_metaclass) } {
        unsafe { crate::baseobjspace::lookup_in_type(w_metaclass, "__new__") }
    } else {
        None
    };

    if let Some(new_fn) = new_fn {
        if unsafe { crate::is_function(new_fn) } {
            // User function: resolve kwargs to kwonly params
            let code_ptr = unsafe { function_get_code(new_fn) };
            let code = unsafe { &*(code_ptr as *const crate::CodeObject) };
            let nparams = code.arg_count as usize; // positional params
            let nkwonly = code.kwonlyarg_count as usize;

            // Build positional args: [mcs, name, bases, ns_dict]
            let mut args = vec![w_metaclass, name, bases, w_namespace_dict];

            // Fill kwonly params from kwargs dict
            for ki in 0..nkwonly {
                let param_idx = nparams + ki;
                if param_idx < code.varnames.len() {
                    let param_name = &code.varnames[param_idx];
                    let key = pyre_object::w_str_new(param_name);
                    if let Some(val) = unsafe { pyre_object::w_dict_lookup(kwargs, key) } {
                        args.push(val);
                    } else {
                        args.push(pyre_object::PY_NULL); // will be filled by defaults
                    }
                }
            }

            return call_user_function_with_args(new_fn, &args);
        }
    }

    // Fallback: call without kwargs
    crate::call_function(w_metaclass, &[name, bases, w_namespace_dict])
}

/// Pack excess positional args into *args tuple, add empty **kwargs dict.
/// PyPy: argument.py _match_signature varargs/varkeywords packing
fn pack_varargs(code: &crate::CodeObject, args: Vec<PyObjectRef>) -> Vec<PyObjectRef> {
    let nparams = (code.arg_count + code.kwonlyarg_count) as usize;
    let has_varargs = code.flags.contains(crate::CodeFlags::VARARGS);
    let has_varkw = code.flags.contains(crate::CodeFlags::VARKEYWORDS);

    if !has_varargs && !has_varkw {
        return args;
    }

    let mut packed = Vec::with_capacity(nparams + 2);
    // Regular positional args
    for i in 0..nparams.min(args.len()) {
        packed.push(args[i]);
    }
    // Fill missing params with PY_NULL
    while packed.len() < nparams {
        packed.push(pyre_object::PY_NULL);
    }
    if has_varargs {
        let extra: Vec<_> = if args.len() > nparams {
            args[nparams..].to_vec()
        } else {
            vec![]
        };
        packed.push(pyre_object::w_tuple_new(extra));
    }
    if has_varkw {
        packed.push(pyre_object::w_dict_new());
    }
    packed
}

/// The real __build_class__(body_fn, name, *bases) implementation.
///
/// PyPy equivalent: pyopcode.py BUILD_CLASS →
///   w_methodsdict = call(body_fn)
///   w_newclass = call(metaclass, name, bases, methodsdict)
/// `__build_class__(func, name, *bases, metaclass=None, **kwds)`
///
/// PyPy: pyopcode.py BUILD_CLASS → build_class()
pub(crate) fn real_build_class(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.len() < 2 {
        return Err(crate::PyError::type_error(
            "__build_class__ requires at least 2 arguments",
        ));
    }
    let body_fn = args[0];
    let name_obj = args[1];

    // Check if last arg is a kwargs dict (from CALL_KW)
    // PyPy: __build_class__(func, name, *bases, metaclass=None, **kwds)
    let (base_args, metaclass, extra_kwargs) = if args.len() > 2 {
        let last = args[args.len() - 1];
        if unsafe { pyre_object::is_dict(last) }
            && unsafe {
                pyre_object::w_dict_lookup(last, pyre_object::w_str_new("__pyre_kw__")).is_some()
            }
        {
            let w_metaclass =
                unsafe { pyre_object::w_dict_lookup(last, pyre_object::w_str_new("metaclass")) };
            // Collect extra kwargs (not metaclass, not __pyre_kw__)
            let extra = pyre_object::w_dict_new();
            unsafe {
                let d = &*(last as *const pyre_object::dictobject::W_DictObject);
                for &(k, v) in &*d.entries {
                    if pyre_object::is_str(k) {
                        let key = pyre_object::w_str_get_value(k);
                        if key != "metaclass" && key != "__pyre_kw__" {
                            pyre_object::w_dict_store(extra, k, v);
                        }
                    }
                }
            }
            (&args[2..args.len() - 1], w_metaclass, Some(extra))
        } else {
            (&args[2..], None, None)
        }
    } else {
        (&args[2..], None, None)
    };

    let name = unsafe { pyre_object::w_str_get_value(name_obj) };
    let bases_tuple = pyre_object::w_tuple_new(base_args.to_vec());

    // If no explicit metaclass, infer from bases (PyPy: calculate_metaclass)
    let w_metaclass = metaclass.or_else(|| {
        unsafe {
            if !pyre_object::is_tuple(bases_tuple) {
                return None;
            }
            let n = pyre_object::w_tuple_len(bases_tuple);
            for i in 0..n {
                if let Some(base) = pyre_object::w_tuple_getitem(bases_tuple, i as i64) {
                    if pyre_object::is_type(base) {
                        // Check if base has a custom metaclass
                        let w_metaclass = crate::baseobjspace::ATTR_TABLE.with(|table| {
                            table
                                .borrow()
                                .get(&(base as usize))
                                .and_then(|d| d.get("__metaclass__").copied())
                        });
                        if w_metaclass.is_some() {
                            return w_metaclass;
                        }
                    }
                }
            }
        }
        None
    });

    build_class_inner(body_fn, name, bases_tuple, w_metaclass, extra_kwargs)
}

fn build_class_inner(
    body_fn: PyObjectRef,
    name: &str,
    bases: PyObjectRef,
    w_metaclass: Option<PyObjectRef>,
    extra_kwargs: Option<PyObjectRef>,
) -> PyResult {
    let code_ptr = unsafe { function_get_code(body_fn) };
    let globals = unsafe { function_get_globals(body_fn) };
    let closure = unsafe { function_get_closure(body_fn) };
    let func_code = code_ptr as *const crate::CodeObject;

    // Call metaclass.__prepare__(name, bases, **kwds) if it exists.
    // PyPy: build_class → metaclass.__prepare__(name, bases, **kwds)
    // Returns the namespace dict to use for the class body.
    let w_namespace = if let Some(w_metaclass) = w_metaclass {
        if unsafe { pyre_object::is_type(w_metaclass) } {
            match crate::baseobjspace::getattr(w_metaclass, "__prepare__") {
                Ok(prepare) => {
                    let ns_obj =
                        crate::call_function(prepare, &[pyre_object::w_str_new(name), bases]);
                    if !ns_obj.is_null() && unsafe { !pyre_object::is_none(ns_obj) } {
                        Some(ns_obj)
                    } else {
                        None
                    }
                }
                Err(_) => None,
            }
        } else {
            None
        }
    } else {
        None
    };

    // Create class namespace — use __prepare__ result or fresh namespace.
    // __prepare__ may return a dict subclass (e.g. EnumDict).
    // dict subclass instances created by w_instance_new store entries in
    // ATTR_TABLE, not in W_DictObject.entries. We handle both cases.
    let mut class_ns = Box::new(PyNamespace::new());
    if let Some(w_prepared_dict) = w_namespace {
        if unsafe { pyre_object::is_dict(w_prepared_dict) } {
            let dict =
                unsafe { &*(w_prepared_dict as *const pyre_object::dictobject::W_DictObject) };
            for &(key, value) in unsafe { &*dict.entries } {
                if !value.is_null() && unsafe { pyre_object::is_str(key) } {
                    crate::namespace_store(
                        &mut class_ns,
                        unsafe { pyre_object::w_str_get_value(key) },
                        value,
                    );
                }
            }
        }
        // dict subclass instance (e.g. EnumDict): backing dict via __dict_data__
        if unsafe { pyre_object::is_instance(w_prepared_dict) } {
            let backing = crate::type_methods::resolve_dict_backing(w_prepared_dict);
            if !backing.is_null() && unsafe { pyre_object::is_dict(backing) } {
                let dict = unsafe { &*(backing as *const pyre_object::dictobject::W_DictObject) };
                for &(key, value) in unsafe { &*dict.entries } {
                    if !value.is_null() && unsafe { pyre_object::is_str(key) } {
                        crate::namespace_store(
                            &mut class_ns,
                            unsafe { pyre_object::w_str_get_value(key) },
                            value,
                        );
                    }
                }
            }
        }
    }
    class_ns.fix_ptr();
    let class_ns_ptr = Box::into_raw(class_ns);

    // w_namespace: if __prepare__ returned a custom dict, we'll replay
    // class body stores into it after execution. This lets EnumDict etc.
    // track member definitions via __setitem__.

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
    // PyPy: typeobject.py — if not bases_w: bases_w = [space.w_object]
    let w_effective_bases = if bases.is_null()
        || !unsafe { pyre_object::is_tuple(bases) }
        || unsafe { pyre_object::w_tuple_len(bases) } == 0
    {
        let w_object = crate::typedef::w_object();
        if !w_object.is_null() {
            pyre_object::w_tuple_new(vec![w_object])
        } else {
            bases
        }
    } else {
        bases
    };
    // Create class via metaclass or default type()
    // PyPy: typeobject.py — metaclass(name, bases, dict_w) or type.__new__
    let w_type = if let Some(w_metaclass) = w_metaclass {
        // Convert class namespace to a dict for metaclass call.
        // If __prepare__ returned a custom dict, replay stores into it
        // so that __setitem__ side effects (e.g. EnumDict tracking) fire.
        let w_namespace_dict = if let Some(w_prepared_dict) = w_namespace {
            // Replay class body stores into prepared dict
            let ns = unsafe { &*class_ns_ptr };
            for (k, &v) in ns.entries() {
                if !v.is_null() {
                    let key = pyre_object::w_str_new(k);
                    // Use setitem to trigger __setitem__ on EnumDict etc.
                    let _ = crate::baseobjspace::setitem(w_prepared_dict, key, v);
                }
            }
            w_prepared_dict
        } else {
            let d = pyre_object::w_dict_new();
            let ns = unsafe { &*class_ns_ptr };
            for (k, &v) in ns.entries() {
                if !v.is_null() {
                    unsafe { pyre_object::w_dict_store(d, pyre_object::w_str_new(k), v) };
                }
            }
            d
        };
        // Call metaclass(name, bases, namespace, **kwds)
        // Pass the ORIGINAL bases (not w_effective_bases) — the metaclass
        // expects the user-declared bases. Default (object,) is added by
        // type.__new__ internally if needed.
        let name_obj = pyre_object::w_str_new(name);
        let result = if let Some(kw) = extra_kwargs {
            call_metaclass_with_kwargs(w_metaclass, name_obj, bases, w_namespace_dict, kw)
        } else {
            crate::call_function(w_metaclass, &[name_obj, bases, w_namespace_dict])
        };
        // If metaclass returned a W_TypeObject, set up MRO and store metaclass ref.
        if unsafe { pyre_object::is_type(result) } {
            let mro = unsafe { crate::baseobjspace::compute_default_mro(result) };
            unsafe { pyre_object::w_type_set_mro(result, mro) };
            // Store metaclass reference in ATTR_TABLE (not class dict)
            // for metatype attribute lookup
            crate::baseobjspace::ATTR_TABLE.with(|table| {
                table
                    .borrow_mut()
                    .entry(result as usize)
                    .or_default()
                    .insert("__metaclass__".to_string(), w_metaclass);
            });
        }
        result
    } else {
        let w = pyre_object::w_type_new(name, w_effective_bases, class_ns_ptr as *mut u8);
        let mro = unsafe { crate::baseobjspace::compute_default_mro(w) };
        unsafe { pyre_object::w_type_set_mro(w, mro) };
        w
    };

    // __set_name__ protocol — CPython: type_new_set_names
    // PyPy: typeobject.py type_new → call __set_name__(owner, name) on each descriptor
    if unsafe { pyre_object::is_type(w_type) } {
        let ns = unsafe { &*class_ns_ptr };
        let entries: Vec<(String, PyObjectRef)> =
            ns.entries().map(|(k, &v)| (k.to_string(), v)).collect();
        for (attr_name, value) in entries {
            if !value.is_null() {
                if let Ok(set_name) = crate::baseobjspace::getattr(value, "__set_name__") {
                    // Call: descriptor.__set_name__(self, owner, name)
                    let _ = crate::call_function(
                        set_name,
                        &[value, w_type, pyre_object::w_str_new(&attr_name)],
                    );
                }
            }
        }
    }

    // CPython: if __classcell__ is in the namespace, set the cell's content
    // to the newly created class. This enables `__class__` references in methods.
    let class_ns = unsafe { &*class_ns_ptr };
    if let Some(&classcell) = class_ns.get("__classcell__") {
        if !classcell.is_null() && unsafe { pyre_object::is_cell(classcell) } {
            unsafe { pyre_object::w_cell_set(classcell, w_type) };
        }
    }

    // Call __init_subclass__ on each base class
    // PyPy: typeobject.py type.__init__ → call __init_subclass__
    if !w_effective_bases.is_null() && unsafe { pyre_object::is_tuple(w_effective_bases) } {
        let n = unsafe { pyre_object::w_tuple_len(w_effective_bases) };
        for i in 0..n {
            if let Some(base) = unsafe { pyre_object::w_tuple_getitem(w_effective_bases, i as i64) }
            {
                if unsafe { pyre_object::is_type(base) } {
                    if let Some(init_sub) =
                        unsafe { crate::baseobjspace::lookup_in_type(base, "__init_subclass__") }
                    {
                        let _ = crate::call_function(init_sub, &[w_type]);
                    }
                }
            }
        }
    }

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

fn type_descr_call(frame: &mut PyFrame, w_type: PyObjectRef, args: &[PyObjectRef]) -> PyResult {
    // Step 1: Look up __new__ via type MRO → allocate instance
    // PyPy: typeobject.py descr_call → w_type.lookup_where('__new__'),
    // then bind/call the resulting descriptor with w_type as the first arg.
    let instance =
        if let Some(new_fn) = unsafe { crate::baseobjspace::lookup_in_type(w_type, "__new__") } {
            // Call __new__(cls, *args)
            let mut new_args = Vec::with_capacity(1 + args.len());
            new_args.push(w_type);
            new_args.extend_from_slice(args);
            call_callable(frame, new_fn, &new_args)?
        } else {
            // Default: allocate bare instance
            pyre_object::w_instance_new(w_type)
        };

    // Step 2: __init__ — only if __new__ returned an instance of w_type.
    // PyPy: descr_call — skips __init__ when __new__ returns a foreign type.
    let call_init = if !instance.is_null() && unsafe { pyre_object::is_instance(instance) } {
        let w_insttype = unsafe { pyre_object::w_instance_get_type(instance) };
        std::ptr::eq(w_insttype, w_type) || issubtype_ptr(w_insttype, w_type)
    } else {
        false
    };

    if call_init {
        if let Some(init_fn) = unsafe { crate::baseobjspace::lookup_in_type(w_type, "__init__") } {
            let mut init_args = Vec::with_capacity(1 + args.len());
            init_args.push(instance);
            init_args.extend_from_slice(args);
            let _ = call_callable(frame, init_fn, &init_args)?;
        }
    }

    Ok(instance)
}
