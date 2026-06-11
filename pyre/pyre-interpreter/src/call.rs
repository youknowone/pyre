//! Function call dispatch — pure interpreter, no JIT dependencies.
//!
//! JIT-specific call infrastructure (force/bridge callbacks, callee frame
//! creation helpers, frame pool) lives in pyre-jit/src/call_jit.rs.

use std::cell::{Cell, RefCell};
use std::sync::OnceLock;

use crate::{
    DictStorage, PyError, PyResult, builtin_code_get, dispatch_callable, function_get_closure,
    function_get_globals, function_get_globals_obj,
};

struct FrameLocalsRoot {
    slot: *mut *mut u8,
    registered: bool,
}

impl FrameLocalsRoot {
    fn new(frame: &PyFrame) -> Self {
        let frame = frame as *const PyFrame as *mut PyFrame;
        let slot = unsafe { std::ptr::addr_of_mut!((*frame).locals_cells_stack_w) } as *mut *mut u8;
        let registered = unsafe { pyre_object::gc_hook::try_gc_add_root(slot) };
        Self { slot, registered }
    }

    fn new_mut(frame: &mut PyFrame) -> Self {
        Self::new(frame)
    }
}

impl Drop for FrameLocalsRoot {
    fn drop(&mut self) {
        if self.registered {
            pyre_object::gc_hook::try_gc_remove_root(self.slot);
        }
    }
}

thread_local! {
    /// Most recent error swallowed by `call_function_impl` /
    /// `call_user_function_with_args`. These functions return a bare
    /// `PyObjectRef` for legacy reasons; when the underlying call raises,
    /// they stash the error here and return PY_NULL / w_none() so that
    /// callers that need the real error can pull it back out via
    /// `take_call_error()`.
    ///
    /// Pattern is the inverse of CPython's `PyErr_Occurred()` — see
    /// `pyerrors.c`. Same idea: a thread-local error indicator paired with
    /// out-of-band NULL returns.
    static PENDING_CALL_ERROR: RefCell<Option<PyError>> = const { RefCell::new(None) };
}

/// Stash an error from the bare-PyObjectRef call path so a caller that
/// recognizes the NULL return can recover the original PyError.
pub fn set_call_error(e: PyError) {
    PENDING_CALL_ERROR.with(|slot| {
        *slot.borrow_mut() = Some(e);
    });
}

/// Take and clear the most recent stashed call error. Returns None if no
/// error is pending. Callers must pair this with the bare-return call
/// helpers (`call_function_impl`, `call_function_impl_raw`,
/// `call_user_function_with_args`) immediately after the call so the
/// error refers to the most recent failed dispatch.
pub fn take_call_error() -> Option<PyError> {
    PENDING_CALL_ERROR.with(|slot| slot.borrow_mut().take())
}

/// Clear any pending stashed error without consuming it.
pub fn clear_call_error() {
    PENDING_CALL_ERROR.with(|slot| {
        slot.borrow_mut().take();
    });
}
use pyre_object::{PY_NULL, PyObjectRef};

use crate::eval::eval_frame_plain;
use crate::pyframe::PyFrame;

// ── Eval function injection ──────────────────────────────────────
type EvalFn = fn(&mut PyFrame) -> PyResult;
static EVAL_OVERRIDE: OnceLock<EvalFn> = OnceLock::new();

type DepthBumpFn = fn() -> Option<Box<dyn std::any::Any>>;
static DEPTH_BUMP_OVERRIDE: OnceLock<DepthBumpFn> = OnceLock::new();

thread_local! {
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

/// Increment call depth and return an RAII guard that decrements on drop.
/// Used by _flat_pycall to match call_user_function's depth tracking.
#[inline(always)]
pub fn increment_call_depth() -> CallDepthGuardPublic {
    CALL_DEPTH.with(|d| d.set(d.get() + 1));
    CallDepthGuardPublic
}

/// RAII guard that decrements CALL_DEPTH on drop.
pub struct CallDepthGuardPublic;
impl Drop for CallDepthGuardPublic {
    #[inline(always)]
    fn drop(&mut self) {
        CALL_DEPTH.with(|d| d.set(d.get().saturating_sub(1)));
    }
}

/// Register the JIT-aware eval function. Called by pyre-jit at startup.
pub fn register_eval_override(f: EvalFn) {
    let _ = EVAL_OVERRIDE.set(f);
}

/// Get the current eval function (JIT-aware if registered, plain otherwise).
/// Respects the force-plain-eval mode.
#[inline]
pub fn get_eval_fn() -> fn(&mut PyFrame) -> PyResult {
    let plain_mode = FORCE_PLAIN_EVAL.with(|c| c.get() > 0);
    if plain_mode {
        eval_frame_plain
    } else {
        EVAL_OVERRIDE.get().copied().unwrap_or(eval_frame_plain)
    }
}

// ── JIT parameter injection ──────────────────────────────────────
//
// `pypy/interpreter/executioncontext.py:296-298 settrace` invokes
// `jit.set_param(None, 'trace_limit', 10000)` on the global default
// jitdriver to widen the trace budget while a tracefunc is installed.
// pyre-interpreter cannot import pyre-jit (its lower-layer crate), so
// the JIT side registers a hook at boot that pyre-interpreter calls
// through.  Mirrors the `EVAL_OVERRIDE` pattern above.
type SetJitParamFn = fn(name: &str, value: i64);
static SET_JIT_PARAM_HOOK: OnceLock<SetJitParamFn> = OnceLock::new();

/// Register the hook that forwards `set_jit_param` calls into the JIT
/// runtime's `WarmState::set_param`. Called by pyre-jit at startup.
pub fn register_set_jit_param_hook(f: SetJitParamFn) {
    let _ = SET_JIT_PARAM_HOOK.set(f);
}

/// `rlib/jit.py:818 jit.set_param(driver=None, name, value)` analogue.
/// No-op when pyre-jit has not registered the hook (e.g. JIT-disabled
/// builds or boot-time callers that fire before the first `eval_with_jit`
/// invocation).
pub fn set_jit_param(name: &str, value: i64) {
    if let Some(hook) = SET_JIT_PARAM_HOOK.get() {
        hook(name, value);
    }
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

/// Snapshot the current thread-local execution context. Residual callers
/// that need to temporarily pin a different context (blackhole's
/// `bh_call_fn_impl` cold path, for example) pair this with
/// `set_last_exec_ctx` to restore the prior value on return.
pub fn take_last_exec_ctx() -> *const crate::PyExecutionContext {
    LAST_EXEC_CTX.with(|c| c.get())
}

/// `pypy/objspace/std/objspace.py space.getexecutioncontext()` analogue.
///
/// PyPy walks thread state and returns the live `ExecutionContext`,
/// creating one on demand.  Pyre stores the active context in a TLS
/// slot seeded at process boot by pyrex (`pyrex/src/lib.rs:185
/// set_last_exec_ctx(Rc::as_ptr(&execution_context))`) and
/// re-stamped on every `eval_frame_plain` entry.  The slot stays
/// pointing at the root EC for the lifetime of the process, so
/// `sys.gettrace`/`settrace`/`getprofile`/`setprofile` and other
/// `space.getexecutioncontext()` callers see the live EC even when
/// no eval frame is currently on the stack.
///
/// TODO: pyre is single-threaded today so the TLS
/// slot is effectively a global.  PyPy's per-thread `threadlocals`
/// dispatch lands when pyre adds its own thread state container.
pub fn getexecutioncontext() -> *const crate::PyExecutionContext {
    take_last_exec_ctx()
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

/// Check if force-plain-eval mode is active.
pub fn is_force_plain_eval() -> bool {
    FORCE_PLAIN_EVAL.with(|c| c.get() > 0)
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

/// Fill positional defaults, kw-only defaults, and pack varargs for a
/// user-function call.  Shared by `call_user_function_with_eval`,
/// `call_user_function_plain_with_ctx` and `call_user_function_with_args`
/// so all positional-only entries apply the same
/// `function.py:217` _flat_pycall_defaults + `argument.py:170-338`
/// _match_signature subset (positional-only — no kwargs path).
///
/// Raises TypeError on too-many positional args (no `*args` to absorb
/// overflow) and on missing required positional / keyword-only args after
/// defaults application, mirroring `argument.py:289-300` ArgErrTooMany and
/// `argument.py:335-338` ArgErrMissing.
fn fill_user_function_args(
    callable: PyObjectRef,
    code_ref: &crate::CodeObject,
    args: &[PyObjectRef],
) -> Result<Vec<PyObjectRef>, crate::PyError> {
    let defaults = unsafe { crate::function_get_defaults(callable) };
    let nparams = code_ref.arg_count as usize;
    let nkwonly = code_ref.kwonlyarg_count as usize;
    let nargs = args.len();
    let has_varargs = code_ref.flags.contains(crate::CodeFlags::VARARGS);

    // argument.py:235-236 — too_many_args when no *vararg to absorb.
    if nargs > nparams && !has_varargs {
        let fname = unsafe { crate::function_get_qualname(callable) };
        let ndefaults = if !defaults.is_null() {
            let defaults = crate::baseobjspace::unwrap_cell(defaults);
            if unsafe { pyre_object::is_tuple(defaults) } {
                unsafe { pyre_object::w_tuple_len(defaults) }
            } else {
                0
            }
        } else {
            0
        };
        let takes_str = if ndefaults > 0 {
            format!(
                "from {} to {} positional arguments",
                nparams - ndefaults,
                nparams
            )
        } else {
            format!(
                "{} positional argument{}",
                nparams,
                if nparams != 1 { "s" } else { "" }
            )
        };
        let given_str = format!("{} {}", nargs, if nargs != 1 { "were" } else { "was" });
        return Err(crate::PyError::type_error(format!(
            "{}() takes {} but {} given",
            fname, takes_str, given_str
        )));
    }

    // Lay out filled_args as `[positional[0..nparams], kwonly[0..nkwonly]]`
    // so the layout matches `pack_varargs`'s expectation that
    // `args[total_params..]` is positional overflow destined for `*args`.
    // Without this split, positional overflow would sit in kwonly slots when
    // `has_varargs && nargs > nparams && nkwonly > 0`
    // (`def f(a, *args, b=K): f(1, 2, 3)` would put `2` in `b`'s slot).
    let total_params = nparams + nkwonly;
    let mut filled_args: Vec<PyObjectRef> = Vec::with_capacity(total_params);
    let n_pos_copied = nargs.min(nparams);
    filled_args.extend_from_slice(&args[..n_pos_copied]);
    for _ in n_pos_copied..total_params {
        filled_args.push(pyre_object::PY_NULL);
    }

    // Fill positional defaults for slots [n_pos_copied..nparams).
    if n_pos_copied < nparams && !defaults.is_null() {
        let defaults = crate::baseobjspace::unwrap_cell(defaults);
        let ndefaults = if unsafe { pyre_object::is_tuple(defaults) } {
            unsafe { pyre_object::w_tuple_len(defaults) }
        } else {
            0
        };
        let first_default = nparams - ndefaults;
        for i in n_pos_copied..nparams {
            if i >= first_default {
                let default_idx = i - first_default;
                if let Some(val) =
                    unsafe { pyre_object::w_tuple_getitem(defaults, default_idx as i64) }
                {
                    filled_args[i] = val;
                }
            }
        }
    }

    // Fill keyword-only defaults from kwdefaults dict.
    if nkwonly > 0 {
        let kwdefaults = unsafe { crate::function_get_kwdefaults(callable) };
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

    // argument.py:302-338 — missing-required after defaults fill.
    let mut missing_positional: Vec<&str> = Vec::new();
    for i in 0..nparams {
        if filled_args[i].is_null() {
            missing_positional.push(code_ref.varnames[i].as_str());
        }
    }
    if !missing_positional.is_empty() {
        let fname = unsafe { crate::function_get_qualname(callable) };
        return Err(crate::PyError::type_error(format_missing_err(
            &fname,
            &missing_positional,
            true,
        )));
    }

    let mut missing_kwonly: Vec<&str> = Vec::new();
    for ki in 0..nkwonly {
        let slot = nparams + ki;
        if filled_args[slot].is_null() {
            missing_kwonly.push(code_ref.varnames[slot].as_str());
        }
    }
    if !missing_kwonly.is_empty() {
        let fname = unsafe { crate::function_get_qualname(callable) };
        return Err(crate::PyError::type_error(format_missing_err(
            &fname,
            &missing_kwonly,
            false,
        )));
    }

    // Append positional overflow AFTER kwonly slots so `pack_varargs` sees
    // `args[total_params..]` as the `*args` source.
    if has_varargs && nargs > nparams {
        filled_args.extend_from_slice(&args[nparams..]);
    }

    Ok(pack_varargs(code_ref, filled_args))
}

/// `argument.py:534-552` ArgErrMissing.getmsg parity.
fn format_missing_err(fname: &str, missing: &[&str], positional: bool) -> String {
    let mut arguments_str = String::new();
    for (i, arg) in missing.iter().enumerate() {
        if i == 0 {
            // no separator
        } else if i == missing.len() - 1 {
            if missing.len() == 2 {
                arguments_str.push_str(" and ");
            } else {
                arguments_str.push_str(", and ");
            }
        } else {
            arguments_str.push_str(", ");
        }
        arguments_str.push('\'');
        arguments_str.push_str(arg);
        arguments_str.push('\'');
    }
    format!(
        "{}() missing {} required {} argument{}: {}",
        fname,
        missing.len(),
        if positional {
            "positional"
        } else {
            "keyword-only"
        },
        if missing.len() != 1 { "s" } else { "" },
        arguments_str
    )
}

fn call_user_function_with_eval(
    frame: &PyFrame,
    callable: PyObjectRef,
    args: &[PyObjectRef],
    eval_fn: EvalFn,
) -> PyResult {
    let w_code = unsafe { crate::getcode(callable) };
    let globals = unsafe { function_get_globals(callable) };
    let w_globals_obj = unsafe { function_get_globals_obj(callable) };
    let closure = unsafe { function_get_closure(callable) };
    let func_code = unsafe {
        crate::w_code_get_ptr(w_code as pyre_object::PyObjectRef) as *const crate::CodeObject
    };
    let code_ref = unsafe { &*func_code };
    let final_args = fill_user_function_args(callable, code_ref, args)?;

    // Generator function: create generator object instead of executing.
    // PyPy: generator.py GeneratorIterator.__init__ wraps PyFrame.
    // RustPython compiler uses CodeFlags::GENERATOR instead of RETURN_GENERATOR opcode.
    if crate::pyframe::code_flags_make_generator(code_ref.flags) {
        let gen_frame =
            crate::pyframe::FrameBox::new(PyFrame::try_new_for_call_with_closure_and_globals_obj(
                w_code,
                &final_args,
                globals,
                w_globals_obj,
                frame.execution_context,
                closure,
            )?);
        return gen_frame.into_generator();
    }

    let mut func_frame =
        crate::pyframe::FrameBox::new(PyFrame::try_new_for_call_with_closure_and_globals_obj(
            w_code,
            &final_args,
            globals,
            w_globals_obj,
            frame.execution_context,
            closure,
        )?);
    func_frame.fix_array_ptrs();
    let _caller_locals_root = FrameLocalsRoot::new(frame);
    let _callee_locals_root = FrameLocalsRoot::new_mut(&mut func_frame);
    eval_fn(&mut func_frame)
}

/// Call a user function with pre-resolved args (scope already packed by
/// resolve_kwargs). Skips defaults-fill and pack_varargs — the caller
/// (call_kw) already produced the final scope via resolve_kwargs which
/// mirrors PyPy's Arguments.parse_into_scope.
pub fn call_user_function_resolved(
    frame: &PyFrame,
    callable: PyObjectRef,
    args: &[PyObjectRef],
) -> PyResult {
    let _depth_guard = increment_call_depth();

    let w_code = unsafe { crate::getcode(callable) };
    let globals = unsafe { function_get_globals(callable) };
    let w_globals_obj = unsafe { function_get_globals_obj(callable) };
    let closure = unsafe { function_get_closure(callable) };
    let func_code = unsafe {
        crate::w_code_get_ptr(w_code as pyre_object::PyObjectRef) as *const crate::CodeObject
    };
    let code_ref = unsafe { &*func_code };

    // Generator function
    if crate::pyframe::code_flags_make_generator(code_ref.flags) {
        let gen_frame =
            crate::pyframe::FrameBox::new(PyFrame::try_new_for_call_with_closure_and_globals_obj(
                w_code,
                args,
                globals,
                w_globals_obj,
                frame.execution_context,
                closure,
            )?);
        return gen_frame.into_generator();
    }

    let eval_fn = get_eval_fn();

    let mut func_frame =
        crate::pyframe::FrameBox::new(PyFrame::try_new_for_call_with_closure_and_globals_obj(
            w_code,
            args,
            globals,
            w_globals_obj,
            frame.execution_context,
            closure,
        )?);
    func_frame.fix_array_ptrs();
    let _caller_locals_root = FrameLocalsRoot::new(frame);
    let _callee_locals_root = FrameLocalsRoot::new_mut(&mut func_frame);
    eval_fn(&mut func_frame)
}

/// Invoke a builtin's function pointer for a positional-only call,
/// packing the `*args` tuple / `**kwargs` dict tail when the builtin's
/// `Signature` declares one.  A variadic builtin is registered with
/// `HOPELESS` arity (see `make_builtin_function_with_arity_and_maybe_sig`),
/// so even a no-keyword call must materialize the (possibly empty) tail
/// slots the `#[pyre_function]` wrapper reads.  Non-variadic builtins keep
/// the raw `func(args)` fast path.
fn call_builtin_code_positional(code: PyObjectRef, args: &[PyObjectRef]) -> PyResult {
    let func = unsafe { builtin_code_get(code) };
    if let Some(sig) = unsafe { crate::builtin_code_get_signature(code) } {
        if sig.has_vararg() || sig.has_kwarg() || sig.num_kwonlyargnames() > 0 {
            let fname = unsafe { crate::builtin_code_name(code) };
            let bound = bind_kwargs_to_signature(sig, fname, args, &[])?;
            return func(&bound);
        }
    }
    func(args)
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

    let frame_ptr = frame as *mut PyFrame;
    dispatch_callable(
        callable,
        |callable| {
            // baseobjspace.py:1243 — `if frame.get_is_being_profiled() and
            // is_builtin_code(w_func): ... return self.call_args_and_c_profile(...)`
            // The `is_builtin_code(w_func)` check is structurally implicit
            // here: dispatch_callable already routed via the builtin arm
            // (runtime_ops.rs:275 `if is_builtin_code(code) { on_builtin }`),
            // so reaching this closure means the callable is a builtin.
            // The remaining condition is the per-frame profile flag, set
            // by `ec.call_trace` (executioncontext.py:150) on frame entry
            // and cleared by `_c_call_return_trace` when profilefunc was
            // turned off (executioncontext.py:122-123).
            let profile_active = unsafe { (*frame_ptr).get_is_being_profiled() };
            if profile_active {
                let w_res = crate::baseobjspace::call_args_and_c_profile(
                    unsafe { &mut *frame_ptr },
                    callable,
                    args,
                );
                if w_res == pyre_object::PY_NULL {
                    return Err(take_call_error()
                        .unwrap_or_else(|| crate::PyError::value_error("call failed")));
                }
                return Ok(w_res);
            }
            let code = unsafe { crate::getcode(callable) };
            call_builtin_code_positional(code as pyre_object::PyObjectRef, args)
        },
        |callable| call_user_function(frame, callable, args),
    )
}

pub fn call_user_function(
    frame: &PyFrame,
    callable: PyObjectRef,
    args: &[PyObjectRef],
) -> PyResult {
    let _depth_guard = increment_call_depth();
    let eval_fn = get_eval_fn();
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
///
/// Mirrors `call_user_function_with_eval`'s arg-fill + generator dispatch
/// so MIFrame Box tracking sees the same result as a concrete-frame call
/// for callees with positional defaults, kw-only defaults, varargs, or
/// the GENERATOR/COROUTINE flags.  The caller-side `FrameLocalsRoot` is
/// skipped because no caller `PyFrame` is available; the callee root is
/// still installed so its locals stay reachable during eval.
pub fn call_user_function_plain_with_ctx(
    execution_context: *const crate::PyExecutionContext,
    callable: PyObjectRef,
    args: &[PyObjectRef],
) -> PyResult {
    let w_code = unsafe { crate::getcode(callable) };
    let globals = unsafe { function_get_globals(callable) };
    let w_globals_obj = unsafe { function_get_globals_obj(callable) };
    let closure = unsafe { function_get_closure(callable) };
    let func_code = unsafe {
        crate::w_code_get_ptr(w_code as pyre_object::PyObjectRef) as *const crate::CodeObject
    };
    let code_ref = unsafe { &*func_code };
    let final_args = fill_user_function_args(callable, code_ref, args)?;

    if crate::pyframe::code_flags_make_generator(code_ref.flags) {
        let gen_frame =
            crate::pyframe::FrameBox::new(PyFrame::try_new_for_call_with_closure_and_globals_obj(
                w_code,
                &final_args,
                globals,
                w_globals_obj,
                execution_context,
                closure,
            )?);
        return gen_frame.into_generator();
    }

    let mut func_frame =
        crate::pyframe::FrameBox::new(PyFrame::try_new_for_call_with_closure_and_globals_obj(
            w_code,
            &final_args,
            globals,
            w_globals_obj,
            execution_context,
            closure,
        )?);
    func_frame.fix_array_ptrs();
    let _callee_locals_root = FrameLocalsRoot::new_mut(&mut func_frame);
    func_frame.run()
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
            let code = unsafe { crate::getcode(callable) };
            call_builtin_code_positional(code as pyre_object::PyObjectRef, args)
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
) -> Result<Vec<PyObjectRef>, crate::PyError> {
    if kwarg_names.is_null() {
        return Ok(args.to_vec());
    }
    let nkw = if unsafe { pyre_object::is_tuple(kwarg_names) } {
        unsafe { pyre_object::w_tuple_len(kwarg_names) }
    } else {
        return Ok(args.to_vec());
    };
    if nkw == 0 {
        return Ok(args.to_vec());
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
        // For type objects, resolve kwargs against __init__ first (most
        // common case: user classes accept kwargs in __init__), falling
        // back to __new__ (e.g. immutable types, metaclasses).
        // PyPy: typeobject.py descr_call → Arguments._match_signature
        //   resolves against the winning __init__ or __new__.
        let init_fn = unsafe { crate::baseobjspace::lookup_in_type(callable, "__init__") };
        if let Some(init_fn) = init_fn {
            if unsafe { crate::is_function(init_fn) } {
                (init_fn, 1usize) // __init__(self, ...) → skip self
            } else {
                // __init__ is builtin → try __new__
                let bases_arg = if args.len() >= nkw + 2 {
                    args[1]
                } else {
                    pyre_object::PY_NULL
                };
                let w_winner = calculate_metaclass(callable, bases_arg).unwrap_or(callable);
                if let Some(new_fn) =
                    unsafe { crate::baseobjspace::lookup_in_type(w_winner, "__new__") }
                {
                    if unsafe { crate::is_function(new_fn) } {
                        (new_fn, 1usize)
                    } else {
                        return Ok(args.to_vec());
                    }
                } else {
                    return Ok(args.to_vec());
                }
            }
        } else {
            return Ok(args.to_vec());
        }
    } else {
        return Ok(args.to_vec());
    };

    let code_ptr = unsafe { crate::get_pycode(target_func) };
    let code = unsafe { &*(code_ptr as *const crate::CodeObject) };
    // Total named params = positional + keyword-only
    let total_params = (code.arg_count + code.kwonlyarg_count) as usize;
    // Effective params = params visible to the caller (excludes implicit cls for types)
    let nparams = total_params - skip_cls;
    let n_pos_params = code.arg_count as usize - skip_cls;
    let n_pos = args.len() - nkw; // number of positional args
    let has_varkw = code.flags.contains(crate::CodeFlags::VARKEYWORDS);
    let has_varargs = code.flags.contains(crate::CodeFlags::VARARGS);
    let posonlyarg_count = code.posonlyarg_count as usize;
    let fname = unsafe { crate::function_get_qualname(target_func) };

    // `argument.py:235-236` — too-many positional args with no *vararg.
    // The kwargs path counts only positional args (`n_pos`); kwargs are
    // matched separately via _match_keywords.
    if n_pos > n_pos_params && !has_varargs {
        let ndefaults = {
            let defaults = unsafe { crate::function_get_defaults(target_func) };
            if !defaults.is_null() {
                let defaults = crate::baseobjspace::unwrap_cell(defaults);
                if unsafe { pyre_object::is_tuple(defaults) } {
                    unsafe { pyre_object::w_tuple_len(defaults) }
                } else {
                    0
                }
            } else {
                0
            }
        };
        let takes_str = if ndefaults > 0 {
            format!(
                "from {} to {} positional arguments",
                n_pos_params - ndefaults,
                n_pos_params
            )
        } else {
            format!(
                "{} positional argument{}",
                n_pos_params,
                if n_pos_params != 1 { "s" } else { "" }
            )
        };
        // argument.py:571 ArgErrTooMany.getmsg
        let nkwonly_given = if nkw > 0 {
            let nkwonly = code.kwonlyarg_count as usize;
            (0..nkw)
                .filter(|&ki| {
                    let kw_name = unsafe { pyre_object::w_tuple_getitem(kwarg_names, ki as i64) };
                    if let Some(kw_obj) = kw_name {
                        if !unsafe { pyre_object::is_str(kw_obj) } {
                            return false;
                        }
                        let kw_s = unsafe { pyre_object::w_str_get_value(kw_obj) };
                        (0..nkwonly).any(|j| &*code.varnames[skip_cls + n_pos_params + j] == kw_s)
                    } else {
                        false
                    }
                })
                .count()
        } else {
            0
        };
        let given_str = if nkwonly_given > 0 {
            format!(
                "{} positional argument{} (and {} keyword-only argument{}) were given",
                n_pos,
                if n_pos != 1 { "s" } else { "" },
                nkwonly_given,
                if nkwonly_given != 1 { "s" } else { "" },
            )
        } else {
            format!(
                "{} {} given",
                n_pos,
                if n_pos != 1 { "were" } else { "was" }
            )
        };
        return Err(crate::PyError::type_error(format!(
            "{}() takes {} but {}",
            fname, takes_str, given_str
        )));
    }

    // Start with PY_NULL for all effective params
    let mut result = vec![pyre_object::PY_NULL; nparams];

    // Fill positional args (PyPy: _match_signature step 1 — argument.py:211-220).
    // Bound at `n_pos_params` so excess positionals never spill into kwonly
    // slots; overflow is packed into *args below if `has_varargs`, otherwise
    // already rejected by the too-many check above.
    for i in 0..n_pos.min(n_pos_params) {
        result[i] = args[i];
    }

    // Match keywords to parameter names (PyPy: _match_keywords)
    // varnames[skip_cls..total_params] are the effective param names
    let mut extra_kwargs: Vec<(PyObjectRef, PyObjectRef)> = Vec::new();
    let mut unmatched_kw_names: Vec<String> = Vec::new();
    for ki in 0..nkw {
        let kw_name = unsafe { pyre_object::w_tuple_getitem(kwarg_names, ki as i64) };
        let Some(kw_name_obj) = kw_name else { continue };
        let kw_value = args[n_pos + ki];

        // argument.py:630 — keywords must be strings (check before access)
        if !unsafe { pyre_object::is_str(kw_name_obj) } {
            return Err(crate::PyError::type_error(format!(
                "{}() keywords must be strings",
                fname
            )));
        }
        let kw_str = unsafe { pyre_object::w_str_get_value(kw_name_obj) };
        let mut matched = false;
        for pi in 0..nparams {
            if &*code.varnames[skip_cls + pi] == kw_str {
                // argument.py:474 — positional-only parameter: if has_kwarg,
                // treat as unmatched (absorb into **kwargs); otherwise error.
                if skip_cls + pi < posonlyarg_count {
                    if has_varkw {
                        break; // fall through to !matched → extra_kwargs
                    }
                    return Err(crate::PyError::type_error(format!(
                        "{}() got some positional-only arguments passed as keyword arguments: '{}'",
                        fname, kw_str
                    )));
                }
                // argument.py:410 — duplicate keyword argument
                if !result[pi].is_null() {
                    return Err(crate::PyError::type_error(format!(
                        "{}() got multiple values for argument '{}'",
                        fname, kw_str
                    )));
                }
                result[pi] = kw_value;
                matched = true;
                break;
            }
        }
        if !matched {
            if has_varkw {
                extra_kwargs.push((kw_name_obj, kw_value));
            } else {
                unmatched_kw_names.push(kw_str.to_string());
            }
        }
    }

    // `argument.py:270-271` ArgErrUnknownKwds — unmatched kwargs and no
    // **kwarg to absorb them.
    if !unmatched_kw_names.is_empty() {
        let msg = if unmatched_kw_names.len() == 1 {
            format!(
                "{}() got an unexpected keyword argument '{}'",
                fname, unmatched_kw_names[0]
            )
        } else {
            let joined = unmatched_kw_names
                .iter()
                .map(|s| format!("'{}'", s))
                .collect::<Vec<_>>()
                .join(", ");
            format!(
                "{}() got {} unexpected keyword arguments: {}",
                fname,
                unmatched_kw_names.len(),
                joined
            )
        };
        return Err(crate::PyError::type_error(msg));
    }

    // Fill positional defaults (PyPy: _match_signature defs_w)
    // Defaults cover the LAST N of the positional params (arg_count).
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

    // `argument.py:302-338` — missing-required positional / kwonly after
    // defaults application.  Errors here mirror ArgErrMissing.
    let mut missing_positional: Vec<&str> = Vec::new();
    for pi in 0..n_pos_params {
        if result[pi].is_null() {
            missing_positional.push(code.varnames[skip_cls + pi].as_str());
        }
    }
    if !missing_positional.is_empty() {
        return Err(crate::PyError::type_error(format_missing_err(
            &fname,
            &missing_positional,
            true,
        )));
    }
    let nkwonly = code.kwonlyarg_count as usize;
    let mut missing_kwonly: Vec<&str> = Vec::new();
    for ki in 0..nkwonly {
        let pi = n_pos_params + ki;
        if result[pi].is_null() {
            missing_kwonly.push(code.varnames[skip_cls + pi].as_str());
        }
    }
    if !missing_kwonly.is_empty() {
        return Err(crate::PyError::type_error(format_missing_err(
            &fname,
            &missing_kwonly,
            false,
        )));
    }

    // Pack *args and **kwargs into scope — PyPy _match_signature lines 207-259.
    // This produces the final scope_w that maps directly to frame locals.
    if has_varargs {
        let extra_pos: Vec<PyObjectRef> = if n_pos > n_pos_params {
            args[n_pos_params..n_pos].to_vec()
        } else {
            vec![]
        };
        result.push(pyre_object::w_tuple_new(extra_pos));
    }
    if has_varkw {
        // `dictmultiobject.py:77-80` — `space.newdict(kwargs=True)` selects
        // EmptyKwargsDictStrategy so the first unicode setitem promotes
        // directly to KwargsDictStrategy (parallel `(keys_w, values_w)`
        // shape) instead of stepping through UnicodeDictStrategy.
        let kw_dict = pyre_object::w_dict_new_kwargs();
        for (key, value) in &extra_kwargs {
            unsafe {
                pyre_object::w_dict_store(kw_dict, *key, *value);
            }
        }
        result.push(kw_dict);
    }

    Ok(result)
}

/// Bind keyword arguments to a builtin's declared `Signature`, producing
/// the flat positional slice the `#[pyre_function]` wrapper reads.
///
/// Mirrors `resolve_kwargs` / argument.py `_match_keywords`, but sources
/// parameter names from the `Signature` rather than a `CodeObject`, and
/// leaves missing slots as `PY_NULL` (the wrapper applies its own
/// `#[default]` values).  Excess positionals pack into the `*args` tuple
/// when `varargname` is set; unmatched keywords pack into the `**kwargs`
/// dict when `kwargname` is set, otherwise raise TypeError.
pub(crate) fn bind_kwargs_to_signature(
    sig: &crate::Signature,
    fname: &str,
    pos_args: &[PyObjectRef],
    kwargs: &[(String, PyObjectRef)],
) -> Result<Vec<PyObjectRef>, crate::PyError> {
    let nparams = sig.argnames.len();
    let n_pos_params = sig.num_argnames(); // positional params (excludes kwonly tail)
    let posonly = sig.posonlyargcount;
    let has_varargs = sig.varargname.is_some();
    let has_varkw = sig.kwargname.is_some();
    let n_pos = pos_args.len();

    // argument.py:235-236 — too many positionals and no `*args` to absorb.
    if n_pos > n_pos_params && !has_varargs {
        return Err(crate::PyError::type_error(format!(
            "{}() takes {} positional argument{} but {} {} given",
            fname,
            n_pos_params,
            if n_pos_params != 1 { "s" } else { "" },
            n_pos,
            if n_pos != 1 { "were" } else { "was" },
        )));
    }

    let mut result = vec![pyre_object::PY_NULL; nparams];
    for i in 0..n_pos.min(n_pos_params) {
        result[i] = pos_args[i];
    }

    // _match_keywords — match each keyword to a param name by index.
    let mut extra_kwargs: Vec<(PyObjectRef, PyObjectRef)> = Vec::new();
    let mut unmatched_kw_names: Vec<String> = Vec::new();
    for (key, value) in kwargs {
        let mut matched = false;
        for pi in 0..nparams {
            if sig.argnames[pi] == key.as_str() {
                // argument.py:474 — positional-only param passed by keyword:
                // absorb into **kwargs if present, else error.
                if pi < posonly {
                    if has_varkw {
                        break;
                    }
                    return Err(crate::PyError::type_error(format!(
                        "{}() got some positional-only arguments passed as keyword arguments: '{}'",
                        fname, key
                    )));
                }
                if !result[pi].is_null() {
                    return Err(crate::PyError::type_error(format!(
                        "{}() got multiple values for argument '{}'",
                        fname, key
                    )));
                }
                result[pi] = *value;
                matched = true;
                break;
            }
        }
        if !matched {
            if has_varkw {
                extra_kwargs.push((pyre_object::w_str_new(key), *value));
            } else {
                unmatched_kw_names.push(key.clone());
            }
        }
    }

    if !unmatched_kw_names.is_empty() {
        let msg = if unmatched_kw_names.len() == 1 {
            format!(
                "{}() got an unexpected keyword argument '{}'",
                fname, unmatched_kw_names[0]
            )
        } else {
            let joined = unmatched_kw_names
                .iter()
                .map(|s| format!("'{}'", s))
                .collect::<Vec<_>>()
                .join(", ");
            format!(
                "{}() got {} unexpected keyword arguments: {}",
                fname,
                unmatched_kw_names.len(),
                joined
            )
        };
        return Err(crate::PyError::type_error(msg));
    }

    // Pack `*args` / `**kwargs` tails — argument.py _match_signature 207-259.
    if has_varargs {
        let extra_pos: Vec<PyObjectRef> = if n_pos > n_pos_params {
            pos_args[n_pos_params..n_pos].to_vec()
        } else {
            vec![]
        };
        result.push(pyre_object::w_tuple_new(extra_pos));
    }
    if has_varkw {
        let kw_dict = pyre_object::w_dict_new_kwargs();
        for (key, value) in &extra_kwargs {
            unsafe {
                pyre_object::w_dict_store(kw_dict, *key, *value);
            }
        }
        result.push(kw_dict);
    }

    Ok(result)
}

/// Call a user function with positional args + keyword args from a dict.
///
/// PyPy: argument.py Arguments._match_signature with keyword handling.
/// Used by CALL_FUNCTION_KW / CALL_KW and CALL_FUNCTION_EX when kwargs
/// are non-empty.
pub fn call_with_kwargs(
    frame: &mut crate::pyframe::PyFrame,
    callable: PyObjectRef,
    pos_args: &[PyObjectRef],
    kwargs: &[(String, PyObjectRef)],
) -> PyResult {
    let callable = crate::baseobjspace::unwrap_cell(callable);

    // Unwrap bound methods: prepend receiver to pos_args.
    if unsafe { pyre_object::is_method(callable) } {
        let func = unsafe { pyre_object::w_method_get_func(callable) };
        let receiver = unsafe { pyre_object::w_method_get_self(callable) };
        let mut full_args = Vec::with_capacity(1 + pos_args.len());
        if !receiver.is_null() && !unsafe { pyre_object::is_none(receiver) } {
            full_args.push(receiver);
        }
        full_args.extend_from_slice(pos_args);
        return call_with_kwargs(frame, func, &full_args, kwargs);
    }

    if unsafe { crate::is_function(callable) } {
        let code = unsafe { crate::getcode(callable) };
        // For builtins: pack kwargs into a dict as last arg.
        //
        // PRE-EXISTING-ADAPTATION (builtin kwargs ABI). PyPy gives every
        // builtin a real Signature (`gateway.py:740 BuiltinCode`, `:804
        // self.sig = app_sig.signature()`) and `funcrun_obj` (`gateway.py:871`)
        // resolves keywords by name through `args.parse_obj` →
        // `_match_signature` (`argument.py:173`), exactly like a user function;
        // there is no marker dict. Pyre's builtin ABI is a flat
        // `&[PyObjectRef]` slice (`BuiltinCodeFn`), so kwargs are smuggled as a
        // trailing dict tagged with the `__pyre_kw__` sentinel and each
        // kwarg-aware builtin reads it manually (`builtins::split_builtin_kwargs`).
        // CONVERGENCE PATH: port the gateway Signature/unwrap_spec surface for
        // builtins, then route builtin kwargs through `Arguments::_match_signature`
        // into named parameter slots and delete `__pyre_kw__`. Deferred: that is
        // a standalone multi-slice epic (no builtin-Signature machinery exists
        // yet) and the JIT inline-call path consumes the same flat tail
        // (`pyre-jit/src/eval.rs:2319`), so it cannot land in one ≤12-file slice.
        // Keep the marker here, in the one builtin kwargs packing site, so
        // CALL_KW and CALL_FUNCTION_EX have the same shape.
        if unsafe { crate::is_builtin_code(code as pyre_object::PyObjectRef) } {
            // Signature-bearing builtins bind keywords into positional
            // order via their declared Signature instead of receiving a
            // trailing `__pyre_kw__` dict.  A null sig (every builtin
            // today) falls through to the dict-packing path below.
            if let Some(sig) =
                unsafe { crate::builtin_code_get_signature(code as pyre_object::PyObjectRef) }
            {
                let fname = unsafe { crate::builtin_code_name(code as pyre_object::PyObjectRef) };
                let bound = bind_kwargs_to_signature(sig, fname, pos_args, kwargs)?;
                // `bound` is already the final flat slice (positional slots
                // plus packed `*args` / `**kwargs` tail), so invoke the
                // builtin directly — routing back through `call_callable`
                // would re-enter `call_builtin_code_positional` and pack the
                // tail a second time.
                let func = unsafe { crate::builtin_code_get(code as pyre_object::PyObjectRef) };
                return func(&bound);
            }
            let mut full_args = pos_args.to_vec();
            if !kwargs.is_empty() {
                let kwargs_dict = pyre_object::w_dict_new();
                unsafe {
                    pyre_object::w_dict_store(
                        kwargs_dict,
                        pyre_object::w_str_new("__pyre_kw__"),
                        pyre_object::w_bool_from(true),
                    );
                }
                for (key, value) in kwargs {
                    unsafe {
                        pyre_object::w_dict_store(kwargs_dict, pyre_object::w_str_new(key), *value);
                    }
                }
                full_args.push(kwargs_dict);
                // Step 2 of the Arguments port: when this is a profiled
                // builtin call AND kwargs are present, route through
                // `call_args_and_c_profile_args` with a structured
                // `Arguments::with_kw(pos_args, keyword_names_w,
                // keywords_w)`.  Otherwise `call_args_and_c_profile`
                // (reached via `call_callable`'s on_builtin closure)
                // would build `Arguments::positional_only(full_args)`
                // and surface the trailing kwargs dict at index 0,
                // breaking the FunctionWithFixedCode rebinding's
                // firstarg() (`argument.py:164-168` returns `None`
                // when positional count is zero, not the kwargs dict).
                let frame_ptr = frame as *mut PyFrame;
                let profile_active = unsafe { (*frame_ptr).get_is_being_profiled() };
                if profile_active {
                    let keyword_names_w: Vec<pyre_object::PyObjectRef> = kwargs
                        .iter()
                        .map(|(k, _)| pyre_object::w_str_new(k))
                        .collect();
                    let keywords_w: Vec<pyre_object::PyObjectRef> =
                        kwargs.iter().map(|(_, v)| *v).collect();
                    let arguments = crate::argument::Arguments::with_kw(
                        pos_args,
                        &keyword_names_w,
                        &keywords_w,
                    );
                    let w_res = crate::baseobjspace::call_args_and_c_profile_args(
                        unsafe { &mut *frame_ptr },
                        callable,
                        &arguments,
                        &full_args,
                    );
                    if w_res == pyre_object::PY_NULL {
                        return Err(take_call_error()
                            .unwrap_or_else(|| crate::PyError::value_error("call failed")));
                    }
                    return Ok(w_res);
                }
            }
            return call_callable(frame, callable, &full_args);
        }

        // For user functions: resolve kwargs to parameter slots
        {
            let w_code = unsafe { crate::getcode(callable) };
            let code = unsafe {
                &*(crate::w_code_get_ptr(w_code as pyre_object::PyObjectRef)
                    as *const crate::CodeObject)
            };
            let total_params = (code.arg_count + code.kwonlyarg_count) as usize;
            let n_pos_params = code.arg_count as usize;
            let has_varkw = code.flags.contains(crate::CodeFlags::VARKEYWORDS);
            let has_varargs = code.flags.contains(crate::CodeFlags::VARARGS);
            let fname = unsafe { crate::function_get_qualname(callable) };

            // `argument.py:235-236` — too-many positional args with no *vararg.
            if pos_args.len() > n_pos_params && !has_varargs {
                let ndefaults = {
                    let defaults = unsafe { crate::function_get_defaults(callable) };
                    if !defaults.is_null() {
                        let defaults = crate::baseobjspace::unwrap_cell(defaults);
                        if unsafe { pyre_object::is_tuple(defaults) } {
                            unsafe { pyre_object::w_tuple_len(defaults) }
                        } else {
                            0
                        }
                    } else {
                        0
                    }
                };
                let takes_str = if ndefaults > 0 {
                    format!(
                        "from {} to {} positional arguments",
                        n_pos_params - ndefaults,
                        n_pos_params
                    )
                } else {
                    format!(
                        "{} positional argument{}",
                        n_pos_params,
                        if n_pos_params != 1 { "s" } else { "" }
                    )
                };
                let given_str = format!(
                    "{} {}",
                    pos_args.len(),
                    if pos_args.len() != 1 { "were" } else { "was" }
                );
                return Err(crate::PyError::type_error(format!(
                    "{}() takes {} but {} given",
                    fname, takes_str, given_str
                )));
            }

            // Build parameter array
            let mut result = vec![pyre_object::PY_NULL; total_params];
            // Fill positional args — bound at `n_pos_params` so excess
            // positionals don't spill into kwonly slots.
            for i in 0..pos_args.len().min(n_pos_params) {
                result[i] = pos_args[i];
            }
            // Match keywords to parameter names
            let mut extra_kwargs: Vec<(String, PyObjectRef)> = Vec::new();
            let mut unmatched_kw_names: Vec<String> = Vec::new();
            for (key, value) in kwargs {
                let mut matched = false;
                for pi in 0..total_params {
                    if code.varnames[pi] == *key {
                        result[pi] = *value;
                        matched = true;
                        break;
                    }
                }
                if !matched {
                    if has_varkw {
                        extra_kwargs.push((key.clone(), *value));
                    } else {
                        unmatched_kw_names.push(key.clone());
                    }
                }
            }

            // `argument.py:270-271` ArgErrUnknownKwds.
            if !unmatched_kw_names.is_empty() {
                let msg = if unmatched_kw_names.len() == 1 {
                    format!(
                        "{}() got an unexpected keyword argument '{}'",
                        fname, unmatched_kw_names[0]
                    )
                } else {
                    let joined = unmatched_kw_names
                        .iter()
                        .map(|s| format!("'{}'", s))
                        .collect::<Vec<_>>()
                        .join(", ");
                    format!(
                        "{}() got {} unexpected keyword arguments: {}",
                        fname,
                        unmatched_kw_names.len(),
                        joined
                    )
                };
                return Err(crate::PyError::type_error(msg));
            }

            // Fill positional defaults from __defaults__ tuple.
            let defaults = unsafe { crate::function_get_defaults(callable) };
            if !defaults.is_null() {
                let defaults = crate::baseobjspace::unwrap_cell(defaults);
                if unsafe { pyre_object::is_tuple(defaults) } {
                    let ndefaults = unsafe { pyre_object::w_tuple_len(defaults) };
                    let first_default = n_pos_params.saturating_sub(ndefaults);
                    for pi in first_default..n_pos_params {
                        if result[pi].is_null() {
                            let di = pi - first_default;
                            if let Some(v) =
                                unsafe { pyre_object::w_tuple_getitem(defaults, di as i64) }
                            {
                                result[pi] = v;
                            }
                        }
                    }
                }
            }
            // Fill keyword-only defaults from __kwdefaults__ dict.
            // function.py Function._apply_defaults — kw-only args take their
            // defaults from the kwdefaults dict by name lookup.
            let nkwonly = code.kwonlyarg_count as usize;
            if nkwonly > 0 {
                let kwdefaults = unsafe { crate::function_get_kwdefaults(callable) };
                if !kwdefaults.is_null() && unsafe { pyre_object::is_dict(kwdefaults) } {
                    for ki in 0..nkwonly {
                        let slot = n_pos_params + ki;
                        if slot < result.len() && result[slot].is_null() {
                            let param_name = &code.varnames[slot];
                            let key = pyre_object::w_str_new(param_name);
                            if let Some(v) = unsafe { pyre_object::w_dict_lookup(kwdefaults, key) }
                            {
                                result[slot] = v;
                            }
                        }
                    }
                }
            }

            // `argument.py:302-338` — missing-required after defaults fill.
            let mut missing_positional: Vec<&str> = Vec::new();
            for pi in 0..n_pos_params {
                if result[pi].is_null() {
                    missing_positional.push(code.varnames[pi].as_str());
                }
            }
            if !missing_positional.is_empty() {
                return Err(crate::PyError::type_error(format_missing_err(
                    &fname,
                    &missing_positional,
                    true,
                )));
            }
            let mut missing_kwonly: Vec<&str> = Vec::new();
            for ki in 0..nkwonly {
                let slot = n_pos_params + ki;
                if result[slot].is_null() {
                    missing_kwonly.push(code.varnames[slot].as_str());
                }
            }
            if !missing_kwonly.is_empty() {
                return Err(crate::PyError::type_error(format_missing_err(
                    &fname,
                    &missing_kwonly,
                    false,
                )));
            }

            // Pack *args and **kwargs
            let mut final_args = result;
            if has_varargs {
                let extra_pos: Vec<PyObjectRef> = if pos_args.len() > n_pos_params {
                    pos_args[n_pos_params..].to_vec()
                } else {
                    vec![]
                };
                final_args.push(pyre_object::w_tuple_new(extra_pos));
            }
            if has_varkw {
                let kw_dict = pyre_object::w_dict_new();
                for (key, value) in &extra_kwargs {
                    unsafe {
                        pyre_object::w_dict_store(kw_dict, pyre_object::w_str_new(key), *value);
                    }
                }
                final_args.push(kw_dict);
            }

            // Create frame and execute
            let globals = unsafe { function_get_globals(callable) };
            let w_globals_obj = unsafe { function_get_globals_obj(callable) };
            let closure = unsafe { function_get_closure(callable) };
            let mut func_frame = crate::pyframe::FrameBox::new(
                crate::pyframe::PyFrame::try_new_for_call_with_closure_and_globals_obj(
                    w_code,
                    &final_args,
                    globals,
                    w_globals_obj,
                    frame.execution_context,
                    closure,
                )?,
            );
            func_frame.fix_array_ptrs();
            let plain_mode = FORCE_PLAIN_EVAL.with(|c| c.get() > 0);
            let eval_fn = if plain_mode {
                crate::eval::eval_frame_plain
            } else {
                EVAL_OVERRIDE
                    .get()
                    .copied()
                    .unwrap_or(crate::eval::eval_frame_plain)
            };
            return eval_fn(&mut func_frame);
        } // end user function branch
    } // end is_function

    // For type objects: allocate via __new__ then call __init__ with kwargs.
    // PyPy: typeobject.py descr_call → __new__ + __init__
    if unsafe { pyre_object::is_type(callable) } {
        // Types with acceptable_as_base_class=false (bool, NoneType) reject kwargs.
        // PyPy: boolobject.py descr_new uses @unwrap_spec (positional only).
        if !kwargs.is_empty()
            && !unsafe { pyre_object::w_type_get_acceptable_as_base_class(callable) }
        {
            let type_name = unsafe { pyre_object::w_type_get_name(callable) };
            return Err(crate::PyError::type_error(format!(
                "{}() takes no keyword arguments",
                type_name,
            )));
        }
        // Calculate the winning metaclass from bases.
        // type(name, bases, dict, **kw) needs to find the correct metaclass
        // and call its __new__ with the kwargs.
        let w_metaclass = if pos_args.len() >= 2 && unsafe { pyre_object::is_tuple(pos_args[1]) } {
            calculate_metaclass(callable, pos_args[1]).unwrap_or(callable)
        } else {
            callable
        };
        // Step 1: __new__(cls, *args, **kwargs)
        let instance = if let Some(new_fn) =
            unsafe { crate::baseobjspace::lookup_in_type(w_metaclass, "__new__") }
        {
            let mut new_args = Vec::with_capacity(1 + pos_args.len());
            new_args.push(w_metaclass);
            new_args.extend_from_slice(pos_args);
            if unsafe { crate::is_function(new_fn) } && !kwargs.is_empty() {
                call_with_kwargs(frame, new_fn, &new_args, kwargs)?
            } else {
                call_callable(frame, new_fn, &new_args)?
            }
        } else {
            pyre_object::w_instance_new(callable)
        };
        // Step 2: __init__(self, *args, **kwargs) with full kwargs support.
        if let Some(w_insttype) = type_call_init_type(instance, callable) {
            if let Some(init_fn) =
                unsafe { crate::baseobjspace::lookup_in_type(w_insttype, "__init__") }
            {
                let mut init_args = Vec::with_capacity(1 + pos_args.len());
                init_args.push(instance);
                init_args.extend_from_slice(pos_args);
                let init_result = if unsafe { crate::is_function(init_fn) } && !kwargs.is_empty() {
                    call_with_kwargs(frame, init_fn, &init_args, kwargs)?
                } else {
                    call_callable(frame, init_fn, &init_args)?
                };
                check_init_returned_none(init_result)?;
            }
        }
        return Ok(instance);
    }

    // For methods: unwrap and retry
    if unsafe { pyre_object::is_method(callable) } {
        let func = unsafe { pyre_object::w_method_get_func(callable) };
        let w_self = unsafe { pyre_object::w_method_get_self(callable) };
        let mut full_args = Vec::with_capacity(1 + pos_args.len());
        if !w_self.is_null() && unsafe { !pyre_object::is_none(w_self) } {
            full_args.push(w_self);
        }
        full_args.extend_from_slice(pos_args);
        return call_with_kwargs(frame, func, &full_args, kwargs);
    }

    // For instances with __call__: dispatch
    if unsafe { pyre_object::is_instance(callable) } {
        let w_type = unsafe { pyre_object::w_instance_get_type(callable) };
        if let Some(call_fn) = unsafe { crate::baseobjspace::lookup_in_type(w_type, "__call__") } {
            let mut call_args = Vec::with_capacity(1 + pos_args.len());
            call_args.push(callable);
            call_args.extend_from_slice(pos_args);
            return call_with_kwargs(frame, call_fn, &call_args, kwargs);
        }
    }

    // Fallback: call_callable with positional args only
    call_callable(frame, callable, pos_args)
}

pub fn register_build_class() {
    crate::typedef::init_typeobjects();
    install_dict_storage_hooks();
}

/// Wire the storage ↔ W_DictObject sync hooks.  Idempotent — guarded
/// by an internal `Once` so repeated invocations
/// (`register_build_class` at runtime startup, `ExecutionContext::new`
/// defensive registration before module allocation) collapse to a
/// single registration without leaking additional function pointers
/// per call.  PyPy's single `W_DictMultiObject` owns both halves of
/// the dict view; pyre's split storage / W_DictObject layout would
/// otherwise let an early `w_dict_setitem_str` (e.g.
/// `Module.__init__`'s `__name__` write) silently miss the storage if
/// these hooks are not yet registered.
pub fn install_dict_storage_hooks() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        pyre_object::dictmultiobject::register_dict_storage_store_hook(
            |ns_ptr, name, value| unsafe {
                let ns = &mut *(ns_ptr as *mut crate::DictStorage);
                crate::dict_storage_store(ns, name, value);
            },
        );
        pyre_object::dictmultiobject::register_dict_storage_delete_hook(|ns_ptr, name| unsafe {
            let ns = &mut *(ns_ptr as *mut crate::DictStorage);
            ns.remove(name);
        });
        pyre_object::dictmultiobject::register_dict_storage_lookup_hook(|ns_ptr, name| unsafe {
            let ns = &*(ns_ptr as *const crate::DictStorage);
            crate::dict_storage_get(ns, name)
        });
        pyre_object::dictmultiobject::register_dict_storage_items_hook(|ns_ptr| unsafe {
            let ns = &*(ns_ptr as *const crate::DictStorage);
            ns.entries().map(|(k, v)| (k.to_string(), *v)).collect()
        });
    });
}

/// `ObjSpace.call_function(callable, *args)` — direct implementation.
///
/// PyPy: baseobjspace.py `call_function`. Now a direct function call
/// (no callback — interpreter and runtime are in the same crate).
/// PyPy: baseobjspace.py `call_function`
///
/// Dispatches to builtins, user functions, and type objects.
/// Type call uses the same __new__ + __init__ protocol as type_descr_call.
/// Re-export for crate-external callers that need a frame-less call path.
///
/// This wrapper preserves the legacy `PyObjectRef`-returning shape used by
/// most call sites. Errors are stashed in `PENDING_CALL_ERROR`; callers
/// recover them via `take_call_error()` after a `PY_NULL` return.
pub fn call_function_impl_raw(callable: PyObjectRef, args: &[PyObjectRef]) -> PyObjectRef {
    match call_function_impl_result(callable, args) {
        Ok(result) => result,
        Err(e) => {
            if std::env::var("PYRE_DEBUG_CALL").is_ok() {
                eprintln!("[call_function_impl] error: {}", e.message);
            }
            set_call_error(e);
            PY_NULL
        }
    }
}

pub(crate) fn call_function_impl(callable: PyObjectRef, args: &[PyObjectRef]) -> PyObjectRef {
    call_function_impl_raw(callable, args)
}

/// pypy/interpreter/baseobjspace.py call_function — Result-returning entry
/// point that mirrors PyPy's OperationError-raising space.call.
///
/// This is the canonical call path. `call_function_impl_raw` (legacy)
/// wraps it for callers that expect a bare `PyObjectRef` and stash the
/// error in `PENDING_CALL_ERROR` instead.
pub fn call_function_impl_result(
    callable: PyObjectRef,
    args: &[PyObjectRef],
) -> Result<PyObjectRef, PyError> {
    // rpython/rlib/rstack.py:42 stack_check(): every interpreter call
    // boundary checks the native stack synchronously, so deep recursion
    // raises RecursionError instead of letting the OS abort on a
    // guard-page hit. Also drain any JIT-prologue pending overflow.
    crate::stack_check::drain_jit_pending_exception()?;
    crate::stack_check::stack_check()?;

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
            return call_function_impl_result(func, &call_args);
        }
        // All callables are Function objects.
        if crate::is_function(callable) {
            let code = crate::getcode(callable);
            if crate::is_builtin_code(code as pyre_object::PyObjectRef) {
                // Builtin function: direct Rust call. Errors propagate
                // naturally through the Result return type — this is the
                // PyPy/OperationError equivalent. Route through the
                // signature-aware positional path so variadic builtins
                // (*args / **kwargs) get their tail packed; non-variadic
                // builtins fall through to the raw fast path.
                return call_builtin_code_positional(code as pyre_object::PyObjectRef, args);
            }
            // User function: create frame + eval. The bare-PyObjectRef
            // helper stashes any error in `PENDING_CALL_ERROR` and returns
            // PY_NULL; recover it here so it propagates as a real Result.
            clear_call_error();
            let result = call_user_function_with_args(callable, args);
            if result.is_null() {
                if let Some(err) = take_call_error() {
                    return Err(err);
                }
            }
            return Ok(result);
        }
        // Type object → descr_call: __new__ + __init__
        // PyPy: typeobject.py descr_call → lookup __new__, call, then __init__
        if pyre_object::is_type(callable) {
            clear_call_error();
            let result = type_descr_call_impl(callable, args);
            if result.is_null() {
                if let Some(err) = take_call_error() {
                    return Err(err);
                }
            }
            return Ok(result);
        }
        // staticmethod → unwrap and call the wrapped function
        // PyPy: function.py StaticMethod.descr_staticmethod__call__
        if pyre_object::is_staticmethod(callable) {
            let func = pyre_object::w_staticmethod_get_func(callable);
            return call_function_impl_result(func, args);
        }
        // classmethod → unwrap and call the wrapped function
        // PyPy: function.py ClassMethod.descr_classmethod__call__
        if pyre_object::is_classmethod(callable) {
            let func = pyre_object::w_classmethod_get_func(callable);
            return call_function_impl_result(func, args);
        }
        // Instance with __call__ — PyPy: descroperation.py
        if pyre_object::is_instance(callable) {
            let w_type = pyre_object::w_instance_get_type(callable);
            if let Some(call_fn) = crate::baseobjspace::lookup_in_type(w_type, "__call__") {
                let mut call_args = Vec::with_capacity(1 + args.len());
                call_args.push(callable);
                call_args.extend_from_slice(args);
                return call_function_impl_result(call_fn, &call_args);
            }
        }
    }
    Err(PyError::type_error(format!(
        "'{}' object is not callable",
        unsafe { (*(*callable).ob_type).name }
    )))
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

    // Step 2: __init__ — only if __new__ returned an instance of w_type.
    // PyPy checks the Python-level type(instance), so builtin-layout subtypes
    // like set subclasses still run __init__.
    if let Some(w_insttype) = type_call_init_type(instance, w_type) {
        if let Some(init_fn) =
            unsafe { crate::baseobjspace::lookup_in_type(w_insttype, "__init__") }
        {
            let mut init_args = Vec::with_capacity(1 + args.len());
            init_args.push(instance);
            init_args.extend_from_slice(args);
            let res = call_function_impl(init_fn, &init_args);
            if res.is_null() {
                // `__init__` raised — error already stashed; propagate it.
                return PY_NULL;
            }
            if let Err(e) = check_init_returned_none(res) {
                set_call_error(e);
                return PY_NULL;
            }
        }
    }

    instance
}

/// `typeobject.py descr_call` — `__init__` must return None.  A non-null,
/// non-None result raises `TypeError: __init__() should return None, not
/// 'X'`.
///
/// A null `result` means `__init__` already raised.  Callers are
/// responsible for detecting that (via `result.is_null()` or a
/// `?`-propagating call) and forwarding the stashed error themselves;
/// this function returns `Ok(())` for null purely as a defensive guard so
/// it never overwrites the original error with a spurious `TypeError`.
fn check_init_returned_none(result: PyObjectRef) -> Result<(), PyError> {
    if result.is_null() || unsafe { pyre_object::is_none(result) } {
        return Ok(());
    }
    let tname = crate::typedef::r#type(result)
        .map(|t| unsafe { pyre_object::w_type_get_name(t) })
        .unwrap_or("object");
    Err(PyError::type_error(format!(
        "__init__() should return None, not '{tname}'"
    )))
}

fn type_call_init_type(instance: PyObjectRef, w_type: PyObjectRef) -> Option<PyObjectRef> {
    let w_insttype = crate::typedef::r#type(instance)?;
    if std::ptr::eq(w_insttype, w_type) || issubtype_ptr(w_insttype, w_type) {
        Some(w_insttype)
    } else {
        None
    }
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
    let w_code = unsafe { crate::getcode(func) };
    let globals = unsafe { function_get_globals(func) };
    let w_globals_obj = unsafe { function_get_globals_obj(func) };
    let closure = unsafe { function_get_closure(func) };
    let func_code = unsafe {
        crate::w_code_get_ptr(w_code as pyre_object::PyObjectRef) as *const crate::CodeObject
    };
    let exec_ctx = BUILD_CLASS_EXEC_CTX.with(|c| c.get());
    let exec_ctx = if exec_ctx.is_null() {
        LAST_EXEC_CTX.with(|c| c.get())
    } else {
        exec_ctx
    };

    let code_ref = unsafe { &*func_code };
    let final_args = match fill_user_function_args(func, code_ref, args) {
        Ok(v) => v,
        Err(e) => {
            set_call_error(e);
            return PY_NULL;
        }
    };

    // Generator function: wrap frame in generator object
    if crate::pyframe::code_flags_make_generator(code_ref.flags) {
        let gen_frame = crate::pyframe::FrameBox::new(
            match PyFrame::try_new_for_call_with_closure_and_globals_obj(
                w_code,
                &final_args,
                globals,
                w_globals_obj,
                exec_ctx,
                closure,
            ) {
                Ok(f) => f,
                Err(e) => {
                    set_call_error(e);
                    return PY_NULL;
                }
            },
        );
        return match gen_frame.into_generator() {
            Ok(v) => v,
            Err(e) => {
                set_call_error(e);
                PY_NULL
            }
        };
    }

    let mut frame = crate::pyframe::FrameBox::new(
        match PyFrame::try_new_for_call_with_closure_and_globals_obj(
            w_code,
            &final_args,
            globals,
            w_globals_obj,
            exec_ctx,
            closure,
        ) {
            Ok(f) => f,
            Err(e) => {
                set_call_error(e);
                return PY_NULL;
            }
        },
    );
    frame.fix_array_ptrs();
    match frame.execute_frame(None, None) {
        Ok(v) => v,
        Err(e) => {
            set_call_error(e);
            PY_NULL
        }
    }
}

/// Invoke a user function with an already-resolved argument scope,
/// frameless (exec ctx pulled from the thread-local set by
/// __build_class__).  Mirrors [`call_user_function_with_args`] but skips
/// `fill_user_function_args` because `args` is the final frame-local
/// layout produced by [`resolve_kwargs`] — re-matching it would treat the
/// packed `*args` / `**kwargs` slots as extra positionals.
fn call_user_function_resolved_frameless(func: PyObjectRef, args: &[PyObjectRef]) -> PyObjectRef {
    let w_code = unsafe { crate::getcode(func) };
    let globals = unsafe { function_get_globals(func) };
    let w_globals_obj = unsafe { function_get_globals_obj(func) };
    let closure = unsafe { function_get_closure(func) };
    let func_code = unsafe {
        crate::w_code_get_ptr(w_code as pyre_object::PyObjectRef) as *const crate::CodeObject
    };
    let exec_ctx = BUILD_CLASS_EXEC_CTX.with(|c| c.get());
    let exec_ctx = if exec_ctx.is_null() {
        LAST_EXEC_CTX.with(|c| c.get())
    } else {
        exec_ctx
    };
    let code_ref = unsafe { &*func_code };

    let mut frame =
        crate::pyframe::FrameBox::new(PyFrame::new_for_call_with_closure_and_globals_obj(
            w_code,
            args,
            globals,
            w_globals_obj,
            exec_ctx,
            closure,
        ));
    frame.fix_array_ptrs();
    if crate::pyframe::code_flags_make_generator(code_ref.flags) {
        return match frame.into_generator() {
            Ok(v) => v,
            Err(e) => {
                set_call_error(e);
                PY_NULL
            }
        };
    }
    match frame.execute_frame(None, None) {
        Ok(v) => v,
        Err(e) => {
            set_call_error(e);
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
    if unsafe { !pyre_object::is_type(w_metaclass) } {
        // compiling.py:213-219 — `space.call_args(w_meta, Arguments(name,
        // bases, ns, **kwds))`; a non-type metaclass receives the
        // class-definition keywords too.
        let kwds: Vec<(String, PyObjectRef)> = if unsafe { pyre_object::is_dict(kwargs) } {
            unsafe {
                pyre_object::w_dict_items(kwargs)
                    .into_iter()
                    .filter(|(k, _)| pyre_object::is_str(*k))
                    .map(|(k, v)| (pyre_object::w_str_get_value(k).to_string(), v))
                    .collect()
            }
        } else {
            Vec::new()
        };
        let frame = {
            let stored = BUILD_CLASS_EXEC_CTX.with(|c| c.get());
            if stored.is_null() {
                std::ptr::null_mut()
            } else {
                unsafe { (*stored).gettopframe() }
            }
        };
        if !kwds.is_empty() && !frame.is_null() {
            return match call_with_kwargs(
                unsafe { &mut *frame },
                w_metaclass,
                &[name, bases, w_namespace_dict],
                &kwds,
            ) {
                Ok(v) => v,
                Err(e) => {
                    set_call_error(e);
                    PY_NULL
                }
            };
        }
        return crate::call_function(w_metaclass, &[name, bases, w_namespace_dict]);
    }
    let kw_items: Vec<(PyObjectRef, PyObjectRef)> = if unsafe { pyre_object::is_dict(kwargs) } {
        unsafe {
            pyre_object::w_dict_items(kwargs)
                .into_iter()
                .filter(|(k, _)| pyre_object::is_str(*k))
                .collect()
        }
    } else {
        Vec::new()
    };
    // Find the metaclass __new__ method
    let new_fn = unsafe { crate::baseobjspace::lookup_in_type(w_metaclass, "__new__") };

    let instance = if let Some(new_fn) = new_fn {
        // Resolve only against a user-defined __new__ with a real code
        // object; the builtin type.__new__ has none, so fall through.
        let is_user_fn = unsafe { crate::is_function(new_fn) }
            && unsafe {
                !crate::is_builtin_code(crate::getcode(new_fn) as pyre_object::PyObjectRef)
            };
        if is_user_fn {
            // [mcs, name, bases, ns] positional + the class-definition
            // kwargs as keywords; resolve_kwargs matches them against the
            // __new__ signature, filling keyword-only params and packing
            // the remainder into a `**kwds` parameter when present.
            let mut call_args = vec![w_metaclass, name, bases, w_namespace_dict];
            let mut names = Vec::new();
            for (k, v) in &kw_items {
                call_args.push(*v);
                names.push(*k);
            }
            let kwarg_names = pyre_object::w_tuple_new(names);
            match resolve_kwargs(new_fn, &call_args, kwarg_names) {
                Ok(resolved) => call_user_function_resolved_frameless(new_fn, &resolved),
                Err(e) => {
                    set_call_error(e);
                    PY_NULL
                }
            }
        } else {
            // Builtin type.__new__ (the metaclass defines no __new__): pass
            // the class-definition keywords through the `__pyre_kw__`
            // trailing-dict ABI so type.__new__ fires __init_subclass__
            // with them, matching the default-metaclass path.
            let mut new_args = vec![w_metaclass, name, bases, w_namespace_dict];
            if !kw_items.is_empty() {
                new_args.push(pack_pyre_kwargs(&kw_items));
            }
            match call_function_impl_result(new_fn, &new_args) {
                Ok(obj) => obj,
                Err(e) => {
                    set_call_error(e);
                    PY_NULL
                }
            }
        }
    } else {
        pyre_object::w_instance_new(w_metaclass)
    };

    if instance.is_null() {
        return PY_NULL;
    }

    if let Some(w_insttype) = type_call_init_type(instance, w_metaclass)
        && let Some(init_fn) =
            unsafe { crate::baseobjspace::lookup_in_type(w_insttype, "__init__") }
    {
        let is_user_fn = unsafe { crate::is_function(init_fn) }
            && unsafe {
                !crate::is_builtin_code(crate::getcode(init_fn) as pyre_object::PyObjectRef)
            };
        if is_user_fn && !kw_items.is_empty() {
            let mut call_args = vec![instance, name, bases, w_namespace_dict];
            let mut names = Vec::with_capacity(kw_items.len());
            for (k, v) in &kw_items {
                call_args.push(*v);
                names.push(*k);
            }
            let kwarg_names = pyre_object::w_tuple_new(names);
            match resolve_kwargs(init_fn, &call_args, kwarg_names) {
                Ok(resolved) => {
                    let res = call_user_function_resolved_frameless(init_fn, &resolved);
                    if res.is_null() {
                        return PY_NULL;
                    }
                    if let Err(e) = check_init_returned_none(res) {
                        set_call_error(e);
                        return PY_NULL;
                    }
                }
                Err(e) => {
                    set_call_error(e);
                    return PY_NULL;
                }
            }
        } else {
            // Builtin __init__ (e.g. type.__init__) ignores the
            // class-definition keywords during 3-arg class creation, and a
            // user __init__ with no kwargs takes only the positional triple.
            // Either way call positionally; the class-definition kwargs flow
            // to the __init_subclass__ forwarding path instead of being
            // rejected here (typeobject.py descr_call passes __args__ to a
            // builtin __init__, which tolerates them).
            if let Err(e) =
                call_function_impl_result(init_fn, &[instance, name, bases, w_namespace_dict])
            {
                set_call_error(e);
                return PY_NULL;
            }
        }
    }

    instance
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
            // Collect extra kwargs (not metaclass, not __pyre_kw__).
            // `w_dict_items` already dispatches `is_module_dict` so a
            // class statement with `**module_dict` (rare but valid)
            // walks the strategy.
            let extra = pyre_object::w_dict_new();
            unsafe {
                for (k, v) in pyre_object::w_dict_items(last) {
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
                        // baseobjspace.py:76 — metaclass from w_class
                        let w_class = (*base).w_class;
                        let w_type_type = crate::typedef::w_type();
                        if !w_class.is_null() && !std::ptr::eq(w_class, w_type_type) {
                            return Some(w_class);
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
    let w_code = unsafe { crate::getcode(body_fn) };
    let globals = unsafe { function_get_globals(body_fn) };
    let w_globals_obj = unsafe { function_get_globals_obj(body_fn) };
    let closure = unsafe { function_get_closure(body_fn) };
    let func_code = unsafe {
        crate::w_code_get_ptr(w_code as pyre_object::PyObjectRef) as *const crate::CodeObject
    };

    // Call metaclass.__prepare__(name, bases, **kwds) if it exists.
    // PyPy: build_class → metaclass.__prepare__(name, bases, **kwds)
    // Returns the namespace dict to use for the class body.
    let w_namespace = if let Some(w_metaclass) = w_metaclass {
        // compiling.py:184 — look up __prepare__ on the metaclass whether
        // or not it is a type; AttributeError falls back to a fresh
        // namespace.
        match crate::baseobjspace::getattr_str(w_metaclass, "__prepare__") {
            Ok(prepare) => {
                // compiling.py:190-196 — call __prepare__ with the
                // class-definition keywords ('metaclass' already popped by
                // the caller).
                let prepare_kwds: Vec<(String, PyObjectRef)> = match extra_kwargs {
                    Some(kw) if unsafe { pyre_object::is_dict(kw) } => unsafe {
                        pyre_object::w_dict_items(kw)
                            .into_iter()
                            .filter(|(k, _)| pyre_object::is_str(*k))
                            .map(|(k, v)| (pyre_object::w_str_get_value(k).to_string(), v))
                            .collect()
                    },
                    _ => Vec::new(),
                };
                let prepare_frame = {
                    let stored = BUILD_CLASS_EXEC_CTX.with(|c| c.get());
                    if stored.is_null() {
                        std::ptr::null_mut()
                    } else {
                        unsafe { (*stored).gettopframe() }
                    }
                };
                let ns_obj = if !prepare_kwds.is_empty() && !prepare_frame.is_null() {
                    call_with_kwargs(
                        unsafe { &mut *prepare_frame },
                        prepare,
                        &[pyre_object::w_str_new(name), bases],
                        &prepare_kwds,
                    )?
                } else {
                    clear_call_error();
                    let r = crate::call_function(prepare, &[pyre_object::w_str_new(name), bases]);
                    if r.is_null() {
                        // __prepare__ was found but raised during execution —
                        // propagate that exception rather than silently using
                        // a fresh namespace.
                        if let Some(err) = take_call_error() {
                            return Err(err);
                        }
                    }
                    r
                };
                // compiling.py:197-204 — a found __prepare__ must return a
                // mapping; None or any sequence is rejected.
                if !ns_obj.is_null() && !crate::baseobjspace::ismapping_w(ns_obj) {
                    let meta_name = if unsafe { pyre_object::is_type(w_metaclass) } {
                        unsafe { pyre_object::w_type_get_name(w_metaclass).to_string() }
                    } else {
                        "<metaclass>".to_string()
                    };
                    let result_type = unsafe {
                        match crate::typedef::r#type(ns_obj) {
                            Some(tp) => pyre_object::w_type_get_name(tp).to_string(),
                            None => (*(*ns_obj).ob_type).name.to_string(),
                        }
                    };
                    return Err(crate::PyError::type_error(format!(
                        "{meta_name}.__prepare__() must return a mapping, not {result_type}"
                    )));
                }
                if ns_obj.is_null() { None } else { Some(ns_obj) }
            }
            // Only a missing __prepare__ (AttributeError) falls back to a
            // fresh namespace; any other lookup error propagates.
            Err(e) if e.kind == crate::PyErrorKind::AttributeError => None,
            Err(e) => return Err(e),
        }
    } else {
        None
    };

    // Create class namespace — use __prepare__ result or fresh namespace.
    // __prepare__ may return a dict subclass (e.g. EnumDict).
    // dict subclass instances created by w_instance_new store entries in
    // ATTR_TABLE, not in W_DictObject.entries. We handle both cases.
    // `w_dict_items` dispatches through `is_module_dict`, so the
    // rare `__prepare__` returning a W_ModuleDictObject still walks
    // correctly.  Both branches share the same shape; collapse them
    // around the dispatching surface.
    let mut class_ns = Box::new(DictStorage::new());
    if let Some(w_prepared_dict) = w_namespace {
        if unsafe { pyre_object::is_dict(w_prepared_dict) } {
            for (key, value) in unsafe { pyre_object::w_dict_items(w_prepared_dict) } {
                if !value.is_null() && unsafe { pyre_object::is_str(key) } {
                    crate::dict_storage_store(
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
                for (key, value) in unsafe { pyre_object::w_dict_items(backing) } {
                    if !value.is_null() && unsafe { pyre_object::is_str(key) } {
                        crate::dict_storage_store(
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

    // When `__prepare__` returned a custom mapping (a dict subclass such as
    // enum._EnumDict), the class body must execute against it directly so its
    // `__setitem__`/`__getitem__` fire mid-body — e.g. `WHITE = RED | GREEN`
    // reads the values `_EnumDict.__setitem__` resolved on assignment, not the
    // stale `auto()` sentinels.  Route the frame's name binding through it via
    // setdictscope_object; an absent or plain-dict namespace keeps the
    // DictStorage fast path.  Restricted to a dict-subclass instance with a
    // resolvable backing so the post-execution mirror below can recover its
    // entries — an exotic non-dict mapping keeps the legacy path unchanged.
    let mapping_namespace = w_namespace.filter(|&w| unsafe {
        pyre_object::is_instance(w) && !crate::type_methods::resolve_dict_backing(w).is_null()
    });

    let mut frame =
        crate::pyframe::FrameBox::new(PyFrame::try_new_for_call_with_closure_and_globals_obj(
            w_code,
            &[],
            globals,
            w_globals_obj,
            exec_ctx,
            closure,
        )?);
    if let Some(w_ns) = mapping_namespace {
        frame.setdictscope_object(w_ns)?;
    } else {
        frame.setdictscope(class_ns_ptr)?;
    }

    frame.execute_frame(None, None)?;

    // The body wrote through the custom mapping; mirror its final contents
    // into class_ns for the downstream type construction (classcell capture,
    // create_all_slots, __set_name__), which read class_ns.
    if let Some(w_ns) = mapping_namespace {
        let backing = crate::type_methods::resolve_dict_backing(w_ns);
        if !backing.is_null() && unsafe { pyre_object::is_dict(backing) } {
            let class_ns = unsafe { &mut *class_ns_ptr };
            // Rebuild from the final backing so names `del`eted from the
            // mapping during body execution don't survive in class_ns.
            class_ns.clear();
            for (key, value) in unsafe { pyre_object::w_dict_items(backing) } {
                if !value.is_null() && unsafe { pyre_object::is_str(key) } {
                    crate::dict_storage_store(
                        class_ns,
                        unsafe { pyre_object::w_str_get_value(key) },
                        value,
                    );
                }
            }
            unsafe { (*class_ns_ptr).fix_ptr() };
        }
    }

    // type_new_classcell (typeobject.c) — capture the `__classcell__` cell
    // so its content can be set to the new class below.  `__class__` /
    // `__classdict__` are cellvar names that `fast2locals` mirrors into
    // the namespace; CPython never exposes them as namespace keys, so drop
    // them now before a metaclass or the class dict ever sees them.
    // `__classcell__` / `__classdictcell__` ARE real namespace entries the
    // class body stores explicitly: a custom metaclass observes them
    // (type.__new__ receives the full namespace) and `type.__new__`
    // (type_new_classcell) consumes them, so they are dropped per
    // construction path below rather than up front.
    let classcell = {
        let class_ns = unsafe { &mut *class_ns_ptr };
        let cell = class_ns.get("__classcell__").copied();
        class_ns.remove("__class__");
        class_ns.remove("__classdict__");
        cell
    };

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
            // Replay class body stores into the prepared dict so __setitem__
            // side effects (EnumDict tracking) fire — but only on the legacy
            // path where the body wrote into class_ns.  When the body executed
            // directly against the mapping (setdictscope_object above) it
            // already holds every store; replaying would re-run __setitem__
            // and, for _EnumDict, reject the duplicate member keys.
            if mapping_namespace.is_none() {
                let ns = unsafe { &*class_ns_ptr };
                for (k, &v) in ns.entries() {
                    if !v.is_null() {
                        let key = pyre_object::w_str_new(k);
                        // Use setitem to trigger __setitem__ on EnumDict etc.
                        let _ = crate::baseobjspace::setitem(w_prepared_dict, key, v);
                    }
                }
            } else {
                // The body wrote directly into the mapping, so the prepared
                // dict already holds every store — but `fast2locals` may have
                // synced the `__class__` / `__classdict__` cellvars into it.
                // Drop that scaffolding before the metaclass observes it.
                for scaffold in ["__class__", "__classdict__"] {
                    let _ = crate::baseobjspace::delitem(
                        w_prepared_dict,
                        pyre_object::w_str_new(scaffold),
                    );
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
        clear_call_error();
        let result = if let Some(kw) = extra_kwargs {
            // Only use kwargs path if there are actual extra kwargs
            let has_extra = unsafe { pyre_object::is_dict(kw) && pyre_object::w_dict_len(kw) > 0 };
            if has_extra {
                call_metaclass_with_kwargs(w_metaclass, name_obj, bases, w_namespace_dict, kw)
            } else {
                crate::call_function(w_metaclass, &[name_obj, bases, w_namespace_dict])
            }
        } else {
            crate::call_function(w_metaclass, &[name_obj, bases, w_namespace_dict])
        };
        // If the metaclass call raised, propagate the original error rather
        // than silently producing a NULL class object.
        if result.is_null() {
            if let Some(err) = take_call_error() {
                return Err(err);
            }
            return Err(PyError::type_error(format!(
                "metaclass call for {name} returned NULL"
            )));
        }
        // baseobjspace.py:76 getclass() — set w_class to the metaclass
        // so type(C) returns the correct metatype.
        if unsafe { pyre_object::is_type(result) } {
            let mro = unsafe { crate::baseobjspace::compute_default_mro(result) };
            unsafe { pyre_object::w_type_set_mro(result, mro) };
            // typeobject.py:373-377 ready() — register self on each
            // base's `weak_subclasses` after MRO is in place.
            unsafe { pyre_object::typeobject::w_type_ready(result) };
            unsafe {
                if (*result).w_class.is_null() {
                    (*result).w_class = w_metaclass;
                }
            }
        }
        result
    } else {
        // No metaclass observes the namespace on the default path, so
        // consume the explicit class cells here (type_new_classcell leaves
        // them out of the class `__dict__`); the captured `classcell` is
        // bound to the new type below.
        unsafe {
            let class_ns = &mut *class_ns_ptr;
            class_ns.remove("__classcell__");
            class_ns.remove("__classdictcell__");
        }
        let w = pyre_object::w_type_new(name, w_effective_bases, class_ns_ptr as *mut u8);
        // typeobject.py:1143-1204 create_all_slots parity.
        unsafe {
            let ns = &*class_ns_ptr;
            create_all_slots(w, ns, w_effective_bases)?;
        }
        // baseobjspace.py:76 — set w_class to 'type' (default metaclass)
        unsafe {
            (*w).w_class = crate::typedef::w_type();
        }
        let mro = unsafe { crate::baseobjspace::compute_default_mro(w) };
        unsafe { pyre_object::w_type_set_mro(w, mro) };
        // typeobject.py:373-377 ready() — register self on each base's
        // `weak_subclasses` so cross-subclass invalidation in
        // `mutated()` and `__subclasses__()` see this class.
        unsafe { pyre_object::typeobject::w_type_ready(w) };
        // __set_name__ protocol — type_new_set_names
        // Only needed here because w_type_new is a raw Rust call that
        // bypasses the type() builtin (builtins.rs) which already calls
        // __set_name__. The metaclass path above goes through type.__new__()
        // which handles __set_name__ in builtins.rs, so we must NOT call it
        // again there to avoid double invocation.
        if unsafe { pyre_object::is_type(w) } {
            let ns = unsafe { &*class_ns_ptr };
            let entries: Vec<(String, PyObjectRef)> =
                ns.entries().map(|(k, &v)| (k.to_string(), v)).collect();
            for (attr_name, value) in entries {
                if !value.is_null() {
                    if let Ok(set_name) = crate::baseobjspace::getattr_str(value, "__set_name__") {
                        crate::builtins::call_and_check(
                            set_name,
                            &[w, pyre_object::w_str_new(&attr_name)],
                        )?;
                    }
                }
            }
        }
        w
    };

    // `_store_type_in_classcell` runs inside type.__new__, which the
    // metaclass path reaches; the default path builds via raw
    // `w_type_new`, so bind the cell here in its stead.  After a
    // metaclass call, compiling.py:232-246 validates that the cell was
    // propagated to type.__new__ and holds the class that came back:
    //
    //     if isinstance(w_cell, Cell) and isinstance(w_class, W_TypeObject):
    //         if w_cell.empty():
    //             raise oefmt(space.w_RuntimeError,
    //                 "__class__ not set defining %S as %S. "
    //                 "Was __classcell__ propagated to type.__new__?", ...)
    //         else:
    //             w_class_from_cell = w_cell.get()
    //             if not space.is_w(w_class, w_class_from_cell):
    //                 raise oefmt(space.w_TypeError,
    //                     "__class__ set to %S defining %S as %S", ...)
    if let Some(classcell) = classcell {
        if !classcell.is_null() && unsafe { pyre_object::is_cell(classcell) } {
            if w_metaclass.is_some() && unsafe { pyre_object::is_type(w_type) } {
                let cell_value = unsafe { pyre_object::w_cell_get(classcell) };
                if cell_value.is_null() {
                    let class_str = unsafe { crate::py_str(w_type) };
                    return Err(PyError::runtime_error(format!(
                        "__class__ not set defining {name} as {class_str}. \
                         Was __classcell__ propagated to type.__new__?"
                    )));
                }
                if !std::ptr::eq(cell_value, w_type) {
                    let cell_str = unsafe { crate::py_str(cell_value) };
                    let class_str = unsafe { crate::py_str(w_type) };
                    return Err(PyError::type_error(format!(
                        "__class__ set to {cell_str} defining {name} as {class_str}"
                    )));
                }
            } else {
                unsafe { pyre_object::w_cell_set(classcell, w_type) };
            }
        }
    }

    // type_new_init_subclass (typeobject.c) — fire __init_subclass__ on
    // the bases with the keywords that reached `type.__new__`.  Only the
    // default-metaclass path builds the class here via `w_type_new`,
    // bypassing `type.__new__`; for it the class-definition keywords are
    // exactly the keywords `type.__new__` would have seen.  The metaclass
    // path routes through `type.__new__` (builtins.rs `type_descr_new`),
    // which fires __init_subclass__ with the subset of keywords the
    // metaclass actually forwarded — so it must NOT be re-fired here.
    if w_metaclass.is_none() {
        let init_subclass_kwargs: Vec<(PyObjectRef, PyObjectRef)> = match extra_kwargs {
            Some(kw) if unsafe { pyre_object::is_dict(kw) } => unsafe {
                pyre_object::w_dict_items(kw)
                    .into_iter()
                    .filter(|(k, _)| pyre_object::is_str(*k))
                    .collect()
            },
            _ => Vec::new(),
        };
        call_init_subclass_on_bases(w_type, w_effective_bases, &init_subclass_kwargs)?;
    }

    Ok(w_type)
}

/// Pack `(name, value)` keyword pairs into the `__pyre_kw__`-tagged
/// trailing dict that the builtin kwargs ABI (`split_builtin_kwargs`)
/// consumes.  Mirrors the producer in `call_with_kwargs`.
fn pack_pyre_kwargs(kw_items: &[(PyObjectRef, PyObjectRef)]) -> PyObjectRef {
    let kw_dict = pyre_object::w_dict_new();
    unsafe {
        pyre_object::w_dict_store(
            kw_dict,
            pyre_object::w_str_new("__pyre_kw__"),
            pyre_object::w_bool_from(true),
        );
        for (k, v) in kw_items {
            pyre_object::w_dict_store(kw_dict, *k, *v);
        }
    }
    kw_dict
}

/// typeobject.py:1020-1026 `_init_subclass` — after a class is created,
/// call `super(w_type, w_type).__init_subclass__(**kwds)` exactly once.
///
/// ```python
/// def _init_subclass(space, w_type, __args__):
///     w_super = space.getattr(space.builtin, space.newtext("super"))
///     w_func = space.getattr(space.call_function(w_super, w_type, w_type),
///                            space.newtext("__init_subclass__"))
///     args = __args__.replace_arguments([])
///     space.call_args(w_func, args)
/// ```
///
/// The super-proxy getattr performs descriptor binding (a classmethod
/// binds `w_type` as cls; a plain function binds `w_type` as the super
/// obj), and the call forwards only the class-definition keywords.
/// `init_subclass_kwargs` must already exclude the `__pyre_kw__` marker
/// and the `metaclass` key.
///
/// `w_effective_bases` is unused now that resolution follows the MRO, but
/// is kept in the signature so the call sites need not change.
pub(crate) fn call_init_subclass_on_bases(
    w_type: PyObjectRef,
    _w_effective_bases: PyObjectRef,
    init_subclass_kwargs: &[(PyObjectRef, PyObjectRef)],
) -> Result<(), crate::PyError> {
    let w_super = pyre_object::superobject::w_super_new(w_type, w_type);
    let w_func = crate::baseobjspace::getattr_str(w_super, "__init_subclass__")?;
    // `__args__.replace_arguments([])` — keywords only, no positionals.
    let kwds: Vec<(String, PyObjectRef)> = init_subclass_kwargs
        .iter()
        .filter(|(k, _)| unsafe { pyre_object::is_str(*k) })
        .map(|(k, v)| (unsafe { pyre_object::w_str_get_value(*k) }.to_string(), *v))
        .collect();
    let frame = {
        let stored = BUILD_CLASS_EXEC_CTX.with(|c| c.get());
        if stored.is_null() {
            std::ptr::null_mut()
        } else {
            unsafe { (*stored).gettopframe() }
        }
    };
    if !frame.is_null() {
        call_with_kwargs(unsafe { &mut *frame }, w_func, &[], &kwds)?;
    } else if kwds.is_empty() {
        // No live frame to thread through call_with_kwargs (direct
        // embedding entry); the bound method carries the receiver.
        clear_call_error();
        let res = crate::call_function(w_func, &[]);
        if res.is_null() {
            if let Some(err) = take_call_error() {
                return Err(err);
            }
        }
    } else {
        return Err(crate::PyError::type_error(
            "__init_subclass__() takes no keyword arguments",
        ));
    }
    Ok(())
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
    if let Some(w_insttype) = type_call_init_type(instance, w_type) {
        if let Some(init_fn) =
            unsafe { crate::baseobjspace::lookup_in_type(w_insttype, "__init__") }
        {
            let mut init_args = Vec::with_capacity(1 + args.len());
            init_args.push(instance);
            init_args.extend_from_slice(args);
            let init_result = call_callable(frame, init_fn, &init_args)?;
            check_init_returned_none(init_result)?;
        }
    }

    Ok(instance)
}

/// typeobject.py:1157-1176 — unpack __slots__ to slot name strings.
///
/// PyPy:
///   if isinstance(w_slots, (bytes, unicode)):
///       slot_names_w = [w_slots]
///   else:
///       slot_names_w = space.unpackiterable(w_slots)
///   for w_slot_name in slot_names_w:
///       slot_name = space.text_w(w_slot_name)
fn collect_slot_names(w_slots: pyre_object::PyObjectRef) -> Result<Vec<String>, crate::PyError> {
    unsafe {
        // typeobject.py:1158-1162: str → single-element list, else unpackiterable
        let slot_names_w = if pyre_object::is_str(w_slots) {
            vec![w_slots]
        } else {
            crate::baseobjspace::unpackiterable(w_slots, -1)?
        };
        let mut names = Vec::new();
        for w_slot_name in slot_names_w {
            if !pyre_object::is_str(w_slot_name) {
                return Err(crate::PyError::type_error(
                    "__slots__ items must be strings, not type".to_string(),
                ));
            }
            let slot_name = pyre_object::w_str_get_value(w_slot_name).to_string();
            // typeobject.py:1208-1209 valid_slot_name
            if !valid_slot_name(&slot_name) {
                return Err(crate::PyError::type_error(
                    "__slots__ must be identifiers".to_string(),
                ));
            }
            names.push(slot_name);
        }
        Ok(names)
    }
}

/// typeobject.py:1234-1240 valid_slot_name:
///   if len(slot_name) == 0 or slot_name[0].isdigit(): return False
///   for c in slot_name: if not c.isalnum() and c != '_': return False
///   return True
fn valid_slot_name(name: &str) -> bool {
    if name.is_empty() {
        return false;
    }
    let mut chars = name.chars();
    let first = chars.next().unwrap();
    if first.is_ascii_digit() {
        return false;
    }
    if !first.is_alphanumeric() && first != '_' {
        return false;
    }
    for c in chars {
        if !c.is_alphanumeric() && c != '_' {
            return false;
        }
    }
    true
}

/// astcompiler/misc.py:78-92 mangle(name, klass):
///   if not name.startswith('__'): return name
///   if name.endswith('__') or '.' in name: return name
///   strip leading underscores from klass
///   return "_%s%s" % (klass[i:], name)
fn mangle(name: &str, klass: &str) -> String {
    if !name.starts_with("__") {
        return name.to_string();
    }
    if name.ends_with("__") || name.contains('.') {
        return name.to_string();
    }
    let stripped = klass.trim_start_matches('_');
    if stripped.is_empty() {
        return name.to_string();
    }
    format!("_{stripped}{name}")
}

/// typeobject.py:1131-1140 copy_flags_from_bases:
///   w_self.hasdict |= w_base.hasdict
///   w_self.weakrefable |= w_base.weakrefable
unsafe fn copy_flags_from_bases(
    w_type: pyre_object::PyObjectRef,
    w_bases: pyre_object::PyObjectRef,
) {
    unsafe {
        if w_bases.is_null() || !pyre_object::is_tuple(w_bases) {
            return;
        }
        let len = pyre_object::w_tuple_len(w_bases);
        for i in 0..len {
            if let Some(base) = pyre_object::w_tuple_getitem(w_bases, i as i64) {
                if pyre_object::is_type(base) {
                    if pyre_object::w_type_get_hasdict(base) {
                        pyre_object::w_type_set_hasdict(w_type, true);
                    }
                    if pyre_object::w_type_get_weakrefable(base) {
                        pyre_object::w_type_set_weakrefable(w_type, true);
                    }
                }
            }
        }
    }
}

/// typeobject.py:1143-1204 create_all_slots.
///
/// Returns `Err` for invalid __slots__ (TypeError), matching PyPy.
///
/// # Safety
/// `w_type` must be a valid W_TypeObject pointer.
pub unsafe fn create_all_slots(
    w_type: pyre_object::PyObjectRef,
    ns: &crate::DictStorage,
    w_bases: pyre_object::PyObjectRef,
) -> Result<(), crate::PyError> {
    unsafe {
        use pyre_object::typeobject::{Layout, leak_layout};

        // typeobject.py:1245: w_bestbase = check_and_find_best_base(space, bases_w)
        let w_bestbase = check_and_find_best_base(w_bases)?;

        // typeobject.py:1507-1508: inherit flag_map_or_seq from bases
        pyre_object::typeobject::inherit_flag_map_or_seq(w_type, w_bases);

        // typeobject.py:1510: copy_flags_from_bases — inherit hasdict/weakrefable
        copy_flags_from_bases(w_type, w_bases);

        // typeobject.py:1146: base_layout = w_bestbase.layout
        let base_layout = if w_bestbase.is_null() {
            std::ptr::null()
        } else {
            pyre_object::w_type_get_layout_ptr(w_bestbase)
        };
        let base_nslots = if base_layout.is_null() {
            0
        } else {
            (*base_layout).nslots
        };

        // typeobject.py:1150-1204 create_all_slots
        let mut newslotnames = Vec::new();
        let (mut wantdict, mut wantweakref);
        if let Some(&w_slots) = ns.get("__slots__") {
            // typeobject.py:1154-1176: has __slots__
            wantdict = false;
            wantweakref = false;
            let all_names = collect_slot_names(w_slots)?;
            for slot_name in &all_names {
                match slot_name.as_str() {
                    // typeobject.py:1165-1169: __dict__ slot
                    "__dict__" => {
                        if wantdict || pyre_object::w_type_get_hasdict(w_type) {
                            return Err(crate::PyError::type_error(
                                "__dict__ slot disallowed: we already got one".to_string(),
                            ));
                        }
                        wantdict = true;
                    }
                    // typeobject.py:1170-1174: __weakref__ slot
                    "__weakref__" => {
                        if wantweakref || pyre_object::w_type_get_weakrefable(w_type) {
                            return Err(crate::PyError::type_error(
                                "__weakref__ slot disallowed: we already got one".to_string(),
                            ));
                        }
                        wantweakref = true;
                    }
                    // typeobject.py:1175-1176: regular slot name
                    _ => newslotnames.push(slot_name.clone()),
                }
            }
            // typeobject.py:1178: string_sort(newslotnames)
            newslotnames.sort();

            // typeobject.py:1183-1189: create_slot loop
            let type_ns = pyre_object::w_type_get_dict_ptr(w_type) as *mut crate::DictStorage;
            let type_name = pyre_object::w_type_get_name(w_type);
            let mut slot_index = base_nslots;
            let mut i = 0;
            while i < newslotnames.len() {
                // typeobject.py:1208-1209: valid_slot_name check
                if !valid_slot_name(&newslotnames[i]) {
                    return Err(crate::PyError::type_error(
                        "__slots__ must be identifiers".to_string(),
                    ));
                }
                // typeobject.py:1211: slot_name = mangle(slot_name, w_self.name)
                let mangled = mangle(&newslotnames[i], type_name);
                if !type_ns.is_null() && (*type_ns).get(mangled.as_str()).is_some() {
                    // typeobject.py:1219-1220: name conflict → skip this slot
                    newslotnames.remove(i);
                } else {
                    // typeobject.py:1216-1217: create_slot
                    newslotnames[i] = mangled.clone();
                    if !type_ns.is_null() {
                        let member = pyre_object::w_member_new(slot_index, mangled.clone(), w_type);
                        (*type_ns).insert(mangled, member);
                    }
                    slot_index += 1;
                    i += 1;
                }
            }
        } else {
            // typeobject.py:1151-1153: no __slots__
            wantdict = true;
            wantweakref = true;
        }

        // typeobject.py:1192-1195: create_dict_slot / create_weakref_slot
        if wantdict {
            create_dict_slot(w_type);
        }
        if wantweakref {
            create_weakref_slot(w_type);
        }

        // typeobject.py:1199-1204: layout computation
        let nslots = base_nslots + newslotnames.len() as u32;
        let typedef = if base_layout.is_null() {
            &pyre_object::pyobject::INSTANCE_TYPE as *const _
        } else {
            (*base_layout).typedef
        };
        let layout = if nslots == base_nslots && !base_layout.is_null() {
            base_layout
        } else {
            leak_layout(Layout {
                typedef,
                nslots,
                newslotnames,
                base_layout,
                acceptable_as_base_class: true,
            })
        };
        pyre_object::w_type_set_layout(w_type, layout);
        Ok(())
    }
}

/// objspace/std/typeobject.py:1222-1226 create_dict_slot.
///
/// ```python
/// def create_dict_slot(w_self):
///     if not w_self.hasdict:
///         w_self.dict_w.setdefault('__dict__',
///             dict_descr.copy_for_type(w_self))
///         w_self.hasdict = True
/// ```
unsafe fn create_dict_slot(w_type: pyre_object::PyObjectRef) {
    unsafe {
        if !pyre_object::w_type_get_hasdict(w_type) {
            let descr =
                crate::typedef::copy_descriptor_for_type(crate::typedef::dict_descr(), w_type);
            let type_ns = pyre_object::w_type_get_dict_ptr(w_type) as *mut crate::DictStorage;
            if !type_ns.is_null() && (*type_ns).get("__dict__").is_none() {
                (*type_ns).insert("__dict__".to_string(), descr);
            }
            pyre_object::w_type_set_hasdict(w_type, true);
        }
    }
}

/// objspace/std/typeobject.py:1228-1232 create_weakref_slot.
///
/// ```python
/// def create_weakref_slot(w_self):
///     if not w_self.weakrefable:
///         w_self.dict_w.setdefault('__weakref__',
///             weakref_descr.copy_for_type(w_self))
///         w_self.weakrefable = True
/// ```
unsafe fn create_weakref_slot(w_type: pyre_object::PyObjectRef) {
    unsafe {
        if !pyre_object::w_type_get_weakrefable(w_type) {
            let descr =
                crate::typedef::copy_descriptor_for_type(crate::typedef::weakref_descr(), w_type);
            let type_ns = pyre_object::w_type_get_dict_ptr(w_type) as *mut crate::DictStorage;
            if !type_ns.is_null() && (*type_ns).get("__weakref__").is_none() {
                (*type_ns).insert("__weakref__".to_string(), descr);
            }
            pyre_object::w_type_set_weakrefable(w_type, true);
        }
    }
}

/// typeobject.py:1089-1105 find_best_base.
unsafe fn find_best_base(w_bases: pyre_object::PyObjectRef) -> pyre_object::PyObjectRef {
    unsafe {
        if w_bases.is_null() || !pyre_object::is_tuple(w_bases) {
            return std::ptr::null_mut();
        }
        let len = pyre_object::w_tuple_len(w_bases);
        let mut w_bestbase: pyre_object::PyObjectRef = std::ptr::null_mut();
        for i in 0..len {
            if let Some(w_candidate) = pyre_object::w_tuple_getitem(w_bases, i as i64) {
                if !pyre_object::is_type(w_candidate) {
                    continue;
                }
                if w_bestbase.is_null() {
                    w_bestbase = w_candidate;
                    continue;
                }
                let cand_layout = pyre_object::w_type_get_layout_ptr(w_candidate);
                let best_layout = pyre_object::w_type_get_layout_ptr(w_bestbase);
                if cand_layout != best_layout
                    && !cand_layout.is_null()
                    && (*cand_layout).issublayout(best_layout)
                {
                    w_bestbase = w_candidate;
                }
            }
        }
        w_bestbase
    }
}

/// typeobject.py:1107-1129 check_and_find_best_base:
///   w_bestbase = find_best_base(bases_w)
///   if w_bestbase is None: raise TypeError
///   if not w_bestbase.layout.typedef.acceptable_as_base_class: raise TypeError
///   for w_base in bases_w: check layout conflicts
unsafe fn check_and_find_best_base(
    w_bases: pyre_object::PyObjectRef,
) -> Result<pyre_object::PyObjectRef, crate::PyError> {
    unsafe {
        let w_bestbase = find_best_base(w_bases);
        // typeobject.py:1113-1115
        if w_bestbase.is_null() {
            return Err(crate::PyError::type_error(
                "a new-style class can't have only classic bases".to_string(),
            ));
        }
        // typeobject.py:1116-1118: acceptable_as_base_class check.
        // typedef.py:43: acceptable = '__new__' in rawdict.
        // bool and NoneType are not acceptable in Python 3.
        if !is_acceptable_base_class(w_bestbase) {
            return Err(crate::PyError::type_error(format!(
                "type '{}' is not an acceptable base class",
                pyre_object::w_type_get_name(w_bestbase),
            )));
        }
        // typeobject.py:1122-1128: check layout conflicts
        let best_layout = pyre_object::w_type_get_layout_ptr(w_bestbase);
        if !best_layout.is_null() && !w_bases.is_null() && pyre_object::is_tuple(w_bases) {
            let len = pyre_object::w_tuple_len(w_bases);
            for i in 0..len {
                if let Some(w_base) = pyre_object::w_tuple_getitem(w_bases, i as i64) {
                    if !pyre_object::is_type(w_base) {
                        continue;
                    }
                    let layout = pyre_object::w_type_get_layout_ptr(w_base);
                    if !layout.is_null() && !(*best_layout).issublayout(layout) {
                        return Err(crate::PyError::type_error(
                            "instance layout conflicts in multiple inheritance".to_string(),
                        ));
                    }
                }
            }
        }
        Ok(w_bestbase)
    }
}

/// typedef.py:43 `acceptable_as_base_class = '__new__' in rawdict`.
/// typeobject.py:1116 checks this flag on the bestbase.
unsafe fn is_acceptable_base_class(w_type: pyre_object::PyObjectRef) -> bool {
    unsafe { pyre_object::w_type_get_acceptable_as_base_class(w_type) }
}
