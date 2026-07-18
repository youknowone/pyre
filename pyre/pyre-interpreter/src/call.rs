//! Function call dispatch — pure interpreter, no JIT dependencies.
//!
//! JIT-specific call infrastructure (force/bridge callbacks, callee frame
//! creation helpers, frame pool) lives in pyre-jit/src/call_jit.rs.

use std::cell::{Cell, RefCell};
use std::sync::OnceLock;

use rustpython_wtf8::Wtf8Buf;

use crate::{
    PyError, PyResult, builtin_code_get, dispatch_callable, function_get_closure,
    function_get_globals_obj,
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
///
/// `dont_look_inside`: the `PENDING_CALL_ERROR` thread-local `.with`
/// read has no extractable graph, so the call stays a residual (the
/// fnaddr is registered in `jit_trace_fnaddrs`).
#[majit_macros::dont_look_inside]
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
#[majit_macros::dont_look_inside]
pub fn take_call_error() -> Option<PyError> {
    PENDING_CALL_ERROR.with(|slot| slot.borrow_mut().take())
}

/// Clear any pending stashed error without consuming it.
#[majit_macros::dont_look_inside]
pub fn clear_call_error() {
    PENDING_CALL_ERROR.with(|slot| {
        slot.borrow_mut().take();
    });
}

/// Root the deferred call error stashed in `PENDING_CALL_ERROR`. Its `PyError`
/// holds up to three GC-managed references — the cached exception object and
/// the lazy NameError/AttributeError name/obj context — none reached by the
/// precise collector through the raw `RefCell`. Forward each non-null slot in
/// place; the context slots may hold movable str/obj, so the store-back the
/// visitor performs matters. Never materialise the lazy-null `exc_object`.
/// The `PyErrorKind` enum carries no object payload (all unit variants), so
/// these three fields are the complete set of GC refs a `PyError` holds.
pub fn walk_pending_call_error(visitor: &mut dyn FnMut(&mut majit_ir::GcRef)) {
    PENDING_CALL_ERROR.with(|slot| {
        // SAFETY: `as_ptr` yields the `Option<PyError>` interior; this closure
        // holds the only reference for its duration and does not re-borrow the
        // cell, so no borrow-flag conflict with a walker-triggered path.
        let opt = unsafe { &mut *slot.as_ptr() };
        if let Some(err) = opt.as_mut() {
            err.walk_gc_refs(visitor);
        }
    });
}

/// Cold debug flag probe for `call_function_impl_raw`. This is a
/// `dont_look_inside` scalar wrapper so the two-phase rtyper sees a plain
/// bool residual instead of `std::env::var`'s `Result<String, VarError>` ABI.
#[majit_macros::dont_look_inside]
pub fn pyre_debug_call_enabled() -> bool {
    // The debug knob reads the real process env; under sandbox host access must
    // go through the seam, so report disabled rather than touch the host env.
    #[cfg(not(feature = "sandbox"))]
    {
        std::env::var("PYRE_DEBUG_CALL").is_ok()
    }
    #[cfg(feature = "sandbox")]
    {
        false
    }
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

    /// Monotonic count of Python frame eval-loop entries — bumped once per
    /// `eval_loop` / `eval_loop_jit` entry (every user-level bytecode frame
    /// that begins running), NEVER decremented.  Unlike [`CALL_DEPTH`] (net
    /// zero after a balanced call returns), this is a cumulative odometer, so
    /// a snapshot taken before a residual call and re-read after it reveals
    /// whether ANY user Python frame ran during the call regardless of how
    /// deeply it nested or whether it returned.  Consumed by the FBW FOR_ITER
    /// Option-C double-apply guard (#57): a value-returning residual whose
    /// concrete execution entered a user frame (a side-effecting `@property`
    /// getter, a user `__add__` / `__iter__` / `__str__`, an imported
    /// module's top level) committed a body effect the Void/helper-tag write
    /// discriminator cannot see.
    static FRAME_ENTRY_COUNT: Cell<u64> = const { Cell::new(0) };
}

/// Get current call depth. Used by pyre-jit for JIT_CALL_DEPTH parity.
#[inline(always)]
pub fn call_depth() -> u32 {
    CALL_DEPTH.with(|d| d.get())
}

/// Snapshot of the monotonic Python frame eval-loop entry odometer
/// ([`FRAME_ENTRY_COUNT`]).  A change between two reads means user-level
/// bytecode ran in between.
#[inline(always)]
pub fn frame_entry_count() -> u64 {
    FRAME_ENTRY_COUNT.with(|c| c.get())
}

/// Bump the monotonic Python frame eval-loop entry odometer
/// ([`FRAME_ENTRY_COUNT`]).  Called once at every `eval_loop` /
/// `eval_loop_jit` entry, i.e. each time a user Python frame begins
/// executing bytecode.
///
/// Touches the runtime-mutable `FRAME_ENTRY_COUNT` thread-local, not a
/// build-time constant, so the JIT residualizes the call instead of tracing
/// into it (`@dont_look_inside`, the `note_alloc` sibling). A `()` return has
/// no discriminant to erase and it cannot raise.
#[majit_macros::dont_look_inside]
pub fn bump_frame_entry_count() {
    FRAME_ENTRY_COUNT.with(|c| c.set(c.get().wrapping_add(1)));
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
///
/// `dont_look_inside`: the `LAST_EXEC_CTX` thread-local `.with` read has no
/// extractable graph (front::mir const-folds the `ThreadLocal` global to
/// None), so the call stays a residual read via the registered fnaddr
/// (`@dont_look_inside`, `rlib/jit.py:139`), the `take_call_error` twin.
#[majit_macros::dont_look_inside]
pub fn take_last_exec_ctx() -> *const crate::PyExecutionContext {
    LAST_EXEC_CTX.with(|c| c.get())
}

pub(crate) fn capture_last_exec_ctx_cell() -> *const () {
    LAST_EXEC_CTX.with(|cell| cell as *const _ as *const ())
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

/// `argument.py:620-626` ArgErrUnknownKwds.getmsg parity.
fn format_unknown_kwds_err(fname: &str, unmatched: &[Wtf8Buf]) -> String {
    if unmatched.len() == 1 {
        format!(
            "{}() got an unexpected keyword argument '{}'",
            fname, unmatched[0]
        )
    } else {
        format!(
            "{}() got {} unexpected keyword arguments",
            fname,
            unmatched.len()
        )
    }
}

fn call_user_function_with_eval(
    frame: &PyFrame,
    callable: PyObjectRef,
    args: &[PyObjectRef],
    eval_fn: EvalFn,
) -> PyResult {
    let w_code = unsafe { crate::getcode(callable) };
    let w_globals = unsafe { function_get_globals_obj(callable) };
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
                w_globals,
                frame.execution_context,
                closure,
                crate::pyframe::FrameLocalsArrayAllocation::OldGenGc,
            )?);
        return gen_frame.into_generator();
    }

    let mut func_frame =
        crate::pyframe::FrameBox::new(PyFrame::try_new_for_call_with_closure_and_globals_obj(
            w_code,
            &final_args,
            w_globals,
            frame.execution_context,
            closure,
            crate::pyframe::FrameLocalsArrayAllocation::OldGenGc,
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
    let w_globals = unsafe { function_get_globals_obj(callable) };
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
                w_globals,
                frame.execution_context,
                closure,
                crate::pyframe::FrameLocalsArrayAllocation::OldGenGc,
            )?);
        return gen_frame.into_generator();
    }

    let eval_fn = get_eval_fn();

    let mut func_frame =
        crate::pyframe::FrameBox::new(PyFrame::try_new_for_call_with_closure_and_globals_obj(
            w_code,
            args,
            w_globals,
            frame.execution_context,
            closure,
            crate::pyframe::FrameLocalsArrayAllocation::OldGenGc,
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
/// GenericAlias.__call__ (`_pypy_generic_alias.py:43-46`) — after calling
/// `__origin__`, set `result.__orig_class__ = self`.  This is wrapped in
/// `try: ... except (AttributeError, TypeError): pass`, so only those two
/// errors are swallowed; anything else propagates.
fn set_orig_class(result: PyObjectRef, alias: PyObjectRef) -> Result<(), crate::PyError> {
    match crate::baseobjspace::setattr_str(result, "__orig_class__", alias) {
        Ok(_) => Ok(()),
        Err(e)
            if e.kind == crate::error::PyErrorKind::AttributeError
                || e.kind == crate::error::PyErrorKind::TypeError =>
        {
            Ok(())
        }
        Err(e) => Err(e),
    }
}

// `dont_look_inside`: a builtin is invoked through a runtime `BuiltinCodeFn`
// value (`func(args)`), a call through a fn-pointer the tracer has no lowering
// for (only static `CallPath`s lower). The builtin body is the residual
// boundary — the JIT residualizes the whole dispatch (signature-aware kwarg
// packing + the C-level call) instead of tracing into it, mirroring
// `cpu.bh_call_*`. This also keeps `builtin_code_get_signature`'s raw-ptr
// `as_ref` read out of any traced graph.
#[majit_macros::dont_look_inside]
fn call_builtin_code_positional(code: PyObjectRef, args: &[PyObjectRef]) -> PyResult {
    // `gateway.py:824 BuiltinCode.funcrun` is translated with both its code
    // object and `Arguments.arguments_w` live across gateway dispatch.  A
    // collection between the outer `space.call_function` reload and this
    // indirect Rust function-pointer call updates the outer shadow slots but
    // not the copied native slice, so mirror the gateway's own root frame and
    // reload immediately before invoking the builtin.
    let _roots = pyre_object::gc_roots::push_roots();
    let root_base = pyre_object::gc_roots::shadow_stack_len();
    pyre_object::gc_roots::pin_root(code);
    for &arg in args {
        pyre_object::gc_roots::pin_root(arg);
    }
    let current_code = || pyre_object::gc_roots::shadow_stack_get(root_base);
    let current_args = || {
        (0..args.len())
            .map(|i| pyre_object::gc_roots::shadow_stack_get(root_base + 1 + i))
            .collect::<Vec<_>>()
    };

    let func = unsafe { builtin_code_get(current_code()) };
    if let Some(sig) = unsafe { crate::builtin_code_get_signature(current_code()) } {
        if sig.has_vararg() || sig.has_kwarg() || sig.num_kwonlyargnames() > 0 {
            let fname = unsafe { crate::builtin_code_name(current_code()) };
            let args = current_args();
            let bound = bind_kwargs_to_signature(sig, fname, &args, &[])?;
            return func(&bound);
        }
    }
    let args = current_args();
    func(&args)
}

/// Leaf execution mode for a user-function call reached through
/// [`call_callable_with_mode`].
///
/// `Jit` routes through the injected JIT-aware eval override
/// (`call_user_function`); `Plain` routes through the interpreter-only
/// eval (`call_user_function_plain`).  The callable-kind dispatch
/// (method / type / staticmethod / classmethod / instance-`__call__`) is
/// identical for both — only the leaf executor differs — so the two entry
/// points share one body instead of a stripped-down copy.
#[derive(Clone, Copy, PartialEq, Eq)]
enum CallMode {
    Jit,
    Plain,
}

pub fn call_callable(frame: &mut PyFrame, callable: PyObjectRef, args: &[PyObjectRef]) -> PyResult {
    call_callable_with_mode(frame, callable, args, CallMode::Jit)
}

/// CALL_FUNCTION_EX helper — unpack `starargs`, merge the `**` mapping, and
/// call.  Factored out of the interpreter's `call_function_ex` so the JIT
/// residual (`bh_call_function_ex_fn`) shares one implementation.  Mirrors
/// `argument.py` unpack_combined_starargs + `_combine_starstarargs_wrapped`:
/// a tuple/list stararg takes the fast path, any other iterable goes through
/// the iter protocol; a non-null `**` mapping accepts the dict fast path or
/// `keys()`/`__getitem__`.  `self_or_null` is the pre-callable stack slot —
/// a non-null value prepends as arg0.
pub fn call_function_ex(
    frame: &mut PyFrame,
    callable: PyObjectRef,
    self_or_null: PyObjectRef,
    starargs: PyObjectRef,
    kwargs_or_null: PyObjectRef,
) -> PyResult {
    let mut args: Vec<PyObjectRef> = unsafe {
        if pyre_object::is_tuple(starargs) {
            let n = pyre_object::w_tuple_len(starargs);
            (0..n as i64)
                .filter_map(|i| pyre_object::w_tuple_getitem(starargs, i))
                .collect()
        } else if pyre_object::is_list(starargs) {
            let n = pyre_object::w_list_len(starargs);
            (0..n as i64)
                .filter_map(|i| pyre_object::w_list_getitem(starargs, i))
                .collect()
        } else {
            // argument.py:92-104 `_combine_starargs_wrapped` — a non-tuple/list
            // stararg unpacks through `fixedview`, and a non-iterable surfaces
            // "argument after * must be an iterable, not %T" (not the bare
            // `iter()` TypeError).
            let mut unpacked: Vec<PyObjectRef> = Vec::new();
            crate::argument::combine_starargs_wrapped(&mut unpacked, starargs, callable)?;
            unpacked
        }
    };
    if !self_or_null.is_null() {
        args.insert(0, self_or_null);
    }

    // Merge the `**` mapping into the call.  argument.py:106-150
    // `_combine_starstarargs_wrapped` accepts any mapping — the dict fast
    // path or an arbitrary object via `keys()` / `__getitem__` — raising
    // "argument after ** must be a mapping" for a non-mapping and
    // "keywords must be strings" for a non-str key.
    if !kwargs_or_null.is_null() {
        let mut keyword_names_w: Vec<PyObjectRef> = Vec::new();
        let mut keywords_w: Vec<PyObjectRef> = Vec::new();
        crate::argument::combine_starstarargs_wrapped(
            &mut keyword_names_w,
            &mut keywords_w,
            kwargs_or_null,
            callable,
        )?;
        if !keyword_names_w.is_empty() {
            let entries: Vec<(Wtf8Buf, PyObjectRef)> = keyword_names_w
                .iter()
                .zip(keywords_w.iter())
                .map(|(&k, &v)| (unsafe { pyre_object::w_str_get_wtf8(k) }.to_owned(), v))
                .collect();
            return call_with_kwargs(frame, callable, &args, &entries);
        }
    }

    call_callable(frame, callable, &args)
}

/// CALL_KW helper — resolve keyword arguments against the callable and
/// call.  Factored out of the interpreter's `call_kw` so the JIT residual
/// (`bh_call_kw_fn`) shares one implementation.  `positional` holds the
/// `arg0..argN-1` values already in positional order (keyword tail
/// included); `kwarg_names` is the constant kwnames tuple (its length is
/// the number of trailing keyword args).  `self_or_null` is the
/// pre-callable stack slot — a non-null value prepends as arg0.
pub fn call_kw(
    frame: &mut PyFrame,
    callable: PyObjectRef,
    self_or_null: PyObjectRef,
    positional: &[PyObjectRef],
    kwarg_names: PyObjectRef,
) -> PyResult {
    let mut args: Vec<PyObjectRef> = positional.to_vec();

    if self_or_null != pyre_object::PY_NULL {
        args.insert(0, self_or_null);
    }

    // Unwrap bound methods: load_method pushes (method, PY_NULL) for
    // bound methods. Extract the underlying function and prepend the
    // receiver so resolve_kwargs sees the correct function signature.
    let callable_unwrapped = callable;
    let callable_unwrapped = if unsafe { pyre_object::is_method(callable_unwrapped) } {
        let func = unsafe { pyre_object::w_method_get_func(callable_unwrapped) };
        let receiver = unsafe { pyre_object::w_method_get_self(callable_unwrapped) };
        if !receiver.is_null() {
            args.insert(0, receiver);
        }
        func
    } else {
        callable_unwrapped
    };

    // function.py:712-713 `StaticMethod.descr_call` receives the original
    // Arguments object.  Preserve the positional/keyword split before the
    // generic signature resolver (a staticmethod wrapper has no Python
    // signature of its own) and forward both collections unchanged.
    if unsafe { pyre_object::is_staticmethod(callable_unwrapped) } {
        let nkw = if unsafe { pyre_object::is_tuple(kwarg_names) } {
            unsafe { pyre_object::w_tuple_len(kwarg_names) }
        } else {
            0
        };
        let n_pos = args.len().saturating_sub(nkw);
        let pos_args = args[..n_pos].to_vec();
        let mut kw_entries = Vec::with_capacity(nkw);
        for ki in 0..nkw {
            if let Some(name_obj) = unsafe { pyre_object::w_tuple_getitem(kwarg_names, ki as i64) }
            {
                let key = unsafe { pyre_object::w_str_get_wtf8(name_obj) }.to_owned();
                kw_entries.push((key, args[n_pos + ki]));
            }
        }
        return call_with_kwargs(frame, callable_unwrapped, &pos_args, &kw_entries);
    }

    // A base classmethod has no tp_call, but a user subtype may define
    // `__call__`. Split the keyword tail before generic signature handling;
    // call_with_kwargs performs the special-method descriptor binding.
    if unsafe { pyre_object::is_classmethod(callable_unwrapped) } {
        let nkw = if unsafe { pyre_object::is_tuple(kwarg_names) } {
            unsafe { pyre_object::w_tuple_len(kwarg_names) }
        } else {
            0
        };
        let n_pos = args.len().saturating_sub(nkw);
        let pos_args = args[..n_pos].to_vec();
        let mut kw_entries = Vec::with_capacity(nkw);
        for ki in 0..nkw {
            if let Some(name_obj) = unsafe { pyre_object::w_tuple_getitem(kwarg_names, ki as i64) }
            {
                let key = unsafe { pyre_object::w_str_get_wtf8(name_obj) }.to_owned();
                kw_entries.push((key, args[n_pos + ki]));
            }
        }
        return call_with_kwargs(frame, callable_unwrapped, &pos_args, &kw_entries);
    }

    // For type objects with kwargs: use call_with_kwargs which handles
    // __new__/__init__ kwargs forwarding correctly.
    if unsafe { pyre_object::is_type(callable_unwrapped) } {
        let nkw = if unsafe { pyre_object::is_tuple(kwarg_names) } {
            unsafe { pyre_object::w_tuple_len(kwarg_names) }
        } else {
            0
        };
        if nkw > 0 {
            let n_pos = args.len() - nkw;
            let pos_args = args[..n_pos].to_vec();
            let mut kw_entries = Vec::with_capacity(nkw);
            for ki in 0..nkw {
                let name = unsafe { pyre_object::w_tuple_getitem(kwarg_names, ki as i64) };
                if let Some(name_obj) = name {
                    let key = unsafe { pyre_object::w_str_get_wtf8(name_obj) }.to_owned();
                    kw_entries.push((key, args[n_pos + ki]));
                }
            }
            return call_with_kwargs(frame, callable_unwrapped, &pos_args, &kw_entries);
        }
    }

    // A generic alias has no signature of its own; its __call__
    // forwards to __origin__(*args, **kwargs).  Split the keyword tail
    // and route through call_with_kwargs so the origin's own kwargs
    // handling (e.g. dict.__init__) sees real keywords.
    if unsafe { pyre_object::is_generic_alias(callable_unwrapped) } {
        let nkw = if unsafe { pyre_object::is_tuple(kwarg_names) } {
            unsafe { pyre_object::w_tuple_len(kwarg_names) }
        } else {
            0
        };
        let n_pos = args.len().saturating_sub(nkw);
        let pos_args = args[..n_pos].to_vec();
        let mut kw_entries = Vec::with_capacity(nkw);
        for ki in 0..nkw {
            let name = unsafe { pyre_object::w_tuple_getitem(kwarg_names, ki as i64) };
            if let Some(name_obj) = name {
                let key = unsafe { pyre_object::w_str_get_wtf8(name_obj) }.to_owned();
                kw_entries.push((key, args[n_pos + ki]));
            }
        }
        return call_with_kwargs(frame, callable_unwrapped, &pos_args, &kw_entries);
    }

    // Resolve keyword args into positional order.
    // argument.py Arguments._match_signature step: match keywords to
    // argnames, fill defaults, pack *args/**kwargs. PyPy's
    // `space.call_args` performs this exactly once; pyre mirrors that
    // by calling resolve_kwargs here and then dispatching directly to
    // call_user_function_resolved — which skips the defaults_fill /
    // pack_varargs replay that call_user_function_with_args performs
    // for positional-only paths.
    let is_builtin = unsafe { crate::is_function(callable_unwrapped) }
        && unsafe {
            crate::is_builtin_code(crate::getcode(callable_unwrapped) as pyre_object::PyObjectRef)
        };
    if is_builtin {
        let nkw = if unsafe { pyre_object::is_tuple(kwarg_names) } {
            unsafe { pyre_object::w_tuple_len(kwarg_names) }
        } else {
            0
        };
        if nkw > 0 {
            let n_pos = args.len() - nkw;
            let pos_args = args[..n_pos].to_vec();
            let mut kw_entries = Vec::with_capacity(nkw);
            for ki in 0..nkw {
                let name = unsafe { pyre_object::w_tuple_getitem(kwarg_names, ki as i64) };
                if let Some(name_obj) = name {
                    let key = unsafe { pyre_object::w_str_get_wtf8(name_obj) }.to_owned();
                    kw_entries.push((key, args[n_pos + ki]));
                }
            }
            // PyPy CALL_FUNCTION_KW builds an Arguments object with
            // keyword_names_w / keywords_w, and the profiled-builtin path
            // passes that same object to call_args_and_c_profile.  Route
            // through call_with_kwargs so pyre's profile path constructs
            // Arguments::with_kw instead of treating the kwargs dict tail
            // as a positional firstarg.
            return call_with_kwargs(frame, callable_unwrapped, &pos_args, &kw_entries);
        }
        return call_callable(frame, callable_unwrapped, &args);
    }

    // pypy/interpreter/function.py Method.call_args parity: unwrap
    // bound method by prepending the receiver, then run resolve_kwargs
    // against the underlying function. This matches
    // `self.space.call_args(w_function, args)` after the MRO-dispatched
    // `im_func` has been extracted.
    let (target_func, mut prepended) = if unsafe { pyre_object::is_method(callable_unwrapped) } {
        let func = unsafe { pyre_object::w_method_get_func(callable_unwrapped) };
        let receiver = unsafe {
            let w_self = pyre_object::w_method_get_self(callable_unwrapped);
            if !w_self.is_null() {
                w_self
            } else {
                pyre_object::w_method_get_class(callable_unwrapped)
            }
        };
        if !receiver.is_null() {
            let mut prepended = Vec::with_capacity(1 + args.len());
            prepended.push(receiver);
            prepended.extend_from_slice(&args);
            (func, Some(prepended))
        } else {
            (func, None)
        }
    } else {
        (callable_unwrapped, None)
    };
    let call_args: &[PyObjectRef] = prepended.as_deref().unwrap_or(&args);
    let resolved = resolve_kwargs(target_func, call_args, kwarg_names)?;
    // Drop the temporary prepended buffer once resolved is built.
    prepended = None;
    let _ = prepended;

    if unsafe { crate::is_function(target_func) } {
        call_user_function_resolved(frame, target_func, &resolved)
    } else {
        call_callable(frame, target_func, &resolved)
    }
}

/// `typeobject.c type_call` is the metaclass's `tp_call`; calling a class
/// dispatches through `type(cls).__call__`.  The base `type` has no
/// `__call__` dict entry — the implicit `__new__`/`__init__` path below is
/// its `tp_call` — so any `__call__` resolved on a *non-`type`* metaclass
/// is a genuine override (enum.EnumType, custom metaclasses with
/// `__call__`).  Returns the override bound to `callable`, or `None` when
/// the default class-instantiation path should run.
fn metaclass_call_override(callable: PyObjectRef) -> Option<PyObjectRef> {
    let metaclass = crate::typedef::r#type(callable)?;
    if std::ptr::eq(metaclass, crate::typedef::w_type()) {
        return None;
    }
    // Resolve WHERE `__call__` is defined first; the default `type.__call__`
    // (the implicit instantiation path) is not an override, so a metaclass
    // that merely inherits it — e.g. ABCMeta — keeps the fast path.  The
    // defining-class half is the cheap guard, so it runs before the value
    // half's second residual walk (avoided on the common fast path).
    let where_defined =
        unsafe { crate::baseobjspace::lookup_where_class_uncached(metaclass, "__call__") }?;
    if std::ptr::eq(where_defined, crate::typedef::w_type()) {
        return None;
    }
    let call_descr =
        unsafe { crate::baseobjspace::lookup_in_type_where_uncached(metaclass, "__call__") }?;
    let bound = unsafe { crate::baseobjspace::get(call_descr, callable, metaclass) }
        .ok()
        .flatten()
        .unwrap_or(call_descr);
    Some(bound)
}

/// Resolve a `__call__` introduced by a classmethod *subtype*. The base
/// classmethod has no such slot in PyPy or CPython 3.14. Descriptor binding
/// is essential here: an ordinary function receives the wrapper, while a
/// staticmethod override does not.
fn classmethod_call_override(callable: PyObjectRef) -> Result<Option<PyObjectRef>, PyError> {
    if !unsafe { pyre_object::is_classmethod(callable) } {
        return Ok(None);
    }
    let Some(w_type) = crate::typedef::r#type(callable) else {
        return Ok(None);
    };
    let Some(call_descr) = (unsafe { crate::baseobjspace::lookup_in_type(w_type, "__call__") })
    else {
        return Ok(None);
    };
    let bound =
        unsafe { crate::baseobjspace::get(call_descr, callable, w_type) }?.unwrap_or(call_descr);
    Ok(Some(bound))
}

/// Resolve a `__call__` introduced by a staticmethod subtype. Exact builtin
/// wrappers use the direct unwrap fast path; subtypes honor their override.
fn staticmethod_call_override(callable: PyObjectRef) -> Result<Option<PyObjectRef>, PyError> {
    if !unsafe { pyre_object::is_staticmethod(callable) }
        || unsafe {
            pyre_object::is_exact_type(callable, &pyre_object::function::STATICMETHOD_TYPE)
        }
    {
        return Ok(None);
    }
    let Some(w_type) = crate::typedef::r#type(callable) else {
        return Ok(None);
    };
    let Some(call_descr) = (unsafe { crate::baseobjspace::lookup_in_type(w_type, "__call__") })
    else {
        return Ok(None);
    };
    let bound =
        unsafe { crate::baseobjspace::get(call_descr, callable, w_type) }?.unwrap_or(call_descr);
    Ok(Some(bound))
}

fn call_callable_with_mode(
    frame: &mut PyFrame,
    callable: PyObjectRef,
    args: &[PyObjectRef],
    mode: CallMode,
) -> PyResult {
    if unsafe { pyre_object::is_method(callable) } {
        let func = unsafe { pyre_object::w_method_get_func(callable) };
        let receiver = unsafe {
            let w_self = pyre_object::w_method_get_self(callable);
            if !w_self.is_null() {
                w_self
            } else {
                pyre_object::w_method_get_class(callable)
            }
        };
        let mut call_args = Vec::with_capacity(1 + args.len());
        if !receiver.is_null() {
            call_args.push(receiver);
        }
        call_args.extend_from_slice(args);
        return call_callable_with_mode(frame, func, &call_args, mode);
    }
    if unsafe { pyre_object::is_type(callable) } {
        if let Some(bound) = metaclass_call_override(callable) {
            return call_callable_with_mode(frame, bound, args, mode);
        }
        return type_descr_call_with_mode(frame, callable, args, mode);
    }

    // staticmethod → unwrap
    // PyPy: function.py StaticMethod.descr_call
    if unsafe { pyre_object::is_exact_type(callable, &pyre_object::function::STATICMETHOD_TYPE) } {
        let func = unsafe { pyre_object::w_staticmethod_get_func(callable) };
        return call_callable_with_mode(frame, func, args, mode);
    }
    if let Some(bound) = classmethod_call_override(callable)? {
        return call_callable_with_mode(frame, bound, args, mode);
    }
    // The base ClassMethod defines no descr_call (function.py), so a raw
    // classmethod object falls through to the not-callable error.

    // Instance with __call__ — PyPy: descroperation.py descr_call
    if unsafe { pyre_object::is_instance(callable) } {
        let w_type = unsafe { pyre_object::w_instance_get_type(callable) };
        if let Some(call_fn) = unsafe { crate::baseobjspace::lookup_in_type(w_type, "__call__") } {
            let mut call_args = Vec::with_capacity(1 + args.len());
            call_args.push(callable);
            call_args.extend_from_slice(args);
            return call_callable_with_mode(frame, call_fn, &call_args, mode);
        }
    }

    // GenericAlias.__call__ (`_pypy_generic_alias.py:41`) —
    // `self.__origin__(*args, **kwargs)`, then best-effort
    // `result.__orig_class__ = self`.
    if unsafe { pyre_object::is_generic_alias(callable) } {
        let origin = unsafe { pyre_object::w_generic_alias_get_origin(callable) };
        let result = call_callable_with_mode(frame, origin, args, mode)?;
        set_orig_class(result, callable)?;
        return Ok(result);
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
        |callable| match mode {
            CallMode::Jit => call_user_function(frame, callable, args),
            CallMode::Plain => call_user_function_plain(frame, callable, args),
        },
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
    let w_globals = unsafe { function_get_globals_obj(callable) };
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
                w_globals,
                execution_context,
                closure,
                crate::pyframe::FrameLocalsArrayAllocation::OldGenGc,
            )?);
        return gen_frame.into_generator();
    }

    let mut func_frame =
        crate::pyframe::FrameBox::new(PyFrame::try_new_for_call_with_closure_and_globals_obj(
            w_code,
            &final_args,
            w_globals,
            execution_context,
            closure,
            crate::pyframe::FrameLocalsArrayAllocation::OldGenGc,
        )?);
    func_frame.fix_array_ptrs();
    let _callee_locals_root = FrameLocalsRoot::new_mut(&mut func_frame);
    func_frame.run()
}

/// Explicit residual-call protocol used by JIT inline framestack concrete
/// execution.
///
/// Residual calls reached from inline execution are opaque slow paths. They
/// must not accidentally reuse the generic JIT-aware `call_user_function()`
/// entry, because that can re-enter portal state that belongs to the outer
/// trace instead of the active inline framestack — hence `CallMode::Plain`,
/// which keeps every user-function leaf (including `__new__`/`__init__`
/// reached through type construction) on the interpreter-only eval. The
/// callable-kind dispatch is otherwise identical to `call_callable`, so both
/// share `call_callable_with_mode` rather than maintaining a divergent copy.
pub fn call_callable_inline_residual(
    frame: &mut PyFrame,
    callable: PyObjectRef,
    args: &[PyObjectRef],
) -> PyResult {
    call_callable_with_mode(frame, callable, args, CallMode::Plain)
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
///
/// Structural note: this is an inline reimplementation of the
/// `_match_signature` / `_match_keywords` / `ArgErr*` steps against the
/// callee's `CodeObject`, not a port of PyPy's `Arguments` class object.
/// Every step cites its `argument.py` line and the observable behavior
/// (fill order, positional-only handling, duplicate/unexpected/missing
/// diagnostics, `*args` / `**kwargs` packing, error message text) matches
/// CPython bit-for-bit.  Reifying an `Arguments` object with the same
/// method surface is a separate, much larger refactor that would also
/// re-thread `call_with_kwargs` / `bind_kwargs_to_signature`; it is out of
/// scope here and tracked as a follow-up.
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
                    let new_fn = unsafe { unwrap_static_new(new_fn) };
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

    // `argument.py:235-236` — flag too-many positional args with no *vararg
    // (kwargs are matched separately). The error is raised after keyword
    // matching (`argument.py:289`) so a duplicate/positional-only/unknown-
    // keyword error on the same call wins first.
    let too_many_args = n_pos > n_pos_params && !has_varargs;

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
    let mut unmatched_kw_names: Vec<Wtf8Buf> = Vec::new();
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
        // A lone-surrogate keyword name (not valid UTF-8) never equals a
        // source-level parameter name, so it falls straight to **kwargs or
        // the unexpected-keyword error below.
        let kw_str = unsafe { pyre_object::w_str_get_value_opt(kw_name_obj) };
        let mut matched = false;
        for pi in 0..nparams {
            let param_name = &*code.varnames[skip_cls + pi];
            if kw_str == Some(param_name) {
                // argument.py:474 — positional-only parameter: if has_kwarg,
                // treat as unmatched (absorb into **kwargs); otherwise error.
                if skip_cls + pi < posonlyarg_count {
                    if has_varkw {
                        break; // fall through to !matched → extra_kwargs
                    }
                    return Err(crate::PyError::type_error(format!(
                        "{}() got some positional-only arguments passed as keyword arguments: '{}'",
                        fname, param_name
                    )));
                }
                // argument.py:410 — duplicate keyword argument
                if !result[pi].is_null() {
                    return Err(crate::PyError::type_error(format!(
                        "{}() got multiple values for argument '{}'",
                        fname, param_name
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
                unmatched_kw_names
                    .push(unsafe { pyre_object::w_str_get_wtf8(kw_name_obj).to_owned() });
            }
        }
    }

    // `argument.py:270-271` ArgErrUnknownKwds — unmatched kwargs and no
    // **kwarg to absorb them.
    if !unmatched_kw_names.is_empty() {
        let msg = format_unknown_kwds_err(&fname, &unmatched_kw_names);
        return Err(crate::PyError::type_error(msg));
    }

    // `argument.py:289` — too-many-positionals raised here, after the
    // keyword-matching errors above.
    if too_many_args {
        let ndefaults = {
            let defaults = unsafe { crate::function_get_defaults(target_func) };
            if !defaults.is_null() {
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
                        let kw_s = unsafe { pyre_object::w_str_get_value_opt(kw_obj) };
                        (0..nkwonly)
                            .any(|j| Some(&*code.varnames[skip_cls + n_pos_params + j]) == kw_s)
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

    // Fill positional defaults (PyPy: _match_signature defs_w)
    // Defaults cover the LAST N of the positional params (arg_count).
    let defaults = unsafe { crate::function_get_defaults(target_func) };
    if !defaults.is_null() {
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
    kwargs: &[(Wtf8Buf, PyObjectRef)],
) -> Result<Vec<PyObjectRef>, crate::PyError> {
    let nparams = sig.argnames.len();
    let n_pos_params = sig.num_argnames(); // positional params (excludes kwonly tail)
    let posonly = sig.posonlyargcount;
    let has_varargs = sig.varargname.is_some();
    let has_varkw = sig.kwargname.is_some();
    let n_pos = pos_args.len();

    // A METH_O-style builtin accepts no keyword arguments — every parameter
    // positional-only, no `**kwargs`, no keyword-only — so any keyword is
    // rejected with the "takes no keyword arguments" form (e.g. `len`, `abs`).
    if !kwargs.is_empty() && !has_varkw && sig.num_kwonlyargnames() == 0 && posonly == n_pos_params
    {
        return Err(crate::PyError::type_error(format!(
            "{fname}() takes no keyword arguments"
        )));
    }

    // argument.py:235-236 — flag too many positionals with no `*args` to
    // absorb, but do not raise yet: argument.py:289 raises it only after
    // keyword matching, so a duplicate/positional-only/unknown-keyword error
    // on the same call wins first.
    let too_many_args = n_pos > n_pos_params && !has_varargs;

    let mut result = vec![pyre_object::PY_NULL; nparams];
    for i in 0..n_pos.min(n_pos_params) {
        result[i] = pos_args[i];
    }

    // _match_keywords — match each keyword to a param name by index.
    let mut extra_kwargs: Vec<(PyObjectRef, PyObjectRef)> = Vec::new();
    let mut unmatched_kw_names: Vec<Wtf8Buf> = Vec::new();
    for (key, value) in kwargs {
        // A lone-surrogate keyword name (not valid UTF-8) never equals a
        // source-level parameter name, so it falls straight to **kwargs or
        // the unexpected-keyword error below.
        let key_str = key.as_str().ok();
        let mut matched = false;
        for pi in 0..nparams {
            if key_str == Some(sig.argnames[pi]) {
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
                extra_kwargs.push((pyre_object::w_str_from_wtf8(key.clone()), *value));
            } else {
                unmatched_kw_names.push(key.clone());
            }
        }
    }

    if !unmatched_kw_names.is_empty() {
        // parse_obj (argument.py:377-380) rewrites the unknown-keyword message
        // to "takes no keyword arguments" when the signature accepts no keywords
        // at all (no **kwargs and no keyword-only params). Every BuiltinCode
        // call routes through parse_obj (gateway.py funcrun / funcrun_obj), so
        // the rewrite applies at any arity, not just the single-argument form.
        let msg = if !has_varkw && sig.num_kwonlyargnames() == 0 {
            format!("{}() takes no keyword arguments", fname)
        } else {
            format_unknown_kwds_err(fname, &unmatched_kw_names)
        };
        return Err(crate::PyError::type_error(msg));
    }

    // argument.py:289 — too-many-positionals is raised last, after the
    // keyword-matching errors above.
    if too_many_args {
        return Err(crate::PyError::type_error(format!(
            "{}() takes {} positional argument{} but {} {} given",
            fname,
            n_pos_params,
            if n_pos_params != 1 { "s" } else { "" },
            n_pos,
            if n_pos != 1 { "were" } else { "was" },
        )));
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
    kwargs: &[(Wtf8Buf, PyObjectRef)],
) -> PyResult {
    // function.py:712-713 StaticMethod.descr_call — the wrapper contributes
    // no implicit argument; forward the original positional and keyword
    // collections unchanged to its w_function.
    if unsafe { pyre_object::is_exact_type(callable, &pyre_object::function::STATICMETHOD_TYPE) } {
        let func = unsafe { pyre_object::w_staticmethod_get_func(callable) };
        return call_with_kwargs(frame, func, pos_args, kwargs);
    }
    if let Some(bound) = staticmethod_call_override(callable)? {
        return call_with_kwargs(frame, bound, pos_args, kwargs);
    }

    if unsafe { pyre_object::is_classmethod(callable) } {
        if let Some(bound) = classmethod_call_override(callable)? {
            return call_with_kwargs(frame, bound, pos_args, kwargs);
        }
        let type_name = crate::typedef::r#type(callable)
            .map(|tp| unsafe { pyre_object::w_type_get_name(tp) })
            .unwrap_or("classmethod");
        return Err(PyError::type_error(format!(
            "'{type_name}' object is not callable"
        )));
    }

    // Unwrap bound methods: prepend receiver to pos_args.
    if unsafe { pyre_object::is_method(callable) } {
        let func = unsafe { pyre_object::w_method_get_func(callable) };
        let receiver = unsafe { pyre_object::w_method_get_self(callable) };
        let mut full_args = Vec::with_capacity(1 + pos_args.len());
        if !receiver.is_null() {
            full_args.push(receiver);
        }
        full_args.extend_from_slice(pos_args);
        return call_with_kwargs(frame, func, &full_args, kwargs);
    }

    // A class call routes through `type(cls).__call__` when the metaclass
    // overrides it (enum functional API passes `module=`/`type=` kwargs).
    if unsafe { pyre_object::is_type(callable) } {
        if let Some(bound) = metaclass_call_override(callable) {
            return call_with_kwargs(frame, bound, pos_args, kwargs);
        }
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
                // Under an active C-level profiler the call must still emit
                // `c_call_trace` / `c_return_trace`, so route the bound flat
                // slice through the profile-aware path like the marker branch
                // below rather than invoking the builtin directly.
                let frame_ptr = frame as *mut PyFrame;
                if unsafe { (*frame_ptr).get_is_being_profiled() } {
                    let keyword_names_w: Vec<pyre_object::PyObjectRef> = kwargs
                        .iter()
                        .map(|(k, _)| pyre_object::w_str_from_wtf8(k.clone()))
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
                        &bound,
                    );
                    if w_res == pyre_object::PY_NULL {
                        return Err(take_call_error()
                            .unwrap_or_else(|| crate::PyError::value_error("call failed")));
                    }
                    return Ok(w_res);
                }
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
                for (key, value) in kwargs {
                    unsafe {
                        pyre_object::w_dict_store(
                            kwargs_dict,
                            pyre_object::w_str_from_wtf8(key.clone()),
                            *value,
                        );
                    }
                }
                // Store the marker last so a user keyword literally named
                // `__pyre_kw__` cannot overwrite the sentinel: the reserved
                // key always resolves to the sentinel value that detection
                // compares by identity.
                unsafe {
                    pyre_object::w_dict_store(
                        kwargs_dict,
                        pyre_object::w_str_new("__pyre_kw__"),
                        pyre_object::kw_marker::w_kw_marker_sentinel(),
                    );
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
                        .map(|(k, _)| pyre_object::w_str_from_wtf8(k.clone()))
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

            // `argument.py:235-236` — flag too-many positional args with no
            // *vararg; the error is raised after keyword matching
            // (`argument.py:289`) so a duplicate/positional-only/unknown-keyword
            // error on the same call wins first.
            let too_many_args = pos_args.len() > n_pos_params && !has_varargs;

            // Build parameter array
            let mut result = vec![pyre_object::PY_NULL; total_params];
            // Fill positional args — bound at `n_pos_params` so excess
            // positionals don't spill into kwonly slots.
            for i in 0..pos_args.len().min(n_pos_params) {
                result[i] = pos_args[i];
            }
            // Match keywords to parameter names
            let posonly = code.posonlyarg_count as usize;
            let mut extra_kwargs: Vec<(Wtf8Buf, PyObjectRef)> = Vec::new();
            let mut unmatched_kw_names: Vec<Wtf8Buf> = Vec::new();
            for (key, value) in kwargs {
                // A lone-surrogate keyword name never equals a source-level
                // parameter name; it falls to **kwargs or the error below.
                let key_str = key.as_str().ok();
                let mut matched = false;
                for pi in 0..total_params {
                    if key_str == Some(code.varnames[pi].as_str()) {
                        // argument.py:474 — positional-only param passed by
                        // keyword: absorb into **kwargs if present, else error.
                        if pi < posonly {
                            if has_varkw {
                                break;
                            }
                            return Err(crate::PyError::type_error(format!(
                                "{}() got some positional-only arguments passed as keyword arguments: '{}'",
                                fname, key
                            )));
                        }
                        // argument.py:495 — ArgErrMultipleValues: keyword
                        // duplicates an already-bound positional argument.
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
                        extra_kwargs.push((key.clone(), *value));
                    } else {
                        unmatched_kw_names.push(key.clone());
                    }
                }
            }

            // `argument.py:270-271` ArgErrUnknownKwds.
            if !unmatched_kw_names.is_empty() {
                let msg = format_unknown_kwds_err(&fname, &unmatched_kw_names);
                return Err(crate::PyError::type_error(msg));
            }

            // `argument.py:289` — too-many-positionals raised here, after the
            // keyword-matching errors above.
            if too_many_args {
                let ndefaults = {
                    let defaults = unsafe { crate::function_get_defaults(callable) };
                    if !defaults.is_null() {
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

            // Fill positional defaults from __defaults__ tuple.
            let defaults = unsafe { crate::function_get_defaults(callable) };
            if !defaults.is_null() {
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
                        pyre_object::w_dict_store(
                            kw_dict,
                            pyre_object::w_str_from_wtf8(key.clone()),
                            *value,
                        );
                    }
                }
                final_args.push(kw_dict);
            }

            // Create frame and execute
            let w_globals = unsafe { function_get_globals_obj(callable) };
            let closure = unsafe { function_get_closure(callable) };
            let mut func_frame = crate::pyframe::FrameBox::new(
                crate::pyframe::PyFrame::try_new_for_call_with_closure_and_globals_obj(
                    w_code,
                    &final_args,
                    w_globals,
                    frame.execution_context,
                    closure,
                    crate::pyframe::FrameLocalsArrayAllocation::OldGenGc,
                )?,
            );
            func_frame.fix_array_ptrs();
            // Generator/coroutine function: return a generator object
            // instead of running the body, matching the positional call
            // path's `code_flags_make_generator` branch.  Without this a
            // generator function invoked with keyword arguments (e.g.
            // `func(*args, **kwds)` from `contextlib.contextmanager`) would
            // execute eagerly and surface the first yielded value.
            if crate::pyframe::code_flags_make_generator(code.flags) {
                return func_frame.into_generator();
            }
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
        // The `function` type is non-acceptable-as-base too, but its
        // `tp_new` (`FunctionType(code, globals, ..., kwdefaults=...)`)
        // does take keyword arguments, so route those through `__new__`.
        let is_function_type = std::ptr::eq(
            callable,
            crate::typedef::gettypeobject(&crate::FUNCTION_TYPE),
        );
        if !kwargs.is_empty()
            && !is_function_type
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
            let new_fn = unsafe { unwrap_static_new(new_fn) };
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
        // `instance` is a movable nursery object held only as a Rust local
        // across the `__init__` dispatch below, which runs arbitrary
        // allocating code and can relocate it during a minor collection.
        // Pin it on the shadow stack and reload the (possibly forwarded)
        // pointer afterwards — the manual equivalent of the translator's
        // shadowstack save/restore around a collecting call
        // (framework.py:853-856).
        let _instance_roots = pyre_object::gc_roots::push_roots();
        let instance_slot = pyre_object::gc_roots::shadow_stack_len();
        pyre_object::gc_roots::pin_root(instance);
        // Step 2: __init__(self, *args, **kwargs) with full kwargs support.
        if let Some(w_insttype) = type_call_init_type(instance, callable)
            && !type_call_type_x_shortcut(callable, pos_args.len(), kwargs.is_empty())
            && let Some(init_fn) =
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
        return Ok(pyre_object::gc_roots::shadow_stack_get(instance_slot));
    }

    // For methods: unwrap and retry
    if unsafe { pyre_object::is_method(callable) } {
        let func = unsafe { pyre_object::w_method_get_func(callable) };
        let w_self = unsafe { pyre_object::w_method_get_self(callable) };
        let mut full_args = Vec::with_capacity(1 + pos_args.len());
        if !w_self.is_null() {
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

    // GenericAlias.__call__ (`_pypy_generic_alias.py:41`) —
    // `self.__origin__(*args, **kwargs)`, then best-effort
    // `result.__orig_class__ = self`.
    if unsafe { pyre_object::is_generic_alias(callable) } {
        let origin = unsafe { pyre_object::w_generic_alias_get_origin(callable) };
        let result = call_with_kwargs(frame, origin, pos_args, kwargs)?;
        set_orig_class(result, callable)?;
        return Ok(result);
    }

    // Fallback: call_callable with positional args only
    call_callable(frame, callable, pos_args)
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
/// Re-export for crate-external callers that need a frame-less call path.
///
/// This wrapper preserves the legacy `PyObjectRef`-returning shape used by
/// most call sites. Errors are stashed in `PENDING_CALL_ERROR`; callers
/// recover them via `take_call_error()` after a `PY_NULL` return.
pub fn call_function_impl_raw(callable: PyObjectRef, args: &[PyObjectRef]) -> PyObjectRef {
    match call_function_impl_result(callable, args) {
        Ok(result) => result,
        Err(e) => {
            log_call_error(&e.message);
            set_call_error(e);
            PY_NULL
        }
    }
}

/// Cold debug-diagnostic sink for `call_function_impl_raw`. Residualized so
/// the `eprintln!` formatting machinery (`fmt::rt::Argument`) stays out of the
/// traced graph — the whole body is a host-stderr write behind the env probe,
/// which the tracer cannot model. No-op under `sandbox` (host access must go
/// through the seam).
#[majit_macros::dont_look_inside]
fn log_call_error(message: &str) {
    #[cfg(not(feature = "sandbox"))]
    if pyre_debug_call_enabled() {
        eprintln!("[call_function_impl] error: {message}");
    }
    #[cfg(feature = "sandbox")]
    let _ = message;
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
    // `baseobjspace.py:1195-1198 call_function` is translated RPython: its
    // callable and `args_w` entries are shadow-stack roots across the entry
    // stack check, and the GC transform reloads their possibly moved
    // addresses afterwards.  Rust's incoming slice only contains copied raw
    // pointers, so establish the same roots before the first collecting call
    // and dispatch from freshly reloaded values.
    let _roots = pyre_object::gc_roots::push_roots();
    let root_base = pyre_object::gc_roots::shadow_stack_len();
    pyre_object::gc_roots::pin_root(callable);
    for &arg in args {
        pyre_object::gc_roots::pin_root(arg);
    }

    // rpython/rlib/rstack.py:42 stack_check(): every interpreter call
    // boundary checks the native stack synchronously, so deep recursion
    // raises RecursionError instead of letting the OS abort on a
    // guard-page hit. Also drain any JIT-prologue pending overflow.
    crate::stack_check::drain_jit_pending_exception()?;
    crate::stack_check::stack_check()?;

    let callable = pyre_object::gc_roots::shadow_stack_get(root_base);
    let rooted_args = (0..args.len())
        .map(|i| pyre_object::gc_roots::shadow_stack_get(root_base + 1 + i))
        .collect::<Vec<_>>();
    let args = rooted_args.as_slice();

    unsafe {
        if pyre_object::is_method(callable) {
            let func = pyre_object::w_method_get_func(callable);
            let w_self = pyre_object::w_method_get_self(callable);
            let receiver = if !w_self.is_null() {
                w_self
            } else {
                pyre_object::w_method_get_class(callable)
            };
            let mut call_args = Vec::with_capacity(1 + args.len());
            if !receiver.is_null() {
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
            if let Some(bound) = metaclass_call_override(callable) {
                return call_function_impl_result(bound, args);
            }
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
        if pyre_object::is_exact_type(callable, &pyre_object::function::STATICMETHOD_TYPE) {
            let func = pyre_object::w_staticmethod_get_func(callable);
            return call_function_impl_result(func, args);
        }
        if let Some(bound) = staticmethod_call_override(callable)? {
            return call_function_impl_result(bound, args);
        }
        if let Some(bound) = classmethod_call_override(callable)? {
            return call_function_impl_result(bound, args);
        }
        // ClassMethod has no descr_call (function.py:718-768; CPython 3.14
        // `PyClassMethod_Type.tp_call = 0`), so a raw wrapper falls through
        // to the ordinary not-callable error.
        // GenericAlias.__call__ (`_pypy_generic_alias.py:41`) —
        // `self.__origin__(*args, **kwargs)`, then best-effort
        // `result.__orig_class__ = self`.  Resolved here because the call
        // path does not consult a typedef `__call__` for builtin W_Roots.
        if pyre_object::is_generic_alias(callable) {
            let origin = pyre_object::w_generic_alias_get_origin(callable);
            let result = call_function_impl_result(origin, args)?;
            set_orig_class(result, callable)?;
            return Ok(result);
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
    let type_name = crate::typedef::r#type(callable)
        .map(|tp| unsafe { pyre_object::w_type_get_name(tp) })
        .unwrap_or_else(|| unsafe { (*(*callable).ob_type).name });
    Err(PyError::type_error(format!(
        "'{type_name}' object is not callable"
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

/// `typeobject.c type_call` — a type whose `tp_new` is NULL
/// (`Py_TPFLAGS_DISALLOW_INSTANTIATION`, e.g. generator) refuses
/// `Type()` with `cannot create 'X' instances`.
pub(crate) fn check_type_instantiable(w_type: PyObjectRef) -> Result<(), PyError> {
    if unsafe { pyre_object::w_type_disallows_instantiation(w_type) } {
        let name = unsafe { pyre_object::w_type_get_name(w_type) };
        return Err(PyError::type_error(format!(
            "cannot create '{name}' instances"
        )));
    }
    // Abstract-class rejection lives in `object.__new__` (objectobject.py:131
    // descr__new__ → `w_type_is_abstract`), the single enforcement point, so
    // the error names the missing methods.  A duplicate check here would fire
    // first with a less specific message.
    Ok(())
}

/// `type.__call__(cls, *args)` — the metaclass-level instantiation entry
/// (`typeobject.c type_call`).  Runs `__new__`/`__init__` directly, WITHOUT
/// re-dispatching through the metaclass (a custom metaclass `__call__` that
/// delegates via `super().__call__` lands here), and surfaces the stashed
/// error instead of a null sentinel.
pub fn type_call_instantiate(
    w_type: PyObjectRef,
    args: &[PyObjectRef],
) -> Result<PyObjectRef, PyError> {
    clear_call_error();
    let result = type_descr_call_impl(w_type, args);
    if result.is_null() {
        if let Some(err) = take_call_error() {
            return Err(err);
        }
    }
    Ok(result)
}

/// Type call without a PyFrame.
/// PyPy: typeobject.py descr_call
fn type_descr_call_impl(w_type: PyObjectRef, args: &[PyObjectRef]) -> PyObjectRef {
    // typeobject.py descr_call keeps `w_type`, every argument, and the new
    // instance live across both Python calls.  In translated RPython the GC
    // transform reloads these from shadow-stack slots after `__new__`; Rust
    // slices are only copies and otherwise retain pre-move addresses.
    let _roots = pyre_object::gc_roots::push_roots();
    let root_base = pyre_object::gc_roots::shadow_stack_len();
    pyre_object::gc_roots::pin_root(w_type);
    for &arg in args {
        pyre_object::gc_roots::pin_root(arg);
    }
    let current_type = || pyre_object::gc_roots::shadow_stack_get(root_base);
    let current_args = || {
        (0..args.len())
            .map(|i| pyre_object::gc_roots::shadow_stack_get(root_base + 1 + i))
            .collect::<Vec<_>>()
    };

    if let Err(e) = check_type_instantiable(current_type()) {
        set_call_error(e);
        return PY_NULL;
    }
    // Step 1: __new__
    let instance = if let Some(new_fn) =
        unsafe { crate::baseobjspace::lookup_in_type(current_type(), "__new__") }
    {
        let mut new_args = Vec::with_capacity(1 + args.len());
        new_args.push(current_type());
        new_args.extend(current_args());
        call_function_impl(new_fn, &new_args)
    } else {
        pyre_object::w_instance_new(current_type())
    };
    let instance_slot = pyre_object::gc_roots::shadow_stack_len();
    pyre_object::gc_roots::pin_root(instance);

    // Step 2: __init__ — only if __new__ returned an instance of w_type.
    // PyPy checks the Python-level type(instance), so builtin-layout subtypes
    // like set subclasses still run __init__.
    if let Some(w_insttype) = type_call_init_type(
        pyre_object::gc_roots::shadow_stack_get(instance_slot),
        current_type(),
    ) && !type_call_type_x_shortcut(current_type(), args.len(), true)
    {
        if let Some(init_fn) =
            unsafe { crate::baseobjspace::lookup_in_type(w_insttype, "__init__") }
        {
            let mut init_args = Vec::with_capacity(1 + args.len());
            init_args.push(pyre_object::gc_roots::shadow_stack_get(instance_slot));
            init_args.extend(current_args());
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

    pyre_object::gc_roots::shadow_stack_get(instance_slot)
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

/// typeobject.py:735-736 — the `type(x)` shortcut: `type.__call__` skips
/// __init__ when self is the `type` builtin, there are no keyword
/// arguments, and exactly one positional argument (`type(x)` returns the
/// class of x, already produced by __new__).
fn type_call_type_x_shortcut(w_type: PyObjectRef, nargs: usize, no_kwargs: bool) -> bool {
    no_kwargs && nargs == 1 && std::ptr::eq(w_type, crate::typedef::w_type())
}

/// Pointer-based subtype check for descr_call __init__ guard — the MRO
/// membership scan lives in `pyre_object::w_type_issubtype`.
fn issubtype_ptr(w_type: PyObjectRef, cls: PyObjectRef) -> bool {
    unsafe { pyre_object::w_type_issubtype(w_type, cls) }
}

/// Helper: call a user function with arbitrary args from descriptor context.
fn call_user_function_with_args(func: PyObjectRef, args: &[PyObjectRef]) -> PyObjectRef {
    let w_code = unsafe { crate::getcode(func) };
    let w_globals = unsafe { function_get_globals_obj(func) };
    let closure = unsafe { function_get_closure(func) };
    let func_code = unsafe {
        crate::w_code_get_ptr(w_code as pyre_object::PyObjectRef) as *const crate::CodeObject
    };
    let exec_ctx = build_class_exec_ctx();
    let exec_ctx = if exec_ctx.is_null() {
        take_last_exec_ctx()
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
                w_globals,
                exec_ctx,
                closure,
                crate::pyframe::FrameLocalsArrayAllocation::OldGenGc,
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
            w_globals,
            exec_ctx,
            closure,
            crate::pyframe::FrameLocalsArrayAllocation::OldGenGc,
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
    let w_globals = unsafe { function_get_globals_obj(func) };
    let closure = unsafe { function_get_closure(func) };
    let func_code = unsafe {
        crate::w_code_get_ptr(w_code as pyre_object::PyObjectRef) as *const crate::CodeObject
    };
    let exec_ctx = build_class_exec_ctx();
    let exec_ctx = if exec_ctx.is_null() {
        take_last_exec_ctx()
    } else {
        exec_ctx
    };
    let code_ref = unsafe { &*func_code };

    let mut frame =
        crate::pyframe::FrameBox::new(PyFrame::new_for_call_with_closure_and_globals_obj(
            w_code,
            args,
            w_globals,
            exec_ctx,
            closure,
            crate::pyframe::FrameLocalsArrayAllocation::OldGenGc,
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
/// `__new__` is stored as an implicit `staticmethod` (type_new_staticmethod);
/// unwrap it to the underlying function so the signature-based dispatch and
/// `is_function` fast-paths below see the real callable rather than the
/// descriptor wrapper. Builtin `__new__` (not a staticmethod) passes through.
unsafe fn unwrap_static_new(f: PyObjectRef) -> PyObjectRef {
    if unsafe { pyre_object::function::is_staticmethod(f) } {
        unsafe { pyre_object::function::w_staticmethod_get_func(f) }
    } else {
        f
    }
}

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
        let kwds: Vec<(Wtf8Buf, PyObjectRef)> = if unsafe { pyre_object::is_dict(kwargs) } {
            unsafe { pyre_object::w_dict_str_entries_wtf8(kwargs) }
        } else {
            Vec::new()
        };
        let frame = {
            let stored = build_class_exec_ctx();
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
        let new_fn = unsafe { unwrap_static_new(new_fn) };
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

/// Resolve `__mro_entries__` for every base that is not a type.
///
/// compiling.py `_update_bases` — for each base that is not a `type`, look up
/// `__mro_entries__` (getattr, not lookup) and, if present, call it with the
/// original bases tuple and splice the returned tuple in place.  Returns the
/// resolved bases and whether any substitution happened.
fn update_bases(
    base_args: &[PyObjectRef],
    w_orig_bases: PyObjectRef,
) -> Result<(Vec<PyObjectRef>, bool), crate::PyError> {
    let mut new_bases: Option<Vec<PyObjectRef>> = None;
    for (i, &w_base) in base_args.iter().enumerate() {
        if unsafe { pyre_object::is_type(w_base) } {
            if let Some(nb) = new_bases.as_mut() {
                nb.push(w_base);
            }
            continue;
        }
        match crate::baseobjspace::getattr_str(w_base, "__mro_entries__") {
            Err(e) if e.kind == crate::PyErrorKind::AttributeError => {
                if let Some(nb) = new_bases.as_mut() {
                    nb.push(w_base);
                }
            }
            Err(e) => return Err(e),
            Ok(w_meth) => {
                let w_new_base = crate::call_function(w_meth, &[w_orig_bases]);
                if w_new_base.is_null() {
                    if let Some(err) = take_call_error() {
                        return Err(err);
                    }
                    return Err(crate::PyError::type_error(
                        "__mro_entries__ must return a tuple",
                    ));
                }
                if !unsafe { pyre_object::is_tuple(w_new_base) } {
                    return Err(crate::PyError::type_error(
                        "__mro_entries__ must return a tuple",
                    ));
                }
                if new_bases.is_none() {
                    new_bases = Some(base_args[..i].to_vec());
                }
                let nb = new_bases.as_mut().unwrap();
                let n = unsafe { pyre_object::w_tuple_len(w_new_base) };
                for j in 0..n {
                    if let Some(item) =
                        unsafe { pyre_object::w_tuple_getitem(w_new_base, j as i64) }
                    {
                        nb.push(item);
                    }
                }
            }
        }
    }
    match new_bases {
        None => Ok((base_args.to_vec(), false)),
        Some(nb) => Ok((nb, true)),
    }
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
                pyre_object::w_dict_lookup(last, pyre_object::w_str_new("__pyre_kw__"))
                    .is_some_and(pyre_object::kw_marker::is_kw_marker_sentinel)
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
                        let key = pyre_object::w_str_get_wtf8(k).as_str();
                        if key != Ok("metaclass") && key != Ok("__pyre_kw__") {
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
    // compiling.py:166-167 — resolve __mro_entries__ before metaclass
    // inference; record the original bases for __orig_bases__ when changed.
    let w_orig_bases = pyre_object::w_tuple_new(base_args.to_vec());
    let (resolved_bases, bases_changed) = update_bases(base_args, w_orig_bases)?;
    let bases_tuple = pyre_object::w_tuple_new(resolved_bases);
    let w_orig_bases = if bases_changed {
        Some(w_orig_bases)
    } else {
        None
    };

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

    build_class_inner(
        body_fn,
        name,
        bases_tuple,
        w_metaclass,
        extra_kwargs,
        w_orig_bases,
    )
}

fn build_class_inner(
    body_fn: PyObjectRef,
    name: &str,
    bases: PyObjectRef,
    w_metaclass: Option<PyObjectRef>,
    extra_kwargs: Option<PyObjectRef>,
    w_orig_bases: Option<PyObjectRef>,
) -> PyResult {
    let w_code = unsafe { crate::getcode(body_fn) };
    let w_globals = unsafe { function_get_globals_obj(body_fn) };
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
                let prepare_kwds: Vec<(Wtf8Buf, PyObjectRef)> = match extra_kwargs {
                    Some(kw) if unsafe { pyre_object::is_dict(kw) } => unsafe {
                        pyre_object::w_dict_str_entries_wtf8(kw)
                    },
                    _ => Vec::new(),
                };
                let prepare_frame = {
                    let stored = build_class_exec_ctx();
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
    // their mapdict instance storage, not in W_DictObject.entries. We
    // handle both cases.
    // `w_dict_items` dispatches through `is_module_dict`, so the
    // rare `__prepare__` returning a W_ModuleDictObject still walks
    // correctly.  Both branches share the same shape; collapse them
    // around the dispatching surface.
    let _class_ns_root = pyre_object::gc_roots::push_roots();
    let w_namespace_root = w_namespace.map(|w_namespace| {
        let root = pyre_object::gc_roots::shadow_stack_len();
        pyre_object::gc_roots::pin_root(w_namespace);
        root
    });
    let class_ns = pyre_object::w_dict_new();
    let class_ns_root = pyre_object::gc_roots::shadow_stack_len();
    pyre_object::gc_roots::pin_root(class_ns);
    if let Some(w_namespace_root) = w_namespace_root {
        let w_prepared_dict = pyre_object::gc_roots::shadow_stack_get(w_namespace_root);
        if unsafe { pyre_object::is_dict(w_prepared_dict) } {
            let keys: Vec<Wtf8Buf> = unsafe {
                pyre_object::w_dict_str_entries_wtf8(w_prepared_dict)
                    .into_iter()
                    .map(|(key, _)| key)
                    .collect()
            };
            for key in keys {
                let w_prepared_dict = pyre_object::gc_roots::shadow_stack_get(w_namespace_root);
                let Some(value) =
                    (unsafe { pyre_object::w_dict_getitem_wtf8(w_prepared_dict, &key) })
                else {
                    continue;
                };
                if value.is_null() {
                    continue;
                }
                let class_ns = pyre_object::gc_roots::shadow_stack_get(class_ns_root);
                unsafe { pyre_object::w_dict_setitem_wtf8_no_proxy(class_ns, &key, value) };
            }
        }
        // dict subclass instance (e.g. EnumDict): backing dict via __dict_data__
        let w_prepared_dict = pyre_object::gc_roots::shadow_stack_get(w_namespace_root);
        if unsafe { pyre_object::is_instance(w_prepared_dict) } {
            let backing = crate::type_methods::resolve_dict_backing(w_prepared_dict);
            if !backing.is_null() && unsafe { pyre_object::is_dict(backing) } {
                let backing_root = pyre_object::gc_roots::shadow_stack_len();
                pyre_object::gc_roots::pin_root(backing);
                let keys: Vec<Wtf8Buf> = unsafe {
                    pyre_object::w_dict_str_entries_wtf8(backing)
                        .into_iter()
                        .map(|(key, _)| key)
                        .collect()
                };
                for key in keys {
                    let backing = pyre_object::gc_roots::shadow_stack_get(backing_root);
                    let Some(value) = (unsafe { pyre_object::w_dict_getitem_wtf8(backing, &key) })
                    else {
                        continue;
                    };
                    if value.is_null() {
                        continue;
                    }
                    let class_ns = pyre_object::gc_roots::shadow_stack_get(class_ns_root);
                    unsafe { pyre_object::w_dict_setitem_wtf8_no_proxy(class_ns, &key, value) };
                }
            }
        }
    }

    // w_namespace: if __prepare__ returned a custom dict, we'll replay
    // class body stores into it after execution. This lets EnumDict etc.
    // track member definitions via __setitem__.

    let stored = build_class_exec_ctx();
    let exec_ctx = if stored.is_null() {
        std::ptr::null::<crate::PyExecutionContext>()
    } else {
        stored
    };

    // Create frame with class_locals set AND closure from enclosing scope.
    // PyPy: executes class body with w_locals = fresh dict, w_globals = module globals,
    // and the closure tuple is passed through for LOAD_DEREF access.
    // Debug: dump code object for __class__ cell investigation. Reads the real
    // process env + writes real stderr, so keep it out of the sandbox build.
    #[cfg(not(feature = "sandbox"))]
    {
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
    }

    // When `__prepare__` returned a custom mapping (a dict subclass such as
    // enum._EnumDict, or any non-dict mapping), the class body must execute
    // against it directly so its `__setitem__`/`__getitem__` fire mid-body —
    // e.g. `WHITE = RED | GREEN` reads the values `_EnumDict.__setitem__`
    // resolved on assignment, not the stale `auto()` sentinels.  Upstream
    // `compiling.py:207-209` runs `frame.setdictscope(w_namespace)`
    // unconditionally; route the frame's name binding through the mapping
    // via setdictscope.  An absent or plain-dict namespace keeps the
    // plain-dict fast path (plain-dict stores have no observable side
    // effects, and the metaclass replay below restores its final contents).
    let mapping_namespace = w_namespace_root
        .map(pyre_object::gc_roots::shadow_stack_get)
        .filter(|&w| unsafe { !pyre_object::is_dict(w) });

    let mut frame =
        crate::pyframe::FrameBox::new(PyFrame::try_new_for_call_with_closure_and_globals_obj(
            w_code,
            &[],
            w_globals,
            exec_ctx,
            closure,
            crate::pyframe::FrameLocalsArrayAllocation::OldGenGc,
        )?);
    // The class body executes against a namespace OBJECT (setdictscope)
    // so STORE_NAME / LOAD_NAME route through the object form, not the raw
    // `*mut DictStorage` w_locals.  A custom non-dict `__prepare__` mapping is
    // used directly (already rooted by the caller); otherwise a fresh dict,
    // seeded from class_ns's pre-body entries and pinned as a GC root for the
    // run.  class_ns is rebuilt from the object after the body for the
    // downstream type construction.
    let _ns_root = pyre_object::gc_roots::push_roots();
    let (body_ns, body_ns_root): (PyObjectRef, Option<usize>) = match mapping_namespace {
        Some(w_ns) => (w_ns, None),
        None => {
            let w_ns = pyre_object::w_dict_new();
            let w_ns_root = pyre_object::gc_roots::shadow_stack_len();
            pyre_object::gc_roots::pin_root(w_ns);
            let class_ns = pyre_object::gc_roots::shadow_stack_get(class_ns_root);
            let keys: Vec<Wtf8Buf> = unsafe {
                pyre_object::w_dict_str_entries_wtf8(class_ns)
                    .into_iter()
                    .map(|(key, _)| key)
                    .collect()
            };
            for key in keys {
                let class_ns = pyre_object::gc_roots::shadow_stack_get(class_ns_root);
                let Some(value) = (unsafe { pyre_object::w_dict_getitem_wtf8(class_ns, &key) })
                else {
                    continue;
                };
                if value.is_null() {
                    continue;
                }
                let w_ns = pyre_object::gc_roots::shadow_stack_get(w_ns_root);
                match key.as_str() {
                    Ok(s) => unsafe {
                        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(w_ns, s, value)
                    },
                    Err(_) => unsafe {
                        pyre_object::dictmultiobject::w_dict_setitem_wtf8_no_proxy(
                            w_ns, &key, value,
                        )
                    },
                }
            }
            (
                pyre_object::gc_roots::shadow_stack_get(w_ns_root),
                Some(w_ns_root),
            )
        }
    };
    frame.setdictscope(body_ns)?;

    // Route the class body through the JIT portal (like the exec / import
    // run-sites) so a hot class-level loop can warm and compile.  The body's
    // NEWLOCALS bindings land in `body_ns`; that object's values are rooted by
    // the `debugdata.w_locals` walk in `walk_pyframe_roots`.
    frame.run_with_jit()?;

    // A minor collection during the body (a hot class-level loop compiles
    // and runs through the JIT portal) can promote the namespace young ->
    // old, relocating it.  The frame's `debugdata.w_locals` slot and the
    // shadow-stack pin are forwarded, keeping the object alive at its new
    // address, but this stack-local `body_ns` copy is not — it is left
    // pointing at the freed nursery slot.  Re-read the forwarded object
    // from the frame before any downstream use; a stale read would
    // dereference reclaimed memory (`resolve_dict_backing` / `w_dict_items`
    // below, then `__set_name__`).
    let body_ns = frame.get_w_locals();
    let mapping_namespace = mapping_namespace.map(|_| body_ns);

    // The body wrote through `body_ns`; mirror its final contents into
    // class_ns for the downstream type construction (classcell capture,
    // create_all_slots, __set_name__), which read class_ns.
    {
        let w_ns = body_ns_root
            .map(pyre_object::gc_roots::shadow_stack_get)
            .unwrap_or(body_ns);
        // Rebuild from the final contents so names `del`eted from the
        // namespace during body execution don't survive in class_ns.
        let class_ns = pyre_object::gc_roots::shadow_stack_get(class_ns_root);
        unsafe { pyre_object::w_dict_clear(class_ns) };
        let backing = crate::type_methods::resolve_dict_backing(w_ns);
        if !backing.is_null() && unsafe { pyre_object::is_dict(backing) } {
            // Dict subclass: read final entries off the backing dict.
            let backing_root = pyre_object::gc_roots::shadow_stack_len();
            pyre_object::gc_roots::pin_root(backing);
            let keys: Vec<Wtf8Buf> = unsafe {
                pyre_object::w_dict_str_entries_wtf8(backing)
                    .into_iter()
                    .map(|(key, _)| key)
                    .collect()
            };
            for key in keys {
                let backing = pyre_object::gc_roots::shadow_stack_get(backing_root);
                let Some(value) = (unsafe { pyre_object::w_dict_getitem_wtf8(backing, &key) })
                else {
                    continue;
                };
                if value.is_null() {
                    continue;
                }
                let class_ns = pyre_object::gc_roots::shadow_stack_get(class_ns_root);
                unsafe { pyre_object::w_dict_setitem_wtf8_no_proxy(class_ns, &key, value) };
            }
        } else if w_metaclass.is_some() {
            // A custom metaclass receives the raw mapping unchanged (passed
            // below) and owns its enumeration — `type.__new__` runs
            // `PyMapping_Keys` itself.  The mapping must therefore not be
            // walked here; only `__classcell__` is consumed locally, for the
            // post-metaclass cell validation below, so lift just that one key
            // via `space.getitem` rather than calling the mapping's `keys()`.
            let w_cellkey = pyre_object::w_str_new("__classcell__");
            match crate::baseobjspace::getitem(w_ns, w_cellkey) {
                Ok(value) if !value.is_null() => {
                    let class_ns = pyre_object::gc_roots::shadow_stack_get(class_ns_root);
                    unsafe {
                        pyre_object::w_dict_setitem_str_no_proxy(class_ns, "__classcell__", value)
                    };
                }
                Ok(_) => {}
                Err(e) if e.kind == crate::PyErrorKind::KeyError => {}
                Err(e) => return Err(e),
            }
        } else {
            // Arbitrary non-dict mapping with the default metaclass: this path
            // builds the type directly from `class_ns`, so materialize the
            // whole namespace via the mapping protocol's `keys()` (the
            // `PyMapping_Keys` path `type.__new__` takes for a non-dict
            // namespace) and read each value back via `space.getitem` so
            // `__getitem__` overrides apply.  `keys()` rather than `iter()`
            // keeps a mapping without `__iter__` working.
            let keys_method = crate::baseobjspace::getattr_str(w_ns, "keys")?;
            let keys_obj = crate::call::call_function_impl_result(keys_method, &[])?;
            let keys = crate::builtins::collect_iterable(keys_obj)?;
            for key in keys {
                if !unsafe { pyre_object::is_str(key) } {
                    continue;
                }
                let value = crate::baseobjspace::getitem(w_ns, key)?;
                if !value.is_null() {
                    let class_ns = pyre_object::gc_roots::shadow_stack_get(class_ns_root);
                    unsafe {
                        pyre_object::w_dict_setitem_wtf8_no_proxy(
                            class_ns,
                            pyre_object::w_str_get_wtf8(key),
                            value,
                        )
                    };
                }
            }
        }
    }

    // compiling.py:211-212 — when __mro_entries__ rewrote the bases, expose
    // the user-declared bases via __orig_bases__ in the class namespace.
    if let Some(w_orig_bases) = w_orig_bases {
        let class_ns = pyre_object::gc_roots::shadow_stack_get(class_ns_root);
        unsafe {
            pyre_object::w_dict_setitem_str_no_proxy(class_ns, "__orig_bases__", w_orig_bases)
        };
        if let Some(w_ns) = mapping_namespace {
            crate::baseobjspace::setitem(
                w_ns,
                pyre_object::w_str_new("__orig_bases__"),
                w_orig_bases,
            )?;
        }
    }

    // type_new_classcell (typeobject.c) — capture the `__classcell__` cell
    // so its content can be set to the new class below.  The `__class__` /
    // `__classdict__` cells themselves never reach the namespace (the body
    // writes through `setdictscope`, and their cells stay empty during the
    // body), so a `__class__` key here is an explicit class-body assignment
    // (e.g. a `__class__` property) that must survive into the class dict.
    // `__classcell__` / `__classdictcell__` ARE real namespace entries the
    // class body stores explicitly: a custom metaclass observes them
    // (type.__new__ receives the full namespace) and `type.__new__`
    // (type_new_classcell) consumes them, so they are dropped per
    // construction path below rather than up front.
    let classcell = {
        let class_ns = pyre_object::gc_roots::shadow_stack_get(class_ns_root);
        unsafe { pyre_object::w_dict_getitem_str(class_ns, "__classcell__") }
    };

    // typeobject.c type_new: every class carries `__doc__` (None when the
    // body has no docstring) so instances inherit it through the type MRO.
    // The compiler only stores `__doc__` when a docstring is present.  Skip
    // the default when `__doc__` is a declared slot — a class variable would
    // collide with the member descriptor (typing._SpecialForm).
    {
        let class_ns = pyre_object::gc_roots::shadow_stack_get(class_ns_root);
        if unsafe { pyre_object::w_dict_getitem_str(class_ns, "__doc__") }.is_none() {
            let class_ns = pyre_object::gc_roots::shadow_stack_get(class_ns_root);
            let doc_is_slot =
                match unsafe { pyre_object::w_dict_getitem_str(class_ns, "__slots__") } {
                    Some(slots)
                        if unsafe {
                            pyre_object::is_str(slots)
                                || pyre_object::is_tuple(slots)
                                || pyre_object::is_list(slots)
                        } =>
                    {
                        collect_slot_names(slots)
                            .map(|names| names.iter().any(|n| n == "__doc__"))
                            .unwrap_or(false)
                    }
                    _ => false,
                };
            if !doc_is_slot {
                let class_ns = pyre_object::gc_roots::shadow_stack_get(class_ns_root);
                unsafe {
                    pyre_object::w_dict_setitem_str_no_proxy(
                        class_ns,
                        "__doc__",
                        pyre_object::w_none(),
                    )
                };
            }
        }
    }

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
    // A custom metaclass owns its bases until (and unless) it invokes
    // type.__new__; do not perform type's C3 validation before dispatch.
    if w_metaclass.is_none() {
        unsafe { crate::baseobjspace::validate_c3_mro(w_effective_bases)? };
    }
    // Create class via metaclass or default type()
    // PyPy: typeobject.py — metaclass(name, bases, dict_w) or type.__new__
    // Keep the default path's fresh managed namespace rooted until slot
    // creation, __set_name__, and classcell binding have all completed.
    let _dict_root = if w_metaclass.is_none() {
        Some(pyre_object::gc_roots::push_roots())
    } else {
        None
    };
    let w_type = if let Some(w_metaclass) = w_metaclass {
        let _metaclass_ns_root = pyre_object::gc_roots::push_roots();
        let mut w_namespace_dict_root = None;
        // Convert class namespace to a dict for metaclass call.
        // If __prepare__ returned a custom dict, replay stores into it
        // so that __setitem__ side effects (e.g. EnumDict tracking) fire.
        let mut w_namespace_dict = if let Some(w_namespace_root) = w_namespace_root {
            let w_prepared_dict = pyre_object::gc_roots::shadow_stack_get(w_namespace_root);
            // Replay class body stores into the prepared dict so __setitem__
            // side effects (EnumDict tracking) fire — but only on the legacy
            // path where the body wrote into class_ns.  When the body executed
            // directly against the mapping (setdictscope above) it
            // already holds every store; replaying would re-run __setitem__
            // and, for _EnumDict, reject the duplicate member keys.
            if mapping_namespace.is_none() {
                let class_ns = pyre_object::gc_roots::shadow_stack_get(class_ns_root);
                let keys: Vec<Wtf8Buf> = unsafe {
                    pyre_object::w_dict_str_entries_wtf8(class_ns)
                        .into_iter()
                        .map(|(key, _)| key)
                        .collect()
                };
                for key in keys {
                    let class_ns = pyre_object::gc_roots::shadow_stack_get(class_ns_root);
                    let Some(value) = (unsafe { pyre_object::w_dict_getitem_wtf8(class_ns, &key) })
                    else {
                        continue;
                    };
                    if value.is_null() {
                        continue;
                    }
                    let w_prepared_dict = pyre_object::gc_roots::shadow_stack_get(w_namespace_root);
                    // `w_prepared_dict` is an exact `dict` on this branch
                    // (`mapping_namespace.is_none()`), so no user code runs.
                    unsafe {
                        pyre_object::w_dict_setitem_wtf8_no_proxy(w_prepared_dict, &key, value)
                    };
                }
            }
            pyre_object::gc_roots::shadow_stack_get(w_namespace_root)
        } else {
            let d = pyre_object::w_dict_new();
            let d_root = pyre_object::gc_roots::shadow_stack_len();
            pyre_object::gc_roots::pin_root(d);
            w_namespace_dict_root = Some(d_root);
            let class_ns = pyre_object::gc_roots::shadow_stack_get(class_ns_root);
            let keys: Vec<String> = unsafe {
                pyre_object::w_dict_str_entries(class_ns)
                    .into_iter()
                    .map(|(key, _)| key)
                    .collect()
            };
            for key in keys {
                let class_ns = pyre_object::gc_roots::shadow_stack_get(class_ns_root);
                let Some(value) = (unsafe { pyre_object::w_dict_getitem_str(class_ns, &key) })
                else {
                    continue;
                };
                if value.is_null() {
                    continue;
                }
                let d = pyre_object::gc_roots::shadow_stack_get(d_root);
                unsafe { pyre_object::w_dict_setitem_str_no_proxy(d, &key, value) };
            }
            pyre_object::gc_roots::shadow_stack_get(d_root)
        };
        // Call metaclass(name, bases, namespace, **kwds)
        // Pass the ORIGINAL bases (not w_effective_bases) — the metaclass
        // expects the user-declared bases. Default (object,) is added by
        // type.__new__ internally if needed.
        let name_obj = pyre_object::w_str_new(name);
        if let Some(root) = w_namespace_dict_root {
            w_namespace_dict = pyre_object::gc_roots::shadow_stack_get(root);
        }
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
        let class_ns = pyre_object::gc_roots::shadow_stack_get(class_ns_root);
        unsafe { pyre_object::w_dict_delitem_str_no_proxy(class_ns, "__classcell__") };
        let class_ns = pyre_object::gc_roots::shadow_stack_get(class_ns_root);
        unsafe { pyre_object::w_dict_delitem_str_no_proxy(class_ns, "__classdictcell__") };
        let class_ns = pyre_object::gc_roots::shadow_stack_get(class_ns_root);
        crate::builtins::type_new_set_hash_if_eq(class_ns);
        let class_ns = pyre_object::gc_roots::shadow_stack_get(class_ns_root);
        crate::builtins::type_new_wrap_special_methods(class_ns);
        let dict_root = pyre_object::gc_roots::shadow_stack_len();
        let dict_obj = pyre_object::w_dict_new();
        pyre_object::gc_roots::pin_root(dict_obj);
        let class_ns = pyre_object::gc_roots::shadow_stack_get(class_ns_root);
        let keys: Vec<Wtf8Buf> = unsafe {
            pyre_object::w_dict_str_entries_wtf8(class_ns)
                .into_iter()
                .map(|(key, _)| key)
                .collect()
        };
        for key in keys {
            let class_ns = pyre_object::gc_roots::shadow_stack_get(class_ns_root);
            let Some(value) = (unsafe { pyre_object::w_dict_getitem_wtf8(class_ns, &key) }) else {
                continue;
            };
            if value.is_null() {
                continue;
            }
            let dict_obj = pyre_object::gc_roots::shadow_stack_get(dict_root);
            match key.as_str() {
                Ok(s) => unsafe { pyre_object::w_dict_setitem_str_no_proxy(dict_obj, s, value) },
                Err(_) => unsafe {
                    pyre_object::w_dict_setitem_wtf8_no_proxy(dict_obj, &key, value)
                },
            }
        }
        let dict_obj = pyre_object::gc_roots::shadow_stack_get(dict_root);
        let w = pyre_object::w_type_new(name, w_effective_bases, dict_obj as *mut u8);
        // typeobject.py:1143-1204 create_all_slots parity.
        unsafe { create_all_slots(w, w_effective_bases)? };
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
            let dict_obj = pyre_object::gc_roots::shadow_stack_get(dict_root);
            let entries = unsafe { pyre_object::w_dict_items(dict_obj) };
            for (w_name, value) in entries {
                if !value.is_null() && unsafe { pyre_object::is_str(w_name) } {
                    unsafe { crate::baseobjspace::set_name(w, w_name, value) }?;
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
                    let class_str = unsafe { crate::py_str(w_type) }?;
                    return Err(PyError::runtime_error(format!(
                        "__class__ not set defining {name} as {class_str}. \
                         Was __classcell__ propagated to type.__new__?"
                    )));
                }
                if !std::ptr::eq(cell_value, w_type) {
                    let cell_str = unsafe { crate::py_str(cell_value) }?;
                    let class_str = unsafe { crate::py_str(w_type) }?;
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
        for (k, v) in kw_items {
            pyre_object::w_dict_store(kw_dict, *k, *v);
        }
        // Marker stored last so a user keyword named `__pyre_kw__` cannot
        // overwrite the sentinel detection compares by identity.
        pyre_object::w_dict_store(
            kw_dict,
            pyre_object::w_str_new("__pyre_kw__"),
            pyre_object::kw_marker::w_kw_marker_sentinel(),
        );
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
    let w_super = pyre_object::descriptor::w_super_new(w_type, w_type);
    let w_func = crate::baseobjspace::getattr_str(w_super, "__init_subclass__")?;
    // `__args__.replace_arguments([])` — keywords only, no positionals.
    let kwds: Vec<(Wtf8Buf, PyObjectRef)> = init_subclass_kwargs
        .iter()
        .filter(|(k, _)| unsafe { pyre_object::is_str(*k) })
        .map(|(k, v)| (unsafe { pyre_object::w_str_get_wtf8(*k) }.to_owned(), *v))
        .collect();
    let frame = {
        let stored = build_class_exec_ctx();
        let stored = if stored.is_null() {
            take_last_exec_ctx()
        } else {
            stored
        };
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

/// Read the __build_class__ execution context.
///
/// `dont_look_inside`: the `BUILD_CLASS_EXEC_CTX` thread-local `.with` read
/// has no extractable graph (front::mir const-folds the `ThreadLocal` global
/// to None), so the call stays a residual read via the registered fnaddr
/// (`@dont_look_inside`, `rlib/jit.py:139`), the `take_last_exec_ctx` twin.
#[majit_macros::dont_look_inside]
pub fn build_class_exec_ctx() -> *const crate::PyExecutionContext {
    BUILD_CLASS_EXEC_CTX.with(|c| c.get())
}

// ── Type calling (instance creation) ─────────────────────────────────
// PyPy equivalent: typeobject.py descr_call → __new__ + __init__

fn type_descr_call_with_mode(
    frame: &mut PyFrame,
    w_type: PyObjectRef,
    args: &[PyObjectRef],
    mode: CallMode,
) -> PyResult {
    check_type_instantiable(w_type)?;
    // Step 1: Look up __new__ via type MRO → allocate instance.
    // PyPy: typeobject.py descr_call → `w_newtype, w_newdescr =
    // self.lookup_where('__new__')`; a missing descriptor (the pathological
    // mro-without-object case) raises, otherwise the descriptor is bound via
    // `space.get(w_newdescr, space.w_None, w_type=self)` and called with
    // w_type as the first arg.
    let Some(new_descr) = (unsafe { crate::baseobjspace::lookup_in_type(w_type, "__new__") })
    else {
        // typeobject.py:715 — `raise oefmt(space.w_TypeError,
        // "cannot create '%N' instances", self)`.
        let name = unsafe { pyre_object::w_type_get_name(w_type) };
        return Err(crate::PyError::type_error(format!(
            "cannot create '{name}' instances"
        )));
    };
    // typeobject.py:726 — `w_newfunc = space.get(w_newdescr, space.w_None,
    // w_type=self)`.  A descriptor with no __get__ (`get` → None) is its own
    // bound value, matching `space.get`'s `if w_get is None: return w_descr`.
    let new_fn = unsafe { crate::baseobjspace::get(new_descr, pyre_object::PY_NULL, w_type)? }
        .unwrap_or(new_descr);
    // typeobject.py:731 — `space.call_obj_args(w_newfunc, self, __args__)`.
    let mut new_args = Vec::with_capacity(1 + args.len());
    new_args.push(w_type);
    new_args.extend_from_slice(args);
    let instance = call_callable_with_mode(frame, new_fn, &new_args, mode)?;

    // Step 2: __init__ — only if __new__ returned an instance of w_type.
    // PyPy: descr_call — skips __init__ when __new__ returns a foreign type.
    if let Some(w_insttype) = type_call_init_type(instance, w_type)
        && !type_call_type_x_shortcut(w_type, args.len(), true)
    {
        if let Some(init_fn) =
            unsafe { crate::baseobjspace::lookup_in_type(w_insttype, "__init__") }
        {
            let mut init_args = Vec::with_capacity(1 + args.len());
            init_args.push(instance);
            init_args.extend_from_slice(args);
            let init_result = call_callable_with_mode(frame, init_fn, &init_args, mode)?;
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
///   typeobject.py:1406 w_self.hasuserdel |= w_base.hasuserdel
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
                    if pyre_object::w_type_get_hasuserdel(base) {
                        pyre_object::w_type_set_hasuserdel(w_type, true);
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
    w_bases: pyre_object::PyObjectRef,
) -> Result<(), crate::PyError> {
    unsafe {
        use pyre_object::typeobject::{Layout, leak_layout};

        // typeobject.py:1245: w_bestbase = check_and_find_best_base(space, bases_w)
        let w_bestbase = check_and_find_best_base(w_bases)?;

        // typeobject.py:1507-1508: inherit flag_map_or_seq from bases
        pyre_object::typeobject::inherit_flag_map_or_seq(w_type, w_bases);

        // typeobject.c type_new: a class body carrying `__abc_tpflags__`
        // (`collections.abc` Mapping = `1<<6`, Sequence = `1<<5`) folds its
        // COLLECTION_FLAGS into the structural-match marker, so subclasses of
        // `abc.Mapping` / `abc.Sequence` match `case {..}` / `case [..]`. The
        // bit lives only in the defining body's namespace, so subclasses pick
        // it up through `inherit_flag_map_or_seq` above rather than re-reading.
        if let Some(w_flags) = crate::type_dict_lookup(w_type, "__abc_tpflags__") {
            if pyre_object::is_int(w_flags) {
                let flags = pyre_object::w_int_get_value(w_flags);
                let collection_flags = flags & ((1 << 6) | (1 << 5));
                if collection_flags == ((1 << 6) | (1 << 5)) {
                    return Err(crate::PyError::type_error(
                        "__abc_tpflags__ cannot be both Py_TPFLAGS_SEQUENCE and Py_TPFLAGS_MAPPING",
                    ));
                }
                if flags & (1 << 6) != 0 {
                    pyre_object::typeobject::w_type_set_flag_map_or_seq(w_type, b'M');
                } else if flags & (1 << 5) != 0 {
                    pyre_object::typeobject::w_type_set_flag_map_or_seq(w_type, b'S');
                }
            }
        }

        // typeobject.py:1510: copy_flags_from_bases — inherit hasdict/weakrefable/hasuserdel
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
        if let Some(w_slots) = crate::type_dict_lookup(w_type, "__slots__") {
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
                if crate::type_dict_contains(w_type, mangled.as_str()) {
                    // typeobject.py:1219-1220: name conflict → skip this slot
                    newslotnames.remove(i);
                } else {
                    // typeobject.py:1216-1217: create_slot
                    newslotnames[i] = mangled.clone();
                    if crate::type_dict_has_storage(w_type) {
                        let member = pyre_object::w_member_new(slot_index, mangled.clone(), w_type);
                        crate::type_dict_store(w_type, &mangled, member);
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
        if crate::type_dict_contains(w_type, "__del__") {
            pyre_object::w_type_set_hasuserdel(w_type, true);
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
                // typedef.py:51-53: inherit typedef.hasdict along the base
                // layout chain (terminates at object's Layout = false).
                typedef_hasdict: if base_layout.is_null() {
                    false
                } else {
                    (*base_layout).typedef_hasdict
                },
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
            if !crate::type_dict_contains(w_type, "__dict__") {
                crate::type_dict_store(w_type, "__dict__", descr);
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
            if !crate::type_dict_contains(w_type, "__weakref__") {
                crate::type_dict_store(w_type, "__weakref__", descr);
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
                "type '{}' is not an acceptable base type",
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
