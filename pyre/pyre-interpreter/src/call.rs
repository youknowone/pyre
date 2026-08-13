//! Function call dispatch — pure interpreter, no JIT dependencies.
//!
//! JIT-specific call infrastructure (force/bridge callbacks, callee frame
//! creation helpers, frame pool) lives in pyre-jit/src/call_jit.rs.

use std::cell::{Cell, RefCell};
use std::sync::OnceLock;

use rustpython_wtf8::{Wtf8, Wtf8Buf};

use crate::runtime_ops::{CallableKind, classify_callable};
use crate::{
    PyError, PyResult, builtin_code_get, function_get_closure, function_get_globals_obj,
    function_get_name, function_get_name_obj, function_get_qualname, function_get_qualname_obj,
};

/// `function.py:131/214/231 new_frame.run(self.name, self.qualname)`.
/// Generator creation must capture the function's writable metadata rather
/// than lazily rereading the immutable code object's names.
pub(crate) fn frame_into_generator_for_function(
    frame: crate::pyframe::FrameBox,
    function: PyObjectRef,
) -> PyResult {
    let _roots = pyre_object::gc_roots::push_roots();
    let function_slot = pyre_object::gc_roots::shadow_stack_len();
    pyre_object::gc_roots::pin_root(function);
    let name_slot = pyre_object::gc_roots::shadow_stack_len();
    pyre_object::gc_roots::pin_root(unsafe { function_get_name_obj(function) });
    let function = pyre_object::gc_roots::shadow_stack_get(function_slot);
    let qualname_slot = pyre_object::gc_roots::shadow_stack_len();
    pyre_object::gc_roots::pin_root(unsafe { function_get_qualname_obj(function) });
    frame.into_generator_named(
        Some(pyre_object::gc_roots::shadow_stack_get(name_slot)),
        Some(pyre_object::gc_roots::shadow_stack_get(qualname_slot)),
    )
}

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
        unsafe { walk_pending_call_error_area(slot as *const _ as *const (), visitor) };
    });
}

pub(crate) fn capture_pending_call_error_area() -> *const () {
    PENDING_CALL_ERROR.with(|slot| slot as *const _ as *const ())
}

/// Walk the deferred call error belonging to one stopped mutator.
///
/// # Safety
/// `data` must come from [`capture_pending_call_error_area`], and the owning
/// mutator must be quiesced.
pub(crate) unsafe fn walk_pending_call_error_area(
    data: *const (),
    visitor: &mut dyn FnMut(&mut majit_ir::GcRef),
) {
    let slot = unsafe { &*(data as *const RefCell<Option<PyError>>) };
    // SAFETY: the owning mutator is stopped, so nothing can borrow or mutate
    // the RefCell while the collector forwards the PyError's object slots.
    let opt = unsafe { &mut *slot.as_ptr() };
    if let Some(err) = opt.as_mut() {
        err.walk_gc_refs(visitor);
    }
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
        // Cached: this sits on the per-call path and is also reachable from
        // compiled code through the fnaddr registry, so an uncached read would
        // pay a `getenv` plus a `String` allocation on every call.
        static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ENABLED.get_or_init(|| std::env::var_os("PYRE_DEBUG_CALL").is_some())
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
type ThreadEntryFn = fn();
static THREAD_ENTRY_HOOK: OnceLock<ThreadEntryFn> = OnceLock::new();

type DepthBumpFn = fn() -> Option<Box<dyn std::any::Any>>;
static DEPTH_BUMP_OVERRIDE: OnceLock<DepthBumpFn> = OnceLock::new();

thread_local! {
    /// Python recursion depth — the number of user Python frames currently
    /// executing bytecode on this thread.  Bumped once at every `eval_loop` /
    /// `eval_loop_jit` entry and dropped when that activation returns, so the
    /// module-level frame, an `exec`ed body and a resumed generator each cost
    /// one unit exactly like a called function does.  `stack_check()` compares
    /// it against `sys.getrecursionlimit()`.
    static PY_RECURSION_DEPTH: Cell<u32> = const { Cell::new(0) };

    /// The innermost frame whose activation has already spent its
    /// [`PY_RECURSION_DEPTH`] unit.  A frame is executed through nested entry
    /// points — the JIT wrapper may run it as compiled code, hand it to the
    /// JIT eval loop, or decline and re-enter the plain evaluator for the very
    /// same frame — and only the outermost of those pays.  Any frame reached
    /// from here is a different, simultaneously-live frame, so its address
    /// cannot collide with the one recorded.
    static ACCOUNTED_ACTIVATION: Cell<usize> = const { Cell::new(0) };

    /// Monotonic count of Python frame eval-loop entries — bumped once per
    /// `eval_loop` / `eval_loop_jit` entry (every user-level bytecode frame
    /// that begins running), NEVER decremented.  Unlike [`PY_RECURSION_DEPTH`] (net
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

/// Number of user Python frames currently executing bytecode on this thread.
/// Used by pyre-jit for JIT_CALL_DEPTH parity.
///
/// The counter is runtime-mutable execution-context state.  Like
/// [`frame_entry_count`], its TLS read has no source-translatable graph and
/// must remain a residual read rather than exposing `LocalKey::with` to the
/// annotator.
#[majit_macros::dont_look_inside]
pub fn py_recursion_depth() -> u32 {
    PY_RECURSION_DEPTH.with(|d| d.get())
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
/// into it (`@dont_look_inside`, the `gc_interp::enabled` sibling). A `()` return has
/// no discriminant to erase and it cannot raise.
#[majit_macros::dont_look_inside]
pub fn bump_frame_entry_count() {
    FRAME_ENTRY_COUNT.with(|c| c.set(c.get().wrapping_add(1)));
}

/// Spend one unit of the recursion budget on `frame`'s activation, returning a
/// guard that gives it back when the activation finishes.  Re-entering for a
/// frame that is already accounted spends nothing, so a frame costs exactly one
/// unit whether it runs as compiled code, through the JIT eval loop, or in the
/// plain evaluator.  `pyframe.py:360` (`execute_frame.insert_stack_check_here`)
/// puts the matching stack check at the same seam.
#[inline]
pub fn enter_recursive_frame(frame: *const PyFrame) -> RecursionDepthGuard {
    let key = frame as usize;
    if ACCOUNTED_ACTIVATION.with(|c| c.get()) == key {
        return RecursionDepthGuard {
            prev: key,
            spent: false,
        };
    }
    PY_RECURSION_DEPTH.with(|d| d.set(d.get() + 1));
    RecursionDepthGuard {
        prev: ACCOUNTED_ACTIVATION.with(|c| c.replace(key)),
        spent: true,
    }
}

/// Spend one unit of the recursion budget on a dispatch level that pushes no
/// Python frame, returning the same guard [`enter_recursive_frame`] returns.
///
/// The self-referential `A.__call__ = A()` chain recurses through
/// `user_call_slot` natively and never reaches a frame activation, so there is
/// no activation to key on: the unit is spent unconditionally and
/// `ACCOUNTED_ACTIVATION` is carried through unchanged, leaving the next real
/// activation to account for itself.
#[inline]
pub fn enter_native_dispatch() -> RecursionDepthGuard {
    PY_RECURSION_DEPTH.with(|d| d.set(d.get() + 1));
    RecursionDepthGuard {
        prev: ACCOUNTED_ACTIVATION.with(|c| c.get()),
        spent: true,
    }
}

/// RAII guard that releases the [`PY_RECURSION_DEPTH`] unit on drop.
pub struct RecursionDepthGuard {
    prev: usize,
    spent: bool,
}

impl Drop for RecursionDepthGuard {
    #[inline]
    fn drop(&mut self) {
        if self.spent {
            PY_RECURSION_DEPTH.with(|d| d.set(d.get().saturating_sub(1)));
            ACCOUNTED_ACTIVATION.with(|c| c.set(self.prev));
        }
    }
}

/// Register the JIT-aware eval function. Called by pyre-jit at startup.
pub fn register_eval_override(f: EvalFn) {
    let _ = EVAL_OVERRIDE.set(f);
}

/// Install the runtime hook for entering a new Python mutator thread.
///
/// `pyre-interpreter` cannot depend on `pyre-jit`, so the JIT registers this
/// reverse hook after it has installed the process-global collector.  Every
/// newly-created Python OS thread invokes it before allocating or evaluating.
pub fn register_thread_entry_hook(f: ThreadEntryFn) {
    let _ = THREAD_ENTRY_HOOK.set(f);
}

pub fn enter_runtime_thread() {
    if let Some(f) = THREAD_ENTRY_HOOK.get() {
        f();
    }
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

// `rlib/jit.py:842 set_user_param` — the positional-string form
// (`"name=value,…"`, `"off"`, `"default"`) that `pypyjit.set_param(str)`
// routes through. The JIT owns the authoritative parser, so this forwards the
// whole string and returns `Err(())` on a malformed string (rlib/jit.py:853).
type SetJitParamStringFn = fn(text: &str) -> Result<(), ()>;
static SET_JIT_PARAM_STRING_HOOK: OnceLock<SetJitParamStringFn> = OnceLock::new();

/// Register the hook that applies a JIT-parameter string via the JIT
/// runtime's `set_user_param`. Called by pyre-jit at startup.
pub fn register_set_jit_param_string_hook(f: SetJitParamStringFn) {
    let _ = SET_JIT_PARAM_STRING_HOOK.set(f);
}

/// Apply a JIT-parameter string. `Ok(())` when the hook is absent (JIT-disabled
/// build) so a `pypyjit.set_param("…")` call is inert rather than an error
/// there; `Err(())` only on a malformed string once the JIT is present.
pub fn set_jit_param_string(text: &str) -> Result<(), ()> {
    match SET_JIT_PARAM_STRING_HOOK.get() {
        Some(hook) => hook(text),
        None => Ok(()),
    }
}

/// jd1 (`unpackiterable_driver`) merge-point hook. pyre-interpreter cannot
/// import pyre-jit (its upper crate), so the JIT registers this at boot and the
/// `unpackiterable_driver.jit_merge_point` marker calls through it. Mirrors the
/// `SET_JIT_PARAM_HOOK` / `EVAL_OVERRIDE` inversion pattern.
/// `greenkey` is the merge-point green; `w_iterator` and `items` are the two
/// `reds='auto'` values the JIT walk backs its InputArgs with.
type UnpackMergeFn = fn(greenkey: PyObjectRef, w_iterator: PyObjectRef, items: PyObjectRef);
static UNPACK_MERGE_HOOK: OnceLock<UnpackMergeFn> = OnceLock::new();

pub fn register_unpack_merge_hook(f: UnpackMergeFn) {
    let _ = UNPACK_MERGE_HOOK.set(f);
}

/// Called from `UnpackIterableJitDriver::jit_merge_point`. No-op until the JIT
/// installs the hook.
#[inline]
pub fn unpack_merge_point(greenkey: PyObjectRef, w_iterator: PyObjectRef, items: PyObjectRef) {
    if let Some(f) = UNPACK_MERGE_HOOK.get() {
        f(greenkey, w_iterator, items);
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
    LAST_EXEC_CTX.with(|c| {
        let previous = c.replace(ctx);
        if previous.is_null() && !ctx.is_null() {
            crate::module::thread::register_execution_context(ctx);
        } else if !previous.is_null() && ctx.is_null() {
            crate::module::thread::unregister_execution_context();
        }
    });
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
                nparams as isize - ndefaults as isize,
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
        // `fname` is WTF-8: `format!` would render it through `Display`, which
        // substitutes U+FFFD for a lone surrogate.
        let mut msg = Wtf8Buf::new();
        msg.push_wtf8(&fname);
        msg.push_str(&format!("() takes {takes_str} but {given_str} given"));
        return Err(crate::PyError::type_error(msg));
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
    // argument.py:218-219 `for i in range(take): scope_w[...] = args_w[i]` —
    // element-by-element positional fill (`args[i]`, a `usize` element index
    // the rtyper lowers to an ArrayRead) rather than a range-slice copy
    // (`&args[..n]` → `core::slice::index`, which has no graph lowering).
    for i in 0..n_pos_copied {
        filled_args.push(args[i]);
    }
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
        // `argument.py:302-315` computes `def_first = co_argcount -
        // len(defaults_w)` in signed arithmetic and keeps `defaults_w[defnum]`
        // for every `defnum = i - def_first` that is not negative.  A
        // `__defaults__` longer than the parameter list makes `def_first`
        // negative, which selects the tail of the tuple; in `usize` the
        // subtraction wrapped instead, no slot ever matched, and the call
        // raised a missing-argument `TypeError`.
        let first_default = nparams as isize - ndefaults as isize;
        for i in n_pos_copied..nparams {
            let default_idx = i as isize - first_default;
            if default_idx >= 0
                && let Some(val) =
                    unsafe { pyre_object::w_tuple_getitem(defaults, default_idx as i64) }
            {
                filled_args[i] = val;
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
    // `args[total_params..]` as the `*args` source.  argument.py:230
    // `starargs_w = args_w[args_left:]` — the one place RPython slices; here
    // spelled as the element-by-element `usize` push loop (`args[i]`, an
    // ArrayRead) the rtyper lowers, rather than the `&args[nparams..]`
    // range-slice (`core::slice::index`, no graph lowering).
    if has_varargs && nargs > nparams {
        for i in nparams..nargs {
            filled_args.push(args[i]);
        }
    }

    Ok(pack_varargs(code_ref, filled_args))
}

/// `argument.py:534-552` ArgErrMissing.getmsg parity.
///
/// `fname` is the function's `__qualname__`, which may carry a lone surrogate,
/// and the message becomes the TypeError's `args[0]` -- so the whole line is
/// assembled as WTF-8 rather than through a `String`.
fn format_missing_err(fname: &Wtf8, missing: &[&str], positional: bool) -> Wtf8Buf {
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
    let mut msg = Wtf8Buf::new();
    msg.push_wtf8(fname);
    msg.push_str(&format!(
        "() missing {} required {} argument{}: {arguments_str}",
        missing.len(),
        if positional {
            "positional"
        } else {
            "keyword-only"
        },
        if missing.len() != 1 { "s" } else { "" },
    ));
    msg
}

/// `argument.py:620-626` ArgErrUnknownKwds.getmsg parity.
fn format_unknown_kwds_err(fname: &Wtf8, unmatched: &[Wtf8Buf]) -> Wtf8Buf {
    let mut msg = Wtf8Buf::new();
    msg.push_wtf8(fname);
    if unmatched.len() == 1 {
        // `argument.py:616` keys this off the keyword's own storage, so a name
        // with a lone surrogate reaches `e.args[0]` as itself.
        msg.push_str("() got an unexpected keyword argument '");
        msg.push_wtf8(&unmatched[0]);
        msg.push_str("'");
    } else {
        msg.push_str(&format!(
            "() got {} unexpected keyword arguments",
            unmatched.len()
        ));
    }
    msg
}

#[cold]
fn raise_if_posonly_kwds(posonly_kwds: &[String], fname: &Wtf8) -> Result<(), PyError> {
    if posonly_kwds.is_empty() {
        return Ok(());
    }
    let mut msg = Wtf8Buf::new();
    msg.push_wtf8(fname);
    msg.push_str(&format!(
        "() got some positional-only arguments passed as keyword arguments: '{}'",
        posonly_kwds.join(", ")
    ));
    Err(crate::PyError::type_error(msg))
}

/// Materialize a Python call frame without retaining the by-value `PyFrame`
/// construction temporary while that frame executes recursively.
///
/// RPython allocates `PyFrame` as a GC object before `execute_frame`; the
/// allocation graph has returned when the recursive interpreter graph runs.
/// Keeping Rust's large `Result<PyFrame, _>` temporary in the executor's
/// native frame multiplies stack use at every Python call, so preserve the
/// same graph boundary explicitly.
#[inline(never)]
fn make_user_call_frame(
    w_code: *const (),
    args: &[PyObjectRef],
    w_globals: PyObjectRef,
    execution_context: *const crate::PyExecutionContext,
    closure: PyObjectRef,
) -> Result<crate::pyframe::FrameBox, crate::PyError> {
    let frame = PyFrame::try_new_for_call_with_closure_and_globals_obj(
        w_code,
        args,
        w_globals,
        execution_context,
        closure,
        crate::pyframe::FrameLocalsArrayAllocation::OldGenGc,
    )?;
    Ok(crate::pyframe::FrameBox::new(frame))
}

enum PreparedUserCall {
    Frame(crate::pyframe::FrameBox),
    Generator(PyObjectRef),
}

/// PyPy `Function.funccall` prepares arguments and allocates the callee frame
/// before entering `PyFrame.execute_frame`.  Keep that preparation in its own
/// native graph as well: none of its argument-matching or frame-construction
/// temporaries are live while the recursive evaluator runs.
#[inline(never)]
fn prepare_user_call(
    execution_context: *const crate::PyExecutionContext,
    callable: PyObjectRef,
    args: &[PyObjectRef],
) -> Result<PreparedUserCall, crate::PyError> {
    let w_code = unsafe { crate::getcode(callable) };
    let w_globals = unsafe { function_get_globals_obj(callable) };
    let closure = unsafe { function_get_closure(callable) };
    let func_code = unsafe {
        crate::w_code_get_ptr(w_code as pyre_object::PyObjectRef) as *const crate::CodeObject
    };
    let code_ref = unsafe { &*func_code };
    let final_args = fill_user_function_args(callable, code_ref, args)?;
    let func_frame =
        make_user_call_frame(w_code, &final_args, w_globals, execution_context, closure)?;

    if crate::pyframe::code_flags_make_generator(code_ref.flags) {
        return frame_into_generator_for_function(func_frame, callable)
            .map(PreparedUserCall::Generator);
    }
    Ok(PreparedUserCall::Frame(func_frame))
}

fn call_user_function_with_eval(
    frame: &PyFrame,
    callable: PyObjectRef,
    args: &[PyObjectRef],
    eval_fn: EvalFn,
) -> PyResult {
    let mut func_frame = match prepare_user_call(frame.execution_context, callable, args)? {
        PreparedUserCall::Frame(func_frame) => func_frame,
        PreparedUserCall::Generator(generator) => return Ok(generator),
    };
    func_frame.fix_array_ptrs();
    let _caller_locals_root = FrameLocalsRoot::new(frame);
    let _callee_locals_root = FrameLocalsRoot::new_mut(&mut func_frame);
    eval_fn(&mut func_frame)
}

/// [`call_user_function`] with the execution context in place of the caller
/// frame — the `function.py:79 Function.call_args(self, args)` shape, which
/// upstream reaches with no frame at all.
///
/// Also the residual-call sibling of [`call_user_function_plain`], keeping the
/// JIT-aware eval function.
///
/// `blackhole.py:1225 bhimpl_residual_call_r_i` is `cpu.bh_call_i(func, ...)`
/// — it invokes the *translated function*, and when that function's graph
/// reaches a `jit_merge_point` (`execute_frame` does) the JIT is entered
/// normally. "Opaque to the trace" does not mean "the JIT is off inside":
/// upstream has no flag that disables it for the extent of a residual call.
/// `bhimpl_recursive_call_*` (`blackhole.py:1095-1132`) is not the only way
/// to reach the portal — it is the path the codewriter emits when the callee
/// is *statically* the portal graph.
///
/// Re-entrant tracing is prevented where upstream prevents it, on the green
/// key: `warmstate.py:473-477` skips a hot back-edge while `JC_TRACING` is
/// set, which pyre mirrors with the `driver.is_tracing()` guard in
/// `maybe_compile_and_run`.
///
/// The execution context is passed in rather than read off a caller frame:
/// `bhimpl_residual_call_r_r` (`blackhole.py:1227`) is
/// `cpu.bh_call_r(func, None, args_r, ...)`, carrying no frame operand, and
/// the callee frame takes its context from the space
/// (`space.getexecutioncontext()`).  A residual helper that resolved the
/// caller frame instead would have to read `topframeref`, and
/// `gettopframe_raw` is `force_vref`: forcing a vref that carries
/// `TOKEN_TRACING_RESCALL` across a residual clears the token, which
/// `tracing_after_residual_call` reads back as "the callee escaped this
/// frame" (`virtualref.py:161-167`).  The recorded escape then materializes
/// the caller frame on every execution of the compiled trace.
///
/// The caller-side `FrameLocalsRoot` is skipped for the same reason it is in
/// [`call_user_function_plain_with_ctx`] — no caller `PyFrame` is available.
/// The callee root is still installed so its locals stay reachable.
pub fn call_user_function_with_ctx(
    execution_context: *const crate::PyExecutionContext,
    callable: PyObjectRef,
    args: &[PyObjectRef],
) -> PyResult {
    let mut func_frame = match prepare_user_call(execution_context, callable, args)? {
        PreparedUserCall::Frame(func_frame) => func_frame,
        PreparedUserCall::Generator(generator) => return Ok(generator),
    };
    func_frame.fix_array_ptrs();
    let _callee_locals_root = FrameLocalsRoot::new_mut(&mut func_frame);
    get_eval_fn()(&mut func_frame)
}

/// Call a user function with pre-resolved args (scope already packed by
/// resolve_kwargs). Skips defaults-fill and pack_varargs — the caller
/// (call_kw) already produced the final scope via resolve_kwargs which
/// mirrors PyPy's Arguments.parse_into_scope.
pub fn call_user_function_resolved(
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

    // Generator function
    if crate::pyframe::code_flags_make_generator(code_ref.flags) {
        let gen_frame =
            crate::pyframe::FrameBox::new(PyFrame::try_new_for_call_with_closure_and_globals_obj(
                w_code,
                args,
                w_globals,
                execution_context,
                closure,
                crate::pyframe::FrameLocalsArrayAllocation::OldGenGc,
            )?);
        return frame_into_generator_for_function(gen_frame, callable);
    }

    let eval_fn = get_eval_fn();

    let mut func_frame =
        crate::pyframe::FrameBox::new(PyFrame::try_new_for_call_with_closure_and_globals_obj(
            w_code,
            args,
            w_globals,
            execution_context,
            closure,
            crate::pyframe::FrameLocalsArrayAllocation::OldGenGc,
        )?);
    func_frame.fix_array_ptrs();
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
pub(crate) fn set_orig_class(
    result: PyObjectRef,
    alias: PyObjectRef,
) -> Result<(), crate::PyError> {
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

/// Invoke a builtin from a slice of raw positional arguments, binding through
/// its `Signature` first when it has a HOPELESS fast arity.  `builtin_code_call`
/// itself never binds — the direct call sites hand it an already-bound flat
/// slice — so any entry that starts from raw positionals (the frame dispatch
/// here, and the JIT residual-call path in `pyre-jit`) must route through this
/// to give a `*args`/optional-positional body the slot shape it reads.
pub fn builtin_code_call_positional(
    current_code: PyObjectRef,
    current_args: &[PyObjectRef],
) -> Result<PyObjectRef, crate::PyError> {
    if let Some(sig) = unsafe { crate::builtin_code_get_signature(current_code) } {
        // Every HOPELESS signature needs `_match_signature`, not only
        // *args/**kwargs/kw-only shapes.  A plain optional positional
        // parameter also has HOPELESS fast arity; bypassing the binder let
        // excess positionals reach the typed wrapper, which consumes its
        // declared prefix and silently ignores the rest.
        if unsafe { crate::builtin_code_get_fast_natural_arity(current_code) } == crate::HOPELESS {
            let fname = unsafe { crate::builtin_code_name(current_code) };
            let bound = bind_kwargs_to_signature(sig, fname, current_args, &[])?;
            return unsafe { crate::builtin_code_call(current_code, &bound) };
        }
    }
    unsafe { crate::builtin_code_call(current_code, current_args) }
}

#[majit_macros::dont_look_inside]
fn call_builtin_code_many_from_roots(root_base: usize, nargs: usize) -> PyResult {
    let mut rooted = vec![pyre_object::PY_NULL; 1 + nargs];
    pyre_object::gc_roots::shadow_stack_copy_range(root_base, &mut rooted);
    builtin_code_call_positional(rooted[0], &rooted[1..])
}

fn call_builtin_code_positional(code: PyObjectRef, args: &[PyObjectRef]) -> PyResult {
    // `gateway.py:824 BuiltinCode.funcrun` is translated with both its code
    // object and `Arguments.arguments_w` live across gateway dispatch.  A
    // collection between the outer `space.call_function` reload and this
    // indirect Rust function-pointer call updates the outer shadow slots but
    // not the copied native slice, so mirror the gateway's own root frame and
    // reload immediately before invoking the builtin.
    let _roots = pyre_object::gc_roots::push_roots();
    let root_base = _roots.base();
    _roots.pin_root(code);
    for &arg in args {
        _roots.pin_root(arg);
    }
    // RPython's pop-roots reload produces ordinary live variables.  Spell the
    // common fixed-arity cases the same way so source translation sees no Rust
    // array slicing/indexing helpers between the live roots and the gateway
    // indirect call.  The uncommon variadic case stays a residual helper.
    let current_code = _roots.get(root_base);
    match args.len() {
        0 => builtin_code_call_positional(current_code, &[]),
        1 => {
            let a0 = _roots.get(root_base + 1);
            builtin_code_call_positional(current_code, &[a0])
        }
        2 => {
            let a0 = _roots.get(root_base + 1);
            let a1 = _roots.get(root_base + 2);
            builtin_code_call_positional(current_code, &[a0, a1])
        }
        3 => {
            let a0 = _roots.get(root_base + 1);
            let a1 = _roots.get(root_base + 2);
            let a2 = _roots.get(root_base + 3);
            builtin_code_call_positional(current_code, &[a0, a1, a2])
        }
        4 => {
            let a0 = _roots.get(root_base + 1);
            let a1 = _roots.get(root_base + 2);
            let a2 = _roots.get(root_base + 3);
            let a3 = _roots.get(root_base + 4);
            builtin_code_call_positional(current_code, &[a0, a1, a2, a3])
        }
        nargs => call_builtin_code_many_from_roots(root_base, nargs),
    }
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

/// `baseobjspace.py:1243 call_valuestack(w_func, nargs, frame, …)` — the one
/// dispatcher upstream gives a frame to.  It settles the C-profile question and
/// hands off to the frameless `space.call_args`, which is
/// [`call_callable_in_ctx`].
///
/// Pyre adds one thing to that shape: `FrameLocalsRoot` on the caller.  RPython
/// roots the caller's locals through the shadowstack of the translated
/// `call_valuestack`; this Rust ABI boundary is outside that transform, so the
/// root is installed here and held across the whole dispatch.
pub fn call_callable(frame: &mut PyFrame, callable: PyObjectRef, args: &[PyObjectRef]) -> PyResult {
    let _caller_locals_root = FrameLocalsRoot::new(frame);
    call_callable_with_mode(frame.execution_context, callable, args, CallMode::Jit)
}

/// `descroperation.py:189 call_args(space, w_obj, args)` — the generic callable
/// dispatcher, which upstream reaches with **no frame**: `Function.call_args`
/// (`function.py:79`) goes straight to `code.funcrun(self, args)` and the callee
/// frame takes its execution context from the space.
pub fn call_callable_in_ctx(
    execution_context: *const crate::PyExecutionContext,
    callable: PyObjectRef,
    args: &[PyObjectRef],
) -> PyResult {
    call_callable_with_mode(execution_context, callable, args, CallMode::Jit)
}

/// The caller frame the C-level profile arm needs, or null when no profiler is
/// installed.
///
/// `baseobjspace.py:1245` gates the arm on `frame.get_is_being_profiled()`, and
/// that flag has exactly one writer: `executioncontext.py:147-149 call_trace`
/// sets it only while `profilefunc is not None`, and `:121-123
/// _c_call_return_trace` clears it and returns the moment `profilefunc is
/// None`.  So an execution context with no profiler installed cannot take the
/// arm, and testing that first leaves the frame unresolved on the ordinary
/// path — `gettopframe_raw` is `force_vref`, and a vref forced while the trace
/// records is marked as escaping (`virtualref.py:161-167`).
fn c_profile_frame(execution_context: *const crate::PyExecutionContext) -> *mut PyFrame {
    if execution_context.is_null() {
        return std::ptr::null_mut();
    }
    let ec = unsafe { &*execution_context };
    if ec.profilefunc.is_none() {
        return std::ptr::null_mut();
    }
    let frame = ec.gettopframe_raw();
    if frame.is_null() || !unsafe { (*frame).get_is_being_profiled() } {
        return std::ptr::null_mut();
    }
    frame
}

/// Function/_BuiltinFunction leaf of ObjSpace call dispatch.
///
/// PyPy does not recursively feed a bound `_Method` back through the generic
/// callable dispatcher.  `Method.call_args` delegates to
/// `space.call_obj_args`, whose Function fast path calls the function
/// directly (`baseobjspace.py:1204-1211`).  Keeping this leaf separate lets
/// the method arm below preserve that shape instead of retaining a second
/// large generic-dispatch Rust frame for every Python method call.
fn call_function_carrier_with_mode(
    execution_context: *const crate::PyExecutionContext,
    callable: PyObjectRef,
    args: &[PyObjectRef],
    mode: CallMode,
) -> PyResult {
    match classify_callable(callable)? {
        CallableKind::Builtin => {
            // baseobjspace.py:1243 — `if frame.get_is_being_profiled() and
            // is_builtin_code(w_func): ... return self.call_args_and_c_profile(...)`
            // The `is_builtin_code(w_func)` check is structurally implicit
            // here: `classify_callable` already selected the builtin arm
            // (`runtime_ops.rs`: `if is_builtin_code(code) { Builtin }`),
            // so reaching this closure means the callable is a builtin.
            let profile_frame = c_profile_frame(execution_context);
            if !profile_frame.is_null() {
                let w_res = crate::baseobjspace::call_args_and_c_profile(
                    unsafe { &mut *profile_frame },
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
        }
        CallableKind::User => match mode {
            CallMode::Jit => call_user_function_with_ctx(execution_context, callable, args),
            CallMode::Plain => call_user_function_plain_with_ctx(execution_context, callable, args),
        },
    }
}

/// [`call_function_ex_in_ctx`] reached from a frame — the `pyopcode.py:1429
/// CALL_FUNCTION_EX` shape, whose else-branch is the frameless
/// `space.call_args(w_function, args)`.  Installs the caller `FrameLocalsRoot`
/// the way [`call_callable`] does.
pub fn call_function_ex(
    frame: &mut PyFrame,
    callable: PyObjectRef,
    self_or_null: PyObjectRef,
    starargs: PyObjectRef,
    kwargs_or_null: PyObjectRef,
) -> PyResult {
    let _caller_locals_root = FrameLocalsRoot::new(frame);
    call_function_ex_in_ctx(
        frame.execution_context,
        callable,
        self_or_null,
        starargs,
        kwargs_or_null,
    )
}

/// CALL_FUNCTION_EX helper — unpack `starargs`, merge the `**` mapping, and
/// call.  Factored out of the interpreter's `call_function_ex` so the JIT
/// residual (`bh_call_function_ex_fn`) shares one implementation.  Mirrors
/// `argument.py` unpack_combined_starargs + `_combine_starstarargs_wrapped`:
/// a tuple/list stararg takes the fast path, any other iterable goes through
/// the iter protocol; a non-null `**` mapping accepts the dict fast path or
/// `keys()`/`__getitem__`.  `self_or_null` is the pre-callable stack slot —
/// a non-null value prepends as arg0.
pub fn call_function_ex_in_ctx(
    execution_context: *const crate::PyExecutionContext,
    callable: PyObjectRef,
    self_or_null: PyObjectRef,
    starargs: PyObjectRef,
    kwargs_or_null: PyObjectRef,
) -> PyResult {
    let mut args: Vec<PyObjectRef> = unsafe {
        // argument.py:92-104 reaches `space.fixedview`, whose fast paths
        // (`objspace.py:519-527`) are a tuple whose `__iter__` is still
        // `tuple.__iter__` and an exact list.  A subtype that replaced
        // `__iter__` falls through to the generic path so the override runs.
        if pyre_object::is_tuple(starargs)
            && crate::baseobjspace::builtin_iter_replacement(starargs, &pyre_object::TUPLE_TYPE)
                .is_none()
        {
            let n = pyre_object::w_tuple_len(starargs);
            (0..n as i64)
                .filter_map(|i| pyre_object::w_tuple_getitem(starargs, i))
                .collect()
        } else if pyre_object::is_exact_list(starargs) {
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
            return call_with_kwargs_in_ctx(execution_context, callable, &args, &entries);
        }
    }

    call_callable_in_ctx(execution_context, callable, &args)
}

/// [`call_kw_in_ctx`] reached from a frame — the `pyopcode.py:1402
/// CALL_FUNCTION_KW` shape, whose else-branch is the frameless
/// `space.call_args(w_function, args)`.  Installs the caller `FrameLocalsRoot`
/// the way [`call_callable`] does.
pub fn call_kw(
    frame: &mut PyFrame,
    callable: PyObjectRef,
    self_or_null: PyObjectRef,
    positional: &[PyObjectRef],
    kwarg_names: PyObjectRef,
) -> PyResult {
    let _caller_locals_root = FrameLocalsRoot::new(frame);
    call_kw_in_ctx(
        frame.execution_context,
        callable,
        self_or_null,
        positional,
        kwarg_names,
    )
}

/// CALL_KW helper — resolve keyword arguments against the callable and
/// call.  Factored out of the interpreter's `call_kw` so the JIT residual
/// (`bh_call_kw_fn`) shares one implementation.  `positional` holds the
/// `arg0..argN-1` values already in positional order (keyword tail
/// included); `kwarg_names` is the constant kwnames tuple (its length is
/// the number of trailing keyword args).  `self_or_null` is the
/// pre-callable stack slot — a non-null value prepends as arg0.
pub fn call_kw_in_ctx(
    execution_context: *const crate::PyExecutionContext,
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
        return call_with_kwargs_in_ctx(
            execution_context,
            callable_unwrapped,
            &pos_args,
            &kw_entries,
        );
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
        return call_with_kwargs_in_ctx(
            execution_context,
            callable_unwrapped,
            &pos_args,
            &kw_entries,
        );
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
            return call_with_kwargs_in_ctx(
                execution_context,
                callable_unwrapped,
                &pos_args,
                &kw_entries,
            );
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
        return call_with_kwargs_in_ctx(
            execution_context,
            callable_unwrapped,
            &pos_args,
            &kw_entries,
        );
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
            return call_with_kwargs_in_ctx(
                execution_context,
                callable_unwrapped,
                &pos_args,
                &kw_entries,
            );
        }
        return call_callable_in_ctx(execution_context, callable_unwrapped, &args);
    }

    // `descroperation.py descr_call` binds an instance's `__call__` and
    // forwards the original Arguments object with its keyword names intact.
    // `resolve_kwargs` only understands Function signatures; running a
    // callable instance through it turns the trailing keyword values into
    // positional arguments before `call_callable` eventually finds
    // `__call__`.  Preserve the Arguments shape by splitting the CALL_KW tail
    // here and letting `call_with_kwargs` perform descriptor binding.
    if !unsafe { crate::is_function(callable_unwrapped) } {
        let nkw = if unsafe { pyre_object::is_tuple(kwarg_names) } {
            unsafe { pyre_object::w_tuple_len(kwarg_names) }
        } else {
            0
        };
        if nkw > 0 {
            let n_pos = args.len().saturating_sub(nkw);
            let pos_args = args[..n_pos].to_vec();
            let mut kw_entries = Vec::with_capacity(nkw);
            for ki in 0..nkw {
                if let Some(name_obj) =
                    unsafe { pyre_object::w_tuple_getitem(kwarg_names, ki as i64) }
                {
                    let key = unsafe { pyre_object::w_str_get_wtf8(name_obj) }.to_owned();
                    kw_entries.push((key, args[n_pos + ki]));
                }
            }
            return call_with_kwargs_in_ctx(
                execution_context,
                callable_unwrapped,
                &pos_args,
                &kw_entries,
            );
        }
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
        call_user_function_resolved(execution_context, target_func, &resolved)
    } else {
        call_callable_in_ctx(execution_context, target_func, &resolved)
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
    if std::ptr::eq(metaclass.as_ptr(), crate::typedef::w_type()) {
        return None;
    }
    // Resolve WHERE `__call__` is defined first; the default `type.__call__`
    // (the implicit instantiation path) is not an override, so a metaclass
    // that merely inherits it — e.g. ABCMeta — keeps the fast path.  The
    // defining-class half is the cheap guard, so it runs before the value
    // half's second residual walk (avoided on the common fast path).
    let where_defined = unsafe {
        crate::baseobjspace::lookup_where_class_uncached(metaclass.as_ptr(), "__call__")
    }?;
    if std::ptr::eq(where_defined, crate::typedef::w_type()) {
        return None;
    }
    let call_descr = unsafe {
        crate::baseobjspace::lookup_in_type_where_uncached(metaclass.as_ptr(), "__call__")
    }?;
    let bound = unsafe { crate::baseobjspace::get(call_descr, callable, metaclass.as_ptr()) }
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
    let Some(call_descr) =
        (unsafe { crate::baseobjspace::lookup_in_type(w_type.as_ptr(), "__call__") })
    else {
        return Ok(None);
    };
    let bound = unsafe { crate::baseobjspace::get(call_descr, callable, w_type.as_ptr()) }?
        .unwrap_or(call_descr);
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
    let Some(call_descr) =
        (unsafe { crate::baseobjspace::lookup_in_type(w_type.as_ptr(), "__call__") })
    else {
        return Ok(None);
    };
    let bound = unsafe { crate::baseobjspace::get(call_descr, callable, w_type.as_ptr()) }?
        .unwrap_or(call_descr);
    Ok(Some(bound))
}

/// descroperation.py `descr__call__` — `space.lookup(w_obj, '__call__')`.
///
/// Consulted once the builtin callables above have had their turn.  PyPy's
/// `space.lookup` applies uniformly to every `W_Root`: the storage layout is
/// irrelevant when the object's dynamic type publishes a `__call__` slot.
fn user_call_slot(callable: PyObjectRef) -> Result<Option<(PyObjectRef, bool)>, PyError> {
    let Some(w_type) = crate::typedef::r#type(callable) else {
        return Ok(None);
    };
    let w_type = w_type.as_ptr();
    let Some(call_fn) = (unsafe { crate::baseobjspace::lookup_in_type(w_type, "__call__") }) else {
        return Ok(None);
    };
    // `A.__call__ = A()` makes this edge feed itself, and the callers below
    // recurse natively.  The interpreter-level check turns that into
    // RecursionError instead of exhausting the machine stack.
    crate::stack_check::stack_check()?;
    // descroperation.py:161-167 `get_and_call_args` — the rule `call_args`
    // uses for `__call__`: `isinstance(w_descr, Function)`, so a builtin
    // function and a fixed-code function take it too.  Such a descriptor
    // takes the object directly as its first positional argument; every other
    // one is bound through `space.get` and then called without an extra
    // receiver. The bool tells all call entrypoints which of the two applies.
    // (`get_and_call_function`'s narrower `type(w_descr) is Function` test is
    // for the *other* entrypoint and must not be used here.)
    if unsafe { crate::is_function(call_fn) } {
        return Ok(Some((call_fn, true)));
    }
    let bound = unsafe { crate::baseobjspace::get(call_fn, callable, w_type) }?.unwrap_or(call_fn);
    Ok(Some((bound, false)))
}

/// baseobjspace.py `call_valuestack` / `call_obj_args` speedhack boundary.
///
/// Keep this wrapper small so exact Functions and bound Methods never enter
/// (and therefore never reserve the native stack frame of) the generic
/// descriptor/type dispatcher below.
#[inline(always)]
fn call_callable_with_mode(
    execution_context: *const crate::PyExecutionContext,
    callable: PyObjectRef,
    args: &[PyObjectRef],
    mode: CallMode,
) -> PyResult {
    // baseobjspace.py:1241-1242 `call_valuestack`: Function is the primary
    // speedhack and calls `funccall_valuestack` directly.  Do not retain the
    // generic descriptor/type dispatcher across every Python frame.
    if unsafe { crate::is_function_carrier(callable) } {
        return call_function_carrier_with_mode(execution_context, callable, args, mode);
    }
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
        if unsafe { crate::is_function_carrier(func) } {
            // function.py `_Method.call_args` -> baseobjspace.py
            // `call_obj_args`: exact Function/BuiltinFunction carriers skip a
            // second generic callable dispatch.
            return call_function_carrier_with_mode(execution_context, func, &call_args, mode);
        }
        return call_callable_with_mode(execution_context, func, &call_args, mode);
    }
    call_non_function_callable_with_mode(execution_context, callable, args, mode)
}

#[inline(never)]
fn call_non_function_callable_with_mode(
    execution_context: *const crate::PyExecutionContext,
    callable: PyObjectRef,
    args: &[PyObjectRef],
    mode: CallMode,
) -> PyResult {
    if unsafe { pyre_object::is_type(callable) } {
        if let Some(bound) = metaclass_call_override(callable) {
            return call_callable_with_mode(execution_context, bound, args, mode);
        }
        return type_descr_call_with_mode(execution_context, callable, args, mode);
    }

    // staticmethod → unwrap
    // PyPy: function.py StaticMethod.descr_call
    if unsafe { pyre_object::is_exact_type(callable, &pyre_object::function::STATICMETHOD_TYPE) } {
        let func = unsafe { pyre_object::w_staticmethod_get_func(callable) };
        return call_callable_with_mode(execution_context, func, args, mode);
    }
    if let Some(bound) = staticmethod_call_override(callable)? {
        return call_callable_with_mode(execution_context, bound, args, mode);
    }
    if let Some(bound) = classmethod_call_override(callable)? {
        return call_callable_with_mode(execution_context, bound, args, mode);
    }
    // The base ClassMethod defines no descr_call (function.py), so a raw
    // classmethod object falls through to the not-callable error.

    // descroperation.py `descr__call__` keeps both `w_obj` and `Arguments`
    // live while binding a non-Function `__call__` descriptor.  `space.get`
    // may execute arbitrary Python and move every nursery argument, so mirror
    // the translated shadow-stack roots and reload them after binding.
    let _user_call_roots = pyre_object::gc_roots::push_roots();
    let user_call_root_base = pyre_object::gc_roots::shadow_stack_len();
    pyre_object::gc_roots::pin_root(callable);
    for &arg in args {
        pyre_object::gc_roots::pin_root(arg);
    }
    if let Some((call_fn, prepend_receiver)) =
        user_call_slot(pyre_object::gc_roots::shadow_stack_get(user_call_root_base))?
    {
        let current_callable = pyre_object::gc_roots::shadow_stack_get(user_call_root_base);
        let current_args: Vec<PyObjectRef> = (0..args.len())
            .map(|i| pyre_object::gc_roots::shadow_stack_get(user_call_root_base + 1 + i))
            .collect();
        if prepend_receiver {
            let mut call_args = Vec::with_capacity(1 + current_args.len());
            call_args.push(current_callable);
            call_args.extend_from_slice(&current_args);
            return call_callable_with_mode(execution_context, call_fn, &call_args, mode);
        }
        // `user_call_slot`'s stack_check bounds a self-referential
        // `A.__call__ = A()` chain only while this self-dispatch recurses
        // natively.  The call-depth guard counts this dispatch level and, by
        // dropping only after the call returns, keeps it off the tail so LLVM
        // cannot rewrite the self-call into a loop that never grows the stack.
        let _depth_guard = enter_native_dispatch();
        return call_callable_with_mode(execution_context, call_fn, &current_args, mode);
    }

    // GenericAlias.__call__ (`_pypy_generic_alias.py:41`) —
    // `self.__origin__(*args, **kwargs)`, then best-effort
    // `result.__orig_class__ = self`.
    if unsafe { pyre_object::is_generic_alias(callable) } {
        let origin = unsafe { pyre_object::w_generic_alias_get_origin(callable) };
        let result = call_callable_with_mode(execution_context, origin, args, mode)?;
        set_orig_class(result, callable)?;
        return Ok(result);
    }

    call_function_carrier_with_mode(execution_context, callable, args, mode)
}

pub fn call_user_function(
    frame: &PyFrame,
    callable: PyObjectRef,
    args: &[PyObjectRef],
) -> PyResult {
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
        return frame_into_generator_for_function(gen_frame, callable);
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
    let _caller_locals_root = FrameLocalsRoot::new(frame);
    call_callable_with_mode(frame.execution_context, callable, args, CallMode::Plain)
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
    // argument.py:469,481-484 — `wrong_posonly` accumulator: every
    // posonly-as-keyword violation is collected across the whole loop and
    // reported together, instead of raising on the first one found.
    let mut posonly_kwds: Vec<String> = Vec::new();
    for ki in 0..nkw {
        let kw_name = unsafe { pyre_object::w_tuple_getitem(kwarg_names, ki as i64) };
        let Some(kw_name_obj) = kw_name else { continue };
        let kw_value = args[n_pos + ki];

        // argument.py:630 — keywords must be strings (check before access).
        // `_PyStack_UnpackDict` names neither the callable nor the key's type.
        if !unsafe { pyre_object::is_str(kw_name_obj) } {
            return Err(crate::PyError::type_error("keywords must be strings"));
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
                    // argument.py:481-484 — collect and keep scanning
                    // remaining keywords instead of raising immediately.
                    posonly_kwds.push(param_name.to_string());
                    matched = true;
                    break;
                }
                // argument.py:410 — duplicate keyword argument
                if !result[pi].is_null() {
                    let mut msg = Wtf8Buf::new();
                    msg.push_wtf8(&fname);
                    msg.push_str(&format!(
                        "() got multiple values for argument '{param_name}'"
                    ));
                    return Err(crate::PyError::type_error(msg));
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

    // argument.py:499-500 — ArgErrPosonlyAsKwds, raised after the full
    // keyword scan (and before ArgErrUnknownKwds, since `_match_keywords`
    // raises this before its caller ever checks unmatched kwds).
    raise_if_posonly_kwds(&posonly_kwds, &fname)?;

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
        let mut msg = Wtf8Buf::new();
        msg.push_wtf8(&fname);
        msg.push_str(&format!("() takes {takes_str} but {given_str}"));
        return Err(crate::PyError::type_error(msg));
    }

    // Fill positional defaults (PyPy: _match_signature defs_w)
    // Defaults cover the LAST N of the positional params (arg_count).
    let defaults = unsafe { crate::function_get_defaults(target_func) };
    if !defaults.is_null() && unsafe { pyre_object::is_tuple(defaults) } {
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
    // `gct_fv_gc_malloc` bracket (`framework.py:853-856`).  Every reference
    // live across the packing is a raw copy: the parameters already bound into
    // `result`, the unmatched keyword pairs, and the two tail objects.  Each
    // allocation below — the tail objects, `w_dict_store`'s strategy promotion
    // — leaves RUNNING before taking `gc_mutex` (`gc_sync.rs:22`), so a
    // collector on another thread relocates whatever the shadow stack does not
    // name.  Pin them all, then read every one back out.
    //
    // A signature with no star parameter allocates nothing here, so it skips
    // the bracket entirely: `pin_root` is `dont_look_inside`, and a pin per
    // bound parameter on every keyword call is a residual the tracer has to
    // carry through the compiled loop.
    if !has_varargs && !has_varkw {
        return Ok(result);
    }
    let roots = pyre_object::gc_roots::push_roots();
    let result_slot = roots.base();
    for &value in &result {
        roots.pin_root(value);
    }
    let extra_slot = result_slot + result.len();
    for &(key, value) in &extra_kwargs {
        roots.pin_root(key);
        roots.pin_root(value);
    }
    let varargs_slot = extra_slot + extra_kwargs.len() * 2;
    if has_varargs {
        let extra_pos: Vec<PyObjectRef> = if n_pos > n_pos_params {
            args[n_pos_params..n_pos].to_vec()
        } else {
            vec![]
        };
        roots.pin_root(pyre_object::w_tuple_new(extra_pos));
    }
    let varkw_slot = varargs_slot + usize::from(has_varargs);
    if has_varkw {
        // `dictmultiobject.py:77-80` — `space.newdict(kwargs=True)` selects
        // EmptyKwargsDictStrategy so the first unicode setitem promotes
        // directly to KwargsDictStrategy (parallel `(keys_w, values_w)`
        // shape) instead of stepping through UnicodeDictStrategy.
        roots.pin_root(pyre_object::w_dict_new_kwargs());
        for i in 0..extra_kwargs.len() {
            unsafe {
                pyre_object::w_dict_store(
                    roots.get(varkw_slot),
                    roots.get(extra_slot + i * 2),
                    roots.get(extra_slot + i * 2 + 1),
                );
            }
        }
    }
    let mut result: Vec<PyObjectRef> = (0..result.len())
        .map(|i| roots.get(result_slot + i))
        .collect();
    if has_varargs {
        result.push(roots.get(varargs_slot));
    }
    if has_varkw {
        result.push(roots.get(varkw_slot));
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
    let mut extra_kwargs: Vec<(Wtf8Buf, PyObjectRef)> = Vec::new();
    let mut unmatched_kw_names: Vec<Wtf8Buf> = Vec::new();
    // argument.py:469,481-484 — collected across the whole loop, reported
    // together instead of raising on the first violation found.
    let mut posonly_kwds: Vec<String> = Vec::new();
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
                    // argument.py:481-484 — collect and keep scanning
                    // remaining keywords instead of raising immediately.
                    posonly_kwds.push(key.to_string());
                    matched = true;
                    break;
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
                // The name stays a `Wtf8Buf` until the packing bracket below.
                // Interning it here would allocate once per unmatched keyword,
                // and every one of those is a safepoint that relocates the
                // values already accumulated — and the bound parameters in
                // `result` — while nothing names them.
                extra_kwargs.push((key.clone(), *value));
            } else {
                unmatched_kw_names.push(key.clone());
            }
        }
    }

    // argument.py:499-500 — ArgErrPosonlyAsKwds, raised after the full
    // keyword scan and before ArgErrUnknownKwds.
    raise_if_posonly_kwds(&posonly_kwds, Wtf8::new(fname))?;

    if !unmatched_kw_names.is_empty() {
        // parse_obj (argument.py:377-380) rewrites the unknown-keyword message
        // to "takes no keyword arguments" when the signature accepts no keywords
        // at all (no **kwargs and no keyword-only params). Every BuiltinCode
        // call routes through parse_obj (gateway.py funcrun / funcrun_obj), so
        // the rewrite applies at any arity, not just the single-argument form.
        let msg = if !has_varkw && sig.num_kwonlyargnames() == 0 {
            let mut msg = Wtf8Buf::new();
            msg.push_wtf8(Wtf8::new(fname));
            msg.push_str("() takes no keyword arguments");
            msg
        } else {
            format_unknown_kwds_err(Wtf8::new(fname), &unmatched_kw_names)
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
    // The bound parameters, the unmatched keyword pairs and both tail objects
    // are all raw copies held across the packing allocations; see the same
    // bracket in `resolve_kwargs`, including why a star-less signature returns
    // before it.
    if !has_varargs && !has_varkw {
        return Ok(result);
    }
    let roots = pyre_object::gc_roots::push_roots();
    let result_slot = roots.base();
    for &value in &result {
        roots.pin_root(value);
    }
    let extra_slot = result_slot + result.len();
    for &(_, value) in &extra_kwargs {
        roots.pin_root(value);
    }
    let varargs_slot = extra_slot + extra_kwargs.len();
    if has_varargs {
        let extra_pos: Vec<PyObjectRef> = if n_pos > n_pos_params {
            pos_args[n_pos_params..n_pos].to_vec()
        } else {
            vec![]
        };
        roots.pin_root(pyre_object::w_tuple_new(extra_pos));
    }
    let varkw_slot = varargs_slot + usize::from(has_varargs);
    if has_varkw {
        roots.pin_root(pyre_object::w_dict_new_kwargs());
        // The index loop lowers to direct element loads; iterator adapters are residual calls.
        #[allow(clippy::needless_range_loop)]
        for i in 0..extra_kwargs.len() {
            let key = &extra_kwargs[i].0;
            unsafe {
                // The key allocation runs first: as the second argument it
                // would be evaluated after the receiver, handing
                // `w_dict_store` the pre-collection dict address.
                let w_key = pyre_object::w_str_from_wtf8_managed(key.clone());
                pyre_object::w_dict_store(roots.get(varkw_slot), w_key, roots.get(extra_slot + i));
            }
        }
    }
    let result_len = result.len();
    let mut result: Vec<PyObjectRef> = Vec::with_capacity(result_len);
    for i in 0..result_len {
        result.push(roots.get(result_slot + i));
    }
    if has_varargs {
        result.push(roots.get(varargs_slot));
    }
    if has_varkw {
        result.push(roots.get(varkw_slot));
    }

    Ok(result)
}

/// [`call_with_kwargs_in_ctx`] reached from a frame.  Installs the caller
/// `FrameLocalsRoot` the way [`call_callable`] does.
pub fn call_with_kwargs(
    frame: &mut crate::pyframe::PyFrame,
    callable: PyObjectRef,
    pos_args: &[PyObjectRef],
    kwargs: &[(Wtf8Buf, PyObjectRef)],
) -> PyResult {
    let _caller_locals_root = FrameLocalsRoot::new(frame);
    call_with_kwargs_in_ctx(frame.execution_context, callable, pos_args, kwargs)
}

/// Call a user function with positional args + keyword args from a dict.
///
/// PyPy: argument.py Arguments._match_signature with keyword handling.
/// Used by CALL_FUNCTION_KW / CALL_KW and CALL_FUNCTION_EX when kwargs
/// are non-empty.
pub fn call_with_kwargs_in_ctx(
    execution_context: *const crate::PyExecutionContext,
    callable: PyObjectRef,
    pos_args: &[PyObjectRef],
    kwargs: &[(Wtf8Buf, PyObjectRef)],
) -> PyResult {
    // RPython's `Arguments` is GC-traced for the whole call. Mirror the GC
    // transform explicitly: keyword binding below allocates tuples, dicts,
    // and keyword-name strings before the callee frame owns these values.
    let _call_roots = pyre_object::gc_roots::push_roots();
    let call_root_base = pyre_object::gc_roots::shadow_stack_len();
    pyre_object::gc_roots::pin_root(callable);
    for &arg in pos_args {
        pyre_object::gc_roots::pin_root(arg);
    }
    for (_, value) in kwargs {
        pyre_object::gc_roots::pin_root(*value);
    }
    let current_callable = || pyre_object::gc_roots::shadow_stack_get(call_root_base);
    let current_pos_arg =
        |index: usize| pyre_object::gc_roots::shadow_stack_get(call_root_base + 1 + index);
    let current_kwarg = |index: usize| {
        pyre_object::gc_roots::shadow_stack_get(call_root_base + 1 + pos_args.len() + index)
    };
    // `Arguments` survives the whole dispatch upstream because its
    // `arguments_w`/`keywords_w` lists are traced and updated in place. The
    // builtin ABI passes a raw `&[PyObjectRef]` copy the collector cannot see,
    // so every forward across an allocating call rebuilds the view from the
    // roots pinned above rather than handing on the incoming slices.
    let extend_current_args = |dst: &mut Vec<PyObjectRef>| {
        for index in 0..pos_args.len() {
            dst.push(current_pos_arg(index));
        }
    };
    let current_kwargs = || -> Vec<(Wtf8Buf, PyObjectRef)> {
        kwargs
            .iter()
            .enumerate()
            .map(|(index, (name, _))| (name.clone(), current_kwarg(index)))
            .collect()
    };

    // function.py:712-713 StaticMethod.descr_call — the wrapper contributes
    // no implicit argument; forward the original positional and keyword
    // collections unchanged to its w_function.
    if unsafe { pyre_object::is_exact_type(callable, &pyre_object::function::STATICMETHOD_TYPE) } {
        let func = unsafe { pyre_object::w_staticmethod_get_func(callable) };
        return call_with_kwargs_in_ctx(execution_context, func, pos_args, kwargs);
    }
    if let Some(bound) = staticmethod_call_override(callable)? {
        return call_with_kwargs_in_ctx(execution_context, bound, pos_args, kwargs);
    }

    if unsafe { pyre_object::is_classmethod(callable) } {
        if let Some(bound) = classmethod_call_override(callable)? {
            return call_with_kwargs_in_ctx(execution_context, bound, pos_args, kwargs);
        }
        let type_name = crate::typedef::r#type(callable)
            .map(|tp| unsafe { pyre_object::w_type_get_name(tp.as_ptr()) })
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
        return call_with_kwargs_in_ctx(execution_context, func, &full_args, kwargs);
    }

    // A class call routes through `type(cls).__call__` when the metaclass
    // overrides it (enum functional API passes `module=`/`type=` kwargs).
    if unsafe { pyre_object::is_type(callable) }
        && let Some(bound) = metaclass_call_override(callable)
    {
        return call_with_kwargs_in_ctx(execution_context, bound, pos_args, kwargs);
    }

    if unsafe { crate::is_function_carrier(callable) } {
        if unsafe { crate::is_slot_wrapper(callable) }
            && let Some(&receiver) = pos_args.first()
        {
            crate::typedef::slot_wrapper_check_instance(callable, receiver)?;
        }
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
                let frame_ptr = c_profile_frame(execution_context);
                if !frame_ptr.is_null() {
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
                return unsafe {
                    crate::builtin_code_call(code as pyre_object::PyObjectRef, &bound)
                };
            }
            let arity = unsafe {
                crate::builtin_code_get_fast_natural_arity(code as pyre_object::PyObjectRef)
            };
            if arity <= 4 && !kwargs.is_empty() {
                return Err(unsafe {
                    crate::builtin_code_no_keyword_arguments(
                        code as pyre_object::PyObjectRef,
                        pos_args.first().copied(),
                    )
                });
            }
            // `gct_fv_gc_malloc` bracket (`framework.py:853-856`) for the kwargs
            // dict below.  `w_dict_new` allocates in the nursery, so the fresh
            // dict is a young object no root names, and every step that follows
            // allocates: the key strings, `w_dict_store`'s strategy promotion,
            // and the call this frame ends in.  An allocation leaves RUNNING
            // before taking `gc_mutex` (`gc_sync.rs:22`), so a collector on
            // another thread runs there and relocates or reclaims anything the
            // shadow stack does not name.  The scope reaches every `return`
            // below, which is where the dict stops being a livevar.
            //
            // The positionals and the keyword values are already pinned by
            // this function's entry bracket, so they are read back through
            // `current_pos_arg` / `current_kwarg` instead of being copied —
            // `pos_args.to_vec()` taken before these allocations would hand
            // the callee pre-collection addresses.
            let kw_roots = pyre_object::gc_roots::push_roots();
            let kw_slot = kw_roots.base();
            if !kwargs.is_empty() {
                kw_roots.pin_root(pyre_object::w_dict_new());
                for (index, (key, _)) in kwargs.iter().enumerate() {
                    unsafe {
                        // The key allocation runs first: as the second argument
                        // it would be evaluated after the receiver, handing
                        // `w_dict_store` the pre-collection dict address.
                        let w_key = pyre_object::w_str_from_wtf8_managed(key.clone());
                        pyre_object::w_dict_store(
                            kw_roots.get(kw_slot),
                            w_key,
                            current_kwarg(index),
                        );
                    }
                }
                // Store the marker last so a user keyword literally named
                // `__pyre_kw__` cannot overwrite the sentinel: the reserved
                // key always resolves to the sentinel value that detection
                // compares by identity.
                unsafe {
                    let marker_key = pyre_object::kw_marker::w_kw_marker_key();
                    let marker_value = pyre_object::kw_marker::w_kw_marker_sentinel();
                    pyre_object::w_dict_store(kw_roots.get(kw_slot), marker_key, marker_value);
                }
            }
            let mut full_args: Vec<PyObjectRef> =
                (0..pos_args.len()).map(current_pos_arg).collect();
            if !kwargs.is_empty() {
                full_args.push(kw_roots.get(kw_slot));
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
                let frame_ptr = c_profile_frame(execution_context);
                if !frame_ptr.is_null() {
                    let keyword_names_w: Vec<pyre_object::PyObjectRef> = kwargs
                        .iter()
                        .map(|(k, _)| pyre_object::w_str_from_wtf8(k.clone()))
                        .collect();
                    // `keyword_names_w` allocated a string per keyword, so
                    // everything read before it — the positionals, the keyword
                    // values, and the dict `full_args` carries — may have moved
                    // since. Everything below reloads from the roots.
                    let keywords_w: Vec<pyre_object::PyObjectRef> =
                        (0..kwargs.len()).map(current_kwarg).collect();
                    let refreshed_pos: Vec<pyre_object::PyObjectRef> =
                        (0..pos_args.len()).map(current_pos_arg).collect();
                    let arguments = crate::argument::Arguments::with_kw(
                        &refreshed_pos,
                        &keyword_names_w,
                        &keywords_w,
                    );
                    full_args = refreshed_pos
                        .iter()
                        .copied()
                        .chain(std::iter::once(kw_roots.get(kw_slot)))
                        .collect();
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
            return call_callable_in_ctx(execution_context, callable, &full_args);
        }

        // For user functions: resolve kwargs to parameter slots
        {
            let w_code = unsafe { crate::getcode(current_callable()) };
            let code = unsafe {
                &*(crate::w_code_get_ptr(w_code as pyre_object::PyObjectRef)
                    as *const crate::CodeObject)
            };
            let total_params = (code.arg_count + code.kwonlyarg_count) as usize;
            let n_pos_params = code.arg_count as usize;
            let has_varkw = code.flags.contains(crate::CodeFlags::VARKEYWORDS);
            let has_varargs = code.flags.contains(crate::CodeFlags::VARARGS);
            let fname = unsafe { crate::function_get_qualname(current_callable()) };

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
                result[i] = current_pos_arg(i);
            }
            // Match keywords to parameter names
            let posonly = code.posonlyarg_count as usize;
            let mut extra_kwargs: Vec<(Wtf8Buf, PyObjectRef)> = Vec::new();
            let mut unmatched_kw_names: Vec<Wtf8Buf> = Vec::new();
            // argument.py:469,481-484 — collected across the whole loop,
            // reported together instead of raising on the first violation.
            let mut posonly_kwds: Vec<String> = Vec::new();
            for (kw_index, (key, _value)) in kwargs.iter().enumerate() {
                let value = current_kwarg(kw_index);
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
                            // argument.py:481-484 — collect and keep scanning
                            // remaining keywords instead of raising immediately.
                            posonly_kwds.push(key.to_string());
                            matched = true;
                            break;
                        }
                        // argument.py:495 — ArgErrMultipleValues: keyword
                        // duplicates an already-bound positional argument.
                        if !result[pi].is_null() {
                            let mut msg = Wtf8Buf::new();
                            msg.push_wtf8(&fname);
                            msg.push_str(&format!("() got multiple values for argument '{key}'"));
                            return Err(crate::PyError::type_error(msg));
                        }
                        result[pi] = value;
                        matched = true;
                        break;
                    }
                }
                if !matched {
                    if has_varkw {
                        extra_kwargs.push((key.clone(), value));
                    } else {
                        unmatched_kw_names.push(key.clone());
                    }
                }
            }

            // argument.py:499-500 — ArgErrPosonlyAsKwds, raised after the
            // full keyword scan and before ArgErrUnknownKwds.
            raise_if_posonly_kwds(&posonly_kwds, &fname)?;

            // `argument.py:270-271` ArgErrUnknownKwds.
            if !unmatched_kw_names.is_empty() {
                let msg = format_unknown_kwds_err(&fname, &unmatched_kw_names);
                return Err(crate::PyError::type_error(msg));
            }

            // `argument.py:289` — too-many-positionals raised here, after the
            // keyword-matching errors above.
            if too_many_args {
                let ndefaults = {
                    let defaults = unsafe { crate::function_get_defaults(current_callable()) };
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
                let mut msg = Wtf8Buf::new();
                msg.push_wtf8(&fname);
                msg.push_str(&format!("() takes {takes_str} but {given_str} given"));
                return Err(crate::PyError::type_error(msg));
            }

            // Fill positional defaults from __defaults__ tuple.
            let defaults = unsafe { crate::function_get_defaults(current_callable()) };
            if !defaults.is_null() && unsafe { pyre_object::is_tuple(defaults) } {
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
            // Fill keyword-only defaults from __kwdefaults__ dict.
            // function.py Function._apply_defaults — kw-only args take their
            // defaults from the kwdefaults dict by name lookup.
            let nkwonly = code.kwonlyarg_count as usize;
            if nkwonly > 0 {
                let kwdefaults = unsafe { crate::function_get_kwdefaults(current_callable()) };
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

            // Keep the matched parameter array live while packing *args and
            // **kwargs: both allocations can relocate values already copied
            // into `result`.
            let _bound_roots = pyre_object::gc_roots::push_roots();
            let bound_root_base = pyre_object::gc_roots::shadow_stack_len();
            for &value in &result {
                pyre_object::gc_roots::pin_root(value);
            }
            let mut packed_tail_slots = Vec::new();
            if has_varargs {
                let extra_pos: Vec<PyObjectRef> = if pos_args.len() > n_pos_params {
                    (n_pos_params..pos_args.len())
                        .map(current_pos_arg)
                        .collect()
                } else {
                    vec![]
                };
                let packed = pyre_object::w_tuple_new(extra_pos);
                let slot = pyre_object::gc_roots::shadow_stack_len();
                pyre_object::gc_roots::pin_root(packed);
                packed_tail_slots.push(slot);
            }
            if has_varkw {
                let kw_dict = pyre_object::w_dict_new();
                let kw_dict_slot = pyre_object::gc_roots::shadow_stack_len();
                pyre_object::gc_roots::pin_root(kw_dict);
                for (key, value) in &extra_kwargs {
                    unsafe {
                        // The key allocation runs first: as the second argument
                        // it would be evaluated after the receiver, handing
                        // `w_dict_store` the pre-collection dict address.
                        let w_key = pyre_object::w_str_from_wtf8_managed(key.clone());
                        pyre_object::w_dict_store(
                            pyre_object::gc_roots::shadow_stack_get(kw_dict_slot),
                            w_key,
                            pyre_object::gc_hook::try_gc_current_object_address(*value as *mut u8)
                                as PyObjectRef,
                        );
                    }
                }
                packed_tail_slots.push(kw_dict_slot);
            }
            let mut final_args: Vec<PyObjectRef> = (0..result.len())
                .map(|i| pyre_object::gc_roots::shadow_stack_get(bound_root_base + i))
                .collect();
            final_args.extend(
                packed_tail_slots
                    .iter()
                    .map(|&slot| pyre_object::gc_roots::shadow_stack_get(slot)),
            );

            // Create frame and execute
            let w_globals = unsafe { function_get_globals_obj(current_callable()) };
            let closure = unsafe { function_get_closure(current_callable()) };
            let mut func_frame = crate::pyframe::FrameBox::new(
                crate::pyframe::PyFrame::try_new_for_call_with_closure_and_globals_obj(
                    w_code,
                    &final_args,
                    w_globals,
                    execution_context,
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
                return frame_into_generator_for_function(func_frame, current_callable());
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
        // `typeobject.py::descr_call` keeps `self` live across the arbitrary
        // Python call to `__new__`.  The translated RPython GC transform
        // reloads it from the shadow stack afterwards; a Rust local would
        // otherwise retain the pre-move address of a heap type.
        let _type_root = pyre_object::gc_roots::push_roots();
        let type_slot = pyre_object::gc_roots::shadow_stack_len();
        pyre_object::gc_roots::pin_root(callable);
        let current_type = || pyre_object::gc_roots::shadow_stack_get(type_slot);
        if let Some(result) = type_call_special_case(current_type(), pos_args, !kwargs.is_empty()) {
            return result;
        }
        // Types with acceptable_as_base_class=false (bool, NoneType) reject kwargs.
        // PyPy: boolobject.py descr_new uses @unwrap_spec (positional only).
        // The `function`, `memoryview`, and deque iterator types are
        // non-acceptable-as-base too, but their `tp_new` functions accept
        // keywords: FunctionType has `kwdefaults=...`, CPython 3.14 exposes
        // `memoryview(object=...)`, and the deque iterator constructors accept
        // (and ignore) `index=...`.  Route them through `__new__`.
        let accepts_keywords_despite_nonbase =
            std::ptr::eq(
                current_type(),
                crate::typedef::gettypeobject(&crate::FUNCTION_TYPE),
            ) || std::ptr::eq(
                current_type(),
                crate::typedef::gettypeobject(&pyre_object::memoryview::MEMORYVIEW_TYPE),
            ) || std::ptr::eq(
                current_type(),
                crate::module::_collections::deque_iter::public_type(),
            ) || std::ptr::eq(
                current_type(),
                crate::module::_collections::deque_rev_iter::public_type(),
            ) || std::ptr::eq(
                current_type(),
                crate::module::_contextvars::context_var_type(),
            ) || crate::_structseq::is_structseq_type(current_type());
        if !kwargs.is_empty()
            && !accepts_keywords_despite_nonbase
            && !unsafe { pyre_object::w_type_get_acceptable_as_base_class(current_type()) }
        {
            let type_name = unsafe { pyre_object::w_type_get_name(current_type()) };
            return Err(crate::PyError::type_error(format!(
                "{}() takes no keyword arguments",
                type_name,
            )));
        }
        // Three-argument `type(name, bases, namespace, **kw)` must select the
        // winning metaclass.  Keep the check at the actual type-construction
        // shape: the former `len >= 2 && args[1] is tuple` condition also
        // captured ordinary constructors such as
        // `_GenericAlias(origin, args, **kw)`.
        let w_metaclass = if pos_args.len() >= 3
            && unsafe { pyre_object::is_str(current_pos_arg(0)) }
            && unsafe { pyre_object::is_tuple(current_pos_arg(1)) }
        {
            calculate_metaclass(current_type(), current_pos_arg(1)).unwrap_or(current_type())
        } else {
            current_type()
        };
        let metaclass_slot = pyre_object::gc_roots::shadow_stack_len();
        pyre_object::gc_roots::pin_root(w_metaclass);
        let current_metaclass = || pyre_object::gc_roots::shadow_stack_get(metaclass_slot);
        // Step 1: __new__(cls, *args, **kwargs)
        let instance = if let Some(new_fn) =
            unsafe { crate::baseobjspace::lookup_in_type(current_metaclass(), "__new__") }
        {
            let new_fn = unsafe { unwrap_static_new(new_fn) };
            let mut new_args = Vec::with_capacity(1 + pos_args.len());
            // `lookup_in_type` interns its name and can collect; reload the
            // winning metaclass and the arguments rather than retaining their
            // pre-lookup addresses.
            new_args.push(current_metaclass());
            extend_current_args(&mut new_args);
            if unsafe { crate::is_function(new_fn) } && !kwargs.is_empty() {
                call_with_kwargs_in_ctx(execution_context, new_fn, &new_args, &current_kwargs())?
            } else {
                call_callable_in_ctx(execution_context, new_fn, &new_args)?
            }
        } else {
            pyre_object::w_instance_new(current_type())
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
        if let Some(w_insttype) = type_call_init_type(
            pyre_object::gc_roots::shadow_stack_get(instance_slot),
            current_type(),
        ) && let Some(init_descr) =
            unsafe { crate::baseobjspace::lookup_in_type(w_insttype, "__init__") }
        {
            // typeobject.py:737-740 `space.get_and_call_args`: exact
            // Function takes the instance explicitly; every other descriptor
            // binds itself and receives only the original constructor args.
            let init_result = if unsafe { crate::is_function(init_descr) } {
                let mut init_args = Vec::with_capacity(1 + pos_args.len());
                // The `__new__` result and the constructor arguments are
                // movable nursery objects, and `__new__` has just run
                // arbitrary allocating code. Reload every one of them from
                // the roots instead of retaining the pre-collection locals
                // across argument binding and descriptor dispatch.
                init_args.push(pyre_object::gc_roots::shadow_stack_get(instance_slot));
                extend_current_args(&mut init_args);
                call_with_kwargs_in_ctx(
                    execution_context,
                    init_descr,
                    &init_args,
                    &current_kwargs(),
                )?
            } else {
                let init_fn = unsafe {
                    crate::baseobjspace::get(
                        init_descr,
                        pyre_object::gc_roots::shadow_stack_get(instance_slot),
                        w_insttype,
                    )?
                }
                .unwrap_or(init_descr);
                // Binding the descriptor allocates, so the arguments are
                // reloaded after it rather than before.
                let mut init_args = Vec::with_capacity(pos_args.len());
                extend_current_args(&mut init_args);
                call_with_kwargs_in_ctx(execution_context, init_fn, &init_args, &current_kwargs())?
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
        return call_with_kwargs_in_ctx(execution_context, func, &full_args, kwargs);
    }

    if let Some((call_fn, prepend_receiver)) = user_call_slot(current_callable())? {
        // `user_call_slot` may run a descriptor and collect.  Rebuild the
        // Arguments view from the roots installed at function entry instead
        // of forwarding the stale incoming slices.
        if prepend_receiver {
            let mut call_args = Vec::with_capacity(1 + pos_args.len());
            call_args.push(current_callable());
            extend_current_args(&mut call_args);
            return call_with_kwargs_in_ctx(
                execution_context,
                call_fn,
                &call_args,
                &current_kwargs(),
            );
        }
        // Depth guard: count this dispatch level and, dropping after the call,
        // keep it off the tail so a self-referential `A.__call__ = A()`
        // recurses natively for stack_check (see call_callable_with_mode).
        let _depth_guard = enter_native_dispatch();
        let mut call_args = Vec::with_capacity(pos_args.len());
        extend_current_args(&mut call_args);
        return call_with_kwargs_in_ctx(execution_context, call_fn, &call_args, &current_kwargs());
    }

    // GenericAlias.__call__ (`_pypy_generic_alias.py:41`) —
    // `self.__origin__(*args, **kwargs)`, then best-effort
    // `result.__orig_class__ = self`.
    if unsafe { pyre_object::is_generic_alias(callable) } {
        let origin = unsafe { pyre_object::w_generic_alias_get_origin(callable) };
        let result = call_with_kwargs_in_ctx(execution_context, origin, pos_args, kwargs)?;
        set_orig_class(result, callable)?;
        return Ok(result);
    }

    // Fallback: call_callable with positional args only
    call_callable_in_ctx(execution_context, callable, pos_args)
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
            log_call_error(&e.message_text());
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
    let root_base = _roots.base();
    _roots.pin_root(callable);
    for &arg in args {
        _roots.pin_root(arg);
    }

    // A JIT prologue may have published an overflow before entering this
    // residual dispatcher, so preserve that pending exception.  Do not run a
    // fresh stack check here: RPython's insert_ll_stackcheck places checks on
    // recursive graph entries (PyPy marks PyFrame.execute_frame), not in front
    // of every ObjSpace call.  In particular, CheckSignalAction must still be
    // able to invoke the non-recursive default_int_handler while the current
    // Python frame is handling a RecursionError.  Python frame call paths carry
    // their stack check in funccall_valuestack / the JIT callee prologue.
    crate::stack_check::drain_jit_pending_exception()?;

    let callable = _roots.get(root_base);
    // RPython's GC transform reloads `Arguments.arguments_w` livevars in
    // place; it does not allocate another list at every ObjSpace call.  Keep
    // the common small call shape allocation-free and retain a Vec only for
    // genuinely wide calls.
    const INLINE_ARGS: usize = 8;
    let mut inline_args = [PY_NULL; INLINE_ARGS];
    let mut wide_args = Vec::new();
    let args = if args.len() <= INLINE_ARGS {
        // The index loop lowers to `setarrayitem`; iterator adapters are residual calls.
        #[allow(clippy::needless_range_loop)]
        for i in 0..args.len() {
            inline_args[i] = _roots.get(root_base + 1 + i);
        }
        &inline_args[..args.len()]
    } else {
        wide_args = Vec::with_capacity(args.len());
        for i in 0..args.len() {
            wide_args.push(_roots.get(root_base + 1 + i));
        }
        wide_args.as_slice()
    };

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
        if crate::is_function_carrier(callable) {
            if crate::is_slot_wrapper(callable)
                && let Some(&receiver) = args.first()
            {
                crate::typedef::slot_wrapper_check_instance(callable, receiver)?;
            }
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
            if result.is_null()
                && let Some(err) = take_call_error()
            {
                return Err(err);
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
            if result.is_null()
                && let Some(err) = take_call_error()
            {
                return Err(err);
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
        if let Some((call_fn, prepend_receiver)) =
            user_call_slot(pyre_object::gc_roots::shadow_stack_get(root_base))?
        {
            // Binding a custom descriptor can collect.  The entry roots above
            // have been updated to forwarded addresses; reconstruct the
            // argument view before recursively dispatching the bound call.
            let current_callable = pyre_object::gc_roots::shadow_stack_get(root_base);
            let mut current_args: Vec<PyObjectRef> = Vec::with_capacity(args.len());
            for i in 0..args.len() {
                current_args.push(pyre_object::gc_roots::shadow_stack_get(root_base + 1 + i));
            }
            if prepend_receiver {
                let mut call_args = Vec::with_capacity(1 + current_args.len());
                call_args.push(current_callable);
                call_args.extend_from_slice(&current_args);
                return call_function_impl_result(call_fn, &call_args);
            }
            // Depth guard: count this dispatch level and, dropping after the
            // call, keep it off the tail so a self-referential
            // `A.__call__ = A()` recurses natively for stack_check.
            let _depth_guard = enter_native_dispatch();
            return call_function_impl_result(call_fn, &current_args);
        }
    }
    let type_name = crate::typedef::r#type(callable)
        .map(|tp| unsafe { pyre_object::w_type_get_name(tp.as_ptr()) })
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
        let w_base_type = unsafe {
            if pyre_object::is_type(base) && !(*base).w_class.is_null() {
                Some((*base).w_class)
            } else {
                crate::typedef::r#type(base).map(|ty| ty.as_ptr())
            }
        };
        let Some(w_base_type) = w_base_type else {
            continue;
        };
        if std::ptr::eq(w_winner, w_base_type) || issubtype_ptr(w_winner, w_base_type) {
            continue;
        }
        if issubtype_ptr(w_base_type, w_winner) {
            w_winner = w_base_type;
            continue;
        }
        return Err(PyError::type_error(
            "metaclass conflict: the metaclass of a derived class must be a \
             (non-strict) subclass of the metaclasses of all its bases",
        ));
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
    if result.is_null()
        && let Some(err) = take_call_error()
    {
        return Err(err);
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
    // `Arguments.prepend` builds one list per call, so both call sites below
    // fill their argument vector straight from the pinned slots — reloading
    // into a throwaway `Vec` first would allocate a second one per call.
    let extend_current_args = |dst: &mut Vec<PyObjectRef>| {
        for i in 0..args.len() {
            dst.push(pyre_object::gc_roots::shadow_stack_get(root_base + 1 + i));
        }
    };

    if let Some(result) = type_call_special_case(current_type(), args, false) {
        return match result {
            Ok(value) => value,
            Err(error) => {
                set_call_error(error);
                PY_NULL
            }
        };
    }

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
        extend_current_args(&mut new_args);
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
    ) && let Some(init_fn) =
        unsafe { crate::baseobjspace::lookup_in_type(w_insttype, "__init__") }
    {
        let mut init_args = Vec::with_capacity(1 + args.len());
        init_args.push(pyre_object::gc_roots::shadow_stack_get(instance_slot));
        extend_current_args(&mut init_args);
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
        .map(|t| unsafe { pyre_object::w_type_get_name(t.as_ptr()) })
        .unwrap_or("object");
    Err(PyError::type_error(format!(
        "__init__() should return None, not '{tname}'"
    )))
}

fn type_call_init_type(instance: PyObjectRef, w_type: PyObjectRef) -> Option<PyObjectRef> {
    let w_insttype = crate::typedef::r#type(instance)?;
    if std::ptr::eq(w_insttype.as_ptr(), w_type) || issubtype_ptr(w_insttype.as_ptr(), w_type) {
        Some(w_insttype.as_ptr())
    } else {
        None
    }
}

/// CPython 3.14 `type_call` / `type_vectorcall` — exact `type` owns the
/// one-argument query form.  `type.__new__` itself accepts only the three
/// class-construction arguments.
fn type_call_special_case(
    w_type: PyObjectRef,
    args: &[PyObjectRef],
    has_kwargs: bool,
) -> Option<PyResult> {
    if !std::ptr::eq(w_type, crate::typedef::w_type()) {
        return None;
    }
    if args.len() == 1 {
        if has_kwargs {
            return Some(Err(PyError::type_error(
                "type() takes no keyword arguments",
            )));
        }
        return Some(Ok(crate::builtins::type_of_object(args[0])));
    }
    if args.len() != 3 {
        return Some(Err(PyError::type_error("type() takes 1 or 3 arguments")));
    }
    None
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
    let exec_ctx = take_last_exec_ctx();

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
        return match frame_into_generator_for_function(gen_frame, func) {
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
    let exec_ctx = take_last_exec_ctx();
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
        return match frame_into_generator_for_function(frame, func) {
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
        // compiling.py:215-221 — `space.call_args(w_meta, Arguments(name,
        // bases, ns, **kwds))`; a non-type metaclass receives the
        // class-definition keywords too, and `call_args`
        // (descroperation.py:189) takes no frame.
        let kwds: Vec<(Wtf8Buf, PyObjectRef)> = if unsafe { pyre_object::is_dict(kwargs) } {
            unsafe { pyre_object::w_dict_str_entries_wtf8(kwargs) }
        } else {
            Vec::new()
        };
        if !kwds.is_empty() {
            return match call_with_kwargs_in_ctx(
                take_last_exec_ctx(),
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
            "__build_class__: not enough arguments",
        ));
    }
    let body_fn = args[0];
    let name_obj = args[1];

    // compiling.py:163-167 — the body must be a Python function carrying a
    // `PyCode`.  Its code object is read directly below, so anything else is
    // rejected here rather than reaching that read.
    if !unsafe { crate::is_function(body_fn) }
        || unsafe { crate::function_has_builtin_code(body_fn) }
    {
        return Err(crate::PyError::type_error(
            "__build_class__: func must be a function",
        ));
    }

    // Check if last arg is a kwargs dict (from CALL_KW)
    // PyPy: __build_class__(func, name, *bases, metaclass=None, **kwds)
    //
    // The class-definition keywords are collected into a fresh dict that only
    // `build_class_inner` consumes, so the guard is opened before the arm that
    // fills it: `update_bases` and both `w_tuple_new` calls run between the
    // two, and a guard scoped to the arm would unpin the dict across them.
    // `build_class_inner` re-pins its own parameter copy. A class statement
    // without keywords opens no scope at all — `pin_root` is
    // `dont_look_inside`, so an unconditional one would residualise in every
    // traced class body.
    let kwds_dict = if args.len() > 2 {
        let last = args[args.len() - 1];
        let is_kwds = unsafe { pyre_object::is_dict(last) }
            && unsafe {
                pyre_object::w_dict_getitem_str(last, "__pyre_kw__")
                    .is_some_and(pyre_object::kw_marker::is_kw_marker_sentinel)
            };
        is_kwds.then_some(last)
    } else {
        None
    };
    let extra_roots = kwds_dict.map(|_| pyre_object::gc_roots::push_roots());
    let (base_args, metaclass, extra_kwargs) = if let Some(last) = kwds_dict {
        {
            let extra_roots = extra_roots.as_ref().expect("opened for a kwargs dict");
            let extra_slot = extra_roots.base();
            let w_metaclass = unsafe { pyre_object::w_dict_getitem_str(last, "metaclass") };
            // Collect extra kwargs (not metaclass, not __pyre_kw__).
            // `w_dict_items` already dispatches `is_module_dict` so a
            // class statement with `**module_dict` (rare but valid)
            // walks the strategy.
            // Born young, and `w_dict_store` allocates on strategy promotion —
            // the bracket in `call_with_kwargs`.
            extra_roots.pin_root(pyre_object::w_dict_new());
            unsafe {
                for (k, v) in pyre_object::w_dict_items(last) {
                    if pyre_object::is_str(k) {
                        let key = pyre_object::w_str_get_wtf8(k).as_str();
                        if key != Ok("metaclass") && key != Ok("__pyre_kw__") {
                            pyre_object::w_dict_store(extra_roots.get(extra_slot), k, v);
                        }
                    }
                }
            }
            (
                &args[2..args.len() - 1],
                w_metaclass,
                Some(extra_roots.get(extra_slot)),
            )
        }
    } else {
        (&args[2..], None, None)
    };

    let name = unsafe { pyre_object::w_str_get_value(name_obj) };
    // compiling.py:166-167 — resolve __mro_entries__ before metaclass
    // inference; record the original bases for __orig_bases__ when changed.
    let w_orig_bases = pyre_object::w_tuple_new(base_args.to_vec());
    let (resolved_bases, bases_changed) = update_bases(base_args, w_orig_bases)?;
    // Non-type bases are not rejected here: `__build_class__` hands the
    // resolved tuple to whichever metaclass was selected, and a metaclass
    // that is not a type may legitimately accept them.  `best_base` performs
    // the `bases must be types` check on the type-construction path.
    let bases_tuple = pyre_object::w_tuple_new(resolved_bases);
    let w_orig_bases = if bases_changed {
        Some(w_orig_bases)
    } else {
        None
    };

    // compiling.py:169-183 — choose the initial metaclass from the first
    // resolved base (or `type` for an empty base list), then run
    // `_calculate_metaclass` across *every* base.  In particular a non-type
    // base without `__mro_entries__` still contributes `space.type(base)`:
    // `(object, None)` therefore raises the metaclass conflict instead of
    // slipping through the default-type fast path.
    //
    // `None` remains pyre's internal spelling for the exact builtin `type`
    // winner, so `build_class_inner` can retain its raw default construction
    // path without changing the app-level algorithm.
    let w_type_type = crate::typedef::w_type();
    let w_metaclass = match metaclass {
        Some(w_meta) if unsafe { pyre_object::is_type(w_meta) } => {
            Some(calculate_metaclass(w_meta, bases_tuple)?)
        }
        Some(w_meta) => Some(w_meta),
        None => {
            let initial = unsafe {
                pyre_object::w_tuple_getitem(bases_tuple, 0)
                    .and_then(crate::typedef::r#type)
                    .map(|w_type| w_type.as_ptr())
                    .unwrap_or(w_type_type)
            };
            let winner = calculate_metaclass(initial, bases_tuple)?;
            if std::ptr::eq(winner, w_type_type) {
                None
            } else {
                Some(winner)
            }
        }
    };

    build_class_inner(
        body_fn,
        name,
        bases_tuple,
        w_metaclass,
        extra_kwargs.map(|_| {
            let roots = extra_roots.as_ref().expect("opened for a kwargs dict");
            roots.get(roots.base())
        }),
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
    // The class-definition keywords survive the class body call and the whole
    // metaclass construction below, and the parameter is a raw copy taken
    // before any of it. Pin it once and read it back at each of the three
    // sites that consume it. A class statement without keywords opens no
    // scope: `pin_root` is `dont_look_inside`, so an unconditional one would
    // residualise in every traced class body.
    let kwds_roots = extra_kwargs.map(|kw| {
        let scope = pyre_object::gc_roots::push_roots();
        let slot = scope.base();
        scope.pin_root(kw);
        (scope, slot)
    });
    let current_kwds = || kwds_roots.as_ref().map(|(scope, slot)| scope.get(*slot));

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
                // compiling.py:194-199 — call __prepare__ with the
                // class-definition keywords ('metaclass' already popped by
                // the caller), through the frameless `space.call_args`.
                let prepare_kwds: Vec<(Wtf8Buf, PyObjectRef)> = match current_kwds() {
                    Some(kw) if unsafe { pyre_object::is_dict(kw) } => unsafe {
                        pyre_object::w_dict_str_entries_wtf8(kw)
                    },
                    _ => Vec::new(),
                };
                let ns_obj = if !prepare_kwds.is_empty() {
                    call_with_kwargs_in_ctx(
                        take_last_exec_ctx(),
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
                            Some(tp) => pyre_object::w_type_get_name(tp.as_ptr()).to_string(),
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

    let exec_ctx = take_last_exec_ctx();

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
    // used directly (already rooted by the caller); otherwise class_ns itself
    // is the body namespace.  The latter identity is semantic: compiler-made
    // `__classdict__` cells close over class_ns, so STORE_NAME must update that
    // same object rather than a temporary copy.
    let _ns_root = pyre_object::gc_roots::push_roots();
    let (body_ns, body_ns_root): (PyObjectRef, Option<usize>) = match mapping_namespace {
        Some(w_ns) => (w_ns, None),
        None => {
            let class_ns = pyre_object::gc_roots::shadow_stack_get(class_ns_root);
            (class_ns, Some(class_ns_root))
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
        let class_ns = pyre_object::gc_roots::shadow_stack_get(class_ns_root);
        // A plain class body now writes directly into class_ns, preserving the
        // `__classdict__` closure identity.  Only a distinct custom prepared
        // mapping needs to be mirrored for downstream type construction.
        let distinct_namespace = !std::ptr::eq(w_ns, class_ns);
        if distinct_namespace {
            // Rebuild from the final contents so names `del`eted from the
            // namespace during body execution don't survive in class_ns.
            unsafe { pyre_object::w_dict_clear(class_ns) };
        }
        let backing = crate::type_methods::resolve_dict_backing(w_ns);
        if distinct_namespace && !backing.is_null() && unsafe { pyre_object::is_dict(backing) } {
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
        } else if distinct_namespace && w_metaclass.is_some() {
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
        } else if distinct_namespace {
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
    if w_metaclass.is_none()
        && classcell.is_some_and(|value| unsafe { !pyre_object::is_cell(value) })
    {
        let value = classcell.unwrap();
        return Err(PyError::type_error(format!(
            "__classcell__ must be a nonlocal cell, not {}",
            unsafe { pyre_object::type_name_of(value) },
        )));
    }
    // CPython 3.14 type_new_set_classdict: compiler-generated annotation
    // functions and comprehensions capture `__classdict__` through this
    // separate cell.  type.__new__ consumes the namespace entry and replaces
    // the cell's provisional body namespace with the completed type dict.
    let classdictcell_root = {
        let class_ns = pyre_object::gc_roots::shadow_stack_get(class_ns_root);
        let cell = unsafe { pyre_object::w_dict_getitem_str(class_ns, "__classdictcell__") };
        cell.filter(|value| unsafe { pyre_object::is_cell(*value) })
            .map(|cell| {
                pyre_object::gc_roots::pin_root(cell);
                pyre_object::gc_roots::shadow_stack_len() - 1
            })
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
    // The C3 validations read `__bases__` off classic bases and
    // `create_all_slots` unpacks `__slots__`; both execute Python, so the
    // tuple cannot stay in an untraced local across them.
    let bases_root = pyre_object::gc_roots::shadow_stack_len();
    pyre_object::gc_roots::pin_root(w_effective_bases);
    // A custom metaclass owns its bases until (and unless) it invokes
    // type.__new__; do not perform type's C3 validation before dispatch.
    if w_metaclass.is_none() {
        let w_effective_bases = pyre_object::gc_roots::shadow_stack_get(bases_root);
        unsafe { crate::baseobjspace::validate_c3_mro(w_effective_bases, false)? };
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
        let result = if let Some(kw) = current_kwds() {
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
        // compiling.py:224 `w_class = space.call_args(w_meta, args)` returns
        // the metaclass result unchanged. `type.__new__` owns classcell, MRO,
        // ready(), and metaclass identity; a custom metaclass that bypasses
        // type.__new__ must not be repaired or overwritten here.
        result
    } else {
        // typeobject.py:1554 `ensure_common_attributes` belongs to type
        // construction, not to `__build_class__` namespace preparation.  A
        // custom non-type metaclass must observe the compiler-produced
        // namespace without an invented `__doc__` key; the default shortcut
        // performs type.__new__'s step here before copying the type dict.
        let class_ns = pyre_object::gc_roots::shadow_stack_get(class_ns_root);
        crate::builtins::type_new_set_doc(class_ns)?;
        // No metaclass observes the namespace on the default path, so
        // consume the explicit class cells here (type_new_classcell leaves
        // them out of the class `__dict__`); the captured `classcell` is
        // bound to the new type below.
        // typeobject.py `_store_type_in_classcell` validates the value before
        // deleting it.  The default-metaclass shortcut bypasses
        // `type.__new__`, so it must preserve that check here too.
        if let Some(w_classcell) = classcell
            && !unsafe { pyre_object::is_cell(w_classcell) }
        {
            return Err(PyError::type_error(format!(
                "__classcell__ must be a nonlocal cell, not {}",
                crate::baseobjspace::object_functionstr_type_name(w_classcell),
            )));
        }
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
        // Slot creation, C3 validation and `__set_name__` below all allocate
        // and can execute Python; the nascent type has no other referrer
        // until its mro is installed, so keep it rooted and reread it after
        // every such step.
        let w_root = pyre_object::gc_roots::shadow_stack_len();
        let w = pyre_object::w_type_new(
            name,
            pyre_object::gc_roots::shadow_stack_get(bases_root),
            dict_obj as *mut u8,
        );
        pyre_object::gc_roots::pin_root(w);
        crate::builtins::type_new_take_qualname(w, dict_obj)?;
        // typeobject.py:1143-1204 create_all_slots parity.
        unsafe { create_all_slots(w, pyre_object::gc_roots::shadow_stack_get(bases_root))? };
        // baseobjspace.py:76 — set w_class to 'type' (default metaclass)
        let w = pyre_object::gc_roots::shadow_stack_get(w_root);
        unsafe {
            (*w).w_class = crate::typedef::w_type();
        }
        // typeobject.py:1560 `compute_mro(w_self)`, reached only once
        // `check_and_find_best_base` inside `create_all_slots` above accepted
        // the tuple.  `compute_default_mro` cannot raise, so `get_mro`'s
        // classic branch runs through the fallible validation here.
        unsafe {
            crate::baseobjspace::validate_c3_mro(
                pyre_object::gc_roots::shadow_stack_get(bases_root),
                true,
            )?
        };
        let w = pyre_object::gc_roots::shadow_stack_get(w_root);
        let mro = unsafe { crate::baseobjspace::compute_default_mro(w) };
        unsafe { pyre_object::w_type_set_mro(w, mro) };
        // typeobject.py:373-377 ready() — register self on each base's
        // `weak_subclasses` so cross-subclass invalidation in
        // `mutated()` and `__subclasses__()` see this class.
        unsafe { pyre_object::typeobject::w_type_ready(w) };
        // CPython type_new_set_classdict binds __classdict__ before
        // type_new_set_names.  A descriptor's __set_name__ may mutate the
        // completed type dict and immediately materialize lazy annotations;
        // the annotation thunk must observe that final dict, not the
        // provisional class-body namespace.
        if let Some(classdictcell_root) = classdictcell_root {
            let classdictcell = pyre_object::gc_roots::shadow_stack_get(classdictcell_root);
            let w = pyre_object::gc_roots::shadow_stack_get(w_root);
            let type_dict = unsafe { pyre_object::w_type_get_dict_ptr(w) as PyObjectRef };
            if !type_dict.is_null() {
                unsafe { pyre_object::w_cell_set(classdictcell, type_dict) };
            }
        }
        // __set_name__ protocol — type_new_set_names
        // Only needed here because w_type_new is a raw Rust call that
        // bypasses the type() builtin (builtins.rs) which already calls
        // __set_name__. The metaclass path above goes through type.__new__()
        // which handles __set_name__ in builtins.rs, so we must NOT call it
        // again there to avoid double invocation.
        if unsafe { pyre_object::is_type(pyre_object::gc_roots::shadow_stack_get(w_root)) } {
            let dict_obj = pyre_object::gc_roots::shadow_stack_get(dict_root);
            let entries = unsafe { pyre_object::w_dict_items(dict_obj) };
            // Every `__set_name__` runs Python, so the snapshot cannot stay in
            // an untraced Vec across the loop.
            let _entry_roots = pyre_object::gc_roots::push_roots();
            let entries_root = pyre_object::gc_roots::shadow_stack_len();
            let mut pinned = 0;
            for (w_name, value) in entries {
                if !value.is_null() && unsafe { pyre_object::is_str(w_name) } {
                    pyre_object::gc_roots::pin_root(w_name);
                    pyre_object::gc_roots::pin_root(value);
                    pinned += 1;
                }
            }
            for i in 0..pinned {
                let w_name = pyre_object::gc_roots::shadow_stack_get(entries_root + i * 2);
                let value = pyre_object::gc_roots::shadow_stack_get(entries_root + i * 2 + 1);
                let w = pyre_object::gc_roots::shadow_stack_get(w_root);
                unsafe { crate::baseobjspace::set_name(w, w_name, value) }?;
            }
        }
        pyre_object::gc_roots::shadow_stack_get(w_root)
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
    if let Some(classcell) = classcell
        && !classcell.is_null()
        && unsafe { pyre_object::is_cell(classcell) }
    {
        if w_metaclass.is_some() && unsafe { pyre_object::is_type(w_type) } {
            let cell_value = unsafe { pyre_object::w_cell_get(classcell) };
            if cell_value.is_null() {
                let class_str = unsafe { crate::py_str_wtf8(w_type) }?;
                return Err(PyError::runtime_error(crate::display::wtf8_format!(
                    format!("__class__ not set defining {name} as "),
                    class_str,
                    ". Was __classcell__ propagated to type.__new__?",
                )));
            }
            if !std::ptr::eq(cell_value, w_type) {
                let cell_str = unsafe { crate::py_str_wtf8(cell_value) }?;
                let class_str = unsafe { crate::py_str_wtf8(w_type) }?;
                return Err(PyError::type_error(crate::display::wtf8_format!(
                    "__class__ set to ",
                    cell_str,
                    format!(" defining {name} as "),
                    class_str,
                )));
            }
        } else {
            unsafe { pyre_object::w_cell_set(classcell, w_type) };
        }
    }

    // The default path bound this before __set_name__; the metaclass path
    // normally binds it inside type.__new__.  Retain this final validation/
    // fallback for a custom metaclass returning a type without delegating.
    if let Some(classdictcell_root) = classdictcell_root
        && unsafe { pyre_object::is_type(w_type) }
    {
        let classdictcell = pyre_object::gc_roots::shadow_stack_get(classdictcell_root);
        let type_dict = unsafe { pyre_object::w_type_get_dict_ptr(w_type) as PyObjectRef };
        if !type_dict.is_null() {
            unsafe { pyre_object::w_cell_set(classdictcell, type_dict) };
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
        let init_subclass_kwargs: Vec<(PyObjectRef, PyObjectRef)> = match current_kwds() {
            Some(kw) if unsafe { pyre_object::is_dict(kw) } => unsafe {
                pyre_object::w_dict_items(kw)
                    .into_iter()
                    .filter(|(k, _)| pyre_object::is_str(*k))
                    .collect()
            },
            _ => Vec::new(),
        };
        call_init_subclass_on_bases(
            w_type,
            pyre_object::gc_roots::shadow_stack_get(bases_root),
            &init_subclass_kwargs,
        )?;
    }

    Ok(w_type)
}

/// Pack `(name, value)` keyword pairs into the `__pyre_kw__`-tagged
/// trailing dict that the builtin kwargs ABI (`split_builtin_kwargs`)
/// consumes.  Mirrors the producer in `call_with_kwargs`.
fn pack_pyre_kwargs(kw_items: &[(PyObjectRef, PyObjectRef)]) -> PyObjectRef {
    // The dict is born young, and `w_dict_store` allocates when it promotes the
    // strategy — see the bracket in `call_with_kwargs`. `kw_items` is the
    // caller's array of raw references, so the pairs still to be installed are
    // pinned as well: the first promotion relocates every one of them.
    let kw_roots = pyre_object::gc_roots::push_roots();
    let item_slot = kw_roots.base();
    for &(k, v) in kw_items {
        kw_roots.pin_root(k);
        kw_roots.pin_root(v);
    }
    let kw_slot = item_slot + kw_items.len() * 2;
    kw_roots.pin_root(pyre_object::w_dict_new());
    unsafe {
        for i in 0..kw_items.len() {
            pyre_object::w_dict_store(
                kw_roots.get(kw_slot),
                kw_roots.get(item_slot + i * 2),
                kw_roots.get(item_slot + i * 2 + 1),
            );
        }
        // Marker stored last so a user keyword named `__pyre_kw__` cannot
        // overwrite the sentinel detection compares by identity.
        let marker_key = pyre_object::kw_marker::w_kw_marker_key();
        let marker_value = pyre_object::kw_marker::w_kw_marker_sentinel();
        pyre_object::w_dict_store(kw_roots.get(kw_slot), marker_key, marker_value);
    }
    kw_roots.get(kw_slot)
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
    // typeobject.py:1022 `space.call_function(w_super, w_type, w_type)`
    // goes through `super.__new__` / `_super_check` before constructing the
    // proxy.  This matters for a custom metaclass mro() that omits the
    // nascent class: `super(w_type, w_type)` must reject that incomplete
    // hierarchy instead of manufacturing an invalid proxy.
    let w_objtype = crate::builtins::super_check(w_type, w_type)?;
    let w_super = pyre_object::descriptor::w_super_new(w_type, w_objtype, w_type);
    let w_func = crate::baseobjspace::getattr_str(w_super, "__init_subclass__")?;
    // typeobject.py:1025-1026 — `args = __args__.replace_arguments([])` then
    // `space.call_args(w_func, args)`: keywords only, no positionals, and no
    // frame, because `call_args` (descroperation.py:189) never takes one.
    let kwds: Vec<(Wtf8Buf, PyObjectRef)> = init_subclass_kwargs
        .iter()
        .filter(|(k, _)| unsafe { pyre_object::is_str(*k) })
        .map(|(k, v)| (unsafe { pyre_object::w_str_get_wtf8(*k) }.to_owned(), *v))
        .collect();
    call_with_kwargs_in_ctx(take_last_exec_ctx(), w_func, &[], &kwds)?;
    Ok(())
}

// ── Type calling (instance creation) ─────────────────────────────────
// PyPy equivalent: typeobject.py descr_call → __new__ + __init__

fn type_descr_call_with_mode(
    execution_context: *const crate::PyExecutionContext,
    w_type: PyObjectRef,
    args: &[PyObjectRef],
    mode: CallMode,
) -> PyResult {
    if let Some(result) = type_call_special_case(w_type, args, false) {
        return result;
    }
    // `typeobject.py:731,738-739` threads one `__args__` through `__new__` and
    // then `__init__`; it needs no reload because `Arguments.arguments_w` is a
    // traced list the moving GC updates in place. This slice is a raw copy the
    // collector cannot see, so pin the type and every argument here and read
    // them back from the shadow stack after `__new__` has run Python code.
    let _roots = pyre_object::gc_roots::push_roots();
    let root_base = pyre_object::gc_roots::shadow_stack_len();
    pyre_object::gc_roots::pin_root(w_type);
    for &arg in args {
        pyre_object::gc_roots::pin_root(arg);
    }
    let current_type = || pyre_object::gc_roots::shadow_stack_get(root_base);
    let extend_current_args = |dst: &mut Vec<PyObjectRef>| {
        for index in 0..args.len() {
            dst.push(pyre_object::gc_roots::shadow_stack_get(
                root_base + 1 + index,
            ));
        }
    };

    check_type_instantiable(current_type())?;
    // Step 1: Look up __new__ via type MRO → allocate instance.
    // PyPy: typeobject.py descr_call → `w_newtype, w_newdescr =
    // self.lookup_where('__new__')`; a missing descriptor (the pathological
    // mro-without-object case) raises, otherwise the descriptor is bound via
    // `space.get(w_newdescr, space.w_None, w_type=self)` and called with
    // w_type as the first arg.
    let Some(new_descr) =
        (unsafe { crate::baseobjspace::lookup_in_type(current_type(), "__new__") })
    else {
        // typeobject.py:715 — `raise oefmt(space.w_TypeError,
        // "cannot create '%N' instances", self)`.
        let name = unsafe { pyre_object::w_type_get_name(current_type()) };
        return Err(crate::PyError::type_error(format!(
            "cannot create '{name}' instances"
        )));
    };
    // typeobject.py:726 — `w_newfunc = space.get(w_newdescr, space.w_None,
    // w_type=self)`.  A descriptor with no __get__ (`get` → None) is its own
    // bound value, matching `space.get`'s `if w_get is None: return w_descr`.
    let new_fn =
        unsafe { crate::baseobjspace::get(new_descr, pyre_object::PY_NULL, current_type())? }
            .unwrap_or(new_descr);
    // typeobject.py:731 — `space.call_obj_args(w_newfunc, self, __args__)`.
    let mut new_args = Vec::with_capacity(1 + args.len());
    new_args.push(current_type());
    extend_current_args(&mut new_args);
    let instance = call_callable_with_mode(execution_context, new_fn, &new_args, mode)?;
    let _instance_roots = pyre_object::gc_roots::push_roots();
    let instance_slot = pyre_object::gc_roots::shadow_stack_len();
    pyre_object::gc_roots::pin_root(instance);
    let current_instance = || pyre_object::gc_roots::shadow_stack_get(instance_slot);

    // Step 2: __init__ — only if __new__ returned an instance of w_type.
    // PyPy: descr_call — skips __init__ when __new__ returns a foreign type.
    if let Some(w_insttype) = type_call_init_type(current_instance(), current_type())
        && let Some(init_descr) =
            unsafe { crate::baseobjspace::lookup_in_type(w_insttype, "__init__") }
    {
        let init_result = if unsafe { crate::is_function(init_descr) } {
            let mut init_args = Vec::with_capacity(1 + args.len());
            init_args.push(current_instance());
            extend_current_args(&mut init_args);
            call_callable_with_mode(execution_context, init_descr, &init_args, mode)?
        } else {
            let init_fn =
                unsafe { crate::baseobjspace::get(init_descr, current_instance(), w_insttype)? }
                    .unwrap_or(init_descr);
            // Binding the descriptor allocates, so the arguments are reloaded
            // after it rather than before.
            let mut init_args = Vec::with_capacity(args.len());
            extend_current_args(&mut init_args);
            call_callable_with_mode(execution_context, init_fn, &init_args, mode)?
        };
        check_init_returned_none(init_result)?;
    }

    Ok(current_instance())
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
pub(crate) fn collect_slot_names(
    w_slots: pyre_object::PyObjectRef,
) -> Result<Vec<String>, crate::PyError> {
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
            // A name with no UTF-8 form is not an identifier, so it takes the
            // same rejection as `('1a',)` rather than aborting for want of a
            // `&str` view: `__slots__ = ('\udc80',)` is a `TypeError`.
            let Some(slot_name) = (unsafe { pyre_object::w_str_get_value_opt(w_slot_name) }) else {
                return Err(crate::PyError::type_error(
                    "__slots__ must be identifiers".to_string(),
                ));
            };
            let slot_name = slot_name.to_string();
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
            if let Some(base) = pyre_object::w_tuple_getitem(w_bases, i as i64)
                && pyre_object::is_type(base)
            {
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
        if let Some(w_flags) = crate::type_dict_lookup(w_type, "__abc_tpflags__")
            && pyre_object::is_int(w_flags)
        {
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
        // CPython 3.14 `type_new_slots`: a variable-sized base may add a
        // managed instance dict, but may not add weakrefs or any explicit
        // `__slots__` entry.  Derive this from the same layout metadata which
        // exposes `tp_itemsize`, so newly ported variable builtins cannot be
        // omitted from type creation semantics.
        let base_has_variable_items = if base_layout.is_null() {
            false
        } else {
            crate::typedef::cpython_type_layout(w_bestbase)
                .is_some_and(|(_, itemsize)| itemsize != 0)
        };

        // typeobject.py:1150-1204 create_all_slots
        let mut newslotnames = Vec::new();
        let (mut wantdict, mut wantweakref);
        if let Some(w_slots) = crate::type_dict_lookup(w_type, "__slots__") {
            // typeobject.py:1154-1176: has __slots__
            wantdict = false;
            wantweakref = false;
            let all_names = collect_slot_names(w_slots)?;
            if base_has_variable_items && !all_names.is_empty() {
                return Err(crate::PyError::type_error(format!(
                    "nonempty __slots__ not supported for subtype of '{}'",
                    pyre_object::w_type_get_name(w_bestbase)
                )));
            }
            if !all_names.iter().any(|name| name == "__doc__")
                && !crate::type_dict_contains(w_type, "__doc__")
            {
                crate::runtime_ops::type_dict_store(w_type, "__doc__", pyre_object::w_none());
            }
            for slot_name in &all_names {
                match slot_name.as_str() {
                    // typeobject.py:1165-1169: __dict__ slot
                    "__dict__" => {
                        // A base whose instances already carry a dict disallows
                        // a second one. Regular classes are flagged `hasdict`;
                        // BaseException subclasses expose their dict through the
                        // native exception slot (`w_exception_getdict`) without
                        // the flag, so check the layout base explicitly.
                        let base_has_dict = pyre_object::w_type_get_hasdict(w_type)
                            || (!w_bestbase.is_null()
                                && crate::builtins::lookup_exc_class("BaseException")
                                    .is_some_and(|base_exc| issubtype_ptr(w_bestbase, base_exc)));
                        if wantdict || base_has_dict {
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
                    // `create_slot`: a name already present is a conflict — a
                    // duplicate `__slots__` entry (a Member of this very type)
                    // is silently ignored, but a class variable of the same
                    // name is a ValueError.
                    let is_dup_slot = crate::type_dict_lookup(w_type, mangled.as_str())
                        .map(|w_prev| unsafe {
                            pyre_object::is_member(w_prev)
                                && std::ptr::eq(pyre_object::w_member_get_cls(w_prev), w_type)
                        })
                        .unwrap_or(false);
                    if !is_dup_slot {
                        return Err(crate::PyError::value_error(format!(
                            "'{mangled}' in __slots__ conflicts with class variable"
                        )));
                    }
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
            wantweakref = !base_has_variable_items;
        }

        // PyPy dict subclasses are W_DictMultiObject instances, so their
        // mapping payload is an intrinsic field independent of whether the
        // Python class requests an instance __dict__.  Pyre composes that
        // payload as `__dict_data__`; reserve it as an inherited layout slot
        // rather than an ordinary mapdict attribute.  Otherwise a slotted
        // dict subclass (PyPy's defaultdict shape) has nowhere to store its
        // mapping and dict operations recurse through the missing backing.
        let dict_type = crate::typedef::gettypeobject(&pyre_object::pyobject::DICT_TYPE);
        let is_dict_subclass = !dict_type.is_null()
            && !std::ptr::eq(w_type, dict_type)
            && crate::baseobjspace::issubtype_w(w_type, dict_type);
        let mut inherited_dict_data = false;
        let mut ancestor_layout = base_layout;
        while !ancestor_layout.is_null() {
            if (*ancestor_layout)
                .newslotnames
                .iter()
                .any(|name| name == "__dict_data__")
            {
                inherited_dict_data = true;
                break;
            }
            ancestor_layout = (*ancestor_layout).base_layout;
        }
        if is_dict_subclass
            && !inherited_dict_data
            && !newslotnames.iter().any(|name| name == "__dict_data__")
        {
            let slot_index = base_nslots + newslotnames.len() as u32;
            if crate::type_dict_has_storage(w_type) {
                let member =
                    pyre_object::w_member_new(slot_index, "__dict_data__".to_string(), w_type);
                crate::type_dict_store(w_type, "__dict_data__", member);
            }
            newslotnames.push("__dict_data__".to_string());
        }

        // `W_WeakrefBase`/`W_Weakref` keep `w_obj_weak`, `w_callable` and
        // `w_hash` as interpreter-owned fields.  A `weakref.ref` subclass that
        // adds no storage keeps the typed layout, and one that gets an instance
        // dict keeps them there; a subclass declaring a non-empty `__slots__`
        // has neither, so reserve private layout slots for them.  Without this
        // the constructor's stores are dropped and every such reference reads
        // back dead — `weakref.KeyedRef` is exactly that shape.
        let weakref_ref_type = crate::module::_weakref::interp__weakref::weakref_type();
        let is_weakref_subclass = !weakref_ref_type.is_null()
            && !std::ptr::eq(w_type, weakref_ref_type)
            && crate::baseobjspace::issubtype_w(w_type, weakref_ref_type);
        if is_weakref_subclass && !newslotnames.is_empty() {
            let reserved = crate::module::_weakref::interp__weakref::RESERVED_FIELD_SLOTS;
            let mut inherited_fields = false;
            let mut ancestor_layout = base_layout;
            while !ancestor_layout.is_null() {
                if (*ancestor_layout)
                    .newslotnames
                    .iter()
                    .any(|name| name == reserved[0])
                {
                    inherited_fields = true;
                    break;
                }
                ancestor_layout = (*ancestor_layout).base_layout;
            }
            if !inherited_fields {
                // No Member is stored for these: they are interpreter storage,
                // not attributes the subclass exposes.
                newslotnames.extend(reserved.iter().map(|name| (*name).to_string()));
            }
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

/// typeobject.py:1335-1353 find_best_base.
unsafe fn find_best_base(
    w_bases: pyre_object::PyObjectRef,
) -> Result<pyre_object::PyObjectRef, crate::PyError> {
    unsafe {
        if w_bases.is_null() || !pyre_object::is_tuple(w_bases) {
            return Ok(std::ptr::null_mut());
        }
        let len = pyre_object::w_tuple_len(w_bases);
        let mut w_bestbase: pyre_object::PyObjectRef = std::ptr::null_mut();
        for i in 0..len {
            if let Some(w_candidate) = pyre_object::w_tuple_getitem(w_bases, i as i64) {
                // typeobject.py:1341-1342 — a non-type base is skipped here,
                // not rejected: it is a classic base, and `get_mro` walks it
                // through `abstract_mro` when the C3 merge reaches it.
                if !pyre_object::is_type(w_candidate) {
                    continue;
                }
                // typeobject.py:1343-1345 — a custom metaclass mro() may
                // expose the nascent type before its MRO is installed, but
                // that incomplete type cannot itself be extended.
                if pyre_object::w_type_get_mro(w_candidate).is_null() {
                    return Err(crate::PyError::type_error(format!(
                        "Cannot extend an incomplete type '{}'",
                        pyre_object::w_type_get_name(w_candidate),
                    )));
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
        Ok(w_bestbase)
    }
}

/// typeobject.py:1107-1129 check_and_find_best_base:
///   w_bestbase = find_best_base(bases_w)
///   if w_bestbase is None: raise TypeError
///   if not w_bestbase.layout.typedef.acceptable_as_base_class: raise TypeError
///   for w_base in bases_w: check layout conflicts
pub(crate) unsafe fn check_and_find_best_base(
    w_bases: pyre_object::PyObjectRef,
) -> Result<pyre_object::PyObjectRef, crate::PyError> {
    unsafe {
        // Every base is rejected here rather than silently skipped: Python 3
        // has no classic classes, so a non-type in a mixed tuple is an error,
        // not something `find_best_base` may drop.  The base survived
        // `__mro_entries__` resolution and metaclass selection, so the winner
        // is a real type being asked to build over a non-type, and the layout
        // walk below would otherwise read `__bases__` off it.
        let len = if w_bases.is_null() || !pyre_object::is_tuple(w_bases) {
            0
        } else {
            pyre_object::w_tuple_len(w_bases)
        };
        for i in 0..len {
            let Some(w_base) = pyre_object::w_tuple_getitem(w_bases, i as i64) else {
                continue;
            };
            if !pyre_object::is_type(w_base) {
                return Err(crate::PyError::type_error("bases must be types"));
            }
        }
        let w_bestbase = find_best_base(w_bases)?;
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
                    let native_layout_conflict = !layout.is_null()
                        && !std::ptr::eq((*best_layout).typedef, (*layout).typedef)
                        && !std::ptr::eq(
                            (*best_layout).typedef,
                            &pyre_object::pyobject::INSTANCE_TYPE,
                        )
                        && !std::ptr::eq((*layout).typedef, &pyre_object::pyobject::INSTANCE_TYPE);
                    if !layout.is_null()
                        && (!(*best_layout).issublayout(layout) || native_layout_conflict)
                    {
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
