use pyre_object::PyObjectRef;
use std::sync::OnceLock;

use crate::PyFrame;

/// Return the live `PyFrame` as the Python-visible `frame` object passed
/// to a trace / profile callback.
///
/// `pyframe.py:_trace` hands the callback the real frame
/// (`jit.hint(frame, access_directly=False)`).  Now that `PyFrame` is a
/// W_Root (`FRAME_TYPE` typedef), pyre does the same: the callback's
/// `frame.f_lineno = N` / `frame.f_trace = local` land directly on the
/// live frame's getsets, so no `sys.namespace` wrapper and no
/// writeback pass is needed.  The frame is the current executing frame
/// (on the `CURRENT_FRAME` chain and thus GC-rooted) for the whole
/// callback, so returning it is safe; mark it escaped so the JIT keeps
/// it materialised while the callback holds a reference
/// (`pyframe.py:176 mark_as_escaped`).
fn wrap_trace_frame(frame: *mut PyFrame) -> PyObjectRef {
    if frame.is_null() {
        return pyre_object::w_none();
    }
    unsafe { (*frame).mark_as_escaped() };
    frame as PyObjectRef
}

/// pypy/interpreter/executioncontext.py:10-15 app_profile_call.
///
/// Module-level low-level trampoline used by `setprofile` to bridge
/// `setllprofile` (which stores a function pointer) to the user
/// callable held in `w_profilefuncarg`.  RPython:
///
/// ```python
/// def app_profile_call(space, w_callable, frame, event, w_arg):
///     frame = jit.hint(frame, access_directly=False)
///     space.call_function(w_callable, frame, space.newtext(event), w_arg)
/// ```
///
/// `jit.hint(frame, access_directly=False)` is a JIT-only annotation
/// that demotes the green frame to a regular w_object before the user
/// callable observes it.  Pyre's internal `PyFrame` is not itself a
/// `PyObject`, so the callback sees a frame wrapper carrying the same
/// Python-visible fields instead of a raw struct pointer.
pub fn app_profile_call(
    _space: PyObjectRef,
    w_callable: PyObjectRef,
    frame: *mut PyFrame,
    event: &str,
    w_arg: PyObjectRef,
) -> Result<(), crate::PyError> {
    let frame_obj = wrap_trace_frame(frame);
    let w_event = pyre_object::w_str_new(event);
    crate::call::call_function_impl_result(w_callable, &[frame_obj, w_event, w_arg]).map(|_| ())
}

/// pypy/interpreter/executioncontext.py:320 `self.profilefunc` — the
/// low-level callback stored by `setllprofile`.  In RPython this is a
/// raw function pointer; pyre mirrors the type with a `fn(...)` so the
/// trampoline (`app_profile_call`) is the actual stored value rather
/// than the user callable, matching upstream's two-slot design
/// (`profilefunc` + `w_profilefuncarg`).
pub type ProfileFunc = fn(
    space: PyObjectRef,
    w_callable: PyObjectRef,
    frame: *mut PyFrame,
    event: &str,
    w_arg: PyObjectRef,
) -> Result<(), crate::PyError>;

pub const TICK_COUNTER_STEP: usize = 100;

#[derive(Debug, Default)]
pub struct WRootFinalizerQueue;

impl WRootFinalizerQueue {
    pub fn finalizer_trigger(&mut self) {
        finalizer_queue_trigger();
    }

    /// `pypy/interpreter/executioncontext.py:640` —
    /// `self.space.finalizer_queue.next_dead()`.
    ///
    /// Returns the next `w_obj` whose finalizer should run, or `None`
    /// when the death queue is empty.
    ///
    pub fn next_dead(&mut self) -> Option<PyObjectRef> {
        let obj = pyre_object::gc_hook::try_gc_finalizer_next_dead(0);
        (!obj.is_null()).then_some(obj)
    }
}

fn finalizer_queue_trigger() {
    let ec = crate::call::getexecutioncontext() as *mut ExecutionContext;
    if ec.is_null() {
        return;
    }
    unsafe {
        let action = (*ec).user_del_action;
        if !action.is_null() {
            (*action).fire();
        }
    }
}

/// objspace.py:486 `allocate_instance` finalizer registration callback.
pub fn maybe_register_user_finalizer(obj: PyObjectRef) {
    let Some(w_type) = crate::typedef::r#type(obj) else {
        return;
    };
    if unsafe { pyre_object::w_type_get_hasuserdel(w_type) } {
        pyre_object::gc_hook::try_gc_register_finalizer(0, obj, finalizer_queue_trigger);
    }
}

/// Register a suspended-generator finalizer so its `finally`/`with` cleanup runs if the
/// generator is collected while suspended inside a handler range. PyPy
/// (generator.py:27) gates registration on `co_flags & CO_YIELD_INSIDE_TRY`; that
/// compile-time flag is unavailable here (external compiler), so the caller gates on a
/// non-empty code exception table — a sound necessary condition for any reachable
/// `finally`/`except`/`with`.
pub fn register_generator_finalizer(obj: PyObjectRef) {
    pyre_object::gc_hook::try_gc_register_finalizer(0, obj, finalizer_queue_trigger);
}

/// Shared execution context for all frames in one interpreter run.
///
/// Holds the builtin module dict used by module-level frames.
#[derive(Clone)]
pub struct ExecutionContext {
    pub space: PyObjectRef,
    pub topframeref: *mut PyFrame,
    pub w_tracefunc: PyObjectRef,
    pub is_tracing: i32,
    pub compiler: PyObjectRef,
    /// pypy/interpreter/executioncontext.py:320 — function pointer to
    /// the low-level profiling trampoline (e.g. `app_profile_call`).
    /// `None` means profiling is disabled.  The user callable lives in
    /// `w_profilefuncarg`; the trampoline forwards to it.
    pub profilefunc: Option<ProfileFunc>,
    pub w_profilefuncarg: PyObjectRef,
    pub thread_disappeared: bool,
    pub w_async_exception_type: PyObjectRef,
    pub actionflag: ActionFlag,
    /// `space.user_del_action`, allocated after the ExecutionContext reaches
    /// its stable process-lifetime address.
    pub user_del_action: *mut UserDelAction,
    /// `pypy/objspace/std/dictmultiobject.py:60-69
    /// allocate_and_init_instance(module=True)` parity — the builtins
    /// module's `w_dict` is a `W_ModuleDictObject` backed by
    /// `ModuleDictStrategy` (`celldict.py:28`).  Populated once at
    /// construction time; pinned with `pin_root` so the strategy
    /// storage survives the EC's lifetime.
    builtins_module: PyObjectRef,
    // `space.builtin.w_dict` is a single W_ModuleDictObject in PyPy
    // (`pypy/interpreter/baseobjspace.py:642`).  Pyre used to keep a
    // parallel `builtins: DictStorage` snapshot here, but that snapshot froze the builtin set at
    // EC construction time — runtime mutations to `__builtins__`
    // weren't visible to new frames.  Reading the live storage from
    // `builtins_module` each call removes the double-storage gap.
    /// Cached dict wrapper over `self.builtins` — pyframe.py:200-204
    /// `space.builtin` returns the same object every call.
    builtin_dict_cache: std::cell::Cell<PyObjectRef>,
    /// `pypy/interpreter/baseobjspace.py` `space.check_signal_action` —
    /// the `CheckSignalAction` registered when the `signal` module loads.
    /// `checksignals` calls its `perform` directly.  Stored as a trait
    /// pointer into the leaked action owned by `module::_signal`
    /// (`install_signal_handling`); `None` until installed.
    pub check_signal_action: Option<*mut dyn AsyncActionOps>,
    /// `executioncontext.py sys_exc_operror` — the active exception for
    /// `sys.exc_info()` / bare `raise`, saved/restored across handler
    /// regions by PUSH_EXC_INFO / POP_EXCEPT.  Single source of truth
    /// (replaces the former `eval::CURRENT_EXCEPTION` thread-local), so
    /// the JIT can read/write it as a GETFIELD_GC/SETFIELD_GC slot on the
    /// per-thread EC pointer and the optimizer can dead-store-eliminate a
    /// balanced save/restore.  GC-rooted via `walk_pyframe_roots`.
    pub sys_exc_value: PyObjectRef,
}

pub type PyExecutionContext = ExecutionContext;

/// `ExecutionContext.get_builtin()` cache read. This concrete accessor keeps
/// the rtyper from having to bind generic `std::cell::Cell<T>::get` while
/// preserving the exact `Cell<PyObjectRef>` storage used by the PyPy-parity
/// builtin-module cache.
#[majit_macros::dont_look_inside]
pub fn execution_context_builtin_cache_get(ec: &ExecutionContext) -> PyObjectRef {
    ec.builtin_dict_cache.get()
}

/// Byte offset of `sys_exc_value` within `ExecutionContext`, for the JIT's
/// GETFIELD_GC/SETFIELD_GC lowering of PUSH_EXC_INFO / POP_EXCEPT.
pub const EC_SYS_EXC_VALUE_OFFSET: usize = std::mem::offset_of!(ExecutionContext, sys_exc_value);

/// Size of `ExecutionContext`, for the JIT's StructPtrInfo SizeDescr
/// describing the (non-GC) EC struct.  The EC is never JIT-allocated;
/// this size only backs the field-tracking SizeDescr.
pub const EC_SIZE: usize = std::mem::size_of::<ExecutionContext>();

impl Default for ExecutionContext {
    fn default() -> Self {
        Self::new()
    }
}

impl ExecutionContext {
    #[inline]
    pub fn new() -> Self {
        let builtins_module = crate::builtins::new_builtin_module_dict();
        pyre_object::gc_roots::pin_root(builtins_module);
        Self {
            space: pyre_object::PY_NULL,
            topframeref: std::ptr::null_mut(),
            w_tracefunc: pyre_object::PY_NULL,
            is_tracing: 0,
            compiler: pyre_object::PY_NULL,
            profilefunc: None,
            w_profilefuncarg: pyre_object::PY_NULL,
            thread_disappeared: false,
            w_async_exception_type: pyre_object::PY_NULL,
            actionflag: ActionFlag::new(),
            user_del_action: std::ptr::null_mut(),
            builtins_module,
            builtin_dict_cache: std::cell::Cell::new(pyre_object::PY_NULL),
            check_signal_action: None,
            sys_exc_value: pyre_object::PY_NULL,
        }
    }

    pub fn __init__(&mut self, space: PyObjectRef) {
        self.space = space;
        self.compiler = pyre_object::w_none();
    }

    pub fn install_user_del_action(&mut self) {
        if self.user_del_action.is_null() {
            let action = UserDelAction::new(self.space, &mut self.actionflag);
            self.user_del_action = Box::into_raw(action);
        }
        pyre_object::gc_hook::register_maybe_finalizer_hook(maybe_register_user_finalizer);
    }

    #[inline]
    pub fn _mark_thread_disappeared(_space: PyObjectRef) {
        let _ = _space;
    }

    #[inline]
    pub fn gettopframe(&self) -> *mut PyFrame {
        self.topframeref
    }

    pub fn gettopframe_nohidden(&self) -> *mut PyFrame {
        let mut frame = self.topframeref;
        while !frame.is_null() {
            // SAFETY: frame pointers are owned by interpreter call stack and can be
            // null-checked before dereference.
            unsafe {
                let current = &*frame;
                if !current.hide() {
                    return frame;
                }
                frame = current.get_f_back();
            }
        }
        frame
    }

    pub fn getnextframe_nohidden(mut frame: *mut PyFrame) -> *mut PyFrame {
        while !frame.is_null() {
            // SAFETY: caller provides a valid frame chain or null.
            unsafe {
                let current = &*frame;
                let next = current.get_f_back();
                if next.is_null() {
                    return std::ptr::null_mut();
                }
                if !(&*next).hide() {
                    return next;
                }
                frame = next;
            }
        }
        frame
    }

    pub fn enter(&mut self, frame: *mut PyFrame) {
        // pypy/interpreter/executioncontext.py:85-89 enter parity.
        if !self.space.is_null() && self.is_tracing > 0 {
            self._revdb_enter(frame);
        }
        unsafe {
            (*frame).f_backref = self.topframeref;
        }
        // `jit.virtual_ref(frame)` — vref allocation, no-op until
        // jit.virtual_ref is ported.
        self.topframeref = frame;
    }

    #[allow(clippy::too_many_arguments)]
    pub fn leave(
        &mut self,
        frame: *mut PyFrame,
        w_exitvalue: PyObjectRef,
        got_exception: bool,
    ) -> Result<(), crate::PyError> {
        // pypy/interpreter/executioncontext.py:91-109 leave parity.
        // The original wraps `_trace('leaveframe', …)` in try/finally so
        // the topframeref restore and the vref dance always run.  We
        // capture the trace result and propagate it after the cleanup
        // block below.
        let trace_result = if self.profilefunc.is_some() {
            self._trace(frame, "leaveframe", w_exitvalue, None)
        } else {
            Ok(())
        };
        let _frame_vref = self.topframeref;
        unsafe {
            self.topframeref = (*frame).f_backref;
            if (*frame).escaped || got_exception {
                let f_back = (*frame).get_f_back();
                if !f_back.is_null() {
                    (*f_back).mark_as_escaped();
                }
                // `frame_vref()` — vref force, no-op until jit.virtual_ref is ported.
            }
        }
        // `jit.virtual_ref_finish(frame_vref, frame)` — no-op until
        // jit.virtual_ref is ported.
        self._revdb_leave(got_exception);
        trace_result
    }

    /// executioncontext.py:113-115 — `c_call_trace(self, frame, w_func, args=None)`.
    pub fn c_call_trace(
        &mut self,
        frame: *mut PyFrame,
        w_func: PyObjectRef,
        args: Option<&crate::argument::Arguments>,
    ) -> Result<(), crate::PyError> {
        self._c_call_return_trace(frame, w_func, args, "c_call")
    }

    /// executioncontext.py:117-119 — `c_return_trace(self, frame, w_func, args=None)`.
    pub fn c_return_trace(
        &mut self,
        frame: *mut PyFrame,
        w_func: PyObjectRef,
        args: Option<&crate::argument::Arguments>,
    ) -> Result<(), crate::PyError> {
        self._c_call_return_trace(frame, w_func, args, "c_return")
    }

    /// executioncontext.py:121-136 — `_c_call_return_trace`.
    pub fn _c_call_return_trace(
        &mut self,
        frame: *mut PyFrame,
        mut w_func: PyObjectRef,
        args: Option<&crate::argument::Arguments>,
        event: &str,
    ) -> Result<(), crate::PyError> {
        if self.profilefunc.is_none() {
            if !frame.is_null() {
                unsafe {
                    (*frame).getorcreatedebug(-1).is_being_profiled = false;
                }
            }
            return Ok(());
        }
        // executioncontext.py:128-134 FunctionWithFixedCode method-call
        // rebinding.  PyPy:
        //   if isinstance(w_func, FunctionWithFixedCode) and args is not None:
        //       w_firstarg = args.firstarg()
        //       if w_firstarg is not None:
        //           w_func = descr_function_get(self.space, w_func,
        //                                       w_firstarg, self.space.type(w_firstarg))
        // BuiltinFunction (function.py:786, the module-level
        // builtin sibling) is intentionally excluded — function.py
        // splits the two so that "builtin function binds
        // differently" (no descriptor rebinding).  Pyre's
        // `is_function_with_fixed_code` (FUNCTION_TYPE && !can_change_code)
        // is the line-by-line port of the isinstance check.
        if let Some(args) = args {
            unsafe {
                if crate::is_function_with_fixed_code(w_func) {
                    if let Some(w_firstarg) = args.firstarg() {
                        if !w_firstarg.is_null() {
                            let w_type =
                                crate::typedef::r#type(w_firstarg).unwrap_or(pyre_object::PY_NULL);
                            w_func = crate::descr_function_get(w_func, w_firstarg, w_type);
                        }
                    }
                }
            }
        }
        self._trace(frame, event, w_func, None)
    }

    pub fn c_exception_trace(
        &mut self,
        frame: *mut PyFrame,
        _w_exc: PyObjectRef,
    ) -> Result<(), crate::PyError> {
        if self.profilefunc.is_none() {
            if !frame.is_null() {
                unsafe {
                    (*frame).getorcreatedebug(-1).is_being_profiled = false;
                }
            }
            return Ok(());
        }
        self._trace(frame, "c_exception", _w_exc, None)
    }

    pub fn call_trace(&mut self, frame: *mut PyFrame) -> Result<(), crate::PyError> {
        if !self.gettrace().is_null() || self.profilefunc.is_some() {
            self._trace(frame, "call", pyre_object::w_none(), None)?;
            if self.profilefunc.is_some() {
                if !frame.is_null() {
                    unsafe {
                        (*frame).getorcreatedebug(-1).is_being_profiled = true;
                    }
                }
            }
        }
        Ok(())
    }

    pub fn return_trace(
        &mut self,
        frame: *mut PyFrame,
        w_retval: PyObjectRef,
    ) -> Result<(), crate::PyError> {
        let _ = (frame, w_retval);
        if !self.gettrace().is_null() {
            self._trace(frame, "return", w_retval, None)?;
        }
        Ok(())
    }

    pub fn bytecode_trace(
        &mut self,
        frame: *mut PyFrame,
        decr_by: usize,
    ) -> Result<(), crate::PyError> {
        // executioncontext.py:158-165 bytecode_trace:
        //   def bytecode_trace(self, frame, decr_by=TICK_COUNTER_STEP):
        //       self.bytecode_only_trace(frame)
        //       actionflag = self.space.actionflag
        //       if actionflag.decrement_ticker(decr_by) < 0:
        //           actionflag.action_dispatcher(self, frame)
        //
        // bytecode_only_trace runs first; if it raises (tracefunc
        // callback exception), the ticker decrement + slow-path
        // action_dispatcher do NOT run.  Use `?` so a tracer error
        // short-circuits before touching actionflag.
        self.bytecode_only_trace(frame)?;
        if self.actionflag.decrement_ticker(decr_by as isize) < 0 {
            // executioncontext.py:165 — `actionflag.action_dispatcher`.
            // Routed through a residual (dont_look_inside) boundary so the
            // tracer never sees the action machinery's trait-object
            // virtual calls + `Result<(), PyError>` propagation, which the
            // JIT codewriter cannot model.  The slow path runs only when
            // the ticker is negative (a signal / fired action), so the
            // residual call adds nothing to the no-signal hot path.
            let self_ptr = self as *mut ExecutionContext;
            if perform_pending_actions(self_ptr as i64, frame as i64) != 0 {
                if let Some(err) = crate::call::take_call_error() {
                    return Err(err);
                }
            }
        }
        Ok(())
    }

    /// Run pending async actions (signal delivery, finalizers).  Split
    /// out so the JIT warm-up loop can take the slow path inline once the
    /// ticker goes negative without re-entering `bytecode_trace`'s tracer
    /// gate.  Mirrors the `actionflag.action_dispatcher(self, frame)` call
    /// in `bytecode_trace` (executioncontext.py:165).
    pub fn perform_actions(&mut self, frame: *mut PyFrame) -> Result<(), crate::PyError> {
        let self_ptr = self as *mut ExecutionContext;
        self.actionflag.action_dispatcher(self_ptr, frame)
    }

    /// pypy/interpreter/executioncontext.py:173-184 `bytecode_only_trace`.
    ///
    /// ```python
    /// def bytecode_only_trace(self, frame):
    ///     if self.space.reverse_debugging:
    ///         self._revdb_potential_stop_point(frame)
    ///     if (frame.get_w_f_trace() is None or self.is_tracing or
    ///         self.gettrace() is None):
    ///         return
    ///     self.run_trace_func(frame)
    /// ```
    ///
    /// reverse_debugging is not implemented in pyre, so the
    /// `_revdb_potential_stop_point` arm reduces to a no-op.  All
    /// three short-circuit conditions stay — including
    /// `frame.get_w_f_trace() is None`, which avoids the trailing
    /// `instr_prev_plus_one` write inside `run_trace_func` when the
    /// per-frame trace is unset (so a later `frame.f_trace = cb`
    /// observes the unmodified instr_prev_plus_one and can fire its
    /// first `line` event correctly).
    pub fn bytecode_only_trace(&mut self, frame: *mut PyFrame) -> Result<(), crate::PyError> {
        if self.space.is_null() || frame.is_null() {
            return Ok(());
        }
        let f_trace_is_none = unsafe { (*frame).get_w_f_trace().is_null() };
        if f_trace_is_none || self.is_tracing != 0 || self.w_tracefunc.is_null() {
            return Ok(());
        }
        self.run_trace_func(frame)
    }

    pub fn _run_finalizers_now(&mut self) {
        if !self.user_del_action.is_null() {
            unsafe { (*self.user_del_action)._run_finalizers() };
        }
    }

    /// pypy/interpreter/executioncontext.py:185-200 `run_trace_func`.
    ///
    /// ```python
    /// def run_trace_func(self, frame):
    ///     code = frame.pycode
    ///     if frame.last_instr == -1:
    ///         return     # don't trace the SETUP_ANNOTATIONS at the very start
    ///     d = frame.getorcreatedebug()
    ///     if d.is_in_line_tracing or d.f_trace_lines:
    ///         lastline = d.f_lineno
    ///         lineno = code._get_lineno_for_pc_tracing(frame.last_instr)
    ///         if lastline != lineno or frame.last_instr < d.instr_prev_plus_one:
    ///             self._trace(frame, 'line', self.space.w_None)
    ///     if d.f_trace_opcodes:
    ///         self._trace(frame, 'opcode', self.space.w_None)
    ///     d.instr_prev_plus_one = frame.last_instr + 1
    /// ```
    pub fn run_trace_func(&mut self, frame: *mut PyFrame) -> Result<(), crate::PyError> {
        if self.space.is_null() || frame.is_null() {
            return Ok(());
        }
        // executioncontext.py:189-192:
        //   d = frame.getorcreatedebug()
        //   lastline = d.f_lineno
        //   lineno = frame.pycode._get_lineno_for_pc_tracing(frame.last_instr)
        //   d.f_lineno = lineno
        let last_instr = unsafe { (*frame).last_instr };
        let lineno = unsafe { (*frame).get_last_lineno() };
        let (lastline, want_line, want_opcode) = unsafe {
            let d = (*frame).getorcreatedebug(lineno);
            let lastline = d.f_lineno;
            // executioncontext.py:192 d.f_lineno = lineno — PERSISTENT
            // write so subsequent run_trace_func invocations see the
            // current line (not the previous).
            d.f_lineno = lineno;
            // executioncontext.py:193-197:
            //   if d.f_trace_lines and lineno != -1:
            //       if lastline != lineno or frame.last_instr < d.instr_prev_plus_one:
            //           self._trace(frame, 'line', self.space.w_None)
            //   if d.f_trace_opcodes:
            //       self._trace(frame, 'opcode', self.space.w_None)
            let want_line = d.f_trace_lines
                && lineno != -1
                && (lastline != lineno || last_instr < d.instr_prev_plus_one);
            let want_opcode = d.f_trace_opcodes;
            (lastline, want_line, want_opcode)
        };
        let _ = lastline;
        if want_line {
            self._trace(frame, "line", pyre_object::w_none(), None)?;
        }
        if want_opcode {
            self._trace(frame, "opcode", pyre_object::w_none(), None)?;
        }
        // executioncontext.py:200 — record the next-PC sentinel so
        // backward jumps re-fire the line event even when staying on
        // the same source line.
        unsafe {
            (*frame).getorcreatedebug(lineno).instr_prev_plus_one = last_instr + 1;
        }
        Ok(())
    }

    /// pypy/interpreter/executioncontext.py:202-208
    /// `bytecode_trace_after_exception`.
    ///
    /// ```python
    /// def bytecode_trace_after_exception(self, frame):
    ///     "Like bytecode_trace(), but without increasing the ticker."
    ///     actionflag = self.space.actionflag
    ///     self.bytecode_only_trace(frame)
    ///     if actionflag.get_ticker() < 0:
    ///         actionflag.action_dispatcher(self, frame)     # slow path
    /// ```
    ///
    /// pyre's action_dispatcher slow path is not yet wired (the
    /// pre-existing stub only decremented the ticker once); the
    /// `bytecode_only_trace` call mirrors upstream so a tracer error
    /// surfaces through this helper's `Result`.
    pub fn bytecode_trace_after_exception(
        &mut self,
        frame: *mut PyFrame,
    ) -> Result<(), crate::PyError> {
        self.bytecode_only_trace(frame)?;
        if self.actionflag.get_ticker() < 0 {
            // executioncontext.py:207-208 — `if actionflag.get_ticker()
            // < 0: actionflag.action_dispatcher(self, frame)`.  Routed
            // through the same residual boundary as `bytecode_trace` so a
            // signal delivered during exception handling propagates.
            let self_ptr = self as *mut ExecutionContext;
            if perform_pending_actions(self_ptr as i64, frame as i64) != 0 {
                if let Some(err) = crate::call::take_call_error() {
                    return Err(err);
                }
            }
        }
        Ok(())
    }

    /// pypy/interpreter/executioncontext.py:430-433 exception_trace.
    ///
    /// ```python
    /// def exception_trace(self, frame, operationerr):
    ///     if self.w_tracefunc is not None:
    ///         self._trace(frame, 'exception',
    ///                     operationerr.get_w_value(self.space), operationerr)
    /// ```
    ///
    /// `_trace` consumes the live `OperationError` (executioncontext.py:
    /// 359-363) and mutates it in place via `normalize_exception`.
    /// Pyre's pyopcode call site does not yet hand the live operr to
    /// `exception_trace` — it forwards a `(w_type, w_value, w_traceback)`
    /// triple — so this wrapper fabricates a temporary `OperationError`
    /// whose lifetime spans the `_trace` call.  PyPy's `pyopcode.py:148
    /// ec.exception_trace(self, operr)` passes the live caller-held
    /// `operr`, but the post-call mutation is unobserved in pyre because
    /// the temp falls out of scope here.  The OperationError
    /// port (caller threads its live operr through) will close that gap.
    pub fn exception_trace(
        &mut self,
        frame: *mut PyFrame,
        w_type: PyObjectRef,
        w_value: PyObjectRef,
        w_traceback: PyObjectRef,
    ) -> Result<(), crate::PyError> {
        if !self.gettrace().is_null() {
            let mut operr = crate::error::OperationError::new(w_type, w_value);
            operr._application_traceback = if w_traceback.is_null() {
                None
            } else {
                Some(w_traceback)
            };
            self._trace(frame, "exception", w_value, Some(&mut operr))?;
        }
        Ok(())
    }

    pub fn sys_exc_info(&self, _for_hidden: bool) -> PyObjectRef {
        let _ = self.gettopframe();
        let _ = _for_hidden;
        pyre_object::PY_NULL
    }

    pub fn set_sys_exc_info(&mut self, _operror: PyObjectRef) {
        let _ = _operror;
        let frame = self.gettopframe_nohidden();
        if !frame.is_null() {
            // Real PyPy stores OperationError in frame.last_exception.
            let _ = frame;
        }
    }

    pub fn clear_sys_exc_info(&mut self) {
        let mut frame = self.gettopframe_nohidden();
        while !frame.is_null() {
            frame = Self::getnextframe_nohidden(frame);
        }
    }

    pub fn settrace(&mut self, w_func: PyObjectRef) {
        self.w_tracefunc = w_func;
        if w_func.is_null() || w_func == pyre_object::w_none() {
            self.w_tracefunc = pyre_object::PY_NULL;
        } else {
            self.force_all_frames(false);
            // executioncontext.py:296-298 — increase the JIT's
            // trace_limit when a tracefunc is installed; tracing
            // generates a ton of extra ops per bytecode.
            crate::call::set_jit_param("trace_limit", 10000);
        }
    }

    pub fn gettrace(&self) -> PyObjectRef {
        self.w_tracefunc
    }

    /// pypy/interpreter/executioncontext.py:303-310 setprofile.
    pub fn setprofile(&mut self, w_func: PyObjectRef) -> Result<(), crate::PyError> {
        if w_func.is_null() || w_func == pyre_object::w_none() {
            self.profilefunc = None;
            self.w_profilefuncarg = pyre_object::PY_NULL;
            Ok(())
        } else {
            self.setllprofile(Some(app_profile_call), w_func)
        }
    }

    /// pypy/interpreter/executioncontext.py:312-313 getprofile.
    pub fn getprofile(&self) -> PyObjectRef {
        self.w_profilefuncarg
    }

    /// pypy/interpreter/executioncontext.py:315-321 setllprofile.
    pub fn setllprofile(
        &mut self,
        func: Option<ProfileFunc>,
        w_arg: PyObjectRef,
    ) -> Result<(), crate::PyError> {
        if func.is_some() {
            // executioncontext.py:317-318 `if w_arg is None: raise
            // ValueError("Cannot call setllprofile with real None")`.
            // The check is against RPython-level None (== null in pyre);
            // Python-level `w_none()` (`space.w_None`) is a valid user
            // argument that flows through unchanged.
            if w_arg.is_null() {
                return Err(crate::PyError::value_error(
                    "Cannot call setllprofile with real None",
                ));
            }
            self.force_all_frames(true);
        }
        self.profilefunc = func;
        self.w_profilefuncarg = w_arg;
        Ok(())
    }

    pub fn force_all_frames(&mut self, is_being_profiled: bool) {
        let mut frame = self.gettopframe_nohidden();
        while !frame.is_null() {
            if is_being_profiled {
                unsafe {
                    (*frame).getorcreatedebug(-1).is_being_profiled = true;
                }
            }
            frame = Self::getnextframe_nohidden(frame);
        }
    }

    pub fn call_tracing(&mut self, w_func: PyObjectRef, w_args: PyObjectRef) -> PyObjectRef {
        let was_tracing = self.is_tracing;
        self.is_tracing = 0;
        let result = crate::baseobjspace::call(w_func, w_args, None);
        self.is_tracing = was_tracing;
        result
    }

    /// pypy/interpreter/executioncontext.py:346-428 _trace.
    ///
    /// Two-arm dispatch: the tracing arm fires every event for the
    /// frame's `w_f_trace` (or the global `w_tracefunc` on `'call'`),
    /// the profiling arm fires only on `call`/`return`/`c_call`/
    /// `c_return`/`c_exception` and routes through the
    /// `profilefunc` low-level trampoline (`app_profile_call` for
    /// `setprofile`-installed callbacks).
    ///
    /// `operr` carries the live `OperationError` instance that
    /// `executioncontext.py:359-363` reads via `operr.w_type`,
    /// `operr.normalize_exception(space)`, and
    /// `operr.get_w_traceback(space)`.  Pyre's `error::OperationError`
    /// is the line-by-line port of `error.OperationError` (the same
    /// `w_type` / `w_value` / `_application_traceback` shape).
    /// Reading the fields here mirrors PyPy 1:1 — `operr.w_type` for
    /// the type, `get_w_value(space)` for the (possibly-normalized)
    /// value, `_application_traceback` for the traceback (or
    /// `space.w_None` when absent).
    pub fn _trace(
        &mut self,
        frame: *mut PyFrame,
        event: &str,
        w_arg: PyObjectRef,
        operr: Option<&mut crate::error::OperationError>,
    ) -> Result<(), crate::PyError> {
        // executioncontext.py:347 if self.is_tracing or frame.hide():
        if self.is_tracing != 0 {
            return Ok(());
        }
        if frame.is_null() {
            return Ok(());
        }
        if unsafe { (*frame).hide() } {
            return Ok(());
        }

        let space = self.space;

        // executioncontext.py:353-356 Tracing cases
        let w_callback = if event == "call" {
            self.gettrace()
        } else {
            unsafe { (*frame).get_w_f_trace() }
        };

        if !w_callback.is_null() && event != "leaveframe" {
            // executioncontext.py:359-363:
            //   if operr is not None:
            //       w_value = operr.normalize_exception(space)
            //       w_arg = space.newtuple([operr.w_type, w_value,
            //                               operr.get_w_traceback(space)])
            //
            // PyPy `normalize_exception` mutates the caller's operr in
            // place (error.py:247 `self.w_type = w_type`); after the
            // call `operr.w_type` is the normalized class.  Pyre takes
            // `&mut OperationError` here so the same mutation reaches
            // the caller's instance instead of a throw-away clone.
            let w_arg = if let Some(operr) = operr {
                let w_value = operr.normalize_exception(space)?;
                let w_type = if operr.w_type.is_null() {
                    crate::typedef::r#type(w_value).unwrap_or_else(pyre_object::w_none)
                } else {
                    operr.w_type
                };
                let w_traceback = operr
                    ._application_traceback
                    .unwrap_or_else(pyre_object::w_none);
                pyre_object::tupleobject::w_tuple_new(vec![w_type, w_value, w_traceback])
            } else {
                w_arg
            };

            let lineno = unsafe { (*frame).get_last_lineno() };
            let init_lineno = if unsafe { (*frame).last_instr } >= 1 {
                lineno
            } else {
                -1
            };
            let had_locals = unsafe {
                let d = (*frame).getorcreatedebug(init_lineno);
                !d.w_locals.is_null()
            };
            if had_locals {
                unsafe { (*frame).fast2locals()? };
            }
            let (prev_line_tracing, old_lineno) = unsafe {
                let d = (*frame).getorcreatedebug(init_lineno);
                (d.is_in_line_tracing, d.f_lineno)
            };

            // executioncontext.py:376 self.is_tracing += 1
            self.is_tracing += 1;
            let call_result = (|| {
                unsafe {
                    let d = (*frame).getorcreatedebug(init_lineno);
                    if event == "line" {
                        d.is_in_line_tracing = true;
                    }
                    d.f_lineno = lineno;
                }
                // executioncontext.py:382-385 space.call_function(w_callback, frame, w_event, w_arg)
                let frame_obj = wrap_trace_frame(frame);
                let w_event = pyre_object::w_str_new(event);
                let call_result = crate::call::call_function_impl_result(
                    w_callback,
                    &[frame_obj, w_event, w_arg],
                );
                // The callback received the live frame, so its
                // `frame.f_trace = local` / `frame.f_lineno = N` setattrs
                // already landed on the frame's getsets — no writeback pass.
                let w_result = call_result?;
                if w_result != pyre_object::w_none() {
                    unsafe {
                        (*frame).getorcreatedebug(init_lineno).w_f_trace = w_result;
                    }
                }
                Ok::<(), crate::PyError>(())
            })();

            if call_result.is_err() {
                self.settrace(pyre_object::w_none());
                unsafe {
                    (*frame).getorcreatedebug(init_lineno).w_f_trace = pyre_object::PY_NULL;
                }
            }

            unsafe {
                let d = (*frame).getorcreatedebug(init_lineno);
                if d.f_lineno == lineno {
                    // executioncontext.py:397-404 — for generator/coroutine
                    // resumptions (`event == 'call'` while `last_instr >= 0`)
                    // keep `d.f_lineno` at the yield line so the instruction
                    // right after `YIELD_VALUE` (still on the same source
                    // line) does not fire a spurious line event; skip the
                    // restore in that case.
                    if event != "call" || (*frame).last_instr < 0 {
                        d.f_lineno = old_lineno;
                    }
                }
                d.is_in_line_tracing = prev_line_tracing;
            }
            self.is_tracing -= 1;
            // executioncontext.py:401-402 reads `d.w_locals is not None`
            // again inside finally — the callback may have installed or
            // cleared w_locals.  Re-read here instead of caching
            // `had_locals` from before the call.
            let post_had_locals = unsafe {
                let d = (*frame).getorcreatedebug(init_lineno);
                !d.w_locals.is_null()
            };
            if post_had_locals {
                unsafe { (*frame).locals2fast(false)? };
            }
            // executioncontext.py:392-395 — re-raise the callback's
            // exception after restoring trace bookkeeping. Caller chain
            // (call_trace/return_trace/bytecode_trace/exception_trace
            // → eval_frame_plain) propagates the error up so the
            // tracefunc behaves like an inline raise from the executing
            // frame.
            if let Err(err) = call_result {
                return Err(err);
            }
        }

        // executioncontext.py:404-428 Profile cases
        if let Some(profilefunc) = self.profilefunc {
            // executioncontext.py:406-411 — only call/return/c_call/
            // c_return/c_exception events fire the profile callback.
            let event = if event == "leaveframe" {
                "return"
            } else if event == "call"
                || event == "c_call"
                || event == "c_return"
                || event == "c_exception"
            {
                event
            } else {
                return Ok(());
            };

            // executioncontext.py:416-417 assert self.is_tracing == 0; self.is_tracing += 1
            assert_eq!(self.is_tracing, 0);
            self.is_tracing += 1;
            // executioncontext.py:420-425 self.profilefunc(...), clearing
            // profile slots on exceptions.
            let profile_result = profilefunc(space, self.w_profilefuncarg, frame, event, w_arg);
            if let Err(err) = profile_result {
                // executioncontext.py:421-425 — clear profile slots and
                // re-raise so the caller observes the failure (matches
                // the bare `raise` after the `except:` block).
                self.profilefunc = None;
                self.w_profilefuncarg = pyre_object::PY_NULL;
                self.is_tracing -= 1;
                return Err(err);
            }
            // executioncontext.py:427-428 self.is_tracing -= 1
            self.is_tracing -= 1;
        }
        Ok(())
    }

    /// executioncontext.py:436-441 `checksignals`:
    /// ```python
    /// if self.space.check_signal_action is not None:
    ///     self.space.check_signal_action.perform(self, None)
    /// ```
    /// Called from the EINTR retry paths (`raise_signal`, `pthread_kill`,
    /// `pthread_sigmask`) to deliver a signal that may have arrived
    /// mid-syscall.  Propagates a handler exception (e.g.
    /// `KeyboardInterrupt`) to the caller.
    pub fn checksignals(&mut self) -> Result<(), crate::PyError> {
        if let Some(action) = self.check_signal_action {
            let self_ptr = self as *mut ExecutionContext;
            unsafe {
                (*action).perform(&mut *self_ptr, std::ptr::null_mut())?;
            }
        }
        Ok(())
    }

    pub fn _revdb_enter(&mut self, _frame: *mut PyFrame) {
        let _ = _frame;
    }

    pub fn _revdb_leave(&mut self, _got_exception: bool) {
        let _ = _got_exception;
    }

    pub fn _revdb_potential_stop_point(&mut self, _frame: *mut PyFrame) {
        let _ = _frame;
    }

    #[allow(unreachable_code)]
    pub fn _freeze_(&self) {
        if !self.topframeref.is_null() {}
    }

    /// Celldict globals for a fresh module (`__main__`, imported source
    /// modules). Seeds the builtins + `__builtins__` directly into
    /// the `W_ModuleDictObject`'s authoritative cell storage so the JIT sees a
    /// stable globals shape up front (same seed-vs-fallback rationale as
    /// module dict. The `IntMutableCell` in-place write stands alone, matching
    /// PyPy's `ModuleDictStrategy`.
    pub fn fresh_module_globals(&self) -> PyObjectRef {
        let dict = pyre_object::dictmultiobject::w_module_dict_new();
        // Root the fresh dict across the seeding loop: each
        // `w_dict_setitem_str_no_proxy` allocates a cell and may trigger a
        // minor collection that would otherwise reclaim the not-yet-referenced
        // dict.
        let _root = pyre_object::gc_roots::push_roots();
        pyre_object::gc_roots::pin_root(dict);
        unsafe {
            for (k, v) in pyre_object::w_dict_str_entries(self.builtins_module) {
                pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(dict, &k, v);
            }
            let w_builtin = self.get_builtin();
            if !w_builtin.is_null() {
                pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                    dict,
                    "__builtins__",
                    w_builtin,
                );
            }
        }
        dict
    }

    /// `pypy/module/__builtin__/moduledef.py:Module.__init__` — `space.builtin`
    /// is a `Module` (not a dict).  Lazily build a `Module` whose
    /// backing dict IS `self.builtins_module`, so subsequent
    /// `module.getdict(space)` access (`pyframe.py:770 fget_f_builtins`)
    /// surfaces the same storage as a dict view.  The cache field stores
    /// the Module identity so identity-sensitive callers (PyPy
    /// `pick_builtin` `if w_builtin is space.builtin: return space.builtin`)
    /// observe the same object every call.
    pub fn get_builtin(&self) -> PyObjectRef {
        let cached = execution_context_builtin_cache_get(self);
        if !cached.is_null() {
            return cached;
        }
        // `pypy/interpreter/module.py:Module.__init__` — `space.builtin`
        // is a `Module` whose `w_dict` is the `W_ModuleDictObject`
        // allocated by `allocate_and_init_instance(module=True)`
        // (`dictmultiobject.py:60-69`).  Pyre's
        // `new_builtin_module_dict` already populated the
        // W_ModuleDictObject in `self.builtins_module`; wrap it
        // through `w_module_new_aliasing_dict` without a raw storage
        // pointer (the strategy storage IS the canonical store).
        let module = pyre_object::w_module_new_aliasing_dict("builtins", self.builtins_module);
        // function.py:797-815 BuiltinFunction.w_moduleobj. The defining
        // module object does not exist while `new_builtin_module_dict` fills
        // the namespace, so bind each builtin's `__self__` now.
        for (_, value) in unsafe { pyre_object::w_dict_str_entries(self.builtins_module) } {
            unsafe { crate::function::builtin_function_set_module_obj(value, module) };
        }
        self.builtin_dict_cache.set(module);
        // `pypy/interpreter/baseobjspace.py:647` —
        // `self.setitem(self.builtin.w_dict, 'builtins',  w_builtin)`.
        // After the builtins module exists, install the self-reference
        // so `__builtins__.__builtins__ is __builtins__` and
        // user-level `import builtins; builtins.__builtins__` round-trips
        // through `space.builtin.w_dict[__builtins__]`.
        unsafe {
            pyre_object::w_dict_setitem_str(self.builtins_module, "__builtins__", module);
        }
        module
    }

    /// Direct lookup into the builtins storage, bypassing the dict-object
    /// wrapper (whose internal hash table does not see entries inserted
    /// through `install_default_builtins` on the underlying DictStorage).
    /// Used by `LOAD_GLOBAL` / `LOAD_NAME` to reach builtins like `print`
    /// when the frame's `w_globals` lacks the name (pypy/interpreter/
    /// pyopcode.py:558-565 LOAD_GLOBAL builtin fallback).
    pub fn lookup_builtin(&self, name: &str) -> Option<PyObjectRef> {
        // `pypy/interpreter/pyopcode.py:558-565 LOAD_GLOBAL` builtin
        // fallback reads through the builtin Module's W_DictObject
        // (`space.getitem(space.builtin.w_dict, name)`).  Pyre's
        // `builtins_module` IS that W_ModuleDictObject; the
        // dispatching `w_dict_getitem_str` routes through
        // `ModuleDictStrategy::getitem_str` (`celldict.py:143-145`).
        unsafe { pyre_object::w_dict_getitem_str(self.builtins_module, name) }
    }
}

/// Residual entry point for the bytecode-trace slow path: run pending
/// async actions (signal delivery, finalizers).  Marked
/// `dont_look_inside` so the tracer treats it as an opaque call and never
/// follows the action machinery's trait-object virtual dispatch +
/// `Result<(), PyError>` propagation (which the JIT codewriter cannot
/// model).  Returns 0 on success and 1 when an action raised, stashing
/// the error in `PENDING_CALL_ERROR` for the caller to re-raise — the
/// same cross-residual error convention as `call_function_impl`.
#[majit_macros::dont_look_inside]
pub extern "C" fn perform_pending_actions(ec_ptr: i64, frame_ptr: i64) -> i64 {
    let ec = ec_ptr as *mut ExecutionContext;
    if ec.is_null() {
        return 0;
    }
    let frame = frame_ptr as *mut PyFrame;
    match unsafe { (*ec).perform_actions(frame) } {
        Ok(()) => 0,
        Err(err) => {
            crate::call::set_call_error(err);
            1
        }
    }
}

#[derive(Clone)]
pub struct AbstractActionFlag {
    _periodic_actions: Vec<*mut dyn AsyncActionOps>,
    _nonperiodic_actions: Vec<*mut dyn AsyncActionOps>,
    pub(crate) _fired_bitmask: usize,
    has_bytecode_counter: bool,
    pub checkinterval_scaled: usize,
}

impl Default for AbstractActionFlag {
    fn default() -> Self {
        Self::new()
    }
}

impl AbstractActionFlag {
    pub fn new() -> Self {
        Self {
            _periodic_actions: Vec::new(),
            _nonperiodic_actions: Vec::new(),
            _fired_bitmask: 0,
            has_bytecode_counter: false,
            checkinterval_scaled: 10000 * TICK_COUNTER_STEP,
        }
    }

    /// pypy/interpreter/executioncontext.py:531-556
    /// `_rebuild_action_dispatcher`.
    ///
    /// TODO(justified): PyPy rebuilds an inner
    /// closure on every `register_*_action` call so that
    /// `periodic_actions = unrolling_iterable(self._periodic_actions)`
    /// captures the current periodic-action list at codegen time;
    /// RPython's JIT then unrolls the periodic loop at trace time.
    /// `unrolling_iterable` is an RPython-only JIT hint with no
    /// runtime semantic effect — the closure body is identical to a
    /// straightforward `for action in self._periodic_actions: …`.
    ///
    /// Pyre's `action_dispatcher` (`ActionFlagOps` trait default,
    /// `executioncontext.rs:1537`) iterates `_periodic_actions`
    /// directly each call.  Rebuilding the closure would produce no
    /// observable change because (a) majit's loop optimizer subsumes
    /// the unroll role for hot dispatchers and (b) the periodic list
    /// is small (typically 1–3 entries — release-the-GIL,
    /// report-the-signals).  Empty body preserved as the canonical
    /// "rebuild is a no-op in pyre" marker.
    pub fn _rebuild_action_dispatcher(&mut self) {}
}

/// `pypy/interpreter/executioncontext.py:458-585` — `AbstractActionFlag`
/// + `ActionFlag` virtual-dispatch interface.
///
/// PyPy's `AbstractActionFlag.fire` (line 482), `setcheckinterval`
/// (line 522), and `action_dispatcher` (line 531) all call
/// `self.reset_ticker(...)`, which Python resolves through the
/// concrete subclass override (`ActionFlag.reset_ticker` at line 574
/// writes `self._ticker = value`; `SignalActionFlag.reset_ticker`
/// writes a C global).  Rust composition cannot virtual-dispatch
/// from a base-struct inherent method, so this trait carries the
/// virtual surface (`reset_ticker` / `get_ticker` /
/// `decrement_ticker`) plus an accessor pair (`abstract_flag` /
/// `abstract_flag_mut`) that lets the default `fire` /
/// `setcheckinterval` / `action_dispatcher` implementations reach
/// the shared `_periodic_actions` / `_nonperiodic_actions` /
/// `_fired_bitmask` / `checkinterval_scaled` state.  Concrete
/// subclasses (`ActionFlag`) implement only the required methods;
/// the default bodies stay a line-for-line port of upstream.
pub trait ActionFlagOps {
    /// pypy/interpreter/executioncontext.py:574-575 `ActionFlag.reset_ticker`.
    fn reset_ticker(&mut self, value: isize);

    /// pypy/interpreter/executioncontext.py:571-572 `ActionFlag.get_ticker`.
    fn get_ticker(&self) -> isize;

    /// pypy/interpreter/executioncontext.py:577-585 `ActionFlag.decrement_ticker`.
    fn decrement_ticker(&mut self, by: isize) -> isize;

    /// Composition accessor: shared `AbstractActionFlag` state.
    fn abstract_flag(&self) -> &AbstractActionFlag;
    /// Composition accessor: shared `AbstractActionFlag` state.
    fn abstract_flag_mut(&mut self) -> &mut AbstractActionFlag;

    /// pypy/interpreter/executioncontext.py:493-510
    /// `register_periodic_action`.
    ///
    /// ```python
    /// if use_bytecode_counter:
    ///     self._periodic_actions.append(action)
    ///     self.has_bytecode_counter = True
    /// else:
    ///     self._periodic_actions.insert(0, action)
    /// self._rebuild_action_dispatcher()
    /// ```
    ///
    /// PyPy comment (line 503-504): "hack to put the release-the-GIL
    /// one at the end of the list, and the report-the-signals one at
    /// the start of the list."  When `use_bytecode_counter` is False
    /// (signal handling), the action is prepended so it runs first.
    ///
    /// Lives on the trait (PyPy hangs `register_periodic_action` on
    /// `AbstractActionFlag` itself, never overridden by `ActionFlag`)
    /// so `*mut dyn ActionFlagOps` callers — the type pyre uses for
    /// `space.actionflag` to admit a future `SignalActionFlag`
    /// analogue — can reach it without downcasting.
    ///
    /// PyPy line 502 asserts `isinstance(action, PeriodicAsyncAction)`.
    /// Pyre enforces the same constraint at compile time by typing the
    /// argument as `*mut dyn PeriodicAsyncActionOps` — non-periodic
    /// `AsyncAction` subclasses do not impl that subtrait, so passing
    /// one is a type error rather than a runtime assert failure.
    fn register_periodic_action(
        &mut self,
        action: *mut dyn PeriodicAsyncActionOps,
        use_bytecode_counter: bool,
    ) {
        // Trait upcast (stable since Rust 1.86) erases the periodic
        // marker so storage stays as the polymorphic
        // `*mut dyn AsyncActionOps` that `action_dispatcher` reads.
        let action_ops: *mut dyn AsyncActionOps = action;
        let flag = self.abstract_flag_mut();
        if use_bytecode_counter {
            flag._periodic_actions.push(action_ops);
            flag.has_bytecode_counter = true;
        } else {
            flag._periodic_actions.insert(0, action_ops);
        }
        flag._rebuild_action_dispatcher();
    }

    /// pypy/interpreter/executioncontext.py:512-517
    /// `register_nonperiodic_action`.  Trait default for the same
    /// reason as `register_periodic_action` above.
    fn register_nonperiodic_action(&mut self, action: *mut dyn AsyncActionOps) -> isize {
        let flag = self.abstract_flag_mut();
        flag._nonperiodic_actions.push(action);
        assert!(flag._nonperiodic_actions.len() < 32);
        flag._rebuild_action_dispatcher();
        (flag._nonperiodic_actions.len() - 1) as isize
    }

    /// pypy/interpreter/executioncontext.py:519-520 `getcheckinterval`.
    fn getcheckinterval(&self) -> usize {
        self.abstract_flag().checkinterval_scaled / TICK_COUNTER_STEP
    }

    /// pypy/interpreter/executioncontext.py:482-490 `fire`.
    ///
    /// ```python
    /// def fire(self, action):
    ///     "Request for the action to be run before the next opcode."
    ///     assert action._action_index >= 0
    ///     mask = r_uint(1) << action._action_index
    ///     if not self._fired_bitmask & mask:
    ///         self._fired_bitmask |= mask
    ///         # set the ticker to -1 in order to force action_dispatcher()
    ///         # to run at the next possible bytecode
    ///         self.reset_ticker(-1)
    /// ```
    fn fire(&mut self, action: *mut dyn AsyncActionOps) {
        if action.is_null() {
            return;
        }
        let base = unsafe { (*action).async_action() };
        // executioncontext.py:484 — `assert action._action_index >= 0`.
        debug_assert!(
            base._action_index >= 0,
            "periodic actions must not call fire"
        );
        let bitmask = base.bitmask;
        let already_fired = self.abstract_flag()._fired_bitmask & bitmask != 0;
        if !already_fired {
            self.abstract_flag_mut()._fired_bitmask |= bitmask;
            self.reset_ticker(-1);
        }
    }

    /// pypy/interpreter/executioncontext.py:522-529 `setcheckinterval`.
    ///
    /// ```python
    /// def setcheckinterval(self, interval):
    ///     MAX = sys.maxint // TICK_COUNTER_STEP
    ///     if interval < 1:
    ///         interval = 1
    ///     elif interval > MAX:
    ///         interval = MAX
    ///     self.checkinterval_scaled = interval * TICK_COUNTER_STEP
    ///     self.reset_ticker(-1)
    /// ```
    fn setcheckinterval(&mut self, interval: usize) {
        let max = usize::MAX / TICK_COUNTER_STEP;
        let interval = interval.max(1).min(max);
        self.abstract_flag_mut().checkinterval_scaled = interval * TICK_COUNTER_STEP;
        self.reset_ticker(-1);
    }

    /// pypy/interpreter/executioncontext.py:531-562 `action_dispatcher`.
    ///
    /// ```python
    /// def action_dispatcher(ec, frame):
    ///     # periodic actions (first reset the bytecode counter)
    ///     self.reset_ticker(self.checkinterval_scaled)
    ///     for action in periodic_actions:
    ///         action.perform(ec, frame)
    ///     # nonperiodic actions
    ///     if self._fired_bitmask:
    ///         for i in range(len(self._nonperiodic_actions)):
    ///             mask = r_uint(1) << i
    ///             if self._fired_bitmask & mask:
    ///                 action = self._nonperiodic_actions[i]
    ///                 self._fired_bitmask &= ~mask
    ///                 action.perform(ec, frame)
    ///         if self._fired_bitmask:
    ///             self.reset_ticker(-1)
    /// ```
    ///
    /// A periodic/nonperiodic `perform` may raise (e.g.
    /// `CheckSignalAction` delivering `KeyboardInterrupt`); the error
    /// propagates out via `?`, aborting the remaining actions, exactly
    /// like PyPy's `action.perform(...)` raising an `OperationError`.
    fn action_dispatcher(
        &mut self,
        ec: *mut ExecutionContext,
        frame: *mut PyFrame,
    ) -> Result<(), crate::PyError> {
        // executioncontext.py:538 — `self.reset_ticker(self.checkinterval_scaled)`.
        let interval = self.abstract_flag().checkinterval_scaled as isize;
        self.reset_ticker(interval);
        // executioncontext.py:539-540 — periodic actions iter.
        // Snapshot pointers before iteration; perform() may register
        // additional actions and mutate `_periodic_actions`.
        let periodic: Vec<*mut dyn AsyncActionOps> = self.abstract_flag()._periodic_actions.clone();
        for action_ptr in periodic {
            if action_ptr.is_null() || ec.is_null() {
                continue;
            }
            unsafe {
                (*action_ptr).perform(&mut *ec, frame)?;
            }
        }
        // executioncontext.py:543-556 — nonperiodic bit-mask scan.
        // Clear each bit before perform() so a fire() during perform()
        // re-arms cleanly (NB at executioncontext.py:544-549).
        if self.abstract_flag()._fired_bitmask != 0 {
            let nactions = self.abstract_flag()._nonperiodic_actions.len();
            for i in 0..nactions {
                let mask: usize = 1usize << i;
                if self.abstract_flag()._fired_bitmask & mask != 0 {
                    let action_ptr = self.abstract_flag()._nonperiodic_actions[i];
                    self.abstract_flag_mut()._fired_bitmask &= !mask;
                    if !action_ptr.is_null() && !ec.is_null() {
                        unsafe {
                            (*action_ptr).perform(&mut *ec, frame)?;
                        }
                    }
                }
            }
            // executioncontext.py:560-561 — if a higher-index action
            // re-fired an earlier bit during iteration, force the next
            // bytecode to re-enter dispatch.
            if self.abstract_flag()._fired_bitmask != 0 {
                self.reset_ticker(-1);
            }
        }
        Ok(())
    }
}

/// pypy/interpreter/executioncontext.py:566-585 `ActionFlag` merged with
/// pypy/module/signal/interp_signal.py:24-51 `SignalActionFlag`.
///
/// PyPy starts with a plain `ActionFlag` (its ticker is a Python field)
/// and, when the `signal` module loads, rebinds `space.actionflag` to a
/// `SignalActionFlag` whose ticker IS the C `pypysig_counter` cell so the
/// OS signal handler can force it negative.  pyre merges the two: the
/// ticker stays a plain `_ticker` field (a plain field read is what the
/// JIT codewriter can model in the per-bytecode hot path — an atomic /
/// volatile global read is not), and the OS signal handler writes -1 into
/// it through a pointer registered at startup
/// (`signalstate::register_ticker` ← `ticker_addr`).  This is the same
/// arrangement as upstream's volatile `pypysig_counter.value`: the
/// handler stores through the cell's address while the interpreter reads
/// the field directly.
#[derive(Clone)]
pub struct ActionFlag {
    base: AbstractActionFlag,
    _ticker: isize,
}

impl Default for ActionFlag {
    fn default() -> Self {
        Self::new()
    }
}

impl ActionFlag {
    pub fn new() -> Self {
        Self {
            base: AbstractActionFlag::new(),
            _ticker: 0,
        }
    }

    pub fn perform_frame_action(&mut self, ec: &mut ExecutionContext, frame: *mut PyFrame) {
        let _ = (ec, frame);
    }

    /// Address of the ticker cell, handed to `signalstate::register_ticker`
    /// so the OS signal handler can force the ticker negative.  Stable for
    /// the process lifetime — the `ExecutionContext` is created once and
    /// never moved (held behind an `Rc` in pyrex).
    pub fn ticker_addr(&mut self) -> *mut isize {
        &mut self._ticker
    }

    /// True when `self._ticker` is the signal-registered ticker cell — the
    /// single cell compiled-loop back-edges poll. Only that ticker drives the
    /// shared async bit; per-EC flags that are not the registered breaker
    /// source must not touch the shared word, or one context's dispatch clear
    /// would drop another context's pending async. wasm builds have no signal
    /// module (no registered ticker), so the mirror is inert there.
    ///
    /// The cell identity is compared as `usize` rather than via
    /// `ptr::eq`: this runs inside the traced eval loop (`decrement_ticker`),
    /// and a raw-pointer equality on `*const isize` can lower to an
    /// `int_eq/ir>i` kind shape that has no blackhole handler; casting both
    /// addresses to `usize` keeps the comparison an int/int equality.
    #[cfg(not(target_arch = "wasm32"))]
    fn is_registered_ticker(&self) -> bool {
        let here = &self._ticker as *const isize as usize;
        let registered = crate::module::signal::signalstate::registered_ticker_ptr() as usize;
        here == registered
    }

    #[cfg(target_arch = "wasm32")]
    fn is_registered_ticker(&self) -> bool {
        false
    }
}

/// pypy/interpreter/executioncontext.py:566-585 — `class ActionFlag(AbstractActionFlag)`.
impl ActionFlagOps for ActionFlag {
    fn abstract_flag(&self) -> &AbstractActionFlag {
        &self.base
    }

    fn abstract_flag_mut(&mut self) -> &mut AbstractActionFlag {
        &mut self.base
    }

    /// interp_signal.py:30-32 `SignalActionFlag.get_ticker` — `p.c_value`.
    /// The cell is the `_ticker` field; the OS handler writes it through
    /// the registered pointer (`ticker_addr`).
    fn get_ticker(&self) -> isize {
        self._ticker
    }

    /// interp_signal.py:34-36 `SignalActionFlag.reset_ticker` —
    /// `p.c_value = value`.
    fn reset_ticker(&mut self, value: isize) {
        self._ticker = value;
        if self.is_registered_ticker() {
            if value < 0 {
                majit_ir::eval_breaker_word::set_async();
            } else {
                majit_ir::eval_breaker_word::clear_async();
                // A signal delivered between the ticker store and this clear
                // rearms the ticker to -1 and re-sets the async bit; the clear
                // would then drop it, leaving a negative ticker with the bit
                // clear so a non-allocating compiled loop misses the signal.
                // Re-read the ticker — the handler writes it through the
                // registered pointer, so force a fresh load — and restore the
                // bit if it was rearmed.
                if unsafe { std::ptr::read_volatile(&self._ticker) } < 0 {
                    majit_ir::eval_breaker_word::set_async();
                }
            }
        }
    }

    /// interp_signal.py:42-51 `SignalActionFlag.decrement_ticker`.
    ///
    /// ```python
    /// def decrement_ticker(self, by):
    ///     p = pypysig_getaddr_occurred()
    ///     value = p.c_value
    ///     if self.has_bytecode_counter:    # this 'if' is constant-folded
    ///         if jit.isconstant(by) and by == 0:
    ///             pass     # normally constant-folded too
    ///         else:
    ///             value -= by
    ///             p.c_value = value
    ///     return value
    /// ```
    ///
    /// `CheckSignalAction` registers with `use_bytecode_counter=False`,
    /// so `has_bytecode_counter` stays false and the ticker is never
    /// decremented here — it only goes negative when the OS handler calls
    /// `signal_pushback` (or `fire` requests a non-periodic action).
    fn decrement_ticker(&mut self, by: isize) -> isize {
        if self.base.has_bytecode_counter {
            self._ticker -= by;
            // This path bypasses reset_ticker, so mirror a future periodic
            // decrement that crosses negative.
            if self.is_registered_ticker() && self._ticker < 0 {
                majit_ir::eval_breaker_word::set_async();
            }
        }
        self._ticker
    }
}

pub struct AsyncAction {
    pub space: PyObjectRef,
    _action_index: isize,
    /// pypy/interpreter/executioncontext.py:600 `self.bitmask = 1 << index`.
    /// Set by `AbstractActionFlag::register_nonperiodic_action` and
    /// consumed by `fire` / `action_dispatcher`'s bit test.
    pub bitmask: usize,
    /// pypy/interpreter/executioncontext.py:603-606 `AsyncAction.fire`
    /// uses `self.space.actionflag` — a constant lookup once the action
    /// is constructed because PyPy's `space.actionflag` is set once at
    /// `pypy/interpreter/baseobjspace.py:447` and never replaced.
    /// Pyre keeps the actionflag on `ExecutionContext` rather than on
    /// `space`, so we cache the back-reference at registration time
    /// (`AsyncAction::new` / `UserDelAction::new` /
    /// `AsyncActionOps::register_periodic_action`).  The pointer is
    /// stable for the process lifetime because pyrex owns the EC for
    /// the entire run.  `fire()` dereferences this slot directly,
    /// matching PyPy's `self.space.actionflag.fire(self)` 1:1 without
    /// the TLS detour.  `null` means the action has not been registered
    /// yet — calling `fire` in that state is a programmer error
    /// (PyPy's docstring at line 605 reads "The action must have been
    /// registered at space initalization time.").
    ///
    /// Stored as `*mut dyn ActionFlagOps` so that `fire()` can dispatch
    /// through the `ActionFlagOps::fire` trait default (which lives on
    /// the trait, mirroring PyPy's polymorphic `actionflag.fire` call)
    /// AND so a future `SignalActionFlag` analogue (PyPy's signal-aware
    /// `AbstractActionFlag` subclass) can be installed without touching
    /// `AsyncAction`.  PyPy types `space.actionflag` as the abstract
    /// base, never as the concrete subclass.
    pub actionflag: *mut dyn ActionFlagOps,
}

impl Default for AsyncAction {
    fn default() -> Self {
        Self {
            space: pyre_object::PY_NULL,
            _action_index: -1,
            bitmask: 0,
            // Fat pointer: null data + ActionFlag vtable.  `fire()` /
            // `register_periodic_action` tests `is_null()` on the data
            // pointer before any deref, so the placeholder vtable never
            // gets reached on an unregistered action.
            actionflag: std::ptr::null_mut::<ActionFlag>(),
        }
    }
}

impl AsyncAction {
    /// pypy/interpreter/executioncontext.py:594-600 `AsyncAction.__init__`.
    ///
    /// ```python
    /// def __init__(self, space):
    ///     self.space = space
    ///     if not isinstance(self, PeriodicAsyncAction):
    ///         self._action_index = self.space.actionflag.register_nonperiodic_action(self)
    ///     else:
    ///         self._action_index = -1
    /// ```
    ///
    /// Returns `Box<Self>` because Rust cannot hand out a stable
    /// pointer to a value before the constructor returns; PyPy's
    /// heap-identity makes `register_nonperiodic_action(self)` safe
    /// in-place.  Caller owns the Box and must keep it alive as long
    /// as the actionflag holds the registered pointer.
    pub fn new(space: PyObjectRef, actionflag: &mut (dyn ActionFlagOps + 'static)) -> Box<Self> {
        let mut action = Box::new(Self {
            space,
            _action_index: -1,
            bitmask: 0,
            actionflag: actionflag as *mut dyn ActionFlagOps,
        });
        let action_ptr: *mut dyn AsyncActionOps = &mut *action;
        let index = actionflag.register_nonperiodic_action(action_ptr);
        action._action_index = index;
        action.bitmask = 1usize << (index as usize);
        action
    }

    /// pypy/interpreter/executioncontext.py:597-600 `else` branch:
    /// `self._action_index = -1`.  Used by PeriodicAsyncAction
    /// subclasses whose `__init__` skips `register_nonperiodic_action`.
    /// The base `AsyncAction` is returned unboxed because
    /// `PeriodicAsyncAction::new` wraps it in a `Box<PeriodicAsyncAction>`
    /// — registration with `register_periodic_action` then captures the
    /// outer Box's stable pointer.
    pub fn new_periodic_base(space: PyObjectRef) -> Self {
        Self {
            space,
            _action_index: -1,
            bitmask: 0,
            // PeriodicAsyncAction subclasses receive their actionflag
            // back-reference at `register_periodic_action` time (the
            // trait default below) rather than at construction —
            // pyjitpl-style mirror of PyPy's two-step `__init__` +
            // `space.actionflag.register_periodic_action(...)` flow.
            //
            // Fat null pointer constructed via ActionFlag upcast — see
            // the `Default` impl above for the same pattern.
            actionflag: std::ptr::null_mut::<ActionFlag>(),
        }
    }
}

/// pypy/interpreter/executioncontext.py:588-609 — `AsyncAction` virtual surface.
///
/// PyPy resolves `action.perform(ec, frame)` through Python class
/// lookup (line 608-609 — base body is "To be overridden.").  Rust
/// composition cannot virtual-dispatch from a base-struct inherent
/// method, so this trait carries the virtual `perform` entry plus an
/// accessor pair (`async_action` / `async_action_mut`) for the shared
/// `_action_index` / `bitmask` / `space` state held on `AsyncAction`.
/// Concrete actions impl the trait; the action lists store
/// `*mut dyn AsyncActionOps` so the dispatcher reaches the override
/// through the embedded vtable, and `fire` / `action_dispatcher` can
/// reach the base bitmask through the accessor.
pub trait AsyncActionOps {
    /// pypy/interpreter/executioncontext.py:608-609 `AsyncAction.perform`:
    /// `def perform(self, executioncontext, frame): "To be overridden."`
    ///
    /// Returns `Result` so an overriding action (e.g. `CheckSignalAction`)
    /// can raise — PyPy's `perform` propagates an `OperationError` up
    /// through `action_dispatcher` to the eval loop.
    fn perform(
        &mut self,
        executioncontext: &mut ExecutionContext,
        frame: *mut PyFrame,
    ) -> Result<(), crate::PyError>;

    /// Composition accessor: shared `AsyncAction` state.
    fn async_action(&self) -> &AsyncAction;
    /// Composition accessor: shared `AsyncAction` state.
    fn async_action_mut(&mut self) -> &mut AsyncAction;

    /// pypy/interpreter/executioncontext.py:602-606 `AsyncAction.fire`:
    ///
    /// ```python
    /// @rgc.no_collect
    /// def fire(self):
    ///     "Request for the action to be run before the next opcode."
    ///     self.space.actionflag.fire(self)
    /// ```
    ///
    /// Pyre stores `actionflag` on `ExecutionContext` rather than on
    /// `space` (TODO — pyre's space is an opaque
    /// `PyObjectRef`).  To keep PyPy's `self.space.actionflag.fire(self)`
    /// directness, the actionflag back-reference is cached on the base
    /// `AsyncAction` at registration time (see
    /// `AsyncAction::new` / `UserDelAction::new` / the trait default
    /// `register_periodic_action` above).  `fire()` dereferences that
    /// back-ref unconditionally; the pointer is stable for the process
    /// lifetime because pyrex owns the EC for the entire run.  The
    /// `Self: Sized + 'static` bound captures the concrete subclass
    /// vtable in the `*mut dyn AsyncActionOps` fat pointer so
    /// `actionflag.fire`'s `(*action).async_action()` reads through the
    /// override-aware vtable rather than the bare-base one.
    ///
    /// PyPy's docstring at line 605 reads "The action must have been
    /// registered at space initalization time."  Pyre asserts the same:
    /// `self.actionflag.is_null()` would mean the action skipped both
    /// `register_nonperiodic_action` and `register_periodic_action`,
    /// which is a programmer error.
    fn fire(&mut self)
    where
        Self: Sized + 'static,
    {
        let actionflag = self.async_action().actionflag;
        debug_assert!(
            !actionflag.is_null(),
            "AsyncAction::fire called before registration (executioncontext.py:605)"
        );
        let action_ptr: *mut dyn AsyncActionOps = self;
        // executioncontext.py:606 — `self.space.actionflag.fire(self)`.
        // The trait dispatch reaches the `ActionFlagOps::fire` default
        // body at line 1454 (which encodes executioncontext.py:482-490
        // `AbstractActionFlag.fire`).
        unsafe { (*actionflag).fire(action_ptr) };
    }
}

/// pypy/interpreter/executioncontext.py:588-609 — bare `AsyncAction`.
/// Default `perform` is the "To be overridden." no-op; concrete
/// subclasses override.
impl AsyncActionOps for AsyncAction {
    fn perform(
        &mut self,
        _executioncontext: &mut ExecutionContext,
        _frame: *mut PyFrame,
    ) -> Result<(), crate::PyError> {
        Ok(())
    }

    fn async_action(&self) -> &AsyncAction {
        self
    }

    fn async_action_mut(&mut self) -> &mut AsyncAction {
        self
    }
}

/// pypy/interpreter/executioncontext.py:612-615 `class
/// PeriodicAsyncAction(AsyncAction)`.  Marker subtrait that narrows
/// `space.actionflag.register_periodic_action(self, ...)` so only
/// types descending from `PeriodicAsyncAction` (the trait analogue of
/// PyPy's class hierarchy) can be registered as periodic.  PyPy
/// guards the same constraint at runtime with
/// `assert isinstance(action, PeriodicAsyncAction)` in
/// `executioncontext.py:502`; pyre lifts the assertion to the type
/// system.
pub trait PeriodicAsyncActionOps: AsyncActionOps {
    /// pypy/interpreter/executioncontext.py:594-600 `else` branch
    /// helper: subclasses of `PeriodicAsyncAction` call
    /// `space.actionflag.register_periodic_action(self, ...)` after
    /// `__init__`.  Lifted into this subtrait so the
    /// `*mut dyn PeriodicAsyncActionOps` cast captures the concrete
    /// subclass vtable (which carries the override of `perform`);
    /// `Self: Sized` keeps trait-object dispatch off this method.
    fn register_periodic_action(
        &mut self,
        actionflag: &mut (dyn ActionFlagOps + 'static),
        use_bytecode_counter: bool,
    ) where
        Self: Sized + 'static,
    {
        // executioncontext.py:603-606 — cache the actionflag back-ref
        // on the base AsyncAction so `fire()` can reach it without a
        // TLS lookup.  PyPy's `self.space.actionflag` is a constant
        // lookup once `space.actionflag` is set at
        // `pypy/interpreter/baseobjspace.py:447` and never replaced;
        // storing the resolved pointer here is the same caching the
        // PyPy implementation does implicitly.
        self.async_action_mut().actionflag = actionflag as *mut dyn ActionFlagOps;
        let action_ptr: *mut dyn PeriodicAsyncActionOps = self;
        actionflag.register_periodic_action(action_ptr, use_bytecode_counter);
    }
}

pub struct PeriodicAsyncAction {
    pub base: AsyncAction,
}

impl PeriodicAsyncAction {
    /// pypy/interpreter/executioncontext.py:594-600 `AsyncAction.__init__`
    /// `else` arm (`isinstance(self, PeriodicAsyncAction)`): only sets
    /// `_action_index = -1`, leaves registration to subclass-specific
    /// `space.actionflag.register_periodic_action(self, use_bytecode_counter)`.
    ///
    /// Returns `Box<Self>` because `register_periodic_action` (called
    /// after construction by the subclass via `register`) takes a
    /// stable `*mut dyn AsyncActionOps`.
    pub fn new(space: PyObjectRef) -> Box<Self> {
        Box::new(Self {
            base: AsyncAction::new_periodic_base(space),
        })
    }

    // pypy/interpreter/executioncontext.py:594-600 — PeriodicAsyncAction
    // subclasses call `space.actionflag.register_periodic_action(self,
    // use_bytecode_counter)` after their own `__init__`.  In pyre this
    // is the trait default `AsyncActionOps::register_periodic_action`
    // — it must be reached through the concrete subclass's vtable so
    // the registered fat pointer's `perform` slot points to the
    // override, not `PeriodicAsyncAction::perform` (which is a no-op
    // per `executioncontext.py:612-615`).  Never call
    // `PeriodicAsyncAction(...).register_periodic_action(...)`
    // directly without a concrete subclass.
}

/// pypy/interpreter/executioncontext.py:612-615 — `class
/// PeriodicAsyncAction(AsyncAction)`: empty body, inherits the
/// "To be overridden." `perform` from `AsyncAction`.  Concrete
/// subclasses (CheckSignalAction etc.) override.
impl AsyncActionOps for PeriodicAsyncAction {
    fn perform(
        &mut self,
        _executioncontext: &mut ExecutionContext,
        _frame: *mut PyFrame,
    ) -> Result<(), crate::PyError> {
        Ok(())
    }

    fn async_action(&self) -> &AsyncAction {
        &self.base
    }

    fn async_action_mut(&mut self) -> &mut AsyncAction {
        &mut self.base
    }
}

/// pypy/interpreter/executioncontext.py:612-615 — `PeriodicAsyncAction`
/// admits `space.actionflag.register_periodic_action(self, ...)`.
impl PeriodicAsyncActionOps for PeriodicAsyncAction {}

pub struct UserDelAction {
    pub base: AsyncAction,
    pub finalizers_lock_count: usize,
    pub enabled_at_app_level: bool,
    pub pending_with_disabled_del: Option<Vec<PyObjectRef>>,
    /// `pypy/interpreter/executioncontext.py:640` —
    /// `self.space.finalizer_queue` access target.
    ///
    /// TODO: PyPy reads the queue via `self.space`
    /// (typed `ObjSpace`).  Pyre's `space` is an opaque `PyObjectRef`,
    /// so the queue is held as a UserDelAction field instead.  When
    /// GC integration lands a typed space surface this
    /// can be folded back into `space.finalizer_queue`.
    pub finalizer_queue: WRootFinalizerQueue,
}

impl UserDelAction {
    /// pypy/interpreter/executioncontext.py:618-631 `UserDelAction.__init__`.
    ///
    /// ```python
    /// class UserDelAction(AsyncAction):
    ///     def __init__(self, space):
    ///         AsyncAction.__init__(self, space)
    ///         self.finalizers_lock_count = 0
    ///         self.enabled_at_app_level = True
    ///         self.pending_with_disabled_del = None
    /// ```
    ///
    /// `AsyncAction.__init__` (executioncontext.py:594-600) registers
    /// the live instance as a nonperiodic action because UserDelAction
    /// is NOT a PeriodicAsyncAction subclass.  Pyre boxes the
    /// UserDelAction so registration captures a stable
    /// `*mut dyn AsyncActionOps` whose vtable dispatches into
    /// `UserDelAction::perform` (executioncontext.py:632-633).
    pub fn new(space: PyObjectRef, actionflag: &mut (dyn ActionFlagOps + 'static)) -> Box<Self> {
        let mut action = Box::new(Self {
            base: AsyncAction {
                space,
                _action_index: -1,
                bitmask: 0,
                // executioncontext.py:603-606 — cache actionflag
                // back-ref for `fire()` (mirror of PyPy's
                // `self.space.actionflag` constant lookup).
                actionflag: actionflag as *mut dyn ActionFlagOps,
            },
            finalizers_lock_count: 0,
            enabled_at_app_level: true,
            pending_with_disabled_del: None,
            finalizer_queue: WRootFinalizerQueue,
        });
        let action_ptr: *mut dyn AsyncActionOps = &mut *action;
        let index = actionflag.register_nonperiodic_action(action_ptr);
        action.base._action_index = index;
        action.base.bitmask = 1usize << (index as usize);
        action
    }

    /// `pypy/interpreter/executioncontext.py:636-643`:
    /// ```python
    /// def _run_finalizers(self):
    ///     # called by perform() when we have to "perform" this action,
    ///     # and also directly at the end of gc.collect).
    ///     while True:
    ///         w_obj = self.space.finalizer_queue.next_dead()
    ///         if w_obj is None:
    ///             break
    ///         self._call_finalizer(w_obj)
    /// ```
    ///
    /// `self.space.finalizer_queue` is read via the local
    /// `self.finalizer_queue` field (see struct doc for
    /// TODO).
    pub fn _run_finalizers(&mut self) {
        loop {
            let w_obj = self.finalizer_queue.next_dead();
            match w_obj {
                None => break,
                Some(w) => {
                    // The deque entry was the object's only root; pin it across the __del__ call.
                    let _roots = pyre_object::gc_roots::push_roots();
                    pyre_object::gc_roots::pin_root(w);
                    self._call_finalizer(w);
                }
            }
        }
    }

    pub fn gc_disabled(&mut self, w_obj: PyObjectRef) -> bool {
        let _ = w_obj;
        if let Some(list) = self.pending_with_disabled_del.as_mut() {
            list.push(w_obj);
            true
        } else {
            false
        }
    }

    pub fn _call_finalizer(&mut self, w_obj: PyObjectRef) {
        if unsafe { pyre_object::generator::is_generator(w_obj) } {
            if self.gc_disabled(w_obj) {
                return;
            }
            if let Err(error) = crate::baseobjspace::generator_finalize(w_obj) {
                report_error(self.base.space, &error, "", w_obj);
            }
            return;
        }
        let Some(w_type) = crate::typedef::r#type(w_obj) else {
            return;
        };
        let Some(w_del) = (unsafe { crate::baseobjspace::lookup_in_type(w_type, "__del__") })
        else {
            return;
        };
        if self.gc_disabled(w_obj) {
            return;
        }
        // pyre's combined helper cannot distinguish get-vs-call errors;
        // report through the call arm (executioncontext.py:680-690).
        if let Err(error) =
            unsafe { crate::baseobjspace::get_and_call_function(w_del, w_obj, w_type, &[]) }
        {
            report_error(self.base.space, &error, "", w_del);
        }
    }
}

/// pypy/interpreter/executioncontext.py:618-633 — `class
/// UserDelAction(AsyncAction)`.  `perform` (line 632-633) calls
/// `self._run_finalizers()`.
impl AsyncActionOps for UserDelAction {
    fn perform(
        &mut self,
        _executioncontext: &mut ExecutionContext,
        _frame: *mut PyFrame,
    ) -> Result<(), crate::PyError> {
        self._run_finalizers();
        Ok(())
    }

    fn async_action(&self) -> &AsyncAction {
        &self.base
    }

    fn async_action_mut(&mut self) -> &mut AsyncAction {
        &mut self.base
    }
}

pub fn report_error(
    space: PyObjectRef,
    error: &crate::PyError,
    where_desc: &str,
    w_obj: PyObjectRef,
) {
    let mut error = error.clone();
    error.write_unraisable(space, where_desc, w_obj);
}

pub fn make_finalizer_queue<WRoot>(w_root: WRoot, _space: PyObjectRef) -> WRootFinalizerQueue {
    let _ = w_root;
    WRootFinalizerQueue
}
