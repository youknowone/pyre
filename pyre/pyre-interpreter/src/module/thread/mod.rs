//! `_thread` — direct port of `pypy/module/thread/`.
//!
//! The ownership shape follows PyPy: each Lock/RLock/ThreadHandle owns its
//! native synchronization state, and each started OS thread gets a distinct
//! ExecutionContext plus GC shadow stack.  No process-global side table is
//! used for per-object state.

use parking_lot::{Condvar, Mutex};
use pyre_object::*;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicUsize, Ordering};
use std::sync::{LazyLock, OnceLock};
use std::time::{Duration, Instant};

const TIMEOUT_MAX: f64 = i64::MAX as f64 / 1_000_000.0;
static THREAD_COUNT: AtomicI64 = AtomicI64::new(0);
static STACK_SIZE: AtomicUsize = AtomicUsize::new(0);
static FINALIZING: AtomicBool = AtomicBool::new(false);
static FINALIZING_THREAD: AtomicI64 = AtomicI64::new(0);
// Number of EC-owned `w_async_exception_type` slots which are non-null.
// This keeps the process eval breaker armed until the targeted free-threaded
// EC observes its own pending exception; another EC must not clear the shared
// breaker first.
static ASYNC_EXCEPTION_COUNT: AtomicUsize = AtomicUsize::new(0);
static TRACE_ALL_GENERATION: AtomicUsize = AtomicUsize::new(0);
static PROFILE_ALL_GENERATION: AtomicUsize = AtomicUsize::new(0);
static TRACE_ALL_HOOK: Mutex<usize> = parking_lot::const_mutex(0);
static PROFILE_ALL_HOOK: Mutex<usize> = parking_lot::const_mutex(0);
// CPython 3.14's `_thread._shutdown` registry, corresponding to PyPy's
// bootstrapper/threadlocals-owned live-thread set.  Values are handle object
// slots, not a parallel copy of handle state.
static SHUTDOWN_HANDLES: Mutex<Vec<usize>> = parking_lot::const_mutex(Vec::new());
// CPython's native thread-handle list, used by `_PyThread_AfterFork()` to
// mark handles owned by vanished threads done before `threading._after_fork`
// walks the Python Thread objects.
static ACTIVE_HANDLES: Mutex<Vec<usize>> = parking_lot::const_mutex(Vec::new());
// `OSThreadLocals._valuedict`: process/interpreter-owned mapping from native
// thread identifiers to their live ExecutionContexts.
static EXECUTION_CONTEXTS: LazyLock<Mutex<indexmap::IndexMap<i64, usize>>> =
    LazyLock::new(|| Mutex::new(indexmap::IndexMap::new()));

/// A blocking host call leaves the free-threaded collector's RUNNING census.
/// This is a GC safepoint transition only; pyre has no process GIL.
pub(crate) fn before_external_block() -> majit_gc::gc_sync::BlockingGuard {
    majit_gc::gc_sync::before_external_block()
}

pub fn set_finalizing() {
    let ident = current_ident();
    // Free-threaded counterpart of PyPy/CPython's final GIL ownership: publish
    // the finalizing owner while every other mutator is stopped at a walkable
    // safepoint.  Mutators which resume into Python bytecode park permanently;
    // mutators already returning from their target finish unregistering before
    // this STW can complete.
    majit_gc::gc_sync::request_stw(|_| {
        FINALIZING_THREAD.store(ident, Ordering::Release);
        FINALIZING.store(true, Ordering::Release);
    });
}

pub fn is_finalizing() -> bool {
    FINALIZING.load(Ordering::Acquire)
}

/// Stop a non-owner mutator from running Python once interpreter teardown has
/// begun.  The forgotten blocking guard keeps it outside the GC RUNNING census;
/// process exit terminates these daemon OS threads after the owner completes
/// finalization, matching the upstream "hang daemon thread" shutdown state.
#[inline]
pub fn park_if_finalizing() {
    if !is_finalizing() || FINALIZING_THREAD.load(Ordering::Acquire) == current_ident() {
        return;
    }
    let blocked = before_external_block();
    std::mem::forget(blocked);
    #[cfg(not(target_arch = "wasm32"))]
    loop {
        std::thread::park();
    }
    #[cfg(target_arch = "wasm32")]
    loop {
        std::hint::spin_loop();
    }
}

pub(crate) fn walk_thread_roots(visitor: &mut dyn FnMut(&mut majit_ir::GcRef)) {
    // PyPy's ExecutionContexts live on the translated interpreter object graph,
    // so every object-valued EC field is traced automatically.  Pyre keeps one
    // Rust Box per OS thread; the process-owned OSThreadLocals registry is
    // therefore the corresponding root owner and must forward those fields in
    // place.  In particular, tracing/profile callbacks must never become stale
    // raw pointers after a moving collection.
    {
        let contexts = EXECUTION_CONTEXTS.lock();
        for &ec_addr in contexts.values() {
            let ec = unsafe { &mut *(ec_addr as *mut crate::PyExecutionContext) };
            let mut forward = |slot: &mut PyObjectRef| {
                if !slot.is_null() {
                    visitor(unsafe { &mut *(slot as *mut PyObjectRef as *mut majit_ir::GcRef) });
                }
            };
            forward(&mut ec.space);
            forward(&mut ec.w_tracefunc);
            forward(&mut ec.compiler);
            forward(&mut ec.w_profilefuncarg);
            forward(&mut ec.w_async_exception_type);
            for wref in &mut ec.thread_local_refs {
                forward(wref);
            }
            if !ec.topframeref.is_null() {
                let mut frame = ec.topframeref as PyObjectRef;
                forward(&mut frame);
                ec.topframeref = frame as *mut crate::PyFrame;
            }
            if !ec.user_del_action.is_null() {
                let action = unsafe { &mut *ec.user_del_action };
                forward(&mut action.base.space);
                if let Some(pending) = action.pending_with_disabled_del.as_mut() {
                    for obj in pending {
                        forward(obj);
                    }
                }
            }
        }
    }
    let mut forwarded = Vec::new();
    for handle in SHUTDOWN_HANDLES.lock().iter_mut() {
        let old = *handle;
        visitor(unsafe { &mut *(handle as *mut usize as *mut majit_ir::GcRef) });
        forwarded.push((old, *handle));
    }
    for handle in ACTIVE_HANDLES.lock().iter_mut() {
        if let Some((_, new)) = forwarded.iter().find(|(old, _)| *old == *handle) {
            *handle = *new;
        } else {
            visitor(unsafe { &mut *(handle as *mut usize as *mut majit_ir::GcRef) });
        }
    }
    visitor(unsafe { &mut *(&mut *TRACE_ALL_HOOK.lock() as *mut usize as *mut majit_ir::GcRef) });
    visitor(unsafe { &mut *(&mut *PROFILE_ALL_HOOK.lock() as *mut usize as *mut majit_ir::GcRef) });
}

pub(crate) fn register_execution_context(ec: *const crate::PyExecutionContext) {
    EXECUTION_CONTEXTS
        .lock()
        .insert(current_ident(), ec as usize);
}

pub(crate) fn unregister_execution_context() {
    EXECUTION_CONTEXTS.lock().shift_remove(&current_ident());
}

pub(crate) fn take_async_exception(ec: *mut crate::PyExecutionContext) -> PyObjectRef {
    let _contexts = EXECUTION_CONTEXTS.lock();
    unsafe {
        let w_type = (*ec).w_async_exception_type;
        (*ec).w_async_exception_type = PY_NULL;
        if !w_type.is_null() {
            ASYNC_EXCEPTION_COUNT.fetch_sub(1, Ordering::AcqRel);
        }
        w_type
    }
}

pub(crate) fn has_pending_async_exception() -> bool {
    ASYNC_EXCEPTION_COUNT.load(Ordering::Acquire) != 0
}

/// CPython 3.14 `sys._settraceallthreads` / `_setprofileallthreads`.
///
/// Publish the requested hook process-wide, then let each OS thread apply it
/// to its own PyPy ExecutionContext at the next bytecode boundary.  This keeps
/// every EC field single-writer under free threading; no caller writes another
/// live thread's `w_tracefunc` / profile tuple.
pub(crate) fn set_trace_all_execution_contexts(w_func: PyObjectRef) {
    *TRACE_ALL_HOOK.lock() = w_func as usize;
    TRACE_ALL_GENERATION.fetch_add(1, Ordering::Release);
    let ec = crate::call::getexecutioncontext() as *mut crate::PyExecutionContext;
    if !ec.is_null() {
        let _ = unsafe { apply_all_thread_hooks(&mut *ec) };
    }
}

pub(crate) fn set_profile_all_execution_contexts(
    w_func: PyObjectRef,
) -> Result<(), crate::PyError> {
    *PROFILE_ALL_HOOK.lock() = w_func as usize;
    PROFILE_ALL_GENERATION.fetch_add(1, Ordering::Release);
    let ec = crate::call::getexecutioncontext() as *mut crate::PyExecutionContext;
    if !ec.is_null() {
        unsafe { apply_all_thread_hooks(&mut *ec)? };
    }
    Ok(())
}

pub(crate) fn apply_all_thread_hooks(
    ec: &mut crate::PyExecutionContext,
) -> Result<(), crate::PyError> {
    let trace_generation = TRACE_ALL_GENERATION.load(Ordering::Acquire);
    if ec.trace_all_generation != trace_generation {
        let w_func = *TRACE_ALL_HOOK.lock() as PyObjectRef;
        ec.settrace(w_func);
        ec.trace_all_generation = trace_generation;
    }
    let profile_generation = PROFILE_ALL_GENERATION.load(Ordering::Acquire);
    if ec.profile_all_generation != profile_generation {
        let w_func = *PROFILE_ALL_HOOK.lock() as PyObjectRef;
        ec.setprofile(w_func)?;
        ec.profile_all_generation = profile_generation;
    }
    Ok(())
}

/// CPython compatibility entry used by `test_threading`; PyPy keeps the same
/// pending exception on `ExecutionContext.w_async_exception_type`.
#[unsafe(no_mangle)]
pub extern "C" fn PyThreadState_SetAsyncExc(ident: usize, w_type: PyObjectRef) -> i32 {
    let contexts = EXECUTION_CONTEXTS.lock();
    let Some(&ec) = contexts.get(&(ident as i64)) else {
        return 0;
    };
    unsafe {
        let slot = &mut (*(ec as *mut crate::PyExecutionContext)).w_async_exception_type;
        match (slot.is_null(), w_type.is_null()) {
            (true, false) => {
                ASYNC_EXCEPTION_COUNT.fetch_add(1, Ordering::AcqRel);
            }
            (false, true) => {
                ASYNC_EXCEPTION_COUNT.fetch_sub(1, Ordering::AcqRel);
            }
            _ => {}
        }
        *slot = w_type;
    }
    // pypy/module/__pypy__/interp_signal.py:_raise_in_thread:
    // `space.actionflag.rearm_ticker()` after updating the EC-owned slot.
    #[cfg(not(target_arch = "wasm32"))]
    crate::module::signal::signalstate::rearm_ticker();
    #[cfg(target_arch = "wasm32")]
    majit_ir::eval_breaker_word::set_async();
    1
}

/// Free-threaded compatibility shims: entering Python does not acquire a GIL.
#[unsafe(no_mangle)]
pub extern "C" fn PyGILState_Ensure() -> i32 {
    0
}

#[unsafe(no_mangle)]
pub extern "C" fn PyGILState_Release(_state: i32) {}

/// `pypy/module/sys/threadmappings.py:_current_frames`.
pub(crate) fn current_frames() -> PyObjectRef {
    let roots = pyre_object::gc_roots::push_roots();
    let base = pyre_object::gc_roots::shadow_stack_len();
    let mut entries = Vec::new();
    majit_gc::gc_sync::request_stw(|_| {
        let contexts = EXECUTION_CONTEXTS.lock();
        for (&ident, &ec) in contexts.iter() {
            let frame =
                unsafe { (*(ec as *const crate::PyExecutionContext)).gettopframe_nohidden() };
            if !frame.is_null() {
                unsafe { (*frame).mark_as_escaped() };
                pyre_object::gc_roots::pin_root(frame as PyObjectRef);
                entries.push(ident);
            }
        }
    });
    let result = w_dict_new();
    for (index, ident) in entries.into_iter().enumerate() {
        let frame = pyre_object::gc_roots::shadow_stack_get(base + index);
        unsafe { w_dict_setitem(result, ident, frame) };
    }
    drop(roots);
    result
}

/// `pypy/module/thread/os_thread.py:reinit_threads`.
pub(crate) fn after_fork_child() {
    let ident = current_ident();
    {
        let mut contexts = EXECUTION_CONTEXTS.lock();
        contexts.retain(|thread_ident, _| *thread_ident == ident);
        if let Some(&ec) = contexts.get(&ident) {
            for &wref in unsafe { &(*(ec as *const crate::PyExecutionContext)).thread_local_refs } {
                let local = unsafe {
                    pyre_object::weakref::w_weakref_deref(
                        wref as *const pyre_object::weakref::Weakref,
                    )
                };
                if let Some(local) = W_Local::from_obj(local) {
                    local.after_fork_reinit();
                }
            }
        }
    }
    let handles = std::mem::take(&mut *ACTIVE_HANDLES.lock());
    for handle in handles {
        if let Some(handle_obj) = W_ThreadHandle::from_obj(handle as PyObjectRef) {
            let mut state = handle_obj.state.lock();
            if state.started && state.ident != ident {
                state.done = true;
                handle_obj.done.notify_all();
            } else if state.started {
                ACTIVE_HANDLES.lock().push(handle);
            }
        }
    }
    SHUTDOWN_HANDLES.lock().clear();
    THREAD_COUNT.store(0, Ordering::SeqCst);
    pyre_object::listobject::list_locks_after_fork_child();
    crate::objspace::std::mapdict::after_fork_child();
    pyre_object::dictmultiobject::module_dict_locks_after_fork_child();
    majit_gc::shadow_stack::after_fork_child();
    majit_gc::gc_sync::after_fork_child();
}

fn parse_acquire_args(
    blocking: i64,
    w_timeout: Option<PyObjectRef>,
) -> Result<Option<Duration>, crate::PyError> {
    // os_lock.py `@unwrap_spec(blocking=int, timeout=float)`: unlike the
    // macro's concrete `f64` receiver, `space.float_w` performs Python's
    // numeric coercion first, so integer timeout arguments retain their
    // value instead of being read through a float object's layout.
    let blocking = blocking != 0;
    let timeout = match w_timeout {
        Some(w_timeout) => crate::baseobjspace::float_w(w_timeout)?,
        None => -1.0,
    };
    if !blocking && timeout != -1.0 {
        return Err(crate::PyError::value_error(
            "can't specify a timeout for a non-blocking call",
        ));
    }
    if timeout < 0.0 && timeout != -1.0 {
        return Err(crate::PyError::value_error(
            "timeout value must be strictly positive",
        ));
    }
    if timeout > TIMEOUT_MAX {
        return Err(crate::PyError::overflow_error("timeout value is too large"));
    }
    Ok(if blocking && timeout >= 0.0 {
        Some(Duration::from_secs_f64(timeout))
    } else {
        None
    })
}

mod lock_class {
    use super::*;

    #[crate::pyre_class("_thread.lock")]
    #[derive(Default)]
    pub struct W_Lock {
        locked: Mutex<bool>,
        ready: Condvar,
    }

    #[crate::pyre_methods(doc = "A lock object is a synchronization primitive.", weakrefable)]
    impl W_Lock {
        #[staticmethod]
        fn __new__(cls: PyObjectRef, args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
            if args.len() > 1 {
                return Err(crate::PyError::type_error(
                    "_thread.lock() takes no arguments",
                ));
            }
            crate::typedef::check_user_subclass(type_object(), cls)?;
            let obj = Self::allocate_stable(Self::default());
            unsafe { (*obj).w_class = cls };
            Ok(obj)
        }

        fn acquire(
            &self,
            #[default(1)] blocking: i64,
            timeout: Option<PyObjectRef>,
        ) -> Result<bool, crate::PyError> {
            let timeout = parse_acquire_args(blocking, timeout)?;
            let blocking = blocking != 0;
            // A potentially blocking native lock wait leaves the collector's
            // RUNNING census.  This does not serialize Python execution.
            let _blocked = before_external_block();
            let mut locked = self.locked.lock();
            if !*locked {
                *locked = true;
                return Ok(true);
            }
            if !blocking {
                return Ok(false);
            }
            match timeout {
                None => {
                    while *locked {
                        self.ready.wait(&mut locked);
                    }
                    *locked = true;
                    Ok(true)
                }
                Some(duration) => {
                    let deadline = Instant::now() + duration;
                    while *locked {
                        let now = Instant::now();
                        if now >= deadline {
                            return Ok(false);
                        }
                        if self.ready.wait_for(&mut locked, deadline - now).timed_out() && *locked {
                            return Ok(false);
                        }
                    }
                    *locked = true;
                    Ok(true)
                }
            }
        }

        fn release(&self) -> Result<(), crate::PyError> {
            let mut locked = self.locked.lock();
            if !*locked {
                return Err(crate::PyError::runtime_error("release unlocked lock"));
            }
            *locked = false;
            self.ready.notify_one();
            Ok(())
        }

        fn locked(&self) -> bool {
            *self.locked.lock()
        }

        fn __enter__(&self) -> Result<PyObjectRef, crate::PyError> {
            self.acquire(1, None)?;
            Ok(self as *const Self as PyObjectRef)
        }

        fn __exit__(&self, _args: &[PyObjectRef]) -> Result<bool, crate::PyError> {
            self.release()?;
            Ok(false)
        }

        fn __repr__(&self) -> String {
            let state = if self.locked() { "locked" } else { "unlocked" };
            format!("<{state} _thread.lock object at {:p}>", self)
        }

        fn _at_fork_reinit(&self) {
            // The old native mutex/condvar may have been owned by a thread which
            // vanished at fork.  PyPy's rthread lock reinit replaces the native
            // lock without trying to acquire or destroy the inherited one.
            let this = self as *const Self as *mut Self;
            unsafe {
                std::ptr::write(&mut (*this).locked, Mutex::new(false));
                std::ptr::write(&mut (*this).ready, Condvar::new());
            }
        }
    }
}
use lock_class::W_Lock;

mod rlock_class {
    use super::*;

    #[derive(Default)]
    struct RLockState {
        count: i64,
        owner: i64,
    }

    #[crate::pyre_class("_thread.RLock")]
    #[derive(Default)]
    pub struct W_RLock {
        state: Mutex<RLockState>,
        ready: Condvar,
    }

    #[crate::pyre_methods(weakrefable)]
    impl W_RLock {
        #[staticmethod]
        fn __new__(cls: PyObjectRef, args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
            if args.len() > 1 && cls == type_object() {
                // CPython 3.14 keeps accepting arguments to the exact native
                // RLock for compatibility, but deprecates them.  PyPy's
                // W_RLock allocator likewise ignores construction arguments;
                // subclasses remain free to consume them in their own __init__.
                crate::warn::warn_deprecation(
                    "Passing arguments to _thread.RLock() is deprecated",
                )?;
            }
            crate::typedef::check_user_subclass(type_object(), cls)?;
            let obj = Self::allocate_stable(Self::default());
            unsafe { (*obj).w_class = cls };
            Ok(obj)
        }

        fn acquire(
            &self,
            #[default(1)] blocking: i64,
            timeout: Option<PyObjectRef>,
        ) -> Result<bool, crate::PyError> {
            let timeout = parse_acquire_args(blocking, timeout)?;
            let blocking = blocking != 0;
            let ident = current_ident();
            let _blocked = before_external_block();
            let mut state = self.state.lock();
            if state.count > 0 && state.owner == ident {
                state.count = state.count.checked_add(1).ok_or_else(|| {
                    crate::PyError::overflow_error("internal lock count overflowed")
                })?;
                return Ok(true);
            }
            if state.count == 0 {
                state.owner = ident;
                state.count = 1;
                return Ok(true);
            }
            if !blocking {
                return Ok(false);
            }
            match timeout {
                None => {
                    while state.count != 0 {
                        self.ready.wait(&mut state);
                    }
                }
                Some(duration) => {
                    let deadline = Instant::now() + duration;
                    while state.count != 0 {
                        let now = Instant::now();
                        if now >= deadline {
                            return Ok(false);
                        }
                        if self.ready.wait_for(&mut state, deadline - now).timed_out()
                            && state.count != 0
                        {
                            return Ok(false);
                        }
                    }
                }
            }
            state.owner = ident;
            state.count = 1;
            Ok(true)
        }

        fn release(&self) -> Result<(), crate::PyError> {
            let ident = current_ident();
            let mut state = self.state.lock();
            if state.count == 0 || state.owner != ident {
                return Err(crate::PyError::runtime_error(
                    "cannot release un-acquired lock",
                ));
            }
            state.count -= 1;
            if state.count == 0 {
                state.owner = 0;
                self.ready.notify_one();
            }
            Ok(())
        }

        fn locked(&self) -> bool {
            self.state.lock().count != 0
        }

        fn _is_owned(&self) -> bool {
            let state = self.state.lock();
            state.count > 0 && state.owner == current_ident()
        }

        fn _recursion_count(&self) -> i64 {
            let state = self.state.lock();
            if state.owner == current_ident() {
                state.count
            } else {
                0
            }
        }

        fn _release_save(&self) -> Result<PyObjectRef, crate::PyError> {
            let mut state = self.state.lock();
            if state.count == 0 {
                return Err(crate::PyError::runtime_error(
                    "cannot release un-acquired lock",
                ));
            }
            let saved = w_tuple_new(vec![w_int_new(state.count), w_int_new(state.owner)]);
            state.count = 0;
            state.owner = 0;
            self.ready.notify_one();
            Ok(saved)
        }

        fn _acquire_restore(&self, saved: PyObjectRef) -> Result<(), crate::PyError> {
            let items = unsafe {
                if !is_tuple(saved) {
                    return Err(crate::PyError::type_error("saved state must be a tuple"));
                }
                w_tuple_items_copy_as_vec(saved)
            };
            if items.len() != 2 || unsafe { !is_int(items[0]) || !is_int(items[1]) } {
                return Err(crate::PyError::type_error("invalid saved state"));
            }
            let count = unsafe { w_int_get_value(items[0]) };
            let owner = unsafe { w_int_get_value(items[1]) };
            let _blocked = before_external_block();
            let mut state = self.state.lock();
            if state.count != 0 {
                while state.count != 0 {
                    self.ready.wait(&mut state);
                }
            }
            state.count = count;
            state.owner = owner;
            Ok(())
        }

        fn __enter__(&self) -> Result<PyObjectRef, crate::PyError> {
            self.acquire(1, None)?;
            Ok(self as *const Self as PyObjectRef)
        }

        fn __exit__(&self, _args: &[PyObjectRef]) -> Result<bool, crate::PyError> {
            self.release()?;
            Ok(false)
        }

        fn __repr__(&self) -> String {
            let state = self.state.lock();
            let locked = if state.count == 0 {
                "unlocked"
            } else {
                "locked"
            };
            format!(
                "<{locked} _thread.RLock object owner={} count={} at {:p}>",
                state.owner, state.count, self
            )
        }

        fn _at_fork_reinit(&self) {
            let this = self as *const Self as *mut Self;
            unsafe {
                std::ptr::write(&mut (*this).state, Mutex::new(RLockState::default()));
                std::ptr::write(&mut (*this).ready, Condvar::new());
            }
        }
    }
}
use rlock_class::W_RLock;

mod handle_class {
    use super::*;

    #[derive(Default)]
    pub(super) struct HandleState {
        pub(super) started: bool,
        pub(super) done: bool,
        pub(super) ident: i64,
        pub(super) daemon: bool,
    }

    #[crate::pyre_class("_thread._ThreadHandle")]
    #[derive(Default)]
    pub struct W_ThreadHandle {
        pub(super) state: Mutex<HandleState>,
        pub(super) done: Condvar,
    }

    #[crate::pyre_methods]
    impl W_ThreadHandle {
        #[staticmethod]
        fn __new__(cls: PyObjectRef, args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
            if args.len() > 1 {
                return Err(crate::PyError::type_error(
                    "_ThreadHandle() takes no arguments",
                ));
            }
            crate::typedef::check_user_subclass(type_object(), cls)?;
            let obj = Self::allocate_stable(Self::default());
            unsafe { (*obj).w_class = cls };
            Ok(obj)
        }

        #[getter]
        fn ident(&self) -> i64 {
            self.state.lock().ident
        }

        fn is_done(&self) -> bool {
            self.state.lock().done
        }

        pub(super) fn join(
            &self,
            #[default(pyre_object::w_none())] timeout: PyObjectRef,
        ) -> Result<(), crate::PyError> {
            let duration = unsafe {
                if is_none(timeout) {
                    None
                } else if is_float(timeout) {
                    Some(Duration::from_secs_f64(
                        floatobject::w_float_get_value(timeout).max(0.0),
                    ))
                } else if is_int(timeout) {
                    Some(Duration::from_secs(w_int_get_value(timeout).max(0) as u64))
                } else {
                    return Err(crate::PyError::type_error(
                        "timeout must be a number or None",
                    ));
                }
            };
            let mut state = self.state.lock();
            if !state.started {
                return Err(crate::PyError::runtime_error("thread not started"));
            }
            if state.ident == current_ident() && !state.done {
                return Err(crate::PyError::runtime_error("Cannot join current thread"));
            }
            if state.done {
                return Ok(());
            }
            if state.daemon && is_finalizing() {
                let cls = crate::builtins::lookup_exc_class("PythonFinalizationError")
                    .expect("PythonFinalizationError must be installed");
                let exc = crate::builtins::exc_exception_new(&[cls])?;
                return Err(unsafe { crate::PyError::from_exc_object(exc) });
            }
            let _blocked = before_external_block();
            match duration {
                None => {
                    while !state.done {
                        self.done.wait(&mut state);
                    }
                }
                Some(timeout) => {
                    let deadline = Instant::now() + timeout;
                    while !state.done {
                        let now = Instant::now();
                        if now >= deadline {
                            break;
                        }
                        if self.done.wait_for(&mut state, deadline - now).timed_out() {
                            break;
                        }
                    }
                }
            }
            Ok(())
        }

        fn _set_done(&self) -> Result<(), crate::PyError> {
            let mut state = self.state.lock();
            if !state.started {
                return Err(crate::PyError::runtime_error("thread not started"));
            }
            state.done = true;
            self.done.notify_all();
            Ok(())
        }
    }
}
use handle_class::W_ThreadHandle;

impl W_ThreadHandle {
    fn start(&self, ident: i64) -> Result<(), crate::PyError> {
        let mut state = self.state.lock();
        if state.started {
            return Err(crate::PyError::runtime_error("thread already started"));
        }
        state.started = true;
        state.ident = ident;
        ACTIVE_HANDLES.lock().push(self as *const Self as usize);
        Ok(())
    }

    fn finish(&self) {
        let mut state = self.state.lock();
        state.done = true;
        self.done.notify_all();
        let address = self as *const Self as usize;
        ACTIVE_HANDLES.lock().retain(|handle| *handle != address);
    }
}

mod local_class {
    use super::*;

    /// `pypy/module/thread/os_local.py Local`.
    ///
    /// `dicts` is deliberately a Python dict, matching upstream's
    /// `self.dicts = {}` and keeping every per-ExecutionContext dictionary on the
    /// object's ordinary GC graph.  The integer key is pyre's stable identity for
    /// the current OS-thread ExecutionContext.
    #[crate::pyre_class("_thread._local")]
    pub struct W_Local {
        dicts: PyObjectRef,
        last_dict: PyObjectRef,
        last_ident: i64,
        state_lock: Mutex<()>,
    }

    impl W_Local {
        pub(super) fn current_dict(&self) -> PyObjectRef {
            let _guard = self.state_lock.lock();
            let this = self as *const Self as *mut Self;
            let ident = current_ident();
            if self.last_ident == ident && !self.last_dict.is_null() {
                return self.last_dict;
            }
            let w_dict =
                unsafe { pyre_object::w_dict_getitem(self.dicts, ident) }.unwrap_or_else(|| {
                    let w_dict = pyre_object::w_dict_new();
                    unsafe { pyre_object::w_dict_setitem(self.dicts, ident, w_dict) };
                    register_local_in_current_ec(self as *const Self as PyObjectRef);
                    w_dict
                });
            unsafe {
                (*this).last_ident = ident;
                (*this).last_dict = w_dict;
            }
            pyre_object::gc_hook::try_gc_write_barrier(this as *mut u8);
            w_dict
        }

        pub(super) fn thread_is_stopping(&self, ident: i64) {
            let _guard = self.state_lock.lock();
            let this = self as *const Self as *mut Self;
            let key = w_int_new(ident);
            unsafe {
                pyre_object::w_dict_delitem(self.dicts, key);
                if (*this).last_ident == ident {
                    (*this).last_ident = 0;
                    (*this).last_dict = PY_NULL;
                }
            }
        }

        pub(super) fn after_fork_reinit(&self) {
            let this = self as *const Self as *mut Self;
            unsafe {
                std::ptr::write(&mut (*this).state_lock, parking_lot::const_mutex(()));
            }
        }
    }

    #[crate::pyre_methods(doc = "Thread-local data", weakrefable)]
    impl W_Local {
        #[staticmethod]
        fn __new__(cls: PyObjectRef, args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
            crate::typedef::check_user_subclass(type_object(), cls)?;
            // os_local.py installs the first dictionary before app-level
            // __init__ is entered, preventing recursive initialization.
            let dicts = pyre_object::w_dict_new();
            let w_dict = pyre_object::w_dict_new();
            let ident = current_ident();
            unsafe { pyre_object::w_dict_setitem(dicts, ident, w_dict) };
            let obj = Self::allocate_stable(Self {
                ob: PyObject::default(),
                dicts,
                last_dict: w_dict,
                last_ident: ident,
                state_lock: parking_lot::const_mutex(()),
            });
            unsafe { (*obj).w_class = cls };
            register_local_in_current_ec(obj);

            // The base `_local` has no app-level initializer accepting arguments.
            // Subclass initialization is dispatched by the ordinary type call
            // after this allocator returns, exactly as for PyPy's TypeDef.
            if args.len() > 1 && cls == type_object() {
                return Err(crate::PyError::type_error(
                    "Initialization arguments are not supported",
                ));
            }
            Ok(obj)
        }

        #[getter]
        fn __dict__(&self) -> PyObjectRef {
            self.current_dict()
        }
    }
}
pub use local_class::W_Local;

fn local_type() -> PyObjectRef {
    local_class::type_object()
}

/// W_Root.getdict dispatch for `os_local.Local.getdict`.
pub(crate) fn local_getdict(obj: PyObjectRef) -> Option<PyObjectRef> {
    let local = W_Local::from_obj(obj)?;
    Some(local.current_dict())
}

/// `os_local.py:Local._register_in_ec`.
fn register_local_in_current_ec(local: PyObjectRef) {
    let ec = crate::call::getexecutioncontext() as *mut crate::PyExecutionContext;
    if ec.is_null() {
        return;
    }
    let root = pyre_object::gc_roots::push_roots();
    pyre_object::gc_roots::pin_root(local);
    let wref = unsafe { pyre_object::weakref::w_weakref_new(local) as PyObjectRef };
    unsafe { (*ec).thread_local_refs.push(wref) };
    drop(root);
}

/// `os_local.py:thread_is_stopping`.
fn thread_is_stopping(ec: &mut crate::PyExecutionContext) {
    let ident = current_ident();
    for wref in std::mem::take(&mut ec.thread_local_refs) {
        let local = unsafe {
            pyre_object::weakref::w_weakref_deref(wref as *const pyre_object::weakref::Weakref)
        };
        if let Some(local) = W_Local::from_obj(local) {
            local.thread_is_stopping(ident);
        }
    }
}

pub(crate) fn current_ident() -> i64 {
    #[cfg(all(
        feature = "host_env",
        not(target_arch = "wasm32"),
        not(feature = "sandbox")
    ))]
    {
        return rustpython_host_env::thread::current_thread_id() as i64;
    }
    #[allow(unreachable_code)]
    1
}

#[crate::pyre_function]
fn get_ident() -> i64 {
    current_ident()
}

#[crate::pyre_function]
fn get_native_id() -> i64 {
    #[cfg(all(not(feature = "sandbox"), target_os = "macos"))]
    {
        let mut tid: u64 = 0;
        if unsafe { libc::pthread_threadid_np(0, &mut tid) } == 0 {
            return tid as i64;
        }
    }
    #[cfg(all(
        not(feature = "sandbox"),
        any(target_os = "linux", target_os = "android")
    ))]
    {
        return unsafe { libc::syscall(libc::SYS_gettid) } as i64;
    }
    current_ident()
}

fn new_lock(_: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    Ok(W_Lock::allocate_stable(W_Lock::default()))
}

fn new_handle(_: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    Ok(W_ThreadHandle::allocate_stable(W_ThreadHandle::default()))
}

fn make_thread_handle(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let ident = unsafe { w_int_get_value(args[0]) };
    let obj = W_ThreadHandle::allocate_stable(W_ThreadHandle::default());
    W_ThreadHandle::from_obj(obj).unwrap().start(ident)?;
    Ok(obj)
}

fn call_thread_target(
    callable: PyObjectRef,
    positional: &[PyObjectRef],
    kwargs: Option<PyObjectRef>,
    ec: *const crate::PyExecutionContext,
) -> Result<PyObjectRef, crate::PyError> {
    if kwargs.is_none_or(|d| unsafe { w_dict_str_entries(d) }.is_empty()) {
        return crate::call::call_function_impl_result(callable, positional);
    }
    let kwargs = kwargs.unwrap();
    let entries = unsafe { w_dict_str_entries(kwargs) };
    let (target, args) = unsafe {
        if is_method(callable) {
            let func = w_method_get_func(callable);
            let receiver = w_method_get_self(callable);
            let mut args = Vec::with_capacity(positional.len() + 1);
            args.push(receiver);
            args.extend_from_slice(positional);
            (func, args)
        } else {
            (callable, positional.to_vec())
        }
    };
    if unsafe { crate::is_function(target) } {
        let mut mixed = args;
        let mut names = Vec::with_capacity(entries.len());
        for (name, value) in entries {
            mixed.push(value);
            names.push(w_str_new(&name));
        }
        let kwarg_names = w_tuple_new(names);
        let resolved = crate::call::resolve_kwargs(target, &mixed, kwarg_names)?;
        crate::call::call_user_function_plain_with_ctx(ec, target, &resolved)
    } else {
        Err(crate::PyError::type_error(
            "keyword arguments for this thread target are not supported",
        ))
    }
}

fn spawn_thread(
    callable: PyObjectRef,
    positional: Vec<PyObjectRef>,
    kwargs: Option<PyObjectRef>,
    handle: Option<PyObjectRef>,
) -> Result<i64, crate::PyError> {
    let parent_ec = crate::call::getexecutioncontext();
    if parent_ec.is_null() {
        return Err(crate::PyError::runtime_error("no execution context"));
    }

    let roots = pyre_object::gc_roots::push_roots();
    let base = pyre_object::gc_roots::shadow_stack_len();
    pyre_object::gc_roots::pin_root(callable);
    for &arg in &positional {
        pyre_object::gc_roots::pin_root(arg);
    }
    if let Some(d) = kwargs {
        pyre_object::gc_roots::pin_root(d);
    }
    if let Some(h) = handle {
        pyre_object::gc_roots::pin_root(h);
    }
    let nargs = positional.len();
    let has_kwargs = kwargs.is_some();
    let has_handle = handle.is_some();
    let callable_addr = callable as usize;
    let args_addr: Vec<usize> = positional.iter().map(|&p| p as usize).collect();
    let kwargs_addr = kwargs.unwrap_or(PY_NULL) as usize;
    let handle_addr = handle.unwrap_or(PY_NULL) as usize;
    let parent_ec_addr = parent_ec as usize;
    let (started_tx, started_rx) = std::sync::mpsc::sync_channel(0);

    let mut builder = std::thread::Builder::new();
    let stack_size = STACK_SIZE.load(Ordering::Relaxed);
    if stack_size != 0 {
        builder = builder.stack_size(stack_size);
    }
    builder
        .spawn(move || {
            crate::call::enter_runtime_thread();
            // The parent holds these in its shadow stack until `started_tx`.
            // Copy them into this mutator's own shadow stack before any
            // interpreter allocation can trigger a moving collection.
            let worker_roots = pyre_object::gc_roots::push_roots();
            let worker_base = pyre_object::gc_roots::shadow_stack_len();
            pyre_object::gc_roots::pin_root(callable_addr as PyObjectRef);
            for addr in &args_addr {
                pyre_object::gc_roots::pin_root(*addr as PyObjectRef);
            }
            if has_kwargs {
                pyre_object::gc_roots::pin_root(kwargs_addr as PyObjectRef);
            }
            if has_handle {
                pyre_object::gc_roots::pin_root(handle_addr as PyObjectRef);
            }

            let mut ec = Box::new(unsafe {
                (*(parent_ec_addr as *const crate::PyExecutionContext)).clone_for_thread()
            });
            let ec_ptr = &*ec as *const crate::PyExecutionContext;
            crate::call::set_last_exec_ctx(ec_ptr);
            // `install_user_del_action` can allocate.  Publish the fresh EC
            // first, matching OSThreadLocals.enter_thread() installing the
            // ExecutionContext before thread bootstrap invokes Python code.
            ec.install_user_del_action();
            let ident = current_ident();
            if has_handle {
                let h = W_ThreadHandle::from_obj(handle_addr as PyObjectRef).unwrap();
                if let Err(e) = h.start(ident) {
                    let _ = started_tx.send(Err(e.message));
                    thread_is_stopping(&mut ec);
                    crate::call::set_last_exec_ctx(std::ptr::null());
                    drop(worker_roots);
                    drop(ec);
                    return;
                }
            }
            THREAD_COUNT.fetch_add(1, Ordering::SeqCst);
            let _ = started_tx.send(Ok(ident));

            let callable = pyre_object::gc_roots::shadow_stack_get(worker_base);
            let args: Vec<PyObjectRef> = (0..nargs)
                .map(|i| pyre_object::gc_roots::shadow_stack_get(worker_base + 1 + i))
                .collect();
            let mut next = worker_base + 1 + nargs;
            let kwargs = if has_kwargs {
                let d = pyre_object::gc_roots::shadow_stack_get(next);
                next += 1;
                Some(d)
            } else {
                None
            };
            let handle = if has_handle {
                Some(pyre_object::gc_roots::shadow_stack_get(next))
            } else {
                None
            };
            // `JitDriver` currently owns a per-TLS background compiler.
            // Creating a compiler thread for every short-lived Python thread
            // makes teardown wait on unrelated compiler polling.  Execute
            // worker frames through the same interpreter source until the
            // driver owner is made interpreter-global.
            let _plain_worker = crate::call::force_plain_eval();
            if let Err(mut error) = call_thread_target(callable, &args, kwargs, ec_ptr) {
                let callable_repr =
                    unsafe { crate::py_repr(callable).unwrap_or_else(|_| "<unknown>".to_string()) };
                error.write_unraisable(
                    w_none(),
                    &format!("Exception ignored in thread started by {callable_repr}"),
                    w_none(),
                );
            }
            thread_is_stopping(&mut ec);
            crate::call::set_last_exec_ctx(std::ptr::null());
            drop(worker_roots);
            drop(ec);
            // `_thread._count()` is the completion signal used by PyPy's
            // thread helpers for non-joinable `start_new_thread` workers.
            // Publish it only after EC unregistration and shadow-root teardown;
            // otherwise the main thread can begin finalization GC while this
            // mutator is still dismantling its runtime state.
            THREAD_COUNT.fetch_sub(1, Ordering::SeqCst);
            // Match the upstream thread-state teardown order: a joinable
            // handle becomes done only after the worker's interpreter roots
            // and ExecutionContext have gone away.  In particular, join()
            // must not return while the worker still roots Thread._bootstrap,
            // which would keep Thread._target argument cycles alive.
            if let Some(handle) = handle {
                W_ThreadHandle::from_obj(handle).unwrap().finish();
            }
        })
        .map_err(|_| crate::PyError::runtime_error("can't start new thread"))?;

    let result = {
        let _blocked = before_external_block();
        started_rx
            .recv()
            .map_err(|_| crate::PyError::runtime_error("can't start new thread"))?
    };
    drop(roots);
    result.map_err(crate::PyError::runtime_error)
}

fn start_new_thread(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if is_finalizing() {
        return Err(crate::PyError::runtime_error(
            "can't create new thread at interpreter shutdown",
        ));
    }
    let (pos, kwargs_marker) = crate::builtins::split_builtin_kwargs(args);
    if pos.len() < 2 || pos.len() > 3 {
        return Err(crate::PyError::type_error(
            "start_new_thread expected 2 or 3 arguments",
        ));
    }
    let callable = pos[0];
    if !crate::baseobjspace::callable_w(callable) {
        return Err(crate::PyError::type_error("first arg must be callable"));
    }
    if unsafe { !is_tuple(pos[1]) } {
        return Err(crate::PyError::type_error("2nd arg must be a tuple"));
    }
    let positional = unsafe { w_tuple_items_copy_as_vec(pos[1]) };
    let kwargs = pos
        .get(2)
        .copied()
        .or_else(|| crate::builtins::kwarg_get(kwargs_marker, "kwargs"));
    if kwargs.is_some_and(|d| unsafe { !is_dict(d) }) {
        return Err(crate::PyError::type_error(
            "optional 3rd arg must be a dictionary",
        ));
    }
    Ok(w_int_new(spawn_thread(callable, positional, kwargs, None)?))
}

fn start_joinable_thread(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if is_finalizing() {
        return Err(crate::PyError::runtime_error(
            "can't create new thread at interpreter shutdown",
        ));
    }
    let (pos, kwargs) = crate::builtins::split_builtin_kwargs(args);
    let callable = pos
        .first()
        .copied()
        .ok_or_else(|| crate::PyError::type_error("missing function argument"))?;
    if !crate::baseobjspace::callable_w(callable) {
        return Err(crate::PyError::type_error("function must be callable"));
    }
    let requested = crate::builtins::kwarg_get(kwargs, "handle");
    let daemon = match crate::builtins::kwarg_get(kwargs, "daemon") {
        Some(value) => crate::baseobjspace::is_true(value)?,
        None => false,
    };
    let handle = match requested {
        Some(obj) if unsafe { !is_none(obj) } => {
            if W_ThreadHandle::from_obj(obj).is_none() {
                return Err(crate::PyError::type_error("handle must be a _ThreadHandle"));
            }
            obj
        }
        _ => W_ThreadHandle::allocate_stable(W_ThreadHandle::default()),
    };
    if let Some(handle_obj) = W_ThreadHandle::from_obj(handle) {
        handle_obj.state.lock().daemon = daemon;
    }
    spawn_thread(callable, Vec::new(), None, Some(handle))?;
    if !daemon {
        SHUTDOWN_HANDLES.lock().push(handle as usize);
    }
    Ok(handle)
}

fn shutdown_threads(_: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    // CPython `thread_shutdown` and PyPy's bootstrapper shutdown both include
    // non-daemon threads started by a thread which is itself being joined.
    // Drain repeatedly because joining one snapshot may publish another.
    loop {
        let handles = std::mem::take(&mut *SHUTDOWN_HANDLES.lock());
        if handles.is_empty() {
            break;
        }
        for handle in handles {
            if let Some(handle) = W_ThreadHandle::from_obj(handle as PyObjectRef) {
                handle.join(w_none())?;
            }
        }
    }
    Ok(w_none())
}

fn stack_size(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.len() > 1 {
        return Err(crate::PyError::type_error(
            "stack_size() takes at most 1 argument",
        ));
    }
    let old = STACK_SIZE.load(Ordering::Relaxed);
    if let Some(&arg) = args.first() {
        let size = unsafe { w_int_get_value(arg) };
        if size < 0 || (size != 0 && size < 32_768) {
            return Err(crate::PyError::value_error(format!(
                "size not valid: {size} bytes"
            )));
        }
        STACK_SIZE.store(size as usize, Ordering::Relaxed);
    }
    Ok(w_int_new(old as i64))
}

fn interrupt_main(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.len() > 1 {
        return Err(crate::PyError::type_error(
            "interrupt_main() takes at most 1 argument",
        ));
    }
    #[cfg(not(target_arch = "wasm32"))]
    let signum = args
        .first()
        .map(|&arg| unsafe { w_int_get_value(arg) as i32 })
        .unwrap_or(libc::SIGINT);
    #[cfg(target_arch = "wasm32")]
    let signum = args
        .first()
        .map(|&arg| unsafe { w_int_get_value(arg) as i32 })
        .unwrap_or(2);
    #[cfg(not(target_arch = "wasm32"))]
    if !(1..crate::module::signal::signalstate::NSIG).contains(&signum) {
        return Err(crate::PyError::value_error("signal number out of range"));
    }
    #[cfg(target_arch = "wasm32")]
    if !(1..65).contains(&signum) {
        return Err(crate::PyError::value_error("signal number out of range"));
    }
    let ec = crate::call::getexecutioncontext();
    #[cfg(target_arch = "wasm32")]
    let _ = ec;
    #[cfg(target_arch = "wasm32")]
    {
        let cls = crate::builtins::lookup_exc_class("KeyboardInterrupt")
            .expect("KeyboardInterrupt must be installed");
        let exc = crate::builtins::exc_exception_new(&[cls])?;
        return Err(unsafe { crate::PyError::from_exc_object(exc) });
    }
    #[cfg(not(target_arch = "wasm32"))]
    if ec.is_null() || unsafe { (*ec).check_signal_action.is_none() } {
        let cls = crate::builtins::lookup_exc_class("KeyboardInterrupt")
            .expect("KeyboardInterrupt must be installed");
        let exc = crate::builtins::exc_exception_new(&[cls])?;
        return Err(unsafe { crate::PyError::from_exc_object(exc) });
    }
    #[cfg(not(target_arch = "wasm32"))]
    crate::module::signal::signalstate::signal_pushback(signum);
    #[cfg(not(target_arch = "wasm32"))]
    Ok(w_none())
}

crate::py_module! {
    "_thread",
    interpleveldefs: {
        "LockType"      => {
            let ty = lock_class::type_object();
            unsafe { pyre_object::w_type_set_acceptable_as_base_class(ty, false) };
            ty
        },
        "RLock"         => rlock_class::type_object(),
        "_ThreadHandle" => handle_class::type_object(),
        "_local"        => local_type(),
        "TIMEOUT_MAX"   => w_float_new(TIMEOUT_MAX),
        "error"         => crate::builtins::lookup_exc_class("RuntimeError")
                               .unwrap_or_else(crate::typedef::w_object),
    },
    functions: {
        "allocate_lock"          / 0 = new_lock,
        "allocate"               / 0 = new_lock,
        "_set_sentinel"          / 0 = new_lock,
        "_make_thread_handle"    / 1 = make_thread_handle,
        "get_ident"              / 0 = get_ident,
        "get_native_id"          / 0 = get_native_id,
        "_count"                 / 0 = |_| Ok(w_int_new(THREAD_COUNT.load(Ordering::SeqCst))),
        "_is_main_interpreter"   / 0 = |_| Ok(w_bool_from(true)),
        "daemon_threads_allowed" / 0 = |_| Ok(w_bool_from(true)),
        "_shutdown"              / 0 = shutdown_threads,
        "stack_size"             / * = stack_size,
        "interrupt_main"         / * = interrupt_main,
        "set_name"               / 1 = |_| Ok(w_none()),
        "_excepthook"            / 1 = |_| Ok(w_none()),
        "_get_main_thread_ident" / 0 = |_| Ok(w_int_new(current_ident())),
        "start_joinable_thread"  / * = start_joinable_thread,
        "start_new_thread"       / * = start_new_thread,
        "start_new"              / * = start_new_thread,
    },
}
