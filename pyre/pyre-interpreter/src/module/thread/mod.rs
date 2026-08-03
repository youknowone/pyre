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

/// `_thread.TIMEOUT_MAX` — the whole-second bound of the nanosecond timestamp
/// an acquire timeout is converted to.  PyPy exposes the microsecond bound
/// instead (`moduledef.py:27` `float(os_lock.TIMEOUT_MAX // 1000000)`), which
/// is a thousand times larger and is its 3.11-era surface.
const TIMEOUT_MAX: f64 = (i64::MAX / 1_000_000_000) as f64;
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

pub mod gil;

/// `rffi.aroundstate.before()`: drop the GIL and leave the collector's RUNNING
/// census for the duration of a blocking host call.
pub fn before_external_block() -> majit_gc::gc_sync::BlockingGuard {
    majit_gc::gc_sync::before_external_block()
}

thread_local! {
    /// Set once this thread owns a mutator registration, so the entry below
    /// stays a single thread-local read on every later entry.
    static RUNTIME_THREAD_ENTERED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };

    /// Armed after `shadow_stack::register_mutator` has captured this thread's
    /// root slots, so the destructor removes the registry entry before those
    /// slots are destroyed and only then gives the GIL back.
    static RUNTIME_THREAD: RuntimeThread = const { RuntimeThread };
}

struct RuntimeThread;

impl Drop for RuntimeThread {
    fn drop(&mut self) {
        majit_gc::shadow_stack::unregister_mutator();
        majit_gc::gc_sync::unregister_thread();
    }
}

/// `rgil.py:186-193 acquire_maybe_in_new_thread`: a thread that has not run
/// pyre code before becomes a GC mutator and takes the GIL before it runs any.
///
/// Upstream reaches every RPython thread through `rpython_startup_code` or
/// rffi's callback path, both of which acquire before the first RPython
/// instruction (entrypoint.c:49,78). pyre is entered from Rust at several
/// points instead — the launcher, a spawned Python thread, and the unit tests
/// that drive the interpreter directly — so the acquire is idempotent per
/// thread and each entry point names it. The registration is given back by
/// `RUNTIME_THREAD`'s destructor when the thread exits, which is what lets a
/// second thread run pyre code afterwards.
#[inline]
pub fn ensure_runtime_thread() {
    if RUNTIME_THREAD_ENTERED.with(|entered| entered.get()) {
        return;
    }
    enter_runtime_thread();
}

#[cold]
fn enter_runtime_thread() {
    RUNTIME_THREAD_ENTERED.with(|entered| entered.set(true));
    majit_gc::gc_sync::register_thread();
    majit_gc::shadow_stack::register_mutator();
    RUNTIME_THREAD.with(|_| {});
}

/// Whether this thread already owns its mutator registration.
///
/// pyre-jit's GC bootstrap has per-thread work of its own to hang off the same
/// registration, so it asks rather than keeping a second flag.
pub fn runtime_thread_entered() -> bool {
    RUNTIME_THREAD_ENTERED.with(|entered| entered.get())
}

/// `rffi.py:193-211 call_external_function`: release the GIL, run the external
/// call, read `errno`, and only then take the GIL back.  The returned `i32` is
/// the saved `errno`, meaningful exactly when the call reports failure.
///
/// The read belongs inside the released window because `_errno_after` runs
/// ahead of `rgil.acquire()` (rffi.py:207-210).  Taking the GIL back can enter
/// the stealer loop, whose mutex and condvar waits overwrite `errno`, so a
/// caller reading it after the guard drops can see the wrong value.
pub(crate) fn call_external_function<R>(f: impl FnOnce() -> R) -> (R, i32) {
    let _blocked = before_external_block();
    let result = f();
    let errno = std::io::Error::last_os_error().raw_os_error().unwrap_or(0);
    (result, errno)
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
        majit_ir::eval_breaker_word::set_finalizing();
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
            forward(&mut ec.sys_exc_value);
            forward(&mut ec.current_gen_or_coroutine);
            forward(&mut ec.w_asyncgen_firstiter_fn);
            forward(&mut ec.w_asyncgen_finalizer_fn);
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
            // `builtins_module` / `builtin_dict_cache` / `thread_local_refs`.
            // `clone_for_thread` copies the builtins reference into the child
            // EC, so each copy is its own slot: forwarding only the parent's
            // leaves the child pointing at the pre-move address once a thread
            // is registered but has not yet armed its own root area.
            ec.walk_builtin_roots(visitor);
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

/// Fast bytecode-boundary gate for CPython's all-thread tracing extensions.
///
/// PyPy's ordinary `bytecode_trace` common path contains only its trace check
/// and action ticker.  Pyre additionally has to notice process-wide
/// `_settraceallthreads` / `_setprofileallthreads` generations, but when
/// neither changed it need not enter the updater (and its mutex-bearing slow
/// arms) at every opcode.
#[inline(always)]
pub(crate) fn all_thread_hooks_current(ec: &crate::PyExecutionContext) -> bool {
    ec.trace_all_generation == TRACE_ALL_GENERATION.load(Ordering::Acquire)
        && ec.profile_all_generation == PROFILE_ALL_GENERATION.load(Ordering::Acquire)
}

/// `dont_look_inside`: the per-thread trace/profile safepoint reads the
/// process-wide generation counters and hook mutexes and applies any change
/// to this thread's `ExecutionContext`.  The JIT does not trace this cold
/// hook-application path; a caller (`ExecutionContext::bytecode_trace`)
/// residualizes the call.
#[majit_macros::dont_look_inside]
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
                // The frame becomes a user-visible value; materialize the
                // virtualizable fields the JIT may still be holding.
                crate::executioncontext::force_frame(frame);
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
    pyre_object::setobject::set_locks_after_fork_child();
    pyre_object::interp_itertools::count_locks_after_fork_child();
    pyre_object::typeobject::subclasses_locks_after_fork_child();
    crate::objspace::std::mapdict::after_fork_child();
    pyre_object::dictmultiobject::module_dict_locks_after_fork_child();
    crate::module::_collections::deque_locks_after_fork_child();
    majit_gc::shadow_stack::after_fork_child();
    majit_gc::gc_sync::after_fork_child();
}

// os_lock.py:20 `RPY_LOCK_FAILURE, RPY_LOCK_ACQUIRED, RPY_LOCK_INTR`.
const RPY_LOCK_FAILURE: i64 = 0;
const RPY_LOCK_ACQUIRED: i64 = 1;
const RPY_LOCK_INTR: i64 = 2;

/// A `pthread_mutex_t` has no poison state — `thread_pthread.c` inspects only
/// the status code — so a panic taken while lock bookkeeping was held must not
/// turn every later acquire into an error.
fn lock_state<T>(mutex: &std::sync::Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// `os_lock.py:23-40 parse_acquire_args`.  The result is the `microseconds`
/// argument of the `RPyThreadAcquireLockTimed` ABI: negative blocks forever,
/// zero polls once.
///
/// The argument is rejected the way `lock_acquire_parse_args` does in 3.14,
/// which is stricter than `os_lock.py`: upstream converts the seconds straight
/// to microseconds, while 3.14 first builds a nanosecond timestamp, so NaN and
/// anything outside the nanosecond range are rejected before the microsecond
/// bound is ever reached — and that bound, being a thousand times wider than
/// the nanosecond one, is then unreachable and is not tested here.
fn parse_acquire_args(
    blocking: i64,
    w_timeout: Option<PyObjectRef>,
) -> Result<i64, crate::PyError> {
    // os_lock.py `@unwrap_spec(blocking=int, timeout=float)`: unlike the
    // macro's concrete `f64` receiver, `space.float_w` performs Python's
    // numeric coercion first, so integer timeout arguments retain their
    // value instead of being read through a float object's layout.
    let blocking = blocking != 0;
    let timeout = match w_timeout {
        Some(w_timeout) => crate::baseobjspace::float_w(w_timeout)?,
        None => -1.0,
    };
    // `_PyTime_FromSecondsObject(&timeout, timeout_obj, _PyTime_ROUND_TIMEOUT)`
    // runs before either check below, so a value it cannot represent is
    // reported even for a non-blocking call.
    if timeout.is_nan() {
        return Err(crate::PyError::value_error(
            "Invalid value NaN (not a number)",
        ));
    }
    // `rarithmetic.ovfcheck_float_to_longlong` bounds, the ones `time.sleep`
    // converts its own argument against (`interp_time.rs:122-126`).
    const NS_MIN: f64 = -9223372036854776832.0;
    const NS_MAX: f64 = 9223372036854775296.0;
    if !(NS_MIN..NS_MAX).contains(&(timeout * 1e9).ceil()) {
        return Err(crate::PyError::overflow_error(
            "timestamp out of range for platform time_t",
        ));
    }
    if !blocking && timeout != -1.0 {
        return Err(crate::PyError::value_error(
            "can't specify a timeout for a non-blocking call",
        ));
    }
    if timeout < 0.0 && timeout != -1.0 {
        return Err(crate::PyError::value_error(
            "timeout value must be a non-negative number",
        ));
    }
    if !blocking {
        Ok(0)
    } else if timeout == -1.0 {
        Ok(-1)
    } else {
        // `_PyTime_ROUND_TIMEOUT` rounds away from zero, both in
        // `lock_acquire_parse_args`' seconds→ns step and in
        // `_PyTime_AsMicroseconds`.  Truncating instead would collapse any
        // positive sub-microsecond timeout to 0, which `acquire_timed` reads
        // as a non-blocking poll rather than a timed wait.
        Ok((timeout * 1e6).ceil() as i64)
    }
}

/// `os_lock.py:49 space.getexecutioncontext().checksignals()`.  The signal
/// module is not built for wasm32 (`module/mod.rs:94`), where no handler can
/// be pending, so there the check has nothing to run.
fn checksignals() -> Result<(), crate::PyError> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        crate::module::signal::interp_signal::checksignals_now()
    }
    #[cfg(target_arch = "wasm32")]
    {
        Ok(())
    }
}

/// `os_lock.py:43-60 acquire_timed` — "Helper to acquire an interruptible lock
/// with a timeout."  `RPY_LOCK_INTR` reports a wait that a signal handler cut
/// short: deliver the signal, then retry with whatever time is left.
///
/// `acquire` is the `RPyThreadAcquireLockTimed` primitive of the lock being
/// taken; upstream reaches it as `lock.acquire_timed`
/// (`rthread.py:192-197 Lock.acquire_timed`, `intr_flag=1`).
fn acquire_timed(
    mut microseconds: i64,
    mut acquire: impl FnMut(i64) -> i64,
) -> Result<i64, crate::PyError> {
    // os_lock.py:45 `endtime`, measured here on the monotonic clock the
    // `time_sleep` retry loop already uses for its deadline.
    let start = Instant::now();
    let endtime = microseconds;
    loop {
        let mut result = acquire(microseconds);
        if result == RPY_LOCK_INTR {
            // Run signal handlers if we were interrupted
            checksignals()?;
            if microseconds >= 0 {
                microseconds = endtime - start.elapsed().as_micros() as i64;
                // Check for negative values, since those mean block forever
                if microseconds <= 0 {
                    result = RPY_LOCK_FAILURE;
                }
            }
        }
        if result != RPY_LOCK_INTR {
            return Ok(result);
        }
    }
}

mod lock_class {
    use super::*;
    // `pthread_cond_wait` returning without the lock is how
    // `RPyThreadAcquireLockTimed` detects a signal (thread_pthread.c:466-471),
    // so the wait must be the bare one-shot call that propagates spurious
    // wakeups.  `parking_lot`'s condition variable retries internally and
    // would swallow exactly the wakeup that carries the interrupt.
    use std::sync::{Condvar, Mutex};

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

        /// `thread_pthread.c:427-485 RPyThreadAcquireLockTimed`, the mutex and
        /// condition-variable build — the shape this lock has — with
        /// `intr_flag=1`, which is what `rthread.py:195` passes.
        fn acquire_timed(&self, microseconds: i64) -> i64 {
            // A potentially blocking native lock wait leaves the collector's
            // RUNNING census.  This does not serialize Python execution.  The
            // guard ends with the primitive, so the signal handlers the caller
            // runs on `RPY_LOCK_INTR` execute back inside the census.
            let _blocked = before_external_block();
            let mut locked = lock_state(&self.locked);
            let mut success;
            if !*locked {
                success = RPY_LOCK_ACQUIRED;
            } else if microseconds == 0 {
                success = RPY_LOCK_FAILURE;
            } else {
                let deadline = (microseconds > 0)
                    .then(|| Instant::now() + Duration::from_micros(microseconds as u64));
                success = RPY_LOCK_FAILURE;
                while success == RPY_LOCK_FAILURE {
                    if let Some(deadline) = deadline {
                        let now = Instant::now();
                        if now >= deadline {
                            break;
                        }
                        let (guard, timeout) = self
                            .ready
                            .wait_timeout(locked, deadline - now)
                            .unwrap_or_else(std::sync::PoisonError::into_inner);
                        locked = guard;
                        if timeout.timed_out() {
                            break;
                        }
                    } else {
                        locked = self
                            .ready
                            .wait(locked)
                            .unwrap_or_else(std::sync::PoisonError::into_inner);
                    }
                    if *locked {
                        // We were woken up, but didn't get the lock.  We
                        // probably received a signal.  Return RPY_LOCK_INTR to
                        // allow the caller to handle it and retry.
                        success = RPY_LOCK_INTR;
                    } else {
                        success = RPY_LOCK_ACQUIRED;
                    }
                }
            }
            if success == RPY_LOCK_ACQUIRED {
                *locked = true;
            }
            success
        }

        /// `os_lock.py:75-85 descr_lock_acquire`.
        fn acquire(
            &self,
            #[default(1)] blocking: i64,
            timeout: Option<PyObjectRef>,
        ) -> Result<bool, crate::PyError> {
            let microseconds = parse_acquire_args(blocking, timeout)?;
            let result = super::acquire_timed(microseconds, |us| self.acquire_timed(us))?;
            Ok(result == RPY_LOCK_ACQUIRED)
        }

        fn release(&self) -> Result<(), crate::PyError> {
            let mut locked = lock_state(&self.locked);
            if !*locked {
                return Err(crate::PyError::runtime_error("release unlocked lock"));
            }
            *locked = false;
            self.ready.notify_one();
            Ok(())
        }

        fn locked(&self) -> bool {
            *lock_state(&self.locked)
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
pub use lock_class::W_Lock;

mod rlock_class {
    use super::*;
    // Same interrupt-detection requirement as `lock_class`.
    use std::sync::{Condvar, Mutex};

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

        /// The native-lock half of `os_lock.py:206-241 acquire_w`, shaped to
        /// the `RPyThreadAcquireLockTimed` ABI (thread_pthread.c:427-485) so
        /// `acquire_timed` can deliver signals between attempts.
        ///
        /// Upstream keeps `rlock_count`/`rlock_owner` outside the native lock
        /// because the GIL serializes them; free-threaded pyre has no such
        /// serialization, so ownership is claimed under the same mutex the
        /// wait releases — the place `thread_pthread.c:479` sets `locked`.
        fn acquire_timed(&self, microseconds: i64, ident: i64) -> i64 {
            let _blocked = before_external_block();
            let mut state = lock_state(&self.state);
            let mut success;
            if state.count == 0 {
                success = RPY_LOCK_ACQUIRED;
            } else if microseconds == 0 {
                success = RPY_LOCK_FAILURE;
            } else {
                let deadline = (microseconds > 0)
                    .then(|| Instant::now() + Duration::from_micros(microseconds as u64));
                success = RPY_LOCK_FAILURE;
                while success == RPY_LOCK_FAILURE {
                    if let Some(deadline) = deadline {
                        let now = Instant::now();
                        if now >= deadline {
                            break;
                        }
                        let (guard, timeout) = self
                            .ready
                            .wait_timeout(state, deadline - now)
                            .unwrap_or_else(std::sync::PoisonError::into_inner);
                        state = guard;
                        if timeout.timed_out() {
                            break;
                        }
                    } else {
                        state = self
                            .ready
                            .wait(state)
                            .unwrap_or_else(std::sync::PoisonError::into_inner);
                    }
                    if state.count != 0 {
                        // Woken without the lock — probably a signal.
                        success = RPY_LOCK_INTR;
                    } else {
                        success = RPY_LOCK_ACQUIRED;
                    }
                }
            }
            if success == RPY_LOCK_ACQUIRED {
                state.owner = ident;
                state.count = 1;
            }
            success
        }

        /// `os_lock.py:206-241 acquire_w`.
        fn acquire(
            &self,
            #[default(1)] blocking: i64,
            timeout: Option<PyObjectRef>,
        ) -> Result<bool, crate::PyError> {
            let microseconds = parse_acquire_args(blocking, timeout)?;
            let tid = current_ident();
            {
                let mut state = lock_state(&self.state);
                if state.count > 0 && state.owner == tid {
                    state.count = state.count.checked_add(1).ok_or_else(|| {
                        crate::PyError::overflow_error("internal lock count overflowed")
                    })?;
                    return Ok(true);
                }
            }
            // os_lock.py:231-235 — `self.lock.acquire(False)` first; only a
            // failed poll waits, and only when `blocking`.  The count check
            // upstream pairs it with is the same predicate the poll tests,
            // because here the count is the lock.
            let mut r = self.acquire_timed(0, tid) == RPY_LOCK_ACQUIRED;
            if !r {
                if blocking == 0 {
                    return Ok(false);
                }
                r = super::acquire_timed(microseconds, |us| self.acquire_timed(us, tid))?
                    == RPY_LOCK_ACQUIRED;
            }
            Ok(r)
        }

        fn release(&self) -> Result<(), crate::PyError> {
            let ident = current_ident();
            let mut state = lock_state(&self.state);
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
            lock_state(&self.state).count != 0
        }

        fn _is_owned(&self) -> bool {
            let state = lock_state(&self.state);
            state.count > 0 && state.owner == current_ident()
        }

        fn _recursion_count(&self) -> i64 {
            let state = lock_state(&self.state);
            if state.owner == current_ident() {
                state.count
            } else {
                0
            }
        }

        fn _release_save(&self) -> Result<PyObjectRef, crate::PyError> {
            let mut state = lock_state(&self.state);
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
            // os_lock.py:286-287 `self.lock.acquire(True)` reaches
            // `RPyThreadAcquireLockTimed` with `intr_flag=0`
            // (rthread.py:169-174), so an interrupted wait is retried rather
            // than reported: restoring a saved state is not a place where a
            // signal may be delivered.
            while self.acquire_timed(-1, owner) != RPY_LOCK_ACQUIRED {}
            lock_state(&self.state).count = count;
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
            let state = lock_state(&self.state);
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
pub use rlock_class::W_RLock;

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
                    // os_lock.py:33-39 parse_acquire_args — a timeout past the
                    // microsecond clock's range is an OverflowError, never a
                    // native abort.  The negated comparison rejects NaN too.
                    let secs = floatobject::w_float_get_value(timeout);
                    if !(secs <= TIMEOUT_MAX) {
                        return Err(crate::PyError::overflow_error("timeout value is too large"));
                    }
                    Some(Duration::from_secs_f64(secs.max(0.0)))
                } else if is_int(timeout) {
                    let secs = w_int_get_value(timeout);
                    if secs as f64 > TIMEOUT_MAX {
                        return Err(crate::PyError::overflow_error("timeout value is too large"));
                    }
                    Some(Duration::from_secs(secs.max(0) as u64))
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
pub use handle_class::W_ThreadHandle;

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
    ///
    /// `initargs` and `initkwargs` are `Local.__init__`'s `self.initargs`
    /// (os_local.py:25).  Upstream keeps one `Arguments`; pyre's call surface
    /// takes the positional and keyword halves separately, so they are stored as
    /// the positional tuple and the construction call's keyword mapping (null
    /// when it had none), and `create_new_dict` replays the call from both.
    #[crate::pyre_class("_thread._local")]
    pub struct W_Local {
        dicts: PyObjectRef,
        initargs: PyObjectRef,
        initkwargs: PyObjectRef,
        last_dict: PyObjectRef,
        last_ident: i64,
        /// Guards `dicts` and the `last_dict`/`last_ident` pair.  Upstream
        /// keeps both unsynchronized — `os_local.py:36` "cache the last seen
        /// dict, works because we are protected by the GIL" — which
        /// free-threaded pyre cannot rely on.
        state_lock: Mutex<()>,
    }

    impl W_Local {
        /// `os_local.py:47-64 create_new_dict`.
        fn create_new_dict(&self, ident: i64) -> Result<PyObjectRef, crate::PyError> {
            let this = self as *const Self as *mut Self;
            let obj = this as PyObjectRef;
            // create a new dict for this thread
            let w_dict = pyre_object::w_dict_new();
            let roots = pyre_object::gc_roots::push_roots();
            pyre_object::gc_roots::pin_root(w_dict);
            let dict_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
            // Published before the initializer runs: the `__init__` about to be
            // entered reaches `getdict` again, and finding this entry is what
            // stops it recursing back into `create_new_dict` (os_local.py:28-31).
            {
                let _guard = self.state_lock.lock();
                unsafe { pyre_object::w_dict_setitem(self.dicts, ident, w_dict) };
            }
            // call __init__ — `space.call_obj_args(w_init, self, self.initargs)`.
            // The argument pointers are copied out of the tuple only after the
            // lookup, which can itself run Python: nothing between the copy and
            // the callee rooting them allocates, so a collection cannot leave
            // the copies naming pre-move addresses.
            let result = crate::typedef::r#type(obj)
                .ok_or_else(|| crate::PyError::type_error("_local instance has no type"))
                .and_then(|w_type| crate::baseobjspace::getattr_str(w_type.as_ptr(), "__init__"))
                .and_then(|w_init| {
                    let mut call_args = vec![obj];
                    call_args.extend(unsafe { w_tuple_items_copy_as_vec(self.initargs) });
                    self.call_init(w_init, &call_args)
                });
            if let Err(err) = result {
                // failed, forget w_dict and propagate the exception
                let key = w_int_new(ident);
                let _guard = self.state_lock.lock();
                unsafe { pyre_object::w_dict_delitem(self.dicts, key) };
                // The initializer reached `current_dict` before it raised —
                // assigning an instance attribute is what publishes the dict —
                // so this thread's cache now names the entry just removed.
                // Leaving it makes the next access return a half-initialized
                // dict instead of rerunning `__init__`; `thread_is_stopping`
                // drops the same pair alongside the same removal.
                unsafe {
                    if (*this).last_ident == ident {
                        (*this).last_ident = 0;
                        (*this).last_dict = PY_NULL;
                    }
                }
                return Err(err);
            }
            // ready.  `register_local_in_current_ec` allocates a weakref, so
            // the dict stays rooted across it and is reloaded afterwards: it is
            // still reachable through `self.dicts`, and a moving collection
            // would relocate it there while this frame's copy kept the pre-move
            // address for `current_dict` to cache and return.
            register_local_in_current_ec(obj);
            let w_dict = pyre_object::gc_roots::shadow_stack_get(dict_slot);
            drop(roots);
            Ok(w_dict)
        }

        /// The call itself of `os_local.py:57 space.call_obj_args(w_init, self,
        /// self.initargs)`.  `args` is the instance followed by the stored
        /// positional arguments; the stored keywords are bound by name.
        ///
        /// Upstream passes a single `Arguments` to `space.call_args`, which
        /// binds keywords whatever the caller is.  Pyre splits that surface:
        /// `call::call_with_kwargs` binds keywords but needs the running frame,
        /// while the frame-less `call_function_impl_result` is positional only.
        /// Reaching the frame through the execution context is the same
        /// resolution `call::call_metaclass_with_kwargs` uses, down to falling
        /// back to the positional call when there is no frame — a receiver only
        /// reaches here from Python code, which always has one.
        fn call_init(
            &self,
            w_init: PyObjectRef,
            args: &[PyObjectRef],
        ) -> Result<PyObjectRef, crate::PyError> {
            let kwds = crate::builtins::builtin_kwarg_entries(
                (!self.initkwargs.is_null()).then_some(self.initkwargs),
            );
            let frame = {
                let ec = crate::call::getexecutioncontext() as *mut crate::PyExecutionContext;
                if ec.is_null() {
                    std::ptr::null_mut()
                } else {
                    unsafe { (*ec).gettopframe_raw() }
                }
            };
            if kwds.is_empty() || frame.is_null() {
                return crate::call::call_function_impl_result(w_init, args);
            }
            crate::call::call_with_kwargs(unsafe { &mut *frame }, w_init, args, &kwds)
        }

        /// `os_local.py:66-76 getdict`.
        pub(super) fn current_dict(&self) -> Result<PyObjectRef, crate::PyError> {
            let ident = current_ident();
            {
                let _guard = self.state_lock.lock();
                if self.last_ident == ident && !self.last_dict.is_null() {
                    return Ok(self.last_dict);
                }
            }
            // `dicts` is mutated under `state_lock` by `create_new_dict` and
            // `thread_is_stopping`, so the probe takes it too rather than
            // reading the native dict beside a concurrent write.  The lock is
            // released before `create_new_dict`, which runs app-level
            // `__init__` and reenters this method.
            let existing = {
                let _guard = self.state_lock.lock();
                unsafe { pyre_object::w_dict_getitem(self.dicts, ident) }
            };
            let w_dict = match existing {
                Some(w_dict) => w_dict,
                None => self.create_new_dict(ident)?,
            };
            let this = self as *const Self as *mut Self;
            let _guard = self.state_lock.lock();
            unsafe {
                (*this).last_ident = ident;
                (*this).last_dict = w_dict;
            }
            pyre_object::gc_hook::try_gc_write_barrier(this as *mut u8);
            Ok(w_dict)
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
        /// `os_local.py:78-88 descr_local__new__`.
        #[staticmethod]
        fn __new__(cls: PyObjectRef, args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
            crate::typedef::check_user_subclass(type_object(), cls)?;
            let (positional, kwargs) = crate::builtins::split_builtin_kwargs(args);
            if positional.len() > 1 || crate::builtins::has_real_kwargs(kwargs) {
                // Construction arguments are rejected by the initializer the
                // subtype inherits, not by the requested type: a subclass that
                // defines its own `__init__` consumes them, and
                // `create_new_dict` replays that call on every further thread.
                // os_local.py:81 runs this ahead of `allocate_instance`, so a
                // refused construction never reaches `_register_in_ec`
                // (os_local.py:40).
                let w_parent_init = unsafe { crate::baseobjspace::lookup_where(cls, "__init__") }
                    .map(|(w_where, _)| w_where);
                if w_parent_init == Some(crate::typedef::w_object()) {
                    return Err(crate::PyError::type_error(
                        "Initialization arguments are not supported",
                    ));
                }
            }
            // os_local.py:23-38 `Local.__init__` installs the first dictionary
            // before app-level __init__ is entered, preventing recursive
            // initialization.
            let roots = pyre_object::gc_roots::push_roots();
            let cls_slot = pyre_object::gc_roots::shadow_stack_len();
            pyre_object::gc_roots::pin_root(cls);
            // A plain `_local()` has no keyword mapping, and a null takes no
            // shadow-stack slot.
            let initkwargs_slot = kwargs.map(|w_kwargs| {
                let slot = pyre_object::gc_roots::shadow_stack_len();
                pyre_object::gc_roots::pin_root(w_kwargs);
                slot
            });
            // Pinned BEFORE the tuple is built: `w_tuple_new` allocates, and
            // while it roots the elements it is given, `cls` and the keyword
            // mapping are raw locals of this frame that a moving collection
            // cannot update.  Pinning them afterwards would record evacuated
            // addresses, so both are read back from their slots below.
            let initargs = w_tuple_new(positional.get(1..).unwrap_or(&[]).to_vec());
            let initargs_slot = pyre_object::gc_roots::shadow_stack_len();
            pyre_object::gc_roots::pin_root(initargs);
            let dicts_slot = pyre_object::gc_roots::shadow_stack_len();
            let dicts = pyre_object::w_dict_new();
            pyre_object::gc_roots::pin_root(dicts);
            let dict_slot = pyre_object::gc_roots::shadow_stack_len();
            let w_dict = pyre_object::w_dict_new();
            pyre_object::gc_roots::pin_root(w_dict);
            let ident = current_ident();
            unsafe {
                pyre_object::w_dict_setitem(
                    pyre_object::gc_roots::shadow_stack_get(dicts_slot),
                    ident,
                    pyre_object::gc_roots::shadow_stack_get(dict_slot),
                )
            };
            // The object slots are filled after the allocation, from the
            // shadow stack, so a collection triggered by `allocate_stable`
            // cannot leave the fresh instance holding pre-move addresses.
            let obj = Self::allocate_stable(Self {
                ob: PyObject::default(),
                dicts: PY_NULL,
                initargs: PY_NULL,
                initkwargs: PY_NULL,
                last_dict: PY_NULL,
                last_ident: ident,
                state_lock: parking_lot::const_mutex(()),
            });
            unsafe {
                let this = obj as *mut Self;
                (*obj).w_class = pyre_object::gc_roots::shadow_stack_get(cls_slot);
                (*this).dicts = pyre_object::gc_roots::shadow_stack_get(dicts_slot);
                (*this).initargs = pyre_object::gc_roots::shadow_stack_get(initargs_slot);
                (*this).initkwargs = initkwargs_slot
                    .map(pyre_object::gc_roots::shadow_stack_get)
                    .unwrap_or(PY_NULL);
                (*this).last_dict = pyre_object::gc_roots::shadow_stack_get(dict_slot);
            }
            pyre_object::gc_hook::try_gc_write_barrier(obj as *mut u8);
            drop(roots);
            register_local_in_current_ec(obj);
            Ok(obj)
        }

        #[getter]
        fn __dict__(&self) -> Result<PyObjectRef, crate::PyError> {
            self.current_dict()
        }
    }
}
pub use local_class::W_Local;

fn local_type() -> PyObjectRef {
    local_class::type_object()
}

/// W_Root.getdict dispatch for `os_local.Local.getdict`.  `None` means the
/// receiver is not a `_local`; `Some(Err(..))` is the app-level `__init__` the
/// first access from a thread runs (`os_local.py:73 create_new_dict`) raising.
pub(crate) fn local_getdict(obj: PyObjectRef) -> Option<Result<PyObjectRef, crate::PyError>> {
    let local = W_Local::from_obj(obj)?;
    Some(local.current_dict())
}

/// True when `obj` is a `_thread._local`, the one receiver whose `getdict`
/// runs app-level code.
pub(crate) fn is_local(obj: PyObjectRef) -> bool {
    W_Local::from_obj(obj).is_some()
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
    // Last: nothing above runs bytecode, so nothing can reach the ticker this
    // action is registered on after it is gone.
    gil::shutdown(ec);
}

/// The calling thread's identity.
///
/// The host thread id is read fresh on every call and is never a build-time
/// constant, so the front end residualizes the read instead of tracing into
/// `rustpython_host_env::thread::current_thread_id`.  This is the single
/// in-tree seam every traced caller reaches it through.
#[majit_macros::dont_look_inside]
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
    // os_thread.py:172 `start_new_thread` begins with `setup_threads(space)`.
    gil::setup_threads(unsafe { &mut *(parent_ec as *mut crate::PyExecutionContext) });

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
    // `os_thread.py`'s Bootstrapper uses the RPython pthread lock: the new
    // thread publishes that bootstrap completed and releases the starter.
    // Do not use Rust's zero-capacity mpsc channel for this hand-off.  On
    // Darwin its waiting side reaches `std::thread::park`, backed by a
    // libdispatch semaphore; libdispatch deliberately traps if that primitive
    // is first created in a child of an ever-multithreaded process.  The
    // release/acquire word is the same one-value bootstrap rendezvous without
    // introducing a second owner for any Python state.
    const START_FAILED: usize = usize::MAX;

    /// The rendezvous word plus the failure detail the starter reports.
    struct Bootstrap {
        word: AtomicUsize,
        error: std::sync::OnceLock<String>,
    }
    /// Worker-side handle on the rendezvous.  A dropped `mpsc` sender signalled
    /// a dead worker for free; a bare word does not, so every exit that never
    /// reached `publish` has to hand `START_FAILED` over instead — including an
    /// unwind, which only a destructor can catch.  Otherwise a worker that dies
    /// during bootstrap leaves the starter spinning forever.
    struct BootstrapSignal(std::sync::Arc<Bootstrap>);
    impl BootstrapSignal {
        /// The word carries the ident *and* two reserved states, so an ident
        /// equal to either would be misread: `0` leaves the starter looping
        /// and `START_FAILED` raises "can't start new thread" for a thread
        /// that started. `current_ident` is the OS thread id — `gettid` on
        /// Linux, `pthread_threadid_np` on Darwin — and neither issues those
        /// two values, which is the invariant this pins.
        fn publish(&self, ident: i64) {
            debug_assert!(
                ident as usize != 0 && ident as usize != START_FAILED,
                "thread ident collides with a reserved rendezvous state"
            );
            self.0.word.store(ident as usize, Ordering::Release);
        }
        /// Preserve the diagnostic the starter raises.  `os_thread.py:145`
        /// reports a bare "can't start new thread", which stays the fallback
        /// when bootstrap died without recording anything.
        fn fail(&self, message: String) {
            let _ = self.0.error.set(message);
        }
    }
    impl Drop for BootstrapSignal {
        fn drop(&mut self) {
            // Takes the word only from its initial state, so this is a no-op
            // once `publish` has run.
            let _ =
                self.0
                    .word
                    .compare_exchange(0, START_FAILED, Ordering::Release, Ordering::Relaxed);
        }
    }

    let started = std::sync::Arc::new(Bootstrap {
        word: AtomicUsize::new(0),
        error: std::sync::OnceLock::new(),
    });
    let worker_started = std::sync::Arc::clone(&started);

    let configured_stack_size = STACK_SIZE.load(Ordering::Relaxed);
    let stack_size = if configured_stack_size == 0 {
        crate::stack_check::DEFAULT_RUNTIME_THREAD_STACK_SIZE
    } else {
        configured_stack_size
    };
    let builder = std::thread::Builder::new().stack_size(stack_size);
    builder
        .spawn(move || {
            // First statement: from here on every exit path, panic included,
            // releases the starter.
            let bootstrap = BootstrapSignal(worker_started);
            crate::stack_check::configure_current_thread_stack_size(stack_size);
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
            // Each mutator owns its ticker, so the GIL-releasing action has to
            // be registered on this thread's own actionflag; without it a
            // worker would hold the GIL until its next external call.
            gil::initialize(&mut ec);
            let ident = current_ident();
            if has_handle {
                let h = W_ThreadHandle::from_obj(handle_addr as PyObjectRef).unwrap();
                if let Err(e) = h.start(ident) {
                    bootstrap.fail(e.message);
                    thread_is_stopping(&mut ec);
                    crate::call::set_last_exec_ctx(std::ptr::null());
                    drop(worker_roots);
                    drop(ec);
                    return;
                }
            }
            THREAD_COUNT.fetch_add(1, Ordering::SeqCst);
            bootstrap.publish(ident);

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
        loop {
            let value = started.word.load(Ordering::Acquire);
            if value == START_FAILED {
                let message = started
                    .error
                    .get()
                    .cloned()
                    .unwrap_or_else(|| "can't start new thread".to_string());
                break Err(crate::PyError::runtime_error(message));
            }
            if value != 0 {
                break Ok(value as i64);
            }
            std::thread::yield_now();
        }
    };
    drop(roots);
    result
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
        // `@unwrap_spec(size=int)` (os_thread.py:216) unwraps through
        // `space.int_w`, which rejects a non-integer instead of reading its
        // payload word as one.
        let size = crate::baseobjspace::int_w(arg)?;
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

/// CPython 3.14 `ExceptHookArgs_desc`: the immutable, non-baseable
/// `_thread._ExceptHookArgs` struct sequence passed to
/// `threading.excepthook`.  The storage and descriptors come from pyre's
/// line-by-line port of PyPy `lib_pypy/_structseq.py`.
fn except_hook_args_type() -> PyObjectRef {
    static TYPE: OnceLock<usize> = OnceLock::new();
    *TYPE.get_or_init(|| {
        let _roots = pyre_object::gc_roots::push_roots();
        let ty = crate::_structseq::make_struct_seq(
            "_thread._ExceptHookArgs",
            &["exc_type", "exc_value", "exc_traceback", "thread"],
        );
        let ty_slot = pin_root_slot(ty);
        unsafe {
            let ty = pyre_object::gc_roots::shadow_stack_get(ty_slot);
            pyre_object::w_type_set_acceptable_as_base_class(ty, false);
            let doc =
                w_str_new("ExceptHookArgs\n\nType used to pass arguments to threading.excepthook.");
            let doc_slot = pin_root_slot(doc);
            let ty = pyre_object::gc_roots::shadow_stack_get(ty_slot);
            let ns = pyre_object::w_type_get_dict_ptr(ty) as PyObjectRef;
            pyre_object::w_dict_setitem_str_no_proxy(
                ns,
                "__doc__",
                pyre_object::gc_roots::shadow_stack_get(doc_slot),
            );
        }
        pyre_object::gc_roots::shadow_stack_get(ty_slot) as usize
    }) as PyObjectRef
}

#[inline]
fn pin_root_slot(value: PyObjectRef) -> usize {
    pyre_object::gc_roots::pin_root(value);
    pyre_object::gc_roots::shadow_stack_len() - 1
}

fn call_method_result(
    obj: PyObjectRef,
    name: &str,
    args: &[PyObjectRef],
) -> Result<PyObjectRef, crate::PyError> {
    let result = crate::baseobjspace::call_method(obj, name, args);
    if result.is_null() {
        Err(crate::call::take_call_error()
            .unwrap_or_else(|| crate::PyError::runtime_error(format!("{name} failed"))))
    } else {
        Ok(result)
    }
}

/// CPython `PyObject_GetOptionalAttr`: only `AttributeError` denotes a missing
/// attribute.  In particular, a `NameError` raised by a descriptor propagates.
fn optional_attr(obj: PyObjectRef, name: &str) -> Result<Option<PyObjectRef>, crate::PyError> {
    match crate::baseobjspace::getattr_str(obj, name) {
        Ok(value) if value.is_null() => Ok(None),
        Ok(value) => Ok(Some(value)),
        Err(err) if err.kind == crate::PyErrorKind::AttributeError => Ok(None),
        Err(err) => Err(err),
    }
}

/// `thread_excepthook_file`: write one string to the live file root.
fn thread_excepthook_write(file_slot: usize, text: PyObjectRef) -> Result<(), crate::PyError> {
    let text_slot = pin_root_slot(text);
    call_method_result(
        pyre_object::gc_roots::shadow_stack_get(file_slot),
        "write",
        &[pyre_object::gc_roots::shadow_stack_get(text_slot)],
    )?;
    Ok(())
}

/// CPython 3.14 `thread_excepthook_file`.
fn thread_excepthook_file(
    file: PyObjectRef,
    exc_value: PyObjectRef,
    exc_traceback: PyObjectRef,
    thread: PyObjectRef,
) -> Result<(), crate::PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    let file_slot = pin_root_slot(file);
    let exc_value_slot = pin_root_slot(exc_value);
    let exc_traceback_slot = pin_root_slot(exc_traceback);
    let thread_slot = pin_root_slot(thread);

    // `PyFile_WriteString("Exception in thread ", file)`.
    thread_excepthook_write(file_slot, w_str_new("Exception in thread "))?;

    // `PyObject_GetOptionalAttr(thread, "name")`; a missing name falls back
    // to the native thread identifier, while a raising descriptor propagates.
    let thread = pyre_object::gc_roots::shadow_stack_get(thread_slot);
    let name = if !unsafe { is_none(thread) } {
        optional_attr(thread, "name")?
    } else {
        None
    };
    if let Some(name) = name {
        let name_slot = pin_root_slot(name);
        let rendered = unsafe {
            crate::display::py_str_wtf8(pyre_object::gc_roots::shadow_stack_get(name_slot))?
        };
        thread_excepthook_write(file_slot, w_str_from_wtf8(rendered))?;
    } else {
        thread_excepthook_write(file_slot, w_str_new(&current_ident().to_string()))?;
    }
    thread_excepthook_write(file_slot, w_str_new(":\n"))?;

    // `_PyErr_Display(file, exc_type, exc_value, exc_traceback)`.  The
    // renderer consumes the explicit traceback field and never rewrites the
    // exception object's own `__traceback__`.
    let mut rendered = Vec::new();
    crate::error::write_exception_from_parts(
        &mut rendered,
        pyre_object::gc_roots::shadow_stack_get(exc_value_slot),
        pyre_object::gc_roots::shadow_stack_get(exc_traceback_slot),
    )
    .map_err(|err| crate::PyError::runtime_error(format!("failed to display exception: {err}")))?;
    let rendered = rustpython_wtf8::Wtf8Buf::from_bytes(rendered)
        .map_err(|_| crate::PyError::runtime_error("invalid WTF-8 exception display"))?;
    thread_excepthook_write(file_slot, pyre_object::w_str_from_wtf8(rendered))?;

    // `_PyFile_Flush(file)`.
    call_method_result(
        pyre_object::gc_roots::shadow_stack_get(file_slot),
        "flush",
        &[],
    )?;
    Ok(())
}

/// CPython 3.14 `thread_excepthook`.
fn thread_excepthook(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let hook_args = args
        .first()
        .copied()
        .ok_or_else(|| crate::PyError::type_error("_excepthook() missing argument"))?;
    let _roots = pyre_object::gc_roots::push_roots();
    let hook_args_slot = pin_root_slot(hook_args);
    let hook_args = pyre_object::gc_roots::shadow_stack_get(hook_args_slot);
    let actual_type = crate::typedef::r#type(hook_args)
        .map(|tp| tp.as_ptr())
        .unwrap_or(PY_NULL);
    if !std::ptr::eq(actual_type, except_hook_args_type()) {
        return Err(crate::PyError::type_error(
            "_thread.excepthook argument type must be ExceptHookArgs",
        ));
    }

    let exc_type = unsafe {
        w_tuple_getitem(pyre_object::gc_roots::shadow_stack_get(hook_args_slot), 0)
            .expect("ExceptHookArgs has four sequence fields")
    };
    let exc_type_slot = pin_root_slot(exc_type);
    if crate::builtins::lookup_exc_class("SystemExit").is_some_and(|system_exit| {
        crate::baseobjspace::is_w(
            pyre_object::gc_roots::shadow_stack_get(exc_type_slot),
            system_exit,
        )
    }) {
        return Ok(w_none());
    }

    let exc_value = unsafe {
        w_tuple_getitem(pyre_object::gc_roots::shadow_stack_get(hook_args_slot), 1)
            .expect("ExceptHookArgs has four sequence fields")
    };
    let exc_traceback = unsafe {
        w_tuple_getitem(pyre_object::gc_roots::shadow_stack_get(hook_args_slot), 2)
            .expect("ExceptHookArgs has four sequence fields")
    };
    let thread = unsafe {
        w_tuple_getitem(pyre_object::gc_roots::shadow_stack_get(hook_args_slot), 3)
            .expect("ExceptHookArgs has four sequence fields")
    };
    let exc_value_slot = pin_root_slot(exc_value);
    let exc_traceback_slot = pin_root_slot(exc_traceback);
    let thread_slot = pin_root_slot(thread);

    // `_PySys_GetOptionalAttr("stderr")` reads the interpreter-owned sys dict,
    // not the replaceable `sys.modules["sys"]` entry.  When stderr is
    // absent/None, use the Thread object's saved `_stderr`, unless both the
    // stream and thread are None.
    let sys = crate::importing::get_interpreter_sys_module()
        .ok_or_else(|| crate::PyError::runtime_error("sys module is unavailable"))?;
    let sys_dict = unsafe { pyre_object::w_module_get_w_dict(sys) };
    let mut file =
        unsafe { pyre_object::w_module_dict_getitem_str(sys_dict, "stderr") }.unwrap_or(PY_NULL);
    if file.is_null() || unsafe { is_none(file) } {
        let thread = pyre_object::gc_roots::shadow_stack_get(thread_slot);
        if unsafe { is_none(thread) } {
            return Ok(w_none());
        }
        file = crate::baseobjspace::getattr_str(thread, "_stderr")?;
        if unsafe { is_none(file) } {
            return Ok(w_none());
        }
    }
    let file_slot = pin_root_slot(file);
    thread_excepthook_file(
        pyre_object::gc_roots::shadow_stack_get(file_slot),
        pyre_object::gc_roots::shadow_stack_get(exc_value_slot),
        pyre_object::gc_roots::shadow_stack_get(exc_traceback_slot),
        pyre_object::gc_roots::shadow_stack_get(thread_slot),
    )?;
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
        "_ExceptHookArgs" => except_hook_args_type(),
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
        "_excepthook"            / 1 = thread_excepthook,
        "_get_main_thread_ident" / 0 = |_| Ok(w_int_new(current_ident())),
        "start_joinable_thread"  / * = start_joinable_thread,
        "start_new_thread"       / * = start_new_thread,
        "start_new"              / * = start_new_thread,
    },
}
