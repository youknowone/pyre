//! gc_sync — Stop-the-world safepoint protocol for free-threading GC.
//!
//! Provides the synchronisation harness around incminimark's collection.
//! Mutators run in parallel; collection pauses all of them via STW.
//! The collector code (`do_collect_nursery`, `do_collect_full`) runs
//! unchanged inside the STW window — it already assumes a single-threaded
//! world during collection.
//!
//! # P0 simplification
//!
//! Every GC operation (alloc, collect, barrier, query) acquires `gc_mutex`
//! briefly. Single-threaded production has zero contention (~20ns
//! uncontended Mutex). cargo test threads serialise correctly.
//! P1 will restore performance with TLAB (per-thread nursery chunks).
//!
//! # Design
//!
//! This is NOT a GIL — mutators do not hold a lock during Python execution.
//! The lock is held only for the duration of each individual GC operation.
//! Collection begins only when every other registered thread is at an
//! entry-style safepoint where all of its live GC references are rooted.
//! A slow `gc_op` leaves RUNNING before blocking on `gc_mutex`, then rejoins
//! RUNNING before its closure executes. A dispatch poll parks directly. There
//! is no exit safepoint: a returned reference remains protected until the
//! caller roots it before its next collection-capable call.

use std::cell::{Cell, UnsafeCell};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex};

use crate::GcAllocator;

/// Process-global GC singleton storage.
/// `UnsafeCell` provides interior mutability; access is serialised by
/// `GC_SYNC.gc_mutex`. `Sync` is sound because all `&mut` access goes
/// through the mutex.
struct GcSingleton(UnsafeCell<Option<Box<dyn GcAllocator>>>);
unsafe impl Sync for GcSingleton {}

static GC_STORE: GcSingleton = GcSingleton(UnsafeCell::new(None));
static GC_INITIALIZED: AtomicBool = AtomicBool::new(false);

/// STW safepoint state.
pub struct GcSync {
    /// Mutex serialising all GC operations. Held briefly per alloc/barrier
    /// (P0). Held for full STW duration during collection.
    gc_mutex: Mutex<()>,
    /// Set while a collector is draining other mutators to entry-style
    /// safepoints. Cleared before the collector releases `gc_mutex`.
    stw_requested: AtomicBool,
    /// RUNNING registered-mutator count. A slow `gc_op` removes itself before
    /// waiting on `gc_mutex`, so a mutex-holding collector can drain every
    /// other mutator while remaining counted itself.
    quiesce: Mutex<QuiesceState>,
    /// Signalled whenever RUNNING decreases toward the collector-inclusive
    /// drain target (one when the collector is counted, otherwise zero).
    quiesced: Condvar,
    /// Signalled when STW ends and parked mutators may become RUNNING again.
    resumed: Condvar,
    /// Generation counter incremented after each STW. Prevents spurious
    /// wake: a thread checks that generation changed before proceeding.
    stw_generation: AtomicUsize,
}

struct QuiesceState {
    running: usize,
}

static GC_SYNC: GcSync = GcSync {
    gc_mutex: Mutex::new(()),
    stw_requested: AtomicBool::new(false),
    quiesce: Mutex::new(QuiesceState { running: 0 }),
    quiesced: Condvar::new(),
    resumed: Condvar::new(),
    stw_generation: AtomicUsize::new(0),
};

// ──────────────────────────────────────────────────────────────
// Singleton management
// ──────────────────────────────────────────────────────────────

/// Store the GC singleton. Idempotent — subsequent calls are no-ops.
/// Must be called before any `gc_op`.
pub fn store_singleton(gc: Box<dyn GcAllocator>) {
    if GC_INITIALIZED.load(Ordering::Acquire) {
        return;
    }
    let _guard = GC_SYNC.gc_mutex.lock().unwrap();
    // Double-check after acquiring mutex.
    if GC_INITIALIZED.load(Ordering::Acquire) {
        return;
    }
    // SAFETY: gc_mutex held, no concurrent access.
    unsafe {
        *GC_STORE.0.get() = Some(gc);
    }
    GC_INITIALIZED.store(true, Ordering::Release);
}

/// Test-support: install a fresh GC singleton, LEAKING the previous one.
///
/// The prior GC's objects must NOT be freed; process-global immortal builtins
/// (a builtin type's `weak_subclasses`, etc.) may still reference them, so
/// dropping the old `OldGen` would leave those references dangling. Forgetting
/// the old singleton keeps them valid. Used by the `gc_stress` harness to give
/// each per-worker test a pristine heap and empty root set, so a prior test's
/// oldgen residue or stale registered roots cannot corrupt this test's
/// collections.
pub fn replace_singleton_leaking_old(gc: Box<dyn GcAllocator>) {
    let _guard = GC_SYNC.gc_mutex.lock().unwrap();
    // SAFETY: gc_mutex held; the gc_stress harness runs tests serially, so no
    // concurrent gc_op is in flight during the swap.
    unsafe {
        if let Some(old) = (*GC_STORE.0.get()).take() {
            std::mem::forget(old);
        }
        *GC_STORE.0.get() = Some(gc);
    }
    GC_INITIALIZED.store(true, Ordering::Release);
}

/// Check if the GC singleton has been initialized.
pub fn is_initialized() -> bool {
    GC_INITIALIZED.load(Ordering::Acquire)
}

/// Access the GC singleton mutably under gc_mutex protection.
/// SAFETY: caller must hold gc_mutex.
unsafe fn singleton_mut() -> &'static mut dyn GcAllocator {
    // SAFETY: caller holds gc_mutex, so there is no concurrent access.
    unsafe { &mut *GC_STORE.0.get() }
        .as_deref_mut()
        .expect("GC singleton not initialized — call store_singleton() first")
}

// ──────────────────────────────────────────────────────────────
// Reentrancy guard — collection-time read-only queries
// ──────────────────────────────────────────────────────────────

/// Per-thread GC-sync facts, kept in one struct like pypy_threadlocal_s
/// (threadlocal.c:46-97). Its address doubles as this thread's ident for the
/// global owner words below, mirroring how _rpygil_get_my_ident reads the
/// ident out of the threadlocal struct (threadlocal.h:143-146).
struct GcThreadState {
    /// Completed `register_thread`, represented in the RUNNING count.
    registered: Cell<bool>,
    /// Registered mutators are normally RUNNING. Outermost slow gc_op
    /// regions waiting on gc_mutex and dispatch safepoint parks flip this
    /// to false.
    running: Cell<bool>,
}

thread_local! {
    static GC_THREAD: GcThreadState = const {
        GcThreadState { registered: Cell::new(false), running: Cell::new(false) }
    };
}

/// Stable nonzero ident of the current thread: the address of its
/// `GC_THREAD` struct.
#[inline]
fn my_ident() -> usize {
    GC_THREAD.with(|t| t as *const GcThreadState as usize)
}

/// Ident of the thread currently holding the exclusive `&mut dyn GcAllocator`
/// (inside a `gc_op` closure); 0 when none. `in_gc_op` compares against
/// `my_ident()`, the `rpy_fastgil == get_ident()` idiom (thread_gil.c:21).
static GC_OP_OWNER: AtomicUsize = AtomicUsize::new(0);

/// Ident of the STW-owning collector thread; 0 when no STW. Only the owner
/// nests (do_collect_full drives do_collect_nursery), so a global owner word
/// plus depth replaces per-thread state.
static STW_OWNER: AtomicUsize = AtomicUsize::new(0);
static STW_DEPTH: AtomicUsize = AtomicUsize::new(0);

/// RAII marker: sets `GC_OP_OWNER` for the exact span of a closure that holds
/// the exclusive `&mut dyn GcAllocator`. Wraps every `singleton_mut()` call.
struct OpGuard {
    prev: usize,
}
impl OpGuard {
    #[inline]
    fn enter() -> Self {
        let prev = GC_OP_OWNER.swap(my_ident(), Ordering::SeqCst);
        assert!(
            prev == 0,
            "GC singleton exclusivity invariant violated: concurrent &mut access"
        );
        OpGuard { prev }
    }
}

/// Releases an acquired fast-path gate on both normal return and unwind.
struct FastPathGate {
    restore: usize,
}
impl Drop for FastPathGate {
    #[inline]
    fn drop(&mut self) {
        IN_FAST_PATH.store(self.restore, Ordering::Release);
    }
}
impl Drop for OpGuard {
    #[inline]
    fn drop(&mut self) {
        GC_OP_OWNER.store(self.prev, Ordering::SeqCst);
    }
}

/// Whether this thread is already inside a `gc_op` / `request_stw` closure
/// (i.e. a collection is running on this thread and holds the `&mut`). Uses an
/// owner-word comparison against this thread's ident.
#[inline]
pub fn in_gc_op() -> bool {
    GC_OP_OWNER.load(Ordering::Relaxed) == my_ident()
}

/// Shared reference to the singleton, re-derived from the static `UnsafeCell`.
///
/// SAFETY: only sound when [`in_gc_op`] holds on this thread — the collector
/// already owns the exclusive `&mut`, all other mutators are parked (STW) or
/// spinning (single-thread fast path), and the returned `&dyn` is used only for
/// a read-only query whose lifetime ends before control returns to the
/// collector. Re-derives from `GC_STORE.0.get()` each call (a pre-`&mut`-cached
/// raw pointer would be invalidated by `singleton_mut`'s reborrow).
#[inline]
unsafe fn singleton_ref_reentrant() -> &'static dyn GcAllocator {
    unsafe { &*GC_STORE.0.get() }
        .as_deref()
        .expect("GC singleton not initialized — call store_singleton() first")
}

/// Read-only query that is safe both at top level and reentrantly from inside a
/// collection (an extra-root walker's `gc_owns_object` / ownership query).
///
/// Top level (`!in_gc_op()`): takes the fully-synchronised [`gc_query`] path.
/// Reentrant (`in_gc_op()`): reads the singleton directly, without re-locking
/// `gc_mutex` (which would deadlock — the lock is non-recursive and, under STW,
/// held by this very collector) or forming a second `&mut`.
#[inline]
pub fn gc_query_reentrant<R>(f: impl FnOnce(&dyn GcAllocator) -> R) -> R {
    if in_gc_op() {
        // SAFETY: in_gc_op() ⇒ this thread holds the &mut and is the sole
        // running mutator (parked/spinning invariant); read-only, bounded to `f`.
        f(unsafe { singleton_ref_reentrant() })
    } else {
        gc_query(f)
    }
}

// ──────────────────────────────────────────────────────────────
// Mutator registry — single-thread fast path
// ──────────────────────────────────────────────────────────────

/// Number of threads that have called `register_thread` and not yet
/// `unregister_thread`.  When ≤ 1, `gc_op` skips the Mutex entirely.
static REGISTERED_THREADS: AtomicUsize = AtomicUsize::new(0);

/// Fast-path lock word: 0 when free, otherwise the holder thread's ident
/// (`my_ident`), following rpy_fastgil (thread_gil.c:7-31). Self-ownership
/// is checked by comparing against `my_ident()`.
static IN_FAST_PATH: AtomicUsize = AtomicUsize::new(0);

/// Register the current thread as a GC mutator. Paired with
/// `unregister_thread`. An unregistered thread may still call `gc_op`; it is
/// not counted for STW, but serialises through the same operation gate.
pub fn register_thread() {
    assert!(
        !GC_THREAD.with(|t| t.registered.get()),
        "GC mutator thread registered twice"
    );
    let old = REGISTERED_THREADS.fetch_add(1, Ordering::SeqCst);
    if old > 0 {
        // A second thread is arriving. Spin until the operation gate is free;
        // after this, existing callers see REGISTERED_THREADS > 1 and take the
        // Mutex path. CAS acquisition makes this stricter than the old marker.
        while IN_FAST_PATH.load(Ordering::Acquire) != 0 {
            std::hint::spin_loop();
        }
    }

    let mut state = GC_SYNC.quiesce.lock().unwrap();
    state = GC_SYNC
        .resumed
        .wait_while(state, |_| GC_SYNC.stw_requested.load(Ordering::Acquire))
        .unwrap();
    state.running += 1;
    GC_THREAD.with(|t| t.running.set(true));
    GC_THREAD.with(|t| t.registered.set(true));
    drop(state);

    if old > 0 && is_initialized() {
        // gc.py:525-531 publishes the nursery slots used by generated inline
        // allocation. A shared free-threaded nursery cannot safely use its
        // non-atomic bump sequence, so the 1→2 transition pins that path off.
        // This is sticky: unregistering threads never republishes the top.
        gc_op(|gc| gc.set_inline_alloc_enabled(false));
    }
}

/// Unregister the current thread. Subsequent `gc_op` calls remain serialised,
/// but this thread no longer participates in STW quiescence.
pub fn unregister_thread() {
    assert!(
        GC_THREAD.with(|t| t.registered.get()),
        "unregistering an unregistered GC mutator thread"
    );
    let mut state = GC_SYNC.quiesce.lock().unwrap();
    assert!(
        GC_THREAD.with(|t| t.running.replace(false)),
        "unregistering a parked GC mutator thread"
    );
    state.running = state
        .running
        .checked_sub(1)
        .expect("RUNNING underflow during unregister_thread");
    GC_THREAD.with(|t| t.registered.set(false));
    let old = REGISTERED_THREADS.fetch_sub(1, Ordering::SeqCst);
    assert!(old > 0, "REGISTERED_THREADS underflow");
    GC_SYNC.quiesced.notify_all();
}

/// Number of registered GC mutators.
#[inline]
pub fn registered_threads() -> usize {
    REGISTERED_THREADS.load(Ordering::Acquire)
}

/// Whether a stop-the-world pause is required for a collection driven by the
/// current thread: true iff at least one *other* thread is a registered mutator.
/// The current thread is excluded because the collector walks its own roots
/// directly (`walk_my_*`); the danger is an unwaited, unscanned OTHER mutator.
/// An unregistered collector with one registered mutator elsewhere still needs
/// STW, which the bare count `> 1` check misses.
#[inline]
pub fn stw_required() -> bool {
    let registered = REGISTERED_THREADS.load(Ordering::Acquire);
    let self_registered = usize::from(GC_THREAD.with(|t| t.registered.get()));
    registered.saturating_sub(self_registered) > 0
}

// ──────────────────────────────────────────────────────────────
// GC operation gate — fast path when single-threaded
// ──────────────────────────────────────────────────────────────

/// Execute a closure with exclusive `&mut dyn GcAllocator` access.
///
/// **Fast path** (single registered thread, no STW): direct access after
/// acquiring `IN_FAST_PATH`, without taking the Mutex.
///
/// **Slow path** (multiple threads or STW): acquires `gc_mutex`.
/// Single-threaded production always takes the fast path.
#[inline]
pub fn gc_op<R>(f: impl FnOnce(&mut dyn GcAllocator) -> R) -> R {
    let ident = my_ident();
    debug_assert!(
        !in_gc_op(),
        "reentrant &mut gc_op — a collection-time query must use gc_query_reentrant"
    );
    // Fast path: single thread, no STW.
    if REGISTERED_THREADS.load(Ordering::Acquire) <= 1
        && !GC_SYNC.stw_requested.load(Ordering::Acquire)
        && IN_FAST_PATH
            .compare_exchange(0, ident, Ordering::Acquire, Ordering::Relaxed)
            .is_ok()
    {
        let _fast_path_gate = FastPathGate { restore: 0 };
        // Recheck after acquiring the fast-path lock: another thread may have
        // registered or requested STW after the eligibility loads above.
        if REGISTERED_THREADS.load(Ordering::Acquire) <= 1
            && !GC_SYNC.stw_requested.load(Ordering::Acquire)
        {
            // SAFETY: the fast-path lock excludes other fast operations, and
            // slow operations drain it before accessing the singleton.
            let r = {
                let _op = OpGuard::enter();
                f(unsafe { singleton_mut() })
            };
            return r;
        }
    }
    gc_op_slow(f)
}

/// Slow path: Mutex-guarded access with STW parking.
#[cold]
fn gc_op_slow<R>(f: impl FnOnce(&mut dyn GcAllocator) -> R) -> R {
    let ident = my_ident();
    let registered = GC_THREAD.with(|t| t.registered.get());
    if registered {
        let mut state = GC_SYNC.quiesce.lock().unwrap();
        assert!(
            GC_THREAD.with(|t| t.running.replace(false)),
            "GC mutator entered a gc_op safepoint twice"
        );
        state.running = state
            .running
            .checked_sub(1)
            .expect("RUNNING underflow entering gc_op safepoint");
        GC_SYNC.quiesced.notify_all();
    }

    // Blocking on gc_mutex is the park: this thread is already excluded from
    // RUNNING, so a collector can hold the mutex while draining other threads.
    let _guard = GC_SYNC.gc_mutex.lock().unwrap();

    // Drain a fast operation before borrowing the singleton. A fast holder
    // never waits on gc_mutex, so this terminates. Together with CAS acquisition
    // this makes fast and slow singleton accesses mutually exclusive.
    let self_holds_fast_path = IN_FAST_PATH.load(Ordering::Acquire) == ident;
    while IN_FAST_PATH.load(Ordering::Acquire) != 0 && !self_holds_fast_path {
        std::hint::spin_loop();
    }
    if !self_holds_fast_path {
        // Claim the gate after draining so a new fast operation cannot start
        // between the final observation above and the singleton borrow.
        while IN_FAST_PATH
            .compare_exchange(0, ident, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            std::hint::spin_loop();
        }
    }
    let _fast_path_gate = (!self_holds_fast_path).then_some(FastPathGate { restore: 0 });

    if registered {
        let mut state = GC_SYNC.quiesce.lock().unwrap();
        // An STW requester holds gc_mutex and clears its request before
        // releasing it, so a thread that acquired gc_mutex cannot need to wait.
        debug_assert!(
            !GC_SYNC.stw_requested.load(Ordering::Acquire),
            "gc_mutex holder observed an active STW request"
        );
        state.running += 1;
        assert!(
            !GC_THREAD.with(|t| t.running.replace(true)),
            "GC mutator re-entered RUNNING twice"
        );
    }

    let result = {
        let _op = OpGuard::enter();
        // There is deliberately no exit park. A returned reference is rooted
        // by the caller before its next entry-style safepoint, matching the
        // single-thread allocation discipline.
        f(unsafe { singleton_mut() })
    };
    result
}

/// Execute a closure with `&dyn GcAllocator` access (read-only query).
/// Same fast/slow path as `gc_op`.
#[inline]
pub fn gc_query<R>(f: impl FnOnce(&dyn GcAllocator) -> R) -> R {
    gc_op(|gc| f(gc))
}

// ──────────────────────────────────────────────────────────────
// STW protocol
// ──────────────────────────────────────────────────────────────

/// RAII stop-the-world guard used by collection drivers.
///
/// Nested guards only raise STW_DEPTH; the outer guard owns the request,
/// RUNNING drain, and resume broadcast.
pub struct StwGuard {
    active: bool,
    owner: bool,
}

/// Quiesce every other registered mutator when the process-global GC is shared.
/// A registered collecting thread stays in RUNNING while collection executes.
pub fn quiesce_mutators() -> StwGuard {
    let ident = my_ident();
    if STW_OWNER.load(Ordering::Acquire) == ident {
        STW_DEPTH.fetch_add(1, Ordering::Relaxed);
        return StwGuard {
            active: true,
            owner: false,
        };
    }

    if !stw_required() {
        return StwGuard {
            active: false,
            owner: false,
        };
    }

    let mut state = GC_SYNC.quiesce.lock().unwrap();
    GC_SYNC.stw_requested.store(true, Ordering::Release);
    majit_ir::eval_breaker_word::set_stw();

    let collector_is_running =
        GC_THREAD.with(|t| t.registered.get()) && GC_THREAD.with(|t| t.running.get());
    let drain_target = usize::from(collector_is_running);
    state = GC_SYNC
        .quiesced
        .wait_while(state, |state| state.running != drain_target)
        .unwrap();
    drop(state);
    STW_OWNER.store(ident, Ordering::Release);
    STW_DEPTH.store(1, Ordering::Relaxed);

    StwGuard {
        active: true,
        owner: true,
    }
}

/// Whether this thread currently owns or is nested inside collector-side STW.
#[inline]
pub fn mutators_quiesced() -> bool {
    STW_OWNER.load(Ordering::Acquire) == my_ident()
}

impl Drop for StwGuard {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        let old = STW_DEPTH.fetch_sub(1, Ordering::Relaxed);
        assert!(old > 0, "STW_DEPTH underflow");
        let remaining = old - 1;
        if !self.owner {
            return;
        }
        assert_eq!(remaining, 0, "outer STW guard dropped before nested guard");
        STW_OWNER.store(0, Ordering::Release);

        let _state = GC_SYNC.quiesce.lock().unwrap();
        GC_SYNC.stw_requested.store(false, Ordering::Release);
        majit_ir::eval_breaker_word::clear_stw();
        GC_SYNC.stw_generation.fetch_add(1, Ordering::Release);
        GC_SYNC.resumed.notify_all();
    }
}

/// Request stop-the-world collection. The calling thread becomes the
/// collector: it waits for all other threads to park, runs `collect_fn`
/// with exclusive GC access, then resumes everyone.
///
/// `collect_fn` receives `&mut dyn GcAllocator` — it can call
/// `collect_nursery`, `collect_full`, etc.
pub fn request_stw(collect_fn: impl FnOnce(&mut dyn GcAllocator)) {
    gc_op(|gc| {
        let _stw = quiesce_mutators();
        collect_fn(gc);
    });
}

/// Park the current thread until the ongoing STW finishes.
fn park_until_stw_done() {
    if !GC_THREAD.with(|t| t.registered.get()) || !GC_THREAD.with(|t| t.running.get()) {
        return;
    }

    let mut state = GC_SYNC.quiesce.lock().unwrap();
    if !GC_SYNC.stw_requested.load(Ordering::Acquire) {
        return;
    }
    assert!(
        GC_THREAD.with(|t| t.running.replace(false)),
        "GC mutator entered a dispatch safepoint twice"
    );
    state.running = state
        .running
        .checked_sub(1)
        .expect("RUNNING underflow entering dispatch safepoint");
    GC_SYNC.quiesced.notify_all();

    state = GC_SYNC
        .resumed
        .wait_while(state, |_| GC_SYNC.stw_requested.load(Ordering::Acquire))
        .unwrap();
    state.running += 1;
    assert!(
        !GC_THREAD.with(|t| t.running.replace(true)),
        "GC mutator left a dispatch safepoint twice"
    );
}

/// Poll for a collector request at a runtime dispatch safepoint.
/// Steady state is one relaxed atomic load.
#[inline]
pub fn safepoint_poll() {
    if GC_SYNC.stw_requested.load(Ordering::Relaxed) {
        park_until_stw_done();
    }
}

// ──────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collector::{GcConfig, MiniMarkGC};
    use crate::trace::TypeInfo;
    use std::cell::UnsafeCell;
    use std::sync::atomic::AtomicUsize;
    use std::sync::{Arc, Barrier, mpsc};
    use std::time::Duration;

    struct GcOpCounter(UnsafeCell<usize>);

    // SAFETY: tests access the cell only inside gc_op, whose singleton gate
    // must provide the same exclusive serialization as it does for the GC.
    unsafe impl Sync for GcOpCounter {}

    fn ensure_gc() {
        if !is_initialized() {
            let gc = Box::new(MiniMarkGC::new());
            store_singleton(gc);
        }
    }

    fn register_test_mutator() {
        crate::shadow_stack::register_mutator();
        register_thread();
    }

    fn unregister_test_mutator() {
        crate::shadow_stack::unregister_mutator();
        unregister_thread();
    }

    fn load_eval_breaker_word() -> usize {
        let addr = majit_ir::eval_breaker_word::eval_breaker_word_addr();
        assert_ne!(addr, 0);
        unsafe { &*(addr as *const AtomicUsize) }.load(Ordering::Relaxed)
    }

    #[test]
    fn gc_op_basic() {
        ensure_gc();
        register_test_mutator();
        let result = gc_op(|gc| gc.nursery_free());
        assert!(!result.is_null());
        unregister_test_mutator();
    }

    #[test]
    #[ignore = "requires exclusive process — quiesces every mutator and drives process-global STW state"]
    fn eval_breaker_word_parks_and_resumes_mutator() {
        majit_ir::eval_breaker_word::clear_async();
        majit_ir::eval_breaker_word::clear_stw();
        majit_ir::eval_breaker_word::publish_addr();

        let observed_poll = Arc::new(AtomicBool::new(false));
        let worker_observed_poll = observed_poll.clone();
        let (ready_tx, ready_rx) = mpsc::channel();
        let (resumed_tx, resumed_rx) = mpsc::channel();
        let worker = std::thread::spawn(move || {
            register_thread();
            ready_tx.send(()).unwrap();
            while load_eval_breaker_word() & majit_ir::eval_breaker_word::EB_STW == 0 {
                std::hint::spin_loop();
            }
            worker_observed_poll.store(true, Ordering::Release);
            // The STW bit and the authoritative request occupy two locations, so
            // a single poll can observe the bit set yet the request not-yet-
            // visible and return without parking. Re-poll until the request is
            // released and the bit is observed cleared — the loop exit is the
            // "resumed mutator sees bit1 cleared" assertion.
            loop {
                safepoint_poll();
                if load_eval_breaker_word() & majit_ir::eval_breaker_word::EB_STW == 0 {
                    break;
                }
            }
            resumed_tx.send(()).unwrap();
            unregister_thread();
        });

        ready_rx.recv().unwrap();
        let stw = quiesce_mutators();
        assert!(
            observed_poll.load(Ordering::Acquire),
            "the bitmask poll must lead the worker into the park gate"
        );
        assert_ne!(
            load_eval_breaker_word() & majit_ir::eval_breaker_word::EB_STW,
            0,
            "bit1 must remain armed throughout the STW episode"
        );
        drop(stw);
        assert_eq!(
            load_eval_breaker_word() & majit_ir::eval_breaker_word::EB_STW,
            0,
            "bit1 must balance to zero before mutators resume"
        );
        resumed_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("the parked mutator must resume cleanly");
        worker.join().unwrap();
        assert_eq!(
            load_eval_breaker_word(),
            0,
            "the eval-breaker word must be balanced after the STW episode"
        );
    }

    #[test]
    fn nested_reentrant_query_inside_gc_op_reads_singleton() {
        ensure_gc();
        register_thread();
        // The reentrant query reads the singleton directly instead of
        // re-entering the operation gate.
        let ok = gc_op(|_outer| gc_query_reentrant(|gc| !gc.nursery_free().is_null()));
        assert!(ok);
        unregister_thread();
    }

    #[test]
    #[ignore = "requires exclusive process — exercises process-global registration state"]
    fn registered_and_unregistered_gc_ops_are_mutually_exclusive() {
        ensure_gc();
        register_test_mutator();

        const OPS_PER_THREAD: usize = 100_000;
        let counter = Arc::new(GcOpCounter(UnsafeCell::new(0)));
        let start = Arc::new(Barrier::new(2));
        let worker = {
            let counter = counter.clone();
            let start = start.clone();
            std::thread::spawn(move || {
                // An unregistered thread's gc_op is legal and must serialize
                // through the same fast-path lock as a registered caller.
                start.wait();
                for _ in 0..OPS_PER_THREAD {
                    gc_op(|_| unsafe { *counter.0.get() += 1 });
                }
            })
        };

        start.wait();
        for _ in 0..OPS_PER_THREAD {
            gc_op(|_| unsafe { *counter.0.get() += 1 });
        }
        worker.join().unwrap();

        assert_eq!(unsafe { *counter.0.get() }, OPS_PER_THREAD * 2);
        unregister_test_mutator();
    }

    #[test]
    #[ignore = "requires exclusive process — mutates the sticky singleton inline-allocation state"]
    fn second_registered_thread_pins_published_nursery_top() {
        ensure_gc();
        register_test_mutator();
        assert_ne!(
            gc_query(
                |gc| unsafe { &*(gc.nursery_top_addr() as *const AtomicUsize) }
                    .load(Ordering::Acquire)
            ),
            0
        );

        std::thread::spawn(|| {
            register_test_mutator();
            unregister_test_mutator();
        })
        .join()
        .unwrap();

        assert_eq!(
            gc_query(
                |gc| unsafe { &*(gc.nursery_top_addr() as *const AtomicUsize) }
                    .load(Ordering::Acquire)
            ),
            0
        );
        unregister_test_mutator();
    }

    #[test]
    #[ignore = "requires exclusive process — conflicts with other majit-gc tests' local GCs"]
    fn two_threads_alloc_no_race() {
        ensure_gc();
        register_test_mutator();

        let counter = Arc::new(AtomicUsize::new(0));
        let barrier = Arc::new(Barrier::new(2));

        let handles: Vec<_> = (0..2)
            .map(|_| {
                let c = counter.clone();
                let b = barrier.clone();
                std::thread::spawn(move || {
                    register_test_mutator();
                    b.wait();
                    for _ in 0..100 {
                        gc_op(|_gc| {
                            // Simulate work under GC lock
                            let v = c.load(Ordering::Relaxed);
                            c.store(v + 1, Ordering::Relaxed);
                        });
                    }
                    unregister_test_mutator();
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        // With gc_mutex serialisation, counter should be exactly 200.
        assert_eq!(counter.load(Ordering::Relaxed), 200);
        unregister_test_mutator();
    }

    #[test]
    #[ignore = "requires exclusive process — conflicts with other majit-gc tests' local GCs"]
    fn stw_blocks_concurrent_gc_ops() {
        ensure_gc();
        register_test_mutator();

        let stw_ran = Arc::new(AtomicBool::new(false));
        let stw_ran2 = stw_ran.clone();

        // Spawn a thread that will try gc_op while STW is in progress.
        let barrier = Arc::new(Barrier::new(2));
        let b2 = barrier.clone();

        let worker = std::thread::spawn(move || {
            register_test_mutator();
            b2.wait();
            while !GC_SYNC.stw_requested.load(Ordering::Acquire) {
                std::hint::spin_loop();
            }
            // This gc_op should block until STW finishes.
            gc_op(|_gc| {
                assert!(
                    stw_ran2.load(Ordering::Acquire),
                    "gc_op should only run after STW completes"
                );
            });
            unregister_test_mutator();
        });

        barrier.wait();
        request_stw(|_gc| {
            stw_ran.store(true, Ordering::Release);
            // Simulate collection work.
            std::thread::sleep(std::time::Duration::from_millis(20));
        });

        worker.join().unwrap();
        unregister_test_mutator();
    }

    #[test]
    #[ignore = "requires exclusive process — conflicts with other majit-gc tests' local GCs"]
    fn entry_only_safepoint_preserves_fresh_gc_op_return() {
        if !is_initialized() {
            let mut gc = MiniMarkGC::with_config(GcConfig {
                nursery_size: 64 * 1024,
                large_object_threshold: 32 * 1024,
                ..GcConfig::default()
            });
            let type_id = gc.register_type(TypeInfo::simple(16));
            assert_eq!(type_id, 0);
            store_singleton(Box::new(gc));
        }

        const MUTATORS: usize = 4;
        const ROUNDS: usize = 128;
        let start = Arc::new(Barrier::new(MUTATORS + 1));
        let finished = Arc::new(AtomicUsize::new(0));

        let collector = {
            let start = start.clone();
            let finished = finished.clone();
            std::thread::spawn(move || {
                register_test_mutator();
                start.wait();
                while finished.load(Ordering::Acquire) != MUTATORS {
                    gc_op(|gc| gc.collect_nursery());
                    std::thread::yield_now();
                }
                unregister_test_mutator();
            })
        };

        let mutators: Vec<_> = (0..MUTATORS)
            .map(|thread_index| {
                let start = start.clone();
                let finished = finished.clone();
                std::thread::spawn(move || {
                    register_test_mutator();
                    start.wait();

                    for round in 0..ROUNDS {
                        let expected =
                            0xA110_C000_0000_0000u64 | ((thread_index as u64) << 32) | round as u64;
                        let fresh = gc_op(|gc| gc.alloc_nursery_typed(0, 16));
                        unsafe { *(fresh.0 as *mut u64) = expected };

                        // Widen the allocation-return window. Collection must
                        // wait for the next entry safepoint, after this fresh
                        // reference has been installed in the shadow stack.
                        std::thread::yield_now();
                        let root_depth = crate::shadow_stack::push(fresh);
                        gc_op(|gc| {
                            gc.collect_nursery();
                            let rooted = crate::shadow_stack::get(root_depth);
                            assert_eq!(unsafe { *(rooted.0 as *const u64) }, expected);
                        });
                        crate::shadow_stack::pop_to(root_depth);
                    }

                    finished.fetch_add(1, Ordering::Release);
                    unregister_test_mutator();
                })
            })
            .collect();

        for mutator in mutators {
            mutator.join().unwrap();
        }
        collector.join().unwrap();
        assert_eq!(registered_threads(), 0);
    }

    #[test]
    #[ignore = "requires exclusive process — conflicts with other majit-gc tests' local GCs"]
    fn multithreaded_collections_preserve_each_mutators_roots() {
        if !is_initialized() {
            let mut gc = MiniMarkGC::with_config(GcConfig {
                nursery_size: 64 * 1024,
                large_object_threshold: 32 * 1024,
                ..GcConfig::default()
            });
            let type_id = gc.register_type(TypeInfo::simple(16));
            assert_eq!(type_id, 0);
            store_singleton(Box::new(gc));
        }

        const THREADS: usize = 4;
        const ROUNDS: usize = 32;
        const ALLOCS_PER_ROUND: usize = 40;
        let start = Arc::new(Barrier::new(THREADS));
        let handles: Vec<_> = (0..THREADS)
            .map(|thread_index| {
                let start = start.clone();
                std::thread::spawn(move || {
                    register_test_mutator();
                    start.wait();

                    let expected = 0xCAFE_0000_0000_0000u64 | thread_index as u64;
                    let root_depth = gc_op(|gc| {
                        let object = gc.alloc_nursery_typed(0, 16);
                        unsafe { *(object.0 as *mut u64) = expected };
                        crate::shadow_stack::push(object)
                    });

                    for _ in 0..ROUNDS {
                        gc_op(|gc| {
                            let object = crate::shadow_stack::get(root_depth);
                            assert_eq!(unsafe { *(object.0 as *const u64) }, expected);
                            for _ in 0..ALLOCS_PER_ROUND {
                                let junk = gc.alloc_nursery_typed(0, 2048);
                                unsafe { *(junk.0 as *mut u64) = 0xBAD0_BAD0_BAD0_BAD0 };
                            }
                            gc.collect_nursery();
                        });

                        gc_op(|_gc| {
                            let object = crate::shadow_stack::get(root_depth);
                            assert_eq!(unsafe { *(object.0 as *const u64) }, expected);
                        });
                    }

                    crate::shadow_stack::pop_to(root_depth);
                    unregister_test_mutator();
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }
        assert_eq!(registered_threads(), 0);

        register_test_mutator();
        let minor_collections = gc_op(|gc| gc.collection_counts().0);
        unregister_test_mutator();
        assert!(minor_collections >= THREADS * ROUNDS);
    }

    #[test]
    #[ignore = "requires exclusive process — conflicts with other majit-gc tests' local GCs"]
    fn oldgen_nonmoving_preserves_other_mutators_roots() {
        if !is_initialized() {
            let mut gc = MiniMarkGC::with_config(GcConfig {
                nursery_size: 64 * 1024,
                large_object_threshold: 32 * 1024,
                ..GcConfig::default()
            });
            let type_id = gc.register_type(TypeInfo::simple(16));
            assert_eq!(type_id, 0);
            store_singleton(Box::new(gc));
        }

        const EXPECTED: u64 = 0xCAFE_D00D_F00D_BAAD;
        let ready = Arc::new(AtomicBool::new(false));
        let done = Arc::new(AtomicBool::new(false));

        register_test_mutator();
        let worker = {
            let ready = ready.clone();
            let done = done.clone();
            std::thread::spawn(move || {
                register_test_mutator();
                let root_depth = gc_op(|gc| {
                    let object = gc.alloc_nursery_typed(0, 16);
                    unsafe { *(object.0 as *mut u64) = EXPECTED };
                    crate::shadow_stack::push(object)
                });
                ready.store(true, Ordering::Release);

                for _ in 0..64 {
                    gc_op(|_gc| {
                        let object = crate::shadow_stack::get(root_depth);
                        assert_eq!(unsafe { *(object.0 as *const u64) }, EXPECTED);
                    });
                    std::thread::yield_now();
                }
                while !done.load(Ordering::Acquire) {
                    gc_op(|_gc| {
                        let object = crate::shadow_stack::get(root_depth);
                        assert_eq!(unsafe { *(object.0 as *const u64) }, EXPECTED);
                    });
                    std::thread::yield_now();
                }
                gc_op(|_gc| {
                    let object = crate::shadow_stack::get(root_depth);
                    assert_eq!(unsafe { *(object.0 as *const u64) }, EXPECTED);
                });

                crate::shadow_stack::pop_to(root_depth);
                unregister_test_mutator();
            })
        };

        while !ready.load(Ordering::Acquire) {
            std::thread::yield_now();
        }
        for _ in 0..3 {
            gc_op(|gc| {
                for _ in 0..40 {
                    let junk = gc.alloc_nursery_typed(0, 2048);
                    unsafe { *(junk.0 as *mut u64) = 0xBAD0_BAD0_BAD0_BAD0 };
                }
                gc.collect_nursery();
            });
        }
        for _ in 0..3 {
            gc_op(|gc| gc.collect_oldgen_nonmoving());
        }

        done.store(true, Ordering::Release);
        worker.join().unwrap();
        unregister_test_mutator();
        assert_eq!(registered_threads(), 0);
    }
}
