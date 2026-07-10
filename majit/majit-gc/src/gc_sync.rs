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
//! The STW protocol is for collection: when nursery is full, the collecting
//! thread requests STW, all other mutators park at their next poll point
//! (which is every GC operation in P0), collection runs, then all resume.

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
    /// Set to true when a thread wants to collect. Other threads park
    /// at their next gc_op or runtime dispatch poll.
    stw_requested: AtomicBool,
    /// RUNNING mutator count. A registered thread is removed before it waits
    /// on gc_mutex, so a collector holding gc_mutex can drain this to zero.
    quiesce: Mutex<QuiesceState>,
    /// Signalled whenever RUNNING decreases, including the transition to zero.
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

thread_local! {
    /// > 0 while this thread already holds exclusive GC access via a
    /// slow-path `gc_op_slow` or a `request_stw` collection. Nested
    /// `gc_op`/`gc_query` on the same thread then run directly on the
    /// singleton instead of re-locking the non-reentrant `gc_mutex`.
    static GATE_HELD_DEPTH: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

struct GateHeldGuard;

impl GateHeldGuard {
    fn enter() -> Self {
        GATE_HELD_DEPTH.with(|c| c.set(c.get() + 1));
        GateHeldGuard
    }
}

impl Drop for GateHeldGuard {
    fn drop(&mut self) {
        GATE_HELD_DEPTH.with(|c| c.set(c.get() - 1));
    }
}

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

thread_local! {
    /// `> 0` while this thread holds an exclusive `&mut` to the singleton
    /// (inside a `gc_op` / `request_stw` closure). A collection fires *inside*
    /// one of those closures, and its extra-root walkers re-enter the GC with
    /// read-only ownership queries (`gc_owns_object` → `is_managed_heap_object`).
    /// `gc_query_reentrant` consults this so such a query reaches the singleton
    /// via a shared read instead of a second `gc_mutex` lock (deadlock) or a
    /// second `&mut` (aliasing).
    static GC_OP_DEPTH: Cell<u32> = const { Cell::new(0) };

    /// Whether this thread has completed `register_thread` and is represented
    /// in the RUNNING count.
    static THREAD_REGISTERED: Cell<bool> = const { Cell::new(false) };

    /// Registered mutators are normally RUNNING. Outermost slow gc_op regions
    /// and safepoint parks flip this to false until they resume.
    static THREAD_RUNNING: Cell<bool> = const { Cell::new(false) };

    /// Reentrancy depth for collector-side quiescence. do_collect_full calls
    /// do_collect_nursery, so only the outer guard owns the STW transition.
    static STW_DEPTH: Cell<u32> = const { Cell::new(0) };
}

/// RAII marker: raises `GC_OP_DEPTH` for the exact span of a closure that holds
/// the exclusive `&mut dyn GcAllocator`. Wraps every `singleton_mut()` call.
struct OpGuard;
impl OpGuard {
    #[inline]
    fn enter() -> Self {
        GC_OP_DEPTH.with(|d| d.set(d.get() + 1));
        OpGuard
    }
}
impl Drop for OpGuard {
    #[inline]
    fn drop(&mut self) {
        GC_OP_DEPTH.with(|d| d.set(d.get() - 1));
    }
}

/// Whether this thread is already inside a `gc_op` / `request_stw` closure
/// (i.e. a collection is running on this thread and holds the `&mut`).
#[inline]
pub fn in_gc_op() -> bool {
    GC_OP_DEPTH.with(|d| d.get() != 0)
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

/// Set by the single-thread fast path while inside `singleton_mut()`.
/// `register_thread` spins on this to prevent the 1→2 transition from
/// racing with a concurrent fast-path gc_op.
static IN_FAST_PATH: AtomicBool = AtomicBool::new(false);

/// Register the current thread as a GC mutator.  Must be called before
/// any `gc_op` on this thread.  Paired with `unregister_thread`.
pub fn register_thread() {
    assert!(
        !THREAD_REGISTERED.with(|registered| registered.get()),
        "GC mutator thread registered twice"
    );
    let old = REGISTERED_THREADS.fetch_add(1, Ordering::SeqCst);
    if old > 0 {
        // A second thread is arriving.  Spin until any in-progress
        // fast-path gc_op completes — after this, the first thread
        // will see REGISTERED_THREADS > 1 and take the Mutex path.
        while IN_FAST_PATH.load(Ordering::Acquire) {
            std::hint::spin_loop();
        }
    }

    let mut state = GC_SYNC.quiesce.lock().unwrap();
    state = GC_SYNC
        .resumed
        .wait_while(state, |_| GC_SYNC.stw_requested.load(Ordering::Acquire))
        .unwrap();
    state.running += 1;
    THREAD_RUNNING.with(|running| running.set(true));
    THREAD_REGISTERED.with(|registered| registered.set(true));
}

/// Unregister the current thread.  After this, gc_op must not be
/// called from this thread.
pub fn unregister_thread() {
    assert!(
        THREAD_REGISTERED.with(|registered| registered.get()),
        "unregistering an unregistered GC mutator thread"
    );
    let mut state = GC_SYNC.quiesce.lock().unwrap();
    assert!(
        THREAD_RUNNING.with(|running| running.replace(false)),
        "unregistering a parked GC mutator thread"
    );
    state.running = state
        .running
        .checked_sub(1)
        .expect("RUNNING underflow during unregister_thread");
    THREAD_REGISTERED.with(|registered| registered.set(false));
    let old = REGISTERED_THREADS.fetch_sub(1, Ordering::SeqCst);
    assert!(old > 0, "REGISTERED_THREADS underflow");
    GC_SYNC.quiesced.notify_all();
}

/// Number of registered GC mutators.
#[inline]
pub fn registered_threads() -> usize {
    REGISTERED_THREADS.load(Ordering::Acquire)
}

// ──────────────────────────────────────────────────────────────
// GC operation gate — fast path when single-threaded
// ──────────────────────────────────────────────────────────────

/// A registered mutator's outermost slow gc_op is a safe region for its
/// complete duration, including time blocked on gc_mutex.
struct SafeRegionGuard {
    active: bool,
}

impl SafeRegionGuard {
    fn enter() -> Self {
        let registered = THREAD_REGISTERED.with(|registered| registered.get());
        let needs_safe_region = registered
            && (REGISTERED_THREADS.load(Ordering::Acquire) > 1
                || GC_SYNC.stw_requested.load(Ordering::Acquire));
        if !needs_safe_region || in_gc_op() {
            return Self { active: false };
        }

        let mut state = GC_SYNC.quiesce.lock().unwrap();
        assert!(
            THREAD_RUNNING.with(|running| running.replace(false)),
            "GC mutator entered a safe region twice"
        );
        state.running = state
            .running
            .checked_sub(1)
            .expect("RUNNING underflow entering gc_op safe region");
        GC_SYNC.quiesced.notify_all();

        state = GC_SYNC
            .resumed
            .wait_while(state, |_| GC_SYNC.stw_requested.load(Ordering::Acquire))
            .unwrap();
        drop(state);
        Self { active: true }
    }
}

impl Drop for SafeRegionGuard {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        let mut state = GC_SYNC.quiesce.lock().unwrap();
        state = GC_SYNC
            .resumed
            .wait_while(state, |_| GC_SYNC.stw_requested.load(Ordering::Acquire))
            .unwrap();
        state.running += 1;
        assert!(
            !THREAD_RUNNING.with(|running| running.replace(true)),
            "GC mutator left a safe region twice"
        );
    }
}

/// Execute a closure with exclusive `&mut dyn GcAllocator` access.
///
/// **Fast path** (single registered thread, no STW): direct access,
/// no Mutex.  Cost: 2 atomic loads + 2 atomic stores (~4ns x86).
///
/// **Slow path** (multiple threads or STW): acquires `gc_mutex`.
/// Single-threaded production always takes the fast path.
#[inline]
pub fn gc_op<R>(f: impl FnOnce(&mut dyn GcAllocator) -> R) -> R {
    if GATE_HELD_DEPTH.with(|c| c.get()) > 0 {
        // This thread already holds exclusive GC access during collection.
        // SAFETY: gc_mutex is held by this thread, so there is no concurrent
        // access to the singleton.
        return f(unsafe { singleton_mut() });
    }

    debug_assert!(
        !in_gc_op(),
        "reentrant &mut gc_op — a collection-time query must use gc_query_reentrant"
    );
    // Fast path: single thread, no STW.
    if REGISTERED_THREADS.load(Ordering::Acquire) <= 1
        && !GC_SYNC.stw_requested.load(Ordering::Acquire)
    {
        IN_FAST_PATH.store(true, Ordering::Release);
        // Double-check: another thread may have registered between
        // our load and the flag set.
        if REGISTERED_THREADS.load(Ordering::Acquire) <= 1
            && !GC_SYNC.stw_requested.load(Ordering::Acquire)
        {
            // SAFETY: single thread, no concurrent access possible.
            let r = {
                let _op = OpGuard::enter();
                f(unsafe { singleton_mut() })
            };
            IN_FAST_PATH.store(false, Ordering::Release);
            return r;
        }
        IN_FAST_PATH.store(false, Ordering::Release);
    }
    gc_op_slow(f)
}

/// Slow path: Mutex-guarded access with STW parking.
#[cold]
fn gc_op_slow<R>(f: impl FnOnce(&mut dyn GcAllocator) -> R) -> R {
    // Enter the safe region before acquiring gc_mutex. A thread blocked on
    // the mutex is therefore already excluded from RUNNING, which lets the
    // collector hold gc_mutex while waiting for RUNNING == 0.
    let _safe = SafeRegionGuard::enter();
    let _guard = GC_SYNC.gc_mutex.lock().unwrap();
    let _held = GateHeldGuard::enter();
    let _op = OpGuard::enter();
    f(unsafe { singleton_mut() })
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
    restore_fast_collector: bool,
}

/// Quiesce every registered mutator when the process-global GC is shared.
/// The collecting thread is normally already safe because collection runs
/// inside an outermost slow gc_op.
pub fn quiesce_mutators() -> StwGuard {
    let nested = STW_DEPTH.with(|depth| {
        let old = depth.get();
        if old != 0 {
            depth.set(old + 1);
            true
        } else {
            false
        }
    });
    if nested {
        return StwGuard {
            active: true,
            owner: false,
            restore_fast_collector: false,
        };
    }

    if REGISTERED_THREADS.load(Ordering::Acquire) <= 1 {
        return StwGuard {
            active: false,
            owner: false,
            restore_fast_collector: false,
        };
    }

    let mut state = GC_SYNC.quiesce.lock().unwrap();
    GC_SYNC.stw_requested.store(true, Ordering::Release);

    // The 1→2 registration transition publishes REGISTERED_THREADS before it
    // waits for the old single-thread fast operation to finish. If that very
    // operation fills the nursery, make its collecting thread safe and release
    // the transition waiter; stw_requested prevents any new fast operation.
    let restore_fast_collector = IN_FAST_PATH.load(Ordering::Acquire)
        && in_gc_op()
        && THREAD_REGISTERED.with(|registered| registered.get())
        && THREAD_RUNNING.with(|running| running.get());
    if restore_fast_collector {
        THREAD_RUNNING.with(|running| running.set(false));
        state.running = state
            .running
            .checked_sub(1)
            .expect("RUNNING underflow quiescing fast-path collector");
        IN_FAST_PATH.store(false, Ordering::Release);
        GC_SYNC.quiesced.notify_all();
    }

    // Any other fast operation began before the multi-thread transition and
    // will finish without consulting the quiesce mutex.
    while IN_FAST_PATH.load(Ordering::Acquire) {
        std::hint::spin_loop();
    }

    state = GC_SYNC
        .quiesced
        .wait_while(state, |state| state.running != 0)
        .unwrap();
    drop(state);
    STW_DEPTH.with(|depth| depth.set(1));

    StwGuard {
        active: true,
        owner: true,
        restore_fast_collector,
    }
}

/// Whether this thread currently owns or is nested inside collector-side STW.
#[inline]
pub fn mutators_quiesced() -> bool {
    STW_DEPTH.with(|depth| depth.get() != 0)
}

impl Drop for StwGuard {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        let remaining = STW_DEPTH.with(|depth| {
            let old = depth.get();
            assert!(old > 0, "STW_DEPTH underflow");
            depth.set(old - 1);
            old - 1
        });
        if !self.owner {
            return;
        }
        assert_eq!(remaining, 0, "outer STW guard dropped before nested guard");

        let mut state = GC_SYNC.quiesce.lock().unwrap();
        if self.restore_fast_collector {
            state.running += 1;
            assert!(
                !THREAD_RUNNING.with(|running| running.replace(true)),
                "fast-path collector resumed twice"
            );
        }
        GC_SYNC.stw_requested.store(false, Ordering::Release);
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
    let safe = SafeRegionGuard::enter();
    drop(safe);
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
    use std::sync::atomic::AtomicUsize;
    use std::sync::{Arc, Barrier};

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

    #[test]
    fn gc_op_basic() {
        ensure_gc();
        register_test_mutator();
        let result = gc_op(|gc| gc.nursery_free());
        assert!(!result.is_null());
        unregister_test_mutator();
    }

    #[test]
    fn nested_gc_query_inside_slow_path_gc_op_does_not_deadlock() {
        ensure_gc();
        register_thread();
        // Outer gc_op holds gc_mutex; the nested gc_query must not re-lock it.
        let ok = gc_op(|_outer| gc_query(|gc| !gc.nursery_free().is_null()));
        assert!(ok);
        unregister_thread();
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
}
