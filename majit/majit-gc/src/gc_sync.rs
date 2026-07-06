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

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex, MutexGuard, OnceLock};

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
    /// at their next poll point (= every gc_op call in P0).
    stw_requested: AtomicBool,
    /// Number of threads currently inside a gc_op (between gc_mutex lock
    /// and unlock). The collector waits for this to reach 0.
    /// Note: in P0, a thread inside gc_op holds gc_mutex, so at most 1
    /// thread is inside at a time. This counter exists for P1 (TLAB)
    /// where alloc won't hold the mutex.
    active_in_gc_op: AtomicUsize,
    /// Signalled when the last active thread exits gc_op during STW.
    all_parked: Condvar,
    /// Signalled when STW ends (collector finished, mutators can resume).
    stw_done: Condvar,
    /// Generation counter incremented after each STW. Prevents spurious
    /// wake: a thread checks that generation changed before proceeding.
    stw_generation: AtomicUsize,
}

static GC_SYNC: GcSync = GcSync {
    gc_mutex: Mutex::new(()),
    stw_requested: AtomicBool::new(false),
    active_in_gc_op: AtomicUsize::new(0),
    all_parked: Condvar::new(),
    stw_done: Condvar::new(),
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

/// Check if the GC singleton has been initialized.
pub fn is_initialized() -> bool {
    GC_INITIALIZED.load(Ordering::Acquire)
}

/// Access the GC singleton mutably under gc_mutex protection.
/// SAFETY: caller must hold gc_mutex.
unsafe fn singleton_mut() -> &'static mut dyn GcAllocator {
    (*GC_STORE.0.get())
        .as_deref_mut()
        .expect("GC singleton not initialized — call store_singleton() first")
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
    let old = REGISTERED_THREADS.fetch_add(1, Ordering::SeqCst);
    if old > 0 {
        // A second thread is arriving.  Spin until any in-progress
        // fast-path gc_op completes — after this, the first thread
        // will see REGISTERED_THREADS > 1 and take the Mutex path.
        while IN_FAST_PATH.load(Ordering::Acquire) {
            std::hint::spin_loop();
        }
    }
}

/// Unregister the current thread.  After this, gc_op must not be
/// called from this thread.
pub fn unregister_thread() {
    REGISTERED_THREADS.fetch_sub(1, Ordering::SeqCst);
}

// ──────────────────────────────────────────────────────────────
// GC operation gate — fast path when single-threaded
// ──────────────────────────────────────────────────────────────

/// Execute a closure with exclusive `&mut dyn GcAllocator` access.
///
/// **Fast path** (single registered thread, no STW): direct access,
/// no Mutex.  Cost: 2 atomic loads + 2 atomic stores (~4ns x86).
///
/// **Slow path** (multiple threads or STW): acquires `gc_mutex`.
/// Single-threaded production always takes the fast path.
#[inline]
pub fn gc_op<R>(f: impl FnOnce(&mut dyn GcAllocator) -> R) -> R {
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
            let r = f(unsafe { singleton_mut() });
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
    if GC_SYNC.stw_requested.load(Ordering::Acquire) {
        park_until_stw_done();
    }
    let _guard = GC_SYNC.gc_mutex.lock().unwrap();
    if GC_SYNC.stw_requested.load(Ordering::Acquire) {
        drop(_guard);
        park_until_stw_done();
        let _guard = GC_SYNC.gc_mutex.lock().unwrap();
        return f(unsafe { singleton_mut() });
    }
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

/// Request stop-the-world collection. The calling thread becomes the
/// collector: it waits for all other threads to park, runs `collect_fn`
/// with exclusive GC access, then resumes everyone.
///
/// `collect_fn` receives `&mut dyn GcAllocator` — it can call
/// `collect_nursery`, `collect_full`, etc.
pub fn request_stw(collect_fn: impl FnOnce(&mut dyn GcAllocator)) {
    GC_SYNC.stw_requested.store(true, Ordering::Release);

    // Wait for any in-progress fast-path gc_op to finish.
    while IN_FAST_PATH.load(Ordering::Acquire) {
        std::hint::spin_loop();
    }

    // Acquire gc_mutex — ensures no thread is mid-slow-path gc_op.
    let guard = GC_SYNC.gc_mutex.lock().unwrap();

    // SAFETY: gc_mutex held + fast path blocked by stw_requested +
    // IN_FAST_PATH drained. No concurrent singleton access.
    collect_fn(unsafe { singleton_mut() });

    // Resume all parked threads.
    GC_SYNC.stw_requested.store(false, Ordering::Release);
    GC_SYNC.stw_generation.fetch_add(1, Ordering::Release);
    GC_SYNC.stw_done.notify_all();
    drop(guard);
}

/// Park the current thread until the ongoing STW finishes.
fn park_until_stw_done() {
    let gen_before = GC_SYNC.stw_generation.load(Ordering::Acquire);
    let guard = GC_SYNC.gc_mutex.lock().unwrap();
    // Re-check: STW may have finished between our check and lock.
    if !GC_SYNC.stw_requested.load(Ordering::Acquire) {
        drop(guard);
        return;
    }
    // Wait until generation advances (STW completed).
    let _guard = GC_SYNC
        .stw_done
        .wait_while(guard, |_| {
            GC_SYNC.stw_generation.load(Ordering::Acquire) == gen_before
                && GC_SYNC.stw_requested.load(Ordering::Acquire)
        })
        .unwrap();
}

// ──────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collector::MiniMarkGC;
    use std::sync::atomic::AtomicUsize;
    use std::sync::{Arc, Barrier};

    fn ensure_gc() {
        if !is_initialized() {
            let gc = Box::new(MiniMarkGC::new());
            store_singleton(gc);
        }
        register_thread();
    }

    #[test]
    fn gc_op_basic() {
        ensure_gc();
        let result = gc_op(|gc| gc.nursery_free());
        assert!(!result.is_null());
    }

    #[test]
    #[ignore = "requires exclusive process — conflicts with other majit-gc tests' local GCs"]
    fn two_threads_alloc_no_race() {
        ensure_gc();

        let counter = Arc::new(AtomicUsize::new(0));
        let barrier = Arc::new(Barrier::new(2));

        let handles: Vec<_> = (0..2)
            .map(|_| {
                let c = counter.clone();
                let b = barrier.clone();
                std::thread::spawn(move || {
                    register_thread();
                    b.wait();
                    for _ in 0..100 {
                        gc_op(|_gc| {
                            // Simulate work under GC lock
                            let v = c.load(Ordering::Relaxed);
                            c.store(v + 1, Ordering::Relaxed);
                        });
                    }
                    unregister_thread();
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        // With gc_mutex serialisation, counter should be exactly 200.
        assert_eq!(counter.load(Ordering::Relaxed), 200);
    }

    #[test]
    #[ignore = "requires exclusive process — conflicts with other majit-gc tests' local GCs"]
    fn stw_blocks_concurrent_gc_ops() {
        ensure_gc();

        let stw_ran = Arc::new(AtomicBool::new(false));
        let stw_ran2 = stw_ran.clone();

        // Spawn a thread that will try gc_op while STW is in progress.
        let barrier = Arc::new(Barrier::new(2));
        let b2 = barrier.clone();

        let worker = std::thread::spawn(move || {
            b2.wait();
            // This gc_op should block until STW finishes.
            gc_op(|_gc| {
                assert!(
                    stw_ran2.load(Ordering::Acquire),
                    "gc_op should only run after STW completes"
                );
            });
        });

        barrier.wait();
        // Small delay to let worker reach gc_op.
        std::thread::sleep(std::time::Duration::from_millis(10));

        request_stw(|_gc| {
            stw_ran.store(true, Ordering::Release);
            // Simulate collection work.
            std::thread::sleep(std::time::Duration::from_millis(20));
        });

        worker.join().unwrap();
    }
}
