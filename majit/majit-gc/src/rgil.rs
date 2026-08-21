//! rgil — the GIL.
//!
//! Ported from thread_gil.c, its fast path in threadlocal.h:150-170, and the
//! rgil.py surface that drives it.
//!
//! # The composite lock (thread_gil.c:2-31)
//!
//! "The GIL" is two locks, and it is held when both are locked:
//!
//! 1. [`RPY_FASTGIL`], a plain word. 0 means unlocked; otherwise it holds the
//!    ident of the owning thread, so a thread checks whether it owns the GIL
//!    with `rpy_fastgil == get_ident()`.
//! 2. `mutex_gil`, a regular mutex which the fast path never unlocks.
//!
//! Releasing is a plain store of 0; acquiring is a compare-and-swap against 0.
//! Whoever loses the compare-and-swap enters [`acquire_slow_path`], becomes
//! "the stealer" by taking `mutex_gil_stealer`, and alternates between
//! retrying the fast path and a timed wait on `mutex_gil`.
//!
//! # Lifetime
//!
//! A thread holds the GIL for as long as it runs pyre code, not for the
//! duration of one GC operation: it is taken once when the thread starts and
//! whenever an external call returns, and dropped before every external call
//! (`rffi.aroundstate.before`/`.after`, spelled here as
//! `gc_sync::before_external_block`). Between two external calls arbitrarily
//! many allocations run with no synchronisation at all, which is what lets
//! `gc_sync::gc_op` be a bare borrow of the singleton.
//!
//! Because a thread can therefore hold the GIL while running a long stretch of
//! bytecode, `GILReleaseAction` (gil.py) yields it from the periodic
//! action; [`yield_thread`] is that yield.

use std::sync::atomic::{AtomicIsize, AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex, MutexGuard};
use std::time::Duration;

use crate::gc_sync;

/// thread_gil.c:86 `Signed rpy_fastgil`. 0 when the GIL is unlocked, otherwise
/// the ident of the thread holding it (point (3)).
static RPY_FASTGIL: AtomicUsize = AtomicUsize::new(0);

/// thread_gil.c:87 `rpy_waiting_threads`, the number of threads inside
/// [`acquire_slow_path`]. Negative until [`allocate`] has run.
static RPY_WAITING_THREADS: AtomicIsize = AtomicIsize::new(GIL_NOT_INITIALIZED);

/// thread_gil.c:87 spells the uninitialised marker -42 and asserts on it.
const GIL_NOT_INITIALIZED: isize = -42;

/// thread_gil.c:88 `rpy_early_poll_n`, the running seed for the randomised
/// early-poll count.
static RPY_EARLY_POLL_N: AtomicIsize = AtomicIsize::new(0);

/// thread_gil.c:111-112.
const RPY_GIL_POKE_MIN: isize = 40;
const RPY_GIL_POKE_MAX: isize = 400;

/// thread_gil.c:217 waits on `mutex_gil` in 0.1 ms intervals.
const MUTEX_GIL_POLL_INTERVAL: Duration = Duration::from_micros(100);

/// thread.h:39 `RPY_FASTGIL_LOCKED`.
#[inline]
fn fastgil_locked() -> bool {
    RPY_FASTGIL.load(Ordering::Relaxed) != 0
}

// ──────────────────────────────────────────────────────────────
// mutex1_t / mutex2_t (thread_pthread.c:547-595)
// ──────────────────────────────────────────────────────────────

/// thread_pthread.c:559-563 `mutex2_t`: a `locked` flag with the mutex and
/// condition variable that guard it. It is not a plain mutex because the
/// stealer must be able to wait on it with a timeout while another thread
/// unlocks it without ever having locked it in the fast path.
struct Mutex2 {
    locked: Mutex<bool>,
    cond: Condvar,
}

impl Mutex2 {
    /// thread_pthread.c:565-569 `mutex2_init_locked`.
    const fn new_locked() -> Self {
        Mutex2 {
            locked: Mutex::new(true),
            cond: Condvar::new(),
        }
    }

    /// thread_pthread.c:570-575 `mutex2_unlock`. The signal is sent after the
    /// mutex is dropped, as in the C.
    fn unlock(&self) {
        *self.locked.lock().unwrap() = false;
        self.cond.notify_one();
    }

    /// thread_win7.c:594-600 `mutex2_signal`: wake a stealer sleeping in
    /// [`Mutex2::lock_timeout`] without changing `locked`, so it can re-check
    /// `rpy_fastgil`. Safe without the mutex held, and a missed wakeup is
    /// benign because that wait has its own timeout.
    #[cfg(windows)]
    fn signal(&self) {
        self.cond.notify_one();
    }

    /// thread_pthread.c:576-578 `mutex2_loop_start`. The returned guard is the
    /// lock the stealer keeps for the whole steal loop, and is dropped in place
    /// of `mutex2_loop_stop` (:579-581).
    fn loop_start(&self) -> MutexGuard<'_, bool> {
        self.locked.lock().unwrap()
    }

    /// thread_pthread.c:582-595 `mutex2_lock_timeout`. Returns the guard along
    /// with whether the mutex was observed unlocked, i.e. whether this thread
    /// just relocked it.
    fn lock_timeout<'a>(
        &'a self,
        guard: MutexGuard<'a, bool>,
        delay: Duration,
    ) -> (MutexGuard<'a, bool>, bool) {
        let mut guard = if *guard {
            self.cond.wait_timeout(guard, delay).unwrap().0
        } else {
            guard
        };
        let result = !*guard;
        *guard = true;
        (guard, result)
    }
}

/// thread_gil.c:89-90. Held in one cell because `rpy_init_mutexes` re-creates
/// both of them together, and a `std::sync::Mutex` can only be reset by
/// overwriting it.
struct GilMutexes {
    stealer: Mutex<()>,
    gil: Mutex2,
}

impl GilMutexes {
    const fn new() -> Self {
        GilMutexes {
            stealer: Mutex::new(()),
            gil: Mutex2::new_locked(),
        }
    }
}

struct GilMutexCell(std::cell::UnsafeCell<GilMutexes>);

// SAFETY: the contents are `Sync`; the cell exists only so that
// `init_mutexes` can replace them in a forked child, where this thread is the
// only one alive.
unsafe impl Sync for GilMutexCell {}

static MUTEXES: GilMutexCell = GilMutexCell(std::cell::UnsafeCell::new(GilMutexes::new()));

#[inline]
fn mutexes() -> &'static GilMutexes {
    // SAFETY: only `init_mutexes` ever writes, and only in a forked child
    // before any other thread exists.
    unsafe { &*MUTEXES.0.get() }
}

/// thread_gil.c:92-98 `rpy_init_mutexes`.
///
/// `mutex1_init` / `mutex2_init_locked` are `pthread_mutex_init` calls, which
/// reset an already-locked mutex and abandon whatever state it held. The
/// overwrite is the same operation: the old value is not dropped, because a
/// thread which vanished in `fork()` may still "hold" it.
fn init_mutexes() {
    // SAFETY: reached only from the surviving thread of a `fork()`, which is
    // the child's only thread, so no reference into the old value is live.
    unsafe { std::ptr::write(MUTEXES.0.get(), GilMutexes::new()) };
}

// ──────────────────────────────────────────────────────────────
// rgil.py surface
// ──────────────────────────────────────────────────────────────

/// rgil.py `allocate` / thread_gil.c:100-109 `RPyGilAllocate`.
///
/// The mutexes are const-initialised, so all [`init_mutexes`] would do at
/// startup is rewrite them with the value they already hold; what is left is
/// publishing that the GIL may now be waited on. Until then
/// [`acquire_slow_path`] is a fatal error, exactly as upstream: a program which
/// never set threads up can only ever take the compare-and-swap.
pub fn allocate() {
    let _ = RPY_WAITING_THREADS.compare_exchange(
        GIL_NOT_INITIALIZED,
        0,
        Ordering::SeqCst,
        Ordering::Relaxed,
    );
}

/// `rpy_init_mutexes` re-runs through `pthread_atfork` (thread_gil.c:105-107).
/// Only the forking thread survives, so nothing can be waiting and the surviving
/// holder's claim, if any, is restored by the caller.
///
/// Both mutexes are re-created rather than only re-flagged: a thread which was
/// in [`acquire_slow_path`] when the parent forked holds `stealer` and the
/// `gil.locked` guard, and the child inherits them locked by a thread that no
/// longer exists.
pub(crate) fn after_fork_child() {
    debug_assert!(
        am_i_holding_the_gil(),
        "the forking thread must hold the GIL across fork()"
    );
    init_mutexes();
    RPY_WAITING_THREADS.store(0, Ordering::SeqCst);
}

/// threadlocal.h:152-156 `_rpygil_acquire_fast_path`.
#[inline]
fn acquire_fast_path(ident: usize) -> bool {
    RPY_FASTGIL
        .compare_exchange(0, ident, Ordering::Acquire, Ordering::Relaxed)
        .is_ok()
}

/// threadlocal.h:158-161 `_RPyGilAcquire`, called on return from an external
/// call and when a thread starts running pyre code.
#[inline]
pub fn acquire() {
    let ident = gc_sync::my_ident();
    if !acquire_fast_path(ident) {
        acquire_slow_path(ident);
    }
}

/// rgil.py `acquire_maybe_in_new_thread`, the acquire used by a thread
/// which has not run pyre code before. Reading [`gc_sync::my_ident`] is what
/// makes sure this thread's thread-locals exist, standing in for
/// `rthread.get_or_make_ident()`.
#[inline]
pub fn acquire_maybe_in_new_thread() {
    allocate();
    acquire();
}

/// threadlocal.h:162-166 `_RPyGilRelease`, called before an external call.
///
/// A plain store — the word is only ever cleared by its owner — followed by
/// [`release_signal`].
#[inline]
pub fn release() {
    debug_assert!(
        am_i_holding_the_gil(),
        "releasing the GIL without holding it"
    );
    RPY_FASTGIL.store(0, Ordering::Release);
    release_signal();
}

/// thread_gil.c:258-276 `RPyGilReleaseSignal`, whose body is `_WIN32`-only and
/// says why: a stealer asleep in `mutex2_lock_timeout` would otherwise have to
/// wait out its timed wait, and Windows' default 10-15 ms timer resolution
/// makes that "severe latency for any code that releases the GIL". On POSIX
/// the 0.1 ms timeout is short enough that the extra wakeup is not needed and
/// its scheduling interference hangs `test_interrupt_non_main`.
#[cfg(windows)]
#[inline]
fn release_signal() {
    if RPY_WAITING_THREADS.load(Ordering::Relaxed) > 0 {
        mutexes().gil.signal();
    }
}

#[cfg(not(windows))]
#[inline]
fn release_signal() {}

/// threadlocal.h:170-172 `_RPyGilGetHolder`.
#[inline]
pub fn gil_get_holder() -> usize {
    RPY_FASTGIL.load(Ordering::Relaxed)
}

/// rgil.py `am_I_holding_the_GIL`, the `rpy_fastgil == get_ident()`
/// idiom of thread_gil.c:19-21.
#[inline]
pub fn am_i_holding_the_gil() -> bool {
    gil_get_holder() == gc_sync::my_ident()
}

/// thread_gil.c:114-233 `RPyGilAcquireSlowPath`: another thread is busy with
/// the GIL.
///
/// A waiting thread runs no pyre code, so it leaves the RUNNING census for the
/// whole wait ([`gc_sync::leave_running_for_gil`]). Upstream needs no such step
/// — a collector there *is* the GIL holder, so anything waiting for the GIL is
/// by construction not running — but pyre's collector drains that census
/// explicitly and would otherwise wait for a thread that is itself waiting for
/// the collector.
#[cold]
fn acquire_slow_path(ident: usize) {
    assert!(
        RPY_WAITING_THREADS.load(Ordering::Relaxed) >= 0,
        "a thread is trying to wait for the GIL, but the GIL was not initialized"
    );

    let census = gc_sync::leave_running_for_gil();

    // Register me as one of the threads that is actively waiting for the GIL.
    let waiting_threads = RPY_WAITING_THREADS.fetch_add(1, Ordering::SeqCst) + 1;

    // Early polling (:144-190): check a bounded number of times whether the GIL
    // becomes free before entering the waiting queue, because there are use
    // cases where it is released very soon after this call. The count is
    // "randomised" between the two pokes to avoid falling into bad cases.
    let mut n = RPY_EARLY_POLL_N.load(Ordering::Relaxed) * 2 + 1;
    while n >= RPY_GIL_POKE_MAX {
        n -= RPY_GIL_POKE_MAX - RPY_GIL_POKE_MIN;
    }
    RPY_EARLY_POLL_N.store(n, Ordering::Relaxed);
    while n >= 0 {
        n -= 1;
        if waiting_threads != RPY_WAITING_THREADS.load(Ordering::Relaxed) {
            // The number changed because another thread entered or left this
            // function. If one left, the GIL has been acquired by it; if one
            // entered, running this loop twice is pointless.
            break;
        }
        std::hint::spin_loop();

        if !fastgil_locked() && acquire_fast_path(ident) {
            // We got the GIL before entering the waiting queue. Wake the
            // stealer thread anyway, for fairness, and go to the waiting queue
            // regardless — the loop below then relocks `mutex_gil` on its first
            // pass, restoring the invariant that the GIL's two locks are both
            // held. Leaving here instead would hand the next stealer a mutex it
            // can take while we still own the word.
            mutexes().gil.unlock();
            break;
        }
    }

    // Now we are in point (3): mutex_gil might be released, but rpy_fastgil
    // might still contain an arbitrary ident.
    //
    // Enter the waiting queue from the end. Assuming a roughly
    // first-in-first-out order, this gives the threads a round-robin chance.
    {
        let mutexes = mutexes();
        let _stealer = mutexes.stealer.lock().unwrap();
        let mut gil = mutexes.gil.loop_start();

        // We are now the stealer thread. Steals!
        loop {
            // Busy-looping here. Look again whether `rpy_fastgil` is released.
            if !fastgil_locked() && acquire_fast_path(ident) {
                // point (8.A)
                break;
            }
            // Sleep for one interval of time. We may be woken up earlier if
            // `mutex_gil` is released. Point (8.B).
            let (guard, relocked) = mutexes.gil.lock_timeout(gil, MUTEX_GIL_POLL_INTERVAL);
            gil = guard;
            if relocked {
                // `mutex_gil` was recently released and we just relocked it;
                // restore the invariant of point (3).
                RPY_FASTGIL.store(ident, Ordering::Release);
                break;
            }
        }
    }

    RPY_WAITING_THREADS.fetch_sub(1, Ordering::SeqCst);
    gc_sync::rejoin_running_after_gil(census);
    // Both exits from the steal loop leave *this* thread's ident in the word,
    // so the exit condition is ownership, not merely that the word is taken.
    debug_assert!(
        am_i_holding_the_gil(),
        "acquire_slow_path returned without owning rpy_fastgil"
    );
}

/// thread_gil.c:235-256 `RPyGilYieldThread`, driven by the periodic
/// `GILReleaseAction` (gil.py). Returns whether the GIL was actually
/// handed over.
///
/// Releasing `mutex_gil` leaves nobody holding the GIL — `rpy_fastgil` is still
/// locked, but the second lock is not — so the immediately following acquire
/// enqueues this thread at the end of the stealer queue behind the threads that
/// were already waiting.
pub fn yield_thread() -> bool {
    debug_assert!(am_i_holding_the_gil(), "yielding a GIL we do not hold");
    if RPY_WAITING_THREADS.load(Ordering::Relaxed) <= 0 {
        return false;
    }
    mutexes().gil.unlock();
    acquire();
    true
}

/// RAII bracket for code that enters pyre from outside — a thread that has to
/// take the GIL before it may touch the GC, and give it back afterwards.
/// `rffi`'s callback wrappers hold it the same way around the RPython side of a
/// call that arrived from C.
pub struct GilGuard {
    _not_send: std::marker::PhantomData<*const ()>,
}

impl GilGuard {
    pub fn acquire() -> Self {
        acquire_maybe_in_new_thread();
        GilGuard {
            _not_send: std::marker::PhantomData,
        }
    }
}

impl Drop for GilGuard {
    fn drop(&mut self) {
        release();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `RPY_FASTGIL` and `RPY_WAITING_THREADS` are process-global and the test
    /// harness runs test functions on concurrent threads, so a second test
    /// holding the GIL would be indistinguishable from a failure here.
    static TEST_SERIAL: Mutex<()> = Mutex::new(());

    #[test]
    fn fast_path_round_trip_leaves_the_word_free() {
        let _serial = TEST_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        allocate();
        let guard = GilGuard::acquire();
        assert!(am_i_holding_the_gil());
        assert_eq!(gil_get_holder(), gc_sync::my_ident());
        drop(guard);
        assert_eq!(gil_get_holder(), 0);
        assert!(!am_i_holding_the_gil());
    }

    #[test]
    fn yield_thread_without_waiters_keeps_the_gil() {
        let _serial = TEST_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        allocate();
        let _guard = GilGuard::acquire();
        assert!(!yield_thread());
        assert!(am_i_holding_the_gil());
    }
}
