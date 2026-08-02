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
//! 2. [`MUTEX_GIL`], a regular mutex which the fast path never unlocks.
//!
//! Releasing is a plain store of 0; acquiring is a compare-and-swap against 0.
//! Whoever loses the compare-and-swap enters [`acquire_slow_path`], becomes
//! "the stealer" by taking [`MUTEX_GIL_STEALER`], and alternates between
//! retrying the fast path and a timed wait on [`MUTEX_GIL`].
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
//! bytecode, `GILReleaseAction` (gil.py:44-50) yields it from the periodic
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

/// thread_gil.c:89-90.
static MUTEX_GIL_STEALER: Mutex<()> = Mutex::new(());
static MUTEX_GIL: Mutex2 = Mutex2::new_locked();

// ──────────────────────────────────────────────────────────────
// rgil.py surface
// ──────────────────────────────────────────────────────────────

/// rgil.py:161-167 `allocate` / thread_gil.c:100-109 `RPyGilAllocate`.
///
/// The mutexes are const-initialised as statics, so all that is left of
/// `rpy_init_mutexes` is publishing that the GIL may now be waited on. Until
/// then [`acquire_slow_path`] is a fatal error, exactly as upstream: a program
/// which never set threads up can only ever take the compare-and-swap.
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
pub(crate) fn after_fork_child() {
    *MUTEX_GIL.locked.lock().unwrap() = true;
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

/// rgil.py:186-193 `acquire_maybe_in_new_thread`, the acquire used by a thread
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
/// A plain store: the word is only ever cleared by its owner. `RPyGilReleaseSignal`
/// (thread_gil.c:258-276) has no counterpart because its body is `_WIN32`-only.
#[inline]
pub fn release() {
    debug_assert!(
        am_i_holding_the_gil(),
        "releasing the GIL without holding it"
    );
    RPY_FASTGIL.store(0, Ordering::Release);
}

/// threadlocal.h:170-172 `_RPyGilGetHolder`.
#[inline]
pub fn gil_get_holder() -> usize {
    RPY_FASTGIL.load(Ordering::Relaxed)
}

/// rgil.py:236-239 `am_I_holding_the_GIL`, the `rpy_fastgil == get_ident()`
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
            MUTEX_GIL.unlock();
            break;
        }
    }

    // Now we are in point (3): mutex_gil might be released, but rpy_fastgil
    // might still contain an arbitrary ident.
    //
    // Enter the waiting queue from the end. Assuming a roughly
    // first-in-first-out order, this gives the threads a round-robin chance.
    {
        let _stealer = MUTEX_GIL_STEALER.lock().unwrap();
        let mut gil = MUTEX_GIL.loop_start();

        // We are now the stealer thread. Steals!
        loop {
            // Busy-looping here. Look again whether `rpy_fastgil` is released.
            if !fastgil_locked() && acquire_fast_path(ident) {
                // point (8.A)
                break;
            }
            // Sleep for one interval of time. We may be woken up earlier if
            // `mutex_gil` is released. Point (8.B).
            let (guard, relocked) = MUTEX_GIL.lock_timeout(gil, MUTEX_GIL_POLL_INTERVAL);
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
    debug_assert!(fastgil_locked());
}

/// thread_gil.c:235-256 `RPyGilYieldThread`, driven by the periodic
/// `GILReleaseAction` (gil.py:44-50). Returns whether the GIL was actually
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
    MUTEX_GIL.unlock();
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

    #[test]
    fn fast_path_round_trip_leaves_the_word_free() {
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
        allocate();
        let _guard = GilGuard::acquire();
        assert!(!yield_thread());
        assert!(am_i_holding_the_gil());
    }
}
