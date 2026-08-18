//! The locks an extension holds -- `Include/cpython/lock.h` and
//! `Include/pythread.h`.
//!
//! A thread that blocks here may be asleep for as long as whoever holds the
//! lock runs, and the collector stops the world by a handshake every mutator
//! has to answer.  Every blocking wait below therefore runs inside
//! [`majit_gc::gc_sync::before_external_block`], which is the same region
//! `Py_BEGIN_ALLOW_THREADS` puts a thread in: the waiter gives the GIL up and
//! stops being a thread a collection waits for.

use std::ffi::{c_int, c_longlong, c_ulong, c_void};
use std::sync::atomic::{AtomicU8, Ordering};
use std::time::{Duration, Instant};

/// The one byte a `PyMutex` occupies, which the caller embeds and
/// zero-initializes.
///
/// Only the low bit is ever set here.  The header's inline fast paths take
/// and release the byte with a compare-exchange against exactly `0` and `1`,
/// so a state this does not write is a state they would never recognize.
#[repr(C)]
pub struct CPyMutex {
    pub bits: u8,
}

const UNLOCKED: u8 = 0;
const LOCKED: u8 = 1;

/// The byte read as the atomic every path below writes it as.
///
/// # Safety
/// `bits` must address one writable byte that outlives the returned reference.
unsafe fn atomic<'a>(bits: *mut u8) -> &'a AtomicU8 {
    unsafe { AtomicU8::from_ptr(bits) }
}

/// Take the byte if it is free, without blocking.
fn try_take(bits: &AtomicU8) -> bool {
    bits.compare_exchange(UNLOCKED, LOCKED, Ordering::Acquire, Ordering::Relaxed)
        .is_ok()
}

/// Take the byte, waiting until `deadline` -- or forever, for `None`.
///
/// The wait spins first, since an uncontended handoff is the common case and
/// a thread that yields immediately pays a context switch for nothing, then
/// backs off to sleeping so that a lock held for a long time costs a core
/// nothing.
fn take_blocking(bits: &AtomicU8, deadline: Option<Instant>) -> bool {
    let blocked = majit_gc::gc_sync::before_external_block();
    let mut spins = 0u32;
    let taken = loop {
        if try_take(bits) {
            break true;
        }
        if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
            break false;
        }
        spins = spins.saturating_add(1);
        if spins < 64 {
            std::hint::spin_loop();
        } else if spins < 256 {
            std::thread::yield_now();
        } else {
            std::thread::sleep(Duration::from_micros(50));
        }
    };
    drop(blocked);
    taken
}

// ── PyMutex ─────────────────────────────────────────────────────────────

/// `PyMutex_Lock(m)` -- the contended half of the header's inline fast path.
///
/// # Safety
/// `m` must address a `PyMutex` that outlives the matching unlock.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyMutex_Lock(m: *mut CPyMutex) {
    if m.is_null() {
        return;
    }
    let bits = unsafe { atomic(&raw mut (*m).bits) };
    if try_take(bits) {
        return;
    }
    take_blocking(bits, None);
}

/// `PyMutex_Unlock(m)` -- releasing a mutex nobody holds is a mistake there is
/// no way to report, so it ends the process the way upstream does.
///
/// # Safety
/// `m` must address a `PyMutex` this thread's caller holds.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyMutex_Unlock(m: *mut CPyMutex) {
    if m.is_null() {
        return;
    }
    let bits = unsafe { atomic(&raw mut (*m).bits) };
    if bits.swap(UNLOCKED, Ordering::Release) == UNLOCKED {
        super::pyerrors::fatal_error(Some("PyMutex_Unlock"), "unlocking mutex that is not locked");
    }
}

/// `PyMutex_IsLocked(m)`.
///
/// # Safety
/// `m` must address a live `PyMutex`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyMutex_IsLocked(m: *mut CPyMutex) -> c_int {
    if m.is_null() {
        return 0;
    }
    (unsafe { atomic(&raw mut (*m).bits) }.load(Ordering::Acquire) != UNLOCKED) as c_int
}

// ── PyThread locks ──────────────────────────────────────────────────────

/// The status codes `PyThread_acquire_lock_timed` answers with.
const PY_LOCK_FAILURE: c_int = 0;
const PY_LOCK_ACQUIRED: c_int = 1;

/// A `PyThread_type_lock` is an opaque pointer, so the byte lives in an
/// allocation of this layer's own rather than in the caller's storage.
///
/// Releasing is not tied to the thread that acquired: this is the binary
/// semaphore `threading.Lock` is built on, which one thread may hand to
/// another.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyThread_allocate_lock() -> *mut c_void {
    Box::into_raw(Box::new(AtomicU8::new(UNLOCKED))) as *mut c_void
}

/// # Safety
/// `lock` must be a lock [`PyThread_allocate_lock`] answered with, not yet
/// freed and not held.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyThread_free_lock(lock: *mut c_void) {
    if lock.is_null() {
        return;
    }
    drop(unsafe { Box::from_raw(lock as *mut AtomicU8) });
}

/// `PyThread_acquire_lock(lock, waitflag)` -- 1 when taken, 0 when not.
///
/// # Safety
/// `lock` must be a live lock [`PyThread_allocate_lock`] answered with.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyThread_acquire_lock(lock: *mut c_void, waitflag: c_int) -> c_int {
    let microseconds = if waitflag != 0 { -1 } else { 0 };
    unsafe { PyThread_acquire_lock_timed(lock, microseconds, 0) }
}

/// `PyThread_acquire_lock_timed(lock, microseconds, intr_flag)` -- 0 asks
/// without waiting, a negative count waits until it is taken, and a positive
/// one waits that long.
///
/// `intr_flag` has nothing to do here: the wait is not interruptible, so
/// `PY_LOCK_INTR` is never answered and a caller that loops on it simply
/// never goes round again.
///
/// # Safety
/// `lock` must be a live lock [`PyThread_allocate_lock`] answered with.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyThread_acquire_lock_timed(
    lock: *mut c_void,
    microseconds: c_longlong,
    _intr_flag: c_int,
) -> c_int {
    if lock.is_null() {
        return PY_LOCK_FAILURE;
    }
    let bits = unsafe { &*(lock as *const AtomicU8) };
    if try_take(bits) {
        return PY_LOCK_ACQUIRED;
    }
    if microseconds == 0 {
        return PY_LOCK_FAILURE;
    }
    let deadline = (microseconds > 0)
        .then(|| Instant::now().checked_add(Duration::from_micros(microseconds as u64)))
        .flatten();
    // A count so large it has no instant to name is a caller asking to wait,
    // which is what an unbounded wait already is.
    if take_blocking(bits, deadline) {
        PY_LOCK_ACQUIRED
    } else {
        PY_LOCK_FAILURE
    }
}

/// # Safety
/// `lock` must be a live lock [`PyThread_allocate_lock`] answered with.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyThread_release_lock(lock: *mut c_void) {
    if lock.is_null() {
        return;
    }
    unsafe { &*(lock as *const AtomicU8) }.store(UNLOCKED, Ordering::Release);
}

/// `PyThread_get_thread_ident()` -- the identity `threading.get_ident()`
/// answers with, which is what a C caller keying a table by thread wants.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyThread_get_thread_ident() -> c_ulong {
    crate::module::thread::current_ident() as c_ulong
}

pub(super) fn ensure_linked() {
    std::hint::black_box(PyMutex_Lock as *const ());
    std::hint::black_box(PyMutex_Unlock as *const ());
    std::hint::black_box(PyMutex_IsLocked as *const ());
    std::hint::black_box(PyThread_allocate_lock as *const ());
    std::hint::black_box(PyThread_free_lock as *const ());
    std::hint::black_box(PyThread_acquire_lock as *const ());
    std::hint::black_box(PyThread_acquire_lock_timed as *const ());
    std::hint::black_box(PyThread_release_lock as *const ());
    std::hint::black_box(PyThread_get_thread_ident as *const ());
}
