//! C-level signal flag state — port of `rpython/translator/c/src/signals.c`.
//!
//! The OS signal handler runs in an async context where almost nothing
//! is safe to do, so it only flips atomic flags: it records the signal
//! number in `SIG_PENDING`, forces `SIG_TICKER` to -1 (so the next
//! `bytecode_trace` runs `action_dispatcher`), and writes one byte to
//! the wakeup fd.  The actual app-level handler is invoked later, from
//! `CheckSignalAction::perform`, at a safe interpreter checkpoint.
//!
//! `SIG_TICKER` is the analogue of C `pypysig_counter.inner.value`; the
//! `ActionFlag` ticker methods (`SignalActionFlag` in upstream) read and
//! write it directly so the handler and the eval loop share one cell.

use std::sync::atomic::{AtomicI32, AtomicI64, AtomicPtr, Ordering};

/// `signals.c:34` — reasonable default cap; the libc crate does not
/// surface `NSIG` portably.  64 fits a single `i64` bitmask.
pub const NSIG: i32 = 64;

/// Address of the `ActionFlag` ticker cell (`signals.c:45-49
/// pypysig_counter`).  The interpreter reads the cell directly as a
/// plain field in the per-bytecode hot path (an atomic / volatile global
/// read there is not modellable by the JIT codewriter); the OS handler
/// stores -1 through this registered pointer, exactly like upstream
/// writing the volatile `pypysig_counter.value` via its address.  Null
/// until `register_ticker` runs at startup.
static TICKER_PTR: AtomicPtr<isize> = AtomicPtr::new(std::ptr::null_mut());

/// `signals.c:51 pypysig_flags_bits` — one bit per pending signal.
static SIG_PENDING: AtomicI64 = AtomicI64::new(0);

/// `signals.c:52 wakeup_fd` — fd to write a byte to on each signal, or
/// -1 when disabled (`signal.set_wakeup_fd`).
static WAKEUP_FD: AtomicI32 = AtomicI32::new(-1);

/// Register the address of the `ActionFlag` ticker so the OS handler can
/// force it negative.  Called once at startup from
/// `install_signal_handling`.
pub fn register_ticker(ptr: *mut isize) {
    TICKER_PTR.store(ptr, Ordering::SeqCst);
}

/// Store -1 into the ticker cell so the next `decrement_ticker` runs
/// `action_dispatcher`.  Async-signal-safe: a single aligned word store.
fn rearm_ticker() {
    let p = TICKER_PTR.load(Ordering::SeqCst);
    if !p.is_null() {
        unsafe { std::ptr::write_volatile(p, -1) };
    }
}

// ── pending-signal bitmask (signals.c pypysig_pushback / pypysig_poll) ──

/// `signals.c:98-114 pypysig_pushback` — set the pending bit for
/// `signum` and force the ticker to -1.  Both the OS handler and
/// `set_interrupt` reach signal delivery through here.
pub fn signal_pushback(signum: i32) {
    if (0..NSIG).contains(&signum) {
        let bitmask = 1i64 << signum;
        SIG_PENDING.fetch_or(bitmask, Ordering::SeqCst);
        rearm_ticker();
    }
}

/// `signals.c:205-223 pypysig_poll` — return the lowest-numbered pending
/// signal, clearing its bit, or -1 when none are pending.
pub fn signal_poll() -> i32 {
    let mut value = SIG_PENDING.load(Ordering::SeqCst);
    while value != 0 {
        let j = value.trailing_zeros() as i32;
        let bit = 1i64 << j;
        match SIG_PENDING.compare_exchange(value, value & !bit, Ordering::SeqCst, Ordering::SeqCst)
        {
            Ok(_) => return j,
            Err(current) => value = current,
        }
    }
    -1
}

// ── wakeup fd (signals.c:246-272 pypysig_set_wakeup_fd) ──

/// `signals.c:246-272 pypysig_set_wakeup_fd` — install `fd`, return the
/// previous one.  pyre always writes the signal-number byte (the
/// `PYPYSIG_WITH_NUL_BYTE` default only matters before any fd is set,
/// during which `WAKEUP_FD` is -1 and nothing is written anyway).
pub fn set_wakeup_fd(fd: i32) -> i32 {
    WAKEUP_FD.swap(fd, Ordering::SeqCst)
}

// ── OS handler + sigaction installation (unix only) ──

/// `signals.c:140-174 signal_setflag_handler` — the actual OS signal
/// handler.  Everything here must be async-signal-safe: flag the signal,
/// then write one byte to the wakeup fd via the raw `write` syscall.
#[cfg(unix)]
extern "C" fn signal_setflag_handler(signum: libc::c_int) {
    signal_pushback(signum as i32);
    let fd = WAKEUP_FD.load(Ordering::SeqCst);
    if fd != -1 {
        let byte = signum as u8;
        // Best-effort, async-signal-safe; ignore the result (signals.c
        // stashes the errno for later — pyre does not surface it yet).
        unsafe {
            let _ = libc::write(fd, &byte as *const u8 as *const libc::c_void, 1);
        }
    }
}

/// Install `handler` for `signum` via `sigaction` with an empty mask and
/// no flags — `signals.c:61-87 / 176-188` (`pypysig_ignore` /
/// `pypysig_default` / `pypysig_setflag`).  Returns whether the
/// `sigaction` call succeeded.
#[cfg(unix)]
fn install_handler(signum: i32, handler: libc::sighandler_t) -> bool {
    unsafe {
        let mut action: libc::sigaction = std::mem::zeroed();
        action.sa_sigaction = handler;
        libc::sigemptyset(&mut action.sa_mask);
        action.sa_flags = 0;
        libc::sigaction(signum, &action, std::ptr::null_mut()) == 0
    }
}

/// `signals.c:176-188 pypysig_setflag` — route `signum` to the flag
/// handler so it is delivered at the next interpreter checkpoint.
#[cfg(unix)]
pub fn pypysig_setflag(signum: i32) -> bool {
    install_handler(signum, signal_setflag_handler as usize)
}

/// `signals.c:75-87 pypysig_default` — restore the OS default action.
#[cfg(unix)]
pub fn pypysig_default(signum: i32) -> bool {
    install_handler(signum, libc::SIG_DFL)
}

/// `signals.c:61-73 pypysig_ignore` — ignore the signal at the OS level.
#[cfg(unix)]
pub fn pypysig_ignore(signum: i32) -> bool {
    install_handler(signum, libc::SIG_IGN)
}

/// `signals.c:190-203 pypysig_reinstall` — no-op on platforms with
/// `SA_RESTART` (sigaction does not reset the handler), which is every
/// unix target pyre builds for.
#[cfg(unix)]
pub fn pypysig_reinstall(_signum: i32) {}

// ── interpreter-thread signal routing ──
//
// pyre runs the interpreter on a thread spawned by `pyrex::main_entry`
// (for a large stack), not the process's original thread.  Process-directed
// async signals (Ctrl-C `SIGINT`, `alarm` `SIGALRM`, `SIGTERM`, …) are
// delivered by the kernel to an arbitrary thread that has them unblocked —
// usually the original thread, which is parked in `join` — so a blocking
// syscall on the interpreter thread is never interrupted.  Blocking the
// async signals on the original thread and unblocking them on the
// interpreter thread makes the interpreter thread the only eligible target,
// so EINTR delivery (`checksignals` in the socket / select / sleep retry
// loops) works.  Synchronous fault signals are left untouched — they must
// reach whichever thread generated them.

/// Build the set of signals routed to the interpreter thread: every signal
/// except the synchronous fault signals.
#[cfg(unix)]
fn async_signal_set() -> libc::sigset_t {
    unsafe {
        let mut set: libc::sigset_t = std::mem::zeroed();
        libc::sigfillset(&mut set);
        for exc in [
            libc::SIGSEGV,
            libc::SIGBUS,
            libc::SIGFPE,
            libc::SIGILL,
            libc::SIGABRT,
            libc::SIGTRAP,
        ] {
            libc::sigdelset(&mut set, exc);
        }
        set
    }
}

/// Block the async signals on the calling thread — called on the process's
/// original thread before the interpreter thread is spawned.
#[cfg(unix)]
pub fn block_async_signals_on_origin_thread() {
    unsafe {
        let set = async_signal_set();
        libc::pthread_sigmask(libc::SIG_BLOCK, &set, std::ptr::null_mut());
    }
}

/// Unblock the async signals on the calling thread — called once on the
/// interpreter thread so process-directed signals land here and interrupt
/// its blocking syscalls.
#[cfg(unix)]
pub fn unblock_async_signals_on_interp_thread() {
    unsafe {
        let set = async_signal_set();
        libc::pthread_sigmask(libc::SIG_UNBLOCK, &set, std::ptr::null_mut());
    }
}

#[cfg(not(unix))]
pub fn block_async_signals_on_origin_thread() {}
#[cfg(not(unix))]
pub fn unblock_async_signals_on_interp_thread() {}

#[cfg(not(unix))]
pub fn pypysig_setflag(_signum: i32) -> bool {
    false
}
#[cfg(not(unix))]
pub fn pypysig_default(_signum: i32) -> bool {
    false
}
#[cfg(not(unix))]
pub fn pypysig_ignore(_signum: i32) -> bool {
    false
}
#[cfg(not(unix))]
pub fn pypysig_reinstall(_signum: i32) {}
