//! The single process-global eval-breaker word polled by JIT-compiled loop
//! back-edges. One `AtomicUsize` whose bits fold the two former back-edge
//! polls into one load + one nonzero branch:
//!   bit0 EB_ASYNC — mirrors a negative async ticker (`ActionFlag._ticker < 0`);
//!                   OR'd in by the OS signal handler and the action dispatcher.
//!   bit1 EB_STW   — mirrors `GC_SYNC.stw_requested`; OR'd in by the collector
//!                   while it drains mutators to safepoints.
//! A compiled loop loads the whole word at the back-edge and deopts to the
//! interpreter when it is non-zero. The interpreter/warm-up loop and the STW
//! park gate remain authoritative; this word is only the JIT's deopt trigger.
//!
//! Immortal process-global, zero from process start, so the guard is harmless
//! until a bit is armed. The published-address holder (`EVAL_BREAKER_WORD_ADDR`)
//! reads `0` until published, and backends treat the back-edge poll as inert
//! in that case.
//!
//! The STW bit and authoritative request occupy two locations. A relaxed poll
//! can briefly deopt before the request is visible, then resume and re-deopt
//! until coherence propagates it; this is a bounded park-latency window.

use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};

thread_local! {
    /// Test-only per-thread override for the baked back-edge poll address.
    static ADDR_OVERRIDE: Cell<usize> = const { Cell::new(0) };
}

/// bit0 — async action / signal pending (mirrors a negative ticker).
pub const EB_ASYNC: usize = 1;
/// bit1 — GC stop-the-world requested (mirrors `GC_SYNC.stw_requested`).
pub const EB_STW: usize = 2;

/// The shared eval-breaker word (see module docs).
static EVAL_BREAKER_WORD: AtomicUsize = AtomicUsize::new(0);

/// Published address of `EVAL_BREAKER_WORD`; `0` until published.
static EVAL_BREAKER_WORD_ADDR: AtomicUsize = AtomicUsize::new(0);

/// Publish the word's address for the backends to bake. Idempotent — the
/// address is an immortal static, so re-publishing stores the same value.
pub fn publish_addr() {
    EVAL_BREAKER_WORD_ADDR.store(
        &EVAL_BREAKER_WORD as *const AtomicUsize as usize,
        Ordering::Relaxed,
    );
}

/// Address the backend bakes into the back-edge poll, or `0` if not published.
pub fn eval_breaker_word_addr() -> usize {
    let ov = ADDR_OVERRIDE.with(|c| c.get());
    if ov != 0 {
        return ov;
    }
    EVAL_BREAKER_WORD_ADDR.load(Ordering::Relaxed)
}

/// Test-only: on the current thread, make the backend bake its back-edge poll
/// against `addr` (a caller-owned word) instead of the process-global word, so
/// unit tests never publish the global address (which would activate every
/// other compiled trace in the test binary). Pass 0 to clear.
pub fn set_addr_override_for_test(addr: usize) {
    ADDR_OVERRIDE.with(|c| c.set(addr));
}

// --- async (bit0): armed by the OS signal handler / action dispatcher ---
// `fetch_or` is a single lock-free atomic RMW → async-signal-safe.
pub fn set_async() {
    EVAL_BREAKER_WORD.fetch_or(EB_ASYNC, Ordering::Relaxed);
}
pub fn clear_async() {
    EVAL_BREAKER_WORD.fetch_and(!EB_ASYNC, Ordering::Relaxed);
}

// --- stw (bit1): armed/cleared by the collector under the quiesce lock ---
pub fn set_stw() {
    EVAL_BREAKER_WORD.fetch_or(EB_STW, Ordering::Release);
}
pub fn clear_stw() {
    EVAL_BREAKER_WORD.fetch_and(!EB_STW, Ordering::Release);
}
