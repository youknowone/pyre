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
//! until a bit is armed. The trace records the address as the poll's constant
//! operand; the published-address holder (`EVAL_BREAKER_WORD_ADDR`) reads `0`
//! until published, and no poll is recorded in that case.
//!
//! The STW bit and authoritative request occupy two locations. A relaxed poll
//! can briefly deopt before the request is visible, then resume and re-deopt
//! until coherence propagates it; this is a bounded park-latency window.

use std::sync::atomic::{AtomicUsize, Ordering};

/// bit0 — async action / signal pending (mirrors a negative ticker).
pub const EB_ASYNC: usize = 1;
/// bit1 — GC stop-the-world requested (mirrors `GC_SYNC.stw_requested`).
pub const EB_STW: usize = 2;

/// The shared eval-breaker word (see module docs).
static EVAL_BREAKER_WORD: AtomicUsize = AtomicUsize::new(0);

/// Width of the word, in bytes. The back-edge poll's load descriptor must use
/// exactly this size: a wider load reads past the word into the adjacent
/// static, so the poll's nonzero test would always be true and every back-edge
/// guard would fail. Pointer-width rather than fixed-64 keeps `fetch_or`
/// lock-free — `set_async` runs inside an OS signal handler.
pub const EVAL_BREAKER_WORD_SIZE: usize = size_of::<AtomicUsize>();

/// Published address of `EVAL_BREAKER_WORD`; `0` until published.
static EVAL_BREAKER_WORD_ADDR: AtomicUsize = AtomicUsize::new(0);

/// Publish the word's address for the tracer to record. Idempotent — the
/// address is an immortal static, so re-publishing stores the same value.
pub fn publish_addr() {
    EVAL_BREAKER_WORD_ADDR.store(
        &EVAL_BREAKER_WORD as *const AtomicUsize as usize,
        Ordering::Relaxed,
    );
}

/// Address the trace records as the back-edge poll's constant, or `0` if not
/// published — in which case no poll is recorded.
pub fn eval_breaker_word_addr() -> usize {
    EVAL_BREAKER_WORD_ADDR.load(Ordering::Relaxed)
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

/// Every flag must fit in the word the poll actually loads. Checked per target,
/// so a flag too wide for a 32-bit `usize` fails the wasm32 build rather than
/// silently reading as unarmed there.
const _: () = assert!((EB_ASYNC | EB_STW) < (1 << (EVAL_BREAKER_WORD_SIZE * 8 - 1)));
