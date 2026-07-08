//! Address of the interpreter's async-action ticker cell, published by the
//! host (pyre) so the JIT backends can bake a `GuardEvalBreaker` that polls it
//! at loop back-edges.
//!
//! The cell is `ExecutionContext.actionflag._ticker`; the OS signal handler
//! forces it negative and the interpreter runs `action_dispatcher` when it
//! goes below zero. Compiled loops mirror the `CHECK_EVAL_BREAKER()` back-edge
//! ticker poll by loading this cell and deopting to the interpreter when it is
//! negative, so async signals/actions are delivered without waiting for the
//! loop to exit naturally.
//!
//! The address is a single process-global registered once at startup
//! (`register_ticker` ← `install_signal_handling`) and stable for the process
//! lifetime (the `ExecutionContext` is held behind an `Rc` and never moves).
//! `0` means no cell has been published yet — backends treat `GuardEvalBreaker`
//! as inert in that case.

use std::sync::atomic::{AtomicUsize, Ordering};

static TICKER_ADDR: AtomicUsize = AtomicUsize::new(0);

/// Publish the ticker cell address. Called once at startup by the host.
pub fn set_ticker_addr(addr: usize) {
    TICKER_ADDR.store(addr, Ordering::Relaxed);
}

/// Address of the ticker cell, or `0` if none has been published.
pub fn ticker_addr() -> usize {
    TICKER_ADDR.load(Ordering::Relaxed)
}
