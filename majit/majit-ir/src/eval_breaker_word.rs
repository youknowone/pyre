//! The single process-global eval-breaker word polled by JIT-compiled loop
//! back-edges. One `AtomicUsize` whose bits fold the two former back-edge
//! polls into one load + one nonzero branch:
//!   bit0 EB_ASYNC — an async ticker request; OR'd in by the OS signal handler
//!                   and action dispatcher, then copied into
//!                   `ActionFlag._ticker` at a safe interpreter checkpoint.
//!   bit1 EB_STW   — mirrors `GC_SYNC.stw_requested`; OR'd in by the collector
//!                   while it drains mutators to safepoints.
//!   bit2 EB_FINALIZING — mirrors interpreter finalization; once armed,
//!                        non-owner mutators park before their next opcode.
//!   bit3 EB_GC_INTERP — process-stable `PYRE_GC_INTERP` dispatch gate.  This
//!                       is masked out of compiled back-edge polls: it avoids
//!                       a second per-opcode atomic load in the interpreter,
//!                       but is not itself a reason to leave machine code.
//!   bit4 EB_GC    — the old-gen allocator reached the next-major threshold;
//!                   OR'd in by the collector, consumed by the interpreter
//!                   dispatch-loop GC safepoint.
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
/// bit2 — interpreter finalization has begun (terminal, never cleared).
pub const EB_FINALIZING: usize = 4;
/// bit3 — interpreter-path allocation/collection integration is enabled.
pub const EB_GC_INTERP: usize = 8;
/// bit4 — the old-gen allocator crossed the next-major threshold and a
/// collection is owed at the next root-complete point.
pub const EB_GC: usize = 16;
/// Bits that require a compiled loop to deopt to the interpreter.
///
/// `EB_GC` belongs here: the allocator only arms the request, and the
/// collection itself runs at the dispatch-loop safepoint, so a compiled loop
/// has to leave machine code for the request to be serviced at all.
pub const JIT_BREAKER_MASK: usize = EB_ASYNC | EB_STW | EB_FINALIZING | EB_GC;

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

#[inline]
pub fn load() -> usize {
    EVAL_BREAKER_WORD.load(Ordering::Relaxed)
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

pub fn set_finalizing() {
    EVAL_BREAKER_WORD.fetch_or(EB_FINALIZING, Ordering::Release);
}

pub fn set_gc_interp() {
    EVAL_BREAKER_WORD.fetch_or(EB_GC_INTERP, Ordering::Release);
}

// --- gc (bit4): armed by the old-gen allocator, consumed by the safepoint ---

/// Request a major collection at the next root-complete point.
///
/// `external_malloc` (incminimark.py) tests `threshold_reached` in the
/// allocator and drives `minor_collection_with_major_progress` there. pyre
/// cannot collect at an arbitrary allocation site — a Rust caller's locals are
/// not roots — so the check stays in the allocator and only the action is
/// deferred: this bit fails the next back-edge poll, and the interpreter
/// dispatch loop, where the frame walker sees the whole root set, performs the
/// collection.
pub fn set_gc() {
    EVAL_BREAKER_WORD.fetch_or(EB_GC, Ordering::Relaxed);
}

/// Consume the pending-collection request, reporting whether one was armed.
///
/// The caller must clear unconditionally — before any decision about whether it
/// can service the request. A bit left armed by a safepoint that declined to
/// collect fails every subsequent back edge, so a compiled loop would deopt on
/// each iteration instead of once.
pub fn take_gc() -> bool {
    // The taker runs on every eligible bytecode dispatch, the armer only when
    // the threshold is crossed; read before the read-modify-write so the
    // common case is a plain load.
    if EVAL_BREAKER_WORD.load(Ordering::Relaxed) & EB_GC == 0 {
        return false;
    }
    EVAL_BREAKER_WORD.fetch_and(!EB_GC, Ordering::Relaxed) & EB_GC != 0
}

/// Depth of the operation chain between the poll's load and its guard.
///
/// The recorder emits `RawLoadI -> IntAnd -> IntIsTrue -> GuardFalse`, so two
/// links separate the guard's condition from the load. The walk below allows
/// a few more so that an optimizer pass inserting or splitting one link does
/// not silently stop matching, and stops well before a long pure chain.
const POLL_CHAIN_DEPTH: usize = 6;

/// Whether `guard`'s condition is a back-edge poll of this word.
///
/// The poll is recorded as
/// `GuardFalse(IntIsTrue(IntAnd(RawLoadI(addr, 0), JIT_BREAKER_MASK)))`. The
/// match anchors on the `RawLoadI` of the published address rather than on the
/// whole shape: that load is the one link the recorder cannot drop (it must
/// stay non-pure and outside the always-pure range, or CSE forwards the
/// preamble's guarded-zero value into the loop body), so walking back to it
/// survives rewrites of the links in between.
///
/// An unpublished address reads `0` and no poll is recorded, so the walk
/// declines rather than matching an unrelated load from a null constant.
pub fn is_back_edge_poll_guard(guard: &crate::resoperation::Op) -> bool {
    let addr = eval_breaker_word_addr();
    if addr == 0 {
        return false;
    }
    // Control-flow guards (`GUARD_NOT_FORCED`, `GUARD_NO_EXCEPTION`, ...) carry
    // no condition operand at all.
    if guard.num_args() == 0 {
        return false;
    }
    let mut operand = guard.arg(0);
    for _ in 0..POLL_CHAIN_DEPTH {
        let Some(op) = operand.bound_op() else {
            return false;
        };
        if op.num_args() == 0 {
            return false;
        }
        if op.opcode == crate::resoperation::OpCode::RawLoadI {
            return op.arg(0).const_int() == Some(addr as i64);
        }
        operand = op.arg(0);
    }
    false
}

/// Every flag must fit in the word the poll actually loads. Checked per target,
/// so a flag too wide for a 32-bit `usize` fails the wasm32 build rather than
/// silently reading as unarmed there.
// `ineffective_bit_mask` reads the fold as a runtime `x | 16` tested against a
// constant; both sides here are constants and the whole item is a static
// assertion, so there is no operand to compare directly.
#[allow(clippy::ineffective_bit_mask)]
const _: () = assert!(
    (EB_ASYNC | EB_STW | EB_FINALIZING | EB_GC_INTERP | EB_GC)
        < (1 << (EVAL_BREAKER_WORD_SIZE * 8 - 1))
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resoperation::{Op, OpCode};
    use crate::value::Const;
    use std::rc::Rc;

    fn bind(op: Op) -> crate::operand::Operand {
        crate::operand::Operand::from_bound_op(&Rc::new(op))
    }

    fn int(value: i64) -> crate::operand::Operand {
        crate::operand::Operand::const_(Const::Int(value))
    }

    /// Build the recorded back-edge poll over `load_addr`, which the caller
    /// varies to separate "this word" from "some other raw load".
    fn poll_guard(load_addr: usize) -> Op {
        let load = bind(Op::new(OpCode::RawLoadI, &[int(load_addr as i64), int(0)]));
        let masked = bind(Op::new(
            OpCode::IntAnd,
            &[load, int(JIT_BREAKER_MASK as i64)],
        ));
        let armed = bind(Op::new(OpCode::IntIsTrue, &[masked]));
        Op::new(OpCode::GuardFalse, &[armed])
    }

    /// The counter split turns on this predicate alone, so it has to separate
    /// the poll from every other guard the same trace records. Only the
    /// address published for this word matches: a raw load of a neighbouring
    /// address reaches the same opcode chain, and a data guard reaches none of
    /// it.
    ///
    /// `publish_addr` is idempotent and writes only the address holder, not the
    /// word, so this leaves the process-global flag state alone.
    #[test]
    fn only_a_poll_of_this_words_address_is_recognised() {
        publish_addr();
        let addr = eval_breaker_word_addr();
        assert_ne!(addr, 0, "publish_addr must make the address readable");

        assert!(is_back_edge_poll_guard(&poll_guard(addr)));
        // Same shape, a different word: the JIT records other raw loads.
        assert!(!is_back_edge_poll_guard(&poll_guard(
            addr + EVAL_BREAKER_WORD_SIZE
        )));
        // A guard on traced values — the population whose failures the
        // `guard_failures` total is meant to describe.
        let value = bind(Op::new(OpCode::IntAdd, &[int(1), int(2)]));
        let is_true = bind(Op::new(OpCode::IntIsTrue, &[value]));
        assert!(!is_back_edge_poll_guard(&Op::new(
            OpCode::GuardTrue,
            &[is_true]
        )));
        // A control-flow guard carries no condition operand at all.
        assert!(!is_back_edge_poll_guard(&Op::new(
            OpCode::GuardNotForced,
            &[]
        )));
    }

    /// One request arms the poll once: the taker reports it, clears it, and the
    /// next taker sees nothing. A taker that failed to clear would keep failing
    /// the back-edge guard and deopt the loop on every iteration. Taking it
    /// must also leave the other two bits alone — they address disjoint parts
    /// of the same word.
    ///
    /// The word is a process-global, so this is deliberately the only test in
    /// the crate that touches it.
    #[test]
    fn gc_request_is_taken_exactly_once_and_leaves_the_other_bits() {
        assert!(!take_gc(), "no request armed at the start of the test");
        set_async();
        set_stw();
        set_gc();
        assert_ne!(load() & EB_GC, 0);
        assert!(take_gc());
        assert_eq!(load() & EB_GC, 0);
        assert!(!take_gc());
        assert_eq!(load() & (EB_ASYNC | EB_STW), EB_ASYNC | EB_STW);
        clear_async();
        clear_stw();
    }
}
