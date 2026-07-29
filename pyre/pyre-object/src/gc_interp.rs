//! Interpreter-path GC integration (experimental, `PYRE_GC_INTERP`).
//!
//! Objects the bytecode interpreter creates via [`crate::lltype::malloc_typed`]
//! (`w_int_new` / `w_float_new`) go through `alloc_with_gc_header`, a bare
//! `std::alloc::alloc` that is never tracked by the collector and never freed —
//! a permanent leak. The JIT-compiled path avoids this because it allocates in
//! the managed nursery; the interpreter path does not, so an interpreter-heavy
//! workload (the wasm benches, or any native run with the JIT cold) grows RSS
//! linearly with the number of objects created.
//!
//! The faithful fix is RPython's model: allocate young objects in the moving
//! nursery and let the allocator trigger a minor collection when it fills. pyre
//! cannot do that yet — the interpreter has no shadowstack pass, so a moving
//! collection would relocate any live `PyObjectRef` held only on the Rust stack
//! of a bytecode handler and leave it dangling (documented at
//! `pyre/pyre-jit/src/eval.rs` `pyre_object_gc_collect_trampoline`).
//!
//! This module is the safe stepping stone: route the interpreter's int/float
//! allocations through the *non-moving* old-gen (`try_gc_alloc_stable`, the same
//! path dict/set/list/instances already use), so they become GC-tracked without
//! the move hazard, and trigger a full mark-sweep at a bytecode-dispatch
//! safepoint (loop top, where the only live refs are in the frame and reachable
//! through the registered `pyframe` root walker). When to collect is not
//! decided here: the safepoint asks the collector's own `threshold_reached`
//! (incminimark.py:1288-1290), which weighs everything the collector is
//! responsible for against the threshold `set_major_threshold_from`
//! (incminimark.py:575-594) derived from the last major's survivors. Upstream
//! asks that question in the allocator; pyre asks it here because here is where
//! it can act on the answer.
//!
//! Gated off by default on native; enabled with `PYRE_GC_INTERP=1`. On wasm it
//! is on by default — the env read returns nothing there, and the interp-path
//! old-gen leak is exactly what makes the wasm benches OOM, so the safepoint
//! major is the mechanism that bounds heap growth. `PYRE_GC_INTERP=0` still
//! turns it off where the env is readable.

use std::cell::Cell;
use std::sync::atomic::{AtomicU8, Ordering};

/// Tri-state: 0 = not yet read from env, 1 = disabled, 2 = enabled.
static STATE: AtomicU8 = AtomicU8::new(0);

thread_local! {
    /// Number of interpreter eval-loop activations currently on this
    /// thread's call stack (both the plain `eval_loop` and the JIT
    /// `eval_loop_jit`). Maintained only while the flag is on, via
    /// [`EvalActivationGuard`]. The dispatch-loop safepoint consults it so
    /// a collection only fires at the OUTERMOST activation — see
    /// [`at_outermost_activation`].
    static EVAL_NESTING: Cell<u32> = const { Cell::new(0) };
}

/// RAII guard that counts one interpreter eval-loop activation. Construct it
/// at the top of `eval_loop` / `eval_loop_jit`; the matching `Drop` rewinds
/// the depth on every exit path (normal return, `?`, unwind). A no-op when
/// the flag is off, so the un-gated interpreter pays nothing.
pub struct EvalActivationGuard {
    armed: bool,
}

/// Count one eval-loop activation on this thread.
///
/// Writes the runtime-mutable `EVAL_NESTING` thread-local, not a build-time
/// constant, so the JIT residualizes the call instead of tracing into it
/// (`@dont_look_inside`, the [`at_outermost_activation`] sibling). No
/// arguments, `()` result, and it cannot raise.
#[majit_macros::dont_look_inside]
pub fn note_eval_activation_enter() {
    EVAL_NESTING.with(|d| d.set(d.get() + 1));
}

/// Rewind one eval-loop activation. The [`note_eval_activation_enter`] twin;
/// same residual contract.
#[majit_macros::dont_look_inside]
pub fn note_eval_activation_exit() {
    EVAL_NESTING.with(|d| d.set(d.get().saturating_sub(1)));
}

impl EvalActivationGuard {
    #[inline]
    pub fn enter() -> Self {
        let armed = enabled();
        if armed {
            note_eval_activation_enter();
        }
        Self { armed }
    }

    #[inline]
    pub fn armed(&self) -> bool {
        self.armed
    }
}

impl Drop for EvalActivationGuard {
    #[inline]
    fn drop(&mut self) {
        if self.armed {
            note_eval_activation_exit();
        }
    }
}

/// Whether the safepoint may fire at this eval-loop nesting depth.
///
/// Depth 1 = module-level loop, depth 2 = one called function's loop.
/// Both are safe: a Python CALL opcode completes before re-entering the
/// eval loop, so no opcode handler holds a Rust-stack `PyObjectRef` across
/// the boundary; the pyframe root walker sees the full `CURRENT_FRAME` /
/// `f_backref` chain.
///
/// Depth ≥ 3 = a callback re-entry from inside an opcode handler (e.g.
/// FOR_ITER → `__getitem__`, exception handler, `_sre.sub` callable,
/// `sorted` key). The outer handler holds live `PyObjectRef`s on the
/// Rust stack that the walker cannot reach — a collection would free
/// still-reachable old-gen objects. Block the safepoint there.
///
/// Reads the runtime-mutable `EVAL_NESTING` thread-local, not a build-time
/// constant, so the JIT residualizes the call instead of tracing into it
/// (`@dont_look_inside`, the [`enabled`] sibling). The `-> bool` return fits
/// a single word and it cannot raise.
#[majit_macros::dont_look_inside]
pub fn at_outermost_activation() -> bool {
    EVAL_NESTING.with(|d| d.get() <= 2)
}

/// Tri-state for the safepoint collection, gated by `PYRE_GC_INTERP_COLLECT`
/// (default on when `PYRE_GC_INTERP` is on). Lets us A/B routing-only vs
/// routing+collection while diagnosing root-completeness.
static COLLECT_STATE: AtomicU8 = AtomicU8::new(0);

/// Dispatches between two `threshold_reached` queries.
///
/// This is a poll rate, not a collection policy: the safepoint holds no model
/// of the heap, and this value cannot make a collection happen that the
/// collector did not ask for. It exists only because reaching the collector
/// costs a thread-local borrow — or, with no backend GC box installed, a
/// `gc_op` compare-and-exchange — which is too much to spend on every bytecode
/// dispatch. At a few tens of bytes allocated per dispatch the interval spans
/// far less than `min_heap_size` (incminimark.py:307), the smallest threshold
/// the collector will ever set, so no answer is reached late enough to matter.
const POLL_INTERVAL: u32 = 1024;

thread_local! {
    /// Eligible dispatches since this thread last asked the collector; see
    /// [`POLL_INTERVAL`]. Advanced only once the cheaper gates have passed, so
    /// it paces the queries actually made rather than the dispatches seen.
    /// Thread-local rather than atomic because the query it paces is itself
    /// per-thread, and a `fetch_add` on every dispatch would cost about what it
    /// is here to avoid.
    static POLL_TICK: Cell<u32> = const { Cell::new(0) };
}

/// Whether this dispatch is the one that asks the collector.
///
/// Reads the runtime-mutable `POLL_TICK` thread-local, not a build-time
/// constant, so the JIT residualizes the call instead of tracing into it
/// (`@dont_look_inside`, the [`at_outermost_activation`] sibling). The
/// `-> bool` return fits a single word and it cannot raise.
#[majit_macros::dont_look_inside]
pub fn poll_due() -> bool {
    POLL_TICK.with(|t| {
        let n = t.get() + 1;
        if n >= POLL_INTERVAL {
            t.set(0);
            true
        } else {
            t.set(n);
            false
        }
    })
}

/// Whether `PYRE_GC_INTERP` routes int/float allocations through the GC and
/// arms the dispatch-loop safepoint. Reads the env once, then caches.
///
/// Reads (and lazily initialises) the runtime `STATE` atomic; the value is not
/// a build-time constant, so the JIT residualises the call instead of tracing
/// into it (`@dont_look_inside`).
#[majit_macros::dont_look_inside]
pub fn enabled() -> bool {
    match STATE.load(Ordering::Relaxed) {
        1 => false,
        2 => true,
        _ => {
            let on = std::env::var_os("PYRE_GC_INTERP")
                .map(|v| !v.is_empty() && v != "0")
                .unwrap_or(cfg!(target_arch = "wasm32"));
            STATE.store(if on { 2 } else { 1 }, Ordering::Relaxed);
            on
        }
    }
}

/// Dispatch-loop safepoint: when the collector says it has reached the
/// threshold it set for its next major, run a non-moving old-gen-only major.
/// A no-op when the flag is off or no collection hook is installed.
///
/// The decision is entirely the collector's. `threshold_reached`
/// (incminimark.py:1288-1290) compares `get_total_memory_used()` — every byte
/// the collector is responsible for, headers included, whatever allocated them
/// — against the threshold `set_major_threshold_from` (incminimark.py:575-594)
/// derived from the last major's survivors under `major_collection_threshold`,
/// `growth_rate_max`, `max_delta`, `min_heap_size` and `max_heap_size`. Running
/// the collection re-derives it, because `do_collect_oldgen_nonmoving` drives
/// `major_collection_step` to the end of the cycle and `finish_incremental_cycle`
/// sets the next threshold there.
///
/// The safepoint deliberately keeps no counter of its own. Upstream asks this
/// question in the allocator, where every allocation passes; a second tally
/// kept beside the collector could only ever see the subset of allocations
/// that remembered to report, and would answer for a heap that is not the one
/// being collected.
///
/// The collection is `try_gc_collect_oldgen` — it seeds roots, marks, and
/// sweeps ONLY the old generation, never touching the nursery (not moved, not
/// freed). So unlike `do_collect_full` (whose leading moving minor would
/// relocate a Rust-stack nursery `PyObjectRef` with no shadowstack root) it is
/// safe with a live nursery, and the `nursery_used == 0` gate is dropped — the
/// reclamation can now fire under an active JIT, exactly when the interp-path
/// old-gen leak accumulates. Reachability stays exact: the marker walks through
/// nursery objects (`old -> nursery -> old` edges are followed) so no live old
/// object is freed. A suspended compiled trace is no obstacle either: a trace
/// suspends by making a residual call into the interpreter, and a call site is
/// where the backend emits a gcmap, so `enumerate_root_walker_values` reaches
/// the suspended frame's live refs by expanding each jitframe on the JF shadow
/// stack through `trace_libc_jitframe`. That is the same basis on which
/// upstream collects with compiled frames on the stack. The safepoint fires
/// only at the outermost eval activation ([`at_outermost_activation`]) so a
/// Python callback nested inside native module code — whose Rust-stack roots
/// the pyframe walker cannot see — never triggers it.
///
/// Dispatches to the installed threshold and collection hooks, neither a
/// build-time constant, so the JIT residualizes the call instead of tracing
/// into it (`@dont_look_inside`, the [`enabled`] sibling). A `()` return has no
/// discriminant to erase and it cannot raise. On the cold interpreter dispatch
/// loop its first act is already the non-inlined `enabled()` early-return; the
/// residual adds no hot-path cost the un-residualized form did not already pay.
#[majit_macros::dont_look_inside]
pub fn safepoint() {
    if !enabled() {
        return;
    }
    if collect_enabled()
        && at_outermost_activation()
        && poll_due()
        && crate::gc_hook::try_gc_major_threshold_reached()
    {
        crate::gc_hook::try_gc_collect_oldgen();
    }
}

/// Whether the safepoint actually collects. Off via `PYRE_GC_INTERP_COLLECT=0`
/// to isolate the allocation routing from the collection while diagnosing.
///
/// Reads (and lazily initialises) the runtime-mutable `COLLECT_STATE` atomic,
/// not a build-time constant, so the JIT residualizes the call instead of
/// tracing into it (`@dont_look_inside`, the [`enabled`] sibling). The
/// `-> bool` return fits a single word and it cannot raise.
#[majit_macros::dont_look_inside]
pub fn collect_enabled() -> bool {
    match COLLECT_STATE.load(Ordering::Relaxed) {
        1 => false,
        2 => true,
        _ => {
            let on = std::env::var_os("PYRE_GC_INTERP_COLLECT")
                .map(|v| !v.is_empty() && v != "0")
                .unwrap_or(true);
            COLLECT_STATE.store(if on { 2 } else { 1 }, Ordering::Relaxed);
            on
        }
    }
}
