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
//! through the registered `pyframe` root walker). The collection is throttled by
//! an allocation counter so the old-gen high-water stays bounded.
//!
//! Gated off by default; enabled with `PYRE_GC_INTERP=1`. On wasm the env read
//! returns nothing, so the flag is always off there for now.

use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};

/// Tri-state: 0 = not yet read from env, 1 = disabled, 2 = enabled.
static STATE: AtomicU8 = AtomicU8::new(0);

/// Tri-state for the safepoint collection, gated by `PYRE_GC_INTERP_COLLECT`
/// (default on when `PYRE_GC_INTERP` is on). Lets us A/B routing-only vs
/// routing+collection while diagnosing root-completeness.
static COLLECT_STATE: AtomicU8 = AtomicU8::new(0);

/// Number of interpreter-routed object allocations since the last collection.
static ALLOC_SINCE_GC: AtomicUsize = AtomicUsize::new(0);

/// Allocations between safepoint collections. At ~24-40 B per int/float this
/// bounds the dead-object high-water to a couple of MB.
const COLLECT_THRESHOLD: usize = 1 << 16;

/// Whether `PYRE_GC_INTERP` routes int/float allocations through the GC and
/// arms the dispatch-loop safepoint. Reads the env once, then caches.
#[inline]
pub fn enabled() -> bool {
    match STATE.load(Ordering::Relaxed) {
        1 => false,
        2 => true,
        _ => {
            let on = std::env::var_os("PYRE_GC_INTERP")
                .map(|v| !v.is_empty() && v != "0")
                .unwrap_or(false);
            STATE.store(if on { 2 } else { 1 }, Ordering::Relaxed);
            on
        }
    }
}

/// Account for one interpreter-routed allocation. Called from `w_int_new` /
/// `w_float_new` after a successful `try_gc_alloc_stable`.
#[inline]
pub fn note_alloc() {
    ALLOC_SINCE_GC.fetch_add(1, Ordering::Relaxed);
}

/// Dispatch-loop safepoint: when enough interpreter objects have accumulated,
/// run a full collection to reclaim the dead ones, then reset the counter. A
/// no-op when the flag is off or no collection hook is installed.
///
/// The collection is gated on an empty nursery. `do_collect_full` runs a moving
/// minor cycle first; with a live nursery, an object referenced only from a
/// Rust-stack temporary (or a jitframe whose gcmap is stale at this PC) would be
/// relocated and dangle — the documented shadowstack gap. An empty nursery
/// makes the minor cycle a no-op, so the collection reduces to a non-moving
/// old-gen mark-sweep, which the registered pyframe root walker covers. When the
/// JIT is active its traces keep the nursery non-empty and trigger their own
/// (gcmap-rooted) collections that sweep old-gen, so skipping here loses no
/// reclamation.
#[inline]
pub fn safepoint() {
    if !enabled() {
        return;
    }
    if collect_enabled()
        && ALLOC_SINCE_GC.load(Ordering::Relaxed) >= COLLECT_THRESHOLD
        && crate::gc_hook::try_gc_nursery_used() == 0
        && crate::gc_hook::try_gc_jitframe_empty()
    {
        crate::gc_hook::try_gc_collect();
        ALLOC_SINCE_GC.store(0, Ordering::Relaxed);
    }
}

/// Whether the safepoint actually collects. Off via `PYRE_GC_INTERP_COLLECT=0`
/// to isolate the allocation routing from the collection while diagnosing.
#[inline]
fn collect_enabled() -> bool {
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
