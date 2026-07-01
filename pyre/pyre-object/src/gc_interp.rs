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
//! Gated off by default on native; enabled with `PYRE_GC_INTERP=1`. On wasm it
//! is on by default — the env read returns nothing there, and the interp-path
//! old-gen leak is exactly what makes the wasm benches OOM, so the safepoint
//! major is the mechanism that bounds heap growth. `PYRE_GC_INTERP=0` still
//! turns it off where the env is readable.

use std::cell::Cell;
use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};

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

impl EvalActivationGuard {
    #[inline]
    pub fn enter() -> Self {
        let armed = enabled();
        if armed {
            EVAL_NESTING.with(|d| d.set(d.get() + 1));
        }
        Self { armed }
    }
}

impl Drop for EvalActivationGuard {
    #[inline]
    fn drop(&mut self) {
        if self.armed {
            EVAL_NESTING.with(|d| d.set(d.get().saturating_sub(1)));
        }
    }
}

/// Whether the current eval-loop activation is the outermost one on this
/// thread. The non-moving major walks the interpreter's roots through the
/// registered pyframe walker, which only sees the `CURRENT_FRAME` /
/// `f_backref` chain. When a NESTED eval loop runs — e.g. a Python callback
/// invoked from a native module (`_sre.sub` with a callable, `sorted` key,
/// …) — that native frame holds live `PyObjectRef`s on the Rust stack that
/// the walker cannot reach, so a collection there would free still-reachable
/// old-gen objects. Firing only at the outermost activation keeps the root
/// set complete; deep pure-Python loops simply collect less often.
#[inline]
fn at_outermost_activation() -> bool {
    EVAL_NESTING.with(|d| d.get() <= 1)
}

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

/// Account for one interpreter-routed allocation. Called from `w_int_new` /
/// `w_float_new` after a successful `try_gc_alloc_stable`.
#[inline]
pub fn note_alloc() {
    ALLOC_SINCE_GC.fetch_add(1, Ordering::Relaxed);
}

/// Dispatch-loop safepoint: when enough interpreter objects have accumulated,
/// run a non-moving old-gen-only major to reclaim the dead ones, then reset the
/// counter. A no-op when the flag is off or no collection hook is installed.
///
/// The collection is `try_gc_collect_oldgen` — it seeds roots, marks, and
/// sweeps ONLY the old generation, never touching the nursery (not moved, not
/// freed). So unlike `do_collect_full` (whose leading moving minor would
/// relocate a Rust-stack nursery `PyObjectRef` with no shadowstack root) it is
/// safe with a live nursery, and the `nursery_used == 0` gate is dropped — the
/// reclamation can now fire under an active JIT, exactly when the interp-path
/// old-gen leak accumulates. Reachability stays exact: the marker walks through
/// nursery objects (`old -> nursery -> old` edges are followed) so no live old
/// object is freed. The `jitframe_empty` gate is kept conservatively: a
/// suspended trace's gcmap roots are seeded by `seed_major_roots`, but until
/// that path is independently re-validated the safepoint stays out of the
/// trace-suspended window. It also fires only at the outermost eval activation
/// ([`at_outermost_activation`]) so a Python callback nested inside native
/// module code — whose Rust-stack roots the pyframe walker cannot see — never
/// triggers it.
#[inline]
pub fn safepoint() {
    if !enabled() {
        return;
    }
    if collect_enabled()
        && at_outermost_activation()
        && ALLOC_SINCE_GC.load(Ordering::Relaxed) >= COLLECT_THRESHOLD
        && crate::gc_hook::try_gc_jitframe_empty()
    {
        crate::gc_hook::try_gc_collect_oldgen();
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
