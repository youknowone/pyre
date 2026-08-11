//! Low-level GC hooks — RPython: `rpython/memory/gc/hook.py`.
//!
//! The collector owns one [`GcHooks`] value, exactly like incminimark's
//! `self.hooks`. Translation supplies the concrete PyPy hooks through the
//! six process-global bare function pointers below. Collector-side calls stay
//! allocation-free: callbacks may only update counters and schedule an
//! app-level async action.

pub type GcHookEnabledFn = fn() -> bool;
pub type GcMinorHookFn = fn(duration: f64, total_memory_used: usize, pinned_objects: usize);
pub type GcCollectStepHookFn = fn(duration: f64, oldstate: u8, newstate: u8);
pub type GcCollectHookFn = fn(
    num_major_collects: usize,
    arenas_count_before: usize,
    arenas_count_after: usize,
    arenas_bytes: usize,
    rawmalloc_bytes_before: usize,
    rawmalloc_bytes_after: usize,
    pinned_objects: usize,
);

crate::global_hook!(static IS_GC_MINOR_ENABLED: GcHookEnabledFn);
crate::global_hook!(static IS_GC_COLLECT_STEP_ENABLED: GcHookEnabledFn);
crate::global_hook!(static IS_GC_COLLECT_ENABLED: GcHookEnabledFn);
crate::global_hook!(static ON_GC_MINOR: GcMinorHookFn);
crate::global_hook!(static ON_GC_COLLECT_STEP: GcCollectStepHookFn);
crate::global_hook!(static ON_GC_COLLECT: GcCollectHookFn);

#[derive(Clone, Copy)]
pub struct GcHookCallbacks {
    pub is_gc_minor_enabled: GcHookEnabledFn,
    pub is_gc_collect_step_enabled: GcHookEnabledFn,
    pub is_gc_collect_enabled: GcHookEnabledFn,
    pub on_gc_minor: GcMinorHookFn,
    pub on_gc_collect_step: GcCollectStepHookFn,
    pub on_gc_collect: GcCollectHookFn,
}

/// Install PyPy's `LowLevelGcHooks` virtual surface for the process.
pub fn register_gc_hooks(callbacks: GcHookCallbacks) {
    IS_GC_MINOR_ENABLED.set(Some(callbacks.is_gc_minor_enabled));
    IS_GC_COLLECT_STEP_ENABLED.set(Some(callbacks.is_gc_collect_step_enabled));
    IS_GC_COLLECT_ENABLED.set(Some(callbacks.is_gc_collect_enabled));
    ON_GC_MINOR.set(Some(callbacks.on_gc_minor));
    ON_GC_COLLECT_STEP.set(Some(callbacks.on_gc_collect_step));
    ON_GC_COLLECT.set(Some(callbacks.on_gc_collect));
}

/// `rpython/memory/gc/hook.py:5-70 GcHooks`.
///
/// This zero-sized value is deliberately a field on `MiniMarkGC`; the global
/// cells provide its translated virtual method bodies, not a parallel store.
#[derive(Default)]
pub struct GcHooks;

impl GcHooks {
    #[inline]
    pub fn fire_gc_minor(&self, duration: f64, total_memory_used: usize, pinned_objects: usize) {
        if IS_GC_MINOR_ENABLED.get().is_some_and(|enabled| enabled())
            && let Some(on_gc_minor) = ON_GC_MINOR.get()
        {
            on_gc_minor(duration, total_memory_used, pinned_objects);
        }
    }

    #[inline]
    pub fn fire_gc_collect_step(&self, duration: f64, oldstate: u8, newstate: u8) {
        if IS_GC_COLLECT_STEP_ENABLED
            .get()
            .is_some_and(|enabled| enabled())
            && let Some(on_gc_collect_step) = ON_GC_COLLECT_STEP.get()
        {
            on_gc_collect_step(duration, oldstate, newstate);
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[inline]
    pub fn fire_gc_collect(
        &self,
        num_major_collects: usize,
        arenas_count_before: usize,
        arenas_count_after: usize,
        arenas_bytes: usize,
        rawmalloc_bytes_before: usize,
        rawmalloc_bytes_after: usize,
        pinned_objects: usize,
    ) {
        if IS_GC_COLLECT_ENABLED.get().is_some_and(|enabled| enabled())
            && let Some(on_gc_collect) = ON_GC_COLLECT.get()
        {
            on_gc_collect(
                num_major_collects,
                arenas_count_before,
                arenas_count_after,
                arenas_bytes,
                rawmalloc_bytes_before,
                rawmalloc_bytes_after,
                pinned_objects,
            );
        }
    }
}
