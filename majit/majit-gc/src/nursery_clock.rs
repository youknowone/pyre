//! Host bytes stand in for nursery bytes on the JIT decay clock.
//!
//! `JitCounter.invoke_after_minor_collection` (installed on
//! `invoke_after_minor_collection`) runs after every minor collection and,
//! every 32 of them, decays the guard and loop counters. Upstream's JIT
//! allocates its trace state in the nursery, so that clock ticks with the
//! JIT's own work. Majit's trace state is allocated on the host heap and
//! never enters the nursery, so those host bytes are what advance the clock
//! here. `IncrementalMiniMarkGC.__init__` puts only objects at or below
//! `nonlarge_max` in the nursery; a larger request is `external_malloc`ed and
//! does not consume it, and this clock uses that same cutoff.

use std::alloc::{GlobalAlloc, Layout};
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::collector::LARGE_OBJECT_THRESHOLD;

/// Bytes of host allocation still owed to the decay clock.
static PENDING_BYTES: AtomicUsize = AtomicUsize::new(0);

/// Nursery size, in bytes. Zero until [`set_quantum`] runs, and while it is
/// zero nothing is counted: there is no nursery yet to stand in for.
static QUANTUM: AtomicUsize = AtomicUsize::new(0);

/// Record the nursery size `MiniMarkGC` just applied. The clock ticks once
/// per this many host bytes.
pub(crate) fn set_quantum(nursery_size: usize) {
    QUANTUM.store(nursery_size, Ordering::Release);
}

/// Add `bytes` toward the next decay-clock tick.
///
/// Allocation-free and lock-free: one compare-exchange updates the pending
/// total, subtracting one quantum per tick it crosses. The hook runs only
/// after that exchange succeeds, so a hook that itself allocates re-enters
/// this function instead of observing a half-updated total.
fn note_allocated(bytes: usize) {
    if bytes == 0 || bytes >= LARGE_OBJECT_THRESHOLD {
        return;
    }
    let mut pending = PENDING_BYTES.load(Ordering::Relaxed);
    loop {
        let quantum = QUANTUM.load(Ordering::Acquire);
        if quantum == 0 {
            return;
        }
        let sum = pending.saturating_add(bytes);
        let ticks = sum / quantum;
        let next = sum % quantum;
        match PENDING_BYTES.compare_exchange_weak(
            pending,
            next,
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => {
                for _ in 0..ticks {
                    crate::invoke_after_minor_collection_hook();
                }
                return;
            }
            Err(observed) => pending = observed,
        }
    }
}

/// `GlobalAlloc` that forwards to `A` and charges a successful non-large
/// request against the nursery quantum.
pub struct HostNurseryClock<A: GlobalAlloc>(pub A);

unsafe impl<A: GlobalAlloc> GlobalAlloc for HostNurseryClock<A> {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { self.0.alloc(layout) };
        if !ptr.is_null() {
            note_allocated(layout.size());
        }
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { self.0.alloc_zeroed(layout) };
        if !ptr.is_null() {
            note_allocated(layout.size());
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { self.0.dealloc(ptr, layout) };
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let ptr = unsafe { self.0.realloc(ptr, layout, new_size) };
        if !ptr.is_null() {
            // A resize allocates a fresh array, so the whole new size counts,
            // the same way a nursery resize does.
            note_allocated(new_size);
        }
        ptr
    }
}
