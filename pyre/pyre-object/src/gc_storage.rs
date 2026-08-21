//! Generic GC-managed boxes for host-owned storage containers.
//!
//! This is the storage-container generalization of the established raw
//! `BigInt` box in `longobject.rs`: [`crate::longobject::BIGINT_PAYLOAD_SIZE`],
//! [`crate::longobject::bigint_destructor`], and
//! [`crate::longobject::alloc_bigint_stable`].

/// Allocate `value` inside a GC-managed, non-moving old-gen box tagged `tid`.
///
/// The box layout is `[GcHeader | T]`, so the returned `*mut T` points at the
/// payload exactly as [`crate::lltype::malloc_raw`] would. This makes it a
/// drop-in replacement at storage-box construction sites. The GC sweep
/// reclaims the box and runs the registered drop-glue destructor (see
/// [`storage_box_destructor`]), so the caller must not use `Box::from_raw` for
/// a GC-managed result.
///
/// The payload size registered in the matching `TypeInfo` must be exactly
/// `size_of::<T>()`, mirroring `longobject.rs`'s `BIGINT_PAYLOAD_SIZE` invariant.
///
/// Allocation uses [`crate::gc_hook::try_gc_alloc_stable_raw`] because storage
/// containers are self-mutating: their methods re-derive `self` from a raw
/// pointer across allocating calls, so the box address must remain stable. This
/// is the same non-moving rule used by
/// [`crate::longobject::alloc_bigint_stable`] and mapdict storage.
///
/// When `tid == 0` or no GC hook is installed (unit tests and pre-init), this
/// falls back to [`crate::lltype::malloc_raw`]. In that case the caller's
/// existing manual-free path remains responsible for the allocation.
#[inline]
pub fn gc_alloc_storage_box<T: 'static>(value: T, tid: u32) -> *mut T {
    if tid != 0 {
        let raw = crate::gc_hook::try_gc_alloc_stable_raw(tid, std::mem::size_of::<T>());
        if !raw.is_null() {
            unsafe {
                std::ptr::write(raw as *mut T, value);
            }
            return raw as *mut T;
        }
    }
    crate::lltype::malloc_raw(value)
}

/// `buffer.py RawByteBuffer.__init__` — report a raw payload's bytes
/// to the collector, the line upstream writes right after the raw malloc.
///
/// Upstream keeps two buffer representations. `ByteBuffer` (`buffer.py`)
/// holds `['\0'] * n`, a GC-heap list the collector allocates and therefore
/// counts by itself. `RawByteBuffer` puts the same bytes in raw memory and
/// pairs the allocation with `rgc.add_memory_pressure(length)`;
/// `rawstorage.alloc_raw_storage` spells it as `add_memory_pressure=True`.
/// A storage box is the raw representation — the collector registers
/// `size_of::<T>()` and never sees the container's own allocation — so it
/// takes the same report. Without it a `BufferedReader` buffer counts as the
/// 24 bytes of its `Vec` rather than the 128KB it holds, and a heap of them
/// never moves the major-collection threshold.
///
/// `size` is the payload alone. `incminimark.py raw_malloc_memory_pressure`'s
/// per-allocation term is added by the collector.
pub fn add_storage_memory_pressure(size: usize) {
    majit_gc::add_memory_pressure_estimate(size as isize);
}

/// GC-sweep destructor for a storage box built by
/// [`gc_alloc_storage_box::<T>`].
///
/// Runs `T`'s drop glue in place, reclaiming the container's owned heap buffer.
/// This is the generic form of `longobject.rs`'s
/// [`crate::longobject::bigint_destructor`], registered through
/// `TypeInfo::with_destructor` (or composed with a custom trace).
///
/// # Safety
///
/// `addr` must point at a live `T` payload allocated by
/// [`gc_alloc_storage_box::<T>`]. The collector must invoke this exactly once.
pub unsafe fn storage_box_destructor<T: 'static>(addr: usize) {
    unsafe { std::ptr::drop_in_place(addr as *mut T) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use std::mem::MaybeUninit;

    thread_local! {
        static DROP_COUNT: Cell<usize> = const { Cell::new(0) };
    }

    struct DropProbe(u32);

    impl Drop for DropProbe {
        fn drop(&mut self) {
            DROP_COUNT.with(|count| count.set(count.get() + 1));
        }
    }

    #[test]
    fn raw_fallback_round_trips_and_destructor_runs_drop_glue_once() {
        DROP_COUNT.with(|count| count.set(0));

        let storage = gc_alloc_storage_box(DropProbe(42), 0);
        assert!(!storage.is_null());
        unsafe {
            assert_eq!((*storage).0, 42);
            storage_box_destructor::<DropProbe>(storage as usize);

            // `tid == 0` guarantees the `malloc_raw` fallback, so reclaim the
            // allocation without running the already-invoked drop glue again.
            drop(Box::from_raw(storage.cast::<MaybeUninit<DropProbe>>()));
        }

        DROP_COUNT.with(|count| assert_eq!(count.get(), 1));
    }
}
