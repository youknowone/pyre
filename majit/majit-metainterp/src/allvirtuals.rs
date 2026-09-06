//! `compile.py AllVirtuals` — the GC object stored in `JITFRAME.jf_savedata`.
//!
//! RPython stores the virtual cache materialized by
//! `ResumeGuardForcedDescr.handle_async_forcing` on the deadframe itself.
//! Keeping the same owner is important: a virtualizable can move in the
//! nursery, whereas the deadframe remains the object whose failing guard will
//! consume the cache.

use std::sync::atomic::{AtomicU32, Ordering};

use majit_ir::GcRef;

const UNREGISTERED: u32 = u32::MAX;
static ALL_VIRTUALS_TYPE_ID: AtomicU32 = AtomicU32::new(UNREGISTERED);

#[repr(C, align(8))]
struct AllVirtuals {
    ptr_count: usize,
    int_count: usize,
    length: usize,
}

const ITEMS_OFFSET: usize = std::mem::size_of::<AllVirtuals>();

unsafe fn all_virtuals_trace(obj_addr: usize, visit: &mut dyn FnMut(*mut GcRef)) {
    let object = obj_addr as *mut AllVirtuals;
    let ptr_count = unsafe { (*object).ptr_count };
    let length = unsafe { (*object).length };
    assert!(
        ptr_count <= length,
        "AllVirtuals pointer count exceeds cache length"
    );
    let items = unsafe { (object as *mut u8).add(ITEMS_OFFSET) as *mut i64 };
    for index in 0..ptr_count {
        // The cache uses i64 slots on both targets; a wasm32 GCREF occupies
        // the low half, but successive references remain eight bytes apart.
        visit(unsafe { items.add(index) as *mut GcRef });
    }
}

/// `compile.py class AllVirtuals` translated as a GC struct with one trailing
/// Signed array.  The prefix says which words are GCREFs; the custom tracer
/// visits exactly that prefix and leaves the integer cache unboxed.
pub fn type_info() -> majit_gc::trace::TypeInfo {
    majit_gc::trace::TypeInfo::varsize_with_custom_trace(
        ITEMS_OFFSET,
        std::mem::size_of::<i64>(),
        std::mem::offset_of!(AllVirtuals, length),
        all_virtuals_trace,
    )
}

/// Publish the type id assigned while the frontend collector is built.
pub fn set_type_id(type_id: u32) {
    ALL_VIRTUALS_TYPE_ID.store(type_id, Ordering::Release);
}

fn type_id() -> u32 {
    let type_id = ALL_VIRTUALS_TYPE_ID.load(Ordering::Acquire);
    assert_ne!(
        type_id, UNREGISTERED,
        "AllVirtuals GC type must be registered before async forcing",
    );
    type_id
}

/// Allocate and initialize the object saved by `handle_async_forcing`.
///
/// `force_from_resumedata` returns its pointer cache in a contiguous vector.
/// Root that whole span across the collecting `malloc_fast`, exactly as the
/// GC transformer roots the live `all_virtuals` list elements around the
/// `AllVirtuals` allocation upstream.
pub fn allocate(ptrs: Vec<i64>, ints: Vec<i64>) -> GcRef {
    let mut roots: Vec<GcRef> = ptrs.iter().map(|&value| GcRef(value as usize)).collect();
    let length = ptrs
        .len()
        .checked_add(ints.len())
        .expect("AllVirtuals cache length overflow");
    let payload_size = ITEMS_OFFSET
        .checked_add(
            length
                .checked_mul(std::mem::size_of::<i64>())
                .expect("AllVirtuals payload size overflow"),
        )
        .expect("AllVirtuals payload size overflow");
    let mut needs_write_barrier = false;
    let object = unsafe {
        majit_gc::alloc_fast_nursery_collecting_typed_roots(
            type_id(),
            payload_size,
            roots.as_mut_ptr(),
            roots.len(),
            &mut needs_write_barrier,
        )
    };
    assert!(!object.is_null(), "AllVirtuals allocation failed");
    if needs_write_barrier && !ptrs.is_empty() {
        majit_gc::gc_write_barrier(object);
    }
    unsafe {
        let header = object.0 as *mut AllVirtuals;
        (*header).ptr_count = ptrs.len();
        (*header).int_count = ints.len();
        (*header).length = length;
        let items = (header as *mut u8).add(ITEMS_OFFSET) as *mut i64;
        for (index, value) in roots.iter().enumerate() {
            *items.add(index) = value.0 as i64;
        }
        std::ptr::copy_nonoverlapping(ints.as_ptr(), items.add(ptrs.len()), ints.len());
    }
    object
}

/// `ResumeGuardForcedDescr.handle_fail`: reveal the cache stored in
/// `deadframe.jf_savedata` for `resume_in_blackhole`.
pub fn reveal(object: GcRef) -> Option<(Vec<i64>, Vec<i64>)> {
    if object.is_null() {
        return None;
    }
    unsafe {
        let header = object.0 as *const AllVirtuals;
        let ptr_count = (*header).ptr_count;
        let int_count = (*header).int_count;
        assert_eq!(
            (*header).length,
            ptr_count + int_count,
            "corrupt AllVirtuals cache length",
        );
        let items = (header as *const u8).add(ITEMS_OFFSET) as *const i64;
        Some((
            std::slice::from_raw_parts(items, ptr_count).to_vec(),
            std::slice::from_raw_parts(items.add(ptr_count), int_count).to_vec(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trace_visits_only_pointer_cache_with_i64_slot_stride() {
        assert_eq!(ITEMS_OFFSET % std::mem::align_of::<i64>(), 0);
        let mut storage = vec![0u64; ITEMS_OFFSET / 8 + 4];
        let object = storage.as_mut_ptr() as *mut AllVirtuals;
        unsafe {
            (*object).ptr_count = 2;
            (*object).int_count = 2;
            (*object).length = 4;
            let items = (object as *mut u8).add(ITEMS_OFFSET) as *mut i64;
            *items = 0x1000;
            *items.add(1) = 0x2000;
            *items.add(2) = 42;
            *items.add(3) = -1;
            let mut slots = Vec::new();
            all_virtuals_trace(object as usize, &mut |slot| {
                slots.push(slot as usize);
                (*slot).0 += 0x100;
            });
            assert_eq!(slots, vec![items as usize, items.add(1) as usize]);
            assert_eq!(
                reveal(GcRef(object as usize)),
                Some((vec![0x1100, 0x2100], vec![42, -1]))
            );
        }
    }
}
