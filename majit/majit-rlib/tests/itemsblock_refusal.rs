//! `NoRoute` vs `Failed`, the split the digit-block allocator reads.
//!
//! Every allocation entry point answers both states with `GcRef(0)`, so the
//! null alone cannot say which one happened, and they are not interchangeable:
//! with nothing owning the heap `std::alloc` *is* the whole heap and a block
//! from it is correct, while a collector that owns the heap and then refuses
//! must not be answered with a block outside it — the caller would store an
//! untraced pointer into a field the collector walks. What separates them is
//! whether an allocator is installed, which is what
//! `try_alloc_typed_items_block_nursery` consults before taking its
//! `std::alloc` path.
//!
//! Its own test binary because installing an allocator writes one
//! process-global cell (`ACTIVE_ALLOC_NURSERY_TYPED`), which every other test
//! in the same binary would then read from a thread holding no GIL.

use majit_gc::{GcAllocOutcome, gc_allocator_installed, set_active_alloc_nursery_typed};
use majit_ir::GcRef;

fn unused_allocator(_type_id: u32, _payload_size: usize) -> GcRef {
    // Never called: `classify` only reads whether a hook is present. Allocating
    // through it would enter the real collector, which needs the GIL.
    unreachable!("the classification never allocates")
}

#[test]
fn a_null_means_no_route_only_while_nothing_owns_the_heap() {
    set_active_alloc_nursery_typed(None);
    assert!(!gc_allocator_installed());
    assert_eq!(GcAllocOutcome::classify(GcRef(0)), GcAllocOutcome::NoRoute);
    // `NoRoute` is the sole licence for the caller's raw path.
    assert!(GcAllocOutcome::NoRoute.allocated_or_abort(32).is_none());

    set_active_alloc_nursery_typed(Some(unused_allocator));
    assert!(gc_allocator_installed());
    assert_eq!(GcAllocOutcome::classify(GcRef(0)), GcAllocOutcome::Failed);
    set_active_alloc_nursery_typed(None);
}

#[test]
fn a_non_null_is_allocated_regardless_of_what_is_installed() {
    let mut probe = 0u8;
    let raw = &mut probe as *mut u8;
    assert_eq!(
        GcAllocOutcome::classify(GcRef(raw as usize)),
        GcAllocOutcome::Allocated(raw)
    );
}
