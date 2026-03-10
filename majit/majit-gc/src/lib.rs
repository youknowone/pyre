/// GC traits and interfaces for the JIT.
///
/// The GC subsystem provides:
/// 1. Object allocation (nursery bump-pointer + old gen)
/// 2. Write barrier insertion
/// 3. GC-aware IR rewriting (NEW_* → inline nursery allocation)
/// 4. Stack maps for compiled code
///
/// Reference: rpython/memory/gc/incminimark.py, rpython/jit/backend/llsupport/gc.py

use majit_ir::{GcRef, Op};

pub mod collector;
pub mod header;
pub mod nursery;
pub mod oldgen;
pub mod trace;

/// GC flags stored in object headers.
///
/// From incminimark.py GCFLAG_* constants.
pub mod flags {
    /// Old object that may point to young objects (needs write barrier check).
    pub const TRACK_YOUNG_PTRS: u64 = 1 << 0;
    /// Prebuilt object with no heap pointers yet.
    pub const NO_HEAP_PTRS: u64 = 1 << 1;
    /// Marked as visited during major collection.
    pub const VISITED: u64 = 1 << 2;
    /// Has a shadow copy for identity hash.
    pub const HAS_SHADOW: u64 = 1 << 3;
    /// Finalizer ordering.
    pub const FINALIZATION_ORDERING: u64 = 1 << 4;
    /// Has card marking enabled (for large arrays).
    pub const HAS_CARDS: u64 = 1 << 5;
    /// At least one card is marked.
    pub const CARDS_SET: u64 = 1 << 6;
    /// Pinned in nursery (won't be moved).
    pub const PINNED: u64 = 1 << 7;
}

/// Write barrier descriptor — information the JIT needs to emit write barrier checks.
///
/// From rpython/jit/backend/llsupport/gc.py WriteBarrierDescr.
#[derive(Debug, Clone)]
pub struct WriteBarrierDescr {
    /// The flag bit to test in the object header.
    pub jit_wb_if_flag: u64,
    /// Byte offset of the flag byte in the header.
    pub jit_wb_if_flag_byteofs: usize,
    /// Single-byte mask to test.
    pub jit_wb_if_flag_singlebyte: u8,
    /// Flag for card marking.
    pub jit_wb_cards_set: u64,
    /// Shift for computing card index.
    pub jit_wb_card_page_shift: u32,
}

/// GC allocator interface.
///
/// Provides allocation and collection primitives.
pub trait GcAllocator: Send {
    /// Allocate a fixed-size object in the nursery.
    fn alloc_nursery(&mut self, size: usize) -> GcRef;

    /// Allocate a variable-size object (array/string).
    fn alloc_varsize(&mut self, base_size: usize, item_size: usize, length: usize) -> GcRef;

    /// Perform a write barrier check on `obj`.
    /// Must be called before storing a GC reference into `obj`.
    fn write_barrier(&mut self, obj: GcRef);

    /// Trigger a minor (nursery) collection.
    fn collect_nursery(&mut self);

    /// Trigger a full collection.
    fn collect_full(&mut self);

    /// Current nursery free pointer.
    fn nursery_free(&self) -> *mut u8;

    /// Nursery top (end) pointer.
    fn nursery_top(&self) -> *const u8;

    /// Maximum size for nursery allocation (larger objects go to old gen directly).
    fn max_nursery_object_size(&self) -> usize;
}

/// GC rewriter — transforms IR operations for GC integration.
///
/// Converts high-level NEW_*/SETFIELD_GC operations into:
/// - Inline nursery bump-pointer allocation (CALL_MALLOC_NURSERY)
/// - Write barrier conditional calls (COND_CALL_GC_WB)
///
/// Reference: rpython/jit/backend/llsupport/rewrite.py GcRewriterAssembler.
pub trait GcRewriter: Send {
    /// Rewrite a list of operations, inserting GC-aware code.
    fn rewrite_for_gc(&self, ops: &[Op]) -> Vec<Op>;
}

/// Stack map — records which frame slots contain GC references at a safepoint.
///
/// At each guard (potential GC safepoint), the backend records a stack map
/// so the GC can find all live references in compiled code.
#[derive(Debug, Clone)]
pub struct GcMap {
    /// Bitmap: bit N is set if frame slot N contains a GC reference.
    pub ref_bitmap: Vec<u64>,
}

impl GcMap {
    pub fn new() -> Self {
        GcMap { ref_bitmap: Vec::new() }
    }

    pub fn set_ref(&mut self, slot: usize) {
        let word = slot / 64;
        let bit = slot % 64;
        if word >= self.ref_bitmap.len() {
            self.ref_bitmap.resize(word + 1, 0);
        }
        self.ref_bitmap[word] |= 1u64 << bit;
    }

    pub fn is_ref(&self, slot: usize) -> bool {
        let word = slot / 64;
        let bit = slot % 64;
        if word >= self.ref_bitmap.len() {
            return false;
        }
        (self.ref_bitmap[word] >> bit) & 1 != 0
    }
}

impl Default for GcMap {
    fn default() -> Self {
        Self::new()
    }
}
