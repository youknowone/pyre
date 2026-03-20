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
pub mod rewrite;
pub mod shadow_stack;
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

    /// Allocate a fixed-size object with a known GC type id.
    fn alloc_nursery_typed(&mut self, type_id: u32, size: usize) -> GcRef {
        let _ = type_id;
        self.alloc_nursery(size)
    }

    /// Allocate a fixed-size object without triggering collection.
    ///
    /// Implementations may fall back to old-gen allocation when the nursery
    /// cannot satisfy the request.
    fn alloc_nursery_no_collect(&mut self, size: usize) -> GcRef;

    /// Allocate a variable-size object (array/string).
    fn alloc_varsize(&mut self, base_size: usize, item_size: usize, length: usize) -> GcRef;

    /// Allocate a variable-size object with a known GC type id.
    fn alloc_varsize_typed(
        &mut self,
        type_id: u32,
        base_size: usize,
        item_size: usize,
        length: usize,
    ) -> GcRef {
        let _ = type_id;
        self.alloc_varsize(base_size, item_size, length)
    }

    /// Allocate a variable-size object without triggering collection.
    ///
    /// Implementations may fall back to old-gen allocation when the nursery
    /// cannot satisfy the request.
    fn alloc_varsize_no_collect(
        &mut self,
        base_size: usize,
        item_size: usize,
        length: usize,
    ) -> GcRef;

    /// Perform a write barrier check on `obj`.
    /// Must be called before storing a GC reference into `obj`.
    fn write_barrier(&mut self, obj: GcRef);

    /// Trigger a minor (nursery) collection.
    fn collect_nursery(&mut self);

    /// Trigger a full collection.
    fn collect_full(&mut self);

    /// Register a stack/root slot that contains a `GcRef`.
    ///
    /// The pointer must remain valid until removed. Backends use this to
    /// expose shadow-root buffers around collecting helper calls.
    ///
    /// # Safety
    /// The caller must ensure the slot remains valid for the duration of the
    /// registration.
    unsafe fn add_root(&mut self, _root: *mut GcRef) {}

    /// Remove a previously-registered root slot.
    fn remove_root(&mut self, _root: *mut GcRef) {}

    /// Current nursery free pointer.
    fn nursery_free(&self) -> *mut u8;

    /// Nursery top (end) pointer.
    fn nursery_top(&self) -> *const u8;

    /// Maximum size for nursery allocation (larger objects go to old gen directly).
    fn max_nursery_object_size(&self) -> usize;

    /// Fast-path write barrier for JIT-compiled code.
    ///
    /// Adds the object directly to the remembered set. The JIT has already
    /// performed the inline flag test (COND_CALL_GC_WB) and determined
    /// that the barrier is needed.
    fn jit_remember_young_pointer(&mut self, obj: GcRef) {
        self.write_barrier(obj);
    }

    /// Whether the GC supports optimized conditional write barriers.
    ///
    /// When true, the JIT emits COND_CALL_GC_WB (inline flag test +
    /// conditional call) instead of a full barrier call.
    fn can_optimize_cond_call(&self) -> bool {
        false
    }

    /// Perform one incremental GC step at a JIT safepoint.
    /// Returns true if any GC work was done.
    fn gc_step(&mut self) -> bool {
        false
    }

    /// Free memory associated with invalidated JIT compiled code.
    fn jit_free(&mut self, _code_ptr: usize, _size: usize) {}

    /// Pin a nursery object so it won't move during minor collection.
    /// Returns true if pinning succeeded.
    fn pin(&mut self, _obj: GcRef) -> bool {
        false
    }

    /// Unpin a previously pinned object.
    fn unpin(&mut self, _obj: GcRef) {}

    /// Check if an object is pinned.
    fn is_pinned(&self, _obj: GcRef) -> bool {
        false
    }
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
        GcMap {
            ref_bitmap: Vec::new(),
        }
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
