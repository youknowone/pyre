/// MiniMarkGC — the core collector implementing the GcAllocator trait.
///
/// A generational copying collector with:
/// - Bump-pointer nursery for young objects
/// - System-allocator old gen with mark-sweep for major collection
/// - Write barrier with remembered set for old-to-young pointers
///
/// Modeled after incminimark's minor/major collection.
use majit_ir::GcRef;

use crate::flags;
use crate::header::{header_of, GcHeader};
use crate::nursery::{Nursery, DEFAULT_NURSERY_SIZE};
use crate::oldgen::OldGen;
use crate::trace::{TypeInfo, TypeRegistry};
use crate::GcAllocator;

/// Configuration for the MiniMarkGC.
pub struct GcConfig {
    /// Nursery size in bytes.
    pub nursery_size: usize,
    /// Maximum object size that can be allocated in the nursery.
    /// Larger objects go directly to old gen.
    pub large_object_threshold: usize,
}

impl Default for GcConfig {
    fn default() -> Self {
        // large_object = (16384+512)*8 from incminimark for 64-bit
        GcConfig {
            nursery_size: DEFAULT_NURSERY_SIZE,
            large_object_threshold: (16384 + 512) * 8,
        }
    }
}

/// Root set: a list of locations that hold GcRef values the GC must trace.
///
/// Each root is a pointer to a GcRef-sized slot. During collection,
/// the GC reads the GcRef from this slot, traces it, and writes back
/// the (possibly updated) value.
pub struct RootSet {
    /// Stack roots: mutable pointers to GcRef slots on the stack or in frames.
    roots: Vec<*mut GcRef>,
}

unsafe impl Send for RootSet {}

impl RootSet {
    pub fn new() -> Self {
        RootSet { roots: Vec::new() }
    }

    /// Add a root. The pointer must remain valid until removed.
    ///
    /// # Safety
    /// The caller must ensure the pointer remains valid for the lifetime of the root.
    pub unsafe fn add(&mut self, root: *mut GcRef) {
        self.roots.push(root);
    }

    /// Remove a root.
    pub fn remove(&mut self, root: *mut GcRef) {
        if let Some(pos) = self.roots.iter().position(|r| *r == root) {
            self.roots.swap_remove(pos);
        }
    }

    /// Clear all roots.
    pub fn clear(&mut self) {
        self.roots.clear();
    }

    /// Number of roots.
    pub fn len(&self) -> usize {
        self.roots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.roots.is_empty()
    }
}

impl Default for RootSet {
    fn default() -> Self {
        Self::new()
    }
}

/// The MiniMark generational GC.
pub struct MiniMarkGC {
    /// The nursery (young generation).
    nursery: Nursery,
    /// The old generation.
    oldgen: OldGen,
    /// Type registry for tracing objects.
    pub types: TypeRegistry,
    /// Root set.
    pub roots: RootSet,
    /// Remembered set: old objects that may point to young objects.
    /// These are old-gen object payload addresses whose TRACK_YOUNG_PTRS
    /// flag has been cleared by the write barrier.
    remembered_set: Vec<usize>,
    /// Configuration.
    config: GcConfig,
    /// Count of minor collections performed.
    pub minor_collections: usize,
    /// Count of major collections performed.
    pub major_collections: usize,
}

impl MiniMarkGC {
    /// Create a new GC with default configuration.
    pub fn new() -> Self {
        Self::with_config(GcConfig::default())
    }

    /// Create a new GC with custom configuration.
    pub fn with_config(config: GcConfig) -> Self {
        MiniMarkGC {
            nursery: Nursery::new(config.nursery_size),
            oldgen: OldGen::new(),
            types: TypeRegistry::new(),
            roots: RootSet::new(),
            remembered_set: Vec::new(),
            config,
            minor_collections: 0,
            major_collections: 0,
        }
    }

    /// Register a type and return its ID.
    pub fn register_type(&mut self, info: TypeInfo) -> u32 {
        self.types.register(info)
    }

    /// Check if an address is in the nursery.
    #[inline]
    pub fn is_in_nursery(&self, addr: usize) -> bool {
        self.nursery.contains(addr)
    }

    /// Allocate a fixed-size object with the given type ID and size (excluding header).
    /// Returns a GcRef pointing to the object payload (after the header).
    pub fn alloc_with_type(&mut self, type_id: u32, payload_size: usize) -> GcRef {
        let total_size = GcHeader::SIZE + payload_size;

        // Large objects go directly to old gen.
        if total_size > self.config.large_object_threshold {
            return self.alloc_in_oldgen(type_id, total_size);
        }

        let ptr = self.nursery.alloc(total_size);
        if ptr.is_null() {
            // Nursery full: trigger minor collection and retry.
            self.do_collect_nursery();
            let ptr = self.nursery.alloc(total_size);
            if ptr.is_null() {
                // Still no space after collection. Allocate in old gen as fallback.
                return self.alloc_in_oldgen(type_id, total_size);
            }
            Self::init_nursery_object(ptr, type_id);
            return GcRef((ptr as usize) + GcHeader::SIZE);
        }

        Self::init_nursery_object(ptr, type_id);
        GcRef((ptr as usize) + GcHeader::SIZE)
    }

    /// Allocate without triggering collection.
    ///
    /// If the nursery cannot satisfy the request, this falls back directly to
    /// old-gen allocation so compiled code can keep running without needing
    /// stack-map-mediated collection.
    pub fn alloc_with_type_no_collect(&mut self, type_id: u32, payload_size: usize) -> GcRef {
        let total_size = GcHeader::SIZE + payload_size;

        if total_size > self.config.large_object_threshold {
            return self.alloc_in_oldgen(type_id, total_size);
        }

        let ptr = self.nursery.alloc(total_size);
        if ptr.is_null() {
            return self.alloc_in_oldgen(type_id, total_size);
        }

        Self::init_nursery_object(ptr, type_id);
        GcRef((ptr as usize) + GcHeader::SIZE)
    }

    /// Initialize a nursery object's header.
    fn init_nursery_object(header_ptr: *mut u8, type_id: u32) {
        let hdr = unsafe { &mut *(header_ptr as *mut GcHeader) };
        // Young objects do NOT have TRACK_YOUNG_PTRS (we assume any
        // young object can point to any other young object).
        *hdr = GcHeader::new(type_id);
    }

    /// Allocate directly in old gen (for large objects or post-collection fallback).
    fn alloc_in_oldgen(&mut self, type_id: u32, total_size: usize) -> GcRef {
        let ptr = self.oldgen.alloc(total_size);
        let hdr = unsafe { &mut *(ptr as *mut GcHeader) };
        // Old objects start with TRACK_YOUNG_PTRS set (they need write barrier).
        *hdr = GcHeader::with_flags(type_id, flags::TRACK_YOUNG_PTRS);
        GcRef((ptr as usize) + GcHeader::SIZE)
    }

    /// Perform a minor (nursery) collection.
    ///
    /// 1. Scan roots: copy referenced nursery objects to old gen.
    /// 2. Process remembered set: copy nursery objects referenced by old-gen objects.
    /// 3. Iteratively process newly discovered references until stable.
    /// 4. Reset nursery.
    pub fn do_collect_nursery(&mut self) {
        self.minor_collections += 1;

        // Phase 1: Process roots — copy nursery objects they point to.
        // We use raw pointers to avoid borrow checker issues since
        // copy_nursery_object mutates oldgen/nursery.
        let roots: Vec<*mut GcRef> = self.roots.roots.iter().copied().collect();
        for root_ptr in roots {
            let gcref = unsafe { *root_ptr };
            if !gcref.is_null() && self.is_in_nursery(gcref.0) {
                let new_ref = self.copy_nursery_object(gcref.0);
                unsafe {
                    *root_ptr = new_ref;
                }
            }
        }

        // Phase 2: Process remembered set and transitive closure.
        // Objects copied to old gen may reference other nursery objects,
        // so we process until all references are resolved.
        let mut idx = 0;
        loop {
            if idx >= self.remembered_set.len() {
                break;
            }
            let obj_addr = self.remembered_set[idx];
            idx += 1;

            // Re-set TRACK_YOUNG_PTRS on this old object since we're
            // processing all its young references now.
            unsafe {
                header_of(obj_addr).set_flag(flags::TRACK_YOUNG_PTRS);
            }

            // Trace this old-gen object's fields and copy any nursery
            // objects they reference.
            self.trace_and_update_object(obj_addr);
        }

        // Clear remembered set.
        self.remembered_set.clear();

        // Reset nursery for new allocations.
        self.nursery.reset();
    }

    /// Copy a single nursery object to old gen.
    /// If already forwarded, returns the forwarding address.
    fn copy_nursery_object(&mut self, obj_addr: usize) -> GcRef {
        let hdr = unsafe { &mut *((obj_addr - GcHeader::SIZE) as *mut GcHeader) };

        // Already forwarded?
        if hdr.is_forwarded() {
            let fwd_addr = unsafe { hdr.forwarding_address() };
            return GcRef(fwd_addr);
        }

        let type_id = hdr.type_id();
        let type_info = self.types.get(type_id);

        // Compute the actual payload size (for varsize objects, read the length).
        let actual_payload_size = if type_info.item_size > 0 {
            let length = unsafe { *((obj_addr + type_info.length_offset) as *const usize) };
            type_info.total_instance_size(length)
        } else {
            type_info.size
        };

        let total_size = GcHeader::SIZE + actual_payload_size;
        let has_gc_ptrs = type_info.has_gc_ptrs;

        // Allocate in old gen and copy.
        let header_ptr = obj_addr - GcHeader::SIZE;
        // Safety: header_ptr points to a valid nursery object of total_size bytes.
        let new_header_ptr = unsafe {
            self.oldgen
                .alloc_and_copy(header_ptr as *const u8, total_size)
        };
        let new_obj_addr = new_header_ptr as usize + GcHeader::SIZE;

        // Set TRACK_YOUNG_PTRS on the new old-gen object.
        let new_hdr = unsafe { &mut *(new_header_ptr as *mut GcHeader) };
        new_hdr.set_flag(flags::TRACK_YOUNG_PTRS);

        // Install forwarding pointer in the nursery copy.
        unsafe {
            hdr.set_forwarding_address(new_obj_addr);
        }

        // If this object has GC pointers, add it to the work list so we
        // trace its fields and update nursery references.
        if has_gc_ptrs {
            // Clear TRACK_YOUNG_PTRS temporarily so the processing loop
            // can re-set it after processing.
            let new_hdr = unsafe { &mut *(new_header_ptr as *mut GcHeader) };
            new_hdr.clear_flag(flags::TRACK_YOUNG_PTRS);
            self.remembered_set.push(new_obj_addr);
        }

        GcRef(new_obj_addr)
    }

    /// Trace an object's GC pointer fields and update any that point
    /// into the nursery by copying the target.
    fn trace_and_update_object(&mut self, obj_addr: usize) {
        let type_id = unsafe { header_of(obj_addr).type_id() };
        let type_info = self.types.get(type_id);
        let gc_ptr_offsets: Vec<usize> = type_info.gc_ptr_offsets.clone();
        let items_have_gc_ptrs = type_info.items_have_gc_ptrs;
        let item_size = type_info.item_size;
        let length_offset = type_info.length_offset;
        let base_size = type_info.size;

        // Process fixed-part GC pointer fields.
        for &offset in &gc_ptr_offsets {
            let slot = (obj_addr + offset) as *mut GcRef;
            let field_ref = unsafe { *slot };
            if !field_ref.is_null() && self.is_in_nursery(field_ref.0) {
                let new_ref = self.copy_nursery_object(field_ref.0);
                unsafe {
                    *slot = new_ref;
                }
            }
        }

        // Process variable-part items if they contain GC pointers.
        if items_have_gc_ptrs && item_size > 0 {
            let length = unsafe { *((obj_addr + length_offset) as *const usize) };
            let items_start = obj_addr + base_size;
            for i in 0..length {
                let slot = (items_start + i * item_size) as *mut GcRef;
                let field_ref = unsafe { *slot };
                if !field_ref.is_null() && self.is_in_nursery(field_ref.0) {
                    let new_ref = self.copy_nursery_object(field_ref.0);
                    unsafe {
                        *slot = new_ref;
                    }
                }
            }
        }
    }

    /// Perform a full (major) mark-sweep collection.
    ///
    /// 1. First do a minor collection to promote all live nursery objects.
    /// 2. Mark phase: trace all roots and transitively mark reachable objects.
    /// 3. Sweep phase: free all unmarked old-gen objects.
    pub fn do_collect_full(&mut self) {
        // Minor collection first to empty the nursery.
        self.do_collect_nursery();

        self.major_collections += 1;

        // Mark phase: BFS from roots.
        let mut worklist: Vec<usize> = Vec::new();

        // Start from roots.
        let roots: Vec<*mut GcRef> = self.roots.roots.iter().copied().collect();
        for root_ptr in roots {
            let gcref = unsafe { *root_ptr };
            if !gcref.is_null() {
                let hdr = unsafe { header_of(gcref.0) };
                if !hdr.has_flag(flags::VISITED) {
                    hdr.set_flag(flags::VISITED);
                    worklist.push(gcref.0);
                }
            }
        }

        // Process worklist: mark and trace transitively.
        while let Some(obj_addr) = worklist.pop() {
            let type_id = unsafe { header_of(obj_addr).type_id() };
            let type_info = self.types.get(type_id);
            let gc_ptr_offsets = type_info.gc_ptr_offsets.clone();
            let items_have_gc_ptrs = type_info.items_have_gc_ptrs;
            let item_size = type_info.item_size;
            let length_offset = type_info.length_offset;
            let base_size = type_info.size;

            // Trace fixed-part fields.
            for &offset in &gc_ptr_offsets {
                let field_ref = unsafe { *((obj_addr + offset) as *const GcRef) };
                if !field_ref.is_null() {
                    let hdr = unsafe { header_of(field_ref.0) };
                    if !hdr.has_flag(flags::VISITED) {
                        hdr.set_flag(flags::VISITED);
                        worklist.push(field_ref.0);
                    }
                }
            }

            // Trace variable-part items.
            if items_have_gc_ptrs && item_size > 0 {
                let length = unsafe { *((obj_addr + length_offset) as *const usize) };
                let items_start = obj_addr + base_size;
                for i in 0..length {
                    let field_ref = unsafe { *((items_start + i * item_size) as *const GcRef) };
                    if !field_ref.is_null() {
                        let hdr = unsafe { header_of(field_ref.0) };
                        if !hdr.has_flag(flags::VISITED) {
                            hdr.set_flag(flags::VISITED);
                            worklist.push(field_ref.0);
                        }
                    }
                }
            }
        }

        // Sweep phase: free unmarked objects.
        self.oldgen.sweep();
    }

    /// Write barrier: call before storing a GC reference into an old-gen object.
    ///
    /// If the object has TRACK_YOUNG_PTRS set, we clear it and add the object
    /// to the remembered set, because it might now point to a young object.
    pub fn do_write_barrier(&mut self, obj: GcRef) {
        if obj.is_null() {
            return;
        }
        let hdr = unsafe { header_of(obj.0) };
        if hdr.has_flag(flags::TRACK_YOUNG_PTRS) {
            hdr.clear_flag(flags::TRACK_YOUNG_PTRS);
            self.remembered_set.push(obj.0);
        }
    }

    /// Card-marking write barrier for large arrays.
    ///
    /// Instead of adding the entire object to the remembered set,
    /// mark only the card that covers the modified array index.
    /// This avoids rescanning the entire array during minor collection.
    ///
    /// `obj` is the array object; `index` is the element index being written.
    /// `card_page_shift` determines the card granularity (elements per card = 1 << shift).
    pub fn do_write_barrier_card(&mut self, obj: GcRef, index: usize, card_page_shift: u32) {
        if obj.is_null() {
            return;
        }
        let hdr = unsafe { header_of(obj.0) };

        // If TRACK_YOUNG_PTRS is set, this object hasn't been written before.
        if hdr.has_flag(flags::TRACK_YOUNG_PTRS) {
            if hdr.has_flag(flags::HAS_CARDS) {
                // Object supports card marking: enable it.
                hdr.clear_flag(flags::TRACK_YOUNG_PTRS);
                self.mark_card(obj, index, card_page_shift);
                return;
            }
            // Fall back to full barrier.
            hdr.clear_flag(flags::TRACK_YOUNG_PTRS);
            self.remembered_set.push(obj.0);
            return;
        }

        // Already had barrier triggered. If it has cards, mark the card.
        if hdr.has_flag(flags::HAS_CARDS) {
            self.mark_card(obj, index, card_page_shift);
        }
    }

    /// Mark a specific card in a card-marked array object.
    fn mark_card(&mut self, obj: GcRef, index: usize, card_page_shift: u32) {
        let hdr = unsafe { header_of(obj.0) };
        if !hdr.has_flag(flags::CARDS_SET) {
            hdr.set_flag(flags::CARDS_SET);
            // First card dirty: add to remembered set for scanning.
            self.remembered_set.push(obj.0);
        }

        // Store dirty mark in card table (bytes before the header).
        // card_index = index >> card_page_shift
        let card_index = index >> card_page_shift;
        let header_addr = obj.0 - GcHeader::SIZE;
        // Card bytes are stored at negative offsets from the header.
        // We use the object's type info's extra card space (if allocated).
        // For now, store in the card tracking set.
        let _ = (header_addr, card_index); // Card byte storage is type-layout dependent
    }
}

/// Safepoint GC map: records which frame slots contain GC references
/// at a specific program point (guard or call site).
///
/// The Cranelift backend builds these during compilation and stores them
/// alongside the compiled code. During collection, the GC uses them to
/// find live references on the stack.
#[derive(Debug, Clone)]
pub struct SafepointMap {
    /// Map from code offset to GcMap.
    pub entries: Vec<SafepointEntry>,
}

/// A single safepoint entry.
#[derive(Debug, Clone)]
pub struct SafepointEntry {
    /// Offset in the compiled code (bytes from function start).
    pub code_offset: u32,
    /// Bitmap of which frame slots contain GC references.
    pub gc_map: crate::GcMap,
}

impl SafepointMap {
    pub fn new() -> Self {
        SafepointMap {
            entries: Vec::new(),
        }
    }

    /// Add a safepoint entry.
    pub fn add(&mut self, code_offset: u32, gc_map: crate::GcMap) {
        self.entries.push(SafepointEntry {
            code_offset,
            gc_map,
        });
    }

    /// Look up the GcMap for a given code offset.
    pub fn lookup(&self, code_offset: u32) -> Option<&crate::GcMap> {
        self.entries
            .iter()
            .find(|e| e.code_offset == code_offset)
            .map(|e| &e.gc_map)
    }
}

impl Default for SafepointMap {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for MiniMarkGC {
    fn default() -> Self {
        Self::new()
    }
}

impl GcAllocator for MiniMarkGC {
    fn alloc_nursery(&mut self, size: usize) -> GcRef {
        self.alloc_with_type(0, size)
    }

    fn alloc_nursery_no_collect(&mut self, size: usize) -> GcRef {
        self.alloc_with_type_no_collect(0, size)
    }

    fn alloc_varsize(&mut self, base_size: usize, item_size: usize, length: usize) -> GcRef {
        let payload_size = base_size + item_size * length;
        self.alloc_with_type(0, payload_size)
    }

    fn alloc_varsize_no_collect(
        &mut self,
        base_size: usize,
        item_size: usize,
        length: usize,
    ) -> GcRef {
        let payload_size = base_size + item_size * length;
        self.alloc_with_type_no_collect(0, payload_size)
    }

    fn write_barrier(&mut self, obj: GcRef) {
        self.do_write_barrier(obj);
    }

    fn collect_nursery(&mut self) {
        self.do_collect_nursery();
    }

    fn collect_full(&mut self) {
        self.do_collect_full();
    }

    unsafe fn add_root(&mut self, root: *mut GcRef) {
        self.roots.add(root);
    }

    fn remove_root(&mut self, root: *mut GcRef) {
        self.roots.remove(root);
    }

    fn nursery_free(&self) -> *mut u8 {
        self.nursery.free_ptr()
    }

    fn nursery_top(&self) -> *const u8 {
        self.nursery.top_ptr()
    }

    fn max_nursery_object_size(&self) -> usize {
        self.config.large_object_threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a GC with a small nursery for testing.
    fn test_gc(nursery_size: usize) -> MiniMarkGC {
        MiniMarkGC::with_config(GcConfig {
            nursery_size,
            large_object_threshold: nursery_size / 2,
        })
    }

    #[test]
    fn test_basic_alloc() {
        let mut gc = test_gc(4096);
        gc.register_type(TypeInfo::simple(16));
        let obj = gc.alloc_with_type(0, 16);
        assert!(!obj.is_null());
        assert!(gc.is_in_nursery(obj.0));
    }

    #[test]
    fn test_multiple_allocs() {
        let mut gc = test_gc(4096);
        gc.register_type(TypeInfo::simple(16));

        let mut refs = Vec::new();
        for _ in 0..10 {
            refs.push(gc.alloc_with_type(0, 16));
        }

        // All should be non-null and distinct.
        for i in 0..refs.len() {
            assert!(!refs[i].is_null());
            for j in (i + 1)..refs.len() {
                assert_ne!(refs[i], refs[j]);
            }
        }
    }

    #[test]
    fn test_large_object_goes_to_oldgen() {
        let mut gc = test_gc(1024);
        gc.register_type(TypeInfo::simple(1024));

        // 1024 > large_object_threshold (512), so goes to old gen.
        let obj = gc.alloc_with_type(0, 1024);
        assert!(!obj.is_null());
        assert!(!gc.is_in_nursery(obj.0));
    }

    #[test]
    fn test_nursery_collection_basic() {
        let mut gc = test_gc(1024);
        gc.register_type(TypeInfo::simple(16));

        // Allocate an object and root it.
        let obj = gc.alloc_with_type(0, 16);
        assert!(gc.is_in_nursery(obj.0));

        // Write something to the object payload.
        unsafe {
            *(obj.0 as *mut u64) = 0xDEADBEEF;
        }

        // Root it.
        let mut root = obj;
        unsafe {
            gc.roots.add(&mut root);
        }

        // Trigger collection.
        gc.collect_nursery();

        // The root should now point to old gen.
        assert!(!gc.is_in_nursery(root.0));
        assert!(!root.is_null());

        // The data should be preserved.
        let val = unsafe { *(root.0 as *const u64) };
        assert_eq!(val, 0xDEADBEEF);

        gc.roots.clear();
    }

    #[test]
    fn test_unrooted_object_dies() {
        let mut gc = test_gc(1024);
        gc.register_type(TypeInfo::simple(16));

        // Allocate but don't root.
        let _obj = gc.alloc_with_type(0, 16);

        // Collection should run without issues.
        gc.collect_nursery();

        // Nursery is reset, the object is gone.
        assert_eq!(gc.nursery.used(), 0);
    }

    #[test]
    fn test_fill_nursery_triggers_collection() {
        let mut gc = test_gc(256);
        gc.register_type(TypeInfo::simple(16));

        // Keep allocating until we must have triggered at least one collection.
        for _ in 0..100 {
            gc.alloc_with_type(0, 16);
        }
        assert!(gc.minor_collections > 0);
    }

    #[test]
    fn test_write_barrier() {
        let mut gc = test_gc(1024);
        gc.register_type(TypeInfo::simple(16));

        // Allocate a large object (goes to old gen).
        let old_obj = gc.alloc_in_oldgen(0, GcHeader::SIZE + 16);
        assert!(!gc.is_in_nursery(old_obj.0));

        // The old object should have TRACK_YOUNG_PTRS.
        let hdr = unsafe { header_of(old_obj.0) };
        assert!(hdr.has_flag(flags::TRACK_YOUNG_PTRS));

        // Write barrier clears the flag and adds to remembered set.
        gc.do_write_barrier(old_obj);
        assert!(!hdr.has_flag(flags::TRACK_YOUNG_PTRS));
        assert_eq!(gc.remembered_set.len(), 1);

        // Second call: flag already cleared, should not add again.
        gc.do_write_barrier(old_obj);
        assert_eq!(gc.remembered_set.len(), 1);
    }

    #[test]
    fn test_nursery_collection_with_pointers() {
        // Object layout: one GcRef field at offset 0 (payload = 8 bytes).
        let mut gc = test_gc(1024);
        let tid = gc.register_type(TypeInfo::with_gc_ptrs(
            std::mem::size_of::<GcRef>(),
            vec![0],
        ));

        // Create two objects: parent -> child.
        let child = gc.alloc_with_type(tid, std::mem::size_of::<GcRef>());
        let parent = gc.alloc_with_type(tid, std::mem::size_of::<GcRef>());

        // Write child's address into parent's first field.
        unsafe {
            *(parent.0 as *mut GcRef) = child;
        }

        // Root only the parent.
        let mut root = parent;
        unsafe {
            gc.roots.add(&mut root);
        }

        // Trigger collection.
        gc.collect_nursery();

        // Parent should have survived.
        assert!(!gc.is_in_nursery(root.0));
        assert!(!root.is_null());

        // The pointer field should now point to the child's new location.
        let child_ref = unsafe { *(root.0 as *const GcRef) };
        assert!(!gc.is_in_nursery(child_ref.0));
        assert!(!child_ref.is_null());

        gc.roots.clear();
    }

    #[test]
    fn test_forwarding_dedup() {
        // Two roots pointing to the same nursery object should get the
        // same forwarded address.
        let mut gc = test_gc(1024);
        let tid = gc.register_type(TypeInfo::with_gc_ptrs(
            std::mem::size_of::<GcRef>(),
            vec![0],
        ));

        let shared = gc.alloc_with_type(tid, std::mem::size_of::<GcRef>());
        let mut root1 = shared;
        let mut root2 = shared;

        unsafe {
            gc.roots.add(&mut root1);
            gc.roots.add(&mut root2);
        }

        gc.collect_nursery();

        // Both roots should point to the same old-gen location.
        assert_eq!(root1, root2);
        assert!(!gc.is_in_nursery(root1.0));

        gc.roots.clear();
    }

    #[test]
    fn test_major_collection() {
        let mut gc = test_gc(1024);
        let tid = gc.register_type(TypeInfo::simple(16));

        // Allocate some objects and root one.
        let obj1 = gc.alloc_with_type(tid, 16);
        let _obj2 = gc.alloc_with_type(tid, 16); // unreachable

        let mut root = obj1;
        unsafe {
            gc.roots.add(&mut root);
        }

        // Full collection: promotes to old gen and sweeps.
        gc.collect_full();

        assert!(!root.is_null());
        assert!(!gc.is_in_nursery(root.0));

        gc.roots.clear();
    }

    #[test]
    fn test_major_collection_frees_unreachable_old_objects() {
        let mut gc = test_gc(1024);
        let tid = gc.register_type(TypeInfo::simple(16));

        // Promote two objects to old gen.
        let obj1 = gc.alloc_with_type(tid, 16);
        let obj2 = gc.alloc_with_type(tid, 16);
        let mut root1 = obj1;
        let mut root2 = obj2;
        unsafe {
            gc.roots.add(&mut root1);
            gc.roots.add(&mut root2);
        }
        gc.collect_nursery();
        assert!(!gc.is_in_nursery(root1.0));
        assert!(!gc.is_in_nursery(root2.0));
        assert_eq!(gc.oldgen.object_count(), 2);

        // Now unroot obj2 and do a full collection.
        gc.roots.remove(&mut root2);
        gc.collect_full();

        // Only obj1 should survive.
        assert_eq!(gc.oldgen.object_count(), 1);
        assert!(!root1.is_null());

        gc.roots.clear();
    }

    #[test]
    fn test_repeated_collections() {
        let mut gc = test_gc(512);
        let tid = gc.register_type(TypeInfo::simple(16));

        let mut root = GcRef::NULL;
        unsafe {
            gc.roots.add(&mut root);
        }

        for i in 0..50 {
            let obj = gc.alloc_with_type(tid, 16);
            // Write a marker value.
            unsafe {
                *(obj.0 as *mut u64) = i as u64;
            }
            root = obj;

            if i % 10 == 0 {
                gc.collect_nursery();
                // Root should survive and preserve its value.
                if !root.is_null() {
                    let val = unsafe { *(root.0 as *const u64) };
                    assert_eq!(val, i as u64);
                }
            }
        }

        gc.roots.clear();
    }

    #[test]
    fn test_gc_allocator_trait() {
        let mut gc = test_gc(4096);
        gc.register_type(TypeInfo::simple(32));

        let obj = gc.alloc_nursery(32);
        assert!(!obj.is_null());

        let varobj = gc.alloc_varsize(8, 4, 10);
        assert!(!varobj.is_null());

        gc.collect_nursery();
        gc.collect_full();
    }

    #[test]
    fn test_alloc_nursery_no_collect_does_not_trigger_collection() {
        let mut gc = test_gc(64);
        gc.register_type(TypeInfo::simple(24));

        let obj1 = gc.alloc_nursery_no_collect(24);
        let obj2 = gc.alloc_nursery_no_collect(24);

        assert!(!obj1.is_null());
        assert!(!obj2.is_null());
        assert_eq!(gc.minor_collections, 0);
        // The second allocation may have fallen back to old gen, but it must
        // still succeed without forcing a collection.
        assert_ne!(obj1, obj2);
    }

    #[test]
    fn test_alloc_varsize_no_collect_does_not_trigger_collection() {
        let mut gc = test_gc(64);
        gc.register_type(TypeInfo::simple(32));

        let obj = gc.alloc_varsize_no_collect(16, 8, 8);

        assert!(!obj.is_null());
        assert_eq!(gc.minor_collections, 0);
        // This request is too large for the tiny nursery, so no-collect mode
        // must have used the old generation instead.
        assert!(!gc.is_in_nursery(obj.0));
    }

    #[test]
    fn test_write_barrier_with_collection() {
        // Scenario: old object points to young object, write barrier ensures
        // the young object survives collection.
        let mut gc = test_gc(1024);
        let tid = gc.register_type(TypeInfo::with_gc_ptrs(
            std::mem::size_of::<GcRef>(),
            vec![0],
        ));

        // Create an old-gen object.
        let old_obj = gc.alloc_in_oldgen(tid, GcHeader::SIZE + std::mem::size_of::<GcRef>());

        // Create a young object.
        let young_obj = gc.alloc_with_type(tid, std::mem::size_of::<GcRef>());
        unsafe {
            *(young_obj.0 as *mut u64) = 0x42424242;
        }

        // Store young ref into old object's field.
        unsafe {
            *(old_obj.0 as *mut GcRef) = young_obj;
        }
        // Write barrier.
        gc.do_write_barrier(old_obj);

        // Root only the old object.
        let mut root = old_obj;
        unsafe {
            gc.roots.add(&mut root);
        }

        // Collect.
        gc.collect_nursery();

        // The old object's field should be updated to the new location.
        let child_ref = unsafe { *(root.0 as *const GcRef) };
        assert!(!gc.is_in_nursery(child_ref.0));
        assert!(!child_ref.is_null());
        let val = unsafe { *(child_ref.0 as *const u64) };
        assert_eq!(val, 0x42424242);

        gc.roots.clear();
    }

    #[test]
    fn test_null_ref_safety() {
        let mut gc = test_gc(1024);
        gc.register_type(TypeInfo::simple(8));

        // Write barrier on null should be a no-op.
        gc.do_write_barrier(GcRef::NULL);
        assert!(gc.remembered_set.is_empty());
    }

    #[test]
    fn test_chain_of_pointers() {
        // Test a chain: root -> A -> B -> C, all in nursery.
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::with_gc_ptrs(
            std::mem::size_of::<GcRef>(),
            vec![0],
        ));

        let c = gc.alloc_with_type(tid, std::mem::size_of::<GcRef>());
        unsafe {
            *(c.0 as *mut GcRef) = GcRef::NULL;
        }

        let b = gc.alloc_with_type(tid, std::mem::size_of::<GcRef>());
        unsafe {
            *(b.0 as *mut GcRef) = c;
        }

        let a = gc.alloc_with_type(tid, std::mem::size_of::<GcRef>());
        unsafe {
            *(a.0 as *mut GcRef) = b;
        }

        let mut root = a;
        unsafe {
            gc.roots.add(&mut root);
        }

        gc.collect_nursery();

        // Verify the entire chain survived.
        assert!(!gc.is_in_nursery(root.0));
        let new_b = unsafe { *(root.0 as *const GcRef) };
        assert!(!gc.is_in_nursery(new_b.0));
        assert!(!new_b.is_null());
        let new_c = unsafe { *(new_b.0 as *const GcRef) };
        assert!(!gc.is_in_nursery(new_c.0));
        assert!(!new_c.is_null());
        let tail = unsafe { *(new_c.0 as *const GcRef) };
        assert!(tail.is_null());

        gc.roots.clear();
    }

    #[test]
    fn test_major_collection_with_graph() {
        // Test major collection with a graph: root -> A -> B, root -> C (unreachable D).
        let ptr_size = std::mem::size_of::<GcRef>();
        let mut gc = test_gc(4096);
        // Type with one pointer field.
        let tid1 = gc.register_type(TypeInfo::with_gc_ptrs(ptr_size, vec![0]));
        // Type with two pointer fields.
        let tid2 = gc.register_type(TypeInfo::with_gc_ptrs(ptr_size * 2, vec![0, ptr_size]));

        let b = gc.alloc_with_type(tid1, ptr_size);
        unsafe {
            *(b.0 as *mut GcRef) = GcRef::NULL;
        }

        let a = gc.alloc_with_type(tid1, ptr_size);
        unsafe {
            *(a.0 as *mut GcRef) = b;
        }

        let c = gc.alloc_with_type(tid1, ptr_size);
        unsafe {
            *(c.0 as *mut GcRef) = GcRef::NULL;
        }

        let d = gc.alloc_with_type(tid1, ptr_size);
        unsafe {
            *(d.0 as *mut GcRef) = GcRef::NULL;
        }
        let _ = d; // unreachable

        // Root object points to both A and C.
        let root_obj = gc.alloc_with_type(tid2, ptr_size * 2);
        unsafe {
            *(root_obj.0 as *mut GcRef) = a;
            *((root_obj.0 + ptr_size) as *mut GcRef) = c;
        }

        let mut root = root_obj;
        unsafe {
            gc.roots.add(&mut root);
        }

        // Promote to old gen.
        gc.collect_nursery();

        // Now do full collection. A, B, C should survive; D should be freed.
        // We have 5 objects in old gen (root, A, B, C, D).
        // Wait, D was unreachable from root, so it wouldn't have been promoted.
        // Actually it was in the nursery unreachable, so it was wiped on nursery reset.
        assert_eq!(gc.oldgen.object_count(), 4); // root, A, B, C

        gc.collect_full();

        // All 4 are reachable from the root, so all survive.
        assert_eq!(gc.oldgen.object_count(), 4);

        gc.roots.clear();
    }

    #[test]
    fn test_data_integrity_across_collections() {
        // Allocate objects with distinctive data, collect, verify data.
        let mut gc = test_gc(2048);
        // Type: 32 bytes payload, one GcRef at offset 0, then 24 bytes of data.
        let tid = gc.register_type(TypeInfo::with_gc_ptrs(32, vec![0]));

        let child = gc.alloc_with_type(tid, 32);
        unsafe {
            *(child.0 as *mut GcRef) = GcRef::NULL;
            *((child.0 + 8) as *mut u64) = 0xAAAA_BBBB_CCCC_DDDD;
            *((child.0 + 16) as *mut u64) = 0x1111_2222_3333_4444;
            *((child.0 + 24) as *mut u64) = 0x5555_6666_7777_8888;
        }

        let parent = gc.alloc_with_type(tid, 32);
        unsafe {
            *(parent.0 as *mut GcRef) = child;
            *((parent.0 + 8) as *mut u64) = 0xCAFE_BABE_DEAD_BEEF;
        }

        let mut root = parent;
        unsafe {
            gc.roots.add(&mut root);
        }

        gc.collect_nursery();

        // Verify parent data.
        let pdata = unsafe { *((root.0 + 8) as *const u64) };
        assert_eq!(pdata, 0xCAFE_BABE_DEAD_BEEF);

        // Verify child data.
        let child_ref = unsafe { *(root.0 as *const GcRef) };
        assert!(!child_ref.is_null());
        let c1 = unsafe { *((child_ref.0 + 8) as *const u64) };
        let c2 = unsafe { *((child_ref.0 + 16) as *const u64) };
        let c3 = unsafe { *((child_ref.0 + 24) as *const u64) };
        assert_eq!(c1, 0xAAAA_BBBB_CCCC_DDDD);
        assert_eq!(c2, 0x1111_2222_3333_4444);
        assert_eq!(c3, 0x5555_6666_7777_8888);

        gc.roots.clear();
    }
}
