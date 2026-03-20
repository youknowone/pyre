/// MiniMarkGC — the core collector implementing the GcAllocator trait.
///
/// A generational copying collector with:
/// - Bump-pointer nursery for young objects
/// - System-allocator old gen with mark-sweep for major collection
/// - Write barrier with remembered set for old-to-young pointers
///
/// Modeled after incminimark's minor/major collection.
use std::collections::{HashMap, HashSet};

use majit_ir::GcRef;

use crate::GcAllocator;
use crate::flags;
use crate::header::{GcHeader, header_of};
use crate::nursery::{DEFAULT_NURSERY_SIZE, Nursery};
use crate::oldgen::OldGen;
use crate::trace::{TypeInfo, TypeRegistry};

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

/// Default card page shift: each card covers 2^7 = 128 array elements.
pub const DEFAULT_CARD_PAGE_SHIFT: u32 = 7;

/// State for incremental major collection.
///
/// Instead of doing a full mark-sweep in one pause, the marking work
/// is spread across multiple minor collections. Each minor collection
/// piggybacks an incremental marking step that processes a bounded
/// number of objects from the gray stack.
struct IncrementalMarkState {
    /// Objects still to be scanned (gray set).
    gray_stack: Vec<usize>,
    /// Whether an incremental cycle is in progress.
    marking_in_progress: bool,
    /// Number of objects marked so far in this cycle.
    objects_marked: usize,
    /// Target number of bytes to trace per increment.
    ///
    /// Mirrors incminimark's `gc_increment_step`, which defaults to
    /// `nursery_size * 2`.
    mark_budget_per_step: usize,
}

impl IncrementalMarkState {
    fn new(nursery_size: usize) -> Self {
        IncrementalMarkState {
            gray_stack: Vec::new(),
            marking_in_progress: false,
            objects_marked: 0,
            mark_budget_per_step: (nursery_size.saturating_mul(2)).max(1),
        }
    }
}

/// Default ratio of old-gen growth that triggers an incremental cycle.
/// When old-gen bytes exceed `last_major_bytes * MAJOR_COLLECT_RATIO`,
/// a new incremental cycle starts.
const MAJOR_COLLECT_RATIO: f64 = 1.82;

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
    /// Card table: maps object address to set of dirty card indices.
    /// Used for large arrays where scanning the entire array during
    /// minor collection is too expensive. Only dirty card ranges are
    /// scanned instead of the whole array.
    card_dirty: HashMap<usize, HashSet<usize>>,
    /// Configuration.
    config: GcConfig,
    /// Count of minor collections performed.
    pub minor_collections: usize,
    /// Count of major collections performed.
    pub major_collections: usize,
    /// State for incremental major collection.
    incr_state: IncrementalMarkState,
    /// Old-gen bytes at the end of the last completed major collection.
    /// Used to decide when to start the next incremental cycle.
    last_major_bytes: usize,
    /// Bytes promoted to old gen since the current incremental cycle started.
    ///
    /// Mirrors incminimark's `size_objects_made_old`.
    bytes_made_old_since_cycle: usize,
    /// Promotion credit granted by completed major-GC steps within the current
    /// incremental cycle.
    ///
    /// Mirrors incminimark's `threshold_objects_made_old`.
    threshold_bytes_made_old: usize,
    /// Pinned nursery objects that must not be moved during minor collection.
    pinned_objects: HashSet<usize>,
    /// Registry of compiled code regions for GC root scanning.
    pub compiled_code_registry: CompiledCodeRegistry,
}

impl MiniMarkGC {
    /// Create a new GC with default configuration.
    pub fn new() -> Self {
        Self::with_config(GcConfig::default())
    }

    /// Create a new GC with custom configuration.
    pub fn with_config(config: GcConfig) -> Self {
        let nursery_size = config.nursery_size;
        MiniMarkGC {
            nursery: Nursery::new(config.nursery_size),
            oldgen: OldGen::new(),
            types: TypeRegistry::new(),
            roots: RootSet::new(),
            remembered_set: Vec::new(),
            card_dirty: HashMap::new(),
            config,
            minor_collections: 0,
            major_collections: 0,
            incr_state: IncrementalMarkState::new(nursery_size),
            last_major_bytes: 0,
            bytes_made_old_since_cycle: 0,
            threshold_bytes_made_old: 0,
            pinned_objects: HashSet::new(),
            compiled_code_registry: CompiledCodeRegistry::new(),
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
        self.bytes_made_old_since_cycle = self
            .bytes_made_old_since_cycle
            .saturating_add(total_size);
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
        // Pinned objects are left in place (not copied to old gen).
        let roots: Vec<*mut GcRef> = self.roots.roots.iter().copied().collect();
        for root_ptr in roots {
            let gcref = unsafe { *root_ptr };
            if !gcref.is_null() && self.is_in_nursery(gcref.0) {
                if self.pinned_objects.contains(&gcref.0) {
                    continue;
                }
                let new_ref = self.copy_nursery_object(gcref.0);
                unsafe {
                    *root_ptr = new_ref;
                }
            }
        }

        // Phase 1b: Process shadow stack roots.
        // RPython gc.py: GcRootMap_shadowstack — walk the thread-local
        // shadow stack to find GC refs pushed by compiled JIT code.
        crate::shadow_stack::walk_roots(|gcref| {
            if self.is_in_nursery(gcref.0) {
                if !self.pinned_objects.contains(&gcref.0) {
                    *gcref = self.copy_nursery_object(gcref.0);
                }
            }
        });

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

            let has_cards = unsafe { header_of(obj_addr).has_flag(flags::CARDS_SET) };

            if has_cards {
                // Card-marked object: scan only dirty card ranges.
                self.scan_cards_for_young_refs(obj_addr, DEFAULT_CARD_PAGE_SHIFT);
                self.clear_cards(obj_addr);
                // Re-set TRACK_YOUNG_PTRS for future write barriers.
                unsafe {
                    header_of(obj_addr).set_flag(flags::TRACK_YOUNG_PTRS);
                }
            } else {
                // Re-set TRACK_YOUNG_PTRS on this old object since we're
                // processing all its young references now.
                unsafe {
                    header_of(obj_addr).set_flag(flags::TRACK_YOUNG_PTRS);
                }

                // Trace this old-gen object's fields and copy any nursery
                // objects they reference.
                self.trace_and_update_object(obj_addr);
            }
        }

        // Clear remembered set and any remaining card entries.
        self.remembered_set.clear();
        self.clear_all_cards();

        // Reset nursery for new allocations, preserving pinned objects.
        if self.pinned_objects.is_empty() {
            self.nursery.reset();
        } else {
            self.reset_nursery_with_pinned();
        }

        // Minor collections must also drive incremental major-collection
        // progress. Like incminimark, take one or more major steps until
        // promoted bytes are back under the current step credit.
        self.run_major_progress_after_minor();
    }

    /// Copy a single nursery object to old gen.
    /// If already forwarded, returns the forwarding address.
    /// Pinned objects are left in place and returned as-is.
    fn copy_nursery_object(&mut self, obj_addr: usize) -> GcRef {
        // Pinned objects must stay in the nursery.
        if self.pinned_objects.contains(&obj_addr) {
            return GcRef(obj_addr);
        }

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
        self.bytes_made_old_since_cycle = self
            .bytes_made_old_since_cycle
            .saturating_add(total_size);

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

    // ── Incremental marking ──

    /// Check whether old-gen growth warrants starting a new incremental
    /// major collection cycle.
    fn should_start_major_cycle(&self) -> bool {
        if self.last_major_bytes == 0 {
            // Never done a major collection. Use a minimum threshold so we
            // don't start an incremental cycle with a nearly empty old gen.
            return self.oldgen.total_bytes() > self.config.nursery_size;
        }
        self.oldgen.total_bytes() as f64 > self.last_major_bytes as f64 * MAJOR_COLLECT_RATIO
    }

    /// Begin a new incremental marking cycle.
    ///
    /// Seeds the gray stack with all root-reachable old-gen objects.
    pub fn start_incremental_cycle(&mut self) {
        self.incr_state.marking_in_progress = true;
        self.incr_state.gray_stack.clear();
        self.incr_state.objects_marked = 0;

        // Seed gray stack from roots.
        let roots: Vec<*mut GcRef> = self.roots.roots.iter().copied().collect();
        for root_ptr in roots {
            let gcref = unsafe { *root_ptr };
            if !gcref.is_null() {
                let hdr = unsafe { header_of(gcref.0) };
                if !hdr.has_flag(flags::VISITED) {
                    hdr.set_flag(flags::VISITED);
                    self.incr_state.gray_stack.push(gcref.0);
                }
            }
        }
    }

    /// Drive incremental major-collection progress after a minor collection.
    ///
    /// This follows incminimark's accounting rule: each major step grants
    /// `nursery_size / 2` bytes of promotion credit, and allocation-heavy
    /// minors may need multiple consecutive steps so old-gen growth does not
    /// outrun marking.
    fn run_major_progress_after_minor(&mut self) {
        if !self.incr_state.marking_in_progress && !self.should_start_major_cycle() {
            return;
        }

        loop {
            self.threshold_bytes_made_old = self
                .threshold_bytes_made_old
                .saturating_add(self.config.nursery_size / 2);

            if !self.incr_state.marking_in_progress {
                self.bytes_made_old_since_cycle = 0;
                self.threshold_bytes_made_old = self.config.nursery_size / 2;
                self.start_incremental_cycle();
            }

            let done = self.incremental_mark_step();
            if done {
                self.finish_incremental_cycle();
                break;
            }

            if self.bytes_made_old_since_cycle <= self.threshold_bytes_made_old {
                break;
            }
        }
    }

    /// Perform one incremental marking step.
    ///
    /// Processes up to `mark_budget_per_step` bytes from the gray stack.
    /// Like incminimark, this is a byte budget, but we always process at least
    /// one object so very small budgets still make forward progress.
    /// Returns `true` if marking is complete (gray stack exhausted).
    pub fn incremental_mark_step(&mut self) -> bool {
        let mut budget = self.incr_state.mark_budget_per_step;
        let mut processed_any = false;
        while budget > 0 || !processed_any {
            let Some(obj_addr) = self.incr_state.gray_stack.pop() else {
                self.incr_state.marking_in_progress = false;
                return true; // marking complete
            };
            let obj_size = self.object_total_size(obj_addr);
            self.mark_object(obj_addr);
            self.incr_state.objects_marked += 1;
            budget = budget.saturating_sub(obj_size.max(1));
            processed_any = true;
        }
        false // more work to do
    }

    fn object_total_size(&self, obj_addr: usize) -> usize {
        let type_id = unsafe { header_of(obj_addr).type_id() };
        let type_info = self.types.get(type_id);
        let payload_size = if type_info.item_size > 0 {
            let length = unsafe { *((obj_addr + type_info.length_offset) as *const usize) };
            type_info.total_instance_size(length)
        } else {
            type_info.size
        };
        GcHeader::SIZE + payload_size
    }

    /// Mark a single object: trace its GC pointer fields and push
    /// unmarked children onto the gray stack.
    fn mark_object(&mut self, obj_addr: usize) {
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
                    self.incr_state.gray_stack.push(field_ref.0);
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
                        self.incr_state.gray_stack.push(field_ref.0);
                    }
                }
            }
        }
    }

    /// Complete the sweep phase after incremental marking finishes.
    fn finish_incremental_cycle(&mut self) {
        self.major_collections += 1;
        self.oldgen.sweep();
        self.last_major_bytes = self.oldgen.total_bytes();
        self.bytes_made_old_since_cycle = 0;
        self.threshold_bytes_made_old = 0;
    }

    /// Whether an incremental marking cycle is currently in progress.
    pub fn is_incremental_marking(&self) -> bool {
        self.incr_state.marking_in_progress
    }

    /// Number of objects marked so far in the current incremental cycle.
    pub fn incremental_objects_marked(&self) -> usize {
        self.incr_state.objects_marked
    }

    /// Set the per-step marking budget in bytes.
    pub fn set_mark_budget(&mut self, budget: usize) {
        self.incr_state.mark_budget_per_step = budget;
    }

    /// Perform a full (major) mark-sweep collection.
    ///
    /// 1. First do a minor collection to promote all live nursery objects.
    /// 2. Mark phase: trace all roots and transitively mark reachable objects.
    /// 3. Sweep phase: free all unmarked old-gen objects.
    pub fn do_collect_full(&mut self) {
        // Minor collection first to empty the nursery.
        // Note: do_collect_nursery may itself start/advance an incremental
        // cycle, but we need a complete mark-sweep here regardless.
        self.do_collect_nursery();

        if self.incr_state.marking_in_progress {
            // An incremental cycle is in progress. Finish it by draining
            // the gray stack without budget limits.
            while !self.incr_state.gray_stack.is_empty() {
                let obj_addr = self.incr_state.gray_stack.pop().unwrap();
                self.mark_object(obj_addr);
                self.incr_state.objects_marked += 1;
            }
            self.incr_state.marking_in_progress = false;
            self.finish_incremental_cycle();
        } else {
            // No incremental cycle in progress. Do a full stop-the-world
            // mark-sweep using the same mark_object infrastructure.
            self.incr_state.gray_stack.clear();

            // Seed from roots.
            let roots: Vec<*mut GcRef> = self.roots.roots.iter().copied().collect();
            for root_ptr in roots {
                let gcref = unsafe { *root_ptr };
                if !gcref.is_null() {
                    let hdr = unsafe { header_of(gcref.0) };
                    if !hdr.has_flag(flags::VISITED) {
                        hdr.set_flag(flags::VISITED);
                        self.incr_state.gray_stack.push(gcref.0);
                    }
                }
            }

            // Drain gray stack completely.
            while let Some(obj_addr) = self.incr_state.gray_stack.pop() {
                self.mark_object(obj_addr);
            }

            self.finish_incremental_cycle();
        }
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

        let card_index = index >> card_page_shift;
        self.card_dirty.entry(obj.0).or_default().insert(card_index);
    }

    /// Check whether a specific card of an object is dirty.
    pub fn is_card_dirty(&self, obj: GcRef, card_index: usize) -> bool {
        self.card_dirty
            .get(&obj.0)
            .is_some_and(|cards| cards.contains(&card_index))
    }

    /// Return all dirty card indices for the given object.
    pub fn dirty_cards(&self, obj: GcRef) -> Vec<usize> {
        self.card_dirty
            .get(&obj.0)
            .map(|cards| {
                let mut v: Vec<usize> = cards.iter().copied().collect();
                v.sort();
                v
            })
            .unwrap_or_default()
    }

    /// Scan only dirty card ranges of a card-marked array object for
    /// young-generation references. Avoids rescanning the entire array.
    fn scan_cards_for_young_refs(&mut self, obj_addr: usize, card_page_shift: u32) {
        let type_id = unsafe { header_of(obj_addr).type_id() };
        let type_info = self.types.get(type_id);

        if !type_info.items_have_gc_ptrs || type_info.item_size == 0 {
            return;
        }

        let item_size = type_info.item_size;
        let base_size = type_info.size;
        let length_offset = type_info.length_offset;
        let length = unsafe { *((obj_addr + length_offset) as *const usize) };
        let items_start = obj_addr + base_size;

        let dirty_cards: Vec<usize> = self
            .card_dirty
            .get(&obj_addr)
            .map(|cards| cards.iter().copied().collect())
            .unwrap_or_default();

        for card_idx in dirty_cards {
            let start = card_idx << card_page_shift;
            let end = ((card_idx + 1) << card_page_shift).min(length);

            for i in start..end {
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

    /// Clear card table entries for a given object.
    pub fn clear_cards(&mut self, obj_addr: usize) {
        self.card_dirty.remove(&obj_addr);
        let hdr = unsafe { header_of(obj_addr) };
        hdr.clear_flag(flags::CARDS_SET);
    }

    /// Clear all card table entries. Called during minor collection
    /// after processing all card-marked objects.
    pub fn clear_all_cards(&mut self) {
        let addrs: Vec<usize> = self.card_dirty.keys().copied().collect();
        for addr in addrs {
            let hdr = unsafe { header_of(addr) };
            hdr.clear_flag(flags::CARDS_SET);
        }
        self.card_dirty.clear();
    }

    // ── JIT integration hooks ──

    /// Fast-path write barrier for JIT-compiled code.
    ///
    /// Called from JIT-compiled code when a write barrier fires.
    /// Adds the object directly to the remembered set without the
    /// full flag-check logic of `do_write_barrier()`, because the
    /// JIT has already determined that the barrier is needed (via
    /// the inline flag test emitted by COND_CALL_GC_WB).
    ///
    /// Equivalent to incminimark's `jit_remember_young_pointer()`.
    pub fn jit_remember_young_pointer(&mut self, obj: GcRef) {
        if obj.is_null() {
            return;
        }
        // Clear TRACK_YOUNG_PTRS so the object won't trigger the
        // barrier again until it is re-processed in a minor collection.
        let hdr = unsafe { header_of(obj.0) };
        hdr.clear_flag(flags::TRACK_YOUNG_PTRS);
        self.remembered_set.push(obj.0);
    }

    /// Returns true if the GC supports optimized conditional write barriers.
    ///
    /// When true, the JIT can emit COND_CALL_GC_WB (an inline flag test +
    /// conditional call) instead of a full write-barrier call. Nursery-based
    /// collectors always support this because the barrier check is a simple
    /// flag test on the object header.
    pub fn can_optimize_cond_call(&self) -> bool {
        true
    }

    /// Perform one incremental GC step. Called from JIT safepoints.
    ///
    /// If an incremental marking cycle should start, it is initiated.
    /// If a cycle is already in progress, one bounded marking step is
    /// performed. Returns true if any GC work was done.
    pub fn gc_step(&mut self) -> bool {
        if self.should_start_major_cycle() && !self.incr_state.marking_in_progress {
            self.start_incremental_cycle();
            let done = self.incremental_mark_step();
            if done {
                self.finish_incremental_cycle();
            }
            true
        } else if self.incr_state.marking_in_progress {
            let done = self.incremental_mark_step();
            if done {
                self.finish_incremental_cycle();
            }
            true
        } else {
            false
        }
    }

    /// Reset the nursery while preserving pinned objects.
    ///
    /// Saves pinned object data, zeroes the nursery, restores pinned objects,
    /// and sets the free pointer past the highest pinned object.
    fn reset_nursery_with_pinned(&mut self) {
        let nursery_start = self.nursery.start_ptr() as usize;

        // Collect (header_start, total_size, data) for each pinned object.
        let mut saved: Vec<(usize, usize, Vec<u8>)> = Vec::new();
        for &obj_addr in &self.pinned_objects {
            let type_id = unsafe { header_of(obj_addr).type_id() };
            let type_info = self.types.get(type_id);
            let payload_size = if type_info.item_size > 0 {
                let length = unsafe { *((obj_addr + type_info.length_offset) as *const usize) };
                type_info.total_instance_size(length)
            } else {
                type_info.size
            };
            let total_size = (GcHeader::SIZE + payload_size).max(GcHeader::MIN_NURSERY_OBJ_SIZE);
            let total_size = (total_size + 7) & !7;
            let header_start = obj_addr - GcHeader::SIZE;
            let data = unsafe {
                std::slice::from_raw_parts(header_start as *const u8, total_size).to_vec()
            };
            saved.push((header_start, total_size, data));
        }

        // Zero-fill the entire nursery.
        self.nursery.reset();

        // Restore pinned objects and compute the highest end.
        let mut max_end = nursery_start;
        for (header_start, total_size, data) in &saved {
            unsafe {
                std::ptr::copy_nonoverlapping(data.as_ptr(), *header_start as *mut u8, *total_size);
            }
            let end = header_start + total_size;
            if end > max_end {
                max_end = end;
            }
        }

        // Set free pointer past the highest pinned object so new allocations
        // don't overwrite it.
        if max_end > nursery_start {
            unsafe {
                self.nursery.set_free_ptr(max_end as *mut u8);
            }
        }
    }

    /// Pin a nursery object so it won't be moved during minor collection.
    /// Sets the PINNED flag in the object header and records the address.
    /// Returns true if pinning succeeded, false if the object is null or
    /// not in the nursery.
    pub fn pin(&mut self, obj: GcRef) -> bool {
        if obj.is_null() || !self.is_in_nursery(obj.0) {
            return false;
        }
        unsafe {
            header_of(obj.0).set_flag(flags::PINNED);
        }
        self.pinned_objects.insert(obj.0);
        true
    }

    /// Unpin a previously pinned object.
    pub fn unpin(&mut self, obj: GcRef) {
        if obj.is_null() {
            return;
        }
        // Clear the header flag if the object is still in the nursery.
        if self.is_in_nursery(obj.0) {
            unsafe {
                header_of(obj.0).clear_flag(flags::PINNED);
            }
        }
        self.pinned_objects.remove(&obj.0);
    }

    /// Check if an object is currently pinned.
    pub fn is_pinned(&self, obj: GcRef) -> bool {
        self.pinned_objects.contains(&obj.0)
    }

    /// Free memory associated with invalidated JIT compiled code.
    ///
    /// `code_ptr` and `size` identify the compiled code region to release.
    /// The region is looked up and removed from the compiled code registry
    /// so the GC no longer scans it for root references.
    pub fn jit_free(&mut self, code_ptr: usize, size: usize) {
        // Find and remove any compiled code region that matches the given range.
        self.compiled_code_registry
            .regions
            .retain(|r| !(r.code_start == code_ptr && r.code_size == size));
    }

    /// Number of objects in the remembered set (for testing / diagnostics).
    pub fn remembered_set_len(&self) -> usize {
        self.remembered_set.len()
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

/// Registry of compiled code regions and their safepoint maps.
///
/// When the GC needs to scan the stack during collection, it uses the return
/// address to find which compiled code region is active, then looks up the
/// safepoint map to determine which frame slots contain GC references.
///
/// From rpython/jit/backend/llsupport/gc.py GcRootMap_asmgcc / GcRootMap_shadowstack.
pub struct CompiledCodeRegistry {
    /// Compiled code regions, sorted by start address for binary search.
    regions: Vec<CompiledCodeRegion>,
}

/// A single compiled code region with its safepoint map.
#[derive(Debug, Clone)]
pub struct CompiledCodeRegion {
    /// Start address of the compiled code.
    pub code_start: usize,
    /// Size of the compiled code in bytes.
    pub code_size: usize,
    /// Safepoint map for this region.
    pub safepoint_map: SafepointMap,
    /// Frame size in slots (each slot = 8 bytes).
    pub frame_size_slots: u32,
    /// JitCellToken number for identification.
    pub loop_token: u64,
}

impl CompiledCodeRegistry {
    pub fn new() -> Self {
        CompiledCodeRegistry {
            regions: Vec::new(),
        }
    }

    /// Register a compiled code region.
    pub fn register(&mut self, region: CompiledCodeRegion) {
        self.regions.push(region);
        // Keep sorted by code_start for binary search
        self.regions.sort_by_key(|r| r.code_start);
    }

    /// Unregister a compiled code region (e.g., when invalidating a loop).
    pub fn unregister(&mut self, loop_token: u64) {
        self.regions.retain(|r| r.loop_token != loop_token);
    }

    /// Look up a compiled code region containing the given return address.
    ///
    /// Returns the region and the offset within it.
    pub fn find_region(&self, return_addr: usize) -> Option<(&CompiledCodeRegion, u32)> {
        // Binary search for the region containing this address
        let idx = self
            .regions
            .binary_search_by(|r| {
                if return_addr < r.code_start {
                    std::cmp::Ordering::Greater
                } else if return_addr >= r.code_start + r.code_size {
                    std::cmp::Ordering::Less
                } else {
                    std::cmp::Ordering::Equal
                }
            })
            .ok()?;

        let region = &self.regions[idx];
        let offset = (return_addr - region.code_start) as u32;
        Some((region, offset))
    }

    /// Scan a compiled frame for GC references using the safepoint map.
    ///
    /// Given a return address (from the call stack) and the frame base pointer,
    /// enumerates all frame slots that contain GC references.
    ///
    /// # Safety
    /// `frame_base` must point to a valid JIT frame with at least
    /// `region.frame_size_slots` slots.
    pub unsafe fn scan_frame(
        &self,
        return_addr: usize,
        frame_base: *const usize,
    ) -> Vec<*mut GcRef> {
        let mut roots = Vec::new();

        let (region, offset) = match self.find_region(return_addr) {
            Some(r) => r,
            None => return roots,
        };

        let gc_map = match region.safepoint_map.lookup(offset) {
            Some(map) => map,
            None => return roots,
        };

        // Enumerate all slots marked as GC references
        for word_idx in 0..gc_map.ref_bitmap.len() {
            let mut bits = gc_map.ref_bitmap[word_idx];
            while bits != 0 {
                let bit = bits.trailing_zeros() as usize;
                let slot_idx = word_idx * 64 + bit;

                if slot_idx < region.frame_size_slots as usize {
                    let slot_ptr = unsafe { frame_base.add(slot_idx) } as *mut GcRef;
                    roots.push(slot_ptr);
                }

                bits &= bits - 1; // Clear lowest set bit
            }
        }

        roots
    }

    /// Number of registered regions.
    pub fn len(&self) -> usize {
        self.regions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.regions.is_empty()
    }
}

impl Default for CompiledCodeRegistry {
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

    fn alloc_nursery_typed(&mut self, type_id: u32, size: usize) -> GcRef {
        self.alloc_with_type(type_id, size)
    }

    fn alloc_nursery_no_collect(&mut self, size: usize) -> GcRef {
        self.alloc_with_type_no_collect(0, size)
    }

    fn alloc_varsize(&mut self, base_size: usize, item_size: usize, length: usize) -> GcRef {
        let payload_size = base_size + item_size * length;
        self.alloc_with_type(0, payload_size)
    }

    fn alloc_varsize_typed(
        &mut self,
        type_id: u32,
        base_size: usize,
        item_size: usize,
        length: usize,
    ) -> GcRef {
        let payload_size = base_size + item_size * length;
        self.alloc_with_type(type_id, payload_size)
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
        unsafe { self.roots.add(root) };
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

    fn jit_remember_young_pointer(&mut self, obj: GcRef) {
        self.jit_remember_young_pointer(obj);
    }

    fn can_optimize_cond_call(&self) -> bool {
        self.can_optimize_cond_call()
    }

    fn gc_step(&mut self) -> bool {
        self.gc_step()
    }

    fn jit_free(&mut self, code_ptr: usize, size: usize) {
        self.jit_free(code_ptr, size);
    }

    fn pin(&mut self, obj: GcRef) -> bool {
        self.pin(obj)
    }

    fn unpin(&mut self, obj: GcRef) {
        self.unpin(obj);
    }

    fn is_pinned(&self, obj: GcRef) -> bool {
        self.is_pinned(obj)
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

    // ── Card marking tests ──

    #[test]
    fn test_card_marking_basic() {
        let mut gc = test_gc(4096);
        gc.register_type(TypeInfo::simple(16));

        // Allocate an old-gen object with HAS_CARDS flag.
        let obj = gc.alloc_in_oldgen(0, GcHeader::SIZE + 16);
        let hdr = unsafe { header_of(obj.0) };
        hdr.set_flag(flags::HAS_CARDS);

        // Card-marking write barrier: mark card for index 5.
        gc.do_write_barrier_card(obj, 5, DEFAULT_CARD_PAGE_SHIFT);

        // The card at index 5 >> 7 = 0 should be dirty.
        assert!(
            gc.is_card_dirty(obj, 0),
            "card 0 should be dirty after writing index 5"
        );
        assert!(
            hdr.has_flag(flags::CARDS_SET),
            "CARDS_SET flag should be set"
        );

        // Mark another index in a different card.
        gc.do_write_barrier_card(obj, 200, DEFAULT_CARD_PAGE_SHIFT);
        let card_idx = 200 >> DEFAULT_CARD_PAGE_SHIFT;
        assert!(
            gc.is_card_dirty(obj, card_idx as usize),
            "card for index 200 should be dirty"
        );
    }

    #[test]
    fn test_card_marking_clear_after_collection() {
        let mut gc = test_gc(4096);
        gc.register_type(TypeInfo::simple(16));

        // Allocate an old-gen object with HAS_CARDS.
        let obj = gc.alloc_in_oldgen(0, GcHeader::SIZE + 16);
        let hdr = unsafe { header_of(obj.0) };
        hdr.set_flag(flags::HAS_CARDS);

        // Mark some cards.
        gc.do_write_barrier_card(obj, 0, DEFAULT_CARD_PAGE_SHIFT);
        gc.do_write_barrier_card(obj, 200, DEFAULT_CARD_PAGE_SHIFT);
        assert!(
            !gc.card_dirty.is_empty(),
            "cards should be dirty before collection"
        );

        // Minor collection clears card table.
        gc.do_collect_nursery();

        assert!(
            gc.card_dirty.is_empty(),
            "card table should be cleared after collection"
        );
        let hdr = unsafe { header_of(obj.0) };
        assert!(
            !hdr.has_flag(flags::CARDS_SET),
            "CARDS_SET flag should be cleared after collection"
        );
    }

    #[test]
    fn test_card_marking_dirty_cards_list() {
        let mut gc = test_gc(4096);
        gc.register_type(TypeInfo::simple(16));

        let obj = gc.alloc_in_oldgen(0, GcHeader::SIZE + 16);
        let hdr = unsafe { header_of(obj.0) };
        hdr.set_flag(flags::HAS_CARDS);

        // Mark cards for indices in different card pages.
        gc.do_write_barrier_card(obj, 0, DEFAULT_CARD_PAGE_SHIFT);
        gc.do_write_barrier_card(obj, 128, DEFAULT_CARD_PAGE_SHIFT);
        gc.do_write_barrier_card(obj, 256, DEFAULT_CARD_PAGE_SHIFT);

        let dirty = gc.dirty_cards(obj);
        assert_eq!(dirty, vec![0, 1, 2], "should have cards 0, 1, 2 dirty");
    }

    #[test]
    fn test_card_marking_fallback_without_has_cards() {
        // Object without HAS_CARDS should fall back to remembered set.
        let mut gc = test_gc(4096);
        gc.register_type(TypeInfo::simple(16));

        let obj = gc.alloc_in_oldgen(0, GcHeader::SIZE + 16);
        // Do NOT set HAS_CARDS.

        gc.do_write_barrier_card(obj, 5, DEFAULT_CARD_PAGE_SHIFT);

        // Should fall back to full remembered set.
        assert_eq!(gc.remembered_set.len(), 1, "should add to remembered set");
        assert!(
            gc.card_dirty.is_empty(),
            "should not mark cards without HAS_CARDS"
        );
    }

    #[test]
    fn test_card_clear_individual() {
        let mut gc = test_gc(4096);
        gc.register_type(TypeInfo::simple(16));

        let obj1 = gc.alloc_in_oldgen(0, GcHeader::SIZE + 16);
        let obj2 = gc.alloc_in_oldgen(0, GcHeader::SIZE + 16);
        let hdr1 = unsafe { header_of(obj1.0) };
        let hdr2 = unsafe { header_of(obj2.0) };
        hdr1.set_flag(flags::HAS_CARDS);
        hdr2.set_flag(flags::HAS_CARDS);

        gc.do_write_barrier_card(obj1, 0, DEFAULT_CARD_PAGE_SHIFT);
        gc.do_write_barrier_card(obj2, 0, DEFAULT_CARD_PAGE_SHIFT);

        // Clear only obj1's cards.
        gc.clear_cards(obj1.0);

        assert!(!gc.is_card_dirty(obj1, 0), "obj1 cards should be cleared");
        assert!(
            gc.is_card_dirty(obj2, 0),
            "obj2 cards should still be dirty"
        );
    }

    // ── SafepointMap tests ──

    #[test]
    fn test_safepoint_map_register_and_lookup() {
        let mut smap = SafepointMap::new();

        let mut gc_map_0 = crate::GcMap::new();
        gc_map_0.set_ref(0);
        gc_map_0.set_ref(3);

        let mut gc_map_1 = crate::GcMap::new();
        gc_map_1.set_ref(1);
        gc_map_1.set_ref(7);

        smap.add(100, gc_map_0);
        smap.add(200, gc_map_1);

        // Lookup existing entries.
        let found_0 = smap.lookup(100).unwrap();
        assert!(found_0.is_ref(0));
        assert!(found_0.is_ref(3));
        assert!(!found_0.is_ref(1));

        let found_1 = smap.lookup(200).unwrap();
        assert!(found_1.is_ref(1));
        assert!(found_1.is_ref(7));
        assert!(!found_1.is_ref(0));

        // Lookup non-existent offset returns None.
        assert!(smap.lookup(999).is_none());
    }

    #[test]
    fn test_safepoint_map_empty() {
        let smap = SafepointMap::new();
        assert!(smap.lookup(0).is_none());
        assert!(smap.entries.is_empty());
    }

    // ── CompiledCodeRegistry tests ──

    #[test]
    fn test_compiled_code_registry_register_and_find() {
        let mut registry = CompiledCodeRegistry::new();
        assert!(registry.is_empty());

        let mut smap = SafepointMap::new();
        let mut gc_map = crate::GcMap::new();
        gc_map.set_ref(0);
        gc_map.set_ref(2);
        smap.add(16, gc_map);

        registry.register(CompiledCodeRegion {
            code_start: 0x1000,
            code_size: 0x100,
            safepoint_map: smap,
            frame_size_slots: 4,
            loop_token: 42,
        });

        assert_eq!(registry.len(), 1);

        // Address inside the region.
        let (region, offset) = registry.find_region(0x1010).unwrap();
        assert_eq!(region.loop_token, 42);
        assert_eq!(offset, 0x10);

        // Address at the start.
        let (region, offset) = registry.find_region(0x1000).unwrap();
        assert_eq!(region.loop_token, 42);
        assert_eq!(offset, 0);

        // Address outside the region.
        assert!(registry.find_region(0x900).is_none());
        assert!(registry.find_region(0x1100).is_none());
    }

    #[test]
    fn test_compiled_code_registry_multiple_regions() {
        let mut registry = CompiledCodeRegistry::new();

        registry.register(CompiledCodeRegion {
            code_start: 0x1000,
            code_size: 0x100,
            safepoint_map: SafepointMap::new(),
            frame_size_slots: 4,
            loop_token: 1,
        });
        registry.register(CompiledCodeRegion {
            code_start: 0x3000,
            code_size: 0x200,
            safepoint_map: SafepointMap::new(),
            frame_size_slots: 8,
            loop_token: 2,
        });
        registry.register(CompiledCodeRegion {
            code_start: 0x2000,
            code_size: 0x80,
            safepoint_map: SafepointMap::new(),
            frame_size_slots: 2,
            loop_token: 3,
        });

        assert_eq!(registry.len(), 3);

        // Each region should be findable.
        assert_eq!(registry.find_region(0x1050).unwrap().0.loop_token, 1);
        assert_eq!(registry.find_region(0x2040).unwrap().0.loop_token, 3);
        assert_eq!(registry.find_region(0x3100).unwrap().0.loop_token, 2);

        // Gap between regions returns None.
        assert!(registry.find_region(0x1200).is_none());
    }

    #[test]
    fn test_compiled_code_registry_unregister() {
        let mut registry = CompiledCodeRegistry::new();

        registry.register(CompiledCodeRegion {
            code_start: 0x1000,
            code_size: 0x100,
            safepoint_map: SafepointMap::new(),
            frame_size_slots: 4,
            loop_token: 10,
        });
        registry.register(CompiledCodeRegion {
            code_start: 0x2000,
            code_size: 0x100,
            safepoint_map: SafepointMap::new(),
            frame_size_slots: 4,
            loop_token: 20,
        });

        assert_eq!(registry.len(), 2);

        registry.unregister(10);
        assert_eq!(registry.len(), 1);
        assert!(registry.find_region(0x1050).is_none());
        assert_eq!(registry.find_region(0x2050).unwrap().0.loop_token, 20);
    }

    #[test]
    fn test_compiled_code_registry_safepoint_lookup_for_root_scanning() {
        let mut registry = CompiledCodeRegistry::new();

        let mut smap = SafepointMap::new();
        let mut gc_map = crate::GcMap::new();
        gc_map.set_ref(0);
        gc_map.set_ref(2);
        smap.add(0x20, gc_map);

        registry.register(CompiledCodeRegion {
            code_start: 0x5000,
            code_size: 0x200,
            safepoint_map: smap,
            frame_size_slots: 4,
            loop_token: 99,
        });

        // Simulate finding a return address and looking up the safepoint map.
        let return_addr = 0x5020;
        let (region, offset) = registry.find_region(return_addr).unwrap();
        let gc_map = region.safepoint_map.lookup(offset).unwrap();

        // Verify the GC map identifies the correct slots.
        assert!(gc_map.is_ref(0), "slot 0 should be a GC ref");
        assert!(!gc_map.is_ref(1), "slot 1 should not be a GC ref");
        assert!(gc_map.is_ref(2), "slot 2 should be a GC ref");
        assert!(!gc_map.is_ref(3), "slot 3 should not be a GC ref");
    }

    #[test]
    fn test_scan_frame_enumerates_gc_ref_slots() {
        let mut registry = CompiledCodeRegistry::new();

        let mut smap = SafepointMap::new();
        let mut gc_map = crate::GcMap::new();
        gc_map.set_ref(0);
        gc_map.set_ref(2);
        smap.add(0x10, gc_map);

        registry.register(CompiledCodeRegion {
            code_start: 0xA000,
            code_size: 0x100,
            safepoint_map: smap,
            frame_size_slots: 4,
            loop_token: 77,
        });

        // Allocate a fake frame on the stack.
        let frame: [usize; 4] = [111, 222, 333, 444];
        let frame_base = frame.as_ptr();

        let return_addr = 0xA010;
        let roots = unsafe { registry.scan_frame(return_addr, frame_base) };

        // Should find slots 0 and 2.
        assert_eq!(roots.len(), 2);
        unsafe {
            assert_eq!(*(roots[0] as *const usize), 111);
            assert_eq!(*(roots[1] as *const usize), 333);
        }
    }

    // ── Incremental marking tests ──

    #[test]
    fn test_incremental_marking_basic() {
        // Start an incremental cycle, run steps, and verify completion.
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::simple(16));

        // Promote some objects to old gen via minor collection.
        let obj1 = gc.alloc_with_type(tid, 16);
        let obj2 = gc.alloc_with_type(tid, 16);
        let mut root1 = obj1;
        let mut root2 = obj2;
        unsafe {
            gc.roots.add(&mut root1);
            gc.roots.add(&mut root2);
        }
        gc.do_collect_nursery();
        assert!(!gc.is_in_nursery(root1.0));
        assert!(!gc.is_in_nursery(root2.0));

        // Manually start an incremental cycle.
        gc.start_incremental_cycle();
        assert!(gc.is_incremental_marking());

        // Run marking steps until complete.
        let mut steps = 0;
        while gc.is_incremental_marking() {
            gc.incremental_mark_step();
            steps += 1;
            if steps > 100 {
                panic!("incremental marking did not complete");
            }
        }

        // Marking should have processed the 2 root objects.
        assert!(gc.incremental_objects_marked() >= 2);

        gc.roots.clear();
    }

    #[test]
    fn test_incremental_marking_piggyback() {
        // Verify that incremental marking progresses during nursery collections.
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::simple(16));

        // Promote several objects to old gen by rooting each and collecting.
        let mut roots_storage = vec![GcRef::NULL; 10];
        for r in roots_storage.iter_mut() {
            unsafe {
                gc.roots.add(r);
            }
        }
        for r in roots_storage.iter_mut() {
            let obj = gc.alloc_with_type(tid, 16);
            *r = obj;
        }
        gc.do_collect_nursery();
        for r in &roots_storage {
            assert!(!gc.is_in_nursery(r.0));
        }

        // Start an incremental cycle with a tiny budget so it takes
        // multiple steps.
        gc.set_mark_budget(1);
        gc.start_incremental_cycle();
        assert!(gc.is_incremental_marking());

        // Each nursery collection should advance the marking.
        let marked_before = gc.incremental_objects_marked();
        gc.do_collect_nursery();
        // After one nursery collection with budget=1, we should have
        // marked at least one more object (if any remained).
        assert!(
            gc.incremental_objects_marked() > marked_before,
            "marking should advance during nursery collection"
        );

        gc.roots.clear();
    }

    #[test]
    fn test_minor_collection_can_take_multiple_major_steps_when_promotions_outpace_credit() {
        let ptr_size = std::mem::size_of::<GcRef>();
        let mut gc = test_gc(1024);
        let tid = gc.register_type(TypeInfo::with_gc_ptrs(ptr_size, vec![0]));

        // Build an old-gen chain large enough that a single budget=1 marking
        // step cannot finish it.
        let mut prev = GcRef::NULL;
        for _ in 0..6 {
            let obj = gc.alloc_in_oldgen(tid, GcHeader::SIZE + ptr_size);
            unsafe {
                *(obj.0 as *mut GcRef) = prev;
            }
            prev = obj;
        }
        let mut root = prev;
        unsafe {
            gc.roots.add(&mut root);
        }

        gc.set_mark_budget(1);
        gc.start_incremental_cycle();
        assert!(gc.incr_state.gray_stack.len() >= 1);

        // Simulate a minor collection that promoted more than one step's worth
        // of objects so incminimark-style accounting demands extra progress.
        gc.bytes_made_old_since_cycle = gc.config.nursery_size;
        gc.threshold_bytes_made_old = 0;
        gc.run_major_progress_after_minor();

        assert!(
            gc.incremental_objects_marked() >= 2
                || gc.threshold_bytes_made_old > gc.config.nursery_size / 2,
            "major progress should take multiple steps when promoted bytes outpace credit"
        );

        gc.roots.clear();
    }

    #[test]
    fn test_incremental_marking_budget() {
        // Each step should process at most `mark_budget_per_step` objects.
        let ptr_size = std::mem::size_of::<GcRef>();
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::with_gc_ptrs(ptr_size, vec![0]));

        // Create a chain of 10 objects so marking has plenty of work.
        let mut prev = GcRef::NULL;
        let mut roots = Vec::new();
        for _ in 0..10 {
            let obj = gc.alloc_with_type(tid, ptr_size);
            unsafe {
                *(obj.0 as *mut GcRef) = prev;
            }
            prev = obj;
            roots.push(prev);
        }

        // Root the head of the chain.
        let mut head = prev;
        unsafe {
            gc.roots.add(&mut head);
        }

        // Promote all to old gen.
        gc.do_collect_nursery();
        assert!(!gc.is_in_nursery(head.0));

        // Start incremental cycle with a tiny byte budget so each step still
        // processes exactly one object.
        gc.set_mark_budget(2);
        gc.start_incremental_cycle();
        assert!(gc.is_incremental_marking());

        // First step: marks at most 1 object.
        let done = gc.incremental_mark_step();
        assert!(!done, "should not be done after marking only 1 out of 10");
        assert_eq!(gc.incremental_objects_marked(), 1);

        // Second step: marks 1 more.
        let done = gc.incremental_mark_step();
        assert!(!done, "should not be done after 2 total");
        assert_eq!(gc.incremental_objects_marked(), 2);

        gc.roots.clear();
    }

    #[test]
    fn test_incremental_marking_completes() {
        // A full incremental cycle (start -> repeated steps -> sweep)
        // produces the same result as a stop-the-world full collection:
        // unreachable old-gen objects are freed.
        let ptr_size = std::mem::size_of::<GcRef>();
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::with_gc_ptrs(ptr_size, vec![0]));

        // Create reachable and unreachable old-gen objects.
        let reachable = gc.alloc_with_type(tid, ptr_size);
        unsafe {
            *(reachable.0 as *mut GcRef) = GcRef::NULL;
        }
        let unreachable = gc.alloc_with_type(tid, ptr_size);
        unsafe {
            *(unreachable.0 as *mut GcRef) = GcRef::NULL;
        }

        let mut root = reachable;
        let mut root2 = unreachable;
        unsafe {
            gc.roots.add(&mut root);
            gc.roots.add(&mut root2);
        }

        // Promote both to old gen.
        gc.do_collect_nursery();
        assert_eq!(gc.oldgen.object_count(), 2);

        // Unroot the unreachable one.
        gc.roots.remove(&mut root2);

        // Run incremental cycle with budget=1 to force multiple steps.
        gc.set_mark_budget(1);
        gc.start_incremental_cycle();

        // Drive the cycle to completion.
        let mut iterations = 0;
        while gc.is_incremental_marking() {
            gc.incremental_mark_step();
            iterations += 1;
            if iterations > 100 {
                panic!("incremental marking did not complete");
            }
        }

        // Finish: sweep unreachable objects.
        gc.finish_incremental_cycle();

        // Only the reachable object should survive.
        assert_eq!(gc.oldgen.object_count(), 1);
        assert!(!root.is_null());

        gc.roots.clear();
    }

    // ── GC stress tests ──

    #[test]
    fn test_gc_stress_with_safepoint_scanning() {
        // Register a compiled code region with a safepoint map, then
        // allocate objects under pressure so nursery collections fire.
        // After collection, verify that roots discovered via scan_frame
        // point to valid, promoted objects.

        let ptr_size = std::mem::size_of::<GcRef>();
        let mut gc = test_gc(512); // small nursery to force frequent collections
        let tid = gc.register_type(TypeInfo::with_gc_ptrs(ptr_size * 2, vec![0, ptr_size]));

        // Build a compiled code registry with a safepoint map marking
        // frame slots 0 and 2 as GC references.
        let mut registry = CompiledCodeRegistry::new();
        let mut smap = SafepointMap::new();
        let mut gc_map = crate::GcMap::new();
        gc_map.set_ref(0);
        gc_map.set_ref(2);
        smap.add(0x50, gc_map);

        registry.register(CompiledCodeRegion {
            code_start: 0x1000,
            code_size: 0x100,
            safepoint_map: smap,
            frame_size_slots: 4,
            loop_token: 1,
        });

        // Simulate a JIT frame: slots 0 and 2 hold GcRefs, slots 1 and 3
        // hold non-pointer data.
        let obj_a = gc.alloc_with_type(tid, ptr_size * 2);
        let obj_b = gc.alloc_with_type(tid, ptr_size * 2);
        unsafe {
            *(obj_a.0 as *mut GcRef) = GcRef::NULL;
            *((obj_a.0 + ptr_size) as *mut GcRef) = GcRef::NULL;
            *(obj_b.0 as *mut GcRef) = GcRef::NULL;
            *((obj_b.0 + ptr_size) as *mut GcRef) = GcRef::NULL;
        }

        let frame: [usize; 4] = [obj_a.0, 0xDEAD, obj_b.0, 0xBEEF];

        // Register frame slots as GC roots (simulating what the backend does
        // at a safepoint).
        let roots_from_frame = unsafe { registry.scan_frame(0x1050, frame.as_ptr()) };
        assert_eq!(roots_from_frame.len(), 2);

        // Register the scanned slots as roots with the GC.
        for root_ptr in &roots_from_frame {
            unsafe {
                gc.roots.add(*root_ptr as *mut GcRef);
            }
        }

        // Allocate many objects to force multiple nursery collections.
        for i in 0..200 {
            let filler = gc.alloc_with_type(tid, ptr_size * 2);
            unsafe {
                *(filler.0 as *mut u64) = i as u64;
            }
        }
        assert!(
            gc.minor_collections > 0,
            "should have triggered nursery collections"
        );

        // Read back the GcRefs from the frame slots (the GC may have updated
        // them when it promoted the objects).
        let ref_a = GcRef(frame[0]);
        let ref_b = GcRef(frame[2]);

        // The original nursery objects should have been forwarded.
        // The frame slots must now point to valid (non-nursery) addresses.
        assert!(!ref_a.is_null());
        assert!(!ref_b.is_null());
        assert!(
            !gc.is_in_nursery(ref_a.0),
            "object A should have been promoted out of nursery"
        );
        assert!(
            !gc.is_in_nursery(ref_b.0),
            "object B should have been promoted out of nursery"
        );

        // Verify non-GC slots are untouched.
        assert_eq!(frame[1], 0xDEAD);
        assert_eq!(frame[3], 0xBEEF);

        gc.roots.clear();
    }

    #[test]
    fn test_incremental_gc_under_allocation_pressure() {
        // Allocate many objects forming a linked list, promote to old gen,
        // then run an incremental major cycle with a tiny budget while
        // continuing to allocate. Verify data integrity throughout.

        let ptr_size = std::mem::size_of::<GcRef>();
        let mut gc = test_gc(1024);
        // Object layout: [next: GcRef][data: u64] = 16 bytes, GC ptr at offset 0
        let tid = gc.register_type(TypeInfo::with_gc_ptrs(ptr_size + 8, vec![0]));

        // Build a linked list of 20 objects.
        let mut prev = GcRef::NULL;
        let mut all_roots: Vec<GcRef> = Vec::new();
        for i in 0..20u64 {
            let obj = gc.alloc_with_type(tid, ptr_size + 8);
            unsafe {
                *(obj.0 as *mut GcRef) = prev;
                *((obj.0 + ptr_size) as *mut u64) = 0xA000 + i;
            }
            prev = obj;
            all_roots.push(obj);
        }

        // Root only the head of the list.
        let mut head = prev;
        unsafe {
            gc.roots.add(&mut head);
        }

        // Promote the whole list to old gen.
        gc.do_collect_nursery();
        assert!(!gc.is_in_nursery(head.0));

        // Start an incremental cycle with budget = 2 so it takes many steps.
        gc.set_mark_budget(2);
        gc.start_incremental_cycle();
        assert!(gc.is_incremental_marking());

        // Interleave incremental marking steps with new allocations.
        let mut step_count = 0;
        while gc.is_incremental_marking() {
            // Allocate a few new (short-lived) objects to maintain pressure.
            for _ in 0..5 {
                let tmp = gc.alloc_with_type(tid, ptr_size + 8);
                unsafe {
                    *(tmp.0 as *mut GcRef) = GcRef::NULL;
                    *((tmp.0 + ptr_size) as *mut u64) = 0xFFFF;
                }
            }
            let done = gc.incremental_mark_step();
            step_count += 1;
            if done {
                break;
            }
            assert!(step_count < 200, "incremental marking should converge");
        }

        // Complete the cycle.
        gc.finish_incremental_cycle();
        assert!(gc.major_collections > 0);

        // Walk the list from the head and verify all data values.
        let mut cursor = head;
        let mut count = 0;
        while !cursor.is_null() {
            let data = unsafe { *((cursor.0 + ptr_size) as *const u64) };
            // data should be 0xA000 + (19 - count) because the list was
            // built in reverse.
            assert_eq!(
                data,
                0xA000 + (19 - count) as u64,
                "data corruption detected at node {count}"
            );
            cursor = unsafe { *(cursor.0 as *const GcRef) };
            count += 1;
        }
        assert_eq!(count, 20, "entire list should be reachable");

        gc.roots.clear();
    }

    #[test]
    fn test_card_marking_under_write_pressure() {
        // Allocate a large array in old gen, write GC refs into many slots,
        // and verify that card marking accurately tracks the dirty ranges
        // so that nursery objects stored in the array survive collection.

        let ptr_size = std::mem::size_of::<GcRef>();
        let mut gc = test_gc(2048);

        // Element type: simple 16-byte object.
        let elem_tid = gc.register_type(TypeInfo::simple(16));

        // Array type: varsize, base_size = 8 (length field at offset 0),
        // each item is a GcRef (item_size = ptr_size), items have GC ptrs.
        let arr_tid = gc.register_type(TypeInfo::varsize(
            8,          // base_size (length field)
            ptr_size,   // item_size
            0,          // length_offset
            true,       // items_have_gc_ptrs
            Vec::new(), // no fixed GC ptr fields
        ));

        let array_length = 512usize;
        let total_payload = 8 + ptr_size * array_length;

        // Allocate the array directly in old gen (it's large).
        let arr = gc.alloc_in_oldgen(arr_tid, GcHeader::SIZE + total_payload);

        // Write the length field.
        unsafe {
            *(arr.0 as *mut usize) = array_length;
        }

        // Enable card marking on this object.
        let hdr = unsafe { header_of(arr.0) };
        hdr.set_flag(flags::HAS_CARDS);

        // Initialize all array slots to NULL.
        let items_start = arr.0 + 8;
        for i in 0..array_length {
            unsafe {
                *((items_start + i * ptr_size) as *mut GcRef) = GcRef::NULL;
            }
        }

        // Write nursery objects into scattered array positions and trigger
        // card-marking write barriers.
        let write_indices: Vec<usize> = vec![0, 1, 5, 64, 127, 128, 200, 255, 256, 400, 511];
        let mut expected_cards: HashSet<usize> = HashSet::new();
        let mut nursery_objs: Vec<(usize, GcRef)> = Vec::new();

        for &idx in &write_indices {
            let obj = gc.alloc_with_type(elem_tid, 16);
            // Write a distinctive marker.
            unsafe {
                *(obj.0 as *mut u64) = 0xCAFE_0000 + idx as u64;
            }
            // Store into the array.
            unsafe {
                *((items_start + idx * ptr_size) as *mut GcRef) = obj;
            }
            // Write barrier with card marking.
            gc.do_write_barrier_card(arr, idx, DEFAULT_CARD_PAGE_SHIFT);
            expected_cards.insert(idx >> DEFAULT_CARD_PAGE_SHIFT);
            nursery_objs.push((idx, obj));
        }

        // Verify the correct cards are dirty.
        let dirty = gc.dirty_cards(arr);
        let dirty_set: HashSet<usize> = dirty.into_iter().collect();
        assert_eq!(
            dirty_set, expected_cards,
            "dirty card set should match expected cards"
        );

        // Root the array so the collection traces it via card scanning.
        let mut root = arr;
        unsafe {
            gc.roots.add(&mut root);
        }

        // Trigger nursery collection — this should use card scanning
        // to find and promote the nursery objects stored in the array.
        gc.do_collect_nursery();

        // After collection, all stored objects should be promoted and
        // their data should be intact.
        for &(idx, _orig) in &nursery_objs {
            let slot_ref = unsafe { *((items_start + idx * ptr_size) as *const GcRef) };
            assert!(
                !slot_ref.is_null(),
                "array slot {idx} should not be null after collection"
            );
            assert!(
                !gc.is_in_nursery(slot_ref.0),
                "array slot {idx} should be promoted to old gen"
            );
            let marker = unsafe { *(slot_ref.0 as *const u64) };
            assert_eq!(
                marker,
                0xCAFE_0000 + idx as u64,
                "data in array slot {idx} should be preserved"
            );
        }

        // Cards should be cleared after collection.
        assert!(
            gc.card_dirty.is_empty(),
            "card table should be cleared after collection"
        );
        let hdr = unsafe { header_of(root.0) };
        assert!(
            !hdr.has_flag(flags::CARDS_SET),
            "CARDS_SET should be cleared after collection"
        );
        assert!(
            hdr.has_flag(flags::TRACK_YOUNG_PTRS),
            "TRACK_YOUNG_PTRS should be re-set after collection"
        );

        // Now do a second round of writes to verify card marking works
        // again after collection.
        let more_indices = vec![10, 300, 450];
        for &idx in &more_indices {
            let obj = gc.alloc_with_type(elem_tid, 16);
            unsafe {
                *(obj.0 as *mut u64) = 0xBEEF_0000 + idx as u64;
                *((items_start + idx * ptr_size) as *mut GcRef) = obj;
            }
            gc.do_write_barrier_card(arr, idx, DEFAULT_CARD_PAGE_SHIFT);
        }

        gc.do_collect_nursery();

        for &idx in &more_indices {
            let slot_ref = unsafe { *((items_start + idx * ptr_size) as *const GcRef) };
            assert!(!slot_ref.is_null());
            assert!(!gc.is_in_nursery(slot_ref.0));
            let marker = unsafe { *(slot_ref.0 as *const u64) };
            assert_eq!(marker, 0xBEEF_0000 + idx as u64);
        }

        gc.roots.clear();
    }

    // ── Write barrier + incremental marking interaction tests ──

    #[test]
    fn test_write_barrier_during_incremental_marking() {
        // Verify that write barriers fired during an active incremental
        // marking cycle correctly add old-gen objects to the remembered set.
        // The remembered set is processed at the next minor collection,
        // ensuring mutated old-gen objects are re-scanned.
        let ptr_size = std::mem::size_of::<GcRef>();
        let mut gc = test_gc(4096);
        // Object layout: [field: GcRef] = ptr_size bytes, GC ptr at offset 0
        let tid = gc.register_type(TypeInfo::with_gc_ptrs(ptr_size, vec![0]));

        // Create 3 old-gen objects (A, B, C) with no pointers initially.
        let obj_a = gc.alloc_with_type(tid, ptr_size);
        let obj_b = gc.alloc_with_type(tid, ptr_size);
        let obj_c = gc.alloc_with_type(tid, ptr_size);
        unsafe {
            *(obj_a.0 as *mut GcRef) = GcRef::NULL;
            *(obj_b.0 as *mut GcRef) = GcRef::NULL;
            *(obj_c.0 as *mut GcRef) = GcRef::NULL;
        }

        let mut root_a = obj_a;
        let mut root_b = obj_b;
        let mut root_c = obj_c;
        unsafe {
            gc.roots.add(&mut root_a);
            gc.roots.add(&mut root_b);
            gc.roots.add(&mut root_c);
        }

        // Promote all to old gen.
        gc.do_collect_nursery();
        assert!(!gc.is_in_nursery(root_a.0));
        assert!(!gc.is_in_nursery(root_b.0));
        assert!(!gc.is_in_nursery(root_c.0));

        // All old-gen objects should have TRACK_YOUNG_PTRS set.
        assert!(unsafe { header_of(root_a.0).has_flag(flags::TRACK_YOUNG_PTRS) });
        assert!(unsafe { header_of(root_b.0).has_flag(flags::TRACK_YOUNG_PTRS) });
        assert!(unsafe { header_of(root_c.0).has_flag(flags::TRACK_YOUNG_PTRS) });

        // Start an incremental marking cycle with budget=1 so it stays
        // active across multiple steps.
        gc.set_mark_budget(1);
        gc.start_incremental_cycle();
        assert!(gc.is_incremental_marking());

        // During marking, perform write barriers on A and B.
        gc.do_write_barrier(root_a);
        gc.do_write_barrier(root_b);

        // A and B should be in the remembered set.
        assert!(
            gc.remembered_set.contains(&root_a.0),
            "write barrier should add A to remembered set during marking"
        );
        assert!(
            gc.remembered_set.contains(&root_b.0),
            "write barrier should add B to remembered set during marking"
        );
        // C was not written to, so it shouldn't be in remembered set.
        assert!(
            !gc.remembered_set.contains(&root_c.0),
            "C should not be in remembered set"
        );

        // TRACK_YOUNG_PTRS should be cleared on A and B.
        assert!(!unsafe { header_of(root_a.0).has_flag(flags::TRACK_YOUNG_PTRS) });
        assert!(!unsafe { header_of(root_b.0).has_flag(flags::TRACK_YOUNG_PTRS) });
        // C still has it.
        assert!(unsafe { header_of(root_c.0).has_flag(flags::TRACK_YOUNG_PTRS) });

        // Drive the incremental cycle to completion via nursery collections.
        for _ in 0..50 {
            if !gc.is_incremental_marking() {
                break;
            }
            gc.do_collect_nursery();
        }

        // No objects should be lost — all 3 are still rooted and should
        // survive the sweep.
        gc.do_collect_full();
        assert_eq!(
            gc.oldgen.object_count(),
            3,
            "all 3 rooted objects should survive full collection"
        );

        gc.roots.clear();
    }

    #[test]
    fn test_mutation_during_incremental_preserves_reachability() {
        // During an incremental marking cycle, mutate an old-gen object to
        // point to a newly promoted object (D). The write barrier ensures D
        // is reachable through the remembered set, so D survives the sweep.
        let ptr_size = std::mem::size_of::<GcRef>();
        let mut gc = test_gc(4096);
        // Object layout: [next: GcRef] = ptr_size bytes, GC ptr at offset 0
        let tid = gc.register_type(TypeInfo::with_gc_ptrs(ptr_size, vec![0]));

        // Build a chain A→B→C in the nursery.
        let obj_c = gc.alloc_with_type(tid, ptr_size);
        unsafe {
            *(obj_c.0 as *mut GcRef) = GcRef::NULL;
        }
        let obj_b = gc.alloc_with_type(tid, ptr_size);
        unsafe {
            *(obj_b.0 as *mut GcRef) = obj_c;
        }
        let obj_a = gc.alloc_with_type(tid, ptr_size);
        unsafe {
            *(obj_a.0 as *mut GcRef) = obj_b;
        }

        let mut root_a = obj_a;
        unsafe {
            gc.roots.add(&mut root_a);
        }

        // Promote A→B→C to old gen.
        gc.do_collect_nursery();
        assert!(!gc.is_in_nursery(root_a.0));
        let promoted_b = unsafe { *(root_a.0 as *const GcRef) };
        let promoted_c = unsafe { *(promoted_b.0 as *const GcRef) };
        assert!(!gc.is_in_nursery(promoted_b.0));
        assert!(!gc.is_in_nursery(promoted_c.0));

        // Start incremental marking with a small budget.
        gc.set_mark_budget(1);
        gc.start_incremental_cycle();
        assert!(gc.is_incremental_marking());

        // Do one marking step — only partially through the graph.
        gc.incremental_mark_step();

        // Now allocate D in the nursery.
        let obj_d = gc.alloc_with_type(tid, ptr_size);
        unsafe {
            *(obj_d.0 as *mut GcRef) = GcRef::NULL;
        }

        // Promote D to old gen by rooting it temporarily.
        let mut root_d = obj_d;
        unsafe {
            gc.roots.add(&mut root_d);
        }
        gc.do_collect_nursery();
        assert!(!gc.is_in_nursery(root_d.0));

        // Mutate A to point to D instead of B (A→D).
        // The write barrier ensures A is in the remembered set.
        gc.do_write_barrier(root_a);
        unsafe {
            *(root_a.0 as *mut GcRef) = root_d;
        }

        // Remove D's direct root — D is now only reachable via A→D.
        gc.roots.remove(&mut root_d);

        // Complete the incremental cycle: drive marking to completion
        // and then sweep.
        gc.do_collect_full();

        // A and D should survive (A is rooted, D reachable via A→D).
        // B and C may or may not survive depending on marking order,
        // but D MUST survive.
        let d_addr = root_d.0;
        let a_field = unsafe { *(root_a.0 as *const GcRef) };
        assert_eq!(
            a_field.0, d_addr,
            "A should still point to D after collection"
        );

        // Verify D is actually alive by checking it's still in old gen.
        assert!(
            !a_field.is_null(),
            "D must survive collection — it's reachable via A"
        );

        gc.roots.clear();
    }

    #[test]
    fn test_nursery_alloc_during_incremental_marking() {
        // Allocate new nursery objects between incremental marking steps.
        // Trigger nursery collections that piggyback marking steps.
        // Verify that newly allocated objects are correctly promoted and
        // that the original old-gen object graph remains intact.
        let ptr_size = std::mem::size_of::<GcRef>();
        let mut gc = test_gc(1024); // small nursery to force frequent collections
        // Object layout: [next: GcRef][data: u64] = ptr_size + 8 bytes
        let tid = gc.register_type(TypeInfo::with_gc_ptrs(ptr_size + 8, vec![0]));

        // Create initial old-gen objects (a small chain of 5).
        let mut prev = GcRef::NULL;
        for i in 0..5u64 {
            let obj = gc.alloc_with_type(tid, ptr_size + 8);
            unsafe {
                *(obj.0 as *mut GcRef) = prev;
                *((obj.0 + ptr_size) as *mut u64) = 0xBB00 + i;
            }
            prev = obj;
        }

        let mut head = prev;
        unsafe {
            gc.roots.add(&mut head);
        }

        // Promote the chain to old gen.
        gc.do_collect_nursery();
        assert!(!gc.is_in_nursery(head.0));
        let old_count_after_promote = gc.oldgen.object_count();
        assert_eq!(old_count_after_promote, 5);

        // Start incremental marking with budget=1.
        gc.set_mark_budget(1);
        gc.start_incremental_cycle();
        assert!(gc.is_incremental_marking());

        let minor_before = gc.minor_collections;

        // Allocate many nursery objects between marking steps. The small
        // nursery (1024 bytes) forces nursery collections that piggyback
        // incremental marking steps.
        for i in 0..100u64 {
            let tmp = gc.alloc_with_type(tid, ptr_size + 8);
            unsafe {
                *(tmp.0 as *mut GcRef) = GcRef::NULL;
                *((tmp.0 + ptr_size) as *mut u64) = 0xDD00 + i;
            }
        }

        let minor_during = gc.minor_collections - minor_before;
        assert!(
            minor_during >= 2,
            "should have triggered multiple nursery collections, got {minor_during}"
        );

        // The incremental marking should have advanced via piggybacking.
        assert!(
            gc.incremental_objects_marked() > 0,
            "piggybacked marking should have processed some objects"
        );

        // Drive any remaining incremental marking to completion and sweep.
        // Use do_collect_full which correctly handles an in-progress cycle.
        gc.do_collect_full();
        assert!(gc.major_collections > 0);

        // Verify the original chain is intact — the 5 rooted objects
        // survived the incremental major cycle.
        let mut cursor = head;
        let mut count = 0;
        while !cursor.is_null() {
            let data = unsafe { *((cursor.0 + ptr_size) as *const u64) };
            assert_eq!(
                data,
                0xBB00 + (4 - count) as u64,
                "original chain data corrupted at node {count}"
            );
            cursor = unsafe { *(cursor.0 as *const GcRef) };
            count += 1;
        }
        assert_eq!(count, 5, "entire chain should be reachable after cycle");

        gc.roots.clear();
    }

    // ── JIT integration hook tests ──

    #[test]
    fn test_jit_remember_young_pointer() {
        let mut gc = test_gc(4096);
        gc.register_type(TypeInfo::simple(16));

        // Allocate an old-gen object.
        let obj = gc.alloc_in_oldgen(0, GcHeader::SIZE + 16);
        assert!(!gc.is_in_nursery(obj.0));

        // Initially TRACK_YOUNG_PTRS is set.
        let hdr = unsafe { header_of(obj.0) };
        assert!(hdr.has_flag(flags::TRACK_YOUNG_PTRS));

        // JIT fast-path barrier: clears flag and adds to remembered set.
        gc.jit_remember_young_pointer(obj);

        assert!(!hdr.has_flag(flags::TRACK_YOUNG_PTRS));
        assert_eq!(gc.remembered_set_len(), 1);

        // Calling again adds a second entry (JIT fast-path does not
        // deduplicate; the collector handles this during minor collection).
        gc.jit_remember_young_pointer(obj);
        assert_eq!(gc.remembered_set_len(), 2);
    }

    #[test]
    fn test_jit_remember_young_pointer_null_is_noop() {
        let mut gc = test_gc(4096);
        gc.jit_remember_young_pointer(GcRef::NULL);
        assert_eq!(gc.remembered_set_len(), 0);
    }

    #[test]
    fn test_jit_remember_young_pointer_survives_collection() {
        // Verify that the remembered-set entry from jit_remember_young_pointer
        // causes a young object to survive minor collection.
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::with_gc_ptrs(
            std::mem::size_of::<GcRef>(),
            vec![0],
        ));

        // Create an old-gen parent and a young child.
        let parent = gc.alloc_in_oldgen(tid, GcHeader::SIZE + std::mem::size_of::<GcRef>());
        let child = gc.alloc_with_type(tid, std::mem::size_of::<GcRef>());
        unsafe {
            *(child.0 as *mut u64) = 0xABCD_1234;
            *(parent.0 as *mut GcRef) = child;
        }

        // Use the JIT hook instead of do_write_barrier.
        gc.jit_remember_young_pointer(parent);

        let mut root = parent;
        unsafe {
            gc.roots.add(&mut root);
        }

        gc.collect_nursery();

        // The child should have been promoted.
        let child_ref = unsafe { *(root.0 as *const GcRef) };
        assert!(!gc.is_in_nursery(child_ref.0));
        assert!(!child_ref.is_null());
        let val = unsafe { *(child_ref.0 as *const u64) };
        assert_eq!(val, 0xABCD_1234);

        gc.roots.clear();
    }

    #[test]
    fn test_can_optimize_cond_call() {
        let gc = test_gc(4096);
        assert!(gc.can_optimize_cond_call());
    }

    #[test]
    fn test_can_optimize_cond_call_via_trait() {
        let gc = test_gc(4096);
        let alloc: &dyn GcAllocator = &gc;
        assert!(alloc.can_optimize_cond_call());
    }

    #[test]
    fn test_gc_step_no_work_when_old_gen_small() {
        let mut gc = test_gc(4096);
        gc.register_type(TypeInfo::simple(16));

        // With an almost-empty old gen, gc_step should do nothing.
        assert!(!gc.gc_step());
        assert!(!gc.is_incremental_marking());
    }

    #[test]
    fn test_gc_step_triggers_incremental() {
        let mut gc = test_gc(256);
        gc.register_type(TypeInfo::simple(16));

        // Force a minor collection to set last_major_bytes baseline.
        let obj = gc.alloc_with_type(0, 16);
        let mut root = obj;
        unsafe {
            gc.roots.add(&mut root);
        }
        gc.collect_full();

        // Fill old gen to trigger the major-cycle ratio threshold.
        // Allocate many objects directly in old gen.
        for _ in 0..200 {
            gc.alloc_in_oldgen(0, GcHeader::SIZE + 16);
        }

        // gc_step should now start an incremental cycle and do work.
        let did_work = gc.gc_step();
        assert!(did_work);

        gc.roots.clear();
    }

    #[test]
    fn test_gc_step_advances_marking() {
        let mut gc = test_gc(256);
        let tid = gc.register_type(TypeInfo::simple(16));

        // Build a chain of old-gen objects so there's marking work to do.
        let mut prev = GcRef::NULL;
        let ptr_tid = gc.register_type(TypeInfo::with_gc_ptrs(
            std::mem::size_of::<GcRef>(),
            vec![0],
        ));
        for _ in 0..20 {
            let obj = gc.alloc_with_type(ptr_tid, std::mem::size_of::<GcRef>());
            unsafe {
                *(obj.0 as *mut GcRef) = prev;
            }
            prev = obj;
        }

        let mut root = prev;
        unsafe {
            gc.roots.add(&mut root);
        }

        // Promote everything to old gen.
        gc.collect_nursery();

        // Add more old-gen objects to trigger the ratio threshold.
        for _ in 0..200 {
            gc.alloc_in_oldgen(tid, GcHeader::SIZE + 16);
        }

        // First step: start cycle.
        let work1 = gc.gc_step();
        assert!(work1);
        let marked_after_1 = gc.incremental_objects_marked();

        // Second step: advance marking further.
        if gc.is_incremental_marking() {
            let work2 = gc.gc_step();
            assert!(work2);
            assert!(gc.incremental_objects_marked() >= marked_after_1);
        }

        gc.roots.clear();
    }

    // ── Pin / Unpin / jit_free tests ──

    #[test]
    fn test_pin_prevents_nursery_move() {
        let mut gc = test_gc(4096);
        gc.register_type(TypeInfo::simple(16));

        let obj = gc.alloc_with_type(0, 16);
        assert!(gc.is_in_nursery(obj.0));

        // Write a marker value.
        unsafe {
            *(obj.0 as *mut u64) = 0xCAFE_BABE;
        }

        // Pin the object and root it.
        assert!(gc.pin(obj));
        assert!(gc.is_pinned(obj));

        let mut root = obj;
        unsafe {
            gc.roots.add(&mut root);
        }

        // Trigger minor collection.
        gc.do_collect_nursery();

        // The root should still point to the same nursery address.
        assert_eq!(root.0, obj.0);
        assert!(gc.is_in_nursery(root.0));

        // Data should be intact.
        let val = unsafe { *(root.0 as *const u64) };
        assert_eq!(val, 0xCAFE_BABE);

        gc.roots.clear();
    }

    #[test]
    fn test_unpin_allows_move() {
        let mut gc = test_gc(4096);
        gc.register_type(TypeInfo::simple(16));

        let obj = gc.alloc_with_type(0, 16);
        assert!(gc.is_in_nursery(obj.0));

        // Write a marker.
        unsafe {
            *(obj.0 as *mut u64) = 0xDEAD_BEEF;
        }

        // Pin, then unpin.
        assert!(gc.pin(obj));
        gc.unpin(obj);
        assert!(!gc.is_pinned(obj));

        let mut root = obj;
        unsafe {
            gc.roots.add(&mut root);
        }

        // Collection should now move the object to old gen.
        gc.do_collect_nursery();

        // The root should now point to old gen (different address).
        assert!(!gc.is_in_nursery(root.0));
        assert_ne!(root.0, obj.0);

        // Data should be preserved after the move.
        let val = unsafe { *(root.0 as *const u64) };
        assert_eq!(val, 0xDEAD_BEEF);

        gc.roots.clear();
    }

    #[test]
    fn test_is_pinned_query() {
        let mut gc = test_gc(4096);
        gc.register_type(TypeInfo::simple(16));

        let obj = gc.alloc_with_type(0, 16);

        // Not pinned by default.
        assert!(!gc.is_pinned(obj));

        // Pin it.
        assert!(gc.pin(obj));
        assert!(gc.is_pinned(obj));

        // Null cannot be pinned.
        assert!(!gc.pin(GcRef(0)));
        assert!(!gc.is_pinned(GcRef(0)));

        // Unpin.
        gc.unpin(obj);
        assert!(!gc.is_pinned(obj));
    }

    #[test]
    fn test_jit_free_unregisters_code() {
        let mut gc = test_gc(4096);

        let smap = SafepointMap::new();
        gc.compiled_code_registry.register(CompiledCodeRegion {
            code_start: 0x1000,
            code_size: 256,
            safepoint_map: smap,
            frame_size_slots: 4,
            loop_token: 1,
        });

        let smap2 = SafepointMap::new();
        gc.compiled_code_registry.register(CompiledCodeRegion {
            code_start: 0x2000,
            code_size: 512,
            safepoint_map: smap2,
            frame_size_slots: 8,
            loop_token: 2,
        });

        assert_eq!(gc.compiled_code_registry.len(), 2);

        // Free the first region.
        gc.jit_free(0x1000, 256);

        assert_eq!(gc.compiled_code_registry.len(), 1);
        assert!(gc.compiled_code_registry.find_region(0x1050).is_none());
        assert!(gc.compiled_code_registry.find_region(0x2050).is_some());

        // Free the second region.
        gc.jit_free(0x2000, 512);
        assert_eq!(gc.compiled_code_registry.len(), 0);
    }
}
